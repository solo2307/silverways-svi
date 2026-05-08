from __future__ import annotations

import base64
import contextlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

DEFAULT_SMOOTHNESS_SCORE_MAP = {
    "excellent": 5,
    "good": 4,
    "intermediate": 3,
    "bad": 2,
    "very_bad": 1,
    "horrible": 1,
    "very_horrible": 1,
    "impassable": 1,
    "not_visible": None,
}


def load_quality_rules(path: str | Path | None) -> dict[str, Any]:
    """Load optional OSM quality rules from YAML."""
    if path is None:
        return {}

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Quality rules file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    return data or {}

def pil_to_data_uri(image: Image.Image) -> str:
    """Convert PIL image to base64 data URI for llama-cpp multimodal input."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def resize_crop_for_vlm(
    image: Image.Image,
    min_short_side: int = 384,
    max_long_side: int = 768,
) -> Image.Image:
    """Upscale small crops for VLM input while preserving aspect ratio."""
    image = image.convert("RGB")
    width, height = image.size

    short_side = min(width, height)
    long_side = max(width, height)

    if short_side >= min_short_side:
        return image

    scale = min_short_side / max(short_side, 1)

    if long_side * scale > max_long_side:
        scale = max_long_side / max(long_side, 1)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from model output."""
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return {
            "error": "no_json_found",
            "raw": text,
        }

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1

            if stack == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return {
                        "error": "json_parse_failed",
                        "raw": text,
                    }

    return {
        "error": "json_parse_failed",
        "raw": text,
    }


@contextlib.contextmanager
def suppress_native_output(enabled: bool = True):
    """
    Suppress stdout/stderr written by native C/C++ libraries.

    contextlib.redirect_stdout only catches Python-level prints.
    llama.cpp / CUDA logs can write directly to file descriptors 1 and 2.
    """
    if not enabled:
        yield
        return

    try:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()

        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()

        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)

        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)

            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()

                os.dup2(saved_stdout_fd, stdout_fd)
                os.dup2(saved_stderr_fd, stderr_fd)

                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)

    except Exception:
        yield


class InternVLTagger:
    """InternVL GGUF tagger using llama-cpp-python."""

    def __init__(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        rules_path: str | Path | None = None,
    ) -> None:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        self.verbose = verbose
        self.rules = load_quality_rules(rules_path)
        self.smoothness_score_map = self.rules.get(
            "smoothness_score_1_to_5",
            DEFAULT_SMOOTHNESS_SCORE_MAP,
        )

        model_path = Path(model_path)
        mmproj_path = Path(mmproj_path)

        if not model_path.exists():
            raise FileNotFoundError(f"InternVL model not found: {model_path}")

        if not mmproj_path.exists():
            raise FileNotFoundError(f"InternVL mmproj file not found: {mmproj_path}")

        with suppress_native_output(enabled=not self.verbose):
            chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))

            self.model = Llama(
                model_path=str(model_path),
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=self.verbose,
            )

    def create_chat_completion_quietly(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 220,
    ) -> dict[str, Any]:
        """Run llama.cpp chat completion while suppressing native logs."""
        with suppress_native_output(enabled=not self.verbose):
            return self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
            )

    def predict_osm_tags(
        self,
        crop: Image.Image,
        feature_label: str,
    ) -> dict[str, Any]:
        """
        Predict OSM-style surface and smoothness tags.

        Supports dedicated sidewalk crops and shared pedestrian/vehicle surface crops.
        """
        crop = resize_crop_for_vlm(crop)
        data_uri = pil_to_data_uri(crop)

        surface_materials = (
            "asphalt, concrete, concrete_plates, paving_stones, paving_slabs, sett, "
            "cobblestone, unhewn_cobblestone, bricks, stone, gravel, fine_gravel, "
            "compacted, dirt, ground, grass, sand, mud, wood, metal, unknown"
        )

        smoothness_tags = (
            "excellent, good, intermediate, bad, very_bad, horrible, "
            "very_horrible, impassable, not_visible"
        )

        walkway_context_tags = (
            "dedicated_sidewalk, shared_pedestrian_vehicle_surface, "
            "driveway_or_private_access, road_only, not_visible"
        )

        schema = {
            "feature": feature_label,
            "dedicated_sidewalk": "<true | false | null>",
            "walkway_context": (
                "<dedicated_sidewalk | shared_pedestrian_vehicle_surface | "
                "driveway_or_private_access | road_only | not_visible>"
            ),
            "surface": "<one value from allowed surface list>",
            "smoothness_osm": (
                "<excellent | good | intermediate | bad | very_bad | horrible | "
                "very_horrible | impassable | not_visible>"
            ),
            "smoothness_score_1_to_5": "<integer 1-5 or null>",
            "confidence": "<low | medium | high>",
            "visible_evidence": "<maximum 8 words>",
        }

        smoothness_definitions = """
OSM smoothness:
excellent: nearly perfect, flat surface.
good: mostly smooth, minor wear or cracks.
intermediate: usable but visibly uneven, patched, jointed, or mildly cracked.
bad: damaged, bumpy, cracked, or uncomfortable for small wheels.
very_bad: very rough or difficult for wheelchairs/strollers.
horrible: passable only with difficulty.
very_horrible: severe rubble, obstacles, or broken surface.
impassable: not usable.
not_visible: cannot judge from the crop.
"""

        system_msg = (
            "Return ONLY a valid JSON object. No markdown. No extra text.\n"
            f"Exact JSON schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Allowed surfaces: {surface_materials}\n\n"
            f"Allowed OSM smoothness values: {smoothness_tags}\n\n"
            f"Allowed walkway_context values: {walkway_context_tags}\n\n"
            f"{smoothness_definitions}\n"
            "Judge only the visible pedestrian-relevant paved or walkable surface. "
            "If there is a curb-separated or clearly dedicated sidewalk/path, set "
            "dedicated_sidewalk=true and walkway_context='dedicated_sidewalk'. "
            "If there is no separate sidewalk but pedestrians likely share the same paved "
            "surface with vehicles, set dedicated_sidewalk=false and "
            "walkway_context='shared_pedestrian_vehicle_surface'. "
            "If the crop is mainly a private driveway or private access area, set "
            "walkway_context='driveway_or_private_access'. "
            "If it is only a vehicle road with no clear pedestrian walking relevance, set "
            "walkway_context='road_only'. "
            "If the walkable surface is too occluded, tiny, blurry, or not visible, set "
            "walkway_context='not_visible', smoothness_osm='not_visible', and "
            "smoothness_score_1_to_5=null. "
            "Base smoothness only on visible flatness, cracks, potholes, bumps, rubble, "
            "missing pavement, ruts, and usability for pedestrians, wheelchairs, strollers, "
            "or small wheeled mobility devices. "
            "Do not judge grass, walls, buildings, parked cars, or sky unless they block "
            "visibility of the target surface."
        )

        messages = [
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Analyze this detected {feature_label} region and return "
                            "the OSM-style walking surface quality JSON."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                ],
            },
        ]

        response = self.create_chat_completion_quietly(
            messages=messages,
            max_tokens=220,
        )

        raw = response["choices"][0]["message"]["content"].strip()
        parsed = extract_json(raw)
        
        # Normalize model quirks before scoring/saving.
        dedicated_value = parsed.get("dedicated_sidewalk")
        if isinstance(dedicated_value, str):
            value = dedicated_value.strip().lower()
            if value == "true":
                parsed["dedicated_sidewalk"] = True
            elif value == "false":
                parsed["dedicated_sidewalk"] = False
            elif value in {"null", "none", "unknown", "not_visible"}:
                parsed["dedicated_sidewalk"] = None
        
        walkway_context = parsed.get("walkway_context")
        if isinstance(walkway_context, str):
            parsed["walkway_context"] = walkway_context.strip()
        
        surface = parsed.get("surface")
        if isinstance(surface, str):
            parsed["surface"] = surface.strip()
        
        smoothness = parsed.get("smoothness_osm")
        if isinstance(smoothness, str):
            parsed["smoothness_osm"] = smoothness.strip()
        
        confidence = parsed.get("confidence")
        if isinstance(confidence, str):
            parsed["confidence"] = confidence.strip().lower()
        
        osm_to_score = self.smoothness_score_map
        
        osm_value = parsed.get("smoothness_osm")

        if isinstance(osm_value, str):
            osm_value = osm_value.strip()
        
        if osm_value in osm_to_score:
            parsed["smoothness_score_1_to_5_raw"] = parsed.get("smoothness_score_1_to_5")
            parsed["smoothness_score_1_to_5"] = osm_to_score[osm_value]
            parsed["smoothness_score_source"] = "mapped_from_smoothness_osm"
        
        parsed["_raw_internvl_response"] = raw
        
        return parsed