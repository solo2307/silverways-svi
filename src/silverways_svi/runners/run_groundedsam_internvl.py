# src/silverways_svi/runners/run_groundedsam_internvl.py

from __future__ import annotations

import base64
import contextlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from PIL import Image
from tqdm import tqdm

from silverways_svi.data.image_dataset import ImageDataset
from silverways_svi.runners.run_grounded_sam import (
    GroundingDinoDetector,
    SAM2Segmenter,
    masks_to_numpy,
    save_annotated,
    save_json,
    save_masks,
)

app = typer.Typer(
    help="Run GroundedSAM localization followed by InternVL sidewalk quality tagging."
)


@app.callback()
def main() -> None:
    """GroundedSAM + InternVL runner."""


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


def pil_to_data_uri(image: Image.Image) -> str:
    """Convert PIL image to base64 data URI for llama-cpp multimodal input."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Get xyxy bounding box from a binary mask."""
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def safe_name(text: str) -> str:
    """Make a safe filename component from a detection label."""
    cleaned = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in text
    ).strip("_")

    return cleaned or "object"


@contextlib.contextmanager
def suppress_native_output(enabled: bool = True):
    """
    Suppress stdout/stderr written by native C/C++ libraries.

    contextlib.redirect_stdout only catches Python-level prints.
    llama.cpp / Metal / CUDA logs can write directly to file descriptors 1 and 2.
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
        # Do not break inference if suppression fails.
        yield


def crop_detection(
    image: Image.Image,
    detection: dict[str, Any],
    mask: np.ndarray | None = None,
    padding_ratio: float = 0.15,
    use_mask: bool = True,
) -> Image.Image:
    """
    Crop a detected object.

    If a SAM mask is available, crop from the mask bounding box.
    If use_mask=True, background outside the mask is whitened.
    """
    image = image.convert("RGB")
    width, height = image.size

    mask_box = bbox_from_mask(mask) if mask is not None else None

    if mask_box is not None:
        x1, y1, x2, y2 = mask_box
    else:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["xyxy"]]

    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    crop = image.crop((x1, y1, x2, y2))

    if use_mask and mask is not None:
        mask_crop = Image.fromarray((mask.astype(np.uint8) * 255)).crop(
            (x1, y1, x2, y2)
        )
        white_bg = Image.new("RGB", crop.size, (255, 255, 255))
        white_bg.paste(crop, mask=mask_crop)
        crop = white_bg

    return crop


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


class InternVLTagger:
    """InternVL GGUF tagger using llama-cpp-python."""

    def __init__(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        self.verbose = verbose

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
        max_tokens: int = 150,
    ) -> dict[str, Any]:
        """
        Run llama.cpp chat completion quietly.

        llama.cpp / multimodal projector logs are native-level prints,
        so we suppress file descriptors, not only Python stdout/stderr.
        """
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
        """Predict OSM-style surface and smoothness tags for a detected crop."""
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

        schema = {
            "feature": feature_label,
            "surface": "<one value from allowed surface list>",
            "smoothness_osm": (
                "<excellent | good | intermediate | bad | very_bad | horrible | "
                "very_horrible | impassable | not_visible>"
            ),
            "smoothness_score_1_to_5": "<integer 1-5 or null>",
            "confidence": "<low | medium | high>",
            "visible_evidence": (
                "<short reason based only on visible surface condition>"
            ),
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
            f"{smoothness_definitions}\n"
            "Judge only the visible cropped or masked sidewalk/path region. "
            "Do not judge nearby road, curb, grass, wall, or building unless it is part of the target surface. "
            "Base smoothness only on visible flatness, cracks, potholes, bumps, rubble, missing pavement, ruts, and usability for pedestrians, wheelchairs, strollers, or small wheeled mobility devices. "
            "If the surface is too occluded, small, blurry, or not visible enough, use smoothness_osm='not_visible' and smoothness_score_1_to_5=null."
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
                            "the OSM-style sidewalk quality JSON."
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
            max_tokens=150,
        )

        raw = response["choices"][0]["message"]["content"].strip()
        parsed = extract_json(raw)

        osm_to_score = {
            "excellent": 5,
            "good": 5,
            "intermediate": 4,
            "bad": 3,
            "very_bad": 2,
            "horrible": 2,
            "very_horrible": 1,
            "impassable": 1,
            "not_visible": None,
        }

        osm_value = parsed.get("smoothness_osm")

        if osm_value in osm_to_score:
            parsed["smoothness_score_1_to_5"] = osm_to_score[osm_value]

        parsed["_raw_internvl_response"] = raw

        return parsed


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/groundedsam_internvl.yaml"),
        "--config",
        "-c",
        help="Path to GroundedSAM + InternVL config YAML.",
    )
) -> None:
    """Run GroundedSAM and enrich detections with InternVL OSM tags."""
    cfg = read_yaml(config)

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    recursive = bool(cfg.get("recursive", False))
    limit = cfg.get("limit")
    device = str(cfg.get("device", "auto"))

    grounding_cfg = cfg["grounding_dino"]
    sam2_cfg = cfg["sam2"]
    prompt_cfg = cfg["prompt"]
    internvl_cfg = cfg["internvl"]
    output_cfg = cfg.get("outputs", {})

    text_labels = prompt_cfg["text_labels"]
    box_threshold = float(prompt_cfg.get("box_threshold", 0.35))
    text_threshold = float(prompt_cfg.get("text_threshold", 0.25))

    dataset = ImageDataset(input_dir=input_dir, recursive=recursive)
    items = list(dataset)

    if limit is not None:
        items = items[: int(limit)]

    if not items:
        typer.echo(f"No images found in: {input_dir}")
        raise typer.Exit(code=0)

    detector = GroundingDinoDetector(
        name_or_path=grounding_cfg["name_or_path"],
        local_dir=grounding_cfg.get("local_dir"),
        device=device,
    )

    segmenter = SAM2Segmenter(
        weights=sam2_cfg["weights"],
        imgsz=int(sam2_cfg.get("imgsz", 1024)),
        device=device,
    )

    tagger = InternVLTagger(
        model_path=internvl_cfg["model_path"],
        mmproj_path=internvl_cfg["mmproj_path"],
        n_ctx=int(internvl_cfg.get("n_ctx", 8192)),
        n_gpu_layers=int(internvl_cfg.get("n_gpu_layers", 0)),
        verbose=bool(internvl_cfg.get("verbose", False)),
    )

    json_dir = output_dir / "json"
    masks_dir = output_dir / "masks"
    annotated_dir = output_dir / "annotated"
    crops_dir = output_dir / "crops"

    for item in tqdm(items, desc="GroundedSAM + InternVL"):
        image = item.load_rgb()

        detections = detector.predict(
            image=image,
            text_labels=text_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        boxes_xyxy = [det["xyxy"] for det in detections]

        if len(boxes_xyxy) == 0:
            payload = {
                "image_id": item.stem,
                "image_path": str(item.path),
                "text_labels": text_labels,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "num_detections": 0,
                "num_masks": 0,
                "detections": [],
            }

            if output_cfg.get("save_json", True):
                save_json(json_dir / f"{item.stem}.json", payload)

            continue

        sam_result = segmenter.predict_from_boxes(
            image_path=item.path,
            boxes_xyxy=boxes_xyxy,
        )

        masks = masks_to_numpy(sam_result)

        enriched_detections = []

        for idx, det in enumerate(detections):
            mask = None

            if masks is not None and idx < masks.shape[0]:
                mask = masks[idx]

            crop = crop_detection(
                image=image,
                detection=det,
                mask=mask,
                padding_ratio=float(cfg.get("crop_padding_ratio", 0.15)),
                use_mask=bool(cfg.get("use_masked_crop", True)),
            )

            if output_cfg.get("save_crops", True):
                crops_dir.mkdir(parents=True, exist_ok=True)
                label_safe = safe_name(str(det["label"]))
                crop.save(crops_dir / f"{item.stem}_det{idx:03d}_{label_safe}.jpg")

            osm_tags = tagger.predict_osm_tags(
                crop=crop,
                feature_label=str(det["label"]),
            )

            enriched_detections.append(
                {
                    **det,
                    "internvl_osm_tags": osm_tags,
                }
            )

        payload = {
            "image_id": item.stem,
            "image_path": str(item.path),
            "text_labels": text_labels,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "num_detections": len(detections),
            "num_masks": 0 if masks is None else int(masks.shape[0]),
            "detections": enriched_detections,
        }

        if output_cfg.get("save_json", True):
            save_json(json_dir / f"{item.stem}.json", payload)

        if output_cfg.get("save_masks", True) and masks is not None:
            save_masks(masks_dir / f"{item.stem}_masks.npy", masks)

        if output_cfg.get("save_annotated", True):
            save_annotated(
                annotated_dir / f"{item.stem}_groundedsam_internvl.jpg",
                image=image,
                detections=detections,
                masks=masks,
            )

    typer.echo(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    app()