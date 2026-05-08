from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import typer
import yaml
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from silverways_svi.data.image_dataset import ImageDataset
from silverways_svi.runners.internvl_tagger import InternVLTagger
from silverways_svi.runners.region_utils import (
    crop_detection,
    safe_name,
    save_json,
    save_masks,
    save_rgb,
)

app = typer.Typer(
    help="Run Mask2Former sidewalk/shared-surface segmentation followed by InternVL tagging."
)


@app.callback()
def main() -> None:
    """Mask2Former + InternVL runner."""


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve torch device."""
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def normalize_label(label: str) -> str:
    """Normalize class label for matching."""
    return label.lower().strip().replace("_", " ").replace("-", " ")


def find_label_id(label: str, id2label: dict[int, str]) -> int | None:
    """Find class ID by normalized exact label match."""
    wanted = normalize_label(label)

    for idx, class_label in id2label.items():
        if normalize_label(class_label) == wanted:
            return int(idx)

    return None


def clean_binary_mask(
    binary_mask: np.ndarray,
    open_kernel_size: int = 0,
    close_kernel_size: int = 0,
) -> np.ndarray:
    """
    Optionally clean a binary mask.

    Opening can break tiny bridges between separate regions.
    Closing can fill small holes inside regions.
    Use 0 to disable each operation.
    """
    mask = binary_mask.astype(np.uint8)

    if open_kernel_size and open_kernel_size > 1:
        kernel = np.ones((open_kernel_size, open_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if close_kernel_size and close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(bool)


def connected_components_from_mask(
    binary_mask: np.ndarray,
    min_area_px: int,
    max_regions: int | None = None,
) -> list[dict[str, Any]]:
    """Split a binary mask into connected regions."""
    mask_uint8 = binary_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    regions: list[dict[str, Any]] = []

    for component_id in range(1, num_labels):
        area_px = int(stats[component_id, cv2.CC_STAT_AREA])

        if area_px < min_area_px:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

        component_mask = labels == component_id

        regions.append(
            {
                "component_id": int(component_id),
                "area_px": area_px,
                "xyxy": [float(x), float(y), float(x + w - 1), float(y + h - 1)],
                "mask": component_mask,
            }
        )

    regions = sorted(regions, key=lambda r: r["area_px"], reverse=True)

    if max_regions is not None:
        regions = regions[:max_regions]

    regions = sorted(regions, key=lambda r: r["xyxy"][0])

    return regions


def overlay_binary_mask(
    image,
    binary_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay candidate mask on top of the original RGB image."""
    image_np = np.asarray(image.convert("RGB")).astype(np.float32)

    color = np.zeros_like(image_np)
    color[..., 1] = 255.0

    mask_bool = binary_mask.astype(bool)

    image_np[mask_bool] = (
        image_np[mask_bool] * (1.0 - alpha)
        + color[mask_bool] * alpha
    )

    return image_np.clip(0, 255).astype(np.uint8)


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/mask2former_internvl.yaml"),
        "--config",
        "-c",
        help="Path to Mask2Former + InternVL config YAML.",
    )
) -> None:
    """Run Mask2Former masks and enrich candidate walking regions with InternVL OSM tags."""
    cfg = read_yaml(config)

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    recursive = bool(cfg.get("recursive", False))
    limit = cfg.get("limit")
    device = resolve_device(str(cfg.get("device", "auto")))

    model_cfg = cfg["mask2former"]
    sidewalk_cfg = cfg.get("sidewalk", {})
    internvl_cfg = cfg["internvl"]
    output_cfg = cfg.get("outputs", {})

    sidewalk_label = str(sidewalk_cfg.get("label", "sidewalk"))
    min_area_fraction = float(sidewalk_cfg.get("min_area_fraction", 0.003))
    max_regions = sidewalk_cfg.get("max_regions", 3)

    if max_regions is not None:
        max_regions = int(max_regions)

    crop_padding_ratio = float(sidewalk_cfg.get("crop_padding_ratio", 0.20))
    use_masked_crop = bool(sidewalk_cfg.get("use_masked_crop", False))

    open_kernel_size = int(sidewalk_cfg.get("open_kernel_size", 0))
    close_kernel_size = int(sidewalk_cfg.get("close_kernel_size", 0))

    fallback_labels = sidewalk_cfg.get("fallback_labels", ["road"])
    if isinstance(fallback_labels, str):
        fallback_labels = [fallback_labels]

    fallback_only_if_no_sidewalk = bool(
        sidewalk_cfg.get("fallback_only_if_no_sidewalk", True)
    )
    fallback_min_y_fraction = float(sidewalk_cfg.get("fallback_min_y_fraction", 0.35))

    overlay_alpha = float(output_cfg.get("overlay_alpha", 0.45))

    local_dir = model_cfg.get("local_dir")
    name_or_path = model_cfg["name_or_path"]

    if local_dir is not None and Path(local_dir).exists():
        model_path = str(local_dir)
    else:
        model_path = name_or_path

    dataset = ImageDataset(input_dir=input_dir, recursive=recursive)
    items = list(dataset)

    if limit is not None:
        items = items[: int(limit)]

    if not items:
        typer.echo(f"No images found in: {input_dir}")
        raise typer.Exit(code=0)

    typer.echo(f"Loading Mask2Former: {model_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The following named arguments are not valid for `Mask2FormerImageProcessor.__init__`.*",
            category=UserWarning,
        )
        processor = AutoImageProcessor.from_pretrained(
            model_path,
            use_fast=False,
        )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    model.eval().to(device)

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    sidewalk_id = find_label_id(sidewalk_label, id2label)

    if sidewalk_id is None:
        available = ", ".join(id2label.values())
        raise ValueError(
            f"Could not find sidewalk label `{sidewalk_label}` in model labels.\n"
            f"Available labels: {available}"
        )

    tagger = InternVLTagger(
        model_path=internvl_cfg["model_path"],
        mmproj_path=internvl_cfg["mmproj_path"],
        n_ctx=int(internvl_cfg.get("n_ctx", 8192)),
        n_gpu_layers=int(internvl_cfg.get("n_gpu_layers", 0)),
        verbose=bool(internvl_cfg.get("verbose", False)),
        rules_path=internvl_cfg.get("rules_path"),
    )

    json_dir = output_dir / "json"
    raw_mask_dir = output_dir / "prediction"
    candidate_mask_dir = output_dir / "candidate_masks"
    overlay_dir = output_dir / "overlay"
    crops_dir = output_dir / "crops"

    for item in tqdm(items, desc="Mask2Former + InternVL"):
        image = item.load_rgb().convert("RGB")
        width, height = image.size

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        semantic_mask = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(height, width)],
        )[0]

        semantic_mask_np = semantic_mask.detach().cpu().numpy().astype(np.uint8)

        min_area_px = int(width * height * min_area_fraction)

        candidate_label = sidewalk_label
        candidate_class_id = int(sidewalk_id)
        candidate_type = "dedicated_sidewalk_candidate"

        candidate_mask_raw = semantic_mask_np == sidewalk_id
        candidate_mask = clean_binary_mask(
            candidate_mask_raw,
            open_kernel_size=open_kernel_size,
            close_kernel_size=close_kernel_size,
        )

        regions = connected_components_from_mask(
            candidate_mask,
            min_area_px=min_area_px,
            max_regions=max_regions,
        )

        if (not regions) and fallback_only_if_no_sidewalk:
            for fallback_label in fallback_labels:
                fallback_id = find_label_id(str(fallback_label), id2label)

                if fallback_id is None:
                    continue

                fallback_mask_raw = semantic_mask_np == fallback_id

                lower_roi = np.zeros_like(fallback_mask_raw, dtype=bool)
                y0 = int(height * fallback_min_y_fraction)
                lower_roi[y0:, :] = True
                fallback_mask_raw = fallback_mask_raw & lower_roi

                fallback_mask = clean_binary_mask(
                    fallback_mask_raw,
                    open_kernel_size=open_kernel_size,
                    close_kernel_size=close_kernel_size,
                )

                fallback_regions = connected_components_from_mask(
                    fallback_mask,
                    min_area_px=min_area_px,
                    max_regions=max_regions,
                )

                if fallback_regions:
                    candidate_label = str(fallback_label)
                    candidate_class_id = int(fallback_id)
                    candidate_type = "shared_surface_fallback"
                    candidate_mask = fallback_mask
                    regions = fallback_regions
                    break

        enriched_detections: list[dict[str, Any]] = []

        for idx, region in enumerate(regions):
            component_mask = region["mask"]

            detection = {
                "label": candidate_label,
                "confidence": None,
                "xyxy": region["xyxy"],
            }

            crop = crop_detection(
                image=image,
                detection=detection,
                mask=component_mask,
                padding_ratio=crop_padding_ratio,
                use_mask=use_masked_crop,
            )

            crop_path = ""
            if output_cfg.get("save_crops", True):
                crops_dir.mkdir(parents=True, exist_ok=True)
                label_safe = safe_name(candidate_label)
                crop_path = str(
                    crops_dir / f"{item.stem}_{label_safe}{idx:03d}.jpg"
                )
                crop.save(crop_path)

            osm_tags = tagger.predict_osm_tags(
                crop=crop,
                feature_label=candidate_label,
            )

            enriched_detections.append(
                {
                    "label": candidate_label,
                    "confidence": None,
                    "source": "mask2former_mapillary",
                    "candidate_type": candidate_type,
                    "class_id": int(candidate_class_id),
                    "component_id": int(region["component_id"]),
                    "area_px": int(region["area_px"]),
                    "area_fraction": float(region["area_px"] / (width * height)),
                    "xyxy": region["xyxy"],
                    "crop_path": crop_path,
                    "internvl_osm_tags": osm_tags,
                }
            )

        raw_mask_path = ""
        if output_cfg.get("save_raw_mask", True):
            raw_mask_path = str(raw_mask_dir / f"{item.stem}_semantic_mask.npy")
            save_masks(
                raw_mask_dir / f"{item.stem}_semantic_mask.npy",
                semantic_mask_np,
            )

        candidate_mask_path = ""
        if output_cfg.get("save_candidate_mask", True):
            candidate_mask_path = str(
                candidate_mask_dir / f"{item.stem}_{safe_name(candidate_label)}_mask.npy"
            )
            save_masks(
                candidate_mask_dir / f"{item.stem}_{safe_name(candidate_label)}_mask.npy",
                candidate_mask.astype(np.uint8),
            )

        overlay_path = ""
        if output_cfg.get("save_overlay", True):
            overlay_path = str(
                overlay_dir / f"{item.stem}_{safe_name(candidate_label)}_overlay.jpg"
            )
            save_rgb(
                overlay_dir / f"{item.stem}_{safe_name(candidate_label)}_overlay.jpg",
                overlay_binary_mask(
                    image=image,
                    binary_mask=candidate_mask,
                    alpha=overlay_alpha,
                ),
            )

        payload = {
            "image_id": item.stem,
            "image_path": str(item.path),
            "model": name_or_path,
            "model_path_used": model_path,
            "localizer": "mask2former_mapillary",
            "sidewalk_label": sidewalk_label,
            "sidewalk_class_id": int(sidewalk_id),
            "candidate_label": candidate_label,
            "candidate_class_id": int(candidate_class_id),
            "candidate_type": candidate_type,
            "fallback_only_if_no_sidewalk": fallback_only_if_no_sidewalk,
            "fallback_labels": fallback_labels,
            "fallback_min_y_fraction": fallback_min_y_fraction,
            "image_width": width,
            "image_height": height,
            "min_area_fraction": min_area_fraction,
            "min_area_px": min_area_px,
            "max_regions": max_regions,
            "crop_padding_ratio": crop_padding_ratio,
            "use_masked_crop": use_masked_crop,
            "open_kernel_size": open_kernel_size,
            "close_kernel_size": close_kernel_size,
            "num_detections": len(enriched_detections),
            "num_masks": len(enriched_detections),
            "raw_mask_path": raw_mask_path,
            "candidate_mask_path": candidate_mask_path,
            "overlay_path": overlay_path,
            "detections": enriched_detections,
        }

        if output_cfg.get("save_json", True):
            save_json(json_dir / f"{item.stem}.json", payload)

    typer.echo(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    app()