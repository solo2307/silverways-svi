"""
Run Mask2Former semantic segmentation using the Mapillary Vistas model.

This runner applies the Hugging Face model:

    facebook/mask2former-swin-large-mapillary-vistas-semantic

to a folder of Street View image crops or regular images.


Mapillary Vistas is a street-scene semantic segmentation dataset with labels
for road environments, including classes such as road, sidewalk, building,
vegetation, terrain, sky, person, car, bus, bicycle, traffic sign, and others.
Official dataset page:

    https://www.mapillary.com/dataset/vistas
    https://research.mapillary.com/publication/iccv17a

The runner saves:
    - a colorized semantic mask for visual inspection
    - an optional raw prediction mask as a NumPy array
    - a JSON file with all class fractions and present classes
    - a CSV file with selected important SVI indicators

This runner does not crop panoramas itself. If the input images are full
panoramas and you want directional crops, generate them first with:

    python scripts/generate_pano_crops.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from PIL import Image
from rich.console import Console
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from silverways_svi.common import (
    append_csv,
    class_fraction,
    hf_token,
    list_images,
    read_yaml,
    resolve_device,
    save_npy,
    save_rgb,
    simple_colorize,
    write_json,
)

app = typer.Typer(help="Run Mask2Former-MapillaryVistas semantic segmentation on SVI images.")
console = Console()


@app.callback()
def main() -> None:
    """Mask2Former runner."""


IMPORTANT_LABELS = [
    "sidewalk",
    "building",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "bench",
]


def normalize_label(label: str) -> str:
    return label.lower().strip().replace("_", " ").replace("-", " ")


def metric_name(label: str) -> str:
    clean = normalize_label(label)
    clean = "_".join(clean.split())
    return f"{clean}_index"


def find_label_id(label: str, id2label: dict[int, str]) -> int | None:
    """Find class ID by exact normalized label match."""
    wanted = normalize_label(label)

    for idx, class_label in id2label.items():
        if normalize_label(class_label) == wanted:
            return idx

    return None


def all_class_fractions(
    mask: np.ndarray,
    id2label: dict[int, str],
) -> dict[str, float]:
    """Return fractions for all classes the model knows about."""
    metrics: dict[str, float] = {}

    for class_id, label in sorted(id2label.items()):
        metrics[metric_name(label)] = class_fraction(mask, class_id)

    return metrics


def important_metrics(
    mask: np.ndarray,
    id2label: dict[int, str],
) -> dict[str, float]:
    """Return compact indicators for important SVI classes."""
    metrics: dict[str, float] = {}

    for label in IMPORTANT_LABELS:
        class_id = find_label_id(label, id2label)
        metrics[metric_name(label)] = (
            0.0 if class_id is None else class_fraction(mask, class_id)
        )

    vegetation = metrics.get("vegetation_index", 0.0)
    terrain = metrics.get("terrain_index", 0.0)

    metrics["green_index"] = vegetation + terrain

    return metrics


def present_classes(
    mask: np.ndarray,
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    """Return only classes present in the prediction."""
    items: list[dict[str, Any]] = []

    for class_id in sorted(np.unique(mask).tolist()):
        label = id2label.get(int(class_id), str(class_id))
        fraction = class_fraction(mask, int(class_id))

        items.append(
            {
                "class_id": int(class_id),
                "class_name": label,
                "fraction": fraction,
            }
        )

    return items


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/mask2former_mapillary.yaml"),
        "--config",
        "-c",
        help="Path to Mask2Former config YAML.",
    )
) -> None:
    cfg = read_yaml(config)

    images = list_images(
        cfg["input_dir"],
        recursive=cfg.get("recursive", True),
        limit=cfg.get("limit"),
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.get("device", "auto"))

    model_cfg = cfg["model"]
    output_cfg = cfg.get("outputs", {})

    local_dir = model_cfg.get("local_dir")
    name_or_path = model_cfg["name_or_path"]

    if local_dir is not None and Path(local_dir).exists():
        model_path = str(local_dir)
    else:
        model_path = name_or_path

    console.print(f"Loading Mask2Former: {model_path}")

    processor = AutoImageProcessor.from_pretrained(
        model_path,
        token=hf_token(),
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_path,
        token=hf_token(),
    )

    model.eval().to(device)

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    predictions_csv = output_dir / "predictions.csv"
    if predictions_csv.exists():
        predictions_csv.unlink()

    for image_path in tqdm(images, desc="Mask2Former"):
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        mask = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
        )[0]

        mask = mask.detach().cpu().numpy().astype(np.uint8)

        compact_metrics = important_metrics(mask, id2label)
        full_metrics = all_class_fractions(mask, id2label)
        classes_present = present_classes(mask, id2label)

        mask_path = ""
        if output_cfg.get("save_color_mask", True):
            mask_path = str(output_dir / "masks" / f"{image_path.stem}.png")
            save_rgb(mask_path, simple_colorize(mask))

        raw_path = ""
        if output_cfg.get("save_raw_mask", False):
            raw_path = str(output_dir / "prediction" / f"{image_path.stem}.npy")
            save_npy(raw_path, mask)

        json_path = ""
        if output_cfg.get("save_json", True):
            json_path = str(output_dir / "json" / f"{image_path.stem}.json")
            write_json(
                json_path,
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "model": name_or_path,
                    "important_metrics": compact_metrics,
                    "present_classes": classes_present,
                    "all_class_fractions": full_metrics,
                },
            )

        append_csv(
            predictions_csv,
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                **compact_metrics,
                "mask_path": mask_path,
                "raw_mask_path": raw_path,
                "json_path": json_path,
            },
        )

    console.print(f"Done: {output_dir}")


if __name__ == "__main__":
    app()