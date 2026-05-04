from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml
from PIL import Image
from tqdm import tqdm

from silverways_svi.data.image_dataset import ImageDataset
from silverways_svi.models.pspnet import PSPNetSegmenter

app = typer.Typer(help="Run PSPNet semantic segmentation on image folders.")


CLASS_IDS = {
    "building": 1,
    "sky": 2,
    "tree": 4,
    "grass": 9,
    "bush": 17,
}

WATER_IDS = [21, 26, 60, 113, 128]

VIZ_COLORS = {
    "tree": (255, 40, 20),
    "grass": (0, 255, 30),
    "bush": (0, 220, 220),
    "building": (150, 150, 150),
    "sky": (135, 206, 235),
    "water": (0, 80, 255),
}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


def class_fraction(mask: np.ndarray, class_ids: int | list[int]) -> float:
    if mask.size == 0:
        return 0.0

    if isinstance(class_ids, int):
        return float((mask == class_ids).sum() / mask.size)

    return float(np.isin(mask, class_ids).sum() / mask.size)


def compute_metrics(mask: np.ndarray) -> dict[str, float]:
    tree = class_fraction(mask, CLASS_IDS["tree"])
    grass = class_fraction(mask, CLASS_IDS["grass"])
    bush = class_fraction(mask, CLASS_IDS["bush"])

    return {
        "sky_index": class_fraction(mask, CLASS_IDS["sky"]),
        "tree_index": tree,
        "grass_index": grass,
        "bush_index": bush,
        "green_index": tree + grass + bush,
        "building_index": class_fraction(mask, CLASS_IDS["building"]),
        "water_index": class_fraction(mask, WATER_IDS),
    }


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)

    color[mask == CLASS_IDS["tree"]] = VIZ_COLORS["tree"]
    color[mask == CLASS_IDS["grass"]] = VIZ_COLORS["grass"]
    color[mask == CLASS_IDS["bush"]] = VIZ_COLORS["bush"]
    color[mask == CLASS_IDS["building"]] = VIZ_COLORS["building"]
    color[mask == CLASS_IDS["sky"]] = VIZ_COLORS["sky"]

    for water_id in WATER_IDS:
        color[mask == water_id] = VIZ_COLORS["water"]

    return color


def make_overlay(
    image: Image.Image,
    color_mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    image_np = np.asarray(image.convert("RGB")).astype(np.float32)
    color_np = color_mask.astype(np.float32)

    foreground = color_mask.sum(axis=-1) > 0

    overlay = image_np.copy()
    overlay[foreground] = (
        image_np[foreground] * (1.0 - alpha)
        + color_np[foreground] * alpha
    )

    return overlay.astype(np.uint8)


def save_visualization(
    image: Image.Image,
    overlay: np.ndarray,
    color_mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(7, 13))

    axes[0].imshow(image)
    axes[0].set_title(f"{title} - original")

    axes[1].imshow(overlay)
    axes[1].set_title("PSPNet overlay")

    axes[2].imshow(color_mask)
    axes[2].set_title("PSPNet selected-class mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/pspnet.yaml"),
        "--config",
        "-c",
        help="Path to PSPNet config YAML.",
    )
) -> None:
    """Run PSPNet on a folder of images."""
    cfg = read_yaml(config)

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    recursive = bool(cfg.get("recursive", False))
    limit = cfg.get("limit")
    device = str(cfg.get("device", "auto"))

    model_cfg = cfg["model"]
    output_cfg = cfg.get("outputs", {})

    dataset = ImageDataset(
        input_dir=input_dir,
        recursive=recursive,
    )

    items = list(dataset)
    if limit is not None:
        items = items[: int(limit)]

    if not items:
        typer.echo(f"No images found in: {input_dir}")
        raise typer.Exit(code=0)

    segmenter = PSPNetSegmenter(
        model_path=model_cfg["local_dir"],
        encoder_name=model_cfg.get("encoder", "resnet101"),
        decoder_name=model_cfg.get("decoder", "upernet"),
        device=device,
    )

    masks_dir = output_dir / "masks"
    prediction_dir = output_dir / "prediction"
    viz_dir = output_dir / "visualizations"
    metrics_csv = output_dir / "indices.csv"

    if metrics_csv.exists():
        metrics_csv.unlink()

    for item in tqdm(items, desc="PSPNet"):
        image = item.load_rgb()
        mask = segmenter.predict(item.path)
        color_mask = colorize_mask(mask)
        metrics = compute_metrics(mask)

        if output_cfg.get("save_color_mask", True):
            masks_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(color_mask).save(masks_dir / f"{item.stem}_mask.png")

        if output_cfg.get("save_raw_prediction", True):
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.save(prediction_dir / f"{item.stem}_raw_mask.npy", mask)

        if output_cfg.get("save_visualization", True):
            overlay = make_overlay(image, color_mask)
            save_visualization(
                image=image,
                overlay=overlay,
                color_mask=color_mask,
                output_path=viz_dir / f"{item.stem}_pspnet_viz.png",
                title=item.name,
            )

        if output_cfg.get("save_metrics_csv", True):
            append_metrics_csv(
                metrics_csv,
                {
                    "image_id": item.stem,
                    "image_path": str(item.path),
                    **metrics,
                },
            )

    typer.echo(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    app()