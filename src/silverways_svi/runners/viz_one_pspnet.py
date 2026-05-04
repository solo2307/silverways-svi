from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from silverways_svi.models.pspnet import PSPNetSegmenter


# ADE20K class ids used by this PSPNet/ADE20K setup.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PSPNet on one SVI image and save a visualization."
    )
    parser.add_argument("--image", required=True, help="Path to one input image.")
    parser.add_argument(
        "--output-dir",
        default="outputs/debug_pspnet",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/pspnet_svi_veg",
        help="Directory for PSPNet model files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay alpha.",
    )
    parser.add_argument(
        "--encoder",
        default="resnet101",
        help="Encoder architecture. Default: resnet101.",
    )
    parser.add_argument(
        "--decoder",
        default="upernet",
        help="Decoder architecture. Default: upernet.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    predictor = PSPNetSegmenter(
        model_path=model_dir,
        device=args.device,
        encoder_name=args.encoder,
        decoder_name=args.decoder,
    )

    mask = predictor.predict(image_path)
    color_mask = colorize_mask(mask)
    overlay = make_overlay(image, color_mask, alpha=args.alpha)
    metrics = compute_metrics(mask)

    output_dir.mkdir(parents=True, exist_ok=True)

    viz_path = output_dir / f"{image_path.stem}_pspnet_viz.png"
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    raw_mask_path = output_dir / f"{image_path.stem}_raw_mask.npy"
    metrics_path = output_dir / f"{image_path.stem}_metrics.json"

    save_visualization(
        image=image,
        overlay=overlay,
        color_mask=color_mask,
        output_path=viz_path,
        title=image_path.name,
    )

    Image.fromarray(color_mask).save(mask_path)
    np.save(raw_mask_path, mask)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved visualization: {viz_path}")
    print(f"Saved color mask:    {mask_path}")
    print(f"Saved raw mask:      {raw_mask_path}")
    print(f"Saved metrics:       {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()