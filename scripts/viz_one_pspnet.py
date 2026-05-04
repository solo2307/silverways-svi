from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

try:
    from mit_semseg.dataset import TestDataset
    from mit_semseg.models import ModelBuilder, SegmentationModule
except ImportError as exc:
    raise ImportError(
        "mit-semseg is missing. Install the Conda environment first."
    ) from exc


MODEL_FILES = [
    "encoder_epoch_50.pth",
    "decoder_epoch_50.pth",
    "color150.mat",
    "color150-labels.txt",
]

DEFAULT_OPTIONS = SimpleNamespace(
    fc_dim=2048,
    num_class=150,
    imgSizes=[300, 400, 500, 600],
    imgMaxSize=1000,
    padding_constant=8,
    segm_downsampling_rate=8,
)

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


def check_model_files(model_dir: Path) -> None:
    """Check that PSPNet model files already exist locally."""
    missing = [name for name in MODEL_FILES if not (model_dir / name).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing PSPNet model files in {model_dir}:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + "\n\nRun this first:\n"
            "  python -m silverways_svi.download_models pspnet"
        )


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # CPU is safest for the older mit-semseg stack.
    return torch.device("cpu")


class PSPNetPredictor:
    """PSPNet/ADE20K predictor using the same architecture as the old working code."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "auto",
        options: SimpleNamespace = DEFAULT_OPTIONS,
        encoder_name: str = "resnet101",
        decoder_name: str = "upernet",
    ) -> None:
        check_model_files(model_dir)

        self.model_dir = model_dir
        self.device = resolve_device(device)
        self.options = options
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.fc_dim = options.fc_dim
        self.num_class = options.num_class
        self.img_sizes = options.imgSizes

        encoder_path = self.model_dir / "encoder_epoch_50.pth"
        decoder_path = self.model_dir / "decoder_epoch_50.pth"

        net_encoder = ModelBuilder.build_encoder(
            arch=self.encoder_name,
            fc_dim=self.fc_dim,
            weights=str(encoder_path),
        )

        net_decoder = ModelBuilder.build_decoder(
            arch=self.decoder_name,
            fc_dim=self.fc_dim,
            num_class=self.num_class,
            weights=str(decoder_path),
            use_softmax=True,
        )

        criterion = torch.nn.NLLLoss(ignore_index=-1)
        self.model = SegmentationModule(net_encoder, net_decoder, criterion)
        self.model.eval()
        self.model.to(self.device)

        torch.set_grad_enabled(False)

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        dataset_test = TestDataset(
            [{"fpath_img": str(image_path)}],
            self.options,
            max_sample=-1,
        )

        batch_data = dataset_test[0]
        seg_size = (
            batch_data["img_ori"].shape[0],
            batch_data["img_ori"].shape[1],
        )

        img_resized_list = batch_data["img_data"]

        scores = torch.zeros(
            1,
            self.num_class,
            seg_size[0],
            seg_size[1],
            device=self.device,
        )

        for img in img_resized_list:
            feed_dict = batch_data.copy()
            feed_dict["img_data"] = img.to(self.device)

            # These are not needed by SegmentationModule forward pass.
            feed_dict.pop("img_ori", None)
            feed_dict.pop("info", None)

            pred_tmp = self.model(feed_dict, segSize=seg_size)
            scores += pred_tmp / len(self.img_sizes)

        _, pred = torch.max(scores, dim=1)

        return pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)


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

    check_model_files(model_dir)

    image = Image.open(image_path).convert("RGB")

    predictor = PSPNetPredictor(
        model_dir=model_dir,
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