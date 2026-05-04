#!/usr/bin/env python3
"""
Mask2Former inference on Mapillary Vistas (semantic segmentation).

- Loads "facebook/mask2former-swin-large-mapillary-vistas-semantic"
- Predicts a semantic map for one or more images
- Saves per image:
    1) <stem>_overlay.png          (image + segmentation legend)
    2) <stem>_labels_raw.png       (grayscale IDs; 8-bit if <=255 else 16-bit)
    3) <stem>_labels_color.png     (colorized labels)
    4) <stem>_labels.npy           (HxW int32 label IDs)
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# -------------------- Logging --------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# -------------------- Utilities --------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def deterministic_color(cid: int) -> Tuple[int, int, int]:
    """Deterministic RGB for class ID via golden-ratio HSV hop."""
    h = (cid * 0.618033988749895) % 1.0
    s, v = 0.65, 1.0
    rgb = (hsv_to_rgb([[h, s, v]])[0] * 255).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def load_palette_csv(
    csv_path: Path, id2label: Dict[int, str]
) -> Dict[int, Tuple[int, int, int]]:
    """
    Load official palette CSV. Expected columns: id,label,r,g,b
    Missing IDs fall back to deterministic colors.
    """
    by_id: Dict[int, Tuple[int, int, int]] = {}
    with Path(csv_path).open("r", newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["id"])
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            by_id[i] = (r, g, b)
    return {i: by_id.get(i, deterministic_color(i)) for i in id2label.keys()}


def colorize_labels(
    label_map: np.ndarray, palette: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """HxW int -> HxWx3 RGB using provided palette."""
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in np.unique(label_map):
        cid = int(cid)
        out[label_map == cid] = palette.get(cid, deterministic_color(cid))
    return out


# -------------------- Core Inference --------------------
class Mask2FormerInferencer:
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic",
        device: Optional[torch.device] = None,
        palette_csv: Optional[Path] = None,
        load_retries: int = 2,
        log_level: str = "INFO",
    ):
        setup_logging(log_level)
        self.model_name = model_name
        self.device = device or pick_device()
        self.palette_csv = palette_csv
        self.load_retries = max(1, load_retries)

        self.processor: Optional[AutoImageProcessor] = None
        self.model: Optional[Mask2FormerForUniversalSegmentation] = None
        self.id2label: Dict[int, str] = {}
        self.num_classes: int = 0
        self.palette: Dict[int, Tuple[int, int, int]] = {}

    def load(self) -> None:
        last_err = None
        for attempt in range(1, self.load_retries + 1):
            try:
                logging.info(
                    f"Loading '{self.model_name}' to device={self.device} "
                    f"(attempt {attempt}/{self.load_retries})"
                )
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    self.model_name
                )
                self.model.to(self.device).eval()
                torch.set_grad_enabled(False)

                cfg = self.model.config
                self.id2label = {
                    int(k): v for k, v in getattr(cfg, "id2label", {}).items()
                }
                self.num_classes = getattr(cfg, "num_labels", None) or len(
                    self.id2label
                )
                if not self.num_classes:
                    raise RuntimeError(
                        "Could not determine number of classes from model config."
                    )

                if self.palette_csv and Path(self.palette_csv).exists():
                    self.palette = load_palette_csv(
                        Path(self.palette_csv), self.id2label
                    )
                    src = "official CSV"
                else:
                    self.palette = {
                        i: deterministic_color(i) for i in self.id2label.keys()
                    }
                    src = "deterministic"
                logging.info(
                    f"Model ready. num_classes={self.num_classes}, palette={src}"
                )
                return
            except Exception as e:
                last_err = e
                logging.exception("Model load failed.")
                time.sleep(1.0)
        raise RuntimeError(
            f"Failed to load model after {self.load_retries} attempts: {last_err}"
        )

    @torch.no_grad()
    def predict_labels(self, image: Image.Image) -> np.ndarray:
        """Return HxW np.int32 label map at original image size."""
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        return seg_map.cpu().numpy().astype(np.int32)

    def save_results(
        self,
        image_path: Path,
        out_dir: Path,
        alpha: float = 0.4,
        with_legend: bool = True,
        legend_max: int = 40,
    ) -> Tuple[Path, Path, Path, Path]:
        """
        Run inference on one image path, save:
          - overlay PNG
          - raw label IDs PNG (8-bit if <=255 else 16-bit)
          - colorized label PNG
          - label IDs NPY
        Returns paths in the order above.
        """
        ensure_dir(out_dir)
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        image = Image.open(img_path).convert("RGB")
        t0 = time.time()
        labels = self.predict_labels(image)
        dt = time.time() - t0
        logging.info(f"Inference: {img_path.name} in {dt:.3f}s | shape={labels.shape}")

        stem = img_path.stem

        # Save raw IDs as .npy
        npy_path = out_dir / f"{stem}_labels.npy"
        np.save(npy_path, labels)

        # Save raw ID PNG (8/16 bit depending on max)
        if labels.max() <= 255:
            raw_img = Image.fromarray(labels.astype(np.uint8), mode="L")
        else:
            raw_img = Image.fromarray(labels.astype(np.uint16), mode="I;16")
        raw_png_path = out_dir / f"{stem}_labels_raw.png"
        raw_img.save(raw_png_path)

        # Save color PNG
        color_map = colorize_labels(labels, self.palette)
        color_png_path = out_dir / f"{stem}_labels_color.png"
        Image.fromarray(color_map).save(color_png_path)

        # Save overlay + legend
        overlay = (
            (1 - alpha) * np.asarray(image).astype(np.float32)
            + alpha * color_map.astype(np.float32)
        ).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(stem)

        if with_legend:
            present = np.unique(labels)
            if present.size > legend_max:
                present = present[:legend_max]
                logging.info("Legend truncated to %d classes.", legend_max)
            patches = []
            for cid in present:
                cid = int(cid)
                name = self.id2label.get(cid, f"class {cid}")
                r, g, b = self.palette.get(cid, deterministic_color(cid))
                patches.append(
                    mpatches.Patch(color=(r / 255, g / 255, b / 255), label=name)
                )
            if patches:
                ax.legend(
                    handles=patches,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                )

        overlay_path = out_dir / f"{stem}_overlay.png"
        plt.tight_layout()
        plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        logging.info(
            f"Saved: {overlay_path.name}, {raw_png_path.name}, {color_png_path.name}, {npy_path.name}"
        )
        return overlay_path, raw_png_path, color_png_path, npy_path

    def run_batch(
        self,
        images: Iterable[Path],
        out_dir: Path,
        alpha: float = 0.4,
        with_legend: bool = True,
        legend_max: int = 40,
    ) -> None:
        ensure_dir(out_dir)
        for p in images:
            try:
                self.save_results(
                    image_path=Path(p),
                    out_dir=out_dir,
                    alpha=alpha,
                    with_legend=with_legend,
                    legend_max=legend_max,
                )
            except Exception:
                logging.exception(f"Failed on image: {p}")


# -------------------- Example usage --------------------
if __name__ == "__main__":
    infer = Mask2FormerInferencer(
        model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
        device=None,  # auto-pick
        palette_csv=None,  # Path("data/mapillary_palette.csv") if you have the official palette
        load_retries=2,
        log_level="INFO",
    )
    infer.load()

    images = [
        Path(""),
        # add more paths if needed
    ]
    out_dir = Path("cache/predict_mask2former")
    infer.run_batch(images, out_dir, alpha=0.4, with_legend=True, legend_max=40)
