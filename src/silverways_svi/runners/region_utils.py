from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_masks(path: Path, masks: np.ndarray | None) -> None:
    """Save mask array as .npy."""
    if masks is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, masks)


def save_rgb(path: Path, image_array: np.ndarray) -> None:
    """Save RGB numpy array as an image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array.astype(np.uint8)).save(path)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Get xyxy bounding box from a binary mask."""
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def safe_name(text: str) -> str:
    """Make a safe filename component."""
    cleaned = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in text
    ).strip("_")

    return cleaned or "object"


def crop_detection(
    image: Image.Image,
    detection: dict[str, Any],
    mask: np.ndarray | None = None,
    padding_ratio: float = 0.15,
    use_mask: bool = True,
) -> Image.Image:
    """
    Crop a detected or segmented region.

    If a mask is available, crop from the mask bounding box.
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