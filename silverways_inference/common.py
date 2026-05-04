from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def read_yaml(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def list_images(input_dir: str | Path, recursive: bool = True, limit: int | None = None) -> list[Path]:
    input_dir = Path(input_dir)
    globber: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = sorted(p for p in globber if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return images[:limit] if limit is not None else images


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device(device: str = "auto") -> str | int:
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        return 0 if resolved.index is None else resolved.index
    return resolved.type


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv(path: str | Path, row: dict[str, object]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)


def save_rgb(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def simple_colorize(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.int64)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        rgb[mask == class_id] = [
            (37 * int(class_id) + 29) % 255,
            (17 * int(class_id) + 71) % 255,
            (97 * int(class_id) + 13) % 255,
        ]
    return rgb


def class_fraction(mask: np.ndarray, class_ids: int | list[int]) -> float:
    if mask.size == 0:
        return 0.0
    if isinstance(class_ids, int):
        return float((mask == class_ids).sum() / mask.size)
    return float(np.isin(mask, class_ids).sum() / mask.size)
