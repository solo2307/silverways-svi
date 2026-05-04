from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import typer
import yaml
from tqdm import tqdm

from silverways_svi.models.sam3 import (
    SAM3Segmenter,
    masks_to_numpy,
    result_to_json,
)

app = typer.Typer(help="Run SAM3 text-prompt segmentation on SVI images.")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


def list_images(input_dir: Path, recursive: bool = True, limit: int | None = None) -> list[Path]:
    globber = input_dir.rglob("*") if recursive else input_dir.glob("*")

    images = sorted(
        path
        for path in globber
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    return images[:limit] if limit is not None else images


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_masks(path: Path, masks: np.ndarray | None) -> None:
    if masks is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, masks)


def save_annotated(path: Path, result: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plotted = result.plot()

    # Ultralytics plot output is usually BGR-compatible for cv2.imwrite.
    cv2.imwrite(str(path), plotted)


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/sam3.yaml"),
        "--config",
        "-c",
        help="Path to SAM3 config YAML.",
    )
) -> None:
    cfg = read_yaml(config)

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    recursive = bool(cfg.get("recursive", True))
    limit = cfg.get("limit")

    model_cfg = cfg["model"]
    prompt_cfg = cfg["prompt"]
    output_cfg = cfg["outputs"]

    prompts = prompt_cfg["text"]
    if isinstance(prompts, str):
        prompts = [prompts]

    segmenter = SAM3Segmenter(
        weights=model_cfg["weights"],
        conf=float(prompt_cfg.get("conf", 0.25)),
        imgsz=int(prompt_cfg.get("imgsz", 1024)),
        half=bool(prompt_cfg.get("half", False)),
        device=str(cfg.get("device", "auto")),
        save=False,
    )

    images = list_images(
        input_dir=input_dir,
        recursive=recursive,
        limit=limit,
    )

    if not images:
        typer.echo(f"No images found in {input_dir}")
        raise typer.Exit(code=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images, desc="SAM3"):
        prediction = segmenter.predict(
            image_path=image_path,
            text_prompts=prompts,
        )

        stem = image_path.stem

        if output_cfg.get("save_json", True):
            save_json(
                output_dir / "json" / f"{stem}.json",
                result_to_json(prediction),
            )

        if output_cfg.get("save_masks", True):
            save_masks(
                output_dir / "masks" / f"{stem}.npy",
                masks_to_numpy(prediction),
            )

        if output_cfg.get("save_annotated", True):
            save_annotated(
                output_dir / "annotated" / f"{stem}.jpg",
                prediction.result,
            )

    typer.echo(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    app()