from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import typer
import yaml
from tqdm import tqdm

from silverways_svi.data.image_dataset import ImageDataset
from silverways_svi.models.sam2 import SAM2Segmenter, masks_to_numpy, result_to_json

app = typer.Typer(help="Run SAM2 on SVI images.")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


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
    cv2.imwrite(str(path), plotted)


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/sam2.yaml"),
        "--config",
        "-c",
        help="Path to SAM2 config YAML.",
    )
) -> None:
    """Run SAM2 on a folder of SVI images."""
    cfg = read_yaml(config)

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    recursive = bool(cfg.get("recursive", True))
    limit = cfg.get("limit")

    model_cfg = cfg["model"]
    predict_cfg = cfg.get("predict", {})
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

    segmenter = SAM2Segmenter(
        weights=model_cfg["weights"],
        device=str(cfg.get("device", "auto")),
        imgsz=int(predict_cfg.get("imgsz", 1024)),
    )

    points = predict_cfg.get("points")
    labels = predict_cfg.get("labels")
    bboxes = predict_cfg.get("bboxes")

    if points is None and bboxes is None:
        typer.echo(
            "Warning: SAM2 usually works best with point or box prompts. "
            "Running without prompts may produce no useful masks depending on the model behavior."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(items, desc="SAM2"):
        prediction = segmenter.predict(
            image_path=item.path,
            points=points,
            labels=labels,
            bboxes=bboxes,
        )

        stem = item.stem

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