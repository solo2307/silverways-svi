from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from tqdm import tqdm
from ultralytics import YOLO

from silverways_svi.common import (
    append_csv,
    list_images,
    read_yaml,
    write_json,
    yolo_device,
)

app = typer.Typer(help="Run YOLO predictions on SVI images.")
console = Console()


@app.callback()
def main() -> None:
    """YOLO runner."""


def get_target_class_ids(
    model_names: dict[int, str],
    target_classes: list[str] | None,
) -> list[int] | None:
    """Convert class names like ['bench'] into YOLO class IDs."""
    if not target_classes:
        return None

    normalized_targets = {name.strip().lower() for name in target_classes}

    class_ids = [
        class_id
        for class_id, class_name in model_names.items()
        if class_name.lower() in normalized_targets
    ]

    missing = normalized_targets - {
        class_name.lower()
        for class_id, class_name in model_names.items()
        if class_id in class_ids
    }

    if missing:
        available = ", ".join(model_names.values())
        raise ValueError(
            f"Target classes not found in YOLO model: {sorted(missing)}\n"
            f"Available classes: {available}"
        )

    return class_ids


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/yolo.yaml"),
        "--config",
        "-c",
        help="Path to YOLO config YAML.",
    )
) -> None:
    cfg = read_yaml(config)

    model_cfg = cfg["model"]
    predict_cfg = cfg.get("predict", {})
    output_cfg = cfg.get("outputs", {})

    weights = model_cfg["weights"]

    images = list_images(
        cfg["input_dir"],
        recursive=cfg.get("recursive", True),
        limit=cfg.get("limit"),
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    device = yolo_device(cfg.get("device", "auto"))

    target_classes = predict_cfg.get("target_classes")
    target_class_ids = get_target_class_ids(model.names, target_classes)

    if target_class_ids is not None:
        console.print(
            f"Detecting only: {target_classes} "
            f"(class IDs: {target_class_ids})"
        )

    predictions_csv = output_dir / "predictions.csv"
    if predictions_csv.exists():
        predictions_csv.unlink()

    for image_path in tqdm(images, desc="YOLO"):
        results = model.predict(
            source=str(image_path),
            conf=predict_cfg.get("conf", 0.25),
            iou=predict_cfg.get("iou", 0.70),
            imgsz=predict_cfg.get("imgsz", 1280),
            classes=target_class_ids,
            device=device,
            verbose=False,
        )

        result = results[0]
        names = result.names
        detections: list[dict[str, Any]] = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confs, classes):
                detections.append(
                    {
                        "class_id": int(cls),
                        "class_name": names[int(cls)],
                        "confidence": float(conf),
                        "xyxy": [float(x) for x in box],
                    }
                )

        json_path = ""
        if output_cfg.get("save_json", True):
            json_path = str(output_dir / "json" / f"{image_path.stem}.json")
            write_json(
                json_path,
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "target_classes": target_classes,
                    "detections": detections,
                },
            )

        annotated_path = ""
        if output_cfg.get("save_annotated", True):
            annotated_path = str(
                output_dir / "annotated" / f"{image_path.stem}.jpg"
            )
            Path(annotated_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=annotated_path)

        append_csv(
            predictions_csv,
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "num_detections": len(detections),
                "classes": "|".join(sorted({d["class_name"] for d in detections})),
                "json_path": json_path,
                "annotated_path": annotated_path,
            },
        )

    console.print(f"Done: {output_dir}")


if __name__ == "__main__":
    app()