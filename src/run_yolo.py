from __future__ import annotations

from pathlib import Path

import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from tqdm import tqdm
from ultralytics import YOLO

from src.common import (
    append_csv,
    hf_token,
    list_images,
    read_yaml,
    write_json,
    yolo_device,
)

app = typer.Typer(help="Run YOLO predictions on SVI images.")
console = Console()


@app.command()
def infer(config: Path = Path("conf/models/yolo.yaml")) -> None:
    cfg = read_yaml(config)
    model_cfg = cfg["model"]
    weights = Path(model_cfg["weights"])

    if model_cfg.get("hf_repo_id") and not weights.exists():
        snapshot_download(
            repo_id=model_cfg["hf_repo_id"],
            repo_type="model",
            local_dir=weights.parent,
            allow_patterns=model_cfg.get("hf_patterns", ["*.pt"]),
            token=hf_token(),
        )

    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    images = list_images(cfg["input_dir"], cfg.get("recursive", True), cfg.get("limit"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    device = yolo_device(cfg.get("device", "auto"))

    for image_path in tqdm(images, desc="YOLO"):
        results = model.predict(
            source=str(image_path),
            conf=cfg["predict"].get("conf", 0.25),
            iou=cfg["predict"].get("iou", 0.7),
            imgsz=cfg["predict"].get("imgsz", 1280),
            device=device,
            verbose=False,
        )

        result = results[0]
        names = result.names
        detections = []

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
        if cfg["outputs"].get("save_json", True):
            json_path = str(output_dir / "json" / f"{image_path.stem}.json")
            write_json(json_path, {"image_path": str(image_path), "detections": detections})

        annotated_path = ""
        if cfg["outputs"].get("save_annotated", True):
            annotated_path = str(output_dir / "annotated" / f"{image_path.stem}.jpg")
            Path(annotated_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=annotated_path)

        append_csv(
            output_dir / "predictions.csv",
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
