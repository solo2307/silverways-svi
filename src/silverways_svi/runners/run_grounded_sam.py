from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import typer
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import SAM

from silverways_svi.data.image_dataset import ImageDataset

app = typer.Typer(help="Run Grounding DINO + SAM2 segmentation.")


@app.callback()
def main() -> None:
    """Grounded-SAM runner."""


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def sam_device(device: torch.device) -> str | int:
    """Convert torch device to Ultralytics device format."""
    if device.type == "cuda":
        return 0 if device.index is None else device.index

    return device.type


def build_prompt(labels: list[str]) -> str:
    """Grounding DINO works well with dot-separated prompts."""
    cleaned = [label.strip().lower() for label in labels if label.strip()]
    return ". ".join(cleaned) + "."


class GroundingDinoDetector:
    """Grounding DINO text-prompt detector."""

    def __init__(
        self,
        name_or_path: str,
        local_dir: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)

        local_path = Path(local_dir) if local_dir is not None else None
        model_path = str(local_path) if local_path is not None and local_path.exists() else name_or_path

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        text_labels: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        image = image.convert("RGB")
        prompt = build_prompt(text_labels)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )

        result = results[0]

        detections: list[dict[str, Any]] = []

        for box, score, label in zip(
            result.get("boxes", []),
            result.get("scores", []),
            result.get("labels", []),
        ):
            detections.append(
                {
                    "label": str(label),
                    "confidence": float(score.detach().cpu().item()),
                    "xyxy": [float(x) for x in box.detach().cpu().tolist()],
                }
            )

        return detections


class SAM2Segmenter:
    """SAM2 segmentation from boxes."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "auto",
        imgsz: int = 1024,
    ) -> None:
        self.weights = Path(weights)

        if not self.weights.exists():
            raise FileNotFoundError(
                f"SAM2 weights not found: {self.weights}\n"
                "Run: python -m silverways_svi.download_models sam2"
            )

        self.device = sam_device(resolve_device(device))
        self.imgsz = imgsz
        self.model = SAM(str(self.weights))

    def predict_from_boxes(
        self,
        image_path: str | Path,
        boxes_xyxy: list[list[float]],
    ) -> Any | None:
        if not boxes_xyxy:
            return None

        results = self.model.predict(
            source=str(image_path),
            bboxes=boxes_xyxy,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results:
            return None

        return results[0]


def masks_to_numpy(result: Any | None) -> np.ndarray | None:
    if result is None:
        return None

    masks = getattr(result, "masks", None)

    if masks is None or masks.data is None:
        return None

    return masks.data.detach().cpu().numpy()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_masks(path: Path, masks: np.ndarray | None) -> None:
    if masks is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, masks)


def draw_boxes(image: Image.Image, detections: list[dict[str, Any]]) -> np.ndarray:
    image_np = np.asarray(image.convert("RGB")).copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        label = det["label"]
        confidence = det["confidence"]

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = f"{label} {confidence:.2f}"
        cv2.putText(
            image_np,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return image_np


def overlay_masks(
    image: Image.Image,
    masks: np.ndarray | None,
    alpha: float = 0.45,
) -> np.ndarray:
    image_np = np.asarray(image.convert("RGB")).astype(np.float32)

    if masks is None:
        return image_np.astype(np.uint8)

    overlay = image_np.copy()

    for idx, mask in enumerate(masks):
        mask_bool = mask.astype(bool)

        color = np.array(
            [
                (37 * idx + 255) % 255,
                (97 * idx + 80) % 255,
                (17 * idx + 40) % 255,
            ],
            dtype=np.float32,
        )

        overlay[mask_bool] = (
            image_np[mask_bool] * (1.0 - alpha)
            + color * alpha
        )

    return overlay.astype(np.uint8)


def save_annotated(
    path: Path,
    image: Image.Image,
    detections: list[dict[str, Any]],
    masks: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    masked = overlay_masks(image, masks)
    boxed = masked.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        label = det["label"]
        confidence = det["confidence"]

        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = f"{label} {confidence:.2f}"
        cv2.putText(
            boxed,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    Image.fromarray(boxed).save(path)


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/grounded_sam.yaml"),
        "--config",
        "-c",
        help="Path to Grounded-SAM config YAML.",
    )
) -> None:
    """Run Grounding DINO boxes + SAM2 segmentation masks."""
    cfg = read_yaml(config)

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    recursive = bool(cfg.get("recursive", False))
    limit = cfg.get("limit")
    device = str(cfg.get("device", "auto"))

    grounding_cfg = cfg["grounding_dino"]
    sam2_cfg = cfg["sam2"]
    prompt_cfg = cfg["prompt"]
    output_cfg = cfg.get("outputs", {})

    text_labels = prompt_cfg["text_labels"]
    box_threshold = float(prompt_cfg.get("box_threshold", 0.35))
    text_threshold = float(prompt_cfg.get("text_threshold", 0.25))

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

    detector = GroundingDinoDetector(
        name_or_path=grounding_cfg["name_or_path"],
        local_dir=grounding_cfg.get("local_dir"),
        device=device,
    )

    segmenter = SAM2Segmenter(
        weights=sam2_cfg["weights"],
        imgsz=int(sam2_cfg.get("imgsz", 1024)),
        device=device,
    )

    json_dir = output_dir / "json"
    masks_dir = output_dir / "masks"
    annotated_dir = output_dir / "annotated"

    for item in tqdm(items, desc="Grounded-SAM"):
        image = item.load_rgb()

        detections = detector.predict(
            image=image,
            text_labels=text_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        boxes_xyxy = [det["xyxy"] for det in detections]

        sam_result = segmenter.predict_from_boxes(
            image_path=item.path,
            boxes_xyxy=boxes_xyxy,
        )

        masks = masks_to_numpy(sam_result)

        payload = {
            "image_id": item.stem,
            "image_path": str(item.path),
            "text_labels": text_labels,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "num_detections": len(detections),
            "detections": detections,
            "num_masks": 0 if masks is None else int(masks.shape[0]),
        }

        if output_cfg.get("save_json", True):
            save_json(json_dir / f"{item.stem}.json", payload)

        if output_cfg.get("save_masks", True):
            save_masks(masks_dir / f"{item.stem}_masks.npy", masks)

        if output_cfg.get("save_annotated", True):
            save_annotated(
                annotated_dir / f"{item.stem}_grounded_sam.jpg",
                image=image,
                detections=detections,
                masks=masks,
            )

    typer.echo(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    app()