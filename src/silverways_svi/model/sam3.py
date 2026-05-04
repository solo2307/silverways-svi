from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor


@dataclass
class SAM3Prediction:
    image_path: Path
    prompts: list[str]
    result: Any


class SAM3Segmenter:
    """SAM3 text-prompt segmentation wrapper using Ultralytics."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        imgsz: int = 1024,
        half: bool = False,
        device: str = "auto",
        save: bool = False,
    ) -> None:
        self.weights = Path(weights)

        if not self.weights.exists():
            raise FileNotFoundError(
                f"SAM3 weights not found: {self.weights}\n"
                "Download sam3.pt manually after Hugging Face access is approved "
                "and place it at the path configured in conf/models/sam3.yaml."
            )

        device_value = None if device == "auto" else device

        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": str(self.weights),
            "imgsz": imgsz,
            "half": half,
            "save": save,
        }

        if device_value is not None:
            overrides["device"] = device_value

        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def predict(
        self,
        image_path: str | Path,
        text_prompts: list[str],
    ) -> SAM3Prediction:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.predictor.set_image(str(image_path))
        results = self.predictor(text=text_prompts)

        # Ultralytics usually returns a list of Results.
        result = results[0] if isinstance(results, list) else results

        return SAM3Prediction(
            image_path=image_path,
            prompts=text_prompts,
            result=result,
        )


def result_to_json(prediction: SAM3Prediction) -> dict[str, Any]:
    """Convert Ultralytics SAM3 result into JSON-safe metadata."""
    result = prediction.result

    payload: dict[str, Any] = {
        "image_name": prediction.image_path.name,
        "image_path": str(prediction.image_path),
        "prompts": prediction.prompts,
        "detections": [],
    }

    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}

    if boxes is None:
        return payload

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

    for i, box in enumerate(xyxy):
        class_id = int(cls[i]) if len(cls) > i else None
        class_name = names.get(class_id, str(class_id)) if class_id is not None else None
        confidence = float(conf[i]) if len(conf) > i else None

        payload["detections"].append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "xyxy": [float(x) for x in box],
            }
        )

    return payload


def masks_to_numpy(prediction: SAM3Prediction) -> np.ndarray | None:
    """Return SAM3 masks as a NumPy array if masks exist."""
    masks = getattr(prediction.result, "masks", None)

    if masks is None or masks.data is None:
        return None

    return masks.data.cpu().numpy()