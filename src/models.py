from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from src.tiling import Tile


@dataclass(frozen=True)
class ModelSpec:
    id: str
    type: str
    source: str
    repo_id: str | None = None
    filename: str | None = None
    local_path: str | None = None


class YoloModelRunner:
    def __init__(self, spec: ModelSpec, device: str | int = 0) -> None:
        self.spec = spec
        self.device = device
        self.model_path = self._resolve_model_path()
        self.model = YOLO(str(self.model_path))

    def _resolve_model_path(self) -> Path:
        if self.spec.source == "local":
            if not self.spec.local_path:
                raise ValueError(f"Model {self.spec.id} is missing local_path.")
            return Path(self.spec.local_path)

        if self.spec.source == "huggingface":
            if not self.spec.repo_id or not self.spec.filename:
                raise ValueError(f"Model {self.spec.id} is missing repo_id or filename.")

            path = hf_hub_download(
                repo_id=self.spec.repo_id,
                filename=self.spec.filename,
                repo_type="model",
                token=os.getenv("HF_TOKEN"),
            )
            return Path(path)

        raise ValueError(f"Unsupported model source: {self.spec.source}")

    def predict_tile(
        self,
        tile: Tile,
        conf: float,
        iou: float,
        imgsz: int,
        pano_width: int,
    ) -> list[dict[str, Any]]:
        arr = np.array(tile.image)

        results = self.model.predict(
            source=arr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        predictions: list[dict[str, Any]] = []

        if not results:
            return predictions

        result = results[0]
        names = result.names

        if result.boxes is None:
            return predictions

        boxes = result.boxes

        for idx in range(len(boxes)):
            xyxy = boxes.xyxy[idx].detach().cpu().numpy().tolist()
            cls_id = int(boxes.cls[idx].detach().cpu().item())
            score = float(boxes.conf[idx].detach().cpu().item())

            x1, y1, x2, y2 = xyxy

            global_x1 = (tile.x + x1) % pano_width
            global_x2 = (tile.x + x2) % pano_width
            global_y1 = tile.y + y1
            global_y2 = tile.y + y2

            predictions.append(
                {
                    "image": tile.image_name,
                    "tile_id": tile.tile_id,
                    "model_id": self.spec.id,
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": score,
                    "bbox_xyxy": [
                        float(global_x1),
                        float(global_y1),
                        float(global_x2),
                        float(global_y2),
                    ],
                    "tile_bbox_xyxy": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "tile_origin_xy": [tile.x, tile.y],
                }
            )

        return predictions