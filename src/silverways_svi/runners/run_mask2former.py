from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from PIL import Image
from rich.console import Console
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from silverways_svi.common import (
    append_csv,
    class_fraction,
    hf_token,
    list_images,
    read_yaml,
    resolve_device,
    save_npy,
    save_rgb,
    simple_colorize,
)

app = typer.Typer(help="Run Mask2Former Cityscapes semantic segmentation.")
console = Console()


def cityscapes_metrics(mask: np.ndarray, id2label: dict[int, str]) -> dict[str, float]:
    label2id = {label.lower(): idx for idx, label in id2label.items()}

    def frac(label: str) -> float:
        idx = label2id.get(label)
        return 0.0 if idx is None else class_fraction(mask, idx)

    vegetation = frac("vegetation")
    terrain = frac("terrain")
    return {
        "road_index": frac("road"),
        "sidewalk_index": frac("sidewalk"),
        "building_index": frac("building"),
        "wall_index": frac("wall"),
        "fence_index": frac("fence"),
        "vegetation_index": vegetation,
        "terrain_index": terrain,
        "green_index": vegetation + terrain,
        "sky_index": frac("sky"),
        "person_index": frac("person"),
        "car_index": frac("car"),
        "bus_index": frac("bus"),
        "bicycle_index": frac("bicycle"),
    }


@app.command()
def infer(config: Path = Path("conf/models/mask2former_mapillary.yaml")) -> None:
    cfg = read_yaml(config)
    images = list_images(cfg["input_dir"], cfg.get("recursive", True), cfg.get("limit"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.get("device", "auto"))
    model_id = cfg["model"]["name_or_path"]

    console.print(f"Loading Mask2Former: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token())
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, token=hf_token())
    model.eval().to(device)

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    for image_path in tqdm(images, desc="Mask2Former"):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        mask = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].cpu().numpy().astype(np.uint8)

        metrics = cityscapes_metrics(mask, id2label)

        mask_path = ""
        if cfg["outputs"].get("save_color_mask", True):
            mask_path = str(output_dir / "masks" / f"{image_path.stem}.png")
            save_rgb(mask_path, simple_colorize(mask))

        raw_path = ""
        if cfg["outputs"].get("save_raw_mask", False):
            raw_path = str(output_dir / "raw_masks" / f"{image_path.stem}.npy")
            save_npy(raw_path, mask)

        append_csv(
            output_dir / "predictions.csv",
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                **metrics,
                "mask_path": mask_path,
                "raw_mask_path": raw_path,
            },
        )

    console.print(f"Done: {output_dir}")


if __name__ == "__main__":
    app()
