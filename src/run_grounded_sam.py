from __future__ import annotations

from pathlib import Path

import torch
import typer
from PIL import Image
from rich.console import Console
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from src.common import (
    hf_token,
    list_images,
    read_yaml,
    resolve_device,
    save_npy,
    write_json,
)

app = typer.Typer(help="Run Grounding DINO, optionally followed by SAM masks.")
console = Console()


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


@app.command()
def infer(config: Path = Path("conf/models/grounded_sam.yaml")) -> None:
    cfg = read_yaml(config)
    images = list_images(cfg["input_dir"], cfg.get("recursive", True), cfg.get("limit"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.get("device", "auto"))

    gdino_id = cfg["grounding_dino"]["model_id"]
    labels = cfg["grounding_dino"]["text_labels"]

    console.print(f"Loading Grounding DINO: {gdino_id}")
    gdino_processor = AutoProcessor.from_pretrained(gdino_id, token=hf_token())
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gdino_id, token=hf_token()
    ).to(device)
    gdino_model.eval()

    sam_enabled = cfg.get("sam", {}).get("enabled", False)
    sam_processor = None
    sam_model = None

    if sam_enabled:
        sam_id = cfg["sam"]["model_id"]
        console.print(f"Loading SAM: {sam_id}")
        sam_processor = SamProcessor.from_pretrained(sam_id, token=hf_token())
        sam_model = SamModel.from_pretrained(sam_id, token=hf_token()).to(device)
        sam_model.eval()

    for image_path in tqdm(images, desc="Grounded-SAM"):
        image = Image.open(image_path).convert("RGB")

        gdino_inputs = gdino_processor(
            images=image,
            text=[labels],
            return_tensors="pt",
        )
        gdino_inputs = to_device(gdino_inputs, device)

        with torch.inference_mode():
            gdino_outputs = gdino_model(**gdino_inputs)

        results = gdino_processor.post_process_grounded_object_detection(
            gdino_outputs,
            gdino_inputs.input_ids,
            threshold=cfg["grounding_dino"].get("box_threshold", 0.35),
            text_threshold=cfg["grounding_dino"].get("text_threshold", 0.25),
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results["boxes"].detach().cpu()
        scores = results["scores"].detach().cpu()
        detected_labels = results["labels"]

        detections = []
        for box, score, label in zip(boxes, scores, detected_labels):
            detections.append(
                {
                    "label": str(label),
                    "confidence": float(score),
                    "xyxy": [float(x) for x in box.tolist()],
                }
            )

        mask_path = ""
        if sam_enabled and sam_processor is not None and sam_model is not None and len(boxes) > 0:
            sam_inputs = sam_processor(
                image,
                input_boxes=[boxes.tolist()],
                return_tensors="pt",
            )
            sam_inputs = to_device(sam_inputs, device)

            with torch.inference_mode():
                sam_outputs = sam_model(**sam_inputs)

            masks = sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0]

            mask_path = str(output_dir / "masks" / f"{image_path.stem}.npy")
            save_npy(mask_path, masks.numpy())

        write_json(
            output_dir / "json" / f"{image_path.stem}.json",
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "text_labels": labels,
                "detections": detections,
                "mask_path": mask_path,
            },
        )

    console.print(f"Done: {output_dir}")


if __name__ == "__main__":
    app()
