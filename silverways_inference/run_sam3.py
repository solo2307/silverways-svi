from __future__ import annotations

from pathlib import Path

import torch
import typer
from PIL import Image
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from silverways_inference.common import (
    hf_token,
    list_images,
    read_yaml,
    resolve_device,
    save_npy,
    write_json,
)

app = typer.Typer(help="Run SAM3 text-prompt segmentation.")
console = Console()


@app.command()
def infer(config: Path = Path("conf/models/sam3.yaml")) -> None:
    cfg = read_yaml(config)
    images = list_images(cfg["input_dir"], cfg.get("recursive", True), cfg.get("limit"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.get("device", "auto"))
    model_id = cfg["model"]["name_or_path"]
    prompt = cfg["prompt"]["text"]
    detection_threshold = cfg["prompt"].get("detection_threshold", 0.30)
    mask_threshold = cfg["prompt"].get("mask_threshold", 0.50)

    if not hf_token():
        console.print(
            "[yellow]HF_TOKEN is not set. facebook/sam3 is gated, so loading may fail with 403.[/yellow]"
        )

    console.print(f"Loading SAM3: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token())
    model = AutoModel.from_pretrained(model_id, token=hf_token()).to(device)
    model.eval()

    for image_path in tqdm(images, desc="SAM3"):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = [image.size[::-1]]

        masks_path = ""
        detections_payload = {}

        if hasattr(processor, "post_process_instance_segmentation"):
            processed = processor.post_process_instance_segmentation(
                outputs,
                threshold=detection_threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )[0]
            masks = processed.get("masks")
            if masks is not None and cfg["outputs"].get("save_masks", True):
                masks_path = str(output_dir / "masks" / f"{image_path.stem}.npy")
                save_npy(masks_path, masks.detach().cpu().numpy())

            detections_payload = {
                key: value.detach().cpu().tolist() if hasattr(value, "detach") else value
                for key, value in processed.items()
                if key != "masks"
            }

        elif hasattr(processor, "post_process_semantic_segmentation"):
            semantic = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_sizes,
                threshold=mask_threshold,
            )[0]
            masks_path = str(output_dir / "masks" / f"{image_path.stem}.npy")
            save_npy(masks_path, semantic.detach().cpu().numpy())
            detections_payload = {"semantic_mask_saved": True}

        else:
            detections_payload = {"warning": "No recognized SAM3 post-processing method found."}

        if cfg["outputs"].get("save_json", True):
            write_json(
                output_dir / "json" / f"{image_path.stem}.json",
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "prompt": prompt,
                    "mask_path": masks_path,
                    "detections": detections_payload,
                },
            )

    console.print(f"Done: {output_dir}")


if __name__ == "__main__":
    app()
