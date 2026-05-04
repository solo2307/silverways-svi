from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from huggingface_hub import snapshot_download
from PIL import Image
from rich.console import Console
from torchvision import transforms
from tqdm import tqdm

from src.common import (
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

try:
    from mit_semseg.models import ModelBuilder, SegmentationModule
except ImportError as exc:
    raise ImportError("Install mit-semseg first. Use environment-cpu.yaml or environment-gpu.yaml.") from exc

app = typer.Typer(help="Run PSPNet semantic segmentation on SVI images.")
console = Console()

ADE20K = {
    "building": 1,
    "sky": 2,
    "tree": 4,
    "grass": 9,
    "bush": 17,
}
WATER_IDS = [21, 26, 60, 113, 128]


def vegetation_metrics(mask: np.ndarray) -> dict[str, float]:
    tree = class_fraction(mask, ADE20K["tree"])
    grass = class_fraction(mask, ADE20K["grass"])
    bush = class_fraction(mask, ADE20K["bush"])
    return {
        "sky_index": class_fraction(mask, ADE20K["sky"]),
        "tree_index": tree,
        "grass_index": grass,
        "bush_index": bush,
        "green_index": tree + grass + bush,
        "building_index": class_fraction(mask, ADE20K["building"]),
        "water_index": class_fraction(mask, WATER_IDS),
    }


class PSPNet:
    def __init__(self, cfg: dict, device: str) -> None:
        self.device = resolve_device(device)
        model_cfg = cfg["model"]

        snapshot_download(
            repo_id=model_cfg["hf_repo_id"],
            repo_type="model",
            local_dir=model_cfg["local_dir"],
            allow_patterns=model_cfg["files"],
            token=hf_token(),
        )

        model_dir = Path(model_cfg["local_dir"])
        encoder = ModelBuilder.build_encoder(
            arch=model_cfg["arch_encoder"],
            fc_dim=model_cfg["fc_dim"],
            weights=str(model_dir / "encoder_epoch_50.pth"),
        )
        decoder = ModelBuilder.build_decoder(
            arch=model_cfg["arch_decoder"],
            fc_dim=model_cfg["fc_dim"],
            num_class=model_cfg["num_class"],
            weights=str(model_dir / "decoder_epoch_50.pth"),
            use_softmax=True,
        )
        self.model = SegmentationModule(encoder, decoder, torch.nn.NLLLoss(ignore_index=-1))
        self.model.eval().to(self.device)

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.inference_mode()
    def predict(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        img_data = self.preprocess(image).unsqueeze(0).to(self.device)
        scores = self.model({"img_data": img_data}, segSize=(height, width))
        return torch.argmax(scores, dim=1)[0].cpu().numpy().astype(np.uint8)


@app.command()
def infer(config: Path = Path("conf/models/pspnet.yaml")) -> None:
    cfg = read_yaml(config)
    images = list_images(cfg["input_dir"], cfg.get("recursive", True), cfg.get("limit"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Running PSPNet on {len(images)} images")
    runner = PSPNet(cfg, cfg.get("device", "auto"))

    for image_path in tqdm(images):
        mask = runner.predict(image_path)
        metrics = vegetation_metrics(mask)

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
