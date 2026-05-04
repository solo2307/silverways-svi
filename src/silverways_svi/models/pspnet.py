"""
PSPNet/ADE20K semantic segmentation model wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = SimpleNamespace(
    fc_dim=2048,
    num_class=150,
    imgSizes=[300, 400, 500, 600],
    imgMaxSize=1000,
    padding_constant=8,
    segm_downsampling_rate=8,
)

def make_options(
    fc_dim: int = 2048,
    num_class: int = 150,
    img_sizes: list[int] | None = None,
    img_max_size: int = 1000,
    padding_constant: int = 8,
    segm_downsampling_rate: int = 8,
) -> SimpleNamespace:
    """Create PSPNet/TestDataset options from config values."""
    return SimpleNamespace(
        fc_dim=fc_dim,
        num_class=num_class,
        imgSizes=img_sizes or [300, 400, 500, 600],
        imgMaxSize=img_max_size,
        padding_constant=padding_constant,
        segm_downsampling_rate=segm_downsampling_rate,
    )

def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class PSPNetSegmenter:
    """PSPNet semantic segmentation model trained on ADE20K."""

    def __init__(
        self,
        model_path: str | Path,
        options: SimpleNamespace = DEFAULT_OPTIONS,
        encoder_name: str = "resnet101",
        decoder_name: str = "upernet",
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.encoder_file = self.model_path / "encoder_epoch_50.pth"
        self.decoder_file = self.model_path / "decoder_epoch_50.pth"

        self.options = options
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.fc_dim = options.fc_dim
        self.num_class = options.num_class
        self.img_sizes = options.imgSizes
        self.device = resolve_device(device)

        self._validate_files()
        self.segmentation_module = self._load_model()

    def _validate_files(self) -> None:
        if not self.encoder_file.exists():
            raise FileNotFoundError(
                f"Encoder weights not found: {self.encoder_file}\n"
                "Run: python -m silverways_svi.download_models pspnet"
            )

        if not self.decoder_file.exists():
            raise FileNotFoundError(
                f"Decoder weights not found: {self.decoder_file}\n"
                "Run: python -m silverways_svi.download_models pspnet"
            )

    def _load_model(self) -> SegmentationModule:
        logger.info("Loading PSPNet encoder and decoder...")

        builder = ModelBuilder()

        net_encoder = builder.build_encoder(
            arch=self.encoder_name,
            weights=str(self.encoder_file),
            fc_dim=self.fc_dim,
        )

        net_decoder = builder.build_decoder(
            arch=self.decoder_name,
            weights=str(self.decoder_file),
            fc_dim=self.fc_dim,
            num_class=self.num_class,
            use_softmax=True,
        )

        segmentation_module = SegmentationModule(
            net_encoder,
            net_decoder,
            torch.nn.NLLLoss(ignore_index=-1),
        )

        segmentation_module.eval()
        segmentation_module.to(self.device)

        logger.info("PSPNet loaded successfully on %s.", self.device)
        return segmentation_module

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> np.ndarray:
        """Predict a semantic segmentation mask for one image."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        dataset_test = TestDataset(
            [{"fpath_img": str(image_path)}],
            self.options,
            max_sample=-1,
        )

        batch_data = dataset_test[0]
        seg_size = (
            batch_data["img_ori"].shape[0],
            batch_data["img_ori"].shape[1],
        )

        img_resized_list = batch_data["img_data"]

        scores = torch.zeros(
            1,
            self.num_class,
            seg_size[0],
            seg_size[1],
            device=self.device,
        )

        for img in img_resized_list:
            feed_dict = batch_data.copy()
            feed_dict["img_data"] = img.to(self.device)
            feed_dict.pop("img_ori", None)
            feed_dict.pop("info", None)

            pred_tmp = self.segmentation_module(feed_dict, segSize=seg_size)
            scores += pred_tmp / len(self.img_sizes)

        _, pred = torch.max(scores, dim=1)

        return pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)