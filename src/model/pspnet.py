"""
pre-trained PSPNet model on ADE20K dataset for semantic segmentation.
"""

import logging

from pathlib import Path
from types import SimpleNamespace
import torch
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.dataset import TestDataset
import matplotlib.pylab as plt


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# Global constants
plt.rcParams["axes.grid"] = False

DEFAULT_OPTIONS = SimpleNamespace(
    fc_dim=2048,
    num_class=150,
    imgSizes=[300, 400, 500, 600],
    imgMaxSize=1000,
    padding_constant=8,
    segm_downsampling_rate=8,
)


class SegModelPSPNet:
    """
    Initializes the SegModelPSPNet class by setting paths and loading the models.

    :param model_path: Path to the models directory.
    :param encoder_name: Name of the encoder architecture.
    :param decoder_name: Name of the decoder architecture.
    :param fc_dim: Dimension of the fully connected layer.
    :param num_class: Number of classes for segmentation.
    :param img_sizes: List of image sizes for multiscale testing.
    """

    def __init__(
        self,
        model_path: Path,
        options: SimpleNamespace = DEFAULT_OPTIONS,
        encoder_name: str = "resnet101",
        decoder_name: str = "upernet",
    ):
        self.model_path = Path(model_path)
        self.encoder_file = self.model_path / "encoder_epoch_50.pth"
        self.decoder_file = self.model_path / "decoder_epoch_50.pth"
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.fc_dim = options.fc_dim
        self.num_class = options.num_class
        self.img_sizes = options.imgSizes

        self._validate_files()
        self.segmentation_module = self._load_model()

    def _validate_files(self):
        """Checks if the models weight files exist before loading."""
        if not self.encoder_file.exists():
            raise FileNotFoundError(f"Encoder weights not found: {self.encoder_file}")
        if not self.decoder_file.exists():
            raise FileNotFoundError(f"Decoder weights not found: {self.decoder_file}")

    def _load_model(self):
        """
        Loads the encoder and decoder models and creates the segmentation module.

        :return: Initialized segmentation module in evaluation mode.
        """
        logger.info("Loading encoder and decoder models...")

        builder = ModelBuilder()
        net_encoder = builder.build_encoder(
            arch=self.encoder_name, weights=str(self.encoder_file), fc_dim=self.fc_dim
        )
        net_decoder = builder.build_decoder(
            arch=self.decoder_name,
            weights=str(self.decoder_file),
            fc_dim=self.fc_dim,
            num_class=self.num_class,
            use_softmax=True,
        )

        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, torch.nn.NLLLoss(ignore_index=-1)
        )
        segmentation_module.eval()
        torch.set_grad_enabled(False)

        logger.info("Model loaded successfully!")
        return segmentation_module

    def predict(self, test_image_path: str):
        """
        Predicts the segmentation map for the given test image.

        :param test_image_path: Path to the test image.
        :return: Predicted segmentation map as a numpy array.
        """
        if not Path(test_image_path).exists():
            raise FileNotFoundError(f"Test image not found: {test_image_path}")

        logger.info(f"Processing image: {test_image_path}")

        dataset_test = TestDataset(
            [{"fpath_img": test_image_path}], DEFAULT_OPTIONS, max_sample=-1
        )
        batch_data = dataset_test[0]
        seg_size = (batch_data["img_ori"].shape[0], batch_data["img_ori"].shape[1])
        img_resized_list = batch_data["img_data"]

        scores = torch.zeros(1, self.num_class, seg_size[0], seg_size[1])

        with torch.no_grad():
            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict["img_data"] = img
                del feed_dict["img_ori"]
                del feed_dict["info"]

                pred_tmp = self.segmentation_module(feed_dict, segSize=seg_size)
                scores += pred_tmp / len(self.img_sizes)

        _, pred = torch.max(scores, dim=1)

        return pred.squeeze(0).cpu().numpy()
