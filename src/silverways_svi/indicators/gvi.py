"""
        Green Index from SVI
        This module calculates the Green Index from the Street View Imagery (SVI)
        Green Index is a measure of the amount of green vegetation (including grass, tree and bush indices) in an image.
"""

from pathlib import Path
import numpy as np
from src.silverways_svi.models.pspnet import PSPNetSegmenter
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def check_image_size(image):
    from PIL import Image

    with Image.open(image) as img:
        width, height = img.size


class GreenIndex_SVI:
    def __init__(self, data_dir: Path):
        """Initialize the Green Index SVI calculator."""
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

        self.data_dir = data_dir
        # self.model_path = data_dir / "green_index_model.pth"
        self.images = self._load_images()

    def _load_images(self):
        return self.data_dir.glob("*.jpg")

    def __getitem__(self, idx):
        image_path = list(self.data_dir.glob("*.jpg"))[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_path} does not exist.")
        return image_path

    def process_images(self, output_file: Path):
        model = PSPNetSegmenter(model_path=Path("models/pspnet"))
        for image in tqdm(self.images, desc="Processing images"):
            # Process prediction
            prediction = model.predict(test_image_path=image)
            green_indices = self.calculate_green_indices(prediction)
            bldg_index, sky_index = self.calculate_enclosure(prediction)
            green_indices.extend([bldg_index, sky_index])
            # Save results to output file
            with open(output_file, "a") as f:
                f.write(f"{image.name},{','.join(map(str, green_indices))}\n")

    def calculate_green_indices(self, prediction: np.ndarray):
        total_size = prediction.size
        tree_index = np.sum(prediction == 4) / total_size
        bush_index = np.sum(prediction == 17) / total_size
        grass_index = np.sum(prediction == 9) / total_size
        green_index = (
            np.sum(prediction == 4) + np.sum(prediction == 9) + np.sum(prediction == 17)
        ) / total_size

        return [green_index, tree_index, bush_index, grass_index]

    def calculate_enclosure(self, prediction: np.ndarray):
        total_size = prediction.size
        if total_size == 0:
            return 0.0
        buildings = float(np.sum(prediction == 1)) / float(total_size)
        sky = float(np.sum(prediction == 2)) / float(total_size)
        return [buildings, sky]


if __name__ == "__main__":
    data_dir = Path("cache/cyclomedia")
    output_file = Path("cache/cyclomedia_indices.csv")

    green_index_svi = GreenIndex_SVI(data_dir)
    green_index_svi.process_images(output_file)

    logger.info(f"Green indices saved to {output_file}")
