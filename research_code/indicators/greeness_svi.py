"""
        Green Index from SVI
        This module calculates the Green Index from the Street View Imagery (SVI)
        Green Index is a measure of the amount of green vegetation (including grass, tree and bush indices) in an image.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from research_code.dl.green_model_svi import SegModelPSPNet
import logging
from tqdm import tqdm
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)
class GreenIndex_SVI:
    def __init__(self, data_dir: Path):
        """Initialize the Green Index SVI calculator."""
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

        self.data_dir = data_dir
        self.model_path = data_dir / "green_index_model.pth"
        self.images = self._load_images()

    def _load_images(self):
        return self.data_dir.glob("*.jpg")

    def __getitem__(self, idx):
        image_path = list(self.data_dir.glob("*.jpg"))[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_path} does not exist.")
        return image_path

    def process_images(self, output_file: Path):
        model = SegModelPSPNet(model_path=self.model_path)
        for image in tqdm(self.images, total = len(self.images), desc='Processing images'):
            # Process prediction
            prediction = model.predict(test_image_path=image)
            green_indices = self.calculate_green_indices(prediction)
            # Save results to output file
            with open(output_file, 'a') as f:
                f.write(f"{image.name},{','.join(map(str, green_indices))}\n")

    def calculate_green_indices(self, prediction: np.ndarray):
        green_index = np.sum(prediction == 1) / prediction.size
        tree_index = np.sum(prediction == 2) / prediction.size
        bush_index = np.sum(prediction == 3) / prediction.size
        grass_index = np.sum(prediction == 4) / prediction.size
        return [green_index, tree_index, bush_index, grass_index]




