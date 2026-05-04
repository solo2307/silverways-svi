"""
This script to generate semantic segmentation predictions for a dataset of images
"""


import logging
from pathlib import Path
import cv2
import numpy as np
from scipy.io import loadmat
from mit_semseg.utils import colorEncode


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_indicators(prediction: np.ndarray):
    total_size = prediction.size
    tree_index = np.sum(prediction == 4) / total_size
    bush_index = np.sum(prediction == 17) / total_size
    grass_index = np.sum(prediction == 9) / total_size
    sky_index = np.sum(prediction == 2) / total_size

    # Optionally, if you want combined green vegetation index:
    green_index = (
        np.sum(prediction == 4) + np.sum(prediction == 9) + np.sum(prediction == 17)
    ) / total_size

    return [sky_index, green_index, tree_index, bush_index, grass_index]


class SVIdataset:
    def __init__(self, input_dir: str, pred_dir: str, model):
        """
        Initializes the dataset processor for semantic segmentation.

        :param input_dir: Directory containing input images.
        :param pred_dir: Directory where predictions will be saved.
        :param model: Segmentation model instance.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(pred_dir)
        self.model = model
        self.model_path = model.model_path
        self.predictions = self.output_dir / "indices.csv"
        if self.predictions.exists():
            self.predictions.unlink()
        # Generate list of valid image files
        self._generate_file_list()

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_file_list(self):
        allowed_exts = {".jpg", ".jpeg", ".png"}

        files = [
            f
            for f in self.input_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in allowed_exts
            and not f.stem.startswith("CAoS")
        ]
        self.items = files
        # sorted([f for f in self.input_dir.glob("*.jpg") if not f.stem.startswith("CAoS")])

    def __getitem__(self, index: int):
        """Retrieve and process an image by index."""
        if index < 0 or index >= len(self.items):
            raise IndexError("Index out of range for dataset items.")
        return self.process_item(self.items[index])

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.items)

    def process_item(
        self, image_path: Path, save_pred: bool = True, gvi: bool = False
    ) -> np.ndarray:
        """
        Process a single image, run segmentation, and save the output.

        :param image_path: Path to the image file.
        :param save_pred: Whether to save the predicted image.
        :param gvi: Whether to process green vegetation index.
        :return: Processed prediction image.
        """
        try:
            # Load and validate the image
            logger.info(f"🔍 Processing: {image_path.name}...")

            # Run segmentation model
            pred = self.model.predict(str(image_path))
            indicators = calculate_indicators(pred)
            # Convert prediction to color
            color_map = loadmat(f"{self.model_path}/color150.mat")["colors"]
            pred_color = colorEncode(pred, color_map)

            # Ensure correct format
            pred_color = pred_color.astype(np.uint8)

            if gvi:
                pred_color = self.process_green(pred_color)
            else:
                pred_color = self.highlight_selected_classes(pred)
                # pred_color = self.highlight_all_classes(pred, self.model_path)
            if save_pred:
                # Save indicators results to output file
                output_indices = [
                    "pano_id",
                    "sky_index",
                    "green_index",
                    "tree_index",
                    "bush_index",
                    "grass_index",
                ]
                output_file = self.output_dir / "indices.csv"
                if not output_file.exists():
                    with open(output_file, "w") as f:
                        f.write(f"{','.join(output_indices)}\n")
                        f.write(f"{image_path.stem},{','.join(map(str, indicators))}\n")
                else:
                    with open(output_file, "a") as f:
                        f.write(f"{image_path.stem},{','.join(map(str, indicators))}\n")

                output_path = self.output_dir / f"{image_path.stem}.png"
                success = cv2.imwrite(str(output_path), pred_color)

                if success:
                    logger.info(f"✅ Saved: {output_path}")
                else:
                    logger.error(f"❌ Failed to save: {output_path}")

            return pred_color

        except Exception as e:
            logger.error(f"❌ Error processing {image_path.name}: {e}")
            return np.array([])

    @staticmethod
    def process_green(image: np.ndarray) -> np.ndarray:
        """
        Identifies and highlights green vegetation in the image.

        :param image: Input segmented image.
        :return: Image with green vegetation highlighted.
        """
        # Define color values for trees, grass, and bushes
        vegetation_colors = {
            "tree": ([4, 200, 3], [255, 0, 0]),  # Red for trees
            "grass": ([4, 250, 7], [0, 255, 0]),  # Green for grass
            "bushes": ([204, 255, 4], [0, 255, 255]),  # Cyan for bushes
        }

        combined_image = np.zeros_like(image)

        for label, (color, patch_color) in vegetation_colors.items():
            mask = np.all(image == color, axis=-1)
            combined_image[mask] = patch_color

        return combined_image

    @staticmethod
    def highlight_selected_classes(pred: np.ndarray) -> np.ndarray:
        """
        Highlight selected semantic classes with predefined colors.

        :param pred: 2D label array from segmentation model.
        :return: RGB image with selected classes highlighted.
        """
        output = np.zeros((*pred.shape, 3), dtype=np.uint8)

        class_colors = {
            4: [0, 0, 255],  # Tree - Red (in BGR)
            9: [0, 255, 0],  # Grass - Green
            17: [255, 255, 0],  # Bushes (plant) - Cyan
            # 11: [0, 255, 255],  # Sidewalk - Yellow
            1: [128, 128, 128],  # Building - Gray
            2: [235, 206, 135],  # Sky - Light Blue (BGR of RGB(135,206,235))
        }

        water_classes = [21, 26, 60, 113, 128]  # All water-related → Blue
        for class_id in water_classes:
            output[pred == class_id] = [0, 0, 255]  # Water - Blue

        for class_id, color in class_colors.items():
            output[pred == class_id] = color

        return output

    @staticmethod
    def highlight_all_classes(pred: np.ndarray, model_path: Path) -> np.ndarray:
        """
        Convert all class labels into their respective colors using the full color150.mat colormap.

        :param pred: 2D label array from segmentation model.
        :param model_path: Path to the model directory containing color150.mat.
        :return: RGB image with all classes colored.
        """
        from scipy.io import loadmat
        from mit_semseg.utils import colorEncode

        color_map = loadmat(f"{model_path}/color150.mat")["colors"]

        if color_map.max() <= 1.0:
            color_map = (color_map * 255).astype(np.uint8)
        else:
            color_map = color_map.astype(np.uint8)

        return colorEncode(pred, color_map)

    def process_all(self, save_pred: bool = True, gvi: bool = False):
        """
        Process all images in the dataset.

        :param save_pred: Whether to save predictions.
        :param gvi: Whether to process green vegetation index.
        """
        logger.info(f"🚀 Starting batch processing for {len(self)} images...")
        for i, image_path in enumerate(self.items):
            self.process_item(image_path, save_pred=save_pred, gvi=gvi)
            logger.info(f"📸 Processed {i + 1}/{len(self)}")

        logger.info("✅ Batch processing complete.")


if __name__ == "__main__":
    # Example of Usage

    # from research_code.dl.mask2former import SegModelPSPNet
    model_path = Path("data/models")
    input_dir = "cache/streetview"
    pred_dir = "cache/prediction"

    # Example of overlay
    # Visualize with indices overlayed in titles
    from research_code.dl.utils import (
        show_single_row_triptych,
    )


    show_single_row_triptych(
        base_path="cache/google-streetview",
        prediction_path="cache/google-prediction",
        uuid="M_IEQajKEKXXeIk7iw69Ag",
        angle=180,
        alpha=0.4,
        save_path=None,
    )
