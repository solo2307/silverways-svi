import os
import logging
from pathlib import Path
import cv2
import numpy as np
from scipy.io import loadmat
from mit_semseg.utils import colorEncode
from research_code.dl.green_model_svi import SegModelPSPNet
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')
def show_overlay_row_with_indices(base_path: str, prediction_path: str, uuid: str, indices_csv: Path, alpha=0.6):
    """
    Show 4 directions (0, 90, 180, 270) of a panorama image with prediction overlay,
    and display sky, green, tree, bush, grass indices as titles.

    :param base_path: Path to folder with original images.
    :param prediction_path: Path to folder with predictions.
    :param uuid: Common UUID prefix of the images.
    :param indices_csv: CSV file path where indices are stored.
    :param alpha: Transparency for overlay.
    """
    import pandas as pd

    # Load indices CSV into DataFrame
    df = pd.read_csv(indices_csv)

    # Filter rows for this uuid prefix (starts with uuid)
    df_filtered = df[df['pano_id'].str.startswith(uuid)]

    # Prepare to plot 4 images
    angles = [0, 90, 180, 270]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, angle in enumerate(angles):
        img_file = os.path.join(base_path, f"{uuid}_{angle}.jpg")
        pred_file = os.path.join(prediction_path, f"{uuid}_{angle}.png")

        img = cv2.imread(img_file)
        pred = cv2.imread(pred_file)

        if img is None or pred is None:
            print(f"Missing image or prediction for angle {angle}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        if img_rgb.shape != pred_rgb.shape:
            pred_rgb = cv2.resize(pred_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Get indices for this specific image from CSV
        row_id = f"{uuid}_{angle}"
        row = df_filtered[df_filtered['pano_id'] == row_id]

        if not row.empty:
            sky_idx = row['sky_index'].values[0]
            green_idx = row['green_index'].values[0]
            tree_idx = row['tree_index'].values[0]
            bush_idx = row['bush_index'].values[0]
            grass_idx = row['grass_index'].values[0]

            title = (f"{angle}°\n"
                     f"Sky: {sky_idx:.3f}\n"
                     f"Green: {green_idx:.3f}\n"
                     f"Tree: {tree_idx:.3f}\n"
                     f"Bush: {bush_idx:.3f}\n"
                     f"Grass: {grass_idx:.3f}")
        else:
            title = f"{angle}°\nNo indices"

        axes[i].imshow(img_rgb)
        axes[i].imshow(pred_rgb, alpha=alpha)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
def show_overlay_row(base_path: str, prediction_path: str, uuid: str, alpha=0.6):
    """
    Show 4 directions (0, 90, 180, 270) of a panorama image with prediction overlay.

    :param base_path: Path to the folder containing original images.
    :param prediction_path: Path to the folder containing predictions.
    :param uuid: The common UUID prefix of the image (e.g., 'abc123').
    :param alpha: Transparency for overlay.
    """
    angles = [0, 90, 180, 270]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, angle in enumerate(angles):
        img_file = os.path.join(base_path, f"{uuid}_{angle}.jpg")
        pred_file = os.path.join(prediction_path, f"{uuid}_{angle}.png")

        img = cv2.imread(img_file)
        pred = cv2.imread(pred_file)

        if img is None or pred is None:
            print(f"Missing image or prediction for angle {angle}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        if img_rgb.shape != pred_rgb.shape:
            pred_rgb = cv2.resize(pred_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        axes[i].imshow(img_rgb)
        axes[i].imshow(pred_rgb, alpha=alpha)
        axes[i].set_title(f"{angle}°")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def calculate_indicators(prediction: np.ndarray):
    total_size = prediction.size
    tree_index = np.sum(prediction == 4) / total_size
    bush_index = np.sum(prediction == 17) / total_size
    grass_index = np.sum(prediction == 9) / total_size
    sky_index = np.sum(prediction == 2) / total_size

    # Optionally, if you want combined green vegetation index:
    green_index = (np.sum(prediction == 4) + np.sum(prediction == 9) + np.sum(prediction == 17)) / total_size

    return [sky_index, green_index, tree_index, bush_index, grass_index]

class DataSet:
    def __init__(self, input_dir: str, pred_dir: str,model):
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
        self.items = sorted([
            f for f in self.input_dir.glob("*.jpg")
            if not f.stem.startswith("CAoS")
        ])

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int):
        """Retrieve and process an image by index."""
        if index < 0 or index >= len(self.items):
            raise IndexError("Index out of range for dataset items.")
        return self.process_item(self.items[index])

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.items)

    def process_item(self, image_path: Path, save_pred: bool = True, gvi: bool = False) -> np.ndarray:
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
            color_map = loadmat(f"{self.model_path}/color150.mat")['colors']
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
                output_indices = ['pano_id','sky_index', 'green_index', 'tree_index', 'bush_index', 'grass_index']
                output_file = self.output_dir / "indices.csv"
                if not output_file.exists():
                    with open(output_file, 'w') as f:
                        f.write(f"{','.join(output_indices)}\n")
                        f.write(f"{image_path.stem},{','.join(map(str, indicators))}\n")
                else:
                    with open(output_file, 'a') as f:
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
            "bushes": ([204, 255, 4], [0, 255, 255])  # Cyan for bushes
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
            11: [0, 255, 255],  # Sidewalk - Yellow
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

        color_map = loadmat(f"{model_path}/color150.mat")['colors']

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
    model_path = Path('data/models')
    input_dir = 'cache/google-streetview'
    pred_dir = 'cache/google-prediction'

    # model = SegModelPSPNet(model_path)
    # dataset = DataSet(input_dir=input_dir, pred_dir=pred_dir, model=model)
    #
    # # Process all images
    # dataset.process_all(save_pred=True, gvi=False)

    #Example of overlay
    # Visualize with indices overlayed in titles
    show_overlay_row_with_indices(
        base_path=input_dir,
        prediction_path=pred_dir,
        uuid="M2a7yeIPC0hRc4IUvnygZA",
        indices_csv='cache/google-prediction/indices.csv',
        alpha=0.4
    )
    # show_overlay_row(
    #     base_path="cache/google-streetview",
    #     prediction_path="cache/google-prediction",
    #     uuid="j9ORkgM5xdMPClKke-6O5Q",
    #     alpha=0.4
    # )