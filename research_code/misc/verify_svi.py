import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
def check_images(directory, delete_corrupt=False):
    corrupt_images = []
    total = 0

    # Walk through all files in the directory
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="Checking images"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                total += 1
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify image without loading it completely
                except (UnidentifiedImageError, OSError) as e:
                    corrupt_images.append(file_path)
                    print(f"Corrupt: {file_path}")
                    if delete_corrupt:
                        os.remove(file_path)

    print(f"\nChecked {total} images.")
    print(f"Found {len(corrupt_images)} corrupt images.")

    return corrupt_images

def run_verification(cfg: DictConfig):
    # 🔧 Set your directory here
    image_directory = f"{cfg.storage.sds}/{cfg.datasets.streetview.images.dir}"
    corrupt = check_images(image_directory, delete_corrupt=False)

    # Save corrupt list to file (optional)
    if corrupt:
        with open("../../cache/corrupt_images.txt", "w") as f:
            for path in corrupt:
                f.write(path + "\n")
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg:DictConfig):
    return run_verification(cfg)

if __name__ == "__main__":
    hydra_main()