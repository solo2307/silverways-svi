import os
import logging
import requests
import pandas as pd
from pathlib import Path
import googlemaps
from datetime import datetime
from tqdm import tqdm

# ---------- Global Logger Setup ----------
def setup_logger():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    log_file = Path(f"logs/google-static-store-{timestamp}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("GoogleStaticStore")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger

logger = setup_logger()

# ---------- Class Definition ----------
class GoogleStaticStore:
    def __init__(self, api_key, cache_folder ="cache/google-streetview"):
        self.api_key = api_key
        self.gmaps = googlemaps.Client(key=api_key)
        self.image_cache_path = Path(cache_folder)
        self.image_cache_path.mkdir(parents=True, exist_ok=True)

    def fetch_image_with_pano(self, pano_id, heading=0, pitch=0, size="640x640"):
        image_name = f"{pano_id}_{heading}.jpg"
        image_path = os.path.join(self.image_cache_path, image_name)

        if os.path.exists(image_path):
            logger.info(f"[SKIP] {image_name} already exists.")
            return image_path

        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "key": self.api_key,
            "pano": pano_id,
            "size": size,
            "heading": heading,
            "fov" : 90,
            "pitch": pitch,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(response.content)
            logger.info(f"[SAVED] {image_name}")
            return image_path
        except requests.RequestException as e:
            logger.error(f"[ERROR] pano_id {pano_id}, heading {heading}: {e}")
            return None
        finally:
            response.close()
# ---------- External CSV Processor Function ----------
def process_pano_csv(csv_path, api_key, output_dir, headings=[0, 90, 180, 270],
                     pitch=0, output_metadata_csv=None, max_requests=10000):
    # Read CSV file and keep only unique panoramic ids
    df = pd.read_csv(csv_path)
    if 'pano_id' not in df.columns:
        raise ValueError("CSV must contain a 'pano_id' column.")
    df = df.dropna(subset=['pano_id'])
    df = df.drop_duplicates(subset='pano_id')

    store = GoogleStaticStore(api_key=api_key, cache_folder=output_dir)
    metadata = []
    request_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing panoramas"):
        pano_id = row['pano_id']
        lat = row.get('lat', None)
        lon = row.get('lon', None)

        for heading in headings:
            image_path = store.fetch_image_with_pano(pano_id, heading=heading, pitch=pitch)
            if image_path:
                request_count += 4
                metadata.append({
                    "pano_id": pano_id,
                    "heading": heading,
                    "lat": lat,
                    "lon": lon,
                    "image_path": image_path
                })
        if request_count >= max_requests:
            break  # Stop outer loop too

    if output_metadata_csv:
        pd.DataFrame(metadata).to_csv(output_metadata_csv, index=False)
        logger.info(f"[METADATA SAVED] {output_metadata_csv}")

if __name__ == "__main__":
    CSV_PATH = 'cache/mannheim_gvi_samples.csv'# 'cache/mannheim_google_panorama_metadata.csv'
    API_KEY = 'YOUR_GOOGLE_API_KEY'  # Replace with your actual Google API key
    OUTPUT_IMAGE_DIR = '/Volumes/sd17f001/ygrin/silverways/mannheim/google-streetview'#'cache/google-streetview'
    process_pano_csv(
        csv_path=CSV_PATH,
        api_key=API_KEY,
        headings=[0, 90, 180, 270],
        output_dir=OUTPUT_IMAGE_DIR,
        output_metadata_csv="cache/downloaded_images.csv",
        max_requests=9000
    )
