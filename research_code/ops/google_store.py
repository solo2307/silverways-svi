import os
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import csv
import hydra
from omegaconf import DictConfig,OmegaConf
from typing import Iterable, Optional, Set, Tuple

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

# ---------- Helpers ----------
def _select(cfg: DictConfig, dotted: str, default=None):
    try:
        return OmegaConf.select(cfg, dotted, default=default)
    except Exception:
        return default
def _load_manifest_existing_pairs(manifest_csv: Path) -> Set[Tuple[str, int]]:
    """
    Read already-logged (pano_id, heading) from manifest to avoid re-downloading.
    """
    pairs: Set[Tuple[str, int]] = set()
    if not manifest_csv.exists() or manifest_csv.stat().st_size == 0:
        return pairs
    try:
        usecols = ["pano_id", "heading"]
        for r in pd.read_csv(manifest_csv, usecols=usecols).itertuples(index=False):
            pid = str(getattr(r, "pano_id"))
            hd = int(getattr(r, "heading"))
            pairs.add((pid, hd))
    except Exception:
        pass
    return pairs
# ---------- Class Definition ----------
class GoogleStaticStore:
    def __init__(self,
                 api_key:str,
                 cache_folder:str ="cache/google-streetview",
                 *,
                 size: str="640X640",
                 fov: int=90,
                 pitch: int = 0
        ):
        if not api_key:
            raise ValueError("Google API key must be provided.")
        self.api_key = api_key
        self.size = size
        self.fov = int(fov)
        self.pitch = int(pitch)
        self.image_cache_path = Path(cache_folder)
        # self.image_cache_path.mkdir(parents=True, exist_ok=True)

    def fetch_image_with_pano(self, pano_id:str, heading:Optional[int]=0):
        """
        Download one SV image by pano_id (+ heading). Returns (path, saved_flag).
        saved_flag = True if newly downloaded, False if already existed, None on error.
        """
        image_name = f"{pano_id}_{heading}.jpg"
        image_path = os.path.join(self.image_cache_path, image_name)

        if os.path.exists(image_path):
            # logger.info(f"[SKIP] {image_name} already exists.")
            return image_path, False

        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "key": self.api_key,
            "pano": pano_id,
            "size": self.size,
            "fov" : self.fov,
            "pitch": self.pitch,
        }
        if heading is not None:
            params["heading"] = heading

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(response.content)
            logger.info(f"[SAVED] {image_name}")
            return image_path, True
        except requests.RequestException as e:
            logger.error(f"[ERROR] pano_id {pano_id}, heading {heading}: {e}")
            return None, None
        finally:
            response.close()

# ---------- External CSV Processor Function ----------
def process_pano_csv(csv_path:Path,
                     api_key:str,
                     output_dir:Path,
                     *,
                     headings:Iterable[int]=[0, 90, 180, 270],
                     pitch:int=0,
                     fov:int=90,
                     size:str="640x640",
                     output_metadata_csv:Optional[str]=None,
                     max_requests:Optional[int]=None) -> int:
    """
       Read pano_ids from metadata CSV and download images.
       - Skips duplicates and invalid rows
       - Respects QPS
       - Resumes based on existing files and manifest_csv
       Returns count of successful downloads (new or existing).
    """
    # Read CSV file and keep only unique panoramic ids
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'pano_id' not in df.columns:
        raise ValueError("CSV must contain a 'pano_id' column.")
    if "status" in df.columns:
        df = df[df["status"] == "OK"]
    df = df.dropna(subset=['pano_id']).drop_duplicates(subset='pano_id')
    logger.info("Unique pano_ids to consider: %d", len(df))

    # Pairs already logged in manifest
    seen_pairs = _load_manifest_existing_pairs(output_metadata_csv) if output_metadata_csv else set()
    # Prepare manifest for appends
    write_header = output_metadata_csv and (not output_metadata_csv.exists() or output_metadata_csv.stat().st_size == 0)
    manifest_f = None
    manifest_writer = None
    if output_metadata_csv:
        output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = output_metadata_csv.open("a", newline="", encoding="utf-8")
        fieldnames = ["pano_id", "heading", "lat", "lon", "image_path"]
        manifest_writer = csv.DictWriter(manifest_f, fieldnames=fieldnames)
        if write_header:
            manifest_writer.writeheader()
    # Send request to download images from GoogleStaticStore
    store = GoogleStaticStore(api_key=api_key,cache_folder=output_dir,
                              fov=fov, pitch=pitch, size=size)
    # metadata = []
    saved_count = 0
    request_count = 0
    if max_requests is None:
        max_requests = len(df) * 4
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing panoramas"):
            pano_id = row['pano_id']
            lat = row.get('lat', None)
            lon = row.get('lon', None)
            # if pano_id.startswith('CAoS'):# logger.warning(f"[SKIP] pano_id {pano_id} is not a valid Google Street View ID.")
            #     continue

            for heading in headings:
                pair = (pano_id, int(heading))
                if manifest_writer and (pano_id,int(heading)) in seen_pairs:
                    saved_count += 1
                    continue
                image_path, saved = store.fetch_image_with_pano(pano_id, heading=int(heading))
                if manifest_writer and image_path is not None:
                    request_count += 1
                    saved_count += 1
                    manifest_writer.writerow(
                        {
                            "pano_id": pano_id,
                            "heading": heading,
                            "lat": lat,
                            "lon": lon,
                            "image_path": image_path
                        }
                    )
                    seen_pairs.add(pair)

                # if saved is not None and saved is True:
                #     request_count += 1
                #     metadata.append({
                #         "pano_id": pano_id,
                #         "heading": heading,
                #         "lat": lat,
                #         "lon": lon,
                #         "image_path": image_path
                #     })
            # if request_count >= max_requests: break
            if max_requests is not None and request_count >= max_requests:
                logger.info("Reached max_requests=%d; stopping.", max_requests)
                return request_count
        # if output_metadata_csv:
        #     pd.DataFrame(metadata).to_csv(output_metadata_csv, index=False)
        #     logger.info(f"[METADATA SAVED] {output_metadata_csv}")
    finally:
        if manifest_f:
            manifest_f.close()
    logger.info("Images present/downloaded: %d/%d → %s", saved_count,request_count, output_dir)
    return request_count

def run_pipeline(cfg: DictConfig) -> int:
    try:
        api_key = cfg.gcp.service_key
        if not api_key:
            logger.error("Missing Google API key (cfg.gcp.service_key).")
            return 1
        # Inputs/outputs from your config
        sds_dir = Path(cfg.storage.sds)

        output_img_dir = sds_dir / _select(cfg, "datasets.streetview.svi_dir", "google-streetview")  # 'cache/google-streetview'
        panorama_metadata_csv = Path(cfg.storage.cache) / cfg.datasets.streetview.output_panorama_metadata #'cache/mannheim_google_panorama_metadata.csv'
        output_metadata_csv = Path(cfg.datasets.streetview.images.manifest)
        max_requests = _select(cfg, "datasets.streetview.images.max_requests", None)

        headings = _select(cfg, "datasets.streetview.images.headings", [0, 90, 180, 270])
        size = _select(cfg, "datasets.streetview.images.size", "640x640")
        fov = int(_select(cfg, "datasets.streetview.images.fov", 90))
        pitch = int(_select(cfg, "datasets.streetview.images.pitch", 0))

        request_num = process_pano_csv(
            csv_path=panorama_metadata_csv,
            api_key=api_key,
            output_metadata_csv=output_metadata_csv,
            headings=headings,
            fov=fov,
            pitch=pitch,
            size=size,
            output_dir=output_img_dir,
            max_requests=max_requests
        )
        logger.info("✅ Pipeline finished - Total requests: %d", request_num)
        return 0
    except Exception:
        logger.exception("❌ Pipeline failed")
        return 1

# Optional Hydra entrypoint (kept for CLI use; safe for imports)
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg: DictConfig):
    return run_pipeline(cfg)

if __name__ == "__main__":
    hydra_main()
