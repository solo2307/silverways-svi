"""
Street View end‑to‑end job


Steps
1) Retreat panorama metadata for generated road points
2) Ensure / generate road points (research_code.dataset.road_points)
3) Read pano_ids and download Street View images
4) (Optional) Upload results to cloud using research_code.ops.google_store


This script is importable and also runnable as a CLI.
It avoids SystemExit so it can be called from other code.


Requirements
- requests, pandas, geopandas, shapely, tqdm
- (optional) google-cloud-storage if your google_store helper needs it


Environment
- GOOGLE_API_KEY (Street View Static API + Metadata enabled)


Config
- Either pass explicit CLI flags, or provide a YAML and load it here.
"""
import csv
import io
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import hydra
from omegaconf import DictConfig,OmegaConf
import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

from research_code.dataset.road_points import run_pipeline as run_panorama_metadata_pipeline
from research_code.ops.google_store import run_pipeline as run_google_store_pipeline
# --------------------------- helpers ---------------------------
def _cfg_get(cfg: DictConfig, dotted: str, default=None):
    try:
        return OmegaConf.select(cfg, dotted, default=default)
    except Exception:
        return default
def _setup_logging(logfile: Optional[Path] = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            *( [logging.FileHandler(logfile)] if logfile else [] ),
            ],
        )
def run_job(cfg: DictConfig) -> int:
    """Run all steps using Hydra cfg and your research_code pipelines."""
    try:
        # 0) Resolve config values
        cache_dir = Path(_cfg_get(cfg, "storage.cache", "cache"))
        api_key = _cfg_get(cfg, "gcp.service_key") or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            logging.error("Missing Google API key (cfg.gcp.service_key or env GOOGLE_API_KEY)")
            return 1
        # Outputs produced by the road_points pipeline
        points_gpkg = cache_dir / _cfg_get(cfg, "datasets.streetview.output_points", "points.gpkg")
        metadata_csv = cache_dir / _cfg_get(cfg, "datasets.streetview.output_panorama_metadata", "pano_metadata.csv")

        # Image settings
        images_dir = Path(_cfg_get(cfg, "datasets.streetview.images.dir", str(cache_dir / "sv_images")))
        size = _cfg_get(cfg, "datasets.streetview.images.size", "640x640")
        fov = int(_cfg_get(cfg, "datasets.streetview.images.fov", 90))
        headings = _cfg_get(cfg, "datasets.streetview.images.headings", [0, 90, 180, 270])
        img_qps = float(_cfg_get(cfg, "datasets.streetview.images.qps", 5.0))
        ### ________________
        # call the panorama metadata pipeline
        # after it call the google store pipeline
        logging.info("✅ Job finished")
        return 0
    except Exception:
        logging.exception("❌ Job failed")
        return 1
@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_main(cfg:DictConfig):
    return run_job(cfg)

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    hydra_main()