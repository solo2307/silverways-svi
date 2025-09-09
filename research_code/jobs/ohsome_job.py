"""
OSM end‑to‑end job


Steps
1)

Environment
- MINIO

"""
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Iterable, Optional, Tuple


from research_code.ops.ohsome_store import run_road_pipeline

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
        # Outputs produced by the road_points pipeline
        metadata_csv = cache_dir / _cfg_get(cfg, "datasets.streetview.output_panorama_metadata", "pano_metadata.csv")
        ### ________________
        # call the panorama metadata pipeline

        logging.info("🚦 Starting road pipeline")
        run_road_pipeline(cfg)
        logging.info("✅ Job finished")
        return 0
    except Exception:
        logging.exception("❌ Job failed")
        return 1
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg:DictConfig):
    return run_job(cfg)

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    hydra_main()