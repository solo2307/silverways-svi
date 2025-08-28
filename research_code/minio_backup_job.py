from research_code.ops.minio_storage import main_back_up_images
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import logging
from typing import Iterable, Optional, Tuple

# --------------------------- Logging ---------------------------
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
        logging.info("🚦 Starting Minio Backup job")
        main_back_up_images(cfg)
        logging.info("✅ Job finished")
        return 0
    except Exception:
        logging.exception("❌ Job failed")
        return 1
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def hydra_main(cfg:DictConfig):
    return run_job(cfg)

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    hydra_main()

