import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from research_code.dl.models import download_models
from research_code.dl.green_model_svi import SegModelPSPNet
from research_code.dl.svi_dataset import GSVIDataSet


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
            *([logging.FileHandler(logfile)] if logfile else []),
        ],
    )


def run(cfg: DictConfig) -> int:
    """Run Inference mode for StreetView images."""
    try:
        print("fff", cfg)
        _ = download_models()
        model = SegModelPSPNet(model_path=Path("data/models"))

        input_dir = _cfg_get(cfg, "input_dir", "data/gsv_images")
        pred_dir = _cfg_get(cfg, "pred_dir", "data/gsv_predictions")
        dataset = GSVIDataSet(input_dir=input_dir, pred_dir=pred_dir, model=model)
        dataset.process_all(save_pred=True, gvi=False)

        logging.info("✅ Job finished")
        return 0
    except Exception:
        logging.exception("❌ Job failed")
        return 1


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg: DictConfig):
    return run(cfg)


# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    hydra_main()
