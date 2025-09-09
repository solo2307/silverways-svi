"""
Road Indicators end-to-end job

Steps
1) Load base road network
2) Run enabled indicator modules (e.g., slope, canopy coverage)
3) Save enriched road network
4) (Optional) Upload results to cloud

This script is importable and also runnable as a CLI.
It avoids SystemExit so it can be called from other code.

Requirements
- geopandas, rasterio, rasterstats, numpy, hydra, omegaconf

Config
- See conf/indicators/ind_config.yaml
"""
import logging
from pathlib import Path
import importlib

import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf

# --------------------------- helpers ---------------------------
def _cfg_get(cfg: DictConfig, dotted: str, default=None):
    try:
        return OmegaConf.select(cfg, dotted, default=default)
    except Exception:
        return default

def _setup_logging(logfile: Path = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(logfile)] if logfile else []),
        ],
    )

# --------------------------- main pipeline ---------------------------
def run_job(cfg: DictConfig) -> int:
    """Run all enabled road indicators."""
    try:
        # Resolve inputs/outputs
        roads_file = Path(_cfg_get(cfg, "indicators.roads_file"))
        out_file = Path(_cfg_get(cfg, "indicators.out_file"))
        col_id = _cfg_get(cfg, "indicators.column_id", "osm_id")

        if not roads_file.exists():
            logging.error(f"❌ Roads file not found: {roads_file}")
            return 1

        # Load roads
        logging.info(f"📂 Loading base roads: {roads_file}")
        roads = gpd.read_file(roads_file)[[col_id, "geometry"]]

        # Loop through indicator features
        for name, params in cfg.indicators.features.items():
            if not params.enabled:
                logging.info(f"⏭️ Skipping {name}")
                continue

            logging.info(f"🚦 Running indicator: {name}")
            try:
                module = importlib.import_module(f"indicators.{name}")
                func = getattr(module, f"add_{name}")
                roads = func(roads, **params)
            except Exception:
                logging.exception(f"❌ Failed while running {name}")
                return 1

        # Save enriched road network
        out_file.parent.mkdir(parents=True, exist_ok=True)
        roads.to_file(out_file, driver="GPKG")
        logging.info(f"✅ Saved enriched roads to {out_file}")
        return 0

    except Exception:
        logging.exception("❌ Job failed")
        return 1

# --------------------------- Hydra entrypoint ---------------------------
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def hydra_main(cfg: DictConfig):
    return run_job(cfg)

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    hydra_main()
