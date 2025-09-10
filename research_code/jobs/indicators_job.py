"""
Road Indicators end-to-end job

Steps
1) Load base road network
2) Run enabled indicator modules (slope, canopy, benches, etc.)
3) Save per-indicator outputs (as defined in config)
4) (Optional) Merge into a single enriched file later

This script is importable and also runnable as a CLI.
It avoids SystemExit so it can be called from other code.

Requirements
- geopandas, rasterio, rasterstats, numpy, hydra, omegaconf

Config
- See conf/indicators/ind_config.yaml
"""

import logging
import importlib
from pathlib import Path
from typing import Optional

import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf

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


# --------------------------- main pipeline ---------------------------


def run(cfg: DictConfig) -> int:
    """Run all enabled road indicators as defined in config."""
    try:
        roads_file = Path(_cfg_get(cfg, "indicators.roads_file"))
        col_id = _cfg_get(cfg, "indicators.column_id", "osm_id")

        if not roads_file.exists():
            logging.error(f"❌ Roads file not found: {roads_file}")
            return 1

        logging.info(f"📂 Loading base road network: {roads_file}")
        roads = gpd.read_file(roads_file)[[col_id, "geometry"]]

        for name, params in cfg.indicators.features.items():
            if not params.enabled:
                logging.info(f"⏭️ Skipping {name}")
                continue

            logging.info(f"🚦 Running indicator: {name}")
            try:
                module = importlib.import_module(f"indicators.{name}")
                func = getattr(module, f"add_{name}")

                # Run indicator
                roads_with_feature = func(roads.copy(), **params)

                # Save per-indicator output
                if "out_file" in params:
                    out_path = Path(params.out_file)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    roads_with_feature.to_file(out_path, driver="GPKG")
                    logging.info(f"💾 Saved {name} results to {out_path}")

            except Exception:
                logging.exception(f"❌ Failed while running {name}")
                return 1

        logging.info("✅ All indicators finished successfully")
        return 0

    except Exception:
        logging.exception("❌ Job failed")
        return 1


# --------------------------- Hydra entrypoint ---------------------------


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg: DictConfig):
    return run(cfg)


# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    hydra_main()
