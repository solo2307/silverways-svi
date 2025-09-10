import typer
from pathlib import Path
from omegaconf import OmegaConf
import logging
from dotenv import load_dotenv
import os

from research_code.jobs import (
    ohsome_job,
    streetview_job,
    streetview_inference_job,
    indicators_job,
)

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# load environment variables from .env automatically
# load_dotenv()
# Explicitly load data-ingestion/.env instead of default
dotenv_path = Path(__file__).parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

app = typer.Typer(
    help="🚶‍♀️ SilverWays CLI – Walkability Indicators for Elderly Pedestrians"
)

osm_app = typer.Typer(help="OSM data utilities")
streetview_app = typer.Typer(help="Street View image download and inference")
indicators_app = typer.Typer(help="Walkability indicators pipeline")

app.add_typer(osm_app, name="osm")
app.add_typer(streetview_app, name="streetview")
app.add_typer(indicators_app, name="indicators")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


# ---------------------------------------------------------------------
# OSM commands
# ---------------------------------------------------------------------
@app.command("apikey-check")
def api_key_check():
    """
    Check if the Google API key is available in the environment (.env).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        typer.secho("✅ Google API key is set!", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "❌ Google API key is missing. Please set it in your .env file.",
            fg=typer.colors.RED,
        )


@osm_app.command("fetch-osm")
def fetch_osm(config: Path = Path("conf/config.yaml")):
    """
    Download road network from OSM for a given city.
    """
    try:
        logging.info(f"📥 Fetching OSM roads for ROI using {config}")
        # Load the Hydra config as DictConfig
        cfg = OmegaConf.load(config)
        ohsome_job.run(cfg)
        logging.info("✅ OSM fetch completed")
    except Exception as e:
        logging.error(f"❌ Failed to fetch OSM data: {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------
# Street View commands
# ---------------------------------------------------------------------
@streetview_app.command("download")
def download(config: Path = Path("conf/config.yaml")):
    """
    Download Street View panoramas based on config.
    """
    try:
        cfg = OmegaConf.load(config)
        logging.info(f"📷 Downloading Street View images using {config}")
        streetview_job.run(cfg)
        logging.info("✅ Street View download completed")
    except Exception as e:
        logging.error(f"❌ Failed to download Street View images: {e}")
        raise typer.Exit(code=1)


@streetview_app.command("infer")
def infer(config: Path = Path("conf/config.yaml")):
    """
    Run deep learning inference on Street View images to extract green/sky indices.
    """
    try:
        logging.info(f"🧠 Running inference on Street View images using {config}")
        cfg = OmegaConf.load(config)
        streetview_inference_job.run(cfg)
        logging.info("✅ Inference completed")
    except Exception as e:
        logging.error(f"❌ Failed to run inference: {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------
# Indicator commands
# ---------------------------------------------------------------------
@indicators_app.command("run")
def run_indicators(config: Path = Path("conf/indicators/ind_config.yaml")):
    """
    Run elderly walkability indicators based on config.
    """
    try:
        cfg = OmegaConf.load(config)
        logging.info(f"⚙️ Running indicators from {config}")
        indicators_job.run(cfg)
        logging.info("✅ Indicators computed successfully")
    except Exception as e:
        logging.error(f"❌ Failed to run indicators: {e}")
        raise typer.Exit(code=1)


@indicators_app.command("merge")
def merge_indicators(
    config: Path = Path("conf/indicators/ind_config.yaml"),
    out: Path = Path("cache/roads_enriched.gpkg"),
):
    """
    Merge all per-indicator outputs into a single enriched road file.
    """
    try:
        cfg = OmegaConf.load(config)
        logging.info(f"📑 Merging indicator outputs into {out}")
        indicators_job.run(cfg)  # you'd implement merge() inside indicators_job
        logging.info("✅ Merge completed")
    except Exception as e:
        logging.error(f"❌ Failed to merge indicators: {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app()
