# 🚶‍♀️ SilverWays Indicators – ETL Pipeline for Elderly Walkability

**data-ingestion** is a semi-automatic pipeline that extracts, transforms, and loads (ETL) geospatial data to analyze **walkability for elderly people** in urban environments.

---

## 💡 Key Features

- 🗺️ Region-based geospatial data extraction
- 🛰️ Satellite and street-level imagery integration
- 🌍 OSM-based infrastructure and amenity data
- 🌳 Urban greenery and infrastructure indicators
- 🧠 Deep learning-based semantic segmentation
- 🏙️ Outputs data for accessibility and walkability analysis

---
## 📂 Project Structure
- **conf/** → Hydra configuration files
  - `config.yaml` = main config
  - `datasets/` = dataset configs (OSM, streetview, etc.)
  - `indicators/` = YAML configs for elderly-friendly indicators

- **research_code/** → Core Python modules
  - `indicators/` = indicator implementations (slope, canopy, benches, etc.)
  - `jobs/` = orchestration jobs (streetview_job, indicator_job, etc.)
  - `ops/` = helper modules (downloads, APIs, storage)
  - `misc/` = utility functions (geospatial, image processing)
  - `dl/` = deep learning models for Street View Imagery

- **cache/** → Local cache for intermediate outputs
- **data/** → Raw data storage
- **environment.yaml** → Python dependencies
- **.env** → Environment variables (e.g., API keys)
---
## ⚙️ Getting Started

### 1. Clone the Repository
`git clone https://gitlab.heigit.org/giscience/disaster-tools/silverways/data-ingestion.git`
### 2. Setting up Python Environment
- Option A – Using mamba/conda (recommended for geospatial libs)

Preinstall mamba and run the following code

`mamba env create -f environment.yaml`

`conda activate silver-ways`

- Option B – Using pip

If you don’t want to use **mamba/conda**, you can install the package directly:

`pip install -e .`

This will install all dependencies from pyproject.toml and expose the CLI tool **_silverways_**.

## 🚀 Usage
### Manual Job Run (mamba/conda) - example of Google Street View Imagery

Step 1: Prepare Road Network File (If  available, skip this step)

Run `ohsome_job.py` to retreat road network from OSM.

Step 2: Download Google Street View Imagery
- call `streetview_job.py`

Step 3: Backup the downloaded imagery to MINIO bucket (optional)
- call `minio_backup_job.py`

### 🖥️ Command Line Interface (CLI)

The silverways CLI wraps all major pipeline steps (OSM fetch, Street View imagery, indicator computation) into simple commands.
It automatically loads environment variables from your .env file (must be placed in the repository root).

Check available commands in terminal:

`silverways --help`

#### 🔑 API Key Check

Verify that your .env file contains a Google API key:

`silverways apikey-check`

 - ✅ Prints confirmation if the key is set
 - ❌ Warns you if it’s missing

#### 🗺️ OSM Data

Fetch the road network from OpenStreetMap:
`silverways osm fetch-osm --config conf/config.yaml`

#### 📸 Street View Imagery

1. Download panoramas:

`silverways streetview download --config conf/config.yaml`


2. Run deep learning inference (e.g., greenery, sky index):

`silverways streetview infer --config conf/config.yaml`

#### 🌳 Walkability Indicators

Compute indicators:

`silverways indicators run --config conf/indicators/ind_config.yaml`


Merge all outputs into one enriched roads file:

`silverways indicators merge \
  --config conf/indicators/ind_config.yaml \
  --out cache/roads_enriched.gpkg`

### 📂 Configuration

All pipeline settings (datasets, storage paths, API keys) are managed in Hydra YAML configs under `conf/config.yml`:
 - assign a `gcp.service_key` that you generated in GCP
 - define the `storage.sds`/`storage.cache` location to store your data

Edit the Street View YAML in `conf/datasets/streetview_config.yaml`:
- define the `input_file` to define your road network file
- define  the output paths `output_points` and `output_panorama_metadata`
- assign point settings such as `point_step` and `merge_distance`
- assign a `request_delay` for retreat panorama images, there is no limit for the request of the Google street view panorama matadata
- define `manifest` to store the downloaded panorama images
- define `dir` to store the downloaded panorama images, note: this folder will be created inside the `storage.sds`/`storage.cache` location
- define `max_requests` to limit the number of downloaded panorama images, if it is **None** or **null** all the panorama images will be downloaded

Environment secrets (Google API key, MinIO credentials, etc.) should be placed in .env at the project root.
