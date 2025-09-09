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
Preinstall mamba and run the following code

`mamba env create -f environment.yaml`

`conda activate silver-ways`

### 3. Usage
#### Example of Google Street View Imagery
Step 0: Prepare Road Network File (If  available, skip this step) 

Run `ohsome_job.py` to retreat road network from OSM.

Step 1: Configure the Pipeline
Edit the YAML file in `conf/config.yml`:
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

Step 3: Download Google Street View Imagery 
- call `streetview_job.py` 

Step 4: Backup the downloaded imagery to MINIO bucket (optional)
- call `minio_backup_job.py`


