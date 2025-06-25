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

---
## ⚙️ Getting Started

### 1. Clone the Repository
`git clone *repository-link*`
### 2. Setting up Python Environment 
Preinstall mamba and run the following code

`mamba env create -f environment.yaml `

`conda activate silverways-env`
### 3. Usage

Step 1: Prepare Your Region of Interest (ROI)
Drop your region files into the data/ folder. Supported formats include .gpkg, .geojson, and .shp.
Step 2: Configure the Pipeline
Edit the YAML file in: `conf/config.yml`


