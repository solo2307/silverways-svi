import json
from shapely import wkt
from pathlib import Path
import pandas as pd
import geopandas as gpd
import logging
import hydra
from omegaconf import DictConfig
from datetime import datetime
from ohsome import OhsomeClient
import requests

# ---------- Global Logger Setup ----------
def setup_logger():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    log_file = Path(f"logs/osm-duckdb-store-{timestamp}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("OsmDuckDBLogger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger

logger = setup_logger()
# ---------- Helpers ----------
def load_bbox_from_gpkg(gpkg_path: str):
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        raise ValueError(f"No features found in {gpkg_path}")
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    return tuple(bounds)


def filter_network_sidewalk(gdf):
	# Always keep geometry and ID fields
	base_keep = ["geometry", "osmId"]

	# Tags of interest
	keywords = [
		"highway", "lanes", "maxspeed", "oneway", "ref",
		"bridge", "tunnel", "surface", "smoothness", "paved",
		"sidewalk", 'bicycle', 'foot','cycleway'
	]
	keep_cols = [c for c in gdf.columns if c in base_keep or c in keywords]
	# Keep columns that contain any of these keywords
	# keep_cols = [c for c in gdf.columns if any(k in c.lower() for k in keywords)]
	#
	# # Merge with base
	# keep_cols = list(dict.fromkeys(base_keep + keep_cols))  # unique, preserve order

	return gdf[keep_cols]
def sanitize_columns(gdf):
	# 1. Ensure all are strings
	gdf.columns = gdf.columns.astype(str)

	# 2. Replace invalid chars
	gdf.columns = gdf.columns.str.replace(":", "_", regex=False).str.replace("@", "", regex=False)

	return gdf

def filter_buildings(gdf):
	base_keep = ["geometry", "osmId"]

	# Tags of interest
	keywords = [
		"building", "access", "addr", "wheelchair",
	]

	# Keep columns that contain any of these keywords
	# keep_cols = [c for c in gdf.columns if c in base_keep or c in keywords]
	keep_cols = [c for c in gdf.columns if any(k in c.lower() for k in keywords)]

	# Merge with base
	keep_cols = list(dict.fromkeys(base_keep + keep_cols))  # unique, preserve order
	return gdf[keep_cols]
# ---------- Class Definition ----------
class OhsomeAPI:
	def __init__(self, geom_path):
		self.url = 'https://api.ohsome.org/v1/elements/geometry'
		self.roi_path = geom_path
		self._assign_geometry()
		self.time = '2025-08-01'

	def _assign_geometry(self):
		# Load bbox and polygons from the provided GeoPackage
		self.bbox = ",".join(str(n) for n in load_bbox_from_gpkg(self.roi_path))
		roi_gdf = gpd.read_file(self.roi_path)
		roi_gdf = roi_gdf[roi_gdf.geometry.type=='Polygon']
		coords = roi_gdf.geometry.apply(lambda g: list(g.exterior.coords) if g.geom_type == "Polygon" else g.coords[:])
		self.bpolys = ",".join([f"{x},{y}" for x, y in coords[0]])

	def get_roads(self,
				 bbox:bool = True,
				 filter:str='highway=* and type:way',
				 output_path:str = None
				):
		if bbox:
			self.data = {"bboxes": self.bbox, "time": self.time, "filter": filter, "properties": "tags"}
		else:
			self.data = {"bpolys": self.bpolys, "time": self.time, "filter": filter,"properties": "tags"}
		try:
			response = requests.post(self.url, data=self.data)
			resp = response.json()
			# Convert directly into a GeoDataFrame
			gdf = gpd.GeoDataFrame.from_features(resp["features"])
			gdf = gdf.set_crs('EPSG:4326')
			# Ensure column names are strings
			gdf = sanitize_columns(gdf)
			gdf = filter_network_sidewalk(gdf)
			if output_path is not None:
				gdf.to_file(output_path, driver='GPKG')
			return gdf
		except Exception:
			logger.exception("❌ OhsomeAPI request failed")
			return None
	def get_buildings(self,
				 bbox:bool = True,
				 filter:str='building=* and (type:relation or type:way)',
				 output_path:str = None
				):
		if bbox:
			self.data = {"bboxes": self.bbox, "time": self.time, "filter": filter, "properties": "tags"}
		else:
			self.data = {"bpolys": self.bpolys, "time": self.time, "filter": filter,"properties": "tags"}
		try:
			response = requests.post(self.url, data=self.data)
			resp = response.json()
			# Convert directly into a GeoDataFrame
			gdf = gpd.GeoDataFrame.from_features(resp["features"])
			gdf = gdf.set_crs('EPSG:4326')
			# Ensure column names are strings
			gdf = sanitize_columns(gdf)
			gdf = filter_buildings(gdf)
			if output_path is not None:
				gdf.to_file(output_path, driver='GPKG')
			return gdf
		except Exception:
			logger.exception("❌ OhsomeAPI request failed")
			return None
	def get_public_transport_stops(self,bbox:bool = True,
					filter: str = "public_transport=stop_position and type:node",
				 output_path: str = None):
		if bbox:
			self.data = {"bboxes": self.bbox, "time": self.time, "filter": filter}
		else:
			self.data = {"bpolys": self.bpolys, "time": self.time, "filter": filter}
		try:
			response = requests.post(self.url, data=self.data)
			resp = response.json()
			# Convert directly into a GeoDataFrame
			gdf = gpd.GeoDataFrame.from_features(resp["features"])
			gdf = gdf.set_crs('EPSG:4326')

			if output_path is not None:
				gdf.to_file(output_path, driver='GPKG')
			return gdf
		except Exception:
			logger.exception("❌ OhsomeAPI request failed")
			return None

	def get_bench(self,
				 bbox:bool = True,
				 filter:str='amenity=bench and type:node',
				 output_path:str = None
				):
		if bbox:
			self.data = {"bboxes": self.bbox, "time": self.time, "filter": filter}
		else:
			self.data = {"bpolys": self.bpolys, "time": self.time, "filter": filter}
		try:
			response = requests.post(self.url, data=self.data)
			resp = response.json()
			# Convert directly into a GeoDataFrame
			gdf = gpd.GeoDataFrame.from_features(resp["features"])
			gdf = gdf.set_crs('EPSG:4326')

			if output_path is not None:
				gdf.to_file(output_path, driver='GPKG')
			return gdf
		except Exception:
			logger.exception("❌ OhsomeAPI request failed")
			return None

# ------------- RUN PIPELINE -------------
def run_road_pipeline(cfg: DictConfig)-> int:
	try:
		client = OhsomeAPI(geom_path = cfg.datasets.osm.region_path)
		_ = client.get_roads(bbox=True,
						 output_path= Path(cfg.storage.sds) / cfg.datasets.osm.dir / 'roads.gpkg')
		return 0
	except Exception:
		logger.exception("❌ OhsomeAPI extraction failed")
		return 1
def run_pipeline(cfg: DictConfig)-> int:
	try:
		client = OhsomeAPI(geom_path = cfg.datasets.osm.region_path)

		if cfg.datasets.osm.roads in ['True', 'true', True]:
			_ = client.get_roads(bbox=True,
							 output_path= Path(cfg.storage.sds) / cfg.datasets.osm.dir / 'roads.gpkg')
		if cfg.datasets.osm.buildings in ['True', 'true', True]:
			_ = client.get_buildings(bbox=True,
									 output_path=Path(cfg.storage.sds) / cfg.datasets.osm.dir / 'buildings.gpkg')
		return 0
	except Exception:
		logger.exception("❌ OhsomeAPI extraction failed")
		return 1
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg: DictConfig):
	return run_road_pipeline(cfg)

if __name__ == "__main__":
	hydra_main()


