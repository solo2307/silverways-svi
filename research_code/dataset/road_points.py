import logging
import uuid
import csv
import time
import requests
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from shapely.geometry import LineString
from shapely.ops import unary_union

# ---------------------- CONFIG ----------------------
API_KEY = 'YOUR_GOOGLE_API_KEY'  # Replace with your actual API key
INPUT_FILE = "/Users/ygrinblat/Documents/HeiGIT_Projects/SilverWays/GeoData/Mannheim/roads_mannheim.gpkg"
OUTPUT_POINTS_FILE = "cache/mannheim_points_osm.gpkg"
OUTPUT_CSV_FILE = "cache/mannheim_google_panorama_metadata.csv"
POINT_STEP = 100
MERGE_DISTANCE = 20
REQUEST_DELAY = 0.1
# ----------------------------------------------------

# ---------------------- LOGGING ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/process_retreat_google_pano.log"),
        logging.StreamHandler()
    ]
)
# ----------------------------------------------------


def get_panorama_metadata(lat: float, lon: float, api_key: str) -> dict:
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "key": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "OK":
            return {
                "pano_lat": data["location"]["lat"],
                "pano_lon": data["location"]["lng"],
                "pano_id": data.get("pano_id"),
                "date": data.get("date"),
                "status": "OK"
            }
        else:
            return {
                "pano_lat": None,
                "pano_lon": None,
                "pano_id": None,
                "date": None,
                "status": data.get("status", "UNKNOWN")
            }

    except Exception as e:
        logging.error(f"Error retrieving panorama for ({lat}, {lon}): {e}")
        return {
            "pano_lat": None,
            "pano_lon": None,
            "pano_id": None,
            "date": None,
            "status": f"ERROR: {e}"
        }


def retreat_panorama_metadata(
    gdf_points: gpd.GeoDataFrame,
    api_key: str,
    output_csv: str,
    delay: float = 0.1
):
    gdf_wgs = gdf_points.to_crs("EPSG:4326").reset_index(drop=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["uuid", "lat", "lon", "osm_id", "pano_lat", "pano_lon", "pano_id", "date", "status"]

    existing_coords = set()
    file_exists = output_path.exists()
    is_empty = not file_exists or output_path.stat().st_size == 0

    if file_exists:
        with open(output_path, "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing_coords.add((float(row["lat"]), float(row["lon"])))
                except Exception:
                    continue

    with open(output_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_empty:
            writer.writeheader()

        for row in tqdm(gdf_wgs.itertuples(), total=len(gdf_wgs), desc="📷 Fetching pano metadata"):
            lat, lon = row.geometry.y, row.geometry.x
            if (lat, lon) in existing_coords:
                continue

            meta = get_panorama_metadata(lat, lon, api_key)

            writer.writerow({
                "uuid": getattr(row, "uuid", str(uuid.uuid4())),
                "lat": lat,
                "lon": lon,
                "osm_id": getattr(row, "osm_id", None),
                "pano_lat": meta["pano_lat"],
                "pano_lon": meta["pano_lon"],
                "pano_id": meta["pano_id"],
                "date": meta["date"],
                "status": meta["status"]
            })

            time.sleep(delay)

    logging.info(f"✅ Panorama metadata written to: {output_csv}")


class RoadDataset:
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.gdf = self._load_data()

    def _load_data(self) -> gpd.GeoDataFrame:
        logging.info(f"📂 Loading: {self.input_path.name}")
        gdf = gpd.read_file(self.input_path)

        if gdf.empty:
            raise ValueError("Input file is empty.")

        if gdf.crs.is_geographic:
            logging.info("🧭 Reprojecting to EPSG:3857...")
            gdf = gdf.to_crs("EPSG:3857")

        return gdf

    def generate_points_along_line(self, line: LineString, step: float = 100) -> List:
        length = line.length
        points = [line.interpolate(distance) for distance in range(0, int(length), int(step))]
        if length % step != 0:
            points.append(line.interpolate(length))
        return points

    @staticmethod
    def merge_close_points(gdf_points: gpd.GeoDataFrame, distance_threshold: float = 30) -> gpd.GeoDataFrame:
        """
        Merge points within a specified distance using spatial index for efficiency.
        Returns a GeoDataFrame with one representative point per cluster.
        """
        logging.info(f"🔗 Merging close points within {distance_threshold} meters...")

        if gdf_points.crs.is_geographic:
            logging.info("🧭 Reprojecting to EPSG:3857 for distance calculations...")
            gdf_points = gdf_points.to_crs(epsg=3857)

        gdf_points = gdf_points.copy().reset_index(drop=True)
        sindex = gdf_points.sindex
        visited = set()
        groups = []

        for idx, geom in enumerate(gdf_points.geometry):
            if idx in visited:
                continue

            # Use spatial index to find nearby points quickly
            possible_matches_idx = list(sindex.intersection(geom.buffer(distance_threshold).bounds))
            nearby_points = gdf_points.iloc[possible_matches_idx]
            cluster = nearby_points[nearby_points.distance(geom) <= distance_threshold]

            # Mark all indices as visited
            visited.update(cluster.index)

            # Take centroid or first point as representative (here: first point)
            groups.append(cluster.iloc[0])

        logging.info(f"✅ Reduced {len(gdf_points)} → {len(groups)} merged points")
        return gpd.GeoDataFrame(groups, geometry='geometry', crs=gdf_points.crs).reset_index(drop=True)

    def generate_points(self) -> gpd.GeoDataFrame:
        gdf = self.gdf[['geometry', 'osm_id', 'highway', 'access']]
        gdf = gdf[gdf['access'] != 'private']

        all_points, osm_ids = [], []
        for idx, geom in tqdm(enumerate(gdf.geometry), total=len(gdf), desc="🛣 Generating points"):
            if geom is None:
                continue

            osm_id = gdf.iloc[idx].get("osm_id", idx)

            if geom.geom_type == "LineString":
                points = self.generate_points_along_line(geom, step=POINT_STEP)
            elif geom.geom_type == "MultiLineString":
                points = [pt for line in geom.geoms for pt in self.generate_points_along_line(line, step=POINT_STEP)]
            else:
                continue

            all_points.extend(points)
            osm_ids.extend([osm_id] * len(points))

        gdf_points = gpd.GeoDataFrame({"geometry": all_points, "osm_id": osm_ids}, crs=gdf.crs)
        gdf_points = self.merge_close_points(gdf_points, distance_threshold=MERGE_DISTANCE)
        gdf_points["uuid"] = [str(uuid.uuid4()) for _ in range(len(gdf_points))]
        return gdf_points


if __name__ == "__main__":
    try:
        logging.info("🚀 Starting road point generation and panorama metadata collection...")
        if not Path(OUTPUT_POINTS_FILE).exists():
            roads = RoadDataset(INPUT_FILE)
            gdf_points = roads.generate_points()
            Path(OUTPUT_POINTS_FILE).parent.mkdir(parents=True, exist_ok=True)
            gdf_points.to_file(OUTPUT_POINTS_FILE, driver="GPKG")
            logging.info(f"✅ Points saved to: {OUTPUT_POINTS_FILE}")
        else:
            gdf_points = gpd.read_file(OUTPUT_POINTS_FILE)
        logging.info("📷 Collecting panorama metadata...")
        retreat_panorama_metadata(gdf_points, API_KEY, OUTPUT_CSV_FILE, delay=REQUEST_DELAY)
    except Exception as e:
        logging.exception(f"❌ Process failed: {e}")
