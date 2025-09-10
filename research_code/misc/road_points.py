import logging
import uuid
import csv
import time
import requests
import pandas as pd
import sys
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from typing import List
from shapely.geometry import LineString
from omegaconf import DictConfig
import hydra
from research_code.ops.ohsome_store import run_road_pipeline

# ----------------------------------------------------
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)


def build_session(timeout: float = 6.0) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        read=6,
        connect=3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "OPTIONS"),
        backoff_factor=0.5,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    _req = s.request

    def _wrapped(method, url, **kw):
        kw.setdefault("timeout", timeout)
        return _req(method, url, **kw)

    s.request = _wrapped  # type: ignore
    return s


# ----------------------------------------------------
def get_panorama_metadata(
    lat: float, lon: float, api_key: str, session: requests.Session
) -> dict:
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "key": api_key}
    try:
        r = session.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        status = data.get("status", "UNKNOWN")
        if status == "OK":
            loc = data.get("location", {})
            return {
                "pano_lat": loc.get("lat"),
                "pano_lon": loc.get("lng"),
                "pano_id": data.get("pano_id"),
                "date": data.get("date"),
                "status": status,
            }
        return {
            "pano_lat": None,
            "pano_lon": None,
            "pano_id": None,
            "date": None,
            "status": status,
        }
    except Exception as e:
        logging.exception("Street View metadata request failed")
        return {
            "pano_lat": None,
            "pano_lon": None,
            "pano_id": None,
            "date": None,
            "status": f"ERROR: {e}",
        }


def retreat_panorama_metadata(
    gdf_points: gpd.GeoDataFrame,
    api_key: str,
    output_csv: str,
    delay: float = 0.1,
    qps: float = 10.0,
    request_timeout: float = 6.0,
):
    gdf_wgs = gdf_points.to_crs(4326).reset_index(drop=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "uuid",
        "lat",
        "lon",
        "osm_id",
        "pano_lat",
        "pano_lon",
        "pano_id",
        "date",
        "status",
    ]
    existing = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            for r in pd.read_csv(output_path, usecols=["lat", "lon"]).itertuples(
                index=False
            ):
                existing.add((float(r.lat), float(r.lon)))
        except Exception:
            pass

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    session = build_session(timeout=request_timeout)

    min_interval = 1.0 / max(qps, 0.1)
    last_call = 0.0
    backoff = 1.0

    with output_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for row in tqdm(
            gdf_wgs.itertuples(), total=len(gdf_wgs), desc="📷 Fetching pano metadata"
        ):
            lat, lon = row.geometry.y, row.geometry.x
            if (lat, lon) in existing:
                continue

            now = time.time()
            sleep_for = last_call + min_interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_call = time.time()

            meta = get_panorama_metadata(lat, lon, api_key, session)
            if meta["status"] == "OVER_QUERY_LIMIT":
                logging.warning("Hit OVER_QUERY_LIMIT; backing off for %.1fs", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)  # cap at 60s
                # retry once after backoff
                meta = get_panorama_metadata(lat, lon, api_key, session)

            w.writerow(
                {
                    "uuid": getattr(row, "uuid", str(uuid.uuid4())),
                    "lat": lat,
                    "lon": lon,
                    "osm_id": getattr(row, "osm_id", None),
                    "pano_lat": meta["pano_lat"],
                    "pano_lon": meta["pano_lon"],
                    "pano_id": meta["pano_id"],
                    "date": meta["date"],
                    "status": meta["status"],
                }
            )
            time.sleep(delay)

    logging.info("✅ Panorama metadata written to: %s", output_csv)


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
        points = [
            line.interpolate(distance) for distance in range(0, int(length), int(step))
        ]
        if length % step != 0:
            points.append(line.interpolate(length))
        return points

    @staticmethod
    def merge_close_points(
        gdf_points: gpd.GeoDataFrame, distance_threshold: float = 30
    ) -> gpd.GeoDataFrame:
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
            possible_matches_idx = list(
                sindex.intersection(geom.buffer(distance_threshold).bounds)
            )
            nearby_points = gdf_points.iloc[possible_matches_idx]
            cluster = nearby_points[nearby_points.distance(geom) <= distance_threshold]

            # Mark all indices as visited
            visited.update(cluster.index)

            # Take centroid or first point as representative (here: first point)
            groups.append(cluster.iloc[0])

        logging.info(f"✅ Reduced {len(gdf_points)} → {len(groups)} merged points")
        return gpd.GeoDataFrame(
            groups, geometry="geometry", crs=gdf_points.crs
        ).reset_index(drop=True)

    def generate_points(
        self, point_step: int = 100, merge_distance: int = 20
    ) -> gpd.GeoDataFrame:
        gdf = self.gdf[["geometry", "osm_id", "highway", "access"]]
        gdf = gdf[gdf["access"] != "private"]

        all_points, osm_ids = [], []
        for idx, geom in tqdm(
            enumerate(gdf.geometry), total=len(gdf), desc="🛣 Generating points"
        ):
            if geom is None:
                continue

            osm_id = gdf.iloc[idx].get("osm_id", idx)

            if geom.geom_type == "LineString":
                points = self.generate_points_along_line(geom, step=point_step)
            elif geom.geom_type == "MultiLineString":
                points = [
                    pt
                    for line in geom.geoms
                    for pt in self.generate_points_along_line(line, step=point_step)
                ]
            else:
                continue

            all_points.extend(points)
            osm_ids.extend([osm_id] * len(points))

        gdf_points = gpd.GeoDataFrame(
            {"geometry": all_points, "osm_id": osm_ids}, crs=gdf.crs
        )
        gdf_points = self.merge_close_points(
            gdf_points, distance_threshold=merge_distance
        )
        gdf_points["uuid"] = [str(uuid.uuid4()) for _ in range(len(gdf_points))]
        return gdf_points


def run_pipeline(cfg: DictConfig):
    # ---------------------- Generate points and fetch panorama metadata ----------------------
    try:
        logs_dir = Path(cfg.datasets.streetview.get("log_dir", "logs"))
        setup_logging(Path(logs_dir) / "process_retreat_google_pano.log")

        input_file = Path(cfg.storage.cache) / cfg.datasets.streetview.input_file
        out_points = Path(cfg.storage.cache) / cfg.datasets.streetview.output_points
        out_csv = (
            Path(cfg.storage.cache) / cfg.datasets.streetview.output_panorama_metadata
        )

        api_key = cfg.gcp.service_key
        if not api_key:
            logging.error("Missing Google API key (cfg.gcp.service_key).")
            return 1
        point_step = int(cfg.datasets.streetview.point_step)
        merge_dist = int(cfg.datasets.streetview.merge_distance)
        delay = float(cfg.datasets.streetview.request_delay)

        try:
            if not input_file.exists():
                logging.info("🚀 Retreating road network...")
                run_road_pipeline(cfg)
        except Exception as e:
            logging.error(f"Road pipeline failed: {e}")
            return 1

        logging.info("🚀 Starting road point generation...")
        if not out_points.exists():
            roads = RoadDataset(str(input_file))
            gdf_points = roads.generate_points(
                point_step=point_step, merge_distance=merge_dist
            )
            out_points.parent.mkdir(parents=True, exist_ok=True)
            gdf_points.to_file(out_points, driver="GPKG")
            logging.info("✅ Points saved to: %s", out_points)
        else:
            gdf_points = gpd.read_file(out_points)

        logging.info("📷 Collecting panorama metadata...")
        retreat_panorama_metadata(
            gdf_points.iloc[:2],
            api_key,
            str(out_csv),
            delay=delay,
            qps=float(cfg.datasets.streetview.get("qps", 10.0)),
            request_timeout=float(cfg.datasets.streetview.get("request_timeout", 6.0)),
        )
        return 0

    except Exception:
        logging.exception("❌ Process failed")
        return 1


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg: DictConfig) -> int:
    return run_pipeline(cfg)


if __name__ == "__main__":
    hydra_main()
