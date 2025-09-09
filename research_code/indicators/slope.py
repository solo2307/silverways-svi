"""
Road Segment Slope Analysis
===========================

This script computes slope values from a DEM and assigns mean, max, and min slope
statistics to road network segments.

Dependencies:
    - geopandas
    - rasterio
    - numpy
    - rasterstats
"""
import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def compute_slope(dem_file: str) -> Tuple[np.ndarray, rasterio.Affine, str]:
    """
    Compute slope in degrees from a DEM raster.

    Parameters
    ----------
    dem_file : str
        Path to DEM GeoTIFF.

    Returns
    -------
    slope_deg : np.ndarray
        2D slope array (degrees).
    transform : rasterio.Affine
        Affine transform for the raster grid.
    crs : str
        Coordinate reference system of the DEM.
    """
    logging.info(f"Loading DEM: {dem_file}")
    try:
        with rasterio.open(dem_file) as src:
            dem = src.read(1, masked=True)
            transform = src.transform
            crs = src.crs
            dx = transform[0]  # pixel size X
            dy = -transform[4]  # pixel size Y
    except Exception as e:
        logging.error(f"Failed to load DEM {dem_file}: {e}")
        raise

    try:
        dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * dx)
        dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * dy)
        slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
        slope_deg = np.degrees(slope_rad).astype("float32")
        slope_deg = np.ma.array(slope_deg, mask=dem.mask)  # preserve nodata mask
    except Exception as e:
        logging.error(f"Error computing slope: {e}")
        raise

    logging.info("Slope computed successfully.")
    return slope_deg, transform, crs
def assign_slope_to_roads(dem_file, roads_file)-> gpd.GeoDataFrame:
    """
    Assign slope statistics (mean, max, min) to road segments.

    Parameters
    ----------
    dem_file : str
        Path to DEM GeoTIFF.
    roads_file : str
        Path to road network (GeoPackage, Shapefile, GeoJSON).
    out_file : str, optional
        If provided, results will be saved to this file.

    Returns
    -------
    gdf_slope : geopandas.GeoDataFrame
        Road network with slope statistics.
    """
    slope, transform, crs = compute_slope(dem_file)

    try:
        gdf_roads = gpd.read_file(roads_file)
    except Exception as e:
        logging.error(f"Failed to load roads {roads_file}: {e}")
        raise

    # Reproject if needed
    if gdf_roads.crs != crs:
        print("⚠️ CRS mismatch! Reprojecting roads...")
        gdf_roads = gdf_roads.to_crs(crs)
    # Fill masked slope
    slope_filled = slope.filled(-9999)
    # Do zonal stats
    try:
        stats = zonal_stats(
            gdf_roads,
            slope_filled,
            affine=transform,
            stats=["mean", "max", "min"],
            nodata=-9999,
            geojson_out=True
        )
    except Exception as e:
        logging.error(f"Zonal stats computation failed: {e}")
        raise


    gdf_slope = gpd.GeoDataFrame.from_features(stats, crs=gdf_roads.crs)
    return gdf_slope

if __name__ == "__main__":
    # Example usage
    dem_file = "cache/mannheim_dem.tif"
    roads_file = "/Volumes/ygrin/silverways/mannheim/osm/roads_mannheim.gpkg"

    gdf_slope = assign_slope_to_roads(dem_file, roads_file)
    print(gdf_slope.shape)