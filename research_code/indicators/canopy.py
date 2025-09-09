"""
Fraction of Tree Canopy Coverage Per Road Segment
"""
import geopandas as gpd
import rasterio
import rasterio.features
import shapely
from shapely.geometry import shape

def add_canopy_fraction(roads: gpd.GeoDataFrame,
                        raster_file: str,
                        threshold: float = 2.0) -> gpd.GeoDataFrame:
    """
    Computes fraction of each road segment intersecting canopy cover.

    Parameters
    ----------
    roads : GeoDataFrame
        Road segments (LineStrings).
    raster_file : str
        Path to canopy height model (CHM) raster.
    threshold : float
        Minimum canopy height in meters to count as canopy.

    Returns
    -------
    roads : GeoDataFrame
        Copy of roads with new column `canopy_fraction`.
    """
    roads = roads.copy()
    roads["canopy_fraction"] = 0.0

    # 1. Read raster
    with rasterio.open(raster_file) as src:
        chm = src.read(1, masked=True)
        mask = chm > threshold   # canopy = True where CHM > threshold

        # 2. Convert canopy raster mask → polygons
        canopy_shapes = list(rasterio.features.shapes(
            mask.astype("uint8"), transform=src.transform
        ))

    # 3. Build canopy polygons (union)
    canopy_polys = [shape(geom) for geom, val in canopy_shapes if val == 1]
    if len(canopy_polys) == 0:
        return roads

    canopy_union = shapely.union_all(canopy_polys)

    # 4. Compute fraction of road length within canopy
    for idx, road in roads.iterrows():
        total_len = road.geometry.length
        if total_len == 0:
            continue
        inter = road.geometry.intersection(canopy_union)
        canopy_len = inter.length if not inter.is_empty else 0
        roads.at[idx, "canopy_fraction"] = canopy_len / total_len

    return roads
