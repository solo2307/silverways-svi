import geopandas as gpd

def add_benches(roads: gpd.GeoDataFrame, benches_file: str, search_radius: int = 50, **kwargs) -> gpd.GeoDataFrame:
    # compute bench proximity here
    roads = roads.copy()
    roads["bench_dist"] = 9999
    return roads