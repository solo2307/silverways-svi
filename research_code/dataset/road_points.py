from pathlib import Path
import os, logging
import geopandas as gpd
from shapely.geometry import box
import mercantile

class Road_Dataset:
    def __init__(self, input_path:str):
        self.input_path = Path(input_path)

    def read_file(self):
        # Define bounding box [min_lon, min_lat, max_lon, max_lat]
        bbox = [8.41416,49.410362, 8.58999,49.59047]
        bbox_geom = box(8.470295, 49.503341, 8.484926, 49.509656)
        zoom = 14  # Define zoom level

        # Generate tiles covering the bounding box at the specified zoom level
        tiles = list(mercantile.tiles(*bbox, zoom))

        # bbox_geom = box(8.470295, 49.503341, 8.484926, 49.509656)
        for tile in tiles:
            gdf = gpd.read_file(filename = self.input_path,
                                mask= box(*list(mercantile.bounds(tile))))
            # Drop roads that only for pedestrian
            gdf_roads = gdf[~gdf['highway'].isin(['steps',
                                                  'service',
                                                  'path',
                                                  'pedestrian',
                                                  'footway'])]

            # Create points along roads;
            break
        return gdf_roads

    # def generate_points(self, gdf, dist):


if __name__ == "__main__":
    rd = Road_Dataset('cache/osm/roads_mannheim.geojson')
    gdf = rd.read_file()
    print(gdf.shape)