import geopandas as gpd
from shapely.geometry import Polygon
import geohash
import shapely


def geohash_inside_polygon(polygon, precision=5):
    unchecked = set()
    inside = set()
    outside = set()

    # To handle the case where the polygon is a MultiPolygon
    if isinstance(polygon, shapely.geometry.MultiPolygon):
        polygons = [poly.exterior.coords for poly in polygon]
    else:
        polygons = [polygon.exterior.coords]

    for bbox_pts in polygons:  # Iterate over each polygon or MultiPolygon
        # Encode all polygon points to geohash at specified precision
        for pt in bbox_pts:
            tst_gh = geohash.encode(pt[1], pt[0], precision)  # Ensure lat/lon order
            unchecked.add(tst_gh)

        # Create a shapely Polygon from the coordinates
        bbox = shapely.geometry.Polygon(bbox_pts)

        # While there are unchecked geohashes, check them
        while unchecked:
            this = unchecked.pop()

            # Decode the geohash to a lat/lon point
            lat, lon = geohash.decode(this)
            # point = Point(lon, lat)  # Point is in (lon, lat) order

            # Check if the geohash boundary (bbox) intersects with the polygon
            lat_min, lon_min, lat_max, lon_max = geohash.bbox(this).values()
            geohash_bbox = Polygon(
                [
                    (lon_min, lat_min),
                    (lon_min, lat_max),
                    (lon_max, lat_max),
                    (lon_max, lat_min),
                    (lon_min, lat_min),
                ]
            )

            if bbox.intersects(geohash_bbox):
                inside.add(this)

                # Add neighboring geohashes for further checking
                for gh in geohash.neighbors(this):
                    if gh not in inside and gh not in outside and gh not in unchecked:
                        unchecked.add(gh)
            else:
                outside.add(this)

    # Create Polygons for geohashes inside or intersecting the polygon
    geohash_polygons = []
    for gh in inside:
        # Get bounding box of the geohash
        bbox = geohash.bbox(gh)
        lat_min = bbox["s"]
        lon_min = bbox["w"]
        lat_max = bbox["n"]
        lon_max = bbox["e"]

        # Create a Polygon representing the bounding box of the geohash
        geohash_polygon = Polygon(
            [
                (lon_min, lat_min),
                (lon_min, lat_max),
                (lon_max, lat_max),
                (lon_max, lat_min),
                (lon_min, lat_min),
            ]
        )

        # Append geohash and its bounding box as a polygon
        geohash_polygons.append({"geohash": gh, "geometry": geohash_polygon})

        # Append geohash and its bounding box as a polygon
        geohash_polygons.append({"geohash": gh, "geometry": geohash_polygon})

    gdf = gpd.GeoDataFrame.from_dict(geohash_polygons, crs=4326)

    return gdf


# Example usage
if __name__ == "__main__":
    # Define a polygon (using a simple square polygon as an example)
    polygon = gpd.read_file("../../data/mannheim.geojson").iloc[0]["geometry"]
    precision = 6
    gdf = geohash_inside_polygon(polygon=polygon, precision=precision)
    gdf.to_file(f"../../cache/mannheim_geohash_{precision}.geojson", driver="GeoJSON")
