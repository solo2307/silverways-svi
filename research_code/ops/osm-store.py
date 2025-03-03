
import json
from shapely import wkt
from pathlib import Path
import pandas as pd
import geopandas as gpd
import logging
import duckdb

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OhsomeDB:
	def __init__(self, dir_cache='cache/osm', key_id='osm-yulia', key_secret='its_ohsome'):
		self._connect()
		self._create_secret(key_id, key_secret)
		self.dir_cache = Path(dir_cache)
		self.dir_cache.mkdir(parents=True, exist_ok=True)

	def _connect(self):
		self.con = duckdb.connect()
		self.con.execute("INSTALL httpfs;")
		self.con.execute("LOAD httpfs;")
		self.con.execute("INSTALL spatial;")
		self.con.execute("LOAD spatial;")
		logger.info("Ohsome DuckDB is connected!")

	def _create_secret(self, key_id, key_secret):
		self.con.execute(f"""
		    CREATE SECRET minio_secret (
		        TYPE s3,
		        KEY_ID '{key_id}',
		        SECRET '{key_secret}',
		        REGION 'eu-central-1',
		        ENDPOINT 'sotm2024.minio.heigit.org',
		        URL_STYLE 'path',
		        USE_SSL false
		    );
		""")
	def _drop_secret(self):
		self.con.execute("""drop secret minio_secret;""")

	def extract_roads(self, output_geojson, minx=8.41416, miny=49.410362, maxx=8.58999, maxy=49.59047):
		query = f"""SELECT osm_id, tags, geometry 
			FROM 
				read_parquet(
				    's3a://heigit-ohsome-sotm24/data/geo_sort_104933/contributions/status=latest/**/*.parquet', 
				    hive_partitioning = true
					) AS mytable
			WHERE  1=1 
				AND mytable.geometry_type='LineString' 
				AND map_contains(mytable.tags, 'highway')
			  	AND mytable.bbox.xmin < {maxx}
			    AND mytable.bbox.ymin < {maxy}
			    AND mytable.bbox.xmax > {minx}
			    AND mytable.bbox.ymax > {miny}
			;
			"""
		# Execute the query and fetch the results into a DataFrame
		df = self.con.execute(query).fetchdf()
		tags_df = df['tags'].apply(pd.Series)
		df = pd.concat([df.drop(columns=['tags']), tags_df], axis=1)
		df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
		gdf = gpd.GeoDataFrame(df, geometry='geometry')
		gdf.to_file(output_geojson, driver="GeoJSON")
	def extract_facilities(self, tag_key='amenity', minx=8.41416, miny=49.410362, maxx=8.58999, maxy=49.59047):
		"""tag_key = ['amenity', 'shop', 'tourism']"""
		query = f"""
			SELECT osm_id, tags, geometry
			FROM 
				read_parquet(
					's3a://heigit-ohsome-sotm24/data/geo_sort_104933/contributions/status=latest/**/*.parquet', 
					hive_partitioning = true
					) AS mytable
			WHERE  1=1 
				AND mytable.geometry_type='Point' 
				AND map_contains(mytable.tags, '{tag_key}')
			  	AND mytable.bbox.xmin > {minx}
			    AND mytable.bbox.ymin > {miny}
			    AND mytable.bbox.xmax < {maxx}
			    AND mytable.bbox.ymax < {maxy}
;
		"""
		# Execute the query and fetch the results into a DataFrame
		df = self.con.execute(query).fetchdf()
		tags_df = df['tags'].apply(pd.Series)
		df = pd.concat([df.drop(columns=['tags']), tags_df], axis=1)
		df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
		gdf = gpd.GeoDataFrame(df, geometry='geometry')
		return gdf

	def extract_buildings(self, output_geojson, minx = 8.41416, miny = 49.410362, maxx = 8.58999, maxy = 49.59047):
		query = f"""SELECT osm_id, tags, geometry 
					FROM 
						read_parquet(
						    's3a://heigit-ohsome-sotm24/data/geo_sort_104933/contributions/status=latest/**/*.parquet', 
						    hive_partitioning = true
							) AS mytable
					WHERE  1=1 
						AND mytable.geometry_type='Polygon' 
						AND map_contains(mytable.tags, 'building')
					  	AND mytable.bbox.xmin > {minx}
						AND mytable.bbox.ymin > {miny}
						AND mytable.bbox.xmax < {maxx}
						AND mytable.bbox.ymax < {maxy}
					;
					"""
		# Execute the query and fetch the results into a DataFrame
		df = self.con.execute(query).fetchdf()
		tags_df = df['tags'].apply(pd.Series)
		df = pd.concat([df.drop(columns=['tags']), tags_df], axis=1)
		df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
		gdf = gpd.GeoDataFrame(df, geometry='geometry')
		gdf.to_file(output_geojson, driver="GeoJSON")


	def extract_benches(self, gdf):
		output_file = self.dir_cache / 'osm_benches.geojson'
		if not output_file.exists():
			gdf = gdf[gdf['amenity']=='bench']
			logger.info(f"Number of Benches in BBOX= {gdf.shape[0]}")
			gdf.to_file(output_file, driver='GeoJSON')
			logger.info(f"OSM amenities BENCH is stored in: {output_file}")
		else:
			logger.info(f"Skipping Extraction. File exists {output_file}")
	def extract_entertainment_landmarks(self, gdf,
				keys=['fountain',
				'social_centre','community_centre']):#memorial=statue
		output_file = self.dir_cache / 'osm_entertainment.geojson'
		if not output_file.exists():
			gdf = gdf[gdf['amenity'].isin(keys)]
			logger.info(f"Number of Benches in BBOX= {gdf.shape[0]}")
			gdf.to_file(output_file, driver='GeoJSON')
			logger.info(f"OSM amenities BENCH is stored in: {output_file}")
		else:
			logger.info(f"Skipping Extraction. File exists {output_file}")
	def extract_food(self, gdf,
					 keys=['bar', 'cafe','fast_food', 'pub', 'biergarten',
					'food_court', 'ice_cream', 'restaurant']):
		output_file = self.dir_cache / 'osm_food.geojson'
		if not output_file.exists():
			gdf = gdf[gdf['amenity'].isin(keys)]
			logger.info(f"Number of Food facilities in BBOX= {gdf.shape[0]}")
			gdf.to_file(output_file, driver='GeoJSON')
			logger.info(f"OSM amenities FOOD is stored in: {output_file}")
		else:
			logger.info(f"Skipping Extraction. File exists {output_file}")


	# def extract_transport_station(self, gdf, keys=['bus_station']):
	# def extract_healthcare(self, gdf, keys=['']):
	# def extract_buildings(self, Polygon)



if __name__ == "__main__":
	db=OhsomeDB()
	gdf = db.extract_facilities(tag_key='shop')
	# db.extract_benches(gdf)
	print(gdf.shape)
