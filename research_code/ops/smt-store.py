import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import requests
from tqdm import tqdm
import os
import uuid



def create_grid(input_geom, tile_width, image_size=(1716, 1436), overlap=0.0):
    pixel_width, pixel_height = image_size
    aspect_ratio = pixel_width / pixel_height
    tile_height = tile_width / aspect_ratio

    if isinstance(input_geom, gpd.GeoDataFrame):
        input_geom = input_geom.geometry
    elif isinstance(input_geom, list):
        input_geom = gpd.GeoSeries(input_geom, crs="EPSG:3857")

    bounds = input_geom.total_bounds
    minx, miny, maxx, maxy = bounds

    step_x = tile_width * (1 - overlap)
    step_y = tile_height * (1 - overlap)

    tiles = []
    fid = 0
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            tile = box(x, y, x + tile_width, y + tile_height)
            if input_geom.intersects(tile).any():
                lx, ly, ux, uy = tile.bounds
                bbox_str = f"{lx},{ly},{ux},{uy}"
                distM = ux - lx
                tiles.append({
                    'fid': fid,
                    'geohash': str(uuid.uuid4()),
                    'bbox': bbox_str,
                    'bbox_geom': tile,
                    'distM': distM
                })
                fid += 1
            x += step_x
        y += step_y

    df = pd.DataFrame(tiles)
    return gpd.GeoDataFrame(df, geometry='bbox_geom', crs='EPSG:3857')


class SketchMapToolClient:
    def __init__(self, base_url="https://test.sketch-map-tool.heigit.org", factor=0.227, image_size=(1716, 1436)):
        self.api_url = f"{base_url}/en/create/results"
        self.status_url = f"{base_url}/api/status"
        self.download_url = f"{base_url}/api/download"
        self.factor = factor
        self.image_width, self.image_height = image_size
        self.size_str = f'{{"width":{self.image_width},"height":{self.image_height}}}'

    def _submit_bbox(self, bbox_str, scale):
        form_data = {
            "format": "A4",
            "orientation": "landscape",
            "bbox": f"[{bbox_str}]",
            "bboxWGS84": "[0,0,0,0]",
            "size": self.size_str,
            "scale": str(scale),
            "layer": "OSM"
        }
        try:
            r = requests.post(self.api_url, data=form_data, allow_redirects=False)
            r.raise_for_status()
            return r.headers['location'].split("/")[4]
        except Exception as e:
            print(f"❌ Submission failed for bbox {bbox_str}: {e}")
            return "ERROR"

    def submit_from_grid(self, grid_df):
        uuids = []
        for _, row in grid_df.iterrows():
            uuid = self._submit_bbox(row['bbox'], row['distM'] * self.factor)
            uuids.append(uuid)

        grid_df['uuid'] = uuids
        grid_df['scale'] = grid_df['distM'] * self.factor
        grid_df['size'] = self.size_str
        return grid_df

    def check_status(self, uuid):
        url = f"{self.status_url}/{uuid}/sketch-map"
        try:
            r = requests.get(url)
            status = r.json().get("status")
            return status #r.status_code == 200
        except Exception as e:
            print(f"⚠️ Status check failed for UUID {uuid}: {e}")
            return False

    def download_map(self, uuid, output_path):
        url = f"{self.download_url}/{uuid}/sketch-map"
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded: {output_path}")
        except Exception as e:
            print(f"❌ Download failed for UUID {uuid}: {e}")

def generate_and_download_sketches_roi(city_name, roi_path, csv_path, tile_width, grid_path, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    # Initialize SketchMapToolClient
    smt = SketchMapToolClient()

    # === Step 1: Create grid ===
    if not os.path.exists(grid_path):
        # === Load and prepare geometry ===
        gdf = gpd.read_file(roi_path)
        if gdf.crs is None or gdf.crs.to_string() != 'EPSG:3857':
            gdf = gdf.to_crs('EPSG:3857')
        if gdf.empty:
            raise ValueError("GeoDataFrame is empty.")
        city_geom = gpd.GeoSeries(gdf.unary_union, crs=gdf.crs)

        grid_df = create_grid(city_geom, tile_width=tile_width, overlap=0.2)

        os.makedirs("cache", exist_ok=True)
        grid_df.to_file(grid_path, driver='GPKG')
        print(f"🗺️ Grid saved to {grid_path} ({len(grid_df)} tiles)")
    else:
        grid_df = gpd.read_file(grid_path)
        print(f"🗺️ Loaded existing grid from {grid_path} ({len(grid_df)} tiles)")
    # === Step 2: Submit tiles to SMT ===
    if not os.path.exists(csv_path):
        submitted_df = smt.submit_from_grid(grid_df)
        submitted_df.to_csv(csv_path, sep=";", index=False)
        print(f"📄 Submissions saved to {csv_path}")
    else:
        submitted_df = pd.read_csv(csv_path, sep=";")
        print(f"📄 Loaded existing submissions from {csv_path} ({len(submitted_df)} entries)")

    # === Step 3: Check status and download ===
    print("\n🔄 Checking status and downloading maps...")
    resave = False
    cnt = 0
    process_cnt = 0
    for idx, row in tqdm(submitted_df.iterrows(), total=len(submitted_df), desc="Processing submissions"):
        uuid = row['uuid']
        fid = row['fid']
        bbox = row['bbox']
        scale = row['scale']
        out_path = os.path.join(download_dir, f"{city_name}_{uuid}.pdf")

        if os.path.exists(out_path):
            # print(f"✅ Already exists, skipping: {out_path}")
            continue

        if uuid == "ERROR":
            # print(f"⚠️ Skipping fid {fid} due to previous submission error.")
            continue

        status = smt.check_status(uuid)

        if status == "SUCCESS":
            smt.download_map(uuid, out_path)
        elif status in ("FAILURE", "PENDING", "ERROR", None):
            # print(f"🔁 Resubmitting map for UUID: {uuid} due to status: {status}")
            new_uuid = smt._submit_bbox(bbox, scale)
            submitted_df.at[idx, 'uuid'] = new_uuid  # Update UUID in the DataFrame
            resave = True
            cnt += 1
        else:
            process_cnt += 1
            print(f"⏳ UUID: {uuid} is {status} ")

    # Save updated UUIDs if any were changed
    if resave:
        submitted_df.to_csv(csv_path, sep=";", index=False)
        # print(f"\n💾 Updated submission file saved to {csv_path}: {cnt} UUIDs resubmitted.")
        grid_df = grid_df.drop(columns=['uuid'], errors='ignore')
        merged_df = grid_df.merge(submitted_df[['geohash', 'uuid']], on='geohash', how='left')
        merged_df.to_file(grid_path, driver='GPKG')
        # print(f"New uuid and geohash ids saved to {grid_path}")
    # else:
    #     print("\n✅ No UUIDs needed resubmission.")

    print(f"\n📊 Summary: {len(submitted_df)} total submissions: "
          f"\n {len(submitted_df) - (process_cnt+cnt)} downloaded,"
          f"\n {process_cnt} still processing,"
          f"\n {cnt} resubmissions made.")


if __name__ == "__main__":
    # Define paths and parameters
    city_name = 'kayseri'
    roi_path = f'data/{city_name}_roi.gpkg'
    csv_path = f"cache/smt_{city_name}_submitted.csv"
    tile_width = 2000
    grid_path = f'cache/{city_name}_grid_3857_{tile_width}.gpkg'
    download_dir = f"cache/smt_{city_name}"

    # Run the main function
    generate_and_download_sketches_roi(city_name, roi_path, csv_path, tile_width, grid_path, download_dir)
