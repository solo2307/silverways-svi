
import numpy as np
import glob
import rasterio
from rasterio.transform import from_origin
from rasterio.merge import merge
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
from typing import List, Optional
def list_xyz_files(directory):
    """List all .xyz files in the given directory/subdirectories."""
    return glob.glob(f"{directory}/**/*.xyz", recursive=True)

def select_xyz_roi( files:List[str], roi_file:Optional[str]=None, tile_file:Optional[str]=None)->List[str]:
    selected_files = []
    if roi_file is None and tile_file is None:
        selected_files = files
    if roi_file is None and tile_file is not None:
        gdf_tile = gpd.read_file(tile_file)
        dopkachel_ids = gdf_tile['dop_kachel']
        tile_ids = [f"dgm1_32_{dop[2:5]}_{dop[-4:]}_1_bw_2022.xyz" for dop in dopkachel_ids]
        # Build lookup of {basename: fullpath}
        file_lookup = {Path(f).name: f for f in files}
        selected_files = [file_lookup[t] for t in tile_ids if t in file_lookup]
    else:
        gdf_roi = gpd.read_file(roi_file)
        if gdf_roi.crs != "28532":
            gdf_roi = gdf_roi.to_crs("EPSG:25832")
        minx, miny, maxx, maxy  = gdf_roi.total_bounds

        for f in files:
            fname = Path(f).name
            fminx = float(fname.split("_")[2])*1000
            fminy = float(fname.split("_")[3])*1000
            fmaxx = fminx + 1000
            fmaxy = fminy + 1000

            # Check if the file's bounding box is fully inside the ROI
            if (fminx >= minx and fmaxx <= maxx and fminy >= miny and fmaxy <= maxy):
                selected_files.append(f)
    return selected_files


def convert_xyz_to_tif(xyz_file, out_file, crs="EPSG:25832", tile_size=1000):
    data = np.loadtxt(xyz_file)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # Compute grid spacing
    dx = np.round(np.min(np.diff(np.unique(x))))
    dy = np.round(np.min(np.diff(np.unique(y))))

    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()

    # Start with NoData grid
    z_grid = np.full((tile_size, tile_size), -9999.0, dtype="float32")

    # Indices
    x_index = ((x - minx) / dx).astype(int)
    y_index = ((maxy - y) / dy).astype(int)
    x_index = np.clip(x_index, 0, tile_size - 1)
    y_index = np.clip(y_index, 0, tile_size - 1)
    z_grid[y_index, x_index] = z

    transform = from_origin(minx, maxy, dx, dy)

    with rasterio.open(
            out_file, "w",
            driver="GTiff",
            height=tile_size,
            width=tile_size,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=-9999.0,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER"
    ) as dst:
        dst.write(z_grid, 1)

    # print(f"✅ Wrote {out_file} with gaps filled as NoData")
    return out_file

def generate_dem(output_tif = "cache/mannheim_dem.tif"):
    # Collect all XYZ files
    files = list_xyz_files("/Volumes/ygrin/silverways/mannheim/dgmdom1")
    selected_files = select_xyz_roi(tile_file="data/opengeodata_mannheim.gpkg",
                                    files=files)
    # XYZ files into GeoTIFF
    if len(selected_files) == 0:
        print("No files to merge!")
    else:
        geotiffs = []
        for f in tqdm(selected_files, desc="Converting XYZ to GeoTIFF", total=len(selected_files)):
            out_tif = f"/Volumes/ygrin/silverways/mannheim/dem1/{Path(f).stem}.tif"
            if not Path(out_tif).exists():  # skip if already done
                convert_xyz_to_tif(f, out_tif)
            geotiffs.append(out_tif)

        # Merge GeoTIFFs
        srcs = [rasterio.open(fp) for fp in geotiffs]
        mosaic, out_transform = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "LZW",
            "tiled": True,
            "BIGTIFF": "IF_SAFER"
        })
        with rasterio.open(output_tif, "w", **meta) as dest:
            dest.write(mosaic)
        print("✅ Saved Mannheim_DEM.tif (optimized for speed and size)")

if __name__ == "__main__":
    generate_dem()

