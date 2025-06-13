import geopandas as gpd
import rasterio
import rasterio.mask
import pandas as pd
import os
from shapely.geometry import mapping
from tqdm import tqdm

# Paths
grid_path = "2_data_processed/grid_with_labels.shp"
raster_path = "1_data_raw/ground_gravity_data/Ground_gravity_asci_grid_geotiff/GEOTIFF/NGPM_BA.tiff"
output_csv = "2_data_processed/ground_gravity_features.csv"

# Load grid
grid = gpd.read_file(grid_path)
grid = grid.to_crs("EPSG:4326")

# Extract mean gravity values from the raster
features = []

with rasterio.open(raster_path) as src:
    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="Extracting ground gravity"):
        geom = [mapping(row["geometry"])]
        try:
            out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
            data = out_image[0]
            data = data[data != src.nodata]
            mean_val = data.mean() if data.size > 0 else None
        except Exception:
            mean_val = None
        features.append(mean_val)

# Add to DataFrame
grid["ground_gravity_mean"] = features

# Save
grid[["Grid_ID", "Label", "ground_gravity_mean"]].to_csv(output_csv, index=False)
print(f"✅ Ground gravity features extracted and saved to: {output_csv}")