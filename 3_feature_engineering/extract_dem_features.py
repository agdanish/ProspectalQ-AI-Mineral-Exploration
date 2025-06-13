import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import os
from tqdm import tqdm

# Input paths
GRID_PATH = "2_data_processed/grid_with_true_labels.shp"
DEM_PATH = "1_data_raw/merged_dem.tif"
OUTPUT_CSV = "2_data_processed/dem_features.csv"

# Load grid
grid = gpd.read_file(GRID_PATH)
grid = grid.to_crs("EPSG:4326")  # Optional: match raster CRS if needed

# Compute zonal stats with progress bar
print("📊 Extracting mean elevation from DEM...")
stats = []
for row in tqdm(grid.itertuples(), total=len(grid), desc="Processing Grid Cells"):
    result = zonal_stats(
        vectors=[row.geometry],
        raster=DEM_PATH,
        stats=["mean"],
        nodata=-9999
    )[0]
    stats.append(result["mean"])

# Create DataFrame with results
grid["DEM_Elevation_Mean"] = stats
output_df = grid[["Grid_ID", "DEM_Elevation_Mean"]]

# Save result
os.makedirs("2_data_processed", exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ DEM feature saved to: {OUTPUT_CSV}")
