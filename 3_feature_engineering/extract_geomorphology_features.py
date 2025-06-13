import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon

# Paths
grid_path = "2_data_processed/grid_with_labels.shp"
geomorphology_path = "1_data_raw/geomorphology_map/geomorphology_250k_gcs_ngdr.shp"
output_path = "2_data_processed/geomorphology_features.csv"

# Load grid
print("🔹 Loading grid...")
grid = gpd.read_file(grid_path)
grid = grid.to_crs(epsg=4326)

# Load geomorphology shapefile
print("🔹 Loading geomorphology map...")
geo_gdf = gpd.read_file(geomorphology_path)
geo_gdf = geo_gdf.to_crs(epsg=4326)

# Prepare output DataFrame
output_df = grid[["Grid_ID"]].copy()
output_df["num_geomorpho_units"] = 0

print("🚀 Performing spatial join...")
joined = gpd.sjoin(grid, geo_gdf, how="left", predicate="intersects")
counts = joined.groupby("Grid_ID").size()
output_df["num_geomorpho_units"] = output_df["Grid_ID"].map(counts).fillna(0).astype(int)

# Save
output_df.to_csv(output_path, index=False)
print(f"✅ Geomorphology features saved to: {output_path}")
