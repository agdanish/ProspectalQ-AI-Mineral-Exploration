import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# Input paths
grid_path = "2_data_processed/grid_with_labels.shp"
geochronology_dirs = {
    "k_ar": "1_data_raw/geochronology_map/k_ar_geochronology_20250224140807718/k_ar_geochronology_20250224140807718.shp",
    "rb_sr": "1_data_raw/geochronology_map/rb_sr_geochronology_20250224140807718/rb_sr_geochronology_20250224140807718.shp",
    "sm_nd": "1_data_raw/geochronology_map/sm_nd_geochronology_20250224140807718/sm_nd_geochronology_20250224140807718.shp",
    "u_pb": "1_data_raw/geochronology_map/u_pb_geochronology_20250224140807718/u_pb_geochronology_20250224140807718.shp"
}

# Load labeled grid
grid = gpd.read_file(grid_path)
output_df = grid[["Grid_ID"]].copy()

# Extract feature: count of points falling inside each grid cell
for method, shp_path in tqdm(geochronology_dirs.items(), desc="Processing Geochronology"):
    try:
        gdf = gpd.read_file(shp_path)
        joined = gpd.sjoin(grid, gdf, how="left", predicate="intersects")
        count_series = joined.groupby("Grid_ID").size()
        output_df[method + "_count"] = output_df["Grid_ID"].map(count_series).fillna(0).astype(int)
    except Exception as e:
        print(f"❌ Error processing {shp_path}: {e}")
        output_df[method + "_count"] = 0

# Save to CSV
output_df.to_csv("2_data_processed/geochronology_features.csv", index=False)
print("✅ Geochronology features saved to: 2_data_processed/geochronology_features.csv")
