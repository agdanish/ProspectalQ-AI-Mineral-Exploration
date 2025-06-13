import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# File paths
shapefile_path = "2_data_processed/grid_with_true_labels.shp"
features_csv_path = "2_data_processed/FINAL_sorted_features_v5.csv"
output_path = "5_outputs_maps/leafmap/clipped_grid_full_enriched.geojson"

# Load data
print("📥 Loading full grid shapefile...")
gdf = gpd.read_file(shapefile_path)

print("📥 Loading full model feature CSV...")
df = pd.read_csv(features_csv_path)

print(f"🔹 Grid geometries: {gdf.shape}, Feature rows: {df.shape}")

# Ensure consistent ID types
gdf["Grid_ID"] = gdf["Grid_ID"].astype(str)
df["Grid_ID"] = df["Grid_ID"].astype(str)

# Merge
print("🔗 Merging on 'Grid_ID' using vectorized join...")
merged = gdf.merge(df, on="Grid_ID", how="left")

print("🔍 Checking grid rows with valid model output...")
matched = 0
for is_valid in tqdm(merged["Predicted_Prob"].notna(), total=len(merged), desc="Validating"):
    if is_valid:
        matched += 1

print(f"✅ {matched:,} / {len(merged):,} grid cells enriched with model output")


# Reproject to EPSG:4326 for Folium
print("🌍 Reprojecting to EPSG:4326...")
merged = merged.to_crs(epsg=4326)

# Save
print(f"💾 Saving to: {output_path}")
merged.to_file(output_path, driver="GeoJSON")
print("✅ GeoJSON saved successfully!")
