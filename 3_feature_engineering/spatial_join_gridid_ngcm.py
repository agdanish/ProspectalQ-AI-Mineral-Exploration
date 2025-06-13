import geopandas as gpd
import pandas as pd
import os

# === Paths ===
grid_path = r"D:\GSI_Hackathon_Project\2_data_processed\grid_with_true_labels.shp"
geochem_path = r"D:\GSI_Hackathon_Project\1_data_raw\geochem_ngcm\stream_sediments_gcs_ngdr_20250221140319808\stream_sediments_gcs_ngdr_with_labels.shp"
output_csv = r"D:\GSI_Hackathon_Project\2_data_processed\geochem_labeled_minerals.csv"

print("🔄 Loading grid and geochem shapefiles...")
grid = gpd.read_file(grid_path)
geochem = gpd.read_file(geochem_path)

# Reproject to a projected CRS (e.g., UTM Zone 43N or any suitable for India)
projected_crs = "EPSG:32643"  # UTM Zone 43N for Karnataka/AP region
grid = grid.to_crs(projected_crs)
geochem = geochem.to_crs(projected_crs)

# Detect correct truncated label column
label_col = [col for col in geochem.columns if col.lower().startswith("target")][0]
print(f"🧠 Found mineral label column: {label_col}")

# Only keep rows with valid mineral label
geochem = geochem[geochem[label_col].str.lower() != "none"]

# Add slight buffer around geochem points for safety
geochem["geometry"] = geochem.geometry.buffer(50)  # 50 meters

# Spatial join (points to grid)
joined = gpd.sjoin(geochem, grid, how="inner", predicate="intersects")

# Use string Grid_IDs directly — do not convert to int
summary = (
    joined.groupby("Grid_ID")[label_col]
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else "None")
    .reset_index()
)

summary = summary.dropna(subset=["Grid_ID"])
summary.rename(columns={label_col: "Target_Mineral"}, inplace=True)

print(f"📊 Matched rows: {len(summary)} / {len(grid)}")
print(f"💾 Saving labeled minerals to: {output_csv}")
summary.to_csv(output_csv, index=False)
print("✅ Done. You can now merge this with predictions for 3D visualization.")
