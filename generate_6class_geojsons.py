import pandas as pd
import geopandas as gpd
import os

# === Paths ===
input_csv = "2_data_processed/FINAL_sorted_features_v5.csv"
grid_shapefile = "2_data_processed/grid_with_true_labels.shp"
output_dir = "5_outputs_maps/leafmap_6class_layers"
os.makedirs(output_dir, exist_ok=True)

# === Load data
df = pd.read_csv(input_csv)
gdf = gpd.read_file(grid_shapefile)

# === Merge on Grid_ID
df["Grid_ID"] = df["Grid_ID"].astype(str)
gdf["Grid_ID"] = gdf["Grid_ID"].astype(str)
merged = gdf.merge(df[["Grid_ID", "Predicted_Prob"]], on="Grid_ID", how="left")
merged = merged.to_crs(epsg=4326)

# === Assign 6-class prospectivity levels (final logic)
def assign_level(prob):
    if prob >= 0.85:
        return "Extremely High"
    elif prob >= 0.70:
        return "Very High"
    elif prob >= 0.50:
        return "High"
    elif prob >= 0.30:
        return "Moderate"
    elif prob >= 0.10:
        return "Low"
    else:
        return "Very Low"

merged["Prospectivity_Level_6Class"] = merged["Predicted_Prob"].apply(assign_level)

# === Filter and export per class
classes = ["Extremely High", "Very High", "High", "Moderate", "Low", "Very Low"]
for level in classes:
    subset = merged[merged["Prospectivity_Level_6Class"] == level]
    out_path = os.path.join(output_dir, f"{level.replace(' ', '_').lower()}.geojson")
    subset.to_file(out_path, driver="GeoJSON")
    print(f"✅ Saved {level}: {out_path}")

print("\n🎉 All 6 geojson layers generated successfully for Leafmap!")
