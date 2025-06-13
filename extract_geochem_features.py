import geopandas as gpd
import pandas as pd

# Paths
geochem_path = "D:/GSI_Hackathon_Project/1_data_raw/geochem_ngcm/stream_sediments_gcs_ngdr_20250221140319808/stream_sediments_gcs_ngdr.shp"
grid_path = "D:/GSI_Hackathon_Project/2_data_processed/grid_with_labels.shp"
output_path = "D:/GSI_Hackathon_Project/2_data_processed/geochem_features.csv"

# Read files
geochem = gpd.read_file(geochem_path).to_crs("EPSG:4326")
grid = gpd.read_file(grid_path).to_crs("EPSG:4326")

# Spatial join
joined = gpd.sjoin(grid, geochem, how="left", predicate="intersects")

# Select relevant columns
selected_cols = ["Grid_ID", "Label", "geometry"] + [col for col in joined.columns if col not in ["index_right", "Grid_ID", "Label", "geometry"]]
geochem_features = joined[selected_cols]

# Save to CSV
geochem_features.to_csv(output_path, index=False)
print("✅ Geochemical features extracted and saved to:", output_path)