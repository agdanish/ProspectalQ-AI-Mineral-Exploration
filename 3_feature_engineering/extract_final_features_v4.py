import pandas as pd
import os

# Input file paths
features_path = "2_data_processed/final_features_v3.csv"
dem_path = "2_data_processed/dem_features.csv"
output_path = "2_data_processed/final_features_v4.csv"

# Load datasets
features_df = pd.read_csv(features_path)
dem_df = pd.read_csv(dem_path)

# Merge on Grid_ID
merged_df = features_df.merge(dem_df, on="Grid_ID", how="left")

# Save the final output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_csv(output_path, index=False)

print(f"✅ Merged DEM features into: {output_path}")
