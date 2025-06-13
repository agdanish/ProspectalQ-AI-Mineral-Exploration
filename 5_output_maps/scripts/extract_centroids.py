import geopandas as gpd
import pandas as pd

# Load shapefile
gdf = gpd.read_file("D:/GSI_Hackathon_Project/2_data_processed/grid_with_true_labels.shp")

# Extract centroids
gdf["Centroid_X"] = gdf.geometry.centroid.x
gdf["Centroid_Y"] = gdf.geometry.centroid.y

# Keep only needed columns
coords_df = gdf[["Grid_ID", "Centroid_X", "Centroid_Y"]]

# Save for merge
coords_df.to_csv("D:/GSI_Hackathon_Project/2_data_processed/grid_centroids.csv", index=False)

df_feat = pd.read_csv("D:/GSI_Hackathon_Project/2_data_processed/FINAL_sorted_features_v4.csv")
df_coords = pd.read_csv("D:/GSI_Hackathon_Project/2_data_processed/grid_centroids.csv")

df_merged = df_feat.merge(df_coords, on="Grid_ID")
df_merged.to_csv("D:/GSI_Hackathon_Project/2_data_processed/final_with_coords.csv", index=False)
