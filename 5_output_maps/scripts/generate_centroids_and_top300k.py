import geopandas as gpd

# === CONFIG: Update this path if needed ===
input_gpkg = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/clipped_grid_final_rebalanced.gpkg"

# === OUTPUT FILES ===
output_polygons = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_polygons_top300k.gpkg"
output_centroids = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_centroids_3lakh.gpkg"

# === LOAD DATA ===
gdf = gpd.read_file(input_gpkg)

# === VALIDATION ===
assert "Prospectivity_Level_6Class_v6" in gdf.columns, "Column 'Prospectivity_Level_6Class_v6' missing!"

# === CREATE CENTROIDS ===
centroids = gdf.copy()
centroids["geometry"] = centroids.geometry.centroid

# === FILTER FOR TOP-300K POLYGONS (exclude 'Very Low') ===
polygon_top = gdf[gdf["Prospectivity_Level_6Class_v6"] != "Very Low"]

# === EXPORT ===
centroids.to_file(output_centroids, driver="GPKG")
polygon_top.to_file(output_polygons, driver="GPKG")

print("✅ Exported centroids and top polygons successfully.")
