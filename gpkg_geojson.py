import geopandas as gpd

# === Input and Output Paths ===
input_gpkg = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_centroids_3lakh_enriched.gpkg"
top300k_polygons = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_polygons_top300k.gpkg"
output_geojson = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_centroids_300k.geojson"

# === Load Data ===
centroids = gpd.read_file(input_gpkg)
top_polygons = gpd.read_file(top300k_polygons)

# === Ensure matching Grid_ID type ===
centroids["Grid_ID"] = centroids["Grid_ID"].astype(str)
top_polygons["Grid_ID"] = top_polygons["Grid_ID"].astype(str)

# === Filter centroids for Top-300K Grid_IDs ===
top_ids = set(top_polygons["Grid_ID"])
filtered_centroids = centroids[centroids["Grid_ID"].isin(top_ids)]

# === Export to GeoJSON ===
filtered_centroids.to_file(output_geojson, driver="GeoJSON")
print("✅ grid_centroids_300k.geojson exported successfully.")
