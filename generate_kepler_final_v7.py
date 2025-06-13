import geopandas as gpd
import pandas as pd
from keplergl import KeplerGl
import json

# === INPUT FILES ===
centroids_path = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_centroids_3lakh_enriched.gpkg"
polygons_path = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_polygons_top300k.gpkg"
config_path = "kepler_config_3layer_final_v2.json"
output_html = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/kepler_final_v7.html"

# === LOAD CONFIG ===
with open(config_path, "r") as f:
    config = json.load(f)

# === LOAD GEO DATA ===
centroids = gpd.read_file(centroids_path).to_crs(epsg=4326)
polygons = gpd.read_file(polygons_path).to_crs(epsg=4326)

# === ADD COORDINATES ===
centroids["lon"] = centroids.geometry.x
centroids["lat"] = centroids.geometry.y
polygons["lon"] = polygons.geometry.centroid.x
polygons["lat"] = polygons.geometry.centroid.y

# === SELECT RELEVANT FIELDS ===
popup_centroids = [
    "Grid_ID", "Prospectivity_Level_6Class_v6", "Contains_Critical_Mineral_Target", "Predicted_Prob",
    "DEM_Elevation_Mean", "SHAP_Insight", "RAG_Insight", "lon", "lat"
]
popup_polygons = [
    "Grid_ID", "Prospectivity_Level_6Class_v6", "Predicted_Prob", "lon", "lat"
]

centroids = centroids[popup_centroids + ["geometry"]]
polygons = polygons[popup_polygons + ["geometry"]]

# === INITIALIZE AND ADD DATA ===
kmap = KeplerGl(height=800, config=config)

# Add Centroids Layer (3 lakh grids) with 6 class color coding
kmap.add_data(data=centroids, name="Centroids_3L")

# Add Polygons Layer (Top 300K grids) for polygons
kmap.add_data(data=polygons, name="Polygons_Top300K")

# === EXPORT FINAL HTML ===
kmap.save_to_html(file_name=output_html)
print("✅ Kepler map saved to:", output_html)
