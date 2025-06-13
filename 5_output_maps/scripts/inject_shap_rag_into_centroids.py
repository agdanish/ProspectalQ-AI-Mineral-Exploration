import geopandas as gpd
import os

# === CONFIG ===
centroid_path = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_centroids_3lakh.gpkg"
top300k_polygons_path = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_polygons_top300k.gpkg"
shap_folder = r"D:/GSI_Hackathon_Project/5_outputs_maps/SHAP/SHAP_plots"
output_path = r"D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/grid_centroids_3lakh_enriched.gpkg"

# === LOAD DATA ===
print("📥 Loading centroid and polygon data...")
centroids = gpd.read_file(centroid_path)
top_polygons = gpd.read_file(top300k_polygons_path)

# === GRID_ID SET FOR TOP-300K ===
top_ids = set(top_polygons["Grid_ID"].astype(str))
centroids["Grid_ID"] = centroids["Grid_ID"].astype(str)

# === INIT SHAP/RAG COLUMNS ===
centroids["SHAP_Insight"] = ""
centroids["RAG_Insight"] = ""

# === INJECT ONLY FOR TOP-300K IDs ===
print("🧠 Injecting SHAP and RAG insights for top 300K centroids...")
for idx, row in centroids.iterrows():
    grid_id = row["Grid_ID"]
    if grid_id not in top_ids:
        continue  # skip non-top300k

    shap_path = os.path.join(shap_folder, grid_id, "shap_explanation.txt")
    rag_path = os.path.join(shap_folder, grid_id, "rag_explanation.txt")

    shap_text = ""
    rag_text = ""

    if os.path.exists(shap_path):
        with open(shap_path, "r", encoding="utf-8") as f:
            shap_text = f.read().strip()

    if os.path.exists(rag_path):
        with open(rag_path, "r", encoding="utf-8") as f:
            rag_text = f.read().strip()

    centroids.at[idx, "SHAP_Insight"] = shap_text
    centroids.at[idx, "RAG_Insight"] = rag_text

# === SAVE NEW FILE ===
print("💾 Saving to:", output_path)
centroids.to_file(output_path, driver="GPKG")
print("✅ Done. SHAP + RAG added to centroids where available.")
