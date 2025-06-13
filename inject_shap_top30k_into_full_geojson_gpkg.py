import geopandas as gpd
import pandas as pd
import json
import os
import re
from tqdm import tqdm

# === FILE PATHS ===
geojson_input = "D:/GSI_Hackathon_Project/5_outputs_maps/leafmap/clipped_grid_full_enriched.geojson"
shap_folder = "D:/GSI_Hackathon_Project/5_outputs_maps/SHAP/SHAP_plots"
rag_metadata_path = "6_rag_pipeline/rag_chunks_metadata_pdf_cleaned.json"
output_path = "5_outputs_maps/leafmap/clipped_grid_full_enriched_final.geojson"

# === LOAD INPUT GEOJSON ===
print("📥 Loading enriched GeoJSON...")
gdf = gpd.read_file(geojson_input)
gdf["Grid_ID"] = gdf["Grid_ID"].astype(str)
print("📊 Loaded:", len(gdf), "grids")

# === PARSE SHAP TXT FILES ===
print("🧠 Extracting SHAP top-3 features from TXT files...")
shap_records = []

for folder in tqdm(os.listdir(shap_folder), desc="🔍 Parsing SHAP folders"):
    shap_txt_path = os.path.join(shap_folder, folder, "shap_explanation.txt")
    if os.path.isfile(shap_txt_path):
        with open(shap_txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            top_feats = [line.strip() for line in lines if re.match(r"^\d+\.\s", line)]
            record = {"Grid_ID": folder}
            for i in range(min(3, len(top_feats))):
                parts = re.split(r"[=\(\)]", top_feats[i])
                if len(parts) >= 4:
                    record[f"Top_Feature_{i+1}"] = parts[1].strip()
                    record[f"Value_{i+1}"] = parts[2].strip()
                    record[f"Direction_{i+1}"] = "INCREASES" if "INCREASES" in top_feats[i] else "DECREASES"
                    record[f"Impact_{i+1}"] = re.findall(r"[-+]?\d*\.\d+|\d+", parts[3])[0]
            shap_records.append(record)

# Merge all SHAP records per Grid_ID
shap_df = pd.DataFrame(shap_records)
print(f"✅ SHAP parsed for {len(shap_df)} grids.")

# === LOAD RAG METADATA ===
print("📥 Loading RAG metadata...")
with open(rag_metadata_path, "r", encoding="utf-8") as f:
    rag_metadata = json.load(f)
supported_keywords = set(chunk['keyword'].lower() for chunk in rag_metadata if 'keyword' in chunk)

# === MERGE SHAP TO GEOJSON ===
print("🔗 Injecting SHAP features...")
merged = gdf.merge(shap_df, on="Grid_ID", how="left")

# === INJECT RAG FOR TOP 30K ===
print("📚 Injecting RAG for top-30K...")
merged["RAG_Insight"] = ""
merged["RAG_Source"] = ""

top30k = merged.nlargest(30000, "Predicted_Prob")

for idx in tqdm(top30k.index, desc="📖 RAG"):
    row = merged.loc[idx]
    mineral = str(row.get("Target_Mineral", "")).lower().strip()
    if mineral and mineral in supported_keywords:
        merged.at[idx, "RAG_Insight"] = f"This area shows {mineral}-related alterations. Refer GSI report section on {mineral}."
        merged.at[idx, "RAG_Source"] = "GSI_Tech_Report_REE_2022.pdf"
    else:
        merged.at[idx, "RAG_Insight"] = "RAG not available for this grid."
        merged.at[idx, "RAG_Source"] = "N/A"

# === SAVE OUTPUTS ===
print("🌍 Reprojecting to EPSG:4326...")
merged = merged.to_crs(epsg=4326)

print(f"💾 Saving enriched GeoJSON to: {output_path}")
merged.to_file(output_path, driver="GeoJSON")

gpkg_path = output_path.replace(".geojson", ".gpkg")
print(f"💾 Also saving GPKG to: {gpkg_path}")
merged.to_file(gpkg_path, driver="GPKG")

print("✅ Final SHAP + RAG enriched GeoJSON and GPKG saved.")
