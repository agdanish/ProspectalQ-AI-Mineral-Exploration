import os
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from tqdm import tqdm

# --- Config ---
grid_path = "2_data_processed/grid_with_true_labels.shp"
aster_root = "1_data_raw/aster_mineral_maps"
output_csv = "2_data_processed/aster_texture_features_with_grid.csv"

# --- Load Grid ---
print("📦 Loading grid...")
grid_gdf = gpd.read_file(grid_path)

# --- Recursively scan for .tif files ---
print("🔍 Scanning ASTER rasters...")
raster_paths = []
for root, _, files in os.walk(aster_root):
    for file in files:
        if file.endswith(".tif") and not file.endswith(".ovr"):
            raster_paths.append(os.path.join(root, file))

# --- Extract features for each raster ---
all_stats = []

print(f"🛰️ Processing {len(raster_paths)} raster files...\n")
for raster_path in tqdm(raster_paths):
    try:
        stats = zonal_stats(
            vectors=grid_gdf,
            raster=raster_path,
            stats=["mean", "std", "min", "max"],
            geojson_out=True,
            nodata=-9999
        )
        basename = os.path.splitext(os.path.basename(raster_path))[0]
        for s in stats:
            s['properties'][f"{basename}_mean"] = s['properties'].pop('mean', None)
            s['properties'][f"{basename}_std"] = s['properties'].pop('std', None)
            s['properties'][f"{basename}_min"] = s['properties'].pop('min', None)
            s['properties'][f"{basename}_max"] = s['properties'].pop('max', None)
            all_stats.append(s['properties'])

    except Exception as e:
        print(f"⚠️ Skipped {raster_path}: {e}")

# --- Convert to DataFrame ---
print("\n🧪 Converting to DataFrame...")
df = pd.DataFrame(all_stats)

# --- Drop duplicate Grid_ID entries, keeping latest ---
df = df.groupby("Grid_ID").last().reset_index()

# --- Save output ---
df.to_csv(output_csv, index=False)
print(f"\n✅ ASTER texture features saved to: {output_csv}")
