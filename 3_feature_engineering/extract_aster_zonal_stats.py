import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio

# ---- CONFIG ----
grid_path = "2_data_processed/grid_with_labels.shp"
texture_csv = "2_data_processed/aster_texture_features.csv"
raster_base_dir = "1_data_raw/aster_mineral_maps/AST_05T TIR"

# ---- 1. Load Grid ----
grid_gdf = gpd.read_file(grid_path)
print("✅ Grid loaded:", grid_path)
print("Grid CRS:", grid_gdf.crs)
print("Grid bounds:", grid_gdf.total_bounds)

# ---- 2. Load Texture Features ----
texture_df = pd.read_csv(texture_csv)
print("✅ Texture features loaded:", texture_csv)
print("Texture CSV rows:", len(texture_df))

# ---- 3. Recursively Find All .tif Files ----
raster_files = []
for root, dirs, files in os.walk(raster_base_dir):
    for f in files:
        if f.lower().endswith(".tif"):
            raster_files.append(os.path.join(root, f))
print("\n--- Raster File Sample ---")
print(raster_files[:5])
print("Total raster files found:", len(raster_files))

# ---- 4. Print Raster CRS and Bounds ----
if len(raster_files) > 0:
    with rasterio.open(raster_files[0]) as src:
        print("Sample raster:", raster_files[0])
        print("Raster CRS:", src.crs)
        print("Raster bounds:", src.bounds)
else:
    print("❌ No raster files found!")
    exit()

# ---- 5. [Optional] Try Extracting Centroids (Debug) ----
centroids = []
for rf in raster_files:
    try:
        with rasterio.open(rf) as src:
            bounds = src.bounds
            cx = (bounds.left + bounds.right) / 2
            cy = (bounds.top + bounds.bottom) / 2
            centroids.append(Point(cx, cy))
    except Exception as e:
        print(f"❌ Error opening {rf}: {e}")
print("Sample centroid:", centroids[0] if centroids else "None")
print("Total centroids computed:", len(centroids))

# ---- 6. [OPTIONAL] Project centroids to grid CRS if needed ----
# If CRS mismatch: you'll need to transform these points! We'll fix after debug.

# ---- 7. STOP HERE for Debugging ----
print("\n--- DEBUG COMPLETE ---")
print("If any above numbers or paths look wrong, copy-paste your output here.")

# (Next step: We will do spatial join & aggregation after debugging)
