
import os
import glob
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm

# === Paths ===
grid_fp = r"D:/GSI_Hackathon_Project/2_data_processed/grid_with_labels.shp"
aster_root = r"D:/GSI_Hackathon_Project/1_data_raw/aster_mineral_maps"
output_csv = r"D:/GSI_Hackathon_Project/2_data_processed/aster_features.csv"

# === ASTER Subfolders ===
aster_subfolders = [
    "AST_05T TIR/1.Silica",
    "AST_05T TIR/2.Gypsum",
    "AST_05T TIR/3.Quartz",
    "AST_07XT VNIR+SWIR/1. AlOH group composition(B5,B7)",
    "AST_07XT VNIR+SWIR/2. Ferrous iron index (B5_B4)",
    "AST_07XT VNIR+SWIR/3. Opaque index (B1_B4)",
    "AST_07XT VNIR+SWIR/4. Ferric oxide content (B4_B3)",
    "AST_07XT VNIR+SWIR/5. FeOH group content (B6+B8_B7)",
    "AST_07XT VNIR+SWIR/6. Ferric oxide composition (B2_B1)",
    "AST_07XT VNIR+SWIR/7. Kaolin group index (B6_B5)",
    "AST_07XT VNIR+SWIR/8. AlOH group content (B5+B7_B6)",
    "AST_07XT VNIR+SWIR/9. MgOH group content (B6+B9_B7+B8)",
    "AST_07XT VNIR+SWIR/10. Ferrous iron content in MgOH_Carbonate (B5_B4)",
    "AST_07XT VNIR+SWIR/11. MgOH group composition (B7_B8)"
]

# === Load grid ===
grid = gpd.read_file(grid_fp).to_crs("EPSG:32643")
features = []

# === Start processing ===
for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="Extracting ASTER features"):
    geom = [mapping(row['geometry'])]
    feature_row = {'Grid_ID': row['Grid_ID'] if 'Grid_ID' in row else idx}

    for folder in aster_subfolders:
        folder_path = os.path.join(aster_root, folder)
        tif_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
        values = []

        for tif_fp in tif_files:
            try:
                with rasterio.open(tif_fp) as src:
                    # Check extent intersection before attempting mask
                    grid_bounds = gpd.GeoSeries(row['geometry']).total_bounds
                    raster_bounds = src.bounds
                    if (
                        raster_bounds.right < grid_bounds[0] or
                        raster_bounds.left > grid_bounds[2] or
                        raster_bounds.top < grid_bounds[1] or
                        raster_bounds.bottom > grid_bounds[3]
                    ):
                        continue  # No overlap, skip

                    out_image, _ = mask(src, geom, crop=True)
                    data = out_image[0]
                    masked_data = data[data != src.nodata]
                    if masked_data.size > 0:
                        values.append(np.mean(masked_data))
            except Exception as e:
                continue  # Skip failed rasters

        feature_row[folder.replace("/", "_")] = np.mean(values) if values else np.nan
    features.append(feature_row)

# === Save to CSV ===
df = pd.DataFrame(features)
df.to_csv(output_csv, index=False)
print(f"✅ ASTER features saved to {output_csv}")
