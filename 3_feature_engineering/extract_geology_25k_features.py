
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Load the labeled grid
grid_path = Path("2_data_processed/grid_with_labels.shp")
grid = gpd.read_file(grid_path)

# Shapefile folders inside geology_25k
folders = [
    "dyke_line_25k_ngdr_20250224140917945",
    "fault_25k_ngdr_20250224140917945",
    "fold_25k_ngdr_20250224140917945",
    "lithology_25k_ngdr_20250224140917945",
    "mine_quarry_25k_ngdr_20250224140917945",
    "mineralization_25k_ngdr_20250224141143411",
    "oriented_structure_line_25k_ngdr_20250224141143411",
    "oriented_structure_plane_25k_ngdr_20250224141143411",
    "shear_zone_25k_ngdr_20250224141143411"
]

# Standardize names for output features
output_names = [
    "dyke_line", "fault", "fold", "lithology", "mine_quarry",
    "mineralization", "oriented_line", "oriented_plane", "shear_zone"
]

# Root path to 25K geology
base_path = Path("1_data_raw/geology_25k")

for folder, feature_name in tqdm(zip(folders, output_names), total=len(folders), desc="Processing Geology 25K"):
    try:
        folder_path = base_path / folder
        shp_files = list(folder_path.glob("*.shp"))
        if not shp_files:
            print(f"⚠️ File not found: {folder_path}")
            continue

        gdf = gpd.read_file(shp_files[0])
        gdf = gdf.to_crs(grid.crs)

        joined = gpd.sjoin(grid, gdf, how="left", predicate="intersects")

        if "index_left" in joined.columns:
            count_series = joined.groupby("index_left").size()
            count = count_series.reindex(range(len(grid))).fillna(0).astype(int).values
        else:
            count = np.zeros(len(grid), dtype=int)

        grid[f"{feature_name}_count"] = count

    except Exception as e:
        print(f"❌ Error processing {folder_path / shp_files[0].name}: {e}")

# Save the extracted features
grid.drop(columns=["geometry"], errors="ignore").to_csv("2_data_processed/geology_25k_features.csv", index=False)
print("✅ Geological 25K features saved to: 2_data_processed/geology_25k_features.csv")
