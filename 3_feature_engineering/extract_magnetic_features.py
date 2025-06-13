
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import pandas as pd
from shapely.geometry import mapping
from tqdm import tqdm

# Paths
grid_path = "D:/GSI_Hackathon_Project/2_data_processed/grid_with_labels.shp"
raster_path = "D:/GSI_Hackathon_Project/1_data_raw/magnetic_data/GRIDS/GEOTIFF/TAIL_TMI_GE.tiff"
output_csv = "D:/GSI_Hackathon_Project/2_data_processed/magnetic_features.csv"

# Load the grid shapefile
grid = gpd.read_file(grid_path)
grid = grid.to_crs("EPSG:4326")  # Ensure grid is in same CRS as raster

# Open the raster
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    if grid.crs != raster_crs:
        grid = grid.to_crs(raster_crs)

    features = []
    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="Extracting magnetic features"):
        try:
            geom = [mapping(row['geometry'])]
            out_image, out_transform = mask(src, geom, crop=True)
            data = out_image[0]
            data = data[data != src.nodata]
            mean_val = data.mean() if data.size > 0 else None
        except Exception:
            mean_val = None
        features.append(mean_val)

grid["magnetic_mean"] = features
grid[["Grid_ID", "Label", "magnetic_mean"]].to_csv(output_csv, index=False)
print(f"✅ Magnetic features extracted and saved to: {output_csv}")
