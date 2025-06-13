
import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm

# Paths
grid_path = "2_data_processed/grid_with_labels.shp"
spec_geotiff_dir = "1_data_raw/spectrometric_data/GRIDS/GEOTIFF"
output_path = "2_data_processed/spectrometric_features.csv"

# Load grid
grid = gpd.read_file(grid_path)
grid = grid.to_crs("EPSG:4326")

# Spectrometric bands to extract
spec_files = {
    "Dose_rate_TC": "Dose_rate_TC.tiff",
    "eTh_ppm": "eTh_ppm.tiff",
    "eU_ppm": "eU_ppm.tiff",
    "K_perc": "K_perc.tiff"
}

# Initialize output DataFrame
features = pd.DataFrame({"Grid_ID": grid["Grid_ID"]})

# Iterate over bands
for band_name, filename in spec_files.items():
    raster_path = os.path.join(spec_geotiff_dir, filename)
    with rasterio.open(raster_path) as src:
        stats = []
        for geom in tqdm(grid.geometry, desc=f"Extracting {band_name}"):
            try:
                out_image, _ = mask(src, [mapping(geom)], crop=True)
                data = out_image[0]
                data = data[data != src.nodata]
                mean_val = data.mean() if data.size > 0 else None
            except Exception:
                mean_val = None
            stats.append(mean_val)
        features[band_name] = stats

# Save
features.to_csv(output_path, index=False)
print(f"✅ Spectrometric features extracted and saved to: {output_path}")
