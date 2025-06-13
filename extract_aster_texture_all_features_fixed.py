# extract_all_aster_texture_features.py
import os
from glob import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops  # Use the new function names!

# Set this to the root ASTER raster folder (it will search all subfolders)
input_dir = "1_data_raw/aster_mineral_maps/"
output_csv = "2_data_processed/aster_texture_features.csv"

def extract_texture_features(image, distances=[1], angles=[0], levels=256):
    """Extract GLCM texture features from an image."""
    if image.dtype != np.uint8:
        # Normalize and convert to uint8
        imin, imax = np.nanmin(image), np.nanmax(image)
        if imax == imin:
            return [np.nan] * 4  # All stats undefined
        image = ((image - imin) / (imax - imin) * 255).astype(np.uint8)
    # Fill NaNs with median or 0 (best effort)
    image = np.nan_to_num(image, nan=0)
    # GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    features = [
        float(graycoprops(glcm, 'contrast')[0, 0]),
        float(graycoprops(glcm, 'homogeneity')[0, 0]),
        float(graycoprops(glcm, 'energy')[0, 0]),
        float(graycoprops(glcm, 'correlation')[0, 0])
    ]
    return features

rows = []

# Recursively find all .tif files
tif_files = [y for x in os.walk(input_dir) for y in glob(os.path.join(x[0], '*.tif'))]
print(f"🔍 Found {len(tif_files)} raster files in {input_dir}")

for tif_path in tqdm(tif_files, desc="Extracting ASTER texture features"):
    try:
        with rasterio.open(tif_path) as src:
            # Use the first band, handle NaN and size
            band = src.read(1)
            if band.size == 0:
                continue
            file_name = os.path.basename(tif_path)
            parent_dir = os.path.basename(os.path.dirname(tif_path))
            group = parent_dir  # e.g., "1.Silica"
            mineral_index = file_name.split('.')[0]  # Remove extension

            # Extract GLCM features
            contrast, homogeneity, energy, correlation = extract_texture_features(band)
            rows.append({
                "file_name": file_name,
                "group": group,
                "mineral_index": mineral_index,
                "contrast": contrast,
                "homogeneity": homogeneity,
                "energy": energy,
                "correlation": correlation
            })
    except Exception as e:
        print(f"❌ Error processing {tif_path}: {e}")

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"✅ Saved texture features for {len(df)} rasters to {output_csv}")
