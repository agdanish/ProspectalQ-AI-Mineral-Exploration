import geopandas as gpd
import os

# Paths
grid_path = "D:/GSI_Hackathon_Project/2_data_processed/grid_with_labels.shp"
lineament_dir = "D:/GSI_Hackathon_Project/1_data_raw/lineaments"

# Load grid
grid = gpd.read_file(grid_path)
print("Loaded grid:", grid.shape)

# All shapefiles in the lineament folder
shapefiles = [f for f in os.listdir(lineament_dir) if f.endswith(".shp")]

for shp in shapefiles:
    feature_path = os.path.join(lineament_dir, shp)
    feature_name = os.path.splitext(shp)[0]
    
    print(f"\nProcessing {shp}...")
    
    # Read shapefile
    try:
        df = gpd.read_file(feature_path)
        if df.crs != grid.crs:
            df = df.to_crs(grid.crs)
        
        # Spatial join
        joined = gpd.sjoin(grid, df, how="left", predicate="intersects")

        if "index_left" in joined.columns:
            count_series = joined.groupby("index_left").size().reindex(range(len(grid))).fillna(0).astype(int)
        else:
            count_series = [0] * len(grid)

        grid[feature_name + "_count"] = count_series

    except Exception as e:
        print(f"❌ Failed to process {shp}: {e}")
        grid[feature_name + "_count"] = 0

# Save final output
output_path = "D:/GSI_Hackathon_Project/2_data_processed/lineament_features.csv"
grid.to_csv(output_path, index=False)
print(f"\n✅ Lineament features extracted and saved to: {output_path}")
