import pandas as pd
from glob import glob

# Paths
processed_path = "2_data_processed/"
output_path = f"{processed_path}combined_features.csv"

# List of feature files (excluding known non-feature files if any)
feature_files = sorted(glob(f"{processed_path}/*_features.csv"))
label_file = f"{processed_path}/grid_with_labels.csv"

# Load the label file
print("🔹 Loading grid labels...")
df_master = pd.read_csv(label_file)
df_master["Grid_ID"] = df_master["Grid_ID"].astype(str).str.strip()

# Merge each feature file
print("🔄 Merging feature files...")
for fpath in feature_files:
    if "grid_with_labels" in fpath:
        continue  # skip label file itself

    print(f"  ➤ Merging: {fpath}")
    df = pd.read_csv(fpath)
    df["Grid_ID"] = df["Grid_ID"].astype(str).str.strip()

    # Drop duplicate columns if they exist (e.g., 'Label')
    cols_to_drop = [col for col in df.columns if col in df_master.columns and col != "Grid_ID"]
    df = df.drop(columns=cols_to_drop)

    # Merge
    df_master = df_master.merge(df, on="Grid_ID", how="left")

# Fill missing values
print("🛠️ Handling missing values...")
for col in df_master.columns:
    if df_master[col].dtype == object:
        df_master[col] = df_master[col].fillna("Unknown").astype(str).str.strip()
    else:
        df_master[col] = df_master[col].fillna(0)

# Save merged output
print(f"💾 Saving merged features to: {output_path}")
df_master.to_csv(output_path, index=False)
print("✅ Merge complete. Combined feature file is ready.")
