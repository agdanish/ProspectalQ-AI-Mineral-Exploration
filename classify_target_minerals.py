import pandas as pd

# Paths
classified_path = r"D:\GSI_Hackathon_Project\2_data_processed\FINAL_post_features_v4.csv"
geochem_labels_path = r"D:\GSI_Hackathon_Project\2_data_processed\geochem_labeled_minerals.csv"
output_path = r"D:\GSI_Hackathon_Project\2_data_processed\FINAL_target_mineral_labeled.csv"

# Load datasets
classified_df = pd.read_csv(classified_path)
geochem_df = pd.read_csv(geochem_labels_path)

# Merge Target_Mineral_Label from geochem labels
merged_df = pd.merge(
    classified_df.drop(columns=["Target_Mineral_Label"], errors='ignore'),
    geochem_df[["Grid_ID", "Target_Mineral_Label"]],
    on="Grid_ID",
    how="left"
)

# Fill empty with "None"
merged_df["Target_Mineral_Label"] = merged_df["Target_Mineral_Label"].fillna("None")

# Save output
merged_df.to_csv(output_path, index=False)
print("✅ Final mineral labels saved to:", output_path)
