import pandas as pd

# Input and output paths
input_path = "2_data_processed/final_features_v4.csv"
output_path = "2_data_processed/FINAL_sorted_features_v4.csv"

# Load data
df = pd.read_csv(input_path)

# Ensure Predicted_Prob column exists
if "Predicted_Prob" not in df.columns:
    raise ValueError("❌ Column 'Predicted_Prob' not found in the input file.")

# Sort by Predicted_Prob descending
df_sorted = df.sort_values("Predicted_Prob", ascending=False)

# Save to new file
df_sorted.to_csv(output_path, index=False)
print(f"✅ Sorted Top-K file saved as: {output_path}")
