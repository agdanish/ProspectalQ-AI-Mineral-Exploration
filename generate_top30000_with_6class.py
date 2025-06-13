import pandas as pd
import numpy as np
import lightgbm as lgb
import json

# === Paths ===
model_path = "4_model_training/final_lgbm_model_v6.txt"
features_path = "2_data_processed/FINAL_features_v5.csv"
mapping_path = "4_model_training/feature_mapping_v6.json"
output_labels = "2_data_processed/labels_top30000_v6.csv"
output_sorted = "2_data_processed/FINAL_sorted_features_v5.csv"

# === Load model
model = lgb.Booster(model_file=model_path)

# === Load and clean features
full_data = pd.read_csv(features_path)
features = [col for col in full_data.columns if col not in ["Grid_ID", "Label", "Prospectivity_Level"]]
X = full_data[features].replace(-9999, np.nan)

# === Drop columns with >60% missing
missing_percent = X.isna().mean() * 100
X = X.loc[:, missing_percent <= 60]

# === Keep only numerical columns
X = X.select_dtypes(include=[np.number])

# === Rename columns using saved mapping
with open(mapping_path, "r") as f:
    col_map = json.load(f)
X_renamed = X.rename(columns=col_map)

# === Subset to model features only
X_final = X_renamed[model.feature_name()]
full_data = full_data.loc[X_final.index]
full_data["Predicted_Prob"] = model.predict(X_final)

# === Assign 6-Class Prospectivity Labels
def assign_6class(prob):
    if prob >= 0.85:
        return "Extremely High"
    elif prob >= 0.70:
        return "Very High"
    elif prob >= 0.50:
        return "High"
    elif prob >= 0.30:
        return "Moderate"
    elif prob >= 0.10:
        return "Low"
    else:
        return "Very Low"

full_data["Prospectivity_Level_6Class"] = full_data["Predicted_Prob"].apply(assign_6class)

# === Save sorted predictions and Top-30,000
sorted_df = full_data.sort_values("Predicted_Prob", ascending=False)
sorted_df.to_csv(output_sorted, index=False)
sorted_df.head(30000).to_csv(output_labels, index=False)

print(f"✅ Saved Top 30,000 to: {output_labels}")
print(f"📊 Full sorted grid list with 6 classes: {output_sorted}")
