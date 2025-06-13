import pandas as pd

# Load the updated dataset
df = pd.read_csv("final_features_v4_with_predictions.csv")

# Define thresholds (you can adjust these if needed)
def assign_prospectivity(prob):
    if prob >= 0.90:
        return "Very High"
    elif prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Moderate"
    elif prob >= 0.10:
        return "Low"
    else:
        return "Very Low"

# Apply prospectivity level
df["Prospectivity_Level"] = df["Predicted_Prob"].apply(assign_prospectivity)

# Save back to the same file
df.to_csv("final_features_v4_with_predictions.csv", index=False)
print("✅ Added 'Prospectivity_Level' to final_features_v4_with_predictions.csv")
