import os
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Suppress all warnings
warnings.filterwarnings("ignore")

# --- File paths ---
data_path = os.path.join("2_data_processed", "FINAL_clustered_features_v4.csv")
model_path = os.path.join("4_model_training", "FINAL_lgbm_model.txt")

# --- Load dataset ---
df = pd.read_csv(data_path)

# --- Identify and drop ID columns if present ---
id_cols = [col for col in ['Grid_ID', 'Centroid_X', 'Centroid_Y'] if col in df.columns]
if id_cols:
    df = df.drop(columns=id_cols)

# --- Detect binary label column ---
label_col = None
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if sorted(unique_vals) == [0, 1] or sorted(unique_vals) == [0.0, 1.0]:
        label_col = col
        break
if not label_col:
    raise ValueError("Binary label column not found.")

X = df.drop(columns=[label_col])
y_true = df[label_col].values

# --- Load the model pipeline ---
model = joblib.load(model_path)

# --- Predict ---
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# --- Evaluation ---
print("\nModel Evaluation on Full Dataset:")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
