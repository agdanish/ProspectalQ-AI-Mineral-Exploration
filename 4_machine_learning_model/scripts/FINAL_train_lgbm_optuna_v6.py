import os
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Load dataset
DATA_PATH = "FINAL_features_v5.csv"
data = pd.read_csv(DATA_PATH)
target_col = "Label"
features = [col for col in data.columns if col not in ["Grid_ID", target_col]]

# Replace -9999 with NaN and drop columns with >60% missing values
X = data[features].replace(-9999, np.nan)
missing_percent = X.isna().mean() * 100
X = X.loc[:, missing_percent <= 60]
print(f"🧹 Cleaned feature set shape: {X.shape}")
print(f"🚫 Dropped {len(missing_percent[missing_percent > 60])} columns with >60% missing values")

X = X.select_dtypes(include=[np.number])
y = data[target_col]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# Impute and apply SMOTE with small k_neighbors
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

sm = SMOTE(sampling_strategy=0.4, random_state=SEED, k_neighbors=3)
X_res, y_res = sm.fit_resample(X_train_imputed, y_train)

# Optuna objective function
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'verbosity': -1,
        'is_unbalance': True,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': 20,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 10)
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(
        lgb.Dataset(X_val, label=y_val), "auc"
    )

    try:
        model = lgb.train(
            params,
            lgb.Dataset(X_res, label=y_res),
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100),
                pruning_callback
            ]
        )
        preds = model.predict(X_val)
        return roc_auc_score(y_val, preds)
    except:
        return float("-inf")

# Run Optuna tuning
study = optuna.create_study(direction="maximize")
for _ in tqdm(range(20), desc="Safe Optuna Trials"):
    study.optimize(objective, n_trials=1, catch=(Exception,))

print("\n✅ Best parameters found by Optuna:")
print(study.best_params)

# Retrain with best parameters
best_model = lgb.train(
    {**study.best_params, 'objective': 'binary', 'metric': 'auc', 'device': 'gpu', 'verbosity': -1, 'min_child_samples': 20, 'is_unbalance': True},
    lgb.Dataset(X_res, label=y_res),
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    num_boost_round=3000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150),
        lgb.log_evaluation(period=100)
    ]
)

# Evaluation
y_pred_prob = best_model.predict(X_val)
y_pred = (y_pred_prob >= 0.1).astype(int)

print("\n📊 Final Evaluation with Cleaned Features:")
print("ROC-AUC:", roc_auc_score(y_val, y_pred_prob))
print("F1 Score:", f1_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))

# Save feature importance
lgb.plot_importance(best_model, max_num_features=20, importance_type='gain')
plt.tight_layout()
plt.savefig("2_data_processed/lgbm_features_v6.png")
print("✅ Feature importance plot saved to: 2_data_processed/lgbm_features_v6")

# Save model
os.makedirs("4_model_training", exist_ok=True)
best_model.save_model("4_model_training/final_lgbm_model_v6.txt")
print("✅ Model saved to: 4_model_training/final_lgbm_model_v6.txt")

# ➕ Run SHAP explanation script
print("\n⚙️ Running SHAP explanation generator...")
os.system("python SHAP_plots.py")
