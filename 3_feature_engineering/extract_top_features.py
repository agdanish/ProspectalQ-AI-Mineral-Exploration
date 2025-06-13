import pandas as pd
import lightgbm as lgb
import numpy as np

# Load data
data = pd.read_csv("2_data_processed/final_features_v3.csv")
features = [col for col in data.columns if col not in ["Grid_ID", "Label"]]
X = data[features].replace(-9999, np.nan)
y = data["Label"]

# Optuna best parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',
    'verbosity': -1,
    'learning_rate': 0.07632645543107966,
    'num_leaves': 35,
    'max_depth': 7,
    'min_data_in_leaf': 59,
    'feature_fraction': 0.6541837079575539,
    'bagging_fraction': 0.7962450640055287,
    'bagging_freq': 9,
    'min_child_samples': 20,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum()
}

# Train model
dtrain = lgb.Dataset(X, label=y)
model = lgb.train(params, dtrain, num_boost_round=451)

# Extract top 20 features
importance = model.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
top_feats = feat_imp.sort_values(by='Importance', ascending=False).head(20)

# Save
top_feats.to_csv("2_data_processed/top_20_features_lightgbm.csv", index=False)
print("✅ Top 20 feature list saved to 2_data_processed/top_20_features_lightgbm.csv")
