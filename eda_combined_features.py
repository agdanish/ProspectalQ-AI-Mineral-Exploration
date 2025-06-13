# Filename: eda_combined_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Cleaned Data
df = pd.read_csv('2_data_processed/combined_features_cleaned_for_ML.csv')

# 2. Class Distribution
print("Class Distribution (Label):")
print(df['Label'].value_counts())
sns.countplot(x='Label', data=df)
plt.title('Class Distribution')
plt.show()

# 3. Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 4. Feature Correlation Matrix (Top 20 by variance for readability)
top_var_cols = df.var().sort_values(ascending=False).head(20).index.tolist()
corr = df[top_var_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix (Top 20 Features)')
plt.show()

# 5. Feature Distributions (First 6 numeric features)
num_cols = df.select_dtypes(include='number').columns.drop('Label')
for col in num_cols[:6]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 6. Quick Feature Importance with Random Forest (Optional Preview)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Features by Random Forest Importance:")
print(importances.head(10))
importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.show()
