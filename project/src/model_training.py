import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from joblib import dump

print("✅ Loading dataset...")
train_df = pd.read_csv("data/processed/training_data.csv", index_col=0, parse_dates=True)

# ✅ Reduce dataset to 2016-2020 for model training
print("✅ Limiting dataset to 2016-2020...")
train_df = train_df[train_df.index >= "2016-01-01"]

# Define features
feature_cols = ["SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "Volatility"]
target_col = "Close"

print("✅ Converting columns to numeric...")
train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
train_df.dropna(inplace=True)
print("✅ Data cleaned!")

X = train_df[feature_cols]
y = (train_df[target_col].shift(-1) > train_df[target_col]).astype(int)

print("✅ Splitting train-test sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
print("✅ Data split complete!")

# ✅ Ensure "models/" directory exists
if not os.path.exists("models"):
    os.makedirs("models")

print("✅ Training Decision Tree...")
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
print("✅ Decision Tree Training Complete!")

print("✅ Training Logistic Regression...")
log_model = LogisticRegression(max_iter=500).fit(X_train, y_train)
print("✅ Logistic Regression Training Complete!")

print("✅ Training AdaBoost...")
adaboost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50).fit(X_train, y_train)
print("✅ AdaBoost Training Complete!")

print("✅ Applying PCA...")
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
print("✅ PCA Applied!")

print("✅ Training K-Means Clustering...")
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train_pca)
print("✅ K-Means Clustering Complete!")

# ✅ Print K-Means results
print(f"K-Means Inertia: {kmeans.inertia_:.2f}")  # Measures cluster compactness
print(f"K-Means Cluster Centers:\n{kmeans.cluster_centers_}")  # Cluster centers
print(f"Example Stock Cluster Assignments:\n{kmeans.labels_[:10]}")  # Show first 10 stocks

# Save models
dump(dt_model, "models/dt_model.joblib")
dump(log_model, "models/log_model.joblib")
dump(adaboost_model, "models/adaboost_model.joblib")
dump(pca, "models/pca_model.joblib")
dump(kmeans, "models/kmeans_model.joblib")

print("✅ Model Training Completed!")

# ✅ Make Predictions
dt_pred = dt_model.predict(X_val)
log_pred = log_model.predict(X_val)
adaboost_pred = adaboost_model.predict(X_val)

# ✅ Print Accuracy Scores
print(f"Decision Tree Accuracy: {accuracy_score(y_val, dt_pred):.2%}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_val, log_pred):.2%}")
print(f"AdaBoost Accuracy: {accuracy_score(y_val, adaboost_pred):.2%}")
