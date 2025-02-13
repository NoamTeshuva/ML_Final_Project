import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, silhouette_score
from joblib import dump
from imblearn.over_sampling import SMOTE

print("✅ Loading dataset...")
train_df = pd.read_csv("data/processed/training_data.csv", index_col=0, parse_dates=True)

# ✅ Ensure dataset is sorted before shifting target
train_df = train_df.sort_index()

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

# ✅ Check class distribution before SMOTE
print("Class Distribution Before SMOTE:")
print(y_train.value_counts())

# ✅ Ensure SMOTE is only applied if imbalance exists
class_ratio = y_train.value_counts(normalize=True)
if abs(class_ratio[0] - class_ratio[1]) < 0.05:  # If classes are within 5% of each other
    print("✅ Classes are already balanced. Skipping SMOTE.")
    X_train_balanced, y_train_balanced = X_train, y_train
else:
    print("✅ Applying SMOTE to balance classes...")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("✅ SMOTE applied successfully!")

# ✅ Reduce training data size to speed up computation
sample_size = 500_000
if len(X_train_balanced) > sample_size:
    X_train_balanced = X_train_balanced[:sample_size]
    y_train_balanced = y_train_balanced[:sample_size]
    print(f"✅ Training on a reduced dataset of {sample_size} samples.")

# ✅ PCA for dimensionality reduction
print("✅ Applying PCA...")
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_balanced)
X_val_pca = pca.transform(X_val)
print(f"✅ PCA Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.2%}")

# ✅ Ensure "models/" directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# ✅ Train Logistic Regression
print("✅ Training Logistic Regression...")
log_model = LogisticRegression(max_iter=500).fit(X_train_pca, y_train_balanced)
print("✅ Logistic Regression Training Complete!")

# ✅ Optimize AdaBoost using RandomizedSearchCV for faster tuning
print("✅ Optimizing AdaBoost...")
adaboost_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.5, 1.0, 1.5]}
adaboost_grid = RandomizedSearchCV(AdaBoostClassifier(), adaboost_params, cv=3, n_iter=3, random_state=42)
adaboost_grid.fit(X_train_pca, y_train_balanced)
best_adaboost = adaboost_grid.best_estimator_
print(f"✅ Best AdaBoost Params: {adaboost_grid.best_params_}")

# ✅ Reduce dataset size for K-Means clustering
sample_size_kmeans = 50_000
X_train_kmeans = X_train_pca[:sample_size_kmeans] if len(X_train_pca) > sample_size_kmeans else X_train_pca
print(f"✅ Running K-Means on {len(X_train_kmeans)} samples instead of full dataset.")

# ✅ Use MiniBatchKMeans for Faster Clustering
print("✅ Training MiniBatchKMeans Clustering...")
kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=10_000, n_init=10)
kmeans.fit(X_train_kmeans)
print("✅ MiniBatchKMeans Clustering Complete!")

# ✅ Save models
dump(log_model, "models/log_model.joblib")
dump(best_adaboost, "models/adaboost_model.joblib")
dump(pca, "models/pca_model.joblib")
dump(kmeans, "models/kmeans_model.joblib")

print("✅ Model Training Completed!")

# ✅ Time-Series Cross Validation (to prevent data leakage)
print("✅ Running Time-Series Cross Validation...")
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_train_pca):
    X_train_fold, X_test_fold = X_train_pca[train_index], X_train_pca[test_index]
    y_train_fold, y_test_fold = y_train_balanced[train_index], y_train_balanced[test_index]

    # Train & evaluate models
    log_fold = LogisticRegression().fit(X_train_fold, y_train_fold)
    ada_fold = AdaBoostClassifier().fit(X_train_fold, y_train_fold)

    print(f"Fold Accuracy | Logistic Regression: {log_fold.score(X_test_fold, y_test_fold):.2%}, AdaBoost: {ada_fold.score(X_test_fold, y_test_fold):.2%}")

# ✅ Make Predictions on Holdout Set
log_pred = log_model.predict(X_val_pca)
adaboost_pred = best_adaboost.predict(X_val_pca)

# ✅ Print Accuracy Scores
print(f"Logistic Regression Accuracy: {accuracy_score(y_val, log_pred):.2%}")
print(f"AdaBoost Accuracy: {accuracy_score(y_val, adaboost_pred):.2%}")
# ✅ Evaluate K-Means Performance
print(f"✅ K-Means Inertia: {kmeans.inertia_:.2f}")  # Measures cluster compactness
print(f"✅ K-Means Silhouette Score: {silhouette_score(X_train_kmeans, kmeans.labels_):.2f}")  # Measures cluster separation
