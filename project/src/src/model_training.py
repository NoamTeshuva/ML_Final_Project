import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
from joblib import dump
from imblearn.over_sampling import SMOTE

# ✅ Load dataset
print("✅ Loading dataset...")
train_df = pd.read_csv("C:/Users/HP/PycharmProjects/ML_final_project/data/processed/training_data.csv")

# ✅ Sort dataset before applying time-based transformations
train_df = train_df.sort_values(by="Date")

# ✅ Define Features & Target
feature_cols = ["SMA_7", "SMA_14", "SMA_50", "SMA_200", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "Volatility"]
target_col = "Target"

print("✅ Converting columns to numeric...")
train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
train_df.dropna(inplace=True)  # Remove NaNs

# ✅ Define X (features) and y (target)
X = train_df[feature_cols]
y = train_df[target_col]

print("✅ Splitting train-test sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
print("✅ Data split complete!")

# ✅ Check class balance before SMOTE
print("Class Distribution Before SMOTE:")
print(y_train.value_counts())

# ✅ Apply SMOTE only if imbalance is significant
if min(y_train.value_counts(normalize=True)) < 0.4:
    print("✅ Applying SMOTE to balance classes...")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("✅ SMOTE applied successfully!")
else:
    print("✅ Classes are balanced. Skipping SMOTE.")

# ✅ Standardize data before PCA
print("✅ Standardizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ✅ Ensure "models/" directory exists before saving models
models_dir = "C:/Users/HP/PycharmProjects/ML_final_project/models/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# ✅ Save the trained scaler
dump(scaler, os.path.join(models_dir, "scaler.joblib"))
print("✅ Scaler saved successfully!")

# ✅ Apply PCA (Keep 99% variance)
print("✅ Applying PCA...")
pca = PCA(n_components=0.99, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
print(f"✅ PCA Explained Variance: {sum(pca.explained_variance_ratio_):.2%}")

# ✅ Save PCA model
dump(pca, os.path.join(models_dir, "pca_model.joblib"))

# ✅ Optimize Logistic Regression
print("✅ Training Logistic Regression...")
log_reg_params = {'C': [0.01, 0.1, 1, 10, 100]}
grid_log = GridSearchCV(LogisticRegression(max_iter=2000), log_reg_params, cv=5)
grid_log.fit(X_train_pca, y_train)
best_log_model = grid_log.best_estimator_
print(f"✅ Best Logistic Regression Params: {grid_log.best_params_}")

# ✅ Train AdaBoost Model
print("✅ Training AdaBoost...")
best_adaboost = AdaBoostClassifier(n_estimators=75, learning_rate=1.5, random_state=42)
best_adaboost.fit(X_train_pca, y_train)
print("✅ AdaBoost Training Complete!")

# ✅ Train K-Means Clustering
print("✅ Training K-Means Clustering...")
kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=10_000, n_init=10)
kmeans.fit(X_train_pca)
print("✅ K-Means Training Complete!")

# ✅ Save models
dump(best_log_model, os.path.join(models_dir, "log_model.joblib"))
dump(best_adaboost, os.path.join(models_dir, "adaboost_model.joblib"))
dump(kmeans, os.path.join(models_dir, "kmeans_model.joblib"))
print("✅ Model Training Completed!")

# ✅ Time-Series Cross Validation
print("✅ Running Time-Series Cross Validation...")
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_train_pca):
    X_train_fold, X_test_fold = X_train_pca[train_index], X_train_pca[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Train & evaluate models
    log_fold = LogisticRegression().fit(X_train_fold, y_train_fold)
    ada_fold = AdaBoostClassifier().fit(X_train_fold, y_train_fold)

    print(f"Fold Accuracy | Logistic: {log_fold.score(X_test_fold, y_test_fold):.2%}, AdaBoost: {ada_fold.score(X_test_fold, y_test_fold):.2%}")

# ✅ Make Predictions on Holdout Set
log_pred = best_log_model.predict(X_val_pca)
adaboost_pred = best_adaboost.predict(X_val_pca)

# ✅ Print Accuracy Scores
print(f"Logistic Regression Accuracy: {accuracy_score(y_val, log_pred):.2%}")
print(f"AdaBoost Accuracy: {accuracy_score(y_val, adaboost_pred):.2%}")

# ✅ Evaluate K-Means Performance
print(f"✅ K-Means Inertia: {kmeans.inertia_:.2e}")
print(f"✅ K-Means Silhouette Score: {silhouette_score(X_train_pca, kmeans.labels_):.2f}")
print(f"✅ K-Means Cluster Centers (First 3 Rows):\n{kmeans.cluster_centers_[:3]}")
