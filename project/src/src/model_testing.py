import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

# ✅ Set paths correctly
data_path = "C:/Users/HP/PycharmProjects/ML_final_project/data/selected_200/testing_data.csv"
model_dir = "C:/Users/HP/PycharmProjects/ML_final_project/models/"

# ✅ Load testing dataset
print("✅ Loading testing dataset...")
test_df = pd.read_csv(data_path)

# ✅ Define features and target
feature_cols = ["SMA_7", "SMA_14", "SMA_50", "SMA_200",
                "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "Volatility"]
target_col = "Target"  # ✅ Use precomputed target

print("✅ Converting columns to numeric...")
test_df[feature_cols] = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
test_df.dropna(inplace=True)  # ✅ Remove NaN values
print("✅ Data cleaned!")

# ✅ Extract Features (X) and Target (y)
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# ✅ Load trained models
print("✅ Loading trained models...")
log_model = load(model_dir + "log_model.joblib")
adaboost_model = load(model_dir + "adaboost_model.joblib")
pca = load(model_dir + "pca_model.joblib")
scaler = load(model_dir + "scaler.joblib")

# ✅ Apply Scaling & PCA
print("✅ Applying scaler and PCA transformation...")
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# ✅ Make Predictions
print("✅ Making predictions...")
log_pred_test = log_model.predict(X_test_pca)
adaboost_pred_test = adaboost_model.predict(X_test_pca)

# ✅ Evaluate models
print("✅ Model Testing Completed!")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred_test):.2%}")
print(f"AdaBoost Accuracy: {accuracy_score(y_test, adaboost_pred_test):.2%}")
