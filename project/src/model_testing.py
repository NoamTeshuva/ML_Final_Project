import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

print("✅ Loading testing dataset...")
test_df = pd.read_csv("data/processed/testing_data.csv", index_col=0, parse_dates=True)

# Define features and target
feature_cols = ["SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "Volatility"]
target_col = "Close"

print("✅ Converting columns to numeric...")
test_df[feature_cols] = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
test_df.dropna(inplace=True)
print("✅ Data cleaned!")

X_test = test_df[feature_cols]
y_test = (test_df[target_col].shift(-1) > test_df[target_col]).astype(int)

# ✅ Load trained models (excluding SVM)
print("✅ Loading trained models...")
dt_model = load("models/dt_model.joblib")
log_model = load("models/log_model.joblib")
adaboost_model = load("models/adaboost_model.joblib")

# ✅ Make Predictions
print("✅ Making predictions...")
dt_pred_test = dt_model.predict(X_test)
log_pred_test = log_model.predict(X_test)
adaboost_pred_test = adaboost_model.predict(X_test)

# ✅ Evaluate models
print("✅ Model Testing Completed")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred_test):.2%}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred_test):.2%}")
print(f"AdaBoost Accuracy: {accuracy_score(y_test, adaboost_pred_test):.2%}")
