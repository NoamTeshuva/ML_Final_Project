import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data
train_df = pd.read_csv("data/processed/training_data.csv", index_col=0, parse_dates=True)

# Reduce dataset size for faster training
train_df = train_df.sample(frac=0.2, random_state=42)  # Use 20% of data

# Define features (matching actual column names)
feature_cols = ["SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "Volatility"]
target_col = "Close"

# Ensure all columns are numeric
train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
train_df.dropna(inplace=True)

X = train_df[feature_cols]
y = (train_df[target_col].shift(-1) > train_df[target_col]).astype(int)  # 1 = Price Increase, 0 = Decrease

# Train-test split within 2015-2020
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train models with reduced complexity
svm_model = SVC(kernel="rbf", probability=True).fit(X_train, y_train)
dt_model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)  # Limit depth
log_model = LogisticRegression(max_iter=300).fit(X_train, y_train)

# Evaluate models
svm_pred = svm_model.predict(X_val)
dt_pred = dt_model.predict(X_val)
log_pred = log_model.predict(X_val)

print("✅ Model Training Completed")
print(f"SVM Accuracy: {accuracy_score(y_val, svm_pred):.2%}")
print(f"Decision Tree Accuracy: {accuracy_score(y_val, dt_pred):.2%}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_val, log_pred):.2%}")
