import pandas as pd

df = pd.read_csv("data/processed/training_data.csv", index_col=0, parse_dates=True)
y = (df["Close"].shift(-1) > df["Close"]).astype(int)

print("Class Distribution in Training Data:")
print(y.value_counts(normalize=True) * 100)
