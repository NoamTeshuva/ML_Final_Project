import pandas as pd

# Load the CSV while skipping metadata rows (use skiprows=1 if needed)
df = pd.read_csv("../../data/processed/Russell2000.csv", index_col=0, parse_dates=True, skiprows=1)

print(df.head())
