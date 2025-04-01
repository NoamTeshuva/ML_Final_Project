import pandas as pd

# Load full dataset
df = pd.read_csv("data/processed/training_data.csv")

# Split the last 20% for testing
split_index = int(len(df) * 0.8)
df_test = df.iloc[split_index:]

# Save as testing data
df_test.to_csv("data/processed/testing_data.csv", index=False)
print("✅ Testing data generated successfully!")
