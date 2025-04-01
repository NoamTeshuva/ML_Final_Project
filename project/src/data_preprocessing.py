import os
import pandas as pd
import random
from shutil import copyfile
from sklearn.preprocessing import MinMaxScaler

# ✅ Define directories
processed_dir = "C:/Users/HP/PycharmProjects/ML_final_project/data/processed/"
selected_dir = "C:/Users/HP/PycharmProjects/ML_final_project/data/selected_200/"

# ✅ Get all available CSV files from processed folder
all_files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]

# ✅ Select 200 random stocks
random.seed(42)  # Ensures reproducibility
selected_files = random.sample(all_files, 200)

print(f"📊 Selecting 200 stocks and saving to {selected_dir}")

# ✅ Ensure the selected_200 folder exists
if not os.path.exists(selected_dir):
    os.makedirs(selected_dir)

# ✅ Copy selected files to selected_200 directory
for file in selected_files:
    src_path = os.path.join(processed_dir, file)
    dest_path = os.path.join(selected_dir, file)
    copyfile(src_path, dest_path)

print(f"✅ 200 random stocks copied to {selected_dir}")

# ✅ Function to process stock data
def add_sma_columns(file):
    file_path = os.path.join(selected_dir, file)

    # ✅ Load existing stock data
    df = pd.read_csv(file_path, parse_dates=["Date"])

    # ✅ Ensure numeric formatting
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ✅ Compute SMA values
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    df["SMA_14"] = df["Close"].rolling(window=14).mean()

    # ✅ Compute Normalized Change if missing
    if "Normalized_Change" not in df.columns:
        df["Normalized_Change"] = (df["Close"] - df["Open"]) / df["Open"]

    # ✅ Drop NaN values after calculations
    df.dropna(inplace=True)

    # ✅ Normalize "Normalized_Change"
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["Normalized_Change"] = scaler.fit_transform(df[["Normalized_Change"]])

    # ✅ Save the updated file
    df.to_csv(file_path, index=False)
    print(f"✅ Processed & Saved: {file}")

# ✅ Process all selected stocks
for file in selected_files:
    add_sma_columns(file)

print(f"✅ SMA update completed for {len(selected_files)} stocks in {selected_dir}")
