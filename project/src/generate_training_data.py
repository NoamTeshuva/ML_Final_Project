import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ Set the directory where `_featured.csv` files are stored
data_dir = "C:/Users/HP/PycharmProjects/ML_final_project/data/selected_200/"

# ✅ Get list of all `_featured.csv` files
stock_files = [f for f in os.listdir(data_dir) if f.endswith("_featured.csv")]

# ✅ Initialize empty list to store all stock data
all_data = []

print(f"📂 Found {len(stock_files)} `_featured.csv` files to process.")

for file in stock_files:
    file_path = os.path.join(data_dir, file)
    print(f"📊 Processing: {file}")

    # ✅ Read stock data
    df = pd.read_csv(file_path)

    # ✅ Ensure Date is properly formatted
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ✅ Check if the file is empty or missing required columns
    if df.empty or "Close" not in df.columns:
        print(f"⚠️ Warning: {file} is empty or missing required columns!")
        continue  # Skip this file

    # ✅ Add Ticker column
    df["Ticker"] = file.replace("_featured.csv", "")

    # ✅ Define the Prediction Target (Binary Classification)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # ✅ Remove last row (because shift(-1) creates a NaN value)
    df.dropna(inplace=True)

    # ✅ Select relevant columns
    selected_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "SMA_7", "SMA_14", "SMA_50", "SMA_200",
                        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "Volatility", "Ticker", "Target"]

    # ✅ Ensure all columns exist before selection
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ Skipping {file} due to missing columns: {missing_columns}")
        continue  # Skip this file

    df = df[selected_columns]
    all_data.append(df)

print(f"✅ Processed {len(all_data)} files.")

# ✅ Combine all stock data into a single dataset
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Total rows in dataset: {full_df.shape[0]}")

    # ✅ Split into training (80%) and testing (20%) sets
    train_df, test_df = train_test_split(full_df, test_size=0.2, shuffle=True, random_state=42)

    # ✅ Ensure the output directory exists
    output_dir = "C:/Users/HP/PycharmProjects/ML_final_project/data/selected_200/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ✅ Save datasets
    train_df.to_csv(os.path.join(output_dir, "training_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "testing_data.csv"), index=False)

    print("✅ `training_data.csv` and `testing_data.csv` created successfully!")
else:
    print("❌ No valid stock data found! Please check your stock CSV files.")
