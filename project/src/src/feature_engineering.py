import os
import pandas as pd
import pandas_ta as ta

def compute_indicators(df):
    """
    Computes technical indicators for stock data.
    """
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-day SMA
    df["SMA_200"] = df["Close"].rolling(window=200).mean()  # 200-day SMA
    df["Volatility"] = df["Close"].rolling(window=20).std()  # 20-day rolling volatility

    # ✅ Add RSI
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # ✅ Add MACD (3 columns: MACD, MACD Signal, MACD Histogram)
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    df.dropna(inplace=True)  # Remove NaN values created by rolling calculations
    return df

if __name__ == "__main__":
    processed_path = "C:/Users/HP/PycharmProjects/ML_final_project/data/selected_200/"
    stock_files = [f for f in os.listdir(processed_path) if f.endswith(".csv")]

    if not stock_files:
        print("⚠️ No cleaned stock files found in selected_200/. Run `data_preprocessing.py` first!")
        exit()

    print(f"📊 Adding features to {len(stock_files)} selected stocks...")

    for file in stock_files:
        file_path = os.path.join(processed_path, file)
        df = pd.read_csv(file_path, parse_dates=["Date"])

        # ✅ Compute indicators
        df = compute_indicators(df)

        # ✅ Save the updated file
        output_file = file.replace(".csv", "_featured.csv")
        df.to_csv(os.path.join(processed_path, output_file), index=False)
        print(f"✅ Features added to: {output_file}")

    print("✅ Feature engineering completed for all selected stocks!")
