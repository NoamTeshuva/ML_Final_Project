import os
import pandas as pd
import pandas_ta as ta

def compute_indicators(df):
    df.ta.sma(length=50, append=True)  # 50-day SMA
    df.ta.sma(length=200, append=True)  # 200-day SMA
    df.ta.rsi(length=14, append=True)  # RSI
    df.ta.macd(append=True)  # MACD
    df["Volatility"] = df["Close"].rolling(window=20).std()  # 20-day rolling volatility
    df.dropna(inplace=True)  # Remove NaNs
    return df

if __name__ == "__main__":
    processed_path = "data/processed/"
    stock_files = [f for f in os.listdir(processed_path) if f.startswith("cleaned_")]

    if not stock_files:
        print("⚠️ No cleaned stock files found. Run `data_preprocessing.py` first!")
        exit()

    for file in stock_files:
        file_path = os.path.join(processed_path, file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        df = compute_indicators(df)

        output_file = file.replace("cleaned_", "featured_")
        df.to_csv(os.path.join(processed_path, output_file))
        print(f"✅ Features added to: {output_file}")
