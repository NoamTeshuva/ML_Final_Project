import pandas as pd
import os
import talib


def compute_indicators(df):
    """
    Compute technical indicators for stock data.

    Parameters:
        df (pd.DataFrame): Cleaned stock data.

    Returns:
        pd.DataFrame: Data with indicators.
    """
    df["SMA_50"] = talib.SMA(df["Close"], timeperiod=50)
    df["SMA_200"] = talib.SMA(df["Close"], timeperiod=200)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["MACD"], df["MACD_signal"], _ = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["Volatility"] = df["Close"].rolling(window=20).std()
    df.dropna(inplace=True)  # Drop rows with NaN values after indicator calculations
    return df


if __name__ == "__main__":
    # Process all cleaned stock files
    stock_files = [f for f in os.listdir("data/processed/") if f.startswith("cleaned_")]

    for file in stock_files:
        file_path = os.path.join("data", "processed", file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        df = compute_indicators(df)

        df.to_csv(f"data/processed/featured_{file.replace('cleaned_', '')}")
        print(f"Features added to: {file}")
