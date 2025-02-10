import os
import pandas as pd

def load_stock_data(ticker):
    """
    Loads the CSV data for a given ticker from the processed data folder.
    """
    file_path = os.path.join("data", "processed", f"{ticker}.csv")

    # Read CSV while skipping metadata rows
    df = pd.read_csv(file_path, skiprows=2, index_col=0, parse_dates=True)

    # Rename columns properly
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Ensure all numeric columns are properly formatted
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)  # Remove any rows with NaN values
    return df

if __name__ == "__main__":
    stock_files = [f for f in os.listdir("data/processed/") if f.endswith(".csv")]
    tickers = [file.replace(".csv", "") for file in stock_files]

    for ticker in tickers:
        df = load_stock_data(ticker)
        if df is not None:
            df.to_csv(f"data/processed/cleaned_{ticker}.csv")
            print(f"✅ Processed: {ticker}")
