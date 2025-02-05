import os
import pandas as pd


def load_stock_data(ticker):
    """
    Loads the CSV data for a given ticker from the processed data folder.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Stock data DataFrame.
    """
    file_path = os.path.join("data", "processed", f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def preprocess_stock_data(df):
    """
    Cleans stock data by handling missing values.

    Parameters:
        df (pd.DataFrame): Raw stock data.

    Returns:
        pd.DataFrame: Cleaned stock data.
    """
    df = df.fillna(method='ffill')  # Forward fill missing values
    df = df.dropna()  # Drop rows that are still missing
    return df


if __name__ == "__main__":
    # List all CSV files in data/processed/
    stock_files = [f for f in os.listdir("data/processed/") if f.endswith(".csv")]
    tickers = [file.replace(".csv", "") for file in stock_files]

    for ticker in tickers:
        df = load_stock_data(ticker)
        if df is not None:
            df_clean = preprocess_stock_data(df)
            df_clean.to_csv(f"data/processed/cleaned_{ticker}.csv")
            print(f"Processed: {ticker}")
