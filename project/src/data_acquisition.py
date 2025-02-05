import yfinance as yf
import datetime
import os
import pandas as pd


def fetch_historical_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker using yfinance.

    Parameters:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with historical stock data.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


def save_data_to_csv(dataframe, ticker):
    """
    Save the fetched DataFrame to a CSV file in the processed data directory.

    Parameters:
        dataframe (pd.DataFrame): The data to save.
        ticker (str): The ticker symbol (used to name the file).
    """
    # Define the directory path (project root/data/processed)
    directory = os.path.join(os.getcwd(), "data", "processed")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{ticker}.csv")
    dataframe.to_csv(filepath)
    print(f"Data for {ticker} saved to {filepath}")


if __name__ == "__main__":
    # Define the 10-year date range
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * 10)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Example ticker (replace with a small-cap ticker of your choice)
    ticker = "AAPL"  # For demonstration; ideally, use a small-cap stock symbol.

    # Fetch the data and display a preview
    df = fetch_historical_data(ticker, start_date_str, end_date_str)
    print(df.head())

    # Save the data to CSV for further processing
    save_data_to_csv(df, ticker)
