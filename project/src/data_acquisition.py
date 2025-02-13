import yfinance as yf
import pandas as pd
import os

# Load tickers from Russell 2000 list
tickers = pd.read_csv("data/iwm_holdings_cleaned.csv")["Ticker"].tolist()

# Track failed tickers
failed_tickers = []

# Create directory if not exists
data_dir = "data/processed/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# ✅ Fetch data for each Russell 2000 ticker
for ticker in tickers:
    try:
        df = yf.download(ticker, start="2015-02-15", end="2025-02-12")

        # Skip empty data
        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        # ✅ Reset index to ensure "Date" is a column (not an index)
        df.reset_index(inplace=True)

        # ✅ Keep only required columns
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # ✅ Save CSV file without extra headers
        df.to_csv(f"{data_dir}/{ticker}.csv", index=False)
        print(f"✅ {ticker}.csv saved correctly!")

    except Exception as e:
        print(f"❌ Failed to download {ticker}: {e}")
        failed_tickers.append(ticker)

# ✅ Save list of failed tickers
pd.DataFrame({"Failed_Tickers": failed_tickers}).to_csv("data/failed_tickers.csv", index=False)
print("✅ Data acquisition completed!")
