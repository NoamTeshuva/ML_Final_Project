import yfinance as yf
import pandas as pd
import os
import time  # ✅ Import time module

# Load tickers from Russell 2000 list
tickers = pd.read_csv("C:/Users/HP/PycharmProjects/ML_final_project/data/iwm_holdings_cleaned.csv")["Ticker"].tolist()

# ✅ Initialize failed tickers list
failed_tickers = []

# Create directory if not exists
data_dir = "C:/Users/HP/PycharmProjects/ML_final_project/data/processed/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# ✅ Fetch data for each ticker
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

        # ✅ Prevent rate limiting by adding a short delay (2 seconds)
        time.sleep(2)

    except Exception as e:
        print(f"❌ Failed to download {ticker}: {e}")
        failed_tickers.append((ticker, str(e)))

        # ✅ If rate limit error occurs, wait longer before next request
        if "rate limited" in str(e).lower():
            print("⏳ Rate limit detected. Pausing for 60 seconds...")
            time.sleep(60)  # Wait 60 seconds before continuing

# ✅ Save list of failed tickers
failed_tickers_path = "C:/Users/HP/PycharmProjects/ML_final_project/data/failed_tickers.csv"
pd.DataFrame(failed_tickers, columns=["Ticker", "Error"]).to_csv(failed_tickers_path, index=False)

print(f"✅ Data acquisition completed! Failed tickers saved to: {failed_tickers_path}")
