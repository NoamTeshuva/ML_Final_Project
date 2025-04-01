import pandas as pd
import os

def combine_stock_data():
    """
    Combine all stock feature data into a single dataset for training.

    Returns:
        pd.DataFrame: Final dataset with all stock indicators.
    """
    stock_files = [f for f in os.listdir("data/processed/") if f.startswith("featured_")]

    dataset = []
    for file in stock_files:
        df = pd.read_csv(os.path.join("data", "processed", file), index_col=0, parse_dates=True)
        df["Ticker"] = file.replace("featured_", "").replace(".csv", "")
        dataset.append(df)

    return pd.concat(dataset)

if __name__ == "__main__":
    final_df = combine_stock_data()

    # Split data into training (2015-2020) and testing (2020-2025)
    train_df = final_df[final_df.index < "2020-01-01"]
    test_df = final_df[final_df.index >= "2020-01-01"]

    # Save datasets
    train_df.to_csv("data/processed/training_data.csv")
    test_df.to_csv("data/processed/testing_data.csv")

    print("Training & Testing datasets prepared.")
