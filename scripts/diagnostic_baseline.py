import os
import sys

import pandas as pd


def run_diagnostic():
    file_path = "data/events.csv"  # CHANGE THIS if your file is named differently

    print("--- 🔍 Checking File Presence ---")
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at {file_path}")
        print("Check if you unzipped the dataset and the folder is named 'data'.")
        return

    print("--- 🚀 Loading Data (This may take 30-60 seconds...) ---")
    try:
        # We only load the columns we need to save memory/time
        df = pd.read_csv(file_path, usecols=["timestamp", "visitorid", "itemid"])
        print(f"✅ Data Loaded! Total rows: {len(df)}")
    except Exception as e:
        print(f"❌ ERROR while reading CSV: {e}")
        return

    print("--- ⚙️ Splitting Data ---")
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Training on {len(train_df)} events...")

    # Get Top 10
    top_10 = train_df["itemid"].value_counts().head(10).index.tolist()
    print(f"🔥 Top 10 Popular Items: {top_10}")

    print("--- 🧪 Running Evaluation (Matching IDs) ---")
    test_users = test_df["visitorid"].unique()
    test_actuals = test_df.groupby("visitorid")["itemid"].apply(set).to_dict()

    hits = 0
    for vid in test_actuals:
        if any(item in test_actuals[vid] for item in top_10):
            hits += 1

    print("\n" + "=" * 30)
    print(f"FINAL RESULT:")
    print(f"Hit Rate @ 10: {hits/len(test_actuals):.6f}")
    print(f"Hits: {hits} | Total Test Users: {len(test_actuals)}")
    print("=" * 30)


if __name__ == "__main__":
    run_diagnostic()
