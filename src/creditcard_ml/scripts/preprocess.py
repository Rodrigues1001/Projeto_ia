import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def main():
    print("🔧 Preprocessing dataset...")

    df = pd.read_csv("data/creditcard.csv")

    # Normalize Time and Amount (feature engineering)
    scaler = StandardScaler()
    df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

    df.to_csv("data/processed.csv", index=False)
    print("✅ Saved: data/processed.csv")

if __name__ == "__main__":
    main()
