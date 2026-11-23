"""Validate and summarize the original credit-card fraud dataset."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"

def main() -> int:
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place the downloaded creditcard.csv in data/raw/ and run again.")
        return 1

    # Read a small sample first, then load the full file for reliable counts.
    sample = pd.read_csv(DATA_PATH, nrows=5)
    print("\n--- Sample rows ---")
    print(sample.to_string(index=False))
    print("\n--- Initial schema ---")
    print("Columns:", list(sample.columns))

    df = pd.read_csv(DATA_PATH)
    print("\n--- Dataset shape ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\n--- Column names and data types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing values ---")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "No missing values found.")

    print("\n--- Duplicate rows ---")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\n--- Target validation ---")
    if "Class" not in df.columns:
        print("ERROR: Expected target column 'Class' was not found.")
        print("Inspect the column names above; do not rename columns blindly.")
        return 2

    print(df["Class"].value_counts(dropna=False).sort_index().to_string())
    print("\nClass proportions (%):")
    print((df["Class"].value_counts(normalize=True, dropna=False).sort_index() * 100).round(4).to_string())

    if set(df["Class"].dropna().unique()).issubset({0, 1}):
        fraud_count = int((df["Class"] == 1).sum())
        total = len(df)
        print(f"\nFraud cases: {fraud_count:,} / {total:,}")
        print(f"Fraud prevalence: {fraud_count / total * 100:.4f}%")

    print("\n--- Numeric summary ---")
    print(df.describe().T.to_string())

    expected = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    actual = set(df.columns)
    if expected == actual:
        print("\nSchema check: PASS (expected 31 columns).")
    else:
        print("\nSchema check: REVIEW")
        print("Missing expected columns:", sorted(expected - actual))
        print("Unexpected columns:", sorted(actual - expected))
        print("If the first column has a blank header, share the output before changing the CSV.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
