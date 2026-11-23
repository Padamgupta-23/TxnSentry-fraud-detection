
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "data" / "processed" / "creditcard_modeling.csv"
DB_DIR = ROOT / "reports"
DB_PATH = DB_DIR / "txnsentry.db"

DB_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {CSV_PATH}"
    )

print("Loading processed dataset...")
df = pd.read_csv(CSV_PATH)

required_columns = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

if not set(df["Class"].unique()).issubset({0, 1}):
    raise ValueError("Class must contain only 0 and 1.")

print(f"Rows loaded: {len(df):,}")

with sqlite3.connect(DB_PATH) as connection:
    df.to_sql(
        "transactions",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_class "
        "ON transactions (Class)"
    )

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_amount "
        "ON transactions (Amount)"
    )

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_time "
        "ON transactions (Time)"
    )

    db_count = connection.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]

print("\nSQLite database created successfully.")
print(f"Database: {DB_PATH}")
print(f"Records stored: {db_count:,}")
