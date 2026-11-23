
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "reports" / "txnsentry.db"
OUTPUT_DIR = ROOT / "reports" / "sql"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DB_PATH.exists():
    raise FileNotFoundError(
        "Database not found. Run build_sql_database.py first."
    )

queries = {
    "transaction_summary": """
        SELECT
            COUNT(*) AS total_transactions,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
                AS fraud_transactions,
            SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END)
                AS legitimate_transactions,
            ROUND(
                100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
                / COUNT(*), 4
            ) AS fraud_percentage
        FROM transactions
    """,

    "amount_statistics": """
        SELECT
            Class,
            COUNT(*) AS transaction_count,
            ROUND(AVG(Amount), 2) AS average_amount,
            ROUND(MIN(Amount), 2) AS minimum_amount,
            ROUND(MAX(Amount), 2) AS maximum_amount
        FROM transactions
        GROUP BY Class
        ORDER BY Class
    """,

    "hourly_distribution": """
        SELECT
            CAST((Time % 86400) / 3600 AS INTEGER)
                AS hour_bucket,
            COUNT(*) AS total_transactions,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
                AS fraud_transactions
        FROM transactions
        GROUP BY hour_bucket
        ORDER BY hour_bucket
    """,

    "feature_class_comparison": """
        SELECT
            Class,
            ROUND(AVG(V14), 5) AS avg_V14,
            ROUND(AVG(V12), 5) AS avg_V12,
            ROUND(AVG(V10), 5) AS avg_V10,
            ROUND(AVG(V4), 5) AS avg_V4,
            ROUND(AVG(V17), 5) AS avg_V17
        FROM transactions
        GROUP BY Class
        ORDER BY Class
    """,

    "top_fraud_transactions": """
        SELECT
            Time, Amount, V14, V12, V10, V4, V17
        FROM transactions
        WHERE Class = 1
        ORDER BY Amount DESC
        LIMIT 20
    """
}


with sqlite3.connect(DB_PATH) as connection:
    for report_name, query in queries.items():
        print(f"\nRunning: {report_name}")

        result = pd.read_sql_query(query, connection)

        output_path = OUTPUT_DIR / f"{report_name}.csv"
        result.to_csv(output_path, index=False)

        print(result.to_string(index=False))
        print(f"Saved: {output_path}")


print("\nAll SQL analytics completed successfully.")
