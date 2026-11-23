"""Prepare a leakage-aware modeling dataset after auditing duplicate rows."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUT_PATH = ROOT / "data" / "processed" / "creditcard_modeling.csv"
REPORT_PATH = ROOT / "reports" / "metrics" / "preparation_summary.json"

def main() -> int:
    if not RAW_PATH.exists():
        print(f"Dataset not found: {RAW_PATH}")
        return 1
    df = pd.read_csv(RAW_PATH)
    if "Class" not in df.columns:
        print("Missing target column 'Class'.")
        return 2
    if df.isna().any().any():
        print("Missing values found. Review before continuing.")
        return 2
    if not set(df["Class"].unique()).issubset({0, 1}):
        print("Unexpected target labels.")
        return 2

    # Exact duplicates across all columns are audited before any removal.
    duplicate_mask = df.duplicated(keep=False)
    duplicate_groups = df.loc[duplicate_mask].groupby(list(df.columns), dropna=False)["Class"].nunique()
    conflicting_exact_duplicate_groups = int((duplicate_groups > 1).sum())

    summary = {
        "raw_rows": int(len(df)),
        "raw_columns": int(df.shape[1]),
        "exact_duplicate_rows_beyond_first_occurrence": int(df.duplicated().sum()),
        "rows_part_of_duplicate_groups": int(duplicate_mask.sum()),
        "conflicting_label_exact_duplicate_groups": conflicting_exact_duplicate_groups,
        "raw_class_counts": {str(k): int(v) for k, v in df["Class"].value_counts().sort_index().items()},
        "duplicate_policy": "Drop only exact duplicate rows after checking for conflicting labels."
    }

    # Exact duplicates across all columns cannot have different Class labels because
    # Class itself is part of the duplicate key; still record and enforce the audit.
    if conflicting_exact_duplicate_groups:
        print("Conflicting duplicate groups detected. Stop and inspect manually.")
        print(json.dumps(summary, indent=2))
        return 3

    cleaned = df.drop_duplicates(keep="first").copy()
    summary["rows_after_exact_deduplication"] = int(len(cleaned))
    summary["rows_removed"] = int(len(df) - len(cleaned))
    summary["cleaned_class_counts"] = {
        str(k): int(v) for k, v in cleaned["Class"].value_counts().sort_index().items()
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_PATH, index=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nPrepared dataset saved to: {OUT_PATH}")
    print("Raw dataset was not modified.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
