"""Exploratory data analysis for TxnSentry.

Reads the untouched raw dataset and writes charts and summary tables to reports/.
Does not modify or delete any source rows.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
FIGURES = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics"

def save_plot(filename: str) -> Path:
    path = FIGURES / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path

def main() -> int:
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place creditcard.csv in data/raw/ and run again.")
        return 1

    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATA_PATH)
    required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    if set(df.columns) != required:
        print("Schema mismatch. Run python src\\inspect_data.py and resolve before EDA.")
        return 2
    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        print("Class must contain only 0 and 1.")
        return 2

    # Core summary
    counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    summary = pd.DataFrame({
        "class_label": [0, 1],
        "class_name": ["Legitimate", "Fraud"],
        "transaction_count": [int(counts.loc[0]), int(counts.loc[1])],
        "percentage": [float(counts.loc[0] / len(df) * 100),
                       float(counts.loc[1] / len(df) * 100)]
    })
    summary.to_csv(METRICS / "class_distribution.csv", index=False)

    amount_summary = df.groupby("Class")["Amount"].agg(
        transaction_count="count", mean="mean", median="median",
        std="std", minimum="min", maximum="max"
    ).reset_index()
    amount_summary["class_name"] = amount_summary["Class"].map({0: "Legitimate", 1: "Fraud"})
    amount_summary.to_csv(METRICS / "amount_summary_by_class.csv", index=False)

    duplicate_mask = df.duplicated(keep=False)
    duplicate_summary = pd.DataFrame([
        {"measure": "rows_total", "value": int(len(df))},
        {"measure": "exact_duplicate_rows_beyond_first_occurrence", "value": int(df.duplicated().sum())},
        {"measure": "rows_part_of_any_duplicate_group", "value": int(duplicate_mask.sum())},
        {"measure": "duplicate_rows_class_0", "value": int((duplicate_mask & (df["Class"] == 0)).sum())},
        {"measure": "duplicate_rows_class_1", "value": int((duplicate_mask & (df["Class"] == 1)).sum())},
    ])
    duplicate_summary.to_csv(METRICS / "duplicate_audit.csv", index=False)

    # Chart 1: class imbalance
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="Class")
    ax.set_title("Transaction Class Distribution")
    ax.set_xlabel("Class (0 = legitimate, 1 = fraud)")
    ax.set_ylabel("Transaction count")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    save_plot("class_distribution.png")

    # Chart 2: amount distributions; use log scale for visibility, preserve zero values
    plt.figure(figsize=(9, 5))
    plot_df = df.copy()
    plot_df["Amount_log1p"] = __import__("numpy").log1p(plot_df["Amount"])
    sns.histplot(data=plot_df, x="Amount_log1p", hue="Class", bins=60,
                 stat="density", common_norm=False, element="step", fill=False)
    plt.title("Transaction Amount Distribution (log1p scale)")
    plt.xlabel("log(1 + Amount)")
    plt.ylabel("Density")
    save_plot("amount_distribution_by_class.png")

    # Chart 3: boxplot of amounts, hide extreme points for readability (not removed from data)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Class", y="Amount", showfliers=False)
    plt.title("Transaction Amount by Class (extreme points hidden for display)")
    plt.xlabel("Class (0 = legitimate, 1 = fraud)")
    plt.ylabel("Amount")
    save_plot("amount_boxplot_by_class.png")

    # Chart 4: fraud prevalence in 1-hour intervals. Time is seconds elapsed in the dataset.
    time_df = df[["Time", "Class"]].copy()
    time_df["hour_bin"] = (time_df["Time"] // 3600).astype(int)
    hourly = time_df.groupby("hour_bin").agg(
        transactions=("Class", "size"),
        fraud_count=("Class", "sum"),
        fraud_rate_pct=("Class", lambda s: s.mean() * 100)
    ).reset_index()
    hourly.to_csv(METRICS / "fraud_rate_by_time_bin.csv", index=False)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly, x="hour_bin", y="fraud_rate_pct", marker="o")
    plt.title("Observed Fraud Rate by One-Hour Time Bin")
    plt.xlabel("Hour bin (relative to dataset start)")
    plt.ylabel("Fraud rate (%)")
    save_plot("fraud_rate_by_time_bin.png")

    # Chart 5: standardized class mean differences for anonymized V1-V28.
    # These are descriptive differences, not causal explanations.
    feature_cols = [f"V{i}" for i in range(1, 29)]
    rows = []
    for col in feature_cols:
        means = df.groupby("Class")[col].mean()
        std = df[col].std(ddof=0)
        difference = float(means.loc[1] - means.loc[0])
        rows.append({
            "feature": col,
            "legitimate_mean": float(means.loc[0]),
            "fraud_mean": float(means.loc[1]),
            "fraud_minus_legitimate_mean": difference,
            "absolute_standardized_mean_difference": abs(difference / std) if std else 0.0
        })
    feature_diff = pd.DataFrame(rows).sort_values(
        "absolute_standardized_mean_difference", ascending=False
    )
    feature_diff.to_csv(METRICS / "pca_feature_mean_differences.csv", index=False)
    top = feature_diff.head(12).sort_values("absolute_standardized_mean_difference")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["absolute_standardized_mean_difference"])
    plt.title("Largest Standardized Mean Differences: PCA Features")
    plt.xlabel("Absolute standardized mean difference")
    plt.ylabel("PCA feature")
    save_plot("pca_feature_mean_differences.png")

    # Concise run summary
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "exact_duplicate_rows_beyond_first_occurrence": int(df.duplicated().sum()),
        "rows_part_of_any_duplicate_group": int(duplicate_mask.sum()),
        "legitimate_count": int(counts.loc[0]),
        "fraud_count": int(counts.loc[1]),
        "fraud_prevalence_percent": float(counts.loc[1] / len(df) * 100),
        "source_csv_modified": False
    }
    with open(METRICS / "eda_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("EDA completed. The original CSV was not modified.")
    print("\nClass distribution:")
    print(summary.to_string(index=False))
    print("\nDuplicate audit:")
    print(duplicate_summary.to_string(index=False))
    print("\nAmount summary:")
    print(amount_summary.to_string(index=False))
    print("\nTop 10 PCA features by absolute standardized mean difference:")
    print(feature_diff.head(10).to_string(index=False))
    print("\nGenerated charts:")
    for name in [
        "class_distribution.png",
        "amount_distribution_by_class.png",
        "amount_boxplot_by_class.png",
        "fraud_rate_by_time_bin.png",
        "pca_feature_mean_differences.png",
    ]:
        print(f"  reports/figures/{name}")
    print("\nGenerated tables are in reports/metrics/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
