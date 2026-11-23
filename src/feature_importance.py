
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"

CHART_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

MODELS = {
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}


def extract_importance(model, model_name):
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{model_name} does not expose feature_importances_."
        )

    values = model.feature_importances_

    if len(values) != len(FEATURES):
        raise ValueError(
            f"Expected {len(FEATURES)} importances, got {len(values)}."
        )

    result = pd.DataFrame({
        "feature": FEATURES,
        "importance": values,
        "model": model_name,
    })

    return result.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


all_results = []

for model_name, model_path in MODELS.items():
    print(f"Loading {model_name}...")

    model = joblib.load(model_path)
    result = extract_importance(model, model_name)

    output_name = model_name.lower().replace(" ", "_")

    result.to_csv(
        METRIC_DIR / f"{output_name}_feature_importance.csv",
        index=False,
    )

    all_results.append(result)

    top = result.head(15).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top 15 Feature Importances - {model_name}")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / f"{output_name}_feature_importance.png",
        dpi=200,
    )
    plt.close()

    print(f"\nTop 10 features: {model_name}")
    print(result.head(10).to_string(index=False))


combined = pd.concat(all_results, ignore_index=True)

combined.to_csv(
    METRIC_DIR / "combined_feature_importance.csv",
    index=False,
)

print("\nFeature importance analysis completed.")
print(f"Charts saved to: {CHART_DIR}")
print(f"CSV reports saved to: {METRIC_DIR}")
