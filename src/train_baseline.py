"""Train and evaluate a first Logistic Regression baseline.

Workflow:
- Prepare data (audits and removes exact duplicate rows).
- Split into train/validation/test with stratification.
- Fit preprocessing and model on train only.
- Select threshold on validation set by F2 score.
- Evaluate the selected threshold once on the held-out test set.
"""
from pathlib import Path
import json
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix, classification_report,
    precision_recall_curve
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "creditcard_modeling.csv"
MODEL_PATH = ROOT / "models" / "logistic_regression_baseline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics"
FIGURES_PATH = ROOT / "reports" / "figures"
TARGET = "Class"

def main() -> int:
    if not DATA_PATH.exists():
        print("Prepared dataset not found. Run: python src\\prepare_data.py")
        return 1

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 60% train, 20% validation, 20% test. Stratification preserves rare-class proportions.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_prob)

    # Select threshold using validation F2; F2 weights recall more than precision.
    # precision/recall have one more element than thresholds, so use all threshold indices.
    if len(thresholds) == 0:
        threshold = 0.5
    else:
        p = precision[:-1]
        r = recall[:-1]
        f2 = (5 * p * r) / (4 * p + r + 1e-12)
        best_idx = int(f2.argmax())
        threshold = float(thresholds[best_idx])

    # Threshold sensitivity table on validation data.
    val_rows = []
    for t in [0.10, 0.20, 0.30, 0.40, 0.50, threshold]:
        pred = (val_prob >= t).astype(int)
        val_rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_val, pred, zero_division=0)),
            "recall": float(recall_score(y_val, pred, zero_division=0)),
            "f1": float(f1_score(y_val, pred, zero_division=0)),
            "f2": float(fbeta_score(y_val, pred, beta=2, zero_division=0)),
            "alerts": int(pred.sum())
        })
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(val_rows).drop_duplicates(subset=["threshold"]).sort_values(
        "threshold"
    ).to_csv(METRICS_PATH / "validation_threshold_analysis.csv", index=False)

    # Final held-out test evaluation at the validation-selected threshold.
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    metrics = {
        "model": "LogisticRegression(class_weight='balanced')",
        "random_state": 42,
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "class_counts": {
            "train": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "validation": {str(k): int(v) for k, v in y_val.value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()}
        },
        "threshold_selection": "Maximum F2 score on validation set",
        "selected_threshold": threshold,
        "validation_average_precision": float(average_precision_score(y_val, val_prob)),
        "test_average_precision_pr_auc": float(average_precision_score(y_test, test_prob)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_f2": float(fbeta_score(y_test, test_pred, beta=2, zero_division=0)),
        "test_alerts": int(test_pred.sum()),
        "test_confusion_matrix_labels_0_1": cm.tolist(),
        "test_classification_report": classification_report(
            y_test, test_pred, labels=[0, 1],
            target_names=["Legitimate", "Fraud"], output_dict=True, zero_division=0
        ),
        "note": "Threshold chosen on validation data; test set used only for final evaluation."
    }
    with open(METRICS_PATH / "logistic_regression_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(f"Test Confusion Matrix (threshold={threshold:.4f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "baseline_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Baseline training completed.")
    print(f"Selected threshold (validation F2): {threshold:.6f}")
    print(f"Validation Average Precision: {metrics['validation_average_precision']:.4f}")
    print("\nHeld-out test metrics:")
    for key in ["test_average_precision_pr_auc", "test_roc_auc", "test_precision",
                "test_recall", "test_f1", "test_f2", "test_alerts"]:
        print(f"{key}: {metrics[key]}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH / 'logistic_regression_baseline.json'}")
    print("This is a baseline, not the final model selection.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
