"""Compare Logistic Regression, Random Forest and XGBoost on one fixed split."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "creditcard_modeling.csv"
REPORTS = ROOT / "reports" / "metrics"
MODELS = ROOT / "models"

def best_f2_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if not len(thresholds):
        return 0.5
    f2 = (5 * precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-12)
    best = np.flatnonzero(np.isclose(f2, f2.max()))
    return float(thresholds[best[-1]])

def main():
    if not DATA.exists():
        raise SystemExit("Prepared dataset missing. Copy your existing data/processed/creditcard_modeling.csv into this project's data/processed/ folder.")
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=["Class"]), df["Class"].astype(int)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=.5, random_state=42, stratify=y_tmp)
    negatives, positives = int((y_train == 0).sum()), int((y_train == 1).sum())
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000, solver="liblinear", random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=42
        )
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=.05, subsample=.85,
            colsample_bytree=.85, min_child_weight=2, reg_lambda=1,
            scale_pos_weight=negatives / max(positives, 1), objective="binary:logistic",
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=42
        )
    except ImportError:
        warnings.warn("XGBoost not installed; skipping it. Run pip install xgboost.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    results, details = [], {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = best_f2_threshold(y_val, val_prob)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        row = {
            "model": name, "selected_threshold": threshold,
            "validation_average_precision": float(average_precision_score(y_val, val_prob)),
            "test_average_precision": float(average_precision_score(y_test, test_prob)),
            "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
            "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "test_f2": float(fbeta_score(y_test, test_pred, beta=2, zero_division=0)),
            "test_alerts": int(test_pred.sum()),
            "true_negatives": int(cm[0,0]), "false_positives": int(cm[0,1]),
            "false_negatives": int(cm[1,0]), "true_positives": int(cm[1,1])
        }
        results.append(row)
        details[name] = {**row, "confusion_matrix_labels_0_1": cm.tolist(),
                         "threshold_selection": "Maximum F2 on validation set",
                         "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}}
        joblib.dump(model, MODELS / (name.lower().replace(" ", "_") + ".joblib"))
        print(f"AP={row['test_average_precision']:.4f} | Precision={row['test_precision']:.4f} | Recall={row['test_recall']:.4f} | F2={row['test_f2']:.4f} | Threshold={threshold:.6f} | Alerts={row['test_alerts']}")
    table = pd.DataFrame(results).sort_values(["test_average_precision", "test_f2"], ascending=False)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_comparison_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("\n=== Model comparison (sorted by test Average Precision) ===")
    print(table.to_string(index=False))
    print("\nReports saved under reports/metrics/. Fitted models saved under models/.")
    print("Use test metrics for final reporting, not repeated tuning.")

if __name__ == "__main__":
    main()
