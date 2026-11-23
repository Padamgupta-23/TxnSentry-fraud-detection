"""Fast validation-only threshold and calibration analysis for TxnSentry."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "creditcard_modeling.csv"
OUT = ROOT / "reports" / "metrics"

def threshold_table(y_true, probabilities):
    """Compute metrics at all unique score thresholds in one vectorized pass."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]

    # Evaluate only at the end of each tied-score group.
    ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(sorted_p) - 1]
    thresholds = sorted_p[ends]
    tp = np.cumsum(sorted_y)[ends].astype(np.int64)
    predicted_positive = ends + 1
    fp = predicted_positive - tp
    total_positive = int(y.sum())
    fn = total_positive - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, total_positive, out=np.zeros_like(tp, dtype=float), where=total_positive != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    f2 = np.divide(5 * precision * recall, 4 * precision + recall,
                   out=np.zeros_like(precision), where=(4 * precision + recall) != 0)
    return pd.DataFrame({
        "threshold": thresholds, "precision": precision, "recall": recall,
        "f1": f1, "f2": f2, "alerts": predicted_positive,
        "false_positives": fp, "true_positives": tp, "false_negatives": fn
    })

def get_models(y_train):
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000,
                                         solver="liblinear", random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=42
        )
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=.05,
            subsample=.85, colsample_bytree=.85, min_child_weight=2,
            reg_lambda=1, scale_pos_weight=neg / max(pos, 1),
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=42
        )
    except ImportError:
        warnings.warn("XGBoost unavailable; install with pip install xgboost.")
    return models

def main():
    if not DATA.exists():
        raise SystemExit("Prepared CSV missing. Run python src\\prepare_data.py first.")
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=["Class"]), df["Class"].astype(int)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=.4, random_state=42, stratify=y
    )
    X_val, _, y_val, _ = train_test_split(
        X_tmp, y_tmp, test_size=.5, random_state=42, stratify=y_tmp
    )
    OUT.mkdir(parents=True, exist_ok=True)
    operating_rows, calibration_rows, summaries = [], [], {}

    for name, model in get_models(y_train).items():
        print(f"Fitting {name}...")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        curve = threshold_table(y_val, prob)
        best = curve.loc[curve["f2"].idxmax()]
        candidates = [
            ("F2_optimal", float(best["threshold"])),
            ("threshold_0.50", .50),
            ("threshold_0.25", .25),
            ("threshold_0.10", .10),
        ]
        recall_candidates = curve[curve["recall"] >= .80]
        if not recall_candidates.empty:
            # Highest threshold that still reaches the target recall.
            candidates.append(("recall_at_least_80pct",
                               float(recall_candidates["threshold"].max())))
        precision_candidates = curve[curve["precision"] >= .90]
        if not precision_candidates.empty:
            # Lowest threshold that reaches precision target, which gives maximum recall
            # among thresholds satisfying it (for monotonic score cutoffs).
            candidates.append(("precision_at_least_90pct",
                               float(precision_candidates["threshold"].min())))

        seen = set()
        for policy, threshold in candidates:
            if threshold in seen:
                continue
            seen.add(threshold)
            # Candidate thresholds may be between observed probabilities; compute vectorized.
            pred_count = int((prob >= threshold).sum())
            tp = int(((prob >= threshold) & (y_val.to_numpy() == 1)).sum())
            fp = pred_count - tp
            fn = int((y_val.to_numpy() == 1).sum()) - tp
            precision = tp / pred_count if pred_count else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
            operating_rows.append({
                "model": name, "policy": policy, "threshold": threshold,
                "precision": precision, "recall": recall, "f1": f1, "f2": f2,
                "alerts": pred_count, "false_positives": fp,
                "true_positives": tp, "false_negatives": fn
            })

        frac_pos, mean_pred = calibration_curve(y_val, prob, n_bins=10, strategy="quantile")
        for idx, (mp, fp_rate) in enumerate(zip(mean_pred, frac_pos), start=1):
            calibration_rows.append({
                "model": name, "bin": idx,
                "mean_predicted_score": float(mp),
                "observed_fraud_rate": float(fp_rate)
            })
        summaries[name] = {
            "validation_average_precision": float(average_precision_score(y_val, prob)),
            "validation_roc_auc": float(roc_auc_score(y_val, prob)),
            "validation_brier_score": float(brier_score_loss(y_val, prob)),
            "validation_fraud_prevalence": float(y_val.mean()),
            "note": "Scores from class-weighted models may be poorly calibrated; Brier score is not sufficient alone to judge calibration."
        }
        print(f"  Average Precision={summaries[name]['validation_average_precision']:.4f}; "
              f"F2-selected threshold={float(best['threshold']):.6f}; "
              f"F2={float(best['f2']):.4f}")

    pd.DataFrame(operating_rows).to_csv(OUT / "validation_operating_points.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "validation_calibration_bins.csv", index=False)
    (OUT / "validation_calibration_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print("\nCompleted quickly using vectorized threshold metrics.")
    print("\nValidation operating points:")
    print(pd.DataFrame(operating_rows).to_string(index=False))
    print("\nSaved reports/metrics/validation_operating_points.csv")
    print("Saved reports/metrics/validation_calibration_bins.csv")
    print("Saved reports/metrics/validation_calibration_summary.json")
    print("No held-out test metrics were used to select thresholds.")

if __name__ == "__main__":
    main()
