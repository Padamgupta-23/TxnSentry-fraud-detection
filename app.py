
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "random_forest.joblib"

COMPARISON_PATH = (
    ROOT / "reports" / "metrics" / "model_comparison.csv"
)

VALIDATION_PATH = (
    ROOT / "reports" / "metrics" / "validation_operating_points.csv"
)

CALIBRATION_PATH = (
    ROOT / "reports" / "metrics" / "validation_calibration_summary.json"
)

DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "creditcard_modeling.csv",
    ROOT / "data" / "raw" / "creditcard.csv",
]

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEFAULT_THRESHOLD = 0.3696021698174826

# Saved dataset profile from the preparation report.
TOTAL_TRANSACTIONS = 283726
LEGITIMATE_TRANSACTIONS = 283253
FRAUD_TRANSACTIONS = 473


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TxnSentry | Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        padding: 13px 15px;
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


@st.cache_data
def load_data(path, modified):
    return pd.read_csv(path)


@st.cache_data
def read_csv(path, modified=None):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass

    return pd.DataFrame()


@st.cache_data
def read_json(path, modified=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass

    return {}


def get_file_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# ============================================================
# MODEL SCORING
# ============================================================

def score(model, frame):
    """
    Generate fraud risk scores using only the 30 model features.

    Class is never passed to the model.
    """

    missing = [c for c in FEATURES if c not in frame.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    X = frame[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "Required feature columns contain missing or non-numeric values."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Required feature columns must contain finite numeric values."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The loaded model does not support predict_proba()."
        )

    return model.predict_proba(X)[:, 1]


# ============================================================
# METRICS
# ============================================================

def metrics_at(y, scores, threshold):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores)

    if len(y) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    predictions = scores >= threshold

    tp = int(((y == 1) & predictions).sum())
    fp = int(((y == 0) & predictions).sum())
    tn = int(((y == 0) & ~predictions).sum())
    fn = int(((y == 1) & ~predictions).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if 4 * precision + recall
        else 0
    )

    alerts = tp + fp

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "alerts": alerts,
        "alert_rate": alerts / len(y),
    }


def format_metric(value, metric):
    if metric in ["precision", "recall"]:
        return f"{float(value):.1%}"

    if metric == "f2":
        return f"{float(value):.3f}"

    if metric == "alerts":
        return f"{int(value):,}"

    return str(value)


# ============================================================
# LOAD MODEL AND REPORTS
# ============================================================

st.title("🛡️ TxnSentry")

st.caption(
    "FINANCIAL FRAUD DETECTION  /  RISK ANALYTICS"
)

st.write(
    "An end-to-end screening prototype to help prioritize "
    "suspicious credit-card transactions for human review."
)

st.warning(
    "Portfolio demonstration only. Scores are not calibrated "
    "real-world fraud probabilities. Do not automatically block "
    "transactions based on this demo."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model not found: {MODEL_PATH}. "
        "Keep app.py in the project root and check "
        "models/random_forest.joblib."
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


comparison = read_csv(
    COMPARISON_PATH,
    get_file_mtime(COMPARISON_PATH),
)

validation = read_csv(
    VALIDATION_PATH,
    get_file_mtime(VALIDATION_PATH),
)

calibration = read_json(
    CALIBRATION_PATH,
    get_file_mtime(CALIBRATION_PATH),
)


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

default_threshold = DEFAULT_THRESHOLD

try:
    if {"model", "policy", "threshold"}.issubset(
        validation.columns
    ):
        rf_policy = validation[
            (validation["model"].astype(str) == "Random Forest")
            & (validation["policy"].astype(str) == "F2_optimal")
        ]

        if not rf_policy.empty:
            default_threshold = float(
                rf_policy.iloc[0]["threshold"]
            )

except (ValueError, TypeError, KeyError):
    default_threshold = DEFAULT_THRESHOLD


default_threshold = float(
    np.clip(default_threshold, 0.001, 0.99)
)


# ============================================================
# LOAD OPTIONAL LABELLED DATASET
# ============================================================

data_path = next(
    (path for path in DATA_CANDIDATES if path.exists()),
    None,
)

data = None

if data_path is not None:
    try:
        loaded_data = load_data(
            str(data_path),
            data_path.stat().st_mtime,
        )

        required_columns = set(FEATURES + ["Class"])

        if required_columns.issubset(loaded_data.columns):
            data = loaded_data

    except Exception:
        data = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Review policy")

    threshold = st.slider(
        "Alert threshold",
        min_value=0.001,
        max_value=0.99,
        value=default_threshold,
        step=0.005,
    )

    st.metric(
        "Active cutoff",
        f"{threshold:.4f}",
    )

    st.caption(
        "A transaction is flagged when its model score "
        "is greater than or equal to the cutoff."
    )

    with st.expander("Threshold trade-off"):

        st.write(
            "Lower cutoff: usually more fraud detected, "
            "but more false alarms."
        )

        st.write(
            "Higher cutoff: usually fewer alerts, "
            "but potentially more missed fraud."
        )

    st.divider()

    st.caption("Required model inputs")

    st.code("Time, V1–V28, Amount")

    st.caption(
        "Class is used only for evaluation, never prediction."
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Executive overview",
        "Transaction explorer",
        "Batch screening",
        "Threshold & errors",
        "Model comparison",
        "Feature guide",
    ]
)


# ============================================================
# SCORE LOCAL DATASET
# ============================================================

@st.cache_data(show_spinner="Scoring local dataset for analytics...")
def score_dataset(
    model_path,
    model_modified,
    data_path_str,
    data_modified,
):
    mdl = joblib.load(model_path)

    df = pd.read_csv(data_path_str)

    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError("Dataset contains invalid feature values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite feature values.")

    scores = mdl.predict_proba(X)[:, 1]

    return df, scores


analytics = None

if data is not None:
    try:
        analytics = score_dataset(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime,
            str(data_path),
            data_path.stat().st_mtime,
        )

    except Exception:
        analytics = None


# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================

with tabs[0]:

    st.subheader("Executive overview")

    st.write(
        "Start here for the model's business purpose, "
        "the selected cutoff, and the scale of the dataset."
    )

    if analytics is not None:

        df, scores = analytics

        y = df["Class"].astype(int).to_numpy()

        m = metrics_at(y, scores, threshold)

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Transactions analysed",
            f"{len(df):,}",
        )

        c2.metric(
            "Fraud prevalence",
            f"{y.mean() * 100:.3f}%",
        )

        c3.metric(
            "Flagged at cutoff",
            f"{m['alerts']:,}",
        )

        c4.metric(
            "Precision at cutoff",
            f"{m['precision']:.1%}",
        )

        c5.metric(
            "Recall at cutoff",
            f"{m['recall']:.1%}",
        )

        st.caption(
            "These numbers use the local labelled CSV and the "
            "currently selected threshold. They are descriptive "
            "on that file, not an independent final test estimate."
        )

        left, right = st.columns(2)

        with left:

            counts = (
                pd.Series(y)
                .map({
                    0: "Legitimate",
                    1: "Fraud",
                })
                .value_counts()
                .rename_axis("Class")
                .reset_index(name="Transactions")
            )

            fig = px.bar(
                counts,
                x="Class",
                y="Transactions",
                color="Class",
                title="Class imbalance: legitimate vs fraud",
                text="Transactions",
            )

            fig.update_layout(showlegend=False)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Shows why accuracy alone can be misleading: "
                "fraud is rare compared with legitimate transactions."
            )

        with right:

            score_frame = pd.DataFrame({
                "Score": scores,
                "Actual label": np.where(
                    y == 1,
                    "Fraud",
                    "Legitimate",
                ),
            })

            fig = px.histogram(
                score_frame,
                x="Score",
                color="Actual label",
                nbins=50,
                barmode="overlay",
                opacity=0.7,
                title="Risk-score distribution by known label",
            )

            fig.add_vline(
                x=threshold,
                line_dash="dash",
                annotation_text="Current cutoff",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "A useful model tends to assign higher scores to "
                "fraud than legitimate transactions. Overlap indicates "
                "cases the model may find difficult."
            )

        st.markdown("#### What happens at the current cutoff?")

        confusion = pd.DataFrame(
            [
                [m["TN"], m["FP"]],
                [m["FN"], m["TP"]],
            ],
            index=[
                "Actual legitimate",
                "Actual fraud",
            ],
            columns=[
                "Predicted not flagged",
                "Predicted flagged",
            ],
        )

        fig = px.imshow(
            confusion,
            text_auto=True,
            aspect="auto",
            title="Confusion matrix at active cutoff",
            color_continuous_scale="Blues",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "TN = legitimate not flagged; FP = legitimate flagged; "
            "FN = fraud missed; TP = fraud flagged. The goal is to "
            "balance missed fraud against review workload."
        )

    else:

        st.info(
            "Showing saved dataset and validation summaries. "
            "Upload a labelled CSV locally to enable live "
            "transaction-level analytics."
        )

        fraud_prevalence = (
            FRAUD_TRANSACTIONS / TOTAL_TRANSACTIONS * 100
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Transactions in dataset",
            f"{TOTAL_TRANSACTIONS:,}",
        )

        c2.metric(
            "Legitimate transactions",
            f"{LEGITIMATE_TRANSACTIONS:,}",
        )

        c3.metric(
            "Fraud transactions",
            f"{FRAUD_TRANSACTIONS:,}",
        )

        c4.metric(
            "Fraud prevalence",
            f"{fraud_prevalence:.3f}%",
        )

        st.caption(
            "Dataset profile from the saved preparation report. "
            "These are historical dataset statistics, not live "
            "transaction counts."
        )

        left, right = st.columns(2)

        with left:

            class_profile = pd.DataFrame({
                "Class": [
                    "Legitimate",
                    "Fraud",
                ],
                "Transactions": [
                    LEGITIMATE_TRANSACTIONS,
                    FRAUD_TRANSACTIONS,
                ],
            })

            fig = px.bar(
                class_profile,
                x="Class",
                y="Transactions",
                color="Class",
                text_auto=True,
                title="Dataset class distribution",
            )

            fig.update_layout(showlegend=False)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:

            st.markdown("#### Selected model: Random Forest")

            st.write(
                "The deployed screening model is a "
                "Random Forest classifier."
            )

            if (
                not validation.empty
                and {
                    "model",
                    "policy",
                    "threshold",
                }.issubset(validation.columns)
            ):

                rf_policy = validation[
                    (validation["model"].astype(str) == "Random Forest")
                    & (
                        validation["policy"].astype(str)
                        == "F2_optimal"
                    )
                ]

            else:
                rf_policy = pd.DataFrame()

            if not rf_policy.empty:

                policy = rf_policy.iloc[0]

                st.metric(
                    "Validation-selected cutoff",
                    f"{float(policy['threshold']):.4f}",
                )

                for metric in [
                    "precision",
                    "recall",
                    "f2",
                    "alerts",
                ]:

                    if metric in rf_policy.columns:

                        st.write(
                            f"**{metric.title()}:** "
                            f"{format_metric(policy[metric], metric)}"
                        )

            else:

                st.caption(
                    "Saved Random Forest validation policy not available."
                )

        st.warning(
            "The saved validation results are historical "
            "model-development results. The sidebar cutoff does "
            "not recalculate these saved metrics. Scores are not "
            "calibrated real-world fraud probabilities."
        )

    st.markdown("#### Workflow")

    cols = st.columns(4)

    for col, number, title, description in zip(
        cols,
        ["01", "02", "03", "04"],
        ["Load", "Score", "Flag", "Review"],
        [
            "30 model inputs",
            "Risk score from model",
            "Compare to cutoff",
            "Human investigation",
        ],
    ):

        with col:

            st.markdown(f"**{number} · {title}**")

            st.write(description)


# ============================================================
# TAB 2: TRANSACTION EXPLORER
# ============================================================

with tabs[1]:

    st.subheader("Transaction explorer")

    st.write(
        "Choose a row from the labelled dataset to compare "
        "the model score with its known label. The label is "
        "never passed into the model."
    )

    if data is not None:

        mode = st.radio(
            "Select example",
            [
                "Fraud example",
                "Legitimate example",
                "Choose row index",
            ],
            horizontal=True,
        )

        if mode == "Fraud example":

            choices = data.index[
                data["Class"].astype(int) == 1
            ]

        elif mode == "Legitimate example":

            choices = data.index[
                data["Class"].astype(int) == 0
            ]

        else:

            choices = data.index

        if len(choices):

            row_idx = int(
                st.selectbox(
                    "Dataset row index",
                    choices.tolist(),
                    index=0,
                )
            )

            row = data.loc[row_idx]

            y = int(row["Class"])

            s = float(
                score(
                    model,
                    pd.DataFrame([
                        {
                            c: row[c]
                            for c in FEATURES
                        }
                    ]),
                )[0]
            )

            a, b, c = st.columns(3)

            a.metric(
                "Model score",
                f"{s:.6f}",
            )

            b.metric(
                "Known dataset label",
                "Fraud" if y else "Legitimate",
            )

            c.metric(
                "Screening result",
                "Flagged" if s >= threshold else "Below cutoff",
            )

            if s >= threshold:

                st.error(
                    "The model flags this row for review; "
                    "that is a screening recommendation."
                )

            else:

                st.info(
                    "The model does not flag this row at the "
                    "current cutoff. A missed fraud is possible."
                )

            st.markdown("#### Selected transaction fields")

            st.dataframe(
                pd.DataFrame([
                    row[FEATURES].to_dict()
                ]),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "V1–V28 are anonymized transformed features. "
                "Their original business meanings are not available."
            )

        else:

            st.warning("No rows found for this label.")

    else:

        st.info(
            "Transaction-level exploration requires a "
            "labelled transaction dataset."
        )

        st.markdown("#### What you can do")

        st.write(
            "Upload a CSV containing the 30 required model features "
            "in the Batch screening section to obtain "
            "transaction-level risk scores."
        )

        st.caption(
            "Individual historical fraud labels are not included "
            "in the public deployment. This keeps the original "
            "dataset separate from the deployed application."
        )


# ============================================================
# TAB 3: BATCH SCREENING
# ============================================================

with tabs[2]:

    st.subheader("Batch screening")

    st.write(
        "Upload a CSV containing Time, V1–V28 and Amount "
        "to screen multiple transactions."
    )

    uploaded = st.file_uploader(
        "Upload CSV with Time, V1–V28, Amount",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded is not None:

        try:

            raw = pd.read_csv(uploaded)

            if raw.empty:

                st.warning("The uploaded CSV is empty.")

            else:

                st.caption(
                    f"{len(raw):,} rows uploaded."
                )

                st.dataframe(
                    raw.head(8),
                    use_container_width=True,
                    hide_index=True,
                )

                missing = [
                    c for c in FEATURES
                    if c not in raw.columns
                ]

                if missing:

                    st.error(
                        "Missing required columns: "
                        + ", ".join(missing)
                    )

                else:

                    if st.button(
                        "Score batch",
                        type="primary",
                    ):

                        scores = score(model, raw)

                        result = raw.copy()

                        result["risk_score"] = scores

                        result["alert_threshold"] = threshold

                        result["alert"] = scores >= threshold

                        a, b, c = st.columns(3)

                        a.metric(
                            "Rows scored",
                            f"{len(result):,}",
                        )

                        b.metric(
                            "Rows flagged",
                            f"{int(result['alert'].sum()):,}",
                        )

                        c.metric(
                            "Flag rate",
                            f"{result['alert'].mean() * 100:.2f}%",
                        )

                        fig = px.histogram(
                            result,
                            x="risk_score",
                            nbins=40,
                            title="Score distribution for uploaded transactions",
                        )

                        fig.add_vline(
                            x=threshold,
                            line_dash="dash",
                            annotation_text="Cutoff",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                        st.caption(
                            "Each bar counts uploaded transactions "
                            "with scores in that range."
                        )

                        st.markdown(
                            "#### Highest-risk transactions"
                        )

                        st.dataframe(
                            result.sort_values(
                                "risk_score",
                                ascending=False,
                            ).head(100),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "Download scored CSV",
                            result.to_csv(
                                index=False
                            ).encode("utf-8"),
                            "txnsentry_scored.csv",
                            "text/csv",
                        )

        except Exception as e:

            st.error(
                f"Unable to process uploaded CSV: {e}"
            )


# ============================================================
# TAB 4: THRESHOLD & ERROR ANALYSIS
# ============================================================

with tabs[3]:

    st.subheader("Threshold trade-offs and error analysis")

    if analytics is None:

        st.info(
            "The original labelled transaction dataset is not "
            "available in the deployed application. The charts "
            "below use saved validation operating points instead."
        )

        if (
            not validation.empty
            and {
                "model",
                "policy",
                "threshold",
            }.issubset(validation.columns)
        ):

            rf_validation = validation[
                validation["model"].astype(str) == "Random Forest"
            ].copy()

            if not rf_validation.empty:

                numeric_columns = [
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "alerts",
                ]

                for col in numeric_columns:

                    if col in rf_validation.columns:

                        rf_validation[col] = pd.to_numeric(
                            rf_validation[col],
                            errors="coerce",
                        )

                rf_validation = rf_validation.dropna(
                    subset=["threshold"]
                ).sort_values("threshold")

                st.markdown(
                    "#### Saved Random Forest validation policies"
                )

                st.dataframe(
                    rf_validation,
                    use_container_width=True,
                    hide_index=True,
                )

                left, right = st.columns(2)

                with left:

                    metric_columns = [
                        c for c in [
                            "precision",
                            "recall",
                            "f1",
                            "f2",
                        ]
                        if c in rf_validation.columns
                    ]

                    if metric_columns:

                        fig = px.line(
                            rf_validation,
                            x="threshold",
                            y=metric_columns,
                            markers=True,
                            title="Validation quality metrics by policy",
                        )

                        fig.update_yaxes(
                            range=[0, 1],
                            title="Metric value",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                        st.caption(
                            "Each point represents a saved validation "
                            "policy, not a newly calculated result."
                        )

                with right:

                    if "alerts" in rf_validation.columns:

                        fig = px.bar(
                            rf_validation,
                            x="policy",
                            y="alerts",
                            title="Validation alert workload by policy",
                            text_auto=True,
                        )

                        fig.update_layout(
                            xaxis_title="Validation policy",
                            yaxis_title="Number of alerts",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                        st.caption(
                            "Shows the number of validation transactions "
                            "that would have been flagged under each "
                            "saved policy."
                        )

                # Display the saved F2 policy separately.
                f2_policy = rf_validation[
                    rf_validation["policy"].astype(str)
                    == "F2_optimal"
                ] if "policy" in rf_validation.columns else pd.DataFrame()

                if not f2_policy.empty:

                    policy = f2_policy.iloc[0]

                    st.markdown(
                        "#### Random Forest F2-optimal validation policy"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Selected threshold",
                        f"{float(policy['threshold']):.4f}",
                    )

                    if "precision" in policy:

                        c2.metric(
                            "Precision",
                            f"{float(policy['precision']):.1%}",
                        )

                    if "recall" in policy:

                        c3.metric(
                            "Recall",
                            f"{float(policy['recall']):.1%}",
                        )

                    if "f2" in policy:

                        c4.metric(
                            "F2 score",
                            f"{float(policy['f2']):.3f}",
                        )

            else:

                st.warning(
                    "Random Forest validation results were not found."
                )

        else:

            st.warning(
                "Saved validation operating points are unavailable. "
                "Check reports/metrics/validation_operating_points.csv."
            )

        if calibration:

            st.markdown(
                "#### Saved model calibration summary"
            )

            calibration_frame = pd.DataFrame([
                {
                    "model": model_name,
                    **metrics,
                }
                for model_name, metrics in calibration.items()
                if isinstance(metrics, dict)
            ])

            if not calibration_frame.empty:

                st.dataframe(
                    calibration_frame,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "Brier score alone cannot establish calibration, "
                "especially with severe class imbalance and "
                "class-weighted models."
            )

        st.warning(
            "These are saved validation results from model development. "
            "Changing the sidebar threshold does not recalculate them. "
            "Live confusion matrices, ROC curves and precision-recall "
            "curves require the labelled dataset."
        )

    else:

        df, scores = analytics

        y = df["Class"].astype(int).to_numpy()

        # Current threshold plus a stable threshold grid.
        grid = np.unique(
            np.r_[
                np.linspace(0.01, 0.99, 100),
                threshold,
            ]
        )

        rows = [
            {
                "threshold": float(t),
                **metrics_at(y, scores, float(t)),
            }
            for t in grid
        ]

        curve = pd.DataFrame(rows).sort_values(
            "threshold"
        )

        left, right = st.columns(2)

        with left:

            fig = px.line(
                curve,
                x="threshold",
                y=[
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                ],
                title="Quality metrics across cutoffs",
            )

            fig.add_vline(
                x=threshold,
                line_dash="dash",
                annotation_text="Current cutoff",
            )

            fig.update_yaxes(
                range=[0, 1],
                title="Metric value",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Precision answers how many alerts were fraud; "
                "recall answers how much fraud we caught. F1 "
                "balances precision and recall; F2 emphasizes recall."
            )

        with right:

            fig = px.line(
                curve,
                x="threshold",
                y="alerts",
                title="Investigation workload across cutoffs",
            )

            fig.add_vline(
                x=threshold,
                line_dash="dash",
                annotation_text="Current cutoff",
            )

            fig.update_yaxes(
                title="Rows to review"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "This approximates how many rows an analyst "
                "would need to inspect at each cutoff."
            )

        m = metrics_at(
            y,
            scores,
            threshold,
        )

        cm = np.array([
            [m["TN"], m["FP"]],
            [m["FN"], m["TP"]],
        ])

        left, right = st.columns(2)

        with left:

            fig = px.imshow(
                cm,
                x=[
                    "Predicted not flagged",
                    "Predicted flagged",
                ],
                y=[
                    "Actual legitimate",
                    "Actual fraud",
                ],
                text_auto=True,
                aspect="auto",
                title="Confusion matrix at current cutoff",
                color_continuous_scale="Blues",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "False positives create review work; "
                "false negatives are fraud cases missed by the model."
            )

        with right:

            fpr, tpr, _ = roc_curve(
                y,
                scores,
            )

            roc_auc = roc_auc_score(
                y,
                scores,
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC curve (AUC={roc_auc:.3f})",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random ranking",
                    line=dict(dash="dash"),
                )
            )

            fig.update_layout(
                title="ROC curve",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                yaxis_range=[0, 1],
                xaxis_range=[0, 1],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "ROC-AUC measures ranking separation across cutoffs. "
                "With severe imbalance, also inspect "
                "precision-recall performance."
            )

        precision_curve, recall_curve, _ = precision_recall_curve(
            y,
            scores,
        )

        ap = average_precision_score(
            y,
            scores,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recall_curve,
                y=precision_curve,
                mode="lines",
                name=f"Precision-recall (AP={ap:.3f})",
            )
        )

        fig.add_hline(
            y=float(y.mean()),
            line_dash="dash",
            annotation_text="Fraud prevalence baseline",
        )

        fig.update_layout(
            title="Precision-recall curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Average Precision summarizes this curve. The dashed "
            "baseline is the fraud prevalence; performance "
            "meaningfully above it is desirable."
        )

        st.markdown("#### Current cutoff metrics")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Precision",
            f"{m['precision']:.1%}",
        )

        c2.metric(
            "Recall",
            f"{m['recall']:.1%}",
        )

        c3.metric(
            "F2",
            f"{m['f2']:.3f}",
        )

        c4.metric(
            "Alerts",
            f"{m['alerts']:,}",
        )

        c5.metric(
            "False negatives",
            f"{m['FN']:,}",
        )

        st.warning(
            "If these metrics are calculated on the full local CSV "
            "used during model development, they are descriptive "
            "diagnostics, not an unbiased final test result. "
            "Use a separate untouched test set for final claims."
        )


# ============================================================
# TAB 5: MODEL COMPARISON
# ============================================================

with tabs[4]:

    st.subheader("Model comparison")

    if not comparison.empty:

        cols = [
            c for c in [
                "model",
                "validation_average_precision",
                "test_average_precision",
                "test_roc_auc",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_f2",
                "test_alerts",
                "false_positives",
                "false_negatives",
            ]
            if c in comparison.columns
        ]

        display = comparison[cols].copy()

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        metric_col = (
            "test_average_precision"
            if "test_average_precision" in display.columns
            else None
        )

        if metric_col:

            fig = px.bar(
                display.sort_values(
                    metric_col,
                    ascending=False,
                ),
                x="model",
                y=metric_col,
                title="Model comparison: test Average Precision",
                text_auto=".3f",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Average Precision is especially relevant when fraud "
                "is rare. The displayed test set has already been "
                "inspected during model comparison."
            )

        st.warning(
            "Treat saved test metrics as exploratory and avoid "
            "repeated model or threshold selection against the "
            "same test set."
        )

    else:

        st.info(
            "Model comparison report not found. Check "
            "reports/metrics/model_comparison.csv."
        )

    if not validation.empty:

        st.markdown("#### Validation threshold policies")

        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True,
        )

    if calibration:

        st.markdown("#### Calibration summary")

        calibration_frame = pd.DataFrame([
            {
                "model": model_name,
                **metrics,
            }
            for model_name, metrics in calibration.items()
            if isinstance(metrics, dict)
        ])

        if not calibration_frame.empty:

            st.dataframe(
                calibration_frame,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Brier score alone cannot establish calibration, "
            "especially with severe class imbalance and "
            "class-weighted models."
        )


# ============================================================
# TAB 6: FEATURE GUIDE
# ============================================================

with tabs[5]:

    st.subheader("Feature guide and responsible use")

    guide = pd.DataFrame([
        {
            "Feature": "Time",
            "Meaning": (
                "Elapsed seconds from the dataset's reference point; "
                "not a calendar timestamp."
            ),
        },
        {
            "Feature": "Amount",
            "Meaning": (
                "Transaction amount as stored in the source dataset."
            ),
        },
        {
            "Feature": "V1–V28",
            "Meaning": (
                "Anonymized PCA-derived transformed numerical features. "
                "Original business meanings are not disclosed."
            ),
        },
        {
            "Feature": "Class",
            "Meaning": (
                "Historical label: 0 = legitimate, 1 = fraud. "
                "Used only for evaluation; never passed to the "
                "scoring model."
            ),
        },
    ])

    st.dataframe(
        guide,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Why accuracy alone is misleading")

    st.write(
        "Fraud is a tiny minority. A model predicting every row "
        "as legitimate could have high accuracy while detecting "
        "no fraud. Use precision, recall, Average Precision, "
        "and alert workload together."
    )

    st.markdown("### Limits")

    st.write(
        "Scores are not calibrated probabilities. The anonymized "
        "features cannot be translated into human-readable causes. "
        "This portfolio demo does not include production "
        "authentication, audit logging, drift monitoring, or "
        "automated case management. Every alert requires human review."
    )

    st.markdown("### Responsible use")

    st.write(
        "TxnSentry is a portfolio demonstration of a machine-learning "
        "screening workflow. It should not be used as a production "
        "fraud prevention system or as the sole basis for financial "
        "decisions."
    )
