# 🛡️ TxnSentry — Financial Fraud Detection & Risk Analytics

An end-to-end machine learning and analytics application that scores credit-card transactions, identifies potentially suspicious activity, and helps prioritize transactions for human review.

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Random Forest · SQLite · Streamlit**

---

## Overview

Credit-card fraud detection is an imbalanced classification problem: fraudulent transactions represent a very small proportion of all transactions.

TxnSentry explores this problem through data preparation, exploratory analysis, model comparison, threshold optimization, feature importance, SQL analytics, and an interactive Streamlit dashboard.

The application supports:

- Transaction-level risk scoring
- Batch CSV screening and downloadable results
- Adjustable alert thresholds
- Precision, recall and review-workload analysis
- Model performance comparison
- Feature importance visualization
- SQL-based transaction analytics

> **Important:** TxnSentry is a portfolio demonstration. Its model scores are not calibrated real-world fraud probabilities. Predictions should support human review, not automatically block transactions.

## Live application

- **Streamlit demo:** https://txnsentry-fraud-detection.streamlit.app/
- **GitHub repository:** https://github.com/Padamgupta-23/TxnSentry-fraud-detection

## Dataset

This project uses the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset contains anonymized credit-card transactions with 30 model input features and a binary target:

| Field | Description |
|---|---|
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset |
| `V1`–`V28` | Anonymized PCA-transformed numerical features |
| `Amount` | Transaction amount |
| `Class` | Target label: 0 = legitimate, 1 = fraud |

The target column `Class` is excluded from model prediction inputs.

### Dataset preparation

| Data quality metric | Result |
|---|---:|
| Original records | 284,807 |
| Exact duplicate rows removed | 1,081 |
| Modeling records | 283,726 |
| Legitimate transactions | 283,253 |
| Fraudulent transactions | 473 |
| Fraud prevalence after deduplication | 0.1667% |

The raw dataset is preserved separately from the processed modeling dataset.

## Machine learning

Three classification approaches were compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

The workflow uses stratified data splitting and selects classification thresholds using validation data. Average Precision (PR-AUC) is emphasized because the target is highly imbalanced.

### Exploratory held-out test comparison

| Model | Average Precision | ROC-AUC | Precision | Recall | F2 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7308 | 0.9670 | 0.8605 | 0.7789 | 0.7940 |
| Random Forest | 0.8564 | 0.9656 | 0.9605 | 0.7684 | 0.8004 |
| XGBoost | 0.8524 | 0.9758 | 0.9500 | 0.8000 | 0.8261 |

These figures are from the project's existing model-comparison run. The test results have already been inspected during model comparison, so they should be treated as **exploratory rather than as a pristine final evaluation**. Avoid further tuning against this same test set.

### Threshold selection

A model's score is converted into a review flag using a configurable cutoff:

`Flag for review when score >= threshold`

Changing the threshold affects:

- Precision: proportion of flagged transactions that are actually fraud
- Recall: proportion of fraudulent transactions detected
- False positives: legitimate transactions sent for review
- False negatives: fraudulent transactions missed
- Alert volume: total transactions requiring review

The dashboard's current default cutoff is approximately `0.3696`. It is an operating demonstration setting, not a universally optimal production threshold.

## Feature importance

Feature importance was generated for Random Forest and XGBoost.

The leading features in both models include `V14`, `V12`, `V10`, and `V4`. `V14` ranked highest in both model reports.

These are model-derived importance measures, not causal explanations. Since `V1`–`V28` are anonymized PCA-transformed features, their real-world meanings cannot be inferred from their names.

## SQL analytics

The processed dataset is also stored in a SQLite database for analytical queries.

The SQL workflow produces:

- Transaction and fraud summary
- Transaction amount statistics by class
- Fraud distribution across elapsed-time buckets
- Average selected feature values by class
- Highest-amount fraudulent transactions

### Selected SQL findings

| Metric | Legitimate | Fraud |
|---|---:|---:|
| Transaction count | 283,253 | 473 |
| Average amount | 88.41 | 123.87 |
| Maximum amount | 25,691.16 | 2,125.87 |

The `Time` field represents elapsed seconds, not clock time. The corresponding SQL grouping is therefore an elapsed-time analysis, not an analysis of time-of-day behavior.

## Dashboard

The Streamlit application provides the following sections:

- **Executive overview:** dataset scale, fraud prevalence and screening summary
- **Transaction explorer:** score an individual transaction
- **Batch screening:** upload a CSV, score multiple transactions and download results
- **Threshold & errors:** explore cutoff trade-offs and confusion-matrix outcomes
- **Model comparison:** compare model metrics and validation threshold policies
- **Feature guide:** explain input fields and model limitations

The overview's full-dataset screening metrics are descriptive results on the available labelled local dataset. They are not independent held-out test metrics.

## Technology stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn, Streamlit charts
- **Database:** SQLite
- **Model persistence:** Joblib
- **Testing:** Python unittest

## Project structure

```text
TxnSentry/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   ├── sql/
│   └── txnsentry.db
│
├── sql/
│
├── src/
│   ├── inspect_data.py
│   ├── eda.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── compare_models.py
│   ├── validation_analysis.py
│   ├── feature_importance.py
│   ├── build_sql_database.py
│   └── run_sql_analytics.py
│
└── tests/
    └── test_model_inference.py
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Padamgupta-23/TxnSentry-fraud-detection
cd TxnSentry
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from the Kaggle dataset and place it at:

```text
data/raw/creditcard.csv
```

### 5. Prepare the data and run analysis

```powershell
python src/inspect_data.py
python src/eda.py
python src/prepare_data.py
```

### 6. Train and compare models

```powershell
python src/train_baseline.py
python src/compare_models.py
python src/validation_analysis.py
python src/feature_importance.py
```

### 7. Run SQL analytics

```powershell
python src/build_sql_database.py
python src/run_sql_analytics.py
```

### 8. Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

### 9. Run tests

```powershell
python -m unittest discover -s tests -v
```

## Limitations and responsible use

- The dataset is historical and anonymized; it does not represent a live banking transaction stream.
- Fraud examples are extremely rare, so accuracy alone is not an adequate performance measure.
- The anonymized PCA features do not provide direct customer or transaction explanations.
- Class-weighted model scores should not automatically be interpreted as calibrated probabilities.
- Threshold selection depends on the review team's capacity and the relative costs of missed fraud and false alerts.
- The existing test comparison has already been inspected and should not be repeatedly used for model selection.
- This prototype is intended for learning, demonstration and human-review prioritization, not production fraud prevention.

## Future improvements

- Evaluate on a new untouched test set or a time-based holdout
- Add robust probability calibration and calibration diagnostics
- Add local explanations for individual predictions where technically appropriate
- Introduce monitoring for data drift and model performance
- Add a production-style inference API and model versioning
- Expand automated tests for malformed CSV files and missing input features

---

## 👨‍💻 About Me

**Padam Gupta**  
B.Tech CSE | Data Science & Machine Learning

- **GitHub:** [@Padamgupta-23](https://github.com/Padamgupta-23)
- **LinkedIn:** [Connect with me](https://www.linkedin.com/in/padam-gupta/)

I'm always learning, building, and turning ideas into useful projects.

If you find this project interesting, consider giving the repository a ⭐!

### © 2026 Padam Gupta

Made with curiosity, code, and love❤️ All rights reserved.
---
