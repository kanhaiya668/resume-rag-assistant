# Loan Approval & Risk Prediction

An end-to-end machine learning system that predicts loan approval outcomes and explains *why* — built with a full ML pipeline, explainable AI (SHAP), and a production-style FastAPI backend.

## Overview

This project goes beyond a simple classifier: it compares multiple models, investigates *why* the best one works, exposes a real REST API, and serves both single and bulk predictions through a styled web interface — with every prediction backed by a transparent, feature-level explanation.

## Key Insight

Exploratory analysis revealed that **CIBIL score alone is the dominant factor** in loan approval decisions — permutation importance places it far above every other feature (~0.43, next highest ~0.05). This mirrors real-world credit underwriting, where credit score is typically the primary gatekeeper before other financial factors are weighed.

## Dataset

[Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) (Kaggle) — 4,269 loan applications with income, CIBIL score, loan amount/term, and asset values (residential, commercial, luxury, bank).

## Model Comparison

Three models were trained and evaluated on a stratified 80/20 split:

| Model | AUC | Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 0.888 | 81% | Baseline linear model |
| Random Forest | 1.000 | 100% | Discarded — perfect score signals overfitting risk on this dataset |
| **HistGradientBoosting** | **1.000** | **99%** | **Selected** — near-perfect with realistic, non-overfit error margin |

**HistGradientBoostingClassifier** was chosen as the final model: strong performance without the perfect-score red flag that made Random Forest unreliable to present.

## Explainability (SHAP)

Every single prediction returns its **top 5 contributing factors** via SHAP (SHapley Additive exPlanations), so the model's decision is never a black box. Example: a rejected application might show `cibil_score: -0.38, loan_term: -0.05` — meaning a low CIBIL score, not income or assets, drove the rejection.

## Features

- **Single Prediction** — web form for one applicant, returns approval decision + probability + SHAP explanation
- **Bulk Prediction** — upload a CSV of applicants, download predictions for all of them at once
- **Dashboard** — dataset-wide statistics (approval rate, average CIBIL by outcome, etc.)
- **REST API** — `/api/predict` JSON endpoint with interactive Swagger docs at `/docs`

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **ML:** scikit-learn (HistGradientBoostingClassifier), SHAP
- **Data:** pandas, NumPy
- **Frontend:** Jinja2 templates, vanilla CSS
- **Model persistence:** joblib

## Project Structure

````
loan-approval-risk-prediction/
├── model/
│   ├── loan_model.pkl
│   └── feature_names.pkl
├── notebooks/
│   └── loan_risk.ipynb          # EDA, feature engineering, model comparison
├── templates/                   # Jinja2 HTML templates
├── static/
│   └── style.css
├── main.py                      # FastAPI application
├── requirements.txt
└── loan_approval_dataset.csv
````

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Single-application prediction form |
| POST | `/predict` | Submit form, get result page |
| POST | `/api/predict` | JSON API — prediction + SHAP explanation |
| GET | `/bulk` | Bulk CSV upload page |
| POST | `/bulk/predict` | Upload CSV, download predictions CSV |
| GET | `/dashboard` | Dataset statistics dashboard |
| GET | `/docs` | Interactive Swagger API documentation |

## Setup

```bash
git clone https://github.com/kanhaiya668/loan-approval-risk-prediction.git
cd loan-approval-risk-prediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` in your browser.

## Author

**Kanhaiya Bhardwaj**
[LinkedIn](https://www.linkedin.com/in/kanhaiya-bhardwaj-57788b375/) · [GitHub](https://github.com/kanhaiya668)
