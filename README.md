# FinTrust Digital Bank — Financial Intelligence & ML Support Solution

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Pytest-15%20Passed-success.svg)](tests/)
[![Phase](https://img.shields.io/badge/Phase-Week%203%20Develop%20%26%20Integrate-orange.svg)]()

Production-grade Machine Learning pipeline, predictive risk models, REST API inference service, and data analytics solution for **FinTrust Digital Bank** (AnalystLab Africa Internship Project).

---

## 🚀 Week 3 Highlights & Key Deliverables

1. **Feature Engineering Pipeline**: Engineered 14 predictive features including cross-border interaction terms, late-night transaction indicators, and income-to-amount ratios.
2. **Model Evaluation & Comparison**: Trained and cross-evaluated 4 classification algorithms (Baseline Logistic Regression, Decision Tree, Random Forest, XGBoost).
3. **Candidate Model Selection**: Candidate model serialized to `models/candidate_model.pkl` (Class-Weighted Logistic Regression selected for optimal 61.5% Recall and 0.6730 ROC-AUC).
4. **FastAPI REST Service**: Production REST API (`src/api.py`) exposing `/health`, `/model-info`, and `/predict` endpoints.
5. **Automated Test Suite**: 15 unit and integration tests (`tests/test_pipeline.py`, `tests/test_api.py`) passing with 100% success rate.
6. **Detailed Documentation**: Comprehensive technical reports in [`docs/Week3_DataScience_MLE_Report.md`](docs/Week3_DataScience_MLE_Report.md) and [`docs/Week3_Project_Summary.md`](docs/Week3_Project_Summary.md).

---

## 📁 Repository Directory Structure

```
fintrust-digital-bank-week3/
│
├── data/
│   ├── raw/                       # Raw customer & transaction CSV files
│   └── processed/                 # Cleaned datasets & model comparison CSVs
│
├── docs/                          # Week 3 Technical Reports & Summaries
│   ├── Week3_DataScience_MLE_Report.md
│   ├── Week3_Project_Summary.md
│   └── Data_Quality_Report.md
│
├── models/                        # Serialized production candidate model
│   └── candidate_model.pkl
│
├── reports/                       # Visualizations & HTML Dashboards
│   ├── figures/                   # Model comparison, confusion matrix & feature importance charts
│   └── dashboard.html             # Interactive Web Dashboard
│
├── src/                           # Reusable Production Modules
│   ├── __init__.py
│   ├── api.py                     # FastAPI REST Service
│   ├── assistant.py               # Grounded GenAI Support Assistant
│   ├── models.py                  # Model training, comparison & serialization
│   ├── pipeline.py                # End-to-End FinTrustMLPipeline Class
│   ├── preprocessing.py           # Feature Engineering & Column Transformer
│   └── validation.py              # Schema & Boundary Validation Suite
│
├── tests/                         # Pytest Automated Test Suite
│   ├── test_api.py                # REST API Endpoint Tests
│   └── test_pipeline.py           # ML Pipeline & Schema Tests
│
├── README.md                      # Project Documentation
├── run_pipeline.py                # CLI Execution Script
└── requirements.txt               # Dependency Locking
```

---

## 🛠️ Quick Start & Reproducibility Guide

### 1. Installation
Ensure Python 3.10+ is installed, then install all required packages:
```bash
pip install -r requirements.txt
```

### 2. Execute End-to-End ML Pipeline
Run the CLI script to execute schema validation, feature engineering, 4-model evaluation, figure generation, and candidate model saving:
```bash
python run_pipeline.py
```

### 3. Run Automated Test Suite
Execute unit and API integration tests:
```bash
pytest -v
```

### 4. Launch REST API Server
Start local FastAPI web server:
```bash
uvicorn src.api:app --reload --port 8000
```
Then navigate to `http://127.0.0.1:8000/docs` in your web browser for the interactive OpenAPI documentation.

---

## 📊 Model Comparison Overview

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | **62.4%** | **28.6%** | **61.5%** | **0.3903** | **0.6730** | **Selected Candidate** |
| Decision Tree Classifier | 63.6% | 28.6% | 57.5% | 0.3822 | 0.6354 | Evaluated |
| Random Forest Classifier | 77.3% | 35.7% | 20.0% | 0.2565 | 0.6487 | Evaluated |
| XGBoost Classifier | 66.8% | 28.3% | 45.1% | 0.3475 | 0.6390 | Evaluated |

---

## ⚠️ Educational Disclaimer
All customer profiles, transactions, and risk review flags are **synthetic educational constructs** created for the AnalystLab Africa internship. The `Risk_Review_Flag` target must not be represented as real commercial fraud detection.
