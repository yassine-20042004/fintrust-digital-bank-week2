# FinTrust Financial Intelligence & Digital Banking Solution (Week 2)

Experience Lab Internship Programme — AnalystLab Africa  
Project Phase: Week 2 — Analyse & Prepare

## Executive Summary
This repository contains the Week 2 deliverables for FinTrust Digital Bank. It transitions strategic planning into practical technical execution across data auditing, SQL business intelligence, exploratory data analysis, baseline machine learning modeling, and grounded customer support assistant development.

## Directory Structure
- `data/`: Raw source data and cleaned/processed datasets.
- `sql/`: 8 production-grade SQL analytical business queries.
- `notebooks/`: Python EDA and visualization notebook.
- `src/`: Core Python modules for data validation, feature engineering, modeling, and GenAI assistant logic.
- `tests/`: Automated unit test suite verifying schema and boundary constraints.
- `docs/`: Evaluation matrices, data quality audits, and the Week 2 Project Summary.

## Installation & Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Components

### Execute Automated Technical Tests:
```bash
pytest tests/test_pipeline.py -v
```

### Run Data Validation and Baseline Modeling:
```bash
python src/train_baseline.py
```

### Run Support Assistant Grounding Test:
```bash
python src/assistant.py
```

## Disclaimer
All data, transactions, customer profiles, and risk flags are synthetic and created for educational purposes within the AnalystLab Africa Experience Lab. The Risk_Review_Flag must not be treated as a real-world fraud determination.
