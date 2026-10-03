# FinTrust Digital Bank — Week 3 Project Summary

**Intern Name**: FinTrust Lead Data Scientist & ML Engineer  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 3 — Develop & Integrate  
**Date**: October 2026  

---

### 1. What You Planned to Accomplish
- Transition the Week 2 baseline risk model into a multi-model comparative study.
- Engineer 6 additional refined behavioural features (increasing feature space from 8 to 14).
- Modularize the ML pipeline into reusable production modules (`validation.py`, `preprocessing.py`, `models.py`, `pipeline.py`).
- Implement an automated REST API service using FastAPI for real-time transaction risk scoring.
- Expand unit and integration test coverage to verify missingness, boundary conditions, API endpoints, and reproducibility.

### 2. What You Completed
- Trained and evaluated 4 distinct classification algorithms (Baseline Logistic Regression, Decision Tree, Random Forest, XGBoost Classifier).
- Built a 14-feature engineering pipeline capturing late-night transaction windows, income ratios, and cross-border interaction terms.
- Produced model comparison tables, error analysis (FP/FN breakdowns), confusion matrices, and feature importance charts saved to `reports/figures/`.
- Built and verified a FastAPI service (`src/api.py`) featuring `/health`, `/model-info`, and `/predict` endpoints.
- Developed a 15-test automated pytest suite achieving 100% pass status.

### 3. Major Development Activities
- **Data Science**: Evaluated recall vs. precision trade-offs across 2,400 test transactions. Selected Class-Weighted Logistic Regression as candidate model due to optimal Recall (61.5%) and ROC-AUC (0.6730).
- **ML Engineering**: Built `FinTrustMLPipeline` class supporting single-record and batch inference. Serialized candidate pipeline to `models/candidate_model.pkl`.
- **API Development**: Defined Pydantic v2 validation models for customer and transaction payloads.

### 4. Key Findings / Results
- **Model Performance Comparison**:
  - Baseline Logistic Regression: Accuracy 62.4%, Recall 61.5%, F1-Score 0.3903, ROC-AUC 0.6730.
  - Decision Tree: Accuracy 63.6%, Recall 57.5%, F1-Score 0.3822, ROC-AUC 0.6354.
  - Random Forest: Accuracy 77.3%, Recall 20.0%, F1-Score 0.2565, ROC-AUC 0.6487.
  - XGBoost: Accuracy 66.8%, Recall 45.1%, F1-Score 0.3475, ROC-AUC 0.6390.
- **Metric Insights**: High accuracy in tree ensembles (77.3%) concealed severe recall degradation (missing 80% of risk cases). Recall-focused tuning proved essential.

### 5. Testing and Validation Performed
- **Pipeline & Schema Validation**: Verified rejection of out-of-range ages, non-positive amounts, unwhitelisted channels, non-numeric strings, and empty dataframes.
- **REST API Testing**: Validated `POST /predict` handling of valid and malformed JSON payloads.
- **Reproducibility Test**: Confirmed zero variance in prediction probability on duplicate inputs.

### 6. Improvements Made
- **Evidence-Based Refinement**:
  `Initial Output (Week 2 Baseline) → Test (Pytest & Model Comparison) → Finding (High FP & low tree recall) → Refinement (Feature expansion & class weighting) → Re-test (15 Pytest cases) → Result (Validated API & Candidate Model)`

### 7. Major Challenges
- **Target Imbalance**: 19.6% synthetic target rate caused tree models to overfit to the majority class ('No Risk'). Solved via `class_weight='balanced'` and `scale_pos_weight`.
- **Data Type Integrity**: Non-numeric strings in numeric schema fields caused Pandas comparison errors. Resolved by adding explicit `pd.to_numeric` parsing in `src/validation.py`.

### 8. Important Decisions
- **Candidate Selection**: Selected Logistic Regression over Random Forest due to 3x higher recall for risk detection (61.5% vs 20.0%).
- **API Framework**: Adopted FastAPI for auto-generated OpenAPI documentation and Pydantic runtime schema enforcement.

### 9. Remaining Limitations
- **Synthetic Educational Target**: `Risk_Review_Flag` is synthetic and must not be used for real commercial credit or fraud decisions.
- **Static In-Memory Pipeline**: Current inference pipeline relies on local pickle serialization without model registry hosting.

### 10. What Must Be Completed in Week 4
- Package the FastAPI service into a Docker container.
- Build final executive slides and integration dashboard for Week 4 final presentation.
- Conduct final end-to-end integration across all 5 tracks.

### 11. Your Final-Week Priorities
- Dockerize API application (`Dockerfile` & `docker-compose.yml`).
- Perform final performance load testing on REST endpoints.
- Complete final project presentation deck and GitHub repository release.
