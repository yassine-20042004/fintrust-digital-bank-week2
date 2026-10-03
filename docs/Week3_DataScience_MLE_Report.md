# FinTrust Digital Bank — Week 3 Data Science & ML Engineering Technical Report

**Author**: Data Science & ML Engineering Lead  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 3 — Develop & Integrate  
**Date**: October 2026  

---

## 1. Data Science Track: Predictive Risk Modelling & Validation

### Part A — Review of Week 2 Baseline Model
In Week 2, a baseline Logistic Regression model was established using 8 initial features. 
- **Baseline Performance**: Accuracy = 62.4%, ROC-AUC = 0.6730, Recall = 61.5%, F1-Score = 0.3903.
- **Weaknesses Identified**:
  1. High False Positive count (722 out of 2,400 test cases) leading to analyst alert fatigue.
  2. Limited feature interactions (omitted late-night transaction windows and historical customer failure ratios).
  3. Lack of tree ensemble evaluation to capture non-linear transaction patterns.

---

### Part B — Development & Evaluation of Additional Models
Four classification algorithms were trained and cross-evaluated using a 80/20 stratified split (12,000 total transaction records):
1. **Baseline Logistic Regression (Class-Weighted)**
2. **Decision Tree Classifier (Tuned Max Depth=8, Min Samples Leaf=5)**
3. **Random Forest Ensemble (150 Estimators, Max Depth=12, Class-Weighted)**
4. **XGBoost Classifier (150 Estimators, Learning Rate=0.05, Scale Pos Weight=4.12)**

---

### Part C — Feature Refinement Table
To improve predictive signal and reduce false alarms, 6 additional features were engineered, bringing the total feature count to 14.

| Feature Name | Feature Decision | Rationale / Business Logic | Expected Effect |
| :--- | :--- | :--- | :--- |
| `Is_Night_Tx` | **Addition** | Binary indicator for transactions executed between 22:00 and 05:00. | Captures non-standard operational hour risk. |
| `International_Weekend_Interact` | **Addition** | Interaction product `Is_International * Is_Weekend`. | Highlights high-risk cross-border weekend transfers. |
| `Amount_to_Income_Ratio` | **Transformation** | Transaction `Amount_NGN` divided by estimated monthly income band. | Detects disproportionately large transfers relative to customer income. |
| `Tenure_to_Age_Ratio` | **Transformation** | Customer tenure in months divided by customer age in years. | Normalizes customer loyalty against age demographic. |
| `Cust_Failed_Tx_Ratio` | **Aggregation** | Proportion of historical failed transactions for each customer. | Flags account friction or potential brute-force testing. |
| `Channel_Risk_Rate` | **Encoding** | Target-encoded historical risk review rate per transaction channel. | Incorporates channel-specific risk weights. |

---

### Part D — Model Comparison Matrix & Trade-Off Analysis

#### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | True Negatives | False Positives | False Negatives | True Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | **0.6238** | **0.2859** | **0.6149** | **0.3903** | **0.6730** | **1,208** | **722** | **181** | **289** |
| Decision Tree Classifier | 0.6362 | 0.2863 | 0.5745 | 0.3822 | 0.6354 | 1,257 | 673 | 200 | 270 |
| Random Forest Classifier | 0.7729 | 0.3574 | 0.2000 | 0.2565 | 0.6487 | 1,761 | 169 | 376 | 94 |
| XGBoost Classifier | 0.6683 | 0.2827 | 0.4511 | 0.3475 | 0.6390 | 1,392 | 538 | 258 | 212 |

#### Metric Trade-Off Analysis
- **Why Accuracy is Insufficient**: Due to synthetic target imbalance (19.6% positive risk flags), Random Forest achieved the highest Accuracy (77.3%) but suffered a severely low Recall of **20.0%** (missing 376 out of 470 risk cases).
- **Selection Rationale**: Baseline Logistic Regression achieves the highest **Recall (61.5%)**, **F1-Score (0.3903)**, and **ROC-AUC (0.6730)**, ensuring the maximum number of potential risk events are captured for manual review.

---

### Part E — Error Analysis
- **False Positives (FP = 722)**: Primarily triggered on high-value domestic transfers (>150,000 NGN) executed by high-income customers during standard business hours.
- **False Negatives (FN = 181)**: Low-value international transactions (<25,000 NGN) executed via POS terminals, which bypassed traditional monetary magnitude filters.

---

### Part F — Model Interpretation & Feature Importance
Top 5 contributing features across tree-based and linear coefficients:
1. `Is_International` (Cross-border flag)
2. `Amount_to_Mean_Ratio` (Deviation from customer spending baseline)
3. `Channel_Risk_Rate` (Channel-level historical vulnerability)
4. `Log_Amount` (Monetary scale)
5. `Cust_Failed_Tx_Ratio` (Historical transaction failure rate)

---

### Part G — Candidate Final Model Selection
- **Selected Candidate**: Class-Weighted Logistic Regression Pipeline.
- **Serialization**: Saved to `models/candidate_model.pkl`.
- **Educational Disclaimer**: Synthetic target construct created solely for technical demonstration; must not be represented as live commercial fraud detection.

---

## 2. Machine Learning Engineering Track: Integration-Ready Workflow & REST API

### Part A — Workflow Architecture
The end-to-end ML pipeline implements standard MLOps modularization:
```
Input Data → Validation Module → Preprocessing → Feature Engineering → Model Inference → Prediction Response
```

### Part B — Reusable Modules Created
- [`src/validation.py`](file:///c:/Users/LENOVO/Desktop/work%20st/fintrust-digital-bank-week2/src/validation.py): Schema integrity, type checking, boundary enforcement.
- [`src/preprocessing.py`](file:///c:/Users/LENOVO/Desktop/work%20st/fintrust-digital-bank-week2/src/preprocessing.py): Feature transformations & scikit-learn column transformer.
- [`src/models.py`](file:///c:/Users/LENOVO/Desktop/work%20st/fintrust-digital-bank-week2/src/models.py): Algorithm training, evaluation, comparison, serialization.
- [`src/pipeline.py`](file:///c:/Users/LENOVO/Desktop/work%20st/fintrust-digital-bank-week2/src/pipeline.py): Orchestration class `FinTrustMLPipeline`.
- [`src/api.py`](file:///c:/Users/LENOVO/Desktop/work%20st/fintrust-digital-bank-week2/src/api.py): FastAPI REST service.

---

### Part C — REST API Specification (`src/api.py`)

#### Endpoints:
1. `GET /` — System status and API documentation URL.
2. `GET /health` — Health check verifying candidate model load status.
3. `GET /model-info` — Metadata on active model, engineered features, and performance.
4. `POST /predict` — Real-time transaction risk scoring.

#### Example Request Payload:
```json
{
  "customer": {
    "Customer_ID": "FT-C10001",
    "Age": 34,
    "Gender": "Female",
    "City": "Lagos",
    "Customer_Segment": "Everyday",
    "Account_Type": "Savings",
    "Tenure_Months": 18,
    "Digital_Engagement_Score": 82.0,
    "Monthly_Income_Band": "250k-499k",
    "Preferred_Channel": "Mobile App"
  },
  "transaction": {
    "Transaction_ID": "FT-T50001",
    "Customer_ID": "FT-C10001",
    "Transaction_DateTime": "2026-03-15 14:30:00",
    "Transaction_Type": "Transfer",
    "Amount_NGN": 150000.0,
    "Channel": "Mobile App",
    "International_Transaction": "No",
    "Transaction_Status": "Successful"
  }
}
```

#### Example Response Payload:
```json
{
  "transaction_id": "FT-T50001",
  "customer_id": "FT-C10001",
  "amount_ngn": 150000.0,
  "channel": "Mobile App",
  "risk_probability": 0.4285,
  "predicted_risk_flag": "No",
  "risk_tier": "Medium",
  "risk_factors": [
    "High monetary value exceeds 85th percentile threshold"
  ]
}
```

---

### Part D — Technical Testing Suite (`pytest`)
15 automated test cases executed across unit and integration suites (100% pass rate):
- `test_1_valid_input`: Schema compliance verification.
- `test_2_empty_input`: Rejection of zero-row DataFrames.
- `test_3_unexpected_categories`: Rejection of unauthorized channels.
- `test_4_invalid_data_types`: Catching non-numeric string data in Age column.
- `test_5_boundary_checks`: Verification of negative amount & out-of-range age boundaries.
- `test_6_feature_engineering`: Verification of 14 engineered features.
- `test_7_model_loading`: Serialized model file integrity check.
- `test_8_prediction_generation`: Output range validation [0.0, 1.0].
- `test_9_output_format`: DataFrame output structure test.
- `test_10_reproducibility`: Deterministic inference verification.
- `test_api_*` (5 API endpoint tests): REST endpoint integration test.

---

### Part E — Reproducibility Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Full ML Pipeline (CLI)**:
   ```bash
   python run_pipeline.py
   ```
3. **Run Automated Test Suite**:
   ```bash
   pytest -v
   ```
4. **Launch Local REST API Server**:
   ```bash
   uvicorn src.api:app --reload --port 8000
   ```
