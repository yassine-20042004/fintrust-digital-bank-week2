# FinTrust Digital Bank — Week 2 Project Summary

**Intern Name**: FinTrust Analyst / Engineer  
**Project**: FinTrust Financial Intelligence & Digital Banking Support Solution  
**Phase**: Week 2 — Analyse & Prepare

---

### 1. Planned vs. Actually Accomplished
- **Planned**: Audit dataset quality, execute SQL queries for business intelligence, engineer candidate risk features, build baseline classification models, and implement a grounded GenAI support prototype.
- **Accomplished**: Cleaned and validated both the Customer (1,500 rows) and Transaction (12,000 rows) datasets, executed 8 production SQL business queries, built an automated technical validation suite (5 tests), evaluated Logistic Regression and Random Forest baseline models, and tested a support assistant prototype against 15 evaluation scenarios.

### 2. Tools Used
- **Data Analytics & Database**: Microsoft Excel, PostgreSQL / DuckDB, Microsoft Power BI.
- **Data Science & ML Engineering**: Python 3.10+, Pandas, NumPy, Scikit-learn, Pytest.
- **Generative AI**: Python rule-based contextual retrieval engine, Knowledge Base grounding matrix.

### 3. Key Findings & Outcomes
- **Financial Scale**: 12,000 transactions totaling 560,477,354.85 NGN were audited, with an average ticket of 46,706.45 NGN.
- **Channel Friction**: Mobile App represents 42.5% of overall volume (5,102 transactions) but exhibits the highest failure rate across digital channels (5.80%).
- **Cross-Border Risk Concentration**: International transactions represent only 4.0% of volume (480 transactions) but trigger risk review flags at 36.88% (compared to 18.88% for domestic).
- **Engagement Value**: Highly engaged digital users (score >= 80) complete 8.41 transactions on average, compared to 7.62 for lower-engagement tiers.

### 4. Major Challenges & Decisions
- **Missingness Handling**: 96 missing records in `Device_Type` and `Location` (0.8%) were imputed with `'Unknown'` rather than dropped, preserving the 560.5M NGN total monetary volume.
- **Target Imbalance**: Synthetic `Risk_Review_Flag` is present in 19.6% of records. ROC-AUC and F1-Score were selected as primary metrics over accuracy to avoid majority-class bias.
- **Hallucination Prevention**: Because the approved knowledge base lacks fee and limit schedules, the GenAI assistant was explicitly programmed to deflect limit inquiries rather than hallucinate numbers.

### 5. Testing & Quality Evaluation
- **MLOps Tests**: 5 pytest test cases passed, verifying empty-dataframe detection, channel whitelist enforcement, age limits, and negative-amount validation.
- **GenAI Tests**: 15/15 evaluation questions passed, maintaining zero credential leakage and zero policy hallucinations.

### 6. Limitations & Disclaimer
- **Synthetic Data Disclaimer**: All customer profiles, transactions, and risk review flags are synthetic educational constructs and must not be used for real-world fraud or credit decisions.
- **Knowledge Base Scope**: The assistant is restricted to standard informational procedures and cannot query real customer account balances.

### 7. Week 3 Proposed Focus
- Transition baseline models into tuned, cross-validated classification pipelines.
- Build interactive Power BI dashboards with user-driven scenario filtering.
- Package the MLOps pipeline into a containerized inference endpoint.
