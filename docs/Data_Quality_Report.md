# FinTrust Digital Bank — Data Quality & Audit Report

**Phase**: Week 2 — Data Audit & Health Check  
**Target Datasets**: `FinTrust_Customer_Data.csv` (1,500 records), `FinTrust_Transaction_Data.csv` (12,000 records)

---

## 1. Executive Summary
A comprehensive data quality audit was conducted across customer demographic profiles and financial transaction logs. Schema integrity, boundary validity, missingness patterns, and categorical distributions were inspected to establish ground-truth reliability for downstream BI and predictive ML modeling.

## 2. Customer Dataset Audit (`FinTrust_Customer_Data`)
- **Total Records**: 1,500 customers
- **Schema Validation**: 100% complete across all 12 required fields (`Customer_ID`, `Customer_Name`, `Age`, `Gender`, `City`, `Customer_Segment`, `Account_Type`, `Tenure_Months`, `Digital_Engagement_Score`, `Monthly_Income_Band`, `Preferred_Channel`, `Account_Status`).
- **Completeness**: 0 missing or null values.
- **Boundary Checks**:
  - `Age`: Ranged strictly between valid boundaries (18–75).
  - `Digital_Engagement_Score`: Valid distribution in [0.0, 100.0].
  - `Tenure_Months`: Non-negative integers.

## 3. Transaction Dataset Audit (`FinTrust_Transaction_Data`)
- **Total Records**: 12,000 transactions
- **Monetary Scale**: 560,477,354.85 NGN total audited volume; mean ticket size 46,706.45 NGN.
- **Missingness Pattern**:
  - `Device_Type`: 96 missing records (0.8%). Imputed with categorical token `'Unknown'`.
  - `Location`: 96 missing records (0.8%). Imputed with categorical token `'Unknown'`.
  - All other fields (Amount, Channel, Status, Risk Flag) present with 0 missingness.
- **Boundary & Validation Constraints**:
  - `Amount_NGN`: All values strictly > 0 NGN.
  - `Channel`: Restricted to whitelist (`Mobile App`, `Web`, `ATM`, `POS`, `USSD`).
  - `Transaction_Status`: Whitelisted (`Successful`, `Failed`, `Reversed`, `Pending`).

## 4. Remediation & Preprocessing Pipeline
1. Categorical Missingness Imputation: Imputed missing `Device_Type` and `Location` values with `'Unknown'` to retain full transaction history and preserve financial volume calculation.
2. Datetime Parsing: Standardized `Transaction_DateTime` into native datetime formats for extraction of hour-of-day, day-of-week, and weekend indicators.
3. Feature Engineering: Created 8 engineered features (`Tx_Hour`, `Tx_DayOfWeek`, `Is_Weekend`, `Log_Amount`, `Is_International`, `Cust_Lifetime_Tx`, `Amount_to_Mean_Ratio`, `Is_High_Value`).
