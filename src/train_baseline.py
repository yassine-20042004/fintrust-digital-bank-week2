"""Baseline Risk Review Classification Training Script."""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from src.preprocessing import engineer_features, get_preprocessor
    from src.validation import clean_datasets, validate_customer_data, validate_transaction_data
except ImportError:
    from preprocessing import engineer_features, get_preprocessor
    from validation import clean_datasets, validate_customer_data, validate_transaction_data


def find_data_file(filenames):
    """Finds existing file from candidate relative/absolute paths."""
    for fn in filenames:
        if os.path.exists(fn):
            return fn
    raise FileNotFoundError(f"Could not locate data file among: {filenames}")


def main():
    print("Loading raw FinTrust datasets...")
    cust_candidates = [
        "data/raw/FinTrust_Customer_Data.csv",
        "data/raw/FinTrust_Customer_Data - FinTrust_Customer_Data (1).csv",
        "FinTrust_Customer_Data - FinTrust_Customer_Data (1).csv",
        "FinTrust_Customer_Data.csv",
        "../data/raw/FinTrust_Customer_Data.csv",
        "../data/raw/FinTrust_Customer_Data - FinTrust_Customer_Data (1).csv"
    ]
    tx_candidates = [
        "data/raw/FinTrust_Transaction_Data.csv",
        "data/raw/FinTrust_Transaction_Data - FinTrust_Transaction_Data (1).csv",
        "FinTrust_Transaction_Data - FinTrust_Transaction_Data (1).csv",
        "FinTrust_Transaction_Data.csv",
        "../data/raw/FinTrust_Transaction_Data.csv",
        "../data/raw/FinTrust_Transaction_Data - FinTrust_Transaction_Data (1).csv"
    ]

    cust_path = find_data_file(cust_candidates)
    tx_path = find_data_file(tx_candidates)

    df_cust = pd.read_csv(cust_path)
    df_tx = pd.read_csv(tx_path)

    validate_customer_data(df_cust)
    validate_transaction_data(df_tx)
    print("Schema and boundary validation passed successfully.")

    df_cust_clean, df_tx_clean = clean_datasets(df_cust, df_tx)
    df_modelling = engineer_features(df_tx_clean, df_cust_clean)

    # Save processed datasets for convenience
    os.makedirs("data/processed", exist_ok=True)
    df_cust_clean.to_csv("data/processed/clean_customers.csv", index=False)
    df_tx_clean.to_csv("data/processed/clean_transactions.csv", index=False)
    df_modelling.to_csv("data/processed/modelling_features.csv", index=False)
    print("Saved processed datasets to data/processed/")

    # Educational Disclaimer: Risk_Review_Flag is synthetic
    y = (df_modelling['Risk_Review_Flag'] == 'Yes').astype(int)
    drop_cols = [
        'Risk_Review_Flag', 'Transaction_ID', 'Customer_ID',
        'Customer_Name', 'Transaction_DateTime', 'Account_Status'
    ]
    X = df_modelling.drop(columns=drop_cols)

    preprocessor, _, _ = get_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\nTraining Baseline Logistic Regression Model...")
    baseline_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    baseline_pipe.fit(X_train, y_train)

    y_pred_base = baseline_pipe.predict(X_test)
    y_prob_base = baseline_pipe.predict_proba(X_test)[:, 1]

    print("\n--- BASELINE LOGISTIC REGRESSION EVALUATION ---")
    print(classification_report(y_test, y_pred_base))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_base):.4f}")

    print("\nTraining Benchmark Random Forest Classifier...")
    rf_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)

    y_pred_rf = rf_pipe.predict(X_test)
    y_prob_rf = rf_pipe.predict_proba(X_test)[:, 1]

    print("\n--- RANDOM FOREST EVALUATION ---")
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")


if __name__ == "__main__":
    main()
