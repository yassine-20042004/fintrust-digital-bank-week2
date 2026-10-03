"""End-to-End Modular ML Pipeline for FinTrust Week 3."""
import os
import joblib
import pandas as pd
import numpy as np

try:
    from src.validation import validate_customer_data, validate_transaction_data, clean_datasets
    from src.preprocessing import engineer_features, get_preprocessor
    from src.models import train_and_evaluate_all_models, save_candidate_model, load_candidate_model
except ImportError:
    from validation import validate_customer_data, validate_transaction_data, clean_datasets
    from preprocessing import engineer_features, get_preprocessor
    from models import train_and_evaluate_all_models, save_candidate_model, load_candidate_model


class FinTrustMLPipeline:
    """Modular End-to-End Pipeline: Input -> Validation -> Preprocessing -> Features -> Model -> Prediction -> Output."""

    def __init__(self, model_path="models/candidate_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.feature_names = None

    def fit_from_raw_data(self, df_cust: pd.DataFrame, df_tx: pd.DataFrame):
        """Validates, cleans, engineers features, trains all models, and saves candidate pipeline."""
        print("[Pipeline Step 1: Input Validation]")
        validate_customer_data(df_cust)
        validate_transaction_data(df_tx)

        print("[Pipeline Step 2: Data Cleaning & Preprocessing]")
        df_cust_clean, df_tx_clean = clean_datasets(df_cust, df_tx)

        print("[Pipeline Step 3: Feature Engineering (14 features)]")
        df_modelling = engineer_features(df_tx_clean, df_cust_clean)

        y = (df_modelling['Risk_Review_Flag'] == 'Yes').astype(int)
        drop_cols = [
            'Risk_Review_Flag', 'Transaction_ID', 'Customer_ID',
            'Customer_Name', 'Transaction_DateTime', 'Account_Status'
        ]
        X = df_modelling.drop(columns=drop_cols, errors='ignore')

        preprocessor, num_feats, cat_feats = get_preprocessor()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        print("[Pipeline Step 4: Model Development & Evaluation]")
        results_df, trained_pipes, best_name, candidate_pipe = train_and_evaluate_all_models(
            X_train, y_train, X_test, y_test, preprocessor
        )

        self.pipeline = candidate_pipe
        save_candidate_model(candidate_pipe, self.model_path)
        print(f"[Pipeline Step 5: Candidate Selected - {best_name}]")

        return results_df, self.pipeline

    def load_pipeline(self):
        """Loads serialized candidate model."""
        self.pipeline = load_candidate_model(self.model_path)
        return self.pipeline

    def predict_dataframe(self, df_cust: pd.DataFrame, df_tx: pd.DataFrame) -> pd.DataFrame:
        """Processes raw input dataframes and returns predictions dataframe."""
        if self.pipeline is None:
            self.load_pipeline()

        validate_customer_data(df_cust)
        validate_transaction_data(df_tx)
        df_cust_clean, df_tx_clean = clean_datasets(df_cust, df_tx)
        df_modelling = engineer_features(df_tx_clean, df_cust_clean)

        drop_cols = [
            'Risk_Review_Flag', 'Transaction_ID', 'Customer_ID',
            'Customer_Name', 'Transaction_DateTime', 'Account_Status'
        ]
        X = df_modelling.drop(columns=drop_cols, errors='ignore')

        probs = self.pipeline.predict_proba(X)[:, 1]
        preds = self.pipeline.predict(X)

        df_results = df_modelling[['Transaction_ID', 'Customer_ID', 'Amount_NGN', 'Channel']].copy()
        df_results['Risk_Probability'] = np.round(probs, 4)
        df_results['Predicted_Risk_Flag'] = np.where(preds == 1, 'Yes', 'No')
        df_results['Risk_Tier'] = pd.cut(
            df_results['Risk_Probability'],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        return df_results

    def predict_single(self, transaction_dict: dict, customer_dict: dict) -> dict:
        """Processes a single transaction request and returns risk assessment payload."""
        df_tx = pd.DataFrame([transaction_dict])
        df_cust = pd.DataFrame([customer_dict])

        res_df = self.predict_dataframe(df_cust, df_tx)
        row = res_df.iloc[0]

        reasons = []
        if transaction_dict.get('International_Transaction') == 'Yes':
            reasons.append("Cross-border international activity detected")
        if float(transaction_dict.get('Amount_NGN', 0)) > 150000.0:
            reasons.append("High monetary value exceeds 85th percentile threshold")
        if transaction_dict.get('Transaction_Status') == 'Failed':
            reasons.append("Historical failed transaction record detected")
        if not reasons:
            reasons.append("Standard transaction pattern within baseline parameters")

        return {
            "transaction_id": row['Transaction_ID'],
            "customer_id": row['Customer_ID'],
            "amount_ngn": float(row['Amount_NGN']),
            "channel": row['Channel'],
            "risk_probability": float(row['Risk_Probability']),
            "predicted_risk_flag": row['Predicted_Risk_Flag'],
            "risk_tier": str(row['Risk_Tier']),
            "risk_factors": reasons
        }
