"""Data Validation Module for FinTrust Financial Data."""
import re
from typing import Tuple
import numpy as np
import pandas as pd

REQUIRED_CUSTOMER_COLS = {
    'Customer_ID', 'Customer_Name', 'Age', 'Gender', 'City',
    'Customer_Segment', 'Account_Type', 'Tenure_Months',
    'Digital_Engagement_Score', 'Monthly_Income_Band',
    'Preferred_Channel', 'Account_Status'
}

REQUIRED_TX_COLS = {
    'Transaction_ID', 'Customer_ID', 'Transaction_DateTime',
    'Transaction_Type', 'Amount_NGN', 'Channel', 'Device_Type',
    'Location', 'International_Transaction', 'Transaction_Status',
    'Risk_Review_Flag'
}

VALID_CHANNELS = {'Mobile App', 'Web', 'ATM', 'POS', 'USSD'}
VALID_STATUSES = {'Successful', 'Failed', 'Reversed', 'Pending'}


class DataValidationError(Exception):
    """Raised when incoming dataset violates required constraints."""
    pass


def validate_customer_data(df: pd.DataFrame) -> bool:
    """Verifies schema and boundaries of Customer dataset."""
    if df.empty:
        raise DataValidationError("Customer DataFrame is empty.")
    missing_cols = REQUIRED_CUSTOMER_COLS - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing customer columns: {missing_cols}")
    try:
        age_numeric = pd.to_numeric(df['Age'])
    except (ValueError, TypeError):
        raise DataValidationError("Age must be a numeric column.")
    if (age_numeric <= 0).any() or (age_numeric > 120).any():
        raise DataValidationError("Invalid Age boundary detected.")
    if (df['Digital_Engagement_Score'] < 0).any() or (df['Digital_Engagement_Score'] > 100).any():
        raise DataValidationError("Engagement score outside valid [0, 100] range.")
    return True


def validate_transaction_data(df: pd.DataFrame) -> bool:
    """Verifies schema, channels, and positive values of Transaction dataset."""
    if df.empty:
        raise DataValidationError("Transaction DataFrame is empty.")
    missing_cols = REQUIRED_TX_COLS - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing transaction columns: {missing_cols}")
    if (df['Amount_NGN'] <= 0).any():
        raise DataValidationError("Non-positive Amount_NGN detected.")
    invalid_channels = set(df['Channel'].dropna().unique()) - VALID_CHANNELS
    if invalid_channels:
        raise DataValidationError(f"Unexpected Channels detected: {invalid_channels}")
    invalid_statuses = set(df['Transaction_Status'].dropna().unique()) - VALID_STATUSES
    if invalid_statuses:
        raise DataValidationError(f"Unexpected Statuses detected: {invalid_statuses}")
    return True


def clean_datasets(df_cust: pd.DataFrame, df_tx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Imputes missing fields and cleans column types."""
    df_tx_clean = df_tx.copy()
    # 96 missing values in Device_Type and Location handled by categorical imputation
    df_tx_clean['Device_Type'] = df_tx_clean['Device_Type'].fillna('Unknown')
    df_tx_clean['Location'] = df_tx_clean['Location'].fillna('Unknown')
    df_tx_clean['Transaction_DateTime'] = pd.to_datetime(df_tx_clean['Transaction_DateTime'])
    
    df_cust_clean = df_cust.copy()
    return df_cust_clean, df_tx_clean
