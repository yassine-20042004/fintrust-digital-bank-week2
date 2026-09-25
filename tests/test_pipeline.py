"""Automated Unit Tests for FinTrust MLOps Pipeline."""
import pytest
import pandas as pd

try:
    from src.validation import validate_customer_data, validate_transaction_data, DataValidationError
except ImportError:
    from validation import validate_customer_data, validate_transaction_data, DataValidationError


def test_1_valid_input():
    """Test 1: Valid input passes schema check without raising exceptions."""
    df_cust = pd.DataFrame([{
        'Customer_ID': 'FT-C99999', 'Customer_Name': 'Test User', 'Age': 30,
        'Gender': 'Male', 'City': 'Lagos', 'Customer_Segment': 'Everyday',
        'Account_Type': 'Savings', 'Tenure_Months': 12,
        'Digital_Engagement_Score': 75.0, 'Monthly_Income_Band': '250k-499k',
        'Preferred_Channel': 'Mobile App', 'Account_Status': 'Active'
    }])
    assert validate_customer_data(df_cust) is True


def test_2_empty_dataset():
    """Test 2: Empty dataset triggers DataValidationError immediately."""
    df_empty = pd.DataFrame()
    with pytest.raises(DataValidationError, match="Customer DataFrame is empty"):
        validate_customer_data(df_empty)


def test_3_unexpected_category():
    """Test 3: Unexpected categorical values in Channel are flagged and rejected."""
    df_tx = pd.DataFrame([{
        'Transaction_ID': 'FT-T99999', 'Customer_ID': 'FT-C99999',
        'Transaction_DateTime': '1/1/2026 10:00', 'Transaction_Type': 'Transfer',
        'Amount_NGN': 5000.0, 'Channel': 'CRYPTO_DESK',  # Invalid channel
        'Device_Type': 'Android', 'Location': 'Lagos',
        'International_Transaction': 'No', 'Transaction_Status': 'Successful',
        'Risk_Review_Flag': 'No'
    }])
    with pytest.raises(DataValidationError, match="Unexpected Channels detected"):
        validate_transaction_data(df_tx)


def test_4_incorrect_data_boundary_age():
    """Test 4: Out-of-bounds numerical values in Age are caught."""
    df_cust = pd.DataFrame([{
        'Customer_ID': 'FT-C99999', 'Customer_Name': 'Test User', 'Age': -5,  # Negative Age
        'Gender': 'Male', 'City': 'Lagos', 'Customer_Segment': 'Everyday',
        'Account_Type': 'Savings', 'Tenure_Months': 12,
        'Digital_Engagement_Score': 75.0, 'Monthly_Income_Band': '250k-499k',
        'Preferred_Channel': 'Mobile App', 'Account_Status': 'Active'
    }])
    with pytest.raises(DataValidationError, match="Invalid Age boundary detected"):
        validate_customer_data(df_cust)


def test_5_negative_transaction_amount():
    """Test 5: Non-positive Amount_NGN values are flagged and halted."""
    df_tx = pd.DataFrame([{
        'Transaction_ID': 'FT-T99999', 'Customer_ID': 'FT-C99999',
        'Transaction_DateTime': '1/1/2026 10:00', 'Transaction_Type': 'Transfer',
        'Amount_NGN': -250.0,  # Negative amount
        'Channel': 'Mobile App', 'Device_Type': 'Android', 'Location': 'Lagos',
        'International_Transaction': 'No', 'Transaction_Status': 'Successful',
        'Risk_Review_Flag': 'No'
    }])
    with pytest.raises(DataValidationError, match="Non-positive Amount_NGN detected"):
        validate_transaction_data(df_tx)
