"""Automated Technical & Reproducibility Test Suite for FinTrust ML Pipeline."""
import pytest
import pandas as pd
import numpy as np
import os

try:
    from src.validation import validate_customer_data, validate_transaction_data, DataValidationError
    from src.pipeline import FinTrustMLPipeline
    from src.preprocessing import engineer_features, get_preprocessor
except ImportError:
    from validation import validate_customer_data, validate_transaction_data, DataValidationError
    from pipeline import FinTrustMLPipeline
    from preprocessing import engineer_features, get_preprocessor


@pytest.fixture
def sample_valid_cust():
    return pd.DataFrame([{
        'Customer_ID': 'FT-C99999', 'Customer_Name': 'Test User', 'Age': 30,
        'Gender': 'Male', 'City': 'Lagos', 'Customer_Segment': 'Everyday',
        'Account_Type': 'Savings', 'Tenure_Months': 12,
        'Digital_Engagement_Score': 75.0, 'Monthly_Income_Band': '250k-499k',
        'Preferred_Channel': 'Mobile App', 'Account_Status': 'Active'
    }])


@pytest.fixture
def sample_valid_tx():
    return pd.DataFrame([{
        'Transaction_ID': 'FT-T99999', 'Customer_ID': 'FT-C99999',
        'Transaction_DateTime': '2026-01-01 10:00:00', 'Transaction_Type': 'Transfer',
        'Amount_NGN': 50000.0, 'Channel': 'Mobile App',
        'Device_Type': 'Android', 'Location': 'Lagos',
        'International_Transaction': 'No', 'Transaction_Status': 'Successful',
        'Risk_Review_Flag': 'No'
    }])


def test_1_valid_input(sample_valid_cust, sample_valid_tx):
    """Test 1: Valid customer and transaction input pass validation without errors."""
    assert validate_customer_data(sample_valid_cust) is True
    assert validate_transaction_data(sample_valid_tx) is True


def test_2_empty_input():
    """Test 2: Empty DataFrames trigger DataValidationError immediately."""
    df_empty = pd.DataFrame()
    with pytest.raises(DataValidationError, match="Customer DataFrame is empty"):
        validate_customer_data(df_empty)
    with pytest.raises(DataValidationError, match="Transaction DataFrame is empty"):
        validate_transaction_data(df_empty)


def test_3_unexpected_categories(sample_valid_tx):
    """Test 3: Unexpected categorical values (e.g. unknown channel) are flagged and rejected."""
    sample_valid_tx['Channel'] = 'UNSUPPORTED_CRYPTO'
    with pytest.raises(DataValidationError, match="Unexpected Channels detected"):
        validate_transaction_data(sample_valid_tx)


def test_4_invalid_data_types(sample_valid_cust):
    """Test 4: Non-numeric strings in Age field are detected during validation."""
    sample_valid_cust['Age'] = 'INVALID_AGE_STRING'
    with pytest.raises(DataValidationError, match="Age must be a numeric column"):
        validate_customer_data(sample_valid_cust)


def test_5_boundary_checks(sample_valid_cust, sample_valid_tx):
    """Test 5: Out-of-bounds Age and negative Amount_NGN are rejected."""
    sample_valid_cust['Age'] = -10
    with pytest.raises(DataValidationError, match="Invalid Age boundary detected"):
        validate_customer_data(sample_valid_cust)

    sample_valid_tx['Amount_NGN'] = -500.0
    with pytest.raises(DataValidationError, match="Non-positive Amount_NGN detected"):
        validate_transaction_data(sample_valid_tx)


def test_6_feature_engineering(sample_valid_cust, sample_valid_tx):
    """Test 6: Feature engineering generates all 14 expected features."""
    df_mod = engineer_features(sample_valid_tx, sample_valid_cust)
    expected_feats = [
        'Tx_Hour', 'Tx_DayOfWeek', 'Is_Weekend', 'Log_Amount',
        'Is_International', 'Cust_Lifetime_Tx', 'Amount_to_Mean_Ratio',
        'Is_High_Value', 'Is_Night_Tx', 'International_Weekend_Interact',
        'Amount_to_Income_Ratio', 'Tenure_to_Age_Ratio',
        'Cust_Failed_Tx_Ratio', 'Channel_Risk_Rate'
    ]
    for feat in expected_feats:
        assert feat in df_mod.columns


def test_7_model_loading():
    """Test 7: Candidate model file exists and loads into memory successfully."""
    pipeline = FinTrustMLPipeline(model_path="models/candidate_model.pkl")
    model = pipeline.load_pipeline()
    assert model is not None


def test_8_prediction_generation(sample_valid_cust, sample_valid_tx):
    """Test 8: ML Pipeline processes single record and returns valid prediction probability."""
    pipeline = FinTrustMLPipeline(model_path="models/candidate_model.pkl")
    res = pipeline.predict_single(sample_valid_tx.iloc[0].to_dict(), sample_valid_cust.iloc[0].to_dict())

    assert "risk_probability" in res
    assert 0.0 <= res["risk_probability"] <= 1.0
    assert res["predicted_risk_flag"] in ["Yes", "No"]
    assert res["risk_tier"] in ["Low", "Medium", "High", "Critical"]


def test_9_output_format(sample_valid_cust, sample_valid_tx):
    """Test 9: Batch prediction returns correct DataFrame columns and formats."""
    pipeline = FinTrustMLPipeline(model_path="models/candidate_model.pkl")
    df_res = pipeline.predict_dataframe(sample_valid_cust, sample_valid_tx)

    expected_cols = ['Transaction_ID', 'Customer_ID', 'Amount_NGN', 'Channel', 'Risk_Probability', 'Predicted_Risk_Flag', 'Risk_Tier']
    for col in expected_cols:
        assert col in df_res.columns


def test_10_reproducibility(sample_valid_cust, sample_valid_tx):
    """Test 10: Model predictions are deterministic for identical input data."""
    pipeline = FinTrustMLPipeline(model_path="models/candidate_model.pkl")
    res1 = pipeline.predict_single(sample_valid_tx.iloc[0].to_dict(), sample_valid_cust.iloc[0].to_dict())
    res2 = pipeline.predict_single(sample_valid_tx.iloc[0].to_dict(), sample_valid_cust.iloc[0].to_dict())

    assert res1["risk_probability"] == res2["risk_probability"]
    assert res1["predicted_risk_flag"] == res2["predicted_risk_flag"]
