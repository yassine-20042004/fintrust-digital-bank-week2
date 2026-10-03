"""Automated Integration Test Suite for FastAPI REST Service."""
import pytest
from fastapi.testclient import TestClient

try:
    from src.api import app
except ImportError:
    from api import app

client = TestClient(app)


def test_api_root():
    """Test root endpoint returns HTTP 200 and system details."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "FinTrust Digital Bank Risk Inference API"
    assert data["status"] == "online"


def test_api_health():
    """Test health check endpoint returns HTTP 200 and status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_api_model_info():
    """Test model metadata endpoint returns features and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["features_engineered"] == 14
    assert "candidate_model" in data


def test_api_predict_valid():
    """Test POST /predict with valid customer & transaction payload."""
    payload = {
        "customer": {
            "Customer_ID": "FT-C10001",
            "Customer_Name": "Alice Johnson",
            "Age": 34,
            "Gender": "Female",
            "City": "Lagos",
            "Customer_Segment": "Everyday",
            "Account_Type": "Savings",
            "Tenure_Months": 18,
            "Digital_Engagement_Score": 82.0,
            "Monthly_Income_Band": "250k-499k",
            "Preferred_Channel": "Mobile App",
            "Account_Status": "Active"
        },
        "transaction": {
            "Transaction_ID": "FT-T50001",
            "Customer_ID": "FT-C10001",
            "Transaction_DateTime": "2026-03-15 14:30:00",
            "Transaction_Type": "Transfer",
            "Amount_NGN": 150000.0,
            "Channel": "Mobile App",
            "Device_Type": "iOS",
            "Location": "Lagos",
            "International_Transaction": "No",
            "Transaction_Status": "Successful",
            "Risk_Review_Flag": "No"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "FT-T50001"
    assert data["customer_id"] == "FT-C10001"
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["predicted_risk_flag"] in ["Yes", "No"]
    assert data["risk_tier"] in ["Low", "Medium", "High", "Critical"]
    assert isinstance(data["risk_factors"], list)


def test_api_predict_invalid_amount():
    """Test POST /predict rejects non-positive transaction amounts with HTTP 422."""
    payload = {
        "customer": {
            "Customer_ID": "FT-C10001",
            "Age": 34, "Gender": "Female", "City": "Lagos",
            "Customer_Segment": "Everyday", "Account_Type": "Savings",
            "Tenure_Months": 18, "Digital_Engagement_Score": 82.0,
            "Monthly_Income_Band": "250k-499k", "Preferred_Channel": "Mobile App"
        },
        "transaction": {
            "Transaction_ID": "FT-T50001", "Customer_ID": "FT-C10001",
            "Transaction_DateTime": "2026-03-15 14:30:00", "Transaction_Type": "Transfer",
            "Amount_NGN": -50.0,  # Invalid amount <= 0
            "Channel": "Mobile App", "International_Transaction": "No",
            "Transaction_Status": "Successful"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
