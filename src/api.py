"""FastAPI REST Service Endpoint for FinTrust Risk Inference Engine."""
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

try:
    from src.pipeline import FinTrustMLPipeline
except ImportError:
    from pipeline import FinTrustMLPipeline

app = FastAPI(
    title="FinTrust Digital Bank Risk Inference API",
    description="Week 3 Real-time Risk Assessment & ML Prediction Endpoint",
    version="1.0.0"
)

# Initialize pipeline instance
ml_pipeline = FinTrustMLPipeline(model_path="models/candidate_model.pkl")


# Pydantic Schemas for API Requests & Responses
class CustomerDataPayload(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    Customer_ID: str = Field(..., example="FT-C10001")
    Customer_Name: Optional[str] = Field("John Doe", example="John Doe")
    Age: int = Field(..., ge=18, le=100, example=35)
    Gender: str = Field(..., example="Male")
    City: str = Field(..., example="Lagos")
    Customer_Segment: str = Field(..., example="Everyday")
    Account_Type: str = Field(..., example="Savings")
    Tenure_Months: int = Field(..., ge=0, example=24)
    Digital_Engagement_Score: float = Field(..., ge=0.0, le=100.0, example=78.5)
    Monthly_Income_Band: str = Field(..., example="250k-499k")
    Preferred_Channel: str = Field(..., example="Mobile App")
    Account_Status: Optional[str] = Field("Active", example="Active")


class TransactionDataPayload(BaseModel):
    model_config = ConfigDict(extra='ignore')

    Transaction_ID: str = Field(..., example="FT-T50001")
    Customer_ID: str = Field(..., example="FT-C10001")
    Transaction_DateTime: str = Field(..., example="2026-03-15 14:30:00")
    Transaction_Type: str = Field(..., example="Transfer")
    Amount_NGN: float = Field(..., gt=0.0, example=125000.0)
    Channel: str = Field(..., example="Mobile App")
    Device_Type: Optional[str] = Field("Android", example="Android")
    Location: Optional[str] = Field("Lagos", example="Lagos")
    International_Transaction: str = Field(..., example="No")
    Transaction_Status: str = Field(..., example="Successful")
    Risk_Review_Flag: Optional[str] = Field("No", example="No")


class RiskPredictionRequest(BaseModel):
    customer: CustomerDataPayload
    transaction: TransactionDataPayload


class RiskPredictionResponse(BaseModel):
    transaction_id: str
    customer_id: str
    amount_ngn: float
    channel: str
    risk_probability: float
    predicted_risk_flag: str
    risk_tier: str
    risk_factors: List[str]


@app.get("/")
def read_root():
    return {
        "service": "FinTrust Digital Bank Risk Inference API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    model_exists = os.path.exists("models/candidate_model.pkl")
    return {
        "status": "healthy" if model_exists else "degraded",
        "model_loaded": model_exists,
        "engine_version": "Week 3 Candidate"
    }


@app.get("/model-info")
def model_info():
    return {
        "candidate_model": "Logistic Regression (Class-Weighted)",
        "features_engineered": 14,
        "primary_metric": "ROC-AUC (0.6730) & F1-Score (0.3903)",
        "target_variable": "Risk_Review_Flag (Synthetic Educational)",
        "pipeline_version": "1.0.0-Week3"
    }


@app.post("/predict", response_model=RiskPredictionResponse, status_code=status.HTTP_200_OK)
def predict_risk(payload: RiskPredictionRequest):
    try:
        cust_dict = payload.customer.model_dump()
        tx_dict = payload.transaction.model_dump()

        result = ml_pipeline.predict_single(tx_dict, cust_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}"
        )
