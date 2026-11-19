import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =====================================
# 1. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn using a trained machine learning model.",
    version="1.0"
)

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
threshold = saved_data["threshold"]
model_name = saved_data["model_name"]

# =====================================
# 3. CUSTOMER INPUT DETAILS
# =====================================

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

# =====================================
# 4. HOME ENDPOINT
# =====================================

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!",
        "model": model_name
    }

# =====================================
# 5. HEALTH CHECK
# =====================================

@app.get("/health")
def health():
    return {"status": "healthy"}

# =====================================
# 6. CUSTOMER CHURN PREDICTION
# =====================================

@app.post("/predict")
def predict_churn(customer: Customer):

    try:
        # Convert customer input into DataFrame
        data = pd.DataFrame([customer.model_dump()])

        # Feature 1: Tenure Group
        data["TenureGroup"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, float("inf")],
            labels=["New", "Medium", "Long", "Loyal"]
        )

        # Feature 2: Average Monthly Spend
        data["AverageMonthlySpend"] = (
            data["TotalCharges"] /
            data["tenure"].replace(0, np.nan)
        )

        # Feature 3: Total Services
        service_columns = [
            "PhoneService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        data["TotalServices"] = (
            data[service_columns] == "Yes"
        ).sum(axis=1)

        # Predict churn probability
        probability = model.predict_proba(data)[0][1]

        # Binary prediction
        prediction = (
            "Yes" if probability >= threshold else "No"
        )

        # Risk classification
        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "model_used": model_name,
            "churn_probability": round(float(probability), 4),
            "churn_prediction": prediction,
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
    