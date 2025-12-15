from __future__ import annotations

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load artifacts
model = joblib.load("models/logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="Credit Risk Prediction API")


class PredictionRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float


class PredictionResponse(BaseModel):
    is_high_risk: int
    probability: float


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    X = np.array([[data.recency, data.frequency, data.monetary]])
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0, 1]
    pred = int(prob >= 0.5)

    return {
        "is_high_risk": pred,
        "probability": float(prob),
    }
