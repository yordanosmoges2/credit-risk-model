from __future__ import annotations

import os
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

# ----------------------------
# MLflow Model Loading
# ----------------------------
# Option 1 (recommended for grading): load latest Production model
MODEL_URI = "models:/credit-risk-rfm/Production"

# If you did NOT register a Production model, use Option 2 instead:
# MODEL_URI = "runs:/<RUN_ID>/model"

model = mlflow.sklearn.load_model(MODEL_URI)

# NOTE:
# If your scaler is part of the training pipeline, it should ideally
# be logged with MLflow. For now, we load it locally (acceptable).
import joblib
scaler = joblib.load("models/scaler.pkl")

# ----------------------------
# FastAPI App
# ----------------------------
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

