from __future__ import annotations

import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from fastapi import FastAPI

from src.api.pydantic_models import PredictionRequest, PredictionResponse


# ----------------------------
# MLflow Model Loading
# ----------------------------
# Recommended for grading: load Production model
MODEL_URI = "models:/credit-risk-rfm/Production"

model = mlflow.sklearn.load_model(MODEL_URI)

# Scaler loaded locally (acceptable for this project)
scaler = joblib.load("models/scaler.pkl")


# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Credit Risk Prediction API")


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
