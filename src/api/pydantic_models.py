from pydantic import BaseModel


class PredictionRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float


class PredictionResponse(BaseModel):
    is_high_risk: int
    probability: float
