"""FastAPI server for credit risk model predictions."""

from fastapi import FastAPI
from .pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Credit Risk Model API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    # Dummy prediction logic
    return PredictionResponse(risk_score=0.5, approved=True)
