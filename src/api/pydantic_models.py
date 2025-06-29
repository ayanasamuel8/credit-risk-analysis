"""Pydantic models for API input/output validation."""

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    # Define input fields
    feature1: float
    feature2: float

class PredictionResponse(BaseModel):
    risk_score: float
    approved: bool
