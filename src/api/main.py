from fastapi import FastAPI
from src.api.pydantic_models import Transaction, PredictionResponse
import pandas as pd
from src.predict import predict

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(tx: Transaction):
    import mlflow
    mlflow.set_tracking_uri("file:///C:/Users/user/Documents/Datasience/credict-risk-analysis/notebooks/mlruns")
    data = pd.DataFrame([tx.dict()])
    prediction = predict(data)
    return PredictionResponse(
        is_high_risk=prediction["is_high_risk"]
    )
