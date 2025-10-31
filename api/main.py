import pandas as pd
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from api.model_loader import model_service
from api.schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="TelemetryGuard Stroke Prediction API",
              description="API for Stroke Prediction using LightGBM model tracked with MLFlow", version="1.0.0")

# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    "tg_requests_total",
    "Total number of prediction requests"
)

@app.get("/health")

def health():
    #readiness check for K8s
    return {
        "status": "ok",
        "model_version": model_service.get_model_version()
    }

@app.post("/predict")

def predict(payload: PredictionResponse):
    start = time.time()


