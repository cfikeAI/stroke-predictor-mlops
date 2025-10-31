import pandas as pd
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from api.model_loader import model_service
from api.schemas import PredictionRequest, PredictionResponse, PredictionResult


app = FastAPI(title="TelemetryGuard Stroke Prediction API",
              description="API for Stroke Prediction using LightGBM model tracked with MLFlow", version="1.0.0")

# Prometheus metrics
PREDICTION_REQUESTS = Counter( #counts total prediction requests
    "tg_requests_total",
    "Total number of prediction requests"
)

PREDICTION_ERRORS = Counter( #counts total prediction errors
    "tg_request_errors_total",
    "Total number of prediction requests that raised an error"
)

PREDICTION_LATENCY = Histogram( #counts time taken for requests
    "tg_request_latency_seconds",
    "Latency for /predict in seconds"
)

MODEL_VERSION_GAUGE = Gauge( #records model version
    "tg_model_version_info",
    "Model version currently loaded (run_id hash truncated),"
    ["run_id"]
)

#init model version gauge once at startup
MODEL_VERSION_GAUGE.labels(run_id=model_service.get_model_version()[:8]).set(1)



@app.get("/health")

def health():
    #readiness check for K8s
    return {
        "status": "ok",
        "model_version": model_service.get_model_version()
    }

@app.post("/predict", response_model=PredictionResponse)

def predict(payload: PredictionRequest):
    #run prediction on one or many rows of data
    start = time.time()
    PREDICTION_REQUESTS.inc()
    try:
        #Normalize payload to list of dicts
        if isinstance(payload.inputs, dict):
            rows = [payload.inputs]
        else:
            rows = [row.dict() for row in payload.inputs]
        #Convert to DataFrame
        df = pd.DataFrame(rows)
        #Get predictions from model service
        probs, labels = model_service.predict_proba_and_label(df)
        results = [
            PredictionResult(prob=float(p), label=int(1))
            for p, 1 in zip(probs, labels)
        ]

        resp = PredictionResponse(
            results=results,
            model_version=model_service.get_model_version()
        )
        return resp
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed = time.time() - start
        PREDICTION_LATENCY.observe(elapsed)

