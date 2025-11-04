import pandas as pd
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from api.model_loader import model_service
from api import PredictionRequest, PredictionResponse, PredictionResult


# Maps human-readable values to numeric encodings from training
ENCODERS = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "ever_married": {"No": 0, "Yes": 1},
    "work_type": {
        "children": 0,
        "Govt_job": 1,
        "Never_worked": 2,
        "Private": 3,
        "Self-employed": 4
    },
    "Residence_type": {"Rural": 0, "Urban": 1},
    "smoking_status": {
        "never smoked": 0,
        "formerly smoked": 1,
        "smokes": 2,
        "Unknown": 3
    },
}

def encode_features(sample: dict) -> dict:
    """Convert human-readable string features to numeric encodings."""
    encoded = {}
    for key, value in sample.items():
        if key in ENCODERS:
            mapping = ENCODERS[key]
            # allow both string and numeric input
            if isinstance(value, str):
                encoded[key] = mapping.get(value, None)
            else:
                encoded[key] = value
        else:
            encoded[key] = value
    return encoded


app = FastAPI(title="TelemetryGuard Stroke Prediction API",
              description="API for Stroke Prediction using LightGBM model tracked with MLFlow", version="1.0.0")

# Prometheus metrics, tagged with *tg_* prefix for TelemetryGuard
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
    "Model version currently loaded (run_id hash truncated)",
    ["run_id"]
)

#init model version gauge once at startup
MODEL_VERSION_GAUGE.labels(run_id=model_service.get_model_version()[:8]).set(1)



@app.get("/health") #surfaces model version for readiness checks. For debugging rollout/rollback later for AKS

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
        encoded_rows = [encode_features(row) for row in rows]
        df = pd.DataFrame(encoded_rows)
        #Get predictions from model service
        probs, labels = model_service.predict_proba_and_label(df)
        results = [
            PredictionResult(prob=float(p), label=int(label))
            for p, label in zip(probs, labels)
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

@app.get("/metrics", response_class=PlainTextResponse)

def metrics():
    #endpoint for Prometheus to scrape
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)