import mlflow
from mlflow.tracking import MlflowClient
import sys

# Tracking URI from your environment
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)

MODEL_NAME = "TelemetryGuard_Stroke_Model"
EXPERIMENT_NAME = "Stroke_Prediction_LightGBM_TelemetryGuard"

print(f"\n[INFO] Connecting to MLflow server at: {tracking_uri}")

try:
    client = MlflowClient()
    print("[INFO] Connected successfully.")
except Exception as e:
    print("[ERROR] Could not connect to MLflow:", e)
    sys.exit(1)

# Try to create the registered model (if not already exists)
try:
    client.create_registered_model(MODEL_NAME)
    print(f"[INFO] Created new registered model: {MODEL_NAME}")
except Exception as e:
    print(f"[WARN] Model may already exist: {e}")

# Find the latest run from your local experiment
try:
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"[ERROR] Experiment '{EXPERIMENT_NAME}' not found on server.")
        sys.exit(1)

    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        print("[ERROR] No runs found in experiment.")
        sys.exit(1)

    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/model"
    print(f"[INFO] Found run: {run.info.run_id}")
    print(f"[INFO] Registering model from URI: {model_uri}")

    version = client.create_model_version(MODEL_NAME, model_uri, run.info.run_id)
    print(f"[INFO] Registered new model version: {version.version}")

    client.set_registered_model_alias(MODEL_NAME, "production", version.version)
    print(f"[INFO] Set alias 'production' → version {version.version}")

except Exception as e:
    print("[ERROR] During model registration:", e)
    sys.exit(1)

print("\n✅ Model successfully registered to AKS MLflow server.")
