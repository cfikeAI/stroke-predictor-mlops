import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import mlflow
import pandas as pd
from typing import Tuple
from mlflow.tracking import MlflowClient
import json
import tempfile

credential = DefaultAzureCredential()
account_url = "https://telemetryguardmlflow.blob.core.windows.net"
blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
try:
    container_client = blob_service_client.get_container_client("mlflow-artifacts")
    container_client.get_container_properties()
    print("✅ Connected to Azure Blob Storage successfully via Managed Identity.")
except Exception as e:
    print("❌ Failed to connect to Blob Storage:", e)


MODEL_NAME = "TelemetryGuard_Stroke_Model"
MODEL_ALIAS = "production"
XTRAIN_PATH = os.path.join("data", "processed", "X_train.csv")

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://telemetryguard-mlflow-service.default.svc.cluster.local:5000")
mlflow.set_tracking_uri(mlflow_uri)


class ModelService:
    """Service wrapper for loading and serving the latest model by alias from MLflow Registry."""

    def __init__(self):
        
        self.client = MlflowClient(tracking_uri=mlflow_uri)

        print(f"Loading model '{MODEL_NAME}' from MLflow Registry (alias: '{MODEL_ALIAS}')...")

        # Load model by alias
        
        self.model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        # Retrieve alias metadata (new API)
        version_info = self.client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        self.model_version = version_info.version
        self.run_id = version_info.run_id

        print(f"Loaded model '{MODEL_NAME}' v{self.model_version} (alias='{MODEL_ALIAS}', run_id={self.run_id})")

        ## Load training feature order
        #if not os.path.exists(XTRAIN_PATH):
        #    raise RuntimeError(
        #        f"Training data not found at '{XTRAIN_PATH}'. Ensure preprocessing completed."
        #    )
        #self.feature_order = pd.read_csv(XTRAIN_PATH, nrows=1).columns.tolist()

        # Download feature_order.json from artifacts of this run
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = self.client.download_artifacts(self.run_id, "model_meta", tmp)
            feature_path = os.path.join(local_dir, "feature_order.json")
            if not os.path.exists(feature_path):
                raise RuntimeError("feature_order.json not found in artifacts. Re-train and log it.")
            with open(feature_path, "r") as f:
                self.feature_order = json.load(f)

    def predict_proba_and_label(self, rows: pd.DataFrame) -> Tuple[list, list]:
        """Predict probabilities and binary labels."""
        rows = rows[self.feature_order]
        probs = self.model.predict(rows)
        labels = (probs >= 0.5).astype(int)
        return probs.tolist(), labels.tolist()

    def get_model_version(self) -> str:
        """Expose version metadata for observability endpoints."""
        return f"{MODEL_NAME}_v{self.model_version}@{MODEL_ALIAS}"


# Singleton instance for FastAPI reuse
model_service = ModelService()
