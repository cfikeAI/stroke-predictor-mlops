import os
import mlflow
import pandas as pd
from typing import Tuple
from mlflow.tracking import MlflowClient

# === Configuration ===
MODEL_NAME = "TelemetryGuard_Stroke_Model"  # Registered model name
MODEL_ALIAS = "production"                  # Alias name (case-sensitive; lowercase)
MLRUNS_PATH = "mlruns"
XTRAIN_PATH = os.path.join("data", "processed", "X_train.csv")


class ModelService:
    """Service wrapper for loading and serving the latest model by alias from MLflow Registry."""

    def __init__(self):
        self.client = MlflowClient(tracking_uri=MLRUNS_PATH)

        print(f"Loading model '{MODEL_NAME}' from MLflow Registry (alias: '{MODEL_ALIAS}')...")

        # Load model by alias
        
        self.model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        # Retrieve alias metadata (new API)
        version_info = self.client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        self.model_version = version_info.version
        self.run_id = version_info.run_id

        print(f"Loaded model '{MODEL_NAME}' v{self.model_version} (alias='{MODEL_ALIAS}', run_id={self.run_id})")

        # Load training feature order
        if not os.path.exists(XTRAIN_PATH):
            raise RuntimeError(
                f"Training data not found at '{XTRAIN_PATH}'. Ensure preprocessing completed."
            )
        self.feature_order = pd.read_csv(XTRAIN_PATH, nrows=1).columns.tolist()

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
