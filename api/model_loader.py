import os
import mlflow
import pandas as pd
from typing import Tuple

EXPERIMENT_NAME = "Stroke_Prediction_LightGBM_TelemetryGuard"
MLRUNS_PATH = "mlruns"
#bridge between MLFlow tracked model and FastAPI inference service

class ModelService:
    def __init__(self):
        #resolve latest model from MLFlow
        self.client = mlflow.tracking.MlflowClient(tracking_uri=MLRUNS_PATH)
        exp = self.client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found in MLFlow at '{MLRUNS_PATH}'") #error if not found
        
        runs = self.client.search_runs(
            exp.experiment_id,
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if len(runs) == 0:
            raise RuntimeError(f"No runs found in experiment '{EXPERIMENT_NAME}. Train a model first'")

        self.run_id = runs[0].info.run_id       
        self.model_uri = f"runs:/{self.run_id}/model" 

        #load model (lighGBM booster) via mlflow
        self.model = mlflow.lightgbm.load_model(self.model_uri)

        #capture feature order from training data
        #canonoical column order = the X_train.csv

        xtrain_path = os.path.join("data", "processed", "X_train.csv")
        if not os.path.exists(xtrain_path):
            raise RuntimeError("Training data not found at 'data/processed/X_train.csv'. Ensure data preprocessing has been completed.")
        self.feature_order = pd.read_csv(xtrain_path, nrows=1).columns.tolist()

    def predict_proba_and_label(self, rows: pd.DataFrame) -> Tuple[list, list]:
        #ensure column order matches training
        rows = rows[self.feature_order]
        #get probability scores from lightGBM model booster
        probs = self.model.predict(rows)
        #binary label using 0.5 threshold
        labels = (probs >= 0.5).astype(int)

        return probs.tolist(), labels.tolist()
    
    def get_model_version(self) -> str:
        #later this will be MLFlow model registry stage/version
        return self.run_id
#singleton-style instance for reuse in FastAPI
model_service = ModelService()

