# tests/test_model.py
import mlflow
import pandas as pd
import sys
import pytest
# Skip entire file if running on Windows or under pytest
if sys.platform.startswith("win"):
    pytest.skip("Skipping LightGBM model load tests on Windows due to threading instability", allow_module_level=True)

from sklearn.metrics import roc_auc_score

MLRUNS_PATH = "mlruns"
EXPERIMENT_NAME = "Stroke_Prediction_LightGBM_TelemetryGuard" #same name as training script

@pytest.fixture(scope="module")
def model():
    import os
    #os.environ["OMP_NUM_THREADS"] = "1"  # Disable OpenMP threading
    #os.environ["LIGHTGBM_USE_MULTITHREADING"] = "0"

    client = mlflow.tracking.MlflowClient(tracking_uri=MLRUNS_PATH)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    return model

@pytest.fixture(scope="module")
def test_data():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_test, y_test

def test_model_predicts_valid_range(model, test_data):
    X_test, _ = test_data
    preds = model.predict(X_test)
    assert (preds >= 0).all() and (preds <= 1).all(), "Predictions outside [0,1]"
    assert len(preds) == len(X_test), "Output length mismatch"

def test_model_auc_above_random(model, test_data):
    X_test, y_test = test_data
    auc = roc_auc_score(y_test, model.predict(X_test))
    assert auc > 0.5, f"AUC below random chance (AUC={auc:.2f})"

def test_model_stability(model, test_data):
    X_test, _ = test_data
    preds_1 = model.predict(X_test)
    preds_2 = model.predict(X_test)
    diff = abs(preds_1 - preds_2).sum()
    assert diff < 1e-6, "Model predictions not deterministic"
