import mlflow
import mlflow.lightgbm

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_name = "TelemetryGuard_Stroke_Model"

# Register your trained model (path must point to the saved model directory or run ID)
result = mlflow.register_model("runs:/<your_latest_run_id>/model", model_name)

# Optional: set alias "production"
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(model_name, "production", result.version)
