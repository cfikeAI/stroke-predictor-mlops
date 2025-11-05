import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Register your model artifact again
result = mlflow.register_model(
    "wasbs://mlflow@telemetryguardmlflow.blob.core.windows.net/<run_id>/artifacts/model",
    "TelemetryGuard_Stroke_Model"
)

client = mlflow.tracking.MlflowClient("http://127.0.0.1:5000")
client.set_registered_model_alias("TelemetryGuard_Stroke_Model", "production", result.version)
