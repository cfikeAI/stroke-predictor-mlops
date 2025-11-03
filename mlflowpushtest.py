import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Register and alias your model again to ensure it’s in the remote store
result = mlflow.register_model(
    "runs:/bd2c3b4ef61f4759938c0706c6e6e538/model",
    "TelemetryGuard_Stroke_Model"
)

client = mlflow.tracking.MlflowClient("http://127.0.0.1:5000")
client.set_registered_model_alias("TelemetryGuard_Stroke_Model", "production", result.version)
print("Model re-registered to remote MLflow tracking server (Blob backend).")
