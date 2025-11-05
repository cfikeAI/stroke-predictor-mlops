
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["0"])  # or your experiment ID
print(runs[0].info.run_id)
