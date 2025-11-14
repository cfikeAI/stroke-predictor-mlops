FROM python:3.10-slim


RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# MLflow + Azure libs used by the server to push to Blob
RUN pip install --no-cache-dir \
    mlflow[azureml]==2.17.0 \
    lightgbm \
    gunicorn \
    azure-identity \
    azure-storage-blob

EXPOSE 5000

ENTRYPOINT ["mlflow"]
CMD ["mlflow", "server","--host", "0.0.0.0", "--port", "5050","--backend-store-uri", "sqlite:///mlflow.db","--default-artifact-root", "file:/mlflow-artifacts"]

