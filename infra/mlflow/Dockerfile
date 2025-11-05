# Use the official MLflow image as a base
FROM ghcr.io/mlflow/mlflow:v2.10.0

# Add Azure SDK dependencies
RUN pip install --no-cache-dir azure-storage-blob==12.19.1 azure-identity==1.16.0

# Optional: for SQLite and Flask if not already included
RUN pip install --no-cache-dir gunicorn

# Expose default MLflow port
EXPOSE 5000

# Entrypoint (same as base, but explicit)
ENTRYPOINT ["mlflow"]
CMD ["server", "--host", "0.0.0.0", "--port", "5000"]
