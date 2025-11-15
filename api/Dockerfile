# Use lightweight Python base image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install dependencies required by LightGBM models
# Install dependencies required by LightGBM models
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

# Set workdir for the app
WORKDIR /app

# Copy root-level requirements.txt (build context = repo root)
COPY api/requirements.txt /app/api/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy source code
COPY api/ /app/api/
COPY src/ /app/src/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
