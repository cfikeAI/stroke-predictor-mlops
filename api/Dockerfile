# Use lightweight Python base image
FROM python:3.10-slim

# Prevent pyc creation and buffer issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ /app/api/
COPY src/ /app/src/

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
