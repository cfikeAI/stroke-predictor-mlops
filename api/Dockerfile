# Use lightweight Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app/api

# Copy root-level requirements
COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all API source code
COPY . /app/api

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
