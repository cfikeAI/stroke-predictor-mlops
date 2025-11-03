**TelemetryGuard - Self-Healing ML Service**

TelemetryGuard is a production-grade MLOps pipeline built for stroke prediction using tabular healthcare data.
It demonstrates data versioning (DVC), experiment tracking (MLflow), model deployment (FastAPI + Docker + Azure AKS), and observability (Prometheus + Grafana) — a complete self-healing ML service lifecycle.

This project reflects the entire MLOps stack in one cohesive implementation:

From raw data to monitored, cloud-deployed inference:


                     ┌──────────────────────────────┐
                     │        Data Versioning        │
                     │          (DVC + Git)          │
                     └──────────────┬────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │      Model Training &        │
                     │   Experiment Tracking (MLflow)│
                     └──────────────┬────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   Model Serving (FastAPI)    │
                     │    + Docker Containerization │
                     └──────────────┬────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   Infrastructure as Code     │
                     │ (Terraform + Azure Services) │
                     └──────────────┬────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  Monitoring & Drift Detection │
                     │ (Prometheus + Grafana future) │
                     └──────────────────────────────┘

Features:
Data versioning with DVC tracking

Model experiement tracking with MLFlow

LightGBM model for tabular data

FastAPI serving 

IaC using Terraform - automated provisioning with Azure ML infrastructure

Monitoring and observability

Scalable deployment orchestrated with Azure Kubernetes Service (AKS)
