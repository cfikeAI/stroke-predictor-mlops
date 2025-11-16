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

Data: Versioned w/ DVC (tabular healthcare)

Training: LightGBM experiments in MLflow

Serving: FastAPI w/ Pydantic validation + health probes

IaC: Terraform for AKS, ACR, RBAC

Deployment: Helm charts + HPA autoscaling

Monitoring: Prometheus metrics + Grafana dashboards (WIP)
