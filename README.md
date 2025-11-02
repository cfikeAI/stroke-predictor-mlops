"# TelemetryGuard - Self-Healing ML Service" 




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

