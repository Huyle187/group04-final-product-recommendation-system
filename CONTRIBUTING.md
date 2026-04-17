# Contributing Guide
## Product Recommendation System — Group 04

---

## Team Members & Responsibilities

### A. Problem Definition & Requirements — All Members
- Define the problem scope and business objectives
- Identify dataset requirements and evaluation criteria
- Agree on success metrics (Precision@k, Hit Rate@k, fairness thresholds)

### B. System Design & Architecture — All Members
- High-level architecture decisions (see [ARCHITECTURE.md](ARCHITECTURE.md))
- Component responsibilities and API contracts
- Infrastructure and deployment topology

### C. Implementation

| Area | Owner | Key Files |
|------|-------|-----------|
| **ML Pipeline** | Minh | `scripts/train_model.py`, `scripts/evaluate_model.py`, `app/model.py`, `app/explainability.py`, `app/fairness.py` |
| **Deployment** | Huy | `Dockerfile`, `docker-compose.yml`, `app/main.py`, `app/config.py`, `app/schemas.py`, `app/middleware.py` |
| **Monitoring** | Muoi | `app/metrics.py`, `prometheus/`, `grafana/` |

#### ML Pipeline (Minh)
- Data ingestion and preprocessing (Retail Rocket dataset)
- EASE collaborative filtering model
- TF-IDF content-based model and hybrid blend
- Per-user temporal train/test split
- SHAP and LIME explainability engine
- Fairness analysis and bias mitigation (MMR, popularity penalty)
- MLflow experiment tracking

#### Deployment (Huy)
- FastAPI REST API implementation
- Docker image and Docker Compose orchestration
- API request/response schemas (Pydantic)
- Logging middleware and error handling
- Model singleton and hot-reload endpoint

#### Monitoring (Muoi)
- Prometheus metrics (counters, histograms, gauges)
- Grafana dashboards (API dashboard, ML dashboard)
- Alert rules (`api_alerts.yml`, `ml_alerts.yml`)

### D. Testing & CI/CD — All Members
- Unit and integration tests (`tests/`)
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`)
- Load testing (`scripts/load_test.py`)

### E. Responsible AI — All Members
- Fairness metrics: popularity bias, category diversity, long-tail exposure, user-segment equity
- Explainability: SHAP latent factor attribution, LIME local approximation
- Ethics discussion and bias mitigation strategies

### F. Documentation — All Members
- `README.md` — setup and usage guide
- `ARCHITECTURE.md` — system design and data flow diagrams
- `ML_PIPELINE.md` — ML pipeline stages and model details
- API documentation via Swagger (`/docs`)
