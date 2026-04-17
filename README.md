# Product Recommendation System

**DDM501 - AI in Production: From Models to Systems**

## Overview

A real-time product recommendation engine for e-commerce, built on the [Retail Rocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). Implements collaborative filtering (EASE), content-based filtering (TF-IDF cosine similarity), and a hybrid blend with full explainability (SHAP/LIME) and fairness analysis.

**Topic:** Topic 1 - Product Recommendation System
**Dataset:** Retail Rocket E-Commerce Dataset (Kaggle)

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design and [ML_PIPELINE.md](ML_PIPELINE.md) for the full ML pipeline.

---

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Retail Rocket dataset CSV files in `./data/`

### Dataset Setup

Download from Kaggle and place in `./data/`:
```
data/
├── events.csv
├── item_properties_part1.csv
└── item_properties_part2.csv
```

### Installation & Setup

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Train the model** (requires `data/events.csv`)
```bash
# Train hybrid model with evaluation and MLflow tracking
python scripts/train_model.py --dataset ./data --model hybrid --evaluate --mlflow

# Train collaborative only (faster)
python scripts/train_model.py --dataset ./data --model collaborative --evaluate
```

3. **Start all services**
```bash
docker-compose up -d
```

4. **Verify services are running**
```bash
curl http://localhost:8000/health
```

### Service URLs
- **API:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)
- **MLflow UI:** `mlflow ui --backend-store-uri ./mlruns`

---

## Project Structure

```
product-recommendation-system/
├── app/                              # FastAPI application
│   ├── main.py                       # REST API endpoints
│   ├── config.py                     # Configuration & env vars
│   ├── schemas.py                    # Pydantic request/response models
│   ├── model.py                      # Model loading & inference (EASE/hybrid/content)
│   ├── metrics.py                    # Prometheus metrics collection
│   ├── middleware.py                 # LoggingMiddleware
│   ├── explainability.py             # SHAP, LIME, collaborative, content explanations
│   └── fairness.py                   # Bias analysis & MMR mitigation
├── tests/                            # Test suite
│   ├── test_api.py                   # API integration tests
│   ├── test_data_quality.py          # Data validation tests
│   ├── test_evaluation.py            # Model evaluation metric tests
│   ├── test_metrics.py               # Prometheus metrics tests
│   └── test_ml_pipeline.py           # End-to-end ML pipeline tests
├── scripts/                          # Utility scripts
│   ├── train_model.py                # Model training pipeline
│   ├── evaluate_model.py             # Standalone evaluation script
│   ├── load_test.py                  # Load/stress testing
│   └── diagnostic_baseline.py        # Baseline diagnostics
├── models/                           # Trained model artifacts
│   ├── model_bundle.pkl              # Serialized model (joblib)
│   ├── model_metadata.json           # Lightweight metadata
│   └── dataset_stats.json            # Dataset statistics
├── data/                             # Retail Rocket dataset (not committed)
├── prometheus/                       # Prometheus config
│   ├── prometheus.yml
│   └── alerts/
│       ├── api_alerts.yml
│       └── ml_alerts.yml
├── grafana/                          # Grafana dashboards
│   ├── dashboards/
│   │   ├── api_dashboard.json        # API & system metrics dashboard
│   │   └── ml_dashboard.json         # ML prediction metrics dashboard
│   └── provisioning/
├── mlruns/                           # MLflow experiment tracking (auto-generated)
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ARCHITECTURE.md                   # System design & architecture diagrams
├── ML_PIPELINE.md                    # ML pipeline detailed documentation
└── README.md
```

---

## API Endpoints

### Health & Status
```bash
GET /health          # Liveness check
GET /ready           # Readiness check (model loaded)
GET /model/info      # Model metadata, version, evaluation metrics
POST /model/reload   # Hot-reload model from disk
```

### Recommendations
```bash
# POST — full control
POST /recommendations
Content-Type: application/json
{
  "user_id": "1150086",
  "num_recommendations": 10,
  "recommendation_type": "hybrid",   # collaborative | content_based | hybrid
  "filters": {"category": "1016"}    # optional
}

# GET — quick access via URL params
GET /recommendations/{user_id}?num_recommendations=10&rec_type=hybrid

# Batch
POST /recommendations/batch
["1150086", "530559", "152963"]
```

### Explainability
```bash
# method: auto | collaborative | content_based | shap | lime | all
GET /recommendations/{user_id}/explain?product_id=461686&method=shap
GET /recommendations/{user_id}/explain?product_id=461686&method=lime
GET /recommendations/{user_id}/explain?product_id=461686&method=all
```

### Fairness
```bash
GET /fairness/check?user_id=1150086&num_recommendations=10
GET /fairness/report?sample_users=100
```

### Metrics
```bash
GET /metrics    # Prometheus scrape endpoint
```

### Test Users

| User ID | Interactions | Profile |
|---------|-------------|---------|
| `1150086` | 8,751 | Most active user — best for testing |
| `530559` | 4,933 | High activity |
| `152963` | 4,083 | High activity |
| `unknown_user_999` | — | Cold-start fallback test |

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run a single test file
pytest tests/test_api.py -v

# Run by marker
pytest -m unit
pytest -m integration

# Load testing
python scripts/load_test.py
```

---

## Training Options

```bash
# Full hybrid model with evaluation and MLflow logging
python scripts/train_model.py \
  --dataset ./data \
  --model hybrid \
  --evaluate \
  --mlflow

# Collaborative only (EASE + SVD)
python scripts/train_model.py --dataset ./data --model collaborative --evaluate

# Content-based only (TF-IDF)
python scripts/train_model.py --dataset ./data --model content_based

# Standalone evaluation against saved bundle
python scripts/evaluate_model.py \
  --bundle models/model_bundle.pkl \
  --dataset ./data \
  --k 10 \
  --output models/evaluation_report.json
```

**Model types:**

| Flag | Algorithm | Use Case |
|------|-----------|----------|
| `collaborative` | EASE item-item + SVD (for SHAP/LIME) | Personalized recs from interaction history |
| `content_based` | TF-IDF cosine similarity | Item feature similarity |
| `hybrid` | 0.6 × EASE + 0.4 × TF-IDF | Best overall quality |

---

## Docker Deployment

```bash
docker-compose up -d          # Start all services
docker-compose logs -f api    # Stream API logs
docker-compose down           # Stop services
docker-compose down -v        # Stop and remove volumes
```

---

## Monitoring

### Prometheus (http://localhost:9090)

Useful queries:
- `http_requests_total` — total requests by method/endpoint
- `http_request_duration_seconds` — request latency histogram
- `ml_predictions_total` — predictions by recommendation type
- `ml_prediction_duration_seconds` — inference latency

### Grafana (http://localhost:3000)

Two pre-provisioned dashboards:
- **API Dashboard** — request rates, error rates, latency percentiles
- **ML Dashboard** — prediction counts, recommendation type breakdown, cache hit rate

---

## Code Quality

```bash
black app/ tests/ scripts/    # Format
isort app/ tests/ scripts/    # Sort imports
flake8 app/ tests/ scripts/   # Lint
mypy app/                     # Type check

# Check only (CI mode)
black --check app/ tests/ scripts/
isort --check-only app/ tests/ scripts/
flake8 app/ tests/ scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Docker Documentation](https://docs.docker.com)
- [Retail Rocket Dataset (Kaggle)](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- [EASE Paper — Steck 2019](https://arxiv.org/abs/1905.03375)
- [Prometheus Documentation](https://prometheus.io/docs)
- [Grafana Documentation](https://grafana.com/docs)
- [Pytest Documentation](https://docs.pytest.org)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
