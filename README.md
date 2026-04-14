# Product Recommendation System

**DDM501 - AI in Production: From Models to Systems**

## Overview

A real-time product recommendation engine for an e-commerce platform. This system implements collaborative filtering, content-based filtering, and hybrid recommendation methods to provide personalized product suggestions.

**Topic:** Topic 1 - Product Recommendation System
**Datasets:** Amazon Product Reviews, Instacart Market Basket, Retail Rocket

---

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- pip

### Installation & Setup

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Start all services**
```bash
docker-compose up -d
```

3. **Verify services are running**
```bash
curl http://localhost:8000/health
```

### Service URLs
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)

---

## Project Structure

```
product-recommendation-system/
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # REST API endpoints
│   ├── config.py                 # Configuration management
│   ├── schemas.py                # Request/response Pydantic models
│   ├── model.py                  # Model loading & inference
│   ├── metrics.py                # Prometheus metrics collection
│   └── middleware.py             # Custom middleware (logging, auth, etc)
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_api.py               # API integration tests
│   └── test_metrics.py           # Metrics collection tests
├── scripts/                      # Utility scripts
│   ├── load_test.py              # Load/stress testing
│   └── train_model.py            # Model training [TODO]
├── models/                       # Trained model storage
├── prometheus/                   # Prometheus monitoring config
│   ├── prometheus.yml            # Prometheus configuration [TODO]
│   └── alerts/
│       ├── api_alerts.yml        # API alert rules [TODO]
│       └── ml_alerts.yml         # ML alert rules [TODO]
├── grafana/                      # Grafana visualization config
│   ├── dashboards/
│   │   ├── api_dashboard.json    # System metrics dashboard [TODO]
│   │   └── ml_dashboard.json     # ML metrics dashboard [TODO]
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboards.yml    # Dashboard provisioning [TODO]
│       └── datasources/
│           └── prometheus.yml    # Datasource configuration [TODO]
├── .github/workflows/
│   └── ci.yml                    # GitHub Actions CI/CD pipeline
├── .gitignore
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Multi-service orchestration
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Team Responsibilities

### Your Work (Deployment & Testing)
- ✅ REST API implementation
- ✅ Docker containerization
- ✅ Docker Compose orchestration
- ✅ Unit tests
- ✅ Integration tests
- ✅ Load testing
- ✅ CI/CD pipeline

### Others' Work
- 📋 Problem Definition & Requirements
- 🤖 ML Pipeline: Data ingestion, preprocessing, feature engineering, model training
- 📊 Experiment tracking (MLflow or equivalent)
- 📈 Monitoring: Prometheus metrics, Grafana dashboards, alerting rules
- 🎯 Responsible AI: Fairness analysis, explainability (SHAP/LIME), ethics discussion
- 📚 Documentation: Architecture, API docs, setup guides

---

## API Endpoints

### Health Check
```bash
GET /health
```
Response: `{"status": "healthy"}`

### Get Recommendations
```bash
POST /recommendations
Content-Type: application/json

{
  "user_id": "user123",
  "num_recommendations": 5,
  "recommendation_type": "collaborative"  # or "content_based", "hybrid"
}
```

### Model Info
```bash
GET /model/info
```

### Metrics
```bash
GET /metrics
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run load tests
python scripts/load_test.py
```

---

## Docker Deployment

### Build and Start
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f api
```

### Stop Services
```bash
docker-compose down
```

### Clean Everything (including volumes)
```bash
docker-compose down -v
```

---

## Monitoring

### Prometheus
- **URL:** http://localhost:9090
- **Query Examples:**
  - `http_requests_total` - Total requests
  - `http_request_duration_seconds` - Request latency
  - `ml_predictions_total` - Total predictions

### Grafana
- **URL:** http://localhost:3000
- **Credentials:** admin / admin
- **Dashboards:** [TODO: List dashboards created]

---

## Development

### Adding Dependencies
```bash
pip install <package>
pip freeze > requirements.txt
```

### Code Quality
```bash
# Format code
black app/

# Type checking
mypy app/

# Linting
flake8 app/
```

---

## Submission Requirements

- ✅ GitHub repository (public or instructor as collaborator)
- ✅ All team members have meaningful commits
- ✅ Meaningful commit messages and proper branching
- ✅ Includes .gitignore
- ✅ Docker files present
- ✅ Tests with good coverage
- ✅ CI/CD pipeline configured

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Docker Documentation](https://docs.docker.com)
- [Prometheus Documentation](https://prometheus.io/docs)
- [Grafana Documentation](https://grafana.com/docs)
- [Pytest Documentation](https://docs.pytest.org)

---

## Contact

For questions about deployment, testing, and API implementation, reach out to the team member handling deployment & testing.
