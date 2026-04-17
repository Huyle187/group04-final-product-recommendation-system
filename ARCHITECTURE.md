# System Design & Architecture
## Product Recommendation System — Group 04

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Component Design & Responsibilities](#2-component-design--responsibilities)
3. [ML Model Stack](#3-ml-model-stack)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [Infrastructure & Deployment](#5-infrastructure--deployment)
6. [Key Design Decisions](#6-key-design-decisions)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE (Training)                              │
│                                                                         │
│  Retail Rocket Dataset                                                  │
│  (events.csv, item_properties.csv)                                      │
│          │                                                              │
│          ▼                                                              │
│  scripts/train_model.py                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. Data Loading → Temporal Split (per-user last 20%)           │   │
│  │  2. Build interaction matrix (user×item, csr_matrix)            │   │
│  │  3. Train EASE (item-item weights, top 16k items)               │   │
│  │  4. Train TruncatedSVD (latent factors, explainability only)    │   │
│  │  5. Compute TF-IDF item-item cosine similarity matrix           │   │
│  │  6. Evaluate: Precision@k, Recall@k, NDCG@k, Hit Rate@k        │   │
│  │  7. Save model_bundle.pkl + log to MLflow                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│          │                                                              │
│          ▼                                                              │
│  models/model_bundle.pkl                                                │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ONLINE (Inference)                              │
│                                                                         │
│   Client (curl / UI / API)                                              │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     FastAPI (app/main.py)                        │  │
│   │   LoggingMiddleware → CORS → Endpoints → RequestMetrics         │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│          ┌───────────────────┼──────────────────────────┐              │
│          ▼                   ▼                           ▼              │
│  RecommendationModel  ExplainabilityEngine     FairnessChecker          │
│  (app/model.py)       (app/explainability.py)  (app/fairness.py)        │
│          │                                                              │
│   ┌──────▼──────────────────────────────────────────────────────────┐  │
│   │   LRU Cache → Strategy Selector → Recommendation List           │  │
│   └──────┬───────────────┬────────────────────────────┬─────────────┘  │
│          │               │                            │                 │
│          ▼               ▼                            ▼                 │
│   EASE Scoring    Content-Based              Popularity Fallback        │
│   (ease_B matrix) (TF-IDF cosine)            (cold-start users)        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY                                   │
│                                                                         │
│   app/metrics.py ──► Prometheus (9090) ──► Grafana Dashboards (3000)   │
│   (Counters, Histograms, Gauges)                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design & Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| **API Layer** | `app/main.py` | HTTP routing, request validation, error handling, orchestrating model/fairness/explain |
| **Model Inference** | `app/model.py` | Singleton model holder; runs EASE/SVD/hybrid/content scoring; LRU cache |
| **Explainability** | `app/explainability.py` | SHAP (exact latent attribution), LIME (local Ridge), collaborative neighbors, content similarity |
| **Fairness** | `app/fairness.py` | Popularity bias (Gini), category diversity (Shannon entropy), long-tail exposure, segment equity, MMR mitigation |
| **Schemas** | `app/schemas.py` | Pydantic I/O contracts — input validation + response serialization |
| **Metrics** | `app/metrics.py` | Prometheus counters/histograms; `RequestMetrics` and `PredictionMetrics` context managers |
| **Config** | `app/config.py` | All env-var driven settings with defaults |
| **Middleware** | `app/middleware.py` | `LoggingMiddleware` — logs method, path, status, latency |
| **Training** | `scripts/train_model.py` | Builds full model bundle: EASE + SVD + TF-IDF + evaluation |
| **Evaluation** | `scripts/evaluate_model.py` | Standalone evaluation: MRR, ILD, novelty, calibration |

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check (model loaded) |
| `POST` | `/recommendations` | Get recommendations for a user |
| `POST` | `/recommendations/batch` | Batch recommendations for multiple users |
| `GET` | `/recommendations/{user_id}` | Recommendations via URL params |
| `GET` | `/recommendations/{user_id}/explain` | Explain a recommendation |
| `GET` | `/model/info` | Model metadata |
| `POST` | `/model/reload` | Hot-reload model from disk |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `GET` | `/fairness/check` | Per-user fairness report |
| `GET` | `/fairness/report` | System-wide bias analysis |

---

## 3. ML Model Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL BUNDLE (model_bundle.pkl)                    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  EASE — Primary Collaborative Filtering                          │   │
│  │                                                                  │   │
│  │  Input:  user interaction history (weighted implicit feedback)  │   │
│  │  Math:   G = X^T · X     (item co-occurrence, 16k×16k)         │   │
│  │          P = (G + λI)^-1                                        │   │
│  │          B = -P / diag(P),  diag(B) = 0                        │   │
│  │  Score:  user_row[active_indices] @ ease_B  → top-N items      │   │
│  │  Size:   ease_B: 16,000×16,000 float32 (~1 GB)                 │   │
│  │  Note:   Covers top 16k items by interaction count             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TruncatedSVD — Explainability Only (not used for recs)         │   │
│  │                                                                  │   │
│  │  Produces: user_factors (39k × 50), item_factors (92k × 50)   │   │
│  │  Used by:  explain_shap(), explain_lime()                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TF-IDF Content-Based Filtering                                  │   │
│  │                                                                  │   │
│  │  Input:  item category / property text                          │   │
│  │  Produces: item_similarity_matrix (cosine, dense)               │   │
│  │  Score:  mean cosine sim of candidate to user's seen items      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Hybrid Blend                                                    │   │
│  │                                                                  │   │
│  │  hybrid_score = 0.6 × EASE_score + 0.4 × content_score         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Interaction Weighting

| Event Type | Weight | Rationale |
|------------|--------|-----------|
| `view` | 1 | Weak implicit signal |
| `addtocart` | 3 | Moderate intent |
| `transaction` | 5 | Strong preference signal |

### Evaluation Results (after EASE + temporal split)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision@10 | 0.0177 | ~1.8% of top-10 are known relevant items |
| Hit Rate@10 | 15.3% | 1 in 6 users gets at least one relevant hit |
| NDCG@10 | ~0.09 | Reasonable for implicit e-commerce feedback |
| ILD (diversity) | 0.83 | Recommendations are highly diverse |
| User Coverage | 100% | All users receive recommendations |

---

## 4. Data Flow Diagrams

### 4.1 Training Data Flow

```
events.csv
(visitorid, event, itemid, timestamp)
        │
        ▼
  Filter: events ∈ {view, addtocart, transaction}
  Weight: view=1, addtocart=3, transaction=5
        │
        ▼
  Per-user temporal split
  ┌─────────────────────────────────────────┐
  │  Each user's interactions sorted by ts  │
  │  First 80% → train_df                   │
  │  Last  20% → test_df                    │
  └─────────────────────────────────────────┘
        │
        ▼
  Build csr_matrix (sparse)
  Rows:   user_idx  (39,635 users)
  Cols:   item_idx  (92,146 items)
  Values: weighted interaction counts
        │
        ├──► EASE training (top 16k items only)
        │         ease_B [16k×16k]
        │         ease_active_indices [16k]
        │
        ├──► SVD training (50 components)
        │         user_factors [39k×50]
        │         item_factors [92k×50]
        │
        └──► TF-IDF similarity
                  item_similarity_matrix [subset×subset]
```

### 4.2 Recommendation Request Flow

```
POST /recommendations
{ user_id, num_recs, rec_type, filters }
        │
        ▼
  Pydantic input validation
        │
        ▼
  LRU cache lookup  key = "{user_id}:{n}:{type}"
  ┌── HIT  ──► return cached result (O(1))
  │
  └── MISS ──►
              │
              ▼
        user_id ∈ user_id_to_idx?
        ┌── NO  ──► popularity fallback (cold-start)
        │           sorted by log-normalized count
        │
        └── YES ──►
                    │
                    ▼
              rec_type dispatch:
              ┌─ "collaborative" ──► EASE: user_row[active] @ ease_B
              │                      mask seen items
              │                      optional category filter
              │                      argsort top-N
              │
              ├─ "content_based" ──► mean cosine sim over user history
              │                      mask seen items
              │                      argsort top-N
              │
              └─ "hybrid" ──────────► over-fetch 3N from each strategy
                                      merge + blend scores (0.6/0.4)
                                      sort, return top-N
              │
              ▼
        store in LRU cache (evict oldest if size ≥ 1000)
              │
              ▼
  Build RecommendationResponse (Pydantic)
  Return JSON
```

### 4.3 SHAP Explanation Flow

```
GET /recommendations/{user_id}/explain
    ?product_id=X&method=shap
        │
        ▼
  ExplainabilityEngine.explain_shap(user_id, product_id)
        │
        ▼
  u_vec = user_factors[user_idx]    # shape: (50,)
  i_vec = item_factors[item_idx]    # shape: (50,)
        │
        ▼
  SHAP_j = u_vec[j] × i_vec[j]     # element-wise product
  (analytically exact for dot-product scoring models)
        │
        ▼
  Top-5 dims by |SHAP_j|
  For each top dim → find 3 catalog items most aligned
        │
        ▼
  Return:
    total_score, top_dimensions,
    representative_items, explanation_text
```

### 4.4 LIME Explanation Flow

```
GET /recommendations/{user_id}/explain
    ?product_id=X&method=lime
        │
        ▼
  ExplainabilityEngine.explain_lime(user_id, product_id, n_samples=50)
        │
        ▼
  item_vec = item_factors[item_idx]     # shape: (50,)
  user_vec = user_factors[user_idx]     # shape: (50,)
        │
        ▼
  noise     = N(0, 0.1) × (50, 50)
  perturbed = item_vec + noise           # (50, 50)
  scores    = perturbed @ user_vec       # (50,)
        │
        ▼
  kernel weights = exp(-||noise||² / 0.5)
        │
        ▼
  Ridge regression (α=0.01):
    scores ~ β · perturbed  (weighted)
        │
        ▼
  Return:
    local_fidelity (R²), feature_importances (top-5 dims)
```

### 4.5 Fairness Check Flow

```
GET /fairness/check?user_id=X
        │
        ▼
  model.get_recommendations(user_id, n=10, type="hybrid")
        │
        ▼
  FairnessChecker.full_fairness_report(user_id, rec_ids)
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  popularity_bias:                                          │
  │    popular_items_fraction = recs above P80 / total        │
  │    gini_coefficient of rec popularity distribution        │
  │    is_biased = fraction > FAIRNESS_POPULARITY_THRESHOLD   │
  │                                                            │
  │  category_diversity:                                       │
  │    Shannon entropy of category distribution               │
  │    normalized to [0, 1]                                    │
  │    meets_threshold = diversity ≥ MIN_CATEGORY_DIVERSITY   │
  │                                                            │
  │  overall_fairness_score:                                   │
  │    = 0.5 × (1 − popularity_bias) + 0.5 × diversity       │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
        │
        ▼
  Return JSON fairness report
```

---

## 5. Infrastructure & Deployment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Docker Compose (3 Services)                          │
│                                                                         │
│  ┌───────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │
│  │     api  :8000        │  │  prometheus  :9090   │  │ grafana:3000 │  │
│  │                       │  │                      │  │              │  │
│  │  FastAPI + Uvicorn    │◄─┤  Scrapes /metrics    │◄─┤  Dashboards  │  │
│  │                       │  │  every 5s            │  │              │  │
│  │  Volumes:             │  │                      │  │  - API       │  │
│  │  ./models → /models   │  │  Alert rules:        │  │  - ML        │  │
│  │  ./logs   → /logs     │  │  api_alerts.yml      │  │              │  │
│  │                       │  │  ml_alerts.yml       │  │              │  │
│  └───────────────────────┘  └─────────────────────┘  └──────────────┘  │
│                                                                         │
│  Network:  recommendation-net  (bridge)                                 │
│  Volumes:  prometheus_data, grafana_data  (persistent named volumes)    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service URLs (local)

| Service | URL | Credentials |
|---------|-----|-------------|
| API + Swagger | http://localhost:8000 / http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **EASE over neural CF** | Closed-form solution — no gradient descent. Deterministic, fast inference, interpretable. State-of-the-art on implicit feedback benchmarks. |
| **MAX_EASE_ITEMS = 16,000** | Full 92k item Gram matrix ≈ 34 GB. Capping at 16k → ~1 GB matrix. Still covers >95% of all user interactions due to power-law item distribution. |
| **SVD retained for explainability only** | SHAP/LIME require per-user and per-item latent factors. EASE `B` matrix doesn't expose these. SVD provides them at low marginal cost. |
| **Per-user temporal split** | Correct evaluation protocol for implicit feedback. Users must appear in both train and test sets. The previous `GroupShuffleSplit` created non-overlapping user sets → near-zero metrics (Precision@10 ≈ 0.000063). Fixed to 0.0177 (280× improvement). |
| **Interaction weights (1 / 3 / 5)** | Weighted implicit feedback better represents preference strength. Transactions signal strong intent; views are weak. Improves model quality without requiring explicit ratings. |
| **LRU cache** | Recommendations (capacity 1000) and explanations (capacity 500) are cached by `(user_id, n, type)` key. Reduces repeated scoring for the same user. |
| **Singleton model** | Loaded once at startup from `model_bundle.pkl`. `POST /model/reload` hot-swaps the singleton without restarting the process. |
| **Popularity fallback (cold-start)** | Unknown users receive log-normalized top-N items. Graceful degradation — system always returns results. |
| **Fairness as first-class endpoints** | Popularity bias, category diversity, long-tail exposure, and user-segment equity are all served via dedicated API endpoints — not buried in logging. |
| **SHAP via element-wise product** | For dot-product scoring `score = u · i`, SHAP attribution is `SHAP_j = u_j × i_j` — analytically exact, no sampling, < 1ms. No external SHAP library required for core attribution. |
