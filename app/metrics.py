"""
Prometheus metrics collection module
"""

import time

from prometheus_client import Counter, Gauge, Histogram, Info

# =============================================================================
# Application Metrics (HTTP Requests)
# =============================================================================

# TODO 1: Define HTTP request counter
# This counter should track total HTTP requests with labels for:
# - method: HTTP method (GET, POST, etc.)
# - endpoint: Request path (/predict, /health, etc.)
# - status: HTTP status code (200, 404, 500, etc.)

# (Removed redundant REQUEST_COUNT)

# TODO 2: Define HTTP request latency histogram
# This histogram should track request duration in seconds with labels for:
# - method: HTTP method
# - endpoint: Request path
# Suggested buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

# (Removed redundant REQUEST_LATENCY)


# =============================================================================
# ML-Specific Metrics
# =============================================================================

# TODO 3: Define prediction counter
# This counter should track total predictions with labels for:
# - model_version: Version of the model used

# (Removed redundant PREDICTION_COUNT)


# TODO 4: Define prediction latency histogram
# This histogram should track prediction duration in seconds
# Suggested buckets for ML predictions (faster than HTTP):
# [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

# (Removed redundant PREDICTION_LATENCY)


# TODO 5: Define prediction value histogram
# This histogram should track the distribution of prediction values

PREDICTION_VALUE = Histogram(
    "ml_prediction_value",
    "Distribution of prediction values",
    ["model_version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)



# ============================================================================
# HTTP Metrics (API performance)
# ============================================================================

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000),
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000),
)


# ============================================================================
# ML Metrics (Model performance)
# ============================================================================

ml_predictions_total = Counter(
    "ml_predictions_total", "Total ML predictions", ["recommendation_type", "status"]
)

ml_prediction_duration_seconds = Histogram(
    "ml_prediction_duration_seconds",
    "ML prediction latency in seconds",
    ["recommendation_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

ml_predictions_cached = Counter(
    "ml_predictions_cached", "Cache hits for recommendations", ["recommendation_type"]
)

# TODO: ====Add model quality metrics====
# - Model accuracy
# - Prediction quality scores
# - Recommendation diversity scores
# - Fairness metrics (bias detection)

# =============================================================================
# Model Status Metrics
# =============================================================================

# TODO 7: Define model loaded gauge
# This gauge should indicate if the model is loaded (1) or not (0)

MODEL_LOADED = Gauge("ml_model_loaded", "Whether the ML model is loaded (1) or not (0)")


# TODO 8: Define model info metric
# This info metric should provide static information about the model:
# - version: Model version
# - type: Model type (SVD, NMF, etc.)
# - path: Path to model file

MODEL_INFO = Info("ml_model", "Information about the loaded ML model")


# TODO 9: Define model last reload timestamp gauge
# This gauge should track when the model was last loaded (Unix timestamp)

MODEL_LAST_RELOAD = Gauge(
    "ml_model_last_reload_timestamp", "Unix timestamp of last model reload"
)


# =============================================================================
# Batch Prediction Metrics (BONUS)
# =============================================================================

# TODO 10 (BONUS): Define batch size histogram
# Track the size of batch prediction requests

BATCH_SIZE = Histogram(
    "ml_batch_prediction_size",
    "Size of batch prediction requests",
    buckets=[1, 5, 10, 25, 50, 100],
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_all_metrics():
    """Return a dictionary of all defined metrics for inspection."""
    return {
        "request_count": http_requests_total,
        "request_latency": http_request_duration_seconds,
        "prediction_count": ml_predictions_total,
        "prediction_latency": ml_prediction_duration_seconds,
        "prediction_value": PREDICTION_VALUE,
        "model_loaded": MODEL_LOADED,
        "model_info": MODEL_INFO,
        "model_last_reload": MODEL_LAST_RELOAD,
        "batch_size": BATCH_SIZE,
    }


def count_implemented_metrics():
    """Count how many metrics have been implemented."""
    metrics = get_all_metrics()
    implemented = sum(1 for m in metrics.values() if m is not None)
    return implemented, len(metrics)


# ============================================================================
# Application Metrics
# ============================================================================

app_info = Gauge("app_info", "Application information", ["version", "model_version"])

recommendation_cache_size = Gauge(
    "recommendation_cache_size",
    "Current size of recommendation cache (items)",
)

active_requests = Gauge("active_requests", "Number of active requests", ["endpoint"])


# ============================================================================
# Error Metrics
# ============================================================================

errors_total = Counter("errors_total", "Total errors", ["error_type", "endpoint"])

validation_errors_total = Counter(
    "validation_errors_total", "Total validation errors", ["field"]
)


# ============================================================================
# TODO: ====Data Quality Metrics====
# - Records processed
# - Invalid records
# - Missing values
# - Data drift detection scores


# ============================================================================
# Helper Functions
# ============================================================================


class RequestMetrics:
    """Context manager for tracking request metrics"""

    def __init__(self, method: str, endpoint: str):
        self.method = method
        self.endpoint = endpoint
        self.start_time = None
        self.status_code = 200

    def __enter__(self):
        self.start_time = time.time()
        active_requests.labels(endpoint=self.endpoint).inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        http_request_duration_seconds.labels(
            method=self.method, endpoint=self.endpoint
        ).observe(duration)

        http_requests_total.labels(
            method=self.method, endpoint=self.endpoint, status=self.status_code
        ).inc()

        active_requests.labels(endpoint=self.endpoint).dec()

        if exc_type is not None:
            errors_total.labels(
                error_type=exc_type.__name__, endpoint=self.endpoint
            ).inc()


class PredictionMetrics:
    """Context manager for tracking prediction metrics"""

    def __init__(self, recommendation_type: str):
        self.recommendation_type = recommendation_type
        self.start_time = None
        self.status = "success"

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is not None:
            self.status = "error"

        ml_prediction_duration_seconds.labels(
            recommendation_type=self.recommendation_type
        ).observe(duration)

        ml_predictions_total.labels(
            recommendation_type=self.recommendation_type, status=self.status
        ).inc()
