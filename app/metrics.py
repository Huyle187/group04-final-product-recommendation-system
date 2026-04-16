"""
Prometheus metrics collection module
"""

import time

from prometheus_client import Counter, Gauge, Histogram

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
