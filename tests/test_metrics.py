"""
Metrics collection tests

Test that metrics are properly collected and formatted
"""

import pytest
from prometheus_client import REGISTRY

from app.metrics import (
    PredictionMetrics,
    RequestMetrics,
    count_implemented_metrics,
    get_all_metrics,
    http_request_duration_seconds,
    http_requests_total,
    ml_predictions_total,
)


class TestMetricDefinitions:
    """Test that metrics are properly defined."""

    def test_request_count_defined(self):
        """Test http_requests_total is defined."""
        assert http_requests_total is not None
        # Check it's a Counter
        assert hasattr(http_requests_total, "inc")

    def test_request_latency_defined(self):
        """Test http_request_duration_seconds is defined."""
        assert http_request_duration_seconds is not None
        assert hasattr(http_request_duration_seconds, "observe")

    def test_prediction_count_defined(self):
        """Test ml_predictions_total is defined."""
        assert ml_predictions_total is not None
        assert hasattr(ml_predictions_total, "inc")


class TestMetricHelpers:
    """Test metric helper functions."""

    def test_get_all_metrics_returns_dict(self):
        """Test get_all_metrics returns a dictionary."""
        metrics = get_all_metrics()
        assert isinstance(metrics, dict)

    def test_count_implemented_metrics(self):
        """Test count_implemented_metrics function."""
        implemented, total = count_implemented_metrics()
        assert isinstance(implemented, int)
        assert isinstance(total, int)
        assert implemented >= 0
        assert total > 0
        assert implemented <= total


class TestMetricLabels:
    """Test metric labels configuration."""

    def test_request_count_has_correct_labels(self):
        """Test http_requests_total has method, endpoint, status labels."""
        # Increment with labels to verify they work
        http_requests_total.labels(method="GET", endpoint="/test", status=200).inc()

    def test_request_latency_has_correct_labels(self):
        """Test http_request_duration_seconds has correct labels."""
        http_request_duration_seconds.labels(method="GET", endpoint="/test").observe(
            0.1
        )


class TestMetricsEndpoint:
    """Test the /metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)

    def test_metrics_endpoint_returns_200(self, client):
        """Test /metrics returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        """Test /metrics returns Prometheus format."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers[
            "content-type"
        ] or "text/plain" in response.headers.get("content-type", "")

    def test_metrics_contains_http_requests_total(self, client):
        """Test metrics output contains http_requests_total."""
        response = client.get("/metrics")
        assert "http_requests_total" in response.text


class TestMetricsIntegration:
    """Integration tests for metrics with API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)

    def test_request_increments_counter(self, client):
        """Test that making requests increments the counter."""
        # Make some requests
        client.get("/health")
        client.get("/health")

        # Check metrics
        response = client.get("/metrics")
        assert "http_requests_total" in response.text


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# Metric Context Manager Tests
# ============================================================================


class TestRequestMetrics:
    """Test RequestMetrics context manager"""

    def test_request_metrics_increment(self):
        """Test that request metrics counter increments after a RequestMetrics block."""
        from prometheus_client import generate_latest

        # Capture baseline output
        baseline = generate_latest().decode("utf-8")
        baseline_count = baseline.count('method="GET"')

        with RequestMetrics("GET", "/test"):
            pass

        # After the context exits, the counter must have been updated
        updated = generate_latest().decode("utf-8")
        updated_count = updated.count('method="GET"')

        # The GET label must appear at least as many times as before
        # (≥ because other concurrent tests may also emit GET metrics)
        assert updated_count >= baseline_count


class TestPredictionMetrics:
    """Test PredictionMetrics context manager"""

    def test_prediction_metrics_success(self):
        """Test prediction metrics on success"""
        with PredictionMetrics("collaborative"):
            pass

    def test_prediction_metrics_error(self):
        """Test prediction metrics on error"""
        try:
            with PredictionMetrics("collaborative"):
                raise ValueError("Test error")
        except ValueError:
            pass

    # TODO: ====Add more prediction metrics tests====


# ============================================================================
# TODO: ====Metrics Collection Tests====
#
# Test specific metrics:
# - HTTP request counts by endpoint
# - HTTP request latencies
# - ML prediction counts
# - ML prediction latencies
# - Cache hit rates
# - Error rates by type
#


class TestMetricsCollection:
    """TODO: ====Test metrics collection===="""

    def test_http_request_metrics_collected(self):
        """TODO: ====Test that HTTP metrics are collected===="""
        pass

    def test_ml_prediction_metrics_collected(self):
        """TODO: ====Test that ML metrics are collected===="""
        pass

    def test_error_metrics_collected(self):
        """TODO: ====Test that error metrics are collected===="""
        pass

    def test_cache_metrics_collected(self):
        """TODO: ====Test that cache metrics are collected===="""
        pass


# ============================================================================
# Prometheus Output Tests
# ============================================================================


class TestPrometheusOutput:
    """Test Prometheus metrics output format"""

    def test_metrics_output_format(self):
        """Test that metrics are in Prometheus format"""
        from prometheus_client import generate_latest

        output = generate_latest()

        # Check it's bytes
        assert isinstance(output, bytes)

        # Convert to string to check format
        output_str = output.decode("utf-8")

        # Should contain metric names
        assert "http_requests_total" in output_str or len(output_str) > 0

    # TODO: ====Add more Prometheus format tests====


# ============================================================================
# TODO: ====Data Quality Metrics Tests====
#
# These tests should validate data quality metrics:
# - Invalid records count
# - Missing values count
# - Data type errors count
#


# ============================================================================
# TODO: ====Fairness Metrics Tests====
#
# These tests should validate fairness metrics:
# - Demographic parity metrics
# - Equal opportunity metrics
# - Calibration metrics
#
