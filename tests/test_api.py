"""
Integration tests for API endpoints

Test coverage for:
- Health check endpoints
- Recommendation endpoints (all types, edge cases)
- Model info endpoints
- Metrics endpoint
- Error handling
- Data quality validation
- Model behavior validation
- Performance / latency
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Test health check endpoints"""

    def test_health_endpoint_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "debug" in data
        assert data["status"] == "healthy"

    def test_ready_endpoint_returns_200(self):
        response = client.get("/ready")
        assert response.status_code in [200, 503]  # 503 if model not loaded


# ============================================================================
# Recommendation Endpoint Tests
# ============================================================================


class TestRecommendationEndpoints:
    """Test recommendation endpoints"""

    def test_recommendations_post_success(self):
        payload = {
            "user_id": "user123",
            "num_recommendations": 5,
            "recommendation_type": "collaborative",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == "user123"
        assert data["recommendation_type"] == "collaborative"
        assert isinstance(data["recommendations"], list)

    def test_recommendations_invalid_type(self):
        payload = {
            "user_id": "user123",
            "num_recommendations": 5,
            "recommendation_type": "invalid_type",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_recommendations_invalid_num_recommendations(self):
        payload = {
            "user_id": "user123",
            "num_recommendations": 101,  # exceeds max of 100
            "recommendation_type": "collaborative",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_recommendations_missing_user_id(self):
        payload = {
            "num_recommendations": 5,
            "recommendation_type": "collaborative",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 422

    def test_recommendations_all_types(self):
        for rec_type in ["collaborative", "content_based", "hybrid"]:
            payload = {
                "user_id": "user123",
                "num_recommendations": 5,
                "recommendation_type": rec_type,
            }
            response = client.post("/recommendations", json=payload)
            assert response.status_code == 200, f"Failed for type={rec_type}"
            assert response.json()["recommendation_type"] == rec_type


class TestRecommendationGetEndpoint:
    """Test GET recommendation endpoint"""

    def test_get_recommendations_success(self):
        response = client.get(
            "/recommendations/user123?num_recommendations=5&rec_type=collaborative"
        )
        assert response.status_code == 200
        assert response.json()["user_id"] == "user123"

    def test_get_recommendations_default_parameters(self):
        response = client.get("/recommendations/user123")
        assert response.status_code == 200
        assert response.json()["num_recommendations_returned"] <= 5


# ============================================================================
# Model Info Endpoint Tests
# ============================================================================


class TestModelInfoEndpoints:
    """Test model information endpoints"""

    def test_model_info_endpoint(self):
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_version" in data
        assert "model_type" in data


# ============================================================================
# Metrics Endpoint Tests
# ============================================================================


class TestMetricsEndpoint:
    """Test metrics endpoint"""

    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_http_metrics(self):
        client.get("/health")  # generate some traffic
        response = client.get("/metrics")
        content = response.text

        assert "http_requests_total" in content
        assert "http_request_duration_seconds" in content


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_endpoint_returns_404(self):
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_error_response_structure(self):
        response = client.post("/recommendations", json={})
        assert response.status_code in [400, 422]


# ============================================================================
# Data Quality Tests
# ============================================================================


class TestDataQuality:
    """Validate data quality of recommendation responses."""

    def test_recommendation_scores_in_range(self):
        """All recommendation scores must be within [0.0, 1.0]."""
        payload = {
            "user_id": "user123",
            "num_recommendations": 10,
            "recommendation_type": "collaborative",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        for rec in response.json()["recommendations"]:
            assert (
                0.0 <= rec["score"] <= 1.0
            ), f"Score out of range: {rec['score']} for product {rec['product_id']}"

    def test_product_ids_not_empty(self):
        """Each recommendation must have a non-empty product_id."""
        payload = {
            "user_id": "user123",
            "num_recommendations": 5,
            "recommendation_type": "collaborative",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        for rec in response.json()["recommendations"]:
            assert rec["product_id"] is not None
            assert len(str(rec["product_id"])) > 0

    def test_recommendations_contain_required_fields(self):
        """Every recommendation dict must contain product_id, product_name, and score."""
        payload = {
            "user_id": "user123",
            "num_recommendations": 3,
            "recommendation_type": "hybrid",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        required = {"product_id", "product_name", "score"}
        for rec in response.json()["recommendations"]:
            missing = required - set(rec.keys())
            assert not missing, f"Missing fields {missing} in recommendation"

    def test_num_recommendations_matches_request(self):
        """num_recommendations_returned must equal len(recommendations)."""
        n = 7
        payload = {
            "user_id": "user123",
            "num_recommendations": n,
            "recommendation_type": "collaborative",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["num_recommendations_returned"] == len(data["recommendations"])
        assert len(data["recommendations"]) <= n

    def test_recommendation_scores_are_floats(self):
        """Score field must be a float (not int, not None)."""
        payload = {
            "user_id": "user123",
            "num_recommendations": 5,
            "recommendation_type": "content_based",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        for rec in response.json()["recommendations"]:
            assert isinstance(
                rec["score"], float
            ), f"Expected float score, got {type(rec['score'])}"


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestModelValidation:
    """Validate model behavior and recommendation quality."""

    def test_model_produces_recommendations(self):
        """Model must return non-empty recommendations for any user_id string."""
        for uid in ["user123", "new_user_xyz", "0", "999999999"]:
            payload = {
                "user_id": uid,
                "num_recommendations": 5,
                "recommendation_type": "collaborative",
            }
            response = client.post("/recommendations", json=payload)
            assert response.status_code == 200, f"Request failed for user_id={uid}"
            recs = response.json()["recommendations"]
            assert len(recs) > 0, f"No recommendations returned for user_id={uid}"

    def test_model_handles_cold_start_users(self):
        """Brand-new users (not in training data) must still receive recommendations."""
        payload = {
            "user_id": "completely_new_user_12345_not_in_training",
            "num_recommendations": 5,
            "recommendation_type": "collaborative",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        recs = response.json()["recommendations"]
        assert (
            len(recs) > 0
        ), "Cold-start user should receive popularity-ranked fallback"

    def test_model_recommendation_quality_no_duplicates(self):
        """Recommended items within a single response must be unique."""
        payload = {
            "user_id": "user123",
            "num_recommendations": 20,
            "recommendation_type": "hybrid",
        }
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        product_ids = [r["product_id"] for r in response.json()["recommendations"]]
        assert len(product_ids) == len(
            set(product_ids)
        ), "Duplicate recommendations found"

    def test_model_info_has_version(self):
        """Model info endpoint must return a non-null model_version."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert data["model_version"] is not None

    @pytest.mark.performance
    def test_recommendation_latency_under_200ms(self):
        """Recommendation endpoint must respond within 200 ms."""
        payload = {
            "user_id": "user123",
            "num_recommendations": 10,
            "recommendation_type": "collaborative",
        }
        start = time.time()
        response = client.post("/recommendations", json=payload)
        elapsed_ms = (time.time() - start) * 1000
        assert response.status_code == 200
        assert elapsed_ms < 200, f"Latency {elapsed_ms:.1f}ms exceeds 200ms threshold"


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Latency and throughput assertions."""

    def test_health_check_latency(self):
        """Health check must complete in under 100 ms."""
        start = time.time()
        response = client.get("/health")
        elapsed_ms = (time.time() - start) * 1000
        assert response.status_code == 200
        assert elapsed_ms < 100, f"Health check took {elapsed_ms:.1f}ms (limit: 100ms)"

    def test_recommendation_latency(self):
        """Recommendation endpoint must complete in under 1000 ms."""
        payload = {
            "user_id": "user123",
            "num_recommendations": 5,
            "recommendation_type": "collaborative",
        }
        start = time.time()
        response = client.post("/recommendations", json=payload)
        elapsed_ms = (time.time() - start) * 1000
        assert response.status_code == 200
        assert (
            elapsed_ms < 1000
        ), f"Recommendation took {elapsed_ms:.1f}ms (limit: 1000ms)"

    def test_cached_recommendation_faster(self):
        """Second identical request must benefit from LRU cache and complete in < 50 ms."""
        payload = {
            "user_id": "cache_perf_test_user",
            "num_recommendations": 5,
            "recommendation_type": "collaborative",
        }
        # First call — populate cache
        client.post("/recommendations", json=payload)
        # Second call — should be cache hit
        start = time.time()
        response = client.post("/recommendations", json=payload)
        elapsed_ms = (time.time() - start) * 1000
        assert response.status_code == 200
        assert (
            elapsed_ms < 50
        ), f"Cached recommendation took {elapsed_ms:.1f}ms (limit: 50ms)"
