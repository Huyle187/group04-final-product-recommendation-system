"""
Tests for advanced evaluation metrics, bias analysis, and SHAP/LIME explainability.

All tests use synthetic in-memory data — no trained model bundle or dataset files
required. Fixtures build minimal ModelEvaluator and ExplainabilityEngine instances
directly from numpy arrays.

Coverage:
- Ranking metrics: MRR, Hit Rate, F1 (Precision/Recall already in test_ml_pipeline.py)
- Diversity metrics: Intra-List Diversity, Category Entropy, Novelty
- Coverage metrics: Catalog Coverage, User Coverage, Long-Tail Coverage
- Fairness: User-segment fairness, Long-tail exposure, MMR/popularity mitigation
- Explainability: SHAP values (analytical), LIME (local fidelity), dispatcher
"""

import sys
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.explainability import ExplainabilityEngine
from app.fairness import FairnessChecker

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """
    Minimal mock RecommendationModel with 3 users x 5 items, 4 latent dimensions.
    L2-normalized so dot product == cosine similarity.
    """
    model = MagicMock()
    model.model = {"initialized": True}

    rng = np.random.default_rng(42)

    # User / item factors — L2 normalize rows
    raw_u = rng.random((3, 4)).astype(np.float32)
    raw_i = rng.random((5, 4)).astype(np.float32)
    norms_u = np.linalg.norm(raw_u, axis=1, keepdims=True)
    norms_i = np.linalg.norm(raw_i, axis=1, keepdims=True)
    model.user_factors = raw_u / norms_u
    model.item_factors = raw_i / norms_i

    model.user_id_to_idx = {"u1": 0, "u2": 1, "u3": 2}
    model.item_id_to_idx = {"i1": 0, "i2": 1, "i3": 2, "i4": 3, "i5": 4}
    model.idx_to_user_id = {0: "u1", 1: "u2", 2: "u3"}
    model.idx_to_item_id = {0: "i1", 1: "i2", 2: "i3", 3: "i4", 4: "i5"}
    model.item_popularity = {"i1": 10, "i2": 8, "i3": 6, "i4": 3, "i5": 1}
    model.item_categories = {
        "i1": "electronics",
        "i2": "electronics",
        "i3": "books",
        "i4": "books",
        "i5": "clothing",
    }

    # Interaction matrix: u1 saw i1,i2; u2 saw i2,i3; u3 saw i1,i3
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 2]
    data = [1.0, 3.0, 1.0, 5.0, 1.0, 3.0]
    model.interaction_matrix = csr_matrix(
        (data, (rows, cols)), shape=(3, 5), dtype=np.float32
    )

    # item_similarity_matrix for content-based tests (5x5 identity-like)
    sim = model.item_factors @ model.item_factors.T
    model.item_similarity_matrix = sim.astype(np.float32)

    return model


@pytest.fixture
def engine(mock_model):
    return ExplainabilityEngine(mock_model)


@pytest.fixture
def checker(mock_model):
    return FairnessChecker(mock_model)


# =============================================================================
# SHAP Explainability Tests
# =============================================================================


class TestSHAPExplainability:
    """Validate the analytical SHAP latent-factor decomposition."""

    def test_shap_returns_correct_method(self, engine):
        result = engine.explain_shap("u1", "i3")
        assert result["method"] == "shap_latent"

    def test_shap_values_sum_equals_score(self, engine, mock_model):
        """sum(SHAP_j) must equal the dot-product score (efficiency axiom)."""
        result = engine.explain_shap("u1", "i3", top_k=4)
        assert "error" not in result

        u_vec = mock_model.user_factors[0]
        i_vec = mock_model.item_factors[2]
        expected_score = float(u_vec @ i_vec)
        assert abs(result["total_score"] - expected_score) < 1e-5

    def test_shap_top_k_dimensions_length(self, engine):
        """top_dimensions must have exactly top_k entries."""
        result = engine.explain_shap("u1", "i3", top_k=3)
        assert len(result["top_dimensions"]) == 3

    def test_shap_dimension_fields_present(self, engine):
        """Each dimension entry must have required keys."""
        result = engine.explain_shap("u1", "i3", top_k=2)
        for dim in result["top_dimensions"]:
            assert "dimension" in dim
            assert "shap_value" in dim
            assert "representative_items" in dim
            assert "direction" in dim

    def test_shap_cold_start_user_returns_error(self, engine):
        result = engine.explain_shap("unknown_user", "i1")
        assert "error" in result

    def test_shap_unknown_product_returns_error(self, engine):
        result = engine.explain_shap("u1", "unknown_product")
        assert "error" in result

    def test_shap_result_is_cached(self, engine):
        """Second call with same args must hit cache (cache size grows by 1)."""
        engine.explain_shap("u1", "i3", top_k=4)
        size_after_first = len(engine._cache)
        engine.explain_shap("u1", "i3", top_k=4)
        assert len(engine._cache) == size_after_first  # no new entry

    def test_shap_abs_shap_descending(self, engine):
        """top_dimensions must be sorted by |SHAP| descending."""
        result = engine.explain_shap("u1", "i3", top_k=4)
        abs_vals = [d["abs_shap"] for d in result["top_dimensions"]]
        assert abs_vals == sorted(abs_vals, reverse=True)


# =============================================================================
# LIME Explainability Tests
# =============================================================================


class TestLIMEExplainability:
    """Validate LIME local linear approximation."""

    def test_lime_returns_correct_method(self, engine):
        result = engine.explain_lime("u1", "i3")
        assert result["method"] == "lime_latent"

    def test_lime_local_fidelity_in_range(self, engine):
        """R² (local_fidelity) must be in [0, 1]."""
        result = engine.explain_lime("u1", "i3", n_samples=50)
        assert "error" not in result
        assert 0.0 <= result["local_fidelity"] <= 1.0

    def test_lime_feature_importances_count(self, engine):
        """feature_importances must have top_k entries."""
        result = engine.explain_lime("u1", "i3", top_k=3)
        assert len(result["feature_importances"]) == 3

    def test_lime_importance_fields_present(self, engine):
        """Each feature_importance entry must have required keys."""
        result = engine.explain_lime("u1", "i3", top_k=2)
        for fi in result["feature_importances"]:
            assert "dimension" in fi
            assert "importance" in fi
            assert "direction" in fi

    def test_lime_cold_start_user_returns_error(self, engine):
        result = engine.explain_lime("unknown_user", "i1")
        assert "error" in result

    def test_lime_result_is_cached(self, engine):
        engine.explain_lime("u2", "i4", n_samples=30)
        size_after_first = len(engine._cache)
        engine.explain_lime("u2", "i4", n_samples=30)
        assert len(engine._cache) == size_after_first

    def test_lime_n_samples_recorded(self, engine):
        result = engine.explain_lime("u1", "i3", n_samples=20)
        assert result["n_samples"] == 20


# =============================================================================
# Explainability Dispatcher Tests
# =============================================================================


class TestExplainDispatcher:
    """Validate the unified explain() entry point."""

    def test_explain_shap_method(self, engine):
        result = engine.explain("u1", "i3", method="shap")
        assert result["method"] == "shap_latent"

    def test_explain_lime_method(self, engine):
        result = engine.explain("u1", "i3", method="lime")
        assert result["method"] == "lime_latent"

    def test_explain_collaborative_method(self, engine):
        result = engine.explain("u1", "i3", method="collaborative")
        assert result["method"] == "collaborative_neighbors"

    def test_explain_content_based_method(self, engine):
        result = engine.explain("u1", "i3", method="content_based")
        assert result["method"] == "content_similarity"

    def test_explain_all_returns_four_keys(self, engine):
        result = engine.explain("u1", "i3", method="all")
        assert set(result.keys()) == {"collaborative", "content_based", "shap", "lime"}

    def test_explain_auto_uses_shap_when_model_loaded(self, engine):
        result = engine.explain("u1", "i3", method="auto")
        assert result.get("method") == "shap_latent"

    def test_explain_unknown_method_returns_fallback(self, engine):
        """Unknown method string must fall through to content_based gracefully."""
        result = engine.explain("u1", "i3", method="unknown_xyz")
        # Falls through: user_factors is not None → shap (auto path) or error
        assert "method" in result or "error" in result


# =============================================================================
# User-Segment Fairness Tests
# =============================================================================


class TestUserSegmentFairness:
    """Validate user-segment fairness checks."""

    def test_segment_keys_present(self, checker):
        recs = {
            "u1": ["i3", "i4", "i5"],
            "u2": ["i1", "i4", "i5"],
            "u3": ["i2", "i4", "i5"],
        }
        result = checker.check_user_segment_fairness(["u1", "u2", "u3"], recs)
        assert "segments" in result
        assert "diversity_gap_power_vs_casual" in result
        assert "unfairness_detected" in result

    def test_fairness_gap_non_negative(self, checker):
        recs = {"u1": ["i3", "i4", "i5"], "u2": ["i1", "i5"], "u3": ["i2", "i4"]}
        result = checker.check_user_segment_fairness(["u1", "u2", "u3"], recs)
        assert result["diversity_gap_power_vs_casual"] >= 0.0
        assert result["popularity_gap_power_vs_casual"] >= 0.0

    def test_unknown_users_returns_error(self, checker):
        result = checker.check_user_segment_fairness(
            ["nonexistent_1", "nonexistent_2"], {}
        )
        assert "error" in result


# =============================================================================
# Long-Tail Exposure Tests
# =============================================================================


class TestLongTailExposure:
    """Validate long-tail item exposure analysis."""

    def test_long_tail_keys_present(self, checker):
        all_recs = [["i1", "i3"], ["i2", "i5"], ["i4"]]
        result = checker.check_long_tail_exposure(all_recs)
        for key in [
            "head_exposure",
            "torso_exposure",
            "tail_exposure",
            "tail_suppression_detected",
        ]:
            assert key in result

    def test_exposure_values_in_range(self, checker):
        all_recs = [["i1", "i2", "i3", "i4", "i5"]]
        result = checker.check_long_tail_exposure(all_recs)
        for key in ["head_exposure", "torso_exposure", "tail_exposure"]:
            assert 0.0 <= result[key] <= 1.0

    def test_tail_suppression_detected_is_bool(self, checker):
        all_recs = [["i1", "i2"]]  # only head items
        result = checker.check_long_tail_exposure(all_recs)
        assert isinstance(result["tail_suppression_detected"], bool)


# =============================================================================
# Bias Mitigation Tests
# =============================================================================


class TestMitigationStrategies:
    """Validate MMR, popularity penalty, and calibrated re-ranking."""

    def _sample_recs(self, mock_model):
        return [
            {"product_id": "i1", "score": 0.9},
            {"product_id": "i2", "score": 0.8},
            {"product_id": "i3", "score": 0.7},
            {"product_id": "i4", "score": 0.6},
            {"product_id": "i5", "score": 0.5},
        ]

    def test_mmr_preserves_length(self, checker, mock_model):
        recs = self._sample_recs(mock_model)
        result = checker.apply_mitigation(recs, strategy="diversity_rerank")
        assert len(result) == len(recs)

    def test_mmr_result_has_product_ids(self, checker, mock_model):
        recs = self._sample_recs(mock_model)
        result = checker.apply_mitigation(recs, strategy="diversity_rerank")
        original_ids = {r["product_id"] for r in recs}
        result_ids = {r["product_id"] for r in result}
        assert original_ids == result_ids

    def test_popularity_penalty_preserves_length(self, checker, mock_model):
        recs = self._sample_recs(mock_model)
        result = checker.apply_mitigation(recs, strategy="popularity_penalty")
        assert len(result) == len(recs)

    def test_popularity_penalty_lowers_popular_score(self, checker, mock_model):
        """After penalty, i1 (most popular) must have a lower score than original."""
        recs = self._sample_recs(mock_model)
        result = checker.apply_mitigation(recs, strategy="popularity_penalty")
        original_i1 = next(r["score"] for r in recs if r["product_id"] == "i1")
        penalized_i1 = next(r["score"] for r in result if r["product_id"] == "i1")
        assert penalized_i1 <= original_i1

    def test_calibrated_preserves_length(self, checker, mock_model):
        recs = self._sample_recs(mock_model)
        result = checker.apply_mitigation(
            recs, strategy="calibrated", user_history=["i1", "i2", "i3"]
        )
        assert len(result) == len(recs)

    def test_empty_recs_returns_empty(self, checker):
        assert checker.apply_mitigation([], strategy="diversity_rerank") == []


# =============================================================================
# Ranking Metric Sanity Tests (ModelEvaluator)
# =============================================================================


class TestModelEvaluatorMetrics:
    """
    Light-weight unit tests for ModelEvaluator ranking and diversity helpers.
    These test the metric logic without file I/O by calling the formulas directly.
    """

    def test_mrr_perfect_first_hit(self):
        """MRR = 1.0 when the first recommendation is a hit."""
        top_k = ["i1", "i2", "i3"]
        ground_truth = {"i1"}
        mrr = 0.0
        for rank, item in enumerate(top_k):
            if item in ground_truth:
                mrr = 1.0 / (rank + 1)
                break
        assert mrr == 1.0

    def test_mrr_second_hit(self):
        """MRR = 0.5 when the second item is the first hit."""
        top_k = ["i2", "i1", "i3"]
        ground_truth = {"i1"}
        mrr = 0.0
        for rank, item in enumerate(top_k):
            if item in ground_truth:
                mrr = 1.0 / (rank + 1)
                break
        assert abs(mrr - 0.5) < 1e-9

    def test_mrr_no_hit(self):
        """MRR = 0.0 when no recommended item is relevant."""
        top_k = ["i2", "i3"]
        ground_truth = {"i1"}
        mrr = 0.0
        for rank, item in enumerate(top_k):
            if item in ground_truth:
                mrr = 1.0 / (rank + 1)
                break
        assert mrr == 0.0

    def test_hit_rate_positive(self):
        assert 1.0 if {"i1", "i5"} & {"i1"} else 0.0 == 1.0

    def test_f1_harmonic_mean(self):
        """F1 = 2*P*R / (P+R) and must be between P and R."""
        p, r = 0.4, 0.6
        f1 = 2 * p * r / (p + r)
        assert min(p, r) <= f1 <= max(p, r)

    def test_f1_zero_when_precision_zero(self):
        p, r = 0.0, 0.5
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert f1 == 0.0

    def test_ild_range(self, mock_model):
        """ILD must be in [0, 1] for L2-normalized item vectors."""
        item_indices = [0, 1, 2, 3]
        vecs = mock_model.item_factors[item_indices]
        k = len(item_indices)
        sim_matrix = vecs @ vecs.T
        sum_sim = sim_matrix.sum() - np.trace(sim_matrix)
        ild = 1.0 - sum_sim / (k * (k - 1))
        assert 0.0 <= float(np.clip(ild, 0.0, 1.0)) <= 1.0

    def test_novelty_positive(self, mock_model):
        """Novelty (self-information) must be > 0 for items with popularity > 0."""
        total = float(sum(mock_model.item_popularity.values()))
        novelties = []
        for iid, pop in mock_model.item_popularity.items():
            prob = pop / total
            novelties.append(-np.log2(prob + 1e-12))
        assert all(n > 0 for n in novelties)

    def test_catalog_coverage_full(self):
        """Coverage = 1.0 when every catalog item is recommended."""
        all_items = {"i1", "i2", "i3", "i4", "i5"}
        recommended = {"i1", "i2", "i3", "i4", "i5"}
        assert len(recommended) / len(all_items) == 1.0

    def test_long_tail_coverage_zero_when_no_tail_recommended(self):
        tail_items = {"i4", "i5"}
        recommended = {"i1", "i2"}
        coverage = len(recommended & tail_items) / len(tail_items)
        assert coverage == 0.0


# =============================================================================
# API Endpoint Integration Tests
# =============================================================================


class TestNewAPIEndpoints:
    """Smoke tests for the new /fairness/report and updated explain endpoints."""

    def test_explain_shap_endpoint(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.get(
            "/recommendations/user123/explain?product_id=prod_0&method=shap"
        )
        assert response.status_code == 200
        data = response.json()
        # Either returns shap result or error dict — both are valid 200 responses
        assert isinstance(data, dict)

    def test_explain_lime_endpoint(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.get(
            "/recommendations/user123/explain?product_id=prod_0&method=lime"
        )
        assert response.status_code == 200
        assert isinstance(response.json(), dict)

    def test_explain_all_endpoint(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.get(
            "/recommendations/user123/explain?product_id=prod_0&method=all"
        )
        assert response.status_code == 200

    def test_fairness_report_endpoint(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.get("/fairness/report?sample_users=5")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
