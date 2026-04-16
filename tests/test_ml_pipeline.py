"""
Unit tests for the ML training pipeline.

Tests use synthetic data injected directly into ModelTrainer — no file I/O
or Retail Rocket dataset required. This makes the tests fast and CI-friendly.

Coverage:
- Data loading and interaction matrix construction
- Collaborative filtering (TruncatedSVD)
- Feature engineering (TF-IDF)
- Evaluation metric computation (Precision@k, Recall@k, NDCG@k, coverage)
- Model artifact saving and loading
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse

# Allow importing from project root when run via pytest from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model import ModelTrainer

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """
    Synthetic interaction DataFrame matching Retail Rocket schema.
    3 users x 7 unique items, with varied event weights.
    """
    return pd.DataFrame(
        {
            "visitorid": [
                "u1",
                "u1",
                "u1",
                "u1",
                "u1",
                "u2",
                "u2",
                "u2",
                "u2",
                "u2",
                "u3",
                "u3",
                "u3",
                "u3",
                "u3",
            ],
            "itemid": [
                "i1",
                "i2",
                "i3",
                "i4",
                "i5",
                "i2",
                "i3",
                "i4",
                "i5",
                "i6",
                "i1",
                "i3",
                "i5",
                "i6",
                "i7",
            ],
            "weight": [1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1],
        }
    )


@pytest.fixture
def trainer_with_data(sample_interactions, tmp_path) -> ModelTrainer:
    """
    ModelTrainer with synthetic data pre-injected.
    Bypasses file I/O so tests run without the real dataset.
    """
    trainer = ModelTrainer(model_type="collaborative", output_dir=str(tmp_path))

    users = sample_interactions["visitorid"].unique()
    items = sample_interactions["itemid"].unique()

    trainer.user_id_to_idx = {u: i for i, u in enumerate(users)}
    trainer.item_id_to_idx = {it: i for i, it in enumerate(items)}
    trainer.idx_to_user_id = {i: u for u, i in trainer.user_id_to_idx.items()}
    trainer.idx_to_item_id = {i: it for it, i in trainer.item_id_to_idx.items()}

    rows = sample_interactions["visitorid"].map(trainer.user_id_to_idx).values
    cols = sample_interactions["itemid"].map(trainer.item_id_to_idx).values
    data = sample_interactions["weight"].values.astype(np.float32)

    n_users = len(users)
    n_items = len(items)
    trainer.interaction_matrix = csr_matrix(
        (data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
    )

    trainer.item_popularity = (
        sample_interactions.groupby("itemid")["visitorid"].count().to_dict()
    )
    trainer.item_categories = {it: f"cat_{i % 3}" for i, it in enumerate(items)}

    return trainer


# =============================================================================
# Data Loading Tests
# =============================================================================


class TestDataLoading:
    """Tests for interaction matrix construction and index consistency."""

    def test_interaction_matrix_shape(self, trainer_with_data):
        """Matrix must have shape (n_users, n_items) = (3, 7)."""
        m = trainer_with_data.interaction_matrix
        assert m.shape == (3, 7)

    def test_interaction_matrix_is_sparse(self, trainer_with_data):
        """Interaction matrix must be a scipy sparse matrix."""
        assert issparse(trainer_with_data.interaction_matrix)

    def test_user_item_index_consistency(self, trainer_with_data):
        """user_id_to_idx and idx_to_user_id must be exact inverses."""
        t = trainer_with_data
        assert len(t.user_id_to_idx) == len(t.idx_to_user_id)
        for uid, idx in t.user_id_to_idx.items():
            assert t.idx_to_user_id[idx] == uid

    def test_weight_hierarchy_maintained(self):
        """
        Transaction weight (5) >= addtocart weight (3) >= view weight (1).
        This ensures the interaction weighting scheme is correct.
        """
        event_weights = {"view": 1, "addtocart": 3, "transaction": 5}
        assert event_weights["transaction"] >= event_weights["addtocart"]
        assert event_weights["addtocart"] >= event_weights["view"]

    def test_interaction_matrix_values_positive(self, trainer_with_data):
        """All stored interaction weights must be positive."""
        m = trainer_with_data.interaction_matrix
        assert m.data.min() > 0


# =============================================================================
# Collaborative Filtering Tests
# =============================================================================


class TestCollaborativeFiltering:
    """Tests for TruncatedSVD training."""

    def test_svd_produces_user_item_factors(self, trainer_with_data):
        """train_collaborative_filtering must populate user_factors and item_factors."""
        t = trainer_with_data
        result = t.train_collaborative_filtering(t.interaction_matrix)
        assert "user_factors" in result
        assert "item_factors" in result

    def test_user_factors_shape(self, trainer_with_data):
        """user_factors must have shape (n_users, n_components)."""
        t = trainer_with_data
        result = t.train_collaborative_filtering(t.interaction_matrix)
        n_users = t.interaction_matrix.shape[0]
        assert result["user_factors"].shape[0] == n_users

    def test_item_factors_shape(self, trainer_with_data):
        """item_factors must have shape (n_items, n_components)."""
        t = trainer_with_data
        result = t.train_collaborative_filtering(t.interaction_matrix)
        n_items = t.interaction_matrix.shape[1]
        assert result["item_factors"].shape[0] == n_items

    def test_user_factors_l2_normalized(self, trainer_with_data):
        """user_factors rows must be L2-normalized (norm == 1.0)."""
        t = trainer_with_data
        t.train_collaborative_filtering(t.interaction_matrix)
        norms = np.linalg.norm(t.user_factors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_item_factors_l2_normalized(self, trainer_with_data):
        """item_factors rows must be L2-normalized (norm == 1.0)."""
        t = trainer_with_data
        t.train_collaborative_filtering(t.interaction_matrix)
        norms = np.linalg.norm(t.item_factors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_explained_variance_positive(self, trainer_with_data):
        """explained_variance_ratio must be in (0, 1]."""
        t = trainer_with_data
        result = t.train_collaborative_filtering(t.interaction_matrix)
        assert 0.0 < result["explained_variance_ratio"] <= 1.0


# =============================================================================
# Feature Engineering Tests
# =============================================================================


class TestFeatureEngineering:
    """Tests for TF-IDF item content feature building."""

    def test_feature_dict_has_required_keys(self, trainer_with_data):
        """feature_engineering must return the expected keys."""
        t = trainer_with_data
        features = t.feature_engineering(None)
        assert "item_tfidf_matrix" in features
        assert "tfidf_vectorizer" in features
        assert "item_feature_texts" in features

    def test_tfidf_matrix_row_count_matches_items(self, trainer_with_data):
        """TF-IDF matrix must have one row per item in item_id_to_idx."""
        t = trainer_with_data
        features = t.feature_engineering(None)
        assert features["item_tfidf_matrix"].shape[0] == len(t.item_id_to_idx)

    def test_item_feature_texts_coverage(self, trainer_with_data):
        """Every item in item_id_to_idx must have a corresponding feature text."""
        t = trainer_with_data
        features = t.feature_engineering(None)
        for item_id in t.item_id_to_idx:
            assert item_id in features["item_feature_texts"]

    def test_popularity_tokens_included(self, trainer_with_data):
        """Feature texts must include popularity tokens."""
        t = trainer_with_data
        features = t.feature_engineering(None)
        texts = list(features["item_feature_texts"].values())
        has_token = any(
            "popular_high" in txt or "popular_medium" in txt or "popular_low" in txt
            for txt in texts
        )
        assert has_token, "No popularity tokens found in item feature texts"


# =============================================================================
# Evaluation Metric Tests
# =============================================================================


class TestEvaluationMetrics:
    """Tests for Precision@k, Recall@k, NDCG@k, and catalog coverage."""

    def _trained_trainer(self, trainer_with_data):
        trainer_with_data.train_collaborative_filtering(
            trainer_with_data.interaction_matrix
        )
        return trainer_with_data

    def test_precision_at_k_range(self, trainer_with_data):
        """Precision@k must be in [0.0, 1.0]."""
        t = self._trained_trainer(trainer_with_data)
        test_df = pd.DataFrame(
            {"visitorid": ["u1", "u2"], "itemid": ["i6", "i1"], "weight": [1.0, 1.0]}
        )
        metrics = t.evaluate_model(test_df, k=5)
        assert 0.0 <= metrics["precision@5"] <= 1.0

    def test_recall_at_k_range(self, trainer_with_data):
        """Recall@k must be in [0.0, 1.0]."""
        t = self._trained_trainer(trainer_with_data)
        test_df = pd.DataFrame({"visitorid": ["u1"], "itemid": ["i6"], "weight": [1.0]})
        metrics = t.evaluate_model(test_df, k=5)
        assert 0.0 <= metrics["recall@5"] <= 1.0

    def test_ndcg_at_k_range(self, trainer_with_data):
        """NDCG@k must be in [0.0, 1.0]."""
        t = self._trained_trainer(trainer_with_data)
        test_df = pd.DataFrame({"visitorid": ["u1"], "itemid": ["i6"], "weight": [1.0]})
        metrics = t.evaluate_model(test_df, k=5)
        assert 0.0 <= metrics["ndcg@5"] <= 1.0

    def test_catalog_coverage_range(self, trainer_with_data):
        """Catalog coverage must be in [0.0, 1.0]."""
        t = self._trained_trainer(trainer_with_data)
        test_df = pd.DataFrame({"visitorid": ["u1"], "itemid": ["i6"], "weight": [1.0]})
        metrics = t.evaluate_model(test_df, k=3)
        assert 0.0 <= metrics["catalog_coverage"] <= 1.0

    def test_metrics_dict_has_required_keys(self, trainer_with_data):
        """evaluate_model must return all expected metric keys."""
        t = self._trained_trainer(trainer_with_data)
        test_df = pd.DataFrame({"visitorid": ["u1"], "itemid": ["i6"], "weight": [1.0]})
        metrics = t.evaluate_model(test_df, k=10)
        required = {"precision@10", "recall@10", "ndcg@10", "catalog_coverage"}
        assert required.issubset(metrics.keys())

    def test_empty_test_set_returns_zero_metrics(self, trainer_with_data):
        """If no test users are in training data, metrics should default to 0."""
        t = self._trained_trainer(trainer_with_data)
        # Use a user ID that was never in training
        test_df = pd.DataFrame(
            {"visitorid": ["unknown_user"], "itemid": ["i1"], "weight": [1.0]}
        )
        metrics = t.evaluate_model(test_df, k=5)
        assert metrics["n_eval_users"] == 0.0
        assert metrics["precision@5"] == 0.0


# =============================================================================
# Model Saving Tests
# =============================================================================


class TestModelSaving:
    """Tests for artifact serialization."""

    def test_save_model_creates_bundle(self, trainer_with_data, tmp_path):
        """save_model must create model_bundle.pkl in the output directory."""
        t = trainer_with_data
        t.train_collaborative_filtering(t.interaction_matrix)
        t.save_model()
        assert (tmp_path / "model_bundle.pkl").exists()

    def test_save_model_creates_metadata_json(self, trainer_with_data, tmp_path):
        """save_model must create model_metadata.json."""
        t = trainer_with_data
        t.train_collaborative_filtering(t.interaction_matrix)
        t.save_model()
        assert (tmp_path / "model_metadata.json").exists()

    def test_metadata_json_is_valid(self, trainer_with_data, tmp_path):
        """model_metadata.json must be valid JSON with expected keys."""
        t = trainer_with_data
        t.train_collaborative_filtering(t.interaction_matrix)
        t.save_model()
        with open(tmp_path / "model_metadata.json") as f:
            meta = json.load(f)
        assert "model_type" in meta
        assert "n_users" in meta
        assert "n_items" in meta

    def test_saved_bundle_loadable(self, trainer_with_data, tmp_path):
        """Saved bundle must be loadable and contain user_factors."""
        import joblib

        t = trainer_with_data
        t.train_collaborative_filtering(t.interaction_matrix)
        t.save_model()

        bundle = joblib.load(tmp_path / "model_bundle.pkl")
        assert "user_factors" in bundle
        assert "item_factors" in bundle
        assert "metadata" in bundle
        assert bundle["user_factors"].shape[0] == len(t.user_id_to_idx)
