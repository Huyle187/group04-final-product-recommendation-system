"""
Model loading and inference module for the Product Recommendation System.

Loads the trained model bundle (model_bundle.pkl) produced by scripts/train_model.py
and exposes get_recommendations() with collaborative, content-based, and hybrid strategies.

Cold-start handling: unknown users receive popularity-ranked fallback recommendations.
LRU cache: recent (user, n, type) tuples are cached up to RECOMMENDATION_CACHE_SIZE entries.
"""

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings
from app.metrics import (
    BATCH_SIZE,
    MODEL_INFO,
    MODEL_LAST_RELOAD,
    MODEL_LOADED,
    PREDICTION_VALUE,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationModel:
    """Handles model loading and recommendation inference."""

    def __init__(self):
        self.model_version = settings.MODEL_VERSION
        self.model_type = settings.MODEL_TYPE

        # Artifacts populated by load_model()
        self.user_factors: Optional[np.ndarray] = None  # (n_users, n_components)
        self.item_factors: Optional[np.ndarray] = None  # (n_items, n_components)
        self.ease_B: Optional[np.ndarray] = None  # (n_active, n_active) EASE weights
        self.ease_active_indices: Optional[np.ndarray] = None  # active item indices
        self.item_similarity_matrix: Optional[np.ndarray] = None  # (n_items, n_items)
        self.user_id_to_idx: Dict[str, int] = {}
        self.item_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        self.idx_to_item_id: Dict[int, str] = {}
        self.item_popularity: Dict[str, int] = {}
        self.item_categories: Dict[str, str] = {}
        self.interaction_matrix = None  # csr_matrix
        self._model_metadata: Dict[str, Any] = {}

        # LRU recommendation cache
        self.cache: OrderedDict = OrderedDict()

        self.load_model()

    # =========================================================================
    # Model Loading
    # =========================================================================

    def load_model(self) -> None:
        """
        Load the model bundle from MODEL_PATH/model_bundle.pkl.
        Falls back to mock mode (health checks still pass) if file not found.
        """
        model_path = Path(settings.MODEL_PATH) / "model_bundle.pkl"

        if not model_path.exists():
            logger.warning(
                f"Model bundle not found at {model_path}. Running in mock mode. "
                "Train a model first: python scripts/train_model.py --dataset ./data"
            )
            self.model = {
                "type": self.model_type,
                "version": self.model_version,
                "initialized": False,
            }
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)
            return

        try:
            import joblib

            logger.info(f"Loading model bundle from {model_path}")
            bundle = joblib.load(model_path)

            self.user_factors = bundle["user_factors"]
            self.item_factors = bundle["item_factors"]
            self.ease_B = bundle.get("ease_B")  # None for old SVD-only bundles
            self.ease_active_indices = bundle.get("ease_active_indices")
            self.item_similarity_matrix = bundle.get("item_similarity_matrix")
            self.user_id_to_idx = bundle["user_id_to_idx"]
            self.item_id_to_idx = bundle["item_id_to_idx"]
            self.idx_to_user_id = bundle["idx_to_user_id"]
            self.idx_to_item_id = bundle["idx_to_item_id"]
            self.item_popularity = bundle["item_popularity"]
            self.item_categories = bundle["item_categories"]
            self.interaction_matrix = bundle["interaction_matrix"]
            self._model_metadata = bundle.get("metadata", {})

            # Precompute sorted popularity list for cold-start fallback
            self._sorted_popular_items = sorted(
                self.item_popularity.items(), key=lambda x: x[1], reverse=True
            )

            self.model = {
                "type": self._model_metadata.get("model_type", self.model_type),
                "version": self._model_metadata.get("version", self.model_version),
                "initialized": True,
            }
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(1)
            if MODEL_LAST_RELOAD is not None:
                MODEL_LAST_RELOAD.set(time.time())
            logger.info(
                f"Model loaded: {self._model_metadata.get('n_users', '?')} users, "
                f"{self._model_metadata.get('n_items', '?')} items, "
                f"components={self._model_metadata.get('svd_n_components', '?')}"
            )
        except Exception as e:
            logger.error(f"Failed to load model bundle: {e}")
            self.model = {
                "type": self.model_type,
                "version": self.model_version,
                "initialized": False,
            }
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)

    # =========================================================================
    # Public Inference Interface
    # =========================================================================

    def get_recommendations(
        self,
        user_id: str,
        num_recommendations: int = 5,
        recommendation_type: str = "collaborative",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate product recommendations for a user.

        Args:
            user_id: Target user identifier
            num_recommendations: How many recommendations to return
            recommendation_type: 'collaborative', 'content_based', or 'hybrid'
            filters: Optional dict with keys like 'category'

        Returns:
            List of dicts matching ProductRecommendation schema
        """
        # Check LRU cache
        cache_key = f"{user_id}:{num_recommendations}:{recommendation_type}"
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        # Fall back to mock if model not loaded
        if not self.model.get("initialized"):
            return self._mock_recommendations(num_recommendations)

        if recommendation_type == "collaborative":
            recs = self._collaborative_recommendations(
                user_id, num_recommendations, filters
            )
        elif recommendation_type == "content_based":
            recs = self._content_based_recommendations(
                user_id, num_recommendations, filters
            )
        elif recommendation_type == "hybrid":
            recs = self._hybrid_recommendations(user_id, num_recommendations, filters)
        else:
            raise ValueError(f"Unknown recommendation_type: {recommendation_type}")

        # LRU eviction
        if len(self.cache) >= settings.RECOMMENDATION_CACHE_SIZE:
            self.cache.popitem(last=False)
        self.cache[cache_key] = recs

        if PREDICTION_VALUE is not None:
            for rec in recs:
                PREDICTION_VALUE.labels(model_version=self.model_version).observe(
                    rec.get("score", 0.0)
                )

        return recs

    def batch_recommendations(
        self,
        user_ids: List[str],
        num_recommendations: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Generate recommendations for multiple users."""
        return {
            uid: self.get_recommendations(uid, num_recommendations) for uid in user_ids
        }

    # =========================================================================
    # Recommendation Strategies
    # =========================================================================

    def _collaborative_recommendations(
        self, user_id: str, n: int, filters: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """
        EASE scoring when ease_B is available; falls back to SVD dot-product.
        Cold-start users receive popularity-ranked fallback.
        """
        if user_id not in self.user_id_to_idx:
            return self._popular_items_fallback(n, filters)

        u_idx = self.user_id_to_idx[user_id]
        n_items = len(self.item_id_to_idx)
        if self.ease_B is not None and self.ease_active_indices is not None:
            # EASE: project user history to active item subset, score, map back
            user_row = np.array(
                self.interaction_matrix[u_idx].todense(), dtype=np.float32
            ).flatten()
            active_user_row = user_row[self.ease_active_indices]
            active_scores = active_user_row @ self.ease_B  # (n_active,)
            scores = np.full(n_items, -np.inf, dtype=np.float32)
            scores[self.ease_active_indices] = active_scores
        else:
            scores = self.user_factors[u_idx] @ self.item_factors.T  # (n_items,)

        # Mask already-seen items
        seen = self.interaction_matrix[u_idx].nonzero()[1]
        scores[seen] = -np.inf

        # Category filter
        if filters and "category" in filters:
            target_cat = str(filters["category"])
            for item_idx in range(len(scores)):
                if scores[item_idx] == -np.inf:
                    continue
                iid = self.idx_to_item_id[item_idx]
                if self.item_categories.get(iid) != target_cat:
                    scores[item_idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        valid = [i for i in top_indices if scores[i] > -np.inf]

        max_score = float(scores[valid[0]]) if valid else 1.0

        results = []
        for idx in valid:
            item_id = self.idx_to_item_id[idx]
            raw = float(scores[idx])
            norm_score = min(1.0, max(0.0, raw / max_score if max_score > 0 else 0.5))
            results.append(
                self._build_rec(
                    item_id,
                    norm_score,
                    "Users with similar behavior also interacted with this item",
                )
            )

        return results

    def _content_based_recommendations(
        self, user_id: str, n: int, filters: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Average item-item cosine similarity across a user's interaction history.
        Falls back to collaborative (or popularity) if content model unavailable.
        """
        if user_id not in self.user_id_to_idx:
            return self._popular_items_fallback(n, filters)

        if self.item_similarity_matrix is None:
            return self._collaborative_recommendations(user_id, n, filters)

        u_idx = self.user_id_to_idx[user_id]
        seen_indices = self.interaction_matrix[u_idx].nonzero()[1]

        if len(seen_indices) == 0:
            return self._popular_items_fallback(n, filters)

        max_content = self.item_similarity_matrix.shape[0]
        valid_seen = [i for i in seen_indices if i < max_content]

        if not valid_seen:
            return self._popular_items_fallback(n, filters)

        # Mean similarity of each candidate item to the user's seen items
        agg_scores = np.mean(
            self.item_similarity_matrix[valid_seen, :], axis=0
        )  # (max_content,)

        # Mask seen items
        valid_seen_arr = np.array([i for i in seen_indices if i < max_content])
        agg_scores[valid_seen_arr] = -1.0

        # Category filter
        if filters and "category" in filters:
            target_cat = str(filters["category"])
            for item_idx in range(len(agg_scores)):
                if agg_scores[item_idx] <= -1.0:
                    continue
                iid = self.idx_to_item_id.get(item_idx)
                if iid and self.item_categories.get(iid) != target_cat:
                    agg_scores[item_idx] = -1.0

        top_indices = np.argsort(agg_scores)[::-1][:n]

        results = []
        for idx in top_indices:
            if agg_scores[idx] <= -1.0:
                break
            item_id = self.idx_to_item_id.get(idx)
            if item_id is None:
                continue
            raw = float(agg_scores[idx])
            # cosine sim in [0, 1] since TF-IDF vectors are non-negative
            norm_score = min(1.0, max(0.0, raw))
            results.append(
                self._build_rec(
                    item_id,
                    norm_score,
                    "Similar item features to products you've viewed",
                )
            )

        return results if results else self._popular_items_fallback(n, filters)

    def _hybrid_recommendations(
        self, user_id: str, n: int, filters: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Weighted blend of collaborative and content-based scores.
        Over-fetches from each strategy then merges.
        """
        w_c = settings.HYBRID_COLLAB_WEIGHT  # default 0.6
        w_cb = settings.HYBRID_CONTENT_WEIGHT  # default 0.4

        k = min(n * 3, 50)
        collab_recs = self._collaborative_recommendations(user_id, k, filters)
        content_recs = self._content_based_recommendations(user_id, k, filters)

        combined: Dict[str, Dict] = {}
        for rec in collab_recs:
            pid = rec["product_id"]
            combined[pid] = {
                "collab_score": rec["score"],
                "content_score": 0.0,
                "rec": rec,
            }
        for rec in content_recs:
            pid = rec["product_id"]
            if pid in combined:
                combined[pid]["content_score"] = rec["score"]
            else:
                combined[pid] = {
                    "collab_score": 0.0,
                    "content_score": rec["score"],
                    "rec": rec,
                }

        scored = []
        for pid, data in combined.items():
            hybrid_score = w_c * data["collab_score"] + w_cb * data["content_score"]
            rec = dict(data["rec"])
            rec["score"] = min(1.0, max(0.0, hybrid_score))
            rec[
                "reason"
            ] = "Recommended based on your browsing patterns and similar item features"
            scored.append(rec)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n]

    def _popular_items_fallback(
        self, n: int, filters: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Return popularity-ranked items as a cold-start fallback."""
        if not self.item_popularity:
            return self._mock_recommendations(n)

        max_pop = self._sorted_popular_items[0][1] if self._sorted_popular_items else 1
        results = []

        for item_id, pop_count in self._sorted_popular_items:
            if item_id not in self.item_id_to_idx:
                continue

            if filters and "category" in filters:
                if self.item_categories.get(item_id) != str(filters["category"]):
                    continue

            score = float(min(1.0, np.log1p(pop_count) / np.log1p(max_pop)))
            results.append(self._build_rec(item_id, score, "Trending item"))

            if len(results) >= n:
                break

        return results

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_rec(self, item_id: str, score: float, reason: str) -> Dict[str, Any]:
        return {
            "product_id": item_id,
            "product_name": f"Product {item_id}",
            "score": round(float(score), 6),
            "reason": reason,
            "category": self.item_categories.get(item_id),
            "price": None,
        }

    def _mock_recommendations(self, n: int) -> List[Dict[str, Any]]:
        """Return deterministic mock recommendations when model is not loaded."""
        return [
            {
                "product_id": f"prod_{i}",
                "product_name": f"Product {i}",
                "score": round(max(0.0, 0.9 - i * 0.1), 6),
                "reason": "Sample recommendation (model not loaded)",
                "category": "electronics",
                "price": None,
            }
            for i in range(n)
        ]

    def clear_cache(self) -> None:
        self.cache.clear()
        logger.info("Recommendation cache cleared")

    # =========================================================================
    # Model Metadata
    # =========================================================================

    def get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "model_version": self.model_version,
            "model_type": self.model_type,
        }
        if self._model_metadata:
            info.update(
                {
                    "last_updated": self._model_metadata.get("created_at"),
                    "training_users": self._model_metadata.get("n_users"),
                    "training_items": self._model_metadata.get("n_items"),
                    "algorithm": self._model_metadata.get("algorithm", "svd"),
                    "svd_components": self._model_metadata.get("svd_n_components"),
                    "ease_lambda": self._model_metadata.get("ease_lambda"),
                    "evaluation_metrics": self._model_metadata.get("metrics", {}),
                }
            )
        return info


# =============================================================================
# Singleton management
# =============================================================================

_model_instance: Optional[RecommendationModel] = None


def get_model() -> RecommendationModel:
    """Return the global singleton RecommendationModel, creating it if needed."""
    global _model_instance
    if _model_instance is None:
        _model_instance = RecommendationModel()
    return _model_instance


def reload_model() -> None:
    """Reload the model from disk (useful after updating the model bundle)."""
    global _model_instance
    _model_instance = RecommendationModel()
    logger.info("Model reloaded")
