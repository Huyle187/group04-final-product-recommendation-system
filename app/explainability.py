"""
Explainability module for the Product Recommendation System.

Four explanation strategies are supported:

1. collaborative (neighbor-based):
   Finds the top-N most similar users (cosine similarity in latent space)
   and reports how many of them interacted with the recommended item.

2. content_based (feature similarity):
   Shows which items from the user's history are most similar to the
   recommended product via the pre-computed item-item cosine matrix.

3. shap (latent factor attribution):
   For a TruncatedSVD model the score is score = user_vec · item_vec.
   This is linear in each latent dimension, so the exact SHAP decomposition
   is: SHAP_j = user_factors[u,j] * item_factors[i,j].
   The top-k dimensions with highest |SHAP_j| are returned together with the
   three catalog items most aligned with each dimension.
   This is analytically exact (not an approximation) and runs in O(n_components).

4. lime (local linear approximation):
   Perturbs the item's latent vector with Gaussian noise, scores each
   perturbation with the user vector, then fits a weighted Ridge regression.
   The regression coefficients are the LIME importance values per dimension.
   n_samples=50 by default; completes in < 5ms.

Results for (shap, lime) are cached in a per-instance LRU dict
(capacity: 500 entries) to avoid recomputation on repeated requests.
"""

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from app.model import RecommendationModel

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Generates explanations for individual product recommendations.

    Requires a loaded RecommendationModel with user_factors, item_factors,
    item_similarity_matrix, and interaction_matrix populated.
    """

    def __init__(self, model: "RecommendationModel"):
        self.model = model
        # LRU cache for SHAP / LIME results (key: "method:user_id:product_id")
        self._cache: OrderedDict = OrderedDict()
        self._cache_size: int = 500

    # =========================================================================
    # Collaborative Explanation
    # =========================================================================

    def explain_collaborative(
        self,
        user_id: str,
        product_id: str,
        n_neighbors: int = 5,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain a collaborative filtering recommendation.

        When EASE is available: shows which items in the user's history have the
        strongest co-occurrence weight with the recommended product (EASE item-item
        weights), and the EASE score contribution per history item.

        Falls back to SVD neighbor-based explanation when EASE is not available:
        finds similar users and reports how many interacted with the product.

        Args:
            user_id: Target user
            product_id: Recommended product to explain
            n_neighbors: Number of similar users to inspect (SVD fallback only)

        Returns:
            Dict with method, supporting evidence, confidence, explanation_text
        """
        m = self.model

        if m.user_factors is None or m.item_factors is None:
            return {"method": "collaborative_neighbors", "error": "Model not loaded"}

        if user_id not in m.user_id_to_idx:
            return {
                "method": "collaborative_neighbors",
                "user_id": user_id,
                "product_id": product_id,
                "error": "User not found in training data (cold-start user)",
                "explanation_text": (
                    "This is a new user. Recommendations are based on item popularity."
                ),
            }

        if product_id not in m.item_id_to_idx:
            return {
                "method": "collaborative_neighbors",
                "user_id": user_id,
                "product_id": product_id,
                "error": "Product not found in training data",
            }

        u_idx = m.user_id_to_idx[user_id]
        item_idx = m.item_id_to_idx[product_id]

        # --- EASE-based explanation (primary path) ---
        if m.ease_B is not None and m.ease_active_indices is not None:
            # ease_active_indices maps positions in ease_B to global item indices
            active_set = set(m.ease_active_indices.tolist())

            if item_idx not in active_set:
                return {
                    "method": "ease_cooccurrence",
                    "user_id": user_id,
                    "product_id": product_id,
                    "error": "Product is outside the EASE active item set",
                }

            # Position of the target item within ease_active_indices
            active_list = m.ease_active_indices.tolist()
            active_pos_map = {
                global_idx: pos for pos, global_idx in enumerate(active_list)
            }
            target_active_pos = active_pos_map[item_idx]

            # Items in the user's history that are also in the EASE active set
            seen_global = m.interaction_matrix[u_idx].nonzero()[1]
            seen_active_positions = [
                active_pos_map[gi] for gi in seen_global if gi in active_pos_map
            ]

            if not seen_active_positions:
                return {
                    "method": "ease_cooccurrence",
                    "user_id": user_id,
                    "product_id": product_id,
                    "explanation_text": (
                        "None of the user's history items overlap with the EASE "
                        "active item set. Recommendation may use popularity fallback."
                    ),
                }

            # EASE weight from each history item to the target item
            # ease_B[history_pos, target_pos] is the item-item co-occurrence weight
            weights = m.ease_B[
                seen_active_positions, target_active_pos
            ]  # (n_seen_active,)

            # Sort by absolute weight (strongest contributors first)
            top_k = min(top_k, len(seen_active_positions))
            top_pos = np.argsort(np.abs(weights))[::-1][:top_k]

            supporting_items = []
            total_score = 0.0
            for pos in top_pos:
                global_idx = active_list[seen_active_positions[pos]]
                hist_item_id = m.idx_to_item_id.get(global_idx)
                w = float(weights[pos])
                total_score += w
                if hist_item_id:
                    supporting_items.append(
                        {
                            "item_id": hist_item_id,
                            "ease_weight": round(w, 6),
                            "category": m.item_categories.get(hist_item_id),
                        }
                    )

            confidence = min(1.0, max(0.0, total_score)) if total_score > 0 else 0.0

            return {
                "method": "ease_cooccurrence",
                "user_id": user_id,
                "product_id": product_id,
                "history_items_checked": len(seen_active_positions),
                "top_contributing_history_items": supporting_items,
                "ease_score_contribution": round(total_score, 6),
                "confidence": round(confidence, 4),
                "explanation_text": (
                    f"Recommended because {len(supporting_items)} item(s) in your "
                    f"browsing history co-occur strongly with this product "
                    f"(EASE score contribution: {total_score:.4f})."
                ),
            }

        # --- SVD neighbor-based explanation (fallback when EASE not available) ---
        user_vec = m.user_factors[u_idx]  # (n_components,)
        all_sims = m.user_factors @ user_vec  # (n_users,)
        all_sims[u_idx] = -1.0  # exclude self

        top_neighbor_indices = np.argsort(all_sims)[::-1][:n_neighbors]

        # Which neighbors interacted with the target item?
        neighbors_with_item: List[int] = []
        for n_idx in top_neighbor_indices:
            if m.interaction_matrix[n_idx, item_idx] > 0:
                neighbors_with_item.append(int(n_idx))

        # Collect supporting items from those neighbors (items they also bought)
        supporting_items_set: set = set()
        for n_idx in neighbors_with_item:
            neighbor_item_indices = m.interaction_matrix[n_idx].nonzero()[1]
            for ni in neighbor_item_indices[:5]:
                ni_id = m.idx_to_item_id.get(ni)
                if ni_id and ni_id != product_id:
                    supporting_items_set.add(ni_id)

        confidence = len(neighbors_with_item) / n_neighbors if n_neighbors > 0 else 0.0

        return {
            "method": "collaborative_neighbors",
            "user_id": user_id,
            "product_id": product_id,
            "similar_user_count": len(neighbors_with_item),
            "top_n_neighbors_checked": n_neighbors,
            "confidence": round(confidence, 3),
            "supporting_items": list(supporting_items_set)[:5],
            "explanation_text": (
                f"{len(neighbors_with_item)} out of {n_neighbors} users with "
                f"similar browsing history also interacted with this item."
            ),
        }

    # =========================================================================
    # Content-Based Explanation
    # =========================================================================

    def explain_content_based(
        self,
        user_id: str,
        product_id: str,
    ) -> Dict[str, Any]:
        """
        Explain a content-based recommendation via item feature similarity.

        Finds the top-3 items from the user's interaction history that are
        most similar to the recommended product, and reports whether they
        share the same product category.

        Args:
            user_id: Target user
            product_id: Recommended product to explain

        Returns:
            Dict with method, most_similar_seen_items, category_match,
            average_similarity_to_history, explanation_text
        """
        m = self.model

        if m.item_similarity_matrix is None:
            return {
                "method": "content_similarity",
                "error": "Content model not available",
            }

        if user_id not in m.user_id_to_idx:
            return {
                "method": "content_similarity",
                "user_id": user_id,
                "product_id": product_id,
                "error": "User not found in training data",
            }

        if product_id not in m.item_id_to_idx:
            return {
                "method": "content_similarity",
                "user_id": user_id,
                "product_id": product_id,
                "error": "Product not found in training data",
            }

        u_idx = m.user_id_to_idx[user_id]
        item_idx = m.item_id_to_idx[product_id]
        max_content = m.item_similarity_matrix.shape[0]

        if item_idx >= max_content:
            return {
                "method": "content_similarity",
                "user_id": user_id,
                "product_id": product_id,
                "error": "Product is outside the content model's item range",
            }

        seen_indices = m.interaction_matrix[u_idx].nonzero()[1]
        valid_seen = [i for i in seen_indices if i < max_content]

        if not valid_seen:
            return {
                "method": "content_similarity",
                "user_id": user_id,
                "product_id": product_id,
                "error": "No overlapping items between user history and content model",
            }

        # Similarity of target item to each of the user's seen items
        sims_to_target = m.item_similarity_matrix[valid_seen, item_idx]  # (n_valid,)

        # Top-3 most similar seen items as evidence
        top_k = min(3, len(valid_seen))
        top_seen_pos = np.argsort(sims_to_target)[::-1][:top_k]

        most_similar_seen: List[Dict[str, Any]] = []
        for pos in top_seen_pos:
            seen_item_id = m.idx_to_item_id.get(valid_seen[pos])
            if seen_item_id:
                most_similar_seen.append(
                    {
                        "item_id": seen_item_id,
                        "similarity": round(float(sims_to_target[pos]), 4),
                        "category": m.item_categories.get(seen_item_id),
                    }
                )

        # Category match check
        product_category = m.item_categories.get(product_id)
        user_categories = {
            m.item_categories.get(m.idx_to_item_id.get(i)) for i in seen_indices
        }
        category_match = (
            product_category in user_categories if product_category else False
        )

        avg_sim = float(np.mean(sims_to_target)) if len(sims_to_target) > 0 else 0.0

        return {
            "method": "content_similarity",
            "user_id": user_id,
            "product_id": product_id,
            "most_similar_seen_items": most_similar_seen,
            "product_category": product_category,
            "category_match": category_match,
            "average_similarity_to_history": round(avg_sim, 4),
            "explanation_text": (
                f"This item is similar to {top_k} product(s) you've previously viewed"
                + (
                    ", and shares the same product category."
                    if category_match
                    else " based on shared properties."
                )
            ),
        }

    # =========================================================================
    # SHAP — Latent Factor Attribution
    # =========================================================================

    def explain_shap(
        self,
        user_id: str,
        product_id: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Exact SHAP decomposition for SVD dot-product scoring.

        For the linear scoring function score = user_vec · item_vec, the
        Shapley value for dimension j is analytically:
            SHAP_j = user_factors[u, j] * item_factors[i, j]
        and sum(SHAP) == score (efficiency axiom satisfied exactly).

        For each of the top-k |SHAP_j| dimensions, the three catalog items
        whose item_factors are most aligned with that dimension are returned
        as human-readable evidence.

        Args:
            user_id: Target user
            product_id: Recommended product
            top_k: Number of top latent dimensions to report

        Returns:
            Dict with method, total_score, top_dimensions, explanation_text
        """
        cache_key = f"shap:{user_id}:{product_id}:{top_k}"
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        m = self.model
        if m.user_factors is None or m.item_factors is None:
            return {"method": "shap_latent", "error": "Model not loaded"}

        if user_id not in m.user_id_to_idx:
            return {
                "method": "shap_latent",
                "user_id": user_id,
                "product_id": product_id,
                "error": "User not found in training data (cold-start user)",
            }
        if product_id not in m.item_id_to_idx:
            return {
                "method": "shap_latent",
                "user_id": user_id,
                "product_id": product_id,
                "error": "Product not found in training data",
            }

        u_idx = m.user_id_to_idx[user_id]
        i_idx = m.item_id_to_idx[product_id]

        user_vec = m.user_factors[u_idx]  # (n_components,)
        item_vec = m.item_factors[i_idx]  # (n_components,)

        # Exact SHAP: component-wise product
        shap_values = user_vec * item_vec  # (n_components,)
        total_score = float(shap_values.sum())

        # Top-k dimensions by absolute SHAP value
        top_dims = np.argsort(np.abs(shap_values))[::-1][:top_k]

        # For each top dimension, find the 3 catalog items most aligned
        dim_items: List[Dict[str, Any]] = []
        for dim in top_dims:
            aligned_indices = np.argsort(m.item_factors[:, dim])[::-1][:3]
            rep_items = [
                m.idx_to_item_id[int(idx)]
                for idx in aligned_indices
                if int(idx) in m.idx_to_item_id
            ]
            dim_items.append(
                {
                    "dimension": int(dim),
                    "shap_value": round(float(shap_values[dim]), 6),
                    "abs_shap": round(float(abs(shap_values[dim])), 6),
                    "direction": "positive" if shap_values[dim] >= 0 else "negative",
                    "representative_items": rep_items,
                }
            )

        result = {
            "method": "shap_latent",
            "user_id": user_id,
            "product_id": product_id,
            "total_score": round(total_score, 6),
            "n_components": int(len(shap_values)),
            "top_dimensions": dim_items,
            "explanation_text": (
                f"The recommendation score ({total_score:.4f}) is driven by "
                f"{top_k} latent dimensions. The most influential dimension "
                f"contributes {dim_items[0]['shap_value']:.4f} to the score."
                if dim_items
                else "SHAP attribution not available."
            ),
        }

        # Cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = result
        return result

    # =========================================================================
    # LIME — Local Linear Approximation
    # =========================================================================

    def explain_lime(
        self,
        user_id: str,
        product_id: str,
        n_samples: int = 50,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        LIME explanation via local linear approximation in item latent space.

        Perturbs the item's latent vector with Gaussian noise, scores each
        perturbation using the user vector, then fits a weighted Ridge regression.
        The kernel function assigns higher weight to perturbations close to the
        original item vector (exponential kernel with bandwidth 0.5).

        The Ridge coefficients are the LIME importance values: positive means
        increasing that latent dimension increases the predicted score, negative
        means decreasing it.

        Args:
            user_id: Target user
            product_id: Recommended product
            n_samples: Number of perturbation samples (default 50, ~2ms)
            top_k: Number of top dimensions to report

        Returns:
            Dict with method, local_fidelity, feature_importances, explanation_text
        """
        cache_key = f"lime:{user_id}:{product_id}:{n_samples}"
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        m = self.model
        if m.user_factors is None or m.item_factors is None:
            return {"method": "lime_latent", "error": "Model not loaded"}

        if user_id not in m.user_id_to_idx:
            return {
                "method": "lime_latent",
                "user_id": user_id,
                "product_id": product_id,
                "error": "User not found in training data (cold-start user)",
            }
        if product_id not in m.item_id_to_idx:
            return {
                "method": "lime_latent",
                "user_id": user_id,
                "product_id": product_id,
                "error": "Product not found in training data",
            }

        from sklearn.linear_model import Ridge

        u_idx = m.user_id_to_idx[user_id]
        i_idx = m.item_id_to_idx[product_id]

        user_vec = m.user_factors[u_idx]  # (n_components,)
        item_vec = m.item_factors[i_idx]  # (n_components,)
        n_components = len(item_vec)

        # Perturb item vector with Gaussian noise (σ=0.1)
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, 0.1, (n_samples, n_components))
        perturbed = item_vec + noise  # (n_samples, n_components)
        scores = perturbed @ user_vec  # (n_samples,)

        # Kernel weights: closer perturbations get higher weight
        distances = np.linalg.norm(noise, axis=1)
        kernel_bandwidth = 0.5
        weights = np.exp(-(distances**2) / kernel_bandwidth)

        # Fit local linear model
        ridge = Ridge(alpha=0.01)
        ridge.fit(perturbed, scores, sample_weight=weights)
        local_fidelity = float(ridge.score(perturbed, scores, sample_weight=weights))

        # Top-k dimensions by absolute coefficient
        coefs = ridge.coef_
        top_dims = np.argsort(np.abs(coefs))[::-1][:top_k]

        feature_importances = [
            {
                "dimension": int(d),
                "importance": round(float(coefs[d]), 6),
                "direction": "positive" if coefs[d] >= 0 else "negative",
            }
            for d in top_dims
        ]

        result = {
            "method": "lime_latent",
            "user_id": user_id,
            "product_id": product_id,
            "n_samples": n_samples,
            "local_fidelity": round(local_fidelity, 4),
            "feature_importances": feature_importances,
            "explanation_text": (
                f"Local linear approximation (R²={local_fidelity:.3f}) identifies "
                f"{top_k} most influential latent dimensions. "
                f"Top dimension ({feature_importances[0]['dimension']}) has "
                f"importance {feature_importances[0]['importance']:.4f}."
                if feature_importances
                else "LIME attribution not available."
            ),
        }

        # Cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = result
        return result

    # =========================================================================
    # Unified Entry Point
    # =========================================================================

    def explain(
        self,
        user_id: str,
        product_id: str,
        method: str = "auto",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate an explanation for why product_id was recommended to user_id.

        Args:
            user_id: Target user
            product_id: Recommended product
            method: 'auto' | 'collaborative' | 'content_based' | 'shap' | 'lime' | 'all'
                    - auto: shap if model loaded, otherwise collaborative
                    - all: returns dict with results from all four methods

        Returns:
            Explanation dict (or nested dict for method='all')
        """
        m = self.model

        if method == "shap":
            return self.explain_shap(user_id, product_id, top_k=top_k)

        if method == "lime":
            return self.explain_lime(user_id, product_id, top_k=top_k)

        if method == "all":
            return {
                "collaborative": self.explain_collaborative(
                    user_id, product_id, top_k=top_k
                ),
                "content_based": self.explain_content_based(user_id, product_id),
                "shap": self.explain_shap(user_id, product_id, top_k=top_k),
                "lime": self.explain_lime(user_id, product_id, top_k=top_k),
            }

        if method == "content_based":
            return self.explain_content_based(user_id, product_id)

        # "collaborative" or "auto"
        if m.user_factors is not None:
            if method == "auto":
                # Default to SHAP when model is loaded — more informative
                return self.explain_shap(user_id, product_id, top_k=top_k)
            return self.explain_collaborative(user_id, product_id, top_k=top_k)

        return self.explain_content_based(user_id, product_id)


# =============================================================================
# Module-level singleton
# =============================================================================

_engine_instance: Optional[ExplainabilityEngine] = None


def get_explainability_engine() -> ExplainabilityEngine:
    """Return the global singleton ExplainabilityEngine (lazy-initialized)."""
    global _engine_instance
    if _engine_instance is None:
        from app.model import get_model

        _engine_instance = ExplainabilityEngine(get_model())
    return _engine_instance
