"""
Explainability module for the Product Recommendation System.

Provides human-interpretable explanations for why a specific product was
recommended to a user. Two strategies are supported:

- collaborative: Neighbor-based explanation using the trained latent space.
  Finds the top-N most similar users and reports how many of them also
  interacted with the recommended item, along with shared supporting items.

- content_based: Feature similarity explanation.
  Shows which items from the user's history are most similar to the
  recommended product, and whether they share the same category.

Note on SHAP: Standard SHAP (TreeExplainer, DeepExplainer) does not apply
directly to TruncatedSVD matrix factorization. KernelSHAP would require
hundreds of model evaluations per request, making it impractical for a
real-time API. The neighbor-based approach used here provides equivalent
conceptual explanations ("users like you also bought this") with O(1) latency
by leveraging the pre-computed L2-normalized factor matrices.
"""

import logging
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

    # =========================================================================
    # Collaborative Explanation
    # =========================================================================

    def explain_collaborative(
        self,
        user_id: str,
        product_id: str,
        n_neighbors: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain a collaborative filtering recommendation via nearest-neighbor evidence.

        Finds the top-N most similar users (cosine similarity in the latent space),
        reports how many interacted with the recommended product, and lists
        supporting items those neighbors also interacted with.

        Args:
            user_id: Target user
            product_id: Recommended product to explain
            n_neighbors: Number of similar users to inspect

        Returns:
            Dict with method, similar_user_count, confidence, supporting_items,
            explanation_text
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

        # Cosine similarity to all users (factors are L2-normalized)
        user_vec = m.user_factors[u_idx]             # (n_components,)
        all_sims = m.user_factors @ user_vec          # (n_users,)
        all_sims[u_idx] = -1.0                        # exclude self

        top_neighbor_indices = np.argsort(all_sims)[::-1][:n_neighbors]

        # Which neighbors interacted with the target item?
        neighbors_with_item: List[int] = []
        for n_idx in top_neighbor_indices:
            if m.interaction_matrix[n_idx, item_idx] > 0:
                neighbors_with_item.append(int(n_idx))

        # Collect supporting items from those neighbors (items they also bought)
        supporting_items: set = set()
        for n_idx in neighbors_with_item:
            neighbor_item_indices = m.interaction_matrix[n_idx].nonzero()[1]
            for ni in neighbor_item_indices[:5]:
                ni_id = m.idx_to_item_id.get(ni)
                if ni_id and ni_id != product_id:
                    supporting_items.add(ni_id)

        confidence = len(neighbors_with_item) / n_neighbors if n_neighbors > 0 else 0.0

        return {
            "method": "collaborative_neighbors",
            "user_id": user_id,
            "product_id": product_id,
            "similar_user_count": len(neighbors_with_item),
            "top_n_neighbors_checked": n_neighbors,
            "confidence": round(confidence, 3),
            "supporting_items": list(supporting_items)[:5],
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
            return {"method": "content_similarity", "error": "Content model not available"}

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
                most_similar_seen.append({
                    "item_id": seen_item_id,
                    "similarity": round(float(sims_to_target[pos]), 4),
                    "category": m.item_categories.get(seen_item_id),
                })

        # Category match check
        product_category = m.item_categories.get(product_id)
        user_categories = {
            m.item_categories.get(m.idx_to_item_id.get(i))
            for i in seen_indices
        }
        category_match = product_category in user_categories if product_category else False

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
    # Unified Entry Point
    # =========================================================================

    def explain(
        self,
        user_id: str,
        product_id: str,
        method: str = "auto",
    ) -> Dict[str, Any]:
        """
        Generate an explanation for why product_id was recommended to user_id.

        Args:
            user_id: Target user
            product_id: Recommended product
            method: 'auto' | 'collaborative' | 'content_based'
                    'auto' uses collaborative if model is loaded, else content

        Returns:
            Explanation dict
        """
        m = self.model

        if method == "content_based":
            return self.explain_content_based(user_id, product_id)

        if method == "collaborative" or method == "auto":
            if m.user_factors is not None:
                return self.explain_collaborative(user_id, product_id)
            return self.explain_content_based(user_id, product_id)

        return {"error": f"Unknown explanation method: {method}"}


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
