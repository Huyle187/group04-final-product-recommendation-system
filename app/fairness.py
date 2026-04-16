"""
Fairness analysis module for the Product Recommendation System.

Assesses recommendation fairness along three measurable axes:

1. Popularity bias — Are recommendations dominated by already-popular items?
   (Filter bubble / rich-get-richer effect)
   Metric: fraction of recommended items above the 80th popularity percentile,
   plus Gini coefficient of the popularity distribution in recommendations.

2. Category diversity — Does a single user's recommendation list span multiple
   product categories, or is it concentrated in one?
   Metric: Normalized Shannon entropy of the category distribution.

3. Catalog coverage — Across all users, what fraction of the product catalog
   is ever recommended? Long-tail items should occasionally surface.
   Metric: unique recommended items / total catalog size, plus top-10%
   item concentration ratio.

Note: The Retail Rocket dataset contains no demographic attributes (age, gender,
location), so classical protected-group fairness metrics (demographic parity,
equal opportunity) cannot be computed. The proxy metrics above capture the
commercially important forms of algorithmic bias in recommender systems.
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from app.config import settings

if TYPE_CHECKING:
    from app.model import RecommendationModel

logger = logging.getLogger(__name__)


class FairnessChecker:
    """
    Computes fairness metrics for recommendation outputs.

    Requires a loaded RecommendationModel with item_popularity and
    item_categories populated.
    """

    def __init__(self, model: "RecommendationModel"):
        self.model = model
        self._popularity_thresholds = self._compute_popularity_thresholds()

    # =========================================================================
    # Initialization helpers
    # =========================================================================

    def _compute_popularity_thresholds(self) -> Dict[str, float]:
        """Precompute percentile thresholds from training popularity counts."""
        if not self.model.item_popularity:
            return {}
        counts = np.array(list(self.model.item_popularity.values()), dtype=float)
        return {
            "p50": float(np.percentile(counts, 50)),
            "p80": float(np.percentile(counts, 80)),
            "p95": float(np.percentile(counts, 95)),
        }

    # =========================================================================
    # Popularity Bias
    # =========================================================================

    def check_popularity_bias(self, recommended_item_ids: List[str]) -> Dict[str, Any]:
        """
        Measure the fraction of recommendations that are "popular" items
        (above the 80th popularity percentile in the training set).

        Also computes the Gini coefficient of popularity scores in the list —
        0 = perfectly equal popularity, 1 = maximally concentrated.

        Flags bias if popular_items_fraction > FAIRNESS_POPULARITY_THRESHOLD (0.3).
        """
        if not recommended_item_ids:
            return {"error": "Empty recommendation list"}
        if not self._popularity_thresholds:
            return {"error": "Popularity data not available"}

        threshold_count = self._popularity_thresholds.get("p80", 0.0)
        popular_count = sum(
            1
            for iid in recommended_item_ids
            if self.model.item_popularity.get(iid, 0) >= threshold_count
        )

        popularity_fraction = popular_count / len(recommended_item_ids)
        is_biased = popularity_fraction > settings.FAIRNESS_POPULARITY_THRESHOLD

        pop_scores = np.array(
            [
                float(self.model.item_popularity.get(iid, 0))
                for iid in recommended_item_ids
            ]
        )
        gini = self._gini_coefficient(pop_scores)

        return {
            "popularity_bias_detected": is_biased,
            "popular_items_fraction": round(popularity_fraction, 4),
            "threshold_used": settings.FAIRNESS_POPULARITY_THRESHOLD,
            "p80_popularity_count": round(threshold_count, 1),
            "gini_coefficient": round(gini, 4),
            "n_recommendations": len(recommended_item_ids),
            "interpretation": (
                "High popularity bias: recommendations over-represent popular items."
                if is_biased
                else "Acceptable popularity distribution."
            ),
        }

    # =========================================================================
    # Category Diversity
    # =========================================================================

    def check_category_diversity(
        self, recommended_item_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Measure category diversity using normalized Shannon entropy.

        H_normalized = -sum(p_i * log(p_i)) / log(n_categories)
        where p_i = fraction of recommendations from category i.

        Score of 1.0 means perfectly uniform across all represented categories.
        Score of 0.0 means all items are from one category.

        Flags low diversity if score < MIN_CATEGORY_DIVERSITY (0.3).
        """
        if not recommended_item_ids:
            return {"error": "Empty recommendation list"}

        categories = [
            self.model.item_categories.get(iid, "unknown")
            for iid in recommended_item_ids
        ]
        cat_counts = Counter(categories)
        n_items = len(categories)
        n_cats = len(cat_counts)

        if n_cats <= 1:
            entropy_normalized = 0.0
        else:
            probs = np.array(
                [count / n_items for count in cat_counts.values()], dtype=float
            )
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            max_entropy = np.log(n_cats)
            entropy_normalized = (
                float(entropy / max_entropy) if max_entropy > 0 else 0.0
            )

        is_diverse = entropy_normalized >= settings.MIN_CATEGORY_DIVERSITY

        return {
            "category_diversity_score": round(entropy_normalized, 4),
            "n_unique_categories": n_cats,
            "category_distribution": dict(cat_counts.most_common(10)),
            "meets_diversity_threshold": is_diverse,
            "threshold": settings.MIN_CATEGORY_DIVERSITY,
            "interpretation": (
                "Good category diversity."
                if is_diverse
                else "Low category diversity: recommendations concentrated in few categories."
            ),
        }

    # =========================================================================
    # Catalog Coverage
    # =========================================================================

    def check_catalog_coverage(
        self, all_recommendation_lists: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Measure system-wide fairness: what fraction of the catalog is reachable
        via recommendations across all users?

        Also reports the concentration ratio: what fraction of all recommendation
        instances go to the top 10% most-recommended items.

        Args:
            all_recommendation_lists: One list of product_ids per user

        Returns:
            Dict with catalog_coverage, unique_items_recommended,
            top_10pct_items_concentration, interpretation
        """
        n_total = len(self.model.item_id_to_idx)
        if n_total == 0:
            return {"error": "Empty catalog"}

        all_recommended: set = set()
        rec_counter: Counter = Counter()

        for rec_list in all_recommendation_lists:
            all_recommended.update(rec_list)
            rec_counter.update(rec_list)

        coverage = len(all_recommended) / n_total

        total_recs = sum(rec_counter.values())
        if total_recs == 0:
            return {"catalog_coverage": 0.0, "error": "No recommendations to analyze"}

        sorted_items = rec_counter.most_common()
        n_top = max(1, int(0.1 * len(sorted_items)))
        top_concentration = sum(count for _, count in sorted_items[:n_top]) / total_recs

        return {
            "catalog_coverage": round(coverage, 4),
            "unique_items_recommended": len(all_recommended),
            "total_catalog_items": n_total,
            "top_10pct_items_concentration": round(top_concentration, 4),
            "interpretation": (
                f"Top 10% of recommended items account for "
                f"{top_concentration * 100:.1f}% of all recommendations. "
                + (
                    "High concentration detected — long-tail items are underserved."
                    if top_concentration > 0.5
                    else "Acceptable distribution across the catalog."
                )
            ),
        }

    # =========================================================================
    # Unified Report
    # =========================================================================

    def full_fairness_report(
        self, user_id: str, recommended_item_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compute all per-user fairness metrics and return a structured report
        suitable for the /fairness/check API endpoint.

        Args:
            user_id: Target user (included in report for context)
            recommended_item_ids: Product IDs returned for this user

        Returns:
            Dict with overall_fairness_score, popularity_bias, category_diversity
        """
        popularity_result = self.check_popularity_bias(recommended_item_ids)
        diversity_result = self.check_category_diversity(recommended_item_ids)

        # Overall score: average of (1 - popular_fraction) and diversity_score
        pop_score = 1.0 - popularity_result.get("popular_items_fraction", 0.5)
        div_score = diversity_result.get("category_diversity_score", 0.0)
        overall_score = round((pop_score + div_score) / 2.0, 4)

        return {
            "user_id": user_id,
            "overall_fairness_score": overall_score,
            "popularity_bias": popularity_result,
            "category_diversity": diversity_result,
            "n_recommendations_analyzed": len(recommended_item_ids),
            "fairness_interpretation": (
                "Good overall fairness."
                if overall_score >= 0.5
                else "Fairness concerns detected — consider diversification."
            ),
            "status": "ok",
        }

    # =========================================================================
    # Statistical Utilities
    # =========================================================================

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """
        Compute the Gini coefficient of an array of non-negative values.
        0 = perfect equality, 1 = maximum inequality.
        """
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        values = np.sort(values.astype(float))
        n = len(values)
        cumulative = np.cumsum(values)
        return float(
            (2 * np.sum(cumulative) - (n + 1) * values.sum()) / (n * values.sum())
        )


# =============================================================================
# Module-level singleton
# =============================================================================

_checker_instance: Optional[FairnessChecker] = None


def get_fairness_checker() -> FairnessChecker:
    """Return the global singleton FairnessChecker (lazy-initialized)."""
    global _checker_instance
    if _checker_instance is None:
        from app.model import get_model

        _checker_instance = FairnessChecker(get_model())
    return _checker_instance
