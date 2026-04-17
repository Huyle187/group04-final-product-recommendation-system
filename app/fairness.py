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
    # User-Segment Fairness
    # =========================================================================

    def check_user_segment_fairness(
        self,
        user_ids: List[str],
        recs_per_user: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Measure whether recommendation quality is equitable across user activity
        segments (power / mid / casual), using interaction count as a proxy for
        engagement level.

        Segments:
          - power:  top 20% by training interaction count
          - casual: bottom 20%
          - mid:    remaining 60%

        For each segment, reports:
          avg_score (mean recommendation score), avg_diversity (category entropy),
          popular_fraction (popularity bias), n_users.

        Flags unfairness when the metric gap between 'power' and 'casual' exceeds
        FAIRNESS_SEGMENT_GAP_THRESHOLD (default 0.2).

        Args:
            user_ids: Users to evaluate
            recs_per_user: {user_id: [product_id, ...]} — pre-generated recommendations

        Returns:
            Dict with per-segment metrics and fairness_gap flags
        """
        if not self.model.interaction_matrix is not None or not user_ids:
            return {"error": "Interaction matrix not available or no users provided"}

        # Compute per-user interaction counts from the training matrix
        user_counts: Dict[str, int] = {}
        for uid in user_ids:
            idx = self.model.user_id_to_idx.get(uid)
            if idx is not None:
                user_counts[uid] = int(self.model.interaction_matrix[idx].nnz)

        if not user_counts:
            return {"error": "None of the provided users found in training data"}

        counts_arr = np.array(list(user_counts.values()))
        p20 = float(np.percentile(counts_arr, 20))
        p80 = float(np.percentile(counts_arr, 80))

        segments: Dict[str, List[str]] = {"power": [], "mid": [], "casual": []}
        for uid, cnt in user_counts.items():
            if cnt >= p80:
                segments["power"].append(uid)
            elif cnt <= p20:
                segments["casual"].append(uid)
            else:
                segments["mid"].append(uid)

        segment_metrics: Dict[str, Any] = {}
        for seg_name, seg_users in segments.items():
            if not seg_users:
                segment_metrics[seg_name] = {"n_users": 0}
                continue

            pop_fractions, div_scores = [], []
            for uid in seg_users:
                rec_ids = recs_per_user.get(uid, [])
                if not rec_ids:
                    continue
                pop_result = self.check_popularity_bias(rec_ids)
                div_result = self.check_category_diversity(rec_ids)
                pop_fractions.append(pop_result.get("popular_items_fraction", 0.0))
                div_scores.append(div_result.get("category_diversity_score", 0.0))

            segment_metrics[seg_name] = {
                "n_users": len(seg_users),
                "avg_popular_fraction": round(
                    float(np.mean(pop_fractions)) if pop_fractions else 0.0, 4
                ),
                "avg_diversity_score": round(
                    float(np.mean(div_scores)) if div_scores else 0.0, 4
                ),
            }

        # Fairness gap: difference between power and casual segments
        power_div = segment_metrics.get("power", {}).get("avg_diversity_score", 0.0)
        casual_div = segment_metrics.get("casual", {}).get("avg_diversity_score", 0.0)
        diversity_gap = round(abs(power_div - casual_div), 4)

        power_pop = segment_metrics.get("power", {}).get("avg_popular_fraction", 0.0)
        casual_pop = segment_metrics.get("casual", {}).get("avg_popular_fraction", 0.0)
        popularity_gap = round(abs(power_pop - casual_pop), 4)

        threshold = settings.FAIRNESS_SEGMENT_GAP_THRESHOLD
        unfair = diversity_gap > threshold or popularity_gap > threshold

        return {
            "segments": segment_metrics,
            "diversity_gap_power_vs_casual": diversity_gap,
            "popularity_gap_power_vs_casual": popularity_gap,
            "segment_fairness_threshold": threshold,
            "unfairness_detected": unfair,
            "interpretation": (
                "Significant disparity between power and casual users detected."
                if unfair
                else "Recommendation quality is relatively equitable across user segments."
            ),
        }

    # =========================================================================
    # Long-Tail Item Exposure
    # =========================================================================

    def check_long_tail_exposure(
        self,
        all_recommendation_lists: List[List[str]],
    ) -> Dict[str, Any]:
        """
        Measure exposure across item popularity tiers: head / torso / tail.

        Tiers (by item popularity percentile):
          - head:  top 20%  (P80 and above)
          - torso: middle 60% (P20–P80)
          - tail:  bottom 20% (below P20)

        For each tier, exposure = (items in tier that appear in any rec list)
                                  / (total items in that tier in the catalog).

        Flags tail suppression if tail_exposure < FAIRNESS_LONG_TAIL_THRESHOLD (0.2).

        Args:
            all_recommendation_lists: One list of product_ids per user

        Returns:
            Dict with per-tier exposure ratios and tail_suppression_detected flag
        """
        if not self._popularity_thresholds:
            return {"error": "Popularity data not available"}

        p20 = float(np.percentile(list(self.model.item_popularity.values()), 20))
        p80 = self._popularity_thresholds.get("p80", 0.0)

        tiers: Dict[str, set] = {"head": set(), "torso": set(), "tail": set()}
        for iid, cnt in self.model.item_popularity.items():
            if iid not in self.model.item_id_to_idx:
                continue
            if cnt >= p80:
                tiers["head"].add(iid)
            elif cnt <= p20:
                tiers["tail"].add(iid)
            else:
                tiers["torso"].add(iid)

        # Flatten all recommended items
        all_recommended: set = set()
        for rec_list in all_recommendation_lists:
            all_recommended.update(rec_list)

        tier_exposure: Dict[str, float] = {}
        tier_counts: Dict[str, int] = {}
        for tier_name, tier_items in tiers.items():
            exposed = len(all_recommended & tier_items)
            total = len(tier_items)
            tier_exposure[f"{tier_name}_exposure"] = round(
                exposed / total if total else 0.0, 4
            )
            tier_counts[f"{tier_name}_total_items"] = total

        tail_exp = tier_exposure.get("tail_exposure", 0.0)
        tail_suppressed = tail_exp < settings.FAIRNESS_LONG_TAIL_THRESHOLD

        return {
            **tier_exposure,
            **tier_counts,
            "tail_suppression_detected": tail_suppressed,
            "long_tail_threshold": settings.FAIRNESS_LONG_TAIL_THRESHOLD,
            "interpretation": (
                f"Long-tail items are underexposed ({tail_exp:.1%} coverage). "
                "Consider diversity re-ranking or popularity penalty."
                if tail_suppressed
                else f"Acceptable long-tail exposure ({tail_exp:.1%})."
            ),
        }

    # =========================================================================
    # Bias Mitigation Strategies
    # =========================================================================

    def apply_mitigation(
        self,
        recommendations: List[Dict[str, Any]],
        strategy: str = "diversity_rerank",
        user_history: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply a post-hoc bias mitigation strategy to a recommendation list.

        Strategies:
          - diversity_rerank: Maximal Marginal Relevance (MMR) re-ranking.
            Balances relevance (original score) with dissimilarity to already-
            selected items.  λ = MMR_LAMBDA (default 0.7).
          - popularity_penalty: Penalise each item's score by its relative
            popularity, boosting long-tail items.
          - calibrated: Reorder so the category distribution matches the
            distribution observed in user_history (requires user_history).

        Args:
            recommendations: List of dicts with 'product_id' and 'score' keys
            strategy: One of 'diversity_rerank', 'popularity_penalty', 'calibrated'
            user_history: Required for 'calibrated' strategy

        Returns:
            Reordered / rescored recommendation list (same length as input)
        """
        if not recommendations:
            return recommendations

        if strategy == "popularity_penalty":
            return self._popularity_penalty(recommendations)
        elif strategy == "calibrated":
            return self._calibrated_rerank(recommendations, user_history or [])
        else:  # default: diversity_rerank (MMR)
            return self._mmr_rerank(recommendations)

    def _mmr_rerank(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance reranking.
        MMR(i) = λ * score(i) - (1-λ) * max_sim(i, already_selected)
        """
        lam = settings.MMR_LAMBDA
        n = len(recommendations)
        if n <= 1:
            return recommendations

        item_ids = [r["product_id"] for r in recommendations]
        scores = np.array([r["score"] for r in recommendations], dtype=float)

        # Build item factor matrix for similarity computation
        indices = [self.model.item_id_to_idx.get(iid) for iid in item_ids]
        has_factors = all(
            i is not None and self.model.item_factors is not None for i in indices
        )

        if not has_factors:
            return recommendations  # fallback: return unchanged

        vecs = np.array(
            [self.model.item_factors[i] for i in indices]  # type: ignore[index]
        )  # (n, n_components) — already L2-normalized

        selected: List[int] = []
        remaining = list(range(n))
        result: List[Dict[str, Any]] = []

        # Always select the highest-scoring item first
        first = int(np.argmax(scores))
        selected.append(first)
        remaining.remove(first)
        result.append(recommendations[first])

        while remaining:
            selected_vecs = vecs[selected]  # (k, n_components)
            mmr_scores = []
            for i in remaining:
                sim_to_selected = float(np.max(vecs[i] @ selected_vecs.T))
                mmr = lam * scores[i] - (1.0 - lam) * sim_to_selected
                mmr_scores.append(mmr)

            best_pos = int(np.argmax(mmr_scores))
            best_idx = remaining[best_pos]
            selected.append(best_idx)
            remaining.remove(best_idx)
            result.append(recommendations[best_idx])

        return result

    def _popularity_penalty(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Penalise score by relative popularity (log-normalized)."""
        if not self.model.item_popularity:
            return recommendations

        max_pop = max(self.model.item_popularity.values()) or 1
        result = []
        for rec in recommendations:
            pop = self.model.item_popularity.get(rec["product_id"], 1)
            penalty = float(np.log1p(pop) / np.log1p(max_pop))
            new_rec = dict(rec)
            new_rec["score"] = round(
                float(np.clip(rec["score"] * (1.0 - 0.5 * penalty), 0.0, 1.0)), 6
            )
            result.append(new_rec)
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    def _calibrated_rerank(
        self,
        recommendations: List[Dict[str, Any]],
        user_history: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Reorder recommendations so their category distribution matches the
        user's historical category distribution.
        Uses greedy calibration: at each step pick the item that minimises
        KL divergence between target and current distribution.
        """
        from collections import Counter

        if not user_history:
            return recommendations

        hist_cats = [
            self.model.item_categories.get(iid, "unknown") for iid in user_history
        ]
        hist_total = len(hist_cats)
        target_dist: Dict[str, float] = {
            cat: cnt / hist_total for cat, cnt in Counter(hist_cats).items()
        }

        selected: List[Dict[str, Any]] = []
        remaining = list(recommendations)
        cat_counts: Counter = Counter()

        while remaining:
            best_item = None
            best_kl = float("inf")

            for rec in remaining:
                cat = self.model.item_categories.get(rec["product_id"], "unknown")
                trial_counts = Counter(cat_counts)
                trial_counts[cat] += 1
                trial_total = sum(trial_counts.values())
                trial_dist = {c: trial_counts[c] / trial_total for c in target_dist}
                kl = sum(
                    p * np.log((p + 1e-12) / (trial_dist.get(c, 1e-12) + 1e-12))
                    for c, p in target_dist.items()
                )
                if kl < best_kl:
                    best_kl = kl
                    best_item = rec

            if best_item is None:
                break
            selected.append(best_item)
            remaining.remove(best_item)
            cat = self.model.item_categories.get(best_item["product_id"], "unknown")
            cat_counts[cat] += 1

        return selected

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
