"""
Comprehensive model evaluation script for the Product Recommendation System.

Loads a saved model bundle and computes a full suite of evaluation metrics:

Ranking Quality:
  - Precision@k, Recall@k, NDCG@k  (already in train_model.py; re-computed here
    for consistency in the standalone report)
  - MRR@k   — Mean Reciprocal Rank
  - Hit Rate@k — fraction of users with ≥1 hit in top-k
  - F1@k    — harmonic mean of Precision@k and Recall@k

Beyond-Accuracy (Diversity & Novelty):
  - Intra-List Diversity (ILD) — mean pairwise dissimilarity within a rec list
  - Category Entropy@k — Shannon entropy of category distribution in rec list
  - Novelty@k — mean self-information of recommended items (surprisingness)

Coverage:
  - Catalog Coverage — fraction of all items ever recommended
  - User Coverage   — fraction of test users who received ≥1 recommendation
  - Long-Tail Coverage — coverage restricted to the bottom-50% popularity items

Usage:
    python scripts/evaluate_model.py \\
      --bundle models/model_bundle.pkl \\
      --dataset ./data \\
      --k 10 \\
      --output models/evaluation_report.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# ModelEvaluator
# =============================================================================


class ModelEvaluator:
    """
    Loads a saved model bundle and evaluates it against held-out test interactions.

    Args:
        bundle_path: Path to model_bundle.pkl produced by train_model.py
        dataset_path: Path to directory containing events.csv
        k: Cut-off for ranking metrics (default 10)
    """

    def __init__(self, bundle_path: str, dataset_path: str, k: int = 10):
        self.k = k
        self.bundle_path = Path(bundle_path)
        self.dataset_path = Path(dataset_path)

        self._load_bundle()
        self._load_test_interactions()

        # Pre-compute total interaction count for novelty denominator
        self._total_interactions = float(sum(self.item_popularity.values()) or 1)

        # Pre-sort popularity for long-tail split
        pop_values = sorted(self.item_popularity.values())
        self._p50_popularity = float(
            np.percentile(pop_values, 50) if pop_values else 1.0
        )

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _load_bundle(self) -> None:
        """Load all artifacts from the saved model bundle."""
        logger.info(f"Loading bundle from {self.bundle_path}")
        bundle = joblib.load(self.bundle_path)

        self.user_factors: np.ndarray = bundle["user_factors"]
        self.item_factors: np.ndarray = bundle["item_factors"]
        self.ease_B: Optional[np.ndarray] = bundle.get("ease_B")
        self.ease_active_indices: Optional[np.ndarray] = bundle.get(
            "ease_active_indices"
        )
        self.user_id_to_idx: Dict[str, int] = bundle["user_id_to_idx"]
        self.item_id_to_idx: Dict[str, int] = bundle["item_id_to_idx"]
        self.idx_to_item_id: Dict[int, str] = bundle["idx_to_item_id"]
        self.item_popularity: Dict[str, int] = bundle["item_popularity"]
        self.item_categories: Dict[str, str] = bundle["item_categories"]
        self.interaction_matrix: csr_matrix = bundle["interaction_matrix"]
        self.metadata: Dict[str, Any] = bundle.get("metadata", {})

        logger.info(
            f"Bundle loaded: {len(self.user_id_to_idx)} users, "
            f"{len(self.item_id_to_idx)} items, "
            f"SVD components={self.metadata.get('svd_n_components', '?')}"
        )

    def _load_test_interactions(self) -> None:
        """
        Per-user temporal split: each user's last 20% of interactions (by timestamp)
        form the test set; the rest is train.  Users appear in BOTH sets, so the
        model can personalize — this is the correct protocol for implicit feedback.
        """
        events_path = self.dataset_path / "events.csv"
        if not events_path.exists():
            raise FileNotFoundError(f"events.csv not found at {events_path}")

        logger.info("Loading events.csv for temporal test split …")
        df = pd.read_csv(events_path)
        df = df.dropna(subset=["itemid"])
        df["visitorid"] = df["visitorid"].astype(str)
        df["itemid"] = df["itemid"].astype(str)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        weight_map = {"view": 1, "addtocart": 3, "transaction": 5}
        df["weight"] = df["event"].map(weight_map).fillna(1).astype(float)

        # Filter to users that exist in the trained model
        df = df[df["visitorid"].isin(self.user_id_to_idx)]

        # Per-user temporal split: sort by timestamp, hold out last 20%
        df = df.sort_values(["visitorid", "timestamp"])
        test_rows = []
        for _uid, grp in df.groupby("visitorid", sort=False):
            n = len(grp)
            if n < 2:
                continue
            split = max(1, int(n * 0.8))
            test_rows.append(grp.iloc[split:])

        if not test_rows:
            raise RuntimeError(
                "No test interactions after temporal split — check dataset"
            )

        test_df = pd.concat(test_rows, ignore_index=True)
        # Aggregate (user, item) pairs in test set
        self.test_interactions = (
            test_df.groupby(["visitorid", "itemid"])["weight"].sum().reset_index()
        )
        logger.info(
            f"Test set: {len(self.test_interactions):,} interactions, "
            f"{self.test_interactions['visitorid'].nunique():,} users"
        )

    # -------------------------------------------------------------------------
    # Prediction helper
    # -------------------------------------------------------------------------

    def _predict_top_k(self, user_id: str) -> List[str]:
        """
        Return top-k item IDs for a user, masking training interactions.
        Uses EASE scoring when ease_B is available; falls back to SVD.
        Returns [] for cold-start users.
        """
        if user_id not in self.user_id_to_idx:
            return []

        u_idx = self.user_id_to_idx[user_id]

        n_items = len(self.item_id_to_idx)
        if self.ease_B is not None and self.ease_active_indices is not None:
            # EASE restricted to active items: project user history to active subset
            user_row = np.array(
                self.interaction_matrix[u_idx].todense(), dtype=np.float32
            ).flatten()
            active_user_row = user_row[self.ease_active_indices]  # (n_active,)
            active_scores = active_user_row @ self.ease_B  # (n_active,)

            # Map back to full item score vector (unscored items stay at -inf)
            scores = np.full(n_items, -np.inf, dtype=np.float32)
            scores[self.ease_active_indices] = active_scores
        else:
            scores = self.user_factors[u_idx] @ self.item_factors.T

        # Mask seen training items
        seen = self.interaction_matrix[u_idx].nonzero()[1]
        scores[seen] = -np.inf

        top_indices = np.argsort(scores)[::-1][: self.k]
        return [self.idx_to_item_id[i] for i in top_indices if scores[i] > -np.inf]

    # -------------------------------------------------------------------------
    # Ranking Metrics
    # -------------------------------------------------------------------------

    def compute_ranking_metrics(self) -> Dict[str, float]:
        """
        Compute Precision@k, Recall@k, NDCG@k, MRR@k, Hit Rate@k, F1@k.
        Only evaluates users present in both test set and training data.
        """
        logger.info(f"Computing ranking metrics @ k={self.k} …")

        test_by_user: Dict[str, set] = defaultdict(set)
        for _, row in self.test_interactions.iterrows():
            test_by_user[row["visitorid"]].add(row["itemid"])

        precisions, recalls, ndcgs, mrrs, hits = [], [], [], [], []
        n_eval = 0

        for user_id, ground_truth in test_by_user.items():
            top_k = self._predict_top_k(user_id)
            if not top_k:
                continue

            n_eval += 1
            hits_set = set(top_k) & ground_truth

            # Precision & Recall
            prec = len(hits_set) / self.k
            rec = len(hits_set) / len(ground_truth) if ground_truth else 0.0
            precisions.append(prec)
            recalls.append(rec)

            # NDCG
            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, item in enumerate(top_k)
                if item in ground_truth
            )
            idcg = sum(
                1.0 / np.log2(rank + 2)
                for rank in range(min(len(ground_truth), self.k))
            )
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            # MRR
            mrr = 0.0
            for rank, item in enumerate(top_k):
                if item in ground_truth:
                    mrr = 1.0 / (rank + 1)
                    break
            mrrs.append(mrr)

            # Hit Rate
            hits.append(1.0 if hits_set else 0.0)

        if n_eval == 0:
            logger.warning("No test users found in training data — all metrics = 0")
            return {
                f"precision@{self.k}": 0.0,
                f"recall@{self.k}": 0.0,
                f"ndcg@{self.k}": 0.0,
                f"mrr@{self.k}": 0.0,
                f"hit_rate@{self.k}": 0.0,
                f"f1@{self.k}": 0.0,
                "n_eval_users": 0,
            }

        mean_prec = float(np.mean(precisions))
        mean_rec = float(np.mean(recalls))
        f1 = (
            2 * mean_prec * mean_rec / (mean_prec + mean_rec)
            if (mean_prec + mean_rec) > 0
            else 0.0
        )

        return {
            f"precision@{self.k}": round(mean_prec, 6),
            f"recall@{self.k}": round(float(np.mean(recalls)), 6),
            f"ndcg@{self.k}": round(float(np.mean(ndcgs)), 6),
            f"mrr@{self.k}": round(float(np.mean(mrrs)), 6),
            f"hit_rate@{self.k}": round(float(np.mean(hits)), 6),
            f"f1@{self.k}": round(f1, 6),
            "n_eval_users": n_eval,
        }

    # -------------------------------------------------------------------------
    # Diversity & Novelty Metrics
    # -------------------------------------------------------------------------

    def compute_diversity_metrics(self) -> Dict[str, float]:
        """
        Compute Intra-List Diversity (ILD), Category Entropy@k, and Novelty@k.
        Sampled over up to 1000 users for performance.
        """
        logger.info("Computing diversity and novelty metrics …")

        sample_users = list(self.user_id_to_idx.keys())[:1000]

        ilds, cat_entropies, novelties = [], [], []

        for user_id in sample_users:
            top_k = self._predict_top_k(user_id)
            if len(top_k) < 2:
                continue

            item_indices = [
                self.item_id_to_idx[iid] for iid in top_k if iid in self.item_id_to_idx
            ]
            if len(item_indices) < 2:
                continue

            # --- ILD: mean pairwise dissimilarity ---
            vecs = self.item_factors[item_indices]  # (k, n_components)
            sim_matrix = vecs @ vecs.T  # (k, k) cosine sims (already L2-normalized)
            k = len(item_indices)
            sum_sim = sim_matrix.sum() - np.trace(sim_matrix)  # exclude diagonal
            ild = 1.0 - sum_sim / (k * (k - 1)) if k > 1 else 0.0
            ilds.append(float(np.clip(ild, 0.0, 1.0)))

            # --- Category Entropy ---
            cats = [self.item_categories.get(iid, "unknown") for iid in top_k]
            cat_counts = np.array(
                list(pd.Series(cats).value_counts(normalize=True).values)
            )
            n_cats = len(cat_counts)
            if n_cats > 1:
                entropy = -np.sum(cat_counts * np.log(cat_counts + 1e-12))
                norm_entropy = float(entropy / np.log(n_cats))
            else:
                norm_entropy = 0.0
            cat_entropies.append(norm_entropy)

            # --- Novelty: mean self-information ---
            item_novelties = []
            for iid in top_k:
                pop = self.item_popularity.get(iid, 1)
                prob = pop / self._total_interactions
                item_novelties.append(-np.log2(prob + 1e-12))
            novelties.append(float(np.mean(item_novelties)))

        return {
            "intra_list_diversity": round(float(np.mean(ilds)) if ilds else 0.0, 4),
            f"category_entropy@{self.k}": round(
                float(np.mean(cat_entropies)) if cat_entropies else 0.0, 4
            ),
            f"novelty@{self.k}": round(
                float(np.mean(novelties)) if novelties else 0.0, 4
            ),
            "n_sampled_users": len(sample_users),
        }

    # -------------------------------------------------------------------------
    # Coverage Metrics
    # -------------------------------------------------------------------------

    def compute_coverage_metrics(self) -> Dict[str, float]:
        """
        Compute Catalog Coverage, User Coverage, and Long-Tail Coverage.
        Sampled over up to 2000 users for performance.
        """
        logger.info("Computing coverage metrics …")

        sample_users = list(self.user_id_to_idx.keys())[:2000]
        total_catalog = len(self.item_id_to_idx)

        all_recommended: set = set()
        users_with_recs = 0

        for user_id in sample_users:
            top_k = self._predict_top_k(user_id)
            if top_k:
                users_with_recs += 1
                all_recommended.update(top_k)

        catalog_coverage = (
            len(all_recommended) / total_catalog if total_catalog else 0.0
        )
        user_coverage = users_with_recs / len(sample_users) if sample_users else 0.0

        # Long-tail coverage: items below median popularity
        tail_items = {
            iid
            for iid, cnt in self.item_popularity.items()
            if cnt <= self._p50_popularity and iid in self.item_id_to_idx
        }
        long_tail_recommended = all_recommended & tail_items
        long_tail_coverage = (
            len(long_tail_recommended) / len(tail_items) if tail_items else 0.0
        )

        return {
            "catalog_coverage": round(catalog_coverage, 4),
            "user_coverage": round(user_coverage, 4),
            "long_tail_coverage": round(long_tail_coverage, 4),
            "unique_items_recommended": len(all_recommended),
            "total_catalog_items": total_catalog,
            "tail_items_in_catalog": len(tail_items),
        }

    # -------------------------------------------------------------------------
    # Full Report
    # -------------------------------------------------------------------------

    def compute_all(self) -> Dict[str, Any]:
        """Run all metric groups and return a unified report dict."""
        ranking = self.compute_ranking_metrics()
        diversity = self.compute_diversity_metrics()
        coverage = self.compute_coverage_metrics()

        report = {
            "metadata": {
                "bundle_path": str(self.bundle_path),
                "evaluated_at": datetime.utcnow().isoformat(),
                "k": self.k,
                "model_type": self.metadata.get("model_type", "unknown"),
                "algorithm": self.metadata.get(
                    "algorithm", "ease" if self.ease_B is not None else "svd"
                ),
                "model_version": self.metadata.get("version", "unknown"),
                "training_users": self.metadata.get("n_users"),
                "training_items": self.metadata.get("n_items"),
                "svd_components": self.metadata.get("svd_n_components"),
                "ease_lambda": self.metadata.get("ease_lambda"),
            },
            "ranking_metrics": ranking,
            "diversity_metrics": diversity,
            "coverage_metrics": coverage,
        }

        # Flat summary for quick scanning
        report["summary"] = {
            **{k: v for k, v in ranking.items() if isinstance(v, float)},
            **{k: v for k, v in diversity.items() if isinstance(v, float)},
            **{k: v for k, v in coverage.items() if isinstance(v, float)},
        }

        return report

    def save_report(self, output_path: str) -> None:
        """Save the full evaluation report as JSON."""
        report = self.compute_all()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {out}")
        self._print_summary(report)

    @staticmethod
    def _print_summary(report: Dict[str, Any]) -> None:
        """Pretty-print key metrics to stdout."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        k = report["metadata"]["k"]
        rm = report["ranking_metrics"]
        dm = report["diversity_metrics"]
        cm = report["coverage_metrics"]

        print(f"\n--- Ranking @ k={k} ({rm.get('n_eval_users', 0)} eval users) ---")
        for key in [
            f"precision@{k}",
            f"recall@{k}",
            f"ndcg@{k}",
            f"mrr@{k}",
            f"hit_rate@{k}",
            f"f1@{k}",
        ]:
            val = rm.get(key, 0.0)
            print(f"  {key:<22} {val:.6f}")

        print(
            f"\n--- Diversity & Novelty ({dm.get('n_sampled_users', 0)} sampled users) ---"
        )
        for key in ["intra_list_diversity", f"category_entropy@{k}", f"novelty@{k}"]:
            val = dm.get(key, 0.0)
            print(f"  {key:<22} {val:.4f}")

        print(
            f"\n--- Coverage ({cm.get('unique_items_recommended', 0)} unique items seen) ---"
        )
        for key in ["catalog_coverage", "user_coverage", "long_tail_coverage"]:
            val = cm.get(key, 0.0)
            print(f"  {key:<22} {val:.4f}")

        print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved recommendation model bundle"
    )
    parser.add_argument(
        "--bundle",
        default="models/model_bundle.pkl",
        help="Path to model_bundle.pkl (default: models/model_bundle.pkl)",
    )
    parser.add_argument(
        "--dataset",
        default="./data",
        help="Path to dataset directory containing events.csv (default: ./data)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cut-off k (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="models/evaluation_report.json",
        help="Output path for JSON report (default: models/evaluation_report.json)",
    )
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        bundle_path=args.bundle,
        dataset_path=args.dataset,
        k=args.k,
    )
    evaluator.save_report(args.output)


if __name__ == "__main__":
    main()
