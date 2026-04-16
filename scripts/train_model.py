"""
Model Training Script for Product Recommendation System
DDM501: AI in Production: From Models to Systems

Trains collaborative filtering, content-based, and hybrid recommendation models
on the Retail Rocket dataset and saves artifacts for serving.

Usage:
    python scripts/train_model.py --dataset ./data --model collaborative --evaluate --mlflow
    python scripts/train_model.py --dataset ./data --model hybrid --evaluate --mlflow
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

# Allow importing app config when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates recommendation models on the Retail Rocket dataset.

    Pipeline:
        load_data → feature_engineering → train_* → evaluate_model → save_model → log_to_mlflow
    """

    def __init__(self, model_type: str = "collaborative", output_dir: str = "./models"):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata: Dict[str, Any] = {
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
        }

        # Model artifacts (populated by train_* methods)
        self.user_factors: Optional[np.ndarray] = None  # (n_users, n_components)
        self.item_factors: Optional[np.ndarray] = None  # (n_items, n_components)
        self.item_similarity_matrix: Optional[np.ndarray] = (
            None  # (n_items, n_items) float32
        )
        self.user_id_to_idx: Dict[str, int] = {}
        self.item_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        self.idx_to_item_id: Dict[int, str] = {}
        self.item_popularity: Dict[str, int] = {}  # item_id -> interaction count
        self.item_categories: Dict[str, str] = {}  # item_id -> category string
        self.interaction_matrix: Optional[csr_matrix] = None
        self.metrics: Dict[str, float] = {}

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_data(self, dataset_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and preprocess the Retail Rocket dataset.

        Args:
            dataset_path: Directory containing events.csv and item_properties CSVs.

        Returns:
            (interactions_df, stats_dict)
            interactions_df has columns [visitorid, itemid, weight]
        """
        data_dir = Path(dataset_path)
        logger.info(f"Loading data from {data_dir}")

        # 1. Load events
        events_path = data_dir / "events.csv"
        if not events_path.exists():
            raise FileNotFoundError(f"events.csv not found at {events_path}")

        events = pd.read_csv(events_path)
        logger.info(f"Loaded {len(events):,} raw events")

        # 2. Convert timestamp (milliseconds epoch)
        events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")

        # 3. Drop null itemid rows; cast to str
        events = events.dropna(subset=["itemid"])
        events["itemid"] = events["itemid"].astype(int).astype(str)
        events["visitorid"] = events["visitorid"].astype(str)

        # 4. Assign interaction weights by event type
        event_weights = {"view": 1, "addtocart": 3, "transaction": 5}
        events["weight"] = events["event"].map(event_weights).fillna(1)

        # 5. Aggregate duplicate (user, item) pairs — sum weights
        interactions = (
            events.groupby(["visitorid", "itemid"])["weight"].sum().reset_index()
        )
        logger.info(f"Aggregated to {len(interactions):,} user-item pairs")

        # 6. Filter to users with >= MIN_USER_INTERACTIONS
        user_counts = interactions.groupby("visitorid")["itemid"].count()
        active_users = user_counts[user_counts >= settings.MIN_USER_INTERACTIONS].index
        interactions = interactions[interactions["visitorid"].isin(active_users)]
        logger.info(
            f"After filtering (>= {settings.MIN_USER_INTERACTIONS} interactions): "
            f"{len(active_users):,} users, {len(interactions):,} pairs"
        )

        # 7. Subsample to MAX_TRAINING_USERS most-active users
        if len(active_users) > settings.MAX_TRAINING_USERS:
            top_users = (
                user_counts[user_counts.index.isin(active_users)]
                .nlargest(settings.MAX_TRAINING_USERS)
                .index
            )
            interactions = interactions[interactions["visitorid"].isin(top_users)]
            logger.info(
                f"Subsampled to {settings.MAX_TRAINING_USERS:,} most-active users"
            )

        # 8. Load item properties → build item_categories
        props = self._load_item_properties(data_dir)
        active_items = set(interactions["itemid"].unique())
        if props is not None:
            props = props[props["itemid"].isin(active_items)]
            cat_props = props[props["property"] == "categoryid"].copy()
            if not cat_props.empty:
                cat_props["timestamp"] = pd.to_datetime(
                    cat_props["timestamp"], unit="ms"
                )
                cat_props = cat_props.sort_values("timestamp").groupby("itemid").last()
                self.item_categories = cat_props["value"].astype(str).to_dict()
                logger.info(
                    f"Loaded categories for {len(self.item_categories):,} items"
                )

        # 9. Item popularity
        item_counts = interactions.groupby("itemid")["visitorid"].count()
        self.item_popularity = item_counts.to_dict()

        # 10. Build index maps
        all_users = interactions["visitorid"].unique()
        all_items = interactions["itemid"].unique()
        self.user_id_to_idx = {uid: i for i, uid in enumerate(all_users)}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(all_items)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}
        self.idx_to_item_id = {i: iid for iid, i in self.item_id_to_idx.items()}

        # 11. Build sparse interaction matrix
        rows = interactions["visitorid"].map(self.user_id_to_idx).values
        cols = interactions["itemid"].map(self.item_id_to_idx).values
        data = interactions["weight"].values.astype(np.float32)
        n_users = len(all_users)
        n_items = len(all_items)
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
        )

        stats = {
            "num_users": n_users,
            "num_items": n_items,
            "num_interactions": len(interactions),
            "sparsity": float(1.0 - len(interactions) / (n_users * n_items)),
            "event_type_counts": events["event"].value_counts().to_dict(),
        }
        logger.info(f"Dataset stats: {stats}")
        return interactions, stats

    def _load_item_properties(self, data_dir: Path) -> Optional[pd.DataFrame]:
        """Load and concatenate item property files."""
        parts = []
        for fname in ["item_properties_part1.csv", "item_properties_part2.csv"]:
            fpath = data_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                df["itemid"] = df["itemid"].astype(str)
                parts.append(df)
        if not parts:
            logger.warning("No item_properties files found; skipping category loading")
            return None
        return pd.concat(parts, ignore_index=True)

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def feature_engineering(self, interactions: Any) -> Dict[str, Any]:
        """
        Build TF-IDF item content features from category and popularity tokens.

        Returns:
            {item_tfidf_matrix, tfidf_vectorizer, item_feature_texts}
        """
        logger.info("Building item content features (TF-IDF)")

        item_ids_ordered = [
            self.idx_to_item_id[i] for i in range(len(self.item_id_to_idx))
        ]

        texts = []
        item_feature_texts: Dict[str, str] = {}
        for item_id in item_ids_ordered:
            tokens = []
            # Category token
            cat = self.item_categories.get(item_id)
            if cat:
                tokens.append(f"cat_{cat}")

            # Popularity bucket token
            pop = self.item_popularity.get(item_id, 0)
            if pop > 100:
                tokens.append("popular_high")
            elif pop > 20:
                tokens.append("popular_medium")
            else:
                tokens.append("popular_low")

            text = " ".join(tokens) if tokens else "unknown"
            item_feature_texts[item_id] = text
            texts.append(text)

        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 1))
        item_tfidf_matrix = tfidf.fit_transform(texts)
        logger.info(
            f"TF-IDF matrix: {item_tfidf_matrix.shape[0]} items x "
            f"{item_tfidf_matrix.shape[1]} features"
        )

        return {
            "item_tfidf_matrix": item_tfidf_matrix,
            "tfidf_vectorizer": tfidf,
            "item_feature_texts": item_feature_texts,
        }

    # =========================================================================
    # Model Training
    # =========================================================================

    def train_collaborative_filtering(self, train_matrix: csr_matrix) -> Dict:
        """
        Train SVD-based matrix factorization for collaborative filtering.

        Decomposes the user-item interaction matrix using TruncatedSVD,
        then L2-normalizes both user and item factor matrices so that
        dot-product lookups are equivalent to cosine similarity.

        Args:
            train_matrix: scipy csr_matrix (n_users x n_items)

        Returns:
            Dict with svd, user_factors, item_factors, explained_variance_ratio
        """
        n_components = min(settings.SVD_N_COMPONENTS, min(train_matrix.shape) - 1)
        logger.info(
            f"Training TruncatedSVD: n_components={n_components}, "
            f"n_iter={settings.SVD_N_ITER}, "
            f"matrix shape={train_matrix.shape}"
        )

        svd = TruncatedSVD(
            n_components=n_components,
            n_iter=settings.SVD_N_ITER,
            random_state=42,
        )

        user_factors = svd.fit_transform(train_matrix)  # (n_users, n_components)
        item_factors = svd.components_.T  # (n_items, n_components)

        # L2-normalize: dot product == cosine similarity
        user_factors = normalize(user_factors, norm="l2")
        item_factors = normalize(item_factors, norm="l2")

        self.user_factors = user_factors
        self.item_factors = item_factors

        evr = float(svd.explained_variance_ratio_.sum())
        logger.info(f"SVD explained variance ratio: {evr:.4f}")

        return {
            "svd": svd,
            "user_factors": user_factors,
            "item_factors": item_factors,
            "explained_variance_ratio": evr,
        }

    def train_content_based_model(self, features: Dict) -> Dict:
        """
        Build item-item cosine similarity matrix from TF-IDF features.

        Args:
            features: Output of feature_engineering()

        Returns:
            Dict with item_similarity_matrix, tfidf_vectorizer
        """
        item_tfidf_matrix = features["item_tfidf_matrix"]
        n_items = item_tfidf_matrix.shape[0]

        # Cap at 15k items to avoid OOM (15k x 15k x 4 bytes ~= 900 MB)
        MAX_CONTENT_ITEMS = 15000
        if n_items > MAX_CONTENT_ITEMS:
            logger.warning(
                f"n_items={n_items:,} > {MAX_CONTENT_ITEMS:,}; "
                f"truncating to {MAX_CONTENT_ITEMS:,} for content model"
            )
            item_tfidf_matrix = item_tfidf_matrix[:MAX_CONTENT_ITEMS]

        logger.info(
            f"Computing item-item cosine similarity ({item_tfidf_matrix.shape[0]:,} items)"
        )
        item_similarity = cosine_similarity(item_tfidf_matrix).astype(np.float32)

        self.item_similarity_matrix = item_similarity
        logger.info(f"Item similarity matrix shape: {item_similarity.shape}")

        return {
            "item_similarity_matrix": item_similarity,
            "tfidf_vectorizer": features["tfidf_vectorizer"],
        }

    def train_hybrid_model(self, collab_result: Dict, content_result: Dict) -> Dict:
        """
        Hybrid model combines collaborative and content scores at inference time.
        Weights are read from config (HYBRID_COLLAB_WEIGHT / HYBRID_CONTENT_WEIGHT).
        No additional training artifact is required.
        """
        logger.info(
            f"Hybrid model configured: "
            f"collab_weight={settings.HYBRID_COLLAB_WEIGHT}, "
            f"content_weight={settings.HYBRID_CONTENT_WEIGHT}"
        )
        return {
            "collab_weight": settings.HYBRID_COLLAB_WEIGHT,
            "content_weight": settings.HYBRID_CONTENT_WEIGHT,
            "collab": collab_result,
            "content": content_result,
        }

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate_model(
        self, test_interactions: pd.DataFrame, k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate collaborative filtering using Precision@k, Recall@k, NDCG@k,
        and catalog coverage on a held-out test set.

        Args:
            test_interactions: DataFrame [visitorid, itemid, weight] — held-out
            k: Ranking cut-off

        Returns:
            Dict with precision@k, recall@k, ndcg@k, catalog_coverage, n_eval_users
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Train collaborative filtering before evaluating")

        logger.info(
            f"Evaluating model on {len(test_interactions):,} test interactions (k={k})"
        )

        precision_list: List[float] = []
        recall_list: List[float] = []
        ndcg_list: List[float] = []
        all_recommended: set = set()

        test_by_user: Dict[str, List[str]] = (
            test_interactions.groupby("visitorid")["itemid"].apply(list).to_dict()
        )

        for user_id, true_items in test_by_user.items():
            if user_id not in self.user_id_to_idx:
                continue  # cold-start user — skip evaluation

            u_idx = self.user_id_to_idx[user_id]
            scores = self.user_factors[u_idx] @ self.item_factors.T  # (n_items,)

            # Mask training items
            seen = self.interaction_matrix[u_idx].nonzero()[1]
            scores[seen] = -np.inf

            top_k_idx = np.argsort(scores)[::-1][:k]
            top_k_items = [
                self.idx_to_item_id[i] for i in top_k_idx if scores[i] > -np.inf
            ]

            true_set = set(true_items)
            hits = len(set(top_k_items) & true_set)
            precision_list.append(hits / k)
            recall_list.append(hits / max(len(true_set), 1))

            # NDCG@k
            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, item in enumerate(top_k_items)
                if item in true_set
            )
            ideal_hits = min(len(true_set), k)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

            all_recommended.update(top_k_items)

        n_catalog = len(self.item_id_to_idx)
        coverage = len(all_recommended) / n_catalog if n_catalog > 0 else 0.0

        self.metrics = {
            f"precision@{k}": float(np.mean(precision_list)) if precision_list else 0.0,
            f"recall@{k}": float(np.mean(recall_list)) if recall_list else 0.0,
            f"ndcg@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
            "catalog_coverage": float(coverage),
            "n_eval_users": float(len(precision_list)),
        }

        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_model(self, stats: Optional[Dict] = None) -> None:
        """
        Save all model artifacts to models/model_bundle.pkl via joblib.
        Also writes a lightweight model_metadata.json for quick inspection.
        """
        model_bundle = {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "item_similarity_matrix": self.item_similarity_matrix,
            "user_id_to_idx": self.user_id_to_idx,
            "item_id_to_idx": self.item_id_to_idx,
            "idx_to_user_id": self.idx_to_user_id,
            "idx_to_item_id": self.idx_to_item_id,
            "item_popularity": self.item_popularity,
            "item_categories": self.item_categories,
            "interaction_matrix": self.interaction_matrix,
            "metadata": {
                "model_type": self.model_type,
                "version": self.metadata["version"],
                "created_at": self.metadata["created_at"],
                "metrics": self.metrics,
                "n_users": len(self.user_id_to_idx),
                "n_items": len(self.item_id_to_idx),
                "svd_n_components": settings.SVD_N_COMPONENTS,
                "hybrid_collab_weight": settings.HYBRID_COLLAB_WEIGHT,
                "hybrid_content_weight": settings.HYBRID_CONTENT_WEIGHT,
            },
        }

        bundle_path = self.output_dir / "model_bundle.pkl"
        joblib.dump(model_bundle, bundle_path, compress=3)
        logger.info(f"Model bundle saved to {bundle_path}")

        # Lightweight metadata JSON (no numpy arrays)
        metadata_path = self.output_dir / "model_metadata.json"
        safe_meta = {k: v for k, v in model_bundle["metadata"].items()}
        with open(metadata_path, "w") as f:
            json.dump(safe_meta, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        # Dataset stats for MLflow artifact
        if stats:
            stats_path = self.output_dir / "dataset_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

    # =========================================================================
    # MLflow Logging
    # =========================================================================

    def log_to_mlflow(self, stats: Optional[Dict] = None) -> None:
        """
        Log experiment run to MLflow using file-based tracking (no server needed).
        Tracking URI defaults to ./mlruns.
        """
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        run_name = f"{self.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model_type": self.model_type,
                    "svd_n_components": settings.SVD_N_COMPONENTS,
                    "svd_n_iter": settings.SVD_N_ITER,
                    "min_user_interactions": settings.MIN_USER_INTERACTIONS,
                    "max_training_users": settings.MAX_TRAINING_USERS,
                    "hybrid_collab_weight": settings.HYBRID_COLLAB_WEIGHT,
                    "hybrid_content_weight": settings.HYBRID_CONTENT_WEIGHT,
                    "n_users": len(self.user_id_to_idx),
                    "n_items": len(self.item_id_to_idx),
                }
            )

            if self.metrics:
                float_metrics = {
                    k: v for k, v in self.metrics.items() if isinstance(v, float)
                }
                mlflow.log_metrics(float_metrics)

            bundle_path = self.output_dir / "model_bundle.pkl"
            metadata_path = self.output_dir / "model_metadata.json"
            stats_path = self.output_dir / "dataset_stats.json"

            if bundle_path.exists():
                mlflow.log_artifact(str(bundle_path), "model")
            if metadata_path.exists():
                mlflow.log_artifact(str(metadata_path), "metadata")
            if stats_path.exists():
                mlflow.log_artifact(str(stats_path), "data")

        logger.info(
            f"MLflow run logged: experiment={settings.MLFLOW_EXPERIMENT_NAME}, "
            f"run_name={run_name}"
        )


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train product recommendation models on Retail Rocket dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="collaborative",
        choices=["collaborative", "content_based", "hybrid"],
        help="Recommendation model type to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data",
        help="Path to directory containing Retail Rocket CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of SVD iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Reserved for future deep learning models",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log experiment to MLflow",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on held-out test set",
    )
    args = parser.parse_args()

    if args.epochs != 10:
        settings.SVD_N_ITER = args.epochs

    trainer = ModelTrainer(model_type=args.model, output_dir=args.output_dir)

    # Step 1: Load data
    interactions, stats = trainer.load_data(args.dataset)

    # Step 2: Train/test split (20% per-user held out)
    train_interactions = interactions
    test_interactions = None

    if args.evaluate:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(
            gss.split(interactions, groups=interactions["visitorid"])
        )
        train_interactions = interactions.iloc[train_idx].reset_index(drop=True)
        test_interactions = interactions.iloc[test_idx].reset_index(drop=True)

        # Rebuild interaction matrix from TRAIN interactions only
        rows = train_interactions["visitorid"].map(trainer.user_id_to_idx).values
        cols = train_interactions["itemid"].map(trainer.item_id_to_idx).values
        data_arr = train_interactions["weight"].values.astype(np.float32)
        trainer.interaction_matrix = csr_matrix(
            (data_arr, (rows, cols)),
            shape=trainer.interaction_matrix.shape,
            dtype=np.float32,
        )
        logger.info(
            f"Train/test split: {len(train_interactions):,} train, "
            f"{len(test_interactions):,} test interactions"
        )

    # Step 3: Feature engineering
    features = trainer.feature_engineering(train_interactions)

    # Step 4: Train
    collab_result = None
    content_result = None

    if args.model in ("collaborative", "hybrid"):
        collab_result = trainer.train_collaborative_filtering(
            trainer.interaction_matrix
        )

    if args.model in ("content_based", "hybrid"):
        content_result = trainer.train_content_based_model(features)

    if args.model == "hybrid":
        trainer.train_hybrid_model(collab_result, content_result)

    # Step 5: Evaluate
    if (
        args.evaluate
        and test_interactions is not None
        and trainer.user_factors is not None
    ):
        trainer.evaluate_model(test_interactions, k=10)

    # Step 6: Save
    trainer.save_model(stats=stats)

    # Step 7: MLflow
    if args.mlflow:
        trainer.log_to_mlflow(stats=stats)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
