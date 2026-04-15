"""
Configuration module for the Product Recommendation System
"""

import os
from typing import Optional


class Settings:
    """Application settings and environment variables"""

    # API Settings
    APP_NAME: str = "Product Recommendation System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Model Settings
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models")
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "collaborative")

    # Metrics Settings
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "True").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8001"))

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Feature Settings
    RECOMMENDATION_CACHE_SIZE: int = int(os.getenv("RECOMMENDATION_CACHE_SIZE", "1000"))
    MAX_RECOMMENDATIONS: int = int(os.getenv("MAX_RECOMMENDATIONS", "100"))
    DEFAULT_RECOMMENDATIONS: int = int(os.getenv("DEFAULT_RECOMMENDATIONS", "5"))

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "retail-rocket-recommendations"
    )

    # Responsible AI Settings
    FAIRNESS_ENABLED: bool = os.getenv("FAIRNESS_ENABLED", "True").lower() == "true"
    FAIRNESS_POPULARITY_THRESHOLD: float = float(
        os.getenv("FAIRNESS_POPULARITY_THRESHOLD", "0.3")
    )
    MIN_CATEGORY_DIVERSITY: float = float(os.getenv("MIN_CATEGORY_DIVERSITY", "0.3"))
    EXPLAINABILITY_METHOD: str = os.getenv("EXPLAINABILITY_METHOD", "similarity")
    MAX_EXPLANATION_FEATURES: int = int(os.getenv("MAX_EXPLANATION_FEATURES", "5"))

    # Training Data Settings
    DATA_PATH: str = os.getenv("DATA_PATH", "./data")
    MIN_USER_INTERACTIONS: int = int(os.getenv("MIN_USER_INTERACTIONS", "5"))
    MAX_TRAINING_USERS: int = int(os.getenv("MAX_TRAINING_USERS", "50000"))

    # Hybrid Model Weights
    HYBRID_COLLAB_WEIGHT: float = float(os.getenv("HYBRID_COLLAB_WEIGHT", "0.6"))
    HYBRID_CONTENT_WEIGHT: float = float(os.getenv("HYBRID_CONTENT_WEIGHT", "0.4"))

    # Collaborative Filtering Hyperparameters
    SVD_N_COMPONENTS: int = int(os.getenv("SVD_N_COMPONENTS", "50"))
    SVD_N_ITER: int = int(os.getenv("SVD_N_ITER", "10"))


settings = Settings()
