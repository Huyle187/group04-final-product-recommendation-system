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
    # TODO: ====Add model loading configuration====

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

    # TODO: ====Database configuration (if using)==== 
    # DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # TODO: ====MLflow configuration====
    # MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # TODO: ====Responsible AI settings====
    # FAIRNESS_ENABLED: bool = os.getenv("FAIRNESS_ENABLED", "True").lower() == "true"
    # EXPLAINABILITY_METHOD: str = os.getenv("EXPLAINABILITY_METHOD", "shap")


settings = Settings()
