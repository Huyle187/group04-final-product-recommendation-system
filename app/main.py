"""
Main FastAPI application for Product Recommendation System

This module defines all API endpoints for the recommendation system:
- Health checks
- Recommendation endpoints
- Model info endpoints
- Metrics endpoint
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.middleware import MetricsMiddleware
from prometheus_client import make_asgi_app

from app.config import settings
from app.explainability import get_explainability_engine
from app.fairness import get_fairness_checker
from app.metrics import (
    PredictionMetrics,
    RequestMetrics,
    app_info,
    http_request_duration_seconds,
    http_requests_total,
    http_response_size_bytes,
)
from app.middleware import LoggingMiddleware
from app.model import get_model
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ProductRecommendation,
    RecommendationRequest,
    RecommendationResponse,
)

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Real-time product recommendation engine for e-commerce platform",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================================================
# Middleware Configuration
# ============================================================================

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: ====Restrict in production====
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom logging middleware
app.add_middleware(LoggingMiddleware)


# Add Metrics middleware
app.add_middleware(MetricsMiddleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize model
    model = get_model()
    logger.info(f"Model initialized: {model.model_type} v{model.model_version}")

    # TODO: ====Initialize other components====
    # - Connect to database
    # - Initialize cache
    # - Load fairness/bias detectors
    # - Initialize MLflow

    # Set app info metric
    app_info.labels(
        version=settings.APP_VERSION, model_version=settings.MODEL_VERSION
    ).set(1)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(f"Shutting down {settings.APP_NAME}")

    # TODO: ====Cleanup resources====
    # - Close database connections
    # - Save cache
    # - Flush metrics


# ============================================================================
# Health Check Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint

    Returns:
        HealthResponse: Application health status
    """
    with RequestMetrics("GET", "/health"):
        return HealthResponse(
            status="healthy", version=settings.APP_VERSION, debug=settings.DEBUG
        )


@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint (for Kubernetes probes)

    Verifies that the application is ready to handle requests:
    - Model is loaded
    - Database is connected [TODO]
    - Dependencies are available
    """
    with RequestMetrics("GET", "/ready"):
        model = get_model()

        # TODO: ====Add readiness checks====
        # - Database connectivity
        # - Cache availability
        # - MLflow connectivity

        if not model.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        return {"status": "ready"}


# ============================================================================
# Recommendation Endpoints
# ============================================================================


@app.post(
    "/recommendations",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get recommendations for a user",
)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get product recommendations for a user

    Parameters:
    - **user_id**: Unique identifier for the user
    - **num_recommendations**: Number of recommendations to return (1-100)
    - **recommendation_type**: Type of recommendation algorithm
      - collaborative: Collaborative filtering
      - content_based: Content-based filtering
      - hybrid: Hybrid approach combining both
    - **filters**: Optional filters for recommendations

    Returns:
        RecommendationResponse: List of recommended products

    Raises:
        HTTPException: If user not found or invalid parameters
    """
    with RequestMetrics("POST", "/recommendations"):
        try:
            model = get_model()

            # Validate recommendation type
            valid_types = ["collaborative", "content_based", "hybrid"]
            if request.recommendation_type not in valid_types:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid recommendation_type. Must be one of: {valid_types}",
                )

            # TODO: ====Validate user exists in dataset====
            # This should check if user_id exists in the training data

            # Get recommendations
            with PredictionMetrics(request.recommendation_type):
                recommendations_data = model.get_recommendations(
                    user_id=request.user_id,
                    num_recommendations=request.num_recommendations,
                    recommendation_type=request.recommendation_type,
                    filters=request.filters,
                )

            # TODO: ====Apply fairness filtering if enabled====
            # - Check for bias in recommendations
            # - Adjust recommendations to improve diversity

            # Build response
            recommendations = [
                ProductRecommendation(**rec) for rec in recommendations_data
            ]

            response = RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                recommendation_type=request.recommendation_type,
                timestamp=datetime.utcnow().isoformat(),
                num_recommendations_returned=len(recommendations),
            )

            logger.info(
                f"Generated {len(recommendations)} {request.recommendation_type} "
                f"recommendations for user {request.user_id}"
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate recommendations",
            )


@app.post("/recommendations/batch", tags=["Recommendations"])
async def batch_recommendations(user_ids: list[str]) -> Dict[str, Any]:
    """
    TODO: ====Implement batch recommendations endpoint====

    Get recommendations for multiple users in a single request

    Optimization considerations:
    - Use vectorized operations
    - Cache results
    - Implement parallel processing
    - Add progress tracking
    """

    with RequestMetrics("POST", "/recommendations/batch"):
        model = get_model()

        # TODO: Remove mock implementation
        results = model.batch_recommendations(user_ids)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "total_users": len(user_ids),
        }


@app.get("/recommendations/{user_id}", tags=["Recommendations"])
async def get_user_recommendations(
    user_id: str,
    num_recommendations: int = Query(5, ge=1, le=100),
    rec_type: str = Query("collaborative", description="Recommendation type"),
) -> RecommendationResponse:
    """
    Get recommendations for a user via URL parameters

    Alternative to POST /recommendations for simple use cases
    """
    request = RecommendationRequest(
        user_id=user_id,
        num_recommendations=num_recommendations,
        recommendation_type=rec_type,
    )

    return await get_recommendations(request)


# ============================================================================
# Model Information Endpoints
# ============================================================================


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """
    Get information about the currently loaded model

    Returns:
        ModelInfoResponse: Model metadata and version information
    """
    with RequestMetrics("GET", "/model/info"):
        model = get_model()
        return ModelInfoResponse(**model.get_model_info())


@app.post("/model/reload", tags=["Model"])
async def reload_model() -> Dict[str, str]:
    """
    TODO: ====Implement model reloading====

    Reload the model from disk (for updating to new versions)

    Requires:
    - Authentication/authorization
    - Graceful request handling
    - Health validation after reload
    - Metrics reset
    """

    # TODO: Add authentication check

    with RequestMetrics("POST", "/model/reload"):
        from app.model import reload_model

        reload_model()
        logger.info("Model reloaded successfully")
        return {"status": "Model reloaded successfully"}


# ============================================================================
# Metrics Endpoint
# ============================================================================


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """
    Prometheus metrics endpoint

    Returns all collected metrics for Prometheus to scrape
    """
    with RequestMetrics("GET", "/metrics"):
        return generate_latest()


# ============================================================================
# Explainability Endpoints
# ============================================================================


@app.get("/recommendations/{user_id}/explain", tags=["Explainability"])
async def explain_recommendation(
    user_id: str,
    product_id: str = Query(..., description="Product ID to explain"),
    method: str = Query(
        "auto",
        description="Explanation method: auto, collaborative, or content_based",
    ),
) -> Dict[str, Any]:
    """
    Explain why a specific product was recommended to a user.

    Methods:
    - auto: Uses collaborative if model is loaded, otherwise content_based
    - collaborative: Neighbor-based — shows similar users who also interacted with this item
    - content_based: Feature similarity — shows which viewed items are similar to this product

    Note: Standard SHAP/KernelSHAP is not used here because it requires hundreds of
    model evaluations per request, making it impractical for real-time serving with
    TruncatedSVD matrix factorization. The neighbor-based approach provides equivalent
    conceptual explanations with O(1) latency using the pre-computed latent space.
    """
    with RequestMetrics("GET", "/recommendations/{user_id}/explain"):
        try:
            engine = get_explainability_engine()
            explanation = engine.explain(user_id, product_id, method)
            return explanation
        except Exception as e:
            logger.error(
                f"Explainability error for user={user_id}, product={product_id}: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate explanation",
            )


# ============================================================================
# Fairness Endpoints
# ============================================================================


@app.get("/fairness/check", tags=["Responsible AI"])
async def check_fairness(
    user_id: str,
    num_recommendations: int = Query(10, ge=1, le=50),
) -> Dict[str, Any]:
    """
    Assess fairness of recommendations for a specific user.

    Computes three proxy fairness metrics (no demographic data available in
    the Retail Rocket dataset):

    - Popularity bias: fraction of recommendations that are already-popular items
      (above 80th percentile). High bias indicates a filter bubble effect.
    - Category diversity: normalized Shannon entropy of the category distribution.
      Low score means recommendations are concentrated in one product category.
    - Overall fairness score: composite of (1 - popularity_bias) and diversity.

    Returns an overall_fairness_score in [0, 1] where higher is fairer.
    """
    with RequestMetrics("GET", "/fairness/check"):
        try:
            model = get_model()
            checker = get_fairness_checker()

            recs_data = model.get_recommendations(
                user_id=user_id,
                num_recommendations=num_recommendations,
                recommendation_type="hybrid",
            )
            rec_ids = [r["product_id"] for r in recs_data]

            report = checker.full_fairness_report(user_id, rec_ids)
            return report
        except Exception as e:
            logger.error(f"Fairness check error for user={user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to compute fairness metrics",
            )


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "error_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "error_code": 500,
        },
    )


# ============================================================================
# Root Endpoint
# ============================================================================


@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
    }


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
