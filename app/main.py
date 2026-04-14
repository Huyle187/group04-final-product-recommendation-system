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
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ProductRecommendation,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from app.model import get_model
from app.middleware import LoggingMiddleware
from app.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_response_size_bytes,
    app_info,
    RequestMetrics,
    PredictionMetrics,
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
        version=settings.APP_VERSION,
        model_version=settings.MODEL_VERSION
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
            status="healthy",
            version=settings.APP_VERSION,
            debug=settings.DEBUG
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
                detail="Model not loaded"
            )
        
        return {"status": "ready"}


# ============================================================================
# Recommendation Endpoints
# ============================================================================

@app.post(
    "/recommendations",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get recommendations for a user"
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
                    detail=f"Invalid recommendation_type. Must be one of: {valid_types}"
                )
            
            # TODO: ====Validate user exists in dataset====
            # This should check if user_id exists in the training data
            
            # Get recommendations
            with PredictionMetrics(request.recommendation_type):
                recommendations_data = model.get_recommendations(
                    user_id=request.user_id,
                    num_recommendations=request.num_recommendations,
                    recommendation_type=request.recommendation_type,
                    filters=request.filters
                )
            
            # TODO: ====Apply fairness filtering if enabled====
            # - Check for bias in recommendations
            # - Adjust recommendations to improve diversity
            
            # Build response
            recommendations = [
                ProductRecommendation(**rec)
                for rec in recommendations_data
            ]
            
            response = RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                recommendation_type=request.recommendation_type,
                timestamp=datetime.utcnow().isoformat(),
                num_recommendations_returned=len(recommendations)
            )
            
            logger.info(
                f"Generated {len(recommendations)} {request.recommendation_type} "
                f"recommendations for user {request.user_id}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate recommendations"
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
            "total_users": len(user_ids)
        }


@app.get("/recommendations/{user_id}", tags=["Recommendations"])
async def get_user_recommendations(
    user_id: str,
    num_recommendations: int = Query(5, ge=1, le=100),
    rec_type: str = Query("collaborative", description="Recommendation type")
) -> RecommendationResponse:
    """
    Get recommendations for a user via URL parameters
    
    Alternative to POST /recommendations for simple use cases
    """
    request = RecommendationRequest(
        user_id=user_id,
        num_recommendations=num_recommendations,
        recommendation_type=rec_type
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
    product_id: str = Query(...),
    method: str = Query("shap", description="Explanation method: shap or lime")
) -> Dict[str, Any]:
    """
    TODO: ====Implement recommendation explanation====
    
    Provide explanation for why a product was recommended
    
    Methods:
    - SHAP: SHapley Additive exPlanations
    - LIME: Local Interpretable Model-agnostic Explanations
    
    Returns:
    - Feature importance
    - Impact scores
    - Decision factors
    """
    
    with RequestMetrics("GET", "/recommendations/{user_id}/explain"):
        # TODO: Implement SHAP/LIME explanations
        return {
            "user_id": user_id,
            "product_id": product_id,
            "explanation_method": method,
            "status": "Not yet implemented"
        }


# ============================================================================
# Fairness Endpoints
# ============================================================================

@app.get("/fairness/check", tags=["Responsible AI"])
async def check_fairness(user_id: str) -> Dict[str, Any]:
    """
    TODO: ====Implement fairness checking====
    
    Check if recommendations for a user have fairness issues:
    - Demographic parity
    - Equal opportunity
    - Calibration
    - Recommendation diversity
    """
    
    with RequestMetrics("GET", "/fairness/check"):
        return {
            "user_id": user_id,
            "fairness_score": 0.85,
            "status": "Not fully implemented"
        }


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
            "error_code": exc.status_code
        }
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
            "error_code": 500
        }
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
        "metrics": "/metrics"
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
        log_level="info"
    )
