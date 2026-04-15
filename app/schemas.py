"""
Pydantic models for request/response validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations"""
    
    user_id: str = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of recommendations to return"
    )
    recommendation_type: str = Field(
        default="collaborative",
        description="Type of recommendation: collaborative, content_based, or hybrid"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for recommendations"
    )

    class Config:
        example = {
            "user_id": "user123",
            "num_recommendations": 5,
            "recommendation_type": "collaborative",
            "filters": {"category": "electronics"}
        }


class BatchRecommendationRequest(BaseModel):
    """Request model for batch recommendations"""
    
    user_ids: List[str] = Field(..., description="List of user IDs")
    num_recommendations: int = Field(default=5, ge=1, le=100)
    recommendation_type: str = Field(default="collaborative")


# ============================================================================
# Response Models
# ============================================================================

class ProductRecommendation(BaseModel):
    """Individual product recommendation"""
    
    product_id: str
    product_name: str
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score")
    reason: Optional[str] = Field(
        default=None,
        description="Explanation for the recommendation"
    )
    category: Optional[str] = None
    price: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    
    user_id: str
    recommendations: List[ProductRecommendation]
    recommendation_type: str
    timestamp: str
    num_recommendations_returned: int


class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations"""
    
    results: Dict[str, RecommendationResponse]
    timestamp: str
    total_users: int


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str
    version: str
    debug: bool


class ModelInfoResponse(BaseModel):
    """Model information response"""

    model_version: str
    model_type: str
    last_updated: Optional[str] = None
    training_users: Optional[int] = None
    training_items: Optional[int] = None
    svd_components: Optional[int] = None
    evaluation_metrics: Optional[Dict[str, float]] = None

    class Config:
        protected_namespaces = ()


class ErrorResponse(BaseModel):
    """Error response model"""
    
    status: str = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
