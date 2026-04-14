"""
Custom middleware for the FastAPI application
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.metrics import RequestMetrics

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging request/response details"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response"""
        
        # Extract request information
        method = request.method
        url = str(request.url)
        path = request.url.path
        
        # Log request
        logger.info(f"Incoming: {method} {path}")
        
        # Record request start time
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {method} {path} - {response.status_code} "
                f"(duration: {process_time:.3f}s)"
            )
            
            # Add custom header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing {method} {path}: {str(e)}")
            raise


# TODO: ====Add more middleware as needed====
# - Authentication/Authorization middleware
# - Rate limiting middleware
# - CORS middleware
# - Request validation middleware
# - Performance monitoring middleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    TODO: ====Implement comprehensive error handling middleware====
    
    Features:
    - Catch and standardize error responses
    - Log error details
    - Implement error recovery strategies
    - Track error metrics
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.exception(f"Unhandled exception: {str(e)}")
            # TODO: Return standardized error response
            raise
