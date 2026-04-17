"""
Custom middleware for the FastAPI application
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.metrics import (
    RequestMetrics,
    http_request_duration_seconds,
    http_requests_total,
)

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect HTTP request metrics.

    This middleware automatically records:
    - Total request count (by method, endpoint, status)
    - Request latency (by method, endpoint)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request and record metrics.

        Args:
            request: The incoming HTTP request
            call_next: The next handler in the chain

        Returns:
            The HTTP response
        """
        # TODO 1: Record the start time
        start_time = time.time()

        # TODO 2: Call the next handler to get the response
        response = await call_next(request)

        # TODO 3: Calculate the duration
        duration = time.time() - start_time

        # TODO 4: Extract request information
        endpoint = request.url.path
        method = request.method
        status = response.status_code

        # TODO: Record the request count metric
        if http_requests_total is not None:
            http_requests_total.labels(
                method=method, endpoint=endpoint, status=status
            ).inc()

        # TODO: Record the request latency metric
        if http_request_duration_seconds is not None:
            http_request_duration_seconds.labels(
                method=method, endpoint=endpoint
            ).observe(duration)

        # TODO 7: Return the response
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log request details (for debugging).

    This is an optional middleware that logs request information.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Log request details and process."""
        import logging

        logger = logging.getLogger(__name__)

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Log response
        logger.info(f"Response: {response.status_code} - {duration:.3f}s")

        return response


# =============================================================================
# Helper function to check if metrics are properly configured
# =============================================================================


def check_metrics_middleware_ready() -> dict:
    """
    Check if the metrics middleware is properly configured.

    Returns:
        Dictionary with status of each metric
    """
    return {
        "request_count_ready": http_requests_total is not None,
        "request_latency_ready": http_request_duration_seconds is not None,
        "middleware_functional": True,
    }


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
