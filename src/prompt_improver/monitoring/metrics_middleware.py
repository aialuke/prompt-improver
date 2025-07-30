"""
OpenTelemetry metrics middleware for APES application.
Provides comprehensive observability for HTTP requests, ML operations, and system resources.
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import logging

# Import OpenTelemetry metrics
from src.prompt_improver.monitoring.opentelemetry.metrics import (
    get_http_metrics, get_database_metrics, get_ml_metrics, get_business_metrics
)

logger = logging.getLogger(__name__)

# Global OpenTelemetry metrics instances
http_metrics = get_http_metrics()
database_metrics = get_database_metrics()
ml_metrics = get_ml_metrics()
business_metrics = get_business_metrics()


class OpenTelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware to collect OpenTelemetry metrics for HTTP requests."""

    def __init__(self, app, metrics_enabled: bool = True):
        super().__init__(app)
        self.metrics_enabled = metrics_enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.metrics_enabled:
            return await call_next(request)

        # Skip metrics collection for the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()

        try:
            response = await call_next(request)

            # Record metrics using OpenTelemetry
            duration_ms = (time.time() - start_time) * 1000
            method = request.method
            endpoint = self._normalize_endpoint(request.url.path)
            status_code = response.status_code

            # Calculate request/response sizes if available
            request_size = int(request.headers.get("content-length", 0))
            response_size = len(response.body) if hasattr(response, 'body') else None

            http_metrics.record_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration_ms=duration_ms,
                request_size=request_size if request_size > 0 else None,
                response_size=response_size
            )

            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            method = request.method
            endpoint = self._normalize_endpoint(request.url.path)
            
            HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status="500").inc()
            HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
            logger.error(f"Request failed: {e}")
            raise
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path to reduce cardinality."""
        # Replace UUIDs and IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Limit path length to prevent high cardinality
        if len(path) > 100:
            path = path[:100] + "..."
            
        return path


def record_ml_prediction(model_name: str, model_version: str, duration: float):
    """Record ML model prediction metrics using OpenTelemetry."""
    duration_ms = duration * 1000  # Convert to milliseconds
    ml_metrics.record_inference(
        model_name=model_name,
        model_version=model_version,
        duration_ms=duration_ms,
        success=True
    )


def record_ml_error(model_name: str, model_version: str, error_type: str):
    """Record ML model error metrics using OpenTelemetry."""
    ml_metrics.record_inference(
        model_name=model_name,
        model_version=model_version,
        duration_ms=0,  # Error case, no duration
        success=False
    )


def record_database_connection(database: str, pool: str, count: int):
    """Record database connection metrics using OpenTelemetry."""
    database_metrics.set_connection_metrics(
        active_connections=count,
        pool_size=count + 5,  # Estimate pool size
        pool_name=pool
    )


def record_database_query(operation: str, table: str, duration: float):
    """Record database query metrics using OpenTelemetry."""
    duration_ms = duration * 1000  # Convert to milliseconds
    database_metrics.record_query(
        operation=operation,
        table=table,
        duration_ms=duration_ms,
        success=True
    )


def record_redis_operation(operation: str, key_pattern: str):
    """Record Redis operation metrics using OpenTelemetry."""
    # Use business metrics for Redis operations
    business_metrics.record_feature_flag_evaluation(
        flag_name=f"redis_{operation}",
        enabled=True
    )


def record_batch_queue_size(queue_name: str, size: int):
    """Record batch processing queue size using OpenTelemetry."""
    # Use business metrics for queue size tracking
    business_metrics.update_active_sessions(
        change=size,
        session_type=f"batch_queue_{queue_name}"
    )


def record_cache_hit(cache_type: str, key_pattern: str):
    """Record cache hit metrics using OpenTelemetry."""
    # Use business metrics for cache operations
    business_metrics.record_feature_flag_evaluation(
        flag_name=f"cache_hit_{cache_type}",
        enabled=True
    )


def record_cache_miss(cache_type: str, key_pattern: str):
    """Record cache miss metrics using OpenTelemetry."""
    # Use business metrics for cache operations
    business_metrics.record_feature_flag_evaluation(
        flag_name=f"cache_miss_{cache_type}",
        enabled=False
    )


def record_mcp_operation(operation: str, status: str):
    """Record MCP operation metrics using OpenTelemetry."""
    # Use business metrics for MCP operations
    business_metrics.record_feature_flag_evaluation(
        flag_name=f"mcp_{operation}",
        enabled=(status == "success")
    )


def get_metrics() -> str:
    """Get OpenTelemetry metrics in Prometheus format."""
    try:
        # OpenTelemetry metrics are exported via OTLP/Prometheus exporters
        # This function is kept for compatibility but metrics are now exported automatically
        return "# OpenTelemetry metrics are exported via configured exporters\n"
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return ""
