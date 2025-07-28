#!/usr/bin/env python3
"""
APES Application Metrics Setup Script
Configures application-specific metrics collection for the monitoring stack.

This script sets up Prometheus metrics endpoints in the APES application
to provide comprehensive observability for ML pipeline operations.

Created: 2025-07-25
SRE Application Instrumentation
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add the src directory to the Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsSetup:
    """Handles setup of application metrics collection."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.src_dir = SRC_DIR
        self.config_dir = PROJECT_ROOT / "config"
        self.metrics_config = {}
        
    def create_metrics_configuration(self):
        """Create comprehensive metrics configuration."""
        logger.info("Creating metrics configuration...")
        
        self.metrics_config = {
            "metrics": {
                "enabled": True,
                "endpoint": "/metrics",
                "port": 8000,
                "collection_interval": 15,
                "retention_days": 30
            },
            "prometheus": {
                "multiproc_dir": self._get_prometheus_multiproc_dir(),
                "registry_type": "CollectorRegistry",
                "enable_gc_metrics": True,
                "enable_platform_collector": True,
                "enable_process_collector": True
            },
            "application_metrics": {
                "http_requests": {
                    "name": "http_requests_total",
                    "description": "Total HTTP requests",
                    "type": "counter",
                    "labels": ["method", "endpoint", "status"]
                },
                "http_request_duration": {
                    "name": "http_request_duration_seconds",
                    "description": "HTTP request duration in seconds",
                    "type": "histogram",
                    "labels": ["method", "endpoint"],
                    "buckets": [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                },
                "ml_model_predictions": {
                    "name": "ml_model_predictions_total",
                    "description": "Total ML model predictions",
                    "type": "counter",
                    "labels": ["model_name", "model_version"]
                },
                "ml_model_inference_duration": {
                    "name": "ml_model_inference_duration_seconds",
                    "description": "ML model inference duration in seconds",
                    "type": "histogram",
                    "labels": ["model_name", "model_version"],
                    "buckets": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
                },
                "ml_model_errors": {
                    "name": "ml_model_errors_total",
                    "description": "Total ML model errors",
                    "type": "counter",
                    "labels": ["model_name", "model_version", "error_type"]
                },
                "database_connections": {
                    "name": "db_connections_active",
                    "description": "Active database connections",
                    "type": "gauge",
                    "labels": ["database", "pool"]
                },
                "database_query_duration": {
                    "name": "db_query_duration_seconds",
                    "description": "Database query duration in seconds",
                    "type": "histogram",
                    "labels": ["operation", "table"],
                    "buckets": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
                },
                "redis_operations": {
                    "name": "redis_operations_total",
                    "description": "Total Redis operations",
                    "type": "counter",
                    "labels": ["operation", "key_pattern"]
                },
                "batch_processing_queue_size": {
                    "name": "batch_processing_queue_size",
                    "description": "Current batch processing queue size",
                    "type": "gauge",
                    "labels": ["queue_name"]
                },
                "batch_processing_duration": {
                    "name": "batch_processing_duration_seconds",
                    "description": "Batch processing duration in seconds",
                    "type": "histogram",
                    "labels": ["batch_type"],
                    "buckets": [1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
                },
                "mcp_operations": {
                    "name": "mcp_operations_total",
                    "description": "Total MCP operations",
                    "type": "counter",
                    "labels": ["operation", "status"]
                },
                "cache_hits": {
                    "name": "cache_hits_total",
                    "description": "Total cache hits",
                    "type": "counter",
                    "labels": ["cache_type", "key_pattern"]
                },
                "cache_misses": {
                    "name": "cache_misses_total",
                    "description": "Total cache misses",
                    "type": "counter",
                    "labels": ["cache_type", "key_pattern"]
                }
            },
            "health_checks": {
                "database": {
                    "enabled": True,
                    "timeout": 5,
                    "critical": True
                },
                "redis": {
                    "enabled": True,
                    "timeout": 3,
                    "critical": True
                },
                "ml_models": {
                    "enabled": True,
                    "timeout": 10,
                    "critical": False
                },
                "external_services": {
                    "enabled": True,
                    "timeout": 5,
                    "critical": False
                }
            },
            "sli_slo": {
                "availability": {
                    "target": 99.9,
                    "measurement_window": "30d"
                },
                "latency": {
                    "target_p95": 2.0,
                    "target_p99": 5.0,
                    "measurement_window": "30d"
                },
                "error_rate": {
                    "target": 0.1,
                    "measurement_window": "30d"
                }
            }
        }
        
        # Save configuration
        config_file = self.config_dir / "metrics_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.metrics_config, f, indent=2)
        
        logger.info(f"Metrics configuration saved to {config_file}")
    
    def _get_prometheus_multiproc_dir(self) -> str:
        """Get Prometheus multiproc directory from centralized configuration."""
        try:
            from prompt_improver.core.config import get_config
            config = get_config()
            return str(config.directory_paths.prometheus_multiproc_dir)
        except ImportError:
            # Fallback to hardcoded value if config not available
            logger.warning("Could not import centralized config, using fallback value for prometheus_multiproc_dir")
            return "/tmp/prometheus_multiproc"
        
    def create_metrics_middleware(self):
        """Create FastAPI metrics middleware."""
        logger.info("Creating metrics middleware...")
        
        middleware_code = '''"""
Prometheus metrics middleware for APES application.
Provides comprehensive observability for HTTP requests, ML operations, and system resources.
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
HTTP_REQUESTS = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ML_MODEL_PREDICTIONS = Counter(
    'ml_model_predictions_total',
    'Total ML model predictions',
    ['model_name', 'model_version']
)

ML_MODEL_INFERENCE_DURATION = Histogram(
    'ml_model_inference_duration_seconds',
    'ML model inference duration in seconds',
    ['model_name', 'model_version'],
    buckets=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
)

ML_MODEL_ERRORS = Counter(
    'ml_model_errors_total',
    'Total ML model errors',
    ['model_name', 'model_version', 'error_type']
)

DATABASE_CONNECTIONS = Gauge(
    'db_connections_active',
    'Active database connections',
    ['database', 'pool']
)

DATABASE_QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
)

REDIS_OPERATIONS = Counter(
    'redis_operations_total',
    'Total Redis operations',
    ['operation', 'key_pattern']
)

BATCH_PROCESSING_QUEUE_SIZE = Gauge(
    'batch_processing_queue_size',
    'Current batch processing queue size',
    ['queue_name']
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type', 'key_pattern']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type', 'key_pattern']
)

MCP_OPERATIONS = Counter(
    'mcp_operations_total',
    'Total MCP operations',
    ['operation', 'status']
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics for HTTP requests."""
    
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
            
            # Record metrics
            duration = time.time() - start_time
            method = request.method
            endpoint = self._normalize_endpoint(request.url.path)
            status = str(response.status_code)
            
            HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=status).inc()
            HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
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
        path = re.sub(r'/\\d+', '/{id}', path)
        
        # Limit path length to prevent high cardinality
        if len(path) > 100:
            path = path[:100] + "..."
            
        return path


def record_ml_prediction(model_name: str, model_version: str, duration: float):
    """Record ML model prediction metrics."""
    ML_MODEL_PREDICTIONS.labels(model_name=model_name, model_version=model_version).inc()
    ML_MODEL_INFERENCE_DURATION.labels(model_name=model_name, model_version=model_version).observe(duration)


def record_ml_error(model_name: str, model_version: str, error_type: str):
    """Record ML model error metrics."""
    ML_MODEL_ERRORS.labels(model_name=model_name, model_version=model_version, error_type=error_type).inc()


def record_database_connection(database: str, pool: str, count: int):
    """Record database connection metrics."""
    DATABASE_CONNECTIONS.labels(database=database, pool=pool).set(count)


def record_database_query(operation: str, table: str, duration: float):
    """Record database query metrics."""
    DATABASE_QUERY_DURATION.labels(operation=operation, table=table).observe(duration)


def record_redis_operation(operation: str, key_pattern: str):
    """Record Redis operation metrics."""
    REDIS_OPERATIONS.labels(operation=operation, key_pattern=key_pattern).inc()


def record_batch_queue_size(queue_name: str, size: int):
    """Record batch processing queue size."""
    BATCH_PROCESSING_QUEUE_SIZE.labels(queue_name=queue_name).set(size)


def record_cache_hit(cache_type: str, key_pattern: str):
    """Record cache hit metrics."""
    CACHE_HITS.labels(cache_type=cache_type, key_pattern=key_pattern).inc()


def record_cache_miss(cache_type: str, key_pattern: str):
    """Record cache miss metrics."""
    CACHE_MISSES.labels(cache_type=cache_type, key_pattern=key_pattern).inc()


def record_mcp_operation(operation: str, status: str):
    """Record MCP operation metrics."""
    MCP_OPERATIONS.labels(operation=operation, status=status).inc()


def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    try:
        return generate_latest()
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return ""
'''
        
        # Write the middleware file
        middleware_file = self.src_dir / "prompt_improver" / "monitoring" / "metrics_middleware.py"
        middleware_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(middleware_file, 'w') as f:
            f.write(middleware_code)
        
        logger.info(f"Metrics middleware created at {middleware_file}")
        
    def create_health_check_endpoint(self):
        """Create comprehensive health check endpoint."""
        logger.info("Creating health check endpoint...")
        
        health_check_code = '''"""
Comprehensive health check endpoint for APES application.
Provides detailed health status for all system components.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
import logging

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class ComponentHealth(BaseModel):
    status: HealthStatus
    response_time: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    last_check: Optional[float] = None

class OverallHealth(BaseModel):
    status: HealthStatus
    timestamp: float
    version: str
    uptime: float
    components: Dict[str, ComponentHealth]
    
router = APIRouter()

class HealthChecker:
    """Health check service for monitoring system components."""
    
    def __init__(self):
        self.start_time = time.time()
        self.version = "1.0.0"  # Should be loaded from config
        
    async def check_database_health(self) -> ComponentHealth:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            # Import database session here to avoid circular imports
            from prompt_improver.database import get_session
            
            async with get_session() as session:
                # Simple query to test connectivity
                result = await session.execute("SELECT 1")
                await result.fetchone()
                
            response_time = time.time() - start_time
            
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={
                    "query_time": response_time,
                    "connection_pool": "available"
                }
            )
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time()
            )
    
    async def check_redis_health(self) -> ComponentHealth:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        
        try:
            from prompt_improver.utils.redis_cache import get_cache
            
            cache = await get_cache()
            
            # Test Redis with a ping
            test_key = "health_check_test"
            await cache.set(test_key, "test_value", expire=10)
            value = await cache.get(test_key)
            await cache.delete(test_key)
            
            if value != "test_value":
                raise Exception("Redis read/write test failed")
            
            response_time = time.time() - start_time
            
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={
                    "ping_time": response_time,
                    "read_write_test": "passed"
                }
            )
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time()
            )
    
    async def check_ml_models_health(self) -> ComponentHealth:
        """Check ML models availability and performance."""
        start_time = time.time()
        
        try:
            # This is a placeholder - implement based on your ML model loading
            # from prompt_improver.ml.models import get_model_registry
            
            # registry = get_model_registry()
            # models = await registry.list_active_models()
            
            response_time = time.time() - start_time
            
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={
                    "models_loaded": 1,  # Replace with actual count
                    "model_check_time": response_time
                }
            )
            
        except Exception as e:
            logger.error(f"ML models health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.DEGRADED,  # Non-critical
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time()
            )
    
    async def check_external_services_health(self) -> ComponentHealth:
        """Check external services connectivity."""
        start_time = time.time()
        
        try:
            # Add checks for external services (APIs, etc.)
            # This is a placeholder implementation
            
            response_time = time.time() - start_time
            
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={
                    "external_apis": "available"
                }
            )
            
        except Exception as e:
            logger.error(f"External services health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.DEGRADED,  # Non-critical
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time()
            )
    
    async def get_overall_health(self) -> OverallHealth:
        """Get comprehensive health status."""
        # Run all health checks concurrently
        checks = await asyncio.gather(
            self.check_database_health(),
            self.check_redis_health(),
            self.check_ml_models_health(),
            self.check_external_services_health(),
            return_exceptions=True
        )
        
        components = {
            "database": checks[0] if not isinstance(checks[0], Exception) else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[0])),
            "redis": checks[1] if not isinstance(checks[1], Exception) else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[1])),
            "ml_models": checks[2] if not isinstance(checks[2], Exception) else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[2])),
            "external_services": checks[3] if not isinstance(checks[3], Exception) else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[3]))
        }
        
        # Determine overall status
        critical_components = ["database", "redis"]
        critical_unhealthy = any(
            components[comp].status == HealthStatus.UNHEALTHY 
            for comp in critical_components
        )
        
        any_unhealthy = any(
            comp.status == HealthStatus.UNHEALTHY 
            for comp in components.values()
        )
        
        any_degraded = any(
            comp.status == HealthStatus.DEGRADED 
            for comp in components.values()
        )
        
        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any_unhealthy or any_degraded:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return OverallHealth(
            status=overall_status,
            timestamp=time.time(),
            version=self.version,
            uptime=time.time() - self.start_time,
            components=components
        )

# Global health checker instance
health_checker = HealthChecker()

@router.get("/health", response_model=OverallHealth)
async def health_check():
    """Comprehensive health check endpoint."""
    return await health_checker.get_overall_health()

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    health = await health_checker.get_overall_health()
    
    if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    # Basic liveness check - just verify the service is responding
    return {"status": "alive", "timestamp": time.time()}
'''
        
        # Write the health check file
        health_file = self.src_dir / "prompt_improver" / "monitoring" / "health_check.py"
        health_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(health_file, 'w') as f:
            f.write(health_check_code)
        
        logger.info(f"Health check endpoint created at {health_file}")
        
    def create_monitoring_init(self):
        """Create monitoring package __init__.py file."""
        logger.info("Creating monitoring package initialization...")
        
        init_code = '''"""
APES Monitoring Package
Provides comprehensive observability for the APES ML Pipeline Orchestrator.
"""

from .metrics_middleware import PrometheusMiddleware, get_metrics
from .health_check import router as health_router

__all__ = [
    "PrometheusMiddleware",
    "get_metrics",  
    "health_router"
]
'''
        
        init_file = self.src_dir / "prompt_improver" / "monitoring" / "__init__.py"
        with open(init_file, 'w') as f:
            f.write(init_code)
            
        logger.info(f"Monitoring package initialized at {init_file}")
        
    def create_usage_examples(self):
        """Create usage examples for integrating metrics into the application."""
        logger.info("Creating usage examples...")
        
        examples_code = '''"""
Usage Examples for APES Monitoring Integration
Shows how to integrate metrics collection into the application.
"""

from fastapi import FastAPI
from prometheus_client import start_http_server, generate_latest
from prompt_improver.monitoring import PrometheusMiddleware, health_router, get_metrics

# Example 1: Basic FastAPI integration
app = FastAPI(title="APES ML Pipeline Orchestrator")

# Add metrics middleware
app.add_middleware(PrometheusMiddleware, metrics_enabled=True)

# Add health check endpoints
app.include_router(health_router, prefix="/api/v1")

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics()

# Example 2: Manual metrics recording in ML operations
from prompt_improver.monitoring.metrics_middleware import (
    record_ml_prediction,
    record_ml_error,
    record_database_query,
    record_cache_hit,
    record_cache_miss
)

async def ml_prediction_example(model_name: str, input_data: dict):
    """Example of recording ML prediction metrics."""
    import time
    
    start_time = time.time()
    
    try:
        # Your ML prediction logic here
        result = await some_ml_model.predict(input_data)
        
        # Record successful prediction
        duration = time.time() - start_time
        record_ml_prediction(model_name, "v1.0", duration)
        
        return result
        
    except Exception as e:
        # Record ML error
        record_ml_error(model_name, "v1.0", type(e).__name__)
        raise

async def database_operation_example():
    """Example of recording database operation metrics."""
    import time
    
    start_time = time.time()
    
    try:
        # Your database operation here
        result = await database.query("SELECT * FROM prompts")
        
        # Record query metrics
        duration = time.time() - start_time
        record_database_query("SELECT", "prompts", duration)
        
        return result
        
    except Exception as e:
        # Still record the query attempt
        duration = time.time() - start_time
        record_database_query("SELECT", "prompts", duration)
        raise

async def cache_operation_example(key: str):
    """Example of recording cache operation metrics."""
    from prompt_improver.utils.redis_cache import get_cache
    
    cache = await get_cache()
    
    try:
        value = await cache.get(key)
        
        if value is not None:
            record_cache_hit("redis", "user_sessions")
            return value
        else:
            record_cache_miss("redis", "user_sessions")
            # Fetch from database and cache
            value = await fetch_from_database(key)
            await cache.set(key, value, expire=3600)
            return value
            
    except Exception as e:
        record_cache_miss("redis", "user_sessions")
        raise

# Example 3: Batch processing metrics
from prompt_improver.monitoring.metrics_middleware import record_batch_queue_size

async def batch_processing_monitor():
    """Example of monitoring batch processing queues."""
    from prompt_improver.ml.optimization.batch.batch_processor import BatchProcessor
    
    processor = BatchProcessor()
    
    # Record queue size periodically
    while True:
        queue_size = await processor.get_queue_size()
        record_batch_queue_size("ml_training", queue_size)
        
        await asyncio.sleep(30)  # Update every 30 seconds

# Example 4: Starting metrics server (alternative to FastAPI endpoint)
def start_metrics_server(port: int = 8001):
    """Start standalone Prometheus metrics server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
    print(f"Metrics available at http://localhost:{port}/metrics")

# Example 5: Custom metric collectors
from prometheus_client import Gauge, Counter

# Custom business metrics
ACTIVE_USERS = Gauge('apes_active_users', 'Number of active users')
PROMPT_IMPROVEMENTS = Counter('apes_prompt_improvements_total', 'Total prompt improvements', ['improvement_type'])

def update_business_metrics():
    """Update custom business metrics."""
    # Your business logic here
    active_users = get_active_user_count()
    ACTIVE_USERS.set(active_users)
    
    # Record specific improvement types
    PROMPT_IMPROVEMENTS.labels(improvement_type='clarity').inc()
    PROMPT_IMPROVEMENTS.labels(improvement_type='specificity').inc()

if __name__ == "__main__":
    # Example usage
    import uvicorn
    
    # Start the application with metrics
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        examples_file = self.project_root / "examples" / "monitoring_integration.py"
        examples_file.parent.mkdir(exist_ok=True)
        
        with open(examples_file, 'w') as f:
            f.write(examples_code)
            
        logger.info(f"Usage examples created at {examples_file}")
    
    def update_requirements(self):
        """Update requirements to include monitoring dependencies."""
        logger.info("Updating requirements for monitoring...")
        
        monitoring_deps = [
            "prometheus_client>=0.19.0",
            "psutil>=5.9.0",  # For system metrics
        ]
        
        requirements_file = self.project_root / "requirements.txt"
        
        if requirements_file.exists():
            # Read existing requirements
            with open(requirements_file, 'r') as f:
                existing_reqs = f.read().strip().split('\n')
            
            # Add new requirements if not already present
            for dep in monitoring_deps:
                dep_name = dep.split('>=')[0].split('==')[0]
                if not any(dep_name in req for req in existing_reqs):
                    existing_reqs.append(dep)
            
            # Write updated requirements
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(existing_reqs) + '\n')
        else:
            # Create new requirements file
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(monitoring_deps) + '\n')
        
        logger.info(f"Requirements updated with monitoring dependencies")
    
    def run(self):
        """Run the complete metrics setup process."""
        logger.info("Starting APES application metrics setup...")
        
        try:
            self.create_metrics_configuration()
            self.create_metrics_middleware()
            self.create_health_check_endpoint()
            self.create_monitoring_init()
            self.create_usage_examples()
            self.update_requirements()
            
            logger.info("✅ APES application metrics setup completed successfully!")
            
            print("\n" + "="*60)
            print("APES Application Metrics Setup Complete")
            print("="*60)
            print("\nNext Steps:")
            print("1. Install monitoring dependencies:")
            print("   pip install -r requirements.txt")
            print("\n2. Integrate metrics into your FastAPI application:")
            print("   See examples/monitoring_integration.py")
            print("\n3. Add health check endpoints to your router")
            print("\n4. Configure Prometheus to scrape your application:")
            print("   - Endpoint: http://your-app:8000/metrics")
            print("   - Health: http://your-app:8000/api/v1/health")
            print("\n5. Start the monitoring stack:")
            print("   cd monitoring && ./start-monitoring.sh")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    setup = MetricsSetup()
    setup.run()


if __name__ == "__main__":
    main()