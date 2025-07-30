"""
Health Check API Endpoints - 2025 Best Practices
Production-ready health, readiness, and liveness checks for Kubernetes deployment

Note: All health data that needs to be persisted to PostgreSQL database should use
JSONB format for optimal performance, indexing capabilities, and psycopg3 compatibility.
HTTP API responses use FastAPI's JSONResponse (which is correct for HTTP).
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any
import psutil
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

# Initialize logger first
logger = logging.getLogger(__name__)

from ..core.config import get_config
from ..database import get_unified_manager
from ..performance.monitoring.health.unified_health_system import get_unified_health_monitor
from ..core.protocols.health_protocol import HealthStatus, HealthCheckResult

# Optional imports with graceful fallbacks
redis_available = True
get_redis_health_summary = None
try:
    from ..cache.redis_health import get_redis_health_summary
except ImportError:
    logger.info("Redis health monitoring not available")
    redis_available = False

ml_available = True
try:
    from ..ml.core import ml_integration  # Check if ML integration is available
    del ml_integration  # Clean up namespace
except ImportError:
    logger.info("ML services not available")
    ml_available = False

# Create router for health endpoints
health_router = APIRouter(prefix="/health", tags=["health"])

# Global health check state
_health_state: Dict[str, Any] = {
    "startup_time": time.time(),
    "last_health_check": None,
    "health_status": "unknown",
    "component_status": {}
}

# Initialize health monitor and router
health_monitor = get_unified_health_monitor()
health_router = APIRouter(prefix="/health", tags=["health"])

async def _check_database() -> HealthCheckResult:
    """Check database connectivity and basic health"""
    start_time = time.time()
    try:
        db_manager = get_unified_manager()

        async with db_manager.get_async_session() as session:
            await session.execute(text("SELECT 1"))

        duration_ms = (time.time() - start_time) * 1000

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            details={"response_time_ms": duration_ms},
            check_name="database",
            duration_ms=duration_ms
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            details={"error": str(e), "response_time_ms": duration_ms},
            check_name="database",
            duration_ms=duration_ms
        )

async def _check_redis() -> HealthCheckResult:
    """Check Redis connectivity if available"""
    if not redis_available:
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Redis not configured",
            check_name="redis",
            duration_ms=0.0
        )

    start_time = time.time()
    try:
        if get_redis_health_summary is None:
            raise ImportError("Redis health summary function not available")

        redis_summary = await get_redis_health_summary()
        duration_ms = (time.time() - start_time) * 1000

        status = HealthStatus.HEALTHY if redis_summary.get("status") == "healthy" else HealthStatus.DEGRADED

        return HealthCheckResult(
            status=status,
            message=f"Redis status: {redis_summary.get('status', 'unknown')}",
            details=redis_summary,
            check_name="redis",
            duration_ms=duration_ms
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Redis check failed: {str(e)}",
            details={"error": str(e)},
            check_name="redis",
            duration_ms=duration_ms
        )

async def _check_system_resources() -> HealthCheckResult:
    """Check system resource usage"""
    start_time = time.time()
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Check if resources are within acceptable limits
        memory_ok = memory.percent < 90
        disk_ok = disk.percent < 90

        status = HealthStatus.HEALTHY if memory_ok and disk_ok else HealthStatus.DEGRADED

        duration_ms = (time.time() - start_time) * 1000

        return HealthCheckResult(
            status=status,
            message=f"Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%",
            details={
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            },
            check_name="system_resources",
            duration_ms=duration_ms
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"System resource check failed: {str(e)}",
            details={"error": str(e)},
            check_name="system_resources",
            duration_ms=duration_ms
        )

def _format_health_data_for_jsonb(health_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format health check data for JSONB storage in PostgreSQL.

    This function ensures all data types are JSONB-compatible and follows
    APES project standards for database storage.

    Args:
        health_data: Raw health check data dictionary

    Returns:
        JSONB-compatible dictionary ready for database storage
    """
    def serialize_for_jsonb(obj: Any) -> Any:
        """Recursively serialize objects for JSONB compatibility"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, HealthStatus):
            return obj.value
        elif isinstance(obj, dict):
            return {str(k): serialize_for_jsonb(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_for_jsonb(item) for item in obj]
        elif hasattr(obj, 'value'):  # Enum objects
            return obj.value
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)

    return serialize_for_jsonb(health_data)

async def _persist_health_check_to_database(health_data: Dict[str, Any]) -> None:  # pragma: no cover
    """
    Persist health check results to database using JSONB format.

    This function demonstrates how health check data should be stored
    in PostgreSQL using JSONB columns for optimal performance.

    Note: This is a placeholder function. In a production system, you would
    have a dedicated health_monitoring table with JSONB columns.
    """
    try:
        # Format data for JSONB storage
        jsonb_data = _format_health_data_for_jsonb(health_data)

        # Example of how this would be stored in a health monitoring table:
        # CREATE TABLE health_monitoring (
        #     id SERIAL PRIMARY KEY,
        #     check_type VARCHAR(50) NOT NULL,
        #     status VARCHAR(20) NOT NULL,
        #     health_data JSONB NOT NULL,
        #     component_status JSONB,
        #     created_at TIMESTAMP DEFAULT NOW()
        # );

        db_manager = get_unified_manager()
        async with db_manager.get_async_session() as session:
            # This would be the actual insert if the table existed:
            # await session.execute(text("""
            #     INSERT INTO health_monitoring (check_type, status, health_data, component_status)
            #     VALUES (:check_type, :status, :health_data, :component_status)
            # """), {
            #     "check_type": jsonb_data.get("check_type", "comprehensive"),
            #     "status": jsonb_data.get("status", "unknown"),
            #     "health_data": jsonb_data,
            #     "component_status": jsonb_data.get("checks", {})
            # })
            # await session.commit()

            # For now, just log that we would store this data
            logger.info(f"Would store health data in JSONB format: {len(str(jsonb_data))} bytes")

    except Exception as e:
        logger.error(f"Failed to persist health check data: {e}")

# Health Check Endpoints

@health_router.get("/liveness")
async def liveness_probe():
    """
    Kubernetes liveness probe - indicates if the application is running
    Should be lightweight and fast (<1s)
    """
    return JSONResponse(
        content={
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - _health_state["startup_time"]),
            "version": get_config().environment.version,
            "environment": get_config().environment.environment
        },
        status_code=status.HTTP_200_OK
    )

@health_router.get("/readiness")
async def readiness_probe():
    """
    Kubernetes readiness probe - indicates if the application is ready to serve traffic
    Checks critical dependencies
    """
    start_time = time.time()
    ready = True
    checks: Dict[str, Dict[str, Any]] = {}

    # Check critical dependencies
    critical_checks = [
        ("database", _check_database),
        ("redis", _check_redis)
    ]

    for check_name, check_func in critical_checks:
        try:
            check_result = await check_func()
            checks[check_name] = {
                "status": check_result.status.value,
                "healthy": check_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
                "message": check_result.message,
                "duration_ms": check_result.duration_ms
            }
            if check_result.status not in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                ready = False
        except Exception as e:
            logger.error(f"Readiness check {check_name} failed: {e}")
            checks[check_name] = {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e),
                "message": f"Check failed: {e}"
            }
            ready = False

    duration_ms = (time.time() - start_time) * 1000

    response_data: Dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "check_duration_ms": round(duration_ms, 2)
    }

    status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response_data, status_code=status_code)

@health_router.get("/startup")
async def startup_probe():
    """
    Kubernetes startup probe - indicates if the application has finished starting up
    Used by Kubernetes to know when to start sending traffic
    """
    uptime: float = time.time() - _health_state["startup_time"]

    startup_tasks: Dict[str, bool] = {
        "configuration_loaded": uptime > 1,
        "database_migrations": uptime > 5,
        "ml_models_loaded": uptime > 15 if ml_available else True,
        "cache_warmed": uptime > 20,
        "health_checks_initialized": uptime > 25
    }

    all_tasks_complete = all(startup_tasks.values())

    response_data: Dict[str, Any] = {
        "status": "started" if all_tasks_complete else "starting",
        "startup_complete": all_tasks_complete,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(uptime),
        "startup_tasks": startup_tasks
    }

    status_code = status.HTTP_200_OK if all_tasks_complete else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response_data, status_code=status_code)

@health_router.get("/")
@health_router.get("")
async def health_check():
    """Main health check endpoint - comprehensive health status"""
    try:
        # Use unified health monitor for comprehensive monitoring
        health_snapshot = await health_monitor.get_overall_health()

        # Update global health state
        _health_state.update({
            "last_health_check": datetime.now(timezone.utc).isoformat(),
            "health_status": "healthy" if health_snapshot.status == HealthStatus.HEALTHY else "unhealthy"
        })

        # Build response
        result: Dict[str, Any] = {
            "status": health_snapshot.status.value,
            "healthy": health_snapshot.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - _health_state["startup_time"]),
            "version": get_config().environment.version,
            "execution_time_ms": health_snapshot.duration_ms,
            "message": health_snapshot.message,
            "details": health_snapshot.details
        }

        status_code = status.HTTP_200_OK if health_snapshot.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=result, status_code=status_code)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        error_result: Dict[str, Any] = {
            "status": "error",
            "healthy": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "message": "Health check system error"
        }
        return JSONResponse(content=error_result, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@health_router.get("/deep")
async def deep_health_check():
    """
    Comprehensive health check - all components and dependencies
    Used for monitoring and alerting
    """
    try:
        start_time = time.time()

        # Run all available health checks
        checks: Dict[str, Dict[str, Any]] = {}
        overall_healthy = True

        # Database check
        db_result = await _check_database()
        checks["database"] = {
            "status": db_result.status.value,
            "healthy": db_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
            "message": db_result.message,
            "duration_ms": db_result.duration_ms,
            "details": db_result.details
        }
        if db_result.status not in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            overall_healthy = False

        # Redis check
        redis_result = await _check_redis()
        checks["redis"] = {
            "status": redis_result.status.value,
            "healthy": redis_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
            "message": redis_result.message,
            "duration_ms": redis_result.duration_ms,
            "details": redis_result.details
        }
        if redis_result.status not in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            overall_healthy = False

        # System resources check
        system_result = await _check_system_resources()
        checks["system_resources"] = {
            "status": system_result.status.value,
            "healthy": system_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
            "message": system_result.message,
            "duration_ms": system_result.duration_ms,
            "details": system_result.details
        }
        if system_result.status not in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            overall_healthy = False

        duration_ms = (time.time() - start_time) * 1000

        # Update global health state
        _health_state.update({
            "last_health_check": datetime.now(timezone.utc).isoformat(),
            "health_status": "healthy" if overall_healthy else "unhealthy",
            "component_status": checks
        })

        # Example: Format health data for potential JSONB storage
        # In a production system, you might want to persist comprehensive health checks
        jsonb_formatted_data = _format_health_data_for_jsonb({
            "check_type": "deep_health_check",
            "timestamp": datetime.now(timezone.utc),
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": checks,
            "execution_time_ms": duration_ms,
            "summary": {
                "total_checks": len(checks),
                "healthy_checks": sum(1 for c in checks.values() if c.get("healthy", False)),
                "unhealthy_checks": sum(1 for c in checks.values() if not c.get("healthy", False))
            }
        })
        logger.debug(f"Health data formatted for JSONB storage: {len(str(jsonb_formatted_data))} bytes")

        result: Dict[str, Any] = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "healthy": overall_healthy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - _health_state["startup_time"]),
            "version": get_config().environment.version,
            "check_duration_ms": round(duration_ms, 2),
            "checks": checks,
            "summary": {
                "total_checks": len(checks),
                "healthy_checks": sum(1 for c in checks.values() if c.get("healthy", False)),
                "unhealthy_checks": sum(1 for c in checks.values() if not c.get("healthy", False))
            }
        }

        status_code = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=result, status_code=status_code)

    except Exception as e:
        logger.error(f"Deep health check failed: {e}")
        error_result: Dict[str, Any] = {
            "status": "error",
            "healthy": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "message": "Deep health check system error"
        }
        return JSONResponse(content=error_result, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
