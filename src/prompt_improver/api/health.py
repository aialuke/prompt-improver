"""Health Check API Endpoints - 2025 Best Practices
Production-ready health, readiness, and liveness checks for Kubernetes deployment.

Updated to use UnifiedMonitoringFacade for consolidated monitoring.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from prompt_improver.core.config import get_config
from prompt_improver.monitoring.unified import (
    HealthStatus,
    UnifiedMonitoringFacade,
    create_monitoring_facade,
)

logger = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health", tags=["health"])

# Global state for health endpoint
_health_state: dict[str, Any] = {
    "startup_time": time.time(),
    "last_health_check": None,
    "health_status": "unknown",
    "component_status": {},
}

# Global monitoring facade instance
_monitoring_facade: UnifiedMonitoringFacade | None = None


async def _get_monitoring_facade() -> UnifiedMonitoringFacade:
    """Get or create the monitoring facade instance."""
    global _monitoring_facade
    if _monitoring_facade is None:
        _monitoring_facade = await create_monitoring_facade()
    return _monitoring_facade


# These individual check functions are replaced by the unified monitoring facade
# but kept for backward compatibility if needed


# Health data formatting and persistence is now handled by the unified monitoring facade


@health_router.get("/liveness")
@health_router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe - indicates if the application is running
    Should be lightweight and fast (<1s).
    """
    try:
        config = get_config()
        return JSONResponse(
            content={
                "status": "alive",
                "timestamp": datetime.now(UTC).isoformat(),
                "uptime_seconds": int(time.time() - _health_state["startup_time"]),
                "version": getattr(config.environment, 'version', 'unknown'),
                "environment": getattr(config.environment, 'environment', 'development'),
            },
            status_code=status.HTTP_200_OK,
        )
    except Exception as e:
        logger.exception(f"Liveness probe failed: {e}")
        return JSONResponse(
            content={
                "status": "alive",
                "timestamp": datetime.now(UTC).isoformat(),
                "uptime_seconds": int(time.time() - _health_state["startup_time"]),
                "error": str(e),
            },
            status_code=status.HTTP_200_OK,
        )


@health_router.get("/readiness")
@health_router.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe - indicates if the application is ready to serve traffic
    Checks critical dependencies using unified monitoring.
    """
    try:
        facade = await _get_monitoring_facade()
        health_summary = await facade.get_system_health()

        # Service is ready if overall status is healthy or degraded
        is_ready = health_summary.overall_status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}

        response_data: dict[str, Any] = {
            "status": "ready" if is_ready else "not_ready",
            "ready": is_ready,
            "timestamp": health_summary.timestamp.isoformat(),
            "health_status": health_summary.overall_status.value,
            "healthy_components": health_summary.healthy_components,
            "total_components": health_summary.total_components,
            "check_duration_ms": round(health_summary.check_duration_ms, 2),
        }

        if not is_ready:
            response_data["critical_issues"] = health_summary.get_critical_issues()

        status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=response_data, status_code=status_code)

    except Exception as e:
        logger.exception(f"Readiness check failed: {e}")
        return JSONResponse(
            content={
                "status": "not_ready",
                "ready": False,
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
                "message": "Readiness check system error",
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@health_router.get("/startup")
async def startup_probe():
    """Kubernetes startup probe - indicates if the application has finished starting up
    Used by Kubernetes to know when to start sending traffic.
    """
    try:
        uptime: float = time.time() - _health_state["startup_time"]

        # Check if unified monitoring system is ready
        facade = await _get_monitoring_facade()
        summary = await facade.get_monitoring_summary()

        startup_tasks: dict[str, bool] = {
            "configuration_loaded": uptime > 1,
            "database_connections": uptime > 5,
            "monitoring_initialized": len(summary.get("components", {}).get("component_names", [])) > 0,
            "health_checks_ready": uptime > 10,
            "system_stable": uptime > 15,
        }

        all_tasks_complete = all(startup_tasks.values())

        response_data: dict[str, Any] = {
            "status": "started" if all_tasks_complete else "starting",
            "startup_complete": all_tasks_complete,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": int(uptime),
            "startup_tasks": startup_tasks,
            "registered_components": summary.get("components", {}).get("registered_count", 0),
        }

        status_code = (
            status.HTTP_200_OK
            if all_tasks_complete
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(content=response_data, status_code=status_code)

    except Exception as e:
        logger.exception(f"Startup probe failed: {e}")
        return JSONResponse(
            content={
                "status": "starting",
                "startup_complete": False,
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@health_router.get("/")
@health_router.get("")
async def health_check():
    """Main health check endpoint - comprehensive health status using unified monitoring."""
    try:
        facade = await _get_monitoring_facade()
        health_summary = await facade.get_system_health()

        _health_state.update({
            "last_health_check": datetime.now(UTC).isoformat(),
            "health_status": "healthy"
            if health_summary.overall_status == HealthStatus.HEALTHY
            else "unhealthy",
            "component_status": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                }
                for name, result in health_summary.component_results.items()
            },
        })

        result: dict[str, Any] = {
            "status": health_summary.overall_status.value,
            "healthy": health_summary.overall_status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED},
            "timestamp": health_summary.timestamp.isoformat(),
            "uptime_seconds": int(time.time() - _health_state["startup_time"]),
            "execution_time_ms": round(health_summary.check_duration_ms, 2),
            "summary": {
                "total_components": health_summary.total_components,
                "healthy_components": health_summary.healthy_components,
                "degraded_components": health_summary.degraded_components,
                "unhealthy_components": health_summary.unhealthy_components,
                "health_percentage": round(health_summary.health_percentage, 2),
            },
            "components": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "category": result.category.value,
                }
                for name, result in health_summary.component_results.items()
            },
        }

        # Add critical issues if any
        critical_issues = health_summary.get_critical_issues()
        if critical_issues:
            result["critical_issues"] = critical_issues

        # Add version info if available
        try:
            config = get_config()
            result["version"] = getattr(config.environment, 'version', 'unknown')
            result["environment"] = getattr(config.environment, 'environment', 'development')
        except Exception:
            pass

        status_code = (
            status.HTTP_200_OK
            if health_summary.overall_status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(content=result, status_code=status_code)

    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        error_result: dict[str, Any] = {
            "status": "error",
            "healthy": False,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": int(time.time() - _health_state["startup_time"]),
            "error": str(e),
            "message": "Health check system error",
        }
        return JSONResponse(
            content=error_result, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@health_router.get("/deep")
async def deep_health_check():
    """Comprehensive health check - all components and dependencies
    Used for monitoring and alerting - uses unified monitoring system.
    """
    try:
        facade = await _get_monitoring_facade()

        # Get comprehensive system health
        health_summary = await facade.get_system_health()

        # Get monitoring summary for additional details
        monitoring_summary = await facade.get_monitoring_summary()

        # Update global state
        _health_state.update({
            "last_health_check": datetime.now(UTC).isoformat(),
            "health_status": "healthy" if health_summary.overall_status == HealthStatus.HEALTHY else "unhealthy",
            "component_status": {
                name: {
                    "status": result.status.value,
                    "healthy": result.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED},
                    "message": result.message,
                    "duration_ms": result.response_time_ms,
                    "details": result.details,
                }
                for name, result in health_summary.component_results.items()
            },
        })

        overall_healthy = health_summary.overall_status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}

        result: dict[str, Any] = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "healthy": overall_healthy,
            "timestamp": health_summary.timestamp.isoformat(),
            "uptime_seconds": int(time.time() - _health_state["startup_time"]),
            "check_duration_ms": round(health_summary.check_duration_ms, 2),
            "checks": {
                name: {
                    "status": result.status.value,
                    "healthy": result.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED},
                    "message": result.message,
                    "duration_ms": result.response_time_ms,
                    "category": result.category.value,
                    "details": result.details,
                }
                for name, result in health_summary.component_results.items()
            },
            "summary": {
                "total_checks": health_summary.total_components,
                "healthy_checks": health_summary.healthy_components,
                "degraded_checks": health_summary.degraded_components,
                "unhealthy_checks": health_summary.unhealthy_components,
                "unknown_checks": health_summary.unknown_components,
                "health_percentage": round(health_summary.health_percentage, 2),
            },
            "monitoring": {
                "registered_components": monitoring_summary.get("components", {}).get("registered_count", 0),
                "metrics_enabled": monitoring_summary.get("metrics", {}).get("collection_enabled", False),
                "parallel_execution": monitoring_summary.get("configuration", {}).get("parallel_execution", False),
            },
        }

        # Add critical issues
        critical_issues = health_summary.get_critical_issues()
        if critical_issues:
            result["critical_issues"] = critical_issues

        # Add version info if available
        try:
            config = get_config()
            result["version"] = getattr(config.environment, 'version', 'unknown')
            result["environment"] = getattr(config.environment, 'environment', 'development')
        except Exception:
            pass

        status_code = (
            status.HTTP_200_OK
            if overall_healthy
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(content=result, status_code=status_code)

    except Exception as e:
        logger.exception(f"Deep health check failed: {e}")
        error_result: dict[str, Any] = {
            "status": "error",
            "healthy": False,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": int(time.time() - _health_state["startup_time"]),
            "error": str(e),
            "message": "Deep health check system error",
        }
        return JSONResponse(
            content=error_result, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@health_router.get("/component/{component_name}")
async def component_health_check(component_name: str):
    """Check health of specific component using unified monitoring."""
    try:
        facade = await _get_monitoring_facade()
        result = await facade.check_component_health(component_name)

        response_data = {
            "component": component_name,
            "status": result.status.value,
            "message": result.message,
            "timestamp": result.timestamp.isoformat(),
            "response_time_ms": result.response_time_ms,
            "category": result.category.value,
            "details": result.details,
        }

        if result.error:
            response_data["error"] = result.error

        # Determine HTTP status code
        if result.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}:
            http_status = status.HTTP_200_OK
        else:
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE

        return JSONResponse(content=response_data, status_code=http_status)

    except Exception as e:
        logger.exception(f"Component health check failed for {component_name}: {e}")
        return JSONResponse(
            content={
                "component": component_name,
                "status": "error",
                "message": f"Health check failed: {e!s}",
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@health_router.get("/metrics")
async def metrics_endpoint():
    """Get current system metrics using unified monitoring."""
    try:
        facade = await _get_monitoring_facade()
        metrics = await facade.collect_all_metrics()

        # Group metrics by category
        metrics_by_category = {}
        for metric in metrics:
            category = "system" if metric.name.startswith("system.") else "application"
            if category not in metrics_by_category:
                metrics_by_category[category] = []

            metrics_by_category[category].append({
                "name": metric.name,
                "value": metric.value,
                "type": metric.metric_type.value,
                "unit": metric.unit,
                "description": metric.description,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags,
            })

        return JSONResponse(
            content={
                "timestamp": datetime.now(UTC).isoformat(),
                "total_metrics": len(metrics),
                "metrics_by_category": metrics_by_category,
            },
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.exception(f"Metrics collection failed: {e}")
        return JSONResponse(
            content={
                "error": f"Metrics collection failed: {e!s}",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@health_router.get("/summary")
async def monitoring_summary():
    """Get comprehensive monitoring summary using unified monitoring."""
    try:
        facade = await _get_monitoring_facade()
        summary = await facade.get_monitoring_summary()

        return JSONResponse(
            content=summary,
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.exception(f"Monitoring summary failed: {e}")
        return JSONResponse(
            content={
                "error": f"Monitoring summary failed: {e!s}",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@health_router.post("/cleanup")
async def cleanup_old_data():
    """Clean up old monitoring data using unified monitoring."""
    try:
        facade = await _get_monitoring_facade()
        cleaned_count = await facade.cleanup_old_monitoring_data()

        return JSONResponse(
            content={
                "message": f"Cleaned up {cleaned_count} old monitoring records",
                "records_cleaned": cleaned_count,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.exception(f"Monitoring data cleanup failed: {e}")
        return JSONResponse(
            content={
                "error": f"Cleanup failed: {e!s}",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
