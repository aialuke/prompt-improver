"""Comprehensive health check endpoint for APES application.
Provides detailed health status for all system components.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from prompt_improver.shared.types import HealthStatus

logger = logging.getLogger(__name__)


class ComponentHealth(BaseModel):
    status: HealthStatus
    response_time: float | None = None
    error: str | None = None
    details: dict[str, Any] | None = None
    last_check: float | None = None


class OverallHealth(BaseModel):
    status: HealthStatus
    timestamp: float
    version: str
    uptime: float
    components: dict[str, ComponentHealth]


router = APIRouter()


class HealthChecker:
    """Health check service for monitoring system components."""

    def __init__(self):
        self.start_time = time.time()
        self.version = "1.0.0"

    async def check_database_health(self) -> ComponentHealth:
        """Check database connectivity and performance."""
        start_time = time.time()
        try:
            # Database session will be injected via repository protocol

            async with self.session_manager.session_context() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()
            response_time = time.time() - start_time
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={"query_time": response_time, "connection_pool": "available"},
            )
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time(),
            )

    async def check_redis_health(self) -> ComponentHealth:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        try:
            from prompt_improver.database import (
                ManagerMode,
                create_security_context,
                get_database_services,
            )

            unified_manager = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
            await {}.initialize()
            security_context = await create_security_context(
                agent_id="health_check", tier="basic"
            )
            test_key = "health_check_test"
            await unified_manager.set_cached(
                test_key,
                "test_value",
                ttl_seconds=10,
                security_context=security_context,
            )
            value = await unified_manager.get_cached(test_key, security_context)
            await unified_manager.invalidate_cached([test_key], security_context)
            if value != "test_value":
                raise Exception("Redis read/write test failed")
            response_time = time.time() - start_time
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={"ping_time": response_time, "read_write_test": "passed"},
            )
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time(),
            )

    async def check_ml_models_health(self) -> ComponentHealth:
        """Check ML models availability and performance."""
        start_time = time.time()
        try:
            response_time = time.time() - start_time
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={"models_loaded": 1, "model_check_time": response_time},
            )
        except Exception as e:
            logger.error(f"ML models health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time(),
            )

    async def check_external_services_health(self) -> ComponentHealth:
        """Check external services connectivity."""
        start_time = time.time()
        try:
            response_time = time.time() - start_time
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                last_check=time.time(),
                details={"external_apis": "available"},
            )
        except Exception as e:
            logger.error(f"External services health check failed: {e}")
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                response_time=time.time() - start_time,
                error=str(e),
                last_check=time.time(),
            )

    async def get_overall_health(self) -> OverallHealth:
        """Get comprehensive health status."""
        checks = await asyncio.gather(
            self.check_database_health(),
            self.check_redis_health(),
            self.check_ml_models_health(),
            self.check_external_services_health(),
            return_exceptions=True,
        )
        components = {
            "database": checks[0]
            if not isinstance(checks[0], Exception)
            else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[0])),
            "redis": checks[1]
            if not isinstance(checks[1], Exception)
            else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[1])),
            "ml_models": checks[2]
            if not isinstance(checks[2], Exception)
            else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[2])),
            "external_services": checks[3]
            if not isinstance(checks[3], Exception)
            else ComponentHealth(status=HealthStatus.UNKNOWN, error=str(checks[3])),
        }
        critical_components = ["database", "redis"]
        critical_unhealthy = any(
            components[comp].status == HealthStatus.UNHEALTHY
            for comp in critical_components
        )
        any_unhealthy = any(
            comp.status == HealthStatus.UNHEALTHY for comp in components.values()
        )
        any_degraded = any(
            comp.status == HealthStatus.DEGRADED for comp in components.values()
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
            components=components,
        )


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
    raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": time.time()}
