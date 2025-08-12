"""Base health checker interface and result types.

This module provides the foundation for component health monitoring extracted from
unified_connection_manager.py, implementing:

- HealthChecker: Abstract base class for component health checks
- HealthResult: Standardized result with status, timing, and error details
- HealthStatus: Status enumeration (HEALTHY, DEGRADED, UNHEALTHY)
- Component-specific health checker implementations

Designed for production reliability with comprehensive error handling.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration for component monitoring."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthResult:
    """Standardized health check result with comprehensive metadata."""

    component: str
    status: HealthStatus
    response_time_ms: float
    message: str = ""
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
        if self.details is None:
            self.details = {}

    def is_healthy(self) -> bool:
        """Check if the component is in a healthy state."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert health result to dictionary for serialization."""
        return {
            "component": self.component,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "message": self.message,
            "error": self.error,
            "details": self.details or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class AggregatedHealthResult:
    """Aggregated health results from multiple components."""

    overall_status: HealthStatus
    components: Dict[str, HealthResult]
    response_time_ms: float
    timestamp: datetime
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0

    def __post_init__(self):
        # Calculate component counts
        for result in self.components.values():
            if result.status == HealthStatus.HEALTHY:
                self.healthy_count += 1
            elif result.status == HealthStatus.DEGRADED:
                self.degraded_count += 1
            elif result.status == HealthStatus.UNHEALTHY:
                self.unhealthy_count += 1

    def get_failed_components(self) -> list[str]:
        """Get list of failed component names."""
        return [
            name
            for name, result in self.components.items()
            if result.status == HealthStatus.UNHEALTHY
        ]

    def get_degraded_components(self) -> list[str]:
        """Get list of degraded component names."""
        return [
            name
            for name, result in self.components.items()
            if result.status == HealthStatus.DEGRADED
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregated result to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_components": len(self.components),
                "healthy_count": self.healthy_count,
                "degraded_count": self.degraded_count,
                "unhealthy_count": self.unhealthy_count,
            },
            "components": {
                name: result.to_dict() for name, result in self.components.items()
            },
            "failed_components": self.get_failed_components(),
            "degraded_components": self.get_degraded_components(),
        }


class HealthChecker(ABC):
    """Abstract base class for component health checkers."""

    def __init__(self, component_name: str, timeout_seconds: float = 5.0):
        self.component_name = component_name
        self.timeout_seconds = timeout_seconds
        self.last_check_time: Optional[datetime] = None
        self.last_result: Optional[HealthResult] = None

    @abstractmethod
    async def check_health(self) -> HealthResult:
        """Perform health check and return standardized result.

        Returns:
            HealthResult with status, timing, and error details
        """
        pass

    async def check_health_with_timing(self) -> HealthResult:
        """Wrapper that adds timing and error handling to health checks."""
        start_time = time.time()

        try:
            # Call the implementation-specific health check
            result = await self.check_health()

            # Update timing
            response_time_ms = (time.time() - start_time) * 1000
            result.response_time_ms = response_time_ms

            # Cache results
            self.last_check_time = datetime.now(UTC)
            self.last_result = result

            return result

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            error_result = HealthResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message=f"Health check failed: {type(e).__name__}",
                error=str(e),
            )

            self.last_check_time = datetime.now(UTC)
            self.last_result = error_result

            logger.error(f"Health check failed for {self.component_name}: {e}")
            return error_result

    def get_cached_result(self) -> Optional[HealthResult]:
        """Get the last cached health check result."""
        return self.last_result

    def is_result_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if the cached result is stale."""
        if not self.last_check_time:
            return True

        age_seconds = (datetime.now(UTC) - self.last_check_time).total_seconds()
        return age_seconds > max_age_seconds


class DatabaseHealthChecker(HealthChecker):
    """Health checker for database connections."""

    def __init__(
        self, component_name: str, connection_pool, timeout_seconds: float = 5.0
    ):
        super().__init__(component_name, timeout_seconds)
        self.connection_pool = connection_pool

    async def check_health(self) -> HealthResult:
        """Check database connection health."""
        try:
            # Basic connectivity check
            async with self.connection_pool.acquire() as conn:
                await conn.execute("SELECT 1")

            # Pool status check
            pool_stats = {
                "size": self.connection_pool.size(),
                "checked_out": self.connection_pool.checkedout(),
                "checked_in": self.connection_pool.checkedin(),
                "invalid": self.connection_pool.invalidated(),
            }

            # Determine status based on pool utilization
            utilization = pool_stats["checked_out"] / pool_stats["size"]

            if utilization > 0.9:
                status = HealthStatus.DEGRADED
                message = f"High connection pool utilization: {utilization:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database connection healthy, utilization: {utilization:.1%}"

            return HealthResult(
                component=self.component_name,
                status=status,
                response_time_ms=0,  # Will be set by wrapper
                message=message,
                details=pool_stats,
            )

        except Exception as e:
            return HealthResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                message="Database connection failed",
                error=str(e),
            )


class RedisHealthChecker(HealthChecker):
    """Health checker for Redis connections."""

    def __init__(self, component_name: str, redis_client, timeout_seconds: float = 2.0):
        super().__init__(component_name, timeout_seconds)
        self.redis_client = redis_client

    async def check_health(self) -> HealthResult:
        """Check Redis connection health."""
        try:
            # Ping Redis
            await self.redis_client.ping()

            # Get Redis info
            info = await self.redis_client.info()
            connected_clients = info.get("connected_clients", 0)
            used_memory_human = info.get("used_memory_human", "unknown")

            return HealthResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=0,  # Will be set by wrapper
                message="Redis connection healthy",
                details={
                    "connected_clients": connected_clients,
                    "used_memory": used_memory_human,
                    "redis_version": info.get("redis_version", "unknown"),
                },
            )

        except Exception as e:
            return HealthResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                message="Redis connection failed",
                error=str(e),
            )


class CacheHealthChecker(HealthChecker):
    """Health checker for cache services."""

    def __init__(
        self, component_name: str, cache_service, timeout_seconds: float = 2.0
    ):
        super().__init__(component_name, timeout_seconds)
        self.cache_service = cache_service

    async def check_health(self) -> HealthResult:
        """Check cache service health."""
        try:
            # Test cache ping if available
            if hasattr(self.cache_service, "ping"):
                ping_result = await self.cache_service.ping()
                if not ping_result:
                    return HealthResult(
                        component=self.component_name,
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=0,
                        message="Cache ping failed",
                    )

            # Get cache statistics if available
            details = {}
            if hasattr(self.cache_service, "get_stats"):
                stats = await self.cache_service.get_stats()
                if isinstance(stats, dict):
                    details = stats

            return HealthResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=0,  # Will be set by wrapper
                message="Cache service healthy",
                details=details,
            )

        except Exception as e:
            return HealthResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                message="Cache service check failed",
                error=str(e),
            )
