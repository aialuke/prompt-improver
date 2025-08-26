"""Shared context for PostgreSQL pool manager components.

Provides shared state and coordination mechanisms between the decomposed
pool manager components while maintaining clean separation of concerns.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from prompt_improver.database.services.connection.connection_metrics import (
    ConnectionMetrics,
)
from prompt_improver.services.cache.cache_facade import CacheFacade as CacheManager

if TYPE_CHECKING:
    import asyncpg
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker


class PoolState(Enum):
    """Connection pool operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    STRESSED = "stressed"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ConnectionMode(Enum):
    """Connection access modes."""
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    ADMIN = "admin"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    echo_sql: bool = False


@dataclass
class PoolConfiguration:
    """PostgreSQL pool configuration with intelligent defaults."""
    pool_size: int
    max_overflow: int
    timeout: float
    enable_ha: bool = False
    enable_circuit_breaker: bool = True
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    application_name: str = "apes_pool_manager"
    skip_connection_test: bool = False

    @classmethod
    def for_environment(
        cls, env: str, service_name: str = "pool_manager"
    ) -> "PoolConfiguration":
        """Create pool configuration optimized for environment."""
        configs = {
            "development": cls(
                pool_size=5,
                max_overflow=2,
                timeout=10.0,
                enable_circuit_breaker=True,
                application_name=f"apes_dev_{service_name}",
            ),
            "testing": cls(
                pool_size=3,
                max_overflow=1,
                timeout=5.0,
                enable_circuit_breaker=False,
                application_name=f"apes_test_{service_name}",
            ),
            "production": cls(
                pool_size=20,
                max_overflow=10,
                timeout=30.0,
                enable_ha=True,
                enable_circuit_breaker=True,
                application_name=f"apes_prod_{service_name}",
            ),
            "mcp_server": cls(
                pool_size=15,
                max_overflow=5,
                timeout=0.5,
                enable_circuit_breaker=True,
                application_name=f"apes_mcp_{service_name}",
            ),
        }
        return configs.get(env, configs["development"])


@dataclass
class ConnectionInfo:
    """Information about a database connection."""
    connection_id: str
    created_at: datetime
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))
    query_count: int = 0
    error_count: int = 0
    pool_name: str = "default"


class PoolSharedContext:
    """Shared context for coordinating pool manager components.

    Contains shared state and provides coordination mechanisms between
    ConnectionPoolCore, PoolScalingManager, and PoolMonitoringService.
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        pool_config: PoolConfiguration,
        service_name: str = "postgres_pool_manager",
        cache_manager: CacheManager | None = None,
    ) -> None:
        # Configuration
        self.db_config = db_config
        self.pool_config = pool_config
        self.service_name = service_name
        self.cache_manager = cache_manager

        # Core engine components (managed by ConnectionPoolCore)
        self.async_engine: AsyncEngine | None = None
        self.async_session_factory: async_sessionmaker | None = None
        self.pg_pools: dict[str, asyncpg.Pool] = {}

        # Initialization state
        self.is_initialized = False
        self.initialization_lock = asyncio.Lock()

        # Pool state management
        self.pool_state = PoolState.INITIALIZING
        self.health_status = HealthStatus.UNKNOWN

        # Scaling configuration and state
        self.min_pool_size = pool_config.pool_size
        self.max_pool_size = min(pool_config.pool_size * 5, 100)
        self.current_pool_size = pool_config.pool_size
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scale_cooldown_seconds = 60
        self.last_scale_time = 0

        # Circuit breaker state
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_timeout = 60

        # Metrics and monitoring
        self.metrics = ConnectionMetrics()
        self.performance_window = deque(maxlen=100)
        self.connection_registry: dict[str, ConnectionInfo] = {}
        self.connection_id_counter = 0

        # Background task handles (managed by components)
        self.background_tasks: dict[str, asyncio.Task | None] = {}

    def record_performance_event(
        self, event_type: str, duration_ms: float, success: bool
    ) -> None:
        """Record a performance event in the sliding window."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "duration_ms": duration_ms,
            "success": success,
        }
        self.performance_window.append(event)

    def is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.pool_config.enable_circuit_breaker:
            return False

        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            # Check if enough time has passed to attempt half-open
            return not time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout

        return False

    def record_circuit_breaker_failure(self, error: Exception) -> None:
        """Record a circuit breaker failure."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()

        # Update health status on repeated failures
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold * 2:
            self.health_status = HealthStatus.UNHEALTHY
            self.pool_state = PoolState.UNHEALTHY

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after successful operations."""
        if self.circuit_breaker_failures > 0:
            self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)

    def update_pool_utilization(self) -> None:
        """Update pool utilization metrics."""
        total_pool_size = self.pool_config.pool_size + self.pool_config.max_overflow
        if total_pool_size > 0:
            self.metrics.pool_utilization = (
                self.metrics.active_connections / total_pool_size * 100
            )

    def register_connection(self) -> str:
        """Register a new connection and return its ID."""
        self.connection_id_counter += 1
        connection_id = f"{self.service_name}_conn_{self.connection_id_counter}"

        info = ConnectionInfo(
            connection_id=connection_id,
            created_at=datetime.now(UTC),
            pool_name="primary",
        )

        self.connection_registry[connection_id] = info
        return connection_id

    def update_connection(self, connection_id: str, operation: str) -> None:
        """Update connection registry on operations."""
        if connection_id in self.connection_registry:
            conn_info = self.connection_registry[connection_id]
            conn_info.last_used = datetime.now(UTC)

            if operation == "checkout":
                conn_info.query_count += 1

    def unregister_connection(self, connection_id: str) -> None:
        """Unregister a connection from the registry."""
        if connection_id in self.connection_registry:
            del self.connection_registry[connection_id]

    def should_scale_up(self) -> bool:
        """Determine if pool should scale up based on current state."""
        if time.time() - self.last_scale_time < self.scale_cooldown_seconds:
            return False

        utilization = self.metrics.pool_utilization / 100.0
        return (
            utilization > self.scale_up_threshold
            and self.current_pool_size < self.max_pool_size
        )

    def should_scale_down(self) -> bool:
        """Determine if pool should scale down based on current state."""
        if time.time() - self.last_scale_time < self.scale_cooldown_seconds:
            return False

        utilization = self.metrics.pool_utilization / 100.0
        return (
            utilization < self.scale_down_threshold
            and self.current_pool_size > self.min_pool_size
            and self.metrics.avg_response_time_ms < 50
        )

    def record_scale_event(self, new_size: int) -> None:
        """Record a scaling event."""
        self.current_pool_size = new_size
        self.pool_config.pool_size = new_size
        self.last_scale_time = time.time()
        self.metrics.last_scale_event = datetime.now(UTC)

    def get_context_stats(self) -> dict[str, Any]:
        """Get shared context statistics."""
        return {
            "service": self.service_name,
            "pool_state": self.pool_state.value,
            "health_status": self.health_status.value,
            "current_pool_size": self.current_pool_size,
            "min_pool_size": self.min_pool_size,
            "max_pool_size": self.max_pool_size,
            "is_initialized": self.is_initialized,
            "circuit_breaker": {
                "enabled": self.pool_config.enable_circuit_breaker,
                "state": "open" if self.is_circuit_breaker_open() else "closed",
                "failures": self.circuit_breaker_failures,
                "threshold": self.circuit_breaker_threshold,
            },
            "connection_registry_size": len(self.connection_registry),
            "performance_window_size": len(self.performance_window),
            "background_tasks": list(self.background_tasks.keys()),
        }
