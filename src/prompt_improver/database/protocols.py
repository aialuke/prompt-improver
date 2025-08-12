"""Service protocols for type-safe database architecture following 2025 best practices.

This module defines protocol interfaces for all database services, enabling
dependency injection with type safety and clean contracts.

Following research-validated best practices:
- Protocol-based interfaces for better type safety
- Enable dependency injection without concrete coupling
- Compatible with existing service implementations
- Support for async operations throughout
"""

import asyncio
from datetime import datetime
from typing import Any, AsyncContextManager, Dict, List, Optional, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from .types import (
    CacheMetrics,
    ConnectionInfo,
    ConnectionMode,
    DatabaseMetrics,
    HealthStatus,
)


class DatabaseServiceProtocol(Protocol):
    """Protocol for database connection services."""

    async def get_session(
        self, mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> AsyncContextManager[AsyncSession]:
        """Get a database session with specified access mode."""
        ...

    async def health_check(self) -> HealthStatus:
        """Check database connection health."""
        ...

    async def get_metrics(self) -> DatabaseMetrics:
        """Get database connection metrics."""
        ...

    async def initialize(self) -> None:
        """Initialize the database service."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the database service and cleanup resources."""
        ...


class CacheServiceProtocol(Protocol):
    """Protocol for cache services (L1, L2, L3)."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key."""
        ...

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    async def clear(self) -> bool:
        """Clear all cache entries."""
        ...

    async def health_check(self) -> HealthStatus:
        """Check cache service health."""
        ...

    async def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        ...


class MultiLevelCacheServiceProtocol(Protocol):
    """Protocol for multi-level cache management."""

    async def get(self, key: str, security_context: Any = None) -> Optional[Any]:
        """Get value with multi-level fallback."""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        security_context: Any = None,
    ) -> bool:
        """Set value across cache levels."""
        ...

    async def invalidate(self, key: str) -> bool:
        """Invalidate key across all cache levels."""
        ...

    async def warm_cache(self, keys: List[str]) -> Dict[str, bool]:
        """Warm cache with specified keys."""
        ...

    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get metrics from all cache levels."""
        ...


class LockServiceProtocol(Protocol):
    """Protocol for distributed locking services."""

    async def acquire(
        self, resource_id: str, timeout_seconds: float = 30.0, ttl_seconds: float = 60.0
    ) -> Optional[str]:
        """Acquire distributed lock, returns lock token if successful."""
        ...

    async def release(self, resource_id: str, lock_token: str) -> bool:
        """Release distributed lock using token."""
        ...

    async def is_locked(self, resource_id: str) -> bool:
        """Check if resource is currently locked."""
        ...

    async def extend_lock(
        self, resource_id: str, lock_token: str, ttl_seconds: float = 60.0
    ) -> bool:
        """Extend lock TTL."""
        ...

    async def health_check(self) -> HealthStatus:
        """Check lock service health."""
        ...


class PubSubServiceProtocol(Protocol):
    """Protocol for publish/subscribe messaging."""

    async def publish(self, channel: str, message: Any) -> bool:
        """Publish message to channel."""
        ...

    async def subscribe(self, channel: str, handler: Any) -> str:
        """Subscribe to channel with message handler, returns subscription ID."""
        ...

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from channel."""
        ...

    async def get_active_subscriptions(self) -> Dict[str, Any]:
        """Get active subscriptions info."""
        ...

    async def health_check(self) -> HealthStatus:
        """Check pub/sub service health."""
        ...


class HealthServiceProtocol(Protocol):
    """Protocol for health monitoring services."""

    async def check_component_health(self, component_name: str) -> HealthStatus:
        """Check health of specific component."""
        ...

    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        ...

    async def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information for all components."""
        ...

    async def register_health_check(
        self, component_name: str, check_function: Any
    ) -> bool:
        """Register health check function for component."""
        ...


class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker functionality."""

    async def call(self, func: Any, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        ...

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        ...

    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        ...

    def get_failure_count(self) -> int:
        """Get current failure count."""
        ...

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        ...


class MetricsServiceProtocol(Protocol):
    """Protocol for metrics collection and reporting."""

    def record_operation(
        self, operation_name: str, duration: float, success: bool = True
    ) -> None:
        """Record operation metrics."""
        ...

    def increment_counter(
        self, metric_name: str, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter metric."""
        ...

    def record_gauge(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record gauge metric."""
        ...

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        ...


class DatabaseServicesProtocol(Protocol):
    """Protocol for composed database services following composition pattern."""

    # Core services
    database: DatabaseServiceProtocol
    cache: MultiLevelCacheServiceProtocol
    lock_manager: LockServiceProtocol
    pubsub: PubSubServiceProtocol
    health_manager: HealthServiceProtocol
    metrics: MetricsServiceProtocol

    async def initialize_all(self) -> None:
        """Initialize all composed services."""
        ...

    async def shutdown_all(self) -> None:
        """Shutdown all composed services."""
        ...

    async def health_check_all(self) -> Dict[str, HealthStatus]:
        """Health check all services."""
        ...

    async def get_metrics_all(self) -> Dict[str, Any]:
        """Get metrics from all services."""
        ...
