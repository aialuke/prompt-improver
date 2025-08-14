"""Cache service protocols for multi-level caching system.

This module defines the protocol interfaces for all cache services following
clean architecture principles with protocol-based dependency injection.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CacheEntry:
    """Cache entry with metadata for tracking access patterns."""
    
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        from datetime import UTC, timedelta
        return datetime.now(UTC) > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        """Update access metadata."""
        from datetime import UTC
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1


@dataclass
class CacheStats:
    """Standardized cache statistics structure."""
    
    hits: int
    misses: int
    hit_rate: float
    size: int
    max_size: Optional[int] = None
    utilization: Optional[float] = None
    estimated_memory_bytes: Optional[int] = None


class CacheServiceProtocol(Protocol):
    """Base protocol for all cache services."""
    
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        ...
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        ...
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        ...
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        ...


class L1CacheServiceProtocol(CacheServiceProtocol, Protocol):
    """Protocol for L1 in-memory cache service."""
    
    def get_memory_usage(self) -> int:
        """Get estimated memory usage in bytes."""
        ...
    
    def get_utilization(self) -> float:
        """Get cache utilization percentage (0.0 to 1.0)."""
        ...
    
    def get_entry_count(self) -> int:
        """Get current number of cached entries."""
        ...


class L2RedisServiceProtocol(CacheServiceProtocol, Protocol):
    """Protocol for L2 Redis cache service."""
    
    async def ping(self) -> bool:
        """Check Redis connection health."""
        ...
    
    async def close(self) -> None:
        """Close Redis connection gracefully."""
        ...
    
    def is_connected(self) -> bool:
        """Check if Redis client is connected."""
        ...
    
    def get_connection_failures(self) -> int:
        """Get count of connection failures."""
        ...


class L3DatabaseServiceProtocol(Protocol):
    """Protocol for L3 database cache service."""
    
    async def get_with_fallback(
        self, 
        key: str, 
        fallback_func: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """Get value with database fallback function."""
        ...
    
    async def cache_result(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Cache a result in the database cache."""
        ...
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        ...
    
    def get_stats(self) -> CacheStats:
        """Get database cache statistics."""
        ...


class CacheMonitoringServiceProtocol(Protocol):
    """Protocol for cache monitoring and metrics service."""
    
    def record_operation(self, operation: str, duration: float, success: bool = True) -> None:
        """Record cache operation metrics."""
        ...
    
    def record_response_time(self, duration: float) -> None:
        """Record response time for SLO monitoring."""
        ...
    
    def handle_cache_error(self, operation: str, error: Exception) -> None:
        """Handle and track cache errors."""
        ...
    
    def get_health_status(self) -> Dict[str, str]:
        """Get health status for all cache components."""
        ...
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        ...
    
    def get_slo_compliance(self) -> Dict[str, Any]:
        """Get SLO compliance metrics."""
        ...
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        ...


class CacheCoordinatorProtocol(Protocol):
    """Protocol for multi-level cache coordination."""
    
    async def get(
        self,
        key: str,
        fallback_func: Optional[Callable[[], Any]] = None,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None,
    ) -> Any | None:
        """Get value from multi-level cache with fallback."""
        ...
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        l2_ttl: Optional[int] = None, 
        l1_ttl: Optional[int] = None
    ) -> None:
        """Set value in multi-level cache."""
        ...
    
    async def delete(self, key: str) -> None:
        """Delete key from all cache levels."""
        ...
    
    async def clear(self) -> None:
        """Clear all cache levels."""
        ...
    
    async def warm_cache(self, keys: List[str]) -> Dict[str, bool]:
        """Manually warm specific keys."""
        ...
    
    async def get_warming_candidates(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get current warming candidates."""
        ...
    
    async def stop_warming(self) -> None:
        """Stop background cache warming."""
        ...
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        ...
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring systems."""
        ...
    
    def get_alert_metrics(self) -> Dict[str, Any]:
        """Get metrics for alerting systems."""
        ...


class CacheFacadeProtocol(CacheCoordinatorProtocol, Protocol):
    """Protocol for unified cache facade providing backwards compatibility."""
    
    async def get_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check for monitoring systems."""
        ...