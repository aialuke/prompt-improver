"""L1 Cache Service for ultra-fast in-memory caching.

Provides high-performance LRU cache with <1ms response times for critical
operations. Focused solely on memory-based caching without external dependencies.
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata for performance tracking."""
    
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now(UTC) > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        """Update access metadata for performance tracking."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1


class L1CacheService:
    """High-performance in-memory LRU cache service.
    
    Designed for <1ms response times with optimized data structures and
    minimal overhead. Focused solely on memory-based caching operations.
    
    Performance targets:
    - GET operations: <0.5ms
    - SET operations: <0.5ms  
    - Memory efficiency: <1KB per entry overhead
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize L1 cache service.
        
        Args:
            max_size: Maximum number of entries in cache
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._total_operations = 0
        self._total_response_time = 0.0
        self._created_at = datetime.now(UTC)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with performance tracking.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.perf_counter()
        
        try:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self._misses += 1
                    return None
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                entry.touch()
                self._hits += 1
                
                return entry.value
            
            self._misses += 1
            return None
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time
            
            # Log slow operations (should be <1ms)
            if response_time > 0.001:
                logger.warning(
                    f"L1 cache GET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache with LRU eviction.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True (always successful for in-memory cache)
        """
        start_time = time.perf_counter()
        
        try:
            now = datetime.now(UTC)
            
            if key in self._cache:
                # Update existing entry
                entry = self._cache[key]
                entry.value = value
                entry.created_at = now
                entry.last_accessed = now
                entry.ttl_seconds = ttl_seconds
                self._cache.move_to_end(key)
            else:
                # Add new entry with eviction if needed
                if len(self._cache) >= self._max_size:
                    # Remove least recently used item
                    self._cache.popitem(last=False)
                
                entry = CacheEntry(
                    value=value,
                    created_at=now,
                    last_accessed=now,
                    ttl_seconds=ttl_seconds,
                )
                self._cache[key] = entry
            
            return True
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time
            
            # Log slow operations (should be <1ms)
            if response_time > 0.001:
                logger.warning(
                    f"L1 cache SET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False if not found
        """
        start_time = time.perf_counter()
        
        try:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time
            
            if response_time > 0.001:
                logger.warning(
                    f"L1 cache DELETE operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        start_time = time.perf_counter()
        
        try:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            # Preserve total_operations and total_response_time for lifetime stats
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and not expired, False otherwise
        """
        if key not in self._cache:
            return False
            
        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            return False
            
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics for monitoring.
        
        Returns:
            Dictionary with performance and utilization statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        avg_response_time = (
            self._total_response_time / self._total_operations 
            if self._total_operations > 0 else 0
        )
        
        # Calculate memory usage estimation
        memory_usage_bytes = self._estimate_memory_usage()
        
        return {
            # Core metrics
            "size": len(self._cache),
            "max_size": self._max_size,
            "utilization": len(self._cache) / self._max_size,
            
            # Performance metrics
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            
            # SLO compliance
            "slo_target_ms": 1.0,
            "slo_compliant": avg_response_time < 0.001,
            
            # Resource usage
            "estimated_memory_bytes": memory_usage_bytes,
            "memory_per_entry_bytes": memory_usage_bytes / max(len(self._cache), 1),
            
            # Operational metrics  
            "total_operations": self._total_operations,
            "uptime_seconds": (datetime.now(UTC) - self._created_at).total_seconds(),
            
            # Health indicators
            "health_status": self._get_health_status(),
        }

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache entries.
        
        Returns:
            Estimated memory usage in bytes
        """
        try:
            import sys
            
            total_size = 0
            for key, entry in self._cache.items():
                # Key size
                total_size += sys.getsizeof(key)
                # Entry object size
                total_size += sys.getsizeof(entry)
                # Value size (approximate)
                total_size += sys.getsizeof(entry.value)
                
            # Add OrderedDict overhead
            total_size += sys.getsizeof(self._cache)
            
            return total_size
            
        except Exception:
            # Fallback estimation
            return len(self._cache) * 1024  # 1KB per entry

    def _get_health_status(self) -> str:
        """Get health status based on performance metrics.
        
        Returns:
            Health status: "healthy", "degraded", or "unhealthy"
        """
        if self._total_operations == 0:
            return "healthy"
            
        avg_response_time = self._total_response_time / self._total_operations
        
        # Health thresholds
        if avg_response_time > 0.005:  # 5ms
            return "unhealthy"
        elif avg_response_time > 0.002:  # 2ms
            return "degraded"
        else:
            return "healthy"

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for dashboards.
        
        Returns:
            Performance summary with key metrics
        """
        stats = self.get_stats()
        
        return {
            "cache_level": "L1",
            "hit_rate": stats["hit_rate"],
            "avg_response_time_ms": stats["avg_response_time_ms"],
            "size": stats["size"],
            "memory_mb": stats["estimated_memory_bytes"] / (1024 * 1024),
            "slo_compliant": stats["slo_compliant"],
            "health_status": stats["health_status"],
        }

    def cleanup_expired_entries(self) -> int:
        """Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"L1 cache cleanup removed {removed_count} expired entries")
        
        return removed_count