"""Protocol interfaces for CacheServiceFacade decomposition.

Multi-level cache architecture with L1 (memory), L2 (Redis), and L3 (database) layers.
Performance targets: L1 <1ms, L2 1-10ms, L3 10-50ms, Overall <200ms.
"""

from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable
from datetime import datetime, timedelta
from enum import Enum


class CacheLevel(Enum):
    """Cache level enumeration."""
    L1_MEMORY = "L1_MEMORY"
    L2_REDIS = "L2_REDIS"
    L3_DATABASE = "L3_DATABASE"


@runtime_checkable
class L1CacheServiceProtocol(Protocol):
    """Protocol for L1 in-memory cache service (<1ms response time)."""
    
    async def get(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from L1 cache.
        
        Args:
            key: Cache key
            namespace: Optional namespace for isolation
            
        Returns:
            Cached value or None if not found
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Set value in L1 cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Optional namespace
            
        Returns:
            Success status
        """
        ...
    
    async def delete(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> bool:
        """Delete value from L1 cache.
        
        Args:
            key: Cache key
            namespace: Optional namespace
            
        Returns:
            Success status
        """
        ...
    
    async def clear(
        self,
        namespace: Optional[str] = None
    ) -> int:
        """Clear L1 cache.
        
        Args:
            namespace: Optional namespace to clear
            
        Returns:
            Number of entries cleared
        """
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics.
        
        Returns:
            Cache statistics including hit rate, size, evictions
        """
        ...
    
    async def warm_up(
        self,
        keys: List[str],
        source: Optional[Any] = None
    ) -> int:
        """Warm up cache with frequently accessed keys.
        
        Args:
            keys: Keys to pre-load
            source: Optional data source
            
        Returns:
            Number of keys loaded
        """
        ...


@runtime_checkable
class L2CacheServiceProtocol(Protocol):
    """Protocol for L2 Redis cache service (1-10ms response time)."""
    
    async def get(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from L2 cache.
        
        Args:
            key: Cache key
            namespace: Optional namespace
            
        Returns:
            Cached value or None
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Set value in L2 cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Optional namespace
            
        Returns:
            Success status
        """
        ...
    
    async def mget(
        self,
        keys: List[str],
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get multiple values from L2 cache.
        
        Args:
            keys: List of cache keys
            namespace: Optional namespace
            
        Returns:
            Dictionary of key-value pairs
        """
        ...
    
    async def mset(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Set multiple values in L2 cache.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            namespace: Optional namespace
            
        Returns:
            Success status
        """
        ...
    
    async def delete_pattern(
        self,
        pattern: str,
        namespace: Optional[str] = None
    ) -> int:
        """Delete keys matching pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            namespace: Optional namespace
            
        Returns:
            Number of keys deleted
        """
        ...
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get Redis connection status.
        
        Returns:
            Connection health and metrics
        """
        ...


@runtime_checkable
class L3CacheServiceProtocol(Protocol):
    """Protocol for L3 database cache service (10-50ms response time)."""
    
    async def get(
        self,
        key: str,
        table: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from L3 cache.
        
        Args:
            key: Cache key
            table: Optional table/collection name
            
        Returns:
            Cached value or None
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        table: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set value in L3 cache with metadata.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live
            table: Optional table name
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        ...
    
    async def query(
        self,
        filters: Dict[str, Any],
        table: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query L3 cache with filters.
        
        Args:
            filters: Query filters
            table: Optional table name
            limit: Result limit
            
        Returns:
            List of matching entries
        """
        ...
    
    async def update_metadata(
        self,
        key: str,
        metadata: Dict[str, Any],
        table: Optional[str] = None
    ) -> bool:
        """Update metadata for cached entry.
        
        Args:
            key: Cache key
            metadata: New metadata
            table: Optional table name
            
        Returns:
            Success status
        """
        ...
    
    async def cleanup_expired(
        self,
        batch_size: int = 100
    ) -> int:
        """Clean up expired entries.
        
        Args:
            batch_size: Cleanup batch size
            
        Returns:
            Number of entries cleaned
        """
        ...


@runtime_checkable
class CacheCoordinatorServiceProtocol(Protocol):
    """Protocol for multi-level cache coordination."""
    
    async def get(
        self,
        key: str,
        levels: Optional[List[CacheLevel]] = None,
        promote: bool = True
    ) -> Tuple[Optional[Any], CacheLevel]:
        """Get value from cache hierarchy.
        
        Args:
            key: Cache key
            levels: Specific levels to check
            promote: Whether to promote to higher levels
            
        Returns:
            Tuple of (value, level_found)
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Dict[CacheLevel, int]] = None,
        levels: Optional[List[CacheLevel]] = None
    ) -> Dict[CacheLevel, bool]:
        """Set value across cache levels.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL per cache level
            levels: Specific levels to set
            
        Returns:
            Success status per level
        """
        ...
    
    async def invalidate(
        self,
        key: str,
        cascade: bool = True
    ) -> Dict[CacheLevel, bool]:
        """Invalidate entry across all levels.
        
        Args:
            key: Cache key
            cascade: Whether to cascade invalidation
            
        Returns:
            Invalidation status per level
        """
        ...
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get multi-level cache performance metrics.
        
        Returns:
            Performance metrics including hit rates, latencies, and efficiency
        """
        ...
    
    async def optimize_cache_distribution(
        self,
        access_patterns: Dict[str, int]
    ) -> Dict[str, CacheLevel]:
        """Optimize cache distribution based on access patterns.
        
        Args:
            access_patterns: Key access frequency data
            
        Returns:
            Recommended cache level per key
        """
        ...
    
    async def health_check(self) -> Dict[CacheLevel, Dict[str, Any]]:
        """Check health of all cache levels.
        
        Returns:
            Health status per cache level
        """
        ...


@runtime_checkable
class CacheServiceFacadeProtocol(Protocol):
    """Protocol for unified CacheServiceFacade."""
    
    async def get(
        self,
        key: str,
        strategy: str = "cascade"
    ) -> Optional[Any]:
        """Get value using specified strategy.
        
        Args:
            key: Cache key
            strategy: Retrieval strategy
            
        Returns:
            Cached value or None
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        strategy: str = "write_through"
    ) -> bool:
        """Set value using specified strategy.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live
            strategy: Write strategy
            
        Returns:
            Success status
        """
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Statistics for all cache levels
        """
        ...
    
    async def configure(
        self,
        config: Dict[str, Any]
    ) -> bool:
        """Configure cache behavior.
        
        Args:
            config: Configuration parameters
            
        Returns:
            Success status
        """
        ...