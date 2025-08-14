"""L1 Cache Service - In-memory cache with <1ms response time.

High-performance LRU-based in-memory cache optimized for ultra-fast access.
Designed to maintain sub-millisecond response times for all operations.
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from prompt_improver.core.protocols.cache_service.cache_protocols import L1CacheServiceProtocol

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
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
        """Update access metadata."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1


class L1CacheService:
    """High-performance in-memory LRU cache for L1 caching with <1ms response time."""
    
    def __init__(self, max_size: int = 1000) -> None:
        """Initialize L1 cache service.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._sets = 0
        self._deletes = 0
        self._namespace_cache: Dict[str, OrderedDict[str, CacheEntry]] = {}
        
    async def get(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from L1 cache with <1ms response time.
        
        Args:
            key: Cache key
            namespace: Optional namespace for isolation
            
        Returns:
            Cached value or None if not found
        """
        operation_start = time.perf_counter()
        
        try:
            cache = self._get_cache_for_namespace(namespace)
            
            if key in cache:
                entry = cache[key]
                if entry.is_expired():
                    del cache[key]
                    self._misses += 1
                    return None
                
                # Move to end (most recently used)
                cache.move_to_end(key)
                entry.touch()
                self._hits += 1
                
                duration = time.perf_counter() - operation_start
                if duration > 0.001:  # 1ms threshold
                    logger.warning(f"L1 cache get exceeded 1ms threshold: {duration:.3f}s for key {key}")
                
                return entry.value
            
            self._misses += 1
            return None
            
        except Exception as e:
            logger.error(f"L1 cache get error for key {key}: {e}")
            self._misses += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Set value in L1 cache with <1ms response time.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Optional namespace
            
        Returns:
            Success status
        """
        operation_start = time.perf_counter()
        
        try:
            cache = self._get_cache_for_namespace(namespace)
            
            if key in cache:
                # Update existing entry
                entry = cache[key]
                entry.value = value
                entry.created_at = datetime.now(UTC)
                entry.ttl_seconds = ttl
                cache.move_to_end(key)
            else:
                # Add new entry, evict if necessary
                if len(cache) >= self._max_size:
                    cache.popitem(last=False)
                    self._evictions += 1
                
                entry = CacheEntry(
                    value=value,
                    created_at=datetime.now(UTC),
                    last_accessed=datetime.now(UTC),
                    ttl_seconds=ttl,
                )
                cache[key] = entry
            
            self._sets += 1
            
            duration = time.perf_counter() - operation_start
            if duration > 0.001:  # 1ms threshold
                logger.warning(f"L1 cache set exceeded 1ms threshold: {duration:.3f}s for key {key}")
            
            return True
            
        except Exception as e:
            logger.error(f"L1 cache set error for key {key}: {e}")
            return False
    
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
        try:
            cache = self._get_cache_for_namespace(namespace)
            
            if key in cache:
                del cache[key]
                self._deletes += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"L1 cache delete error for key {key}: {e}")
            return False
    
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
        try:
            if namespace is None:
                count = len(self._cache)
                self._cache.clear()
                self._hits = 0
                self._misses = 0
                self._evictions = 0
                self._sets = 0
                self._deletes = 0
                return count
            else:
                if namespace in self._namespace_cache:
                    count = len(self._namespace_cache[namespace])
                    del self._namespace_cache[namespace]
                    return count
                return 0
                
        except Exception as e:
            logger.error(f"L1 cache clear error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics.
        
        Returns:
            Cache statistics including hit rate, size, evictions
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        # Calculate memory usage estimation
        memory_usage = self._estimate_memory_usage()
        
        # Get namespace stats
        namespace_stats = {}
        for ns, cache in self._namespace_cache.items():
            namespace_stats[ns] = {
                "size": len(cache),
                "keys": list(cache.keys())[:10],  # Sample of keys
            }
        
        return {
            "cache_level": "L1_MEMORY",
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self._max_size,
            "evictions": self._evictions,
            "sets": self._sets,
            "deletes": self._deletes,
            "estimated_memory_bytes": memory_usage,
            "namespaces": namespace_stats,
            "performance_target": "<1ms",
            "last_updated": datetime.now(UTC).isoformat(),
        }
    
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
        warmed_count = 0
        
        try:
            for key in keys:
                # Skip if already in cache
                if key in self._cache:
                    continue
                
                # If source provided and callable, use it to get value
                if source and callable(source):
                    try:
                        value = await source(key) if hasattr(source, '__call__') else None
                        if value is not None:
                            await self.set(key, value)
                            warmed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to warm key {key}: {e}")
                        
        except Exception as e:
            logger.error(f"Cache warm-up error: {e}")
        
        return warmed_count
    
    def _get_cache_for_namespace(self, namespace: Optional[str]) -> OrderedDict[str, CacheEntry]:
        """Get cache instance for namespace."""
        if namespace is None:
            return self._cache
        
        if namespace not in self._namespace_cache:
            self._namespace_cache[namespace] = OrderedDict()
        
        return self._namespace_cache[namespace]
    
    def _estimate_memory_usage(self) -> int:
        """Estimate L1 cache memory usage in bytes."""
        try:
            import sys
            
            total_size = 0
            
            # Calculate main cache
            for key, entry in self._cache.items():
                total_size += sys.getsizeof(key)
                total_size += sys.getsizeof(entry.value)
                total_size += sys.getsizeof(entry)
            
            # Calculate namespace caches
            for ns, cache in self._namespace_cache.items():
                total_size += sys.getsizeof(ns)
                for key, entry in cache.items():
                    total_size += sys.getsizeof(key)
                    total_size += sys.getsizeof(entry.value)
                    total_size += sys.getsizeof(entry)
            
            return total_size
            
        except Exception:
            # Fallback estimation
            total_entries = len(self._cache) + sum(len(cache) for cache in self._namespace_cache.values())
            return total_entries * 1024  # ~1KB per entry estimate
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries across all namespaces.
        
        Returns:
            Number of entries cleaned
        """
        cleaned_count = 0
        
        try:
            # Clean main cache
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                cleaned_count += 1
            
            # Clean namespace caches
            for cache in self._namespace_cache.values():
                expired_keys = [
                    key for key, entry in cache.items()
                    if entry.is_expired()
                ]
                for key in expired_keys:
                    del cache[key]
                    cleaned_count += 1
            
        except Exception as e:
            logger.error(f"L1 cache cleanup error: {e}")
        
        return cleaned_count
    
    async def get_hot_keys(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed keys.
        
        Args:
            limit: Maximum number of keys to return
            
        Returns:
            List of hot key information
        """
        try:
            all_entries = []
            
            # Collect from main cache
            for key, entry in self._cache.items():
                all_entries.append({
                    "key": key,
                    "namespace": None,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed.isoformat(),
                    "created_at": entry.created_at.isoformat(),
                })
            
            # Collect from namespace caches
            for ns, cache in self._namespace_cache.items():
                for key, entry in cache.items():
                    all_entries.append({
                        "key": key,
                        "namespace": ns,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed.isoformat(),
                        "created_at": entry.created_at.isoformat(),
                    })
            
            # Sort by access count and return top entries
            hot_keys = sorted(all_entries, key=lambda x: x["access_count"], reverse=True)
            return hot_keys[:limit]
            
        except Exception as e:
            logger.error(f"Error getting hot keys: {e}")
            return []