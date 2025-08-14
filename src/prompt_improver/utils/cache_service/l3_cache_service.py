"""L3 Cache Service - Database-backed cache with 10-50ms response time.

Persistent cache layer using database storage for long-term cache entries.
Designed to maintain 10-50ms response times with repository pattern and optimization.
"""

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from prompt_improver.core.protocols.cache_service.cache_protocols import L3CacheServiceProtocol

logger = logging.getLogger(__name__)


class L3CacheService:
    """Database-backed L3 cache service with 10-50ms response time target."""
    
    def __init__(
        self,
        repository: Optional[Any] = None,
        default_ttl: timedelta = timedelta(hours=24),
        cleanup_interval: int = 3600
    ) -> None:
        """Initialize L3 cache service.
        
        Args:
            repository: Database repository for persistence
            default_ttl: Default time to live for entries
            cleanup_interval: Cleanup interval in seconds
        """
        self._repository = repository
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = datetime.now(UTC)
        self._operations_count = 0
        self._response_times = []
        self._max_response_time_samples = 1000
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_accessed_keys = {}
        
    async def get(
        self,
        key: str,
        table: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from L3 cache with 10-50ms response time.
        
        Args:
            key: Cache key
            table: Optional table/collection name
            
        Returns:
            Cached value or None
        """
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            if self._repository is None:
                logger.warning("L3 cache repository not available")
                return None
            
            # Use repository pattern for database access
            table_name = table or "cache_entries"
            
            # Query for the cache entry
            entry = await self._repository.get_by_key(
                table=table_name,
                key=key
            )
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.05:  # 50ms threshold
                logger.warning(f"L3 cache get exceeded 50ms threshold: {duration:.3f}s for key {key}")
            
            if entry is None:
                self._cache_misses += 1
                return None
            
            # Check if entry is expired
            if self._is_expired(entry):
                await self._repository.delete(table=table_name, key=key)
                self._cache_misses += 1
                return None
            
            # Update access tracking
            self._last_accessed_keys[key] = datetime.now(UTC)
            await self._update_access_metadata(key, table_name)
            
            self._cache_hits += 1
            
            # Deserialize value
            if isinstance(entry.get("value"), str):
                try:
                    return json.loads(entry["value"])
                except json.JSONDecodeError:
                    return entry["value"]
            
            return entry.get("value")
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.error(f"L3 cache get error for key {key}: {e}")
            self._cache_misses += 1
            return None
    
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
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            if self._repository is None:
                logger.warning("L3 cache repository not available")
                return False
            
            table_name = table or "cache_entries"
            ttl_to_use = ttl or self._default_ttl
            expiry_time = datetime.now(UTC) + ttl_to_use
            
            # Serialize value
            serialized_value = json.dumps(value, default=str) if not isinstance(value, str) else value
            
            # Prepare entry data
            entry_data = {
                "key": key,
                "value": serialized_value,
                "created_at": datetime.now(UTC),
                "expires_at": expiry_time,
                "last_accessed": datetime.now(UTC),
                "access_count": 1,
                "metadata": metadata or {}
            }
            
            # Use repository to persist
            success = await self._repository.upsert(
                table=table_name,
                data=entry_data
            )
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.05:  # 50ms threshold
                logger.warning(f"L3 cache set exceeded 50ms threshold: {duration:.3f}s for key {key}")
            
            if success:
                self._last_accessed_keys[key] = datetime.now(UTC)
            
            return success
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.error(f"L3 cache set error for key {key}: {e}")
            return False
    
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
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            if self._repository is None:
                logger.warning("L3 cache repository not available")
                return []
            
            table_name = table or "cache_entries"
            
            # Add expiry filter to exclude expired entries
            current_time = datetime.now(UTC)
            filters["expires_at__gt"] = current_time
            
            results = await self._repository.query(
                table=table_name,
                filters=filters,
                limit=limit
            )
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.05:  # 50ms threshold
                logger.warning(f"L3 cache query exceeded 50ms threshold: {duration:.3f}s")
            
            # Deserialize values
            processed_results = []
            for result in results:
                if isinstance(result.get("value"), str):
                    try:
                        result["value"] = json.loads(result["value"])
                    except json.JSONDecodeError:
                        pass
                processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.error(f"L3 cache query error: {e}")
            return []
    
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
        try:
            if self._repository is None:
                return False
            
            table_name = table or "cache_entries"
            
            return await self._repository.update_metadata(
                table=table_name,
                key=key,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"L3 cache metadata update error for key {key}: {e}")
            return False
    
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
        operation_start = time.perf_counter()
        
        try:
            if self._repository is None:
                return 0
            
            current_time = datetime.now(UTC)
            
            # Check if cleanup is needed based on interval
            if (current_time - self._last_cleanup).total_seconds() < self._cleanup_interval:
                return 0
            
            deleted_count = await self._repository.delete_expired(
                table="cache_entries",
                current_time=current_time,
                batch_size=batch_size
            )
            
            self._last_cleanup = current_time
            
            duration = time.perf_counter() - operation_start
            if duration > 0.05:  # 50ms threshold
                logger.warning(f"L3 cache cleanup exceeded 50ms threshold: {duration:.3f}s")
            
            if deleted_count > 0:
                logger.info(f"L3 cache cleanup removed {deleted_count} expired entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"L3 cache cleanup error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L3 cache statistics.
        
        Returns:
            Cache statistics including hit rate, size, response times
        """
        try:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
            
            # Get repository stats if available
            repo_stats = {}
            if self._repository and hasattr(self._repository, 'get_stats'):
                repo_stats = await self._repository.get_stats()
            
            # Calculate response time percentiles
            response_time_stats = self._calculate_response_time_percentiles()
            
            return {
                "cache_level": "L3_DATABASE",
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": hit_rate,
                "operations_count": self._operations_count,
                "response_times": response_time_stats,
                "repository_stats": repo_stats,
                "last_cleanup": self._last_cleanup.isoformat(),
                "performance_target": "10-50ms",
                "recently_accessed_keys": len(self._last_accessed_keys),
                "last_updated": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"L3 cache stats error: {e}")
            return {
                "cache_level": "L3_DATABASE",
                "error": str(e),
                "last_updated": datetime.now(UTC).isoformat(),
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on L3 cache.
        
        Returns:
            Health status and performance metrics
        """
        try:
            # Test basic functionality
            test_key = f"health_check_{int(time.time())}"
            test_value = {"test": True, "timestamp": time.time()}
            
            # Test set operation
            set_start = time.perf_counter()
            set_success = await self.set(test_key, test_value, ttl=timedelta(minutes=1))
            set_duration = time.perf_counter() - set_start
            
            # Test get operation
            get_start = time.perf_counter()
            get_result = await self.get(test_key) if set_success else None
            get_duration = time.perf_counter() - get_start
            
            # Cleanup test entry
            if self._repository:
                await self._repository.delete(table="cache_entries", key=test_key)
            
            # Determine health status
            avg_response_time = self._get_avg_response_time()
            p95_response_time = self._get_p95_response_time()
            
            health_status = "healthy"
            if p95_response_time > 0.05:  # 50ms threshold
                health_status = "degraded"
            if p95_response_time > 0.1:  # 100ms threshold
                health_status = "unhealthy"
            
            get_success = get_result is not None and get_result.get("test") is True
            
            return {
                "healthy": health_status == "healthy" and set_success and get_success,
                "status": health_status,
                "performance": {
                    "set_duration_ms": set_duration * 1000,
                    "get_duration_ms": get_duration * 1000,
                    "avg_response_time_ms": avg_response_time * 1000,
                    "p95_response_time_ms": p95_response_time * 1000,
                    "target_response_time": "10-50ms",
                },
                "operations": {
                    "set_success": set_success,
                    "get_success": get_success,
                    "repository_available": self._repository is not None,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        expires_at = entry.get("expires_at")
        if expires_at is None:
            return False
        
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        
        return datetime.now(UTC) > expires_at
    
    async def _update_access_metadata(self, key: str, table: str) -> None:
        """Update access metadata for cache entry."""
        try:
            if self._repository and hasattr(self._repository, 'update_access'):
                await self._repository.update_access(
                    table=table,
                    key=key,
                    access_time=datetime.now(UTC)
                )
        except Exception as e:
            logger.warning(f"Failed to update access metadata for key {key}: {e}")
    
    def _record_response_time(self, duration: float) -> None:
        """Record response time for performance monitoring."""
        self._response_times.append(duration)
        if len(self._response_times) > self._max_response_time_samples:
            self._response_times = self._response_times[-self._max_response_time_samples // 2:]
    
    def _get_avg_response_time(self) -> float:
        """Get average response time."""
        if not self._response_times:
            return 0.0
        return sum(self._response_times) / len(self._response_times)
    
    def _get_p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        if not self._response_times:
            return 0.0
        sorted_times = sorted(self._response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    def _calculate_response_time_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles for monitoring."""
        if not self._response_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
        
        import statistics
        
        sorted_times = sorted(self._response_times)
        n = len(sorted_times)
        
        return {
            "p50": sorted_times[int(n * 0.5)] if n > 0 else 0.0,
            "p95": sorted_times[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_times[int(n * 0.99)] if n > 0 else 0.0,
            "mean": statistics.mean(sorted_times),
            "max": max(sorted_times),
            "count": n,
        }
    
    async def close(self) -> None:
        """Close L3 cache service gracefully."""
        try:
            if self._repository and hasattr(self._repository, 'close'):
                await self._repository.close()
            logger.info("L3 cache service closed successfully")
        except Exception as e:
            logger.warning(f"Error closing L3 cache service: {e}")