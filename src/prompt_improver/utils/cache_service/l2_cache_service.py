"""L2 Cache Service - Redis cache with 1-10ms response time.

High-performance Redis-based cache for shared data across multiple processes.
Optimized for 1-10ms response times with connection pooling and error resilience.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from prompt_improver.core.protocols.cache_service.cache_protocols import L2CacheServiceProtocol

logger = logging.getLogger(__name__)


class L2CacheService:
    """Redis-based L2 cache service with 1-10ms response time target."""
    
    def __init__(self) -> None:
        """Initialize L2 cache service with Redis connection."""
        self._client: Optional["coredis.Redis"] = None
        self._connection_error_logged = False
        self._ever_connected: bool = False
        self._last_reconnect: bool = False
        self._connection_failures = 0
        self._operations_count = 0
        self._response_times = []
        self._max_response_time_samples = 1000
        
    async def _get_client(self) -> Optional["coredis.Redis"]:
        """Get or create Redis client with graceful error handling."""
        if self._client is not None:
            return self._client
            
        try:
            import coredis
            
            try:
                from prompt_improver.core.config import get_config
                
                config = get_config()
                redis_config = config.redis
                self._client = coredis.Redis(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.database,
                    password=redis_config.password,
                    username=redis_config.username or None,
                    socket_connect_timeout=redis_config.connection_timeout,
                    socket_timeout=redis_config.socket_timeout,
                    max_connections=redis_config.max_connections,
                    decode_responses=False,
                )
                await self._client.ping()
                logger.info("L2 Redis cache connection established successfully")
                self._ever_connected = True
                self._last_reconnect = True
                self._connection_error_logged = False
                self._connection_failures = 0
                
            except ImportError:
                # Fallback to environment variables
                redis_host = os.getenv("REDIS_HOST", "redis.external.service")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                redis_db = int(os.getenv("REDIS_DB", "0"))
                redis_password = os.getenv("REDIS_PASSWORD")
                
                self._client = coredis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    max_connections=20,
                    decode_responses=False,
                )
                await self._client.ping()
                logger.info("L2 Redis cache connection established using environment variables")
                self._connection_error_logged = False
                self._ever_connected = True
                self._last_reconnect = True
                self._connection_failures = 0
                
        except ImportError:
            if not self._connection_error_logged:
                logger.warning("coredis not available - Redis L2 cache disabled")
                self._connection_error_logged = True
            return None
            
        except Exception as e:
            self._connection_failures += 1
            if not self._connection_error_logged:
                logger.warning(f"Failed to connect to Redis - L2 cache disabled: {e}")
                self._connection_error_logged = True
            self._client = None
            self._last_reconnect = False
            return None
            
        return self._client
    
    async def get(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from L2 cache with 1-10ms response time.
        
        Args:
            key: Cache key
            namespace: Optional namespace
            
        Returns:
            Cached value or None
        """
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            client = await self._get_client()
            if client is None:
                return None
            
            cache_key = self._build_key(key, namespace)
            result = await client.get(cache_key)
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.01:  # 10ms threshold
                logger.warning(f"L2 cache get exceeded 10ms threshold: {duration:.3f}s for key {cache_key}")
            
            if result is not None:
                return json.loads(result.decode("utf-8"))
            
            return None
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.warning(f"L2 cache get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Set value in L2 cache with 1-10ms response time.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Optional namespace
            
        Returns:
            Success status
        """
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            client = await self._get_client()
            if client is None:
                return False
            
            cache_key = self._build_key(key, namespace)
            serialized_value = json.dumps(value, default=str).encode("utf-8")
            
            if ttl and ttl > 0:
                await client.set(cache_key, serialized_value, ex=ttl)
            else:
                await client.set(cache_key, serialized_value)
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.01:  # 10ms threshold
                logger.warning(f"L2 cache set exceeded 10ms threshold: {duration:.3f}s for key {cache_key}")
            
            return True
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.warning(f"L2 cache set error for key {key}: {e}")
            return False
    
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
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            client = await self._get_client()
            if client is None:
                return {}
            
            cache_keys = [self._build_key(key, namespace) for key in keys]
            results = await client.mget(cache_keys)
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.01:  # 10ms threshold
                logger.warning(f"L2 cache mget exceeded 10ms threshold: {duration:.3f}s for {len(keys)} keys")
            
            # Build result dictionary
            result_dict = {}
            for i, (original_key, result) in enumerate(zip(keys, results)):
                if result is not None:
                    result_dict[original_key] = json.loads(result.decode("utf-8"))
            
            return result_dict
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.warning(f"L2 cache mget error: {e}")
            return {}
    
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
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            client = await self._get_client()
            if client is None:
                return False
            
            # Prepare data for mset
            cache_items = {}
            for key, value in items.items():
                cache_key = self._build_key(key, namespace)
                cache_items[cache_key] = json.dumps(value, default=str).encode("utf-8")
            
            await client.mset(cache_items)
            
            # Set TTL for each key if specified
            if ttl and ttl > 0:
                for cache_key in cache_items.keys():
                    await client.expire(cache_key, ttl)
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.01:  # 10ms threshold
                logger.warning(f"L2 cache mset exceeded 10ms threshold: {duration:.3f}s for {len(items)} items")
            
            return True
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.warning(f"L2 cache mset error: {e}")
            return False
    
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
        operation_start = time.perf_counter()
        self._operations_count += 1
        
        try:
            client = await self._get_client()
            if client is None:
                return 0
            
            search_pattern = self._build_key(pattern, namespace)
            
            # Find matching keys
            matching_keys = []
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=search_pattern, count=100)
                matching_keys.extend(keys)
                if cursor == 0:
                    break
            
            # Delete matching keys
            deleted_count = 0
            if matching_keys:
                deleted_count = await client.delete(matching_keys)
            
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            
            if duration > 0.01:  # 10ms threshold
                logger.warning(f"L2 cache delete_pattern exceeded 10ms threshold: {duration:.3f}s")
            
            return deleted_count
            
        except Exception as e:
            duration = time.perf_counter() - operation_start
            self._record_response_time(duration)
            logger.warning(f"L2 cache delete_pattern error: {e}")
            return 0
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get Redis connection status.
        
        Returns:
            Connection health and metrics
        """
        try:
            client = await self._get_client()
            if client is None:
                return {
                    "connected": False,
                    "error": "No client available",
                    "connection_failures": self._connection_failures,
                }
            
            # Test connection with ping
            ping_start = time.perf_counter()
            await client.ping()
            ping_duration = time.perf_counter() - ping_start
            
            # Get Redis info
            try:
                info = await client.info()
                redis_version = info.get("redis_version", "unknown")
                used_memory = info.get("used_memory_human", "unknown")
                connected_clients = info.get("connected_clients", 0)
            except Exception:
                redis_version = "unknown"
                used_memory = "unknown"
                connected_clients = 0
            
            return {
                "connected": True,
                "ping_duration_ms": ping_duration * 1000,
                "redis_version": redis_version,
                "used_memory": used_memory,
                "connected_clients": connected_clients,
                "connection_failures": self._connection_failures,
                "operations_count": self._operations_count,
                "avg_response_time_ms": self._get_avg_response_time() * 1000,
                "p95_response_time_ms": self._get_p95_response_time() * 1000,
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "connection_failures": self._connection_failures,
            }
    
    def _build_key(self, key: str, namespace: Optional[str]) -> str:
        """Build cache key with optional namespace."""
        if namespace:
            return f"{namespace}:{key}"
        return key
    
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
    
    async def close(self) -> None:
        """Close Redis connection gracefully."""
        if self._client is not None:
            try:
                # Prefer async close if available
                if hasattr(self._client, "close") and hasattr(self._client.close, "__call__"):
                    if hasattr(self._client.close, "__await__"):
                        await self._client.close()
                    else:
                        self._client.close()
                elif hasattr(self._client, "aclose"):
                    await self._client.aclose()
                elif hasattr(self._client, "connection_pool"):
                    pool = getattr(self._client, "connection_pool", None)
                    if pool is not None and hasattr(pool, "disconnect"):
                        try:
                            pool.disconnect()
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on L2 cache.
        
        Returns:
            Health status and performance metrics
        """
        try:
            status = await self.get_connection_status()
            
            if not status["connected"]:
                return {
                    "healthy": False,
                    "status": "unhealthy",
                    "error": status.get("error", "Connection failed"),
                    "connection_failures": self._connection_failures,
                }
            
            # Check if response times are within acceptable range
            avg_response_time = self._get_avg_response_time()
            p95_response_time = self._get_p95_response_time()
            
            health_status = "healthy"
            if p95_response_time > 0.01:  # 10ms threshold
                health_status = "degraded"
            if p95_response_time > 0.05:  # 50ms threshold
                health_status = "unhealthy"
            
            return {
                "healthy": health_status == "healthy",
                "status": health_status,
                "performance": {
                    "avg_response_time_ms": avg_response_time * 1000,
                    "p95_response_time_ms": p95_response_time * 1000,
                    "operations_count": self._operations_count,
                    "target_response_time": "1-10ms",
                },
                "connection": status,
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "unhealthy",
                "error": str(e),
            }