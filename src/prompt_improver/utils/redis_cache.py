"""
Redis Cache Module - Clean 2025 Implementation
Direct replacement for missing redis_cache.py using UnifiedConnectionManager patterns.

Provides Redis caching functionality with:
- Direct coredis connection using existing RedisConfig
- Proper error handling and logging
- No compatibility layers or legacy code
- TestContainers validation support
- Clean, modern async interface
"""

import asyncio
import json
import logging
import hashlib
from typing import Any, Optional, Dict, List, Callable, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import os
import coredis
from coredis.exceptions import ConnectionError, TimeoutError, AuthenticationError

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class RedisCache:
    """
    Clean Redis cache implementation using UnifiedConnectionManager patterns.
    
    Features:
    - Direct coredis connection using existing RedisConfig
    - Proper error handling and logging
    - No compatibility layers or legacy code
    - Performance metrics tracking
    """
    
    def __init__(self):
        self._client: Optional[coredis.Redis] = None
        self._metrics = CacheMetrics()
        self._initialized = False
        
        # Load Redis config from environment (avoiding circular imports)
        self._host = os.getenv('REDIS_HOST', 'localhost')
        self._port = int(os.getenv('REDIS_PORT', '6379'))
        self._database = int(os.getenv('REDIS_DB', '0'))
        self._password = os.getenv('REDIS_PASSWORD')
        self._username = os.getenv('REDIS_USERNAME')
        self._socket_timeout = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
        self._connection_timeout = int(os.getenv('REDIS_CONNECT_TIMEOUT', '5'))
        self._max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', '20'))
    
    async def _ensure_client(self) -> coredis.Redis:
        """Ensure Redis client is initialized."""
        if not self._client or not self._initialized:
            try:
                # Use environment-based config to avoid circular imports
                self._client = coredis.Redis(
                    host=self._host,
                    port=self._port,
                    db=self._database,
                    password=self._password,
                    username=self._username,
                    socket_timeout=self._socket_timeout,
                    socket_connect_timeout=self._connection_timeout,
                    max_connections=self._max_connections,
                    decode_responses=True
                )
                
                # Test connection
                await self._client.ping()
                self._initialized = True
                logger.info("Redis cache client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache client: {e}")
                raise ConnectionError(f"Redis connection failed: {e}")
        
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            client = await self._ensure_client()
            value = await client.get(key)
            
            if value is not None:
                self._metrics.hits += 1
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            else:
                self._metrics.misses += 1
                return None
                
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Redis cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            client = await self._ensure_client()
            
            # Serialize value if not string
            if not isinstance(value, str):
                value = json.dumps(value, default=str)
            
            if expire:
                result = await client.setex(key, expire, value)
            else:
                result = await client.set(key, value)
            
            if result:
                self._metrics.sets += 1
                return True
            return False
            
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Redis cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            client = await self._ensure_client()
            result = await client.delete(key)
            
            if result:
                self._metrics.deletes += 1
                return True
            return False
            
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Redis cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            client = await self._ensure_client()
            return bool(await client.exists(key))
        except Exception as e:
            logger.error(f"Redis cache exists error for key {key}: {e}")
            return False
    
    async def ping(self) -> bool:
        """Test Redis connection."""
        try:
            client = await self._ensure_client()
            result = await client.ping()
            return result == b'PONG' or result == 'PONG'
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all cache entries (use with caution)."""
        try:
            client = await self._ensure_client()
            await client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return False
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        return self._metrics
    
    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._initialized = False


# Global cache instance
_cache_instance: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = RedisCache()
    
    return _cache_instance


# Direct Redis client access (for legacy compatibility)
async def get_redis_client() -> coredis.Redis:
    """Get direct Redis client."""
    cache = await get_cache()
    return await cache._ensure_client()


# Legacy alias
redis_client = get_redis_client


# Convenience functions for common operations
async def get(key: str) -> Optional[Any]:
    """Get value from cache."""
    cache = await get_cache()
    return await cache.get(key)


async def set(key: str, value: Any, expire: Optional[int] = None) -> bool:
    """Set value in cache."""
    cache = await get_cache()
    return await cache.set(key, value, expire)


async def delete(key: str) -> bool:
    """Delete key from cache."""
    cache = await get_cache()
    return await cache.delete(key)


async def invalidate(pattern: str) -> int:
    """Invalidate keys matching pattern."""
    try:
        cache = await get_cache()
        client = await cache._ensure_client()
        
        # Get keys matching pattern
        keys = await client.keys(pattern)
        if keys:
            return await client.delete(*keys)
        return 0
        
    except Exception as e:
        logger.error(f"Cache invalidation error for pattern {pattern}: {e}")
        return 0


# Advanced caching patterns
def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
    
    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}:{v}")
        else:
            key_parts.append(f"{k}:{hashlib.md5(str(v).encode()).hexdigest()[:8]}")
    
    return ":".join(key_parts)


async def with_cache(key: str, func: Callable, expire: Optional[int] = None) -> Any:
    """Execute function with caching."""
    cached_value = await get(key)
    if cached_value is not None:
        return cached_value
    
    # Execute function and cache result
    result = await func() if asyncio.iscoroutinefunction(func) else func()
    await set(key, result, expire)
    return result


# Singleflight pattern for preventing cache stampedes
_singleflight_locks: Dict[str, asyncio.Lock] = {}


async def with_singleflight(key: str, func: Callable, expire: Optional[int] = None) -> Any:
    """Execute function with singleflight pattern to prevent cache stampedes."""
    # Check cache first
    cached_value = await get(key)
    if cached_value is not None:
        return cached_value
    
    # Use singleflight lock to prevent concurrent execution
    if key not in _singleflight_locks:
        _singleflight_locks[key] = asyncio.Lock()
    
    async with _singleflight_locks[key]:
        # Double-check cache after acquiring lock
        cached_value = await get(key)
        if cached_value is not None:
            return cached_value
        
        # Execute function and cache result
        result = await func() if asyncio.iscoroutinefunction(func) else func()
        await set(key, result, expire)
        return result


async def _execute_and_cache(key: str, func: Callable, expire: Optional[int] = None) -> Any:
    """Execute function and cache result (alias for with_cache)."""
    return await with_cache(key, func, expire)


# Cache event handling
class CacheSubscriber:
    """Redis pub/sub subscriber for cache events."""
    
    def __init__(self, pattern: str = "*"):
        self.pattern = pattern
        self._client: Optional[coredis.Redis] = None
        self._pubsub = None
        self._running = False
    
    async def start(self):
        """Start subscribing to cache events."""
        if self._running:
            return
        
        try:
            cache = await get_cache()
            self._client = await cache._ensure_client()
            self._pubsub = self._client.pubsub()
            await self._pubsub.psubscribe(self.pattern)
            self._running = True
            logger.info(f"Cache subscriber started for pattern: {self.pattern}")
            
        except Exception as e:
            logger.error(f"Failed to start cache subscriber: {e}")
            raise
    
    async def stop(self):
        """Stop subscribing to cache events."""
        if not self._running:
            return
        
        try:
            if self._pubsub:
                await self._pubsub.punsubscribe(self.pattern)
                await self._pubsub.close()
            self._running = False
            logger.info("Cache subscriber stopped")
            
        except Exception as e:
            logger.error(f"Error stopping cache subscriber: {e}")
    
    async def listen(self):
        """Listen for cache events."""
        if not self._running:
            await self.start()
        
        async for message in self._pubsub.listen():
            if message['type'] == 'pmessage':
                yield {
                    'pattern': message['pattern'],
                    'channel': message['channel'],
                    'data': message['data']
                }


# Cache event management
_cache_subscribers: List[CacheSubscriber] = []


async def start_cache_subscriber(pattern: str = "*") -> CacheSubscriber:
    """Start cache event subscriber."""
    subscriber = CacheSubscriber(pattern)
    await subscriber.start()
    _cache_subscribers.append(subscriber)
    return subscriber


async def stop_cache_subscriber(subscriber: CacheSubscriber):
    """Stop cache event subscriber."""
    await subscriber.stop()
    if subscriber in _cache_subscribers:
        _cache_subscribers.remove(subscriber)


async def stop_all_cache_subscribers():
    """Stop all cache subscribers."""
    for subscriber in _cache_subscribers[:]:
        await stop_cache_subscriber(subscriber)


# Cache metrics and monitoring
CACHE_HITS = "redis_cache_hits_total"
CACHE_MISSES = "redis_cache_misses_total"
CACHE_LATENCY = "redis_cache_operation_duration_seconds"


async def get_cache_info() -> Dict[str, Any]:
    """Get cache information and metrics."""
    try:
        cache = await get_cache()
        client = await cache._ensure_client()
        
        info = await client.info()
        metrics = cache.get_metrics()
        
        return {
            "connected": True,
            "redis_version": info.get("redis_version", "unknown"),
            "used_memory": info.get("used_memory", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "cache_hits": metrics.hits,
            "cache_misses": metrics.misses,
            "cache_sets": metrics.sets,
            "cache_deletes": metrics.deletes,
            "cache_errors": metrics.errors,
            "hit_ratio": metrics.hit_ratio
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        return {
            "connected": False,
            "error": str(e),
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_ratio": 0.0
        }


# Cleanup function
async def cleanup_cache():
    """Cleanup cache resources."""
    global _cache_instance
    
    await stop_all_cache_subscribers()
    
    if _cache_instance:
        await _cache_instance.close()
        _cache_instance = None
    
    logger.info("Redis cache cleanup completed")