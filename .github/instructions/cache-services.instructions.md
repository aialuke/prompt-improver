---
applyTo: "**/cache/**/*.py"
---

## Cache Implementation Guidelines

### Redis Cache Patterns
- Use async Redis client (coredis) for all cache operations
- Implement proper TTL management based on data volatility
- Design cache keys with hierarchical namespace structure
- Handle cache failures gracefully with database fallback

### Required Cache Service Pattern
```python
from typing import Optional, Any, Dict, List
import json
import hashlib
from datetime import timedelta
from coredis import Redis
from opentelemetry import trace
from src.prompt_improver.utils.logging import get_logger
from src.prompt_improver.utils.exceptions import CacheError

class PromptCacheService:
    """Redis-based caching service for prompt analysis results."""
    
    def __init__(self, redis_client: Redis, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.logger = get_logger(__name__)
        self.tracer = trace.get_tracer(__name__)
    
    async def get_analysis_result(
        self,
        prompt_hash: str,
        user_id: str,
        analysis_type: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result.
        
        Args:
            prompt_hash: SHA-256 hash of the prompt text
            user_id: User identifier for personalized caching
            analysis_type: Type of analysis performed
            
        Returns:
            Cached analysis result or None if not found
        """
        cache_key = self._build_cache_key("analysis", prompt_hash, user_id, analysis_type)
        
        with self.tracer.start_as_current_span("cache_get") as span:
            span.set_attribute("cache.key", cache_key)
            
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    span.set_attribute("cache.hit", True)
                    return json.loads(cached_data)
                else:
                    span.set_attribute("cache.hit", False)
                    return None
            except Exception as e:
                span.record_exception(e)
                self.logger.warning(f"Cache get failed: {e}", extra={"cache_key": cache_key})
                return None
```

### Cache Key Design
- Use hierarchical naming: `service:entity:identifier:qualifier`
- Include version information for schema changes
- Use consistent separator characters (`:` recommended)
- Include user context for personalized caching

### Cache Key Examples
```python
def _build_cache_key(self, *parts: str) -> str:
    """Build hierarchical cache key from parts."""
    # Examples:
    # prompt:analysis:sha256hash:user123:default
    # prompt:suggestions:sha256hash:user123:v2
    # ml:model:prediction:model_v1.2:sha256hash
    safe_parts = [str(part).replace(":", "_") for part in parts if part]
    return ":".join(safe_parts)
```

### TTL Management Strategy
```python
class CacheTTLConfig:
    """Cache TTL configuration for different data types."""
    
    PROMPT_ANALYSIS = 3600      # 1 hour - moderate volatility
    ML_PREDICTIONS = 7200       # 2 hours - stable predictions
    USER_PREFERENCES = 86400    # 24 hours - user settings
    SYSTEM_CONFIG = 300         # 5 minutes - system configuration
    RATE_LIMIT = 60            # 1 minute - rate limiting data
    
    @classmethod
    def get_ttl(cls, cache_type: str, custom_ttl: Optional[int] = None) -> int:
        """Get TTL for cache type with optional override."""
        if custom_ttl:
            return custom_ttl
        return getattr(cls, cache_type.upper(), cls.PROMPT_ANALYSIS)
```

### Cache Invalidation Patterns
```python
async def invalidate_user_analysis_cache(self, user_id: str) -> None:
    """Invalidate all analysis cache entries for a user."""
    pattern = f"prompt:analysis:*:{user_id}:*"
    
    with self.tracer.start_as_current_span("cache_invalidate") as span:
        span.set_attribute("cache.pattern", pattern)
        
        try:
            # Use Redis SCAN for safe key iteration
            async for key in self.redis.scan_iter(match=pattern, count=100):
                await self.redis.delete(key)
                
            span.set_attribute("cache.invalidated", True)
            self.logger.info(f"Invalidated cache for user: {user_id}")
        except Exception as e:
            span.record_exception(e)
            self.logger.error(f"Cache invalidation failed: {e}")
            raise CacheError(f"Failed to invalidate cache for user {user_id}")
```

### Batch Operations
```python
async def set_multiple(
    self,
    cache_entries: Dict[str, Any],
    ttl: Optional[int] = None
) -> None:
    """Set multiple cache entries in a single pipeline operation."""
    if not cache_entries:
        return
    
    pipeline = self.redis.pipeline()
    effective_ttl = ttl or self.default_ttl
    
    try:
        for key, value in cache_entries.items():
            serialized_value = json.dumps(value, default=str)
            pipeline.setex(key, effective_ttl, serialized_value)
        
        await pipeline.execute()
        self.logger.debug(f"Set {len(cache_entries)} cache entries")
    except Exception as e:
        self.logger.error(f"Batch cache set failed: {e}")
        raise CacheError("Failed to set multiple cache entries")
```

### Error Handling & Fallback
```python
async def get_with_fallback(
    self,
    cache_key: str,
    fallback_func: Callable[[], Awaitable[Any]],
    ttl: Optional[int] = None
) -> Any:
    """Get from cache with database fallback."""
    
    # Try cache first
    try:
        cached_result = await self.get(cache_key)
        if cached_result is not None:
            return cached_result
    except Exception as e:
        self.logger.warning(f"Cache get failed, using fallback: {e}")
    
    # Fallback to database
    result = await fallback_func()
    
    # Cache the result for next time
    try:
        await self.set(cache_key, result, ttl)
    except Exception as e:
        self.logger.warning(f"Cache set failed: {e}")
    
    return result
```

### Performance Monitoring
```python
async def get_cache_stats(self) -> Dict[str, Any]:
    """Get cache performance statistics."""
    try:
        info = await self.redis.info("stats")
        return {
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": self._calculate_hit_rate(info),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory_human", "unknown"),
            "total_commands_processed": info.get("total_commands_processed", 0)
        }
    except Exception as e:
        self.logger.error(f"Failed to get cache stats: {e}")
        return {"error": str(e)}

def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
    """Calculate cache hit rate percentage."""
    hits = info.get("keyspace_hits", 0)
    misses = info.get("keyspace_misses", 0)
    total = hits + misses
    return (hits / total * 100) if total > 0 else 0.0
```

### Configuration Management
```python
class CacheConfig:
    """Cache configuration with environment variable support."""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        self.socket_connect_timeout = int(os.getenv("REDIS_CONNECT_TIMEOUT", "5"))
        self.retry_on_timeout = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
        self.health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
```

### Required Imports
```python
import json
import hashlib
from typing import Optional, Any, Dict, List, Callable, Awaitable
from datetime import timedelta
import os
from coredis import Redis, ConnectionPool
from opentelemetry import trace
from src.prompt_improver.utils.logging import get_logger
from src.prompt_improver.utils.exceptions import CacheError
```

### Testing Cache Services
- Mock Redis client for unit tests
- Test cache hit/miss scenarios
- Verify TTL behavior and expiration
- Test error handling and fallback mechanisms
- Use Redis fake/mock for integration tests
