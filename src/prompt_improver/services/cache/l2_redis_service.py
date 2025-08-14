"""L2 Redis Cache Service for shared caching across processes.

Provides Redis-based caching with 1-10ms response times for shared data.
Handles connection management, serialization, and error recovery gracefully.
"""

import asyncio
import contextlib
import inspect
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class L2RedisService:
    """Redis cache service for L2 caching operations.
    
    Designed for 1-10ms response times with robust connection management
    and graceful error handling. Handles serialization and Redis-specific
    operations while maintaining clean separation from other cache levels.
    
    Performance targets:
    - GET operations: <5ms
    - SET operations: <5ms
    - Connection recovery: <100ms
    """

    def __init__(self) -> None:
        """Initialize L2 Redis service."""
        self._client: Optional["coredis.Redis"] = None
        self._connection_error_logged = False
        self._ever_connected = False
        self._last_reconnect = False
        self._created_at = datetime.now(UTC)
        
        # Performance tracking
        self._total_operations = 0
        self._successful_operations = 0
        self._failed_operations = 0
        self._total_response_time = 0.0
        self._connection_attempts = 0
        self._last_health_check = None

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Deserialized value or None if not found/error
        """
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                return None
                
            raw_value = await client.get(key)
            if raw_value is None:
                return None
                
            # Deserialize value
            value = json.loads(raw_value.decode("utf-8"))
            self._successful_operations += 1
            
            return value
            
        except Exception as e:
            self._failed_operations += 1
            logger.warning(f"L2 Redis GET error for key {key}: {e}")
            return None
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time
            
            # Log slow operations (should be <10ms)
            if response_time > 0.01:
                logger.warning(
                    f"L2 Redis GET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                return False
                
            # Serialize value
            serialized_value = json.dumps(value, default=str).encode("utf-8")
            
            # Set with optional expiration
            if ttl_seconds and ttl_seconds > 0:
                await client.set(key, serialized_value, ex=ttl_seconds)
            else:
                await client.set(key, serialized_value)
            
            self._successful_operations += 1
            return True
            
        except Exception as e:
            self._failed_operations += 1
            logger.warning(f"L2 Redis SET error for key {key}: {e}")
            return False
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time
            
            if response_time > 0.01:
                logger.warning(
                    f"L2 Redis SET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False otherwise
        """
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                return False
                
            # Delete key (coredis expects list of keys)
            result = await client.delete([key])
            success = (result or 0) > 0
            
            if success:
                self._successful_operations += 1
            else:
                self._failed_operations += 1
                
            return success
            
        except Exception as e:
            self._failed_operations += 1
            logger.warning(f"L2 Redis DELETE error for key {key}: {e}")
            return False
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time

    async def clear(self) -> None:
        """Clear cache (not implemented for Redis for safety).
        
        Note: Redis FLUSHDB is dangerous and not implemented here.
        Use delete() for specific keys or implement pattern-based clearing.
        """
        logger.info("L2 Redis clear requested - use specific key deletion for safety")

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            client = await self._get_client()
            if client is None:
                return False
                
            result = await client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.warning(f"L2 Redis EXISTS error for key {key}: {e}")
            return False

    async def ping(self) -> bool:
        """Ping Redis to check connection health.
        
        Returns:
            True if Redis is responding, False otherwise
        """
        try:
            client = await self._get_client()
            if client is None:
                return False
                
            await client.ping()
            return True
            
        except Exception as e:
            logger.warning(f"L2 Redis ping failed: {e}")
            return False

    async def _get_client(self) -> Optional["coredis.Redis"]:
        """Get or create Redis client with connection management.
        
        Returns:
            Redis client or None if connection failed
        """
        if self._client is not None:
            return self._client

        try:
            import coredis
            
            self._connection_attempts += 1
            
            # Try to get configuration
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

            # Test connection
            await self._client.ping()
            
            logger.info("L2 Redis service connected successfully")
            self._ever_connected = True
            self._last_reconnect = True
            self._connection_error_logged = False
            
            return self._client

        except ImportError:
            if not self._connection_error_logged:
                logger.warning("coredis not available - L2 Redis cache disabled")
                self._connection_error_logged = True
            return None
            
        except Exception as e:
            if not self._connection_error_logged:
                logger.warning(f"Failed to connect to Redis - L2 cache disabled: {e}")
                self._connection_error_logged = True
            
            self._client = None
            self._last_reconnect = False
            return None

    async def close(self) -> None:
        """Close Redis connection gracefully."""
        if self._client is not None:
            try:
                # Try multiple close methods depending on coredis version
                if hasattr(self._client, "close") and inspect.iscoroutinefunction(
                    self._client.close
                ):
                    await self._client.close()
                elif hasattr(self._client, "aclose") and inspect.iscoroutinefunction(
                    self._client.aclose
                ):
                    await self._client.aclose()
                elif hasattr(self._client, "close"):
                    try:
                        self._client.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                elif hasattr(self._client, "connection_pool"):
                    pool = getattr(self._client, "connection_pool", None)
                    if pool is not None and hasattr(pool, "disconnect"):
                        try:
                            pool.disconnect()
                        except Exception:
                            pass
                            
            except Exception as e:
                logger.warning(f"Error closing L2 Redis connection: {e}")
            finally:
                self._client = None

    def get_stats(self) -> dict[str, Any]:
        """Get Redis cache performance statistics.
        
        Returns:
            Dictionary with performance and connection statistics
        """
        total_ops = self._total_operations
        success_rate = (
            self._successful_operations / total_ops if total_ops > 0 else 0
        )
        avg_response_time = (
            self._total_response_time / total_ops if total_ops > 0 else 0
        )
        
        return {
            # Core metrics
            "total_operations": total_ops,
            "successful_operations": self._successful_operations,
            "failed_operations": self._failed_operations,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            
            # Connection metrics
            "connection_attempts": self._connection_attempts,
            "ever_connected": self._ever_connected,
            "currently_connected": self._client is not None,
            "last_reconnect": self._last_reconnect,
            
            # SLO compliance
            "slo_target_ms": 10.0,
            "slo_compliant": avg_response_time < 0.01,
            
            # Health indicators
            "health_status": self._get_health_status(),
            "uptime_seconds": (datetime.now(UTC) - self._created_at).total_seconds(),
        }

    def _get_health_status(self) -> str:
        """Get health status based on performance and connection metrics.
        
        Returns:
            Health status: "healthy", "degraded", or "unhealthy"
        """
        if self._total_operations == 0:
            return "healthy" if self._client is not None else "degraded"
            
        success_rate = self._successful_operations / self._total_operations
        avg_response_time = self._total_response_time / self._total_operations
        
        # Health thresholds
        if not self._ever_connected or success_rate < 0.5:
            return "unhealthy"
        elif success_rate < 0.9 or avg_response_time > 0.02:  # 20ms
            return "degraded"
        else:
            return "healthy"

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check for Redis service.
        
        Returns:
            Health check results with detailed status
        """
        start_time = time.perf_counter()
        
        try:
            # Test ping
            ping_start = time.perf_counter()
            ping_success = await self.ping()
            ping_time = time.perf_counter() - ping_start
            
            # Test basic operations if ping succeeds
            operations_success = False
            operations_time = 0.0
            
            if ping_success:
                test_key = f"health_check_{int(time.time())}"
                test_value = {"test": True, "timestamp": time.time()}
                
                ops_start = time.perf_counter()
                
                # Test set
                set_success = await self.set(test_key, test_value, ttl_seconds=60)
                
                # Test get
                get_result = None
                if set_success:
                    get_result = await self.get(test_key)
                
                # Test delete
                if set_success:
                    await self.delete(test_key)
                
                operations_time = time.perf_counter() - ops_start
                operations_success = (
                    set_success and 
                    get_result is not None and 
                    get_result.get("test") is True
                )
            
            total_time = time.perf_counter() - start_time
            
            return {
                "healthy": ping_success and operations_success,
                "checks": {
                    "ping": {
                        "success": ping_success,
                        "response_time_ms": ping_time * 1000,
                    },
                    "operations": {
                        "success": operations_success,
                        "response_time_ms": operations_time * 1000,
                    },
                },
                "performance": {
                    "total_check_time_ms": total_time * 1000,
                    "meets_slo": total_time < 0.1,  # 100ms health check SLO
                },
                "stats": self.get_stats(),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"L2 Redis health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "stats": self.get_stats(),
                "timestamp": datetime.now(UTC).isoformat(),
            }