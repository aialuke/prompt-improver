"""L2 Redis Cache Service for shared caching across processes.

Provides Redis-based caching with 1-10ms response times for shared data.
Handles connection management, serialization, and error recovery gracefully.
"""

import inspect
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from coredis import Redis as CoredisRedis
else:
    try:
        from coredis import Redis as CoredisRedis
    except ImportError:
        CoredisRedis = None

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
        self._client: CoredisRedis | None = None
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

    def _track_operation(self, start_time: float, success: bool, operation: str, key: str = "") -> None:
        """Track operation performance and log slow operations."""
        response_time = time.perf_counter() - start_time
        self._total_operations += 1
        self._total_response_time += response_time
        
        if success:
            self._successful_operations += 1
        else:
            self._failed_operations += 1
            
        # Log slow operations (should be <10ms)
        if response_time > 0.01:
            logger.warning(f"L2 Redis {operation} took {response_time*1000:.2f}ms (key: {key[:50]}...)")

    async def get(self, key: str) -> Any | None:
        """Get value from Redis cache."""
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                self._track_operation(start_time, False, "GET", key)
                return None
                
            raw_value = await client.get(key)
            if raw_value is None:
                self._track_operation(start_time, True, "GET", key)
                return None
                
            value = json.loads(raw_value.decode("utf-8"))
            self._track_operation(start_time, True, "GET", key)
            return value
            
        except Exception as e:
            logger.warning(f"L2 Redis GET error for key {key}: {e}")
            self._track_operation(start_time, False, "GET", key)
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Set value in Redis cache."""
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                self._track_operation(start_time, False, "SET", key)
                return False
                
            serialized_value = json.dumps(value, default=str).encode("utf-8")
            
            if ttl_seconds and ttl_seconds > 0:
                await client.set(key, serialized_value, ex=ttl_seconds)
            else:
                await client.set(key, serialized_value)
            
            self._track_operation(start_time, True, "SET", key)
            return True
            
        except Exception as e:
            logger.warning(f"L2 Redis SET error for key {key}: {e}")
            self._track_operation(start_time, False, "SET", key)
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                self._track_operation(start_time, False, "DELETE", key)
                return False
                
            result = await client.delete([key])
            success = (result or 0) > 0
            self._track_operation(start_time, success, "DELETE", key)
            return success
            
        except Exception as e:
            logger.warning(f"L2 Redis DELETE error for key {key}: {e}")
            self._track_operation(start_time, False, "DELETE", key)
            return False

    async def clear(self) -> None:
        """Clear cache - not implemented for Redis safety."""
        logger.info("L2 Redis clear requested - use specific key deletion for safety")

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                self._track_operation(start_time, False, "EXISTS", key)
                return False
                
            result = await client.exists([key])
            exists = result > 0
            self._track_operation(start_time, True, "EXISTS", key)
            return exists
            
        except Exception as e:
            logger.warning(f"L2 Redis EXISTS error for key {key}: {e}")
            self._track_operation(start_time, False, "EXISTS", key)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate L2 cache entries matching pattern.
        
        Args:
            pattern: Pattern to match against cache keys (supports Redis glob patterns)
            
        Returns:
            Number of entries invalidated
        """
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            if client is None:
                return 0
                
            # Use Redis SCAN to find matching keys
            keys = [key async for key in client.scan_iter(match=pattern)]
            
            if keys:
                # Delete all matching keys
                result = await client.delete(keys)
                deleted_count = result or 0
            else:
                deleted_count = 0
                
            self._track_operation(start_time, True, "INVALIDATE_PATTERN", pattern)
            return deleted_count
            
        except Exception as e:
            logger.warning(f"L2 Redis pattern invalidation failed '{pattern}': {e}")
            self._track_operation(start_time, False, "INVALIDATE_PATTERN", pattern)
            return 0

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

    async def _get_client(self) -> CoredisRedis | None:
        """Get or create Redis client with connection management.
        
        Returns:
            Redis client or None if connection failed
        """
        if self._client is not None:
            return self._client

        try:
            if CoredisRedis is None:
                if not self._connection_error_logged:
                    logger.warning("coredis not available - L2 Redis cache disabled")
                    self._connection_error_logged = True
                return None
            
            self._connection_attempts += 1
            
            # Modern environment-based configuration with security enhancements
            redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"
            redis_ssl_cert_reqs = os.getenv("REDIS_SSL_CERT_REQS", "required")
            redis_ssl_ca_certs = os.getenv("REDIS_SSL_CA_CERTS")
            redis_ssl_certfile = os.getenv("REDIS_SSL_CERTFILE")  
            redis_ssl_keyfile = os.getenv("REDIS_SSL_KEYFILE")
            
            # SECURITY: Configure TLS/SSL for production security
            ssl_config = {}
            if redis_ssl:
                ssl_config.update({
                    "ssl": True,
                    "ssl_cert_reqs": redis_ssl_cert_reqs,
                })
                if redis_ssl_ca_certs:
                    ssl_config["ssl_ca_certs"] = redis_ssl_ca_certs
                if redis_ssl_certfile:
                    ssl_config["ssl_certfile"] = redis_ssl_certfile  
                if redis_ssl_keyfile:
                    ssl_config["ssl_keyfile"] = redis_ssl_keyfile
                    
                logger.info("Redis client configured with TLS/SSL encryption")
            else:
                logger.warning("Redis client configured WITHOUT TLS/SSL - not recommended for production")
            
            self._client = CoredisRedis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD"),
                username=os.getenv("REDIS_USERNAME"),
                socket_connect_timeout=int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")),
                socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
                decode_responses=False,
                **ssl_config,  # Apply SSL configuration
            )

            # Test connection
            await self._client.ping()
            
            logger.info("L2 Redis service connected successfully")
            self._ever_connected = True
            self._last_reconnect = True
            self._connection_error_logged = False
            
            return self._client

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
                if hasattr(self._client, "close"):
                    if inspect.iscoroutinefunction(self._client.close):
                        await self._client.close()
                    else:
                        self._client.close()
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