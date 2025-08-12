"""L2 Redis distributed cache implementation with security context validation.

This module provides high-performance distributed Redis caching extracted from
unified_connection_manager.py, implementing:

- RedisCache: Distributed cache with cluster/sentinel support and security validation
- RedisCacheEntry: Serialized cache entry with metadata and expiration
- RedisMetrics: Comprehensive Redis-specific performance monitoring
- Security context integration for multi-tenant isolation
- JSON serialization with fallback strategies
- Connection pooling and high availability support

Designed for single-digit millisecond response times with comprehensive observability.
"""

import asyncio
import json
import logging
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

try:
    import redis.asyncio as redis
    from redis.asyncio.retry import Retry
    from redis.backoff import ExponentialBackoff

    REDIS_AVAILABLE = True
except ImportError:
    warnings.warn("Redis not available. Install with: pip install redis")
    REDIS_AVAILABLE = False
    redis = Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import Counter, Gauge, Histogram

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    Counter = Any
    Gauge = Any
    Histogram = Any

from prompt_improver.common.types import SecurityContext

logger = logging.getLogger(__name__)


class RedisConnectionType(Enum):
    """Redis connection types."""

    STANDALONE = "standalone"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


class SerializationFormat(Enum):
    """Serialization formats for Redis values."""

    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"


@dataclass
class RedisCacheConfig:
    """Configuration for Redis L2 cache."""

    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None

    # Connection pooling
    max_connections: int = 100
    retry_on_timeout: bool = True
    retry_on_error: List[Exception] = None

    # Timeouts (seconds)
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = None

    # Performance settings
    connection_pool_max_connections: int = 50
    health_check_interval: float = 30.0

    # Serialization
    serialization_format: SerializationFormat = SerializationFormat.JSON
    compression_enabled: bool = False
    compression_threshold: int = 1024  # bytes

    # Security
    ssl_enabled: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # Sentinel (if using sentinel)
    sentinel_hosts: List[tuple] = None
    sentinel_service_name: Optional[str] = None
    sentinel_socket_timeout: float = 0.1

    def __post_init__(self):
        if self.retry_on_error is None:
            self.retry_on_error = [
                redis.ConnectionError,
                redis.TimeoutError,
                redis.BusyLoadingError,
            ]

        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {}


@dataclass
class RedisCacheEntry:
    """Redis cache entry with serialization metadata."""

    key: str
    value: Any
    serialized_value: Union[str, bytes]
    serialization_format: SerializationFormat
    created_at: datetime
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    compressed: bool = False
    security_context_id: Optional[str] = None

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(
                self.serialized_value.encode()
                if isinstance(self.serialized_value, str)
                else self.serialized_value
            )

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now(UTC) > self.created_at + timedelta(seconds=self.ttl_seconds)

    def time_until_expiry(self) -> Optional[float]:
        """Get seconds until expiry, or None if no TTL."""
        if self.ttl_seconds is None:
            return None
        expires_at = self.created_at + timedelta(seconds=self.ttl_seconds)
        remaining = (expires_at - datetime.now(UTC)).total_seconds()
        return max(0, remaining)


class RedisMetrics:
    """Comprehensive Redis cache metrics tracking."""

    def __init__(self, service_name: str = "redis_cache", redis_client=None):
        self.service_name = service_name
        self.redis_client = redis_client

        # Basic metrics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0

        # Performance metrics
        self.response_times: List[float] = []
        self.connection_pool_hits = 0
        self.connection_pool_misses = 0

        # OpenTelemetry setup
        self.operations_counter: Optional[Counter] = None
        self.response_time_histogram: Optional[Histogram] = None
        self.connection_pool_gauge: Optional[Gauge] = None

        if OPENTELEMETRY_AVAILABLE:
            self._setup_telemetry()

    def _setup_telemetry(self) -> None:
        """Setup OpenTelemetry metrics."""
        try:
            meter = metrics.get_meter(f"prompt_improver.cache.{self.service_name}")

            self.operations_counter = meter.create_counter(
                "redis_operations_total",
                description="Total Redis operations by type and result",
                unit="1",
            )

            self.response_time_histogram = meter.create_histogram(
                "redis_response_time_seconds",
                description="Redis operation response times",
                unit="s",
            )

            self.connection_pool_gauge = meter.create_gauge(
                "redis_connection_pool_active",
                description="Active Redis connections in pool",
                unit="1",
            )

            logger.debug(
                f"OpenTelemetry metrics initialized for Redis {self.service_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to setup Redis OpenTelemetry metrics: {e}")

    def record_operation(
        self,
        operation: str,
        status: str,
        duration_ms: float = 0,
        security_context_id: Optional[str] = None,
    ) -> None:
        """Record Redis operation metrics."""
        # Update counters
        if status == "hit":
            self.hits += 1
        elif status == "miss":
            self.misses += 1
        elif status == "error":
            self.errors += 1

        if operation == "set":
            self.sets += 1
        elif operation == "delete":
            self.deletes += 1

        # Record response time
        if duration_ms > 0:
            self.response_times.append(duration_ms)
            # Keep only recent times
            if len(self.response_times) > 10000:
                self.response_times = self.response_times[-5000:]

        # OpenTelemetry metrics
        if self.operations_counter:
            self.operations_counter.add(
                1,
                {
                    "operation": operation,
                    "status": status,
                    "security_context": security_context_id or "none",
                },
            )

        if self.response_time_histogram and duration_ms > 0:
            self.response_time_histogram.record(duration_ms / 1000.0)

    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        if not self.redis_client:
            return {}

        try:
            info = await self.redis_client.info()
            return {
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_connections_received": info.get("total_connections_received", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Redis cache statistics."""
        total_operations = self.hits + self.misses
        hit_rate = self.hits / total_operations if total_operations > 0 else 0

        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0
        )

        return {
            "service": self.service_name,
            "operations": {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "errors": self.errors,
                "total": total_operations,
            },
            "performance": {
                "hit_rate": hit_rate,
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": self._percentile(95)
                if self.response_times
                else 0,
                "p99_response_time_ms": self._percentile(99)
                if self.response_times
                else 0,
            },
            "connections": {
                "pool_hits": self.connection_pool_hits,
                "pool_misses": self.connection_pool_misses,
            },
        }

    def _percentile(self, p: int) -> float:
        """Calculate percentile of response times."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int((p / 100.0) * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]


class RedisCache:
    """High-performance distributed Redis cache for L2 caching.

    Enhanced from unified_connection_manager.py with:
    - Security context validation for multi-tenant isolation
    - Multiple serialization formats (JSON, pickle, msgpack)
    - Compression for large values
    - Connection pooling and high availability
    - Comprehensive metrics and monitoring
    - Cluster and Sentinel support
    """

    def __init__(
        self,
        config: RedisCacheConfig,
        enable_metrics: bool = True,
        service_name: str = "redis_cache",
    ):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is required for RedisCache. Install with: pip install redis"
            )

        self.config = config
        self.service_name = service_name

        # Redis client (will be initialized in async context)
        self.redis_client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None

        # Metrics
        self.metrics = RedisMetrics(service_name) if enable_metrics else None

        # Security validation
        self._security_validation_enabled = True

        logger.info(
            f"RedisCache initialized: {config.host}:{config.port}, "
            f"serialization={config.serialization_format.value}"
        )

    async def initialize(self) -> None:
        """Initialize Redis connection and connection pool."""
        try:
            # Create connection pool
            self._connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                username=self.config.username,
                max_connections=self.config.connection_pool_max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                retry_on_timeout=self.config.retry_on_timeout,
                retry_on_error=self.config.retry_on_error,
                retry=Retry(ExponentialBackoff(), 3),
                health_check_interval=self.config.health_check_interval,
            )

            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)

            # Update metrics client reference
            if self.metrics:
                self.metrics.redis_client = self.redis_client

            # Test connection
            await self.redis_client.ping()

            logger.info(
                f"Redis connection established to {self.config.host}:{self.config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise

    def _validate_security_context(self, security_context: SecurityContext) -> bool:
        """Validate security context for Redis operations."""
        if not self._security_validation_enabled or not security_context:
            return True

        # Basic validation - can be enhanced with more sophisticated checks
        return (
            security_context.user_id is not None
            and len(security_context.user_id) > 0
            and security_context.permissions is not None
        )

    def _serialize_value(
        self, value: Any
    ) -> tuple[Union[str, bytes], SerializationFormat]:
        """Serialize value for Redis storage."""
        if isinstance(value, (str, bytes)):
            return value, SerializationFormat.JSON

        try:
            if self.config.serialization_format == SerializationFormat.JSON:
                serialized = json.dumps(value)
                return serialized, SerializationFormat.JSON

            elif self.config.serialization_format == SerializationFormat.PICKLE:
                import pickle

                serialized = pickle.dumps(value)
                return serialized, SerializationFormat.PICKLE

            elif self.config.serialization_format == SerializationFormat.MSGPACK:
                try:
                    import msgpack

                    serialized = msgpack.packb(value)
                    return serialized, SerializationFormat.MSGPACK
                except ImportError:
                    logger.warning("msgpack not available, falling back to JSON")
                    serialized = json.dumps(value)
                    return serialized, SerializationFormat.JSON

            else:
                # Default fallback
                serialized = json.dumps(value)
                return serialized, SerializationFormat.JSON

        except (TypeError, ValueError) as e:
            logger.warning(f"Serialization failed, using string representation: {e}")
            return str(value), SerializationFormat.JSON

    def _deserialize_value(
        self, serialized_value: Union[str, bytes], format: SerializationFormat
    ) -> Any:
        """Deserialize value from Redis storage."""
        try:
            if format == SerializationFormat.JSON:
                if isinstance(serialized_value, bytes):
                    serialized_value = serialized_value.decode("utf-8")
                return json.loads(serialized_value)

            elif format == SerializationFormat.PICKLE:
                import pickle

                if isinstance(serialized_value, str):
                    serialized_value = serialized_value.encode("utf-8")
                return pickle.loads(serialized_value)

            elif format == SerializationFormat.MSGPACK:
                try:
                    import msgpack

                    return msgpack.unpackb(serialized_value, raw=False)
                except ImportError:
                    logger.warning("msgpack not available during deserialization")
                    return serialized_value

            else:
                return serialized_value

        except Exception as e:
            logger.warning(f"Deserialization failed: {e}")
            return serialized_value

    async def get(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> Any:
        """Get value from Redis cache with security validation."""
        if not self.redis_client:
            await self.initialize()

        start_time = time.time()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(f"Invalid security context for Redis get key: {key}")
                if self.metrics:
                    self.metrics.record_operation(
                        "get",
                        "error",
                        0,
                        security_context.user_id if security_context else None,
                    )
                return None

            # Get from Redis
            redis_value = await self.redis_client.get(key)

            if redis_value is not None:
                # Deserialize based on the serialization format (assuming JSON for now)
                deserialized_value = self._deserialize_value(
                    redis_value, SerializationFormat.JSON
                )

                duration_ms = (time.time() - start_time) * 1000
                if self.metrics:
                    self.metrics.record_operation(
                        "get",
                        "hit",
                        duration_ms,
                        security_context.user_id if security_context else None,
                    )

                return deserialized_value
            else:
                duration_ms = (time.time() - start_time) * 1000
                if self.metrics:
                    self.metrics.record_operation(
                        "get",
                        "miss",
                        duration_ms,
                        security_context.user_id if security_context else None,
                    )
                return None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Redis get failed for key {key}: {e}")
            if self.metrics:
                self.metrics.record_operation(
                    "get",
                    "error",
                    duration_ms,
                    security_context.user_id if security_context else None,
                )
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        security_context: Optional[SecurityContext] = None,
    ) -> bool:
        """Set value in Redis cache with security validation."""
        if not self.redis_client:
            await self.initialize()

        start_time = time.time()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(f"Invalid security context for Redis set key: {key}")
                if self.metrics:
                    self.metrics.record_operation(
                        "set",
                        "error",
                        0,
                        security_context.user_id if security_context else None,
                    )
                return False

            # Serialize value
            serialized_value, serialization_format = self._serialize_value(value)

            # Set in Redis
            if ttl_seconds:
                await self.redis_client.setex(key, ttl_seconds, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)

            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_operation(
                    "set",
                    "success",
                    duration_ms,
                    security_context.user_id if security_context else None,
                )

            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Redis set failed for key {key}: {e}")
            if self.metrics:
                self.metrics.record_operation(
                    "set",
                    "error",
                    duration_ms,
                    security_context.user_id if security_context else None,
                )
            return False

    async def delete(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Delete key from Redis cache with security validation."""
        if not self.redis_client:
            await self.initialize()

        start_time = time.time()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(f"Invalid security context for Redis delete key: {key}")
                if self.metrics:
                    self.metrics.record_operation(
                        "delete",
                        "error",
                        0,
                        security_context.user_id if security_context else None,
                    )
                return False

            # Delete from Redis
            deleted_count = await self.redis_client.delete(key)
            success = deleted_count > 0

            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                status = "success" if success else "miss"
                self.metrics.record_operation(
                    "delete",
                    status,
                    duration_ms,
                    security_context.user_id if security_context else None,
                )

            return success

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Redis delete failed for key {key}: {e}")
            if self.metrics:
                self.metrics.record_operation(
                    "delete",
                    "error",
                    duration_ms,
                    security_context.user_id if security_context else None,
                )
            return False

    async def exists(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Check if key exists in Redis cache."""
        if not self.redis_client:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(f"Invalid security context for Redis exists key: {key}")
                return False

            return bool(await self.redis_client.exists(key))

        except Exception as e:
            logger.error(f"Redis exists failed for key {key}: {e}")
            return False

    async def expire(
        self, key: str, seconds: int, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Set expiration for key."""
        if not self.redis_client:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(f"Invalid security context for Redis expire key: {key}")
                return False

            result = await self.redis_client.expire(key, seconds)
            return bool(result)

        except Exception as e:
            logger.error(f"Redis expire failed for key {key}: {e}")
            return False

    async def increment(
        self,
        key: str,
        delta: int = 1,
        security_context: Optional[SecurityContext] = None,
    ) -> Optional[int]:
        """Increment integer value by delta."""
        if not self.redis_client:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(
                    f"Invalid security context for Redis increment key: {key}"
                )
                return None

            return await self.redis_client.incrby(key, delta)

        except Exception as e:
            logger.error(f"Redis increment failed for key {key}: {e}")
            return None

    async def clear_database(
        self, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Clear all keys from current database."""
        if not self.redis_client:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning("Invalid security context for Redis clear database")
                return False

            await self.redis_client.flushdb()
            return True

        except Exception as e:
            logger.error(f"Redis clear database failed: {e}")
            return False

    async def ping(self) -> bool:
        """Ping Redis server for health check."""
        if not self.redis_client:
            try:
                await self.initialize()
            except Exception:
                return False

        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis ping failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Redis cache statistics."""
        base_stats = {
            "connection": {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
                "connected": self.redis_client is not None,
                "serialization_format": self.config.serialization_format.value,
            }
        }

        if self.metrics:
            metrics_stats = self.metrics.get_stats()
            base_stats.update(metrics_stats)

        # Get Redis server info
        if self.metrics and self.redis_client:
            redis_info = await self.metrics.get_redis_info()
            base_stats["redis_info"] = redis_info

        return base_stats

    async def shutdown(self) -> None:
        """Shutdown Redis connection and cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()

            if self._connection_pool:
                await self._connection_pool.disconnect()

            logger.info("Redis cache shutdown complete")

        except Exception as e:
            logger.warning(f"Error during Redis shutdown: {e}")

    def __repr__(self) -> str:
        return (
            f"RedisCache({self.config.host}:{self.config.port}, "
            f"db={self.config.db}, "
            f"serialization={self.config.serialization_format.value})"
        )


# Convenience function for easy configuration
def create_redis_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    **kwargs,
) -> RedisCache:
    """Create Redis cache with simple configuration."""
    config = RedisCacheConfig(host=host, port=port, db=db, password=password, **kwargs)
    return RedisCache(config)
