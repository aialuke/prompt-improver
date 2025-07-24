"""Redis cache utility module with async helpers and Prometheus metrics.

Provides async Redis cache operations with automatic compression (lz4) and
Prometheus metrics for monitoring cache hits/misses and latency.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Optional, Union

import lz4.frame
import redis.asyncio as redis
import yaml
from prometheus_client import Counter, Histogram, REGISTRY
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Configure logging
logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Lightweight circuit breaker for Redis operations."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = None  # Will be initialized per event loop

    def _get_lock(self):
        """Get or create the lock for current event loop."""
        try:
            # Always create a new lock to avoid event loop binding issues
            if self._lock is None or self._lock._loop != asyncio.get_event_loop():
                self._lock = asyncio.Lock()
            return self._lock
        except RuntimeError:
            # Different event loop, create new lock
            self._lock = asyncio.Lock()
            return self._lock

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        lock = self._get_lock()
        async with lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - operation blocked")

            try:
                result = await func(*args, **kwargs)
                # Success - reset circuit breaker
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

                raise e

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"

    def reset(self):
        """Manually reset the circuit breaker."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0
        logger.info("Circuit breaker manually reset")

class RedisConfig(BaseModel):
    """Redis configuration with validation and sane defaults."""

    # Connection settings
    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis server port")
    cache_db: int = Field(default=2, ge=0, le=15, description="Redis database number for cache")

    # Pool settings
    pool_size: int = Field(default=10, ge=1, description="Connection pool size")
    max_connections: int = Field(default=50, ge=1, description="Maximum connections")

    # Timeout settings
    connect_timeout: int = Field(default=5, ge=0, description="Connection timeout in seconds")
    socket_timeout: int = Field(default=5, ge=0, description="Socket timeout in seconds")

    # Keep-alive settings
    socket_keepalive: bool = Field(default=True, description="Enable socket keep-alive")
    socket_keepalive_options: dict = Field(default_factory=dict, description="Socket keep-alive options")

    # SSL settings
    use_ssl: bool = Field(default=False, description="Use SSL connection")

    # Monitoring settings
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )

    @classmethod
    def load_from_yaml(cls, path: str) -> "RedisConfig":
        """Load Redis configuration from YAML file with validation.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Validated RedisConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValidationError: If configuration is invalid
            ValueError: If configuration structure is invalid
        """
        try:
            with open(path) as file:
                data = yaml.safe_load(file)

            if not data or 'connection' not in data:
                logger.warning(f"No 'connection' section found in {path}, using defaults")
                return cls()

            connection_config = data['connection']

            # Map YAML keys to model fields
            mapped_config = {
                'host': connection_config.get('host', 'localhost'),
                'port': connection_config.get('port', 6379),
                'cache_db': connection_config.get('cache_db', 2),
                'pool_size': connection_config.get('pool_size', 10),
                'max_connections': connection_config.get('max_connections', 50),
                'connect_timeout': connection_config.get('connect_timeout', 5),
                'socket_timeout': connection_config.get('socket_timeout', 5),
                'socket_keepalive': connection_config.get('socket_keepalive', True),
                'socket_keepalive_options': connection_config.get('socket_keepalive_options', {}),
                'use_ssl': connection_config.get('ssl', {}).get('enabled', False),
                'monitoring_enabled': data.get('monitoring', {}).get('enabled', True)
            }

            config = cls(**mapped_config)
            logger.info(f"Redis configuration loaded from {path}")
            return config

        except FileNotFoundError:
            logger.warning(f"Redis config file {path} not found, using defaults")
            return cls()
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {path}: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        except ValidationError as e:
            logger.error(f"Invalid Redis configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading Redis config: {e}")
            raise ValueError(f"Failed to load Redis configuration: {e}")

    def validate_config(self) -> None:
        """Validate the Redis configuration and log warnings for potential issues."""
        warnings = []

        # Check pool size vs max connections
        if self.pool_size > self.max_connections:
            warnings.append(f"Pool size ({self.pool_size}) exceeds max connections ({self.max_connections})")

        # Check timeout values
        if self.connect_timeout > 30:
            warnings.append(f"Connect timeout ({self.connect_timeout}s) is quite high")

        if self.socket_timeout > 30:
            warnings.append(f"Socket timeout ({self.socket_timeout}s) is quite high")

        # Check for production readiness
        if self.host == "localhost" and self.monitoring_enabled:
            warnings.append("Using localhost - ensure this is appropriate for your deployment")

        if not self.use_ssl and self.host != "localhost":
            warnings.append("SSL disabled for remote connection - consider enabling for security")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Redis config warning: {warning}")

        # Log successful validation
        if not warnings:
            logger.info("Redis configuration validation passed")
        else:
            logger.info(f"Redis configuration loaded with {len(warnings)} warnings")

        # Log final configuration
        logger.info(f"Redis config: {self.host}:{self.port}, DB {self.cache_db}, pool {self.pool_size}/{self.max_connections}")

# Decorator to cache function results

def cached(ttl=3600, key_func=None):
    """Decorator to cache function results in Redis with graceful fallback.

    Args:
        ttl: Time-to-live in seconds for cached values
        key_func: Optional function to generate cache key from args/kwargs

    Returns:
        Decorator function
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            if key_func:
                try:
                    cache_key = key_func(*args, **kwargs)
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Key function failed: {e} - proceeding without cache")
                    return await fn(*args, **kwargs)
            else:
                # Generate cache key from function name and arguments
                key_data = {
                    'fn': fn.__name__,
                    'args': str(args),
                    'kwargs': str(sorted(kwargs.items()))
                }
                key_str = json.dumps(key_data, sort_keys=True, default=str)
                cache_key = f"{fn.__name__}:{hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()}"

            # Try fetching from cache first
            try:
                cached_result = await RedisCache.get(cache_key)
                if cached_result:
                    return json.loads(cached_result.decode())
            except Exception as e:
                logger.warning(f"Redis unavailable for cache get - proceeding without cache: {e}")

            # Call the function and cache the result
            result = await fn(*args, **kwargs)

            # Try to cache the result
            try:
                if result is not None:
                    cache_data = json.dumps(result, default=str).encode()
                    await RedisCache.set(cache_key, cache_data, ttl)
            except Exception as e:
                logger.warning(f"Failed to cache result - continuing: {e}")

            return result
        return wrapper
    return decorator

# Lazy import to prevent circular imports
def _get_cache_metrics():
    """Lazy initialization of cache metrics."""
    try:
        from ..performance.monitoring.metrics_registry import get_metrics_registry, StandardMetrics
        metrics_registry = get_metrics_registry()
        return {
            'hits': metrics_registry.get_or_create_counter(
                StandardMetrics.CACHE_HITS_TOTAL,
                'Total number of Redis cache hits'
            ),
            'misses': metrics_registry.get_or_create_counter(
                StandardMetrics.CACHE_MISSES_TOTAL,
                'Total number of Redis cache misses'
            ),
            'latency': metrics_registry.get_or_create_histogram(
                StandardMetrics.CACHE_OPERATION_DURATION,
                'Redis cache operation latency in seconds',
                ['operation']
            )
        }
    except ImportError:
        # Fallback to mock metrics
        class MockMetric:
            def inc(self, *args, **kwargs): pass
            def observe(self, *args, **kwargs): pass
            def labels(self, *args, **kwargs): return self

        return {
            'hits': MockMetric(),
            'misses': MockMetric(),
            'latency': MockMetric()
        }

# Initialize metrics lazily
_cache_metrics = None

def get_cache_metrics():
    global _cache_metrics
    if _cache_metrics is None:
        _cache_metrics = _get_cache_metrics()
    return _cache_metrics

def _get_legacy_metric(metric_name):
    """Get legacy metric with lazy loading."""
    metrics = get_cache_metrics()
    return metrics.get(metric_name, metrics['hits'])  # Fallback to hits metric

CACHE_HITS = property(lambda self: get_cache_metrics()['hits'])
CACHE_MISSES = property(lambda self: get_cache_metrics()['misses'])
CACHE_LATENCY = property(lambda self: get_cache_metrics()['latency'])
CACHE_LATENCY_MS = property(lambda self: get_cache_metrics()['latency'])  # Same as CACHE_LATENCY

try:
    CACHE_ERRORS = Counter('redis_cache_errors_total', 'Total number of Redis cache errors', ['operation'])
except ValueError:
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_name') and collector._name == 'redis_cache_errors_total':
            CACHE_ERRORS = collector
            break
    else:
        CACHE_ERRORS = Counter('redis_cache_errors_total_v2', 'Total number of Redis cache errors', ['operation'])

# Load and validate Redis configuration
config_path = "config/redis_config.yaml"
redis_config = RedisConfig.load_from_yaml(config_path)
redis_config.validate_config()

# Redis client initialization with configurable connection pool
# Using ConnectionPool for better connection reuse and performance
connection_pool = redis.ConnectionPool(
    host=redis_config.host,
    port=redis_config.port,
    db=redis_config.cache_db,
    max_connections=redis_config.max_connections,
    socket_connect_timeout=redis_config.connect_timeout,
    socket_timeout=redis_config.socket_timeout,
    socket_keepalive=redis_config.socket_keepalive,
    socket_keepalive_options=redis_config.socket_keepalive_options,
    ssl=redis_config.use_ssl
)

redis_client = redis.Redis(connection_pool=connection_pool)

# Circuit breaker instances for Redis operations
_redis_get_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)
_redis_set_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)
_redis_delete_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

# Singleflight pattern - tracks ongoing operations
_ongoing_operations: dict[str, asyncio.Future] = {}
_operations_lock = None  # Will be initialized per event loop

def _get_operations_lock():
    """Get or create the operations lock for current event loop."""
    global _operations_lock
    try:
        # Always create a new lock to avoid event loop binding issues
        if _operations_lock is None or _operations_lock._loop != asyncio.get_event_loop():
            _operations_lock = asyncio.Lock()
        return _operations_lock
    except RuntimeError:
        # Different event loop, create new lock
        _operations_lock = asyncio.Lock()
        return _operations_lock

class RedisCache:
    """Redis cache with automatic compression and Prometheus metrics."""

    @staticmethod
    async def get(key: str, retries: int = 3, backoff_factor: float = 0.5) -> bytes | None:
        """Retrieve a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Decompressed bytes if found, None otherwise
        """
        # Check circuit breaker first
        if _redis_get_breaker.is_open():
            logger.warning(f"Circuit breaker is open for Redis get operation on key {key}")
            return None

        async def _redis_get_operation():
            """Internal Redis get operation."""
            start_time = time.perf_counter()
            metrics = get_cache_metrics()
            with metrics['latency'].labels(operation='get').time():
                data = await redis_client.get(key)
            end_time = time.perf_counter()
            metrics['latency'].labels(operation='get').observe((end_time - start_time) * 1000)
            if data:
                metrics['hits'].inc()
                try:
                    return lz4.frame.decompress(data)
                except Exception as e:
                    logger.warning(f"Failed to decompress cache data for key {key}: {e}")
                    # Clean up corrupted cache entry
                    await redis_client.delete(key)
                    metrics['misses'].inc()
                    return None
            else:
                metrics['misses'].inc()
                return None

        for attempt in range(retries):
            try:
                # Use circuit breaker for Redis operation
                return await _redis_get_breaker.call(_redis_get_operation)
            except (redis.ConnectionError, redis.TimeoutError, redis.BusyLoadingError) as e:
                # Calculate sleep time with exponential back-off
                sleep_time = backoff_factor * (2 ** attempt)
                logger.warning(f"Retrying Redis get for key {key} after error: {e}. Attempt {attempt + 1}/{retries}, sleeping for {sleep_time} seconds.")
                if attempt < retries - 1:  # Only sleep if not the last attempt
                    await asyncio.sleep(sleep_time)
            except Exception as e:
                CACHE_ERRORS.labels(operation='get').inc()
                logger.error(f"Redis get error for key {key}: {e}")
                return None

        # All retries exhausted
        CACHE_ERRORS.labels(operation='get').inc()
        logger.error(f"Redis get failed after {retries} attempts for key {key}")
        return None

    @staticmethod
    async def set(key: str, value: bytes, expire: int | None = None, retries: int = 3, backoff_factor: float = 0.5) -> bool:
        """Store a value in the cache with optional expiration.

        Args:
            key: Cache key to store
            value: Bytes to store (will be compressed)
            expire: Expiration time in seconds (optional)

        Returns:
            True if successful, False otherwise
        """
        # Check circuit breaker first
        if _redis_set_breaker.is_open():
            logger.warning(f"Circuit breaker is open for Redis set operation on key {key}")
            return False

        async def _redis_set_operation():
            """Internal Redis set operation."""
            start_time = time.perf_counter()
            with CACHE_LATENCY.labels(operation='set').time():
                compressed_data = lz4.frame.compress(value)
                await redis_client.set(key, compressed_data, ex=expire)
            end_time = time.perf_counter()
            CACHE_LATENCY_MS.labels(operation='set').observe((end_time - start_time) * 1000)
            return True

        for attempt in range(retries):
            try:
                # Use circuit breaker for Redis operation
                return await _redis_set_breaker.call(_redis_set_operation)
            except (redis.ConnectionError, redis.TimeoutError, redis.BusyLoadingError) as e:
                # Calculate sleep time with exponential back-off
                sleep_time = backoff_factor * (2 ** attempt)
                logger.warning(f"Retrying Redis set for key {key} after error: {e}. Attempt {attempt + 1}/{retries}, sleeping for {sleep_time} seconds.")
                if attempt < retries - 1:  # Only sleep if not the last attempt
                    await asyncio.sleep(sleep_time)
            except Exception as e:
                CACHE_ERRORS.labels(operation='set').inc()
                logger.error(f"Redis set error for key {key}: {e}")
                return False

        # All retries exhausted
        CACHE_ERRORS.labels(operation='set').inc()
        logger.error(f"Redis set failed after {retries} attempts for key {key}")
        return False

    @staticmethod
    async def invalidate(key: str, retries: int = 3, backoff_factor: float = 0.5) -> bool:
        """Remove a value from the cache.

        Args:
            key: Cache key to remove
            retries: Number of retry attempts
            backoff_factor: Exponential backoff factor

        Returns:
            True if successful, False otherwise
        """
        # Check circuit breaker first
        if _redis_delete_breaker.is_open():
            logger.warning(f"Circuit breaker is open for Redis delete operation on key {key}")
            return False

        async def _redis_delete_operation():
            """Internal Redis delete operation."""
            start_time = time.perf_counter()
            with CACHE_LATENCY.labels(operation='delete').time():
                result = await redis_client.delete(key)
            end_time = time.perf_counter()
            CACHE_LATENCY_MS.labels(operation='delete').observe((end_time - start_time) * 1000)
            return result > 0

        for attempt in range(retries):
            try:
                # Use circuit breaker for Redis operation
                return await _redis_delete_breaker.call(_redis_delete_operation)
            except (redis.ConnectionError, redis.TimeoutError, redis.BusyLoadingError) as e:
                # Calculate sleep time with exponential back-off
                sleep_time = backoff_factor * (2 ** attempt)
                logger.warning(f"Retrying Redis delete for key {key} after error: {e}. Attempt {attempt + 1}/{retries}, sleeping for {sleep_time} seconds.")
                if attempt < retries - 1:  # Only sleep if not the last attempt
                    await asyncio.sleep(sleep_time)
            except Exception as e:
                CACHE_ERRORS.labels(operation='delete').inc()
                logger.error(f"Redis delete error for key {key}: {e}")
                return False

        # All retries exhausted
        CACHE_ERRORS.labels(operation='delete').inc()
        logger.error(f"Redis delete failed after {retries} attempts for key {key}")
        return False

    @staticmethod
    def with_singleflight(cache_key_fn=None, expire: int | None = None):
        """Decorator to ensure a single flight of concurrent cache requests.

        Prevents thundering herd problem by ensuring only one instance of
        the same function call executes at a time.

        Args:
            cache_key_fn: Function to generate cache key from args/kwargs
            expire: Cache expiration time in seconds

        Returns:
            Decorator function
        """
        def decorator(fn):
            @wraps(fn)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_fn:
                    cache_key = cache_key_fn(*args, **kwargs)
                else:
                    # Default key generation
                    key_data = {
                        'fn': fn.__name__,
                        'args': args,
                        'kwargs': kwargs
                    }
                    key_str = json.dumps(key_data, sort_keys=True, default=str)
                    cache_key = hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

                # Check cache first
                cached_value = await RedisCache.get(cache_key)
                if cached_value is not None:
                    try:
                        # Assume cached value is JSON-serialized
                        return json.loads(cached_value.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Return raw bytes if not JSON
                        return cached_value

                # Singleflight pattern - check if operation is already in progress
                ops_lock = _get_operations_lock()
                async with ops_lock:
                    if cache_key in _ongoing_operations:
                        # Wait for ongoing operation to complete
                        try:
                            result = await _ongoing_operations[cache_key]
                            return result
                        except Exception:
                            # If ongoing operation failed, continue with new attempt
                            pass

                    # Start new operation
                    future = asyncio.create_task(_execute_and_cache(fn, cache_key, expire, *args, **kwargs))
                    _ongoing_operations[cache_key] = future

                try:
                    result = await future
                    return result
                finally:
                    # Clean up ongoing operation
                    async with ops_lock:
                        _ongoing_operations.pop(cache_key, None)

            return wrapper
        return decorator

async def _execute_and_cache(fn, cache_key: str, expire: int | None, *args, **kwargs):
    """Execute function and cache the result."""
    try:
        result = await fn(*args, **kwargs)

        # Cache the result
        if result is not None:
            if isinstance(result, bytes):
                cache_data = result
            else:
                # JSON serialize non-bytes results
                cache_data = json.dumps(result, default=str).encode()

            await RedisCache.set(cache_key, cache_data, expire)

        return result
    except Exception as e:
        logger.error(f"Function execution failed in singleflight: {e}")
        raise

class CacheSubscriber:
    """Cache subscriber service to handle pattern.invalidate events."""

    def __init__(self):
        self.is_running = False
        self.subscriber_task = None
        self.pubsub = None

    async def start(self):
        """Start the cache subscriber service."""
        if self.is_running:
            return

        try:
            self.pubsub = redis_client.pubsub()
            await self.pubsub.subscribe('pattern.invalidate')
            self.is_running = True

            # Start the subscriber task
            self.subscriber_task = asyncio.create_task(self._listen_for_events())
            logger.info("Cache subscriber started, listening for pattern.invalidate events")

        except Exception as e:
            logger.error(f"Failed to start cache subscriber: {e}")

    async def stop(self):
        """Stop the cache subscriber service."""
        if not self.is_running:
            return

        self.is_running = False

        if self.subscriber_task:
            self.subscriber_task.cancel()
            try:
                await self.subscriber_task
            except asyncio.CancelledError:
                pass

        if self.pubsub:
            await self.pubsub.close()

        logger.info("Cache subscriber stopped")

    async def _listen_for_events(self):
        """Listen for pattern.invalidate events and handle cache invalidation."""
        try:
            while self.is_running:
                try:
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message:
                        await self._handle_invalidate_event(message)
                except TimeoutError:
                    # Timeout is expected, continue listening
                    continue
                except Exception as e:
                    logger.error(f"Error processing invalidate event: {e}")
                    await asyncio.sleep(1)  # Brief pause before retrying

        except asyncio.CancelledError:
            logger.info("Cache subscriber task cancelled")
            raise
        except Exception as e:
            logger.error(f"Cache subscriber error: {e}")

    async def _handle_invalidate_event(self, message):
        """Handle a single pattern.invalidate event."""
        try:
            if message['type'] != 'message':
                return

            # Parse the event data
            event_data = json.loads(message['data'])
            event_type = event_data.get('type')
            cache_prefixes = event_data.get('cache_prefixes', [])

            logger.info(f"Received cache invalidation event: {event_type}")

            if not cache_prefixes:
                logger.warning("No cache prefixes specified in invalidation event")
                return

            # Invalidate cache keys by prefix
            invalidated_count = 0
            for prefix in cache_prefixes:
                count = await self._invalidate_by_prefix(prefix)
                invalidated_count += count

            logger.info(f"Cache invalidation complete: {invalidated_count} keys invalidated for event {event_type}")

        except Exception as e:
            logger.error(f"Error handling invalidate event: {e}")

    async def _invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all cache keys matching a prefix."""
        try:
            # Use Redis SCAN to find keys matching the prefix
            keys_to_delete = []
            cursor = 0

            while True:
                cursor, keys = await redis_client.scan(cursor, match=f"{prefix}*", count=100)
                keys_to_delete.extend(keys)

                if cursor == 0:
                    break

            # Delete the keys in batches
            if keys_to_delete:
                # Convert bytes keys to strings if needed
                str_keys = [k.decode() if isinstance(k, bytes) else k for k in keys_to_delete]
                deleted_count = await redis_client.delete(*str_keys)
                logger.debug(f"Deleted {deleted_count} keys with prefix '{prefix}'")
                return deleted_count
            logger.debug(f"No keys found with prefix '{prefix}'")
            return 0

        except Exception as e:
            logger.error(f"Error invalidating keys with prefix '{prefix}': {e}")
            return 0

# Global cache subscriber instance
_cache_subscriber = None

async def start_cache_subscriber():
    """Start the global cache subscriber service."""
    global _cache_subscriber
    if _cache_subscriber is None:
        _cache_subscriber = CacheSubscriber()
    await _cache_subscriber.start()

async def stop_cache_subscriber():
    """Stop the global cache subscriber service."""
    global _cache_subscriber
    if _cache_subscriber:
        await _cache_subscriber.stop()

# Convenience functions for module-level access
get = RedisCache.get
set = RedisCache.set
invalidate = RedisCache.invalidate
with_singleflight = RedisCache.with_singleflight
