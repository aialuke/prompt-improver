"""Real Redis testcontainer for comprehensive cache testing.

Provides real Redis instance using testcontainers for testing multi-level caching,
Redis-based services, and performance validation with actual Redis behavior.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
from testcontainers.redis import RedisContainer

logger = logging.getLogger(__name__)


class RealRedisTestContainer:
    """Real Redis testcontainer for comprehensive cache testing.

    Provides authentic Redis behavior for testing:
    - Multi-level caching strategies (L2 Redis layer)
    - Redis-based services and session management
    - Cache performance and eviction policies
    - Real Redis connection pooling and clustering behavior
    - Redis-specific features (pub/sub, streams, etc.)
    """

    def __init__(
        self,
        redis_version: str = "7.2",
        port: int | None = None,
        max_memory: str = "256mb",
        max_memory_policy: str = "allkeys-lru",
        enable_persistence: bool = False,
    ):
        """Initialize Redis testcontainer.

        Args:
            redis_version: Redis version to use
            port: Container port (auto-assigned if None)
            max_memory: Maximum memory for Redis instance
            max_memory_policy: Memory eviction policy
            enable_persistence: Enable Redis persistence (slower for tests)
        """
        self.redis_version = redis_version
        self.port = port
        self.max_memory = max_memory
        self.max_memory_policy = max_memory_policy
        self.enable_persistence = enable_persistence

        self._container: RedisContainer | None = None
        self._redis_client: redis.Redis | None = None
        self._connection_url: str | None = None
        self._container_id = str(uuid.uuid4())[:8]

    async def start(self) -> "RealRedisTestContainer":
        """Start Redis container and initialize connections."""
        try:
            # Create Redis container with optimized test configuration
            self._container = RedisContainer(
                image=f"redis:{self.redis_version}"
            )

            # Configure Redis for testing
            redis_config = [
                f"maxmemory {self.max_memory}",
                f"maxmemory-policy {self.max_memory_policy}",
                "timeout 30",  # Connection timeout
                "tcp-keepalive 60",  # Keep connections alive
            ]

            # Disable persistence for faster tests unless explicitly enabled
            if not self.enable_persistence:
                redis_config.extend([
                    "save ''",  # Disable RDB snapshots
                    "appendonly no"  # Disable AOF
                ])

            # Apply configuration
            for config_line in redis_config:
                self._container = self._container.with_command(
                    f'sh -c "echo \"{config_line}\" >> /usr/local/etc/redis/redis.conf && redis-server /usr/local/etc/redis/redis.conf"'
                )

            if self.port:
                self._container = self._container.with_bind_ports(6379, self.port)

            self._container.start()

            # Get connection details
            host = self._container.get_container_host_ip()
            port = self._container.get_exposed_port(6379)

            self._connection_url = f"redis://{host}:{port}"

            # Create Redis client with connection pooling
            self._redis_client = redis.from_url(
                self._connection_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                health_check_interval=30,
            )

            # Wait for Redis to be ready
            await self._wait_for_readiness()

            # Configure Redis runtime settings
            await self._configure_redis_instance()

            logger.info(
                f"Redis testcontainer started: {self._container_id} "
                f"(version {self.redis_version}, port {port})"
            )

            return self

        except Exception as e:
            logger.exception(f"Failed to start Redis testcontainer: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop Redis container and clean up resources."""
        try:
            if self._redis_client:
                await self._redis_client.close()
                self._redis_client = None

            if self._container:
                self._container.stop()
                self._container = None

            logger.info(f"Redis testcontainer stopped: {self._container_id}")

        except Exception as e:
            logger.warning(f"Error stopping Redis testcontainer: {e}")

    async def _wait_for_readiness(self, max_retries: int = 30, retry_delay: float = 1.0):
        """Wait for Redis to be ready for connections."""
        for attempt in range(max_retries):
            try:
                await self._redis_client.ping()
                logger.debug(f"Redis ready after {attempt + 1} attempts")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Redis not ready after {max_retries} attempts: {e}")
                await asyncio.sleep(retry_delay)

    async def _configure_redis_instance(self):
        """Configure Redis instance for optimal testing performance."""
        try:
            # Set additional runtime configuration
            config_updates = {
                "tcp-keepalive": "60",
                "timeout": "30",
                "client-output-buffer-limit": "normal 0 0 0",
                "client-output-buffer-limit": "replica 256mb 64mb 60",
                "client-output-buffer-limit": "pubsub 32mb 8mb 60",
            }

            for key, value in config_updates.items():
                try:
                    await self._redis_client.config_set(key, value)
                except Exception as e:
                    logger.debug(f"Could not set Redis config {key}={value}: {e}")

            logger.debug(f"Redis instance configured for container: {self._container_id}")

        except Exception as e:
            logger.exception(f"Failed to configure Redis instance: {e}")
            raise

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[redis.Redis, None]:
        """Get Redis client with proper cleanup."""
        if not self._redis_client:
            raise RuntimeError("Container not started. Call start() first.")

        try:
            yield self._redis_client
        finally:
            # Client cleanup is handled by connection pooling
            pass

    async def flush_all(self) -> None:
        """Flush all Redis databases."""
        async with self.get_client() as client:
            await client.flushall()

    async def flush_db(self, db: int = 0) -> None:
        """Flush specific Redis database.

        Args:
            db: Database number to flush
        """
        async with self.get_client() as client:
            await client.select(db)
            await client.flushdb()

    async def get_info(self) -> dict[str, Any]:
        """Get Redis instance information."""
        async with self.get_client() as client:
            return await client.info()

    async def get_memory_usage(self) -> dict[str, Any]:
        """Get Redis memory usage statistics."""
        info = await self.get_info()
        return {
            "used_memory": info.get("used_memory", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "used_memory_peak": info.get("used_memory_peak", 0),
            "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
            "maxmemory": info.get("maxmemory", 0),
            "maxmemory_human": info.get("maxmemory_human", "0B"),
            "maxmemory_policy": info.get("maxmemory_policy", "unknown"),
        }

    async def get_connection_stats(self) -> dict[str, Any]:
        """Get Redis connection statistics."""
        info = await self.get_info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "blocked_clients": info.get("blocked_clients", 0),
            "total_connections_received": info.get("total_connections_received", 0),
            "rejected_connections": info.get("rejected_connections", 0),
        }

    async def test_performance(
        self,
        operation_count: int = 1000,
        key_size: int = 100,
        value_size: int = 1000
    ) -> dict[str, Any]:
        """Test Redis performance with various operations.

        Args:
            operation_count: Number of operations to perform
            key_size: Size of keys in bytes
            value_size: Size of values in bytes

        Returns:
            Performance test results
        """
        async with self.get_client() as client:
            # Generate test data
            test_data = {}
            for i in range(operation_count):
                key = f"perf_test_key_{i:05d}_{'x' * (key_size - 20)}"
                value = f"perf_test_value_{i:05d}_{'y' * (value_size - 25)}"
                test_data[key] = value

            # Test SET operations
            set_start = time.perf_counter()
            for key, value in test_data.items():
                await client.set(key, value)
            set_duration = time.perf_counter() - set_start

            # Test GET operations
            get_start = time.perf_counter()
            for key in test_data:
                await client.get(key)
            get_duration = time.perf_counter() - get_start

            # Test MGET operations (batch)
            keys_list = list(test_data.keys())
            mget_start = time.perf_counter()
            await client.mget(keys_list)
            mget_duration = time.perf_counter() - mget_start

            # Clean up test data
            await client.delete(*test_data.keys())

            return {
                "operation_count": operation_count,
                "key_size_bytes": key_size,
                "value_size_bytes": value_size,
                "set_operations": {
                    "total_duration_ms": set_duration * 1000,
                    "ops_per_second": operation_count / set_duration,
                    "avg_latency_ms": (set_duration / operation_count) * 1000,
                },
                "get_operations": {
                    "total_duration_ms": get_duration * 1000,
                    "ops_per_second": operation_count / get_duration,
                    "avg_latency_ms": (get_duration / operation_count) * 1000,
                },
                "mget_operation": {
                    "total_duration_ms": mget_duration * 1000,
                    "ops_per_second": operation_count / mget_duration,
                    "avg_latency_ms": mget_duration * 1000,
                },
            }

    async def test_connection_pooling(self, concurrent_connections: int = 20) -> dict[str, Any]:
        """Test Redis connection pooling behavior.

        Args:
            concurrent_connections: Number of concurrent connections to test

        Returns:
            Connection pooling test results
        """
        async def make_request(client_id: int):
            start_time = time.perf_counter()
            async with self.get_client() as client:
                # Perform some operations
                key = f"pool_test_{client_id}"
                await client.set(key, f"value_{client_id}")
                value = await client.get(key)
                await client.delete(key)
                return time.perf_counter() - start_time

        # Test concurrent connections
        start_time = time.perf_counter()
        tasks = [make_request(i) for i in range(concurrent_connections)]
        request_times = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        return {
            "concurrent_connections": concurrent_connections,
            "total_time_ms": total_time * 1000,
            "avg_request_time_ms": sum(request_times) / len(request_times) * 1000,
            "min_request_time_ms": min(request_times) * 1000,
            "max_request_time_ms": max(request_times) * 1000,
            "requests_per_second": concurrent_connections / total_time,
        }

    def get_connection_url(self) -> str:
        """Get Redis connection URL."""
        if not self._connection_url:
            raise RuntimeError("Container not started. Call start() first.")
        return self._connection_url

    def get_port(self) -> int:
        """Get exposed Redis port."""
        if not self._container:
            raise RuntimeError("Container not started. Call start() first.")
        return self._container.get_exposed_port(6379)

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection information."""
        if not self._container:
            raise RuntimeError("Container not started. Call start() first.")

        return {
            "host": self._container.get_container_host_ip(),
            "port": self.get_port(),
            "connection_url": self._connection_url,
            "container_id": self._container_id,
            "version": self.redis_version,
        }

    async def create_test_data(self, key_patterns: list[str], data_size: str = "small") -> dict[str, Any]:
        """Create test data for various testing scenarios.

        Args:
            key_patterns: List of key patterns to create
            data_size: Size of test data ("small", "medium", "large")

        Returns:
            Test data creation results
        """
        size_config = {
            "small": {"count": 100, "value_size": 100},
            "medium": {"count": 1000, "value_size": 1000},
            "large": {"count": 10000, "value_size": 10000},
        }

        config = size_config.get(data_size, size_config["small"])

        async with self.get_client() as client:
            created_keys = []

            for pattern in key_patterns:
                for i in range(config["count"]):
                    key = f"{pattern}:{i:05d}"
                    value = f"test_data_{pattern}_{i:05d}_{'x' * (config['value_size'] - 30)}"
                    await client.set(key, value, ex=3600)  # 1 hour expiry
                    created_keys.append(key)

            return {
                "created_keys": len(created_keys),
                "patterns": key_patterns,
                "data_size": data_size,
                "value_size_bytes": config["value_size"],
            }

    async def __aenter__(self) -> "RealRedisTestContainer":
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class RedisTestFixture:
    """Test fixture helper for Redis testcontainers."""

    def __init__(self, container: RealRedisTestContainer):
        self.container = container

    async def setup_cache_layers(self) -> dict[str, Any]:
        """Setup cache layers for multi-level cache testing."""
        async with self.container.get_client() as client:
            # Setup different cache layers with different TTLs
            cache_layers = {
                "l1_cache": {"db": 0, "ttl": 300},   # 5 minutes
                "l2_cache": {"db": 1, "ttl": 1800},  # 30 minutes
                # L3 cache no longer used in L1/L2 architecture
            }

            for layer_name, config in cache_layers.items():
                await client.select(config["db"])
                await client.flushdb()

                # Create sample data for each layer
                for i in range(10):
                    key = f"{layer_name}:item_{i:03d}"
                    value = f"{layer_name}_value_{i:03d}"
                    await client.set(key, value, ex=config["ttl"])

            return cache_layers

    async def validate_cache_performance_targets(self) -> dict[str, Any]:
        """Validate cache performance against targets."""
        # Performance targets for Redis operations
        targets = {
            "set_operation_ms": 1.0,    # <1ms for SET operations
            "get_operation_ms": 0.5,    # <0.5ms for GET operations
            "mget_batch_ms": 5.0,       # <5ms for batch operations
        }

        results = await self.container.test_performance(
            operation_count=1000,
            key_size=50,
            value_size=200
        )

        validation = {}
        for operation, target_ms in targets.items():
            if operation == "set_operation_ms":
                actual_ms = results["set_operations"]["avg_latency_ms"]
            elif operation == "get_operation_ms":
                actual_ms = results["get_operations"]["avg_latency_ms"]
            elif operation == "mget_batch_ms":
                actual_ms = results["mget_operation"]["avg_latency_ms"]
            else:
                continue

            validation[operation] = {
                "target_ms": target_ms,
                "actual_ms": actual_ms,
                "performance_met": actual_ms < target_ms,
            }

        return {
            "targets": targets,
            "validation": validation,
            "overall_performance": all(v["performance_met"] for v in validation.values()),
            "full_results": results,
        }
