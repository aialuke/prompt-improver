"""Redis connection manager with clustering, sentinel, and health monitoring.

Extracted from database.unified_connection_manager.py to provide:
- Redis cluster and standalone connection management
- Redis Sentinel integration for high availability
- Connection pooling with circuit breaker patterns
- Health monitoring and failover capabilities
- Distributed locking and pub/sub coordination
- Performance monitoring and metrics collection

This centralizes all Redis functionality from the monolithic manager.
"""

import asyncio
import json
import logging
import time
import warnings
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

try:
    import redis.asyncio as redis
    from redis.asyncio import ConnectionPool, Redis, Sentinel
    from redis.asyncio.cluster import RedisCluster
    from redis.exceptions import (
        ClusterDownError,
        ConnectionError as RedisConnectionError,
        ResponseError,
        TimeoutError as RedisTimeoutError,
    )

    REDIS_AVAILABLE = True
except ImportError:
    warnings.warn("Redis not available. Install with: pip install redis")
    REDIS_AVAILABLE = False
    # Mock classes for type hints
    Redis = Any
    ConnectionPool = Any
    Sentinel = Any
    RedisCluster = Any
    RedisConnectionError = Exception
    RedisTimeoutError = Exception
    ClusterDownError = Exception
    ResponseError = Exception

from prompt_improver.database.services.connection.connection_metrics import (
    ConnectionMetrics,
)

logger = logging.getLogger(__name__)


class RedisMode(Enum):
    """Redis deployment modes."""

    STANDALONE = "standalone"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


class RedisHealthStatus(Enum):
    """Redis health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    # Connection settings
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0

    # Pool settings
    max_connections: int = 100
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    # Cluster settings
    cluster_nodes: Optional[List[str]] = None
    skip_full_coverage_check: bool = False

    # Sentinel settings
    sentinel_hosts: Optional[List[tuple[str, int]]] = None
    sentinel_service_name: str = "mymaster"
    sentinel_socket_timeout: float = 0.5

    # Health and monitoring
    health_check_interval: int = 30
    max_idle_time: int = 300

    @property
    def connection_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @property
    def mode(self) -> RedisMode:
        """Determine Redis deployment mode."""
        if self.cluster_nodes:
            return RedisMode.CLUSTER
        elif self.sentinel_hosts:
            return RedisMode.SENTINEL
        else:
            return RedisMode.STANDALONE

    @classmethod
    def for_environment(cls, env: str, service_name: str = "redis") -> "RedisConfig":
        """Create Redis configuration optimized for environment."""
        configs = {
            "development": cls(
                max_connections=20,
                socket_timeout=10.0,
                health_check_interval=60,
            ),
            "testing": cls(
                max_connections=10,
                socket_timeout=2.0,
                health_check_interval=30,
                db=15,  # Use test database
            ),
            "production": cls(
                max_connections=200,
                socket_timeout=5.0,
                socket_connect_timeout=2.0,
                health_check_interval=15,
                retry_on_timeout=True,
            ),
            "mcp_server": cls(
                max_connections=50,
                socket_timeout=1.0,
                socket_connect_timeout=0.5,
                health_check_interval=10,
                retry_on_timeout=False,  # Fast fail for MCP
            ),
        }
        return configs.get(env, configs["development"])


@dataclass
class RedisNodeInfo:
    """Information about a Redis node."""

    host: str
    port: int
    node_id: Optional[str] = None
    role: str = "master"
    is_available: bool = True
    last_check: datetime = field(default_factory=lambda: datetime.now(UTC))
    response_time_ms: float = 0.0
    memory_usage: Optional[int] = None
    connected_clients: Optional[int] = None


class RedisManager:
    """Advanced Redis connection manager with clustering and sentinel support.

    Provides comprehensive Redis management with:
    - Multi-mode support (standalone, cluster, sentinel)
    - Connection pooling with health monitoring
    - Automatic failover and cluster discovery
    - Circuit breaker pattern for resilience
    - Performance metrics and monitoring
    - Distributed operations (locks, pub/sub)
    """

    def __init__(self, config: RedisConfig, service_name: str = "redis_manager"):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis package not available. Install with: pip install redis"
            )

        self.config = config
        self.service_name = service_name

        # Connection management
        self._redis_client: Optional[Union[Redis, RedisCluster]] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._sentinel_client: Optional[Sentinel] = None

        # Health and state management
        self._health_status = RedisHealthStatus.UNKNOWN
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._cluster_nodes: Dict[str, RedisNodeInfo] = {}

        # Circuit breaker
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60
        self._circuit_breaker_last_failure = 0

        # Metrics and monitoring
        self.metrics = ConnectionMetrics()
        self._operation_history = deque(maxlen=1000)
        self._node_health_cache: Dict[str, RedisNodeInfo] = {}

        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cluster_discovery_task: Optional[asyncio.Task] = None

        logger.info(
            f"RedisManager initialized: {service_name} ({config.mode.value} mode)"
        )
        logger.info(
            f"Config: host={config.host}:{config.port}, max_conn={config.max_connections}"
        )

    async def initialize(self) -> bool:
        """Initialize the Redis manager with connections."""
        async with self._initialization_lock:
            if self._is_initialized:
                return True

            try:
                # Initialize based on deployment mode
                if self.config.mode == RedisMode.CLUSTER:
                    await self._setup_cluster_client()
                elif self.config.mode == RedisMode.SENTINEL:
                    await self._setup_sentinel_client()
                else:
                    await self._setup_standalone_client()

                # Test connectivity
                await self._test_connections()

                # Start background monitoring
                self._health_monitor_task = asyncio.create_task(
                    self._health_monitor_loop()
                )

                if self.config.mode == RedisMode.CLUSTER:
                    self._cluster_discovery_task = asyncio.create_task(
                        self._cluster_discovery_loop()
                    )

                self._health_status = RedisHealthStatus.HEALTHY
                self._is_initialized = True

                logger.info(
                    f"RedisManager initialized successfully: {self.service_name}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to initialize RedisManager: {e}")
                self._health_status = RedisHealthStatus.UNHEALTHY
                raise

    async def _setup_standalone_client(self) -> None:
        """Setup standalone Redis client with connection pooling."""
        pool_kwargs = {
            "host": self.config.host,
            "port": self.config.port,
            "password": self.config.password,
            "db": self.config.db,
            "max_connections": self.config.max_connections,
            "retry_on_timeout": self.config.retry_on_timeout,
            "socket_timeout": self.config.socket_timeout,
            "socket_connect_timeout": self.config.socket_connect_timeout,
            "socket_keepalive": True,
            "socket_keepalive_options": {},
        }

        self._connection_pool = ConnectionPool(**pool_kwargs)
        self._redis_client = Redis(connection_pool=self._connection_pool)

        logger.info(f"Standalone Redis client created: {self.config.connection_url}")

    async def _setup_cluster_client(self) -> None:
        """Setup Redis cluster client."""
        if not self.config.cluster_nodes:
            raise ValueError("Cluster nodes must be specified for cluster mode")

        startup_nodes = []
        for node in self.config.cluster_nodes:
            if ":" in node:
                host, port = node.split(":", 1)
                startup_nodes.append({"host": host, "port": int(port)})
            else:
                startup_nodes.append({"host": node, "port": 6379})

        cluster_kwargs = {
            "startup_nodes": startup_nodes,
            "password": self.config.password,
            "max_connections": self.config.max_connections,
            "socket_timeout": self.config.socket_timeout,
            "socket_connect_timeout": self.config.socket_connect_timeout,
            "skip_full_coverage_check": self.config.skip_full_coverage_check,
            "retry_on_timeout": self.config.retry_on_timeout,
        }

        self._redis_client = RedisCluster(**cluster_kwargs)

        # Discover cluster topology
        await self._discover_cluster_topology()

        logger.info(
            f"Redis cluster client created with {len(startup_nodes)} startup nodes"
        )

    async def _setup_sentinel_client(self) -> None:
        """Setup Redis Sentinel client for high availability."""
        if not self.config.sentinel_hosts:
            raise ValueError("Sentinel hosts must be specified for sentinel mode")

        sentinel_kwargs = {
            "sentinels": self.config.sentinel_hosts,
            "socket_timeout": self.config.sentinel_socket_timeout,
            "password": self.config.password,
        }

        self._sentinel_client = Sentinel(**sentinel_kwargs)

        # Get master connection
        self._redis_client = self._sentinel_client.master_for(
            service_name=self.config.sentinel_service_name,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            max_connections=self.config.max_connections,
            retry_on_timeout=self.config.retry_on_timeout,
        )

        logger.info(
            f"Sentinel client created for service: {self.config.sentinel_service_name}"
        )

    async def _test_connections(self) -> None:
        """Test Redis connections to ensure they work."""
        if not self._redis_client:
            raise RuntimeError("Redis client not initialized")

        # Basic connectivity test
        await self._redis_client.ping()

        # Test basic operations
        test_key = f"_test:{self.service_name}:{int(time.time())}"
        await self._redis_client.set(test_key, "test_value", ex=10)
        result = await self._redis_client.get(test_key)
        await self._redis_client.delete(test_key)

        if result != b"test_value":
            raise RuntimeError("Redis basic operations test failed")

        logger.debug("Redis connection tests passed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[Union[Redis, RedisCluster]]:
        """Get Redis connection with automatic error handling and metrics."""
        if not self._is_initialized:
            await self.initialize()

        if not self._redis_client:
            raise RuntimeError("Redis client not available")

        if self._is_circuit_breaker_open():
            raise RuntimeError("Circuit breaker is open - Redis unavailable")

        start_time = time.time()

        try:
            yield self._redis_client

            # Record successful operation
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(duration_ms, success=True)
            self._record_operation("connection", duration_ms, True)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(duration_ms, success=False)
            self._handle_connection_failure(e)
            self._record_operation("connection", duration_ms, False)
            logger.error(f"Redis connection error: {e}")
            raise

    async def get(self, key: str) -> Optional[bytes]:
        """Get value from Redis with error handling."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.get(key)

    async def set(
        self,
        key: str,
        value: Union[str, bytes],
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in Redis with options."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.set(key, value, ex=ex, px=px, nx=nx, xx=xx)

    async def delete(self, *keys: str) -> int:
        """Delete keys from Redis."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Check if keys exist in Redis."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.exists(*keys)

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.expire(key, seconds)

    async def hget(self, name: str, key: str) -> Optional[bytes]:
        """Get hash field value."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.hget(name, key)

    async def hset(self, name: str, key: str, value: Union[str, bytes]) -> int:
        """Set hash field value."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.hset(name, key, value)

    async def hmget(self, name: str, keys: List[str]) -> List[Optional[bytes]]:
        """Get multiple hash field values."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.hmget(name, keys)

    async def hmset(self, name: str, mapping: Dict[str, Union[str, bytes]]) -> bool:
        """Set multiple hash fields."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.hmset(name, mapping)

    async def sadd(self, name: str, *values: Union[str, bytes]) -> int:
        """Add members to a set."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.sadd(name, *values)

    async def sismember(self, name: str, value: Union[str, bytes]) -> bool:
        """Check if value is a member of set."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.sismember(name, value)

    async def smembers(self, name: str) -> Set[bytes]:
        """Get all members of a set."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.smembers(name)

    async def lpush(self, name: str, *values: Union[str, bytes]) -> int:
        """Push values to the left of a list."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.lpush(name, *values)

    async def rpop(self, name: str) -> Optional[bytes]:
        """Pop value from the right of a list."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.rpop(name)

    async def llen(self, name: str) -> int:
        """Get length of a list."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.llen(name)

    async def publish(self, channel: str, message: Union[str, bytes]) -> int:
        """Publish message to a channel."""
        async with self.get_connection() as redis_conn:
            return await redis_conn.publish(channel, message)

    async def execute_lua_script(
        self, script: str, keys: List[str] = None, args: List[Union[str, bytes]] = None
    ) -> Any:
        """Execute Lua script on Redis."""
        async with self.get_connection() as redis_conn:
            sha = await redis_conn.script_load(script)
            return await redis_conn.evalsha(
                sha, len(keys or []), *(keys or []), *(args or [])
            )

    async def acquire_lock(
        self,
        lock_name: str,
        timeout: float = 10.0,
        sleep: float = 0.1,
        blocking_timeout: Optional[float] = None,
    ) -> Optional[str]:
        """Acquire distributed lock using Redis."""
        identifier = (
            f"{self.service_name}:{asyncio.current_task().get_name()}:{time.time()}"
        )
        lock_key = f"lock:{lock_name}"

        end_time = time.time() + (blocking_timeout or timeout)

        while time.time() < end_time:
            # Try to acquire lock (ensure timeout is at least 1 second)
            expire_seconds = max(1, int(timeout))
            if await self.set(lock_key, identifier, ex=expire_seconds, nx=True):
                return identifier

            await asyncio.sleep(sleep)

        return None

    async def release_lock(self, lock_name: str, identifier: str) -> bool:
        """Release distributed lock."""
        lock_key = f"lock:{lock_name}"

        # Lua script to atomically check and release lock
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = await self.execute_lua_script(lua_script, [lock_key], [identifier])
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to release lock {lock_name}: {e}")
            return False

    async def _discover_cluster_topology(self) -> None:
        """Discover Redis cluster topology and node information."""
        if not isinstance(self._redis_client, RedisCluster):
            return

        try:
            cluster_info = await self._redis_client.cluster_nodes()

            self._cluster_nodes.clear()

            for node_info in cluster_info.values():
                if "host" in node_info and "port" in node_info:
                    node_key = f"{node_info['host']}:{node_info['port']}"

                    self._cluster_nodes[node_key] = RedisNodeInfo(
                        host=node_info["host"],
                        port=node_info["port"],
                        node_id=node_info.get("node_id"),
                        role="master"
                        if "master" in node_info.get("flags", [])
                        else "slave",
                        is_available=True,
                        last_check=datetime.now(UTC),
                    )

            logger.debug(f"Discovered {len(self._cluster_nodes)} cluster nodes")

        except Exception as e:
            logger.error(f"Failed to discover cluster topology: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of Redis manager."""
        start_time = time.time()

        health_info = {
            "status": "unknown",
            "timestamp": start_time,
            "service": self.service_name,
            "mode": self.config.mode.value,
            "components": {},
            "metrics": await self.get_metrics(),
            "response_time_ms": 0,
        }

        try:
            # Test basic connectivity
            async with self.get_connection() as redis_conn:
                await redis_conn.ping()
            health_info["components"]["connectivity"] = "healthy"

            # Test basic operations
            test_key = f"_health_check:{self.service_name}:{int(time.time())}"
            await self.set(test_key, "health_check", ex=5)
            result = await self.get(test_key)
            await self.delete(test_key)

            if result == b"health_check":
                health_info["components"]["operations"] = "healthy"
            else:
                health_info["components"]["operations"] = "degraded"

            # Check cluster/sentinel specific health
            if self.config.mode == RedisMode.CLUSTER:
                await self._check_cluster_health(health_info)
            elif self.config.mode == RedisMode.SENTINEL:
                await self._check_sentinel_health(health_info)

            # Overall status
            failed_components = [
                k for k, v in health_info["components"].items() if v != "healthy"
            ]
            if not failed_components:
                health_info["status"] = "healthy"
                self._health_status = RedisHealthStatus.HEALTHY
            elif len(failed_components) < len(health_info["components"]) / 2:
                health_info["status"] = "degraded"
                self._health_status = RedisHealthStatus.DEGRADED
            else:
                health_info["status"] = "unhealthy"
                self._health_status = RedisHealthStatus.UNHEALTHY

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            self._health_status = RedisHealthStatus.UNHEALTHY
            logger.error(f"Redis health check failed: {e}")

        health_info["response_time_ms"] = (time.time() - start_time) * 1000
        return health_info

    async def _check_cluster_health(self, health_info: Dict[str, Any]) -> None:
        """Check Redis cluster specific health."""
        if not isinstance(self._redis_client, RedisCluster):
            return

        try:
            # Check cluster state
            cluster_info = await self._redis_client.cluster_info()
            if cluster_info.get("cluster_state") == "ok":
                health_info["components"]["cluster_state"] = "healthy"
            else:
                health_info["components"]["cluster_state"] = "degraded"

            # Check node availability
            healthy_nodes = sum(
                1 for node in self._cluster_nodes.values() if node.is_available
            )
            total_nodes = len(self._cluster_nodes)

            if healthy_nodes == total_nodes:
                health_info["components"]["cluster_nodes"] = "healthy"
            elif healthy_nodes > total_nodes / 2:
                health_info["components"]["cluster_nodes"] = "degraded"
            else:
                health_info["components"]["cluster_nodes"] = "unhealthy"

            health_info["cluster_stats"] = {
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "cluster_state": cluster_info.get("cluster_state", "unknown"),
            }

        except Exception as e:
            health_info["components"]["cluster_health"] = f"error: {e}"
            logger.warning(f"Cluster health check failed: {e}")

    async def _check_sentinel_health(self, health_info: Dict[str, Any]) -> None:
        """Check Redis Sentinel specific health."""
        if not self._sentinel_client:
            return

        try:
            # Check master availability
            master_info = await self._sentinel_client.discover_master(
                self.config.sentinel_service_name
            )
            if master_info:
                health_info["components"]["sentinel_master"] = "healthy"
            else:
                health_info["components"]["sentinel_master"] = "unhealthy"

            # Check sentinel connectivity
            sentinels = self._sentinel_client.sentinels
            healthy_sentinels = 0
            for sentinel in sentinels:
                try:
                    await sentinel.ping()
                    healthy_sentinels += 1
                except Exception:
                    pass

            if healthy_sentinels == len(sentinels):
                health_info["components"]["sentinels"] = "healthy"
            elif healthy_sentinels > 0:
                health_info["components"]["sentinels"] = "degraded"
            else:
                health_info["components"]["sentinels"] = "unhealthy"

            health_info["sentinel_stats"] = {
                "total_sentinels": len(sentinels),
                "healthy_sentinels": healthy_sentinels,
                "master_host": master_info[0] if master_info else None,
                "master_port": master_info[1] if master_info else None,
            }

        except Exception as e:
            health_info["components"]["sentinel_health"] = f"error: {e}"
            logger.warning(f"Sentinel health check failed: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Redis manager metrics."""
        base_metrics = self.metrics.to_dict()

        return {
            "service": self.service_name,
            "mode": self.config.mode.value,
            "health_status": self._health_status.value,
            "connection_metrics": base_metrics,
            "circuit_breaker": {
                "enabled": True,
                "state": "open" if self._is_circuit_breaker_open() else "closed",
                "failures": self._circuit_breaker_failures,
                "threshold": self._circuit_breaker_threshold,
            },
            "redis_info": await self._get_redis_info(),
            "operation_history_size": len(self._operation_history),
            "cluster_nodes": len(self._cluster_nodes) if self._cluster_nodes else 0,
        }

    async def _get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            async with self.get_connection() as redis_conn:
                info = await redis_conn.info()
                return {
                    "version": info.get("redis_version", "unknown"),
                    "used_memory": info.get("used_memory", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
        except Exception as e:
            logger.debug(f"Failed to get Redis info: {e}")
            return {}

    def _handle_connection_failure(self, error: Exception) -> None:
        """Handle connection failure and update circuit breaker."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()

        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            logger.error(
                f"Redis circuit breaker opened due to {self._circuit_breaker_failures} failures"
            )

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            # Check if enough time has passed to attempt half-open
            if (
                time.time() - self._circuit_breaker_last_failure
            ) > self._circuit_breaker_timeout:
                logger.info("Redis circuit breaker attempting half-open state")
                return False
            return True
        return False

    def _record_operation(
        self, operation_type: str, duration_ms: float, success: bool
    ) -> None:
        """Record operation in history for analysis."""
        operation = {
            "timestamp": time.time(),
            "operation": operation_type,
            "duration_ms": duration_ms,
            "success": success,
        }
        self._operation_history.append(operation)

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._is_initialized:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.health_check()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _cluster_discovery_loop(self) -> None:
        """Background cluster topology discovery loop."""
        while self._is_initialized and self.config.mode == RedisMode.CLUSTER:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._discover_cluster_topology()
            except Exception as e:
                logger.error(f"Cluster discovery error: {e}")

    async def shutdown(self) -> None:
        """Shutdown Redis manager and cleanup resources."""
        logger.info(f"Shutting down RedisManager: {self.service_name}")

        try:
            # Cancel background tasks
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass

            if self._cluster_discovery_task:
                self._cluster_discovery_task.cancel()
                try:
                    await self._cluster_discovery_task
                except asyncio.CancelledError:
                    pass

            # Close Redis connections
            if self._redis_client:
                await self._redis_client.aclose()
                logger.info("Redis client closed")

            if self._connection_pool:
                await self._connection_pool.disconnect()
                logger.info("Redis connection pool closed")

            # Clear state
            self._is_initialized = False
            self._cluster_nodes.clear()
            self._operation_history.clear()

            logger.info(f"RedisManager shutdown complete: {self.service_name}")

        except Exception as e:
            logger.error(f"Error during RedisManager shutdown: {e}")

    def __repr__(self) -> str:
        return (
            f"RedisManager(service={self.service_name}, "
            f"mode={self.config.mode.value}, health={self._health_status.value})"
        )
