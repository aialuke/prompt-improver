"""Redis Sentinel manager for high availability and automatic failover.

Extracted from database.unified_connection_manager.py to provide:
- Redis Sentinel discovery and monitoring
- Automatic master/replica failover coordination
- Sentinel health monitoring and validation
- Multi-sentinel redundancy management
- Failover event handling and notifications
- Master election and promotion coordination

This centralizes all Redis Sentinel functionality from the monolithic manager.
"""

import asyncio
import logging
import time
import warnings
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis, Sentinel
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
        ResponseError,
        TimeoutError as RedisTimeoutError,
    )

    REDIS_AVAILABLE = True
except ImportError:
    warnings.warn("Redis not available. Install with: pip install redis", stacklevel=2)
    REDIS_AVAILABLE = False
    # Mock classes for type hints
    Redis = Any
    Sentinel = Any
    RedisConnectionError = Exception
    RedisTimeoutError = Exception
    ResponseError = Exception

from prompt_improver.database.services.connection.connection_metrics import (
    ConnectionMetrics,
)

logger = logging.getLogger(__name__)


class SentinelState(Enum):
    """Sentinel operational states."""

    INITIALIZING = "initializing"
    MONITORING = "monitoring"
    FAILOVER_IN_PROGRESS = "failover_in_progress"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class FailoverEvent(Enum):
    """Types of failover events."""

    MASTER_DOWN = "master_down"
    MASTER_SWITCH = "master_switch"
    REPLICA_PROMOTED = "replica_promoted"
    SENTINEL_DOWN = "sentinel_down"
    SENTINEL_UP = "sentinel_up"
    QUORUM_LOST = "quorum_lost"
    QUORUM_RESTORED = "quorum_restored"


@dataclass
class SentinelConfig:
    """Redis Sentinel configuration."""

    # Sentinel connection settings
    sentinel_hosts: list[tuple[str, int]]
    service_name: str = "mymaster"
    socket_timeout: float = 0.5
    password: str | None = None

    # Master/replica settings
    master_timeout: float = 2.0
    replica_timeout: float = 1.0
    max_connections: int = 50

    # Health and monitoring
    health_check_interval: int = 10
    failover_timeout: int = 180
    monitor_interval: int = 5

    # Quorum settings
    quorum: int = 2
    down_after_milliseconds: int = 30000
    failover_timeout_ms: int = 180000
    parallel_syncs: int = 1

    @classmethod
    def for_environment(
        cls, env: str, sentinels: list[tuple[str, int]] | None = None
    ) -> "SentinelConfig":
        """Create Sentinel configuration optimized for environment."""
        default_sentinels = sentinels or [
            ("localhost", 26379),
            ("localhost", 26380),
            ("localhost", 26381),
        ]

        configs = {
            "development": cls(
                sentinel_hosts=default_sentinels,
                socket_timeout=2.0,
                health_check_interval=30,
                monitor_interval=10,
                quorum=1,  # Lower quorum for dev
            ),
            "testing": cls(
                sentinel_hosts=default_sentinels[:2],  # Only 2 sentinels for testing
                socket_timeout=1.0,
                health_check_interval=15,
                monitor_interval=5,
                quorum=1,
                failover_timeout=60,
            ),
            "production": cls(
                sentinel_hosts=default_sentinels,
                socket_timeout=0.5,
                health_check_interval=5,
                monitor_interval=2,
                quorum=2,
                down_after_milliseconds=10000,  # Faster detection in prod
                failover_timeout_ms=120000,  # Faster failover in prod
            ),
        }
        return configs.get(env, configs["development"])


@dataclass
class SentinelInfo:
    """Information about a Redis Sentinel instance."""

    host: str
    port: int
    is_available: bool = True
    last_check: datetime = field(default_factory=lambda: datetime.now(UTC))
    response_time_ms: float = 0.0
    runid: str | None = None
    flags: set[str] = field(default_factory=set)
    pending_commands: int = 0

    @property
    def address(self) -> str:
        """Get sentinel address string."""
        return f"{self.host}:{self.port}"


@dataclass
class MasterInfo:
    """Information about Redis master instance."""

    name: str
    host: str
    port: int
    runid: str | None = None
    flags: set[str] = field(default_factory=set)
    last_ping: datetime = field(default_factory=lambda: datetime.now(UTC))
    num_slaves: int = 0
    num_other_sentinels: int = 0
    quorum: int = 0
    failover_timeout: int = 0
    down_after_period: int = 0

    @property
    def address(self) -> str:
        """Get master address string."""
        return f"{self.host}:{self.port}"

    @property
    def is_down(self) -> bool:
        """Check if master is marked as down."""
        return "s_down" in self.flags or "o_down" in self.flags


@dataclass
class FailoverEventInfo:
    """Information about a failover event."""

    event_type: FailoverEvent
    service_name: str
    old_master: str | None = None
    new_master: str | None = None
    sentinel: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)


class SentinelManager:
    """Advanced Redis Sentinel manager for high availability.

    Provides comprehensive Sentinel management with:
    - Multi-sentinel monitoring and coordination
    - Automatic master/replica failover handling
    - Sentinel health monitoring and validation
    - Failover event tracking and notifications
    - Master election and promotion coordination
    - Quorum management and split-brain prevention
    """

    def __init__(self, config: SentinelConfig, service_name: str = "sentinel_manager") -> None:
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis package not available. Install with: pip install redis"
            )

        self.config = config
        self.service_name = service_name

        # Sentinel connections
        self._sentinel_client: Sentinel | None = None
        self._master_client: Redis | None = None
        self._replica_clients: dict[str, Redis] = {}

        # State management
        self._state = SentinelState.INITIALIZING
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()

        # Sentinel tracking
        self._sentinel_info: dict[str, SentinelInfo] = {}
        self._master_info: MasterInfo | None = None
        self._current_master_address: tuple[str, int] | None = None

        # Failover tracking
        self._failover_events = deque(maxlen=100)
        self._last_failover_time = 0.0
        self._failover_in_progress = False

        # Metrics and monitoring
        self.metrics = ConnectionMetrics()
        self._operation_history = deque(maxlen=500)

        # Background tasks
        self._monitor_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None

        # Event callbacks
        self._failover_callbacks: list[callable] = []

        logger.info(f"SentinelManager initialized: {service_name}")
        logger.info(f"Sentinels: {[f'{h}:{p}' for h, p in config.sentinel_hosts]}")
        logger.info(f"Service: {config.service_name}, Quorum: {config.quorum}")

    async def initialize(self) -> bool:
        """Initialize the Sentinel manager."""
        async with self._initialization_lock:
            if self._is_initialized:
                return True

            try:
                await self._setup_sentinel_client()
                await self._discover_master()
                await self._discover_sentinels()

                # Start background monitoring
                self._monitor_task = asyncio.create_task(self._monitor_loop())
                self._health_check_task = asyncio.create_task(self._health_check_loop())

                self._state = SentinelState.MONITORING
                self._is_initialized = True

                logger.info(
                    f"SentinelManager initialized successfully: {self.service_name}"
                )
                return True

            except Exception as e:
                logger.exception(f"Failed to initialize SentinelManager: {e}")
                self._state = SentinelState.UNAVAILABLE
                raise

    async def _setup_sentinel_client(self) -> None:
        """Setup Redis Sentinel client."""
        sentinel_kwargs = {
            "sentinels": self.config.sentinel_hosts,
            "socket_timeout": self.config.socket_timeout,
        }

        if self.config.password:
            sentinel_kwargs["password"] = self.config.password

        self._sentinel_client = Sentinel(**sentinel_kwargs)

        logger.info(
            f"Sentinel client created with {len(self.config.sentinel_hosts)} sentinels"
        )

    async def _discover_master(self) -> None:
        """Discover current master from Sentinels."""
        if not self._sentinel_client:
            raise RuntimeError("Sentinel client not initialized")

        try:
            master_address = await self._sentinel_client.discover_master(
                self.config.service_name
            )

            if master_address:
                self._current_master_address = master_address

                # Get master client
                self._master_client = self._sentinel_client.master_for(
                    service_name=self.config.service_name,
                    socket_timeout=self.config.master_timeout,
                    max_connections=self.config.max_connections,
                )

                # Get master info
                master_info = await self._get_master_info()
                if master_info:
                    self._master_info = master_info

                logger.info(
                    f"Master discovered: {master_address[0]}:{master_address[1]}"
                )
            else:
                raise RuntimeError(
                    f"No master found for service: {self.config.service_name}"
                )

        except Exception as e:
            logger.exception(f"Failed to discover master: {e}")
            raise

    async def _discover_sentinels(self) -> None:
        """Discover and connect to all available Sentinels."""
        self._sentinel_info.clear()

        for host, port in self.config.sentinel_hosts:
            try:
                # Create individual sentinel connection for info
                individual_sentinel = Sentinel(
                    [(host, port)],
                    socket_timeout=self.config.socket_timeout,
                )

                # Try to ping this sentinel
                sentinel_conn = individual_sentinel.sentinels[0]
                await sentinel_conn.ping()

                # Get sentinel info
                info = SentinelInfo(
                    host=host,
                    port=port,
                    is_available=True,
                    last_check=datetime.now(UTC),
                    response_time_ms=0.0,  # Would measure in real implementation
                )

                self._sentinel_info[f"{host}:{port}"] = info
                logger.debug(f"Sentinel discovered: {host}:{port}")

            except Exception as e:
                logger.warning(f"Failed to connect to sentinel {host}:{port}: {e}")

                # Still track it as unavailable
                info = SentinelInfo(
                    host=host,
                    port=port,
                    is_available=False,
                    last_check=datetime.now(UTC),
                )
                self._sentinel_info[f"{host}:{port}"] = info

        available_count = sum(
            1 for info in self._sentinel_info.values() if info.is_available
        )
        logger.info(
            f"Sentinels discovered: {available_count}/{len(self.config.sentinel_hosts)} available"
        )

    async def _get_master_info(self) -> MasterInfo | None:
        """Get detailed information about the current master."""
        if not self._sentinel_client or not self._current_master_address:
            return None

        try:
            # Get master info from first available sentinel
            sentinel_conn = None
            for sentinel in self._sentinel_client.sentinels:
                try:
                    await sentinel.ping()
                    sentinel_conn = sentinel
                    break
                except Exception:
                    continue

            if not sentinel_conn:
                return None

            # Get master info
            master_info_raw = await sentinel_conn.execute_command(
                "SENTINEL", "MASTER", self.config.service_name
            )

            if master_info_raw:
                # Parse sentinel master info response
                info_dict = {}
                for i in range(0, len(master_info_raw), 2):
                    key = master_info_raw[i].decode("utf-8")
                    value = master_info_raw[i + 1].decode("utf-8")
                    info_dict[key] = value

                return MasterInfo(
                    name=self.config.service_name,
                    host=self._current_master_address[0],
                    port=self._current_master_address[1],
                    runid=info_dict.get("runid"),
                    flags=set(info_dict.get("flags", "").split(",")),
                    num_slaves=int(info_dict.get("num-slaves", 0)),
                    num_other_sentinels=int(info_dict.get("num-other-sentinels", 0)),
                    quorum=int(info_dict.get("quorum", 0)),
                    failover_timeout=int(info_dict.get("failover-timeout", 0)),
                    down_after_period=int(info_dict.get("down-after-milliseconds", 0)),
                )

        except Exception as e:
            logger.exception(f"Failed to get master info: {e}")

        return None

    @asynccontextmanager
    async def get_master_connection(self) -> AsyncIterator[Redis]:
        """Get connection to current master with failover support."""
        if not self._is_initialized:
            await self.initialize()

        if not self._master_client:
            raise RuntimeError("Master client not available")

        start_time = time.time()

        try:
            yield self._master_client

            # Record successful operation
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(duration_ms, success=True)
            self._record_operation("master_connection", duration_ms, True)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(duration_ms, success=False)
            self._record_operation("master_connection", duration_ms, False)

            # Handle potential failover
            await self._handle_master_failure(e)

            logger.exception(f"Master connection error: {e}")
            raise

    async def get_replica_connection(self, readonly: bool = True) -> Redis | None:
        """Get connection to a replica for read operations."""
        if not self._is_initialized:
            await self.initialize()

        if not self._sentinel_client:
            raise RuntimeError("Sentinel client not available")

        try:
            replica_client = self._sentinel_client.slave_for(
                service_name=self.config.service_name,
                socket_timeout=self.config.replica_timeout,
                max_connections=self.config.max_connections,
            )

            if readonly:
                await replica_client.readonly()

            return replica_client

        except Exception as e:
            logger.exception(f"Failed to get replica connection: {e}")
            return None

    async def _handle_master_failure(self, error: Exception) -> None:
        """Handle master connection failure and potential failover."""
        logger.warning(f"Master connection failed: {error}")

        # Record failover event
        event = FailoverEventInfo(
            event_type=FailoverEvent.MASTER_DOWN,
            service_name=self.config.service_name,
            old_master=f"{self._current_master_address[0]}:{self._current_master_address[1]}"
            if self._current_master_address
            else None,
            details={"error": str(error)},
        )
        self._failover_events.append(event)

        # Set failover state
        self._failover_in_progress = True
        self._state = SentinelState.FAILOVER_IN_PROGRESS
        self._last_failover_time = time.time()

        # Try to rediscover master
        try:
            await asyncio.sleep(1)  # Brief delay
            old_master = self._current_master_address
            await self._discover_master()

            # Check if master changed
            if self._current_master_address != old_master:
                switch_event = FailoverEventInfo(
                    event_type=FailoverEvent.MASTER_SWITCH,
                    service_name=self.config.service_name,
                    old_master=f"{old_master[0]}:{old_master[1]}"
                    if old_master
                    else None,
                    new_master=f"{self._current_master_address[0]}:{self._current_master_address[1]}"
                    if self._current_master_address
                    else None,
                )
                self._failover_events.append(switch_event)

                logger.info(
                    f"Master switched: {old_master} -> {self._current_master_address}"
                )

                # Notify callbacks
                await self._notify_failover_callbacks(switch_event)

            self._failover_in_progress = False
            self._state = SentinelState.MONITORING

        except Exception as e:
            logger.exception(f"Failed to handle master failure: {e}")
            self._state = SentinelState.DEGRADED

    async def add_failover_callback(self, callback: callable) -> None:
        """Add callback for failover events."""
        self._failover_callbacks.append(callback)

    async def _notify_failover_callbacks(self, event: FailoverEventInfo) -> None:
        """Notify all registered failover callbacks."""
        for callback in self._failover_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.exception(f"Failover callback error: {e}")

    async def force_failover(self) -> bool:
        """Force a failover to promote a replica to master."""
        if not self._sentinel_client:
            logger.error("Cannot force failover - Sentinel client not available")
            return False

        try:
            logger.info(f"Forcing failover for service: {self.config.service_name}")

            # Execute sentinel failover command
            sentinel_conn = self._sentinel_client.sentinels[0]  # Use first available
            result = await sentinel_conn.execute_command(
                "SENTINEL", "FAILOVER", self.config.service_name
            )

            if result:
                # Record forced failover event
                event = FailoverEventInfo(
                    event_type=FailoverEvent.REPLICA_PROMOTED,
                    service_name=self.config.service_name,
                    old_master=f"{self._current_master_address[0]}:{self._current_master_address[1]}"
                    if self._current_master_address
                    else None,
                    details={"forced": True},
                )
                self._failover_events.append(event)

                # Wait for failover to complete
                await asyncio.sleep(5)
                await self._discover_master()

                logger.info("Forced failover completed successfully")
                return True

        except Exception as e:
            logger.exception(f"Failed to force failover: {e}")

        return False

    async def get_sentinel_status(self) -> dict[str, Any]:
        """Get comprehensive Sentinel status."""
        available_sentinels = sum(
            1 for info in self._sentinel_info.values() if info.is_available
        )
        total_sentinels = len(self._sentinel_info)

        quorum_status = "healthy"
        if available_sentinels < self.config.quorum:
            quorum_status = "lost"
        elif available_sentinels < (total_sentinels / 2):
            quorum_status = "degraded"

        return {
            "service": self.service_name,
            "state": self._state.value,
            "sentinel_service": self.config.service_name,
            "sentinels": {
                "total": total_sentinels,
                "available": available_sentinels,
                "required_quorum": self.config.quorum,
                "quorum_status": quorum_status,
            },
            "master": {
                "address": f"{self._current_master_address[0]}:{self._current_master_address[1]}"
                if self._current_master_address
                else None,
                "is_down": self._master_info.is_down if self._master_info else None,
                "num_slaves": self._master_info.num_slaves if self._master_info else 0,
            },
            "failover": {
                "in_progress": self._failover_in_progress,
                "last_failover_ago_seconds": time.time() - self._last_failover_time
                if self._last_failover_time > 0
                else None,
                "recent_events": len([
                    e
                    for e in self._failover_events
                    if (datetime.now(UTC) - e.timestamp).seconds < 300
                ]),
            },
            "metrics": self.metrics.to_dict(),
        }

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of Sentinel manager."""
        start_time = time.time()

        health_info = {
            "status": "unknown",
            "timestamp": start_time,
            "service": self.service_name,
            "components": {},
            "response_time_ms": 0,
        }

        try:
            # Test sentinel connectivity
            available_sentinels = 0
            for address, info in self._sentinel_info.items():
                try:
                    host, port = address.split(":")
                    test_sentinel = Sentinel([(host, int(port))], socket_timeout=1.0)
                    await test_sentinel.sentinels[0].ping()
                    info.is_available = True
                    available_sentinels += 1
                    health_info["components"][f"sentinel_{address}"] = "healthy"
                except Exception as e:
                    info.is_available = False
                    health_info["components"][f"sentinel_{address}"] = (
                        f"error: {str(e)[:50]}"
                    )

            # Test master connectivity
            if self._master_client:
                try:
                    await self._master_client.ping()
                    health_info["components"]["master"] = "healthy"
                except Exception as e:
                    health_info["components"]["master"] = f"error: {str(e)[:50]}"
            else:
                health_info["components"]["master"] = "unavailable"

            # Overall health assessment
            quorum_met = available_sentinels >= self.config.quorum
            master_healthy = health_info["components"].get("master") == "healthy"

            if quorum_met and master_healthy and not self._failover_in_progress:
                health_info["status"] = "healthy"
            elif quorum_met and (master_healthy or not self._failover_in_progress):
                health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.exception(f"Sentinel health check failed: {e}")

        health_info["response_time_ms"] = (time.time() - start_time) * 1000
        return health_info

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

    async def _monitor_loop(self) -> None:
        """Background monitoring loop for Sentinel and master state."""
        while self._is_initialized:
            try:
                await asyncio.sleep(self.config.monitor_interval)

                # Check master status
                if self._master_client:
                    try:
                        await self._master_client.ping()
                    except Exception as e:
                        logger.warning(f"Master ping failed in monitor loop: {e}")
                        await self._handle_master_failure(e)

                # Update sentinel information
                await self._update_sentinel_info()

                # Check for quorum issues
                available_sentinels = sum(
                    1 for info in self._sentinel_info.values() if info.is_available
                )
                if available_sentinels < self.config.quorum and not any(
                    e.event_type == FailoverEvent.QUORUM_LOST
                    for e in [
                        e
                        for e in self._failover_events
                        if (datetime.now(UTC) - e.timestamp).seconds < 60
                    ]
                ):
                    event = FailoverEventInfo(
                        event_type=FailoverEvent.QUORUM_LOST,
                        service_name=self.config.service_name,
                        details={
                            "available": available_sentinels,
                            "required": self.config.quorum,
                        },
                    )
                    self._failover_events.append(event)
                    logger.error(
                        f"Quorum lost: {available_sentinels}/{self.config.quorum}"
                    )

            except Exception as e:
                logger.exception(f"Monitor loop error: {e}")

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._is_initialized:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.health_check()
            except Exception as e:
                logger.exception(f"Health check loop error: {e}")

    async def _update_sentinel_info(self) -> None:
        """Update information about all Sentinels."""
        for address, info in self._sentinel_info.items():
            try:
                host, port = address.split(":")
                test_sentinel = Sentinel([(host, int(port))], socket_timeout=1.0)

                start_time = time.time()
                await test_sentinel.sentinels[0].ping()
                duration = (time.time() - start_time) * 1000

                info.is_available = True
                info.last_check = datetime.now(UTC)
                info.response_time_ms = duration

            except Exception as e:
                info.is_available = False
                info.last_check = datetime.now(UTC)
                logger.debug(f"Sentinel {address} unavailable: {e}")

    async def shutdown(self) -> None:
        """Shutdown Sentinel manager and cleanup resources."""
        logger.info(f"Shutting down SentinelManager: {self.service_name}")

        try:
            # Cancel background tasks
            if self._monitor_task:
                self._monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._monitor_task

            if self._health_check_task:
                self._health_check_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._health_check_task

            # Close connections
            if self._master_client:
                await self._master_client.aclose()
                logger.info("Master client closed")

            for name, replica_client in self._replica_clients.items():
                try:
                    await replica_client.aclose()
                    logger.info(f"Replica client {name} closed")
                except Exception as e:
                    logger.warning(f"Error closing replica client {name}: {e}")

            # Clear state
            self._is_initialized = False
            self._sentinel_info.clear()
            self._replica_clients.clear()
            self._failover_callbacks.clear()

            logger.info(f"SentinelManager shutdown complete: {self.service_name}")

        except Exception as e:
            logger.exception(f"Error during SentinelManager shutdown: {e}")

    def __repr__(self) -> str:
        return (
            f"SentinelManager(service={self.service_name}, "
            f"state={self._state.value}, "
            f"sentinels={len(self._sentinel_info)}, "
            f"master={self._current_master_address})"
        )
