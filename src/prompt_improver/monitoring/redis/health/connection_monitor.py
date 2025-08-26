"""Redis Connection Monitor Service.

Focused Redis connection pool monitoring service for connection management and failover detection.
Designed for real-time connection health monitoring following SRE best practices.
"""

import asyncio
import contextlib
import logging
import time
from typing import Any

import coredis

from prompt_improver.monitoring.redis.health.types import (
    ConnectionPoolMetrics,
    ConnectionStatus,
)
from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry
from prompt_improver.shared.interfaces.protocols.monitoring import (
    RedisClientProviderProtocol,
)

logger = logging.getLogger(__name__)
_metrics_registry = get_metrics_registry()

# Connection monitoring metrics
CONNECTION_POOL_SIZE = _metrics_registry.get_or_create_gauge(
    "redis_connection_pool_size",
    "Current Redis connection pool size",
    ["pool_type"]
)

CONNECTION_UTILIZATION = _metrics_registry.get_or_create_gauge(
    "redis_connection_utilization_percent",
    "Redis connection pool utilization percentage"
)

CONNECTION_FAILURES = _metrics_registry.get_or_create_counter(
    "redis_connection_failures_total",
    "Total Redis connection failures",
    ["failure_type"]
)

CONNECTION_CREATION_TIME = _metrics_registry.get_or_create_histogram(
    "redis_connection_creation_duration_ms",
    "Time to create Redis connection in milliseconds"
)

CONNECTION_TIMEOUTS = _metrics_registry.get_or_create_counter(
    "redis_connection_timeouts_total",
    "Total Redis connection timeouts"
)


class RedisConnectionMonitor:
    """Redis connection monitor service for pool management and failover detection.

    Provides comprehensive connection health monitoring with real-time metrics
    and automated issue detection for SRE incident response.
    """

    def __init__(
        self,
        client_provider: RedisClientProviderProtocol,
        pool_monitor_interval: float = 30.0,
        connection_timeout: float = 5.0
    ) -> None:
        """Initialize Redis connection monitor.

        Args:
            client_provider: Redis client provider for connections
            pool_monitor_interval: Interval for pool monitoring in seconds
            connection_timeout: Timeout for connection operations
        """
        self.client_provider = client_provider
        self.pool_monitor_interval = pool_monitor_interval
        self.connection_timeout = connection_timeout

        # Connection state tracking
        self._last_metrics: ConnectionPoolMetrics | None = None
        self._connection_history: list[ConnectionPoolMetrics] = []
        self._max_history_size = 100

        # Issue detection
        self._detected_issues: list[str] = []
        self._issue_threshold_utilization = 80.0
        self._issue_threshold_failures = 5

        # Monitoring state
        self._is_monitoring = False
        self._monitor_task: asyncio.Task | None = None

    async def monitor_connections(self) -> ConnectionPoolMetrics:
        """Monitor connection pool health and utilization.

        Returns:
            Current connection pool metrics with detailed status
        """
        start_time = time.time()

        try:
            metrics = ConnectionPoolMetrics()
            metrics.status = ConnectionStatus.CONNECTED

            client = await self.client_provider.get_client()
            if not client:
                metrics.status = ConnectionStatus.DISCONNECTED
                return metrics

            # Collect connection info from Redis
            await self._collect_connection_info(client, metrics)

            # Analyze connection pool health
            await self._analyze_pool_health(metrics)

            # Update metrics registry
            CONNECTION_POOL_SIZE.labels(pool_type="active").set(metrics.active_connections)
            CONNECTION_POOL_SIZE.labels(pool_type="idle").set(metrics.idle_connections)
            CONNECTION_UTILIZATION.set(metrics.pool_utilization)

            # Cache metrics and update history
            self._last_metrics = metrics
            self._update_history(metrics)

            return metrics

        except Exception as e:
            logger.exception(f"Connection monitoring failed: {e}")

            CONNECTION_FAILURES.labels(failure_type="monitoring_error").inc()

            # Return failure metrics
            metrics = ConnectionPoolMetrics()
            metrics.status = ConnectionStatus.DISCONNECTED
            return metrics

    async def detect_connection_issues(self) -> list[str]:
        """Detect connection-related issues and bottlenecks.

        Returns:
            List of detected connection issues
        """
        issues = []

        if not self._last_metrics:
            await self.monitor_connections()

        if self._last_metrics:
            metrics = self._last_metrics

            # High utilization check
            if metrics.pool_utilization > self._issue_threshold_utilization:
                issues.append(
                    f"High connection pool utilization: {metrics.pool_utilization:.1f}%"
                )

            # Connection failure rate check
            if metrics.connection_failures > self._issue_threshold_failures:
                issues.append(
                    f"High connection failure rate: {metrics.connection_failures} failures"
                )

            # Connection timeout check
            if metrics.connection_timeouts > 0:
                issues.append(
                    f"Connection timeouts detected: {metrics.connection_timeouts}"
                )

            # Pool exhaustion check
            if metrics.active_connections >= metrics.max_pool_size * 0.95:
                issues.append(
                    "Connection pool near exhaustion - consider increasing pool size"
                )

            # Slow connection creation check
            if metrics.avg_connection_time_ms > 1000:
                issues.append(
                    f"Slow connection creation: {metrics.avg_connection_time_ms:.1f}ms"
                )

            # Connection leak detection
            idle_ratio = (
                metrics.idle_connections / (metrics.active_connections + metrics.idle_connections)
                if (metrics.active_connections + metrics.idle_connections) > 0 else 0
            )
            if idle_ratio < 0.1 and metrics.active_connections > 10:
                issues.append(
                    "Possible connection leak - very few idle connections"
                )

        self._detected_issues = issues
        return issues

    async def get_connection_stats(self) -> dict[str, Any]:
        """Get detailed connection statistics.

        Returns:
            Comprehensive connection statistics dictionary
        """
        try:
            client = await self.client_provider.get_client()
            if not client:
                return {"error": "Redis client not available"}

            # Get Redis info
            info = await client.info()

            # Get client list for detailed analysis
            client_list = await self._get_client_list(client)

            return {
                "connection_info": {
                    "connected_clients": self._safe_int(info.get("connected_clients", 0)),
                    "maxclients": self._safe_int(info.get("maxclients", 10000)),
                    "total_connections_received": self._safe_int(
                        info.get("total_connections_received", 0)
                    ),
                    "rejected_connections": self._safe_int(
                        info.get("rejected_connections", 0)
                    ),
                    "blocked_clients": self._safe_int(info.get("blocked_clients", 0)),
                    "tracking_clients": self._safe_int(info.get("tracking_clients", 0)),
                },
                "client_analysis": client_list,
                "buffer_info": {
                    "client_recent_max_input_buffer": self._safe_int(
                        info.get("client_recent_max_input_buffer", 0)
                    ),
                    "client_recent_max_output_buffer": self._safe_int(
                        info.get("client_recent_max_output_buffer", 0)
                    ),
                },
                "last_metrics": (
                    self._last_metrics.__dict__ if self._last_metrics else None
                ),
                "detected_issues": self._detected_issues,
                "monitoring_status": {
                    "is_monitoring": self._is_monitoring,
                    "history_size": len(self._connection_history),
                }
            }

        except Exception as e:
            logger.exception(f"Failed to get connection stats: {e}")
            return {"error": str(e)}

    async def validate_connection_pool(self) -> bool:
        """Validate connection pool health.

        Returns:
            True if connection pool is healthy
        """
        try:
            metrics = await self.monitor_connections()

            # Pool is healthy if:
            # 1. Connection status is good
            # 2. Utilization is reasonable
            # 3. No critical issues detected

            if metrics.status == ConnectionStatus.DISCONNECTED:
                return False

            if metrics.pool_utilization > 90.0:
                return False

            issues = await self.detect_connection_issues()
            critical_issues = [
                issue for issue in issues
                if any(keyword in issue.lower() for keyword in ["timeout", "failure", "exhaustion"])
            ]

            return len(critical_issues) == 0

        except Exception as e:
            logger.exception(f"Connection pool validation failed: {e}")
            return False

    async def start_monitoring(self) -> None:
        """Start continuous connection monitoring."""
        if self._is_monitoring:
            logger.warning("Connection monitoring already started")
            return

        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Redis connection monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop continuous connection monitoring."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None

        logger.info("Redis connection monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Continuous monitoring loop."""
        logger.info("Starting Redis connection monitor loop")

        while self._is_monitoring:
            try:
                await self.monitor_connections()
                await self.detect_connection_issues()

                # Log critical issues
                if self._detected_issues:
                    logger.warning(f"Redis connection issues detected: {self._detected_issues}")

                await asyncio.sleep(self.pool_monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in connection monitor loop: {e}")
                await asyncio.sleep(5)  # Brief delay on error

    async def _collect_connection_info(
        self,
        client: coredis.Redis,
        metrics: ConnectionPoolMetrics
    ) -> None:
        """Collect connection information from Redis.

        Args:
            client: Redis client instance
            metrics: ConnectionPoolMetrics object to populate
        """
        try:
            # Get Redis info
            info = await asyncio.wait_for(
                client.info(),
                timeout=self.connection_timeout
            )

            # Basic connection metrics
            metrics.active_connections = self._safe_int(info.get("connected_clients", 0))
            metrics.max_pool_size = self._safe_int(info.get("maxclients", 10000))

            # Connection lifecycle metrics
            metrics.connections_created = self._safe_int(
                info.get("total_connections_received", 0)
            )
            metrics.connection_failures = self._safe_int(
                info.get("rejected_connections", 0)
            )

            # Calculate pool utilization
            metrics.calculate_utilization()

            # Measure connection creation time
            await self._measure_connection_time(metrics)

        except TimeoutError:
            logger.warning("Timeout collecting connection info")
            metrics.status = ConnectionStatus.DEGRADED
            CONNECTION_TIMEOUTS.inc()
        except Exception as e:
            logger.exception(f"Failed to collect connection info: {e}")
            metrics.status = ConnectionStatus.DEGRADED

    async def _analyze_pool_health(self, metrics: ConnectionPoolMetrics) -> None:
        """Analyze connection pool health and set status.

        Args:
            metrics: ConnectionPoolMetrics object to analyze
        """
        # Determine connection status based on utilization and issues
        if metrics.pool_utilization > 95.0 or metrics.pool_utilization > 85.0 or metrics.connection_failures > self._issue_threshold_failures:
            metrics.status = ConnectionStatus.DEGRADED
        else:
            metrics.status = ConnectionStatus.CONNECTED

    async def _measure_connection_time(self, metrics: ConnectionPoolMetrics) -> None:
        """Measure connection creation time.

        Args:
            metrics: ConnectionPoolMetrics object to update
        """
        try:
            # Test connection creation time
            start_time = time.time()

            test_client = await self.client_provider.get_client()
            if test_client:
                await asyncio.wait_for(
                    test_client.ping(),
                    timeout=self.connection_timeout
                )

            end_time = time.time()
            connection_time_ms = (end_time - start_time) * 1000

            metrics.avg_connection_time_ms = connection_time_ms
            CONNECTION_CREATION_TIME.observe(connection_time_ms)

        except Exception as e:
            logger.debug(f"Failed to measure connection time: {e}")
            # Don't fail the entire monitoring for connection timing issues

    async def _get_client_list(self, client: coredis.Redis) -> dict[str, Any]:
        """Get detailed client list analysis.

        Args:
            client: Redis client instance

        Returns:
            Client analysis dictionary
        """
        try:
            # Get client list (this can be expensive, so we limit it)
            client_info = await asyncio.wait_for(
                client.client_list(),
                timeout=self.connection_timeout
            )

            # Analyze client patterns
            analysis = {
                "total_clients": len(client_info),
                "client_types": {},
                "idle_clients": 0,
                "long_running_clients": 0,
                "blocked_clients": 0,
            }

            for client_data in client_info[:50]:  # Limit analysis to first 50 clients
                # Client type analysis
                name = client_data.get("name", "unnamed")
                client_type = name.split(":")[0] if ":" in name else "unknown"
                analysis["client_types"][client_type] = (
                    analysis["client_types"].get(client_type, 0) + 1
                )

                # Idle analysis
                idle_time = self._safe_int(client_data.get("idle", 0))
                if idle_time > 300:  # 5 minutes
                    analysis["idle_clients"] += 1

                # Long-running analysis
                age = self._safe_int(client_data.get("age", 0))
                if age > 3600:  # 1 hour
                    analysis["long_running_clients"] += 1

                # Blocked client analysis
                if "flags" in client_data and "b" in client_data["flags"]:
                    analysis["blocked_clients"] += 1

            return analysis

        except Exception as e:
            logger.debug(f"Failed to get client list: {e}")
            return {"error": str(e)}

    def _update_history(self, metrics: ConnectionPoolMetrics) -> None:
        """Update connection metrics history.

        Args:
            metrics: Current metrics to add to history
        """
        self._connection_history.append(metrics)

        # Keep history size manageable
        if len(self._connection_history) > self._max_history_size:
            self._connection_history = self._connection_history[-self._max_history_size:]

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def get_connection_trend(self) -> dict[str, Any]:
        """Get connection trend analysis from history.

        Returns:
            Connection trend analysis
        """
        if not self._connection_history:
            return {"error": "No connection history available"}

        recent_metrics = self._connection_history[-10:]  # Last 10 samples

        avg_utilization = sum(m.pool_utilization for m in recent_metrics) / len(recent_metrics)
        avg_active = sum(m.active_connections for m in recent_metrics) / len(recent_metrics)
        total_failures = sum(m.connection_failures for m in recent_metrics)

        return {
            "sample_count": len(recent_metrics),
            "avg_utilization_percent": round(avg_utilization, 2),
            "avg_active_connections": round(avg_active, 1),
            "total_failures": total_failures,
            "trend": "increasing" if recent_metrics[-1].pool_utilization > avg_utilization else "stable"
        }
