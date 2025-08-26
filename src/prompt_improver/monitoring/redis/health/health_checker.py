"""Redis Health Checker Service.

Focused Redis health checking service for connection status and response time monitoring.
Designed for <25ms operations with real-time monitoring capabilities following SRE best practices.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from prompt_improver.monitoring.redis.health.types import (
    HealthMetrics,
    RedisHealthStatus,
)
from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry
from prompt_improver.shared.interfaces.protocols.monitoring import (
    RedisClientProviderProtocol,
)

logger = logging.getLogger(__name__)
_metrics_registry = get_metrics_registry()

# Metrics for monitoring
HEALTH_CHECK_DURATION = _metrics_registry.get_or_create_histogram(
    "redis_health_check_duration_ms",
    "Redis health check duration in milliseconds",
    ["operation"]
)

HEALTH_CHECK_ERRORS = _metrics_registry.get_or_create_counter(
    "redis_health_check_errors_total",
    "Total Redis health check errors",
    ["error_type", "operation"]
)

CONNECTIVITY_STATUS = _metrics_registry.get_or_create_gauge(
    "redis_connectivity_status",
    "Redis connectivity status (1=healthy, 0=unhealthy)"
)

PING_LATENCY = _metrics_registry.get_or_create_histogram(
    "redis_ping_latency_ms",
    "Redis ping latency in milliseconds"
)


class RedisHealthChecker:
    """Redis health checker service for connection status and response time monitoring.

    Provides fast, reliable health checking with SRE-focused metrics and alerting.
    All operations are designed to complete in <25ms for real-time monitoring.
    """

    def __init__(
        self,
        client_provider: RedisClientProviderProtocol,
        timeout_seconds: float = 5.0,
        max_retries: int = 3
    ) -> None:
        """Initialize Redis health checker.

        Args:
            client_provider: Redis client provider for connections
            timeout_seconds: Operation timeout for health checks
            max_retries: Maximum retry attempts for failed checks
        """
        self.client_provider = client_provider
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Health state tracking
        self._last_metrics: HealthMetrics | None = None
        self._consecutive_failures = 0
        self._last_success_time: datetime | None = None
        self._is_monitoring = False

        # Performance tracking
        self._ping_samples: list[float] = []
        self._max_ping_samples = 100

    async def check_health(self) -> HealthMetrics:
        """Perform comprehensive health check with timing and error tracking.

        Returns:
            Current health metrics with detailed status information
        """
        start_time = time.time()

        try:
            with HEALTH_CHECK_DURATION.labels(operation="comprehensive").time():
                metrics = HealthMetrics()
                metrics.last_check_time = datetime.now(UTC)

                # Test basic connectivity
                ping_latency = await self._ping_with_retry()
                metrics.ping_latency_ms = ping_latency

                if ping_latency > 0:
                    # Successful ping - collect additional metrics
                    await self._collect_basic_info(metrics)

                    # Update success tracking
                    self._consecutive_failures = 0
                    self._last_success_time = datetime.now(UTC)
                    metrics.is_available = True
                    CONNECTIVITY_STATUS.set(1)

                else:
                    # Failed ping - mark as unavailable
                    self._consecutive_failures += 1
                    metrics.is_available = False
                    metrics.error_count = self._consecutive_failures
                    CONNECTIVITY_STATUS.set(0)

                    HEALTH_CHECK_ERRORS.labels(
                        error_type="connectivity",
                        operation="ping"
                    ).inc()

                # Calculate final status
                metrics.consecutive_failures = self._consecutive_failures
                metrics.status = metrics.get_status()

                # Update timing
                end_time = time.time()
                metrics.check_duration_ms = (end_time - start_time) * 1000

                # Cache metrics for fast access
                self._last_metrics = metrics

                return metrics

        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            self._consecutive_failures += 1

            HEALTH_CHECK_ERRORS.labels(
                error_type="exception",
                operation="check_health"
            ).inc()

            # Return failure metrics
            metrics = HealthMetrics()
            metrics.last_check_time = datetime.now(UTC)
            metrics.is_available = False
            metrics.consecutive_failures = self._consecutive_failures
            metrics.last_error = str(e)
            metrics.status = RedisHealthStatus.FAILED
            metrics.check_duration_ms = (time.time() - start_time) * 1000

            CONNECTIVITY_STATUS.set(0)
            self._last_metrics = metrics

            return metrics

    async def ping(self) -> float:
        """Check Redis connectivity with latency measurement.

        Returns:
            Ping latency in milliseconds, or -1 if failed
        """
        try:
            with HEALTH_CHECK_DURATION.labels(operation="ping").time():
                return await self._ping_with_retry()

        except Exception as e:
            logger.exception(f"Ping failed: {e}")
            HEALTH_CHECK_ERRORS.labels(
                error_type="ping_failure",
                operation="ping"
            ).inc()
            return -1.0

    async def is_available(self) -> bool:
        """Quick availability check for circuit breaker logic.

        Returns:
            True if Redis is available based on last health check
        """
        if self._last_metrics:
            return self._last_metrics.is_available

        # If no cached metrics, do a quick ping
        ping_result = await self.ping()
        return ping_result > 0

    def get_last_metrics(self) -> HealthMetrics | None:
        """Get cached health metrics for fast access.

        Returns:
            Last collected health metrics or None if no checks performed
        """
        return self._last_metrics

    async def _ping_with_retry(self) -> float:
        """Ping Redis with retry logic and latency measurement.

        Returns:
            Average ping latency in milliseconds across retries
        """
        latencies = []
        last_error = None

        for attempt in range(self.max_retries):
            try:
                client = await self.client_provider.get_client()
                if not client:
                    raise ConnectionError("Redis client not available")

                # Measure ping latency
                start_time = time.time()
                await asyncio.wait_for(
                    client.ping(),
                    timeout=self.timeout_seconds
                )
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                # Record latency metric
                PING_LATENCY.observe(latency_ms)

                # Update ping samples for trend analysis
                self._ping_samples.append(latency_ms)
                if len(self._ping_samples) > self._max_ping_samples:
                    self._ping_samples = self._ping_samples[-self._max_ping_samples:]

                # Return average latency if we have multiple samples
                if len(latencies) > 1:
                    return sum(latencies) / len(latencies)
                return latency_ms

            except (TimeoutError, ConnectionError, OSError) as e:
                last_error = e
                logger.warning(f"Ping attempt {attempt + 1} failed: {e}")

                # Short delay before retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.1)

        # All retries failed
        if last_error:
            logger.error(f"All ping attempts failed: {last_error}")

        return -1.0

    async def _collect_basic_info(self, metrics: HealthMetrics) -> None:
        """Collect basic Redis information for health assessment.

        Args:
            metrics: HealthMetrics object to populate
        """
        try:
            client = await self.client_provider.get_client()
            if not client:
                return

            # Get basic info with timeout
            info = await asyncio.wait_for(
                client.info(),
                timeout=self.timeout_seconds
            )

            # Extract key metrics
            metrics.connection_count = self._safe_int(info.get("connected_clients", 0))
            metrics.max_connections = self._safe_int(info.get("maxclients", 10000))

            if metrics.max_connections > 0:
                metrics.connection_utilization = (
                    metrics.connection_count / metrics.max_connections * 100
                )

            # Memory metrics
            used_memory = self._safe_int(info.get("used_memory", 0))
            metrics.memory_usage_mb = used_memory / 1024 / 1024
            metrics.memory_fragmentation_ratio = self._safe_float(
                info.get("mem_fragmentation_ratio", 1.0)
            )

            # Performance metrics
            hits = self._safe_int(info.get("keyspace_hits", 0))
            misses = self._safe_int(info.get("keyspace_misses", 0))
            total_ops = hits + misses

            if total_ops > 0:
                metrics.hit_rate = (hits / total_ops) * 100

            metrics.ops_per_second = self._safe_float(
                info.get("instantaneous_ops_per_sec", 0)
            )

        except Exception as e:
            logger.warning(f"Failed to collect basic info: {e}")
            # Don't fail the entire health check for info collection failures

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def get_ping_trend(self) -> dict[str, float]:
        """Get ping latency trend analysis.

        Returns:
            Dictionary with ping latency statistics
        """
        if not self._ping_samples:
            return {
                "count": 0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "recent_avg_ms": 0.0
            }

        # Calculate statistics
        avg_ms = sum(self._ping_samples) / len(self._ping_samples)
        min_ms = min(self._ping_samples)
        max_ms = max(self._ping_samples)

        # Recent average (last 10 samples)
        recent_samples = self._ping_samples[-10:]
        recent_avg_ms = sum(recent_samples) / len(recent_samples)

        return {
            "count": len(self._ping_samples),
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "recent_avg_ms": round(recent_avg_ms, 2)
        }

    def reset_failure_count(self) -> None:
        """Reset consecutive failure count for manual recovery."""
        self._consecutive_failures = 0
        logger.info("Redis health checker failure count reset")

    def get_uptime_info(self) -> dict[str, Any]:
        """Get uptime and availability information.

        Returns:
            Dictionary with uptime statistics
        """
        now = datetime.now(UTC)

        return {
            "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
            "consecutive_failures": self._consecutive_failures,
            "seconds_since_last_success": (
                (now - self._last_success_time).total_seconds()
                if self._last_success_time else None
            ),
            "is_available": self._last_metrics.is_available if self._last_metrics else False,
            "current_status": self._last_metrics.status.value if self._last_metrics else "unknown"
        }
