"""Cache Monitoring Service for health checks and performance metrics.

Provides comprehensive monitoring, alerting, and observability for the
multi-level cache system. Includes SLO compliance tracking, health checks,
and metrics collection for monitoring systems.
"""

import logging
import statistics
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from prompt_improver.services.cache.l1_cache_service import L1CacheService
from prompt_improver.services.cache.l2_redis_service import L2RedisService

logger = logging.getLogger(__name__)

# Optional OpenTelemetry integration
try:
    from opentelemetry import metrics, trace

    _opentelemetry_available = True
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)

    # Create metrics
    cache_operations_counter = meter.create_counter(
        "cache_operations_total",
        description="Total cache operations by type, level, and status",
        unit="1",
    )
    cache_hit_ratio_gauge = meter.create_gauge(
        "cache_hit_ratio",
        description="Cache hit ratio by level and type",
        unit="ratio",
    )
    cache_latency_histogram = meter.create_histogram(
        "cache_operation_duration_seconds",
        description="Cache operation duration by type and level",
        unit="s",
    )
    cache_size_gauge = meter.create_gauge(
        "cache_size_current",
        description="Current cache size by level",
        unit="1",
    )
    cache_memory_usage_gauge = meter.create_gauge(
        "cache_memory_usage_bytes",
        description="Estimated cache memory usage",
        unit="bytes",
    )
    cache_error_counter = meter.create_counter(
        "cache_errors_total",
        description="Cache errors by type and operation",
        unit="1",
    )

except ImportError:
    _opentelemetry_available = False
    tracer = None
    meter = None
    cache_operations_counter = None
    cache_hit_ratio_gauge = None
    cache_latency_histogram = None
    cache_size_gauge = None
    cache_memory_usage_gauge = None
    cache_error_counter = None

# Provide access to the flag
OPENTELEMETRY_AVAILABLE = _opentelemetry_available


class CacheMonitoringService:
    """Monitoring service for multi-level cache system.

    Provides health checks, performance metrics, SLO compliance tracking,
    and alerting capabilities. Integrates with OpenTelemetry when available.

    Monitoring capabilities:
    - Real-time health checks (<25ms)
    - Performance metrics collection
    - SLO compliance tracking (95% < 200ms)
    - Alert thresholds and notifications
    - OpenTelemetry integration
    """

    def __init__(self, l1_cache: L1CacheService, l2_cache: L2RedisService | None = None) -> None:
        """Initialize cache monitoring service.

        Args:
            l1_cache: L1 cache service to monitor
            l2_cache: L2 cache service to monitor (optional)
        """
        self._l1_cache = l1_cache
        self._l2_cache = l2_cache
        self._created_at = datetime.now(UTC)

        # Response time tracking for SLO calculation (optimized with deque)
        self._response_times: deque[float] = deque(maxlen=1000)  # Automatic size limit for performance
        self._max_response_time_samples = 1000

        # Error tracking with bounded memory usage
        self._error_counts: dict[str, int] = {}
        self._max_error_types = 500  # Prevent memory leak
        self._last_error_time: datetime | None = None
        self._consecutive_errors = 0

        # Performance tracking for direct cache operations
        self._total_requests = 0
        self._total_response_time = 0.0
        self._l1_hits = 0
        self._l2_hits = 0

        # Health status tracking (coordinator removed)
        self._health_status = {
            "overall_health": "healthy",
            "l1_health": "healthy",
            "l2_health": "healthy",
            "last_health_check": datetime.now(UTC),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check on cache system.

        Tests all cache levels and coordinator functionality.
        Target: Complete health check in <100ms.

        Returns:
            Comprehensive health check results
        """
        start_time = time.perf_counter()

        try:
            # Test direct cache operations (no coordinator overhead)
            test_key = f"health_check_{int(time.time())}"
            test_value = {"test": True, "timestamp": time.time()}

            # Test L1 set operation
            set_start = time.perf_counter()
            await self._l1_cache.set(test_key, test_value)
            if self._l2_cache:
                await self._l2_cache.set(test_key, test_value, 300)  # 5 min TTL
            set_time = time.perf_counter() - set_start

            # Test L1 get operation
            get_start = time.perf_counter()
            retrieved_value = await self._l1_cache.get(test_key)
            get_time = time.perf_counter() - get_start

            # Test delete operation
            delete_start = time.perf_counter()
            await self._l1_cache.delete(test_key)
            if self._l2_cache:
                await self._l2_cache.delete(test_key)
            delete_time = time.perf_counter() - delete_start

            # Validate results (direct operations should be much faster)
            operations_healthy = (
                retrieved_value is not None and
                retrieved_value.get("test") is True and
                set_time < 0.01 and  # 10ms (much faster without coordination)
                get_time < 0.005 and  # 5ms for L1 get
                delete_time < 0.01   # 10ms
            )

            # Get individual cache health directly
            l1_stats = self._l1_cache.get_stats()
            l2_stats = self._l2_cache.get_stats() if self._l2_cache else None

            # Update health status (coordinator removed)
            self._health_status.update({
                "l1_health": l1_stats["health_status"],
                "l2_health": l2_stats["health_status"] if l2_stats else "disabled",
                "last_health_check": datetime.now(UTC),
            })

            # Determine overall health (coordinator removed)
            component_healths = [
                self._health_status["l1_health"],
                self._health_status["l2_health"] if l2_stats else "healthy",
            ]

            if "unhealthy" in component_healths:
                self._health_status["overall_health"] = "unhealthy"
            elif "degraded" in component_healths:
                self._health_status["overall_health"] = "degraded"
            else:
                self._health_status["overall_health"] = "healthy"

            total_time = time.perf_counter() - start_time

            return {
                "healthy": operations_healthy and self._health_status["overall_health"] != "unhealthy",
                "overall_status": self._health_status["overall_health"],
                "checks": {
                    "direct_operations": {
                        "healthy": operations_healthy,
                        "set_time_ms": set_time * 1000,
                        "get_time_ms": get_time * 1000,
                        "delete_time_ms": delete_time * 1000,
                        "value_match": retrieved_value == test_value,
                        "architecture": "direct_cache_aside_pattern",
                    },
                    "cache_levels": {
                        "l1": {
                            "healthy": l1_stats["health_status"] == "healthy",
                            "status": l1_stats["health_status"],
                            "hit_rate": l1_stats["hit_rate"],
                            "size": l1_stats["size"],
                        },
                        "l2": {
                            "healthy": l2_stats["health_status"] == "healthy" if l2_stats else False,
                            "status": l2_stats["health_status"] if l2_stats else "disabled",
                            "enabled": l2_stats is not None,
                        } if l2_stats else {"enabled": False, "status": "disabled"},
                    },
                },
                "performance": {
                    "total_check_time_ms": total_time * 1000,
                    "meets_slo": total_time < 0.025,  # 25ms SLO (much faster without coordination)
                    "l1_hit_rate": l1_stats["hit_rate"],
                    "l2_hit_rate": l2_stats["hit_rate"] if l2_stats else 0.0,
                    "coordination_overhead": "eliminated",
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Cache health check failed: {e}")
            self._record_error("health_check", e)

            return {
                "healthy": False,
                "overall_status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def get_monitoring_metrics(self) -> dict[str, Any]:
        """Get metrics optimized for monitoring systems and dashboards.

        Returns:
            Flattened metrics suitable for time-series databases
        """
        stats = self._get_combined_performance_stats()

        metrics = {
            # Overall performance
            "cache.hit_rate.overall": stats["overall_hit_rate"],
            "cache.response_time.avg_ms": stats["avg_response_time_ms"],
            "cache.requests.total": stats["total_requests"],
            "cache.health.overall": self._health_status["overall_health"],

            # L1 metrics
            "cache.l1.hit_rate": stats["l1_hit_rate"],
            "cache.l1.hits": stats["l1_hits"],
            "cache.l1.size": stats["l1_cache_stats"]["size"],
            "cache.l1.memory_bytes": stats["l1_cache_stats"]["estimated_memory_bytes"],
            "cache.l1.health": stats["l1_cache_stats"]["health_status"],

            # L2 metrics (if available)
            "cache.l2.enabled": stats["l2_cache_stats"] is not None,
            "cache.l2.hit_rate": stats["l2_hit_rate"],
            "cache.l2.hits": stats["l2_hits"],
        }

        # Add L2-specific metrics if available
        if stats["l2_cache_stats"]:
            l2_stats = stats["l2_cache_stats"]
            metrics.update({
                "cache.l2.success_rate": l2_stats["success_rate"],
                "cache.l2.health": l2_stats["health_status"],
                "cache.l2.connected": l2_stats["currently_connected"],
            })

        # Database cache eliminated per 2025 best practices
        metrics["cache.database.enabled"] = False

        # Cache warming metrics
        metrics.update({
            "cache.warming.enabled": stats["warming_enabled"],
            "cache.warming.tracked_patterns": stats["tracked_patterns"],
        })

        return metrics

    def get_alert_metrics(self) -> dict[str, Any]:
        """Get metrics for alerting systems with threshold evaluations.

        Returns:
            Alert metrics with boolean flags and threshold comparisons
        """
        stats = self._get_combined_performance_stats()
        slo_compliance = self.calculate_slo_compliance()

        # Define alert thresholds
        THRESHOLDS = {
            "hit_rate_critical": 0.5,
            "hit_rate_warning": 0.7,
            "response_time_critical_ms": 200,
            "response_time_warning_ms": 100,
            "slo_compliance_critical": 0.90,
            "slo_compliance_warning": 0.95,
            "error_rate_critical": 0.1,
            "error_rate_warning": 0.05,
            "consecutive_errors_critical": 10,
            "memory_usage_warning": 0.8,
        }

        # Calculate current values
        hit_rate = stats["overall_hit_rate"]
        response_time_ms = stats["avg_response_time_ms"]
        compliance_rate = slo_compliance["compliance_rate"]
        error_rate = self._calculate_error_rate()
        memory_usage = self._calculate_memory_utilization(stats)

        return {
            "alerts": {
                # Hit rate alerts
                "hit_rate_critical": hit_rate < THRESHOLDS["hit_rate_critical"],
                "hit_rate_warning": hit_rate < THRESHOLDS["hit_rate_warning"],

                # Response time alerts
                "response_time_critical": response_time_ms > THRESHOLDS["response_time_critical_ms"],
                "response_time_warning": response_time_ms > THRESHOLDS["response_time_warning_ms"],

                # SLO compliance alerts
                "slo_compliance_critical": compliance_rate < THRESHOLDS["slo_compliance_critical"],
                "slo_compliance_warning": compliance_rate < THRESHOLDS["slo_compliance_warning"],

                # Error rate alerts
                "error_rate_critical": error_rate > THRESHOLDS["error_rate_critical"],
                "error_rate_warning": error_rate > THRESHOLDS["error_rate_warning"],

                # System health alerts
                "system_unhealthy": self._health_status["overall_health"] == "unhealthy",
                "system_degraded": self._health_status["overall_health"] == "degraded",
                "consecutive_errors_critical": self._consecutive_errors > THRESHOLDS["consecutive_errors_critical"],
                "memory_usage_warning": memory_usage > THRESHOLDS["memory_usage_warning"],
            },
            "values": {
                "hit_rate": hit_rate,
                "response_time_ms": response_time_ms,
                "slo_compliance": compliance_rate,
                "error_rate": error_rate,
                "consecutive_errors": self._consecutive_errors,
                "memory_utilization": memory_usage,
                "overall_health": self._health_status["overall_health"],
            },
            "thresholds": THRESHOLDS,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def calculate_slo_compliance(self) -> dict[str, Any]:
        """Calculate SLO compliance metrics (target: 95% of operations < 200ms).

        Returns:
            SLO compliance statistics
        """
        if not self._response_times:
            return {
                "compliant": True,
                "slo_target_ms": 200,
                "compliance_rate": 1.0,
                "sample_count": 0,
            }

        slo_target = 0.2  # 200ms
        compliant_requests = sum(1 for t in self._response_times if t <= slo_target)
        total_requests = len(self._response_times)
        compliance_rate = compliant_requests / total_requests

        # Calculate percentiles with proper interpolation
        sorted_times = sorted(self._response_times)
        percentiles: dict[str, float] = {}
        for p in [50, 95, 99]:
            percentiles[f"p{p}"] = self._calculate_percentile(sorted_times, p)

        return {
            "compliant": compliance_rate >= 0.95,
            "slo_target_ms": slo_target * 1000,
            "compliance_rate": compliance_rate,
            "compliant_requests": compliant_requests,
            "total_requests": total_requests,
            "violations": total_requests - compliant_requests,
            "percentiles": {key: value * 1000 for key, value in percentiles.items()},  # Convert to ms
            "mean_ms": statistics.mean(self._response_times) * 1000,
            "sample_count": total_requests,
        }

    def record_operation(self, operation: str, duration: float, success: bool = True) -> None:
        """Record cache operation for monitoring.

        Args:
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation succeeded
        """
        # Record response time for SLO calculation (deque auto-manages size)
        self._response_times.append(duration)

        # Track errors
        if not success:
            self._record_error(operation, Exception(f"{operation} failed"))
        else:
            self._consecutive_errors = 0

        # Update OpenTelemetry metrics if available
        if OPENTELEMETRY_AVAILABLE:
            self._update_opentelemetry_metrics(operation, duration, success)

    def _record_error(self, operation: str, error: Exception) -> None:
        """Record error for monitoring and alerting."""
        error_key = f"{operation}_{type(error).__name__}"
        error_count = self._error_counts.get(error_key, 0) + 1
        self._error_counts[error_key] = error_count
        self._consecutive_errors += 1
        self._last_error_time = datetime.now(UTC)

        # Prevent memory leak: clean up old error counts when limit reached
        if len(self._error_counts) > self._max_error_types:
            self._cleanup_old_error_counts()

        # Throttle error logging to prevent log spam - only log first occurrence and every 50th occurrence per error type
        if error_count == 1 or error_count % 50 == 0:
            logger.warning(f"Cache error in {operation}: {error} (occurrences: {error_count}, consecutive_total: {self._consecutive_errors})")

    def _cleanup_old_error_counts(self) -> None:
        """Clean up old error counts to prevent memory leak."""
        if len(self._error_counts) <= self._max_error_types // 2:
            return

        # Keep only the most frequent errors (LRU-style cleanup)
        sorted_errors = sorted(
            self._error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Keep top half of errors by frequency
        self._error_counts = dict(sorted_errors[:self._max_error_types // 2])
        logger.info(f"Cleaned up error counts, kept {len(self._error_counts)} most frequent errors")

    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate."""
        stats = self._get_combined_performance_stats()
        total_errors = sum(self._error_counts.values())
        total_requests = stats["total_requests"]

        return total_errors / max(total_requests, 1)

    def _calculate_memory_utilization(self, stats: dict[str, Any]) -> float:
        """Calculate memory utilization across cache levels."""
        l1_stats = stats["l1_cache_stats"]
        return l1_stats["utilization"]

    def _calculate_percentile(self, sorted_values: list[float], percentile: int) -> float:
        """Calculate percentile with proper interpolation.

        Args:
            sorted_values: Pre-sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value with interpolation
        """
        if not sorted_values:
            return 0.0

        # Use proper percentile calculation with interpolation
        index = percentile / 100.0 * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]

        # Interpolate between adjacent values
        lower_index = int(index)
        upper_index = lower_index + 1

        if upper_index >= len(sorted_values):
            return sorted_values[-1]

        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]

        # Linear interpolation
        return lower_value + (upper_value - lower_value) * (index - lower_index)

    def _update_opentelemetry_metrics(self, operation: str, duration: float, success: bool) -> None:
        """Update OpenTelemetry metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Record operation
        if cache_operations_counter:
            cache_operations_counter.add(
                1,
                {
                    "operation": operation,
                    "status": "success" if success else "error",
                    "cache_type": "multi_level",
                }
            )

        # Record latency
        if cache_latency_histogram:
            cache_latency_histogram.record(
                duration,
                {
                    "operation": operation,
                    "cache_type": "multi_level",
                }
            )

        # Record errors
        if not success and cache_error_counter:
            cache_error_counter.add(
                1,
                {
                    "operation": operation,
                    "cache_type": "multi_level",
                }
            )

    def update_cache_metrics(self) -> None:
        """Update cache-specific OpenTelemetry metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        stats = self._get_combined_performance_stats()

        # Hit ratios
        if cache_hit_ratio_gauge:
            cache_hit_ratio_gauge.set(
                stats["overall_hit_rate"],
                {"cache_type": "multi_level", "level": "overall"}
            )
            cache_hit_ratio_gauge.set(
                stats["l1_hit_rate"],
                {"cache_type": "multi_level", "level": "l1"}
            )
            cache_hit_ratio_gauge.set(
                stats["l2_hit_rate"],
                {"cache_type": "multi_level", "level": "l2"}
            )

        # Cache sizes
        if cache_size_gauge:
            l1_stats = stats["l1_cache_stats"]
            cache_size_gauge.set(
                l1_stats["size"],
                {"cache_type": "multi_level", "level": "l1"}
            )

        # Memory usage
        if cache_memory_usage_gauge:
            l1_stats = stats["l1_cache_stats"]
            cache_memory_usage_gauge.set(
                l1_stats["estimated_memory_bytes"],
                {"cache_type": "multi_level", "level": "l1"}
            )

    def get_health_summary(self) -> dict[str, Any]:
        """Get concise health summary for dashboards.

        Returns:
            Concise health summary
        """
        stats = self._get_combined_performance_stats()
        slo_compliance = self.calculate_slo_compliance()

        return {
            "overall_healthy": self._health_status["overall_health"] == "healthy",
            "overall_status": self._health_status["overall_health"],
            "hit_rate": stats["overall_hit_rate"],
            "avg_response_time_ms": stats["avg_response_time_ms"],
            "slo_compliant": slo_compliance["compliant"],
            "total_requests": stats["total_requests"],
            "error_rate": self._calculate_error_rate(),
            "uptime_hours": (datetime.now(UTC) - self._created_at).total_seconds() / 3600,
            "last_check": (
                self._health_status["last_health_check"].isoformat()
                if isinstance(self._health_status["last_health_check"], datetime)
                else str(self._health_status["last_health_check"])
            ),
        }

    def _get_combined_performance_stats(self) -> dict[str, Any]:
        """Get combined performance statistics from L1 and L2 cache services directly.

        Returns:
            Combined performance statistics matching coordinator format
        """
        # Get direct stats from cache services
        l1_stats = self._l1_cache.get_stats()
        l2_stats = self._l2_cache.get_stats() if self._l2_cache else None

        # Calculate combined metrics
        total_hits = self._l1_hits + self._l2_hits
        total_requests = self._total_requests if self._total_requests > 0 else max(l1_stats.get("requests", 0), 1)
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        avg_response_time_ms = (self._total_response_time / total_requests * 1000) if total_requests > 0 else 0.0

        # Overall health status
        l1_healthy = l1_stats["health_status"] == "healthy"
        l2_healthy = l2_stats["health_status"] == "healthy" if l2_stats else True

        if not l1_healthy or not l2_healthy:
            health_status = "unhealthy"
        elif l1_stats["health_status"] == "degraded" or (l2_stats and l2_stats["health_status"] == "degraded"):
            health_status = "degraded"
        else:
            health_status = "healthy"

        return {
            "total_requests": total_requests,
            "overall_hit_rate": overall_hit_rate,
            "avg_response_time_ms": avg_response_time_ms,
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "l1_hit_rate": self._l1_hits / total_requests if total_requests > 0 else 0.0,
            "l2_hit_rate": self._l2_hits / total_requests if total_requests > 0 else 0.0,
            "l1_cache_stats": l1_stats,
            "l2_cache_stats": l2_stats,
            "health_status": health_status,
            "uptime_seconds": (datetime.now(UTC) - self._created_at).total_seconds(),
            "architecture": "direct_cache_aside_pattern",
            "coordination_overhead": "eliminated",
        }
