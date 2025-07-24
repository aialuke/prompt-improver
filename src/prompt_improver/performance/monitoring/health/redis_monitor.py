"""
Redis health monitoring task for periodic liveness and performance checks.

Enhanced with 2025 best practices:
- Connection pool monitoring and health tracking
- Adaptive thresholds based on performance trends
- Circuit breaker patterns for resilience
- SLI/SLO integration for observability
- Multi-dimensional metrics and percentile tracking
- Graceful degradation and fallback mechanisms
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Union, Dict, List, Optional
from enum import Enum
import statistics

try:
    from ...utils.redis_cache import CACHE_ERRORS, CACHE_LATENCY_MS, redis_client
except ImportError:
    # Fallback if relative imports fail
    from src.prompt_improver.utils.redis_cache import (
        CACHE_ERRORS,
        CACHE_LATENCY_MS,
        redis_client,
    )

import logging

from prometheus_client import Counter, Histogram, Gauge

from .background_manager import get_background_task_manager
from .base import HealthChecker, HealthResult, HealthStatus

class CircuitBreakerState(Enum):
    """Circuit breaker states for Redis health monitoring."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class RedisHealthMetrics:
    """Enhanced metrics collection for Redis health monitoring."""

    def __init__(self):
        self.latency_samples: List[float] = []
        self.error_count = 0
        self.success_count = 0
        self.connection_pool_size = 0
        self.active_connections = 0
        self.last_update = datetime.now(timezone.utc)

    def add_latency_sample(self, latency_ms: float):
        """Add a latency sample and maintain rolling window."""
        self.latency_samples.append(latency_ms)
        # Keep only last 100 samples for percentile calculations
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not self.latency_samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_samples = sorted(self.latency_samples)
        n = len(sorted_samples)

        return {
            "p50": sorted_samples[int(n * 0.5)] if n > 0 else 0.0,
            "p95": sorted_samples[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_samples[int(n * 0.99)] if n > 0 else 0.0
        }

    def get_error_rate(self) -> float:
        """Calculate current error rate."""
        total = self.error_count + self.success_count
        return (self.error_count / total) if total > 0 else 0.0

    def reset(self):
        """Reset metrics for new measurement period."""
        self.error_count = 0
        self.success_count = 0
        self.last_update = datetime.now(timezone.utc)

# Configure logging
logger = logging.getLogger("redis_health_monitor")

# Use centralized metrics registry
from ..metrics_registry import get_metrics_registry

metrics_registry = get_metrics_registry()
REDIS_CHECK_FAILURES = metrics_registry.get_or_create_counter(
    'redis_check_failures_total',
    'Total number of Redis check failures'
)

class RedisHealthMonitor(HealthChecker):
    """
    Enhanced Redis health monitoring with 2025 best practices.

    features:
    - Connection pool monitoring and health tracking
    - Adaptive thresholds based on performance trends
    - Circuit breaker patterns for resilience
    - Multi-dimensional metrics and percentile tracking
    - SLI/SLO integration for observability
    """

    def __init__(self, config):
        super().__init__("redis")
        self.config = config
        self.failure_count = 0
        self.last_check_time = None
        self.is_connected = True

        # Enhanced 2025 features
        self.metrics = RedisHealthMetrics()
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_failure_count = 0
        self.circuit_breaker_last_failure_time = None
        self.circuit_breaker_timeout = config.get('circuit_breaker_timeout', 60)  # seconds
        self.circuit_breaker_failure_threshold = config.get('circuit_breaker_failure_threshold', 5)

        # Adaptive thresholds (2025 best practice)
        self.adaptive_thresholds = {
            'latency_warning': config.get('latency_threshold', 100),  # ms
            'latency_critical': config.get('latency_threshold', 100) * 2,  # ms
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.10,  # 10%
        }

        # SLI/SLO tracking
        self.sli_targets = {
            'availability': 99.9,  # 99.9% availability
            'latency_p95': 200.0,  # 200ms P95 latency
            'error_rate': 1.0,     # <1% error rate
        }

        # Enhanced metrics
        self.latency_histogram = None
        self.connection_pool_gauge = None
        self._setup_enhanced_metrics()

    def _setup_enhanced_metrics(self):
        """Setup enhanced Prometheus metrics for 2025 observability."""
        try:
            # Latency histogram with percentile buckets
            self.latency_histogram = metrics_registry.get_or_create_histogram(
                'redis_operation_duration_seconds',
                'Redis operation latency in seconds',
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            )

            # Connection pool metrics
            self.connection_pool_gauge = metrics_registry.get_or_create_gauge(
                'redis_connection_pool_size',
                'Redis connection pool size and utilization',
                ['metric']
            )
        except Exception as e:
            logger.warning(f"Failed to setup enhanced metrics: {e}")

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (blocking requests)."""
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if timeout period has passed
            if (self.circuit_breaker_last_failure_time and
                time.time() - self.circuit_breaker_last_failure_time > self.circuit_breaker_timeout):
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return False
            return True
        return False

    def _record_circuit_breaker_success(self):
        """Record successful operation for circuit breaker."""
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.circuit_breaker_failure_count = 0
            logger.info("Circuit breaker moved to CLOSED state")
        elif self.circuit_breaker_state == CircuitBreakerState.CLOSED:
            self.circuit_breaker_failure_count = max(0, self.circuit_breaker_failure_count - 1)

    def _record_circuit_breaker_failure(self):
        """Record failed operation for circuit breaker."""
        self.circuit_breaker_failure_count += 1
        self.circuit_breaker_last_failure_time = time.time()

        if (self.circuit_breaker_failure_count >= self.circuit_breaker_failure_threshold and
            self.circuit_breaker_state != CircuitBreakerState.OPEN):
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker OPENED after {self.circuit_breaker_failure_count} failures")

    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent performance (2025 best practice)."""
        try:
            percentiles = self.metrics.get_percentiles()
            error_rate = self.metrics.get_error_rate()

            # Adjustment factor for threshold changes
            adjustment_factor = 0.1  # 10% adjustment

            # Adaptive latency thresholds (adjust based on P95)
            if percentiles["p95"] > 0:
                current_p95 = percentiles["p95"]

                if current_p95 > self.adaptive_thresholds['latency_critical']:
                    # Performance degraded, relax thresholds slightly
                    self.adaptive_thresholds['latency_warning'] *= (1 + adjustment_factor)
                    self.adaptive_thresholds['latency_critical'] *= (1 + adjustment_factor)
                elif current_p95 < self.adaptive_thresholds['latency_warning'] * 0.7:
                    # Performance improved, tighten thresholds slightly
                    self.adaptive_thresholds['latency_warning'] *= (1 - adjustment_factor)
                    self.adaptive_thresholds['latency_critical'] *= (1 - adjustment_factor)

            # Adaptive error rate thresholds
            if error_rate > self.adaptive_thresholds['error_rate_critical']:
                # High error rate, relax thresholds temporarily
                self.adaptive_thresholds['error_rate_warning'] *= (1 + adjustment_factor)
                self.adaptive_thresholds['error_rate_critical'] *= (1 + adjustment_factor)
            elif error_rate < self.adaptive_thresholds['error_rate_warning'] * 0.5:
                # Low error rate, tighten thresholds
                self.adaptive_thresholds['error_rate_warning'] *= (1 - adjustment_factor)
                self.adaptive_thresholds['error_rate_critical'] *= (1 - adjustment_factor)

        except Exception as e:
            logger.warning(f"Failed to update adaptive thresholds: {e}")

    async def ping_check(self) -> str | None:
        """Perform a Redis PING command to check if the server is alive."""
        # Circuit breaker check
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker is OPEN, skipping ping check")
            return None

        start_time = time.time()
        try:
            response = await redis_client.ping()
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            if response:
                # Record success metrics
                self.metrics.success_count += 1
                self.metrics.add_latency_sample(latency_ms)
                self._record_circuit_breaker_success()

                # Record enhanced metrics
                if self.latency_histogram:
                    self.latency_histogram.labels(operation='ping').observe(end_time - start_time)

                return f"PING successful ({latency_ms:.1f}ms)"
        except Exception as e:
            logger.error(f"PING failed: {e}")
            self.metrics.error_count += 1
            self._record_circuit_breaker_failure()
            CACHE_ERRORS.labels(operation='ping').inc()
        return None

    async def get_check(self) -> str | None:
        """Perform a Redis GET command to check if data retrieval is working."""
        # Circuit breaker check
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker is OPEN, skipping get check")
            return None

        test_key = "health:check"
        start_time = time.time()
        try:
            await redis_client.set(test_key, "ok", ex=60)
            set_time = time.time()
            response = await redis_client.get(test_key)
            end_time = time.time()

            if response == b"ok":
                latency_ms = (end_time - start_time) * 1000
                get_latency_ms = (end_time - set_time) * 1000

                # Record success metrics
                self.metrics.success_count += 1
                self.metrics.add_latency_sample(get_latency_ms)
                self._record_circuit_breaker_success()

                # Record enhanced metrics
                CACHE_LATENCY_MS.labels(operation='get').observe(get_latency_ms)
                if self.latency_histogram:
                    self.latency_histogram.labels(operation='get').observe(end_time - set_time)

                return f"GET successful ({get_latency_ms:.1f}ms)"
        except Exception as e:
            logger.error(f"GET failed: {e}")
            self.metrics.error_count += 1
            self._record_circuit_breaker_failure()
            CACHE_ERRORS.labels(operation='get').inc()
        return None

    async def check(self) -> HealthResult:
        """Perform enhanced health check with 2025 best practices."""
        try:
            # Update connection pool metrics
            await self._update_connection_pool_metrics()

            # Perform health checks
            ping_result = await self.ping_check()
            get_result = await self.get_check()

            # Update adaptive thresholds based on recent performance
            self._update_adaptive_thresholds()

            # Calculate SLI metrics
            sli_metrics = self._calculate_sli_metrics()

            # Determine health status based on enhanced criteria
            if not ping_result or not get_result:
                self.failure_count += 1
                REDIS_CHECK_FAILURES.inc()
                logger.warning(f"Redis check failed: {self.failure_count} failures")

                if self.failure_count >= self.config['failure_threshold']:
                    await self.trigger_reconnection_logic()

                # Enhanced failure analysis
                status = self._determine_health_status(sli_metrics, has_connectivity=False)
                return HealthResult(
                    status=status,
                    component=self.name,
                    message=f"{self.failure_count} consecutive failures. Circuit breaker: {self.circuit_breaker_state.value}"
                )

            # Success case
            self.failure_count = 0
            status = self._determine_health_status(sli_metrics, has_connectivity=True)

            return HealthResult(
                status=status,
                component=self.name,
                message=f"Redis is {status.value}. P95 latency: {sli_metrics.get('latency_p95', 0):.1f}ms, Error rate: {sli_metrics.get('error_rate', 0):.2%}"
            )

        except Exception as e:
            logger.error(f"Unhandled exception during Redis check: {e}")
            self.metrics.error_count += 1
            self._record_circuit_breaker_failure()

            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message=f"Redis check failed due to exception. Circuit breaker: {self.circuit_breaker_state.value}"
            )

    async def trigger_reconnection_logic(self) -> None:
        """Trigger reconnection logic after threshold failures."""
        max_retries = self.config.get('reconnection', {}).get('max_retries', 3)
        backoff_factor = self.config.get('reconnection', {}).get('backoff_factor', 2)

        retry_count = 0
        backoff = 1

        while retry_count < max_retries:
            try:
                await asyncio.sleep(backoff)
                # Attempt reconnection
                pool = redis_client.connection_pool
                pool.disconnect()
                await asyncio.sleep(1)
                self.failure_count = 0  # Reset failure count
                logger.info("Reconnected to Redis")
                break  # Exit loop on successful reconnection
            except Exception as e:
                logger.warning(f"Reconnection attempt {retry_count} failed: {e}")
                retry_count += 1
                backoff *= backoff_factor

        if retry_count == max_retries:
            logger.error("Redis reconnection failed after maximum retries")

    async def _update_connection_pool_metrics(self):
        """Update connection pool metrics for monitoring."""
        try:
            # Get connection pool information from Redis client
            if hasattr(redis_client, 'connection_pool'):
                pool = redis_client.connection_pool
                self.metrics.connection_pool_size = getattr(pool, 'max_connections', 0)
                self.metrics.active_connections = len(getattr(pool, '_available_connections', []))

                # Update Prometheus gauge
                if self.connection_pool_gauge:
                    self.connection_pool_gauge.labels(metric='total').set(self.metrics.connection_pool_size)
                    self.connection_pool_gauge.labels(metric='active').set(self.metrics.active_connections)
        except Exception as e:
            logger.warning(f"Failed to update connection pool metrics: {e}")

    def _calculate_sli_metrics(self) -> Dict[str, float]:
        """Calculate Service Level Indicators (SLI) for Redis."""
        try:
            percentiles = self.metrics.get_percentiles()
            error_rate = self.metrics.get_error_rate()
            total_requests = self.metrics.success_count + self.metrics.error_count

            # Calculate availability (successful requests / total requests)
            availability = (self.metrics.success_count / total_requests * 100) if total_requests > 0 else 100.0

            return {
                "availability": availability,
                "latency_p50": percentiles["p50"],
                "latency_p95": percentiles["p95"],
                "latency_p99": percentiles["p99"],
                "error_rate": error_rate,
                "total_requests": total_requests,
                "circuit_breaker_state": self.circuit_breaker_state.value
            }
        except Exception as e:
            logger.warning(f"Failed to calculate SLI metrics: {e}")
            return {}

    def _determine_health_status(self, sli_metrics: Dict[str, float], has_connectivity: bool) -> HealthStatus:
        """Determine health status based on SLI metrics and adaptive thresholds."""
        if not has_connectivity:
            return HealthStatus.FAILED

        try:
            # Check circuit breaker state
            if self.circuit_breaker_state == CircuitBreakerState.OPEN:
                return HealthStatus.FAILED
            elif self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                return HealthStatus.WARNING

            # Check SLI against targets
            availability = sli_metrics.get("availability", 100.0)
            latency_p95 = sli_metrics.get("latency_p95", 0.0)
            error_rate = sli_metrics.get("error_rate", 0.0)

            # Critical conditions
            if (availability < 95.0 or  # Less than 95% availability
                latency_p95 > self.adaptive_thresholds['latency_critical'] or
                error_rate > self.adaptive_thresholds['error_rate_critical']):
                return HealthStatus.FAILED

            # Warning conditions
            if (availability < self.sli_targets['availability'] or
                latency_p95 > self.adaptive_thresholds['latency_warning'] or
                error_rate > self.adaptive_thresholds['error_rate_warning']):
                return HealthStatus.WARNING

            # All good
            return HealthStatus.HEALTHY

        except Exception as e:
            logger.warning(f"Failed to determine health status: {e}")
            return HealthStatus.WARNING

    def get_enhanced_status(self) -> Dict[str, any]:
        """Get enhanced status information for monitoring dashboards."""
        try:
            sli_metrics = self._calculate_sli_metrics()
            percentiles = self.metrics.get_percentiles()

            return {
                "component": self.name,
                "circuit_breaker_state": self.circuit_breaker_state.value,
                "failure_count": self.failure_count,
                "sli_metrics": sli_metrics,
                "adaptive_thresholds": self.adaptive_thresholds,
                "sli_targets": self.sli_targets,
                "connection_pool": {
                    "size": self.metrics.connection_pool_size,
                    "active": self.metrics.active_connections,
                    "utilization": (self.metrics.active_connections / self.metrics.connection_pool_size * 100)
                                 if self.metrics.connection_pool_size > 0 else 0
                },
                "performance": {
                    "latency_percentiles": percentiles,
                    "error_rate": self.metrics.get_error_rate(),
                    "total_requests": self.metrics.success_count + self.metrics.error_count,
                    "success_rate": (self.metrics.success_count / (self.metrics.success_count + self.metrics.error_count) * 100)
                                  if (self.metrics.success_count + self.metrics.error_count) > 0 else 100.0
                },
                "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
                "is_connected": self.is_connected
            }
        except Exception as e:
            logger.error(f"Failed to get enhanced status: {e}")
            return {"component": self.name, "error": str(e)}

async def schedule_redis_health_checks(config):
    """Schedule and manage periodic Redis health checks."""
    task_manager = get_background_task_manager()
    health_monitor = RedisHealthMonitor(config)

    task_id = "redis-health-check"
    check_interval = config.get('check_interval', 60)

    # Run periodic health checks
    while True:
        try:
            await task_manager.submit_task(f"{task_id}-{int(time.time())}", health_monitor.check)
            await asyncio.sleep(check_interval)
        except Exception as e:
            logger.error(f"Error scheduling Redis health check: {e}")
            await asyncio.sleep(check_interval)

# Initialize the Redis Health Monitor
default_monitor_config = {
    'check_interval': 60,  # Time between checks in seconds
    'failure_threshold': 3,
    'latency_threshold': 100,
    'reconnection': {'max_retries': 5, 'backoff_factor': 2}
}

async def start_redis_health_monitor(config=None):
    """Start the Redis health monitoring task based on the configuration."""
    if config is None:
        config = default_monitor_config
    await schedule_redis_health_checks(config)
