"""Unified health management system for database services.

This module provides comprehensive health monitoring functionality extracted from
unified_connection_manager.py, implementing:

- HealthManager: Central coordinator for all component health checks
- HealthManagerConfig: Configurable monitoring intervals and thresholds
- Background monitoring with automatic recovery detection
- Circuit breaker integration for fault tolerance
- Aggregated health reporting with component-level details

Designed for production monitoring with <10ms health check response times.
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from prompt_improver.database.services.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from prompt_improver.database.services.health.health_checker import (
    AggregatedHealthResult,
    CacheHealthChecker,
    DatabaseHealthChecker,
    HealthChecker,
    HealthResult,
    HealthStatus,
    RedisHealthChecker,
)

logger = logging.getLogger(__name__)


@dataclass
class HealthManagerConfig:
    """Configuration for health management system."""

    # Health check intervals
    check_interval_seconds: float = 30.0
    fast_check_interval_seconds: float = 5.0  # For degraded components

    # Background monitoring
    enable_background_monitoring: bool = True
    max_concurrent_checks: int = 10

    # Circuit breaker settings
    enable_circuit_breakers: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0

    # Health result caching
    cache_results_seconds: float = 10.0
    stale_result_threshold_seconds: float = 60.0

    # Failure tracking
    consecutive_failure_threshold: int = 3
    degraded_threshold_ratio: float = 0.5  # 50% components failing = degraded

    # Performance thresholds
    response_time_warning_ms: float = 100.0
    response_time_error_ms: float = 1000.0

    def __post_init__(self):
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be greater than 0")
        if self.max_concurrent_checks <= 0:
            raise ValueError("max_concurrent_checks must be greater than 0")


class HealthManager:
    """Unified health management system for database services.

    Central coordinator that manages health checks for multiple components
    including databases, Redis, caches, and other services with:
    - Parallel health check execution
    - Circuit breaker integration
    - Background monitoring loops
    - Aggregated health reporting
    - Automatic failure recovery detection
    """

    def __init__(
        self,
        config: HealthManagerConfig | None = None,
        service_name: str = "health_manager",
    ) -> None:
        self.config = config or HealthManagerConfig()
        self.service_name = service_name

        # Component tracking
        self._health_checkers: dict[str, HealthChecker] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # State management
        self._last_check_results: dict[str, HealthResult] = {}
        self._last_aggregated_result: AggregatedHealthResult | None = None
        self._last_full_check_time: datetime | None = None

        # Background monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False
        self._shutdown_event = asyncio.Event()

        # Failure tracking
        self._consecutive_failures: dict[str, int] = {}
        self._component_recovery_times: dict[str, datetime] = {}

        # Performance metrics
        self.total_checks = 0
        self.successful_checks = 0
        self.failed_checks = 0
        self.avg_response_time_ms = 0.0

        logger.info(
            f"HealthManager initialized: {self.service_name}, "
            f"check_interval={self.config.check_interval_seconds}s"
        )

    def register_health_checker(self, name: str, checker: HealthChecker) -> None:
        """Register a health checker for a component."""
        if name in self._health_checkers:
            logger.warning(f"Health checker {name} already registered, replacing")

        self._health_checkers[name] = checker
        self._consecutive_failures[name] = 0

        # Create circuit breaker if enabled
        if self.config.enable_circuit_breakers:
            cb_config = CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout_seconds=self.config.circuit_breaker_recovery_timeout,
            )
            self._circuit_breakers[name] = CircuitBreaker(
                f"{self.service_name}_{name}", cb_config
            )

        logger.info(f"Registered health checker: {name}")

    def unregister_health_checker(self, name: str) -> bool:
        """Unregister a health checker."""
        if name in self._health_checkers:
            del self._health_checkers[name]
            self._last_check_results.pop(name, None)
            self._consecutive_failures.pop(name, None)
            self._component_recovery_times.pop(name, None)

            if name in self._circuit_breakers:
                del self._circuit_breakers[name]

            logger.info(f"Unregistered health checker: {name}")
            return True
        return False

    def register_database_checker(
        self, name: str, connection_pool, timeout_seconds: float = 5.0
    ) -> None:
        """Convenience method to register database health checker."""
        checker = DatabaseHealthChecker(name, connection_pool, timeout_seconds)
        self.register_health_checker(name, checker)

    def register_redis_checker(
        self, name: str, redis_client, timeout_seconds: float = 2.0
    ) -> None:
        """Convenience method to register Redis health checker."""
        checker = RedisHealthChecker(name, redis_client, timeout_seconds)
        self.register_health_checker(name, checker)

    def register_cache_checker(
        self, name: str, cache_service, timeout_seconds: float = 2.0
    ) -> None:
        """Convenience method to register cache service health checker."""
        checker = CacheHealthChecker(name, cache_service, timeout_seconds)
        self.register_health_checker(name, checker)

    async def check_component_health(
        self, component_name: str, use_circuit_breaker: bool = True
    ) -> HealthResult | None:
        """Check health of a specific component."""
        if component_name not in self._health_checkers:
            logger.error(
                f"No health checker registered for component: {component_name}"
            )
            return None

        checker = self._health_checkers[component_name]
        circuit_breaker = self._circuit_breakers.get(component_name)

        # Check circuit breaker if enabled
        if (
            use_circuit_breaker
            and circuit_breaker
            and not circuit_breaker.is_call_permitted()
        ):
            return HealthResult(
                component=component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                message="Circuit breaker open",
                error="Component circuit breaker is open due to repeated failures",
            )

        try:
            # Perform health check
            result = await checker.check_health_with_timing()

            # Update circuit breaker
            if use_circuit_breaker and circuit_breaker:
                if result.is_healthy():
                    circuit_breaker.record_success(result.response_time_ms)
                    # Reset consecutive failures on success
                    if self._consecutive_failures[component_name] > 0:
                        self._consecutive_failures[component_name] = 0
                        self._component_recovery_times[component_name] = datetime.now(
                            UTC
                        )
                        logger.info(
                            f"Component {component_name} recovered after failures"
                        )
                else:
                    circuit_breaker.record_failure(
                        response_time_ms=result.response_time_ms
                    )
                    self._consecutive_failures[component_name] += 1

            # Cache result
            self._last_check_results[component_name] = result

            # Update global metrics
            self.total_checks += 1
            if result.is_healthy():
                self.successful_checks += 1
            else:
                self.failed_checks += 1

            # Update average response time
            if result.response_time_ms > 0:
                total_response_time = self.avg_response_time_ms * (
                    self.total_checks - 1
                )
                self.avg_response_time_ms = (
                    total_response_time + result.response_time_ms
                ) / self.total_checks

            return result

        except Exception as e:
            logger.exception(f"Health check failed for {component_name}: {e}")

            error_result = HealthResult(
                component=component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                message=f"Health check exception: {type(e).__name__}",
                error=str(e),
            )

            # Update circuit breaker on exception
            if use_circuit_breaker and circuit_breaker:
                circuit_breaker.record_failure(e)

            self._consecutive_failures[component_name] += 1
            self._last_check_results[component_name] = error_result

            self.total_checks += 1
            self.failed_checks += 1

            return error_result

    async def check_all_components_health(
        self, parallel: bool = True
    ) -> AggregatedHealthResult:
        """Check health of all registered components."""
        if not self._health_checkers:
            return AggregatedHealthResult(
                overall_status=HealthStatus.UNKNOWN,
                components={},
                response_time_ms=0,
                timestamp=datetime.now(UTC),
            )

        start_time = time.time()
        component_results: dict[str, HealthResult] = {}

        if parallel and len(self._health_checkers) > 1:
            # Execute health checks in parallel with concurrency limit
            semaphore = asyncio.Semaphore(self.config.max_concurrent_checks)

            async def check_with_semaphore(name: str) -> tuple[str, HealthResult]:
                async with semaphore:
                    result = await self.check_component_health(name)
                    return name, result

            # Create tasks for parallel execution
            tasks = [
                check_with_semaphore(name) for name in self._health_checkers
            ]

            # Wait for all checks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    name, health_result = result
                    if health_result:
                        component_results[name] = health_result
                elif isinstance(result, Exception):
                    logger.error(f"Health check task failed: {result}")

        else:
            # Sequential execution
            for name in self._health_checkers:
                result = await self.check_component_health(name)
                if result:
                    component_results[name] = result

        # Calculate overall status
        response_time_ms = (time.time() - start_time) * 1000
        overall_status = self._calculate_overall_status(component_results)

        # Create aggregated result
        aggregated_result = AggregatedHealthResult(
            overall_status=overall_status,
            components=component_results,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(UTC),
        )

        # Cache the result
        self._last_aggregated_result = aggregated_result
        self._last_full_check_time = datetime.now(UTC)

        logger.debug(
            f"Health check completed: {overall_status.value}, "
            f"{len(component_results)} components, "
            f"{response_time_ms:.1f}ms"
        )

        return aggregated_result

    def _calculate_overall_status(
        self, component_results: dict[str, HealthResult]
    ) -> HealthStatus:
        """Calculate overall health status from component results."""
        if not component_results:
            return HealthStatus.UNKNOWN

        healthy_count = sum(
            1
            for result in component_results.values()
            if result.status == HealthStatus.HEALTHY
        )
        degraded_count = sum(
            1
            for result in component_results.values()
            if result.status == HealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1
            for result in component_results.values()
            if result.status == HealthStatus.UNHEALTHY
        )

        total_components = len(component_results)

        # All components healthy
        if healthy_count == total_components:
            return HealthStatus.HEALTHY

        # More than threshold unhealthy = overall unhealthy
        unhealthy_ratio = unhealthy_count / total_components
        if unhealthy_ratio > self.config.degraded_threshold_ratio:
            return HealthStatus.UNHEALTHY

        # Some components degraded or unhealthy = overall degraded
        if degraded_count > 0 or unhealthy_count > 0:
            return HealthStatus.DEGRADED

        return HealthStatus.UNKNOWN

    def get_cached_result(self, component_name: str) -> HealthResult | None:
        """Get cached health result for a component."""
        result = self._last_check_results.get(component_name)

        if result and self._is_result_fresh(result):
            return result

        return None

    def get_last_aggregated_result(self) -> AggregatedHealthResult | None:
        """Get the last aggregated health check result."""
        if self._last_aggregated_result and self._last_full_check_time:
            age_seconds = (
                datetime.now(UTC) - self._last_full_check_time
            ).total_seconds()
            if age_seconds <= self.config.cache_results_seconds:
                return self._last_aggregated_result

        return None

    def _is_result_fresh(self, result: HealthResult) -> bool:
        """Check if a health result is still fresh."""
        if not result.timestamp:
            return False

        age_seconds = (datetime.now(UTC) - result.timestamp).total_seconds()
        return age_seconds <= self.config.cache_results_seconds

    async def start_background_monitoring(self) -> None:
        """Start background health monitoring loop."""
        if self._is_monitoring:
            logger.warning("Background monitoring already running")
            return

        if not self.config.enable_background_monitoring:
            logger.info("Background monitoring disabled")
            return

        self._is_monitoring = True
        self._shutdown_event.clear()
        self._monitoring_task = asyncio.create_task(self._background_monitoring_loop())

        logger.info("Background health monitoring started")

    async def stop_background_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False
        self._shutdown_event.set()

        if self._monitoring_task:
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=10.0)
            except TimeoutError:
                logger.warning("Background monitoring task did not stop gracefully")
                self._monitoring_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._monitoring_task

        logger.info("Background health monitoring stopped")

    async def _background_monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_monitoring:
            try:
                # Determine check interval based on current health status
                interval = self.config.check_interval_seconds
                if (
                    self._last_aggregated_result
                    and self._last_aggregated_result.overall_status
                    == HealthStatus.DEGRADED
                ):
                    interval = self.config.fast_check_interval_seconds

                # Wait for next check or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=interval
                    )
                    break  # Shutdown requested
                except TimeoutError:
                    pass  # Normal timeout, continue with health check

                # Perform health check
                await self.check_all_components_health()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Background monitoring error: {e}")
                await asyncio.sleep(min(interval * 2, 60))  # Back off on error

        logger.debug("Background monitoring loop exited")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive health manager statistics."""
        circuit_breaker_stats = {}
        if self.config.enable_circuit_breakers:
            circuit_breaker_stats = {
                name: cb.get_stats() for name, cb in self._circuit_breakers.items()
            }

        component_status = {}
        for name, result in self._last_check_results.items():
            component_status[name] = {
                "status": result.status.value,
                "last_check": result.timestamp.isoformat()
                if result.timestamp
                else None,
                "consecutive_failures": self._consecutive_failures.get(name, 0),
                "recovery_time": (
                    self._component_recovery_times[name].isoformat()
                    if name in self._component_recovery_times
                    else None
                ),
            }

        return {
            "service": self.service_name,
            "monitoring": {
                "background_enabled": self.config.enable_background_monitoring,
                "is_running": self._is_monitoring,
                "check_interval_seconds": self.config.check_interval_seconds,
                "registered_components": len(self._health_checkers),
            },
            "performance": {
                "total_checks": self.total_checks,
                "successful_checks": self.successful_checks,
                "failed_checks": self.failed_checks,
                "success_rate": self.successful_checks / self.total_checks
                if self.total_checks > 0
                else 0,
                "avg_response_time_ms": self.avg_response_time_ms,
            },
            "components": component_status,
            "circuit_breakers": circuit_breaker_stats,
            "last_aggregated_check": (
                self._last_aggregated_result.to_dict()
                if self._last_aggregated_result
                else None
            ),
        }

    async def shutdown(self) -> None:
        """Shutdown health manager and cleanup resources."""
        logger.info(f"Shutting down HealthManager: {self.service_name}")

        await self.stop_background_monitoring()

        # Clear all state
        self._health_checkers.clear()
        self._circuit_breakers.clear()
        self._last_check_results.clear()
        self._consecutive_failures.clear()
        self._component_recovery_times.clear()

        logger.info("HealthManager shutdown complete")

    def __repr__(self) -> str:
        status = "unknown"
        if self._last_aggregated_result:
            status = self._last_aggregated_result.overall_status.value

        return (
            f"HealthManager(service={self.service_name}, "
            f"components={len(self._health_checkers)}, "
            f"status={status}, monitoring={self._is_monitoring})"
        )


# Convenience function for creating health managers
def create_health_manager(
    service_name: str = "health_manager",
    check_interval_seconds: float = 30.0,
    enable_background_monitoring: bool = True,
    enable_circuit_breakers: bool = True,
    **kwargs,
) -> HealthManager:
    """Create health manager with simple configuration."""
    config = HealthManagerConfig(
        check_interval_seconds=check_interval_seconds,
        enable_background_monitoring=enable_background_monitoring,
        enable_circuit_breakers=enable_circuit_breakers,
        **kwargs,
    )
    return HealthManager(config, service_name)
