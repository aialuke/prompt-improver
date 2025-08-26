"""Internal services for the unified monitoring facade.

Provides the internal service layer for health checking coordination
and metrics collection, following clean architecture principles.
"""

import asyncio
import logging
import time

from prompt_improver.monitoring.unified.health_checkers import (
    DatabaseHealthChecker,
    MLModelsHealthChecker,
    RedisHealthChecker,
    SystemResourcesHealthChecker,
)
from prompt_improver.monitoring.unified.types import (
    HealthCheckResult,
    HealthStatus,
    MetricPoint,
    MetricType,
    MonitoringConfig,
    SystemHealthSummary,
)
from prompt_improver.shared.interfaces.protocols.monitoring import (
    HealthCheckComponentProtocol,
    MonitoringRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class HealthCheckService:
    """Internal service for coordinating health checks."""

    def __init__(
        self,
        config: MonitoringConfig,
        repository: MonitoringRepositoryProtocol | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self._components: dict[str, HealthCheckComponentProtocol] = {}
        self._last_results: dict[str, HealthCheckResult] = {}
        self._last_check_time: float | None = None

        # Register default health checkers
        self._register_default_components()

    def _register_default_components(self) -> None:
        """Register default health check components."""
        default_checkers = [
            DatabaseHealthChecker(timeout_seconds=self.config.health_check_timeout_seconds),
            RedisHealthChecker(timeout_seconds=self.config.health_check_timeout_seconds),
            MLModelsHealthChecker(timeout_seconds=self.config.health_check_timeout_seconds),
            SystemResourcesHealthChecker(timeout_seconds=self.config.health_check_timeout_seconds),
        ]

        for checker in default_checkers:
            self._components[checker.get_component_name()] = checker

    async def run_all_checks(self) -> SystemHealthSummary:
        """Run all registered health checks."""
        start_time = time.time()

        try:
            if self.config.health_check_parallel_enabled:
                results = await self._run_checks_parallel()
            else:
                results = await self._run_checks_sequential()

            # Store results in cache
            self._last_results = {r.component_name: r for r in results}
            self._last_check_time = time.time()

            # Store in repository if available
            if self.repository:
                try:
                    for result in results:
                        await self.repository.store_health_result(result)
                except Exception as e:
                    logger.warning(f"Failed to store health results: {e}")

            # Calculate summary
            return self._calculate_health_summary(results, start_time)

        except Exception as e:
            logger.exception(f"Health check execution failed: {e}")

            # Return emergency summary
            return SystemHealthSummary(
                overall_status=HealthStatus.UNKNOWN,
                total_components=len(self._components),
                healthy_components=0,
                degraded_components=0,
                unhealthy_components=0,
                unknown_components=len(self._components),
                check_duration_ms=(time.time() - start_time) * 1000,
            )

    async def _run_checks_parallel(self) -> list[HealthCheckResult]:
        """Run health checks in parallel."""
        if not self._components:
            return []

        # Create semaphore to limit concurrent checks
        semaphore = asyncio.Semaphore(self.config.max_concurrent_checks)

        async def run_single_check(name: str, checker: HealthCheckComponentProtocol) -> HealthCheckResult:
            async with semaphore:
                try:
                    return await checker.check_health()
                except Exception as e:
                    logger.exception(f"Health check failed for {name}: {e}")
                    return HealthCheckResult(
                        status=HealthStatus.UNKNOWN,
                        component_name=name,
                        message=f"Health check error: {e!s}",
                        error=str(e),
                    )

        tasks = [
            run_single_check(name, checker)
            for name, checker in self._components.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component_name = list(self._components.keys())[i]
                logger.error(f"Health check exception for {component_name}: {result}")
                valid_results.append(
                    HealthCheckResult(
                        status=HealthStatus.UNKNOWN,
                        component_name=component_name,
                        message=f"Health check exception: {result!s}",
                        error=str(result),
                    )
                )
            else:
                valid_results.append(result)

        return valid_results

    async def _run_checks_sequential(self) -> list[HealthCheckResult]:
        """Run health checks sequentially."""
        results = []

        for name, checker in self._components.items():
            try:
                result = await checker.check_health()
                results.append(result)
            except Exception as e:
                logger.exception(f"Health check failed for {name}: {e}")
                results.append(
                    HealthCheckResult(
                        status=HealthStatus.UNKNOWN,
                        component_name=name,
                        message=f"Health check error: {e!s}",
                        error=str(e),
                    )
                )

        return results

    def _calculate_health_summary(
        self, results: list[HealthCheckResult], start_time: float
    ) -> SystemHealthSummary:
        """Calculate overall health summary from individual results."""
        total_components = len(results)
        healthy_components = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        degraded_components = sum(1 for r in results if r.status == HealthStatus.DEGRADED)
        unhealthy_components = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
        unknown_components = sum(1 for r in results if r.status == HealthStatus.UNKNOWN)

        # Determine overall status
        if unhealthy_components > 0:
            # Check if any critical components are unhealthy
            critical_unhealthy = any(
                r.status == HealthStatus.UNHEALTHY and r.component_name in self.config.critical_components
                for r in results
            )
            overall_status = HealthStatus.UNHEALTHY if critical_unhealthy else HealthStatus.DEGRADED
        elif degraded_components > 0 or unknown_components > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        component_results = {r.component_name: r for r in results}

        return SystemHealthSummary(
            overall_status=overall_status,
            total_components=total_components,
            healthy_components=healthy_components,
            degraded_components=degraded_components,
            unhealthy_components=unhealthy_components,
            unknown_components=unknown_components,
            component_results=component_results,
            check_duration_ms=(time.time() - start_time) * 1000,
        )

    async def run_component_check(self, component_name: str) -> HealthCheckResult:
        """Run health check for specific component."""
        if component_name not in self._components:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                component_name=component_name,
                message="Component not found",
                error="Component not registered",
            )

        try:
            checker = self._components[component_name]
            result = await checker.check_health()

            # Store result
            self._last_results[component_name] = result

            if self.repository:
                try:
                    await self.repository.store_health_result(result)
                except Exception as e:
                    logger.warning(f"Failed to store health result for {component_name}: {e}")

            return result

        except Exception as e:
            logger.exception(f"Component health check failed for {component_name}: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                component_name=component_name,
                message=f"Health check failed: {e!s}",
                error=str(e),
            )

    def register_component(self, checker: HealthCheckComponentProtocol) -> None:
        """Register health check component."""
        name = checker.get_component_name()
        self._components[name] = checker
        logger.info(f"Registered health check component: {name}")

    def unregister_component(self, component_name: str) -> bool:
        """Unregister health check component."""
        if component_name in self._components:
            del self._components[component_name]
            if component_name in self._last_results:
                del self._last_results[component_name]
            logger.info(f"Unregistered health check component: {component_name}")
            return True
        return False

    def get_registered_components(self) -> list[str]:
        """Get list of registered component names."""
        return list(self._components.keys())


class MetricsCollectionService:
    """Internal service for metrics collection."""

    def __init__(
        self,
        config: MonitoringConfig,
        repository: MonitoringRepositoryProtocol | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self._custom_metrics: list[MetricPoint] = []

    async def collect_system_metrics(self) -> list[MetricPoint]:
        """Collect system-level metrics."""
        metrics = []

        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(
                MetricPoint(
                    name="system.cpu.usage_percent",
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="CPU usage percentage",
                )
            )

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.extend([
                MetricPoint(
                    name="system.memory.usage_percent",
                    value=memory.percent,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="Memory usage percentage",
                ),
                MetricPoint(
                    name="system.memory.available_bytes",
                    value=memory.available,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Available memory in bytes",
                ),
            ])

            # Disk metrics
            disk = psutil.disk_usage("/")
            metrics.extend([
                MetricPoint(
                    name="system.disk.usage_percent",
                    value=(disk.used / disk.total) * 100,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="Disk usage percentage",
                ),
                MetricPoint(
                    name="system.disk.free_bytes",
                    value=disk.free,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Free disk space in bytes",
                ),
            ])

        except ImportError:
            logger.warning("psutil not available for system metrics collection")
        except Exception as e:
            logger.exception(f"Failed to collect system metrics: {e}")

        return metrics

    async def collect_application_metrics(self) -> list[MetricPoint]:
        """Collect application-level metrics."""
        metrics = []

        try:
            # Process metrics
            import os

            import psutil

            process = psutil.Process(os.getpid())

            # Process CPU and memory
            metrics.extend([
                MetricPoint(
                    name="app.process.cpu_percent",
                    value=process.cpu_percent(),
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="Process CPU usage percentage",
                ),
                MetricPoint(
                    name="app.process.memory_rss_bytes",
                    value=process.memory_info().rss,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Process RSS memory usage",
                ),
                MetricPoint(
                    name="app.process.open_files",
                    value=len(process.open_files()),
                    metric_type=MetricType.GAUGE,
                    unit="count",
                    description="Number of open files",
                ),
            ])

        except ImportError:
            logger.warning("psutil not available for application metrics collection")
        except Exception as e:
            logger.exception(f"Failed to collect application metrics: {e}")

        return metrics

    async def collect_component_metrics(self, component_name: str) -> list[MetricPoint]:
        """Collect metrics for specific component."""
        return []

        # This could be extended to collect component-specific metrics
        # For now, return empty list as component-specific metrics would
        # require integration with individual components

    def record_metric(self, metric: MetricPoint) -> None:
        """Record a single metric point."""
        self._custom_metrics.append(metric)

        # Store in repository if available
        if self.repository:
            asyncio.create_task(self._store_metric(metric))

    async def _store_metric(self, metric: MetricPoint) -> None:
        """Store metric in repository."""
        try:
            await self.repository.store_metrics([metric])
        except Exception as e:
            logger.warning(f"Failed to store metric {metric.name}: {e}")

    async def get_all_metrics(self) -> list[MetricPoint]:
        """Get all collected metrics."""
        all_metrics = []

        if self.config.metrics_collection_enabled:
            # Collect system and application metrics
            system_metrics = await self.collect_system_metrics()
            app_metrics = await self.collect_application_metrics()

            all_metrics.extend(system_metrics)
            all_metrics.extend(app_metrics)

        # Add custom metrics
        all_metrics.extend(self._custom_metrics)

        # Store all metrics
        if self.repository and all_metrics:
            try:
                await self.repository.store_metrics(all_metrics)
            except Exception as e:
                logger.warning(f"Failed to store metrics batch: {e}")

        return all_metrics

    def clear_custom_metrics(self) -> None:
        """Clear stored custom metrics."""
        self._custom_metrics.clear()
