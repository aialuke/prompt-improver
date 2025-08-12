"""Unified Health System with Plugin Architecture

Consolidates 15+ health checkers into a plugin-based architecture for:
- EnhancedMLServiceHealthChecker
- AnalyticsServiceHealthChecker
- MLServiceHealthChecker
- Database health checkers (5 types)
- Redis health checkers (3 variants)
- API endpoint health checkers (4 types)

Features:
- Plugin-based architecture with runtime registration
- Category-based health reporting (ML, Database, Redis, API, System)
- Performance optimized (<10ms per health check)
- Environment-specific health check profiles
- Unified configuration management
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from prompt_improver.core.protocols.health_protocol import (
    HealthCheckResult,
    HealthStatus,
)
from prompt_improver.performance.monitoring.health.base import (
    AggregatedHealthResult,
    HealthChecker,
    HealthResult,
    HealthStatus as BaseHealthStatus,
)

logger = logging.getLogger(__name__)


def _get_background_task_manager():
    """Lazy import background task manager"""
    from prompt_improver.performance.monitoring.health.background_manager import (
        get_background_task_manager,
    )

    return get_background_task_manager()


def _get_task_priority():
    """Lazy import TaskPriority"""
    from prompt_improver.performance.monitoring.health.background_manager import (
        TaskPriority,
    )

    return TaskPriority


def _get_enhanced_health_checker():
    """Lazy import EnhancedHealthChecker"""
    from prompt_improver.performance.monitoring.health.enhanced_base import (
        EnhancedHealthChecker,
    )

    return EnhancedHealthChecker


def _get_health_protocol_types():
    """Lazy import health protocol types"""
    from prompt_improver.core.protocols.health_protocol import (
        HealthCheckResult,
        HealthMonitorProtocol,
        HealthStatus,
    )

    return (HealthMonitorProtocol, HealthCheckResult, HealthStatus)


def _convert_base_status_to_protocol(status: BaseHealthStatus) -> HealthStatus:
    """Convert base.HealthStatus to health_protocol.HealthStatus"""
    _, _, HealthStatus = _get_health_protocol_types()
    mapping = {
        BaseHealthStatus.HEALTHY: HealthStatus.HEALTHY,
        BaseHealthStatus.WARNING: HealthStatus.DEGRADED,
        BaseHealthStatus.FAILED: HealthStatus.UNHEALTHY,
    }
    return mapping.get(status, HealthStatus.UNKNOWN)


def _convert_protocol_status_to_base(status: HealthStatus) -> BaseHealthStatus:
    """Convert health_protocol.HealthStatus to base.HealthStatus"""
    _, _, HealthStatus = _get_health_protocol_types()
    mapping = {
        HealthStatus.HEALTHY: BaseHealthStatus.HEALTHY,
        HealthStatus.DEGRADED: BaseHealthStatus.WARNING,
        HealthStatus.UNHEALTHY: BaseHealthStatus.FAILED,
        HealthStatus.UNKNOWN: BaseHealthStatus.FAILED,
    }
    return mapping.get(status, BaseHealthStatus.FAILED)


def _convert_health_result_to_check_result(
    health_result: HealthResult,
) -> HealthCheckResult:
    """Convert HealthResult to HealthCheckResult"""
    _, HealthCheckResult, _ = _get_health_protocol_types()
    return HealthCheckResult(
        status=_convert_base_status_to_protocol(health_result.status),
        message=health_result.message or "",
        details=health_result.details or {},
        check_name=health_result.component,
        duration_ms=health_result.response_time_ms or 0.0,
    )


def _convert_check_result_to_health_result(
    check_result: HealthCheckResult,
) -> HealthResult:
    """Convert HealthCheckResult to HealthResult"""
    return HealthResult(
        status=_convert_protocol_status_to_base(check_result.status),
        component=check_result.check_name,
        response_time_ms=check_result.duration_ms,
        message=check_result.message,
        error=check_result.details.get("error") if check_result.details else None,
        details=check_result.details,
        timestamp=datetime.now(UTC),
    )


class HealthCheckCategory(Enum):
    """Health check categories for organization and reporting"""

    ML = "ml"
    DATABASE = "database"
    REDIS = "redis"
    API = "api"
    SYSTEM = "system"
    EXTERNAL = "external"
    CUSTOM = "custom"


@dataclass
class HealthCheckPluginConfig:
    """Configuration for health check plugins"""

    enabled: bool = True
    timeout_seconds: float = 10.0
    critical: bool = False
    interval_seconds: float | None = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


class HealthCheckPlugin(ABC):
    """Base class for all health check plugins.

    Provides standardized interface for health checking with:
    - Category-based organization
    - Configuration management
    - Performance monitoring
    - Error handling
    """

    def __init__(
        self,
        name: str,
        category: HealthCheckCategory,
        config: HealthCheckPluginConfig | None = None,
    ):
        self.name = name
        self.category = category
        self.config = config or HealthCheckPluginConfig()
        self._last_check_time: datetime | None = None
        self._last_result: HealthCheckResult | None = None
        self._check_count = 0
        self._failure_count = 0

    @abstractmethod
    async def execute_check(self) -> HealthCheckResult:
        """Execute the actual health check logic"""

    async def check_health(self) -> HealthCheckResult:
        """Perform health check with timing, error handling, and retry logic"""
        start_time = time.time()
        attempt = 0
        max_attempts = self.config.retry_count + 1
        while attempt < max_attempts:
            try:
                result = await asyncio.wait_for(
                    self.execute_check(), timeout=self.config.timeout_seconds
                )
                duration_ms = (time.time() - start_time) * 1000
                result.duration_ms = duration_ms
                result.check_name = self.name
                self._last_check_time = datetime.now(UTC)
                self._last_result = result
                self._check_count += 1
                if result.status != HealthStatus.HEALTHY:
                    self._failure_count += 1
                return result
            except TimeoutError:
                attempt += 1
                if attempt >= max_attempts:
                    duration_ms = (time.time() - start_time) * 1000

                    result = HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check timed out after {self.config.timeout_seconds}s",
                        details={"timeout": True, "attempts": attempt},
                        check_name=self.name,
                        duration_ms=duration_ms,
                    )
                    self._last_result = result
                    self._failure_count += 1
                    return result
                if self.config.retry_delay_seconds > 0:
                    await asyncio.sleep(self.config.retry_delay_seconds)
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    duration_ms = (time.time() - start_time) * 1000

                    result = HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {e!s}",
                        details={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempts": attempt,
                        },
                        check_name=self.name,
                        duration_ms=duration_ms,
                    )
                    self._last_result = result
                    self._failure_count += 1
                    return result
                if self.config.retry_delay_seconds > 0:
                    await asyncio.sleep(self.config.retry_delay_seconds)

    def get_plugin_metrics(self) -> dict[str, Any]:
        """Get plugin-specific metrics"""
        return {
            "name": self.name,
            "category": self.category.value,
            "enabled": self.config.enabled,
            "check_count": self._check_count,
            "failure_count": self._failure_count,
            "success_rate": (self._check_count - self._failure_count)
            / max(self._check_count, 1),
            "last_check_time": self._last_check_time.isoformat()
            if self._last_check_time
            else None,
            "last_status": self._last_result.status.value
            if self._last_result
            else None,
            "configuration": {
                "timeout_seconds": self.config.timeout_seconds,
                "critical": self.config.critical,
                "retry_count": self.config.retry_count,
                "tags": list(self.config.tags),
            },
        }


@dataclass
class HealthProfile:
    """Environment-specific health check profile"""

    name: str
    enabled_plugins: set[str] = field(default_factory=set)
    disabled_plugins: set[str] = field(default_factory=set)
    category_configs: dict[HealthCheckCategory, HealthCheckPluginConfig] = field(
        default_factory=dict
    )
    global_timeout: float = 30.0
    parallel_execution: bool = True
    critical_only: bool = False


class UnifiedHealthMonitor:
    """Unified health monitoring system with plugin architecture.

    Manages health check plugins with:
    - Runtime registration/deregistration
    - Category-based organization
    - Performance optimization
    - Configuration profiles
    - Comprehensive reporting
    """

    def __init__(self, default_profile: HealthProfile | None = None):
        self._plugins: dict[str, HealthCheckPlugin] = {}
        self._categories: dict[HealthCheckCategory, set[str]] = {
            category: set() for category in HealthCheckCategory
        }
        self._active_profile = default_profile or HealthProfile(name="default")
        self._health_profiles: dict[str, HealthProfile] = {
            "default": self._active_profile
        }
        self._background_tasks: set[asyncio.Task] = set()
        self._periodic_checkers: dict[str, str] = {}

    async def register_plugin(
        self, plugin: HealthCheckPlugin, auto_enable: bool = True
    ) -> bool:
        """Register a health check plugin.

        Args:
            plugin: The health check plugin to register
            auto_enable: Whether to automatically enable the plugin

        Returns:
            True if successfully registered, False if name already exists
        """
        if plugin.name in self._plugins:
            logger.warning(f"Plugin {plugin.name} already registered")
            return False
        self._plugins[plugin.name] = plugin
        self._categories[plugin.category].add(plugin.name)
        if auto_enable:
            self._active_profile.enabled_plugins.add(plugin.name)
        logger.info(
            f"Registered health check plugin: {plugin.name} ({plugin.category.value})"
        )
        if plugin.config.interval_seconds:
            await self._start_periodic_check(plugin)
        return True

    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a health check plugin.

        Args:
            plugin_name: Name of the plugin to remove

        Returns:
            True if plugin was removed, False if not found
        """
        if plugin_name not in self._plugins:
            return False
        plugin = self._plugins[plugin_name]
        if plugin_name in self._periodic_checkers:
            # Cancel periodic checker via background manager asynchronously
            task_id = self._periodic_checkers[plugin_name]
            asyncio.create_task(_get_background_task_manager().cancel_task(task_id))
            del self._periodic_checkers[plugin_name]
        del self._plugins[plugin_name]
        self._categories[plugin.category].discard(plugin_name)
        for profile in self._health_profiles.values():
            profile.enabled_plugins.discard(plugin_name)
            profile.disabled_plugins.discard(plugin_name)
        logger.info(f"Unregistered health check plugin: {plugin_name}")
        return True

    def get_registered_plugins(self) -> list[str]:
        """Get list of all registered plugin names"""
        return list(self._plugins.keys())

    def get_plugins_by_category(self, category: HealthCheckCategory) -> list[str]:
        """Get plugins by category"""
        return list(self._categories[category])

    async def check_health(
        self,
        plugin_name: str | None = None,
        category: HealthCheckCategory | None = None,
        include_details: bool = True,
    ) -> dict[str, HealthCheckResult]:
        """Perform health checks on registered plugins.

        Args:
            plugin_name: Specific plugin to check, None for enabled plugins
            category: Check only plugins in this category
            include_details: Whether to include detailed information

        Returns:
            Dictionary mapping plugin names to health results
        """
        start_time = time.time()
        plugins_to_check = self._get_plugins_to_check(plugin_name, category)
        if not plugins_to_check:
            return {}
        if self._active_profile.parallel_execution and len(plugins_to_check) > 1:
            results = await self._execute_parallel_checks(plugins_to_check)
        else:
            results = await self._execute_sequential_checks(plugins_to_check)
        total_duration = time.time() - start_time
        if total_duration > self._active_profile.global_timeout:
            logger.warning(
                f"Health check took {total_duration:.2f}s, exceeding global timeout of {self._active_profile.global_timeout:.2f}s"
            )
        return results

    async def get_overall_health(self):
        """Get overall system health status.

        Returns:
            Combined health result for the entire system
        """
        individual_results = await self.check_health()
        if not individual_results:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="No health checks registered",
                check_name="overall_health",
            )
        critical_failures: list[str] = []
        warnings: list[str] = []
        all_healthy = True
        for name, result in individual_results.items():
            plugin = self._plugins[name]
            if result.status == HealthStatus.UNHEALTHY:
                all_healthy = False
                if plugin.config.critical:
                    critical_failures.append(name)
                else:
                    warnings.append(name)
            elif result.status == HealthStatus.DEGRADED:
                warnings.append(name)

        if critical_failures:
            overall_status = HealthStatus.UNHEALTHY
            message = f"Critical health check failures: {', '.join(critical_failures)}"
        elif warnings:
            overall_status = HealthStatus.DEGRADED
            message = f"Health check warnings: {', '.join(warnings)}"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All health checks passing"
        return HealthCheckResult(
            status=overall_status,
            message=message,
            details={
                "total_checks": len(individual_results),
                "critical_failures": critical_failures,
                "warnings": warnings,
                "healthy_count": len([
                    r
                    for r in individual_results.values()
                    if r.status == HealthStatus.HEALTHY
                ]),
                "individual_results": {
                    name: {
                        "status": result.status.value,
                        "message": result.message,
                        "duration_ms": result.duration_ms,
                    }
                    for name, result in individual_results.items()
                },
            },
            check_name="overall_health",
        )

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health monitoring summary.

        Returns:
            Dictionary containing health monitoring metrics and statistics
        """
        return {
            "registered_plugins": len(self._plugins),
            "enabled_plugins": len(self._active_profile.enabled_plugins),
            "disabled_plugins": len(self._active_profile.disabled_plugins),
            "categories": {
                category.value: len(plugins)
                for category, plugins in self._categories.items()
            },
            "active_profile": self._active_profile.name,
            "periodic_checkers": len(self._periodic_checkers),
            "plugin_metrics": {
                name: plugin.get_plugin_metrics()
                for name, plugin in self._plugins.items()
            },
            "configuration": {
                "global_timeout": self._active_profile.global_timeout,
                "parallel_execution": self._active_profile.parallel_execution,
                "critical_only": self._active_profile.critical_only,
            },
        }

    def create_health_profile(
        self, name: str, enabled_plugins: set[str] | None = None, **kwargs
    ) -> HealthProfile:
        """Create a new health profile"""
        profile = HealthProfile(
            name=name, enabled_plugins=enabled_plugins or set(), **kwargs
        )
        self._health_profiles[name] = profile
        return profile

    def activate_profile(self, profile_name: str) -> bool:
        """Activate a health profile"""
        if profile_name not in self._health_profiles:
            return False
        self._active_profile = self._health_profiles[profile_name]
        logger.info(f"Activated health profile: {profile_name}")
        return True

    def _get_plugins_to_check(
        self, plugin_name: str | None, category: HealthCheckCategory | None
    ) -> list[HealthCheckPlugin]:
        """Determine which plugins to check based on filters"""
        if plugin_name:
            if plugin_name in self._plugins:
                return [self._plugins[plugin_name]]
            return []
        plugins_to_check = []
        for name, plugin in self._plugins.items():
            if not plugin.config.enabled:
                continue
            if name not in self._active_profile.enabled_plugins:
                continue
            if name in self._active_profile.disabled_plugins:
                continue
            if category and plugin.category != category:
                continue
            if self._active_profile.critical_only and (not plugin.config.critical):
                continue
            plugins_to_check.append(plugin)
        return plugins_to_check

    async def _execute_parallel_checks(
        self, plugins: list[HealthCheckPlugin]
    ) -> dict[str, HealthCheckResult]:
        """Execute health checks in parallel using asyncio tasks to collect results"""
        tasks: dict[str, asyncio.Task[HealthCheckResult]] = {
            plugin.name: asyncio.create_task(plugin.check_health())
            for plugin in plugins
        }
        results: dict[str, HealthCheckResult] = {}
        for name, task in tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=30.0)
                results[name] = result
            except Exception as e:
                logger.error(f"Health check {name} failed with exception: {e}")

                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e!s}",
                    check_name=name,
                )
        return results

    async def _execute_sequential_checks(
        self, plugins: list[HealthCheckPlugin]
    ) -> dict[str, HealthCheckResult]:
        """Execute health checks sequentially"""
        results = {}
        for plugin in plugins:
            try:
                results[plugin.name] = await plugin.check_health()
            except Exception as e:
                logger.error(f"Health check {plugin.name} failed with exception: {e}")

                results[plugin.name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e!s}",
                    check_name=plugin.name,
                )
        return results

    async def _start_periodic_check(self, plugin: HealthCheckPlugin) -> None:
        """Start periodic health checking for a plugin using enhanced task management"""
        if not plugin.config.interval_seconds:
            return

        async def periodic_checker():
            while True:
                try:
                    await asyncio.sleep(plugin.config.interval_seconds)
                    await plugin.check_health()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Periodic health check {plugin.name} failed: {e}")

        task_manager = _get_background_task_manager()
        TaskPriority = _get_task_priority()
        task_id = await task_manager.submit_enhanced_task(
            task_id=f"periodic_health_check_{plugin.name}",
            coroutine=periodic_checker(),
            priority=TaskPriority.NORMAL,
            tags={
                "service": "health_system",
                "type": "periodic_check",
                "plugin": plugin.name,
            },
        )
        self._periodic_checkers[plugin.name] = task_id

    async def shutdown(self) -> None:
        """Shutdown the health monitor and cancel all background tasks"""
        task_manager = _get_background_task_manager()
        for task_id in self._periodic_checkers.values():
            await task_manager.cancel_task(task_id)
        self._periodic_checkers.clear()
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        logger.info("Unified health monitor shutdown completed")


_unified_health_monitor: UnifiedHealthMonitor | None = None


def get_unified_health_monitor() -> UnifiedHealthMonitor:
    """Get the global unified health monitor instance"""
    global _unified_health_monitor
    if _unified_health_monitor is None:
        _unified_health_monitor = UnifiedHealthMonitor()
    return _unified_health_monitor


async def register_health_plugin(plugin: HealthCheckPlugin) -> bool:
    """Register a health check plugin with the global monitor"""
    return await get_unified_health_monitor().register_plugin(plugin)


def create_simple_health_plugin(
    name: str,
    category: HealthCheckCategory,
    check_func: Callable[[], bool | dict[str, Any]],
    config: HealthCheckPluginConfig | None = None,
) -> HealthCheckPlugin:
    """Create a simple health check plugin from a callable.

    Args:
        name: Plugin name
        category: Plugin category
        check_func: Function that returns health status (bool, dict, or HealthCheckResult)
        config: Plugin configuration

    Returns:
        Health check plugin ready for registration
    """

    class SimpleHealthPlugin(HealthCheckPlugin):
        async def execute_check(self):
            try:
                result = check_func()
                if hasattr(result, "status") and hasattr(result, "message"):
                    return result
                if isinstance(result, bool):
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY
                        if result
                        else HealthStatus.UNHEALTHY,
                        message="Health check " + ("passed" if result else "failed"),
                        check_name=self.name,
                    )
                if isinstance(result, dict):
                    status = result.get("status", HealthStatus.HEALTHY)
                    if isinstance(status, str):
                        status = HealthStatus(status)
                    return HealthCheckResult(
                        status=status,
                        message=result.get("message", ""),
                        details=result.get("details"),
                        check_name=self.name,
                    )
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    check_name=self.name,
                )
            except Exception as e:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e!s}",
                    details={"error": str(e)},
                    check_name=self.name,
                )

    return SimpleHealthPlugin(name, category, config)
