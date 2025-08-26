"""Base classes to eliminate inheritance duplication patterns.

Provides common base classes for:
- Configuration models
- Health checkers
- Service classes
- Monitor classes

Consolidates common initialization and utility patterns.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from prompt_improver.core.common.config_utils import ConfigMixin
from prompt_improver.core.common.metrics_utils import MetricsMixin
from prompt_improver.performance.monitoring.health.background_manager import (
    TaskPriority,
    get_background_task_manager,
)


class ServiceState(StrEnum):
    """Common service state enumeration."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    STOPPED = "stopped"


class HealthStatus(StrEnum):
    """Common health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Standard health check result structure."""

    status: HealthStatus
    message: str
    details: dict[str, Any]
    timestamp: datetime
    duration_ms: float
    critical: bool = False


class BaseConfigModel(BaseModel):
    """Base configuration model with common validation patterns.

    Consolidates duplicate validation logic across config classes.
    """

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "use_enum_values": True,
    }

    @classmethod
    def validate_port(cls, v: Any, field_name: str = "port") -> int:
        """Common port validation."""
        if not isinstance(v, int) or v < 1 or v > 65535:
            raise ValueError(f"{field_name} must be between 1 and 65535, got {v}")
        return v

    @classmethod
    def validate_timeout(cls, v: Any, field_name: str = "timeout") -> int | float:
        """Common timeout validation."""
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v

    @classmethod
    def validate_ratio(cls, v: Any, field_name: str = "ratio") -> float:
        """Common ratio validation."""
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError(f"{field_name} must be between 0 and 1, got {v}")
        return float(v)

    @classmethod
    def validate_percentage(cls, v: Any, field_name: str = "percentage") -> float:
        """Common percentage validation."""
        if not isinstance(v, (int, float)) or v < 0 or v > 100:
            raise ValueError(f"{field_name} must be between 0 and 100, got {v}")
        return float(v)


class BaseService(ConfigMixin, MetricsMixin, ABC):
    """Base service class with common patterns.

    Consolidates common service initialization and lifecycle patterns.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.state = ServiceState.INITIALIZING
        self.start_time: datetime | None = None
        self.last_error: str | None = None
        self.error_count = 0

    @abstractmethod
    async def start(self) -> bool:
        """Start the service."""

    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service."""

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform health check."""

    async def restart(self) -> bool:
        """Restart the service."""
        self.logger.info(f"Restarting service: {self.name}")
        if not await self.stop():
            self.logger.error(f"Failed to stop service: {self.name}")
            return False
        if not await self.start():
            self.logger.error(f"Failed to start service: {self.name}")
            return False
        self.logger.info(f"Service restarted successfully: {self.name}")
        return True

    def get_uptime(self) -> timedelta | None:
        """Get service uptime."""
        if self.start_time is None:
            return None
        return datetime.now(UTC) - self.start_time

    def record_error(self, error: str) -> None:
        """Record an error for this service."""
        self.last_error = error
        self.error_count += 1
        self.logger.error(f"Service {self.name} error: {error}")
        self.increment_counter(
            "service_errors_total", {"service": self.name, "error": error}
        )

    def get_status(self) -> dict[str, Any]:
        """Get service status information."""
        uptime = self.get_uptime()
        return {
            "name": self.name,
            "state": self.state.value,
            "uptime_seconds": uptime.total_seconds() if uptime is not None else None,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "config_status": self.config_status.success
            if hasattr(self, "config_status")
            else None,
            "metrics_available": self.metrics_available,
        }


class BaseHealthChecker(MetricsMixin, ABC):
    """Base health checker class with common patterns.

    Consolidates duplicate health checking patterns across monitoring modules.
    """

    def __init__(
        self, name: str, timeout: float = 5.0, critical: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.timeout = timeout
        self.critical = critical
        self.last_result: HealthCheckResult | None = None
        self.consecutive_failures = 0
        self.max_failures = 3

    @abstractmethod
    async def _perform_check(self) -> HealthCheckResult:
        """Perform the actual health check."""

    async def check(self) -> HealthCheckResult:
        """Perform health check with timeout and error handling.

        Returns:
            HealthCheckResult with status and details
        """
        start_time = datetime.now(UTC)
        try:
            result = await asyncio.wait_for(self._perform_check(), timeout=self.timeout)
            if result.status == HealthStatus.HEALTHY:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            self.last_result = result
            self._record_health_metrics(result)
            return result
        except TimeoutError:
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self.consecutive_failures += 1
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                details={
                    "timeout": True,
                    "consecutive_failures": self.consecutive_failures,
                },
                timestamp=start_time,
                duration_ms=duration_ms,
                critical=self.critical,
            )
            self.last_result = result
            self._record_health_metrics(result)
            return result
        except Exception as e:
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self.consecutive_failures += 1
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e!s}",
                details={
                    "error": str(e),
                    "consecutive_failures": self.consecutive_failures,
                },
                timestamp=start_time,
                duration_ms=duration_ms,
                critical=self.critical,
            )
            self.last_result = result
            self.logger.exception(f"Health check {self.name} failed: {e}")
            self._record_health_metrics(result)
            return result

    def _record_health_metrics(self, result: HealthCheckResult) -> None:
        """Record health check metrics."""
        self.observe_histogram(
            "health_check_duration_seconds",
            result.duration_ms / 1000,
            {"check": self.name, "status": result.status.value},
        )
        self.increment_counter(
            "health_checks_total",
            {
                "check": self.name,
                "status": result.status.value,
                "critical": str(result.critical),
            },
        )
        if self.consecutive_failures > 0:
            self.set_gauge(
                "health_check_consecutive_failures",
                self.consecutive_failures,
                {"check": self.name},
            )

    def is_degraded(self) -> bool:
        """Check if the health checker is in a degraded state."""
        return self.consecutive_failures >= self.max_failures

    def get_status(self) -> dict[str, Any]:
        """Get health checker status."""
        return {
            "name": self.name,
            "timeout": self.timeout,
            "critical": self.critical,
            "consecutive_failures": self.consecutive_failures,
            "max_failures": self.max_failures,
            "is_degraded": self.is_degraded(),
            "last_result": {
                "status": self.last_result.status.value,
                "message": self.last_result.message,
                "timestamp": self.last_result.timestamp.isoformat(),
                "duration_ms": self.last_result.duration_ms,
            }
            if self.last_result
            else None,
        }


class BaseMonitor(BaseService):
    """Base monitor class with common monitoring patterns.

    Consolidates duplicate monitoring initialization and operation patterns.
    """

    def __init__(self, name: str, check_interval: float = 30.0, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self.check_interval = check_interval
        self.monitoring_task_id: str | None = None
        self.health_checkers: dict[str, BaseHealthChecker] = {}
        self.monitoring_enabled = True

    def add_health_checker(self, checker: BaseHealthChecker) -> None:
        """Add a health checker to this monitor."""
        self.health_checkers[checker.name] = checker
        self.logger.info(f"Added health checker: {checker.name}")

    def remove_health_checker(self, name: str) -> None:
        """Remove a health checker from this monitor."""
        if name in self.health_checkers:
            del self.health_checkers[name]
            self.logger.info(f"Removed health checker: {name}")

    async def start(self) -> bool:
        """Start monitoring."""
        if self.monitoring_task and (not self.monitoring_task.done()):
            self.logger.warning(f"Monitor {self.name} already running")
            return True
        self.state = ServiceState.HEALTHY
        self.start_time = datetime.now(UTC)
        self.monitoring_enabled = True
        task_manager = get_background_task_manager()
        self.monitoring_task_id = await task_manager.submit_enhanced_task(
            task_id=f"monitor_{self.name}_{str(uuid.uuid4())[:8]}",
            coroutine=self._monitoring_loop(),
            priority=TaskPriority.HIGH,
            tags={
                "service": "monitoring",
                "type": "health_monitoring",
                "component": self.name,
                "operation": "continuous_monitoring",
            },
        )
        self.logger.info(f"Started monitor: {self.name}")
        return True

    async def stop(self) -> bool:
        """Stop monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self.monitoring_task_id)
            self.monitoring_task_id = None
        self.state = ServiceState.STOPPED
        self.logger.info(f"Stopped monitor: {self.name}")
        return True

    async def health_check(self) -> HealthCheckResult:
        """Perform health check for this monitor."""
        if not self.monitoring_enabled:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Monitoring is disabled",
                details={"enabled": False},
                timestamp=datetime.now(UTC),
                duration_ms=0,
                critical=True,
            )
        if not self.monitoring_task_id:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Monitoring task is not running",
                details={"task_running": False},
                timestamp=datetime.now(UTC),
                duration_ms=0,
                critical=True,
            )
        task_manager = get_background_task_manager()
        task_status = await task_manager.get_task_status(self.monitoring_task_id)
        if not task_status or task_status.get("status") not in {"running", "pending"}:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Monitoring task is not running",
                details={"task_running": False, "task_status": task_status},
                timestamp=datetime.now(UTC),
                duration_ms=0,
                critical=True,
            )
        uptime = self.get_uptime()
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Monitor is running normally",
            details={
                "checkers_count": len(self.health_checkers),
                "uptime_seconds": uptime.total_seconds() if uptime is not None else 0,
            },
            timestamp=datetime.now(UTC),
            duration_ms=0,
            critical=False,
        )

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._perform_monitoring_cycle()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.record_error(f"Monitoring cycle error: {e}")
                await asyncio.sleep(self.check_interval)

    @abstractmethod
    async def _perform_monitoring_cycle(self) -> None:
        """Perform one monitoring cycle."""

    async def get_all_health_results(self) -> dict[str, HealthCheckResult]:
        """Get health check results for all registered checkers."""
        results: dict[str, HealthCheckResult] = {}
        for name, checker in self.health_checkers.items():
            try:
                result = await checker.check()
                results[name] = result
            except Exception as e:
                self.logger.exception(f"Failed to check {name}: {e}")
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                    details={"error": str(e)},
                    timestamp=datetime.now(UTC),
                    duration_ms=0,
                    critical=checker.critical,
                )
        return results
