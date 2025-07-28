"""
Base classes to eliminate inheritance duplication patterns.

Provides common base classes for:
- Configuration models
- Health checkers
- Service classes
- Monitor classes

Consolidates common initialization and utility patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta, UTC
from enum import Enum
from dataclasses import dataclass
import asyncio

from pydantic import BaseModel
from .logging_utils import LoggerMixin
from .config_utils import ConfigMixin
from .metrics_utils import MetricsMixin

class ServiceState(str, Enum):
    """Common service state enumeration."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    STOPPED = "stopped"

class HealthStatus(str, Enum):
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
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    critical: bool = False

class BaseConfigModel(BaseModel):
    """
    Base configuration model with common validation patterns.
    
    Consolidates duplicate validation logic across config classes.
    """
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
        use_enum_values = True  # Use enum values in output
    
    @classmethod
    def validate_port(cls, v: int, field_name: str = "port") -> int:
        """Common port validation."""
        if not isinstance(v, int) or v < 1 or v > 65535:
            raise ValueError(f"{field_name} must be between 1 and 65535, got {v}")
        return v
    
    @classmethod
    def validate_timeout(cls, v: Union[int, float], field_name: str = "timeout") -> Union[int, float]:
        """Common timeout validation."""
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v
    
    @classmethod
    def validate_ratio(cls, v: float, field_name: str = "ratio") -> float:
        """Common ratio validation."""
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError(f"{field_name} must be between 0 and 1, got {v}")
        return v
    
    @classmethod
    def validate_percentage(cls, v: float, field_name: str = "percentage") -> float:
        """Common percentage validation."""
        if not isinstance(v, (int, float)) or v < 0 or v > 100:
            raise ValueError(f"{field_name} must be between 0 and 100, got {v}")
        return v

class BaseService(LoggerMixin, ConfigMixin, MetricsMixin, ABC):
    """
    Base service class with common patterns.
    
    Consolidates common service initialization and lifecycle patterns.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.state = ServiceState.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.error_count = 0
        
    @abstractmethod
    async def start(self) -> bool:
        """Start the service."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        pass
    
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
    
    def get_uptime(self) -> Optional[timedelta]:
        """Get service uptime."""
        if self.start_time is None:
            return None
        return datetime.now(UTC) - self.start_time
    
    def record_error(self, error: str) -> None:
        """Record an error for this service."""
        self.last_error = error
        self.error_count += 1
        self.logger.error(f"Service {self.name} error: {error}")
        
        # Record error metric if available
        self.increment_counter("service_errors_total", {
            "service": self.name,
            "error": error
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "name": self.name,
            "state": self.state.value,
            "uptime_seconds": self.get_uptime().total_seconds() if self.get_uptime() else None,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "config_status": self.config_status.success if hasattr(self, 'config_status') else None,
            "metrics_available": self.metrics_available
        }

class BaseHealthChecker(LoggerMixin, MetricsMixin, ABC):
    """
    Base health checker class with common patterns.
    
    Consolidates duplicate health checking patterns across monitoring modules.
    """
    
    def __init__(self, name: str, timeout: float = 5.0, critical: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.timeout = timeout
        self.critical = critical
        self.last_result: Optional[HealthCheckResult] = None
        self.consecutive_failures = 0
        self.max_failures = 3
    
    @abstractmethod
    async def _perform_check(self) -> HealthCheckResult:
        """Perform the actual health check."""
        pass
    
    async def check(self) -> HealthCheckResult:
        """
        Perform health check with timeout and error handling.
        
        Returns:
            HealthCheckResult with status and details
        """
        start_time = datetime.now(UTC)
        
        try:
            # Perform check with timeout
            result = await asyncio.wait_for(
                self._perform_check(),
                timeout=self.timeout
            )
            
            # Update consecutive failures counter
            if result.status == HealthStatus.HEALTHY:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            self.last_result = result
            
            # Record metrics
            self._record_health_metrics(result)
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self.consecutive_failures += 1
            
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                details={"timeout": True, "consecutive_failures": self.consecutive_failures},
                timestamp=start_time,
                duration_ms=duration_ms,
                critical=self.critical
            )
            
            self.last_result = result
            self._record_health_metrics(result)
            
            return result
            
        except Exception as e:
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self.consecutive_failures += 1
            
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "consecutive_failures": self.consecutive_failures},
                timestamp=start_time,
                duration_ms=duration_ms,
                critical=self.critical
            )
            
            self.last_result = result
            self.logger.error(f"Health check {self.name} failed: {e}")
            self._record_health_metrics(result)
            
            return result
    
    def _record_health_metrics(self, result: HealthCheckResult) -> None:
        """Record health check metrics."""
        # Record check duration
        self.observe_histogram("health_check_duration_seconds", result.duration_ms / 1000, {
            "check": self.name,
            "status": result.status.value
        })
        
        # Record check status
        self.increment_counter("health_checks_total", {
            "check": self.name,
            "status": result.status.value,
            "critical": str(result.critical)
        })
        
        # Record consecutive failures if any
        if self.consecutive_failures > 0:
            self.set_gauge("health_check_consecutive_failures", self.consecutive_failures, {
                "check": self.name
            })
    
    def is_degraded(self) -> bool:
        """Check if the health checker is in a degraded state."""
        return self.consecutive_failures >= self.max_failures
    
    def get_status(self) -> Dict[str, Any]:
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
                "duration_ms": self.last_result.duration_ms
            } if self.last_result else None
        }

class BaseMonitor(BaseService):
    """
    Base monitor class with common monitoring patterns.
    
    Consolidates duplicate monitoring initialization and operation patterns.
    """
    
    def __init__(self, name: str, check_interval: float = 30.0, **kwargs):
        super().__init__(name, **kwargs)
        self.check_interval = check_interval
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_checkers: Dict[str, BaseHealthChecker] = {}
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
        if self.monitoring_task and not self.monitoring_task.done():
            self.logger.warning(f"Monitor {self.name} already running")
            return True
        
        self.state = ServiceState.HEALTHY
        self.start_time = datetime.now(UTC)
        self.monitoring_enabled = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"Started monitor: {self.name}")
        return True
    
    async def stop(self) -> bool:
        """Stop monitoring."""
        self.monitoring_enabled = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
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
                critical=True
            )
        
        # Check if monitoring task is running
        if not self.monitoring_task or self.monitoring_task.done():
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Monitoring task is not running",
                details={"task_running": False},
                timestamp=datetime.now(UTC),
                duration_ms=0,
                critical=True
            )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Monitor is running normally",
            details={
                "checkers_count": len(self.health_checkers),
                "uptime_seconds": self.get_uptime().total_seconds() if self.get_uptime() else 0
            },
            timestamp=datetime.now(UTC),
            duration_ms=0,
            critical=False
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
        pass
    
    async def get_all_health_results(self) -> Dict[str, HealthCheckResult]:
        """Get health check results for all registered checkers."""
        results = {}
        
        for name, checker in self.health_checkers.items():
            try:
                result = await checker.check()
                results[name] = result
            except Exception as e:
                self.logger.error(f"Failed to check {name}: {e}")
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                    details={"error": str(e)},
                    timestamp=datetime.now(UTC),
                    duration_ms=0,
                    critical=checker.critical
                )
        
        return results