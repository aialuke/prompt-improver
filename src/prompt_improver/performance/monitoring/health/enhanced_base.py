"""Enhanced Base Health Checker with 2025 features
Integrates circuit breaker, SLA monitoring, structured logging, and OpenTelemetry
"""

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, Optional

from prompt_improver.performance.monitoring.health.base import (
    HealthChecker,
    HealthResult,
)
from prompt_improver.performance.monitoring.health.circuit_breaker import (
    CircuitBreakerConfig,
    circuit_breaker_registry,
)
from prompt_improver.performance.monitoring.health.sla_monitor import (
    SLAConfiguration,
    get_or_create_sla_monitor,
)
from prompt_improver.performance.monitoring.health.structured_logging import (
    StructuredLogger,
    get_metrics_logger,
)
from prompt_improver.performance.monitoring.health.telemetry import TelemetryContext

logger = logging.getLogger(__name__)


class EnhancedHealthChecker(HealthChecker):
    """Enhanced health checker with 2025 observability features"""

    def __init__(
        self,
        component_name: str,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        sla_config: SLAConfiguration | None = None,
        enable_telemetry: bool = True,
    ):
        super().__init__(component_name)
        self.logger = StructuredLogger(f"health.{component_name}")
        self.circuit_breaker = circuit_breaker_registry.get_or_create(
            f"health_{component_name}", circuit_breaker_config or CircuitBreakerConfig()
        )
        self.sla_monitor = get_or_create_sla_monitor(
            component_name, sla_config or SLAConfiguration(service_name=component_name)
        )
        self.metrics_logger = get_metrics_logger(component_name)
        self.telemetry_enabled = enable_telemetry
        if enable_telemetry:
            self.telemetry_context = TelemetryContext(component_name)

    async def check(self) -> HealthResult:
        """Enhanced health check with all 2025 features"""
        start_time = time.time()
        span_context = (
            self.telemetry_context.span("health_check")
            if self.telemetry_enabled
            else None
        )
        if span_context:
            with span_context:
                try:
                    result = await self.circuit_breaker.call(self._execute_health_check)
                except Exception as e:
                    if self.sla_monitor:
                        self.sla_monitor.record_health_check(
                            success=False, response_time_ms=0
                        )
                    from prompt_improver.performance.monitoring.health.base import (
                        HealthResult,
                        HealthStatus,
                    )

                    return HealthResult(
                        status=HealthStatus.FAILED,
                        component=self.name,
                        response_time_ms=0,
                        message=f"Health check failed: {e!s}",
                        error=str(e),
                        details={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                    )
                else:
                    if self.sla_monitor:
                        self.sla_monitor.record_health_check(
                            success=True, response_time_ms=result.response_time_ms or 0
                        )
                    return result
        else:
            try:
                result = await self.circuit_breaker.call(self._execute_health_check)
            except Exception as e:
                if self.sla_monitor:
                    self.sla_monitor.record_health_check(
                        success=False, response_time_ms=0
                    )
                from prompt_improver.performance.monitoring.health.base import (
                    HealthResult,
                    HealthStatus,
                )

                return HealthResult(
                    status=HealthStatus.FAILED,
                    component=self.name,
                    response_time_ms=0,
                    message=f"Health check failed: {e!s}",
                    error=str(e),
                    details={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            else:
                if self.sla_monitor:
                    self.sla_monitor.record_health_check(
                        success=True, response_time_ms=result.response_time_ms or 0
                    )
                return result

    @abstractmethod
    async def _execute_health_check(self) -> HealthResult:
        """Actual health check implementation to be provided by subclasses"""

    def _get_sla_summary(self) -> dict[str, Any]:
        """Get summarized SLA compliance status"""
        report = self.sla_monitor.get_sla_report()
        return {
            "overall_status": report.get("overall_status"),
            "availability": report.get("overall_availability"),
            "breaching_targets": [
                name
                for name, target in report.get("sla_targets", {}).items()
                if target.get("status") == "breaching"
            ],
        }

    @staticmethod
    def _null_context():
        """Null context manager for when telemetry is disabled"""

        class NullContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                pass

        return NullContext()

    def get_enhanced_status(self) -> dict[str, Any]:
        """Get comprehensive status including all monitoring data"""
        return {
            "component": self.name,
            "circuit_breaker": self.circuit_breaker.get_metrics(),
            "sla_report": self.sla_monitor.get_sla_report(),
            "sla_metrics": self.sla_monitor.get_sla_metrics_for_export(),
        }


def create_enhanced_health_checker(
    component_name: str,
    health_check_func,
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    sla_config: SLAConfiguration | None = None,
) -> EnhancedHealthChecker:
    """Factory function to create enhanced health checker with custom implementation"""

    class CustomHealthChecker(EnhancedHealthChecker):
        async def _execute_health_check(self) -> HealthResult:
            return await health_check_func()

    return CustomHealthChecker(
        component_name=component_name,
        circuit_breaker_config=circuit_breaker_config,
        sla_config=sla_config,
    )
