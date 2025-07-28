"""
Enhanced Base Health Checker with 2025 features
Integrates circuit breaker, SLA monitoring, structured logging, and OpenTelemetry
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import time
import logging

from .base import HealthChecker, HealthResult
from .circuit_breaker import circuit_breaker_registry, CircuitBreakerConfig
from .structured_logging import StructuredLogger, get_metrics_logger
from .sla_monitor import get_or_create_sla_monitor, SLAConfiguration
from .telemetry import TelemetryContext

logger = logging.getLogger(__name__)

class EnhancedHealthChecker(HealthChecker):
    """
    Enhanced health checker with 2025 observability features
    """

    def __init__(
        self,
        component_name: str,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        sla_config: Optional[SLAConfiguration] = None,
        enable_telemetry: bool = True
    ):
        super().__init__(component_name)

        # Initialize structured logger
        self.logger = StructuredLogger(f"health.{component_name}")

        # Initialize circuit breaker
        self.circuit_breaker = circuit_breaker_registry.get_or_create(
            f"health_{component_name}",
            circuit_breaker_config or CircuitBreakerConfig()
        )

        # Initialize SLA monitor
        self.sla_monitor = get_or_create_sla_monitor(
            component_name,
            sla_config or SLAConfiguration(service_name=component_name)
        )

        # Initialize metrics logger
        self.metrics_logger = get_metrics_logger(component_name)

        # Telemetry context
        self.telemetry_enabled = enable_telemetry
        if enable_telemetry:
            self.telemetry_context = TelemetryContext(component_name)

    async def check(self) -> HealthResult:
        """
        Enhanced health check with all 2025 features
        """
        start_time = time.time()

        # Create telemetry span context
        span_context = (
            self.telemetry_context.span("health_check")
            if self.telemetry_enabled else None
        )

        # Handle telemetry context (convert sync context manager to async compatible)
        if span_context:
            # Use the sync context manager directly but handle it properly
            with span_context:
                try:
                    # Execute with circuit breaker protection
                    result = await self.circuit_breaker.call(
                        self._execute_health_check
                    )
                except Exception as e:
                    # Record failure in SLA monitoring
                    if self.sla_monitor:
                        self.sla_monitor.record_health_check(
                            success=False,
                            response_time_ms=0
                        )
                    # Return failed health result instead of raising
                    from prompt_improver.performance.monitoring.health.base import HealthResult, HealthStatus
                    return HealthResult(
                        status=HealthStatus.FAILED,
                        component=self.name,
                        response_time_ms=0,
                        message=f"Health check failed: {str(e)}",
                        error=str(e),
                        details={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    )
                else:
                    # Record success in SLA monitoring
                    if self.sla_monitor:
                        self.sla_monitor.record_health_check(
                            success=True,
                            response_time_ms=result.response_time_ms or 0
                        )
                    return result
        else:
            # No telemetry context
            try:
                # Execute with circuit breaker protection
                result = await self.circuit_breaker.call(
                    self._execute_health_check
                )
            except Exception as e:
                # Record failure in SLA monitoring
                if self.sla_monitor:
                    self.sla_monitor.record_health_check(
                        success=False,
                        response_time_ms=0
                    )
                # Return failed health result instead of raising
                from prompt_improver.performance.monitoring.health.base import HealthResult, HealthStatus
                return HealthResult(
                    status=HealthStatus.FAILED,
                    component=self.name,
                    response_time_ms=0,
                    message=f"Health check failed: {str(e)}",
                    error=str(e),
                    details={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
            else:
                # Record success in SLA monitoring
                if self.sla_monitor:
                    self.sla_monitor.record_health_check(
                        success=True,
                        response_time_ms=result.response_time_ms or 0
                    )
                return result

    @abstractmethod
    async def _execute_health_check(self) -> HealthResult:
        """
        Actual health check implementation to be provided by subclasses
        """
        pass

    def _get_sla_summary(self) -> Dict[str, Any]:
        """Get summarized SLA compliance status"""
        report = self.sla_monitor.get_sla_report()
        return {
            "overall_status": report.get("overall_status"),
            "availability": report.get("overall_availability"),
            "breaching_targets": [
                name for name, target in report.get("sla_targets", {}).items()
                if target.get("status") == "breaching"
            ]
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

    def get_enhanced_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status including all monitoring data
        """
        return {
            "component": self.name,
            "circuit_breaker": self.circuit_breaker.get_metrics(),
            "sla_report": self.sla_monitor.get_sla_report(),
            "sla_metrics": self.sla_monitor.get_sla_metrics_for_export()
        }

def create_enhanced_health_checker(
    component_name: str,
    health_check_func,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    sla_config: Optional[SLAConfiguration] = None
) -> EnhancedHealthChecker:
    """
    Factory function to create enhanced health checker with custom implementation
    """
    class CustomHealthChecker(EnhancedHealthChecker):
        async def _execute_health_check(self) -> HealthResult:
            return await health_check_func()

    return CustomHealthChecker(
        component_name=component_name,
        circuit_breaker_config=circuit_breaker_config,
        sla_config=sla_config
    )
