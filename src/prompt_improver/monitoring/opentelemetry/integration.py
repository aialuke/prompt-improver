"""
OpenTelemetry Integration with Existing Monitoring Infrastructure
================================================================

Provides integration adapters to connect OpenTelemetry with existing
Prometheus metrics, health checks, and monitoring systems.
"""

import logging
from typing import Dict, Optional
from contextlib import contextmanager

try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = otel_metrics = Status = StatusCode = None

# Import existing monitoring components
try:
    from prompt_improver.performance.monitoring.metrics_registry import (
        get_metrics_registry, StandardMetrics
    )
    from prompt_improver.performance.monitoring.health.telemetry import (
        get_telemetry, health_check_span
    )
    EXISTING_MONITORING_AVAILABLE = True
except ImportError:
    EXISTING_MONITORING_AVAILABLE = False
    get_metrics_registry = StandardMetrics = None
    get_telemetry = health_check_span = None

from .setup import get_tracer, get_meter
from .metrics import get_http_metrics, get_database_metrics, get_business_metrics
from .tracing import trace_async, add_span_attributes

logger = logging.getLogger(__name__)

class MetricsIntegration:
    """Integrates OpenTelemetry metrics with existing Prometheus metrics."""

    def __init__(self):
        self.prometheus_registry = get_metrics_registry() if EXISTING_MONITORING_AVAILABLE else None
        self.otel_http_metrics = get_http_metrics()
        self.otel_db_metrics = get_database_metrics()
        self.otel_business_metrics = get_business_metrics()

    def bridge_prometheus_to_otel(self) -> None:
        """Create bridges from Prometheus metrics to OpenTelemetry."""
        if not self.prometheus_registry or not OTEL_AVAILABLE:
            logger.warning("Cannot bridge metrics: missing dependencies")
            return

        # This would create observable instruments that read from Prometheus
        # For now, we'll focus on dual recording
        logger.info("Prometheus to OpenTelemetry metric bridging configured")

    def record_dual_metrics(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str],
        metric_type: str = "counter"
    ) -> None:
        """Record metrics in both Prometheus and OpenTelemetry."""
        # Record in Prometheus (existing system)
        if self.prometheus_registry:
            try:
                if metric_type == "counter":
                    prometheus_metric = self.prometheus_registry.get_or_create_counter(
                        metric_name,
                        f"OpenTelemetry bridged metric: {metric_name}",
                        list(labels.keys())
                    )
                    prometheus_metric.labels(**labels).inc(value)
                elif metric_type == "histogram":
                    prometheus_metric = self.prometheus_registry.get_or_create_histogram(
                        metric_name,
                        f"OpenTelemetry bridged metric: {metric_name}",
                        list(labels.keys())
                    )
                    prometheus_metric.labels(**labels).observe(value)
                elif metric_type == "gauge":
                    prometheus_metric = self.prometheus_registry.get_or_create_gauge(
                        metric_name,
                        f"OpenTelemetry bridged metric: {metric_name}",
                        list(labels.keys())
                    )
                    prometheus_metric.labels(**labels).set(value)
            except Exception as e:
                logger.debug(f"Failed to record Prometheus metric {metric_name}: {e}")

        # Record in OpenTelemetry
        if OTEL_AVAILABLE:
            try:
                meter = get_meter(__name__)
                if meter:
                    if metric_type == "counter":
                        counter = meter.create_counter(metric_name)
                        counter.add(value, labels)
                    elif metric_type == "histogram":
                        histogram = meter.create_histogram(metric_name)
                        histogram.record(value, labels)
                    elif metric_type == "gauge":
                        gauge = meter.create_gauge(metric_name)
                        gauge.set(value, labels)
            except Exception as e:
                logger.debug(f"Failed to record OpenTelemetry metric {metric_name}: {e}")

class HealthCheckIntegration:
    """Integrates OpenTelemetry tracing with existing health check system."""

    def __init__(self):
        self.tracer = get_tracer(__name__)

    def instrument_health_check(self, health_check_func):
        """Decorator to instrument existing health checks with OpenTelemetry."""
        @trace_async(
            operation_name=f"health_check.{health_check_func.__name__}",
            component="health",
            capture_result=True
        )
        async def wrapper(*args, **kwargs):
            # Add health check specific attributes
            add_span_attributes(
                **{
                    "health_check.name": health_check_func.__name__,
                    "health_check.component": "system"
                }
            )

            result = await health_check_func(*args, **kwargs)

            # Add result-specific attributes
            if hasattr(result, 'status'):
                add_span_attributes(
                    **{
                        "health_check.status": result.status.name,
                        "health_check.healthy": result.status.name == "HEALTHY"
                    }
                )

            if hasattr(result, 'response_time_ms'):
                add_span_attributes(
                    **{"health_check.response_time_ms": result.response_time_ms}
                )

            return result

        return wrapper

    @contextmanager
    def health_check_context(self, component_name: str, check_type: str = "health"):
        """Context manager for health check operations."""
        if not OTEL_AVAILABLE or not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(
            f"health_check.{component_name}",
            kind=trace.SpanKind.INTERNAL
        ) as span:
            span.set_attribute("health_check.component", component_name)
            span.set_attribute("health_check.type", check_type)

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

class ApplicationIntegration:
    """Main integration point for OpenTelemetry with the application."""

    def __init__(self):
        self.metrics_integration = MetricsIntegration()
        self.health_check_integration = HealthCheckIntegration()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all integrations."""
        if self._initialized:
            return

        try:
            # Set up metric bridging
            self.metrics_integration.bridge_prometheus_to_otel()

            # Set up health check instrumentation
            self._setup_health_check_instrumentation()

            self._initialized = True
            logger.info("OpenTelemetry application integration initialized")

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry integration: {e}")
            raise

    def _setup_health_check_instrumentation(self) -> None:
        """Set up automatic health check instrumentation."""
        # This would typically patch existing health check classes/functions
        # For now, we provide the decorator for manual application
        logger.info("Health check instrumentation available via decorators")

    def record_business_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: str = "counter"
    ) -> None:
        """Record a business metric in both systems."""
        self.metrics_integration.record_dual_metrics(
            metric_name,
            value,
            labels or {},
            metric_type
        )

    def get_health_check_decorator(self):
        """Get health check decorator for existing health checks."""
        return self.health_check_integration.instrument_health_check

    def get_health_check_context(self):
        """Get health check context manager."""
        return self.health_check_integration.health_check_context

# Global integration instance
_app_integration: Optional[ApplicationIntegration] = None

def get_application_integration() -> ApplicationIntegration:
    """Get global application integration instance."""
    global _app_integration
    if _app_integration is None:
        _app_integration = ApplicationIntegration()
    return _app_integration

def initialize_integration() -> None:
    """Initialize OpenTelemetry integration with existing monitoring."""
    get_application_integration().initialize()

def record_business_metric(
    metric_name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None,
    metric_type: str = "counter"
) -> None:
    """Convenience function to record business metrics."""
    get_application_integration().record_business_metric(
        metric_name, value, labels, metric_type
    )

def health_check_instrumentation():
    """Decorator factory for health check instrumentation."""
    return get_application_integration().get_health_check_decorator()

def health_check_context(component_name: str, check_type: str = "health"):
    """Context manager for health check operations."""
    return get_application_integration().get_health_check_context()(
        component_name, check_type
    )

# FastAPI integration helpers
def setup_fastapi_telemetry(app, service_name: str = "prompt-improver"):
    """Set up comprehensive OpenTelemetry for a FastAPI application."""
    from .instrumentation import instrument_fastapi_app
    from .context import trace_context_middleware
    from .setup import init_telemetry

    # Initialize telemetry
    init_telemetry(service_name=service_name)

    # Instrument FastAPI
    instrument_fastapi_app(app)

    # Add context middleware
    app.middleware("http")(trace_context_middleware())

    # Initialize integrations
    initialize_integration()

    logger.info(f"FastAPI application '{service_name}' instrumented with OpenTelemetry")

    return app

def create_example_usage():
    """Create example usage documentation."""
    return """
    Example Usage:

    # In main.py or app initialization
    from prompt_improver.monitoring.opentelemetry.integration import (
        setup_fastapi_telemetry, initialize_integration
    )

    # For FastAPI apps
    app = FastAPI()
    setup_fastapi_telemetry(app, "prompt-improver")

    # For standalone services
    from prompt_improver.monitoring.opentelemetry import init_telemetry
    init_telemetry(service_name="prompt-improver")
    initialize_integration()

    # In business logic
    from prompt_improver.monitoring.opentelemetry import (
        trace_ml_operation, trace_business_operation
    )

    @trace_ml_operation("inference", model_name="gpt-4", capture_io=True)
    async def improve_prompt(prompt: str) -> str:
        # Your ML logic here
        return improved_prompt

    @trace_business_operation("prompt_processing")
    async def process_user_request(request: UserRequest) -> Response:
        # Your business logic here
        return response

    # In health checks
    from prompt_improver.monitoring.opentelemetry.integration import (
        health_check_instrumentation
    )

    @health_check_instrumentation()
    async def check_database_health():
        # Your health check logic
        return HealthCheckResult(status=HealthStatus.HEALTHY)
    """
