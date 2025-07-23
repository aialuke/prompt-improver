"""
OpenTelemetry Integration for Distributed Tracing
2025 Best Practices for Microservices Observability
"""

import time
from typing import Dict, Optional, Any, Callable
from contextlib import contextmanager
from functools import wraps
import logging

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")

# Semantic conventions
HEALTH_CHECK_SPAN_NAME = "health_check"
HEALTH_CHECK_KIND = "internal"


class TelemetryProvider:
    """
    Manages OpenTelemetry instrumentation for health checks
    """

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        environment: str = "production",
        otlp_endpoint: Optional[str] = None
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.otlp_endpoint = otlp_endpoint or "localhost:4317"

        if OTEL_AVAILABLE:
            self._setup_telemetry()
        else:
            self.tracer = None
            self.meter = None

    def _setup_telemetry(self):
        """Initialize OpenTelemetry providers"""
        # Create resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.environment,
        })

        # Setup tracing
        trace_provider = TracerProvider(resource=resource)

        # Add OTLP exporter for traces
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=self.otlp_endpoint,
            insecure=True  # Use secure=False for local development
        )
        trace_provider.add_span_processor(
            BatchSpanProcessor(otlp_trace_exporter)
        )

        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(self.service_name, self.service_version)

        # Setup metrics
        metric_reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(
                endpoint=self.otlp_endpoint,
                insecure=True
            ),
            export_interval_millis=60000  # Export every minute
        )

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )

        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(self.service_name, self.service_version)

        # Create metrics instruments
        self._create_metrics_instruments()

    def _create_metrics_instruments(self):
        """Create reusable metrics instruments"""
        if not self.meter:
            return

        # Health check duration histogram
        self.health_check_duration = self.meter.create_histogram(
            name="health_check_duration",
            description="Duration of health checks in milliseconds",
            unit="ms"
        )

        # Health check status counter
        self.health_check_counter = self.meter.create_counter(
            name="health_check_total",
            description="Total number of health checks",
            unit="1"
        )

        # Current health status gauge
        self.health_status_gauge = self.meter.create_up_down_counter(
            name="health_status",
            description="Current health status (1=healthy, 0=unhealthy)",
            unit="1"
        )

        # SLA compliance gauge
        self.sla_compliance_gauge = self.meter.create_observable_gauge(
            name="sla_compliance",
            description="SLA compliance ratio (0-1)",
            unit="1",
            callbacks=[]  # Will be set by SLA monitor
        )


# Global telemetry provider
_telemetry_provider: Optional[TelemetryProvider] = None


def init_telemetry(
    service_name: str,
    service_version: str = "1.0.0",
    environment: str = "production",
    otlp_endpoint: Optional[str] = None
) -> TelemetryProvider:
    """Initialize global telemetry provider"""
    global _telemetry_provider
    _telemetry_provider = TelemetryProvider(
        service_name, service_version, environment, otlp_endpoint
    )
    return _telemetry_provider


def get_telemetry() -> Optional[TelemetryProvider]:
    """Get global telemetry provider"""
    return _telemetry_provider


@contextmanager
def health_check_span(
    component_name: str,
    check_type: str = "health_check",
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Context manager for creating health check spans
    """
    provider = get_telemetry()
    if not provider or not provider.tracer:
        yield None
        return

    # Start span
    with provider.tracer.start_as_current_span(
        f"{HEALTH_CHECK_SPAN_NAME}.{component_name}",
        kind=trace.SpanKind.INTERNAL
    ) as span:
        # Add standard attributes
        span.set_attribute("health_check.component", component_name)
        span.set_attribute("health_check.type", check_type)
        span.set_attribute("service.name", provider.service_name)

        # Add correlation ID if available
        try:
            from src.middleware.correlation_context import get_correlation_id
            correlation_id = get_correlation_id()
            if correlation_id:
                span.set_attribute("correlation.id", correlation_id)
        except ImportError:
            pass

        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(f"health_check.{key}", value)

        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            # Success
            span.set_status(Status(StatusCode.OK))


def instrument_health_check(
    component_name: str,
    check_type: str = "health_check"
):
    """
    Decorator to automatically instrument health check methods with OpenTelemetry
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            provider = get_telemetry()

            # Start timing
            start_time = time.time()

            # Create span context
            with health_check_span(
                component_name=component_name,
                check_type=check_type,
                attributes={
                    "class_name": self.__class__.__name__,
                    "method_name": func.__name__
                }
            ) as span:
                try:
                    # Execute health check
                    result = await func(self, *args, **kwargs)

                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000

                    # Add result attributes to span
                    if span and hasattr(result, 'status'):
                        span.set_attribute("health_check.status", result.status.name)
                        span.set_attribute("health_check.healthy",
                                         result.status.name == "HEALTHY")

                    if span and hasattr(result, 'response_time_ms'):
                        span.set_attribute("health_check.response_time_ms",
                                         result.response_time_ms)

                    # Record metrics
                    if provider and provider.meter:
                        # Duration histogram
                        provider.health_check_duration.record(
                            duration_ms,
                            attributes={
                                "component": component_name,
                                "status": getattr(result, 'status', 'unknown').name
                                          if hasattr(result, 'status') else "unknown"
                            }
                        )

                        # Status counter
                        provider.health_check_counter.add(
                            1,
                            attributes={
                                "component": component_name,
                                "status": getattr(result, 'status', 'unknown').name
                                          if hasattr(result, 'status') else "unknown"
                            }
                        )

                        # Update health gauge
                        is_healthy = (hasattr(result, 'status') and
                                    result.status.name == "HEALTHY")
                        provider.health_status_gauge.add(
                            1 if is_healthy else -1,
                            attributes={"component": component_name}
                        )

                    return result

                except Exception as e:
                    # Record failure metrics
                    if provider and provider.meter:
                        provider.health_check_counter.add(
                            1,
                            attributes={
                                "component": component_name,
                                "status": "ERROR"
                            }
                        )

                        provider.health_status_gauge.add(
                            -1,
                            attributes={"component": component_name}
                        )

                    raise

        return wrapper
    return decorator


class TelemetryContext:
    """
    Helper class for managing telemetry context in health checks
    """

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.provider = get_telemetry()
        self._span_stack = []

    @contextmanager
    def span(self, operation: str, **attributes):
        """Create a nested span for sub-operations"""
        if not self.provider or not self.provider.tracer:
            yield None
            return

        with self.provider.tracer.start_as_current_span(
            f"{self.component_name}.{operation}",
            kind=trace.SpanKind.INTERNAL
        ) as span:
            # Add attributes
            span.set_attribute("component", self.component_name)
            span.set_attribute("operation", operation)

            for key, value in attributes.items():
                span.set_attribute(key, value)

            self._span_stack.append(span)
            try:
                yield span
            finally:
                self._span_stack.pop()

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current span"""
        if self._span_stack:
            current_span = self._span_stack[-1]
            current_span.add_event(name, attributes=attributes or {})

    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span"""
        if self._span_stack:
            current_span = self._span_stack[-1]
            current_span.set_attribute(key, value)

    def record_exception(self, exception: Exception):
        """Record exception in current span"""
        if self._span_stack:
            current_span = self._span_stack[-1]
            current_span.record_exception(exception)


# Utility function for manual span creation
def create_health_check_span(
    component_name: str,
    operation: str = "check"
) -> Optional[Any]:
    """Create a span for manual instrumentation"""
    provider = get_telemetry()
    if not provider or not provider.tracer:
        return None

    return provider.tracer.start_as_current_span(
        f"{component_name}.health_check.{operation}",
        kind=trace.SpanKind.INTERNAL
    )