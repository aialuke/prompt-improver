"""OpenTelemetry Integration for Prompt Improver
============================================

Comprehensive distributed tracing, metrics, and observability infrastructure
following 2025 best practices for production-ready OpenTelemetry implementation.

This module provides:
- Auto-instrumentation for HTTP, database, and external API calls
- Custom business logic tracing for ML pipelines and prompt processing
- RED (Rate, Errors, Duration) metrics collection
- Context propagation across async boundaries
- Resource-efficient sampling strategies
- Integration with existing monitoring infrastructure

Usage:
    from prompt_improver.monitoring.opentelemetry import init_telemetry, get_tracer

    # Initialize at application startup
    init_telemetry(
        service_name="prompt-improver",
        environment="production",
        otlp_endpoint="http://otel-collector:4317"
    )

    # Use in application code
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("business_operation"):
        # Your code here
        pass
"""

from prompt_improver.monitoring.opentelemetry.context import (
    context_scope,
    extract_context,
    get_correlation_id,
    get_request_id,
    get_user_id,
    inject_context,
    propagate_context,
    set_correlation_id,
    set_request_id,
    set_user_id,
    trace_context_middleware,
    with_context,
)
from prompt_improver.monitoring.opentelemetry.instrumentation import (
    instrument_database,
    instrument_external_apis,
    instrument_fastapi_app,
    instrument_http,
    instrument_ml_pipeline,
    instrument_redis,
    trace_business_operation,
    trace_cache_operation,
    trace_database_operation,
    trace_ml_operation,
)
from prompt_improver.monitoring.opentelemetry.metrics import (
    BusinessMetrics,
    DatabaseMetrics,
    HttpMetrics,
    MLMetrics,
    get_business_metrics,
    get_database_metrics,
    get_http_metrics,
    get_ml_metrics,
    record_counter,
    record_gauge,
    record_histogram,
)
from prompt_improver.monitoring.opentelemetry.setup import (
    TelemetryConfig,
    get_meter,
    get_tracer,
    init_telemetry,
    shutdown_telemetry,
)
from prompt_improver.monitoring.opentelemetry.tracing import (
    add_span_attributes,
    create_span_link,
    record_exception,
    trace_async,
    trace_sync,
)

try:
    from prompt_improver.monitoring.opentelemetry.metrics import (
        MLMetrics,
        get_ml_metrics,
    )
    from prompt_improver.monitoring.opentelemetry.ml_utils import (
        MLMonitoringMixin,
        MLPerformanceTracker,
        create_ml_monitoring_context,
        ml_monitor,
        monitor_classification,
        monitor_failure_analysis,
    )

    ML_FRAMEWORK_AVAILABLE = True
except ImportError:
    ML_FRAMEWORK_AVAILABLE = False
__all__ = [
    "BusinessMetrics",
    "DatabaseMetrics",
    "HttpMetrics",
    "MLMetrics",
    "TelemetryConfig",
    "add_span_attributes",
    "context_scope",
    "create_span_link",
    "extract_context",
    "get_business_metrics",
    "get_correlation_id",
    "get_database_metrics",
    "get_http_metrics",
    "get_meter",
    "get_ml_metrics",
    "get_request_id",
    "get_tracer",
    "get_user_id",
    "init_telemetry",
    "inject_context",
    "instrument_database",
    "instrument_external_apis",
    "instrument_fastapi_app",
    "instrument_http",
    "instrument_ml_pipeline",
    "instrument_redis",
    "propagate_context",
    "record_counter",
    "record_exception",
    "record_gauge",
    "record_histogram",
    "set_correlation_id",
    "set_request_id",
    "set_user_id",
    "shutdown_telemetry",
    "trace_async",
    "trace_business_operation",
    "trace_cache_operation",
    "trace_context_middleware",
    "trace_database_operation",
    "trace_ml_operation",
    "trace_sync",
    "with_context",
]
if ML_FRAMEWORK_AVAILABLE:
    __all__.extend([
        "MLMetrics",
        "MLMonitoringMixin",
        "MLPerformanceTracker",
        "create_ml_monitoring_context",
        "get_ml_metrics",
        "ml_monitor",
        "monitor_classification",
        "monitor_failure_analysis",
    ])
