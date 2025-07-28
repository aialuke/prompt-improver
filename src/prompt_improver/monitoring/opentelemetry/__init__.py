"""
OpenTelemetry Integration for Prompt Improver
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
        otlp_endpoint="http://localhost:4317"
    )
    
    # Use in application code
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("business_operation"):
        # Your code here
        pass
"""

from .setup import (
    init_telemetry,
    get_tracer,
    get_meter,
    shutdown_telemetry,
    TelemetryConfig,
)

from .instrumentation import (
    instrument_http,
    instrument_database,
    instrument_redis,
    instrument_ml_pipeline,
    instrument_external_apis,
    trace_ml_operation,
    trace_database_operation,
    trace_cache_operation,
    trace_business_operation,
    instrument_fastapi_app,
)

from .metrics import (
    HttpMetrics,
    DatabaseMetrics,
    MLMetrics,
    BusinessMetrics,
    record_counter,
    record_histogram,
    record_gauge,
)

from .tracing import (
    trace_async,
    trace_sync,
    add_span_attributes,
    record_exception,
    create_span_link,
)

from .context import (
    get_correlation_id,
    set_correlation_id,
    propagate_context,
    inject_context,
    extract_context,
    get_request_id,
    set_request_id,
    get_user_id,
    set_user_id,
    context_scope,
    with_context,
    trace_context_middleware,
)

__all__ = [
    # Core setup
    "init_telemetry",
    "get_tracer",
    "get_meter", 
    "shutdown_telemetry",
    "TelemetryConfig",
    
    # Instrumentation
    "instrument_http",
    "instrument_database",
    "instrument_redis",
    "instrument_ml_pipeline",
    "instrument_external_apis",
    "trace_ml_operation",
    "trace_database_operation",
    "trace_cache_operation",
    "trace_business_operation",
    "instrument_fastapi_app",
    
    # Metrics
    "HttpMetrics",
    "DatabaseMetrics",
    "MLMetrics",
    "BusinessMetrics",
    "record_counter",
    "record_histogram",
    "record_gauge",
    
    # Tracing
    "trace_async",
    "trace_sync",
    "add_span_attributes",
    "record_exception",
    "create_span_link",
    
    # Context
    "get_correlation_id",
    "set_correlation_id",
    "propagate_context",
    "inject_context",
    "extract_context",
    "get_request_id",
    "set_request_id", 
    "get_user_id",
    "set_user_id",
    "context_scope",
    "with_context",
    "trace_context_middleware",
]