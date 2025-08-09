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
        otlp_endpoint="http://localhost:4317"
    )

    # Use in application code
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("business_operation"):
        # Your code here
        pass
"""
from prompt_improver.monitoring.opentelemetry.context import context_scope, extract_context, get_correlation_id, get_request_id, get_user_id, inject_context, propagate_context, set_correlation_id, set_request_id, set_user_id, trace_context_middleware, with_context
from prompt_improver.monitoring.opentelemetry.instrumentation import instrument_database, instrument_external_apis, instrument_fastapi_app, instrument_http, instrument_ml_pipeline, instrument_redis, trace_business_operation, trace_cache_operation, trace_database_operation, trace_ml_operation
from prompt_improver.monitoring.opentelemetry.metrics import BusinessMetrics, DatabaseMetrics, HttpMetrics, MLMetrics, get_business_metrics, get_database_metrics, get_http_metrics, get_ml_metrics, record_counter, record_gauge, record_histogram
from prompt_improver.monitoring.opentelemetry.setup import TelemetryConfig, get_meter, get_tracer, init_telemetry, shutdown_telemetry
from prompt_improver.monitoring.opentelemetry.tracing import add_span_attributes, create_span_link, record_exception, trace_async, trace_sync
try:
    from prompt_improver.monitoring.opentelemetry.metrics import MLMetrics, get_ml_metrics
    from prompt_improver.monitoring.opentelemetry.ml_utils import MLMonitoringMixin, MLPerformanceTracker, create_ml_monitoring_context, ml_monitor, monitor_classification, monitor_failure_analysis
    ML_FRAMEWORK_AVAILABLE = True
except ImportError:
    ML_FRAMEWORK_AVAILABLE = False
__all__ = ['init_telemetry', 'get_tracer', 'get_meter', 'shutdown_telemetry', 'TelemetryConfig', 'instrument_http', 'instrument_database', 'instrument_redis', 'instrument_ml_pipeline', 'instrument_external_apis', 'trace_ml_operation', 'trace_database_operation', 'trace_cache_operation', 'trace_business_operation', 'instrument_fastapi_app', 'HttpMetrics', 'DatabaseMetrics', 'MLMetrics', 'BusinessMetrics', 'get_http_metrics', 'get_database_metrics', 'get_ml_metrics', 'get_business_metrics', 'record_counter', 'record_histogram', 'record_gauge', 'trace_async', 'trace_sync', 'add_span_attributes', 'record_exception', 'create_span_link', 'get_correlation_id', 'set_correlation_id', 'propagate_context', 'inject_context', 'extract_context', 'get_request_id', 'set_request_id', 'get_user_id', 'set_user_id', 'context_scope', 'with_context', 'trace_context_middleware']
if ML_FRAMEWORK_AVAILABLE:
    __all__.extend(['get_ml_metrics', 'MLMetrics', 'MLMonitoringMixin', 'ml_monitor', 'create_ml_monitoring_context', 'MLPerformanceTracker', 'monitor_failure_analysis', 'monitor_classification'])
