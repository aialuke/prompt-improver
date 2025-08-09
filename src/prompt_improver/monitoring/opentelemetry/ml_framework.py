"""OpenTelemetry ML Monitoring Framework
====================================

Comprehensive ML-specific monitoring using OpenTelemetry standards.
Provides instrumentation for machine learning workflows, model training,
and inference pipelines with distributed tracing and metrics collection.
"""

def create_otel_http_server(port=8080, host='0.0.0.0'):
    """Create OpenTelemetry HTTP metrics server."""
    return OTelHTTPServer(port, host)
