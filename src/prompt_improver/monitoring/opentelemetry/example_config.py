"""
OpenTelemetry Configuration Examples
===================================

Provides example configurations for different deployment environments
and common use cases.
"""

import os
from .setup import TelemetryConfig, ExporterType, SamplingStrategy


def get_development_config() -> TelemetryConfig:
    """Configuration for local development."""
    return TelemetryConfig(
        service_name="prompt-improver-dev",
        service_version="1.0.0-dev",
        environment="development",
        
        # Local OTLP collector
        otlp_endpoint_grpc="http://localhost:4317",
        otlp_endpoint_http="http://localhost:4318",
        otlp_insecure=True,
        
        # Console output for debugging
        trace_exporter=ExporterType.CONSOLE,
        metric_exporter=ExporterType.CONSOLE,
        
        # High sampling for development
        sampling_strategy=SamplingStrategy.ALWAYS_ON,
        sampling_rate=1.0,
        
        # Fast export for immediate feedback
        export_timeout_millis=5000,
        schedule_delay_millis=1000,
        metric_export_interval_millis=10000,  # 10 seconds
        
        # Enable all features
        enable_tracing=True,
        enable_metrics=True,
        enable_logging=True,
        enable_auto_instrumentation=True,
        
        # Development-specific attributes
        resource_attributes={
            "development.mode": True,
            "developer.name": os.getenv("USER", "unknown")
        }
    )


def get_staging_config() -> TelemetryConfig:
    """Configuration for staging environment."""
    return TelemetryConfig(
        service_name="prompt-improver-staging",
        service_version=os.getenv("APP_VERSION", "1.0.0"),
        environment="staging",
        
        # Staging OTLP collector
        otlp_endpoint_grpc=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
        otlp_endpoint_http=os.getenv("OTEL_EXPORTER_OTLP_HTTP_ENDPOINT", "http://otel-collector:4318"),
        otlp_insecure=False,
        
        # OTLP export to observability platform
        trace_exporter=ExporterType.OTLP_GRPC,
        metric_exporter=ExporterType.OTLP_GRPC,
        
        # Moderate sampling for staging
        sampling_strategy=SamplingStrategy.PARENT_BASED,
        sampling_rate=0.5,  # 50% sampling
        
        # Balanced performance
        export_timeout_millis=30000,
        schedule_delay_millis=5000,
        metric_export_interval_millis=30000,  # 30 seconds
        
        # All features enabled
        enable_tracing=True,
        enable_metrics=True,
        enable_logging=True,
        enable_auto_instrumentation=True,
        
        # Staging-specific attributes
        resource_attributes={
            "deployment.environment": "staging",
            "k8s.cluster.name": os.getenv("K8S_CLUSTER_NAME", "staging-cluster"),
            "k8s.namespace.name": os.getenv("K8S_NAMESPACE", "prompt-improver-staging")
        }
    )


def get_production_config() -> TelemetryConfig:
    """Configuration for production environment."""
    return TelemetryConfig(
        service_name="prompt-improver",
        service_version=os.getenv("APP_VERSION", "1.0.0"),
        environment="production",
        
        # Production OTLP collector with authentication
        otlp_endpoint_grpc=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://api.honeycomb.io:443"),
        otlp_endpoint_http=os.getenv("OTEL_EXPORTER_OTLP_HTTP_ENDPOINT", "https://api.honeycomb.io"),
        otlp_headers={
            "x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY", ""),
            "x-honeycomb-dataset": "prompt-improver-prod"
        },
        otlp_insecure=False,
        
        # Secure OTLP export
        trace_exporter=ExporterType.OTLP_GRPC,
        metric_exporter=ExporterType.OTLP_GRPC,
        
        # Conservative sampling for production
        sampling_strategy=SamplingStrategy.PARENT_BASED,
        sampling_rate=0.1,  # 10% sampling
        rate_limit_per_second=100,
        
        # Production performance tuning
        max_export_batch_size=512,
        export_timeout_millis=30000,
        schedule_delay_millis=5000,
        max_queue_size=2048,
        metric_export_interval_millis=60000,  # 1 minute
        metric_export_timeout_millis=30000,
        
        # All features enabled
        enable_tracing=True,
        enable_metrics=True,
        enable_logging=True,
        enable_auto_instrumentation=True,
        
        # Production-specific attributes
        resource_attributes={
            "deployment.environment": "production",
            "service.namespace": "prompt-improver",
            "k8s.cluster.name": os.getenv("K8S_CLUSTER_NAME", "prod-cluster"),
            "k8s.namespace.name": os.getenv("K8S_NAMESPACE", "prompt-improver"),
            "k8s.pod.name": os.getenv("K8S_POD_NAME", ""),
            "k8s.node.name": os.getenv("K8S_NODE_NAME", ""),
            "cloud.provider": os.getenv("CLOUD_PROVIDER", "aws"),
            "cloud.region": os.getenv("AWS_REGION", "us-west-2")
        }
    )


def get_testing_config() -> TelemetryConfig:
    """Configuration for automated testing."""
    return TelemetryConfig(
        service_name="prompt-improver-test",
        service_version="test",
        environment="test",
        
        # No external exports during testing
        trace_exporter=ExporterType.NONE,
        metric_exporter=ExporterType.NONE,
        
        # Minimal sampling
        sampling_strategy=SamplingStrategy.ALWAYS_OFF,
        sampling_rate=0.0,
        
        # Fast export settings (though not used)
        export_timeout_millis=1000,
        schedule_delay_millis=100,
        metric_export_interval_millis=1000,
        
        # Disable features that might interfere with tests
        enable_tracing=False,
        enable_metrics=False,
        enable_logging=False,
        enable_auto_instrumentation=False,
        
        # Test-specific attributes
        resource_attributes={
            "test.mode": True,
            "test.framework": "pytest"
        }
    )


def get_observability_config() -> TelemetryConfig:
    """Configuration optimized for maximum observability (development/debugging)."""
    return TelemetryConfig(
        service_name="prompt-improver-debug",
        service_version="debug",
        environment="debug",
        
        # Local collectors for detailed analysis
        otlp_endpoint_grpc="http://localhost:4317",
        otlp_endpoint_http="http://localhost:4318", 
        otlp_insecure=True,
        
        # Export to both console and OTLP for analysis
        trace_exporter=ExporterType.OTLP_GRPC,
        metric_exporter=ExporterType.OTLP_GRPC,
        
        # Maximum sampling for complete visibility
        sampling_strategy=SamplingStrategy.ALWAYS_ON,
        sampling_rate=1.0,
        
        # Aggressive export for real-time analysis
        max_export_batch_size=128,  # Smaller batches
        export_timeout_millis=5000,
        schedule_delay_millis=500,   # Very frequent exports
        max_queue_size=1024,
        metric_export_interval_millis=5000,  # 5 seconds
        
        # All features enabled
        enable_tracing=True,
        enable_metrics=True,
        enable_logging=True,
        enable_auto_instrumentation=True,
        
        # Debug-specific attributes
        resource_attributes={
            "debug.mode": True,
            "observability.level": "maximum",
            "sampling.debug": True
        }
    )


def get_minimal_config() -> TelemetryConfig:
    """Minimal configuration with basic tracing only."""
    return TelemetryConfig(
        service_name="prompt-improver-minimal",
        service_version="1.0.0",
        environment="minimal",
        
        # Basic OTLP export
        otlp_endpoint_grpc="http://localhost:4317",
        otlp_insecure=True,
        
        # Only traces
        trace_exporter=ExporterType.OTLP_GRPC,
        metric_exporter=ExporterType.NONE,
        
        # Minimal sampling
        sampling_strategy=SamplingStrategy.PROBABILISTIC,
        sampling_rate=0.01,  # 1% sampling
        
        # Basic export settings
        export_timeout_millis=30000,
        schedule_delay_millis=5000,
        
        # Only essential features
        enable_tracing=True,
        enable_metrics=False,
        enable_logging=False,
        enable_auto_instrumentation=True,
        
        resource_attributes={}
    )


# Configuration factory
def get_config_for_environment(env: str = None) -> TelemetryConfig:
    """Get configuration based on environment variable or parameter."""
    env = env or os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": get_development_config,
        "dev": get_development_config,
        "staging": get_staging_config,
        "stage": get_staging_config,
        "production": get_production_config,
        "prod": get_production_config,
        "test": get_testing_config,
        "testing": get_testing_config,
        "debug": get_observability_config,
        "observability": get_observability_config,
        "minimal": get_minimal_config
    }
    
    config_func = config_map.get(env, get_development_config)
    return config_func()


# Usage examples
def create_usage_examples():
    """Create comprehensive usage examples."""
    return """
    OpenTelemetry Configuration Examples:
    
    # 1. Environment-based configuration
    from prompt_improver.monitoring.opentelemetry import init_telemetry
    from prompt_improver.monitoring.opentelemetry.example_config import get_config_for_environment
    
    config = get_config_for_environment()  # Uses ENVIRONMENT env var
    init_telemetry(config=config)
    
    # 2. Explicit environment configuration
    config = get_config_for_environment("production")
    init_telemetry(config=config)
    
    # 3. Custom configuration
    from prompt_improver.monitoring.opentelemetry.setup import TelemetryConfig, ExporterType
    
    custom_config = TelemetryConfig(
        service_name="my-service",
        environment="custom",
        otlp_endpoint_grpc="https://my-collector.com:4317",
        trace_exporter=ExporterType.OTLP_GRPC,
        sampling_rate=0.25
    )
    init_telemetry(config=custom_config)
    
    # 4. FastAPI integration
    from fastapi import FastAPI
    from prompt_improver.monitoring.opentelemetry.integration import setup_fastapi_telemetry
    
    app = FastAPI()
    setup_fastapi_telemetry(app, "my-fastapi-service")
    
    # 5. Manual instrumentation
    from prompt_improver.monitoring.opentelemetry import (
        trace_async, trace_ml_operation, get_correlation_id
    )
    
    @trace_ml_operation("text_analysis", model_name="bert", capture_io=True)
    async def analyze_text(text: str) -> dict:
        # Your ML logic
        correlation_id = get_correlation_id()  # For logging
        return {"sentiment": "positive", "correlation_id": correlation_id}
    
    @trace_async("business_logic", component="prompt_processor")
    async def process_prompt(prompt: str) -> str:
        # Your business logic
        analyzed = await analyze_text(prompt)
        return f"Processed: {prompt}"
    
    # 6. Health check instrumentation
    from prompt_improver.monitoring.opentelemetry.integration import health_check_instrumentation
    
    @health_check_instrumentation()
    async def check_database():
        # Your health check logic
        return {"status": "healthy", "response_time": 5.2}
    
    # 7. Context propagation
    from prompt_improver.monitoring.opentelemetry import (
        with_context, propagate_context, set_user_id
    )
    
    @with_context(user_id="user123", session_id="session456")
    async def user_operation():
        # This operation will have user context
        pass
    
    # 8. Manual context management
    from prompt_improver.monitoring.opentelemetry import context_scope
    
    async def handle_request(user_id: str):
        with context_scope(user_id=user_id, request_id="req123"):
            # All operations in this scope have context
            await process_prompt("Hello world")
    
    # 9. Metrics recording
    from prompt_improver.monitoring.opentelemetry import record_counter, record_histogram
    
    record_counter("user_requests_total", labels={"user_type": "premium"})
    record_histogram("request_duration_ms", 123.45, labels={"endpoint": "/api/improve"})
    
    # 10. Integration with existing Prometheus metrics
    from prompt_improver.monitoring.opentelemetry.integration import record_business_metric
    
    record_business_metric(
        "prompt_improvements_total",
        1.0,
        labels={"improvement_type": "clarity", "user_tier": "premium"},
        metric_type="counter"
    )
    """
