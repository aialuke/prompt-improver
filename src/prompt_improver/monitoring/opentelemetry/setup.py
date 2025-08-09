"""Core OpenTelemetry Setup and Configuration Module
================================================

Production-ready OpenTelemetry SDK initialization with:
- Async-compatible configuration
- Resource-efficient sampling strategies
- OTLP exporters for traces, metrics, and logs
- Environment-based configuration
- Graceful fallback when telemetry is unavailable
"""
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Union
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as OTLPMetricExporterHTTP
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider, sampling
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
    ResourceAttributes = None
    try:
        from opentelemetry.semconv.resource import ResourceAttributes
    except ImportError:
        try:
            from opentelemetry.semantic_conventions.resource import ResourceAttributes
        except ImportError:

            class _ResourceAttributesFallback:
                DEPLOYMENT_ENVIRONMENT = 'deployment.environment'
                SERVICE_NAMESPACE = 'service.namespace'
                SERVICE_INSTANCE_ID = 'service.instance.id'
            ResourceAttributes = _ResourceAttributesFallback()
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None
    ResourceAttributes = None
logger = logging.getLogger(__name__)

class _NoOpSpan:

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

class _NoOpInstrument:

    def add(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

class _NoOpTracer:

    def start_span(self, *args, **kwargs):
        return _NoOpSpan()

    def start_as_current_span(self, *args, **kwargs):
        return _NoOpSpan()

class _NoOpMeter:

    def create_counter(self, *args, **kwargs):
        return _NoOpInstrument()

    def create_histogram(self, *args, **kwargs):
        return _NoOpInstrument()

    def create_gauge(self, *args, **kwargs):
        return _NoOpInstrument()

class ExporterType(Enum):
    """Supported exporter types."""
    OTLP_GRPC = 'otlp_grpc'
    OTLP_HTTP = 'otlp_http'
    CONSOLE = 'console'
    NONE = 'none'

class SamplingStrategy(Enum):
    """Trace sampling strategies for performance optimization."""
    ALWAYS_ON = 'always_on'
    ALWAYS_OFF = 'always_off'
    PROBABILISTIC = 'probabilistic'
    RATE_LIMITING = 'rate_limiting'
    PARENT_BASED = 'parent_based'

@dataclass
class TelemetryConfig:
    """OpenTelemetry configuration with production defaults."""
    service_name: str = 'prompt-improver'
    service_version: str = '1.0.0'
    environment: str = 'production'
    otlp_endpoint_grpc: str | None = None
    otlp_endpoint_http: str | None = None
    otlp_headers: dict[str, str] = field(default_factory=dict)
    otlp_insecure: bool = True
    trace_exporter: ExporterType = ExporterType.OTLP_GRPC
    metric_exporter: ExporterType = ExporterType.OTLP_GRPC
    sampling_strategy: SamplingStrategy = SamplingStrategy.PARENT_BASED
    sampling_rate: float = 0.1
    rate_limit_per_second: int = 100
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000
    schedule_delay_millis: int = 5000
    max_queue_size: int = 2048
    metric_export_interval_millis: int = 60000
    metric_export_timeout_millis: int = 30000
    resource_attributes: dict[str, str | int | float | bool] = field(default_factory=dict)
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_auto_instrumentation: bool = True

    @classmethod
    def from_environment(cls) -> 'TelemetryConfig':
        """Create configuration from environment variables with LGTM stack defaults."""
        default_headers = {'X-JSONB-Compatible': 'true', 'X-APES-Service': 'prompt-improver', 'X-LGTM-Integration': 'enabled'}
        return cls(service_name=os.getenv('OTEL_SERVICE_NAME', 'apes-prompt-improver'), service_version=os.getenv('OTEL_SERVICE_VERSION', '1.0.0'), environment=os.getenv('OTEL_ENVIRONMENT', 'development'), otlp_endpoint_grpc=os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317'), otlp_endpoint_http=os.getenv('OTEL_EXPORTER_OTLP_HTTP_ENDPOINT', 'http://localhost:4318'), otlp_insecure=os.getenv('OTEL_EXPORTER_OTLP_INSECURE', 'true').lower() == 'true', otlp_headers=default_headers, trace_exporter=ExporterType(os.getenv('OTEL_TRACE_EXPORTER', 'console')), metric_exporter=ExporterType(os.getenv('OTEL_METRIC_EXPORTER', 'console')), sampling_strategy=SamplingStrategy(os.getenv('OTEL_SAMPLING_STRATEGY', 'parent_based')), sampling_rate=float(os.getenv('OTEL_SAMPLING_RATE', '0.1')), rate_limit_per_second=int(os.getenv('OTEL_RATE_LIMIT_PER_SECOND', '100')), enable_tracing=os.getenv('OTEL_TRACING_ENABLED', 'true').lower() == 'true', enable_metrics=os.getenv('OTEL_METRICS_ENABLED', 'true').lower() == 'true', enable_logging=os.getenv('OTEL_LOGGING_ENABLED', 'true').lower() == 'true', enable_auto_instrumentation=os.getenv('OTEL_AUTO_INSTRUMENTATION_ENABLED', 'true').lower() == 'true')

class TelemetryManager:
    """Production-ready OpenTelemetry manager with async support."""

    def __init__(self, config: TelemetryConfig | None=None):
        self.config = config or TelemetryConfig.from_environment()
        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._initialized = False
        if not OTEL_AVAILABLE:
            logger.warning('OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp')

    def initialize(self) -> None:
        """Initialize OpenTelemetry providers and exporters."""
        if not OTEL_AVAILABLE or self._initialized:
            return
        try:
            self._setup_resource()
            if self.config.enable_tracing:
                self._setup_tracing()
            if self.config.enable_metrics:
                self._setup_metrics()
            self._initialized = True
            logger.info('OpenTelemetry initialized successfully')
        except Exception as e:
            logger.error('Failed to initialize OpenTelemetry: %s', e)
            raise

    def _setup_resource(self):
        """Create OpenTelemetry resource with service metadata."""
        attributes = {SERVICE_NAME: self.config.service_name, SERVICE_VERSION: self.config.service_version, **self.config.resource_attributes}
        if ResourceAttributes:
            attributes.update({ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment, ResourceAttributes.SERVICE_NAMESPACE: 'prompt-improver', ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv('HOSTNAME', 'unknown')})
        else:
            attributes.update({'deployment.environment': self.config.environment, 'service.namespace': 'prompt-improver', 'service.instance.id': os.getenv('HOSTNAME', 'unknown')})
        if OTEL_AVAILABLE:
            self._resource = Resource.create(attributes)
        else:
            self._resource = None
        return self._resource

    def _setup_tracing(self) -> None:
        """Configure tracing with sampling and OTLP export."""
        sampler = self._create_sampler()
        self._tracer_provider = TracerProvider(resource=self._resource, sampler=sampler)
        if self.config.trace_exporter != ExporterType.NONE:
            exporter = self._create_trace_exporter()
            processor = BatchSpanProcessor(exporter, max_export_batch_size=self.config.max_export_batch_size, export_timeout_millis=self.config.export_timeout_millis, schedule_delay_millis=self.config.schedule_delay_millis, max_queue_size=self.config.max_queue_size)
            self._tracer_provider.add_span_processor(processor)
        if OTEL_AVAILABLE and trace:
            trace.set_tracer_provider(self._tracer_provider)

    def _setup_metrics(self) -> None:
        """Configure metrics collection and export."""
        readers = []
        if self.config.metric_exporter != ExporterType.NONE:
            exporter = self._create_metric_exporter()
            reader = PeriodicExportingMetricReader(exporter=exporter, export_interval_millis=self.config.metric_export_interval_millis, export_timeout_millis=self.config.metric_export_timeout_millis)
            readers.append(reader)
        self._meter_provider = MeterProvider(resource=self._resource, metric_readers=readers)
        if OTEL_AVAILABLE and metrics:
            metrics.set_meter_provider(self._meter_provider)

    def _create_sampler(self):
        """Create appropriate sampler based on configuration."""
        if not OTEL_AVAILABLE:
            return None
        strategy = self.config.sampling_strategy
        if strategy == SamplingStrategy.ALWAYS_ON:
            return sampling.ALWAYS_ON
        if strategy == SamplingStrategy.ALWAYS_OFF:
            return sampling.ALWAYS_OFF
        if strategy == SamplingStrategy.PROBABILISTIC:
            return sampling.TraceIdRatioBased(self.config.sampling_rate)
        if strategy == SamplingStrategy.RATE_LIMITING:
            logger.warning('Rate limiting sampler not available, using probabilistic')
            return sampling.TraceIdRatioBased(self.config.sampling_rate)
        if strategy == SamplingStrategy.PARENT_BASED:
            return sampling.ParentBased(root=sampling.TraceIdRatioBased(self.config.sampling_rate))
        logger.warning('Unknown sampling strategy: %s, using parent-based', strategy)
        return sampling.ParentBased(root=sampling.TraceIdRatioBased(self.config.sampling_rate))

    def _create_trace_exporter(self):
        """Create trace exporter based on configuration."""
        exporter_type = self.config.trace_exporter
        if exporter_type == ExporterType.OTLP_GRPC:
            return OTLPSpanExporter(endpoint=self.config.otlp_endpoint_grpc, headers=self.config.otlp_headers, insecure=self.config.otlp_insecure)
        if exporter_type == ExporterType.OTLP_HTTP:
            return OTLPSpanExporterHTTP(endpoint=self.config.otlp_endpoint_http, headers=self.config.otlp_headers)
        if exporter_type == ExporterType.CONSOLE:
            return ConsoleSpanExporter()
        raise ValueError(f'Unsupported trace exporter: {exporter_type}')

    def _create_metric_exporter(self):
        """Create metric exporter based on configuration."""
        exporter_type = self.config.metric_exporter
        if exporter_type == ExporterType.OTLP_GRPC:
            return OTLPMetricExporter(endpoint=self.config.otlp_endpoint_grpc, headers=self.config.otlp_headers, insecure=self.config.otlp_insecure)
        if exporter_type == ExporterType.OTLP_HTTP:
            return OTLPMetricExporterHTTP(endpoint=self.config.otlp_endpoint_http, headers=self.config.otlp_headers)
        if exporter_type == ExporterType.CONSOLE:
            return ConsoleMetricExporter()
        raise ValueError(f'Unsupported metric exporter: {exporter_type}')

    def get_tracer(self, name: str, version: str | None=None):
        """Get a tracer instance."""
        if not OTEL_AVAILABLE or not self._initialized:
            return _NoOpTracer()
        if trace:
            return trace.get_tracer(name, version)
        return _NoOpTracer()

    def get_meter(self, name: str, version: str | None=None):
        """Get a meter instance."""
        if not OTEL_AVAILABLE or not self._initialized:
            return _NoOpMeter()
        if metrics:
            return metrics.get_meter(name, version or '1.0.0')
        return _NoOpMeter()

    def shutdown(self, timeout_millis: int=30000) -> None:
        """Gracefully shutdown telemetry providers."""
        if not self._initialized:
            return
        try:
            if self._tracer_provider:
                self._tracer_provider.shutdown()
            if self._meter_provider:
                self._meter_provider.shutdown(timeout_millis / 1000)
            logger.info('OpenTelemetry shutdown completed')
        except Exception as e:
            logger.error('Error during OpenTelemetry shutdown: %s', e)
        finally:
            self._initialized = False
_telemetry_manager: TelemetryManager | None = None

def init_telemetry(service_name: str='prompt-improver', service_version: str='1.0.0', environment: str='production', otlp_endpoint: str | None=None, config: TelemetryConfig | None=None) -> TelemetryManager:
    """Initialize global OpenTelemetry telemetry.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Deployment environment (dev/staging/production)
        otlp_endpoint: OTLP collector endpoint (if not using config)
        config: Complete telemetry configuration

    Returns:
        TelemetryManager instance
    """
    global _telemetry_manager
    if config is None:
        config = TelemetryConfig.from_environment()
        config.service_name = service_name
        config.service_version = service_version
        config.environment = environment
        if otlp_endpoint:
            config.otlp_endpoint_grpc = otlp_endpoint
    _telemetry_manager = TelemetryManager(config)
    _telemetry_manager.initialize()
    return _telemetry_manager

def get_tracer(name: str, version: str | None=None):
    """Get a tracer instance from the global telemetry manager."""
    if _telemetry_manager:
        return _telemetry_manager.get_tracer(name, version)
    return _NoOpTracer() if not OTEL_AVAILABLE else None

def get_meter(name: str, version: str | None=None):
    """Get a meter instance from the global telemetry manager."""
    if _telemetry_manager:
        return _telemetry_manager.get_meter(name, version)
    return _NoOpMeter() if not OTEL_AVAILABLE else None

def shutdown_telemetry(timeout_millis: int=30000) -> None:
    """Shutdown global telemetry."""
    if _telemetry_manager:
        _telemetry_manager.shutdown(timeout_millis)

@contextmanager
def telemetry_context(config: TelemetryConfig | None=None):
    """Context manager for temporary telemetry setup."""
    manager = TelemetryManager(config)
    manager.initialize()
    try:
        yield manager
    finally:
        manager.shutdown()

@asynccontextmanager
async def async_telemetry_context(config: TelemetryConfig | None=None):
    """Async context manager for temporary telemetry setup."""
    manager = TelemetryManager(config)
    manager.initialize()
    try:
        yield manager
    finally:
        manager.shutdown()
