"""Unified Monitoring Manager - Complete Monitoring and Telemetry Integration

Consolidates all monitoring functionality following the unified manager pattern:
- OpenTelemetry telemetry management (tracing, metrics, logging)
- Auto-instrumentation for all components (HTTP, database, Redis)
- Performance monitoring and health checks
- Distributed tracing and observability
- Custom metrics collection and export

Replaces and consolidates:
- TelemetryManager (OpenTelemetry setup and management)
- InstrumentationManager (auto-instrumentation)
- PerformanceMonitor (performance tracking)
- HealthMonitor (health checks)

Implements 2025 best practices for unified monitoring:
- Mode-based monitoring configuration
- Async telemetry operations with proper lifecycle
- Integration with DatabaseServices for persistence
- Comprehensive observability with minimal overhead
"""

import asyncio
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Health and alerting imports
from pydantic import BaseModel
from sqlmodel import Field, SQLModel

# Health system imports
try:
    from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent
    from prompt_improver.performance.monitoring.health.background_manager import (
        TaskPriority,
        get_background_task_manager,
    )
    from prompt_improver.performance.monitoring.health.unified_health_system import (
        HealthCheckCategory,
        HealthCheckPluginConfig,
        HealthProfile,
    )

    HEALTH_SYSTEM_AVAILABLE = True
except ImportError:
    HEALTH_SYSTEM_AVAILABLE = False

# Import all telemetry dependencies with fallbacks
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.console import (
        ConsoleMetricExporter,
        ConsoleSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Environment types for different deployment contexts"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    source_component: str | None
    affected_components: list[str]
    metrics: dict[str, Any]
    tags: dict[str, str]
    created_at: datetime
    updated_at: datetime
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    resolution_note: str | None = None


@dataclass
class AlertRule:
    """Alert rule configuration."""

    rule_id: str
    rule_name: str
    metric_name: str
    condition: str
    threshold: float
    duration_seconds: int
    severity: AlertSeverity
    enabled: bool
    tags: dict[str, str]


@dataclass
class AlertStats:
    """Alert statistics."""

    total_alerts: int = 0
    active_alerts: int = 0
    critical_alerts: int = 0
    warning_alerts: int = 0
    alerts_last_hour: int = 0
    alerts_last_day: int = 0
    avg_resolution_time_hours: float = 0.0
    top_alert_types: list[dict[str, Any]] = field(default_factory=list)


class HealthMonitoringPolicy(BaseModel):
    """Health monitoring policy configuration"""

    enabled: bool = Field(default=True)
    global_timeout_seconds: float = Field(default=30.0)
    parallel_execution: bool = Field(default=True)
    max_concurrent_checks: int = Field(default=10)
    failure_retry_count: int = Field(default=1)
    failure_retry_delay_seconds: float = Field(default=2.0)
    critical_alert_threshold: int = Field(default=1)
    degraded_alert_threshold: int = Field(default=3)
    health_check_interval_seconds: float = Field(default=60.0)
    metrics_retention_hours: int = Field(default=24)
    enable_telemetry: bool = Field(default=True)
    enable_circuit_breaker: bool = Field(default=True)
    enable_sla_monitoring: bool = Field(default=True)


class CategoryThresholds(BaseModel):
    """Performance thresholds for health check categories"""

    timeout_seconds: float = Field(default=10.0)
    critical_response_time_ms: float = Field(default=5000.0)
    warning_response_time_ms: float = Field(default=2000.0)
    max_failure_rate: float = Field(default=0.1)
    retry_count: int = Field(default=1)
    retry_delay_seconds: float = Field(default=1.0)


class MonitoringMode(Enum):
    """Monitoring operation modes."""

    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    MINIMAL = "minimal"


class ExporterType(Enum):
    """Telemetry exporter types."""

    CONSOLE = "console"
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"
    NONE = "none"


@dataclass
class UnifiedMonitoringConfig:
    """Unified monitoring configuration with health and alerting support."""

    mode: MonitoringMode = MonitoringMode.DEVELOPMENT
    service_name: str = "prompt-improver"
    service_version: str = "1.0.0"
    environment: str = "development"

    # Telemetry configuration
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_auto_instrumentation: bool = True

    # Export configuration
    trace_exporter: ExporterType = ExporterType.CONSOLE
    metric_exporter: ExporterType = ExporterType.CONSOLE
    otlp_endpoint_grpc: str = "http://otel-collector:4317"
    otlp_endpoint_http: str = "http://otel-collector:4318"

    # Sampling configuration
    sampling_rate: float = 0.1
    rate_limit_per_second: int = 100

    # Health monitoring configuration
    enable_health_monitoring: bool = True
    enable_alerting: bool = True
    health_environment: EnvironmentType = EnvironmentType.DEVELOPMENT

    # Alert configuration
    alert_retention_days: int = 30
    max_alerts_in_memory: int = 1000
    escalation_timeout_minutes: int = 30
    suppression_timeout_minutes: int = 60
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    enable_pagerduty_alerts: bool = False

    @classmethod
    def from_environment(cls) -> "UnifiedMonitoringConfig":
        """Create configuration from environment variables."""
        return cls(
            mode=MonitoringMode(os.getenv("MONITORING_MODE", "development")),
            service_name=os.getenv("OTEL_SERVICE_NAME", "prompt-improver"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            environment=os.getenv("OTEL_ENVIRONMENT", "development"),
            enable_tracing=os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true",
            enable_metrics=os.getenv("OTEL_METRICS_ENABLED", "true").lower() == "true",
            enable_logging=os.getenv("OTEL_LOGGING_ENABLED", "true").lower() == "true",
            enable_auto_instrumentation=os.getenv(
                "OTEL_AUTO_INSTRUMENTATION_ENABLED", "true"
            ).lower()
            == "true",
            trace_exporter=ExporterType(os.getenv("OTEL_TRACE_EXPORTER", "console")),
            metric_exporter=ExporterType(os.getenv("OTEL_METRIC_EXPORTER", "console")),
            otlp_endpoint_grpc=os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
            ),
            otlp_endpoint_http=os.getenv(
                "OTEL_EXPORTER_OTLP_HTTP_ENDPOINT", "http://otel-collector:4318"
            ),
            sampling_rate=float(os.getenv("OTEL_SAMPLING_RATE", "0.1")),
            rate_limit_per_second=int(os.getenv("OTEL_RATE_LIMIT_PER_SECOND", "100")),
            # Health and alerting configuration
            enable_health_monitoring=os.getenv(
                "HEALTH_MONITORING_ENABLED", "true"
            ).lower()
            == "true",
            enable_alerting=os.getenv("ALERTING_ENABLED", "true").lower() == "true",
            health_environment=EnvironmentType(
                os.getenv("HEALTH_ENVIRONMENT", "development")
            ),
            alert_retention_days=int(os.getenv("ALERT_RETENTION_DAYS", "30")),
            max_alerts_in_memory=int(os.getenv("MAX_ALERTS_IN_MEMORY", "1000")),
            escalation_timeout_minutes=int(
                os.getenv("ESCALATION_TIMEOUT_MINUTES", "30")
            ),
            suppression_timeout_minutes=int(
                os.getenv("SUPPRESSION_TIMEOUT_MINUTES", "60")
            ),
            enable_email_alerts=os.getenv("EMAIL_ALERTS_ENABLED", "false").lower()
            == "true",
            enable_slack_alerts=os.getenv("SLACK_ALERTS_ENABLED", "false").lower()
            == "true",
            enable_pagerduty_alerts=os.getenv(
                "PAGERDUTY_ALERTS_ENABLED", "false"
            ).lower()
            == "true",
        )


class UnifiedMonitoringManager:
    """Unified Monitoring Manager - Complete Monitoring and Telemetry Integration.

    Consolidates all monitoring functionality:
    - OpenTelemetry telemetry management (tracing, metrics, logging)
    - Auto-instrumentation for all components
    - Health configuration management with environment-specific policies
    - Alert management with escalation and notification systems
    - Performance monitoring and health checks
    - Distributed tracing and observability
    - Custom metrics collection and export

    Replaces and consolidates:
    - TelemetryManager (OpenTelemetry setup and management)
    - InstrumentationManager (auto-instrumentation)
    - HealthConfigurationManager (health configuration)
    - AlertManager (alert management and escalation)
    - PerformanceMonitor (performance tracking)
    - HealthMonitor (health checks)
    """

    def __init__(self, config: UnifiedMonitoringConfig | None = None, event_bus=None):
        """Initialize unified monitoring manager.

        Args:
            config: Monitoring configuration (auto-detected from environment if not provided)
            event_bus: Event bus for ML events (optional)
        """
        self.config = config or UnifiedMonitoringConfig.from_environment()
        self.event_bus = event_bus

        # Telemetry components
        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._initialized = False
        self._instrumented: dict[str, bool] = {}

        # Health configuration components
        self._health_policies: dict[EnvironmentType, HealthMonitoringPolicy] = {}
        self._category_thresholds: dict = {}
        self._health_profiles: dict[str, Any] = {}

        # Alert management components
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.alert_rules: dict[str, AlertRule] = {}
        self.alert_stats = AlertStats()
        self.is_monitoring = False
        self.monitor_task = None

        # Initialize health and alert components if available
        if HEALTH_SYSTEM_AVAILABLE and self.config.enable_health_monitoring:
            self._initialize_health_configuration()
        if HEALTH_SYSTEM_AVAILABLE and self.config.enable_alerting:
            self._initialize_alert_management()

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available - telemetry functionality limited"
            )

        if not HEALTH_SYSTEM_AVAILABLE:
            logger.warning(
                "Health system not available - health and alerting functionality limited"
            )

    def initialize(self) -> None:
        """Initialize unified monitoring with telemetry and instrumentation."""
        if not OTEL_AVAILABLE or self._initialized:
            return

        try:
            self._setup_resource()

            if self.config.enable_tracing:
                self._setup_tracing()

            if self.config.enable_metrics:
                self._setup_metrics()

            if self.config.enable_auto_instrumentation:
                self._setup_auto_instrumentation()

            self._initialized = True
            logger.info(
                f"Unified monitoring initialized successfully (mode: {self.config.mode.value})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize unified monitoring: {e}")
            raise

    def _setup_resource(self):
        """Create OpenTelemetry resource with service metadata."""
        if not OTEL_AVAILABLE:
            return None

        attributes = {
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "deployment.environment": self.config.environment,
            "service.namespace": "prompt-improver",
            "service.instance.id": os.getenv("HOSTNAME", "unknown"),
            "unified.monitoring.enabled": True,
            "monitoring.mode": self.config.mode.value,
        }

        self._resource = Resource.create(attributes)
        return self._resource

    def _setup_tracing(self) -> None:
        """Configure distributed tracing with sampling and export."""
        if not OTEL_AVAILABLE:
            return

        # Create sampler based on configuration
        sampler = ParentBased(root=TraceIdRatioBased(self.config.sampling_rate))

        self._tracer_provider = TracerProvider(resource=self._resource, sampler=sampler)

        # Configure trace exporter
        if self.config.trace_exporter == ExporterType.OTLP_GRPC:
            exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint_grpc)
        elif self.config.trace_exporter == ExporterType.CONSOLE:
            exporter = ConsoleSpanExporter()
        else:
            return

        # Add batch processor
        processor = BatchSpanProcessor(exporter)
        self._tracer_provider.add_span_processor(processor)

        # Set global tracer provider
        trace.set_tracer_provider(self._tracer_provider)
        logger.info(
            f"Tracing configured with {self.config.trace_exporter.value} exporter"
        )

    def _setup_metrics(self) -> None:
        """Configure metrics collection and export."""
        if not OTEL_AVAILABLE:
            return

        self._meter_provider = MeterProvider(resource=self._resource)

        # Configure metric exporter
        if self.config.metric_exporter == ExporterType.OTLP_GRPC:
            exporter = OTLPMetricExporter(endpoint=self.config.otlp_endpoint_grpc)
        elif self.config.metric_exporter == ExporterType.CONSOLE:
            exporter = ConsoleMetricExporter()
        else:
            return

        # Set global meter provider
        metrics.set_meter_provider(self._meter_provider)
        logger.info(
            f"Metrics configured with {self.config.metric_exporter.value} exporter"
        )

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for all components."""
        if not OTEL_AVAILABLE:
            return

        self.instrument_http()
        self.instrument_database()
        self.instrument_redis()
        logger.info("Auto-instrumentation enabled for all components")

    def instrument_http(self) -> None:
        """Instrument HTTP client and server operations."""
        if not OTEL_AVAILABLE or self._instrumented.get("http", False):
            return

        try:
            AioHttpClientInstrumentor().instrument()
            RequestsInstrumentor().instrument()
            self._instrumented["http"] = True
            logger.info("HTTP instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument HTTP: {e}")

    def instrument_database(self) -> None:
        """Instrument database operations."""
        if not OTEL_AVAILABLE or self._instrumented.get("database", False):
            return

        try:
            AsyncPGInstrumentor().instrument()
            self._instrumented["database"] = True
            logger.info("Database instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument database: {e}")

    def instrument_redis(self) -> None:
        """Instrument Redis operations."""
        if not OTEL_AVAILABLE or self._instrumented.get("redis", False):
            return

        try:
            RedisInstrumentor().instrument()
            self._instrumented["redis"] = True
            logger.info("Redis instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument Redis: {e}")

    def get_tracer(self, name: str) -> Any:
        """Get tracer for component."""
        if OTEL_AVAILABLE and self._initialized:
            return trace.get_tracer(name)
        return None

    def get_meter(self, name: str) -> Any:
        """Get meter for component."""
        if OTEL_AVAILABLE and self._initialized:
            return metrics.get_meter(name)
        return None

    def is_initialized(self) -> bool:
        """Check if monitoring is initialized."""
        return self._initialized

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "initialized": self._initialized,
            "mode": self.config.mode.value,
            "opentelemetry_available": OTEL_AVAILABLE,
            "tracing_enabled": self.config.enable_tracing
            and self._tracer_provider is not None,
            "metrics_enabled": self.config.enable_metrics
            and self._meter_provider is not None,
            "auto_instrumentation_enabled": self.config.enable_auto_instrumentation,
            "instrumented_components": self._instrumented,
            "service_info": {
                "name": self.config.service_name,
                "version": self.config.service_version,
                "environment": self.config.environment,
            },
            "export_configuration": {
                "trace_exporter": self.config.trace_exporter.value,
                "metric_exporter": self.config.metric_exporter.value,
                "otlp_endpoint_grpc": self.config.otlp_endpoint_grpc,
                "sampling_rate": self.config.sampling_rate,
            },
        }

    async def shutdown(self) -> None:
        """Graceful shutdown of monitoring components."""
        if not self._initialized:
            return

        try:
            # Shutdown trace provider
            if self._tracer_provider:
                if hasattr(self._tracer_provider, "shutdown"):
                    self._tracer_provider.shutdown()

            # Shutdown meter provider
            if self._meter_provider:
                if hasattr(self._meter_provider, "shutdown"):
                    self._meter_provider.shutdown()

            self._initialized = False
            logger.info("Unified monitoring shutdown completed")

            # Shutdown alert monitoring
            if self.is_monitoring:
                await self.stop_alert_monitoring()

        except Exception as e:
            logger.error(f"Error during monitoring shutdown: {e}")

    # === HEALTH CONFIGURATION MANAGEMENT ===

    def _initialize_health_configuration(self) -> None:
        """Initialize health configuration management"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return

        # Create default policies for each environment
        self._health_policies = self._create_default_policies()
        self._category_thresholds = self._create_default_thresholds()
        self._health_profiles = self._create_default_profiles()

    def _create_default_policies(self) -> dict[EnvironmentType, HealthMonitoringPolicy]:
        """Create default monitoring policies for each environment"""
        return {
            EnvironmentType.DEVELOPMENT: HealthMonitoringPolicy(
                global_timeout_seconds=60.0,
                parallel_execution=True,
                max_concurrent_checks=5,
                failure_retry_count=0,
                health_check_interval_seconds=120.0,
                enable_telemetry=False,
                enable_circuit_breaker=False,
                enable_sla_monitoring=False,
            ),
            EnvironmentType.TESTING: HealthMonitoringPolicy(
                global_timeout_seconds=30.0,
                parallel_execution=True,
                max_concurrent_checks=8,
                failure_retry_count=1,
                health_check_interval_seconds=60.0,
                enable_telemetry=True,
                enable_circuit_breaker=True,
                enable_sla_monitoring=True,
            ),
            EnvironmentType.STAGING: HealthMonitoringPolicy(
                global_timeout_seconds=25.0,
                parallel_execution=True,
                max_concurrent_checks=12,
                failure_retry_count=2,
                critical_alert_threshold=1,
                degraded_alert_threshold=2,
                health_check_interval_seconds=45.0,
                enable_telemetry=True,
                enable_circuit_breaker=True,
                enable_sla_monitoring=True,
            ),
            EnvironmentType.PRODUCTION: HealthMonitoringPolicy(
                global_timeout_seconds=20.0,
                parallel_execution=True,
                max_concurrent_checks=15,
                failure_retry_count=3,
                failure_retry_delay_seconds=1.0,
                critical_alert_threshold=1,
                degraded_alert_threshold=2,
                health_check_interval_seconds=30.0,
                metrics_retention_hours=72,
                enable_telemetry=True,
                enable_circuit_breaker=True,
                enable_sla_monitoring=True,
            ),
            EnvironmentType.LOCAL: HealthMonitoringPolicy(
                global_timeout_seconds=45.0,
                parallel_execution=False,
                max_concurrent_checks=3,
                failure_retry_count=0,
                health_check_interval_seconds=300.0,
                enable_telemetry=False,
                enable_circuit_breaker=False,
                enable_sla_monitoring=False,
            ),
        }

    def _create_default_thresholds(self) -> dict:
        """Create default thresholds for each health check category"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return {}

        base_thresholds = {
            EnvironmentType.DEVELOPMENT: {
                HealthCheckCategory.ML: CategoryThresholds(
                    timeout_seconds=30.0,
                    critical_response_time_ms=10000.0,
                    warning_response_time_ms=5000.0,
                    retry_count=0,
                ),
                HealthCheckCategory.DATABASE: CategoryThresholds(
                    timeout_seconds=15.0,
                    critical_response_time_ms=3000.0,
                    warning_response_time_ms=1000.0,
                    retry_count=1,
                ),
                HealthCheckCategory.REDIS: CategoryThresholds(
                    timeout_seconds=10.0,
                    critical_response_time_ms=1000.0,
                    warning_response_time_ms=500.0,
                    retry_count=1,
                ),
                HealthCheckCategory.API: CategoryThresholds(
                    timeout_seconds=20.0,
                    critical_response_time_ms=5000.0,
                    warning_response_time_ms=2000.0,
                    retry_count=1,
                ),
                HealthCheckCategory.SYSTEM: CategoryThresholds(
                    timeout_seconds=10.0,
                    critical_response_time_ms=2000.0,
                    warning_response_time_ms=1000.0,
                    retry_count=0,
                ),
                HealthCheckCategory.EXTERNAL: CategoryThresholds(
                    timeout_seconds=30.0,
                    critical_response_time_ms=15000.0,
                    warning_response_time_ms=10000.0,
                    retry_count=2,
                ),
                HealthCheckCategory.CUSTOM: CategoryThresholds(
                    timeout_seconds=15.0, retry_count=1
                ),
            }
        }

        # Production adjustments
        production_adjustments = {
            HealthCheckCategory.ML: CategoryThresholds(
                timeout_seconds=20.0,
                critical_response_time_ms=5000.0,
                warning_response_time_ms=2000.0,
                retry_count=2,
            ),
            HealthCheckCategory.DATABASE: CategoryThresholds(
                timeout_seconds=10.0,
                critical_response_time_ms=1000.0,
                warning_response_time_ms=500.0,
                retry_count=2,
            ),
            HealthCheckCategory.REDIS: CategoryThresholds(
                timeout_seconds=5.0,
                critical_response_time_ms=500.0,
                warning_response_time_ms=200.0,
                retry_count=2,
            ),
            HealthCheckCategory.API: CategoryThresholds(
                timeout_seconds=15.0,
                critical_response_time_ms=3000.0,
                warning_response_time_ms=1000.0,
                retry_count=2,
            ),
            HealthCheckCategory.SYSTEM: CategoryThresholds(
                timeout_seconds=8.0,
                critical_response_time_ms=1000.0,
                warning_response_time_ms=500.0,
                retry_count=1,
            ),
            HealthCheckCategory.EXTERNAL: CategoryThresholds(
                timeout_seconds=25.0,
                critical_response_time_ms=10000.0,
                warning_response_time_ms=5000.0,
                retry_count=3,
            ),
        }

        base_thresholds[EnvironmentType.PRODUCTION] = production_adjustments
        base_thresholds[EnvironmentType.STAGING] = production_adjustments
        base_thresholds[EnvironmentType.TESTING] = base_thresholds[
            EnvironmentType.DEVELOPMENT
        ]
        base_thresholds[EnvironmentType.LOCAL] = base_thresholds[
            EnvironmentType.DEVELOPMENT
        ]

        return base_thresholds[self.config.health_environment]

    def _create_default_profiles(self) -> dict[str, Any]:
        """Create default health check profiles"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return {}

        policy = self._health_policies[self.config.health_environment]
        profiles = {
            "full": HealthProfile(
                name="full",
                enabled_plugins={
                    "ml_service",
                    "enhanced_ml_service",
                    "ml_model",
                    "ml_data_quality",
                    "ml_training",
                    "ml_performance",
                    "ml_orchestrator",
                    "ml_component_registry",
                    "ml_resource_manager",
                    "ml_workflow_engine",
                    "ml_event_bus",
                    "database",
                    "database_connection_pool",
                    "database_query_performance",
                    "database_index_health",
                    "database_bloat",
                    "redis",
                    "redis_detailed",
                    "redis_memory",
                    "analytics_service",
                    "enhanced_analytics_service",
                    "mcp_server",
                    "system_resources",
                    "queue_service",
                },
                global_timeout=policy.global_timeout_seconds,
                parallel_execution=policy.parallel_execution,
                critical_only=False,
            ),
            "critical": HealthProfile(
                name="critical",
                enabled_plugins={
                    "enhanced_ml_service",
                    "database",
                    "redis",
                    "enhanced_analytics_service",
                    "mcp_server",
                },
                global_timeout=policy.global_timeout_seconds * 0.7,
                parallel_execution=True,
                critical_only=True,
            ),
            "minimal": HealthProfile(
                name="minimal",
                enabled_plugins={"database", "redis", "system_resources"},
                global_timeout=15.0,
                parallel_execution=True,
                critical_only=False,
            ),
            "ml_focused": HealthProfile(
                name="ml_focused",
                enabled_plugins={
                    "enhanced_ml_service",
                    "ml_model",
                    "ml_data_quality",
                    "ml_training",
                    "ml_performance",
                    "ml_orchestrator",
                    "database",
                    "redis",
                },
                global_timeout=policy.global_timeout_seconds,
                parallel_execution=True,
                critical_only=False,
            ),
            "infrastructure": HealthProfile(
                name="infrastructure",
                enabled_plugins={
                    "database",
                    "database_connection_pool",
                    "database_query_performance",
                    "redis",
                    "redis_detailed",
                    "redis_memory",
                    "system_resources",
                    "queue_service",
                },
                global_timeout=policy.global_timeout_seconds,
                parallel_execution=True,
                critical_only=False,
            ),
        }

        # Set default profile based on environment
        if self.config.health_environment == EnvironmentType.PRODUCTION:
            profiles["default"] = profiles["critical"]
        elif self.config.health_environment == EnvironmentType.DEVELOPMENT:
            profiles["default"] = profiles["minimal"]
        else:
            profiles["default"] = profiles["full"]

        return profiles

    def get_health_policy(self) -> HealthMonitoringPolicy | None:
        """Get monitoring policy for current environment"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return None
        return self._health_policies.get(self.config.health_environment)

    def get_category_thresholds(self, category) -> CategoryThresholds | None:
        """Get thresholds for a specific health check category"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return None
        return self._category_thresholds.get(category, CategoryThresholds())

    def get_health_profile(self, profile_name: str = "default") -> Any:
        """Get a health check profile by name"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return None
        return self._health_profiles.get(profile_name)

    def get_available_health_profiles(self) -> list[str]:
        """Get list of available profile names"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return []
        return list(self._health_profiles.keys())

    def create_plugin_config(
        self, category, critical: bool = False, **overrides
    ) -> Any:
        """Create plugin configuration based on category and environment"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return None

        thresholds = self.get_category_thresholds(category)
        if not thresholds:
            return None

        policy = self.get_health_policy()
        if not policy:
            return None

        config = HealthCheckPluginConfig(
            enabled=True,
            timeout_seconds=thresholds.timeout_seconds,
            critical=critical,
            retry_count=thresholds.retry_count,
            retry_delay_seconds=thresholds.retry_delay_seconds,
            tags=set(),
            metadata={
                "environment": self.config.health_environment.value,
                "category": category.value,
                "policy_version": "1.0",
            },
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def get_health_alerting_config(self) -> dict[str, Any]:
        """Get health alerting configuration"""
        policy = self.get_health_policy()
        if not policy:
            return {}

        return {
            "enabled": self.config.health_environment
            in [EnvironmentType.STAGING, EnvironmentType.PRODUCTION],
            "critical_threshold": policy.critical_alert_threshold,
            "degraded_threshold": policy.degraded_alert_threshold,
            "notification_channels": self._get_notification_channels(),
            "escalation_timeout_minutes": self.config.escalation_timeout_minutes,
            "recovery_notification": True,
        }

    def _get_notification_channels(self) -> list[str]:
        """Get notification channels based on environment"""
        channels = []
        if self.config.enable_email_alerts:
            channels.append("email")
        if self.config.enable_slack_alerts:
            channels.append("slack")
        if self.config.enable_pagerduty_alerts:
            channels.append("pagerduty")
        return channels

    # === ALERT MANAGEMENT ===

    def _initialize_alert_management(self) -> None:
        """Initialize alert management system"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return

        self._initialize_default_alert_rules()

    def _initialize_default_alert_rules(self) -> None:
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="component_unhealthy",
                rule_name="Component Unhealthy",
                metric_name="component_health_status",
                condition="equals",
                threshold=0,
                duration_seconds=60,
                severity=AlertSeverity.CRITICAL,
                enabled=True,
                tags={"category": "component_health"},
            ),
            AlertRule(
                rule_id="high_error_rate",
                rule_name="High Error Rate",
                metric_name="error_rate_percentage",
                condition="greater_than",
                threshold=10.0,
                duration_seconds=300,
                severity=AlertSeverity.WARNING,
                enabled=True,
                tags={"category": "performance"},
            ),
            AlertRule(
                rule_id="high_response_time",
                rule_name="High Response Time",
                metric_name="average_response_time_ms",
                condition="greater_than",
                threshold=1000.0,
                duration_seconds=180,
                severity=AlertSeverity.WARNING,
                enabled=True,
                tags={"category": "performance"},
            ),
            AlertRule(
                rule_id="resource_exhaustion",
                rule_name="Resource Exhaustion",
                metric_name="resource_utilization_percentage",
                condition="greater_than",
                threshold=90.0,
                duration_seconds=120,
                severity=AlertSeverity.CRITICAL,
                enabled=True,
                tags={"category": "resources"},
            ),
            AlertRule(
                rule_id="workflow_failure_rate",
                rule_name="High Workflow Failure Rate",
                metric_name="workflow_failure_rate_percentage",
                condition="greater_than",
                threshold=20.0,
                duration_seconds=600,
                severity=AlertSeverity.CRITICAL,
                enabled=True,
                tags={"category": "workflow"},
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    async def start_alert_monitoring(self) -> None:
        """Start alert monitoring"""
        if not HEALTH_SYSTEM_AVAILABLE or not self.config.enable_alerting:
            return

        if self.is_monitoring:
            logger.warning("Alert monitoring is already running")
            return

        logger.info("Starting alert monitoring")
        self.is_monitoring = True

        try:
            task_manager = get_background_task_manager()
            self.monitor_task_id = await task_manager.submit_enhanced_task(
                task_id=f"unified_alert_monitor_{str(uuid.uuid4())[:8]}",
                coroutine=self._alert_monitoring_loop(),
                priority=TaskPriority.HIGH,
                tags={
                    "service": "unified_monitoring",
                    "type": "monitoring",
                    "component": "alert_monitor",
                    "module": "unified_monitoring_manager",
                },
            )

            if self.event_bus:
                await self.event_bus.emit(
                    MLEvent(
                        event_type=EventType.MONITORING_STARTED,
                        source="unified_monitoring_manager",
                        data={
                            "monitor_type": "unified_alert_manager",
                            "started_at": datetime.now(UTC).isoformat(),
                        },
                    )
                )
        except Exception as e:
            logger.error(f"Failed to start alert monitoring: {e}")
            self.is_monitoring = False

    async def stop_alert_monitoring(self) -> None:
        """Stop alert monitoring"""
        if not self.is_monitoring:
            return

        logger.info("Stopping alert monitoring")
        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        if self.event_bus:
            await self.event_bus.emit(
                MLEvent(
                    event_type=EventType.MONITORING_STOPPED,
                    source="unified_monitoring_manager",
                    data={
                        "monitor_type": "unified_alert_manager",
                        "stopped_at": datetime.now(UTC).isoformat(),
                    },
                )
            )

    async def _alert_monitoring_loop(self) -> None:
        """Main alert monitoring loop"""
        try:
            while self.is_monitoring:
                await self._process_alert_escalations()
                await self._cleanup_old_alerts()
                await self._update_alert_statistics()
                await self._check_alert_storms()
                await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            logger.info("Alert monitoring loop cancelled")
        except Exception as e:
            logger.error("Error in alert monitoring loop: %s", e)
            self.is_monitoring = False

    async def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        source_component: str = None,
        affected_components: list[str] = None,
        metrics: dict[str, Any] = None,
        tags: dict[str, str] = None,
    ) -> str:
        """Create a new alert"""
        if not HEALTH_SYSTEM_AVAILABLE or not self.config.enable_alerting:
            return ""

        try:
            alert_id = f"alert_{int(datetime.now().timestamp())}_{alert_type}"
            timestamp = datetime.now(UTC)

            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                status=AlertStatus.ACTIVE,
                title=title,
                description=description,
                source_component=source_component,
                affected_components=affected_components or [],
                metrics=metrics or {},
                tags=tags or {},
                created_at=timestamp,
                updated_at=timestamp,
            )

            # Check for duplicate alerts
            duplicate_alert = await self._check_duplicate_alert(alert)
            if duplicate_alert:
                logger.debug("Suppressing duplicate alert: %s", alert_id)
                return duplicate_alert.alert_id

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Update statistics
            self.alert_stats.total_alerts += 1
            self.alert_stats.active_alerts += 1
            if severity == AlertSeverity.CRITICAL:
                self.alert_stats.critical_alerts += 1
            elif severity == AlertSeverity.WARNING:
                self.alert_stats.warning_alerts += 1

            logger.warning(f"Alert created: {title} ({severity.value}) - {description}")

            # Emit event
            if self.event_bus:
                await self.event_bus.emit(
                    MLEvent(
                        event_type=EventType.ALERT_CREATED,
                        source="unified_monitoring_manager",
                        data={
                            "alert": asdict(alert),
                            "timestamp": timestamp.isoformat(),
                        },
                    )
                )

            # Send external notifications
            await self._send_external_notifications(alert)

            return alert_id

        except Exception as e:
            logger.error("Error creating alert: %s", e)
            return ""

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str, note: str = None
    ) -> bool:
        """Acknowledge an alert"""
        if not HEALTH_SYSTEM_AVAILABLE or alert_id not in self.active_alerts:
            return False

        try:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(UTC)
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now(UTC)

            if note:
                alert.description += f"\n\nAcknowledgment note: {note}"

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

            if self.event_bus:
                await self.event_bus.emit(
                    MLEvent(
                        event_type=EventType.ALERT_ACKNOWLEDGED,
                        source="unified_monitoring_manager",
                        data={
                            "alert_id": alert_id,
                            "acknowledged_by": acknowledged_by,
                            "note": note,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                )

            return True

        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    async def resolve_alert(self, alert_id: str, resolution_note: str = None) -> bool:
        """Resolve an alert"""
        if not HEALTH_SYSTEM_AVAILABLE or alert_id not in self.active_alerts:
            return False

        try:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(UTC)
            alert.resolution_note = resolution_note
            alert.updated_at = datetime.now(UTC)

            del self.active_alerts[alert_id]

            # Update statistics
            self.alert_stats.active_alerts -= 1
            if alert.severity == AlertSeverity.CRITICAL:
                self.alert_stats.critical_alerts -= 1
            elif alert.severity == AlertSeverity.WARNING:
                self.alert_stats.warning_alerts -= 1

            logger.info(
                f"Alert {alert_id} resolved: {resolution_note or 'No note provided'}"
            )

            if self.event_bus:
                duration_minutes = (
                    alert.resolved_at - alert.created_at
                ).total_seconds() / 60
                await self.event_bus.emit(
                    MLEvent(
                        event_type=EventType.ALERT_RESOLVED,
                        source="unified_monitoring_manager",
                        data={
                            "alert_id": alert_id,
                            "resolution_note": resolution_note,
                            "duration_minutes": duration_minutes,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                )

            return True

        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False

    async def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get all active alerts"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return []
        return [asdict(alert) for alert in self.active_alerts.values()]

    async def get_alert_statistics(self) -> dict[str, Any]:
        """Get alert statistics"""
        if not HEALTH_SYSTEM_AVAILABLE:
            return {}
        return asdict(self.alert_stats)

    async def _check_duplicate_alert(self, new_alert: Alert) -> Alert | None:
        """Check for duplicate alerts to prevent spam"""
        for alert in self.active_alerts.values():
            if (
                alert.alert_type == new_alert.alert_type
                and alert.source_component == new_alert.source_component
                and alert.status == AlertStatus.ACTIVE
            ):
                time_diff = (new_alert.created_at - alert.created_at).total_seconds()
                if time_diff < self.config.suppression_timeout_minutes * 60:
                    return alert
        return None

    async def _process_alert_escalations(self) -> None:
        """Process alert escalations"""
        try:
            current_time = datetime.now(UTC)
            for alert in self.active_alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    time_since_creation = (
                        current_time - alert.created_at
                    ).total_seconds()
                    escalation_threshold = self.config.escalation_timeout_minutes * 60

                    if time_since_creation > escalation_threshold:
                        await self._escalate_alert(alert)
        except Exception as e:
            logger.error("Error processing escalations: %s", e)

    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an unacknowledged alert"""
        try:
            if alert.severity == AlertSeverity.WARNING:
                alert.severity = AlertSeverity.CRITICAL
                self.alert_stats.warning_alerts -= 1
                self.alert_stats.critical_alerts += 1
            elif alert.severity == AlertSeverity.CRITICAL:
                alert.severity = AlertSeverity.EMERGENCY

            alert.updated_at = datetime.now(UTC)
            logger.warning(
                f"Alert {alert.alert_id} escalated to {alert.severity.value}"
            )

            await self._send_escalation_notifications(alert)

            if self.event_bus:
                await self.event_bus.emit(
                    MLEvent(
                        event_type=EventType.ALERT_ESCALATED,
                        source="unified_monitoring_manager",
                        data={
                            "alert_id": alert.alert_id,
                            "new_severity": alert.severity.value,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                )
        except Exception as e:
            logger.error(f"Error escalating alert {alert.alert_id}: {e}")

    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts"""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(
                days=self.config.alert_retention_days
            )
            original_count = len(self.alert_history)

            self.alert_history = [
                alert for alert in self.alert_history if alert.created_at > cutoff_time
            ]

            if len(self.alert_history) > self.config.max_alerts_in_memory:
                self.alert_history = self.alert_history[
                    -self.config.max_alerts_in_memory :
                ]

            cleaned_count = original_count - len(self.alert_history)
            if cleaned_count > 0:
                logger.debug("Cleaned up %s old alerts", cleaned_count)
        except Exception as e:
            logger.error("Error cleaning up alerts: %s", e)

    async def _update_alert_statistics(self) -> None:
        """Update alert statistics"""
        try:
            current_time = datetime.now(UTC)
            one_hour_ago = current_time - timedelta(hours=1)
            one_day_ago = current_time - timedelta(days=1)

            # Update time-based statistics
            self.alert_stats.alerts_last_hour = sum(
                1 for alert in self.alert_history if alert.created_at > one_hour_ago
            )
            self.alert_stats.alerts_last_day = sum(
                1 for alert in self.alert_history if alert.created_at > one_day_ago
            )

            # Calculate average resolution time
            resolved_alerts = [
                alert
                for alert in self.alert_history
                if alert.resolved_at is not None and alert.created_at > one_day_ago
            ]

            if resolved_alerts:
                total_resolution_time = sum(
                    (alert.resolved_at - alert.created_at).total_seconds()
                    for alert in resolved_alerts
                )
                self.alert_stats.avg_resolution_time_hours = (
                    total_resolution_time / len(resolved_alerts) / 3600
                )

            # Update top alert types
            alert_type_counts = {}
            for alert in self.alert_history:
                if alert.created_at > one_day_ago:
                    alert_type_counts[alert.alert_type] = (
                        alert_type_counts.get(alert.alert_type, 0) + 1
                    )

            self.alert_stats.top_alert_types = [
                {"alert_type": alert_type, "count": count}
                for alert_type, count in sorted(
                    alert_type_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]
        except Exception as e:
            logger.error("Error updating alert stats: %s", e)

    async def _check_alert_storms(self) -> None:
        """Check for alert storms and implement suppression"""
        try:
            current_time = datetime.now(UTC)
            last_hour = current_time - timedelta(hours=1)
            recent_alerts = [
                alert for alert in self.alert_history if alert.created_at > last_hour
            ]

            if len(recent_alerts) > 50:  # Alert storm threshold
                logger.warning(
                    "Alert storm detected: %s alerts in the last hour",
                    len(recent_alerts),
                )

                if self.event_bus:
                    await self.event_bus.emit(
                        MLEvent(
                            event_type=EventType.ALERT_STORM_DETECTED,
                            source="unified_monitoring_manager",
                            data={
                                "alert_count": len(recent_alerts),
                                "time_window_hours": 1,
                                "timestamp": current_time.isoformat(),
                            },
                        )
                    )
        except Exception as e:
            logger.error("Error checking alert storms: %s", e)

    async def _send_external_notifications(self, alert: Alert) -> None:
        """Send notifications to external systems"""
        try:
            if self.config.enable_email_alerts:
                await self._send_email_notification(alert)
            if self.config.enable_slack_alerts:
                await self._send_slack_notification(alert)
            if self.config.enable_pagerduty_alerts and alert.severity in [
                AlertSeverity.CRITICAL,
                AlertSeverity.EMERGENCY,
            ]:
                await self._send_pagerduty_notification(alert)
        except Exception as e:
            logger.error("Error sending external notifications: %s", e)

    async def _send_escalation_notifications(self, alert: Alert) -> None:
        """Send escalation notifications"""
        try:
            if self.config.enable_email_alerts:
                await self._send_email_notification(alert, is_escalation=True)
            if self.config.enable_slack_alerts:
                await self._send_slack_notification(alert, is_escalation=True)
            if self.config.enable_pagerduty_alerts:
                await self._send_pagerduty_notification(alert, is_escalation=True)
        except Exception as e:
            logger.error("Error sending escalation notifications: %s", e)

    async def _send_email_notification(
        self, alert: Alert, is_escalation: bool = False
    ) -> None:
        """Send email notification"""
        try:
            subject = f"{'ESCALATED: ' if is_escalation else ''}{alert.severity.value.upper()}: {alert.title}"
            logger.debug("Email notification: %s", subject)
            # EXTENSION POINT: Integrate with email service (SMTP/SendGrid/etc.)
            # Implementation would go here when email notifications are required
        except Exception as e:
            logger.error("Error sending email notification: %s", e)

    async def _send_slack_notification(
        self, alert: Alert, is_escalation: bool = False
    ) -> None:
        """Send Slack notification"""
        try:
            message = f"{'🚨 ESCALATED: ' if is_escalation else '⚠️ '}{alert.title}"
            logger.debug("Slack notification: %s", message)
            # EXTENSION POINT: Integrate with Slack API
            # Implementation would go here when Slack notifications are required
        except Exception as e:
            logger.error("Error sending Slack notification: %s", e)

    async def _send_pagerduty_notification(
        self, alert: Alert, is_escalation: bool = False
    ) -> None:
        """Send PagerDuty notification"""
        try:
            incident_key = f"unified_monitoring_{alert.alert_id}"
            logger.debug("PagerDuty notification: %s", incident_key)
            # EXTENSION POINT: Integrate with PagerDuty Events API
            # Implementation would go here when PagerDuty integration is required
        except Exception as e:
            logger.error("Error sending PagerDuty notification: %s", e)


# Create default instance
unified_monitoring_manager = UnifiedMonitoringManager()

# Backward compatibility aliases - to be removed in future version
TelemetryManager = UnifiedMonitoringManager
InstrumentationManager = UnifiedMonitoringManager
HealthConfigurationManager = UnifiedMonitoringManager
AlertManager = UnifiedMonitoringManager


def get_unified_monitoring_manager() -> UnifiedMonitoringManager:
    """Get the default unified monitoring manager instance."""
    return unified_monitoring_manager


# Backward compatibility functions
def get_health_config() -> UnifiedMonitoringManager:
    """Get the unified monitoring manager (replaces HealthConfigurationManager)"""
    return unified_monitoring_manager


def reset_health_config() -> None:
    """Reset the global health configuration manager"""
    # The unified manager handles its own lifecycle


def load_config_from_environment() -> UnifiedMonitoringManager:
    """Load configuration from environment variables"""
    return UnifiedMonitoringManager(UnifiedMonitoringConfig.from_environment())


def get_default_profile():
    """Get the default health profile for current environment"""
    return unified_monitoring_manager.get_health_profile("default")


def get_critical_profile():
    """Get the critical health profile"""
    return unified_monitoring_manager.get_health_profile("critical")


def create_category_config(category, critical: bool = False, **overrides):
    """Create plugin configuration for a category"""
    return unified_monitoring_manager.create_plugin_config(
        category, critical, **overrides
    )
