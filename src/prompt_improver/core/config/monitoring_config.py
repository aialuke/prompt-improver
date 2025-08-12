"""Monitoring and Observability Configuration Module

Configuration for metrics collection, health checks, OpenTelemetry,
and system observability with environment-specific settings.
"""

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class OpenTelemetryConfig(BaseModel):
    """OpenTelemetry and telemetry configuration."""

    # Service Information
    service_name: str = Field(
        default="apes-prompt-improver", description="Service name for telemetry"
    )
    service_version: str = Field(
        default="1.0.0", description="Service version for telemetry"
    )
    environment: str = Field(
        default="development", description="Environment for telemetry"
    )

    # Exporter Configuration
    otlp_endpoint: str | None = Field(
        default=None, description="OTLP gRPC endpoint URL"
    )
    otlp_http_endpoint: str | None = Field(
        default=None, description="OTLP HTTP endpoint URL"
    )
    otlp_insecure: bool = Field(
        default=True, description="Use insecure connection for OTLP"
    )

    # Telemetry Features
    tracing_enabled: bool = Field(
        default=True, description="Enable distributed tracing"
    )
    metrics_enabled: bool = Field(
        default=True, description="Enable metrics collection"
    )
    logging_enabled: bool = Field(
        default=True, description="Enable structured logging"
    )
    auto_instrumentation_enabled: bool = Field(
        default=True, description="Enable automatic instrumentation"
    )

    # Sampling Configuration
    sampling_strategy: str = Field(
        default="parent_based", description="Sampling strategy"
    )
    sampling_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Sampling rate"
    )
    rate_limit_per_second: int = Field(
        default=100, ge=1, le=10000, description="Rate limit for telemetry"
    )

    @field_validator("sampling_strategy")
    @classmethod
    def validate_sampling_strategy(cls, v):
        """Validate sampling strategy."""
        allowed = ["always_on", "always_off", "parent_based", "trace_id_ratio"]
        if v not in allowed:
            raise ValueError(f"Sampling strategy must be one of {allowed}")
        return v


class HealthCheckConfig(BaseModel):
    """Health check system configuration."""

    interval_seconds: float = Field(
        default=30.0, gt=0, le=300, description="Health check interval"
    )
    timeout_seconds: float = Field(
        default=10.0, gt=0, le=60, description="Health check timeout"
    )
    db_query_timeout: float = Field(
        default=5.0, gt=0, le=30, description="Database health check timeout"
    )
    redis_timeout: float = Field(
        default=2.0, gt=0, le=10, description="Redis health check timeout"
    )
    failure_threshold: int = Field(
        default=3, ge=1, le=10, description="Consecutive failures before unhealthy"
    )
    success_threshold: int = Field(
        default=2, ge=1, le=10, description="Consecutive successes to recover"
    )
    enable_detailed_health_info: bool = Field(
        default=True, description="Include detailed component health info"
    )


class MetricsConfig(BaseModel):
    """Metrics collection configuration."""

    collection_interval_seconds: float = Field(
        default=60.0, gt=0, le=3600, description="Metrics collection interval"
    )
    export_interval_seconds: float = Field(
        default=30.0, gt=0, le=300, description="Metrics export interval"
    )
    enable_system_metrics: bool = Field(
        default=True, description="Collect system metrics (CPU, memory, etc.)"
    )
    enable_application_metrics: bool = Field(
        default=True, description="Collect application-specific metrics"
    )
    enable_database_metrics: bool = Field(
        default=True, description="Collect database performance metrics"
    )
    enable_cache_metrics: bool = Field(
        default=True, description="Collect cache performance metrics"
    )
    enable_http_metrics: bool = Field(
        default=True, description="Collect HTTP request metrics"
    )
    enable_ml_metrics: bool = Field(
        default=True, description="Collect ML pipeline metrics"
    )


class LoggingConfig(BaseModel):
    """Logging system configuration."""

    level: str = Field(default="INFO", description="Default logging level")
    format: str = Field(default="json", description="Log format")
    enable_structured_logging: bool = Field(
        default=True, description="Enable structured logging"
    )
    enable_correlation_ids: bool = Field(
        default=True, description="Enable request correlation IDs"
    )
    enable_performance_logging: bool = Field(
        default=True, description="Enable performance logging"
    )
    enable_security_logging: bool = Field(
        default=True, description="Enable security event logging"
    )
    max_log_line_length: int = Field(
        default=8192, gt=0, description="Maximum log line length"
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate logging level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format."""
        allowed = ["json", "text", "structured"]
        if v.lower() not in allowed:
            raise ValueError(f"Log format must be one of {allowed}")
        return v.lower()


class MonitoringConfig(BaseSettings):
    """Main monitoring and observability configuration."""

    # Global Monitoring Settings
    enabled: bool = Field(default=True, description="Enable monitoring system")
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    # Component Configurations
    opentelemetry: OpenTelemetryConfig = Field(default_factory=OpenTelemetryConfig)
    health_checks: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # External Endpoints
    prometheus_endpoint: str | None = Field(
        default=None, description="Prometheus metrics endpoint"
    )
    grafana_endpoint: str | None = Field(
        default=None, description="Grafana dashboard endpoint"
    )
    alertmanager_endpoint: str | None = Field(
        default=None, description="Alertmanager endpoint"
    )

    # Alert Configuration
    enable_alerting: bool = Field(default=True, description="Enable alerting system")
    alert_webhook_url: str | None = Field(
        default=None, description="Webhook URL for alerts"
    )
    critical_alert_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Critical alert threshold"
    )
    warning_alert_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Warning alert threshold"
    )

    # Performance Monitoring
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    performance_baseline_enabled: bool = Field(
        default=True, description="Enable performance baseline tracking"
    )
    regression_detection_enabled: bool = Field(
        default=True, description="Enable performance regression detection"
    )

    @field_validator("critical_alert_threshold")
    @classmethod
    def validate_critical_threshold(cls, v, info):
        """Ensure critical threshold is higher than warning threshold."""
        if "warning_alert_threshold" in info.data:
            warning = info.data["warning_alert_threshold"]
            if v <= warning:
                raise ValueError("Critical threshold must be higher than warning threshold")
        return v

    model_config = {
        "env_prefix": "MONITORING_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_assignment": True,
    }