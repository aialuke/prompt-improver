"""Unified monitoring types and enums.

Provides standardized types for the unified monitoring facade,
consolidating the various health status and result types across
different monitoring systems.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class HealthStatus(Enum):
    """Standardized health status enum."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class ComponentCategory(Enum):
    """Categories for organizing health check components."""

    DATABASE = "database"
    CACHE = "cache"
    ML_MODELS = "ml_models"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    QUEUE = "queue"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Standardized health check result."""

    status: HealthStatus
    component_name: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    category: ComponentCategory = ComponentCategory.CUSTOM

    def is_healthy(self) -> bool:
        """Check if status indicates healthy state."""
        return self.status == HealthStatus.HEALTHY

    def is_degraded(self) -> bool:
        """Check if status indicates degraded state."""
        return self.status == HealthStatus.DEGRADED

    def is_unhealthy(self) -> bool:
        """Check if status indicates unhealthy state."""
        return self.status == HealthStatus.UNHEALTHY


@dataclass
class MetricPoint:
    """Individual metric data point."""

    name: str
    value: float
    metric_type: MetricType
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    unit: str = ""
    description: str = ""


@dataclass
class SystemHealthSummary:
    """Overall system health summary."""

    overall_status: HealthStatus
    total_components: int
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    unknown_components: int
    component_results: dict[str, HealthCheckResult] = field(default_factory=dict)
    check_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def health_percentage(self) -> float:
        """Calculate percentage of healthy components."""
        if self.total_components == 0:
            return 0.0
        return (self.healthy_components / self.total_components) * 100.0

    def get_critical_issues(self) -> list[str]:
        """Get list of critical health issues."""
        issues = []
        for name, result in self.component_results.items():
            if result.is_unhealthy():
                issues.append(f"{name}: {result.message}")
        return issues


@dataclass
class MonitoringConfig:
    """Configuration for unified monitoring system."""

    health_check_timeout_seconds: float = 30.0
    health_check_parallel_enabled: bool = True
    max_concurrent_checks: int = 10
    metrics_collection_enabled: bool = True
    metrics_retention_hours: int = 24
    performance_monitoring_enabled: bool = True
    cache_health_results_seconds: int = 60
    enable_detailed_logging: bool = True
    critical_components: list[str] = field(default_factory=lambda: ["database", "cache"])


@dataclass
class ComponentHealthChecker:
    """Configuration for individual health checker component."""

    name: str
    category: ComponentCategory
    enabled: bool = True
    timeout_seconds: float = 10.0
    critical: bool = False
    check_interval_seconds: float | None = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    tags: dict[str, str] = field(default_factory=dict)
