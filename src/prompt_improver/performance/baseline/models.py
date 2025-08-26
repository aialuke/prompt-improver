"""Performance baseline data models and metrics definitions."""

import statistics
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from sqlmodel import Field

try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
try:
    METRICS_REGISTRY_AVAILABLE = True
except ImportError:
    METRICS_REGISTRY_AVAILABLE = False


class MetricType(Enum):
    """Types of performance metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE = "resource"
    BUSINESS = "business"
    SATURATION = "saturation"
    AVAILABILITY = "availability"


class TrendDirection(Enum):
    """Performance trend directions."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricDefinition(BaseModel):
    """Definition of a performance metric."""

    name: str
    metric_type: MetricType
    description: str
    unit: str
    target_value: float | None = Field(default=None, le=10000.0, ge=0.0)
    warning_threshold: float | None = Field(default=None, le=10000.0, ge=0.0)
    critical_threshold: float | None = Field(default=None, le=10000.0, ge=0.0)
    lower_is_better: bool = Field(default=True)
    tags: dict[str, str] = Field(default_factory=dict)

    def __post_init__(self):
        """Validate metric definition."""
        if (
            self.metric_type in {MetricType.THROUGHPUT, MetricType.AVAILABILITY}
        ):
            self.lower_is_better = False


class MetricValue(BaseModel):
    """A single metric measurement."""

    metric_name: str = Field(min_length=1, max_length=255)
    value: float = Field(le=1000000.0, ge=-1000000.0)
    timestamp: datetime
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricValue":
        """Create from dictionary."""
        return cls(
            metric_name=data["metric_name"],
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


class BaselineMetrics(BaseModel):
    """Complete set of baseline performance metrics."""

    baseline_id: str = Field(min_length=1, max_length=100)
    collection_timestamp: datetime
    duration_seconds: float = Field(gt=0.0, le=86400.0)
    response_times: list[float] = Field(default_factory=list)
    error_rates: list[float] = Field(default_factory=list)
    throughput_values: list[float] = Field(default_factory=list)
    cpu_utilization: list[float] = Field(default_factory=list)
    memory_utilization: list[float] = Field(default_factory=list)
    disk_usage: list[float] = Field(default_factory=list)
    network_io: list[float] = Field(default_factory=list)
    database_connection_time: list[float] = Field(default_factory=list)
    cache_hit_rate: list[float] = Field(default_factory=list)
    queue_depth: list[float] = Field(default_factory=list)
    business_transactions: list[float] = Field(default_factory=list)
    user_satisfaction_score: list[float] = Field(default_factory=list)
    custom_metrics: dict[str, list[float]] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self):
        """Initialize baseline ID if not provided."""
        if not self.baseline_id:
            self.baseline_id = str(uuid.uuid4())

    def add_metric_value(self, metric_name: str, value: float):
        """Add a metric value to the appropriate list."""
        metric_mapping = {
            "response_time": "response_times",
            "error_rate": "error_rates",
            "throughput": "throughput_values",
            "cpu_utilization": "cpu_utilization",
            "memory_utilization": "memory_utilization",
            "disk_usage": "disk_usage",
            "network_io": "network_io",
            "database_connection_time": "database_connection_time",
            "cache_hit_rate": "cache_hit_rate",
            "queue_depth": "queue_depth",
            "business_transactions": "business_transactions",
            "user_satisfaction_score": "user_satisfaction_score",
        }
        if metric_name in metric_mapping:
            attr_name = metric_mapping[metric_name]
            getattr(self, attr_name).append(value)
        else:
            if metric_name not in self.custom_metrics:
                self.custom_metrics[metric_name] = []
            self.custom_metrics[metric_name].append(value)

    def get_metric_statistics(self, metric_name: str) -> dict[str, float]:
        """Calculate statistics for a specific metric."""
        values = self._get_metric_values(metric_name)
        if not values:
            return {}
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": self._percentile(values, 50),
            "p90": self._percentile(values, 90),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def _get_metric_values(self, metric_name: str) -> list[float]:
        """Get values for a specific metric."""
        metric_mapping = {
            "response_time": self.response_times,
            "error_rate": self.error_rates,
            "throughput": self.throughput_values,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "database_connection_time": self.database_connection_time,
            "cache_hit_rate": self.cache_hit_rate,
            "queue_depth": self.queue_depth,
            "business_transactions": self.business_transactions,
            "user_satisfaction_score": self.user_satisfaction_score,
        }
        if metric_name in metric_mapping:
            return metric_mapping[metric_name]
        if metric_name in self.custom_metrics:
            return self.custom_metrics[metric_name]
        return []

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = percentile / 100 * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "baseline_id": self.baseline_id,
            "collection_timestamp": self.collection_timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "response_times": self.response_times,
            "error_rates": self.error_rates,
            "throughput_values": self.throughput_values,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "database_connection_time": self.database_connection_time,
            "cache_hit_rate": self.cache_hit_rate,
            "queue_depth": self.queue_depth,
            "business_transactions": self.business_transactions,
            "user_satisfaction_score": self.user_satisfaction_score,
            "custom_metrics": self.custom_metrics,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineMetrics":
        """Create from dictionary."""
        return cls(
            baseline_id=data["baseline_id"],
            collection_timestamp=datetime.fromisoformat(data["collection_timestamp"]),
            duration_seconds=data["duration_seconds"],
            response_times=data.get("response_times", []),
            error_rates=data.get("error_rates", []),
            throughput_values=data.get("throughput_values", []),
            cpu_utilization=data.get("cpu_utilization", []),
            memory_utilization=data.get("memory_utilization", []),
            disk_usage=data.get("disk_usage", []),
            network_io=data.get("network_io", []),
            database_connection_time=data.get("database_connection_time", []),
            cache_hit_rate=data.get("cache_hit_rate", []),
            queue_depth=data.get("queue_depth", []),
            business_transactions=data.get("business_transactions", []),
            user_satisfaction_score=data.get("user_satisfaction_score", []),
            custom_metrics=data.get("custom_metrics", {}),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


class PerformanceTrend(BaseModel):
    """Performance trend analysis results."""

    metric_name: str = Field(min_length=1, max_length=255)
    direction: TrendDirection
    magnitude: float = Field(ge=-1000.0, le=1000.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    timeframe_start: datetime
    timeframe_end: datetime
    sample_count: int = Field(ge=0, le=1000000)
    baseline_mean: float = Field(ge=-1000000.0, le=1000000.0)
    current_mean: float = Field(ge=-1000000.0, le=1000000.0)
    variance_ratio: float = Field(ge=0.0, le=1000.0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_value_24h: float | None = Field(default=None, ge=-1000000.0, le=1000000.0)
    predicted_value_7d: float | None = Field(default=None, ge=-1000000.0, le=1000000.0)

    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if trend is statistically significant."""
        return self.p_value is not None and self.p_value < threshold


class RegressionAlert(BaseModel):
    """Performance regression alert."""

    alert_id: str = Field(min_length=1, max_length=100)
    metric_name: str = Field(min_length=1, max_length=255)
    severity: AlertSeverity
    message: str = Field(min_length=1, max_length=1000)
    current_value: float = Field(ge=-1000000.0, le=1000000.0)
    baseline_value: float = Field(ge=-1000000.0, le=1000000.0)
    threshold_value: float = Field(ge=-1000000.0, le=1000000.0)
    degradation_percentage: float = Field(ge=-1000.0, le=1000.0)
    detection_timestamp: datetime
    alert_timestamp: datetime
    affected_operations: list[str] = Field(default_factory=list)
    probable_causes: list[str] = Field(default_factory=list)
    remediation_suggestions: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self):
        """Initialize alert ID if not provided."""
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "message": self.message,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "threshold_value": self.threshold_value,
            "degradation_percentage": self.degradation_percentage,
            "detection_timestamp": self.detection_timestamp.isoformat(),
            "alert_timestamp": self.alert_timestamp.isoformat(),
            "affected_operations": self.affected_operations,
            "probable_causes": self.probable_causes,
            "remediation_suggestions": self.remediation_suggestions,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class ProfileData(BaseModel):
    """Code profiling data."""

    profile_id: str = Field(min_length=1, max_length=100)
    operation_name: str = Field(min_length=1, max_length=255)
    start_time: datetime
    end_time: datetime
    duration_ms: float = Field(ge=0.0, le=3600000.0)
    function_calls: list[dict[str, Any]] = Field(default_factory=list)
    memory_snapshots: list[dict[str, Any]] = Field(default_factory=list)
    cpu_samples: list[float] = Field(default_factory=list)
    slow_functions: list[dict[str, Any]] = Field(default_factory=list)
    memory_allocations: list[dict[str, Any]] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self):
        """Initialize profile ID if not provided."""
        if not self.profile_id:
            self.profile_id = str(uuid.uuid4())


class BaselineComparison(BaseModel):
    """Comparison between two baselines."""

    baseline_a_id: str = Field(min_length=1, max_length=100)
    baseline_b_id: str = Field(min_length=1, max_length=100)
    comparison_timestamp: datetime
    metric_comparisons: dict[str, dict[str, Any]] = Field(default_factory=dict)
    overall_improvement: bool
    significant_regressions: list[str] = Field(default_factory=list)
    significant_improvements: list[str] = Field(default_factory=list)
    total_metrics_compared: int = Field(default=0, ge=0)
    metrics_improved: int = Field(default=0, ge=0)
    metrics_degraded: int = Field(default=0, ge=0)
    metrics_stable: int = Field(default=0, ge=0)

    def add_metric_comparison(
        self,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        change_percentage: float,
        is_significant: bool,
        lower_is_better: bool = True,
    ):
        """Add a metric comparison result."""
        self.metric_comparisons[metric_name] = {
            "baseline_value": baseline_value,
            "current_value": current_value,
            "change_percentage": change_percentage,
            "is_significant": is_significant,
            "lower_is_better": lower_is_better,
            "assessment": self._assess_change(
                change_percentage, is_significant, lower_is_better
            ),
        }
        self.total_metrics_compared += 1
        if is_significant:
            if (lower_is_better and change_percentage < 0) or (
                not lower_is_better and change_percentage > 0
            ):
                self.metrics_improved += 1
                self.significant_improvements.append(metric_name)
            elif (lower_is_better and change_percentage > 0) or (
                not lower_is_better and change_percentage < 0
            ):
                self.metrics_degraded += 1
                self.significant_regressions.append(metric_name)
        else:
            self.metrics_stable += 1
        self.overall_improvement = self.metrics_improved > self.metrics_degraded

    def _assess_change(
        self, change_percentage: float, is_significant: bool, lower_is_better: bool
    ) -> str:
        """Assess the nature of a metric change."""
        if not is_significant:
            return "stable"
        if lower_is_better:
            return "improvement" if change_percentage < 0 else "regression"
        return "improvement" if change_percentage > 0 else "regression"


STANDARD_METRICS = {
    "response_time_p95": MetricDefinition(
        name="response_time_p95",
        metric_type=MetricType.LATENCY,
        description="95th percentile response time",
        unit="milliseconds",
        target_value=200.0,
        warning_threshold=150.0,
        critical_threshold=200.0,
        lower_is_better=True,
    ),
    "error_rate": MetricDefinition(
        name="error_rate",
        metric_type=MetricType.ERROR_RATE,
        description="Percentage of failed requests",
        unit="percentage",
        target_value=1.0,
        warning_threshold=5.0,
        critical_threshold=10.0,
        lower_is_better=True,
    ),
    "throughput_rps": MetricDefinition(
        name="throughput_rps",
        metric_type=MetricType.THROUGHPUT,
        description="Requests processed per second",
        unit="requests/second",
        target_value=10.0,
        warning_threshold=5.0,
        critical_threshold=1.0,
        lower_is_better=False,
    ),
    "cpu_utilization": MetricDefinition(
        name="cpu_utilization",
        metric_type=MetricType.RESOURCE,
        description="CPU usage percentage",
        unit="percentage",
        target_value=70.0,
        warning_threshold=80.0,
        critical_threshold=90.0,
        lower_is_better=True,
    ),
    "memory_utilization": MetricDefinition(
        name="memory_utilization",
        metric_type=MetricType.RESOURCE,
        description="Memory usage percentage",
        unit="percentage",
        target_value=70.0,
        warning_threshold=80.0,
        critical_threshold=90.0,
        lower_is_better=True,
    ),
    "database_response_time": MetricDefinition(
        name="database_response_time",
        metric_type=MetricType.LATENCY,
        description="Database query response time",
        unit="milliseconds",
        target_value=50.0,
        warning_threshold=100.0,
        critical_threshold=200.0,
        lower_is_better=True,
    ),
    "cache_hit_rate": MetricDefinition(
        name="cache_hit_rate",
        metric_type=MetricType.THROUGHPUT,
        description="Cache hit rate percentage",
        unit="percentage",
        target_value=90.0,
        warning_threshold=80.0,
        critical_threshold=70.0,
        lower_is_better=False,
    ),
}


def get_metric_definition(metric_name: str) -> MetricDefinition | None:
    """Get standard metric definition by name."""
    return STANDARD_METRICS.get(metric_name)


def list_standard_metrics() -> list[str]:
    """List all standard metric names."""
    return list(STANDARD_METRICS.keys())
