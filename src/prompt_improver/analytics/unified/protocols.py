"""Analytics Component Protocols for Unified Architecture.

This module defines the protocols and interfaces for the unified analytics service components.
Each component specializes in a specific analytics domain while maintaining consistent interfaces.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel


class AnalyticsEvent(BaseModel):
    """Base analytics event structure."""
    event_id: str
    event_type: str
    source: str
    timestamp: datetime
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


class AnalyticsMetrics(BaseModel):
    """Base analytics metrics structure."""
    metric_name: str
    value: float | int | str | bool
    timestamp: datetime
    tags: dict[str, str] | None = None
    context: dict[str, Any] | None = None


class PerformanceMetrics(BaseModel):
    """Performance metrics structure."""
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    timestamp: datetime


class ExperimentResult(BaseModel):
    """A/B testing experiment result."""
    experiment_id: str
    control_group_size: int
    treatment_group_size: int
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    winner: str | None


class SessionMetrics(BaseModel):
    """Session analytics metrics."""
    session_id: str
    user_id: str | None
    duration_seconds: float
    events_count: int
    conversion_rate: float
    quality_score: float
    timestamp: datetime


class MLModelMetrics(BaseModel):
    """ML model performance metrics."""
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    timestamp: datetime


class DataCollectionProtocol(Protocol):
    """Protocol for data collection components."""

    @abstractmethod
    async def collect_event(self, event: AnalyticsEvent) -> bool:
        """Collect an analytics event."""
        ...

    @abstractmethod
    async def collect_metrics(self, metrics: AnalyticsMetrics) -> bool:
        """Collect analytics metrics."""
        ...

    @abstractmethod
    async def aggregate_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Aggregate events within a time range."""
        ...

    @abstractmethod
    async def get_events(
        self,
        filter_criteria: dict[str, Any],
        limit: int = 100
    ) -> list[AnalyticsEvent]:
        """Get events matching filter criteria."""
        ...


class PerformanceAnalyticsProtocol(Protocol):
    """Protocol for performance analytics components."""

    @abstractmethod
    async def track_performance(self, metrics: PerformanceMetrics) -> bool:
        """Track performance metrics."""
        ...

    @abstractmethod
    async def analyze_performance_trends(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> dict[str, Any]:
        """Analyze performance trends over time."""
        ...

    @abstractmethod
    async def detect_performance_anomalies(
        self,
        threshold: float = 2.0
    ) -> list[dict[str, Any]]:
        """Detect performance anomalies."""
        ...

    @abstractmethod
    async def generate_performance_report(
        self,
        report_type: str = "summary"
    ) -> dict[str, Any]:
        """Generate performance analysis report."""
        ...


class ABTestingProtocol(Protocol):
    """Protocol for A/B testing components."""

    @abstractmethod
    async def create_experiment(
        self,
        name: str,
        control_variant: dict[str, Any],
        treatment_variants: list[dict[str, Any]],
        success_metric: str,
        target_significance: float = 0.05
    ) -> str:
        """Create a new A/B testing experiment."""
        ...

    @abstractmethod
    async def record_experiment_event(
        self,
        experiment_id: str,
        variant_id: str,
        event_type: str,
        user_id: str,
        value: float | None = None
    ) -> bool:
        """Record an event for an experiment."""
        ...

    @abstractmethod
    async def analyze_experiment(
        self,
        experiment_id: str
    ) -> ExperimentResult:
        """Analyze experiment results."""
        ...

    @abstractmethod
    async def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get list of active experiments."""
        ...


class SessionAnalyticsProtocol(Protocol):
    """Protocol for session analytics components."""

    @abstractmethod
    async def track_session(self, session_metrics: SessionMetrics) -> bool:
        """Track session metrics."""
        ...

    @abstractmethod
    async def analyze_session_patterns(
        self,
        session_ids: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None
    ) -> dict[str, Any]:
        """Analyze patterns in session data."""
        ...

    @abstractmethod
    async def compare_sessions(
        self,
        session_a_id: str,
        session_b_id: str
    ) -> dict[str, Any]:
        """Compare two sessions."""
        ...

    @abstractmethod
    async def generate_session_report(
        self,
        session_id: str,
        report_format: str = "summary"
    ) -> dict[str, Any]:
        """Generate session analysis report."""
        ...


class MLAnalyticsProtocol(Protocol):
    """Protocol for ML analytics components."""

    @abstractmethod
    async def track_model_performance(self, metrics: MLModelMetrics) -> bool:
        """Track ML model performance metrics."""
        ...

    @abstractmethod
    async def analyze_model_drift(
        self,
        model_id: str,
        time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Analyze model performance drift."""
        ...

    @abstractmethod
    async def compare_models(
        self,
        model_ids: list[str]
    ) -> dict[str, Any]:
        """Compare multiple ML models."""
        ...

    @abstractmethod
    async def predict_model_issues(
        self,
        model_id: str
    ) -> list[dict[str, Any]]:
        """Predict potential model issues."""
        ...


class CacheProtocol(Protocol):
    """Protocol for caching components."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get cached value."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set cached value with optional TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        ...

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        ...


class AnalyticsComponentProtocol(Protocol):
    """Base protocol for all analytics components."""

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check component health status."""
        ...

    @abstractmethod
    async def get_metrics(self) -> dict[str, Any]:
        """Get component performance metrics."""
        ...

    @abstractmethod
    async def configure(self, config: dict[str, Any]) -> bool:
        """Configure component with settings."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown component."""
        ...


class AnalyticsServiceProtocol(Protocol):
    """Main protocol for the unified analytics service facade."""

    @abstractmethod
    async def collect_data(self, data_type: str, data: dict[str, Any]) -> bool:
        """Collect data of specified type."""
        ...

    @abstractmethod
    async def analyze_performance(
        self,
        analysis_type: str,
        parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Perform performance analysis."""
        ...

    @abstractmethod
    async def run_experiment(
        self,
        experiment_config: dict[str, Any]
    ) -> str:
        """Run A/B testing experiment."""
        ...

    @abstractmethod
    async def analyze_sessions(
        self,
        analysis_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze user sessions."""
        ...

    @abstractmethod
    async def monitor_models(
        self,
        model_ids: list[str]
    ) -> dict[str, Any]:
        """Monitor ML model performance."""
        ...

    @abstractmethod
    async def generate_insights(
        self,
        insight_type: str,
        context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate actionable insights."""
        ...


class AnalyticsConfiguration(BaseModel):
    """Configuration for analytics components."""
    component_name: str
    enabled: bool = True
    cache_ttl_seconds: int = 300
    batch_size: int = 100
    memory_limit_mb: int = 500
    performance_targets: dict[str, float] = {
        "max_response_time_ms": 200,
        "min_throughput_rps": 100,
        "max_error_rate": 0.01
    }
    alert_thresholds: dict[str, float] = {
        "high_response_time_ms": 1000,
        "high_error_rate": 0.05,
        "high_memory_usage_mb": 400
    }


class ComponentHealth(BaseModel):
    """Component health status."""
    component_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    response_time_ms: float
    error_rate: float
    memory_usage_mb: float
    alerts: list[str] = []
    details: dict[str, Any] = {}


class AnalyticsInsight(BaseModel):
    """Analytics insight structure."""
    insight_id: str
    insight_type: str
    title: str
    description: str
    confidence_score: float
    impact_level: str  # "low", "medium", "high", "critical"
    recommendations: list[str]
    supporting_data: dict[str, Any]
    timestamp: datetime
    expires_at: datetime | None = None
