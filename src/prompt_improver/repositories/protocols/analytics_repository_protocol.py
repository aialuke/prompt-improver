"""Analytics repository protocol for session analysis and performance metrics.

Defines the interface for analytics data access, including:
- Session performance analysis
- Time-series metrics and trending
- Dashboard aggregations
- Statistical analytics
"""

from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# CLEAN ARCHITECTURE 2025: Use domain DTOs instead of database models
from prompt_improver.core.domain.types import (
    ImprovementSessionData,
    PromptSessionData,
    RuleEffectivenessData,
    UserSatisfactionData,
)


class TimeGranularity(Enum):
    """Time granularity for analytics queries."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MetricType(Enum):
    """Types of metrics for analytics."""

    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    SUCCESS_RATE = "success_rate"
    DURATION = "duration"
    RESOURCE_USAGE = "resource_usage"


class TimeSeriesPoint(BaseModel):
    """Data point in a time series."""

    timestamp: datetime
    value: float
    metadata: dict[str, Any] | None = None


class PerformanceTrend(BaseModel):
    """Performance trend analysis results."""

    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    data_points: list[TimeSeriesPoint]
    statistical_significance: float
    confidence_interval: tuple[float, float]


class SessionAnalytics(BaseModel):
    """Session analytics summary."""

    total_sessions: int
    avg_improvement_score: float
    avg_quality_score: float
    avg_confidence_level: float
    top_performing_rules: list[dict[str, Any]]
    performance_distribution: dict[str, int]


@runtime_checkable
class AnalyticsRepositoryProtocol(Protocol):
    """Protocol for analytics data access operations."""

    # Session Analytics
    async def get_session_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: dict[str, Any] | None = None,
    ) -> SessionAnalytics:
        """Get comprehensive session analytics for a date range."""
        ...

    async def get_prompt_sessions(
        self,
        session_ids: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PromptSessionData]:
        """Retrieve prompt sessions with filters."""
        ...

    async def get_improvement_sessions(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_improvement_score: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ImprovementSessionData]:
        """Retrieve improvement sessions with filters."""
        ...

    # Performance Trending
    async def get_performance_trend(
        self,
        metric_type: MetricType,
        granularity: TimeGranularity,
        start_date: datetime,
        end_date: datetime,
        rule_ids: list[str] | None = None,
    ) -> PerformanceTrend:
        """Get performance trend analysis for specified metrics."""
        ...

    async def get_time_series_data(
        self,
        metric_name: str,
        granularity: TimeGranularity,
        start_date: datetime,
        end_date: datetime,
        aggregation: str = "avg",  # avg, sum, min, max, count
    ) -> list[TimeSeriesPoint]:
        """Get time-series data for any metric."""
        ...

    # Rule Effectiveness Analysis
    async def get_rule_effectiveness_stats(
        self,
        rule_ids: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[RuleEffectivenessData]:
        """Get detailed rule effectiveness statistics."""
        ...

    async def get_rule_performance_comparison(
        self,
        rule_ids: list[str],
        start_date: datetime,
        end_date: datetime,
        metric: str = "improvement_score",
    ) -> dict[str, dict[str, float]]:
        """Compare performance metrics between rules."""
        ...

    # User Satisfaction Analytics
    async def get_user_satisfaction_stats(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        granularity: TimeGranularity = TimeGranularity.DAY,
    ) -> list[UserSatisfactionData]:
        """Get user satisfaction statistics over time."""
        ...

    async def get_satisfaction_correlation(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, float]:
        """Analyze correlation between satisfaction and performance metrics."""
        ...

    # Dashboard Aggregations
    async def get_dashboard_summary(self, period_hours: int = 24) -> dict[str, Any]:
        """Get summary metrics for dashboard display."""
        ...

    async def get_top_performers(
        self, metric: str, period_days: int = 7, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get top performing entities by metric."""
        ...

    async def get_anomaly_detection(
        self, metric_name: str, lookback_hours: int = 24, threshold_stddev: float = 2.0
    ) -> list[dict[str, Any]]:
        """Detect anomalies in metrics using statistical analysis."""
        ...

    # Performance Optimization Insights
    async def get_slow_queries_analysis(
        self,
        min_duration_ms: int = 100,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze slow-performing operations."""
        ...

    async def get_resource_usage_patterns(
        self, granularity: TimeGranularity, start_date: datetime, end_date: datetime
    ) -> dict[str, list[TimeSeriesPoint]]:
        """Get resource usage patterns over time."""
        ...

    # Export and Reporting
    async def export_analytics_data(
        self,
        export_type: str,  # "csv", "json", "parquet"
        start_date: datetime,
        end_date: datetime,
        filters: dict[str, Any] | None = None,
    ) -> bytes:
        """Export analytics data in specified format."""
        ...

    async def generate_analytics_report(
        self,
        report_type: str,  # "summary", "detailed", "comparative"
        start_date: datetime,
        end_date: datetime,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate structured analytics report."""
        ...

    # Advanced Analytics Methods (migrated from analytics_query_interface.py)

    async def get_performance_distribution_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        bucket_count: int = 20,
    ) -> dict[str, Any]:
        """Get performance distribution analysis with histogram data."""
        ...

    async def get_correlation_analysis(
        self,
        metrics: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Get correlation analysis between different session metrics."""
        ...
