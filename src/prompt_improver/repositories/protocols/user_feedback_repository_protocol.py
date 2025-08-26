"""User feedback repository protocol for user satisfaction and feedback management.

Defines the interface for user feedback operations, including:
- Feedback collection and storage
- Satisfaction analysis
- User sentiment tracking
- Feedback correlation with performance metrics
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# Domain models - no database coupling


class FeedbackFilter(BaseModel):
    """Filter criteria for feedback queries."""

    rating_min: int | None = None
    rating_max: int | None = None
    is_processed: bool | None = None
    ml_optimized: bool | None = None
    model_id: str | None = None
    improvement_areas: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    has_text: bool | None = None


class FeedbackAnalysis(BaseModel):
    """Comprehensive feedback analysis results."""

    total_feedback_count: int
    average_rating: float
    rating_distribution: dict[str, int]
    satisfaction_trend: str  # "improving", "declining", "stable"
    common_improvement_areas: list[dict[str, Any]]
    sentiment_analysis: dict[str, float]
    correlation_with_performance: dict[str, float]
    actionable_insights: list[str]


class UserSentimentAnalysis(BaseModel):
    """User sentiment analysis results."""

    session_id: str
    overall_sentiment: str  # "positive", "negative", "neutral"
    sentiment_score: float  # -1.0 to 1.0
    emotion_breakdown: dict[str, float]
    key_phrases: list[str]
    satisfaction_factors: dict[str, float]


class FeedbackCorrelation(BaseModel):
    """Correlation analysis between feedback and performance."""

    correlation_metric: str
    correlation_strength: float
    statistical_significance: float
    confidence_interval: tuple[float, float]
    insights: list[str]


class SatisfactionTrend(BaseModel):
    """Satisfaction trend analysis."""

    period: str
    trend_direction: str
    trend_strength: float
    data_points: list[dict[str, Any]]
    seasonal_patterns: dict[str, Any] | None
    anomalies: list[dict[str, Any]]


@runtime_checkable
class UserFeedbackRepositoryProtocol(Protocol):
    """Protocol for user feedback data access operations."""

    # Feedback Management
    async def create_feedback(self, feedback_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user feedback record."""
        ...

    async def get_feedback_by_id(self, feedback_id: int) -> dict[str, Any] | None:
        """Get feedback record by ID."""
        ...

    async def get_feedback_by_session(self, session_id: str) -> dict[str, Any] | None:
        """Get feedback for specific session."""
        ...

    async def get_feedback_list(
        self,
        filters: FeedbackFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get feedback records with filtering and pagination."""
        ...

    async def update_feedback(
        self, feedback_id: int, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update feedback record."""
        ...

    async def mark_feedback_processed(
        self, feedback_id: int, ml_optimized: bool = False
    ) -> bool:
        """Mark feedback as processed."""
        ...

    async def delete_feedback(self, feedback_id: int) -> bool:
        """Delete feedback record."""
        ...

    # Satisfaction Analysis
    async def get_satisfaction_stats(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        granularity: str = "day",  # "hour", "day", "week", "month"
    ) -> list[dict[str, Any]]:
        """Get satisfaction statistics over time."""
        ...

    async def get_feedback_analysis(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        filters: FeedbackFilter | None = None,
    ) -> FeedbackAnalysis:
        """Get comprehensive feedback analysis."""
        ...

    async def get_rating_distribution(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        group_by: str | None = None,  # "rule_category", "model_id", etc.
    ) -> dict[str, dict[str, int]]:
        """Get rating distribution analysis."""
        ...

    async def get_satisfaction_trend(
        self, days_back: int = 30, granularity: str = "day"
    ) -> SatisfactionTrend:
        """Get satisfaction trend analysis."""
        ...

    # Sentiment Analysis
    async def analyze_feedback_sentiment(
        self, feedback_id: int
    ) -> UserSentimentAnalysis | None:
        """Analyze sentiment for specific feedback."""
        ...

    async def get_sentiment_trends(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        granularity: str = "day",
    ) -> list[dict[str, Any]]:
        """Get sentiment trends over time."""
        ...

    async def get_negative_feedback_analysis(
        self,
        rating_threshold: int = 2,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze negative feedback for improvement opportunities."""
        ...

    # Improvement Area Analysis
    async def get_improvement_area_frequency(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> dict[str, int]:
        """Get frequency of improvement areas mentioned."""
        ...

    async def get_improvement_area_trends(
        self, area: str, days_back: int = 30
    ) -> list[dict[str, Any]]:
        """Track trends for specific improvement area."""
        ...

    async def get_improvement_priorities(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_mentions: int = 5,
    ) -> list[dict[str, Any]]:
        """Get prioritized list of improvement areas."""
        ...

    # Correlation Analysis
    async def correlate_feedback_with_performance(
        self,
        performance_metric: str = "improvement_score",
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> FeedbackCorrelation:
        """Correlate user feedback with performance metrics."""
        ...

    async def correlate_feedback_with_rules(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> dict[str, FeedbackCorrelation]:
        """Correlate feedback with rule effectiveness."""
        ...

    async def analyze_satisfaction_drivers(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> dict[str, float]:
        """Identify key drivers of user satisfaction."""
        ...

    # Advanced Analytics
    async def get_user_journey_analysis(self, session_ids: list[str]) -> dict[str, Any]:
        """Analyze user journey and satisfaction path."""
        ...

    async def predict_user_satisfaction(
        self, session_characteristics: dict[str, Any]
    ) -> dict[str, float]:
        """Predict user satisfaction based on session characteristics."""
        ...

    async def get_feedback_anomalies(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        threshold_stddev: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in feedback patterns."""
        ...

    # Segmentation and Cohort Analysis
    async def segment_users_by_satisfaction(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> dict[str, dict[str, Any]]:
        """Segment users based on satisfaction patterns."""
        ...

    async def get_cohort_satisfaction_analysis(
        self,
        cohort_period: str = "month",  # "week", "month"
        periods_back: int = 6,
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze satisfaction by user cohorts."""
        ...

    # Reporting and Export
    async def generate_satisfaction_report(
        self,
        report_type: str,  # "summary", "detailed", "trend"
        date_from: datetime,
        date_to: datetime,
        include_charts: bool = False,
    ) -> dict[str, Any]:
        """Generate satisfaction analysis report."""
        ...

    async def export_feedback_data(
        self,
        format_type: str,  # "csv", "json", "excel"
        filters: FeedbackFilter | None = None,
        include_analytics: bool = True,
    ) -> bytes:
        """Export feedback data in specified format."""
        ...

    # Maintenance and Cleanup
    async def cleanup_processed_feedback(
        self, days_old: int = 365, keep_negative: bool = True
    ) -> int:
        """Clean up old processed feedback."""
        ...

    async def anonymize_feedback_text(
        self, feedback_ids: list[int] | None = None, days_old: int = 90
    ) -> int:
        """Anonymize feedback text for privacy."""
        ...
