"""Prompt repository protocol for prompt improvement session management.

Defines the interface for prompt-specific data access operations, including:
- Improvement session management
- A/B experiment management
- Discovered pattern management
- Cross-domain analytics for optimization
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# CLEAN ARCHITECTURE 2025: Use domain DTOs instead of database models
from prompt_improver.core.domain.types import (
    ABExperimentData,
    DiscoveredPatternData,
    ImprovementSessionCreateData,
    ImprovementSessionData,
)


class SessionFilter(BaseModel):
    """Filter criteria for session queries."""

    session_id: str | None = None
    user_context_contains: str | None = None
    min_quality_score: float | None = None
    min_improvement_score: float | None = None
    min_confidence_level: float | None = None
    has_user_feedback: bool | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class ExperimentFilter(BaseModel):
    """Filter criteria for A/B experiment queries."""

    status: str | None = None
    target_metric: str | None = None
    min_sample_size: int | None = None
    started_after: datetime | None = None
    completed_before: datetime | None = None


class PatternFilter(BaseModel):
    """Filter criteria for discovered pattern queries."""

    pattern_type: str | None = None
    min_effectiveness: float | None = None
    min_support_count: int | None = None
    discovery_run_id: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class SessionAnalytics(BaseModel):
    """Analytics results for improvement sessions."""

    total_sessions: int
    avg_improvement_score: float
    avg_confidence_level: float
    sessions_with_feedback: int
    most_common_rules_applied: list[dict[str, Any]]
    improvement_trends: dict[str, list[dict[str, Any]]]
    user_satisfaction_correlation: dict[str, float]


class OptimizationData(BaseModel):
    """Data for ML optimization operations."""

    features: list[list[float]]
    effectiveness_scores: list[float]
    rule_ids: list[str] | None
    metadata: dict[str, Any]


@runtime_checkable
class PromptRepositoryProtocol(Protocol):
    """Protocol for prompt improvement session and experiment data access."""

    # Session Management
    async def create_session(
        self, session_data: ImprovementSessionCreateData
    ) -> ImprovementSessionData:
        """Create a new improvement session."""
        ...

    async def get_session(self, session_id: str) -> ImprovementSessionData | None:
        """Get session by session ID."""
        ...

    async def get_sessions(
        self,
        filters: SessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ImprovementSessionData]:
        """Get sessions with filtering and pagination."""
        ...

    async def update_session(
        self, session_id: str, update_data: dict[str, Any]
    ) -> ImprovementSessionData | None:
        """Update session data."""
        ...

    async def get_session_analytics(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        filters: SessionFilter | None = None,
    ) -> SessionAnalytics:
        """Get comprehensive session analytics."""
        ...

    async def get_recent_sessions(
        self, hours_back: int = 24, limit: int = 50
    ) -> list[ImprovementSessionData]:
        """Get recent improvement sessions."""
        ...

    # A/B Experiment Management
    async def create_experiment(self, experiment_data: dict[str, Any]) -> ABExperimentData:
        """Create a new A/B experiment."""
        ...

    async def get_experiment(self, experiment_id: str) -> ABExperimentData | None:
        """Get experiment by ID."""
        ...

    async def get_experiments(
        self,
        filters: ExperimentFilter | None = None,
        sort_by: str = "started_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ABExperimentData]:
        """Get experiments with filtering."""
        ...

    async def update_experiment(
        self, experiment_id: str, update_data: dict[str, Any]
    ) -> ABExperimentData | None:
        """Update experiment data."""
        ...

    async def get_active_experiments(
        self, target_metric: str | None = None
    ) -> list[ABExperimentData]:
        """Get currently running experiments."""
        ...

    async def complete_experiment(
        self, experiment_id: str, results: dict[str, Any]
    ) -> ABExperimentData | None:
        """Mark experiment as completed with results."""
        ...

    # Discovered Pattern Management
    async def create_pattern(self, pattern_data: dict[str, Any]) -> DiscoveredPatternData:
        """Store a discovered pattern."""
        ...

    async def get_pattern(self, pattern_id: str) -> DiscoveredPatternData | None:
        """Get pattern by ID."""
        ...

    async def get_patterns(
        self,
        filters: PatternFilter | None = None,
        sort_by: str = "avg_effectiveness",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DiscoveredPatternData]:
        """Get patterns with filtering."""
        ...

    async def get_effective_patterns(
        self, min_effectiveness: float = 0.7, min_support: int = 5, limit: int = 20
    ) -> list[DiscoveredPatternData]:
        """Get most effective discovered patterns."""
        ...

    async def update_pattern(
        self, pattern_id: str, update_data: dict[str, Any]
    ) -> DiscoveredPatternData | None:
        """Update pattern data."""
        ...

    # Cross-Domain Analytics for Optimization
    async def get_optimization_training_data(
        self,
        rule_ids: list[str] | None = None,
        min_samples: int = 20,
        lookback_days: int = 30,
        include_synthetic: bool = True,
        synthetic_ratio: float = 0.3,
    ) -> OptimizationData:
        """Get training data for ML optimization."""
        ...

    async def get_rule_performance_with_metadata(
        self,
        rule_ids: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        include_parameters: bool = True,
    ) -> list[dict[str, Any]]:
        """Get rule performance data with metadata for optimization."""
        ...

    async def analyze_session_effectiveness(
        self,
        session_ids: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze overall session effectiveness for optimization."""
        ...

    async def get_pattern_experiment_candidates(
        self,
        min_effectiveness: float = 0.75,
        max_existing_experiments: int = 3,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get patterns that are candidates for A/B experiments."""
        ...

    # Cross-Domain Queries for Prompt Improvement Service
    async def get_sessions_with_feedback_and_performance(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_rating: int = 3,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get sessions with both user feedback and rule performance data."""
        ...

    async def get_rule_effectiveness_by_session_context(
        self,
        context_filters: dict[str, Any] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Analyze rule effectiveness by session context."""
        ...

    # Bulk Operations and Maintenance
    async def archive_old_sessions(
        self, days_old: int = 90, keep_successful_sessions: bool = True
    ) -> int:
        """Archive old sessions."""
        ...

    async def cleanup_incomplete_sessions(self, hours_old: int = 24) -> int:
        """Clean up incomplete/orphaned sessions."""
        ...

    async def batch_update_session_scores(
        self, score_updates: dict[str, dict[str, float]]
    ) -> int:
        """Batch update session quality/improvement scores."""
        ...

    # Health and Diagnostics
    async def health_check(self) -> dict[str, Any]:
        """Perform repository health check."""
        ...

    async def get_connection_info(self) -> dict[str, Any]:
        """Get database connection information."""
        ...
