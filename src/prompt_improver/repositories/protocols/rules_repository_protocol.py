"""Rules repository protocol for rule engine data access operations.

Defines the interface for rule management operations, including:
- Rule metadata management
- Rule performance tracking
- Rule intelligence caching
- Rule effectiveness analysis
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# Domain models - no database coupling


class RuleFilter(BaseModel):
    """Filter criteria for rule queries."""

    enabled: bool | None = None
    category: str | None = None
    min_priority: int | None = None
    max_priority: int | None = None
    rule_version: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None


class RulePerformanceFilter(BaseModel):
    """Filter criteria for rule performance queries."""

    rule_ids: list[str] | None = None
    prompt_type: str | None = None
    prompt_category: str | None = None
    min_improvement_score: float | None = None
    min_confidence_level: float | None = None
    max_execution_time_ms: int | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class RuleEffectivenessAnalysis(BaseModel):
    """Rule effectiveness analysis results."""

    rule_id: str
    rule_name: str
    total_applications: int
    avg_improvement_score: float
    improvement_score_stddev: float
    avg_confidence_level: float
    avg_execution_time_ms: float
    success_rate: float
    trend_analysis: dict[str, float]
    performance_by_category: dict[str, dict[str, float]]
    recommendations: list[str]


class RuleIntelligenceMetrics(BaseModel):
    """Rule intelligence cache metrics."""

    cache_key: str
    rule_id: str
    effectiveness_score: float
    total_score: float
    confidence_level: float
    sample_size: int
    performance_trend: str | None
    pattern_insights: dict[str, Any] | None
    last_updated: datetime


class RuleComparisonResult(BaseModel):
    """Results of comparing rule effectiveness."""

    rule_comparisons: list[dict[str, Any]]
    statistical_significance: dict[str, float]
    recommendations: list[str]
    best_performers: list[str]
    underperformers: list[str]


@runtime_checkable
class RulesRepositoryProtocol(Protocol):
    """Protocol for rule engine data access operations."""

    # Rule Metadata Management
    async def create_rule(self, rule_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new rule metadata record."""
        ...

    async def get_rules(
        self,
        filters: RuleFilter | None = None,
        sort_by: str = "priority",
        sort_desc: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve rules with filtering and sorting."""
        ...

    async def get_rule_by_id(self, rule_id: str) -> dict[str, Any] | None:
        """Get rule metadata by rule ID."""
        ...

    async def get_rules_by_category(
        self, category: str, enabled_only: bool = True
    ) -> list[dict[str, Any]]:
        """Get all rules in a specific category."""
        ...

    async def update_rule(
        self, rule_id: str, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update rule metadata."""
        ...

    async def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        ...

    async def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        ...

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule and its associated data."""
        ...

    # Rule Performance Tracking
    async def create_rule_performance(
        self, performance_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Record rule performance metrics."""
        ...

    async def get_rule_performances(
        self,
        filters: RulePerformanceFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get rule performance records with filtering."""
        ...

    async def get_performance_by_rule(
        self,
        rule_id: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get performance history for specific rule."""
        ...

    async def get_recent_performances(
        self, rule_ids: list[str] | None = None, hours_back: int = 24, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get recent rule performance data."""
        ...

    # Rule Effectiveness Analysis
    async def get_rule_effectiveness_analysis(
        self,
        rule_id: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> RuleEffectivenessAnalysis | None:
        """Get comprehensive effectiveness analysis for rule."""
        ...

    async def get_top_performing_rules(
        self,
        metric: str = "improvement_score",
        category: str | None = None,
        min_applications: int = 10,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get top performing rules by metric."""
        ...

    async def get_underperforming_rules(
        self,
        metric: str = "improvement_score",
        threshold: float = 0.5,
        min_applications: int = 10,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Identify underperforming rules."""
        ...

    async def compare_rule_effectiveness(
        self,
        rule_ids: list[str],
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> RuleComparisonResult:
        """Compare effectiveness between multiple rules."""
        ...

    # Rule Intelligence Cache Management
    async def create_rule_intelligence_cache(
        self, cache_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create rule intelligence cache entry."""
        ...

    async def get_rule_intelligence_cache(
        self, cache_key: str
    ) -> dict[str, Any] | None:
        """Get cached rule intelligence by key."""
        ...

    async def get_intelligence_by_rule(
        self,
        rule_id: str,
        prompt_characteristics_hash: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get cached intelligence for specific rule."""
        ...

    async def update_rule_intelligence_cache(
        self, cache_key: str, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update rule intelligence cache."""
        ...

    async def cleanup_expired_cache(self) -> int:
        """Clean up expired intelligence cache entries."""
        ...

    async def get_intelligence_metrics(
        self, rule_id: str
    ) -> RuleIntelligenceMetrics | None:
        """Get intelligence metrics for rule."""
        ...

    # Rule Performance Analytics
    async def get_rule_performance_trends(
        self,
        rule_ids: list[str] | None = None,
        days_back: int = 30,
        granularity: str = "day",  # "hour", "day", "week"
    ) -> dict[str, list[dict[str, Any]]]:
        """Get performance trends over time."""
        ...

    async def get_rule_usage_statistics(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> dict[str, Any]:
        """Get overall rule usage statistics."""
        ...

    async def get_performance_correlations(
        self,
        rule_ids: list[str],
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, dict[str, float]]:
        """Analyze correlations between rule performances."""
        ...

    async def get_category_performance_summary(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> dict[str, dict[str, Any]]:
        """Get performance summary by rule category."""
        ...

    # Rule Configuration and Optimization
    async def get_optimal_rule_parameters(
        self, rule_id: str, prompt_characteristics: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Get optimal parameters for rule based on performance data."""
        ...

    async def get_rule_combination_recommendations(
        self, target_categories: list[str], max_rules: int = 5
    ) -> list[list[str]]:
        """Get recommended rule combinations for categories."""
        ...

    async def analyze_rule_conflicts(
        self, rule_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze potential conflicts between rules."""
        ...

    # Bulk Operations and Maintenance
    async def batch_update_rule_priorities(
        self, rule_priority_updates: dict[str, int]
    ) -> int:
        """Update priorities for multiple rules."""
        ...

    async def archive_old_performances(
        self, days_old: int = 90, keep_summaries: bool = True
    ) -> int:
        """Archive old performance data."""
        ...

    async def recalculate_effectiveness_scores(
        self, rule_ids: list[str] | None = None
    ) -> int:
        """Recalculate effectiveness scores for rules."""
        ...
