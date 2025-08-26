"""Apriori repository protocol for association rule mining and pattern discovery.

Defines the interface for Apriori-specific data access operations, including:
- Association rule management
- Frequent itemset storage and retrieval
- Pattern discovery tracking
- Advanced pattern analysis results
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# CLEAN ARCHITECTURE 2025: Use domain DTOs instead of database models
from prompt_improver.core.domain.types import (
    AdvancedPatternResultsData,
    AprioriAnalysisRequestData,
    AprioriAnalysisResponseData,
    AprioriAssociationRuleData,
    AprioriPatternDiscoveryData,
    FrequentItemsetData,
    PatternDiscoveryRequestData,
    PatternDiscoveryResponseData,
    PatternEvaluationData,
)


class AssociationRuleFilter(BaseModel):
    """Filter criteria for association rule queries."""

    min_support: float | None = None
    min_confidence: float | None = None
    min_lift: float | None = None
    pattern_category: str | None = None
    discovery_run_id: str | None = None
    antecedents_contains: list[str] | None = None
    consequents_contains: list[str] | None = None


class PatternDiscoveryFilter(BaseModel):
    """Filter criteria for pattern discovery queries."""

    status: str | None = None
    min_execution_time: float | None = None
    max_execution_time: float | None = None
    min_patterns_found: int | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class ItemsetAnalysis(BaseModel):
    """Itemset analysis results."""

    itemset: str
    support: float
    itemset_length: int
    business_relevance: str | None
    related_rules: list[dict[str, Any]]
    trend_analysis: dict[str, float] | None


class PatternInsights(BaseModel):
    """Comprehensive pattern insights."""

    discovery_run_id: str
    total_patterns: int
    pattern_quality_distribution: dict[str, int]
    top_patterns_by_metric: dict[str, list[dict[str, Any]]]
    business_recommendations: list[str]
    actionable_insights: list[dict[str, Any]]


@runtime_checkable
class AprioriRepositoryProtocol(Protocol):
    """Protocol for Apriori pattern mining data access operations."""

    # Association Rules Management
    async def create_association_rule(
        self, rule_data: dict[str, Any]
    ) -> AprioriAssociationRuleData:
        """Create a new association rule."""
        ...

    async def get_association_rules(
        self,
        filters: AssociationRuleFilter | None = None,
        sort_by: str = "lift",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AprioriAssociationRuleData]:
        """Retrieve association rules with filtering and sorting."""
        ...

    async def get_association_rule_by_id(
        self, rule_id: int
    ) -> AprioriAssociationRuleData | None:
        """Get association rule by ID."""
        ...

    async def get_association_rules_by_pattern(
        self, antecedents: list[str] | None = None, consequents: list[str] | None = None
    ) -> list[AprioriAssociationRuleData]:
        """Find rules containing specific antecedents or consequents."""
        ...

    async def update_association_rule(
        self, rule_id: int, update_data: dict[str, Any]
    ) -> AprioriAssociationRuleData | None:
        """Update association rule metadata."""
        ...

    async def delete_association_rule(self, rule_id: int) -> bool:
        """Delete association rule by ID."""
        ...

    # Frequent Itemsets Management
    async def create_frequent_itemset(
        self, itemset_data: dict[str, Any]
    ) -> FrequentItemsetData:
        """Create a new frequent itemset."""
        ...

    async def get_frequent_itemsets(
        self,
        discovery_run_id: str | None = None,
        min_support: float | None = None,
        itemset_length: int | None = None,
        itemset_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FrequentItemsetData]:
        """Retrieve frequent itemsets with filters."""
        ...

    async def get_itemset_analysis(
        self, itemset: str, discovery_run_id: str | None = None
    ) -> ItemsetAnalysis | None:
        """Get detailed analysis for a specific itemset."""
        ...

    async def get_itemsets_by_length(
        self,
        length: int,
        discovery_run_id: str | None = None,
        min_support: float | None = None,
    ) -> list[FrequentItemsetData]:
        """Get itemsets by specific length."""
        ...

    # Pattern Discovery Management
    async def create_pattern_discovery(
        self, discovery_data: dict[str, Any]
    ) -> AprioriPatternDiscoveryData:
        """Create pattern discovery run record."""
        ...

    async def get_pattern_discoveries(
        self,
        filters: PatternDiscoveryFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AprioriPatternDiscoveryData]:
        """Retrieve pattern discovery runs."""
        ...

    async def get_pattern_discovery_by_id(
        self, discovery_run_id: str
    ) -> AprioriPatternDiscoveryData | None:
        """Get pattern discovery run by ID."""
        ...

    async def update_pattern_discovery(
        self, discovery_run_id: str, update_data: dict[str, Any]
    ) -> AprioriPatternDiscoveryData | None:
        """Update pattern discovery run."""
        ...

    async def get_discovery_results_summary(
        self, discovery_run_id: str
    ) -> dict[str, Any] | None:
        """Get comprehensive summary for discovery run."""
        ...

    # Advanced Pattern Results
    async def create_advanced_pattern_results(
        self, results_data: dict[str, Any]
    ) -> AdvancedPatternResultsData:
        """Store advanced pattern discovery results."""
        ...

    async def get_advanced_pattern_results(
        self, discovery_run_id: str
    ) -> AdvancedPatternResultsData | None:
        """Get advanced pattern results by discovery run ID."""
        ...

    async def get_all_advanced_results(
        self,
        min_quality_score: float | None = None,
        algorithms_used: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AdvancedPatternResultsData]:
        """Get all advanced pattern results with filters."""
        ...

    # Pattern Evaluation and Validation
    async def create_pattern_evaluation(
        self, evaluation_data: dict[str, Any]
    ) -> PatternEvaluationData:
        """Create pattern evaluation record."""
        ...

    async def get_pattern_evaluations(
        self,
        pattern_type: str | None = None,
        discovery_run_id: str | None = None,
        evaluation_status: str | None = None,
        min_validation_score: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PatternEvaluationData]:
        """Get pattern evaluations with filters."""
        ...

    async def update_pattern_evaluation(
        self, evaluation_id: int, update_data: dict[str, Any]
    ) -> PatternEvaluationData | None:
        """Update pattern evaluation."""
        ...

    # Analytics and Insights
    async def get_pattern_insights(
        self, discovery_run_id: str
    ) -> PatternInsights | None:
        """Get comprehensive pattern insights for discovery run."""
        ...

    async def get_pattern_trends(
        self, pattern_category: str | None = None, days_back: int = 30
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze pattern discovery trends over time."""
        ...

    async def get_rule_effectiveness_comparison(
        self,
        rule_ids: list[int],
        metrics: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare effectiveness metrics between rules."""
        ...

    async def get_top_patterns_by_metric(
        self,
        metric: str,  # "support", "confidence", "lift", "conviction"
        discovery_run_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top-performing patterns by specified metric."""
        ...

    # Bulk Operations and Analytics
    async def run_apriori_analysis(
        self, request: AprioriAnalysisRequestData
    ) -> AprioriAnalysisResponseData:
        """Execute complete Apriori analysis workflow."""
        ...

    async def run_pattern_discovery(
        self, request: PatternDiscoveryRequestData
    ) -> PatternDiscoveryResponseData:
        """Execute comprehensive pattern discovery workflow."""
        ...

    async def cleanup_old_discoveries(
        self, days_old: int = 90, keep_successful: bool = True
    ) -> int:
        """Clean up old discovery runs, returns count deleted."""
        ...

    async def export_patterns(
        self,
        discovery_run_id: str,
        format_type: str = "json",  # "json", "csv", "xml"
    ) -> bytes:
        """Export pattern data in specified format."""
        ...
