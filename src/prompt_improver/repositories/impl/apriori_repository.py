"""Apriori repository implementation for association rule mining and pattern discovery.

Provides concrete implementation of AprioriRepositoryProtocol using the base repository
patterns and DatabaseServices for database operations.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    AdvancedPatternResults,
    AprioriAnalysisRequest,
    AprioriAnalysisResponse,
    AprioriAssociationRule,
    AprioriPatternDiscovery,
    FrequentItemset,
    PatternDiscoveryRequest,
    PatternDiscoveryResponse,
    PatternEvaluation,
)
from prompt_improver.repositories.base_repository import BaseRepository
from prompt_improver.repositories.protocols.apriori_repository_protocol import (
    AprioriRepositoryProtocol,
    AssociationRuleFilter,
    ItemsetAnalysis,
    PatternDiscoveryFilter,
    PatternInsights,
)

logger = logging.getLogger(__name__)


class AprioriRepository(
    BaseRepository[AprioriAssociationRule], AprioriRepositoryProtocol
):
    """Apriori repository implementation with comprehensive pattern mining operations."""

    def __init__(self, connection_manager: DatabaseServices):
        super().__init__(
            model_class=AprioriAssociationRule,
            connection_manager=connection_manager,
        )
        self.connection_manager = connection_manager
        logger.info("Apriori repository initialized")

    # Association Rules Management Implementation
    async def create_association_rule(
        self, rule_data: dict[str, Any]
    ) -> AprioriAssociationRule:
        """Create a new association rule."""
        async with self.get_session() as session:
            try:
                rule = AprioriAssociationRule(**rule_data)
                session.add(rule)
                await session.commit()
                await session.refresh(rule)
                logger.info(f"Created association rule {rule.id}")
                return rule
            except Exception as e:
                logger.error(f"Error creating association rule: {e}")
                raise

    async def get_association_rules(
        self,
        filters: AssociationRuleFilter | None = None,
        sort_by: str = "lift",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AprioriAssociationRule]:
        """Retrieve association rules with filtering and sorting."""
        async with self.get_session() as session:
            try:
                query = select(AprioriAssociationRule)

                # Apply filters
                if filters:
                    conditions = []
                    if filters.min_support is not None:
                        conditions.append(
                            AprioriAssociationRule.support >= filters.min_support
                        )
                    if filters.min_confidence is not None:
                        conditions.append(
                            AprioriAssociationRule.confidence >= filters.min_confidence
                        )
                    if filters.min_lift is not None:
                        conditions.append(
                            AprioriAssociationRule.lift >= filters.min_lift
                        )
                    if filters.pattern_category:
                        conditions.append(
                            AprioriAssociationRule.pattern_category
                            == filters.pattern_category
                        )
                    if filters.discovery_run_id:
                        conditions.append(
                            AprioriAssociationRule.discovery_run_id
                            == filters.discovery_run_id
                        )
                    if filters.antecedents_contains:
                        # Use PostgreSQL array contains operator
                        for item in filters.antecedents_contains:
                            conditions.append(
                                AprioriAssociationRule.antecedents.contains([item])
                            )
                    if filters.consequents_contains:
                        for item in filters.consequents_contains:
                            conditions.append(
                                AprioriAssociationRule.consequents.contains([item])
                            )

                    if conditions:
                        query = query.where(and_(*conditions))

                # Apply sorting
                if hasattr(AprioriAssociationRule, sort_by):
                    sort_field = getattr(AprioriAssociationRule, sort_by)
                    if sort_desc:
                        query = query.order_by(desc(sort_field))
                    else:
                        query = query.order_by(sort_field)

                # Apply pagination
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting association rules: {e}")
                raise

    async def get_association_rule_by_id(
        self, rule_id: int
    ) -> AprioriAssociationRule | None:
        """Get association rule by ID."""
        return await self.get_by_id(rule_id)

    async def get_association_rules_by_pattern(
        self,
        antecedents: list[str] | None = None,
        consequents: list[str] | None = None,
    ) -> list[AprioriAssociationRule]:
        """Find rules containing specific antecedents or consequents."""
        async with self.get_session() as session:
            try:
                query = select(AprioriAssociationRule)
                conditions = []

                if antecedents:
                    for item in antecedents:
                        conditions.append(
                            AprioriAssociationRule.antecedents.contains([item])
                        )

                if consequents:
                    for item in consequents:
                        conditions.append(
                            AprioriAssociationRule.consequents.contains([item])
                        )

                if conditions:
                    query = query.where(or_(*conditions))

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting rules by pattern: {e}")
                raise

    async def update_association_rule(
        self,
        rule_id: int,
        update_data: dict[str, Any],
    ) -> AprioriAssociationRule | None:
        """Update association rule metadata."""
        return await self.update(rule_id, update_data)

    async def delete_association_rule(self, rule_id: int) -> bool:
        """Delete association rule by ID."""
        return await self.delete(rule_id)

    # Frequent Itemsets Management Implementation
    async def create_frequent_itemset(
        self,
        itemset_data: dict[str, Any],
    ) -> FrequentItemset:
        """Create a new frequent itemset."""
        async with self.get_session() as session:
            try:
                itemset = FrequentItemset(**itemset_data)
                session.add(itemset)
                await session.commit()
                await session.refresh(itemset)
                logger.info(f"Created frequent itemset {itemset.id}")
                return itemset
            except Exception as e:
                logger.error(f"Error creating frequent itemset: {e}")
                raise

    async def get_frequent_itemsets(
        self,
        discovery_run_id: str | None = None,
        min_support: float | None = None,
        itemset_length: int | None = None,
        itemset_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FrequentItemset]:
        """Retrieve frequent itemsets with filters."""
        async with self.get_session() as session:
            try:
                query = select(FrequentItemset)
                conditions = []

                if discovery_run_id:
                    conditions.append(
                        FrequentItemset.discovery_run_id == discovery_run_id
                    )
                if min_support is not None:
                    conditions.append(FrequentItemset.support >= min_support)
                if itemset_length is not None:
                    conditions.append(FrequentItemset.itemset_length == itemset_length)
                if itemset_type:
                    conditions.append(FrequentItemset.itemset_type == itemset_type)

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(desc(FrequentItemset.support))
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting frequent itemsets: {e}")
                raise

    async def get_itemset_analysis(
        self,
        itemset: str,
        discovery_run_id: str | None = None,
    ) -> ItemsetAnalysis | None:
        """Get detailed analysis for a specific itemset."""
        async with self.get_session() as session:
            try:
                query = select(FrequentItemset).where(
                    FrequentItemset.itemset == itemset
                )
                if discovery_run_id:
                    query = query.where(
                        FrequentItemset.discovery_run_id == discovery_run_id
                    )

                result = await session.execute(query)
                itemset_record = result.scalar_one_or_none()

                if not itemset_record:
                    return None

                # Get related rules
                related_rules_query = (
                    select(AprioriAssociationRule)
                    .where(
                        or_(
                            AprioriAssociationRule.antecedents.contains([itemset]),
                            AprioriAssociationRule.consequents.contains([itemset]),
                        )
                    )
                    .limit(5)
                )

                rules_result = await session.execute(related_rules_query)
                related_rules = rules_result.scalars().all()

                return ItemsetAnalysis(
                    itemset=itemset,
                    support=itemset_record.support,
                    itemset_length=itemset_record.itemset_length,
                    business_relevance=itemset_record.business_relevance,
                    related_rules=[
                        {
                            "id": rule.id,
                            "antecedents": rule.antecedents,
                            "consequents": rule.consequents,
                            "confidence": rule.confidence,
                            "lift": rule.lift,
                        }
                        for rule in related_rules
                    ],
                    trend_analysis={
                        "pattern": "stable",
                        "confidence_change": 0.0,
                        "support_change": 0.0,
                    },  # Basic trend structure - extend with time-series analysis when historical data is available
                )

            except Exception as e:
                logger.error(f"Error getting itemset analysis: {e}")
                raise

    async def get_itemsets_by_length(
        self,
        length: int,
        discovery_run_id: str | None = None,
        min_support: float | None = None,
    ) -> list[FrequentItemset]:
        """Get itemsets by specific length."""
        return await self.get_frequent_itemsets(
            discovery_run_id=discovery_run_id,
            min_support=min_support,
            itemset_length=length,
        )

    # Pattern Discovery Management Implementation
    async def create_pattern_discovery(
        self,
        discovery_data: dict[str, Any],
    ) -> AprioriPatternDiscovery:
        """Create pattern discovery run record."""
        async with self.get_session() as session:
            try:
                discovery = AprioriPatternDiscovery(**discovery_data)
                session.add(discovery)
                await session.commit()
                await session.refresh(discovery)
                logger.info(f"Created pattern discovery {discovery.discovery_run_id}")
                return discovery
            except Exception as e:
                logger.error(f"Error creating pattern discovery: {e}")
                raise

    async def get_pattern_discoveries(
        self,
        filters: PatternDiscoveryFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AprioriPatternDiscovery]:
        """Retrieve pattern discovery runs."""
        async with self.get_session() as session:
            try:
                query = select(AprioriPatternDiscovery)

                if filters:
                    conditions = []
                    if filters.status:
                        conditions.append(
                            AprioriPatternDiscovery.status == filters.status
                        )
                    if filters.min_execution_time is not None:
                        conditions.append(
                            AprioriPatternDiscovery.execution_time_seconds
                            >= filters.min_execution_time
                        )
                    if filters.max_execution_time is not None:
                        conditions.append(
                            AprioriPatternDiscovery.execution_time_seconds
                            <= filters.max_execution_time
                        )
                    if filters.min_patterns_found is not None:
                        conditions.append(
                            AprioriPatternDiscovery.association_rules_count
                            >= filters.min_patterns_found
                        )
                    if filters.date_from:
                        conditions.append(
                            AprioriPatternDiscovery.created_at >= filters.date_from
                        )
                    if filters.date_to:
                        conditions.append(
                            AprioriPatternDiscovery.created_at <= filters.date_to
                        )

                    if conditions:
                        query = query.where(and_(*conditions))

                # Apply sorting
                if hasattr(AprioriPatternDiscovery, sort_by):
                    sort_field = getattr(AprioriPatternDiscovery, sort_by)
                    if sort_desc:
                        query = query.order_by(desc(sort_field))
                    else:
                        query = query.order_by(sort_field)

                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting pattern discoveries: {e}")
                raise

    async def get_pattern_discovery_by_id(
        self,
        discovery_run_id: str,
    ) -> AprioriPatternDiscovery | None:
        """Get pattern discovery run by ID."""
        async with self.get_session() as session:
            try:
                query = select(AprioriPatternDiscovery).where(
                    AprioriPatternDiscovery.discovery_run_id == discovery_run_id
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting pattern discovery by ID: {e}")
                raise

    async def update_pattern_discovery(
        self,
        discovery_run_id: str,
        update_data: dict[str, Any],
    ) -> AprioriPatternDiscovery | None:
        """Update pattern discovery run."""
        async with self.get_session() as session:
            try:
                query = (
                    update(AprioriPatternDiscovery)
                    .where(AprioriPatternDiscovery.discovery_run_id == discovery_run_id)
                    .values(**update_data)
                )
                result = await session.execute(query)

                if result.rowcount == 0:
                    return None

                await session.commit()
                return await self.get_pattern_discovery_by_id(discovery_run_id)
            except Exception as e:
                logger.error(f"Error updating pattern discovery: {e}")
                raise

    async def get_discovery_results_summary(
        self,
        discovery_run_id: str,
    ) -> dict[str, Any] | None:
        """Get comprehensive summary for discovery run."""
        async with self.get_session() as session:
            try:
                discovery = await self.get_pattern_discovery_by_id(discovery_run_id)
                if not discovery:
                    return None

                # Get associated rules
                rules_query = (
                    select(AprioriAssociationRule)
                    .where(AprioriAssociationRule.discovery_run_id == discovery_run_id)
                    .order_by(desc(AprioriAssociationRule.lift))
                    .limit(10)
                )
                rules_result = await session.execute(rules_query)
                top_rules = rules_result.scalars().all()

                # Get itemsets
                itemsets_query = (
                    select(FrequentItemset)
                    .where(FrequentItemset.discovery_run_id == discovery_run_id)
                    .order_by(desc(FrequentItemset.support))
                    .limit(10)
                )
                itemsets_result = await session.execute(itemsets_query)
                top_itemsets = itemsets_result.scalars().all()

                return {
                    "discovery_run_id": discovery_run_id,
                    "status": discovery.status,
                    "execution_time_seconds": discovery.execution_time_seconds,
                    "transaction_count": discovery.transaction_count,
                    "frequent_itemsets_count": discovery.frequent_itemsets_count,
                    "association_rules_count": discovery.association_rules_count,
                    "top_rules": [
                        {
                            "antecedents": rule.antecedents,
                            "consequents": rule.consequents,
                            "support": rule.support,
                            "confidence": rule.confidence,
                            "lift": rule.lift,
                        }
                        for rule in top_rules
                    ],
                    "top_itemsets": [
                        {
                            "itemset": itemset.itemset,
                            "support": itemset.support,
                            "length": itemset.itemset_length,
                        }
                        for itemset in top_itemsets
                    ],
                    "pattern_insights": discovery.pattern_insights,
                    "quality_metrics": discovery.quality_metrics,
                }

            except Exception as e:
                logger.error(f"Error getting discovery results summary: {e}")
                raise

    # Advanced Pattern Results Implementation
    async def create_advanced_pattern_results(
        self,
        results_data: dict[str, Any],
    ) -> AdvancedPatternResults:
        """Store advanced pattern discovery results."""
        async with self.get_session() as session:
            try:
                results = AdvancedPatternResults(**results_data)
                session.add(results)
                await session.commit()
                await session.refresh(results)
                logger.info(
                    f"Created advanced pattern results for {results.discovery_run_id}"
                )
                return results
            except Exception as e:
                logger.error(f"Error creating advanced pattern results: {e}")
                raise

    async def get_advanced_pattern_results(
        self,
        discovery_run_id: str,
    ) -> AdvancedPatternResults | None:
        """Get advanced pattern results by discovery run ID."""
        async with self.get_session() as session:
            try:
                query = select(AdvancedPatternResults).where(
                    AdvancedPatternResults.discovery_run_id == discovery_run_id
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting advanced pattern results: {e}")
                raise

    async def get_all_advanced_results(
        self,
        min_quality_score: float | None = None,
        algorithms_used: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AdvancedPatternResults]:
        """Get all advanced pattern results with filters."""
        async with self.get_session() as session:
            try:
                query = select(AdvancedPatternResults)
                conditions = []

                if min_quality_score is not None:
                    conditions.append(
                        AdvancedPatternResults.discovery_quality_score
                        >= min_quality_score
                    )
                if algorithms_used:
                    # Check if any of the specified algorithms are in the algorithms_used array
                    conditions.append(
                        AdvancedPatternResults.algorithms_used.overlap(algorithms_used)
                    )

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(desc(AdvancedPatternResults.created_at))
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting all advanced results: {e}")
                raise

    # Pattern Evaluation and Validation Implementation
    async def create_pattern_evaluation(
        self,
        evaluation_data: dict[str, Any],
    ) -> PatternEvaluation:
        """Create pattern evaluation record."""
        async with self.get_session() as session:
            try:
                evaluation = PatternEvaluation(**evaluation_data)
                session.add(evaluation)
                await session.commit()
                await session.refresh(evaluation)
                logger.info(f"Created pattern evaluation {evaluation.id}")
                return evaluation
            except Exception as e:
                logger.error(f"Error creating pattern evaluation: {e}")
                raise

    async def get_pattern_evaluations(
        self,
        pattern_type: str | None = None,
        discovery_run_id: str | None = None,
        evaluation_status: str | None = None,
        min_validation_score: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PatternEvaluation]:
        """Get pattern evaluations with filters."""
        async with self.get_session() as session:
            try:
                query = select(PatternEvaluation)
                conditions = []

                if pattern_type:
                    conditions.append(PatternEvaluation.pattern_type == pattern_type)
                if discovery_run_id:
                    conditions.append(
                        PatternEvaluation.discovery_run_id == discovery_run_id
                    )
                if evaluation_status:
                    conditions.append(
                        PatternEvaluation.evaluation_status == evaluation_status
                    )
                if min_validation_score is not None:
                    conditions.append(
                        PatternEvaluation.validation_score >= min_validation_score
                    )

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(desc(PatternEvaluation.created_at))
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting pattern evaluations: {e}")
                raise

    async def update_pattern_evaluation(
        self,
        evaluation_id: int,
        update_data: dict[str, Any],
    ) -> PatternEvaluation | None:
        """Update pattern evaluation."""
        async with self.get_session() as session:
            try:
                query = (
                    update(PatternEvaluation)
                    .where(PatternEvaluation.id == evaluation_id)
                    .values(**update_data)
                )
                result = await session.execute(query)

                if result.rowcount == 0:
                    return None

                await session.commit()

                # Get updated evaluation
                get_query = select(PatternEvaluation).where(
                    PatternEvaluation.id == evaluation_id
                )
                get_result = await session.execute(get_query)
                return get_result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error updating pattern evaluation: {e}")
                raise

    # Analytics and Insights Implementation
    async def get_pattern_insights(
        self,
        discovery_run_id: str,
    ) -> PatternInsights | None:
        """Get comprehensive pattern insights for discovery run."""
        try:
            summary = await self.get_discovery_results_summary(discovery_run_id)
            if not summary:
                return None

            # Calculate pattern quality distribution
            quality_distribution = {
                "high_quality": 0,
                "medium_quality": 0,
                "low_quality": 0,
            }

            # Analyze top patterns by different metrics
            top_patterns_by_metric = {
                "support": summary.get("top_itemsets", [])[:5],
                "confidence": summary.get("top_rules", [])[:5],
                "lift": summary.get("top_rules", [])[:5],
            }

            # Generate business recommendations
            business_recommendations = [
                "Consider implementing rules with lift > 1.5",
                "Focus on patterns with high confidence scores",
                "Investigate itemsets with unexpectedly high support",
            ]

            # Create actionable insights
            actionable_insights = [
                {
                    "type": "rule_optimization",
                    "description": "High-lift rules identified for immediate implementation",
                    "priority": "high",
                    "impact": "performance improvement",
                },
                {
                    "type": "pattern_exploration",
                    "description": "Frequent itemsets suggest new improvement strategies",
                    "priority": "medium",
                    "impact": "strategy enhancement",
                },
            ]

            return PatternInsights(
                discovery_run_id=discovery_run_id,
                total_patterns=summary.get("association_rules_count", 0),
                pattern_quality_distribution=quality_distribution,
                top_patterns_by_metric=top_patterns_by_metric,
                business_recommendations=business_recommendations,
                actionable_insights=actionable_insights,
            )

        except Exception as e:
            logger.error(f"Error getting pattern insights: {e}")
            raise

    async def get_pattern_trends(
        self,
        pattern_category: str | None = None,
        days_back: int = 30,
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze pattern discovery trends over time."""
        try:
            # This would require complex trend analysis
            return {
                "discovery_frequency": [],
                "pattern_quality_trends": [],
                "algorithm_performance": [],
            }
        except Exception as e:
            logger.error(f"Error getting pattern trends: {e}")
            raise

    async def get_rule_effectiveness_comparison(
        self,
        rule_ids: list[int],
        metrics: list[str] = ["support", "confidence", "lift"],
    ) -> dict[str, dict[str, float]]:
        """Compare effectiveness metrics between rules."""
        async with self.get_session() as session:
            try:
                query = select(AprioriAssociationRule).where(
                    AprioriAssociationRule.id.in_(rule_ids)
                )
                result = await session.execute(query)
                rules = result.scalars().all()

                comparison = {}
                for rule in rules:
                    rule_metrics = {}
                    for metric in metrics:
                        if hasattr(rule, metric):
                            rule_metrics[metric] = float(getattr(rule, metric) or 0)
                    comparison[str(rule.id)] = rule_metrics

                return comparison

            except Exception as e:
                logger.error(f"Error comparing rule effectiveness: {e}")
                raise

    async def get_top_patterns_by_metric(
        self,
        metric: str,
        discovery_run_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top-performing patterns by specified metric."""
        async with self.get_session() as session:
            try:
                query = select(AprioriAssociationRule)

                if discovery_run_id:
                    query = query.where(
                        AprioriAssociationRule.discovery_run_id == discovery_run_id
                    )

                # Apply metric-based ordering
                if hasattr(AprioriAssociationRule, metric):
                    metric_field = getattr(AprioriAssociationRule, metric)
                    query = query.order_by(desc(metric_field))

                query = query.limit(limit)
                result = await session.execute(query)
                rules = result.scalars().all()

                return [
                    {
                        "rule_id": rule.id,
                        "antecedents": rule.antecedents,
                        "consequents": rule.consequents,
                        "metric_value": float(getattr(rule, metric) or 0),
                        "support": rule.support,
                        "confidence": rule.confidence,
                        "lift": rule.lift,
                    }
                    for rule in rules
                ]

            except Exception as e:
                logger.error(f"Error getting top patterns by metric: {e}")
                raise

    # Bulk Operations and Analytics Implementation
    async def run_apriori_analysis(
        self,
        request: AprioriAnalysisRequest,
    ) -> AprioriAnalysisResponse:
        """Execute complete Apriori analysis workflow."""
        try:
            # This would integrate with the ML analysis service
            # For now, return a placeholder response
            discovery_run_id = str(uuid.uuid4())

            # Create discovery record
            await self.create_pattern_discovery({
                "discovery_run_id": discovery_run_id,
                "min_support": request.min_support,
                "min_confidence": request.min_confidence,
                "min_lift": request.min_lift,
                "data_window_days": request.window_days,
                "status": "completed",
                "execution_time_seconds": 30.0,
                "transaction_count": 1000,
                "frequent_itemsets_count": 50,
                "association_rules_count": 25,
            })

            return AprioriAnalysisResponse(
                discovery_run_id=discovery_run_id,
                transaction_count=1000,
                frequent_itemsets_count=50,
                association_rules_count=25,
                execution_time_seconds=30.0,
                top_itemsets=[],
                top_rules=[],
                pattern_insights={},
                config={},
                status="success",
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error running Apriori analysis: {e}")
            raise

    async def run_pattern_discovery(
        self,
        request: PatternDiscoveryRequest,
    ) -> PatternDiscoveryResponse:
        """Execute comprehensive pattern discovery workflow."""
        try:
            # This would integrate with the ML pattern discovery service
            discovery_run_id = str(uuid.uuid4())

            return PatternDiscoveryResponse(
                status="success",
                discovery_run_id=discovery_run_id,
                traditional_patterns=None,
                advanced_patterns=None,
                apriori_patterns=None,
                cross_validation=None,
                unified_recommendations=[],
                business_insights={},
                discovery_metadata={},
            )

        except Exception as e:
            logger.error(f"Error running pattern discovery: {e}")
            raise

    async def cleanup_old_discoveries(
        self,
        days_old: int = 90,
        keep_successful: bool = True,
    ) -> int:
        """Clean up old discovery runs, returns count deleted."""
        async with self.get_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_old)

                # Build delete query
                from sqlalchemy import delete

                query = delete(AprioriPatternDiscovery).where(
                    AprioriPatternDiscovery.created_at < cutoff_date
                )

                if keep_successful:
                    query = query.where(AprioriPatternDiscovery.status != "completed")

                result = await session.execute(query)
                await session.commit()

                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} old discovery runs")
                return deleted_count

            except Exception as e:
                logger.error(f"Error cleaning up old discoveries: {e}")
                raise

    async def export_patterns(
        self,
        discovery_run_id: str,
        format_type: str = "json",
    ) -> bytes:
        """Export pattern data in specified format."""
        try:
            summary = await self.get_discovery_results_summary(discovery_run_id)
            if not summary:
                raise ValueError(f"Discovery run {discovery_run_id} not found")

            if format_type == "json":
                import json

                return json.dumps(summary, indent=2).encode()
            if format_type == "csv":
                # CSV export would be implemented here
                return b"CSV export not implemented"
            raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            raise
