"""ML Pattern Discovery Service for advanced pattern analysis.

Combines traditional ML pattern discovery with Apriori association rules
and advanced pattern discovery algorithms for comprehensive insights.
"""

import json
import logging
import time
from typing import Any, Dict, List

# import numpy as np  # Converted to lazy loading
from prompt_improver.core.utils.lazy_ml_loader import get_numpy
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.models import RuleMetadata, RulePerformance
from prompt_improver.utils.datetime_utils import aware_utc_now
from .protocols import PatternDiscoveryServiceProtocol

logger = logging.getLogger(__name__)


class MLPatternDiscoveryService(PatternDiscoveryServiceProtocol):
    """Service for ML-based pattern discovery and analysis."""

    def __init__(self, db_manager=None):
        """Initialize pattern discovery service.
        
        Args:
            db_manager: Database manager for data access
        """
        self.db_manager = db_manager
        
        # Initialize advanced pattern discovery with proper sync database manager
        self.pattern_discovery = None
        if db_manager:
            self._initialize_advanced_discovery()
            
        logger.info("ML Pattern Discovery Service initialized")

    def _initialize_advanced_discovery(self):
        """Initialize advanced pattern discovery component."""
        try:
            # Lazy import to break circular dependency
            from ..learning.patterns.advanced_pattern_discovery import AdvancedPatternDiscovery
            from ...database import get_database_services, ManagerMode
            
            # Use unified manager for sync operations
            sync_db_manager = get_database_services(ManagerMode.SYNC_HEAVY)
            self.pattern_discovery = AdvancedPatternDiscovery(db_manager=sync_db_manager)
        except ImportError as e:
            logger.warning(f"Advanced pattern discovery not available: {e}")

    async def discover_patterns(
        self,
        db_session: AsyncSession,
        min_effectiveness: float = 0.7,
        min_support: int = 5,
        use_advanced_discovery: bool = True,
        include_apriori: bool = True,
    ) -> Dict[str, Any]:
        """Enhanced pattern discovery combining traditional ML with Apriori association rules.

        Args:
            db_session: Database session
            min_effectiveness: Minimum effectiveness threshold
            min_support: Minimum number of occurrences
            use_advanced_discovery: Use advanced pattern discovery with HDBSCAN/FP-Growth
            include_apriori: Include Apriori association rule mining

        Returns:
            Comprehensive pattern discovery results with ML and Apriori insights
        """
        start_time = time.time()
        logger.info(
            f"Starting enhanced pattern discovery (advanced: {use_advanced_discovery}, apriori: {include_apriori})"
        )

        try:
            results = {
                "status": "success",
                "discovery_metadata": {
                    "start_time": start_time,
                    "algorithms_used": [],
                    "discovery_modes": [],
                },
            }

            # 1. Traditional ML Pattern Discovery (existing implementation)
            traditional_results = await self._discover_traditional_patterns(
                db_session, min_effectiveness, min_support
            )
            results["traditional_patterns"] = traditional_results
            results["discovery_metadata"]["algorithms_used"].append("traditional_ml")
            results["discovery_metadata"]["discovery_modes"].append(
                "parameter_analysis"
            )

            # 2. Advanced Pattern Discovery (HDBSCAN, FP-Growth, Semantic Analysis)
            if use_advanced_discovery and self.pattern_discovery:
                advanced_results = (
                    await self.pattern_discovery.discover_advanced_patterns(
                        db_session=db_session,
                        min_effectiveness=min_effectiveness,
                        min_support=min_support,
                        pattern_types=[
                            "parameter",
                            "sequence",
                            "performance",
                            "semantic",
                        ],
                        use_ensemble=True,
                        include_apriori=include_apriori,
                    )
                )
                results["advanced_patterns"] = advanced_results
                results["discovery_metadata"]["algorithms_used"].extend([
                    "hdbscan",
                    "fp_growth",
                    "semantic_clustering",
                ])
                results["discovery_metadata"]["discovery_modes"].extend([
                    "density_clustering",
                    "frequent_patterns",
                    "semantic_analysis",
                ])

                # Include Apriori patterns if they were discovered
                if include_apriori and "apriori_patterns" in advanced_results:
                    results["apriori_patterns"] = advanced_results["apriori_patterns"]
                    results["discovery_metadata"]["algorithms_used"].append("apriori")
                    results["discovery_metadata"]["discovery_modes"].append(
                        "association_rules"
                    )

            # 3. Cross-validation and ensemble analysis
            if use_advanced_discovery and "advanced_patterns" in results:
                cross_validation = self._cross_validate_pattern_discovery(
                    traditional_results, results["advanced_patterns"]
                )
                results["cross_validation"] = cross_validation

            # 4. Generate unified recommendations
            unified_recommendations = self._generate_unified_recommendations(results)
            results["unified_recommendations"] = unified_recommendations

            # 5. Business insights from all discovery methods
            business_insights = self._generate_business_insights(results)
            results["business_insights"] = business_insights

            # Add execution metadata
            execution_time = time.time() - start_time
            results["discovery_metadata"].update({
                "execution_time_seconds": execution_time,
                "total_patterns_discovered": self._count_total_patterns(results),
                "discovery_quality_score": self._calculate_discovery_quality(results),
                "timestamp": aware_utc_now().isoformat(),
                "algorithms_count": len(
                    results["discovery_metadata"]["algorithms_used"]
                ),
            })

            logger.info(
                f"Enhanced pattern discovery completed in {execution_time:.2f}s with "
                f"{len(results['discovery_metadata']['algorithms_used'])} algorithms"
            )

            return results

        except Exception as e:
            logger.error(f"Enhanced pattern discovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_seconds": time.time() - start_time,
            }

    async def get_contextualized_patterns(
        self,
        context_items: List[str],
        db_session: AsyncSession,
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """Get patterns relevant to a specific context using advanced pattern discovery.

        This method leverages both traditional ML and Apriori association rules
        to find patterns relevant to the current prompt improvement context.

        Args:
            context_items: Items representing current context (rules, characteristics)
            db_session: Database session
            min_confidence: Minimum confidence for returned patterns

        Returns:
            Dictionary with contextualized patterns and recommendations
        """
        try:
            if not self.pattern_discovery or not hasattr(self.pattern_discovery, "get_contextualized_patterns"):
                logger.warning(
                    "Advanced pattern discovery not available for contextualized patterns"
                )
                return {"error": "Advanced pattern discovery not configured"}

            # Use advanced pattern discovery for contextualized analysis
            results = await self.pattern_discovery.get_contextualized_patterns(
                context_items=context_items,
                db_session=db_session,
                min_confidence=min_confidence,
            )

            # Enhance with traditional ML insights
            traditional_context = await self._get_traditional_context_patterns(
                context_items, db_session
            )

            # Combine results
            if "error" not in results:
                results["traditional_insights"] = traditional_context
                results["combined_recommendations"] = (
                    self._combine_context_recommendations(
                        results.get("recommendations", []),
                        traditional_context.get("recommendations", []),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error getting contextualized patterns: {e}")
            return {"error": f"Contextualized pattern analysis failed: {e!s}"}

    async def _discover_traditional_patterns(
        self,
        db_session: AsyncSession,
        min_effectiveness: float,
        min_support: int,
    ) -> Dict[str, Any]:
        """Traditional pattern discovery (existing implementation)."""
        try:
            # Query rule performance data
            stmt = (
                select(RulePerformance, RuleMetadata)
                .join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id)
                .where(RulePerformance.improvement_score >= min_effectiveness)
            )

            result = await db_session.execute(stmt)
            performance_data = result.fetchall()

            if len(performance_data) < min_support:
                return {
                    "status": "insufficient_data",
                    "message": f"Only {len(performance_data)} high-performing samples found (minimum: {min_support})",
                    "data_points": len(performance_data),
                }

            # Analyze rule patterns
            rule_patterns = {}
            for row in performance_data:
                rule_id = row.rule_id
                params = row.default_parameters or {}
                effectiveness = row.improvement_score

                # Extract parameter patterns
                pattern_key = json.dumps(params, sort_keys=True)
                if pattern_key not in rule_patterns:
                    rule_patterns[pattern_key] = {
                        "parameters": params,
                        "effectiveness_scores": [],
                        "rule_ids": [],
                        "count": 0,
                    }

                rule_patterns[pattern_key]["effectiveness_scores"].append(effectiveness)
                rule_patterns[pattern_key]["rule_ids"].append(rule_id)
                rule_patterns[pattern_key]["count"] += 1

            # Filter patterns by support and effectiveness
            discovered_patterns = []
            for pattern_key, pattern_data in rule_patterns.items():
                if pattern_data["count"] >= min_support:
                    avg_effectiveness = get_numpy().mean(pattern_data["effectiveness_scores"])
                    if avg_effectiveness >= min_effectiveness:
                        discovered_patterns.append({
                            "parameters": pattern_data["parameters"],
                            "avg_effectiveness": avg_effectiveness,
                            "support_count": pattern_data["count"],
                            "rule_ids": pattern_data["rule_ids"],
                            "effectiveness_range": [
                                min(pattern_data["effectiveness_scores"]),
                                max(pattern_data["effectiveness_scores"]),
                            ],
                            "pattern_type": "traditional_parameter_pattern",
                        })

            # Sort by effectiveness
            discovered_patterns.sort(key=lambda x: x["avg_effectiveness"], reverse=True)

            return {
                "status": "success",
                "patterns_discovered": len(discovered_patterns),
                "patterns": discovered_patterns[:10],  # Top 10 patterns
                "total_analyzed": len(performance_data),
                "discovery_type": "traditional_ml",
                "algorithm": "parameter_analysis",
            }

        except Exception as e:
            logger.error(f"Traditional pattern discovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "discovery_type": "traditional_ml",
            }

    async def _get_traditional_context_patterns(
        self, context_items: List[str], db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Get traditional ML patterns relevant to context."""
        try:
            # Extract rule names from context items
            rule_contexts = [item for item in context_items if item.startswith("rule_")]

            if not rule_contexts:
                return {"recommendations": [], "context_match": 0.0}

            # Query performance data for context rules
            rule_ids = [rule.replace("rule_", "") for rule in rule_contexts]

            stmt = (
                select(RulePerformance)
                .where(RulePerformance.rule_id.in_(rule_ids))
                .where(RulePerformance.improvement_score >= 0.6)
            )

            result = await db_session.execute(stmt)
            context_performance = result.fetchall()

            recommendations = []
            if context_performance:
                avg_performance = get_numpy().mean([
                    row.improvement_score for row in context_performance
                ])
                recommendations.append({
                    "type": "traditional_context",
                    "action": f"Context rules show {avg_performance:.1%} average performance",
                    "confidence": min(len(context_performance) / 10, 1.0),
                    "priority": "high" if avg_performance > 0.8 else "medium",
                })

            return {
                "recommendations": recommendations,
                "context_match": len(context_performance) / len(rule_ids)
                if rule_ids
                else 0.0,
                "performance_data": len(context_performance),
            }

        except Exception as e:
            logger.error(f"Error getting traditional context patterns: {e}")
            return {"recommendations": [], "context_match": 0.0}

    def _cross_validate_pattern_discovery(
        self, traditional_results: Dict[str, Any], advanced_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cross-validate patterns discovered by different methods."""
        validation = {
            "consistency_score": 0.0,
            "complementary_insights": [],
            "confidence_boost": [],
            "pattern_overlap": 0.0,
        }

        try:
            # Compare traditional vs advanced patterns
            traditional_patterns = traditional_results.get("patterns", [])
            advanced_pattern_types = [
                "parameter_patterns",
                "sequence_patterns",
                "performance_patterns",
            ]

            total_advanced_patterns = 0
            overlapping_patterns = 0

            for pattern_type in advanced_pattern_types:
                if pattern_type in advanced_results:
                    patterns = advanced_results[pattern_type].get("patterns", [])
                    total_advanced_patterns += len(patterns)

                    # Check for overlapping insights
                    for advanced_pattern in patterns:
                        for traditional_pattern in traditional_patterns:
                            if self._patterns_overlap(
                                traditional_pattern, advanced_pattern
                            ):
                                overlapping_patterns += 1
                                validation["confidence_boost"].append({
                                    "traditional_pattern": traditional_pattern.get(
                                        "parameters"
                                    ),
                                    "advanced_pattern": advanced_pattern.get(
                                        "pattern_id"
                                    ),
                                    "overlap_reason": "parameter_similarity",
                                })

            # Calculate metrics
            if total_advanced_patterns > 0:
                validation["pattern_overlap"] = (
                    overlapping_patterns / total_advanced_patterns
                )
                validation["consistency_score"] = min(
                    validation["pattern_overlap"] * 2, 1.0
                )

            # Identify complementary insights
            if "apriori_patterns" in advanced_results:
                apriori_insights = advanced_results["apriori_patterns"].get(
                    "pattern_insights", {}
                )
                validation["complementary_insights"].extend([
                    {"type": "apriori_association", "insight": insight}
                    for insights_list in apriori_insights.values()
                    for insight in (
                        insights_list if isinstance(insights_list, list) else []
                    )
                ])

            return validation

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return validation

    def _patterns_overlap(
        self, traditional_pattern: Dict, advanced_pattern: Dict
    ) -> bool:
        """Check if patterns from different methods overlap."""
        try:
            # Simple overlap check based on parameter similarity
            trad_params = traditional_pattern.get("parameters", {})
            adv_params = advanced_pattern.get("parameters", {})

            if not trad_params or not adv_params:
                return False

            common_keys = set(trad_params.keys()).intersection(set(adv_params.keys()))
            return len(common_keys) > 0

        except Exception:
            return False

    def _generate_unified_recommendations(
        self, results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate unified recommendations from all discovery methods."""
        recommendations = []

        try:
            # From traditional patterns
            traditional_patterns = results.get("traditional_patterns", {}).get(
                "patterns", []
            )
            for pattern in traditional_patterns[:3]:  # Top 3
                recommendations.append({
                    "type": "parameter_optimization",
                    "source": "traditional_ml",
                    "action": f"Optimize parameters: {pattern.get('parameters')}",
                    "effectiveness": pattern.get("avg_effectiveness", 0),
                    "confidence": "high"
                    if pattern.get("support_count", 0) > 10
                    else "medium",
                    "priority": "high"
                    if pattern.get("avg_effectiveness", 0) > 0.8
                    else "medium",
                })

            # From Apriori patterns
            apriori_patterns = results.get("apriori_patterns", {}).get("patterns", [])
            for pattern in apriori_patterns[:3]:  # Top 3
                recommendations.append({
                    "type": "association_rule",
                    "source": "apriori",
                    "action": pattern.get(
                        "business_insight", "Apply discovered association rule"
                    ),
                    "confidence": pattern.get("confidence", 0),
                    "lift": pattern.get("lift", 0),
                    "priority": "high" if pattern.get("lift", 0) > 2.0 else "medium",
                })

            # From advanced patterns
            advanced_results = results.get("advanced_patterns", {})
            for pattern_type in ["parameter_patterns", "performance_patterns"]:
                if pattern_type in advanced_results:
                    patterns = advanced_results[pattern_type].get("patterns", [])
                    for pattern in patterns[:2]:  # Top 2 from each type
                        recommendations.append({
                            "type": pattern_type,
                            "source": "advanced_ml",
                            "action": f"Apply {pattern_type} insights: {pattern.get('pattern_id', 'pattern')}",
                            "effectiveness": pattern.get("effectiveness", 0),
                            "confidence": pattern.get("confidence", 0),
                            "priority": "medium",
                        })

            # Sort by priority and effectiveness
            priority_order = {"high": 3, "medium": 2, "low": 1}
            recommendations.sort(
                key=lambda x: (
                    priority_order.get(x.get("priority", "low"), 1),
                    x.get("effectiveness", x.get("confidence", 0)),
                ),
                reverse=True,
            )

            return recommendations[:10]  # Top 10 recommendations

        except Exception as e:
            logger.error(f"Error generating unified recommendations: {e}")
            return []

    def _generate_business_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business insights from comprehensive pattern discovery."""
        insights = {
            "key_findings": [],
            "performance_drivers": [],
            "optimization_opportunities": [],
            "risk_factors": [],
        }

        try:
            # Insights from traditional patterns
            traditional_patterns = results.get("traditional_patterns", {}).get(
                "patterns", []
            )
            if traditional_patterns:
                top_pattern = traditional_patterns[0]
                insights["key_findings"].append(
                    f"Top performing parameter configuration achieves {top_pattern.get('avg_effectiveness', 0):.1%} effectiveness"
                )

            # Insights from Apriori patterns
            apriori_insights = results.get("apriori_patterns", {}).get(
                "pattern_insights", {}
            )
            for category, patterns in apriori_insights.items():
                if isinstance(patterns, list) and patterns:
                    insights["performance_drivers"].extend(
                        patterns[:2]
                    )  # Top 2 per category

            # Insights from advanced discovery
            advanced_results = results.get("advanced_patterns", {})
            if "ensemble_analysis" in advanced_results:
                ensemble = advanced_results["ensemble_analysis"]
                insights["optimization_opportunities"].append(
                    f"Ensemble analysis reveals {ensemble.get('consensus_patterns', 0)} consensus patterns for optimization"
                )

            # Cross-validation insights
            cross_val = results.get("cross_validation", {})
            if cross_val.get("consistency_score", 0) > 0.7:
                insights["key_findings"].append(
                    f"High consistency score ({cross_val['consistency_score']:.1%}) between discovery methods increases confidence"
                )
            elif cross_val.get("consistency_score", 0) < 0.3:
                insights["risk_factors"].append(
                    "Low consistency between discovery methods suggests need for more data or refined parameters"
                )

            return insights

        except Exception as e:
            logger.error(f"Error generating business insights: {e}")
            return insights

    def _combine_context_recommendations(
        self, apriori_recommendations: List, traditional_recommendations: List
    ) -> List[Dict[str, Any]]:
        """Combine recommendations from different discovery methods."""
        combined = []

        # Add Apriori recommendations with source tag
        for rec in apriori_recommendations:
            rec_copy = rec.copy()
            rec_copy["source"] = "apriori"
            combined.append(rec_copy)

        # Add traditional recommendations with source tag
        for rec in traditional_recommendations:
            rec_copy = rec.copy()
            rec_copy["source"] = "traditional"
            combined.append(rec_copy)

        # Sort by priority and confidence
        priority_order = {"high": 3, "medium": 2, "low": 1}
        combined.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 1),
                x.get("confidence", 0),
            ),
            reverse=True,
        )

        return combined[:8]  # Top 8 combined recommendations

    def _count_total_patterns(self, results: Dict[str, Any]) -> int:
        """Count total patterns discovered across all methods."""
        total = 0
        try:
            # Traditional patterns
            total += len(results.get("traditional_patterns", {}).get("patterns", []))

            # Advanced patterns
            advanced_results = results.get("advanced_patterns", {})
            for pattern_type in [
                "parameter_patterns",
                "sequence_patterns",
                "performance_patterns",
                "semantic_patterns",
            ]:
                if pattern_type in advanced_results:
                    total += len(advanced_results[pattern_type].get("patterns", []))

            # Apriori patterns
            total += len(results.get("apriori_patterns", {}).get("patterns", []))

            return total
        except Exception:
            return 0

    def _calculate_discovery_quality(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score for pattern discovery."""
        try:
            scores = []

            # Traditional discovery quality
            traditional = results.get("traditional_patterns", {})
            if traditional.get("status") == "success":
                scores.append(min(traditional.get("patterns_discovered", 0) / 10, 1.0))

            # Advanced discovery quality
            advanced = results.get("advanced_patterns", {})
            if "discovery_metadata" in advanced:
                execution_time = advanced["discovery_metadata"].get(
                    "execution_time", float("inf")
                )
                # Quality inversely related to execution time (penalty for slow discovery)
                time_score = max(
                    0, 1.0 - (execution_time / 60)
                )  # Penalty after 60 seconds
                scores.append(time_score)

            # Cross-validation quality
            cross_val = results.get("cross_validation", {})
            consistency_score = cross_val.get("consistency_score", 0)
            scores.append(consistency_score)

            # Return average quality score
            return float(get_numpy().mean(scores)) if scores else 0.0

        except Exception:
            return 0.0