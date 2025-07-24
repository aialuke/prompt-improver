"""
Performance Gap Analyzer - 2025 Best Practices Implementation
Clean implementation with correlation-driven stopping criteria and intelligent gap detection.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ...database.models import ImprovementSession, RulePerformance, UserFeedback


@dataclass
class PerformanceGap:
    """Represents a detected performance gap."""
    rule_id: str
    gap_type: str  # 'effectiveness', 'consistency', 'coverage'
    severity: float  # 0.0 to 1.0
    current_performance: float
    target_performance: float
    gap_magnitude: float
    improvement_potential: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class GapAnalysisResult:
    """Results of performance gap analysis."""
    session_id: str
    analysis_timestamp: datetime
    total_gaps_detected: int
    critical_gaps: List[PerformanceGap]
    improvement_opportunities: List[PerformanceGap]
    stopping_criteria_met: bool
    correlation_score: float
    plateau_detected: bool
    recommended_actions: List[str]
    metadata: Dict[str, Any]


class PerformanceGapAnalyzer:
    """
    2025 best practices performance gap analyzer with correlation-driven stopping.

    Features:
    - Intelligent gap detection using statistical analysis
    - Correlation-driven stopping criteria
    - Plateau detection with confidence intervals
    - Multi-dimensional performance assessment
    - Adaptive threshold adjustment
    """

    def __init__(self):
        self.logger = logging.getLogger("apes.performance_gap_analyzer")

        # 2025 best practice thresholds
        self.config = {
            "effectiveness_threshold": 0.75,  # 75% effectiveness target
            "consistency_threshold": 0.85,    # 85% consistency target
            "coverage_threshold": 0.90,       # 90% coverage target
            "improvement_threshold": 0.02,    # 2% minimum improvement
            "plateau_window": 5,              # 5 iterations for plateau detection
            "correlation_threshold": 0.95,    # 95% correlation for stopping
            "confidence_interval": 0.95,      # 95% confidence interval
            "min_samples": 20,                # Minimum samples for analysis
        }

    async def analyze_performance_gaps(
        self,
        session: AsyncSession,
        rule_ids: Optional[List[str]] = None,
        baseline_window: int = 10
    ) -> GapAnalysisResult:
        """
        Analyze performance gaps with 2025 best practices.

        Args:
            session: Database session
            rule_ids: Specific rules to analyze (None for all)
            baseline_window: Number of recent sessions for baseline

        Returns:
            Comprehensive gap analysis results
        """
        self.logger.info(f"Starting performance gap analysis for {len(rule_ids) if rule_ids else 'all'} rules")

        analysis_start = datetime.now(timezone.utc)

        try:
            # 1. Gather performance data
            performance_data = await self._gather_performance_data(session, rule_ids, baseline_window)

            # 2. Detect performance gaps
            gaps = await self._detect_performance_gaps(performance_data)

            # 3. Analyze correlation patterns
            correlation_score = await self._analyze_correlation_patterns(performance_data)

            # 4. Check plateau detection
            plateau_detected = await self._detect_performance_plateau(performance_data)

            # 5. Evaluate stopping criteria
            stopping_criteria_met = await self._evaluate_stopping_criteria(
                gaps, correlation_score, plateau_detected
            )

            # 6. Generate recommendations
            recommendations = await self._generate_recommendations(gaps, performance_data)

            # Categorize gaps by severity
            critical_gaps = [gap for gap in gaps if gap.severity >= 0.7]
            improvement_opportunities = [gap for gap in gaps if 0.3 <= gap.severity < 0.7]

            result = GapAnalysisResult(
                session_id=f"gap_analysis_{int(analysis_start.timestamp())}",
                analysis_timestamp=analysis_start,
                total_gaps_detected=len(gaps),
                critical_gaps=critical_gaps,
                improvement_opportunities=improvement_opportunities,
                stopping_criteria_met=stopping_criteria_met,
                correlation_score=correlation_score,
                plateau_detected=plateau_detected,
                recommended_actions=recommendations,
                metadata={
                    "analysis_duration_ms": (datetime.now(timezone.utc) - analysis_start).total_seconds() * 1000,
                    "rules_analyzed": len(rule_ids) if rule_ids else "all",
                    "baseline_window": baseline_window,
                    "config": self.config
                }
            )

            self.logger.info(f"Gap analysis completed: {len(gaps)} gaps detected, stopping_criteria_met={stopping_criteria_met}")
            return result

        except Exception as e:
            self.logger.error(f"Error in performance gap analysis: {e}")
            raise

    async def _gather_performance_data(
        self,
        session: AsyncSession,
        rule_ids: Optional[List[str]],
        window: int
    ) -> Dict[str, Any]:
        """Gather comprehensive performance data for analysis."""

        # Get recent improvement sessions
        sessions_query = select(ImprovementSession).order_by(
            ImprovementSession.created_at.desc()
        ).limit(window)

        sessions_result = await session.execute(sessions_query)
        recent_sessions = sessions_result.scalars().all()

        # Get rule performance data
        rule_perf_query = select(RulePerformance)
        if rule_ids:
            rule_perf_query = rule_perf_query.where(RulePerformance.rule_id.in_(rule_ids))

        rule_perf_result = await session.execute(rule_perf_query)
        rule_performances = rule_perf_result.scalars().all()

        # Get user feedback data
        feedback_query = select(UserFeedback).order_by(
            UserFeedback.created_at.desc()
        ).limit(window * 10)  # More feedback samples

        feedback_result = await session.execute(feedback_query)
        user_feedback = feedback_result.scalars().all()

        return {
            "sessions": recent_sessions,
            "rule_performances": rule_performances,
            "user_feedback": user_feedback,
            "analysis_window": window
        }

    async def _detect_performance_gaps(self, performance_data: Dict[str, Any]) -> List[PerformanceGap]:
        """Detect performance gaps using statistical analysis."""
        gaps = []

        # Analyze rule performance gaps
        rule_performances = performance_data["rule_performances"]

        # Group by rule_id for analysis
        rule_groups = {}
        for perf in rule_performances:
            if perf.rule_id not in rule_groups:
                rule_groups[perf.rule_id] = []
            rule_groups[perf.rule_id].append(perf)

        for rule_id, performances in rule_groups.items():
            if len(performances) < self.config["min_samples"]:
                continue

            # Calculate performance metrics
            effectiveness_scores = [p.effectiveness_score for p in performances if p.effectiveness_score is not None]
            consistency_scores = [p.consistency_score for p in performances if p.consistency_score is not None]

            if effectiveness_scores:
                # Effectiveness gap analysis
                current_effectiveness = np.mean(effectiveness_scores)
                target_effectiveness = self.config["effectiveness_threshold"]

                if current_effectiveness < target_effectiveness:
                    gap_magnitude = target_effectiveness - current_effectiveness
                    severity = min(gap_magnitude / target_effectiveness, 1.0)

                    gaps.append(PerformanceGap(
                        rule_id=rule_id,
                        gap_type="effectiveness",
                        severity=severity,
                        current_performance=current_effectiveness,
                        target_performance=target_effectiveness,
                        gap_magnitude=gap_magnitude,
                        improvement_potential=gap_magnitude * 0.8,  # 80% achievable
                        confidence=self._calculate_confidence(effectiveness_scores),
                        metadata={
                            "sample_count": len(effectiveness_scores),
                            "std_dev": np.std(effectiveness_scores),
                            "trend": self._calculate_trend(effectiveness_scores)
                        }
                    ))

            if consistency_scores:
                # Consistency gap analysis
                current_consistency = np.mean(consistency_scores)
                target_consistency = self.config["consistency_threshold"]

                if current_consistency < target_consistency:
                    gap_magnitude = target_consistency - current_consistency
                    severity = min(gap_magnitude / target_consistency, 1.0)

                    gaps.append(PerformanceGap(
                        rule_id=rule_id,
                        gap_type="consistency",
                        severity=severity,
                        current_performance=current_consistency,
                        target_performance=target_consistency,
                        gap_magnitude=gap_magnitude,
                        improvement_potential=gap_magnitude * 0.7,  # 70% achievable
                        confidence=self._calculate_confidence(consistency_scores),
                        metadata={
                            "sample_count": len(consistency_scores),
                            "std_dev": np.std(consistency_scores),
                            "variability": np.std(consistency_scores) / np.mean(consistency_scores)
                        }
                    ))

        return gaps

    async def _analyze_correlation_patterns(self, performance_data: Dict[str, Any]) -> float:
        """Analyze correlation patterns for stopping criteria."""
        try:
            sessions = performance_data["sessions"]
            if len(sessions) < 3:
                return 0.0

            # Extract performance trends
            session_scores = []
            for session in sessions:
                if hasattr(session, 'overall_improvement_score') and session.overall_improvement_score:
                    session_scores.append(session.overall_improvement_score)

            if len(session_scores) < 3:
                return 0.0

            # Calculate correlation with time (trend analysis)
            time_indices = list(range(len(session_scores)))
            correlation = np.corrcoef(time_indices, session_scores)[0, 1]

            # Return absolute correlation (strength of relationship)
            return abs(correlation) if not np.isnan(correlation) else 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating correlation patterns: {e}")
            return 0.0

    async def _detect_performance_plateau(self, performance_data: Dict[str, Any]) -> bool:
        """Detect if performance has plateaued using statistical analysis."""
        try:
            sessions = performance_data["sessions"]
            if len(sessions) < self.config["plateau_window"]:
                return False

            # Get recent performance scores
            recent_scores = []
            for session in sessions[:self.config["plateau_window"]]:
                if hasattr(session, 'overall_improvement_score') and session.overall_improvement_score:
                    recent_scores.append(session.overall_improvement_score)

            if len(recent_scores) < self.config["plateau_window"]:
                return False

            # Check for plateau using coefficient of variation
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)

            if mean_score == 0:
                return True  # No improvement

            coefficient_of_variation = std_score / mean_score

            # Plateau detected if variation is very low
            return coefficient_of_variation < 0.05  # 5% variation threshold

        except Exception as e:
            self.logger.warning(f"Error detecting plateau: {e}")
            return False

    async def _evaluate_stopping_criteria(
        self,
        gaps: List[PerformanceGap],
        correlation_score: float,
        plateau_detected: bool
    ) -> bool:
        """Evaluate whether stopping criteria are met using 2025 best practices."""

        # Criteria 1: No critical gaps remaining
        critical_gaps = [gap for gap in gaps if gap.severity >= 0.7]
        no_critical_gaps = len(critical_gaps) == 0

        # Criteria 2: High correlation (stable performance)
        high_correlation = correlation_score >= self.config["correlation_threshold"]

        # Criteria 3: Performance plateau detected
        plateau_criterion = plateau_detected

        # Criteria 4: All gaps below improvement threshold
        all_gaps_minor = all(gap.gap_magnitude < self.config["improvement_threshold"] for gap in gaps)

        # Stopping criteria met if any major criterion is satisfied
        stopping_criteria_met = (
            (no_critical_gaps and high_correlation) or
            (plateau_criterion and all_gaps_minor) or
            (no_critical_gaps and all_gaps_minor)
        )

        self.logger.info(f"Stopping criteria evaluation: no_critical_gaps={no_critical_gaps}, "
                        f"high_correlation={high_correlation}, plateau={plateau_criterion}, "
                        f"all_gaps_minor={all_gaps_minor}, stopping_met={stopping_criteria_met}")

        return stopping_criteria_met

    async def _generate_recommendations(
        self,
        gaps: List[PerformanceGap],
        performance_data: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on gap analysis."""
        recommendations = []

        # Analyze gap patterns
        effectiveness_gaps = [gap for gap in gaps if gap.gap_type == "effectiveness"]
        consistency_gaps = [gap for gap in gaps if gap.gap_type == "consistency"]

        if effectiveness_gaps:
            high_severity_effectiveness = [gap for gap in effectiveness_gaps if gap.severity >= 0.7]
            if high_severity_effectiveness:
                recommendations.append(
                    f"Focus on effectiveness improvement for {len(high_severity_effectiveness)} rules with high-severity gaps"
                )
                recommendations.append(
                    "Generate targeted synthetic data for underperforming rule patterns"
                )

        if consistency_gaps:
            high_variability_rules = [gap for gap in consistency_gaps if gap.metadata.get("variability", 0) > 0.3]
            if high_variability_rules:
                recommendations.append(
                    f"Stabilize {len(high_variability_rules)} rules with high performance variability"
                )
                recommendations.append(
                    "Apply ensemble methods to reduce prediction variance"
                )

        # General recommendations
        if len(gaps) > 10:
            recommendations.append("Consider batch optimization for multiple rules simultaneously")

        if not gaps:
            recommendations.append("Performance targets achieved - consider increasing thresholds")

        return recommendations

    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence score for performance measurements."""
        if len(scores) < 2:
            return 0.5

        # Use coefficient of variation as confidence measure
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if mean_score == 0:
            return 0.0

        cv = std_score / mean_score
        confidence = max(0.0, min(1.0, 1.0 - cv))  # Higher confidence with lower variation

        return confidence

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate performance trend direction."""
        if len(scores) < 2:
            return "insufficient_data"

        # Simple trend analysis
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]

        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)

        if second_mean > first_mean * 1.05:  # 5% improvement
            return "improving"
        elif second_mean < first_mean * 0.95:  # 5% decline
            return "declining"
        else:
            return "stable"

    # ===== 2025 ENHANCED GAP IDENTIFICATION FOR ADAPTIVE DATA GENERATION =====

    async def analyze_gaps_for_targeted_generation(
        self,
        session: AsyncSession,
        rule_ids: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced gap analysis specifically for targeted data generation (2025 best practice).

        Provides detailed gap characterization for adaptive synthetic data generation
        including hardness characterization, focus area analysis, and generation strategy recommendations.

        Args:
            session: Database session
            rule_ids: Specific rules to analyze
            focus_areas: Specific areas to focus analysis on

        Returns:
            Comprehensive gap analysis for targeted generation
        """
        self.logger.info("Starting enhanced gap analysis for targeted data generation")

        try:
            # 1. Perform standard gap analysis
            standard_result = await self.analyze_performance_gaps(session, rule_ids)

            # 2. Enhanced gap characterization
            enhanced_gaps = await self._characterize_gaps_for_generation(
                standard_result.critical_gaps + standard_result.improvement_opportunities,
                session,
                focus_areas
            )

            # 3. Hardness analysis
            hardness_analysis = await self._analyze_example_hardness(session, rule_ids)

            # 4. Focus area prioritization
            focus_priorities = await self._prioritize_focus_areas(enhanced_gaps, focus_areas)

            # 5. Generation strategy recommendations
            strategy_recommendations = await self._recommend_generation_strategies(
                enhanced_gaps, hardness_analysis, focus_priorities
            )

            # 6. Difficulty distribution analysis
            difficulty_distribution = await self._analyze_difficulty_distribution(
                session, rule_ids, enhanced_gaps
            )

            return {
                "standard_analysis": standard_result,
                "enhanced_gaps": enhanced_gaps,
                "hardness_analysis": hardness_analysis,
                "focus_priorities": focus_priorities,
                "strategy_recommendations": strategy_recommendations,
                "difficulty_distribution": difficulty_distribution,
                "generation_config": {
                    "recommended_strategy": strategy_recommendations.get("primary_strategy", "statistical"),
                    "focus_areas": focus_priorities.get("top_areas", []),
                    "difficulty_weights": difficulty_distribution.get("recommended_weights", {}),
                    "target_samples": self._calculate_optimal_sample_size(enhanced_gaps),
                    "hardness_threshold": hardness_analysis.get("optimal_threshold", 0.7)
                },
                "metadata": {
                    "analysis_timestamp": datetime.now(timezone.utc),
                    "total_enhanced_gaps": len(enhanced_gaps),
                    "generation_ready": True
                }
            }

        except Exception as e:
            self.logger.error(f"Error in enhanced gap analysis: {e}")
            raise

    async def _characterize_gaps_for_generation(
        self,
        gaps: List[PerformanceGap],
        session: AsyncSession,
        focus_areas: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Characterize gaps with additional metadata for targeted generation."""
        enhanced_gaps = []

        for gap in gaps:
            # Get additional rule context
            rule_context = await self._get_rule_context(session, gap.rule_id)

            # Calculate generation difficulty
            generation_difficulty = self._calculate_generation_difficulty(gap, rule_context)

            # Determine optimal generation method
            optimal_method = self._determine_optimal_generation_method(gap, rule_context)

            # Calculate improvement potential with confidence intervals
            improvement_potential = self._calculate_detailed_improvement_potential(gap, rule_context)

            # Focus area relevance scoring
            focus_relevance = self._calculate_focus_area_relevance(gap, focus_areas)

            enhanced_gap = {
                "original_gap": gap,
                "rule_context": rule_context,
                "generation_difficulty": generation_difficulty,
                "optimal_generation_method": optimal_method,
                "improvement_potential": improvement_potential,
                "focus_area_relevance": focus_relevance,
                "priority_score": self._calculate_priority_score(gap, generation_difficulty, focus_relevance),
                "estimated_samples_needed": self._estimate_samples_needed(gap, generation_difficulty),
                "confidence_level": self._calculate_enhanced_confidence(gap, rule_context)
            }

            enhanced_gaps.append(enhanced_gap)

        # Sort by priority score
        enhanced_gaps.sort(key=lambda x: x["priority_score"], reverse=True)

        return enhanced_gaps

    async def _analyze_example_hardness(
        self,
        session: AsyncSession,
        rule_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Analyze example hardness for targeted generation (2025 best practice)."""
        try:
            # Get performance data for hardness analysis
            performance_data = await self._gather_performance_data(session, rule_ids, 20)

            # Calculate hardness metrics
            hardness_scores = []
            rule_hardness = {}

            for rule_perf in performance_data["rule_performances"]:
                if rule_perf.effectiveness_score is not None:
                    # Hardness inversely related to effectiveness
                    hardness = 1.0 - rule_perf.effectiveness_score
                    hardness_scores.append(hardness)

                    if rule_perf.rule_id not in rule_hardness:
                        rule_hardness[rule_perf.rule_id] = []
                    rule_hardness[rule_perf.rule_id].append(hardness)

            if not hardness_scores:
                return {"optimal_threshold": 0.7, "distribution": {}, "recommendations": []}

            # Calculate optimal hardness threshold
            hardness_array = np.array(hardness_scores)
            optimal_threshold = np.percentile(hardness_array, 70)  # 70th percentile

            # Analyze hardness distribution
            distribution = {
                "mean": np.mean(hardness_array),
                "std": np.std(hardness_array),
                "median": np.median(hardness_array),
                "q25": np.percentile(hardness_array, 25),
                "q75": np.percentile(hardness_array, 75),
                "hard_examples_ratio": np.sum(hardness_array > optimal_threshold) / len(hardness_array)
            }

            # Generate recommendations
            recommendations = []
            if distribution["hard_examples_ratio"] > 0.3:
                recommendations.append("High proportion of hard examples - focus on neural generation methods")
            if distribution["std"] > 0.2:
                recommendations.append("High hardness variance - use adaptive difficulty distribution")

            return {
                "optimal_threshold": optimal_threshold,
                "distribution": distribution,
                "rule_hardness": rule_hardness,
                "recommendations": recommendations
            }

        except Exception as e:
            self.logger.warning(f"Error in hardness analysis: {e}")
            return {"optimal_threshold": 0.7, "distribution": {}, "recommendations": []}

    async def _prioritize_focus_areas(
        self,
        enhanced_gaps: List[Dict[str, Any]],
        focus_areas: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Prioritize focus areas based on gap analysis."""
        if not enhanced_gaps:
            return {"top_areas": focus_areas or [], "priorities": {}}

        # Calculate area impact scores
        area_impacts = {}

        # Default focus areas if none provided
        if not focus_areas:
            focus_areas = ["clarity", "specificity", "effectiveness", "consistency"]

        for area in focus_areas:
            total_impact = 0.0
            relevant_gaps = 0

            for gap_data in enhanced_gaps:
                gap = gap_data["original_gap"]
                relevance = gap_data["focus_area_relevance"].get(area, 0.0)

                if relevance > 0.1:  # Relevant threshold
                    impact = gap.severity * gap.improvement_potential * relevance
                    total_impact += impact
                    relevant_gaps += 1

            if relevant_gaps > 0:
                area_impacts[area] = {
                    "total_impact": total_impact,
                    "average_impact": total_impact / relevant_gaps,
                    "relevant_gaps": relevant_gaps
                }

        # Sort areas by total impact
        sorted_areas = sorted(
            area_impacts.items(),
            key=lambda x: x[1]["total_impact"],
            reverse=True
        )

        top_areas = [area for area, _ in sorted_areas[:3]]  # Top 3 areas

        return {
            "top_areas": top_areas,
            "priorities": area_impacts,
            "focus_distribution": {area: data["total_impact"] for area, data in area_impacts.items()}
        }

    async def _recommend_generation_strategies(
        self,
        enhanced_gaps: List[Dict[str, Any]],
        hardness_analysis: Dict[str, Any],
        focus_priorities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend optimal generation strategies based on gap analysis."""
        if not enhanced_gaps:
            return {"primary_strategy": "statistical", "strategies": {}}

        # Analyze gap characteristics
        high_severity_gaps = [g for g in enhanced_gaps if g["original_gap"].severity >= 0.7]
        complex_gaps = [g for g in enhanced_gaps if g["generation_difficulty"] >= 0.6]

        # Strategy scoring
        strategy_scores = {
            "statistical": 0.0,
            "neural_enhanced": 0.0,
            "rule_focused": 0.0,
            "diversity_enhanced": 0.0
        }

        # Score based on gap characteristics
        for gap_data in enhanced_gaps:
            gap = gap_data["original_gap"]
            weight = gap.severity * gap.improvement_potential

            if gap.gap_type == "effectiveness" and gap.severity >= 0.7:
                strategy_scores["neural_enhanced"] += weight * 2.0
            elif gap.gap_type == "consistency":
                strategy_scores["rule_focused"] += weight * 1.5
            elif gap_data["generation_difficulty"] >= 0.6:
                strategy_scores["neural_enhanced"] += weight * 1.5
            else:
                strategy_scores["statistical"] += weight

        # Adjust based on hardness analysis
        hard_ratio = hardness_analysis.get("distribution", {}).get("hard_examples_ratio", 0.3)
        if hard_ratio > 0.4:
            strategy_scores["neural_enhanced"] *= 1.5
            strategy_scores["diversity_enhanced"] *= 1.2

        # Determine primary strategy
        primary_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]

        return {
            "primary_strategy": primary_strategy,
            "strategy_scores": strategy_scores,
            "recommendations": {
                "primary": primary_strategy,
                "secondary": sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[1][0],
                "reasoning": self._generate_strategy_reasoning(strategy_scores, hardness_analysis)
            }
        }

    async def _analyze_difficulty_distribution(
        self,
        session: AsyncSession,
        rule_ids: Optional[List[str]],
        enhanced_gaps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze optimal difficulty distribution for targeted generation."""
        if not enhanced_gaps:
            return {"recommended_weights": {"easy": 0.33, "medium": 0.34, "hard": 0.33}}

        # Calculate gap severity distribution
        severities = [gap["original_gap"].severity for gap in enhanced_gaps]
        high_severity_ratio = sum(1 for s in severities if s >= 0.7) / len(severities)
        medium_severity_ratio = sum(1 for s in severities if 0.3 <= s < 0.7) / len(severities)

        # Determine optimal weights based on gap distribution
        if high_severity_ratio > 0.5:
            # Many high-severity gaps - focus on hard examples
            weights = {"easy": 0.15, "medium": 0.25, "hard": 0.60}
        elif medium_severity_ratio > 0.6:
            # Many medium gaps - balanced approach
            weights = {"easy": 0.25, "medium": 0.50, "hard": 0.25}
        else:
            # Few gaps - uniform distribution
            weights = {"easy": 0.33, "medium": 0.34, "hard": 0.33}

        return {
            "recommended_weights": weights,
            "gap_severity_distribution": {
                "high": high_severity_ratio,
                "medium": medium_severity_ratio,
                "low": 1.0 - high_severity_ratio - medium_severity_ratio
            },
            "reasoning": self._generate_difficulty_reasoning(high_severity_ratio, medium_severity_ratio)
        }

    # Helper methods for enhanced gap analysis

    async def _get_rule_context(self, session: AsyncSession, rule_id: str) -> Dict[str, Any]:
        """Get additional context for a rule."""
        try:
            # Get rule performance history
            rule_perf_query = select(RulePerformance).where(
                RulePerformance.rule_id == rule_id
            ).order_by(RulePerformance.created_at.desc()).limit(10)

            result = await session.execute(rule_perf_query)
            performances = result.scalars().all()

            if not performances:
                return {"performance_history": [], "trend": "unknown", "stability": 0.5}

            # Calculate trend and stability
            scores = [p.effectiveness_score for p in performances if p.effectiveness_score is not None]

            if len(scores) >= 2:
                trend = "improving" if scores[0] > scores[-1] else "declining" if scores[0] < scores[-1] else "stable"
                stability = 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.0
            else:
                trend = "unknown"
                stability = 0.5

            return {
                "performance_history": scores,
                "trend": trend,
                "stability": max(0.0, min(1.0, stability)),
                "sample_count": len(scores)
            }

        except Exception as e:
            self.logger.warning(f"Error getting rule context for {rule_id}: {e}")
            return {"performance_history": [], "trend": "unknown", "stability": 0.5}

    def _calculate_generation_difficulty(self, gap: PerformanceGap, rule_context: Dict[str, Any]) -> float:
        """Calculate how difficult it will be to generate data for this gap."""
        base_difficulty = gap.severity  # Higher severity = more difficult

        # Adjust based on gap type
        type_multipliers = {
            "effectiveness": 1.0,
            "consistency": 1.2,  # Consistency gaps are harder to address
            "coverage": 0.8       # Coverage gaps are easier
        }

        difficulty = base_difficulty * type_multipliers.get(gap.gap_type, 1.0)

        # Adjust based on rule stability
        stability = rule_context.get("stability", 0.5)
        if stability < 0.3:  # Unstable rules are harder to generate for
            difficulty *= 1.3

        # Adjust based on trend
        trend = rule_context.get("trend", "unknown")
        if trend == "declining":
            difficulty *= 1.2
        elif trend == "improving":
            difficulty *= 0.9

        return max(0.0, min(1.0, difficulty))

    def _determine_optimal_generation_method(self, gap: PerformanceGap, rule_context: Dict[str, Any]) -> str:
        """Determine the optimal generation method for this specific gap."""
        if gap.severity >= 0.8:
            return "neural_enhanced"
        elif gap.gap_type == "consistency" and rule_context.get("stability", 0.5) < 0.4:
            return "rule_focused"
        elif gap.gap_type == "coverage":
            return "diversity_enhanced"
        else:
            return "statistical"

    def _calculate_detailed_improvement_potential(self, gap: PerformanceGap, rule_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed improvement potential with confidence intervals."""
        base_potential = gap.improvement_potential

        # Adjust based on rule trend
        trend = rule_context.get("trend", "unknown")
        if trend == "improving":
            potential_multiplier = 1.2
        elif trend == "declining":
            potential_multiplier = 0.8
        else:
            potential_multiplier = 1.0

        adjusted_potential = base_potential * potential_multiplier

        # Calculate confidence intervals based on stability
        stability = rule_context.get("stability", 0.5)
        confidence_width = (1.0 - stability) * 0.3  # Less stable = wider intervals

        return {
            "expected": adjusted_potential,
            "lower_bound": max(0.0, adjusted_potential - confidence_width),
            "upper_bound": min(1.0, adjusted_potential + confidence_width),
            "confidence": stability
        }

    def _calculate_focus_area_relevance(self, gap: PerformanceGap, focus_areas: Optional[List[str]]) -> Dict[str, float]:
        """Calculate relevance of this gap to specified focus areas."""
        if not focus_areas:
            return {}

        relevance_scores = {}

        for area in focus_areas:
            # Base relevance on gap type and area alignment
            if area.lower() in gap.gap_type.lower():
                relevance_scores[area] = 0.9
            elif gap.gap_type == "effectiveness" and area in ["clarity", "specificity"]:
                relevance_scores[area] = 0.7
            elif gap.gap_type == "consistency" and area in ["stability", "reliability"]:
                relevance_scores[area] = 0.8
            else:
                # Check metadata for area mentions
                metadata_text = str(gap.metadata).lower()
                if area.lower() in metadata_text:
                    relevance_scores[area] = 0.6
                else:
                    relevance_scores[area] = 0.3  # Default low relevance

        return relevance_scores

    def _calculate_priority_score(self, gap: PerformanceGap, generation_difficulty: float, focus_relevance: Dict[str, float]) -> float:
        """Calculate overall priority score for gap addressing."""
        # Base score from gap severity and improvement potential
        base_score = gap.severity * gap.improvement_potential

        # Adjust for generation difficulty (easier gaps get slight priority)
        difficulty_adjustment = 1.0 - (generation_difficulty * 0.2)

        # Adjust for focus area relevance
        max_relevance = max(focus_relevance.values()) if focus_relevance else 0.5
        relevance_adjustment = 0.8 + (max_relevance * 0.4)  # 0.8 to 1.2 multiplier

        # Adjust for confidence
        confidence_adjustment = 0.7 + (gap.confidence * 0.6)  # 0.7 to 1.3 multiplier

        priority_score = base_score * difficulty_adjustment * relevance_adjustment * confidence_adjustment

        return max(0.0, min(1.0, priority_score))

    def _estimate_samples_needed(self, gap: PerformanceGap, generation_difficulty: float) -> int:
        """Estimate number of samples needed to address this gap."""
        # Base samples based on gap magnitude
        base_samples = int(gap.gap_magnitude * 1000)  # 1000 samples per unit gap

        # Adjust for difficulty
        difficulty_multiplier = 1.0 + generation_difficulty

        # Adjust for gap type
        type_multipliers = {
            "effectiveness": 1.0,
            "consistency": 1.5,  # Need more samples for consistency
            "coverage": 0.8       # Fewer samples for coverage
        }

        type_multiplier = type_multipliers.get(gap.gap_type, 1.0)

        estimated_samples = int(base_samples * difficulty_multiplier * type_multiplier)

        # Reasonable bounds
        return max(50, min(2000, estimated_samples))

    def _calculate_enhanced_confidence(self, gap: PerformanceGap, rule_context: Dict[str, Any]) -> float:
        """Calculate enhanced confidence score incorporating rule context."""
        base_confidence = gap.confidence

        # Adjust based on sample count
        sample_count = rule_context.get("sample_count", 0)
        if sample_count >= 20:
            sample_adjustment = 1.0
        elif sample_count >= 10:
            sample_adjustment = 0.9
        else:
            sample_adjustment = 0.7

        # Adjust based on stability
        stability = rule_context.get("stability", 0.5)
        stability_adjustment = 0.5 + (stability * 0.5)  # 0.5 to 1.0

        enhanced_confidence = base_confidence * sample_adjustment * stability_adjustment

        return max(0.0, min(1.0, enhanced_confidence))

    def _calculate_optimal_sample_size(self, enhanced_gaps: List[Dict[str, Any]]) -> int:
        """Calculate optimal total sample size for targeted generation."""
        if not enhanced_gaps:
            return 500  # Default

        # Sum estimated samples for top priority gaps
        top_gaps = enhanced_gaps[:5]  # Top 5 priority gaps
        total_estimated = sum(gap["estimated_samples_needed"] for gap in top_gaps)

        # Add buffer for diversity
        total_with_buffer = int(total_estimated * 1.2)

        # Reasonable bounds
        return max(200, min(2000, total_with_buffer))

    def _generate_strategy_reasoning(self, strategy_scores: Dict[str, float], hardness_analysis: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for strategy selection."""
        top_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        hard_ratio = hardness_analysis.get("distribution", {}).get("hard_examples_ratio", 0.3)

        if top_strategy == "neural_enhanced":
            return f"Neural enhancement recommended due to high complexity gaps and {hard_ratio:.1%} hard examples"
        elif top_strategy == "rule_focused":
            return "Rule-focused generation recommended due to consistency gaps and rule-specific issues"
        elif top_strategy == "diversity_enhanced":
            return "Diversity enhancement recommended to improve pattern coverage"
        else:
            return "Statistical generation sufficient for current gap profile"

    def _generate_difficulty_reasoning(self, high_severity_ratio: float, medium_severity_ratio: float) -> str:
        """Generate reasoning for difficulty distribution recommendations."""
        if high_severity_ratio > 0.5:
            return f"High proportion ({high_severity_ratio:.1%}) of severe gaps requires focus on hard examples"
        elif medium_severity_ratio > 0.6:
            return f"Balanced gap distribution ({medium_severity_ratio:.1%} medium) suggests balanced difficulty approach"
        else:
            return "Low gap severity allows uniform difficulty distribution"
