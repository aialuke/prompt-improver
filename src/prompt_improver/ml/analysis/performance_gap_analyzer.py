"""
Performance Gap Analyzer - 2025 Best Practices Implementation
Clean implementation with correlation-driven stopping criteria and intelligent gap detection.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
# import numpy as np  # Converted to lazy loading
from ...database.models import ImprovementSession, RulePerformance, UserFeedback
from prompt_improver.core.utils.lazy_ml_loader import get_numpy

@dataclass
class PerformanceGap:
    """Represents a detected performance gap."""
    rule_id: str
    gap_type: str
    severity: float
    current_performance: float
    target_performance: float
    gap_magnitude: float
    improvement_potential: float
    confidence: float
    metadata: dict[str, Any]

@dataclass
class GapAnalysisResult:
    """Results of performance gap analysis."""
    session_id: str
    analysis_timestamp: datetime
    total_gaps_detected: int
    critical_gaps: list[PerformanceGap]
    improvement_opportunities: list[PerformanceGap]
    stopping_criteria_met: bool
    correlation_score: float
    plateau_detected: bool
    recommended_actions: list[str]
    metadata: dict[str, Any]

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
        self.logger = logging.getLogger('apes.performance_gap_analyzer')
        self.config = {'effectiveness_threshold': 0.75, 'consistency_threshold': 0.85, 'coverage_threshold': 0.9, 'improvement_threshold': 0.02, 'plateau_window': 5, 'correlation_threshold': 0.95, 'confidence_interval': 0.95, 'min_samples': 20}

    async def analyze_performance_gaps(self, session: AsyncSession, rule_ids: list[str] | None=None, baseline_window: int=10) -> GapAnalysisResult:
        """
        Analyze performance gaps with 2025 best practices.

        Args:
            session: Database session
            rule_ids: Specific rules to analyze (None for all)
            baseline_window: Number of recent sessions for baseline

        Returns:
            Comprehensive gap analysis results
        """
        self.logger.info('Starting performance gap analysis for %s rules', len(rule_ids) if rule_ids else 'all')
        analysis_start = datetime.now(timezone.utc)
        try:
            performance_data = await self._gather_performance_data(session, rule_ids, baseline_window)
            gaps = await self._detect_performance_gaps(performance_data)
            correlation_score = await self._analyze_correlation_patterns(performance_data)
            plateau_detected = await self._detect_performance_plateau(performance_data)
            stopping_criteria_met = await self._evaluate_stopping_criteria(gaps, correlation_score, plateau_detected)
            recommendations = await self._generate_recommendations(gaps, performance_data)
            critical_gaps = [gap for gap in gaps if gap.severity >= 0.7]
            improvement_opportunities = [gap for gap in gaps if 0.3 <= gap.severity < 0.7]
            result = GapAnalysisResult(session_id=f'gap_analysis_{int(analysis_start.timestamp())}', analysis_timestamp=analysis_start, total_gaps_detected=len(gaps), critical_gaps=critical_gaps, improvement_opportunities=improvement_opportunities, stopping_criteria_met=stopping_criteria_met, correlation_score=correlation_score, plateau_detected=plateau_detected, recommended_actions=recommendations, metadata={'analysis_duration_ms': (datetime.now(timezone.utc) - analysis_start).total_seconds() * 1000, 'rules_analyzed': len(rule_ids) if rule_ids else 'all', 'baseline_window': baseline_window, 'config': self.config})
            self.logger.info('Gap analysis completed: {len(gaps)} gaps detected, stopping_criteria_met=%s', stopping_criteria_met)
            return result
        except Exception as e:
            self.logger.error('Error in performance gap analysis: %s', e)
            raise

    async def _gather_performance_data(self, session: AsyncSession, rule_ids: list[str] | None, window: int) -> dict[str, Any]:
        """Gather comprehensive performance data for analysis."""
        sessions_query = select(ImprovementSession).order_by(ImprovementSession.created_at.desc()).limit(window)
        sessions_result = await session.execute(sessions_query)
        recent_sessions = sessions_result.scalars().all()
        rule_perf_query = select(RulePerformance)
        if rule_ids:
            rule_perf_query = rule_perf_query.where(RulePerformance.rule_id.in_(rule_ids))
        rule_perf_result = await session.execute(rule_perf_query)
        rule_performances = rule_perf_result.scalars().all()
        feedback_query = select(UserFeedback).order_by(UserFeedback.created_at.desc()).limit(window * 10)
        feedback_result = await session.execute(feedback_query)
        user_feedback = feedback_result.scalars().all()
        return {'sessions': recent_sessions, 'rule_performances': rule_performances, 'user_feedback': user_feedback, 'analysis_window': window}

    async def _detect_performance_gaps(self, performance_data: dict[str, Any]) -> list[PerformanceGap]:
        """Detect performance gaps using statistical analysis."""
        gaps = []
        rule_performances = performance_data['rule_performances']
        rule_groups = {}
        for perf in rule_performances:
            if perf.rule_id not in rule_groups:
                rule_groups[perf.rule_id] = []
            rule_groups[perf.rule_id].append(perf)
        for rule_id, performances in rule_groups.items():
            if len(performances) < self.config['min_samples']:
                continue
            effectiveness_scores = [p.effectiveness_score for p in performances if p.effectiveness_score is not None]
            consistency_scores = [p.consistency_score for p in performances if p.consistency_score is not None]
            if effectiveness_scores:
                current_effectiveness = get_numpy().mean(effectiveness_scores)
                target_effectiveness = self.config['effectiveness_threshold']
                if current_effectiveness < target_effectiveness:
                    gap_magnitude = target_effectiveness - current_effectiveness
                    severity = min(gap_magnitude / target_effectiveness, 1.0)
                    gaps.append(PerformanceGap(rule_id=rule_id, gap_type='effectiveness', severity=severity, current_performance=current_effectiveness, target_performance=target_effectiveness, gap_magnitude=gap_magnitude, improvement_potential=gap_magnitude * 0.8, confidence=self._calculate_confidence(effectiveness_scores), metadata={'sample_count': len(effectiveness_scores), 'std_dev': get_numpy().std(effectiveness_scores), 'trend': self._calculate_trend(effectiveness_scores)}))
            if consistency_scores:
                current_consistency = get_numpy().mean(consistency_scores)
                target_consistency = self.config['consistency_threshold']
                if current_consistency < target_consistency:
                    gap_magnitude = target_consistency - current_consistency
                    severity = min(gap_magnitude / target_consistency, 1.0)
                    gaps.append(PerformanceGap(rule_id=rule_id, gap_type='consistency', severity=severity, current_performance=current_consistency, target_performance=target_consistency, gap_magnitude=gap_magnitude, improvement_potential=gap_magnitude * 0.7, confidence=self._calculate_confidence(consistency_scores), metadata={'sample_count': len(consistency_scores), 'std_dev': get_numpy().std(consistency_scores), 'variability': get_numpy().std(consistency_scores) / get_numpy().mean(consistency_scores)}))
        return gaps

    async def _analyze_correlation_patterns(self, performance_data: dict[str, Any]) -> float:
        """Analyze correlation patterns for stopping criteria."""
        try:
            sessions = performance_data['sessions']
            if len(sessions) < 3:
                return 0.0
            session_scores = []
            for session in sessions:
                if hasattr(session, 'overall_improvement_score') and session.overall_improvement_score:
                    session_scores.append(session.overall_improvement_score)
            if len(session_scores) < 3:
                return 0.0
            time_indices = list(range(len(session_scores)))
            correlation = get_numpy().corrcoef(time_indices, session_scores)[0, 1]
            return abs(correlation) if not get_numpy().isnan(correlation) else 0.0
        except Exception as e:
            self.logger.warning('Error calculating correlation patterns: %s', e)
            return 0.0

    async def _detect_performance_plateau(self, performance_data: dict[str, Any]) -> bool:
        """Detect if performance has plateaued using statistical analysis."""
        try:
            sessions = performance_data['sessions']
            if len(sessions) < self.config['plateau_window']:
                return False
            recent_scores = []
            for session in sessions[:self.config['plateau_window']]:
                if hasattr(session, 'overall_improvement_score') and session.overall_improvement_score:
                    recent_scores.append(session.overall_improvement_score)
            if len(recent_scores) < self.config['plateau_window']:
                return False
            mean_score = get_numpy().mean(recent_scores)
            std_score = get_numpy().std(recent_scores)
            if mean_score == 0:
                return True
            coefficient_of_variation = std_score / mean_score
            return coefficient_of_variation < 0.05
        except Exception as e:
            self.logger.warning('Error detecting plateau: %s', e)
            return False

    async def _evaluate_stopping_criteria(self, gaps: list[PerformanceGap], correlation_score: float, plateau_detected: bool) -> bool:
        """Evaluate whether stopping criteria are met using 2025 best practices."""
        critical_gaps = [gap for gap in gaps if gap.severity >= 0.7]
        no_critical_gaps = len(critical_gaps) == 0
        high_correlation = correlation_score >= self.config['correlation_threshold']
        plateau_criterion = plateau_detected
        all_gaps_minor = all(gap.gap_magnitude < self.config['improvement_threshold'] for gap in gaps)
        stopping_criteria_met = no_critical_gaps and high_correlation or (plateau_criterion and all_gaps_minor) or (no_critical_gaps and all_gaps_minor)
        self.logger.info('Stopping criteria evaluation: no_critical_gaps=%s, high_correlation=%s, plateau=%s, all_gaps_minor=%s, stopping_met=%s', no_critical_gaps, high_correlation, plateau_criterion, all_gaps_minor, stopping_criteria_met)
        return stopping_criteria_met

    async def _generate_recommendations(self, gaps: list[PerformanceGap], performance_data: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on gap analysis."""
        recommendations = []
        effectiveness_gaps = [gap for gap in gaps if gap.gap_type == 'effectiveness']
        consistency_gaps = [gap for gap in gaps if gap.gap_type == 'consistency']
        if effectiveness_gaps:
            high_severity_effectiveness = [gap for gap in effectiveness_gaps if gap.severity >= 0.7]
            if high_severity_effectiveness:
                recommendations.append(f'Focus on effectiveness improvement for {len(high_severity_effectiveness)} rules with high-severity gaps')
                recommendations.append('Generate targeted synthetic data for underperforming rule patterns')
        if consistency_gaps:
            high_variability_rules = [gap for gap in consistency_gaps if gap.metadata.get('variability', 0) > 0.3]
            if high_variability_rules:
                recommendations.append(f'Stabilize {len(high_variability_rules)} rules with high performance variability')
                recommendations.append('Apply ensemble methods to reduce prediction variance')
        if len(gaps) > 10:
            recommendations.append('Consider batch optimization for multiple rules simultaneously')
        if not gaps:
            recommendations.append('Performance targets achieved - consider increasing thresholds')
        return recommendations

    def _calculate_confidence(self, scores: list[float]) -> float:
        """Calculate confidence score for performance measurements."""
        if len(scores) < 2:
            return 0.5
        mean_score = get_numpy().mean(scores)
        std_score = get_numpy().std(scores)
        if mean_score == 0:
            return 0.0
        cv = std_score / mean_score
        confidence = max(0.0, min(1.0, 1.0 - cv))
        return confidence

    def _calculate_trend(self, scores: list[float]) -> str:
        """Calculate performance trend direction."""
        if len(scores) < 2:
            return 'insufficient_data'
        first_half = scores[:len(scores) // 2]
        second_half = scores[len(scores) // 2:]
        first_mean = get_numpy().mean(first_half)
        second_mean = get_numpy().mean(second_half)
        if second_mean > first_mean * 1.05:
            return 'improving'
        elif second_mean < first_mean * 0.95:
            return 'declining'
        else:
            return 'stable'

    async def analyze_gaps_for_targeted_generation(self, session: AsyncSession, rule_ids: list[str] | None=None, focus_areas: list[str] | None=None) -> dict[str, Any]:
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
        self.logger.info('Starting enhanced gap analysis for targeted data generation')
        try:
            standard_result = await self.analyze_performance_gaps(session, rule_ids)
            enhanced_gaps = await self._characterize_gaps_for_generation(standard_result.critical_gaps + standard_result.improvement_opportunities, session, focus_areas)
            hardness_analysis = await self._analyze_example_hardness(session, rule_ids)
            focus_priorities = await self._prioritize_focus_areas(enhanced_gaps, focus_areas)
            strategy_recommendations = await self._recommend_generation_strategies(enhanced_gaps, hardness_analysis, focus_priorities)
            difficulty_distribution = await self._analyze_difficulty_distribution(session, rule_ids, enhanced_gaps)
            return {'standard_analysis': standard_result, 'enhanced_gaps': enhanced_gaps, 'hardness_analysis': hardness_analysis, 'focus_priorities': focus_priorities, 'strategy_recommendations': strategy_recommendations, 'difficulty_distribution': difficulty_distribution, 'generation_config': {'recommended_strategy': strategy_recommendations.get('primary_strategy', 'statistical'), 'focus_areas': focus_priorities.get('top_areas', []), 'difficulty_weights': difficulty_distribution.get('recommended_weights', {}), 'target_samples': self._calculate_optimal_sample_size(enhanced_gaps), 'hardness_threshold': hardness_analysis.get('optimal_threshold', 0.7)}, 'metadata': {'analysis_timestamp': datetime.now(timezone.utc), 'total_enhanced_gaps': len(enhanced_gaps), 'generation_ready': True}}
        except Exception as e:
            self.logger.error('Error in enhanced gap analysis: %s', e)
            raise

    async def _characterize_gaps_for_generation(self, gaps: list[PerformanceGap], session: AsyncSession, focus_areas: list[str] | None) -> list[dict[str, Any]]:
        """Characterize gaps with additional metadata for targeted generation."""
        enhanced_gaps = []
        for gap in gaps:
            rule_context = await self._get_rule_context(session, gap.rule_id)
            generation_difficulty = self._calculate_generation_difficulty(gap, rule_context)
            optimal_method = self._determine_optimal_generation_method(gap, rule_context)
            improvement_potential = self._calculate_detailed_improvement_potential(gap, rule_context)
            focus_relevance = self._calculate_focus_area_relevance(gap, focus_areas)
            enhanced_gap = {'original_gap': gap, 'rule_context': rule_context, 'generation_difficulty': generation_difficulty, 'optimal_generation_method': optimal_method, 'improvement_potential': improvement_potential, 'focus_area_relevance': focus_relevance, 'priority_score': self._calculate_priority_score(gap, generation_difficulty, focus_relevance), 'estimated_samples_needed': self._estimate_samples_needed(gap, generation_difficulty), 'confidence_level': self._calculate_enhanced_confidence(gap, rule_context)}
            enhanced_gaps.append(enhanced_gap)
        enhanced_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
        return enhanced_gaps

    async def _analyze_example_hardness(self, session: AsyncSession, rule_ids: list[str] | None) -> dict[str, Any]:
        """Analyze example hardness for targeted generation (2025 best practice)."""
        try:
            performance_data = await self._gather_performance_data(session, rule_ids, 20)
            hardness_scores = []
            rule_hardness = {}
            for rule_perf in performance_data['rule_performances']:
                if rule_perf.effectiveness_score is not None:
                    hardness = 1.0 - rule_perf.effectiveness_score
                    hardness_scores.append(hardness)
                    if rule_perf.rule_id not in rule_hardness:
                        rule_hardness[rule_perf.rule_id] = []
                    rule_hardness[rule_perf.rule_id].append(hardness)
            if not hardness_scores:
                return {'optimal_threshold': 0.7, 'distribution': {}, 'recommendations': []}
            hardness_array = get_numpy().array(hardness_scores)
            optimal_threshold = get_numpy().percentile(hardness_array, 70)
            distribution = {'mean': get_numpy().mean(hardness_array), 'std': get_numpy().std(hardness_array), 'median': get_numpy().median(hardness_array), 'q25': get_numpy().percentile(hardness_array, 25), 'q75': get_numpy().percentile(hardness_array, 75), 'hard_examples_ratio': get_numpy().sum(hardness_array > optimal_threshold) / len(hardness_array)}
            recommendations = []
            if distribution['hard_examples_ratio'] > 0.3:
                recommendations.append('High proportion of hard examples - focus on neural generation methods')
            if distribution['std'] > 0.2:
                recommendations.append('High hardness variance - use adaptive difficulty distribution')
            return {'optimal_threshold': optimal_threshold, 'distribution': distribution, 'rule_hardness': rule_hardness, 'recommendations': recommendations}
        except Exception as e:
            self.logger.warning('Error in hardness analysis: %s', e)
            return {'optimal_threshold': 0.7, 'distribution': {}, 'recommendations': []}

    async def _prioritize_focus_areas(self, enhanced_gaps: list[dict[str, Any]], focus_areas: list[str] | None) -> dict[str, Any]:
        """Prioritize focus areas based on gap analysis."""
        if not enhanced_gaps:
            return {'top_areas': focus_areas or [], 'priorities': {}}
        area_impacts = {}
        if not focus_areas:
            focus_areas = ['clarity', 'specificity', 'effectiveness', 'consistency']
        for area in focus_areas:
            total_impact = 0.0
            relevant_gaps = 0
            for gap_data in enhanced_gaps:
                gap = gap_data['original_gap']
                relevance = gap_data['focus_area_relevance'].get(area, 0.0)
                if relevance > 0.1:
                    impact = gap.severity * gap.improvement_potential * relevance
                    total_impact += impact
                    relevant_gaps += 1
            if relevant_gaps > 0:
                area_impacts[area] = {'total_impact': total_impact, 'average_impact': total_impact / relevant_gaps, 'relevant_gaps': relevant_gaps}
        sorted_areas = sorted(area_impacts.items(), key=lambda x: x[1]['total_impact'], reverse=True)
        top_areas = [area for area, _ in sorted_areas[:3]]
        return {'top_areas': top_areas, 'priorities': area_impacts, 'focus_distribution': {area: data['total_impact'] for area, data in area_impacts.items()}}

    async def _recommend_generation_strategies(self, enhanced_gaps: list[dict[str, Any]], hardness_analysis: dict[str, Any], focus_priorities: dict[str, Any]) -> dict[str, Any]:
        """Recommend optimal generation strategies based on gap analysis."""
        if not enhanced_gaps:
            return {'primary_strategy': 'statistical', 'strategies': {}}
        high_severity_gaps = [g for g in enhanced_gaps if g['original_gap'].severity >= 0.7]
        complex_gaps = [g for g in enhanced_gaps if g['generation_difficulty'] >= 0.6]
        strategy_scores = {'statistical': 0.0, 'neural_enhanced': 0.0, 'rule_focused': 0.0, 'diversity_enhanced': 0.0}
        for gap_data in enhanced_gaps:
            gap = gap_data['original_gap']
            weight = gap.severity * gap.improvement_potential
            if gap.gap_type == 'effectiveness' and gap.severity >= 0.7:
                strategy_scores['neural_enhanced'] += weight * 2.0
            elif gap.gap_type == 'consistency':
                strategy_scores['rule_focused'] += weight * 1.5
            elif gap_data['generation_difficulty'] >= 0.6:
                strategy_scores['neural_enhanced'] += weight * 1.5
            else:
                strategy_scores['statistical'] += weight
        hard_ratio = hardness_analysis.get('distribution', {}).get('hard_examples_ratio', 0.3)
        if hard_ratio > 0.4:
            strategy_scores['neural_enhanced'] *= 1.5
            strategy_scores['diversity_enhanced'] *= 1.2
        primary_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        return {'primary_strategy': primary_strategy, 'strategy_scores': strategy_scores, 'recommendations': {'primary': primary_strategy, 'secondary': sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[1][0], 'reasoning': self._generate_strategy_reasoning(strategy_scores, hardness_analysis)}}

    async def _analyze_difficulty_distribution(self, session: AsyncSession, rule_ids: list[str] | None, enhanced_gaps: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze optimal difficulty distribution for targeted generation."""
        if not enhanced_gaps:
            return {'recommended_weights': {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}}
        severities = [gap['original_gap'].severity for gap in enhanced_gaps]
        high_severity_ratio = sum(1 for s in severities if s >= 0.7) / len(severities)
        medium_severity_ratio = sum(1 for s in severities if 0.3 <= s < 0.7) / len(severities)
        if high_severity_ratio > 0.5:
            weights = {'easy': 0.15, 'medium': 0.25, 'hard': 0.6}
        elif medium_severity_ratio > 0.6:
            weights = {'easy': 0.25, 'medium': 0.5, 'hard': 0.25}
        else:
            weights = {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}
        return {'recommended_weights': weights, 'gap_severity_distribution': {'high': high_severity_ratio, 'medium': medium_severity_ratio, 'low': 1.0 - high_severity_ratio - medium_severity_ratio}, 'reasoning': self._generate_difficulty_reasoning(high_severity_ratio, medium_severity_ratio)}

    async def _get_rule_context(self, session: AsyncSession, rule_id: str) -> dict[str, Any]:
        """Get additional context for a rule."""
        try:
            rule_perf_query = select(RulePerformance).where(RulePerformance.rule_id == rule_id).order_by(RulePerformance.created_at.desc()).limit(10)
            result = await session.execute(rule_perf_query)
            performances = result.scalars().all()
            if not performances:
                return {'performance_history': [], 'trend': 'unknown', 'stability': 0.5}
            scores = [p.effectiveness_score for p in performances if p.effectiveness_score is not None]
            if len(scores) >= 2:
                trend = 'improving' if scores[0] > scores[-1] else 'declining' if scores[0] < scores[-1] else 'stable'
                stability = 1.0 - get_numpy().std(scores) / get_numpy().mean(scores) if get_numpy().mean(scores) > 0 else 0.0
            else:
                trend = 'unknown'
                stability = 0.5
            return {'performance_history': scores, 'trend': trend, 'stability': max(0.0, min(1.0, stability)), 'sample_count': len(scores)}
        except Exception as e:
            self.logger.warning('Error getting rule context for {rule_id}: %s', e)
            return {'performance_history': [], 'trend': 'unknown', 'stability': 0.5}

    def _calculate_generation_difficulty(self, gap: PerformanceGap, rule_context: dict[str, Any]) -> float:
        """Calculate how difficult it will be to generate data for this gap."""
        base_difficulty = gap.severity
        type_multipliers = {'effectiveness': 1.0, 'consistency': 1.2, 'coverage': 0.8}
        difficulty = base_difficulty * type_multipliers.get(gap.gap_type, 1.0)
        stability = rule_context.get('stability', 0.5)
        if stability < 0.3:
            difficulty *= 1.3
        trend = rule_context.get('trend', 'unknown')
        if trend == 'declining':
            difficulty *= 1.2
        elif trend == 'improving':
            difficulty *= 0.9
        return max(0.0, min(1.0, difficulty))

    def _determine_optimal_generation_method(self, gap: PerformanceGap, rule_context: dict[str, Any]) -> str:
        """Determine the optimal generation method for this specific gap."""
        if gap.severity >= 0.8:
            return 'neural_enhanced'
        elif gap.gap_type == 'consistency' and rule_context.get('stability', 0.5) < 0.4:
            return 'rule_focused'
        elif gap.gap_type == 'coverage':
            return 'diversity_enhanced'
        else:
            return 'statistical'

    def _calculate_detailed_improvement_potential(self, gap: PerformanceGap, rule_context: dict[str, Any]) -> dict[str, float]:
        """Calculate detailed improvement potential with confidence intervals."""
        base_potential = gap.improvement_potential
        trend = rule_context.get('trend', 'unknown')
        if trend == 'improving':
            potential_multiplier = 1.2
        elif trend == 'declining':
            potential_multiplier = 0.8
        else:
            potential_multiplier = 1.0
        adjusted_potential = base_potential * potential_multiplier
        stability = rule_context.get('stability', 0.5)
        confidence_width = (1.0 - stability) * 0.3
        return {'expected': adjusted_potential, 'lower_bound': max(0.0, adjusted_potential - confidence_width), 'upper_bound': min(1.0, adjusted_potential + confidence_width), 'confidence': stability}

    def _calculate_focus_area_relevance(self, gap: PerformanceGap, focus_areas: list[str] | None) -> dict[str, float]:
        """Calculate relevance of this gap to specified focus areas."""
        if not focus_areas:
            return {}
        relevance_scores = {}
        for area in focus_areas:
            if area.lower() in gap.gap_type.lower():
                relevance_scores[area] = 0.9
            elif gap.gap_type == 'effectiveness' and area in ['clarity', 'specificity']:
                relevance_scores[area] = 0.7
            elif gap.gap_type == 'consistency' and area in ['stability', 'reliability']:
                relevance_scores[area] = 0.8
            else:
                metadata_text = str(gap.metadata).lower()
                if area.lower() in metadata_text:
                    relevance_scores[area] = 0.6
                else:
                    relevance_scores[area] = 0.3
        return relevance_scores

    def _calculate_priority_score(self, gap: PerformanceGap, generation_difficulty: float, focus_relevance: dict[str, float]) -> float:
        """Calculate overall priority score for gap addressing."""
        base_score = gap.severity * gap.improvement_potential
        difficulty_adjustment = 1.0 - generation_difficulty * 0.2
        max_relevance = max(focus_relevance.values()) if focus_relevance else 0.5
        relevance_adjustment = 0.8 + max_relevance * 0.4
        confidence_adjustment = 0.7 + gap.confidence * 0.6
        priority_score = base_score * difficulty_adjustment * relevance_adjustment * confidence_adjustment
        return max(0.0, min(1.0, priority_score))

    def _estimate_samples_needed(self, gap: PerformanceGap, generation_difficulty: float) -> int:
        """Estimate number of samples needed to address this gap."""
        base_samples = int(gap.gap_magnitude * 1000)
        difficulty_multiplier = 1.0 + generation_difficulty
        type_multipliers = {'effectiveness': 1.0, 'consistency': 1.5, 'coverage': 0.8}
        type_multiplier = type_multipliers.get(gap.gap_type, 1.0)
        estimated_samples = int(base_samples * difficulty_multiplier * type_multiplier)
        return max(50, min(2000, estimated_samples))

    def _calculate_enhanced_confidence(self, gap: PerformanceGap, rule_context: dict[str, Any]) -> float:
        """Calculate enhanced confidence score incorporating rule context."""
        base_confidence = gap.confidence
        sample_count = rule_context.get('sample_count', 0)
        if sample_count >= 20:
            sample_adjustment = 1.0
        elif sample_count >= 10:
            sample_adjustment = 0.9
        else:
            sample_adjustment = 0.7
        stability = rule_context.get('stability', 0.5)
        stability_adjustment = 0.5 + stability * 0.5
        enhanced_confidence = base_confidence * sample_adjustment * stability_adjustment
        return max(0.0, min(1.0, enhanced_confidence))

    def _calculate_optimal_sample_size(self, enhanced_gaps: list[dict[str, Any]]) -> int:
        """Calculate optimal total sample size for targeted generation."""
        if not enhanced_gaps:
            return 500
        top_gaps = enhanced_gaps[:5]
        total_estimated = sum(gap['estimated_samples_needed'] for gap in top_gaps)
        total_with_buffer = int(total_estimated * 1.2)
        return max(200, min(2000, total_with_buffer))

    def _generate_strategy_reasoning(self, strategy_scores: dict[str, float], hardness_analysis: dict[str, Any]) -> str:
        """Generate human-readable reasoning for strategy selection."""
        top_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        hard_ratio = hardness_analysis.get('distribution', {}).get('hard_examples_ratio', 0.3)
        if top_strategy == 'neural_enhanced':
            return f'Neural enhancement recommended due to high complexity gaps and {hard_ratio:.1%} hard examples'
        elif top_strategy == 'rule_focused':
            return 'Rule-focused generation recommended due to consistency gaps and rule-specific issues'
        elif top_strategy == 'diversity_enhanced':
            return 'Diversity enhancement recommended to improve pattern coverage'
        else:
            return 'Statistical generation sufficient for current gap profile'

    def _generate_difficulty_reasoning(self, high_severity_ratio: float, medium_severity_ratio: float) -> str:
        """Generate reasoning for difficulty distribution recommendations."""
        if high_severity_ratio > 0.5:
            return f'High proportion ({high_severity_ratio:.1%}) of severe gaps requires focus on hard examples'
        elif medium_severity_ratio > 0.6:
            return f'Balanced gap distribution ({medium_severity_ratio:.1%} medium) suggests balanced difficulty approach'
        else:
            return 'Low gap severity allows uniform difficulty distribution'