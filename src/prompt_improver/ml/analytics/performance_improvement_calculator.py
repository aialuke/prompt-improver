"""
Performance Improvement Calculator for Training Sessions
Implements 2025 best practices for calculating performance improvements, trend analysis, and plateau detection.
"""
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlmodel import SQLModel, Field
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from ...database.models import TrainingIteration, TrainingSession
from ...utils.datetime_utils import naive_utc_now
logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction classification"""
    IMPROVING = 'improving'
    DECLINING = 'declining'
    STABLE = 'stable'
    VOLATILE = 'volatile'

class PlateauStatus(Enum):
    """Plateau detection status"""
    NO_PLATEAU = 'no_plateau'
    EARLY_PLATEAU = 'early_plateau'
    CONFIRMED_PLATEAU = 'confirmed_plateau'
    PERFORMANCE_DECLINE = 'performance_decline'

class PerformanceMetrics(SQLModel):
    """Standardized performance metrics for training sessions"""
    model_accuracy: float = Field(ge=0.0, le=1.0, description='Model prediction accuracy')
    rule_effectiveness: float = Field(ge=0.0, le=1.0, description='Rule optimization effectiveness')
    pattern_coverage: float = Field(ge=0.0, le=1.0, description='Pattern discovery coverage')
    training_efficiency: float = Field(ge=0.0, description='Resource utilization efficiency')
    timestamp: datetime = Field(description='Metrics collection timestamp')
    iteration: int = Field(ge=0, description='Training iteration number')
    session_id: str = Field(description='Training session identifier')

class ImprovementCalculation(SQLModel):
    """Performance improvement calculation result"""
    absolute_improvement: float = Field(description='Absolute performance improvement')
    relative_improvement: float = Field(description='Relative improvement as percentage')
    weighted_improvement: float = Field(description='Weighted improvement across metrics')
    confidence_score: float = Field(ge=0.0, le=1.0, description='Confidence in improvement calculation')
    statistical_significance: bool = Field(description='Statistical significance of improvement')
    effect_size: float = Field(ge=0.0, description="Cohen's d effect size")
    improvement_rate: float = Field(description='Rate of improvement per hour')

class TrendAnalysis(SQLModel):
    """Trend analysis result"""
    direction: TrendDirection = Field(description='Overall trend direction')
    slope: float = Field(description='Linear regression slope')
    correlation: float = Field(ge=-1.0, le=1.0, description='Pearson correlation coefficient')
    p_value: float = Field(ge=0.0, le=1.0, description='Statistical significance p-value')
    trend_strength: str = Field(description='Qualitative trend strength assessment')
    volatility_score: float = Field(ge=0.0, description='Coefficient of variation')
    prediction_confidence: float = Field(ge=0.0, le=1.0, description='Confidence in trend predictions')

class PlateauDetection(SQLModel):
    """Plateau detection result"""
    status: PlateauStatus = Field(description='Current plateau status')
    plateau_start_iteration: Optional[int] = Field(default=None, ge=0, description='Iteration where plateau began')
    plateau_duration: int = Field(ge=0, description='Number of iterations in plateau')
    improvement_stagnation_score: float = Field(ge=0.0, le=1.0, description='Degree of improvement stagnation')
    correlation_analysis: float = Field(ge=-1.0, le=1.0, description='Correlation-based plateau indicator')
    recommendation: str = Field(description='Recommended action based on plateau analysis')

class PerformanceImprovementCalculator:
    """
    Advanced performance improvement calculator implementing 2025 best practices.

    Features:
    - Research-validated weighted improvement calculations
    - Correlation-driven plateau detection
    - Statistical significance testing
    - Trend analysis with confidence intervals
    - Multi-dimensional performance assessment
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.metric_weights = {'model_accuracy': 0.4, 'rule_effectiveness': 0.3, 'pattern_coverage': 0.2, 'training_efficiency': 0.1}
        self.significance_level = 0.05
        self.plateau_threshold = 0.02
        self.plateau_consecutive_iterations = 3
        self.correlation_plateau_threshold = 0.1

    async def calculate_improvement(self, current_metrics: PerformanceMetrics, previous_metrics: PerformanceMetrics) -> ImprovementCalculation:
        """
        Calculate comprehensive performance improvement between two metric points.

        Uses research-validated weighted improvement calculation with statistical significance testing.
        """
        try:
            improvements = {}
            for metric in self.metric_weights.keys():
                current_val = getattr(current_metrics, metric)
                previous_val = getattr(previous_metrics, metric)
                if previous_val > 0:
                    absolute_imp = current_val - previous_val
                    relative_imp = absolute_imp / previous_val
                    improvements[metric] = {'absolute': absolute_imp, 'relative': relative_imp}
                else:
                    improvements[metric] = {'absolute': 0.0, 'relative': 0.0}
            weighted_improvement = sum((improvements[metric]['relative'] * weight for metric, weight in self.metric_weights.items()))
            absolute_improvement = np.mean([improvements[metric]['absolute'] for metric in self.metric_weights.keys()])
            relative_improvement = np.mean([improvements[metric]['relative'] for metric in self.metric_weights.keys()])
            time_diff = (current_metrics.timestamp - previous_metrics.timestamp).total_seconds()
            improvement_rate = weighted_improvement / (time_diff / 3600) if time_diff > 0 else 0.0
            significance, effect_size = await self._calculate_statistical_significance(current_metrics, previous_metrics)
            confidence_score = self._calculate_confidence_score(weighted_improvement, significance, effect_size, time_diff)
            return ImprovementCalculation(absolute_improvement=absolute_improvement, relative_improvement=relative_improvement, weighted_improvement=weighted_improvement, confidence_score=confidence_score, statistical_significance=significance, effect_size=effect_size, improvement_rate=improvement_rate)
        except Exception as e:
            self.logger.error('Error calculating improvement: %s', e)
            return ImprovementCalculation(absolute_improvement=0.0, relative_improvement=0.0, weighted_improvement=0.0, confidence_score=0.0, statistical_significance=False, effect_size=0.0, improvement_rate=0.0)

    async def analyze_performance_trend(self, session_id: str, lookback_iterations: int=10) -> TrendAnalysis:
        """
        Analyze performance trend over recent iterations using correlation analysis.

        Implements 2025 best practices for trend detection with statistical validation.
        """
        try:
            iterations = await self._get_recent_iterations(session_id, lookback_iterations)
            if len(iterations) < 3:
                return TrendAnalysis(direction=TrendDirection.STABLE, slope=0.0, correlation=0.0, p_value=1.0, trend_strength='insufficient_data', volatility_score=0.0, prediction_confidence=0.0)
            performance_scores = []
            iteration_numbers = []
            for iteration in iterations:
                metrics = self._extract_performance_metrics(iteration)
                if metrics:
                    weighted_score = self._calculate_weighted_score(metrics)
                    performance_scores.append(weighted_score)
                    iteration_numbers.append(iteration.iteration)
            if len(performance_scores) < 3:
                return TrendAnalysis(direction=TrendDirection.STABLE, slope=0.0, correlation=0.0, p_value=1.0, trend_strength='insufficient_data', volatility_score=0.0, prediction_confidence=0.0)
            correlation, p_value = pearsonr(iteration_numbers, performance_scores)
            slope, intercept, r_value, p_val, std_err = stats.linregress(iteration_numbers, performance_scores)
            direction = self._classify_trend_direction(correlation, p_value)
            volatility_score = np.std(performance_scores) / np.mean(performance_scores) if np.mean(performance_scores) > 0 else 0.0
            trend_strength = self._classify_trend_strength(abs(correlation), p_value)
            prediction_confidence = self._calculate_prediction_confidence(correlation, p_value, len(performance_scores), volatility_score)
            return TrendAnalysis(direction=direction, slope=slope, correlation=correlation, p_value=p_value, trend_strength=trend_strength, volatility_score=volatility_score, prediction_confidence=prediction_confidence)
        except Exception as e:
            self.logger.error('Error analyzing trend for session {session_id}: %s', e)
            return TrendAnalysis(direction=TrendDirection.STABLE, slope=0.0, correlation=0.0, p_value=1.0, trend_strength='error', volatility_score=0.0, prediction_confidence=0.0)

    async def detect_plateau(self, session_id: str, improvement_history: List[float], lookback_iterations: int=10) -> PlateauDetection:
        """
        Detect performance plateau using correlation-driven analysis (2025 best practice).

        Implements advanced plateau detection with multiple validation methods.
        """
        try:
            if len(improvement_history) < self.plateau_consecutive_iterations:
                return PlateauDetection(status=PlateauStatus.NO_PLATEAU, plateau_start_iteration=None, plateau_duration=0, improvement_stagnation_score=0.0, correlation_analysis=0.0, recommendation='Continue training - insufficient data for plateau detection')
            recent_improvements = improvement_history[-self.plateau_consecutive_iterations:]
            below_threshold_count = sum((1 for imp in recent_improvements if imp < self.plateau_threshold))
            if len(improvement_history) >= 10:
                iterations = list(range(len(improvement_history)))
                correlation, p_value = pearsonr(iterations, improvement_history)
                correlation_indicates_plateau = abs(correlation) < self.correlation_plateau_threshold
            else:
                correlation = 0.0
                correlation_indicates_plateau = False
            stagnation_score = self._calculate_stagnation_score(improvement_history)
            status = self._determine_plateau_status(below_threshold_count, correlation_indicates_plateau, stagnation_score)
            plateau_start = self._find_plateau_start(improvement_history) if status != PlateauStatus.NO_PLATEAU else None
            plateau_duration = len(improvement_history) - plateau_start if plateau_start else 0
            recommendation = self._generate_plateau_recommendation(status, stagnation_score, correlation)
            return PlateauDetection(status=status, plateau_start_iteration=plateau_start, plateau_duration=plateau_duration, improvement_stagnation_score=stagnation_score, correlation_analysis=correlation, recommendation=recommendation)
        except Exception as e:
            self.logger.error('Error detecting plateau for session {session_id}: %s', e)
            return PlateauDetection(status=PlateauStatus.NO_PLATEAU, plateau_start_iteration=None, plateau_duration=0, improvement_stagnation_score=0.0, correlation_analysis=0.0, recommendation='Error in plateau detection - continue training')

    async def _calculate_statistical_significance(self, current_metrics: PerformanceMetrics, previous_metrics: PerformanceMetrics) -> Tuple[bool, float]:
        """Calculate statistical significance and effect size"""
        try:
            iterations = await self._get_recent_iterations(current_metrics.session_id, 20)
            if len(iterations) < 10:
                return (False, 0.0)
            scores = []
            for iteration in iterations:
                metrics = self._extract_performance_metrics(iteration)
                if metrics:
                    scores.append(self._calculate_weighted_score(metrics))
            if len(scores) < 10:
                return (False, 0.0)
            mid_point = len(scores) // 2
            before_scores = scores[:mid_point]
            after_scores = scores[mid_point:]
            statistic, p_value = stats.ttest_ind(after_scores, before_scores)
            pooled_std = np.sqrt(((len(after_scores) - 1) * np.var(after_scores, ddof=1) + (len(before_scores) - 1) * np.var(before_scores, ddof=1)) / (len(after_scores) + len(before_scores) - 2))
            cohens_d = (np.mean(after_scores) - np.mean(before_scores)) / pooled_std if pooled_std > 0 else 0.0
            return (p_value < self.significance_level, abs(cohens_d))
        except Exception as e:
            self.logger.error('Error calculating statistical significance: %s', e)
            return (False, 0.0)

    def _calculate_confidence_score(self, improvement: float, significance: bool, effect_size: float, time_diff: float) -> float:
        """Calculate confidence score for improvement calculation"""
        confidence = 0.0
        confidence += min(0.4, abs(improvement) * 2)
        if significance:
            confidence += 0.3
        confidence += min(0.2, effect_size * 0.1)
        if time_diff > 300:
            confidence += 0.1
        return min(1.0, confidence)

    async def _get_recent_iterations(self, session_id: str, limit: int) -> List[TrainingIteration]:
        """Get recent training iterations for analysis"""
        try:
            query = select(TrainingIteration).where(TrainingIteration.session_id == session_id).order_by(desc(TrainingIteration.iteration)).limit(limit)
            result = await self.db_session.execute(query)
            iterations = result.scalars().all()
            return list(reversed(iterations))
        except Exception as e:
            self.logger.error('Error getting recent iterations: %s', e)
            return []

    def _extract_performance_metrics(self, iteration: TrainingIteration) -> Optional[Dict[str, float]]:
        """Extract performance metrics from training iteration"""
        try:
            metrics = iteration.performance_metrics
            if not metrics:
                return None
            return {'model_accuracy': metrics.get('model_accuracy', metrics.get('accuracy', 0.0)), 'rule_effectiveness': metrics.get('rule_effectiveness', metrics.get('rule_score', 0.0)), 'pattern_coverage': metrics.get('pattern_coverage', metrics.get('coverage', 0.0)), 'training_efficiency': metrics.get('training_efficiency', metrics.get('efficiency', 0.0))}
        except Exception as e:
            self.logger.error('Error extracting metrics from iteration: %s', e)
            return None

    def _calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted performance score"""
        return sum((metrics.get(metric, 0.0) * weight for metric, weight in self.metric_weights.items()))

    def _classify_trend_direction(self, correlation: float, p_value: float) -> TrendDirection:
        """Classify trend direction based on correlation analysis"""
        if p_value > self.significance_level:
            return TrendDirection.STABLE
        if abs(correlation) < 0.3:
            return TrendDirection.VOLATILE
        elif correlation > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING

    def _classify_trend_strength(self, abs_correlation: float, p_value: float) -> str:
        """Classify trend strength"""
        if p_value > self.significance_level:
            return 'not_significant'
        if abs_correlation >= 0.7:
            return 'strong'
        elif abs_correlation >= 0.5:
            return 'moderate'
        elif abs_correlation >= 0.3:
            return 'weak'
        else:
            return 'very_weak'

    def _calculate_prediction_confidence(self, correlation: float, p_value: float, sample_size: int, volatility: float) -> float:
        """Calculate prediction confidence"""
        confidence = 0.0
        confidence += min(0.4, abs(correlation))
        if p_value < self.significance_level:
            confidence += 0.3
        confidence += min(0.2, sample_size / 50)
        confidence -= min(0.3, volatility)
        return max(0.0, min(1.0, confidence))

    def _calculate_stagnation_score(self, improvement_history: List[float]) -> float:
        """Calculate improvement stagnation score"""
        if len(improvement_history) < 5:
            return 0.0
        recent_improvements = improvement_history[-5:]
        avg_recent = np.mean(recent_improvements)
        if len(improvement_history) >= 10:
            earlier_improvements = improvement_history[-10:-5]
            avg_earlier = np.mean(earlier_improvements)
            if avg_earlier > 0:
                stagnation = max(0.0, (avg_earlier - avg_recent) / avg_earlier)
            else:
                stagnation = 1.0 if avg_recent <= 0 else 0.0
        else:
            stagnation = 1.0 - min(1.0, avg_recent / self.plateau_threshold)
        return min(1.0, stagnation)

    def _determine_plateau_status(self, below_threshold_count: int, correlation_indicates_plateau: bool, stagnation_score: float) -> PlateauStatus:
        """Determine plateau status from multiple indicators"""
        if stagnation_score > 0.8:
            return PlateauStatus.PERFORMANCE_DECLINE
        elif below_threshold_count >= self.plateau_consecutive_iterations and correlation_indicates_plateau and (stagnation_score > 0.7):
            return PlateauStatus.CONFIRMED_PLATEAU
        elif below_threshold_count >= 2 and (correlation_indicates_plateau or stagnation_score > 0.5):
            return PlateauStatus.EARLY_PLATEAU
        else:
            return PlateauStatus.NO_PLATEAU

    def _find_plateau_start(self, improvement_history: List[float]) -> Optional[int]:
        """Find the iteration where plateau started"""
        for i in range(len(improvement_history) - self.plateau_consecutive_iterations + 1):
            consecutive_low = all((improvement_history[i + j] < self.plateau_threshold for j in range(self.plateau_consecutive_iterations)))
            if consecutive_low:
                return i
        return None

    def _generate_plateau_recommendation(self, status: PlateauStatus, stagnation_score: float, correlation: float) -> str:
        """Generate recommendation based on plateau analysis"""
        if status == PlateauStatus.CONFIRMED_PLATEAU:
            return 'Performance plateau confirmed. Consider: 1) Adjusting learning parameters, 2) Adding new training data, 3) Changing generation strategy, or 4) Stopping training.'
        elif status == PlateauStatus.EARLY_PLATEAU:
            return 'Early plateau indicators detected. Monitor closely and consider parameter adjustments if trend continues.'
        elif status == PlateauStatus.PERFORMANCE_DECLINE:
            return 'Performance decline detected. Consider: 1) Reverting to previous checkpoint, 2) Reducing learning rate, or 3) Investigating data quality issues.'
        else:
            return 'No plateau detected. Continue training with current configuration.'
