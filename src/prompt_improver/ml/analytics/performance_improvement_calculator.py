"""
Performance Improvement Calculator for Training Sessions
Implements 2025 best practices for calculating performance improvements, trend analysis, and plateau detection.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from ...database.models import TrainingSession, TrainingIteration
from ...utils.datetime_utils import naive_utc_now

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction classification"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


class PlateauStatus(Enum):
    """Plateau detection status"""
    NO_PLATEAU = "no_plateau"
    EARLY_PLATEAU = "early_plateau"
    CONFIRMED_PLATEAU = "confirmed_plateau"
    PERFORMANCE_DECLINE = "performance_decline"


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics for training sessions"""
    model_accuracy: float
    rule_effectiveness: float
    pattern_coverage: float
    training_efficiency: float
    timestamp: datetime
    iteration: int
    session_id: str


@dataclass
class ImprovementCalculation:
    """Performance improvement calculation result"""
    absolute_improvement: float
    relative_improvement: float
    weighted_improvement: float
    confidence_score: float
    statistical_significance: bool
    effect_size: float
    improvement_rate: float


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    direction: TrendDirection
    slope: float
    correlation: float
    p_value: float
    trend_strength: str
    volatility_score: float
    prediction_confidence: float


@dataclass
class PlateauDetection:
    """Plateau detection result"""
    status: PlateauStatus
    plateau_start_iteration: Optional[int]
    plateau_duration: int
    improvement_stagnation_score: float
    correlation_analysis: float
    recommendation: str


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

        # Research-validated weights for performance metrics (2025 best practices)
        self.metric_weights = {
            "model_accuracy": 0.4,      # Primary metric - model performance
            "rule_effectiveness": 0.3,   # Rule optimization quality
            "pattern_coverage": 0.2,     # Discovery completeness
            "training_efficiency": 0.1   # Resource utilization
        }

        # Statistical thresholds
        self.significance_level = 0.05
        self.plateau_threshold = 0.02  # 2% minimum improvement
        self.plateau_consecutive_iterations = 3
        self.correlation_plateau_threshold = 0.1

    async def calculate_improvement(
        self,
        current_metrics: PerformanceMetrics,
        previous_metrics: PerformanceMetrics
    ) -> ImprovementCalculation:
        """
        Calculate comprehensive performance improvement between two metric points.

        Uses research-validated weighted improvement calculation with statistical significance testing.
        """
        try:
            # Calculate individual metric improvements
            improvements = {}
            for metric in self.metric_weights.keys():
                current_val = getattr(current_metrics, metric)
                previous_val = getattr(previous_metrics, metric)

                if previous_val > 0:
                    absolute_imp = current_val - previous_val
                    relative_imp = absolute_imp / previous_val
                    improvements[metric] = {
                        "absolute": absolute_imp,
                        "relative": relative_imp
                    }
                else:
                    improvements[metric] = {"absolute": 0.0, "relative": 0.0}

            # Calculate weighted improvement
            weighted_improvement = sum(
                improvements[metric]["relative"] * weight
                for metric, weight in self.metric_weights.items()
            )

            # Calculate absolute improvement (average)
            absolute_improvement = np.mean([
                improvements[metric]["absolute"]
                for metric in self.metric_weights.keys()
            ])

            # Calculate relative improvement (average)
            relative_improvement = np.mean([
                improvements[metric]["relative"]
                for metric in self.metric_weights.keys()
            ])

            # Calculate improvement rate (per time unit)
            time_diff = (current_metrics.timestamp - previous_metrics.timestamp).total_seconds()
            improvement_rate = weighted_improvement / (time_diff / 3600) if time_diff > 0 else 0.0

            # Statistical significance and effect size
            significance, effect_size = await self._calculate_statistical_significance(
                current_metrics, previous_metrics
            )

            # Confidence score based on multiple factors
            confidence_score = self._calculate_confidence_score(
                weighted_improvement, significance, effect_size, time_diff
            )

            return ImprovementCalculation(
                absolute_improvement=absolute_improvement,
                relative_improvement=relative_improvement,
                weighted_improvement=weighted_improvement,
                confidence_score=confidence_score,
                statistical_significance=significance,
                effect_size=effect_size,
                improvement_rate=improvement_rate
            )

        except Exception as e:
            self.logger.error(f"Error calculating improvement: {e}")
            return ImprovementCalculation(
                absolute_improvement=0.0,
                relative_improvement=0.0,
                weighted_improvement=0.0,
                confidence_score=0.0,
                statistical_significance=False,
                effect_size=0.0,
                improvement_rate=0.0
            )

    async def analyze_performance_trend(
        self,
        session_id: str,
        lookback_iterations: int = 10
    ) -> TrendAnalysis:
        """
        Analyze performance trend over recent iterations using correlation analysis.

        Implements 2025 best practices for trend detection with statistical validation.
        """
        try:
            # Get recent iterations
            iterations = await self._get_recent_iterations(session_id, lookback_iterations)

            if len(iterations) < 3:
                return TrendAnalysis(
                    direction=TrendDirection.STABLE,
                    slope=0.0,
                    correlation=0.0,
                    p_value=1.0,
                    trend_strength="insufficient_data",
                    volatility_score=0.0,
                    prediction_confidence=0.0
                )

            # Extract performance scores
            performance_scores = []
            iteration_numbers = []

            for iteration in iterations:
                metrics = self._extract_performance_metrics(iteration)
                if metrics:
                    weighted_score = self._calculate_weighted_score(metrics)
                    performance_scores.append(weighted_score)
                    iteration_numbers.append(iteration.iteration)

            if len(performance_scores) < 3:
                return TrendAnalysis(
                    direction=TrendDirection.STABLE,
                    slope=0.0,
                    correlation=0.0,
                    p_value=1.0,
                    trend_strength="insufficient_data",
                    volatility_score=0.0,
                    prediction_confidence=0.0
                )

            # Calculate correlation between iteration and performance
            correlation, p_value = pearsonr(iteration_numbers, performance_scores)

            # Calculate slope using linear regression
            slope, intercept, r_value, p_val, std_err = stats.linregress(
                iteration_numbers, performance_scores
            )

            # Determine trend direction
            direction = self._classify_trend_direction(correlation, p_value)

            # Calculate volatility
            volatility_score = np.std(performance_scores) / np.mean(performance_scores) if np.mean(performance_scores) > 0 else 0.0

            # Trend strength classification
            trend_strength = self._classify_trend_strength(abs(correlation), p_value)

            # Prediction confidence
            prediction_confidence = self._calculate_prediction_confidence(
                correlation, p_value, len(performance_scores), volatility_score
            )

            return TrendAnalysis(
                direction=direction,
                slope=slope,
                correlation=correlation,
                p_value=p_value,
                trend_strength=trend_strength,
                volatility_score=volatility_score,
                prediction_confidence=prediction_confidence
            )

        except Exception as e:
            self.logger.error(f"Error analyzing trend for session {session_id}: {e}")
            return TrendAnalysis(
                direction=TrendDirection.STABLE,
                slope=0.0,
                correlation=0.0,
                p_value=1.0,
                trend_strength="error",
                volatility_score=0.0,
                prediction_confidence=0.0
            )

    async def detect_plateau(
        self,
        session_id: str,
        improvement_history: List[float],
        lookback_iterations: int = 10
    ) -> PlateauDetection:
        """
        Detect performance plateau using correlation-driven analysis (2025 best practice).

        Implements advanced plateau detection with multiple validation methods.
        """
        try:
            if len(improvement_history) < self.plateau_consecutive_iterations:
                return PlateauDetection(
                    status=PlateauStatus.NO_PLATEAU,
                    plateau_start_iteration=None,
                    plateau_duration=0,
                    improvement_stagnation_score=0.0,
                    correlation_analysis=0.0,
                    recommendation="Continue training - insufficient data for plateau detection"
                )

            # Method 1: Check recent improvements against threshold
            recent_improvements = improvement_history[-self.plateau_consecutive_iterations:]
            below_threshold_count = sum(1 for imp in recent_improvements if imp < self.plateau_threshold)

            # Method 2: Correlation analysis for plateau detection
            if len(improvement_history) >= 10:
                iterations = list(range(len(improvement_history)))
                correlation, p_value = pearsonr(iterations, improvement_history)
                correlation_indicates_plateau = abs(correlation) < self.correlation_plateau_threshold
            else:
                correlation = 0.0
                correlation_indicates_plateau = False

            # Method 3: Improvement stagnation score
            stagnation_score = self._calculate_stagnation_score(improvement_history)

            # Determine plateau status
            status = self._determine_plateau_status(
                below_threshold_count, correlation_indicates_plateau, stagnation_score
            )

            # Find plateau start if detected
            plateau_start = self._find_plateau_start(improvement_history) if status != PlateauStatus.NO_PLATEAU else None
            plateau_duration = len(improvement_history) - plateau_start if plateau_start else 0

            # Generate recommendation
            recommendation = self._generate_plateau_recommendation(status, stagnation_score, correlation)

            return PlateauDetection(
                status=status,
                plateau_start_iteration=plateau_start,
                plateau_duration=plateau_duration,
                improvement_stagnation_score=stagnation_score,
                correlation_analysis=correlation,
                recommendation=recommendation
            )

        except Exception as e:
            self.logger.error(f"Error detecting plateau for session {session_id}: {e}")
            return PlateauDetection(
                status=PlateauStatus.NO_PLATEAU,
                plateau_start_iteration=None,
                plateau_duration=0,
                improvement_stagnation_score=0.0,
                correlation_analysis=0.0,
                recommendation="Error in plateau detection - continue training"
            )

    # Helper methods for performance calculations

    async def _calculate_statistical_significance(
        self,
        current_metrics: PerformanceMetrics,
        previous_metrics: PerformanceMetrics
    ) -> Tuple[bool, float]:
        """Calculate statistical significance and effect size"""
        try:
            # Get historical data for comparison
            iterations = await self._get_recent_iterations(current_metrics.session_id, 20)

            if len(iterations) < 10:
                return False, 0.0

            # Extract performance scores
            scores = []
            for iteration in iterations:
                metrics = self._extract_performance_metrics(iteration)
                if metrics:
                    scores.append(self._calculate_weighted_score(metrics))

            if len(scores) < 10:
                return False, 0.0

            # Split into before/after groups
            mid_point = len(scores) // 2
            before_scores = scores[:mid_point]
            after_scores = scores[mid_point:]

            # Perform t-test
            statistic, p_value = stats.ttest_ind(after_scores, before_scores)

            # Calculate Cohen's d (effect size)
            pooled_std = np.sqrt(
                ((len(after_scores) - 1) * np.var(after_scores, ddof=1) +
                 (len(before_scores) - 1) * np.var(before_scores, ddof=1)) /
                (len(after_scores) + len(before_scores) - 2)
            )

            cohens_d = (np.mean(after_scores) - np.mean(before_scores)) / pooled_std if pooled_std > 0 else 0.0

            return p_value < self.significance_level, abs(cohens_d)

        except Exception as e:
            self.logger.error(f"Error calculating statistical significance: {e}")
            return False, 0.0

    def _calculate_confidence_score(
        self,
        improvement: float,
        significance: bool,
        effect_size: float,
        time_diff: float
    ) -> float:
        """Calculate confidence score for improvement calculation"""
        confidence = 0.0

        # Base confidence from improvement magnitude
        confidence += min(0.4, abs(improvement) * 2)

        # Statistical significance bonus
        if significance:
            confidence += 0.3

        # Effect size contribution
        confidence += min(0.2, effect_size * 0.1)

        # Time stability factor
        if time_diff > 300:  # More than 5 minutes
            confidence += 0.1

        return min(1.0, confidence)

    async def _get_recent_iterations(
        self,
        session_id: str,
        limit: int
    ) -> List[TrainingIteration]:
        """Get recent training iterations for analysis"""
        try:
            query = (
                select(TrainingIteration)
                .where(TrainingIteration.session_id == session_id)
                .order_by(desc(TrainingIteration.iteration))
                .limit(limit)
            )

            result = await self.db_session.execute(query)
            iterations = result.scalars().all()

            # Return in chronological order
            return list(reversed(iterations))

        except Exception as e:
            self.logger.error(f"Error getting recent iterations: {e}")
            return []

    def _extract_performance_metrics(self, iteration: TrainingIteration) -> Optional[Dict[str, float]]:
        """Extract performance metrics from training iteration"""
        try:
            metrics = iteration.performance_metrics
            if not metrics:
                return None

            # Extract standard metrics with fallbacks
            return {
                "model_accuracy": metrics.get("model_accuracy", metrics.get("accuracy", 0.0)),
                "rule_effectiveness": metrics.get("rule_effectiveness", metrics.get("rule_score", 0.0)),
                "pattern_coverage": metrics.get("pattern_coverage", metrics.get("coverage", 0.0)),
                "training_efficiency": metrics.get("training_efficiency", metrics.get("efficiency", 0.0))
            }

        except Exception as e:
            self.logger.error(f"Error extracting metrics from iteration: {e}")
            return None

    def _calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted performance score"""
        return sum(
            metrics.get(metric, 0.0) * weight
            for metric, weight in self.metric_weights.items()
        )

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
            return "not_significant"

        if abs_correlation >= 0.7:
            return "strong"
        elif abs_correlation >= 0.5:
            return "moderate"
        elif abs_correlation >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def _calculate_prediction_confidence(
        self,
        correlation: float,
        p_value: float,
        sample_size: int,
        volatility: float
    ) -> float:
        """Calculate prediction confidence"""
        confidence = 0.0

        # Correlation strength
        confidence += min(0.4, abs(correlation))

        # Statistical significance
        if p_value < self.significance_level:
            confidence += 0.3

        # Sample size factor
        confidence += min(0.2, sample_size / 50)

        # Volatility penalty
        confidence -= min(0.3, volatility)

        return max(0.0, min(1.0, confidence))

    def _calculate_stagnation_score(self, improvement_history: List[float]) -> float:
        """Calculate improvement stagnation score"""
        if len(improvement_history) < 5:
            return 0.0

        recent_improvements = improvement_history[-5:]
        avg_recent = np.mean(recent_improvements)

        # Compare with earlier improvements
        if len(improvement_history) >= 10:
            earlier_improvements = improvement_history[-10:-5]
            avg_earlier = np.mean(earlier_improvements)

            # Stagnation score: how much recent performance lags behind earlier
            if avg_earlier > 0:
                stagnation = max(0.0, (avg_earlier - avg_recent) / avg_earlier)
            else:
                stagnation = 1.0 if avg_recent <= 0 else 0.0
        else:
            # For shorter histories, just check if recent improvements are low
            stagnation = 1.0 - min(1.0, avg_recent / self.plateau_threshold)

        return min(1.0, stagnation)

    def _determine_plateau_status(
        self,
        below_threshold_count: int,
        correlation_indicates_plateau: bool,
        stagnation_score: float
    ) -> PlateauStatus:
        """Determine plateau status from multiple indicators"""

        # Performance decline (check first - highest priority)
        if stagnation_score > 0.8:
            return PlateauStatus.PERFORMANCE_DECLINE

        # Strong plateau indicators
        elif (below_threshold_count >= self.plateau_consecutive_iterations and
              correlation_indicates_plateau and
              stagnation_score > 0.7):
            return PlateauStatus.CONFIRMED_PLATEAU

        # Early plateau indicators
        elif (below_threshold_count >= 2 and
              (correlation_indicates_plateau or stagnation_score > 0.5)):
            return PlateauStatus.EARLY_PLATEAU

        else:
            return PlateauStatus.NO_PLATEAU

    def _find_plateau_start(self, improvement_history: List[float]) -> Optional[int]:
        """Find the iteration where plateau started"""
        for i in range(len(improvement_history) - self.plateau_consecutive_iterations + 1):
            consecutive_low = all(
                improvement_history[i + j] < self.plateau_threshold
                for j in range(self.plateau_consecutive_iterations)
            )
            if consecutive_low:
                return i
        return None

    def _generate_plateau_recommendation(
        self,
        status: PlateauStatus,
        stagnation_score: float,
        correlation: float
    ) -> str:
        """Generate recommendation based on plateau analysis"""

        if status == PlateauStatus.CONFIRMED_PLATEAU:
            return "Performance plateau confirmed. Consider: 1) Adjusting learning parameters, 2) Adding new training data, 3) Changing generation strategy, or 4) Stopping training."

        elif status == PlateauStatus.EARLY_PLATEAU:
            return "Early plateau indicators detected. Monitor closely and consider parameter adjustments if trend continues."

        elif status == PlateauStatus.PERFORMANCE_DECLINE:
            return "Performance decline detected. Consider: 1) Reverting to previous checkpoint, 2) Reducing learning rate, or 3) Investigating data quality issues."

        else:
            return "No plateau detected. Continue training with current configuration."
