"""
Session Comparison Analyzer
Implements comprehensive session comparison and historical analysis following 2025 best practices.

Key Features (2025 Standards):
- Multi-dimensional session comparison algorithms
- Statistical significance testing for performance differences
- Pattern identification across training sessions
- Optimization opportunity detection
- Benchmarking and performance ranking
- Historical trend analysis
- Correlation analysis between session parameters and outcomes
"""

import logging
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func, text
from sqlalchemy.orm import selectinload

from ...database.models import TrainingSession, TrainingIteration, GenerationSession
from .performance_improvement_calculator import PerformanceImprovementCalculator
from .session_summary_reporter import SessionSummaryReporter, SessionSummary
from ...utils.datetime_utils import naive_utc_now

logger = logging.getLogger(__name__)


class ComparisonDimension(Enum):
    """Dimensions for session comparison"""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"
    SPEED = "speed"
    QUALITY = "quality"
    RESOURCE_USAGE = "resource_usage"


class ComparisonMethod(Enum):
    """Statistical methods for comparison"""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CORRELATION = "correlation"
    EFFECT_SIZE = "effect_size"


@dataclass
class SessionComparisonResult:
    """Result of comparing two training sessions"""
    session_a_id: str
    session_b_id: str
    comparison_dimension: ComparisonDimension

    # Statistical results
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]

    # Performance metrics
    session_a_score: float
    session_b_score: float
    performance_difference: float
    relative_improvement: float

    # Insights and recommendations
    winner: str  # "session_a", "session_b", or "no_significant_difference"
    insights: List[str]
    recommendations: List[str]


@dataclass
class MultiSessionAnalysis:
    """Analysis of multiple training sessions"""
    session_ids: List[str]
    analysis_period: Tuple[datetime, datetime]

    # Performance rankings
    performance_ranking: List[Tuple[str, float]]  # (session_id, score)
    efficiency_ranking: List[Tuple[str, float]]
    stability_ranking: List[Tuple[str, float]]

    # Pattern identification
    high_performers: List[str]
    low_performers: List[str]
    outliers: List[str]

    # Optimization opportunities
    common_success_patterns: List[Dict[str, Any]]
    common_failure_patterns: List[Dict[str, Any]]
    optimization_recommendations: List[str]

    # Statistical insights
    performance_distribution: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    cluster_analysis: Dict[str, List[str]]


@dataclass
class BenchmarkResult:
    """Benchmarking result against historical data"""
    session_id: str
    benchmark_period: Tuple[datetime, datetime]

    # Percentile rankings
    performance_percentile: float
    efficiency_percentile: float
    speed_percentile: float

    # Comparisons
    vs_average: Dict[str, float]
    vs_best: Dict[str, float]
    vs_recent: Dict[str, float]

    # Classification
    performance_tier: str  # "excellent", "good", "average", "below_average", "poor"
    improvement_potential: float

    # Insights
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class SessionComparisonAnalyzer:
    """
    Advanced session comparison analyzer implementing 2025 best practices.

    Features:
    - Multi-dimensional statistical comparison
    - Pattern identification and clustering
    - Optimization opportunity detection
    - Historical benchmarking
    - Correlation analysis
    - Performance ranking and classification
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.performance_calculator = PerformanceImprovementCalculator(db_session)
        self.summary_reporter = SessionSummaryReporter(db_session)

        # Statistical thresholds (2025 best practices)
        self.significance_level = 0.05
        self.effect_size_thresholds = {
            "small": 0.2,
            "medium": 0.5,
            "large": 0.8
        }

        # Performance classification thresholds
        self.performance_tiers = {
            "excellent": 0.9,
            "good": 0.75,
            "average": 0.5,
            "below_average": 0.25,
            "poor": 0.0
        }

    async def compare_sessions(
        self,
        session_a_id: str,
        session_b_id: str,
        dimension: ComparisonDimension = ComparisonDimension.PERFORMANCE,
        method: ComparisonMethod = ComparisonMethod.T_TEST
    ) -> SessionComparisonResult:
        """
        Compare two training sessions across specified dimension.

        Implements 2025 best practices:
        - Statistical significance testing
        - Effect size calculation
        - Confidence intervals
        - Multi-dimensional analysis
        """
        try:
            # Get session data
            session_a_task = self._get_session_with_iterations(session_a_id)
            session_b_task = self._get_session_with_iterations(session_b_id)

            session_a_data, session_b_data = await asyncio.gather(session_a_task, session_b_task)

            if not session_a_data or not session_b_data:
                raise ValueError("One or both sessions not found")

            session_a, iterations_a = session_a_data
            session_b, iterations_b = session_b_data

            # Extract comparison metrics
            metrics_a = await self._extract_comparison_metrics(session_a, iterations_a, dimension)
            metrics_b = await self._extract_comparison_metrics(session_b, iterations_b, dimension)

            # Perform statistical comparison
            statistical_result = await self._perform_statistical_comparison(
                metrics_a, metrics_b, method
            )

            # Calculate performance scores
            score_a = np.mean(metrics_a) if metrics_a else 0.0
            score_b = np.mean(metrics_b) if metrics_b else 0.0

            performance_difference = score_b - score_a
            relative_improvement = (performance_difference / score_a) if score_a > 0 else 0.0

            # Determine winner
            if statistical_result["significant"]:
                winner = "session_b" if performance_difference > 0 else "session_a"
            else:
                winner = "no_significant_difference"

            # Generate insights and recommendations
            insights = await self._generate_comparison_insights(
                session_a, session_b, iterations_a, iterations_b, dimension, statistical_result
            )

            recommendations = await self._generate_comparison_recommendations(
                session_a, session_b, winner, dimension, statistical_result
            )

            return SessionComparisonResult(
                session_a_id=session_a_id,
                session_b_id=session_b_id,
                comparison_dimension=dimension,
                statistical_significance=statistical_result["significant"],
                p_value=statistical_result["p_value"],
                effect_size=statistical_result["effect_size"],
                confidence_interval=statistical_result["confidence_interval"],
                session_a_score=score_a,
                session_b_score=score_b,
                performance_difference=performance_difference,
                relative_improvement=relative_improvement,
                winner=winner,
                insights=insights,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Error comparing sessions {session_a_id} and {session_b_id}: {e}")
            raise

    async def analyze_multiple_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50
    ) -> MultiSessionAnalysis:
        """
        Analyze multiple training sessions for patterns and optimization opportunities.

        Implements 2025 best practices:
        - Clustering analysis
        - Pattern identification
        - Performance ranking
        - Correlation analysis
        """
        try:
            # Get sessions to analyze
            if session_ids:
                sessions_data = await self._get_multiple_sessions_by_ids(session_ids)
            else:
                sessions_data = await self._get_sessions_by_date_range(start_date, end_date, limit)

            if len(sessions_data) < 2:
                raise ValueError("Need at least 2 sessions for multi-session analysis")

            # Extract features for analysis
            session_features = {}
            for session, iterations in sessions_data:
                features = await self._extract_session_features(session, iterations)
                session_features[session.session_id] = features

            # Performance rankings
            performance_ranking = await self._calculate_performance_ranking(session_features)
            efficiency_ranking = await self._calculate_efficiency_ranking(session_features)
            stability_ranking = await self._calculate_stability_ranking(session_features)

            # Pattern identification
            patterns = await self._identify_patterns(session_features)

            # Clustering analysis
            clusters = await self._perform_clustering_analysis(session_features)

            # Correlation analysis
            correlation_matrix = await self._calculate_correlation_matrix(session_features)

            # Statistical distribution
            performance_distribution = await self._calculate_performance_distribution(session_features)

            # Generate optimization recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(
                session_features, patterns, clusters
            )

            analysis_period = (
                min(session.started_at for session, _ in sessions_data),
                max(session.completed_at or session.started_at for session, _ in sessions_data)
            )

            return MultiSessionAnalysis(
                session_ids=list(session_features.keys()),
                analysis_period=analysis_period,
                performance_ranking=performance_ranking,
                efficiency_ranking=efficiency_ranking,
                stability_ranking=stability_ranking,
                high_performers=patterns["high_performers"],
                low_performers=patterns["low_performers"],
                outliers=patterns["outliers"],
                common_success_patterns=patterns["success_patterns"],
                common_failure_patterns=patterns["failure_patterns"],
                optimization_recommendations=optimization_recommendations,
                performance_distribution=performance_distribution,
                correlation_matrix=correlation_matrix,
                cluster_analysis=clusters
            )

        except Exception as e:
            self.logger.error(f"Error analyzing multiple sessions: {e}")
            raise

    async def benchmark_session(
        self,
        session_id: str,
        benchmark_period_days: int = 30
    ) -> BenchmarkResult:
        """
        Benchmark a session against historical performance data.

        Implements 2025 best practices:
        - Percentile ranking
        - Multi-dimensional comparison
        - Performance tier classification
        - Improvement potential analysis
        """
        try:
            # Get target session
            target_session_data = await self._get_session_with_iterations(session_id)
            if not target_session_data:
                raise ValueError(f"Session {session_id} not found")

            target_session, target_iterations = target_session_data

            # Get historical sessions for benchmarking
            end_date = target_session.started_at
            start_date = end_date - timedelta(days=benchmark_period_days)

            historical_sessions = await self._get_sessions_by_date_range(start_date, end_date, 100)

            if len(historical_sessions) < 5:
                self.logger.warning(f"Limited historical data for benchmarking: {len(historical_sessions)} sessions")

            # Extract features for target session
            target_features = await self._extract_session_features(target_session, target_iterations)

            # Extract features for historical sessions
            historical_features = {}
            for session, iterations in historical_sessions:
                if session.session_id != session_id:  # Exclude target session
                    features = await self._extract_session_features(session, iterations)
                    historical_features[session.session_id] = features

            # Calculate percentile rankings
            percentiles = await self._calculate_percentile_rankings(target_features, historical_features)

            # Compare against averages, best, and recent
            comparisons = await self._calculate_benchmark_comparisons(target_features, historical_features)

            # Classify performance tier
            performance_tier = await self._classify_performance_tier(target_features, historical_features)

            # Calculate improvement potential
            improvement_potential = await self._calculate_improvement_potential(target_features, historical_features)

            # Generate insights
            strengths, weaknesses = await self._identify_strengths_weaknesses(target_features, historical_features)
            recommendations = await self._generate_benchmark_recommendations(
                target_features, historical_features, performance_tier
            )

            return BenchmarkResult(
                session_id=session_id,
                benchmark_period=(start_date, end_date),
                performance_percentile=percentiles["performance"],
                efficiency_percentile=percentiles["efficiency"],
                speed_percentile=percentiles["speed"],
                vs_average=comparisons["vs_average"],
                vs_best=comparisons["vs_best"],
                vs_recent=comparisons["vs_recent"],
                performance_tier=performance_tier,
                improvement_potential=improvement_potential,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Error benchmarking session {session_id}: {e}")
            raise

    # Helper methods for data retrieval and analysis

    async def _get_session_with_iterations(self, session_id: str) -> Optional[Tuple[TrainingSession, List[TrainingIteration]]]:
        """Get session with its iterations"""
        try:
            # Get session
            session_query = select(TrainingSession).where(TrainingSession.session_id == session_id)
            session_result = await self.db_session.execute(session_query)
            session = session_result.scalar_one_or_none()

            if not session:
                return None

            # Get iterations
            iterations_query = (
                select(TrainingIteration)
                .where(TrainingIteration.session_id == session_id)
                .order_by(TrainingIteration.iteration)
            )
            iterations_result = await self.db_session.execute(iterations_query)
            iterations = iterations_result.scalars().all()

            return (session, list(iterations))

        except Exception as e:
            self.logger.error(f"Error getting session with iterations: {e}")
            return None

    async def _get_multiple_sessions_by_ids(self, session_ids: List[str]) -> List[Tuple[TrainingSession, List[TrainingIteration]]]:
        """Get multiple sessions by their IDs"""
        try:
            sessions_data = []

            for session_id in session_ids:
                session_data = await self._get_session_with_iterations(session_id)
                if session_data:
                    sessions_data.append(session_data)

            return sessions_data

        except Exception as e:
            self.logger.error(f"Error getting multiple sessions by IDs: {e}")
            return []

    async def _get_sessions_by_date_range(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> List[Tuple[TrainingSession, List[TrainingIteration]]]:
        """Get sessions within date range"""
        try:
            # Build query
            query = select(TrainingSession)

            if start_date:
                query = query.where(TrainingSession.started_at >= start_date)
            if end_date:
                query = query.where(TrainingSession.started_at <= end_date)

            query = query.order_by(desc(TrainingSession.started_at)).limit(limit)

            # Execute query
            result = await self.db_session.execute(query)
            sessions = result.scalars().all()

            # Get iterations for each session
            sessions_data = []
            for session in sessions:
                iterations_query = (
                    select(TrainingIteration)
                    .where(TrainingIteration.session_id == session.session_id)
                    .order_by(TrainingIteration.iteration)
                )
                iterations_result = await self.db_session.execute(iterations_query)
                iterations = iterations_result.scalars().all()

                sessions_data.append((session, list(iterations)))

            return sessions_data

        except Exception as e:
            self.logger.error(f"Error getting sessions by date range: {e}")
            return []

    async def _extract_comparison_metrics(
        self,
        session: TrainingSession,
        iterations: List[TrainingIteration],
        dimension: ComparisonDimension
    ) -> List[float]:
        """Extract metrics for comparison based on dimension"""
        try:
            metrics = []

            if dimension == ComparisonDimension.PERFORMANCE:
                # Extract performance scores from iterations
                for iteration in iterations:
                    if iteration.performance_metrics:
                        score = iteration.performance_metrics.get('model_accuracy', 0.0)
                        metrics.append(score)

                # Add session-level performance
                if session.current_performance:
                    metrics.append(session.current_performance)

            elif dimension == ComparisonDimension.EFFICIENCY:
                # Calculate efficiency metrics (performance per time)
                for iteration in iterations:
                    if iteration.duration_seconds and iteration.improvement_score:
                        efficiency = iteration.improvement_score / (iteration.duration_seconds / 3600)
                        metrics.append(efficiency)

            elif dimension == ComparisonDimension.STABILITY:
                # Calculate stability (consistency of performance)
                performance_scores = []
                for iteration in iterations:
                    if iteration.performance_metrics:
                        score = iteration.performance_metrics.get('model_accuracy', 0.0)
                        performance_scores.append(score)

                if len(performance_scores) > 1:
                    stability = 1.0 - (np.std(performance_scores) / np.mean(performance_scores))
                    metrics.append(max(0.0, stability))

            elif dimension == ComparisonDimension.SPEED:
                # Extract iteration durations
                for iteration in iterations:
                    if iteration.duration_seconds:
                        # Convert to speed (inverse of duration)
                        speed = 1.0 / (iteration.duration_seconds / 60)  # iterations per minute
                        metrics.append(speed)

            elif dimension == ComparisonDimension.QUALITY:
                # Extract quality metrics
                for iteration in iterations:
                    if iteration.performance_metrics:
                        quality = iteration.performance_metrics.get('quality_score', 0.0)
                        metrics.append(quality)

            elif dimension == ComparisonDimension.RESOURCE_USAGE:
                # Extract resource usage efficiency
                for iteration in iterations:
                    if iteration.performance_metrics:
                        memory_usage = iteration.performance_metrics.get('memory_usage_mb', 1000)
                        performance = iteration.performance_metrics.get('model_accuracy', 0.0)
                        if memory_usage > 0:
                            efficiency = performance / (memory_usage / 1000)  # performance per GB
                            metrics.append(efficiency)

            return metrics

        except Exception as e:
            self.logger.error(f"Error extracting comparison metrics: {e}")
            return []

    async def _perform_statistical_comparison(
        self,
        metrics_a: List[float],
        metrics_b: List[float],
        method: ComparisonMethod
    ) -> Dict[str, Any]:
        """Perform statistical comparison between two metric sets"""
        try:
            if not metrics_a or not metrics_b:
                return {
                    "significant": False,
                    "p_value": 1.0,
                    "effect_size": 0.0,
                    "confidence_interval": (0.0, 0.0)
                }

            if method == ComparisonMethod.T_TEST:
                # Perform independent t-test
                statistic, p_value = ttest_ind(metrics_a, metrics_b)

                # Calculate Cohen's d (effect size)
                pooled_std = np.sqrt(
                    ((len(metrics_a) - 1) * np.var(metrics_a, ddof=1) +
                     (len(metrics_b) - 1) * np.var(metrics_b, ddof=1)) /
                    (len(metrics_a) + len(metrics_b) - 2)
                )

                cohens_d = (np.mean(metrics_b) - np.mean(metrics_a)) / pooled_std if pooled_std > 0 else 0.0

                # Calculate confidence interval for difference in means
                diff_mean = np.mean(metrics_b) - np.mean(metrics_a)
                se_diff = pooled_std * np.sqrt(1/len(metrics_a) + 1/len(metrics_b))
                ci_lower = diff_mean - 1.96 * se_diff
                ci_upper = diff_mean + 1.96 * se_diff

                return {
                    "significant": p_value < self.significance_level,
                    "p_value": p_value,
                    "effect_size": abs(cohens_d),
                    "confidence_interval": (ci_lower, ci_upper)
                }

            elif method == ComparisonMethod.MANN_WHITNEY:
                # Perform Mann-Whitney U test (non-parametric)
                statistic, p_value = mannwhitneyu(metrics_a, metrics_b, alternative='two-sided')

                # Calculate effect size (r = Z / sqrt(N))
                n1, n2 = len(metrics_a), len(metrics_b)
                z_score = abs(statistic - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                effect_size = z_score / np.sqrt(n1 + n2)

                return {
                    "significant": p_value < self.significance_level,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "confidence_interval": (0.0, 0.0)  # Not applicable for Mann-Whitney
                }

            elif method == ComparisonMethod.CORRELATION:
                # Calculate correlation between metrics
                if len(metrics_a) == len(metrics_b):
                    correlation, p_value = pearsonr(metrics_a, metrics_b)
                    return {
                        "significant": p_value < self.significance_level,
                        "p_value": p_value,
                        "effect_size": abs(correlation),
                        "confidence_interval": (0.0, 0.0)  # Not applicable for correlation
                    }
                else:
                    return {
                        "significant": False,
                        "p_value": 1.0,
                        "effect_size": 0.0,
                        "confidence_interval": (0.0, 0.0)
                    }

            else:
                raise ValueError(f"Unsupported comparison method: {method}")

        except Exception as e:
            self.logger.error(f"Error performing statistical comparison: {e}")
            return {
                "significant": False,
                "p_value": 1.0,
                "effect_size": 0.0,
                "confidence_interval": (0.0, 0.0)
            }

    async def _extract_session_features(
        self,
        session: TrainingSession,
        iterations: List[TrainingIteration]
    ) -> Dict[str, float]:
        """Extract comprehensive features for session analysis"""
        try:
            features = {}

            # Basic session metrics
            features["total_iterations"] = len(iterations)
            features["duration_hours"] = session.total_training_time_seconds / 3600 if session.total_training_time_seconds else 0.0
            features["initial_performance"] = session.initial_performance or 0.0
            features["final_performance"] = session.current_performance or 0.0
            features["best_performance"] = session.best_performance or 0.0

            # Performance improvement
            if session.initial_performance and session.current_performance:
                features["total_improvement"] = session.current_performance - session.initial_performance
                features["improvement_rate"] = features["total_improvement"] / features["duration_hours"] if features["duration_hours"] > 0 else 0.0
            else:
                features["total_improvement"] = 0.0
                features["improvement_rate"] = 0.0

            # Iteration-level statistics
            if iterations:
                durations = [it.duration_seconds for it in iterations if it.duration_seconds]
                improvements = [it.improvement_score for it in iterations if it.improvement_score]

                features["avg_iteration_duration"] = np.mean(durations) if durations else 0.0
                features["std_iteration_duration"] = np.std(durations) if len(durations) > 1 else 0.0
                features["avg_improvement_per_iteration"] = np.mean(improvements) if improvements else 0.0
                features["std_improvement_per_iteration"] = np.std(improvements) if len(improvements) > 1 else 0.0

                # Success rate
                successful = sum(1 for it in iterations if it.status == 'completed')
                features["success_rate"] = successful / len(iterations)

                # Efficiency metrics
                if durations and improvements:
                    efficiency_scores = [imp / (dur / 3600) for imp, dur in zip(improvements, durations) if dur > 0]
                    features["avg_efficiency"] = np.mean(efficiency_scores) if efficiency_scores else 0.0

                # Stability (consistency)
                performance_scores = []
                for iteration in iterations:
                    if iteration.performance_metrics:
                        score = iteration.performance_metrics.get('model_accuracy', 0.0)
                        performance_scores.append(score)

                if len(performance_scores) > 1:
                    features["performance_stability"] = 1.0 - (np.std(performance_scores) / np.mean(performance_scores))
                else:
                    features["performance_stability"] = 1.0
            else:
                features.update({
                    "avg_iteration_duration": 0.0,
                    "std_iteration_duration": 0.0,
                    "avg_improvement_per_iteration": 0.0,
                    "std_improvement_per_iteration": 0.0,
                    "success_rate": 0.0,
                    "avg_efficiency": 0.0,
                    "performance_stability": 0.0
                })

            # Resource utilization
            memory_usages = []
            for iteration in iterations:
                if iteration.performance_metrics:
                    memory = iteration.performance_metrics.get('memory_usage_mb', 0)
                    if memory > 0:
                        memory_usages.append(memory)

            if memory_usages:
                features["avg_memory_usage"] = np.mean(memory_usages)
                features["peak_memory_usage"] = max(memory_usages)
                features["memory_efficiency"] = features["final_performance"] / (features["avg_memory_usage"] / 1000) if features["avg_memory_usage"] > 0 else 0.0
            else:
                features["avg_memory_usage"] = 0.0
                features["peak_memory_usage"] = 0.0
                features["memory_efficiency"] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"Error extracting session features: {e}")
            return {}

    async def _generate_comparison_insights(
        self,
        session_a: TrainingSession,
        session_b: TrainingSession,
        iterations_a: List[TrainingIteration],
        iterations_b: List[TrainingIteration],
        dimension: ComparisonDimension,
        statistical_result: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from session comparison"""
        try:
            insights = []

            # Statistical significance insight
            if statistical_result["significant"]:
                effect_size = statistical_result["effect_size"]
                if effect_size >= self.effect_size_thresholds["large"]:
                    insights.append(f"Large effect size ({effect_size:.3f}) indicates substantial difference in {dimension.value}")
                elif effect_size >= self.effect_size_thresholds["medium"]:
                    insights.append(f"Medium effect size ({effect_size:.3f}) indicates moderate difference in {dimension.value}")
                else:
                    insights.append(f"Small effect size ({effect_size:.3f}) indicates minor difference in {dimension.value}")
            else:
                insights.append(f"No statistically significant difference found in {dimension.value}")

            # Session-specific insights
            if len(iterations_a) != len(iterations_b):
                insights.append(f"Sessions have different iteration counts: {len(iterations_a)} vs {len(iterations_b)}")

            # Performance insights
            if session_a.current_performance and session_b.current_performance:
                perf_diff = abs(session_a.current_performance - session_b.current_performance)
                if perf_diff > 0.1:
                    insights.append(f"Significant performance difference: {perf_diff:.1%}")

            # Duration insights
            duration_a = session_a.total_training_time_seconds or 0
            duration_b = session_b.total_training_time_seconds or 0
            if abs(duration_a - duration_b) > 3600:  # More than 1 hour difference
                insights.append(f"Substantial training time difference: {abs(duration_a - duration_b) / 3600:.1f} hours")

            return insights

        except Exception as e:
            self.logger.error(f"Error generating comparison insights: {e}")
            return []

    async def _generate_comparison_recommendations(
        self,
        session_a: TrainingSession,
        session_b: TrainingSession,
        winner: str,
        dimension: ComparisonDimension,
        statistical_result: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations from session comparison"""
        try:
            recommendations = []

            if winner == "no_significant_difference":
                recommendations.append(f"No clear winner in {dimension.value} - both approaches are comparable")
                recommendations.append("Consider other dimensions for comparison or gather more data")
            else:
                winning_session = session_b if winner == "session_b" else session_a
                losing_session = session_a if winner == "session_b" else session_b

                recommendations.append(f"Adopt configuration from {winning_session.session_id} for better {dimension.value}")

                # Specific recommendations based on dimension
                if dimension == ComparisonDimension.PERFORMANCE:
                    recommendations.append("Analyze successful session's hyperparameters and data generation strategy")
                elif dimension == ComparisonDimension.EFFICIENCY:
                    recommendations.append("Investigate resource allocation and optimization techniques from winning session")
                elif dimension == ComparisonDimension.STABILITY:
                    recommendations.append("Review training consistency factors and error handling approaches")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating comparison recommendations: {e}")
            return []

    async def _calculate_performance_ranking(self, session_features: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Calculate performance ranking for sessions"""
        try:
            rankings = []
            for session_id, features in session_features.items():
                # Composite performance score
                score = (
                    features.get("final_performance", 0.0) * 0.4 +
                    features.get("total_improvement", 0.0) * 0.3 +
                    features.get("improvement_rate", 0.0) * 0.2 +
                    features.get("success_rate", 0.0) * 0.1
                )
                rankings.append((session_id, score))

            return sorted(rankings, key=lambda x: x[1], reverse=True)

        except Exception as e:
            self.logger.error(f"Error calculating performance ranking: {e}")
            return []

    async def _calculate_efficiency_ranking(self, session_features: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Calculate efficiency ranking for sessions"""
        try:
            rankings = []
            for session_id, features in session_features.items():
                # Efficiency score (performance per resource)
                score = (
                    features.get("improvement_rate", 0.0) * 0.4 +
                    features.get("avg_efficiency", 0.0) * 0.3 +
                    features.get("memory_efficiency", 0.0) * 0.2 +
                    (1.0 / max(features.get("avg_iteration_duration", 1.0), 1.0)) * 0.1
                )
                rankings.append((session_id, score))

            return sorted(rankings, key=lambda x: x[1], reverse=True)

        except Exception as e:
            self.logger.error(f"Error calculating efficiency ranking: {e}")
            return []

    async def _calculate_stability_ranking(self, session_features: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Calculate stability ranking for sessions"""
        try:
            rankings = []
            for session_id, features in session_features.items():
                # Stability score
                score = (
                    features.get("performance_stability", 0.0) * 0.5 +
                    features.get("success_rate", 0.0) * 0.3 +
                    (1.0 - min(features.get("std_improvement_per_iteration", 1.0), 1.0)) * 0.2
                )
                rankings.append((session_id, score))

            return sorted(rankings, key=lambda x: x[1], reverse=True)

        except Exception as e:
            self.logger.error(f"Error calculating stability ranking: {e}")
            return []

    async def _identify_patterns(self, session_features: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Identify patterns in session performance"""
        try:
            # Calculate percentiles for classification
            all_scores = [features.get("final_performance", 0.0) for features in session_features.values()]
            if not all_scores:
                return {
                    "high_performers": [],
                    "low_performers": [],
                    "outliers": [],
                    "success_patterns": [],
                    "failure_patterns": []
                }

            high_threshold = np.percentile(all_scores, 80)
            low_threshold = np.percentile(all_scores, 20)

            high_performers = []
            low_performers = []
            outliers = []

            for session_id, features in session_features.items():
                score = features.get("final_performance", 0.0)

                if score >= high_threshold:
                    high_performers.append(session_id)
                elif score <= low_threshold:
                    low_performers.append(session_id)

                # Detect outliers (sessions with unusual characteristics)
                if (features.get("duration_hours", 0.0) > np.percentile([f.get("duration_hours", 0.0) for f in session_features.values()], 95) or
                    features.get("total_iterations", 0) > np.percentile([f.get("total_iterations", 0) for f in session_features.values()], 95)):
                    outliers.append(session_id)

            # Identify success patterns
            success_patterns = []
            if high_performers:
                high_perf_features = [session_features[sid] for sid in high_performers]
                avg_success_rate = np.mean([f.get("success_rate", 0.0) for f in high_perf_features])
                avg_efficiency = np.mean([f.get("avg_efficiency", 0.0) for f in high_perf_features])

                if avg_success_rate > 0.9:
                    success_patterns.append({"pattern": "high_success_rate", "value": avg_success_rate})
                if avg_efficiency > np.mean([f.get("avg_efficiency", 0.0) for f in session_features.values()]):
                    success_patterns.append({"pattern": "high_efficiency", "value": avg_efficiency})

            # Identify failure patterns
            failure_patterns = []
            if low_performers:
                low_perf_features = [session_features[sid] for sid in low_performers]
                avg_success_rate = np.mean([f.get("success_rate", 0.0) for f in low_perf_features])
                avg_duration = np.mean([f.get("duration_hours", 0.0) for f in low_perf_features])

                if avg_success_rate < 0.7:
                    failure_patterns.append({"pattern": "low_success_rate", "value": avg_success_rate})
                if avg_duration > np.mean([f.get("duration_hours", 0.0) for f in session_features.values()]) * 1.5:
                    failure_patterns.append({"pattern": "excessive_duration", "value": avg_duration})

            return {
                "high_performers": high_performers,
                "low_performers": low_performers,
                "outliers": outliers,
                "success_patterns": success_patterns,
                "failure_patterns": failure_patterns
            }

        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")
            return {
                "high_performers": [],
                "low_performers": [],
                "outliers": [],
                "success_patterns": [],
                "failure_patterns": []
            }

    async def _perform_clustering_analysis(self, session_features: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Perform clustering analysis on sessions"""
        try:
            if len(session_features) < 3:
                return {"cluster_0": list(session_features.keys())}

            # Prepare feature matrix
            feature_names = ["final_performance", "total_improvement", "improvement_rate", "success_rate", "avg_efficiency"]
            feature_matrix = []
            session_ids = []

            for session_id, features in session_features.items():
                feature_vector = [features.get(name, 0.0) for name in feature_names]
                feature_matrix.append(feature_vector)
                session_ids.append(session_id)

            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # Perform K-means clustering
            n_clusters = min(3, len(session_features))  # Max 3 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

            # Group sessions by cluster
            clusters = {}
            for i, session_id in enumerate(session_ids):
                cluster_id = f"cluster_{cluster_labels[i]}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(session_id)

            return clusters

        except Exception as e:
            self.logger.error(f"Error performing clustering analysis: {e}")
            return {"cluster_0": list(session_features.keys())}

    async def _calculate_correlation_matrix(self, session_features: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between features"""
        try:
            if len(session_features) < 3:
                return {}

            # Prepare feature matrix
            feature_names = ["final_performance", "total_improvement", "improvement_rate", "success_rate", "avg_efficiency", "performance_stability"]
            feature_matrix = []

            for features in session_features.values():
                feature_vector = [features.get(name, 0.0) for name in feature_names]
                feature_matrix.append(feature_vector)

            # Calculate correlation matrix
            feature_matrix = np.array(feature_matrix)
            correlation_matrix = np.corrcoef(feature_matrix.T)

            # Convert to dictionary format
            result = {}
            for i, name_i in enumerate(feature_names):
                result[name_i] = {}
                for j, name_j in enumerate(feature_names):
                    result[name_i][name_j] = float(correlation_matrix[i, j])

            return result

        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return {}

    async def _calculate_performance_distribution(self, session_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate performance distribution statistics"""
        try:
            performance_scores = [features.get("final_performance", 0.0) for features in session_features.values()]

            if not performance_scores:
                return {}

            return {
                "mean": float(np.mean(performance_scores)),
                "median": float(np.median(performance_scores)),
                "std": float(np.std(performance_scores)),
                "min": float(np.min(performance_scores)),
                "max": float(np.max(performance_scores)),
                "q25": float(np.percentile(performance_scores, 25)),
                "q75": float(np.percentile(performance_scores, 75))
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance distribution: {e}")
            return {}

    async def _generate_optimization_recommendations(
        self,
        session_features: Dict[str, Dict[str, float]],
        patterns: Dict[str, Any],
        clusters: Dict[str, List[str]]
    ) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        try:
            recommendations = []

            # Recommendations based on high performers
            if patterns["high_performers"]:
                high_perf_features = [session_features[sid] for sid in patterns["high_performers"]]
                avg_success_rate = np.mean([f.get("success_rate", 0.0) for f in high_perf_features])
                avg_efficiency = np.mean([f.get("avg_efficiency", 0.0) for f in high_perf_features])

                if avg_success_rate > 0.9:
                    recommendations.append(f"Target success rate above {avg_success_rate:.1%} based on high-performing sessions")

                if avg_efficiency > 0.1:
                    recommendations.append(f"Optimize for efficiency scores above {avg_efficiency:.3f}")

            # Recommendations based on failure patterns
            if patterns["failure_patterns"]:
                for pattern in patterns["failure_patterns"]:
                    if pattern["pattern"] == "low_success_rate":
                        recommendations.append("Investigate and address causes of iteration failures")
                    elif pattern["pattern"] == "excessive_duration":
                        recommendations.append("Implement timeout mechanisms and optimize training speed")

            # Cluster-based recommendations
            if len(clusters) > 1:
                recommendations.append("Consider different optimization strategies for different session types")

            # General recommendations
            all_scores = [f.get("final_performance", 0.0) for f in session_features.values()]
            if all_scores and np.std(all_scores) > 0.1:
                recommendations.append("High performance variance detected - standardize training procedures")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return []

    # Additional helper methods for benchmarking

    async def _calculate_percentile_rankings(
        self,
        target_features: Dict[str, float],
        historical_features: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate percentile rankings for target session"""
        try:
            if not historical_features:
                return {"performance": 50.0, "efficiency": 50.0, "speed": 50.0}

            # Performance percentile
            historical_performance = [f.get("final_performance", 0.0) for f in historical_features.values()]
            target_performance = target_features.get("final_performance", 0.0)
            performance_percentile = stats.percentileofscore(historical_performance, target_performance)

            # Efficiency percentile
            historical_efficiency = [f.get("avg_efficiency", 0.0) for f in historical_features.values()]
            target_efficiency = target_features.get("avg_efficiency", 0.0)
            efficiency_percentile = stats.percentileofscore(historical_efficiency, target_efficiency)

            # Speed percentile (inverse of duration)
            historical_speed = [1.0 / max(f.get("avg_iteration_duration", 1.0), 1.0) for f in historical_features.values()]
            target_speed = 1.0 / max(target_features.get("avg_iteration_duration", 1.0), 1.0)
            speed_percentile = stats.percentileofscore(historical_speed, target_speed)

            return {
                "performance": performance_percentile,
                "efficiency": efficiency_percentile,
                "speed": speed_percentile
            }

        except Exception as e:
            self.logger.error(f"Error calculating percentile rankings: {e}")
            return {"performance": 50.0, "efficiency": 50.0, "speed": 50.0}

    async def _calculate_benchmark_comparisons(
        self,
        target_features: Dict[str, float],
        historical_features: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate benchmark comparisons (vs average, best, recent)"""
        try:
            if not historical_features:
                return {"vs_average": {}, "vs_best": {}, "vs_recent": {}}

            # Calculate averages
            avg_performance = np.mean([f.get("final_performance", 0.0) for f in historical_features.values()])
            avg_efficiency = np.mean([f.get("avg_efficiency", 0.0) for f in historical_features.values()])
            avg_improvement = np.mean([f.get("total_improvement", 0.0) for f in historical_features.values()])

            # Calculate best values
            best_performance = max([f.get("final_performance", 0.0) for f in historical_features.values()])
            best_efficiency = max([f.get("avg_efficiency", 0.0) for f in historical_features.values()])
            best_improvement = max([f.get("total_improvement", 0.0) for f in historical_features.values()])

            # Calculate recent averages (last 25% of sessions)
            recent_count = max(1, len(historical_features) // 4)
            recent_sessions = list(historical_features.values())[-recent_count:]
            recent_performance = np.mean([f.get("final_performance", 0.0) for f in recent_sessions])
            recent_efficiency = np.mean([f.get("avg_efficiency", 0.0) for f in recent_sessions])
            recent_improvement = np.mean([f.get("total_improvement", 0.0) for f in recent_sessions])

            # Calculate differences
            target_performance = target_features.get("final_performance", 0.0)
            target_efficiency = target_features.get("avg_efficiency", 0.0)
            target_improvement = target_features.get("total_improvement", 0.0)

            return {
                "vs_average": {
                    "performance": target_performance - avg_performance,
                    "efficiency": target_efficiency - avg_efficiency,
                    "improvement": target_improvement - avg_improvement
                },
                "vs_best": {
                    "performance": target_performance - best_performance,
                    "efficiency": target_efficiency - best_efficiency,
                    "improvement": target_improvement - best_improvement
                },
                "vs_recent": {
                    "performance": target_performance - recent_performance,
                    "efficiency": target_efficiency - recent_efficiency,
                    "improvement": target_improvement - recent_improvement
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating benchmark comparisons: {e}")
            return {"vs_average": {}, "vs_best": {}, "vs_recent": {}}

    async def _classify_performance_tier(
        self,
        target_features: Dict[str, float],
        historical_features: Dict[str, Dict[str, float]]
    ) -> str:
        """Classify session into performance tier"""
        try:
            if not historical_features:
                return "average"

            # Calculate composite score
            target_score = (
                target_features.get("final_performance", 0.0) * 0.4 +
                target_features.get("total_improvement", 0.0) * 0.3 +
                target_features.get("avg_efficiency", 0.0) * 0.2 +
                target_features.get("success_rate", 0.0) * 0.1
            )

            # Calculate historical distribution
            historical_scores = []
            for features in historical_features.values():
                score = (
                    features.get("final_performance", 0.0) * 0.4 +
                    features.get("total_improvement", 0.0) * 0.3 +
                    features.get("avg_efficiency", 0.0) * 0.2 +
                    features.get("success_rate", 0.0) * 0.1
                )
                historical_scores.append(score)

            # Determine percentile
            percentile = stats.percentileofscore(historical_scores, target_score)

            # Classify into tier
            if percentile >= 90:
                return "excellent"
            elif percentile >= 75:
                return "good"
            elif percentile >= 25:
                return "average"
            elif percentile >= 10:
                return "below_average"
            else:
                return "poor"

        except Exception as e:
            self.logger.error(f"Error classifying performance tier: {e}")
            return "average"

    async def _calculate_improvement_potential(
        self,
        target_features: Dict[str, float],
        historical_features: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate improvement potential for target session"""
        try:
            if not historical_features:
                return 0.5

            # Find best historical performance
            best_performance = max([f.get("final_performance", 0.0) for f in historical_features.values()])
            target_performance = target_features.get("final_performance", 0.0)

            # Calculate potential improvement
            if best_performance > target_performance:
                potential = (best_performance - target_performance) / best_performance
                return min(1.0, potential)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating improvement potential: {e}")
            return 0.5

    async def _identify_strengths_weaknesses(
        self,
        target_features: Dict[str, float],
        historical_features: Dict[str, Dict[str, float]]
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses of target session"""
        try:
            strengths = []
            weaknesses = []

            if not historical_features:
                return strengths, weaknesses

            # Calculate percentiles for each feature
            features_to_check = ["final_performance", "total_improvement", "avg_efficiency", "success_rate", "performance_stability"]

            for feature in features_to_check:
                historical_values = [f.get(feature, 0.0) for f in historical_features.values()]
                target_value = target_features.get(feature, 0.0)

                if historical_values:
                    percentile = stats.percentileofscore(historical_values, target_value)

                    if percentile >= 80:
                        strengths.append(f"High {feature.replace('_', ' ')}: {percentile:.0f}th percentile")
                    elif percentile <= 20:
                        weaknesses.append(f"Low {feature.replace('_', ' ')}: {percentile:.0f}th percentile")

            return strengths, weaknesses

        except Exception as e:
            self.logger.error(f"Error identifying strengths and weaknesses: {e}")
            return [], []

    async def _generate_benchmark_recommendations(
        self,
        target_features: Dict[str, float],
        historical_features: Dict[str, Dict[str, float]],
        performance_tier: str
    ) -> List[str]:
        """Generate benchmark-based recommendations"""
        try:
            recommendations = []

            if performance_tier == "excellent":
                recommendations.append("Maintain current approach - performance is in top 10%")
                recommendations.append("Consider sharing best practices with other sessions")
            elif performance_tier == "good":
                recommendations.append("Good performance - minor optimizations could reach excellence")
            elif performance_tier == "average":
                recommendations.append("Analyze high-performing sessions for improvement opportunities")
                recommendations.append("Focus on efficiency and consistency improvements")
            elif performance_tier == "below_average":
                recommendations.append("Significant improvement needed - review training configuration")
                recommendations.append("Consider adopting proven strategies from successful sessions")
            else:  # poor
                recommendations.append("Major performance issues detected - comprehensive review required")
                recommendations.append("Implement fundamental changes to training approach")

            # Specific recommendations based on features
            if not historical_features:
                return recommendations

            # Success rate recommendations
            target_success_rate = target_features.get("success_rate", 0.0)
            avg_success_rate = np.mean([f.get("success_rate", 0.0) for f in historical_features.values()])

            if target_success_rate < avg_success_rate * 0.8:
                recommendations.append("Investigate and address iteration failure causes")

            # Efficiency recommendations
            target_efficiency = target_features.get("avg_efficiency", 0.0)
            avg_efficiency = np.mean([f.get("avg_efficiency", 0.0) for f in historical_features.values()])

            if target_efficiency < avg_efficiency * 0.8:
                recommendations.append("Optimize resource utilization and training speed")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating benchmark recommendations: {e}")
            return []
