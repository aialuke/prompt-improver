"""Statistical analysis engine for performance baselines and trend detection."""

import logging
import statistics
from datetime import UTC, datetime, timedelta
from typing import Any

from prompt_improver.core.utils.lazy_ml_loader import get_scipy_stats
from prompt_improver.performance.baseline.models import (
    BaselineComparison,
    BaselineMetrics,
    PerformanceTrend,
    TrendDirection,
    get_metric_definition,
)

try:
    # import numpy as np  # Converted to lazy loading
    # from scipy import stats  # Converted to lazy loading

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

    class MockNumpy:
        @staticmethod
        def array(data):
            return data

        @staticmethod
        def mean(data):
            return statistics.mean(data) if data else 0

        @staticmethod
        def std(data):
            return statistics.stdev(data) if len(data) > 1 else 0

        @staticmethod
        def percentile(data, p):
            return StatisticalAnalyzer._percentile_fallback(data, p)

    np = MockNumpy()
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Advanced statistical analysis for performance baselines.

    Provides trend detection, anomaly identification, performance forecasting,
    and statistical significance testing for performance metrics.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        trend_minimum_samples: int = 10,
        anomaly_threshold: float = 2.0,
        enable_advanced_analysis: bool = True,
    ) -> None:
        """Initialize statistical analyzer.

        Args:
            significance_threshold: P-value threshold for statistical significance
            trend_minimum_samples: Minimum samples required for trend analysis
            anomaly_threshold: Z-score threshold for anomaly detection
            enable_advanced_analysis: Enable advanced statistical methods
        """
        self.significance_threshold = significance_threshold
        self.trend_minimum_samples = trend_minimum_samples
        self.anomaly_threshold = anomaly_threshold
        self.enable_advanced_analysis = enable_advanced_analysis and SCIPY_AVAILABLE
        logger.info(
            f"StatisticalAnalyzer initialized (advanced={'enabled' if self.enable_advanced_analysis else 'disabled'})"
        )

    async def analyze_trend(
        self,
        metric_name: str,
        baseline_data: list[BaselineMetrics],
        timeframe_hours: int = 24,
    ) -> PerformanceTrend:
        """Analyze performance trend for a specific metric.

        Args:
            metric_name: Name of the metric to analyze
            baseline_data: List of baseline measurements
            timeframe_hours: Time window for analysis

        Returns:
            Performance trend analysis results
        """
        if len(baseline_data) < self.trend_minimum_samples:
            return PerformanceTrend(
                metric_name=metric_name,
                direction=TrendDirection.UNKNOWN,
                magnitude=0.0,
                confidence_score=0.0,
                timeframe_start=datetime.now(UTC) - timedelta(hours=timeframe_hours),
                timeframe_end=datetime.now(UTC),
                sample_count=len(baseline_data),
                baseline_mean=0.0,
                current_mean=0.0,
                variance_ratio=0.0,
            )
        cutoff_time = datetime.now(UTC) - timedelta(hours=timeframe_hours)
        filtered_data = [
            b for b in baseline_data if b.collection_timestamp >= cutoff_time
        ]
        if len(filtered_data) < self.trend_minimum_samples:
            filtered_data = baseline_data[-self.trend_minimum_samples :]
        metric_values = []
        timestamps = []
        for baseline in filtered_data:
            values = self._extract_metric_values(baseline, metric_name)
            if values:
                metric_values.extend(values)
                timestamps.extend([baseline.collection_timestamp] * len(values))
        if len(metric_values) < self.trend_minimum_samples:
            return PerformanceTrend(
                metric_name=metric_name,
                direction=TrendDirection.UNKNOWN,
                magnitude=0.0,
                confidence_score=0.0,
                timeframe_start=cutoff_time,
                timeframe_end=datetime.now(UTC),
                sample_count=len(metric_values),
                baseline_mean=0.0,
                current_mean=0.0,
                variance_ratio=0.0,
            )
        return await self._analyze_trend_data(
            metric_name, metric_values, timestamps, cutoff_time
        )

    async def _analyze_trend_data(
        self,
        metric_name: str,
        values: list[float],
        timestamps: list[datetime],
        timeframe_start: datetime,
    ) -> PerformanceTrend:
        """Analyze trend from metric values and timestamps."""
        mean_value = statistics.mean(values)
        median_value = statistics.median(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        mid_point = len(values) // 2
        baseline_values = values[:mid_point] if mid_point > 0 else values[:1]
        current_values = values[mid_point:] if mid_point < len(values) else values[-1:]
        baseline_mean = statistics.mean(baseline_values)
        current_mean = statistics.mean(current_values)
        if baseline_mean != 0:
            magnitude = (current_mean - baseline_mean) / baseline_mean * 100
        else:
            magnitude = 0.0
        direction = self._determine_trend_direction(magnitude, values)
        confidence_score = self._calculate_confidence_score(
            values, baseline_values, current_values
        )
        p_value = None
        if (
            self.enable_advanced_analysis
            and len(baseline_values) > 1
            and (len(current_values) > 1)
        ):
            try:
                _t_stat, p_value = get_scipy_stats().ttest_ind(baseline_values, current_values)
            except Exception as e:
                logger.debug(f"T-test failed for {metric_name}: {e}")
        baseline_var = (
            statistics.variance(baseline_values) if len(baseline_values) > 1 else 0
        )
        current_var = (
            statistics.variance(current_values) if len(current_values) > 1 else 0
        )
        variance_ratio = current_var / baseline_var if baseline_var > 0 else 0
        predicted_24h, predicted_7d = self._predict_future_values(
            values, timestamps, baseline_mean, current_mean, magnitude
        )
        return PerformanceTrend(
            metric_name=metric_name,
            direction=direction,
            magnitude=magnitude,
            confidence_score=confidence_score,
            timeframe_start=timeframe_start,
            timeframe_end=datetime.now(UTC),
            sample_count=len(values),
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            variance_ratio=variance_ratio,
            p_value=p_value,
            predicted_value_24h=predicted_24h,
            predicted_value_7d=predicted_7d,
        )

    def _extract_metric_values(
        self, baseline: BaselineMetrics, metric_name: str
    ) -> list[float]:
        """Extract values for a specific metric from baseline data."""
        if metric_name == "response_time" and baseline.response_times:
            return baseline.response_times
        if metric_name == "error_rate" and baseline.error_rates:
            return baseline.error_rates
        if metric_name == "throughput" and baseline.throughput_values:
            return baseline.throughput_values
        if metric_name == "cpu_utilization" and baseline.cpu_utilization:
            return baseline.cpu_utilization
        if metric_name == "memory_utilization" and baseline.memory_utilization:
            return baseline.memory_utilization
        if (
            metric_name == "database_connection_time"
            and baseline.database_connection_time
        ):
            return baseline.database_connection_time
        if metric_name == "cache_hit_rate" and baseline.cache_hit_rate:
            return baseline.cache_hit_rate
        if metric_name in baseline.custom_metrics:
            return baseline.custom_metrics[metric_name]
        if metric_name == "response_time_p95" and baseline.response_times:
            return [self._percentile(baseline.response_times, 95)]
        if metric_name == "response_time_p99" and baseline.response_times:
            return [self._percentile(baseline.response_times, 99)]
        return []

    def _determine_trend_direction(
        self, magnitude: float, values: list[float]
    ) -> TrendDirection:
        """Determine trend direction based on magnitude and data characteristics."""
        if len(values) > 1:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            cv = std_val / mean_val if mean_val > 0 else 0
            if cv > 0.5:
                return TrendDirection.VOLATILE
        abs_magnitude = abs(magnitude)
        if abs_magnitude < 5:
            return TrendDirection.STABLE
        if magnitude > 0:
            return (
                TrendDirection.DEGRADING
                if abs_magnitude > 15
                else TrendDirection.DEGRADING
            )
        return TrendDirection.IMPROVING

    def _calculate_confidence_score(
        self,
        all_values: list[float],
        baseline_values: list[float],
        current_values: list[float],
    ) -> float:
        """Calculate confidence score for trend analysis (0-1 scale)."""
        factors = []
        sample_factor = min(len(all_values) / (self.trend_minimum_samples * 2), 1.0)
        factors.append(sample_factor)
        if len(all_values) > 1:
            mean_val = statistics.mean(all_values)
            std_val = statistics.stdev(all_values)
            cv = std_val / mean_val if mean_val > 0 else 1
            consistency_factor = max(0, 1 - cv)
            factors.append(consistency_factor)
        if len(baseline_values) > 1 and len(current_values) > 1:
            baseline_mean = statistics.mean(baseline_values)
            current_mean = statistics.mean(current_values)
            baseline_std = statistics.stdev(baseline_values)
            if baseline_std > 0:
                effect_size = abs(current_mean - baseline_mean) / baseline_std
                significance_factor = min(effect_size / 2, 1.0)
                factors.append(significance_factor)
        return statistics.mean(factors) if factors else 0.0

    def _predict_future_values(
        self,
        values: list[float],
        timestamps: list[datetime],
        baseline_mean: float,
        current_mean: float,
        magnitude: float,
    ) -> tuple[float | None, float | None]:
        """Predict future metric values using trend analysis."""
        if len(values) < 3:
            return (None, None)
        try:
            trend_rate = magnitude / 100
            predicted_24h = current_mean * (1 + trend_rate * 0.1)
            predicted_7d = current_mean * (1 + trend_rate * 0.5)
            return (predicted_24h, predicted_7d)
        except Exception as e:
            logger.debug(f"Prediction failed: {e}")
            return (None, None)

    async def detect_anomalies(
        self,
        baseline: BaselineMetrics,
        historical_baselines: list[BaselineMetrics],
        metric_name: str,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in current baseline compared to historical data.

        Args:
            baseline: Current baseline to check for anomalies
            historical_baselines: Historical baselines for comparison
            metric_name: Specific metric to analyze

        Returns:
            List of detected anomalies with details
        """
        anomalies = []
        current_values = self._extract_metric_values(baseline, metric_name)
        if not current_values:
            return anomalies
        historical_values = []
        for hist_baseline in historical_baselines:
            hist_values = self._extract_metric_values(hist_baseline, metric_name)
            historical_values.extend(hist_values)
        if len(historical_values) < self.trend_minimum_samples:
            return anomalies
        hist_mean = statistics.mean(historical_values)
        hist_std = (
            statistics.stdev(historical_values) if len(historical_values) > 1 else 0
        )
        if hist_std == 0:
            return anomalies
        for _i, value in enumerate(current_values):
            z_score = abs(value - hist_mean) / hist_std
            if z_score > self.anomaly_threshold:
                anomaly = {
                    "metric_name": metric_name,
                    "value": value,
                    "historical_mean": hist_mean,
                    "historical_std": hist_std,
                    "z_score": z_score,
                    "severity": "critical" if z_score > 3 else "warning",
                    "timestamp": baseline.collection_timestamp,
                    "baseline_id": baseline.baseline_id,
                }
                anomalies.append(anomaly)
        return anomalies

    async def compare_baselines(
        self,
        baseline_a: BaselineMetrics,
        baseline_b: BaselineMetrics,
        metrics_to_compare: list[str] | None = None,
    ) -> BaselineComparison:
        """Compare two baselines and analyze performance differences.

        Args:
            baseline_a: First baseline (typically older/reference)
            baseline_b: Second baseline (typically newer/current)
            metrics_to_compare: Specific metrics to compare (if None, compare all)

        Returns:
            Detailed comparison results
        """
        comparison = BaselineComparison(
            baseline_a_id=baseline_a.baseline_id,
            baseline_b_id=baseline_b.baseline_id,
            comparison_timestamp=datetime.now(UTC),
            overall_improvement=False,
        )
        if metrics_to_compare is None:
            metrics_to_compare = self._get_common_metrics(baseline_a, baseline_b)
        for metric_name in metrics_to_compare:
            values_a = self._extract_metric_values(baseline_a, metric_name)
            values_b = self._extract_metric_values(baseline_b, metric_name)
            if not values_a or not values_b:
                continue
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            change_percentage = (mean_b - mean_a) / mean_a * 100 if mean_a != 0 else 0.0
            is_significant = False
            if (
                self.enable_advanced_analysis
                and len(values_a) > 1
                and (len(values_b) > 1)
            ):
                try:
                    _, p_value = get_scipy_stats().ttest_ind(values_a, values_b)
                    is_significant = p_value < self.significance_threshold
                except Exception:
                    pass
            else:
                is_significant = abs(change_percentage) > 10
            metric_def = get_metric_definition(metric_name)
            lower_is_better = metric_def.lower_is_better if metric_def else True
            comparison.add_metric_comparison(
                metric_name=metric_name,
                baseline_value=mean_a,
                current_value=mean_b,
                change_percentage=change_percentage,
                is_significant=is_significant,
                lower_is_better=lower_is_better,
            )
        return comparison

    def _get_common_metrics(
        self, baseline_a: BaselineMetrics, baseline_b: BaselineMetrics
    ) -> list[str]:
        """Get list of metrics present in both baselines."""
        metrics_a = set()
        metrics_b = set()
        standard_mapping = {
            "response_times": "response_time",
            "error_rates": "error_rate",
            "throughput_values": "throughput",
            "cpu_utilization": "cpu_utilization",
            "memory_utilization": "memory_utilization",
            "database_connection_time": "database_connection_time",
            "cache_hit_rate": "cache_hit_rate",
        }
        for attr, metric_name in standard_mapping.items():
            if getattr(baseline_a, attr):
                metrics_a.add(metric_name)
            if getattr(baseline_b, attr):
                metrics_b.add(metric_name)
        metrics_a.update(baseline_a.custom_metrics.keys())
        metrics_b.update(baseline_b.custom_metrics.keys())
        return list(metrics_a.intersection(metrics_b))

    async def calculate_performance_score(
        self, baseline: BaselineMetrics, weight_config: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """Calculate overall performance score for a baseline.

        Args:
            baseline: Baseline to score
            weight_config: Custom weights for different metrics

        Returns:
            Performance score and breakdown
        """
        default_weights = {
            "response_time": 0.3,
            "error_rate": 0.2,
            "throughput": 0.2,
            "cpu_utilization": 0.1,
            "memory_utilization": 0.1,
            "database_connection_time": 0.1,
        }
        weights = weight_config or default_weights
        scores = {}
        weighted_total = 0.0
        total_weight = 0.0
        for metric_name, weight in weights.items():
            values = self._extract_metric_values(baseline, metric_name)
            if not values:
                continue
            metric_score = self._calculate_metric_score(metric_name, values)
            scores[metric_name] = {
                "score": metric_score,
                "weight": weight,
                "weighted_score": metric_score * weight,
                "value_count": len(values),
                "mean_value": statistics.mean(values),
            }
            weighted_total += metric_score * weight
            total_weight += weight
        overall_score = weighted_total / total_weight if total_weight > 0 else 0.0
        return {
            "overall_score": overall_score,
            "grade": self._score_to_grade(overall_score),
            "metric_scores": scores,
            "baseline_id": baseline.baseline_id,
            "collection_timestamp": baseline.collection_timestamp.isoformat(),
        }

    def _calculate_metric_score(self, metric_name: str, values: list[float]) -> float:
        """Calculate score (0-100) for a specific metric."""
        if not values:
            return 0.0
        metric_def = get_metric_definition(metric_name)
        mean_value = statistics.mean(values)
        if metric_def and metric_def.target_value:
            target = metric_def.target_value
            if metric_def.lower_is_better:
                if mean_value <= target:
                    return 100.0
                if (
                    metric_def.critical_threshold
                    and mean_value >= metric_def.critical_threshold
                ):
                    return 0.0
                critical = metric_def.critical_threshold or target * 2
                return max(0, 100 * (critical - mean_value) / (critical - target))
            if mean_value >= target:
                return 100.0
            if (
                metric_def.critical_threshold
                and mean_value <= metric_def.critical_threshold
            ):
                return 0.0
            critical = metric_def.critical_threshold or target * 0.5
            return max(0, 100 * (mean_value - critical) / (target - critical))
        return 50.0

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """Calculate percentile of values (fallback implementation)."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = percentile / 100 * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))

    @staticmethod
    def _percentile_fallback(values: list[float], percentile: float) -> float:
        """Fallback percentile calculation for MockNumpy."""
        return StatisticalAnalyzer._percentile(values, percentile)


async def analyze_metric_trend(
    metric_name: str, baselines: list[BaselineMetrics], timeframe_hours: int = 24
) -> PerformanceTrend:
    """Convenience function to analyze trend for a specific metric."""
    analyzer = StatisticalAnalyzer()
    return await analyzer.analyze_trend(metric_name, baselines, timeframe_hours)


async def detect_performance_anomalies(
    current_baseline: BaselineMetrics,
    historical_baselines: list[BaselineMetrics],
    metrics: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Detect anomalies across multiple metrics."""
    analyzer = StatisticalAnalyzer()
    if metrics is None:
        metrics = [
            "response_time",
            "error_rate",
            "throughput",
            "cpu_utilization",
            "memory_utilization",
        ]
    all_anomalies = {}
    for metric_name in metrics:
        anomalies = await analyzer.detect_anomalies(
            current_baseline, historical_baselines, metric_name
        )
        if anomalies:
            all_anomalies[metric_name] = anomalies
    return all_anomalies


async def calculate_baseline_score(
    baseline: BaselineMetrics, custom_weights: dict[str, float] | None = None
) -> dict[str, Any]:
    """Calculate performance score for a baseline."""
    analyzer = StatisticalAnalyzer()
    return await analyzer.calculate_performance_score(baseline, custom_weights)
