"""Statistical analysis engine for performance baselines and trend detection."""

import asyncio
import logging
import statistics
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import math

# Scientific computing
try:
    import numpy as np
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback implementations
    class MockNumpy:
        @staticmethod
        def array(data): return data
        @staticmethod
        def mean(data): return statistics.mean(data) if data else 0
        @staticmethod
        def std(data): return statistics.stdev(data) if len(data) > 1 else 0
        @staticmethod
        def percentile(data, p): return StatisticalAnalyzer._percentile_fallback(data, p)
    np = MockNumpy()

from .models import (
    BaselineMetrics, PerformanceTrend, TrendDirection, MetricDefinition,
    get_metric_definition, BaselineComparison
)

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
        anomaly_threshold: float = 2.0,  # Standard deviations
        enable_advanced_analysis: bool = True
    ):
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
        
        logger.info(f"StatisticalAnalyzer initialized (advanced={'enabled' if self.enable_advanced_analysis else 'disabled'})")

    async def analyze_trend(
        self,
        metric_name: str,
        baseline_data: List[BaselineMetrics],
        timeframe_hours: int = 24
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
                timeframe_start=datetime.now(timezone.utc) - timedelta(hours=timeframe_hours),
                timeframe_end=datetime.now(timezone.utc),
                sample_count=len(baseline_data),
                baseline_mean=0.0,
                current_mean=0.0,
                variance_ratio=0.0
            )
        
        # Filter data to timeframe
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=timeframe_hours)
        filtered_data = [
            b for b in baseline_data 
            if b.collection_timestamp >= cutoff_time
        ]
        
        if len(filtered_data) < self.trend_minimum_samples:
            filtered_data = baseline_data[-self.trend_minimum_samples:]
        
        # Extract metric values with timestamps
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
                timeframe_end=datetime.now(timezone.utc),
                sample_count=len(metric_values),
                baseline_mean=0.0,
                current_mean=0.0,
                variance_ratio=0.0
            )
        
        # Perform trend analysis
        return await self._analyze_trend_data(
            metric_name, metric_values, timestamps, cutoff_time
        )

    async def _analyze_trend_data(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime],
        timeframe_start: datetime
    ) -> PerformanceTrend:
        """Analyze trend from metric values and timestamps."""
        
        # Basic statistics
        mean_value = statistics.mean(values)
        median_value = statistics.median(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Split into baseline (first half) and current (second half) periods
        mid_point = len(values) // 2
        baseline_values = values[:mid_point] if mid_point > 0 else values[:1]
        current_values = values[mid_point:] if mid_point < len(values) else values[-1:]
        
        baseline_mean = statistics.mean(baseline_values)
        current_mean = statistics.mean(current_values)
        
        # Calculate trend magnitude (percentage change)
        if baseline_mean != 0:
            magnitude = ((current_mean - baseline_mean) / baseline_mean) * 100
        else:
            magnitude = 0.0
        
        # Determine trend direction
        direction = self._determine_trend_direction(magnitude, values)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(values, baseline_values, current_values)
        
        # Statistical significance testing
        p_value = None
        if self.enable_advanced_analysis and len(baseline_values) > 1 and len(current_values) > 1:
            try:
                # Perform t-test to determine if difference is significant
                t_stat, p_value = stats.ttest_ind(baseline_values, current_values)
            except Exception as e:
                logger.debug(f"T-test failed for {metric_name}: {e}")
        
        # Variance analysis
        baseline_var = statistics.variance(baseline_values) if len(baseline_values) > 1 else 0
        current_var = statistics.variance(current_values) if len(current_values) > 1 else 0
        variance_ratio = current_var / baseline_var if baseline_var > 0 else 0
        
        # Predictions
        predicted_24h, predicted_7d = self._predict_future_values(
            values, timestamps, baseline_mean, current_mean, magnitude
        )

        return PerformanceTrend(
            metric_name=metric_name,
            direction=direction,
            magnitude=magnitude,
            confidence_score=confidence_score,
            timeframe_start=timeframe_start,
            timeframe_end=datetime.now(timezone.utc),
            sample_count=len(values),
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            variance_ratio=variance_ratio,
            p_value=p_value,
            predicted_value_24h=predicted_24h,
            predicted_value_7d=predicted_7d
        )

    def _extract_metric_values(self, baseline: BaselineMetrics, metric_name: str) -> List[float]:
        """Extract values for a specific metric from baseline data."""
        # Check standard metric lists first
        if metric_name == 'response_time' and baseline.response_times:
            return baseline.response_times
        elif metric_name == 'error_rate' and baseline.error_rates:
            return baseline.error_rates
        elif metric_name == 'throughput' and baseline.throughput_values:
            return baseline.throughput_values
        elif metric_name == 'cpu_utilization' and baseline.cpu_utilization:
            return baseline.cpu_utilization
        elif metric_name == 'memory_utilization' and baseline.memory_utilization:
            return baseline.memory_utilization
        elif metric_name == 'database_connection_time' and baseline.database_connection_time:
            return baseline.database_connection_time
        elif metric_name == 'cache_hit_rate' and baseline.cache_hit_rate:
            return baseline.cache_hit_rate
        
        # Check custom metrics
        if metric_name in baseline.custom_metrics:
            return baseline.custom_metrics[metric_name]
        
        # Calculate derived metrics
        if metric_name == 'response_time_p95' and baseline.response_times:
            return [self._percentile(baseline.response_times, 95)]
        elif metric_name == 'response_time_p99' and baseline.response_times:
            return [self._percentile(baseline.response_times, 99)]
        
        return []

    def _determine_trend_direction(self, magnitude: float, values: List[float]) -> TrendDirection:
        """Determine trend direction based on magnitude and data characteristics."""
        # Check for volatility (high coefficient of variation)
        if len(values) > 1:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            cv = std_val / mean_val if mean_val > 0 else 0
            
            if cv > 0.5:  # High volatility
                return TrendDirection.VOLATILE
        
        # Determine direction based on magnitude
        abs_magnitude = abs(magnitude)
        
        if abs_magnitude < 5:  # Less than 5% change
            return TrendDirection.STABLE
        elif magnitude > 0:
            return TrendDirection.DEGRADING if abs_magnitude > 15 else TrendDirection.DEGRADING
        else:
            return TrendDirection.IMPROVING
    
    def _calculate_confidence_score(
        self, 
        all_values: List[float], 
        baseline_values: List[float], 
        current_values: List[float]
    ) -> float:
        """Calculate confidence score for trend analysis (0-1 scale)."""
        factors = []
        
        # Sample size factor
        sample_factor = min(len(all_values) / (self.trend_minimum_samples * 2), 1.0)
        factors.append(sample_factor)
        
        # Consistency factor (lower coefficient of variation = higher confidence)
        if len(all_values) > 1:
            mean_val = statistics.mean(all_values)
            std_val = statistics.stdev(all_values)
            cv = std_val / mean_val if mean_val > 0 else 1
            consistency_factor = max(0, 1 - cv)
            factors.append(consistency_factor)
        
        # Difference significance factor
        if len(baseline_values) > 1 and len(current_values) > 1:
            baseline_mean = statistics.mean(baseline_values)
            current_mean = statistics.mean(current_values)
            baseline_std = statistics.stdev(baseline_values)
            
            if baseline_std > 0:
                # Effect size (Cohen's d)
                effect_size = abs(current_mean - baseline_mean) / baseline_std
                significance_factor = min(effect_size / 2, 1.0)  # Normalize to 0-1
                factors.append(significance_factor)
        
        # Return average of all factors
        return statistics.mean(factors) if factors else 0.0

    def _predict_future_values(
        self,
        values: List[float],
        timestamps: List[datetime],
        baseline_mean: float,
        current_mean: float,
        magnitude: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Predict future metric values using trend analysis."""
        if len(values) < 3:
            return None, None
        
        try:
            # Simple linear extrapolation based on recent trend
            trend_rate = magnitude / 100  # Convert percentage to ratio
            
            # 24-hour prediction
            predicted_24h = current_mean * (1 + trend_rate * 0.1)  # Damped projection
            
            # 7-day prediction  
            predicted_7d = current_mean * (1 + trend_rate * 0.5)  # More damped for longer term
            
            return predicted_24h, predicted_7d
            
        except Exception as e:
            logger.debug(f"Prediction failed: {e}")
            return None, None

    async def detect_anomalies(
        self,
        baseline: BaselineMetrics,
        historical_baselines: List[BaselineMetrics],
        metric_name: str
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in current baseline compared to historical data.
        
        Args:
            baseline: Current baseline to check for anomalies
            historical_baselines: Historical baselines for comparison
            metric_name: Specific metric to analyze
            
        Returns:
            List of detected anomalies with details
        """
        anomalies = []
        
        # Get current metric values
        current_values = self._extract_metric_values(baseline, metric_name)
        if not current_values:
            return anomalies
        
        # Collect historical values
        historical_values = []
        for hist_baseline in historical_baselines:
            hist_values = self._extract_metric_values(hist_baseline, metric_name)
            historical_values.extend(hist_values)
        
        if len(historical_values) < self.trend_minimum_samples:
            return anomalies
        
        # Calculate historical statistics
        hist_mean = statistics.mean(historical_values)
        hist_std = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
        
        if hist_std == 0:
            return anomalies
        
        # Check each current value for anomalies
        for i, value in enumerate(current_values):
            z_score = abs(value - hist_mean) / hist_std
            
            if z_score > self.anomaly_threshold:
                anomaly = {
                    'metric_name': metric_name,
                    'value': value,
                    'historical_mean': hist_mean,
                    'historical_std': hist_std,
                    'z_score': z_score,
                    'severity': 'critical' if z_score > 3 else 'warning',
                    'timestamp': baseline.collection_timestamp,
                    'baseline_id': baseline.baseline_id
                }
                anomalies.append(anomaly)

        return anomalies

    async def compare_baselines(
        self,
        baseline_a: BaselineMetrics,
        baseline_b: BaselineMetrics,
        metrics_to_compare: Optional[List[str]] = None
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
            comparison_timestamp=datetime.now(timezone.utc),
            overall_improvement=False
        )

        # Determine metrics to compare
        if metrics_to_compare is None:
            metrics_to_compare = self._get_common_metrics(baseline_a, baseline_b)

        for metric_name in metrics_to_compare:
            values_a = self._extract_metric_values(baseline_a, metric_name)
            values_b = self._extract_metric_values(baseline_b, metric_name)

            if not values_a or not values_b:
                continue

            # Calculate means for comparison
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)

            # Calculate change percentage
            if mean_a != 0:
                change_percentage = ((mean_b - mean_a) / mean_a) * 100
            else:
                change_percentage = 0.0

            # Determine if change is statistically significant
            is_significant = False
            if self.enable_advanced_analysis and len(values_a) > 1 and len(values_b) > 1:
                try:
                    _, p_value = stats.ttest_ind(values_a, values_b)
                    is_significant = p_value < self.significance_threshold
                except Exception:
                    pass
            else:
                # Simple threshold-based significance
                is_significant = abs(change_percentage) > 10  # 10% threshold

            # Determine if lower is better for this metric
            metric_def = get_metric_definition(metric_name)
            lower_is_better = metric_def.lower_is_better if metric_def else True

            # Add to comparison
            comparison.add_metric_comparison(
                metric_name=metric_name,
                baseline_value=mean_a,
                current_value=mean_b,
                change_percentage=change_percentage,
                is_significant=is_significant,
                lower_is_better=lower_is_better
            )

        return comparison

    def _get_common_metrics(self, baseline_a: BaselineMetrics, baseline_b: BaselineMetrics) -> List[str]:
        """Get list of metrics present in both baselines."""
        metrics_a = set()
        metrics_b = set()

        # Standard metrics
        standard_mapping = {
            'response_times': 'response_time',
            'error_rates': 'error_rate',
            'throughput_values': 'throughput',
            'cpu_utilization': 'cpu_utilization',
            'memory_utilization': 'memory_utilization',
            'database_connection_time': 'database_connection_time',
            'cache_hit_rate': 'cache_hit_rate'
        }

        for attr, metric_name in standard_mapping.items():
            if getattr(baseline_a, attr):
                metrics_a.add(metric_name)
            if getattr(baseline_b, attr):
                metrics_b.add(metric_name)

        # Custom metrics
        metrics_a.update(baseline_a.custom_metrics.keys())
        metrics_b.update(baseline_b.custom_metrics.keys())

        return list(metrics_a.intersection(metrics_b))

    async def calculate_performance_score(
        self,
        baseline: BaselineMetrics,
        weight_config: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate overall performance score for a baseline.

        Args:
            baseline: Baseline to score
            weight_config: Custom weights for different metrics

        Returns:
            Performance score and breakdown
        """
        # Default weights for different metric types
        default_weights = {
            'response_time': 0.3,
            'error_rate': 0.2,
            'throughput': 0.2,
            'cpu_utilization': 0.1,
            'memory_utilization': 0.1,
            'database_connection_time': 0.1
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
                'score': metric_score,
                'weight': weight,
                'weighted_score': metric_score * weight,
                'value_count': len(values),
                'mean_value': statistics.mean(values)
            }

            weighted_total += metric_score * weight
            total_weight += weight

        overall_score = weighted_total / total_weight if total_weight > 0 else 0.0

        return {
            'overall_score': overall_score,
            'grade': self._score_to_grade(overall_score),
            'metric_scores': scores,
            'baseline_id': baseline.baseline_id,
            'collection_timestamp': baseline.collection_timestamp.isoformat()
        }

    def _calculate_metric_score(self, metric_name: str, values: List[float]) -> float:
        """Calculate score (0-100) for a specific metric."""
        if not values:
            return 0.0

        metric_def = get_metric_definition(metric_name)
        mean_value = statistics.mean(values)

        if metric_def and metric_def.target_value:
            # Score based on target value
            target = metric_def.target_value

            if metric_def.lower_is_better:
                # Lower is better (latency, error rate, resource usage)
                if mean_value <= target:
                    return 100.0  # Perfect score
                elif metric_def.critical_threshold and mean_value >= metric_def.critical_threshold:
                    return 0.0    # Worst score
                else:
                    # Linear interpolation between target and critical
                    critical = metric_def.critical_threshold or target * 2
                    return max(0, 100 * (critical - mean_value) / (critical - target))
            else:
                # Higher is better (throughput, availability)
                if mean_value >= target:
                    return 100.0  # Perfect score
                elif metric_def.critical_threshold and mean_value <= metric_def.critical_threshold:
                    return 0.0    # Worst score
                else:
                    # Linear interpolation between critical and target
                    critical = metric_def.critical_threshold or target * 0.5
                    return max(0, 100 * (mean_value - critical) / (target - critical))

        # Default scoring for metrics without defined targets
        return 50.0  # Neutral score

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile of values (fallback implementation)."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    @staticmethod
    def _percentile_fallback(values: List[float], percentile: float) -> float:
        """Fallback percentile calculation for MockNumpy."""
        return StatisticalAnalyzer._percentile(values, percentile)

# Utility functions for common statistical operations\n\nasync def analyze_metric_trend(\n    metric_name: str,\n    baselines: List[BaselineMetrics],\n    timeframe_hours: int = 24\n) -> PerformanceTrend:\n    \"\"\"Convenience function to analyze trend for a specific metric.\"\"\"\n    analyzer = StatisticalAnalyzer()\n    return await analyzer.analyze_trend(metric_name, baselines, timeframe_hours)\n\nasync def detect_performance_anomalies(\n    current_baseline: BaselineMetrics,\n    historical_baselines: List[BaselineMetrics],\n    metrics: Optional[List[str]] = None\n) -> Dict[str, List[Dict[str, Any]]]:\n    \"\"\"Detect anomalies across multiple metrics.\"\"\"\n    analyzer = StatisticalAnalyzer()\n    \n    if metrics is None:\n        metrics = ['response_time', 'error_rate', 'throughput', 'cpu_utilization', 'memory_utilization']\n    \n    all_anomalies = {}\n    \n    for metric_name in metrics:\n        anomalies = await analyzer.detect_anomalies(\n            current_baseline, historical_baselines, metric_name\n        )\n        if anomalies:\n            all_anomalies[metric_name] = anomalies\n    \n    return all_anomalies\n\nasync def calculate_baseline_score(\n    baseline: BaselineMetrics,\n    custom_weights: Optional[Dict[str, float]] = None\n) -> Dict[str, Any]:\n    \"\"\"Calculate performance score for a baseline.\"\"\"\n    analyzer = StatisticalAnalyzer()\n    return await analyzer.calculate_performance_score(baseline, custom_weights)