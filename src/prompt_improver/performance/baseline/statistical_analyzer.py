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
        predicted_24h, predicted_7d = self._predict_future_values(\n            values, timestamps, baseline_mean, current_mean, magnitude\n        )\n        \n        return PerformanceTrend(\n            metric_name=metric_name,\n            direction=direction,\n            magnitude=magnitude,\n            confidence_score=confidence_score,\n            timeframe_start=timeframe_start,\n            timeframe_end=datetime.now(timezone.utc),\n            sample_count=len(values),\n            baseline_mean=baseline_mean,\n            current_mean=current_mean,\n            variance_ratio=variance_ratio,\n            p_value=p_value,\n            predicted_value_24h=predicted_24h,\n            predicted_value_7d=predicted_7d\n        )

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
                }\n                anomalies.append(anomaly)\n        \n        return anomalies\n\n    async def compare_baselines(\n        self,\n        baseline_a: BaselineMetrics,\n        baseline_b: BaselineMetrics,\n        metrics_to_compare: Optional[List[str]] = None\n    ) -> BaselineComparison:\n        \"\"\"Compare two baselines and analyze performance differences.\n        \n        Args:\n            baseline_a: First baseline (typically older/reference)\n            baseline_b: Second baseline (typically newer/current)\n            metrics_to_compare: Specific metrics to compare (if None, compare all)\n            \n        Returns:\n            Detailed comparison results\n        \"\"\"\n        comparison = BaselineComparison(\n            baseline_a_id=baseline_a.baseline_id,\n            baseline_b_id=baseline_b.baseline_id,\n            comparison_timestamp=datetime.now(timezone.utc),\n            overall_improvement=False\n        )\n        \n        # Determine metrics to compare\n        if metrics_to_compare is None:\n            metrics_to_compare = self._get_common_metrics(baseline_a, baseline_b)\n        \n        for metric_name in metrics_to_compare:\n            values_a = self._extract_metric_values(baseline_a, metric_name)\n            values_b = self._extract_metric_values(baseline_b, metric_name)\n            \n            if not values_a or not values_b:\n                continue\n            \n            # Calculate means for comparison\n            mean_a = statistics.mean(values_a)\n            mean_b = statistics.mean(values_b)\n            \n            # Calculate change percentage\n            if mean_a != 0:\n                change_percentage = ((mean_b - mean_a) / mean_a) * 100\n            else:\n                change_percentage = 0.0\n            \n            # Determine if change is statistically significant\n            is_significant = False\n            if self.enable_advanced_analysis and len(values_a) > 1 and len(values_b) > 1:\n                try:\n                    _, p_value = stats.ttest_ind(values_a, values_b)\n                    is_significant = p_value < self.significance_threshold\n                except Exception:\n                    pass\n            else:\n                # Simple threshold-based significance\n                is_significant = abs(change_percentage) > 10  # 10% threshold\n            \n            # Determine if lower is better for this metric\n            metric_def = get_metric_definition(metric_name)\n            lower_is_better = metric_def.lower_is_better if metric_def else True\n            \n            # Add to comparison\n            comparison.add_metric_comparison(\n                metric_name=metric_name,\n                baseline_value=mean_a,\n                current_value=mean_b,\n                change_percentage=change_percentage,\n                is_significant=is_significant,\n                lower_is_better=lower_is_better\n            )\n        \n        return comparison

    def _get_common_metrics(self, baseline_a: BaselineMetrics, baseline_b: BaselineMetrics) -> List[str]:
        """Get list of metrics present in both baselines."""
        metrics_a = set()\n        metrics_b = set()\n        \n        # Standard metrics\n        standard_mapping = {\n            'response_times': 'response_time',\n            'error_rates': 'error_rate',\n            'throughput_values': 'throughput',\n            'cpu_utilization': 'cpu_utilization',\n            'memory_utilization': 'memory_utilization',\n            'database_connection_time': 'database_connection_time',\n            'cache_hit_rate': 'cache_hit_rate'\n        }\n        \n        for attr, metric_name in standard_mapping.items():\n            if getattr(baseline_a, attr):\n                metrics_a.add(metric_name)\n            if getattr(baseline_b, attr):\n                metrics_b.add(metric_name)\n        \n        # Custom metrics\n        metrics_a.update(baseline_a.custom_metrics.keys())\n        metrics_b.update(baseline_b.custom_metrics.keys())\n        \n        return list(metrics_a.intersection(metrics_b))

    async def calculate_performance_score(\n        self, \n        baseline: BaselineMetrics,\n        weight_config: Optional[Dict[str, float]] = None\n    ) -> Dict[str, Any]:\n        \"\"\"Calculate overall performance score for a baseline.\n        \n        Args:\n            baseline: Baseline to score\n            weight_config: Custom weights for different metrics\n            \n        Returns:\n            Performance score and breakdown\n        \"\"\"\n        # Default weights for different metric types\n        default_weights = {\n            'response_time': 0.3,\n            'error_rate': 0.2,\n            'throughput': 0.2,\n            'cpu_utilization': 0.1,\n            'memory_utilization': 0.1,\n            'database_connection_time': 0.1\n        }\n        \n        weights = weight_config or default_weights\n        \n        scores = {}\n        weighted_total = 0.0\n        total_weight = 0.0\n        \n        for metric_name, weight in weights.items():\n            values = self._extract_metric_values(baseline, metric_name)\n            if not values:\n                continue\n            \n            metric_score = self._calculate_metric_score(metric_name, values)\n            scores[metric_name] = {\n                'score': metric_score,\n                'weight': weight,\n                'weighted_score': metric_score * weight,\n                'value_count': len(values),\n                'mean_value': statistics.mean(values)\n            }\n            \n            weighted_total += metric_score * weight\n            total_weight += weight\n        \n        overall_score = weighted_total / total_weight if total_weight > 0 else 0.0\n        \n        return {\n            'overall_score': overall_score,\n            'grade': self._score_to_grade(overall_score),\n            'metric_scores': scores,\n            'baseline_id': baseline.baseline_id,\n            'collection_timestamp': baseline.collection_timestamp.isoformat()\n        }

    def _calculate_metric_score(self, metric_name: str, values: List[float]) -> float:\n        \"\"\"Calculate score (0-100) for a specific metric.\"\"\"\n        if not values:\n            return 0.0\n        \n        metric_def = get_metric_definition(metric_name)\n        mean_value = statistics.mean(values)\n        \n        if metric_def and metric_def.target_value:\n            # Score based on target value\n            target = metric_def.target_value\n            \n            if metric_def.lower_is_better:\n                # Lower is better (latency, error rate, resource usage)\n                if mean_value <= target:\n                    return 100.0  # Perfect score\n                elif metric_def.critical_threshold and mean_value >= metric_def.critical_threshold:\n                    return 0.0    # Worst score\n                else:\n                    # Linear interpolation between target and critical\n                    critical = metric_def.critical_threshold or target * 2\n                    return max(0, 100 * (critical - mean_value) / (critical - target))\n            else:\n                # Higher is better (throughput, availability)\n                if mean_value >= target:\n                    return 100.0  # Perfect score\n                elif metric_def.critical_threshold and mean_value <= metric_def.critical_threshold:\n                    return 0.0    # Worst score\n                else:\n                    # Linear interpolation between critical and target\n                    critical = metric_def.critical_threshold or target * 0.5\n                    return max(0, 100 * (mean_value - critical) / (target - critical))\n        \n        # Default scoring for metrics without defined targets\n        return 50.0  # Neutral score\n\n    def _score_to_grade(self, score: float) -> str:\n        \"\"\"Convert numerical score to letter grade.\"\"\"\n        if score >= 90:\n            return 'A'\n        elif score >= 80:\n            return 'B'\n        elif score >= 70:\n            return 'C'\n        elif score >= 60:\n            return 'D'\n        else:\n            return 'F'\n\n    @staticmethod\n    def _percentile(values: List[float], percentile: float) -> float:\n        \"\"\"Calculate percentile of values (fallback implementation).\"\"\"\n        if not values:\n            return 0.0\n        \n        sorted_values = sorted(values)\n        index = (percentile / 100) * (len(sorted_values) - 1)\n        \n        if index.is_integer():\n            return sorted_values[int(index)]\n        else:\n            lower = sorted_values[int(index)]\n            upper = sorted_values[int(index) + 1]\n            return lower + (upper - lower) * (index - int(index))\n\n    @staticmethod\n    def _percentile_fallback(values: List[float], percentile: float) -> float:\n        \"\"\"Fallback percentile calculation for MockNumpy.\"\"\"\n        return StatisticalAnalyzer._percentile(values, percentile)

# Utility functions for common statistical operations\n\nasync def analyze_metric_trend(\n    metric_name: str,\n    baselines: List[BaselineMetrics],\n    timeframe_hours: int = 24\n) -> PerformanceTrend:\n    \"\"\"Convenience function to analyze trend for a specific metric.\"\"\"\n    analyzer = StatisticalAnalyzer()\n    return await analyzer.analyze_trend(metric_name, baselines, timeframe_hours)\n\nasync def detect_performance_anomalies(\n    current_baseline: BaselineMetrics,\n    historical_baselines: List[BaselineMetrics],\n    metrics: Optional[List[str]] = None\n) -> Dict[str, List[Dict[str, Any]]]:\n    \"\"\"Detect anomalies across multiple metrics.\"\"\"\n    analyzer = StatisticalAnalyzer()\n    \n    if metrics is None:\n        metrics = ['response_time', 'error_rate', 'throughput', 'cpu_utilization', 'memory_utilization']\n    \n    all_anomalies = {}\n    \n    for metric_name in metrics:\n        anomalies = await analyzer.detect_anomalies(\n            current_baseline, historical_baselines, metric_name\n        )\n        if anomalies:\n            all_anomalies[metric_name] = anomalies\n    \n    return all_anomalies\n\nasync def calculate_baseline_score(\n    baseline: BaselineMetrics,\n    custom_weights: Optional[Dict[str, float]] = None\n) -> Dict[str, Any]:\n    \"\"\"Calculate performance score for a baseline.\"\"\"\n    analyzer = StatisticalAnalyzer()\n    return await analyzer.calculate_performance_score(baseline, custom_weights)