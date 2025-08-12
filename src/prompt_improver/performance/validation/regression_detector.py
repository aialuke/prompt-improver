"""Performance Regression Detection System

This module provides automated performance regression detection and monitoring
for validation operations. It tracks performance trends, alerts on degradation,
and integrates with CI/CD pipelines for continuous performance monitoring.

Key Features:
1. Statistical trend analysis with confidence intervals
2. Automated alerting on performance regressions
3. CI/CD integration hooks
4. Historical performance tracking
5. Intelligent baseline adaptation
"""

import asyncio
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""

    LOW = "low"  # 5-15% degradation
    MEDIUM = "medium"  # 15-30% degradation
    HIGH = "high"  # 30-50% degradation
    CRITICAL = "critical"  # >50% degradation


class TrendDirection(Enum):
    """Performance trend directions."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class PerformanceDataPoint:
    """Single performance measurement."""

    timestamp: datetime
    operation_name: str
    latency_us: float
    memory_kb: float
    success_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation_name": self.operation_name,
            "latency_us": self.latency_us,
            "memory_kb": self.memory_kb,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceDataPoint":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            operation_name=data["operation_name"],
            latency_us=data["latency_us"],
            memory_kb=data["memory_kb"],
            success_rate=data["success_rate"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegressionAlert:
    """Performance regression alert."""

    timestamp: datetime
    operation_name: str
    severity: RegressionSeverity
    current_performance_us: float
    baseline_performance_us: float
    degradation_percent: float
    confidence_level: float
    trend_direction: TrendDirection
    affected_metrics: list[str]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation_name": self.operation_name,
            "severity": self.severity.value,
            "current_performance_us": self.current_performance_us,
            "baseline_performance_us": self.baseline_performance_us,
            "degradation_percent": self.degradation_percent,
            "confidence_level": self.confidence_level,
            "trend_direction": self.trend_direction.value,
            "affected_metrics": self.affected_metrics,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceTrend:
    """Performance trend analysis results."""

    operation_name: str
    direction: TrendDirection
    slope: float  # μs per day
    r_squared: float  # Trend strength
    confidence_interval: tuple[float, float]
    data_points_count: int
    analysis_period_days: int
    volatility: float  # Standard deviation of residuals
    projected_performance_7d: float  # Projected performance in 7 days


class PerformanceRegressionDetector:
    """Advanced performance regression detection system."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("performance_data")
        self.data_dir.mkdir(exist_ok=True)

        # Detection thresholds
        self.regression_thresholds = {
            RegressionSeverity.LOW: 0.05,  # 5%
            RegressionSeverity.MEDIUM: 0.15,  # 15%
            RegressionSeverity.HIGH: 0.30,  # 30%
            RegressionSeverity.CRITICAL: 0.50,  # 50%
        }

        # Statistical parameters
        self.confidence_level = 0.95
        self.min_samples_for_trend = 10
        self.baseline_window_days = 30
        self.detection_window_days = 7

        # Performance history
        self.performance_history: dict[str, list[PerformanceDataPoint]] = {}
        self.alerts_history: list[RegressionAlert] = []

    async def initialize(self) -> None:
        """Initialize detector with historical data."""
        await self._load_performance_history()
        await self._load_alerts_history()
        logger.info("Regression detector initialized with historical data")

    async def record_performance_data(self, data_point: PerformanceDataPoint) -> None:
        """Record a new performance measurement."""
        operation_name = data_point.operation_name

        if operation_name not in self.performance_history:
            self.performance_history[operation_name] = []

        self.performance_history[operation_name].append(data_point)

        # Trim old data to keep memory manageable
        cutoff_date = datetime.now(UTC) - timedelta(days=90)
        self.performance_history[operation_name] = [
            dp
            for dp in self.performance_history[operation_name]
            if dp.timestamp >= cutoff_date
        ]

        # Save to disk periodically
        if len(self.performance_history[operation_name]) % 100 == 0:
            await self._save_performance_history()

    async def check_for_regressions(self, operation_name: str) -> list[RegressionAlert]:
        """Check for performance regressions in a specific operation."""
        if operation_name not in self.performance_history:
            logger.warning(f"No performance history available for {operation_name}")
            return []

        data_points = self.performance_history[operation_name]
        if len(data_points) < self.min_samples_for_trend:
            logger.info(
                f"Insufficient data points for regression analysis: {len(data_points)}"
            )
            return []

        alerts = []

        # Analyze latency regression
        latency_alert = await self._detect_latency_regression(data_points)
        if latency_alert:
            alerts.append(latency_alert)

        # Analyze memory regression
        memory_alert = await self._detect_memory_regression(data_points)
        if memory_alert:
            alerts.append(memory_alert)

        # Analyze success rate regression
        success_rate_alert = await self._detect_success_rate_regression(data_points)
        if success_rate_alert:
            alerts.append(success_rate_alert)

        # Save new alerts
        for alert in alerts:
            self.alerts_history.append(alert)
            logger.warning(
                f"Performance regression detected: {alert.operation_name} - {alert.severity.value}"
            )

        if alerts:
            await self._save_alerts_history()

        return alerts

    async def _detect_latency_regression(
        self, data_points: list[PerformanceDataPoint]
    ) -> RegressionAlert | None:
        """Detect latency performance regression."""
        # Extract latency values and timestamps
        recent_data = [
            dp
            for dp in data_points
            if dp.timestamp
            >= datetime.now(UTC) - timedelta(days=self.detection_window_days)
        ]
        baseline_data = [
            dp
            for dp in data_points
            if dp.timestamp
            >= datetime.now(UTC) - timedelta(days=self.baseline_window_days)
        ]

        if len(recent_data) < 5 or len(baseline_data) < 10:
            return None

        recent_latencies = [dp.latency_us for dp in recent_data]
        baseline_latencies = [
            dp.latency_us for dp in baseline_data[: -len(recent_data)]
        ]  # Exclude recent data from baseline

        # Statistical comparison
        recent_mean = statistics.mean(recent_latencies)
        baseline_mean = statistics.mean(baseline_latencies)

        if recent_mean <= baseline_mean:
            return None  # No regression, performance improved or stable

        degradation_percent = (recent_mean - baseline_mean) / baseline_mean

        # Determine severity
        severity = self._determine_severity(degradation_percent)
        if severity is None:
            return None  # Below threshold

        # Statistical significance test (t-test)
        try:
            t_stat, p_value = stats.ttest_ind(recent_latencies, baseline_latencies)
            confidence = 1 - p_value
        except Exception:
            confidence = 0.5  # Fallback

        # Only alert if statistically significant
        if confidence < self.confidence_level:
            return None

        # Trend analysis
        trend = self._analyze_trend(
            [dp.latency_us for dp in data_points], [dp.timestamp for dp in data_points]
        )

        return RegressionAlert(
            timestamp=datetime.now(UTC),
            operation_name=data_points[0].operation_name,
            severity=severity,
            current_performance_us=recent_mean,
            baseline_performance_us=baseline_mean,
            degradation_percent=degradation_percent * 100,
            confidence_level=confidence,
            trend_direction=trend.direction,
            affected_metrics=["latency"],
            recommendations=self._generate_latency_recommendations(
                severity, degradation_percent
            ),
            metadata={
                "recent_samples": len(recent_data),
                "baseline_samples": len(baseline_latencies),
                "p_value": p_value if "t_stat" in locals() else None,
                "trend_slope": trend.slope,
                "trend_r_squared": trend.r_squared,
            },
        )

    async def _detect_memory_regression(
        self, data_points: list[PerformanceDataPoint]
    ) -> RegressionAlert | None:
        """Detect memory usage regression."""
        recent_data = [
            dp
            for dp in data_points
            if dp.timestamp
            >= datetime.now(UTC) - timedelta(days=self.detection_window_days)
        ]
        baseline_data = [
            dp
            for dp in data_points
            if dp.timestamp
            >= datetime.now(UTC) - timedelta(days=self.baseline_window_days)
        ]

        if len(recent_data) < 5 or len(baseline_data) < 10:
            return None

        recent_memory = [dp.memory_kb for dp in recent_data]
        baseline_memory = [dp.memory_kb for dp in baseline_data[: -len(recent_data)]]

        recent_mean = statistics.mean(recent_memory)
        baseline_mean = statistics.mean(baseline_memory)

        if (
            recent_mean <= baseline_mean * 1.1
        ):  # Allow 10% memory growth before alerting
            return None

        degradation_percent = (recent_mean - baseline_mean) / baseline_mean
        severity = self._determine_severity(degradation_percent)

        if severity is None:
            return None

        # Statistical significance test
        try:
            t_stat, p_value = stats.ttest_ind(recent_memory, baseline_memory)
            confidence = 1 - p_value
        except Exception:
            confidence = 0.5

        if confidence < self.confidence_level:
            return None

        trend = self._analyze_trend(
            [dp.memory_kb for dp in data_points], [dp.timestamp for dp in data_points]
        )

        return RegressionAlert(
            timestamp=datetime.now(UTC),
            operation_name=data_points[0].operation_name,
            severity=severity,
            current_performance_us=recent_mean,  # Using memory value in performance field
            baseline_performance_us=baseline_mean,
            degradation_percent=degradation_percent * 100,
            confidence_level=confidence,
            trend_direction=trend.direction,
            affected_metrics=["memory"],
            recommendations=self._generate_memory_recommendations(
                severity, degradation_percent
            ),
            metadata={
                "recent_samples": len(recent_data),
                "baseline_samples": len(baseline_memory),
                "p_value": p_value if "t_stat" in locals() else None,
                "trend_slope": trend.slope,
                "trend_r_squared": trend.r_squared,
            },
        )

    async def _detect_success_rate_regression(
        self, data_points: list[PerformanceDataPoint]
    ) -> RegressionAlert | None:
        """Detect success rate regression."""
        recent_data = [
            dp
            for dp in data_points
            if dp.timestamp
            >= datetime.now(UTC) - timedelta(days=self.detection_window_days)
        ]
        baseline_data = [
            dp
            for dp in data_points
            if dp.timestamp
            >= datetime.now(UTC) - timedelta(days=self.baseline_window_days)
        ]

        if len(recent_data) < 5 or len(baseline_data) < 10:
            return None

        recent_success_rates = [dp.success_rate for dp in recent_data]
        baseline_success_rates = [
            dp.success_rate for dp in baseline_data[: -len(recent_data)]
        ]

        recent_mean = statistics.mean(recent_success_rates)
        baseline_mean = statistics.mean(baseline_success_rates)

        # For success rate, we alert on degradation (lower values are worse)
        if recent_mean >= baseline_mean * 0.95:  # Allow 5% drop before alerting
            return None

        degradation_percent = (
            baseline_mean - recent_mean
        ) / baseline_mean  # Reversed for success rate
        severity = self._determine_severity(degradation_percent)

        if severity is None:
            return None

        try:
            t_stat, p_value = stats.ttest_ind(
                recent_success_rates, baseline_success_rates
            )
            confidence = 1 - p_value
        except Exception:
            confidence = 0.5

        if confidence < self.confidence_level:
            return None

        trend = self._analyze_trend(
            [dp.success_rate for dp in data_points],
            [dp.timestamp for dp in data_points],
        )

        return RegressionAlert(
            timestamp=datetime.now(UTC),
            operation_name=data_points[0].operation_name,
            severity=severity,
            current_performance_us=recent_mean * 100,  # Convert to percentage
            baseline_performance_us=baseline_mean * 100,
            degradation_percent=degradation_percent * 100,
            confidence_level=confidence,
            trend_direction=trend.direction,
            affected_metrics=["success_rate"],
            recommendations=self._generate_success_rate_recommendations(
                severity, degradation_percent
            ),
            metadata={
                "recent_samples": len(recent_data),
                "baseline_samples": len(baseline_success_rates),
                "p_value": p_value if "t_stat" in locals() else None,
                "trend_slope": trend.slope,
                "trend_r_squared": trend.r_squared,
            },
        )

    def _determine_severity(
        self, degradation_percent: float
    ) -> RegressionSeverity | None:
        """Determine regression severity based on degradation percentage."""
        if (
            degradation_percent
            >= self.regression_thresholds[RegressionSeverity.CRITICAL]
        ):
            return RegressionSeverity.CRITICAL
        if degradation_percent >= self.regression_thresholds[RegressionSeverity.HIGH]:
            return RegressionSeverity.HIGH
        if degradation_percent >= self.regression_thresholds[RegressionSeverity.MEDIUM]:
            return RegressionSeverity.MEDIUM
        if degradation_percent >= self.regression_thresholds[RegressionSeverity.LOW]:
            return RegressionSeverity.LOW
        return None

    def _analyze_trend(
        self, values: list[float], timestamps: list[datetime]
    ) -> PerformanceTrend:
        """Analyze performance trend using linear regression."""
        if len(values) < self.min_samples_for_trend:
            return PerformanceTrend(
                operation_name="unknown",
                direction=TrendDirection.STABLE,
                slope=0.0,
                r_squared=0.0,
                confidence_interval=(0.0, 0.0),
                data_points_count=len(values),
                analysis_period_days=0,
                volatility=0.0,
                projected_performance_7d=statistics.mean(values) if values else 0.0,
            )

        # Convert timestamps to days since first measurement
        first_timestamp = min(timestamps)
        x_days = [
            (ts - first_timestamp).total_seconds() / (24 * 3600) for ts in timestamps
        ]
        y_values = values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_days, y_values)
        r_squared = r_value**2

        # Calculate confidence interval for slope
        alpha = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(values) - 2)
        confidence_interval = (
            slope - t_critical * std_err,
            slope + t_critical * std_err,
        )

        # Determine trend direction
        if r_squared < 0.1:  # Weak correlation
            direction = TrendDirection.VOLATILE
        elif slope > std_err * 2:  # Significantly positive
            direction = TrendDirection.DEGRADING  # For latency, positive slope is bad
        elif slope < -std_err * 2:  # Significantly negative
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.STABLE

        # Calculate volatility (residuals standard deviation)
        predicted_values = [slope * x + intercept for x in x_days]
        residuals = [
            actual - predicted
            for actual, predicted in zip(y_values, predicted_values, strict=False)
        ]
        volatility = statistics.stdev(residuals) if len(residuals) > 1 else 0.0

        # Project performance 7 days ahead
        current_day = max(x_days) if x_days else 0
        projected_day = current_day + 7
        projected_performance_7d = slope * projected_day + intercept

        analysis_period = (max(timestamps) - min(timestamps)).days if timestamps else 0

        return PerformanceTrend(
            operation_name="trend_analysis",
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            confidence_interval=confidence_interval,
            data_points_count=len(values),
            analysis_period_days=analysis_period,
            volatility=volatility,
            projected_performance_7d=projected_performance_7d,
        )

    def _generate_latency_recommendations(
        self, severity: RegressionSeverity, degradation_percent: float
    ) -> list[str]:
        """Generate recommendations for latency regression."""
        recommendations = []

        if severity == RegressionSeverity.CRITICAL:
            recommendations.extend([
                "CRITICAL: Immediate action required - rollback recent changes if possible",
                "Profile application to identify bottlenecks causing >50% degradation",
                "Check for resource exhaustion (CPU, memory, disk I/O)",
                "Consider emergency scaling or load shedding",
            ])
        elif severity == RegressionSeverity.HIGH:
            recommendations.extend([
                "HIGH: Review recent code changes for performance impact",
                "Enable detailed performance profiling and monitoring",
                "Check database query performance and connection pool health",
                "Verify cache hit rates and cache invalidation patterns",
            ])
        elif severity == RegressionSeverity.MEDIUM:
            recommendations.extend([
                "MEDIUM: Investigate gradual performance degradation",
                "Review validation logic complexity and object creation patterns",
                "Consider optimization opportunities (caching, async processing)",
                "Monitor trend continuation and escalate if worsening",
            ])
        else:  # LOW
            recommendations.extend([
                "LOW: Minor degradation detected - monitor for trend continuation",
                "Review recent changes for unintended performance impact",
                "Consider proactive optimization to prevent further degradation",
            ])

        return recommendations

    def _generate_memory_recommendations(
        self, severity: RegressionSeverity, degradation_percent: float
    ) -> list[str]:
        """Generate recommendations for memory regression."""
        recommendations = []

        if severity == RegressionSeverity.CRITICAL:
            recommendations.extend([
                "CRITICAL: Memory usage increased >50% - risk of OOM conditions",
                "Check for memory leaks in validation object lifecycle",
                "Review object retention and garbage collection patterns",
                "Consider immediate memory optimization or scaling",
            ])
        elif severity == RegressionSeverity.HIGH:
            recommendations.extend([
                "HIGH: Significant memory increase - investigate object creation patterns",
                "Profile memory allocation and deallocation patterns",
                "Check for unnecessary object retention or caching issues",
                "Review validation data structure efficiency",
            ])
        else:
            recommendations.extend([
                "Monitor memory growth trend and investigate if continuing",
                "Review validation object lifecycle and cleanup procedures",
                "Consider optimizing data structures and reducing object creation",
            ])

        return recommendations

    def _generate_success_rate_recommendations(
        self, severity: RegressionSeverity, degradation_percent: float
    ) -> list[str]:
        """Generate recommendations for success rate regression."""
        recommendations = []

        if severity == RegressionSeverity.CRITICAL:
            recommendations.extend([
                "CRITICAL: Success rate dropped >50% - major reliability issue",
                "Investigate error patterns and exception causes immediately",
                "Check external dependencies (database, cache, services)",
                "Consider rollback or emergency fixes",
            ])
        elif severity == RegressionSeverity.HIGH:
            recommendations.extend([
                "HIGH: Significant increase in validation failures",
                "Analyze error logs for common failure patterns",
                "Check input data quality and validation rule changes",
                "Review timeout and retry configuration",
            ])
        else:
            recommendations.extend([
                "Monitor error rates and investigate root causes",
                "Review recent validation rule changes or data format updates",
                "Check for intermittent external service issues",
            ])

        return recommendations

    async def get_performance_summary(
        self, operation_name: str, days: int = 30
    ) -> dict[str, Any]:
        """Get performance summary for an operation."""
        if operation_name not in self.performance_history:
            return {"error": f"No data available for {operation_name}"}

        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        data_points = [
            dp
            for dp in self.performance_history[operation_name]
            if dp.timestamp >= cutoff_date
        ]

        if not data_points:
            return {"error": f"No recent data available for {operation_name}"}

        latencies = [dp.latency_us for dp in data_points]
        memory_usage = [dp.memory_kb for dp in data_points]
        success_rates = [dp.success_rate for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]

        # Trend analysis
        latency_trend = self._analyze_trend(latencies, timestamps)
        memory_trend = self._analyze_trend(memory_usage, timestamps)

        # Recent alerts
        recent_alerts = [
            alert
            for alert in self.alerts_history
            if alert.operation_name == operation_name
            and alert.timestamp >= datetime.now(UTC) - timedelta(days=7)
        ]

        return {
            "operation_name": operation_name,
            "analysis_period_days": days,
            "data_points_count": len(data_points),
            "latency_stats": {
                "mean_us": statistics.mean(latencies),
                "median_us": statistics.median(latencies),
                "p95_us": np.percentile(latencies, 95) if latencies else 0,
                "p99_us": np.percentile(latencies, 99) if latencies else 0,
                "min_us": min(latencies),
                "max_us": max(latencies),
                "std_us": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            },
            "memory_stats": {
                "mean_kb": statistics.mean(memory_usage),
                "median_kb": statistics.median(memory_usage),
                "min_kb": min(memory_usage),
                "max_kb": max(memory_usage),
            },
            "success_rate_stats": {
                "mean": statistics.mean(success_rates),
                "min": min(success_rates),
                "max": max(success_rates),
            },
            "trends": {
                "latency": {
                    "direction": latency_trend.direction.value,
                    "slope_us_per_day": latency_trend.slope,
                    "r_squared": latency_trend.r_squared,
                    "projected_7d_us": latency_trend.projected_performance_7d,
                    "volatility": latency_trend.volatility,
                },
                "memory": {
                    "direction": memory_trend.direction.value,
                    "slope_kb_per_day": memory_trend.slope,
                    "r_squared": memory_trend.r_squared,
                    "projected_7d_kb": memory_trend.projected_performance_7d,
                },
            },
            "recent_alerts_count": len(recent_alerts),
            "recent_alerts": [
                asdict(alert) for alert in recent_alerts[:10]
            ],  # Last 10 alerts
        }

    async def generate_ci_report(self) -> dict[str, Any]:
        """Generate CI/CD integration report."""
        all_operations = list(self.performance_history.keys())

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "regression_summary": {
                "total_operations_monitored": len(all_operations),
                "operations_with_regressions": 0,
                "critical_regressions": 0,
                "high_regressions": 0,
                "medium_regressions": 0,
                "low_regressions": 0,
            },
            "operation_status": {},
            "overall_status": "PASS",
            "recommendations": [],
        }

        # Check each operation for recent regressions
        for operation_name in all_operations:
            alerts = await self.check_for_regressions(operation_name)
            recent_alerts = [
                alert
                for alert in self.alerts_history
                if alert.operation_name == operation_name
                and alert.timestamp >= datetime.now(UTC) - timedelta(hours=24)
            ]

            has_regression = len(recent_alerts) > 0
            if has_regression:
                report["regression_summary"]["operations_with_regressions"] += 1

                # Count by severity
                for alert in recent_alerts:
                    if alert.severity == RegressionSeverity.CRITICAL:
                        report["regression_summary"]["critical_regressions"] += 1
                    elif alert.severity == RegressionSeverity.HIGH:
                        report["regression_summary"]["high_regressions"] += 1
                    elif alert.severity == RegressionSeverity.MEDIUM:
                        report["regression_summary"]["medium_regressions"] += 1
                    else:
                        report["regression_summary"]["low_regressions"] += 1

            # Get performance summary
            summary = await self.get_performance_summary(operation_name, days=7)

            report["operation_status"][operation_name] = {
                "has_regression": has_regression,
                "recent_alerts_count": len(recent_alerts),
                "current_latency_us": summary.get("latency_stats", {}).get(
                    "mean_us", 0
                ),
                "trend_direction": summary.get("trends", {})
                .get("latency", {})
                .get("direction", "unknown"),
                "status": "FAIL" if has_regression else "PASS",
            }

        # Determine overall status
        if report["regression_summary"]["critical_regressions"] > 0:
            report["overall_status"] = "CRITICAL"
        elif report["regression_summary"]["high_regressions"] > 0:
            report["overall_status"] = "HIGH"
        elif report["regression_summary"]["medium_regressions"] > 0:
            report["overall_status"] = "MEDIUM"
        elif report["regression_summary"]["low_regressions"] > 0:
            report["overall_status"] = "LOW"

        # Generate recommendations
        if report["regression_summary"]["critical_regressions"] > 0:
            report["recommendations"].append(
                "BLOCK DEPLOYMENT: Critical performance regressions detected"
            )
        elif report["regression_summary"]["high_regressions"] > 0:
            report["recommendations"].append(
                "CAUTION: High severity regressions - consider delaying deployment"
            )
        elif report["regression_summary"]["medium_regressions"] > 0:
            report["recommendations"].append(
                "REVIEW: Medium severity regressions - investigate before deployment"
            )
        else:
            report["recommendations"].append(
                "PROCEED: No significant performance regressions detected"
            )

        return report

    async def _load_performance_history(self) -> None:
        """Load performance history from disk."""
        history_file = self.data_dir / "performance_history.json"
        if not history_file.exists():
            return

        try:
            async with aiofiles.open(history_file) as f:
                content = await f.read()
                data = json.loads(content)

            for operation_name, data_points_data in data.items():
                self.performance_history[operation_name] = [
                    PerformanceDataPoint.from_dict(dp_data)
                    for dp_data in data_points_data
                ]

            logger.info(
                f"Loaded performance history for {len(self.performance_history)} operations"
            )
        except Exception as e:
            logger.error(f"Failed to load performance history: {e}")

    async def _save_performance_history(self) -> None:
        """Save performance history to disk."""
        history_file = self.data_dir / "performance_history.json"

        # Convert to serializable format
        data = {
            operation_name: [asdict(dp) for dp in data_points]
            for operation_name, data_points in self.performance_history.items()
        }

        try:
            async with aiofiles.open(history_file, "w") as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")

    async def _load_alerts_history(self) -> None:
        """Load alerts history from disk."""
        alerts_file = self.data_dir / "alerts_history.json"
        if not alerts_file.exists():
            return

        try:
            async with aiofiles.open(alerts_file) as f:
                content = await f.read()
                alerts_data = json.loads(content)

            self.alerts_history = []
            for alert_data in alerts_data:
                alert = RegressionAlert(
                    timestamp=datetime.fromisoformat(alert_data["timestamp"]),
                    operation_name=alert_data["operation_name"],
                    severity=RegressionSeverity(alert_data["severity"]),
                    current_performance_us=alert_data["current_performance_us"],
                    baseline_performance_us=alert_data["baseline_performance_us"],
                    degradation_percent=alert_data["degradation_percent"],
                    confidence_level=alert_data["confidence_level"],
                    trend_direction=TrendDirection(alert_data["trend_direction"]),
                    affected_metrics=alert_data["affected_metrics"],
                    recommendations=alert_data["recommendations"],
                    metadata=alert_data.get("metadata", {}),
                )
                self.alerts_history.append(alert)

            logger.info(f"Loaded {len(self.alerts_history)} alerts from history")
        except Exception as e:
            logger.error(f"Failed to load alerts history: {e}")

    async def _save_alerts_history(self) -> None:
        """Save alerts history to disk."""
        alerts_file = self.data_dir / "alerts_history.json"

        # Keep only recent alerts (last 90 days)
        cutoff_date = datetime.now(UTC) - timedelta(days=90)
        recent_alerts = [
            alert for alert in self.alerts_history if alert.timestamp >= cutoff_date
        ]

        alerts_data = [
            asdict(alert) for alert in recent_alerts[-1000:]
        ]  # Keep last 1000 alerts max

        try:
            async with aiofiles.open(alerts_file, "w") as f:
                await f.write(json.dumps(alerts_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save alerts history: {e}")


# Factory function
def get_regression_detector(
    data_dir: Path | None = None,
) -> PerformanceRegressionDetector:
    """Get or create regression detector instance."""
    return PerformanceRegressionDetector(data_dir)


# CLI integration for CI/CD
async def run_ci_regression_check(data_dir: str | None = None) -> int:
    """Run regression check for CI/CD integration.

    Returns:
        0 for success (no regressions)
        1 for low/medium regressions (warning)
        2 for high/critical regressions (failure)
    """
    detector = get_regression_detector(Path(data_dir) if data_dir else None)
    await detector.initialize()

    report = await detector.generate_ci_report()

    # Print report for CI logs
    print(json.dumps(report, indent=2))

    # Return appropriate exit code
    if report["overall_status"] in ["CRITICAL", "HIGH"]:
        return 2
    if report["overall_status"] in ["MEDIUM", "LOW"]:
        return 1
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Performance Regression Detection")
    parser.add_argument("--data-dir", type=str, help="Performance data directory")
    parser.add_argument(
        "--ci-check", action="store_true", help="Run CI/CD regression check"
    )

    args = parser.parse_args()

    async def main():
        if args.ci_check:
            exit_code = await run_ci_regression_check(args.data_dir)
            sys.exit(exit_code)
        else:
            detector = get_regression_detector(
                Path(args.data_dir) if args.data_dir else None
            )
            await detector.initialize()
            print("Regression detector initialized successfully")

    asyncio.run(main())
