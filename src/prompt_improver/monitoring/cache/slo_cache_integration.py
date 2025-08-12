"""SLO Cache Performance Integration
=================================

Enhanced integration between cache performance monitoring and SLO systems,
providing comprehensive correlation, predictive alerting, and performance
optimization recommendations.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from prompt_improver.monitoring.cache.unified_cache_monitoring import (
    AlertSeverity,
    get_unified_cache_monitor,
)
from prompt_improver.monitoring.slo.unified_observability import get_slo_observability

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
    slo_cache_tracer = trace.get_tracer(__name__ + ".slo_cache")
    slo_cache_meter = metrics.get_meter(__name__ + ".slo_cache")
    slo_cache_performance_impact = slo_cache_meter.create_gauge(
        "slo_cache_performance_impact",
        description="Impact of cache performance on SLO compliance",
        unit="ratio",
    )
    slo_cache_budget_consumption = slo_cache_meter.create_gauge(
        "slo_cache_error_budget_consumption",
        description="Error budget consumption due to cache performance",
        unit="percent",
    )
    cache_sli_violations = slo_cache_meter.create_counter(
        "cache_sli_violations_total",
        description="Total cache SLI violations by type and severity",
        unit="1",
    )
    predictive_cache_alerts = slo_cache_meter.create_counter(
        "predictive_cache_alerts_total",
        description="Total predictive cache performance alerts",
        unit="1",
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    slo_cache_tracer = None
    slo_cache_meter = None
    slo_cache_performance_impact = None
    slo_cache_budget_consumption = None
    cache_sli_violations = None
    predictive_cache_alerts = None
logger = logging.getLogger(__name__)


class CacheSLIType(Enum):
    """Cache-specific SLI types."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    HIT_RATE = "hit_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class PredictionConfidence(Enum):
    """Confidence levels for predictive alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class CacheSLI:
    """Cache Service Level Indicator."""

    sli_type: CacheSLIType
    target_value: float
    current_value: float
    compliance_ratio: float
    measurement_window: timedelta
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CachePerformanceTrend:
    """Cache performance trend analysis."""

    metric_name: str
    trend_direction: str
    slope: float
    confidence: PredictionConfidence
    projected_value: float
    projection_time: datetime
    historical_data: list[float] = field(default_factory=list)


@dataclass
class PredictiveAlert:
    """Predictive cache performance alert."""

    alert_id: str
    alert_type: str
    predicted_violation_time: datetime
    confidence: PredictionConfidence
    current_trend: CachePerformanceTrend
    impact_assessment: dict[str, Any]
    recommended_actions: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class CacheSLOIntegration:
    """Enhanced integration between cache monitoring and SLO systems.

    Provides predictive alerting, performance correlation, and
    automated optimization recommendations.
    """

    def __init__(self):
        """Initialize cache SLO integration."""
        self.slo_observability = get_slo_observability()
        self.cache_monitor = get_unified_cache_monitor()
        self._sli_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._trend_analyzers: dict[str, list[float]] = defaultdict(list)
        self._cache_slo_targets = {
            CacheSLIType.AVAILABILITY: 0.999,
            CacheSLIType.LATENCY: 50.0,
            CacheSLIType.HIT_RATE: 0.85,
            CacheSLIType.ERROR_RATE: 0.001,
            CacheSLIType.THROUGHPUT: 1000.0,
        }
        self._prediction_window = timedelta(minutes=30)
        self._trend_analysis_window = timedelta(hours=2)
        self._minimum_data_points = 20
        self._correlated_alerts: dict[str, list[str]] = defaultdict(list)
        self._alert_storm_prevention = {}
        self._performance_impact_history = deque(maxlen=1000)
        logger.info("CacheSLOIntegration initialized with predictive capabilities")

    async def calculate_cache_slis(
        self, measurement_window: timedelta = None
    ) -> dict[CacheSLIType, CacheSLI]:
        """Calculate cache Service Level Indicators.

        Args:
            measurement_window: Time window for SLI calculation

        Returns:
            Dictionary of calculated SLIs
        """
        if measurement_window is None:
            measurement_window = timedelta(minutes=5)
        cache_stats = self.cache_monitor.get_comprehensive_stats()
        slis = {}
        availability = self._calculate_availability_sli(cache_stats, measurement_window)
        if availability:
            slis[CacheSLIType.AVAILABILITY] = availability
        latency = self._calculate_latency_sli(cache_stats, measurement_window)
        if latency:
            slis[CacheSLIType.LATENCY] = latency
        hit_rate = self._calculate_hit_rate_sli(cache_stats, measurement_window)
        if hit_rate:
            slis[CacheSLIType.HIT_RATE] = hit_rate
        error_rate = self._calculate_error_rate_sli(cache_stats, measurement_window)
        if error_rate:
            slis[CacheSLIType.ERROR_RATE] = error_rate
        for sli_type, sli in slis.items():
            self._sli_history[sli_type.value].append({
                "timestamp": sli.timestamp.timestamp(),
                "value": sli.current_value,
                "compliance": sli.compliance_ratio,
            })
        if OPENTELEMETRY_AVAILABLE:
            for sli_type, sli in slis.items():
                if cache_sli_violations and sli.compliance_ratio < 1.0:
                    cache_sli_violations.add(
                        1,
                        {
                            "sli_type": sli_type.value,
                            "severity": "warning"
                            if sli.compliance_ratio > 0.8
                            else "critical",
                        },
                    )
        return slis

    def _calculate_availability_sli(
        self, cache_stats: dict, window: timedelta
    ) -> CacheSLI | None:
        """Calculate cache availability SLI."""
        try:
            total_ops = 0
            successful_ops = 0
            if "enhanced_monitoring" in cache_stats:
                performance_metrics = cache_stats["enhanced_monitoring"].get(
                    "performance_metrics", {}
                )
                for metric_key, metrics in performance_metrics.items():
                    if "operation_count" in metrics:
                        total_ops += metrics["operation_count"]
                        error_rate = metrics.get("error_rate", 0)
                        successful_ops += metrics["operation_count"] * (1 - error_rate)
            if total_ops == 0:
                return None
            availability = successful_ops / total_ops
            target = self._cache_slo_targets[CacheSLIType.AVAILABILITY]
            compliance = 1.0 if availability >= target else availability / target
            return CacheSLI(
                sli_type=CacheSLIType.AVAILABILITY,
                target_value=target,
                current_value=availability,
                compliance_ratio=compliance,
                measurement_window=window,
            )
        except Exception as e:
            logger.error(f"Error calculating availability SLI: {e}")
            return None

    def _calculate_latency_sli(
        self, cache_stats: dict, window: timedelta
    ) -> CacheSLI | None:
        """Calculate cache latency SLI (P95)."""
        try:
            p95_latencies = []
            if "enhanced_monitoring" in cache_stats:
                performance_metrics = cache_stats["enhanced_monitoring"].get(
                    "performance_metrics", {}
                )
                for metrics in performance_metrics.values():
                    if "p95_duration_ms" in metrics and metrics["p95_duration_ms"] > 0:
                        p95_latencies.append(metrics["p95_duration_ms"])
            if not p95_latencies:
                return None
            current_p95 = statistics.mean(p95_latencies)
            target = self._cache_slo_targets[CacheSLIType.LATENCY]
            compliance = 1.0 if current_p95 <= target else target / current_p95
            return CacheSLI(
                sli_type=CacheSLIType.LATENCY,
                target_value=target,
                current_value=current_p95,
                compliance_ratio=compliance,
                measurement_window=window,
            )
        except Exception as e:
            logger.error(f"Error calculating latency SLI: {e}")
            return None

    def _calculate_hit_rate_sli(
        self, cache_stats: dict, window: timedelta
    ) -> CacheSLI | None:
        """Calculate cache hit rate SLI."""
        try:
            overall_hit_rate = cache_stats.get("overall_hit_rate", 0)
            target = self._cache_slo_targets[CacheSLIType.HIT_RATE]
            compliance = (
                1.0 if overall_hit_rate >= target else overall_hit_rate / target
            )
            return CacheSLI(
                sli_type=CacheSLIType.HIT_RATE,
                target_value=target,
                current_value=overall_hit_rate,
                compliance_ratio=compliance,
                measurement_window=window,
            )
        except Exception as e:
            logger.error(f"Error calculating hit rate SLI: {e}")
            return None

    def _calculate_error_rate_sli(
        self, cache_stats: dict, window: timedelta
    ) -> CacheSLI | None:
        """Calculate cache error rate SLI."""
        try:
            total_errors = 0
            total_operations = 0
            if "enhanced_monitoring" in cache_stats:
                performance_metrics = cache_stats["enhanced_monitoring"].get(
                    "performance_metrics", {}
                )
                for metrics in performance_metrics.values():
                    if "operation_count" in metrics and "error_rate" in metrics:
                        ops = metrics["operation_count"]
                        error_rate = metrics["error_rate"]
                        total_operations += ops
                        total_errors += ops * error_rate
            if total_operations == 0:
                return None
            current_error_rate = total_errors / total_operations
            target = self._cache_slo_targets[CacheSLIType.ERROR_RATE]
            compliance = (
                1.0 if current_error_rate <= target else target / current_error_rate
            )
            return CacheSLI(
                sli_type=CacheSLIType.ERROR_RATE,
                target_value=target,
                current_value=current_error_rate,
                compliance_ratio=compliance,
                measurement_window=window,
            )
        except Exception as e:
            logger.error(f"Error calculating error rate SLI: {e}")
            return None

    async def analyze_performance_trends(self) -> dict[str, CachePerformanceTrend]:
        """Analyze cache performance trends for predictive alerting.

        Returns:
            Dictionary of performance trends by metric
        """
        trends = {}
        for sli_type_str, history in self._sli_history.items():
            if len(history) < self._minimum_data_points:
                continue
            try:
                trend = self._calculate_trend(sli_type_str, list(history))
                if trend:
                    trends[sli_type_str] = trend
            except Exception as e:
                logger.error(f"Error analyzing trend for {sli_type_str}: {e}")
        return trends

    def _calculate_trend(
        self, metric_name: str, data_points: list[dict]
    ) -> CachePerformanceTrend | None:
        """Calculate performance trend using linear regression."""
        if len(data_points) < self._minimum_data_points:
            return None
        values = [point["value"] for point in data_points]
        timestamps = [point["timestamp"] for point in data_points]
        n = len(values)
        x = list(range(n))
        y = values
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return None
        slope = numerator / denominator
        if abs(slope) < 0.001:
            trend_direction = "stable"
            confidence = PredictionConfidence.HIGH
        elif slope > 0:
            trend_direction = (
                "improving"
                if metric_name in ["hit_rate", "availability"]
                else "degrading"
            )
            confidence = (
                PredictionConfidence.MEDIUM
                if abs(slope) < 0.01
                else PredictionConfidence.HIGH
            )
        else:
            trend_direction = (
                "degrading"
                if metric_name in ["hit_rate", "availability"]
                else "improving"
            )
            confidence = (
                PredictionConfidence.MEDIUM
                if abs(slope) < 0.01
                else PredictionConfidence.HIGH
            )
        time_steps_ahead = int(self._prediction_window.total_seconds() / 60)
        projected_value = y[-1] + slope * time_steps_ahead
        projection_time = datetime.now(UTC) + self._prediction_window
        return CachePerformanceTrend(
            metric_name=metric_name,
            trend_direction=trend_direction,
            slope=slope,
            confidence=confidence,
            projected_value=projected_value,
            projection_time=projection_time,
            historical_data=values[-50:],
        )

    async def generate_predictive_alerts(self) -> list[PredictiveAlert]:
        """Generate predictive alerts based on performance trends.

        Returns:
            List of predictive alerts
        """
        trends = await self.analyze_performance_trends()
        alerts = []
        for metric_name, trend in trends.items():
            if self._will_violate_sli(metric_name, trend):
                alert = self._create_predictive_alert(metric_name, trend)
                if alert:
                    alerts.append(alert)
        if OPENTELEMETRY_AVAILABLE and predictive_cache_alerts:
            for alert in alerts:
                predictive_cache_alerts.add(
                    1,
                    {
                        "alert_type": alert.alert_type,
                        "confidence": alert.confidence.value,
                    },
                )
        return alerts

    def _will_violate_sli(self, metric_name: str, trend: CachePerformanceTrend) -> bool:
        """Check if trend indicates future SLI violation."""
        try:
            sli_type = CacheSLIType(metric_name)
            target = self._cache_slo_targets[sli_type]
            if sli_type in [CacheSLIType.HIT_RATE, CacheSLIType.AVAILABILITY]:
                return (
                    trend.projected_value < target
                    and trend.trend_direction == "degrading"
                )
            return (
                trend.projected_value > target and trend.trend_direction == "degrading"
            )
        except ValueError:
            return False

    def _create_predictive_alert(
        self, metric_name: str, trend: CachePerformanceTrend
    ) -> PredictiveAlert | None:
        """Create predictive alert from trend analysis."""
        try:
            sli_type = CacheSLIType(metric_name)
            target = self._cache_slo_targets[sli_type]
            if trend.slope == 0:
                return None
            current_value = trend.historical_data[-1] if trend.historical_data else 0
            steps_to_violation = abs((target - current_value) / trend.slope)
            violation_time = datetime.now(UTC) + timedelta(minutes=steps_to_violation)
            recommendations = self._generate_recommendations(sli_type, trend)
            impact = self._assess_performance_impact(sli_type, trend)
            alert = PredictiveAlert(
                alert_id=f"predictive_{metric_name}_{int(time.time())}",
                alert_type=f"predicted_{metric_name}_violation",
                predicted_violation_time=violation_time,
                confidence=trend.confidence,
                current_trend=trend,
                impact_assessment=impact,
                recommended_actions=recommendations,
            )
            return alert
        except Exception as e:
            logger.error(f"Error creating predictive alert for {metric_name}: {e}")
            return None

    def _generate_recommendations(
        self, sli_type: CacheSLIType, trend: CachePerformanceTrend
    ) -> list[str]:
        """Generate recommendations based on SLI type and trend."""
        recommendations = []
        if sli_type == CacheSLIType.HIT_RATE:
            recommendations.extend([
                "Increase cache warming frequency for popular keys",
                "Review cache TTL settings to prevent premature expiration",
                "Analyze access patterns for cache optimization opportunities",
                "Consider increasing cache capacity if memory allows",
            ])
        elif sli_type == CacheSLIType.LATENCY:
            recommendations.extend([
                "Review cache backend performance and connections",
                "Optimize serialization/deserialization processes",
                "Check for network latency issues to cache backends",
                "Consider cache data size optimization",
            ])
        elif sli_type == CacheSLIType.AVAILABILITY:
            recommendations.extend([
                "Check cache backend health and connectivity",
                "Review error handling and retry logic",
                "Consider implementing cache failover mechanisms",
                "Monitor resource utilization on cache backends",
            ])
        elif sli_type == CacheSLIType.ERROR_RATE:
            recommendations.extend([
                "Investigate root causes of cache operation failures",
                "Review timeout configurations",
                "Check for resource contention or capacity issues",
                "Implement better error handling and logging",
            ])
        return recommendations

    def _assess_performance_impact(
        self, sli_type: CacheSLIType, trend: CachePerformanceTrend
    ) -> dict[str, Any]:
        """Assess potential impact of performance degradation."""
        impact = {
            "severity": "medium",
            "affected_services": ["cache_layer"],
            "business_impact": "potential_latency_increase",
            "user_experience_impact": "minimal",
        }
        if sli_type == CacheSLIType.AVAILABILITY and trend.projected_value < 0.99:
            impact["severity"] = "high"
            impact["business_impact"] = "service_degradation"
            impact["user_experience_impact"] = "noticeable"
        elif sli_type == CacheSLIType.LATENCY and trend.projected_value > 200:
            impact["severity"] = "high"
            impact["business_impact"] = "performance_impact"
            impact["user_experience_impact"] = "significant"
        elif sli_type == CacheSLIType.HIT_RATE and trend.projected_value < 0.7:
            impact["severity"] = "medium"
            impact["business_impact"] = "increased_backend_load"
            impact["user_experience_impact"] = "minor"
        return impact

    async def correlate_with_slo_monitoring(self):
        """Correlate cache performance with overall SLO monitoring."""
        try:
            cache_slis = await self.calculate_cache_slis()
            for sli_type, sli in cache_slis.items():
                async with self.slo_observability.observe_slo_operation(
                    operation="cache_sli_measurement",
                    service_name="unified_cache",
                    target_name=sli_type.value,
                    component="cache_monitoring",
                ):
                    await self.slo_observability.record_slo_compliance(
                        service_name="unified_cache",
                        target_name=sli_type.value,
                        compliance_ratio=sli.compliance_ratio,
                        error_budget_remaining=(1.0 - sli.compliance_ratio) * 100,
                        time_window=sli.measurement_window,
                        additional_metrics={
                            "current_value": sli.current_value,
                            "target_value": sli.target_value,
                        },
                    )
            if OPENTELEMETRY_AVAILABLE and slo_cache_performance_impact:
                overall_impact = self._calculate_overall_performance_impact(cache_slis)
                slo_cache_performance_impact.set(overall_impact)
        except Exception as e:
            logger.error(f"Error correlating with SLO monitoring: {e}")

    def _calculate_overall_performance_impact(
        self, cache_slis: dict[CacheSLIType, CacheSLI]
    ) -> float:
        """Calculate overall performance impact score."""
        if not cache_slis:
            return 0.0
        weights = {
            CacheSLIType.AVAILABILITY: 0.3,
            CacheSLIType.LATENCY: 0.25,
            CacheSLIType.HIT_RATE: 0.25,
            CacheSLIType.ERROR_RATE: 0.2,
        }
        weighted_impact = 0.0
        total_weight = 0.0
        for sli_type, sli in cache_slis.items():
            if sli_type in weights:
                impact = 1.0 - sli.compliance_ratio
                weighted_impact += impact * weights[sli_type]
                total_weight += weights[sli_type]
        return weighted_impact / total_weight if total_weight > 0 else 0.0

    async def get_slo_cache_report(self) -> dict[str, Any]:
        """Generate comprehensive SLO cache performance report.

        Returns:
            Dictionary with SLO cache performance data
        """
        cache_slis = await self.calculate_cache_slis()
        trends = await self.analyze_performance_trends()
        predictive_alerts = await self.generate_predictive_alerts()
        health_score = self._calculate_cache_health_score(cache_slis)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "cache_slis": {
                sli_type.value: {
                    "current_value": sli.current_value,
                    "target_value": sli.target_value,
                    "compliance_ratio": sli.compliance_ratio,
                    "status": "compliant"
                    if sli.compliance_ratio >= 0.95
                    else "at_risk"
                    if sli.compliance_ratio >= 0.8
                    else "violation",
                }
                for sli_type, sli in cache_slis.items()
            },
            "performance_trends": {
                metric: {
                    "direction": trend.trend_direction,
                    "confidence": trend.confidence.value,
                    "projected_value": trend.projected_value,
                    "projection_time": trend.projection_time.isoformat(),
                }
                for metric, trend in trends.items()
            },
            "predictive_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "type": alert.alert_type,
                    "predicted_time": alert.predicted_violation_time.isoformat(),
                    "confidence": alert.confidence.value,
                    "recommendations": alert.recommended_actions[:3],
                }
                for alert in predictive_alerts
            ],
            "overall_health": {
                "score": health_score,
                "status": "healthy"
                if health_score > 0.8
                else "degraded"
                if health_score > 0.6
                else "critical",
            },
            "integration_status": {
                "slo_observability_connected": self.slo_observability is not None,
                "cache_monitor_connected": self.cache_monitor is not None,
                "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
            },
        }

    def _calculate_cache_health_score(
        self, cache_slis: dict[CacheSLIType, CacheSLI]
    ) -> float:
        """Calculate overall cache health score (0.0 to 1.0)."""
        if not cache_slis:
            return 0.0
        total_compliance = sum(sli.compliance_ratio for sli in cache_slis.values())
        return total_compliance / len(cache_slis)

    async def start_monitoring(self):
        """Start background monitoring tasks."""
        from prompt_improver.performance.monitoring.health.background_manager import (
            TaskPriority,
            get_background_task_manager,
        )

        task_manager = get_background_task_manager()
        if task_manager:
            await task_manager.submit_enhanced_task(
                task_id="cache_slo_correlation",
                coroutine=self._periodic_slo_correlation(),
                priority=TaskPriority.NORMAL,
                tags={
                    "service": "cache_monitoring",
                    "type": "slo_correlation",
                    "component": "cache_slo_integration",
                },
            )
            await task_manager.submit_enhanced_task(
                task_id="cache_predictive_analysis",
                coroutine=self._periodic_predictive_analysis(),
                priority=TaskPriority.BACKGROUND,
                tags={
                    "service": "cache_monitoring",
                    "type": "predictive_analysis",
                    "component": "cache_slo_integration",
                },
            )
        logger.info("Cache SLO monitoring started")

    async def _periodic_slo_correlation(self):
        """Periodic SLO correlation task."""
        while True:
            try:
                await self.correlate_with_slo_monitoring()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Periodic SLO correlation failed: {e}")
                await asyncio.sleep(30)

    async def _periodic_predictive_analysis(self):
        """Periodic predictive analysis task."""
        while True:
            try:
                alerts = await self.generate_predictive_alerts()
                if alerts:
                    logger.info(f"Generated {len(alerts)} predictive cache alerts")
                    for alert in alerts:
                        logger.warning(
                            f"Predictive Alert: {alert.alert_type} - {alert.predicted_violation_time}"
                        )
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Periodic predictive analysis failed: {e}")
                await asyncio.sleep(60)


_cache_slo_integration: CacheSLOIntegration | None = None


def get_cache_slo_integration() -> CacheSLOIntegration:
    """Get or create global cache SLO integration instance."""
    global _cache_slo_integration
    if _cache_slo_integration is None:
        _cache_slo_integration = CacheSLOIntegration()
    return _cache_slo_integration


async def initialize_cache_slo_monitoring():
    """Initialize and start cache SLO monitoring."""
    integration = get_cache_slo_integration()
    await integration.start_monitoring()
    logger.info("Cache SLO monitoring initialized and started")
