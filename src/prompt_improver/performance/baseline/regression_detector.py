"""Performance regression detection and alerting system."""

import logging
import statistics
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from prompt_improver.performance.baseline.models import (
    AlertSeverity,
    BaselineMetrics,
    RegressionAlert,
    get_metric_definition,
)
from prompt_improver.performance.baseline.statistical_analyzer import (
    StatisticalAnalyzer,
)

try:
    import aiohttp

    WEBHOOK_AVAILABLE = True
except ImportError:
    WEBHOOK_AVAILABLE = False
logger = logging.getLogger(__name__)


class RegressionThresholds:
    """Configuration for regression detection thresholds."""

    def __init__(
        self,
        warning_degradation_percent: float = 15.0,
        critical_degradation_percent: float = 25.0,
        emergency_degradation_percent: float = 50.0,
        minimum_samples: int = 5,
        significance_threshold: float = 0.05,
        consecutive_violations: int = 3,
    ) -> None:
        """Initialize regression thresholds.

        Args:
            warning_degradation_percent: % degradation to trigger warning
            critical_degradation_percent: % degradation to trigger critical alert
            emergency_degradation_percent: % degradation to trigger emergency alert
            minimum_samples: Minimum samples required for detection
            significance_threshold: Statistical significance threshold
            consecutive_violations: Required consecutive violations
        """
        self.warning_degradation_percent = warning_degradation_percent
        self.critical_degradation_percent = critical_degradation_percent
        self.emergency_degradation_percent = emergency_degradation_percent
        self.minimum_samples = minimum_samples
        self.significance_threshold = significance_threshold
        self.consecutive_violations = consecutive_violations


class AlertChannel:
    """Base class for alert channels."""

    async def send_alert(self, alert: RegressionAlert) -> bool:
        """Send alert through this channel."""
        raise NotImplementedError


class LogAlertChannel(AlertChannel):
    """Log-based alert channel."""

    def __init__(self, logger_name: str = __name__) -> None:
        self.logger = logging.getLogger(logger_name)

    async def send_alert(self, alert: RegressionAlert) -> bool:
        """Send alert to logs."""
        try:
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.EMERGENCY: logging.CRITICAL,
            }.get(alert.severity, logging.WARNING)
            self.logger.log(
                log_level,
                f"Performance Regression Detected: {alert.message} (Metric: {alert.metric_name}, Degradation: {alert.degradation_percentage:.1f}%)",
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to send log alert: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Webhook-based alert channel."""

    def __init__(self, webhook_url: str, timeout: int = 30) -> None:
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def send_alert(self, alert: RegressionAlert) -> bool:
        """Send alert via webhook."""
        if not WEBHOOK_AVAILABLE:
            logger.warning("Webhook alerts not available (aiohttp not installed)")
            return False
        try:
            payload = {
                "alert_type": "performance_regression",
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "message": alert.message,
                "current_value": alert.current_value,
                "baseline_value": alert.baseline_value,
                "degradation_percentage": alert.degradation_percentage,
                "timestamp": alert.alert_timestamp.isoformat(),
                "alert_id": alert.alert_id,
                "affected_operations": alert.affected_operations,
                "probable_causes": alert.probable_causes,
                "remediation_suggestions": alert.remediation_suggestions,
            }
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response,
            ):
                if response.status == 200:
                    logger.info(
                        f"Webhook alert sent successfully for {alert.metric_name}"
                    )
                    return True
                logger.error(f"Webhook alert failed with status {response.status}")
                return False
        except Exception as e:
            logger.exception(f"Failed to send webhook alert: {e}")
            return False


class RegressionDetector:
    """Advanced performance regression detection with alerting.

    Monitors performance baselines and detects significant degradations,
    sending alerts through configured channels.
    """

    def __init__(
        self,
        thresholds: RegressionThresholds | None = None,
        alert_channels: list[AlertChannel] | None = None,
        enable_statistical_analysis: bool = True,
        cooldown_period: int = 300,
        max_alerts_per_hour: int = 10,
    ) -> None:
        """Initialize regression detector.

        Args:
            thresholds: Detection threshold configuration
            alert_channels: List of alert channels to use
            enable_statistical_analysis: Use statistical significance testing
            cooldown_period: Minimum time between alerts for same metric (seconds)
            max_alerts_per_hour: Maximum alerts per hour per metric
        """
        self.thresholds = thresholds or RegressionThresholds()
        self.alert_channels = alert_channels or [LogAlertChannel()]
        self.enable_statistical_analysis = enable_statistical_analysis
        self.cooldown_period = cooldown_period
        self.max_alerts_per_hour = max_alerts_per_hour
        self._violation_counts: dict[str, int] = {}
        self._last_alert_times: dict[str, datetime] = {}
        self._alert_counts: dict[str, list[datetime]] = {}
        self._active_alerts: dict[str, RegressionAlert] = {}
        self._baseline_history: list[BaselineMetrics] = []
        if self.enable_statistical_analysis:
            self.analyzer = StatisticalAnalyzer(
                significance_threshold=self.thresholds.significance_threshold
            )
        logger.info(
            f"RegressionDetector initialized with {len(self.alert_channels)} channels"
        )

    async def check_for_regressions(
        self,
        current_baseline: BaselineMetrics,
        reference_baselines: list[BaselineMetrics] | None = None,
    ) -> list[RegressionAlert]:
        """Check current baseline for performance regressions.

        Args:
            current_baseline: Current baseline to analyze
            reference_baselines: Historical baselines for comparison

        Returns:
            List of detected regression alerts
        """
        alerts = []
        if reference_baselines is None:
            reference_baselines = self._baseline_history[-20:]
        if len(reference_baselines) < self.thresholds.minimum_samples:
            logger.debug(
                f"Insufficient reference baselines ({len(reference_baselines)} < {self.thresholds.minimum_samples})"
            )
            return alerts
        metrics_to_check = self._get_metrics_to_check(
            current_baseline, reference_baselines
        )
        for metric_name in metrics_to_check:
            try:
                alert = await self._check_metric_regression(
                    metric_name, current_baseline, reference_baselines
                )
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.exception(f"Error checking regression for {metric_name}: {e}")
        self._baseline_history.append(current_baseline)
        if len(self._baseline_history) > 100:
            self._baseline_history = self._baseline_history[-100:]
        for alert in alerts:
            await self._process_alert(alert)
        return alerts

    async def _check_metric_regression(
        self,
        metric_name: str,
        current_baseline: BaselineMetrics,
        reference_baselines: list[BaselineMetrics],
    ) -> RegressionAlert | None:
        """Check a specific metric for regression."""
        current_values = self._extract_metric_values(current_baseline, metric_name)
        if not current_values:
            return None
        reference_values = []
        for baseline in reference_baselines:
            values = self._extract_metric_values(baseline, metric_name)
            reference_values.extend(values)
        if len(reference_values) < self.thresholds.minimum_samples:
            return None
        current_mean = statistics.mean(current_values)
        reference_mean = statistics.mean(reference_values)
        metric_def = get_metric_definition(metric_name)
        lower_is_better = metric_def.lower_is_better if metric_def else True
        if reference_mean == 0:
            return None
        if lower_is_better:
            degradation_pct = (current_mean - reference_mean) / reference_mean * 100
        else:
            degradation_pct = (reference_mean - current_mean) / reference_mean * 100
        if degradation_pct <= 0:
            self._violation_counts[metric_name] = 0
            return None
        severity = self._determine_severity(degradation_pct)
        if severity is None:
            self._violation_counts[metric_name] = 0
            return None
        is_significant = True
        if (
            self.enable_statistical_analysis
            and len(current_values) > 1
            and (len(reference_values) > 1)
        ):
            try:
                trend = await self.analyzer.analyze_trend(
                    metric_name, [*reference_baselines, current_baseline]
                )
                is_significant = trend.is_significant()
            except Exception as e:
                logger.debug(f"Statistical analysis failed for {metric_name}: {e}")
        if not is_significant:
            logger.debug(f"Regression in {metric_name} not statistically significant")
            return None
        self._violation_counts[metric_name] = (
            self._violation_counts.get(metric_name, 0) + 1
        )
        if self._violation_counts[metric_name] < self.thresholds.consecutive_violations:
            logger.debug(
                f"Regression in {metric_name}: {self._violation_counts[metric_name]}/{self.thresholds.consecutive_violations} violations"
            )
            return None
        if not self._is_alert_allowed(metric_name):
            return None
        return RegressionAlert(
            alert_id=str(uuid.uuid4()),
            metric_name=metric_name,
            severity=severity,
            message=self._generate_alert_message(
                metric_name, degradation_pct, severity
            ),
            current_value=current_mean,
            baseline_value=reference_mean,
            threshold_value=self._get_threshold_value(degradation_pct),
            degradation_percentage=degradation_pct,
            detection_timestamp=current_baseline.collection_timestamp,
            alert_timestamp=datetime.now(UTC),
            affected_operations=self._identify_affected_operations(
                metric_name, current_baseline
            ),
            probable_causes=self._identify_probable_causes(
                metric_name, degradation_pct
            ),
            remediation_suggestions=self._generate_remediation_suggestions(
                metric_name, severity
            ),
        )

    def _extract_metric_values(
        self, baseline: BaselineMetrics, metric_name: str
    ) -> list[float]:
        """Extract values for a specific metric from baseline."""
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
        return []

    def _get_metrics_to_check(
        self, current: BaselineMetrics, reference: list[BaselineMetrics]
    ) -> list[str]:
        """Get list of metrics to check for regressions."""
        metrics = set()
        if current.response_times and any(b.response_times for b in reference):
            metrics.add("response_time")
        if current.error_rates and any(b.error_rates for b in reference):
            metrics.add("error_rate")
        if current.throughput_values and any(b.throughput_values for b in reference):
            metrics.add("throughput")
        if current.cpu_utilization and any(b.cpu_utilization for b in reference):
            metrics.add("cpu_utilization")
        if current.memory_utilization and any(b.memory_utilization for b in reference):
            metrics.add("memory_utilization")
        if current.database_connection_time and any(
            b.database_connection_time for b in reference
        ):
            metrics.add("database_connection_time")
        if current.cache_hit_rate and any(b.cache_hit_rate for b in reference):
            metrics.add("cache_hit_rate")
        for metric_name in current.custom_metrics:
            if any(metric_name in b.custom_metrics for b in reference):
                metrics.add(metric_name)
        return list(metrics)

    def _determine_severity(self, degradation_pct: float) -> AlertSeverity | None:
        """Determine alert severity based on degradation percentage."""
        if degradation_pct >= self.thresholds.emergency_degradation_percent:
            return AlertSeverity.EMERGENCY
        if degradation_pct >= self.thresholds.critical_degradation_percent:
            return AlertSeverity.CRITICAL
        if degradation_pct >= self.thresholds.warning_degradation_percent:
            return AlertSeverity.WARNING
        return None

    def _get_threshold_value(self, degradation_pct: float) -> float:
        """Get the threshold value that was exceeded."""
        if degradation_pct >= self.thresholds.emergency_degradation_percent:
            return self.thresholds.emergency_degradation_percent
        if degradation_pct >= self.thresholds.critical_degradation_percent:
            return self.thresholds.critical_degradation_percent
        return self.thresholds.warning_degradation_percent

    def _is_alert_allowed(self, metric_name: str) -> bool:
        """Check if alert is allowed based on cooldown and rate limits."""
        current_time = datetime.now(UTC)
        if metric_name in self._last_alert_times:
            time_since_last = (
                current_time - self._last_alert_times[metric_name]
            ).total_seconds()
            if time_since_last < self.cooldown_period:
                return False
        if metric_name not in self._alert_counts:
            self._alert_counts[metric_name] = []
        one_hour_ago = current_time - timedelta(hours=1)
        self._alert_counts[metric_name] = [
            alert_time
            for alert_time in self._alert_counts[metric_name]
            if alert_time > one_hour_ago
        ]
        return not len(self._alert_counts[metric_name]) >= self.max_alerts_per_hour

    def _generate_alert_message(
        self, metric_name: str, degradation_pct: float, severity: AlertSeverity
    ) -> str:
        """Generate human-readable alert message."""
        metric_display = metric_name.replace("_", " ").title()
        if severity == AlertSeverity.EMERGENCY:
            return f"EMERGENCY: {metric_display} has severely degraded by {degradation_pct:.1f}%"
        if severity == AlertSeverity.CRITICAL:
            return f"CRITICAL: {metric_display} has critically degraded by {degradation_pct:.1f}%"
        return f"WARNING: {metric_display} has degraded by {degradation_pct:.1f}%"

    def _identify_affected_operations(
        self, metric_name: str, baseline: BaselineMetrics
    ) -> list[str]:
        """Identify operations affected by the regression."""
        affected = []
        for custom_metric in baseline.custom_metrics:
            if metric_name in custom_metric and "_" in custom_metric:
                operation = custom_metric.split("_")[0]
                if operation not in affected:
                    affected.append(operation)
        if metric_name in {"response_time", "database_connection_time"}:
            affected.extend(["API endpoints", "Database operations"])
        elif metric_name == "error_rate":
            affected.extend(["All operations", "User requests"])
        elif metric_name in {"cpu_utilization", "memory_utilization"}:
            affected.extend(["System performance", "All operations"])
        elif metric_name == "cache_hit_rate":
            affected.extend(["Cache operations", "Data retrieval"])
        return list(set(affected))

    def _identify_probable_causes(
        self, metric_name: str, degradation_pct: float
    ) -> list[str]:
        """Identify probable causes of the regression."""
        causes = []
        if metric_name == "response_time":
            causes.extend([
                "Increased load or traffic",
                "Database performance issues",
                "Network latency",
                "Resource contention",
                "Inefficient code changes",
            ])
        elif metric_name == "error_rate":
            causes.extend([
                "Recent code deployment",
                "External service failures",
                "Database connectivity issues",
                "Configuration changes",
                "Resource exhaustion",
            ])
        elif metric_name in {"cpu_utilization", "memory_utilization"}:
            causes.extend([
                "Increased workload",
                "Memory leaks",
                "Inefficient algorithms",
                "Resource-intensive operations",
                "Background processes",
            ])
        elif metric_name == "database_connection_time":
            causes.extend([
                "Database server overload",
                "Network connectivity issues",
                "Database configuration changes",
                "Connection pool exhaustion",
            ])
        elif metric_name == "cache_hit_rate":
            causes.extend([
                "Cache invalidation issues",
                "Memory pressure",
                "Cache configuration changes",
                "Data access pattern changes",
            ])
        if degradation_pct > 50:
            causes.extend([
                "System outage or partial failure",
                "Critical resource exhaustion",
                "Major configuration error",
            ])
        return causes

    def _generate_remediation_suggestions(
        self, metric_name: str, severity: AlertSeverity
    ) -> list[str]:
        """Generate remediation suggestions."""
        suggestions = []
        if severity == AlertSeverity.EMERGENCY:
            suggestions.extend([
                "Investigate immediately - system may be in critical state",
                "Consider rolling back recent changes",
                "Check system resource availability",
                "Escalate to on-call engineer",
            ])
        elif severity == AlertSeverity.CRITICAL:
            suggestions.extend([
                "Investigate within 30 minutes",
                "Review recent deployments",
                "Check system metrics and logs",
            ])
        if metric_name == "response_time":
            suggestions.extend([
                "Review database query performance",
                "Check for resource bottlenecks",
                "Analyze request patterns",
                "Consider scaling resources",
            ])
        elif metric_name == "error_rate":
            suggestions.extend([
                "Check application logs for error patterns",
                "Verify external service status",
                "Review recent code changes",
                "Test critical user flows",
            ])
        elif metric_name in {"cpu_utilization", "memory_utilization"}:
            suggestions.extend([
                "Monitor resource usage trends",
                "Identify resource-intensive processes",
                "Consider horizontal scaling",
                "Review application efficiency",
            ])
        suggestions.extend([
            "Monitor metric trends for recovery",
            "Document findings for future reference",
            "Update monitoring thresholds if needed",
        ])
        return suggestions

    async def _process_alert(self, alert: RegressionAlert) -> None:
        """Process and send an alert through all channels."""
        self._last_alert_times[alert.metric_name] = alert.alert_timestamp
        if alert.metric_name not in self._alert_counts:
            self._alert_counts[alert.metric_name] = []
        self._alert_counts[alert.metric_name].append(alert.alert_timestamp)
        self._active_alerts[alert.alert_id] = alert
        for channel in self.alert_channels:
            try:
                success = await channel.send_alert(alert)
                if success:
                    logger.info(
                        f"Alert sent via {type(channel).__name__} for {alert.metric_name}"
                    )
                else:
                    logger.error(f"Failed to send alert via {type(channel).__name__}")
            except Exception as e:
                logger.exception(f"Error sending alert via {type(channel).__name__}: {e}")

    async def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """Mark an alert as resolved."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.metadata["resolved"] = True
            alert.metadata["resolution_time"] = datetime.now(UTC).isoformat()
            alert.metadata["resolution_note"] = resolution_note
            self._violation_counts[alert.metric_name] = 0
            logger.info(f"Alert {alert_id} resolved for metric {alert.metric_name}")
            return True
        return False

    def get_active_alerts(self) -> list[RegressionAlert]:
        """Get all active alerts."""
        return [
            alert
            for alert in self._active_alerts.values()
            if not alert.metadata.get("resolved", False)
        ]

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get statistics about alert activity."""
        current_time = datetime.now(UTC)
        alerts_24h = 0
        for alert_times in self._alert_counts.values():
            alerts_24h += len([
                t for t in alert_times if (current_time - t).total_seconds() < 24 * 3600
            ])
        return {
            "active_alerts": len(self.get_active_alerts()),
            "total_alerts_24h": alerts_24h,
            "metrics_with_violations": len([
                metric for metric, count in self._violation_counts.items() if count > 0
            ]),
            "alert_channels": len(self.alert_channels),
            "violation_counts": self._violation_counts.copy(),
            "thresholds": {
                "warning": self.thresholds.warning_degradation_percent,
                "critical": self.thresholds.critical_degradation_percent,
                "emergency": self.thresholds.emergency_degradation_percent,
            },
        }


_global_detector: RegressionDetector | None = None


def get_regression_detector() -> RegressionDetector:
    """Get the global regression detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = RegressionDetector()
    return _global_detector


def set_regression_detector(detector: RegressionDetector) -> None:
    """Set the global regression detector instance."""
    global _global_detector
    _global_detector = detector


async def check_baseline_for_regressions(
    baseline: BaselineMetrics, reference_baselines: list[BaselineMetrics] | None = None
) -> list[RegressionAlert]:
    """Check a baseline for regressions using the global detector."""
    detector = get_regression_detector()
    return await detector.check_for_regressions(baseline, reference_baselines)


async def setup_webhook_alerts(webhook_url: str) -> None:
    """Setup webhook alerts for the global detector."""
    detector = get_regression_detector()
    webhook_channel = WebhookAlertChannel(webhook_url)
    detector.alert_channels.append(webhook_channel)
    logger.info(f"Added webhook alert channel: {webhook_url}")
