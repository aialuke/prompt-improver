"""Health Alerting and Notification Service.

Provides comprehensive health alerting, threshold monitoring, and alert
escalation management. This service focuses on health assessment, issue
identification, and recommendation generation.

Features:
- Health score calculation based on comprehensive metrics
- Issue identification with severity classification
- Actionable recommendation generation
- Threshold monitoring and alerting
- Alert escalation and management
- Customizable alerting rules and conditions
"""

from datetime import UTC, datetime
from typing import Any

from prompt_improver.core.common import get_logger
from prompt_improver.database.health.services.health_types import (
    HealthAlert,
    HealthThreshold,
)

logger = get_logger(__name__)


class AlertingService:
    """Service for database health alerting and notification management.

    This service provides comprehensive health assessment, issue identification,
    and recommendation generation with configurable thresholds and alerting.
    """

    def __init__(self) -> None:
        """Initialize the alerting service with default thresholds."""
        # Default threshold configurations
        self.thresholds = {
            "connection_pool_utilization": HealthThreshold(
                metric_name="connection_pool_utilization",
                warning_threshold=80.0,
                critical_threshold=95.0,
                comparison_operator="greater_than",
                description="Connection pool utilization percentage"
            ),
            "slow_queries_count": HealthThreshold(
                metric_name="slow_queries_count",
                warning_threshold=5.0,
                critical_threshold=10.0,
                comparison_operator="greater_than",
                description="Number of slow queries detected"
            ),
            "cache_hit_ratio": HealthThreshold(
                metric_name="cache_hit_ratio",
                warning_threshold=95.0,
                critical_threshold=90.0,
                comparison_operator="less_than",
                description="Database cache hit ratio percentage"
            ),
            "replication_lag_seconds": HealthThreshold(
                metric_name="replication_lag_seconds",
                warning_threshold=60.0,
                critical_threshold=300.0,
                comparison_operator="greater_than",
                description="Replication lag in seconds"
            ),
            "rollback_ratio_percent": HealthThreshold(
                metric_name="rollback_ratio_percent",
                warning_threshold=10.0,
                critical_threshold=20.0,
                comparison_operator="greater_than",
                description="Transaction rollback ratio percentage"
            ),
            "blocking_locks": HealthThreshold(
                metric_name="blocking_locks",
                warning_threshold=1.0,
                critical_threshold=5.0,
                comparison_operator="greater_than",
                description="Number of blocking locks"
            ),
        }

        # Alert history for deduplication and tracking
        self._alert_history: list[HealthAlert] = []
        self._max_alert_history = 1000

    def calculate_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate overall health score based on all metrics (0-100).

        Args:
            metrics: Comprehensive database health metrics

        Returns:
            Health score between 0 and 100
        """
        try:
            score = 100.0

            # Connection pool health (25% weight)
            pool_metrics = metrics.get("connection_pool", {})
            if isinstance(pool_metrics, dict) and "utilization_percent" in pool_metrics:
                utilization = pool_metrics["utilization_percent"]
                if utilization > 95:
                    score -= 20  # Critical impact
                elif utilization > 80:
                    score -= 10  # Warning impact

            # Query performance health (30% weight)
            query_metrics = metrics.get("query_performance", {})
            if isinstance(query_metrics, dict):
                slow_queries = query_metrics.get("slow_queries_count", 0)
                if slow_queries > 10:
                    score -= 25  # Critical impact
                elif slow_queries > 5:
                    score -= 15  # Warning impact
                elif slow_queries > 0:
                    score -= min(10, slow_queries * 2)  # Gradual impact

            # Cache performance health (20% weight)
            cache_metrics = metrics.get("cache", {})
            if isinstance(cache_metrics, dict):
                hit_ratio = cache_metrics.get("overall_cache_hit_ratio_percent", 100)
                if hit_ratio < 90:
                    score -= 20  # Critical impact
                elif hit_ratio < 95:
                    score -= 10  # Warning impact

            # Replication health (15% weight)
            replication_metrics = metrics.get("replication", {})
            if isinstance(replication_metrics, dict) and replication_metrics.get("replication_enabled"):
                lag_seconds = replication_metrics.get("lag_seconds", 0)
                if lag_seconds > 300:
                    score -= 15  # Critical impact
                elif lag_seconds > 60:
                    score -= 8   # Warning impact

            # Lock health (5% weight)
            lock_metrics = metrics.get("locks", {})
            if isinstance(lock_metrics, dict):
                blocking_locks = lock_metrics.get("blocking_locks", 0)
                long_running_locks = lock_metrics.get("long_running_locks", 0)
                score -= min(5, (blocking_locks + long_running_locks) * 1)

            # Transaction health (5% weight)
            txn_metrics = metrics.get("transactions", {})
            if isinstance(txn_metrics, dict):
                rollback_ratio = txn_metrics.get("rollback_ratio_percent", 0)
                if rollback_ratio > 20:
                    score -= 5   # Critical impact
                elif rollback_ratio > 10:
                    score -= 3   # Warning impact

            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.exception(f"Failed to calculate health score: {e}")
            return 0.0

    def identify_health_issues(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify specific health issues based on metrics.

        Args:
            metrics: Comprehensive database health metrics

        Returns:
            List of identified health issues with severity and details
        """
        issues = []
        timestamp = datetime.now(UTC).isoformat()

        try:
            # Connection pool issues
            pool_metrics = metrics.get("connection_pool", {})
            if isinstance(pool_metrics, dict):
                utilization = pool_metrics.get("utilization_percent", 0)
                threshold = self.thresholds["connection_pool_utilization"]

                if utilization > threshold.critical_threshold:
                    issues.append({
                        "severity": "critical",
                        "category": "connection_pool",
                        "message": f"Connection pool utilization critically high: {utilization:.1f}%",
                        "metric_value": utilization,
                        "threshold": threshold.critical_threshold,
                        "timestamp": timestamp,
                    })
                elif utilization > threshold.warning_threshold:
                    issues.append({
                        "severity": "warning",
                        "category": "connection_pool",
                        "message": f"Connection pool utilization high: {utilization:.1f}%",
                        "metric_value": utilization,
                        "threshold": threshold.warning_threshold,
                        "timestamp": timestamp,
                    })

            # Query performance issues
            query_metrics = metrics.get("query_performance", {})
            if isinstance(query_metrics, dict):
                slow_queries = query_metrics.get("slow_queries_count", 0)
                threshold = self.thresholds["slow_queries_count"]

                if slow_queries > threshold.critical_threshold:
                    issues.append({
                        "severity": "critical",
                        "category": "query_performance",
                        "message": f"High number of slow queries: {slow_queries}",
                        "metric_value": slow_queries,
                        "threshold": threshold.critical_threshold,
                        "timestamp": timestamp,
                    })
                elif slow_queries > threshold.warning_threshold:
                    issues.append({
                        "severity": "warning",
                        "category": "query_performance",
                        "message": f"Slow queries detected: {slow_queries}",
                        "metric_value": slow_queries,
                        "threshold": threshold.warning_threshold,
                        "timestamp": timestamp,
                    })

            # Cache performance issues
            cache_metrics = metrics.get("cache", {})
            if isinstance(cache_metrics, dict):
                hit_ratio = cache_metrics.get("overall_cache_hit_ratio_percent", 100)
                threshold = self.thresholds["cache_hit_ratio"]

                if hit_ratio < threshold.critical_threshold:
                    issues.append({
                        "severity": "critical",
                        "category": "cache_performance",
                        "message": f"Cache hit ratio critically low: {hit_ratio:.1f}%",
                        "metric_value": hit_ratio,
                        "threshold": threshold.critical_threshold,
                        "timestamp": timestamp,
                    })
                elif hit_ratio < threshold.warning_threshold:
                    issues.append({
                        "severity": "warning",
                        "category": "cache_performance",
                        "message": f"Cache hit ratio below optimal: {hit_ratio:.1f}%",
                        "metric_value": hit_ratio,
                        "threshold": threshold.warning_threshold,
                        "timestamp": timestamp,
                    })

            # Replication issues
            replication_metrics = metrics.get("replication", {})
            if isinstance(replication_metrics, dict) and replication_metrics.get("replication_enabled"):
                lag_seconds = replication_metrics.get("lag_seconds", 0)
                threshold = self.thresholds["replication_lag_seconds"]

                if lag_seconds > threshold.critical_threshold:
                    issues.append({
                        "severity": "critical",
                        "category": "replication",
                        "message": f"Replication lag critically high: {lag_seconds:.1f} seconds",
                        "metric_value": lag_seconds,
                        "threshold": threshold.critical_threshold,
                        "timestamp": timestamp,
                    })
                elif lag_seconds > threshold.warning_threshold:
                    issues.append({
                        "severity": "warning",
                        "category": "replication",
                        "message": f"Replication lag elevated: {lag_seconds:.1f} seconds",
                        "metric_value": lag_seconds,
                        "threshold": threshold.warning_threshold,
                        "timestamp": timestamp,
                    })

            # Lock issues
            lock_metrics = metrics.get("locks", {})
            if isinstance(lock_metrics, dict):
                blocking_locks = lock_metrics.get("blocking_locks", 0)
                threshold = self.thresholds["blocking_locks"]

                if blocking_locks > threshold.critical_threshold:
                    issues.append({
                        "severity": "critical",
                        "category": "locks",
                        "message": f"High number of blocking locks: {blocking_locks}",
                        "metric_value": blocking_locks,
                        "threshold": threshold.critical_threshold,
                        "timestamp": timestamp,
                    })
                elif blocking_locks > threshold.warning_threshold:
                    issues.append({
                        "severity": "warning",
                        "category": "locks",
                        "message": f"Blocking locks detected: {blocking_locks}",
                        "metric_value": blocking_locks,
                        "threshold": threshold.warning_threshold,
                        "timestamp": timestamp,
                    })

            # Transaction issues
            txn_metrics = metrics.get("transactions", {})
            if isinstance(txn_metrics, dict):
                rollback_ratio = txn_metrics.get("rollback_ratio_percent", 0)
                threshold = self.thresholds["rollback_ratio_percent"]

                if rollback_ratio > threshold.critical_threshold:
                    issues.append({
                        "severity": "critical",
                        "category": "transactions",
                        "message": f"High transaction rollback ratio: {rollback_ratio:.1f}%",
                        "metric_value": rollback_ratio,
                        "threshold": threshold.critical_threshold,
                        "timestamp": timestamp,
                    })
                elif rollback_ratio > threshold.warning_threshold:
                    issues.append({
                        "severity": "warning",
                        "category": "transactions",
                        "message": f"Elevated transaction rollback ratio: {rollback_ratio:.1f}%",
                        "metric_value": rollback_ratio,
                        "threshold": threshold.warning_threshold,
                        "timestamp": timestamp,
                    })

                # Long-running transaction issues
                long_txns = len(txn_metrics.get("long_running_transactions", []))
                if long_txns > 0:
                    issues.append({
                        "severity": "warning",
                        "category": "transactions",
                        "message": f"Long-running transactions detected: {long_txns}",
                        "metric_value": long_txns,
                        "threshold": 0,
                        "timestamp": timestamp,
                    })

        except Exception as e:
            logger.exception(f"Failed to identify health issues: {e}")
            issues.append({
                "severity": "error",
                "category": "monitoring",
                "message": f"Health issue identification failed: {e}",
                "metric_value": 0,
                "threshold": 0,
                "timestamp": timestamp,
            })

        return issues

    def generate_recommendations(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate actionable recommendations based on metrics and identified issues.

        Args:
            metrics: Comprehensive database health metrics

        Returns:
            List of actionable recommendations with priority and expected impact
        """
        recommendations = []

        try:
            # Connection pool recommendations
            pool_metrics = metrics.get("connection_pool", {})
            if isinstance(pool_metrics, dict):
                utilization = pool_metrics.get("utilization_percent", 0)
                if utilization > 90:
                    recommendations.append({
                        "category": "connection_pool",
                        "priority": "critical",
                        "action": "increase_pool_size",
                        "description": f"Increase connection pool size (current utilization: {utilization:.1f}%)",
                        "expected_impact": "Reduced connection wait times and improved throughput",
                    })
                elif utilization > 80:
                    recommendations.append({
                        "category": "connection_pool",
                        "priority": "high",
                        "action": "monitor_pool_usage",
                        "description": f"Monitor connection pool usage closely (utilization: {utilization:.1f}%)",
                        "expected_impact": "Prevent connection pool exhaustion",
                    })

                waiting_requests = pool_metrics.get("waiting_requests", 0)
                if waiting_requests > 0:
                    recommendations.append({
                        "category": "connection_pool",
                        "priority": "high",
                        "action": "optimize_connection_usage",
                        "description": f"Optimize connection usage - {waiting_requests} requests waiting",
                        "expected_impact": "Reduced connection contention and wait times",
                    })

            # Query performance recommendations
            query_metrics = metrics.get("query_performance", {})
            if isinstance(query_metrics, dict):
                slow_queries = query_metrics.get("slow_queries_count", 0)
                if slow_queries > 10:
                    recommendations.append({
                        "category": "query_performance",
                        "priority": "critical",
                        "action": "optimize_slow_queries",
                        "description": f"Urgent: Optimize {slow_queries} slow queries",
                        "expected_impact": "Significant improvement in response times",
                    })
                elif slow_queries > 0:
                    recommendations.append({
                        "category": "query_performance",
                        "priority": "high",
                        "action": "analyze_slow_queries",
                        "description": f"Analyze and optimize {slow_queries} slow queries",
                        "expected_impact": "Improved query performance and reduced resource usage",
                    })

                missing_indexes = query_metrics.get("missing_indexes_count", 0)
                if missing_indexes > 0:
                    recommendations.append({
                        "category": "indexing",
                        "priority": "medium",
                        "action": "add_missing_indexes",
                        "description": f"Add {missing_indexes} missing indexes for frequently queried columns",
                        "expected_impact": "Faster query execution and reduced I/O",
                    })

            # Cache performance recommendations
            cache_metrics = metrics.get("cache", {})
            if isinstance(cache_metrics, dict):
                hit_ratio = cache_metrics.get("overall_cache_hit_ratio_percent", 100)
                if hit_ratio < 90:
                    recommendations.append({
                        "category": "cache_tuning",
                        "priority": "critical",
                        "action": "increase_shared_buffers",
                        "description": f"Increase shared_buffers - cache hit ratio critically low: {hit_ratio:.1f}%",
                        "expected_impact": "Significantly improved cache performance and reduced I/O",
                    })
                elif hit_ratio < 95:
                    recommendations.append({
                        "category": "cache_tuning",
                        "priority": "high",
                        "action": "tune_cache_settings",
                        "description": f"Tune cache settings - hit ratio below optimal: {hit_ratio:.1f}%",
                        "expected_impact": "Improved cache efficiency and query performance",
                    })

            # Storage recommendations
            storage_metrics = metrics.get("storage", {})
            if isinstance(storage_metrics, dict) and "bloat_metrics" in storage_metrics:
                bloat_info = storage_metrics["bloat_metrics"]
                if isinstance(bloat_info, dict):
                    bloated_tables = bloat_info.get("bloated_tables_count", 0)
                    if bloated_tables > 5:
                        recommendations.append({
                            "category": "maintenance",
                            "priority": "high",
                            "action": "vacuum_analyze",
                            "description": f"Run VACUUM ANALYZE on {bloated_tables} bloated tables",
                            "expected_impact": "Reclaimed storage space and improved query performance",
                        })
                    elif bloated_tables > 0:
                        recommendations.append({
                            "category": "maintenance",
                            "priority": "medium",
                            "action": "schedule_maintenance",
                            "description": f"Schedule maintenance for {bloated_tables} tables with bloat",
                            "expected_impact": "Preventive maintenance to avoid performance degradation",
                        })

            # Replication recommendations
            replication_metrics = metrics.get("replication", {})
            if isinstance(replication_metrics, dict) and replication_metrics.get("replication_enabled"):
                lag_seconds = replication_metrics.get("lag_seconds", 0)
                if lag_seconds > 300:
                    recommendations.append({
                        "category": "replication",
                        "priority": "critical",
                        "action": "investigate_replication_lag",
                        "description": f"Investigate critical replication lag: {lag_seconds:.1f} seconds",
                        "expected_impact": "Restored replication performance and data consistency",
                    })
                elif lag_seconds > 60:
                    recommendations.append({
                        "category": "replication",
                        "priority": "high",
                        "action": "monitor_replication",
                        "description": f"Monitor replication lag closely: {lag_seconds:.1f} seconds",
                        "expected_impact": "Prevented replication issues and maintained consistency",
                    })

            # Add timestamp to all recommendations
            timestamp = datetime.now(UTC).isoformat()
            for rec in recommendations:
                rec["timestamp"] = timestamp

        except Exception as e:
            logger.exception(f"Failed to generate recommendations: {e}")
            recommendations.append({
                "category": "monitoring",
                "priority": "error",
                "action": "fix_monitoring",
                "description": f"Fix recommendation generation system: {e}",
                "expected_impact": "Restored health monitoring capabilities",
                "timestamp": datetime.now(UTC).isoformat(),
            })

        return recommendations

    def check_thresholds(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Check if any metrics exceed defined thresholds.

        Args:
            metrics: Metrics to check against thresholds

        Returns:
            List of threshold violations
        """
        violations = []
        timestamp = datetime.now(UTC).isoformat()

        try:
            for metric_name, threshold in self.thresholds.items():
                if not threshold.enabled:
                    continue

                # Extract metric value based on metric name
                metric_value = self._extract_metric_value(metrics, metric_name)
                if metric_value is None:
                    continue

                # Check threshold violation
                is_violation = False
                severity = "info"

                if threshold.comparison_operator == "greater_than":
                    if metric_value > threshold.critical_threshold:
                        is_violation = True
                        severity = "critical"
                    elif metric_value > threshold.warning_threshold:
                        is_violation = True
                        severity = "warning"
                elif threshold.comparison_operator == "less_than":
                    if metric_value < threshold.critical_threshold:
                        is_violation = True
                        severity = "critical"
                    elif metric_value < threshold.warning_threshold:
                        is_violation = True
                        severity = "warning"

                if is_violation:
                    violations.append({
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "threshold_type": severity,
                        "threshold_value": (
                            threshold.critical_threshold if severity == "critical"
                            else threshold.warning_threshold
                        ),
                        "comparison_operator": threshold.comparison_operator,
                        "description": threshold.description,
                        "timestamp": timestamp,
                    })

        except Exception as e:
            logger.exception(f"Failed to check thresholds: {e}")
            violations.append({
                "metric_name": "system_error",
                "metric_value": 0,
                "threshold_type": "error",
                "threshold_value": 0,
                "comparison_operator": "error",
                "description": f"Threshold checking failed: {e}",
                "timestamp": timestamp,
            })

        return violations

    async def send_alert(self, alert: dict[str, Any]) -> bool:
        """Send alert notification.

        Args:
            alert: Alert information to send

        Returns:
            True if alert was sent successfully, False otherwise
        """
        try:
            # For now, just log the alert - in production this would integrate
            # with notification systems like email, Slack, PagerDuty, etc.
            logger.warning(
                f"HEALTH ALERT [{alert.get('severity', 'unknown').upper()}] "
                f"{alert.get('category', 'unknown')}: {alert.get('message', 'No message')}"
            )

            # Add to alert history for tracking
            alert_obj = HealthAlert(
                severity=alert.get("severity", "unknown"),
                category=alert.get("category", "unknown"),
                message=alert.get("message", ""),
                metric_name=alert.get("metric_name", ""),
                metric_value=alert.get("metric_value", 0.0),
                threshold=alert.get("threshold", 0.0),
                source_service="alerting_service",
                alert_id=f"{alert.get('category', 'unknown')}_{datetime.now(UTC).timestamp()}",
            )

            self._alert_history.append(alert_obj)

            # Limit alert history size
            if len(self._alert_history) > self._max_alert_history:
                self._alert_history = self._alert_history[-self._max_alert_history:]

            return True

        except Exception as e:
            logger.exception(f"Failed to send alert: {e}")
            return False

    def set_threshold(
        self, metric_name: str, warning_threshold: float, critical_threshold: float
    ) -> None:
        """Set threshold for a specific metric.

        Args:
            metric_name: Name of the metric
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
        """
        try:
            # Determine comparison operator based on metric type
            comparison_operator = "greater_than"
            if metric_name in {"cache_hit_ratio"}:
                comparison_operator = "less_than"

            self.thresholds[metric_name] = HealthThreshold(
                metric_name=metric_name,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                comparison_operator=comparison_operator,
                enabled=True,
                description=f"Custom threshold for {metric_name}"
            )

            logger.info(
                f"Updated threshold for {metric_name}: "
                f"warning={warning_threshold}, critical={critical_threshold}"
            )

        except Exception as e:
            logger.exception(f"Failed to set threshold for {metric_name}: {e}")

    def get_alert_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent alert history.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent alerts
        """
        try:
            cutoff_time = datetime.now(UTC).timestamp() - (hours * 3600)
            return [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "category": alert.category,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self._alert_history
                if alert.timestamp.timestamp() > cutoff_time
            ]
        except Exception as e:
            logger.exception(f"Failed to get alert history: {e}")
            return []

    def _extract_metric_value(self, metrics: dict[str, Any], metric_name: str) -> float | None:
        """Extract metric value from metrics dictionary based on metric name.

        Args:
            metrics: Metrics dictionary
            metric_name: Name of the metric to extract

        Returns:
            Metric value or None if not found
        """
        try:
            if metric_name == "connection_pool_utilization":
                pool_metrics = metrics.get("connection_pool", {})
                return pool_metrics.get("utilization_percent")

            if metric_name == "slow_queries_count":
                query_metrics = metrics.get("query_performance", {})
                return query_metrics.get("slow_queries_count", 0)

            if metric_name == "cache_hit_ratio":
                cache_metrics = metrics.get("cache", {})
                return cache_metrics.get("overall_cache_hit_ratio_percent")

            if metric_name == "replication_lag_seconds":
                replication_metrics = metrics.get("replication", {})
                return replication_metrics.get("lag_seconds")

            if metric_name == "rollback_ratio_percent":
                txn_metrics = metrics.get("transactions", {})
                return txn_metrics.get("rollback_ratio_percent")

            if metric_name == "blocking_locks":
                lock_metrics = metrics.get("locks", {})
                return lock_metrics.get("blocking_locks")

            return None

        except Exception as e:
            logger.exception(f"Failed to extract metric value for {metric_name}: {e}")
            return None
