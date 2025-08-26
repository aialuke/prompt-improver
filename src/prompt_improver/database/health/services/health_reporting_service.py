"""Health Reporting and Analytics Service.

Provides comprehensive health reporting, historical analysis, and report generation.
This service focuses on health data visualization, trend analysis, and comprehensive
reporting for database health monitoring.

Features:
- Health metrics history tracking and storage
- Trend analysis and pattern recognition
- Comprehensive health report generation
- Historical data analysis and insights
- Multiple export formats (JSON, CSV, etc.)
- Performance trend visualization data
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from prompt_improver.core.common import get_logger

logger = get_logger(__name__)


class HealthReportingService:
    """Service for database health reporting and historical analysis.

    This service provides comprehensive reporting capabilities for database health
    metrics, including trend analysis, historical comparisons, and data export.
    """

    def __init__(self) -> None:
        """Initialize the health reporting service."""
        # Metrics history storage
        self._metrics_history: list[dict[str, Any]] = []
        self._max_history_size = 1000  # Keep last 1000 data points

        # Trend analysis configuration
        self._trend_analysis_window = 10  # Points to consider for trend calculation
        self._trend_threshold = 0.05      # 5% change threshold for trend detection

    def add_metrics_to_history(self, metrics: dict[str, Any]) -> None:
        """Add metrics to history for trend analysis.

        Args:
            metrics: Health metrics to add to history
        """
        try:
            # Ensure timestamp is present
            if "timestamp" not in metrics:
                metrics["timestamp"] = datetime.now(UTC).isoformat()

            # Add to history
            self._metrics_history.append(metrics.copy())

            # Limit history size
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history = self._metrics_history[-self._max_history_size:]

            logger.debug(f"Added metrics to history. Total history size: {len(self._metrics_history)}")

        except Exception as e:
            logger.exception(f"Failed to add metrics to history: {e}")

    def get_health_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get health trends over the specified time period.

        Args:
            hours: Number of hours to analyze for trends

        Returns:
            Dictionary containing trend analysis results
        """
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

            # Filter recent metrics
            recent_metrics = [
                m for m in self._metrics_history
                if datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")) >= cutoff_time
            ]

            if len(recent_metrics) < 2:
                return {
                    "status": "insufficient_data",
                    "message": f"Need at least 2 data points in last {hours} hours",
                    "data_points_found": len(recent_metrics),
                    "hours_requested": hours,
                }

            # Extract trend data for key metrics
            health_scores = self._extract_metric_values(recent_metrics, "health_score")
            connection_utilizations = self._extract_nested_metric_values(
                recent_metrics, "connection_pool", "utilization_percent"
            )
            cache_hit_ratios = self._extract_nested_metric_values(
                recent_metrics, "cache", "overall_cache_hit_ratio_percent"
            )
            slow_query_counts = self._extract_nested_metric_values(
                recent_metrics, "query_performance", "slow_queries_count"
            )

            # Calculate trends
            trends = {
                "health_score": self._calculate_trend(health_scores),
                "connection_utilization": self._calculate_trend(connection_utilizations),
                "cache_hit_ratio": self._calculate_trend(cache_hit_ratios),
                "slow_query_count": self._calculate_trend(slow_query_counts),
            }

            # Generate trend summary
            trend_summary = self._generate_trend_summary(recent_metrics)

            return {
                "status": "success",
                "period_hours": hours,
                "data_points": len(recent_metrics),
                "time_range": {
                    "start": recent_metrics[0]["timestamp"] if recent_metrics else None,
                    "end": recent_metrics[-1]["timestamp"] if recent_metrics else None,
                },
                "trends": trends,
                "current_values": {
                    "health_score": health_scores[-1] if health_scores else None,
                    "connection_utilization": connection_utilizations[-1] if connection_utilizations else None,
                    "cache_hit_ratio": cache_hit_ratios[-1] if cache_hit_ratios else None,
                    "slow_query_count": slow_query_counts[-1] if slow_query_counts else None,
                },
                "average_values": {
                    "health_score": sum(health_scores) / len(health_scores) if health_scores else None,
                    "connection_utilization": sum(connection_utilizations) / len(connection_utilizations) if connection_utilizations else None,
                    "cache_hit_ratio": sum(cache_hit_ratios) / len(cache_hit_ratios) if cache_hit_ratios else None,
                    "slow_query_count": sum(slow_query_counts) / len(slow_query_counts) if slow_query_counts else None,
                },
                "summary": trend_summary,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Failed to get health trends: {e}")
            return {
                "status": "error",
                "error": str(e),
                "hours_requested": hours,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
            }

    def generate_health_report(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive health report.

        Args:
            metrics: Current health metrics

        Returns:
            Comprehensive health report with analysis and recommendations
        """
        try:
            report_timestamp = datetime.now(UTC)

            # Extract key metrics for reporting
            health_score = metrics.get("health_score", 0)
            issues = metrics.get("issues", [])
            recommendations = metrics.get("recommendations", [])

            # Categorize issues by severity
            critical_issues = [issue for issue in issues if issue.get("severity") == "critical"]
            warning_issues = [issue for issue in issues if issue.get("severity") == "warning"]
            info_issues = [issue for issue in issues if issue.get("severity") == "info"]

            # Categorize recommendations by priority
            critical_recommendations = [rec for rec in recommendations if rec.get("priority") == "critical"]
            high_recommendations = [rec for rec in recommendations if rec.get("priority") == "high"]
            medium_recommendations = [rec for rec in recommendations if rec.get("priority") == "medium"]

            # Determine overall health status
            if health_score >= 90:
                overall_status = "excellent"
                status_description = "Database health is excellent with minimal issues."
            elif health_score >= 75:
                overall_status = "good"
                status_description = "Database health is good with some minor issues to monitor."
            elif health_score >= 50:
                overall_status = "fair"
                status_description = "Database health is fair with several issues requiring attention."
            elif health_score >= 25:
                overall_status = "poor"
                status_description = "Database health is poor with significant issues requiring immediate attention."
            else:
                overall_status = "critical"
                status_description = "Database health is critical with severe issues requiring urgent intervention."

            # Component health summary
            component_health = self._assess_component_health(metrics)

            # Performance summary
            performance_summary = self._generate_performance_summary(metrics)

            # Resource utilization summary
            resource_summary = self._generate_resource_summary(metrics)

            # Historical context if available
            historical_context = self._get_historical_context()

            return {
                "report_metadata": {
                    "generated_at": report_timestamp.isoformat(),
                    "report_type": "comprehensive_health_report",
                    "version": "2025.1.0",
                    "data_timestamp": metrics.get("timestamp", report_timestamp.isoformat()),
                },
                "executive_summary": {
                    "overall_health_score": health_score,
                    "overall_status": overall_status,
                    "status_description": status_description,
                    "total_issues": len(issues),
                    "critical_issues_count": len(critical_issues),
                    "warning_issues_count": len(warning_issues),
                    "total_recommendations": len(recommendations),
                    "urgent_actions_required": len(critical_recommendations) + len(high_recommendations),
                },
                "component_health": component_health,
                "performance_analysis": performance_summary,
                "resource_utilization": resource_summary,
                "issues_analysis": {
                    "critical_issues": critical_issues,
                    "warning_issues": warning_issues,
                    "info_issues": info_issues,
                    "issue_categories": self._categorize_issues(issues),
                },
                "recommendations": {
                    "critical_priority": critical_recommendations,
                    "high_priority": high_recommendations,
                    "medium_priority": medium_recommendations,
                    "recommendation_categories": self._categorize_recommendations(recommendations),
                },
                "historical_context": historical_context,
                "raw_metrics": {
                    "connection_pool": metrics.get("connection_pool", {}),
                    "query_performance": metrics.get("query_performance", {}),
                    "cache_metrics": metrics.get("cache", {}),
                    "storage_metrics": metrics.get("storage", {}),
                    "replication_metrics": metrics.get("replication", {}),
                    "lock_metrics": metrics.get("locks", {}),
                    "transaction_metrics": metrics.get("transactions", {}),
                },
            }

        except Exception as e:
            logger.exception(f"Failed to generate health report: {e}")
            return {
                "report_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "report_type": "error_report",
                    "error": str(e),
                },
                "executive_summary": {
                    "overall_health_score": 0,
                    "overall_status": "error",
                    "status_description": f"Failed to generate report: {e}",
                },
            }

    def generate_trend_summary(self, recent_metrics: list[dict[str, Any]]) -> str:
        """Generate a human-readable trend summary.

        Args:
            recent_metrics: List of recent health metrics

        Returns:
            Human-readable trend summary string
        """
        try:
            if not recent_metrics:
                return "No data available for trend analysis."

            latest = recent_metrics[-1]
            health_score = latest.get("health_score", 0)
            issues_count = len(latest.get("issues", []))

            # Determine health status
            if health_score >= 90:
                health_status = "excellent"
            elif health_score >= 75:
                health_status = "good"
            elif health_score >= 50:
                health_status = "fair"
            else:
                health_status = "poor"

            # Trend direction analysis
            if len(recent_metrics) >= 2:
                previous_score = recent_metrics[-2].get("health_score", health_score)
                if health_score > previous_score + 5:
                    trend_direction = "improving"
                elif health_score < previous_score - 5:
                    trend_direction = "degrading"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"

            summary = (
                f"Database health is {health_status} (score: {health_score:.1f}/100) "
                f"with {issues_count} active issues. "
                f"Trend: {trend_direction} over recent monitoring period."
            )

            # Add component-specific insights
            latest_connection = latest.get("connection_pool", {})
            if isinstance(latest_connection, dict):
                utilization = latest_connection.get("utilization_percent", 0)
                if utilization > 90:
                    summary += f" Connection pool utilization is high ({utilization:.1f}%)."

            latest_cache = latest.get("cache", {})
            if isinstance(latest_cache, dict):
                hit_ratio = latest_cache.get("overall_cache_hit_ratio_percent", 100)
                if hit_ratio < 95:
                    summary += f" Cache hit ratio needs attention ({hit_ratio:.1f}%)."

            return summary

        except Exception as e:
            logger.exception(f"Failed to generate trend summary: {e}")
            return f"Failed to generate trend summary: {e}"

    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format.

        Args:
            format_type: Export format ("json", "csv", or "summary")

        Returns:
            Exported metrics data as string
        """
        try:
            if format_type.lower() == "json":
                return json.dumps(self._metrics_history, indent=2, default=str)

            if format_type.lower() == "csv":
                return self._export_to_csv()

            if format_type.lower() == "summary":
                return self._export_summary()

            raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            logger.exception(f"Failed to export metrics in format {format_type}: {e}")
            return f"Export failed: {e}"

    def get_metrics_history(self) -> list[dict[str, Any]]:
        """Get historical metrics data.

        Returns:
            List of historical metrics
        """
        return self._metrics_history.copy()

    def clear_history(self) -> None:
        """Clear metrics history (for testing or reset purposes)."""
        self._metrics_history.clear()
        logger.info("Metrics history cleared")

    def _extract_metric_values(self, metrics_list: list[dict[str, Any]], key: str) -> list[float]:
        """Extract values for a specific metric key from metrics list."""
        values = []
        for metrics in metrics_list:
            value = metrics.get(key)
            if value is not None and isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _extract_nested_metric_values(
        self, metrics_list: list[dict[str, Any]], parent_key: str, child_key: str
    ) -> list[float]:
        """Extract values for a nested metric key from metrics list."""
        values = []
        for metrics in metrics_list:
            parent_dict = metrics.get(parent_key, {})
            if isinstance(parent_dict, dict):
                value = parent_dict.get(child_key)
                if value is not None and isinstance(value, (int, float)):
                    values.append(float(value))
        return values

    def _calculate_trend(self, values: list[float]) -> dict[str, Any]:
        """Calculate trend information for a series of values."""
        if len(values) < 2:
            return {
                "direction": "unknown",
                "change_percent": 0.0,
                "confidence": "low",
                "data_points": len(values),
            }

        # Calculate trend using simple linear approach
        window_size = min(self._trend_analysis_window, len(values))
        recent_values = values[-window_size:]

        # Compare recent average to earlier average
        mid_point = len(recent_values) // 2
        if mid_point == 0:
            mid_point = 1

        earlier_avg = sum(recent_values[:mid_point]) / mid_point
        recent_avg = sum(recent_values[mid_point:]) / (len(recent_values) - mid_point)

        if earlier_avg == 0:
            change_percent = 0.0
        else:
            change_percent = ((recent_avg - earlier_avg) / earlier_avg) * 100

        # Determine trend direction
        if abs(change_percent) < self._trend_threshold * 100:
            direction = "stable"
        elif change_percent > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Confidence based on data points and consistency
        confidence = "high" if len(values) >= 10 else "medium" if len(values) >= 5 else "low"

        return {
            "direction": direction,
            "change_percent": round(change_percent, 2),
            "confidence": confidence,
            "data_points": len(values),
            "current_value": values[-1] if values else None,
            "average_value": sum(values) / len(values) if values else None,
        }

    def _assess_component_health(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Assess health of individual components."""
        component_health = {}

        # Connection pool health
        pool_metrics = metrics.get("connection_pool", {})
        if isinstance(pool_metrics, dict):
            utilization = pool_metrics.get("utilization_percent", 0)
            efficiency = pool_metrics.get("pool_efficiency_score", 100)

            if utilization > 95 or efficiency < 50:
                status = "critical"
            elif utilization > 80 or efficiency < 70:
                status = "warning"
            else:
                status = "healthy"

            component_health["connection_pool"] = {
                "status": status,
                "utilization_percent": utilization,
                "efficiency_score": efficiency,
            }

        # Query performance health
        query_metrics = metrics.get("query_performance", {})
        if isinstance(query_metrics, dict):
            slow_queries = query_metrics.get("slow_queries_count", 0)
            assessment = query_metrics.get("overall_assessment", "unknown")

            if slow_queries > 10 or assessment == "poor":
                status = "critical"
            elif slow_queries > 5 or assessment == "moderate":
                status = "warning"
            else:
                status = "healthy"

            component_health["query_performance"] = {
                "status": status,
                "slow_queries_count": slow_queries,
                "assessment": assessment,
            }

        # Cache health
        cache_metrics = metrics.get("cache", {})
        if isinstance(cache_metrics, dict):
            hit_ratio = cache_metrics.get("overall_cache_hit_ratio_percent", 100)

            if hit_ratio < 90:
                status = "critical"
            elif hit_ratio < 95:
                status = "warning"
            else:
                status = "healthy"

            component_health["cache"] = {
                "status": status,
                "hit_ratio_percent": hit_ratio,
            }

        return component_health

    def _generate_performance_summary(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate performance analysis summary."""
        query_metrics = metrics.get("query_performance", {})
        if not isinstance(query_metrics, dict):
            return {"status": "no_data"}

        return {
            "slow_queries_count": query_metrics.get("slow_queries_count", 0),
            "frequent_queries_count": len(query_metrics.get("frequent_queries", [])),
            "io_intensive_queries_count": len(query_metrics.get("io_intensive_queries", [])),
            "pg_stat_statements_available": query_metrics.get("pg_stat_statements_available", False),
            "overall_assessment": query_metrics.get("overall_assessment", "unknown"),
            "performance_summary": query_metrics.get("performance_summary", ""),
        }

    def _generate_resource_summary(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate resource utilization summary."""
        summary = {}

        # Connection pool resources
        pool_metrics = metrics.get("connection_pool", {})
        if isinstance(pool_metrics, dict):
            summary["connection_pool"] = {
                "utilization_percent": pool_metrics.get("utilization_percent", 0),
                "total_connections": pool_metrics.get("total_connections", 0),
                "active_connections": pool_metrics.get("active_connections", 0),
            }

        # Cache resources
        cache_metrics = metrics.get("cache", {})
        if isinstance(cache_metrics, dict):
            summary["cache"] = {
                "hit_ratio_percent": cache_metrics.get("overall_cache_hit_ratio_percent", 100),
                "efficiency": cache_metrics.get("cache_efficiency", "unknown"),
            }

        # Storage resources
        storage_metrics = metrics.get("storage", {})
        if isinstance(storage_metrics, dict):
            summary["storage"] = {
                "database_size_pretty": storage_metrics.get("database_size_pretty", "unknown"),
                "index_to_table_ratio": storage_metrics.get("index_to_table_ratio", 0),
            }

        return summary

    def _get_historical_context(self) -> dict[str, Any]:
        """Get historical context for the current metrics."""
        if len(self._metrics_history) < 2:
            return {"status": "insufficient_data"}

        # Compare with metrics from 24 hours ago (or as far back as we have data)
        target_time = datetime.now(UTC) - timedelta(hours=24)
        historical_metrics = None

        for metrics in self._metrics_history:
            metrics_time = datetime.fromisoformat(metrics["timestamp"].replace("Z", "+00:00"))
            if metrics_time <= target_time:
                historical_metrics = metrics

        if not historical_metrics:
            historical_metrics = self._metrics_history[0]

        current_metrics = self._metrics_history[-1]

        # Compare key metrics
        current_score = current_metrics.get("health_score", 0)
        historical_score = historical_metrics.get("health_score", 0)

        score_change = current_score - historical_score

        return {
            "status": "available",
            "comparison_period": "24_hours",
            "health_score_change": round(score_change, 2),
            "health_score_trend": "improving" if score_change > 5 else "degrading" if score_change < -5 else "stable",
            "historical_timestamp": historical_metrics["timestamp"],
            "current_timestamp": current_metrics["timestamp"],
        }

    def _categorize_issues(self, issues: list[dict[str, Any]]) -> dict[str, int]:
        """Categorize issues by category."""
        categories = {}
        for issue in issues:
            category = issue.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _categorize_recommendations(self, recommendations: list[dict[str, Any]]) -> dict[str, int]:
        """Categorize recommendations by category."""
        categories = {}
        for rec in recommendations:
            category = rec.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _export_to_csv(self) -> str:
        """Export metrics history to CSV format."""
        if not self._metrics_history:
            return "No data available for CSV export"

        # Simple CSV export with key metrics
        csv_lines = ["timestamp,health_score,connection_utilization,cache_hit_ratio,slow_queries"]

        for metrics in self._metrics_history:
            timestamp = metrics.get("timestamp", "")
            health_score = metrics.get("health_score", 0)

            connection_util = 0
            pool_metrics = metrics.get("connection_pool", {})
            if isinstance(pool_metrics, dict):
                connection_util = pool_metrics.get("utilization_percent", 0)

            cache_hit_ratio = 100
            cache_metrics = metrics.get("cache", {})
            if isinstance(cache_metrics, dict):
                cache_hit_ratio = cache_metrics.get("overall_cache_hit_ratio_percent", 100)

            slow_queries = 0
            query_metrics = metrics.get("query_performance", {})
            if isinstance(query_metrics, dict):
                slow_queries = query_metrics.get("slow_queries_count", 0)

            csv_lines.append(f"{timestamp},{health_score},{connection_util},{cache_hit_ratio},{slow_queries}")

        return "\n".join(csv_lines)

    def _export_summary(self) -> str:
        """Export a summary of metrics history."""
        if not self._metrics_history:
            return "No metrics data available"

        latest = self._metrics_history[-1]
        total_points = len(self._metrics_history)

        summary_lines = [
            "Database Health Metrics Summary",
            "=" * 35,
            f"Total data points: {total_points}",
            f"Latest timestamp: {latest.get('timestamp', 'unknown')}",
            f"Current health score: {latest.get('health_score', 0):.1f}/100",
            f"Data collection span: {self._get_data_span()}",
            "",
            "Recent Issues:",
        ]

        issues = latest.get("issues", [])
        if issues:
            # Show top 5 issues
            summary_lines.extend(f"- [{issue.get('severity', 'unknown').upper()}] {issue.get('message', 'No message')}" for issue in issues[:5])
        else:
            summary_lines.append("- No issues detected")

        summary_lines.append("")
        summary_lines.append("Recent Recommendations:")

        recommendations = latest.get("recommendations", [])
        if recommendations:
            # Show top 5 recommendations
            summary_lines.extend(f"- [{rec.get('priority', 'unknown').upper()}] {rec.get('description', 'No description')}" for rec in recommendations[:5])
        else:
            summary_lines.append("- No recommendations")

        return "\n".join(summary_lines)

    def _get_data_span(self) -> str:
        """Get the time span of collected data."""
        if len(self._metrics_history) < 2:
            return "Single data point"

        try:
            earliest = datetime.fromisoformat(self._metrics_history[0]["timestamp"].replace("Z", "+00:00"))
            latest = datetime.fromisoformat(self._metrics_history[-1]["timestamp"].replace("Z", "+00:00"))
            span = latest - earliest

            if span.days > 0:
                return f"{span.days} days, {span.seconds // 3600} hours"
            return f"{span.seconds // 3600} hours, {(span.seconds % 3600) // 60} minutes"
        except Exception:
            return "Unknown span"
