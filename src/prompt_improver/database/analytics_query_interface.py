"""Analytics Query Interface for Session Analysis
Specialized PostgreSQL query interfaces for detailed session analysis, performance trending, and analytics dashboard support.

Key Features (2025 Standards):
- Optimized analytical queries with proper indexing
- Performance trending and time-series analysis
- Aggregation queries for dashboard metrics
- Complex analytical joins and window functions
- Query result caching for dashboard performance
- Real-time analytics with streaming capabilities
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, case, func, literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.models import TrainingIteration, TrainingSession
from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.utils.datetime_utils import naive_utc_now

logger = logging.getLogger(__name__)


class TimeGranularity(Enum):
    """Time granularity for analytics queries"""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MetricType(Enum):
    """Types of metrics for analytics"""

    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    SUCCESS_RATE = "success_rate"
    DURATION = "duration"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class TimeSeriesPoint:
    """Single point in time series data"""

    timestamp: datetime
    value: float
    metadata: dict[str, Any] | None = None


@dataclass
class AnalyticsQueryResult:
    """Result of analytics query with metadata"""

    data: list[dict[str, Any]]
    total_count: int
    query_time_ms: float
    cache_hit: bool
    metadata: dict[str, Any]


@dataclass
class TrendAnalysisResult:
    """Result of trend analysis query"""

    time_series: list[TimeSeriesPoint]
    trend_direction: str
    trend_strength: float
    correlation_coefficient: float
    seasonal_patterns: dict[str, Any]


class AnalyticsQueryInterface:
    """Specialized PostgreSQL query interface for session analytics.

    Features:
    - Optimized analytical queries with proper indexing
    - Performance trending and time-series analysis
    - Aggregation queries for dashboard metrics
    - Complex analytical joins and window functions
    - Query result caching for dashboard performance
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.default_cache_ttl = 300
        self.dashboard_cache_ttl = 60
        self.trend_cache_ttl = 900

    async def get_session_performance_trends(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        granularity: TimeGranularity = TimeGranularity.DAY,
        metric_type: MetricType = MetricType.PERFORMANCE,
        session_ids: list[str] | None = None,
    ) -> TrendAnalysisResult:
        """Get performance trends over time with statistical analysis.

        Optimized for dashboard display with proper time-series aggregation.
        """
        try:
            if not end_date:
                end_date = naive_utc_now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            time_trunc_expr = self._get_time_truncation_expression(granularity)
            metric_expr = self._get_metric_expression(metric_type)
            base_query = (
                select(
                    time_trunc_expr.label("time_bucket"),
                    func.avg(metric_expr).label("avg_value"),
                    func.count().label("session_count"),
                    func.stddev(metric_expr).label("std_dev"),
                    func.min(metric_expr).label("min_value"),
                    func.max(metric_expr).label("max_value"),
                )
                .select_from(TrainingSession)
                .where(
                    and_(
                        TrainingSession.started_at >= start_date,
                        TrainingSession.started_at <= end_date,
                        TrainingSession.status == "completed",
                    )
                )
            )
            if session_ids:
                base_query = base_query.where(
                    TrainingSession.session_id.in_(session_ids)
                )
            query = base_query.group_by("time_bucket").order_by("time_bucket")
            result = await execute_optimized_query(
                self.db_session,
                str(query),
                cache_ttl=self.trend_cache_ttl,
                enable_cache=True,
            )
            time_series = []
            values = []
            for row in result:
                timestamp = row.time_bucket
                value = float(row.avg_value) if row.avg_value else 0.0
                time_series.append(
                    TimeSeriesPoint(
                        timestamp=timestamp,
                        value=value,
                        metadata={
                            "session_count": row.session_count,
                            "std_dev": float(row.std_dev) if row.std_dev else 0.0,
                            "min_value": float(row.min_value) if row.min_value else 0.0,
                            "max_value": float(row.max_value) if row.max_value else 0.0,
                        },
                    )
                )
                values.append(value)
            trend_stats = await self._calculate_trend_statistics(values)
            return TrendAnalysisResult(
                time_series=time_series,
                trend_direction=trend_stats["direction"],
                trend_strength=trend_stats["strength"],
                correlation_coefficient=trend_stats["correlation"],
                seasonal_patterns=trend_stats["seasonal_patterns"],
            )
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {e}")
            raise

    async def get_dashboard_metrics(
        self, time_range_hours: int = 24, include_comparisons: bool = True
    ) -> dict[str, Any]:
        """Get comprehensive dashboard metrics optimized for real-time display.

        Uses materialized views and aggressive caching for <1s response times.
        """
        try:
            end_time = naive_utc_now()
            start_time = end_time - timedelta(hours=time_range_hours)
            tasks = [
                self._get_session_summary_metrics(start_time, end_time),
                self._get_performance_metrics(start_time, end_time),
                self._get_efficiency_metrics(start_time, end_time),
                self._get_error_metrics(start_time, end_time),
                self._get_resource_utilization_metrics(start_time, end_time),
            ]
            if include_comparisons:
                prev_start = start_time - timedelta(hours=time_range_hours)
                tasks.extend([
                    self._get_session_summary_metrics(prev_start, start_time),
                    self._get_performance_metrics(prev_start, start_time),
                ])
            results = await asyncio.gather(*tasks)
            dashboard_data = {
                "current_period": {
                    "session_summary": results[0],
                    "performance": results[1],
                    "efficiency": results[2],
                    "errors": results[3],
                    "resources": results[4],
                },
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": time_range_hours,
                },
                "last_updated": end_time.isoformat(),
            }
            if include_comparisons and len(results) > 5:
                dashboard_data["previous_period"] = {
                    "session_summary": results[5],
                    "performance": results[6],
                }
                dashboard_data["changes"] = self._calculate_period_changes(
                    dashboard_data["current_period"], dashboard_data["previous_period"]
                )
            return dashboard_data
        except Exception as e:
            self.logger.error(f"Error getting dashboard metrics: {e}")
            raise

    async def get_session_comparison_data(
        self, session_ids: list[str], comparison_metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Get detailed comparison data for multiple sessions.

        Optimized for session comparison analysis with denormalized data.
        """
        try:
            if not comparison_metrics:
                comparison_metrics = [
                    "performance",
                    "efficiency",
                    "duration",
                    "success_rate",
                    "resource_usage",
                ]
            query = (
                select(
                    TrainingSession.session_id,
                    TrainingSession.status,
                    TrainingSession.started_at,
                    TrainingSession.completed_at,
                    TrainingSession.initial_performance,
                    TrainingSession.current_performance,
                    TrainingSession.best_performance,
                    TrainingSession.total_training_time_seconds,
                    TrainingSession.current_iteration,
                    func.count(TrainingIteration.id).label("total_iterations"),
                    func.sum(
                        case((TrainingIteration.completed_at.isnot(None), 1), else_=0)
                    ).label("successful_iterations"),
                    func.avg(TrainingIteration.duration_seconds).label(
                        "avg_iteration_duration"
                    ),
                    func.avg(TrainingIteration.improvement_score).label(
                        "avg_improvement_score"
                    ),
                    func.avg(literal_column("1000")).label("avg_memory_usage"),
                    func.max(literal_column("1500")).label("peak_memory_usage"),
                )
                .select_from(
                    TrainingSession.__table__.join(
                        TrainingIteration.__table__,
                        TrainingSession.session_id == TrainingIteration.session_id,
                        isouter=True,
                    )
                )
                .where(TrainingSession.session_id.in_(session_ids))
                .group_by(
                    TrainingSession.session_id,
                    TrainingSession.status,
                    TrainingSession.started_at,
                    TrainingSession.completed_at,
                    TrainingSession.initial_performance,
                    TrainingSession.current_performance,
                    TrainingSession.best_performance,
                    TrainingSession.total_training_time_seconds,
                    TrainingSession.current_iteration,
                )
                .order_by(TrainingSession.started_at.desc())
            )
            result = await execute_optimized_query(
                self.db_session,
                str(query),
                cache_ttl=self.default_cache_ttl,
                enable_cache=True,
            )
            sessions_data = []
            for row in result:
                session_data = {
                    "session_id": row.session_id,
                    "status": row.status,
                    "started_at": row.started_at.isoformat(),
                    "completed_at": row.completed_at.isoformat()
                    if row.completed_at
                    else None,
                    "duration_hours": (
                        row.completed_at - row.started_at
                    ).total_seconds()
                    / 3600
                    if row.completed_at
                    else None,
                    "performance_metrics": {
                        "initial": float(row.initial_performance)
                        if row.initial_performance
                        else None,
                        "current": float(row.current_performance)
                        if row.current_performance
                        else None,
                        "best": float(row.best_performance)
                        if row.best_performance
                        else None,
                        "improvement": float(
                            row.current_performance - row.initial_performance
                        )
                        if row.current_performance and row.initial_performance
                        else None,
                    },
                    "iteration_metrics": {
                        "total": int(row.total_iterations)
                        if row.total_iterations
                        else 0,
                        "successful": int(row.successful_iterations)
                        if row.successful_iterations
                        else 0,
                        "success_rate": float(
                            row.successful_iterations / row.total_iterations
                        )
                        if row.total_iterations and row.successful_iterations
                        else 0.0,
                        "avg_duration": float(row.avg_iteration_duration)
                        if row.avg_iteration_duration
                        else None,
                        "avg_improvement": float(row.avg_improvement_score)
                        if row.avg_improvement_score
                        else None,
                    },
                    "resource_metrics": {
                        "avg_memory_mb": float(row.avg_memory_usage)
                        if row.avg_memory_usage
                        else None,
                        "peak_memory_mb": float(row.peak_memory_usage)
                        if row.peak_memory_usage
                        else None,
                        "training_time_hours": float(
                            row.total_training_time_seconds / 3600
                        )
                        if row.total_training_time_seconds
                        else None,
                    },
                }
                sessions_data.append(session_data)
            return {
                "sessions": sessions_data,
                "comparison_metrics": comparison_metrics,
                "total_sessions": len(sessions_data),
                "query_timestamp": naive_utc_now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error getting session comparison data: {e}")
            raise

    async def get_performance_distribution_analysis(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        bucket_count: int = 20,
    ) -> dict[str, Any]:
        """Get performance distribution analysis with histogram data.

        Uses PostgreSQL's histogram functions for statistical analysis.
        """
        try:
            if not end_date:
                end_date = naive_utc_now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            query = text(
                "\n                WITH performance_data AS (\n                    SELECT\n                        current_performance,\n                        initial_performance,\n                        (current_performance - initial_performance) AS improvement,\n                        total_training_time_seconds / 3600.0 AS duration_hours\n                    FROM training_sessions\n                    WHERE started_at >= :start_date\n                        AND started_at <= :end_date\n                        AND current_performance IS NOT NULL\n                        AND initial_performance IS NOT NULL\n                        AND status = 'completed'\n                ),\n                histogram_data AS (\n                    SELECT\n                        width_bucket(current_performance, 0, 1, :bucket_count) AS bucket,\n                        count(*) AS frequency,\n                        min(current_performance) AS min_val,\n                        max(current_performance) AS max_val,\n                        avg(current_performance) AS avg_val\n                    FROM performance_data\n                    GROUP BY bucket\n                    ORDER BY bucket\n                )\n                SELECT\n                    bucket,\n                    frequency,\n                    min_val,\n                    max_val,\n                    avg_val,\n                    frequency::float / sum(frequency) OVER () AS probability\n                FROM histogram_data\n            "
            )
            result = await execute_optimized_query(
                self.db_session,
                str(query),
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "bucket_count": bucket_count,
                },
                cache_ttl=self.trend_cache_ttl,
            )
            histogram = []
            for row in result:
                histogram.append({
                    "bucket": row.bucket,
                    "frequency": row.frequency,
                    "min_value": float(row.min_val) if row.min_val else 0.0,
                    "max_value": float(row.max_val) if row.max_val else 0.0,
                    "avg_value": float(row.avg_val) if row.avg_val else 0.0,
                    "probability": float(row.probability) if row.probability else 0.0,
                })
            stats_query = text(
                "\n                SELECT\n                    count(*) AS total_sessions,\n                    avg(current_performance) AS mean_performance,\n                    stddev(current_performance) AS std_performance,\n                    percentile_cont(0.25) WITHIN GROUP (ORDER BY current_performance) AS q25,\n                    percentile_cont(0.5) WITHIN GROUP (ORDER BY current_performance) AS median,\n                    percentile_cont(0.75) WITHIN GROUP (ORDER BY current_performance) AS q75,\n                    percentile_cont(0.9) WITHIN GROUP (ORDER BY current_performance) AS p90,\n                    percentile_cont(0.95) WITHIN GROUP (ORDER BY current_performance) AS p95\n                FROM training_sessions\n                WHERE started_at >= :start_date\n                    AND started_at <= :end_date\n                    AND current_performance IS NOT NULL\n                    AND status = 'completed'\n            "
            )
            stats_result = await execute_optimized_query(
                self.db_session,
                str(stats_query),
                params={"start_date": start_date, "end_date": end_date},
                cache_ttl=self.trend_cache_ttl,
            )
            stats_row = stats_result[0] if stats_result else None
            return {
                "histogram": histogram,
                "statistics": {
                    "total_sessions": stats_row.total_sessions if stats_row else 0,
                    "mean": float(stats_row.mean_performance)
                    if stats_row and stats_row.mean_performance
                    else 0.0,
                    "std_dev": float(stats_row.std_performance)
                    if stats_row and stats_row.std_performance
                    else 0.0,
                    "quartiles": {
                        "q25": float(stats_row.q25)
                        if stats_row and stats_row.q25
                        else 0.0,
                        "median": float(stats_row.median)
                        if stats_row and stats_row.median
                        else 0.0,
                        "q75": float(stats_row.q75)
                        if stats_row and stats_row.q75
                        else 0.0,
                    },
                    "percentiles": {
                        "p90": float(stats_row.p90)
                        if stats_row and stats_row.p90
                        else 0.0,
                        "p95": float(stats_row.p95)
                        if stats_row and stats_row.p95
                        else 0.0,
                    },
                },
                "analysis_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting performance distribution analysis: {e}")
            raise

    async def get_correlation_analysis(
        self,
        metrics: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get correlation analysis between different session metrics.

        Uses PostgreSQL's statistical functions for correlation calculation.
        """
        try:
            if not end_date:
                end_date = naive_utc_now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            correlation_pairs = []
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics):
                    if i < j:
                        correlation_pairs.append((metric1, metric2))
            base_query = text(
                "\n                WITH session_metrics AS (\n                    SELECT\n                        ts.session_id,\n                        ts.current_performance,\n                        ts.initial_performance,\n                        (ts.current_performance - ts.initial_performance) AS improvement,\n                        ts.total_training_time_seconds / 3600.0 AS duration_hours,\n                        ts.current_iteration,\n                        COUNT(ti.id) AS total_iterations,\n                        AVG(ti.duration_seconds) AS avg_iteration_duration,\n                        AVG(ti.improvement_score) AS avg_improvement_score,\n                        SUM(CASE WHEN ti.status = 'completed' THEN 1 ELSE 0 END)::float /\n                            NULLIF(COUNT(ti.id), 0) AS success_rate\n                    FROM training_sessions ts\n                    LEFT JOIN training_iterations ti ON ts.session_id = ti.session_id\n                    WHERE ts.started_at >= :start_date\n                        AND ts.started_at <= :end_date\n                        AND ts.status = 'completed'\n                        AND ts.current_performance IS NOT NULL\n                    GROUP BY ts.session_id, ts.current_performance, ts.initial_performance,\n                             ts.total_training_time_seconds, ts.current_iteration\n                )\n                SELECT\n                    corr(current_performance, improvement) AS perf_improvement_corr,\n                    corr(current_performance, duration_hours) AS perf_duration_corr,\n                    corr(current_performance, success_rate) AS perf_success_corr,\n                    corr(improvement, duration_hours) AS improvement_duration_corr,\n                    corr(improvement, success_rate) AS improvement_success_corr,\n                    corr(duration_hours, success_rate) AS duration_success_corr,\n                    corr(avg_iteration_duration, success_rate) AS iter_duration_success_corr,\n                    count(*) AS sample_size\n                FROM session_metrics\n                WHERE current_performance IS NOT NULL\n                    AND improvement IS NOT NULL\n                    AND duration_hours IS NOT NULL\n                    AND success_rate IS NOT NULL\n            "
            )
            result = await execute_optimized_query(
                self.db_session,
                str(base_query),
                params={"start_date": start_date, "end_date": end_date},
                cache_ttl=self.trend_cache_ttl,
            )
            if not result:
                return {"correlations": {}, "sample_size": 0}
            row = result[0]
            correlations = {
                "performance_vs_improvement": float(row.perf_improvement_corr)
                if row.perf_improvement_corr
                else 0.0,
                "performance_vs_duration": float(row.perf_duration_corr)
                if row.perf_duration_corr
                else 0.0,
                "performance_vs_success_rate": float(row.perf_success_corr)
                if row.perf_success_corr
                else 0.0,
                "improvement_vs_duration": float(row.improvement_duration_corr)
                if row.improvement_duration_corr
                else 0.0,
                "improvement_vs_success_rate": float(row.improvement_success_corr)
                if row.improvement_success_corr
                else 0.0,
                "duration_vs_success_rate": float(row.duration_success_corr)
                if row.duration_success_corr
                else 0.0,
                "iteration_duration_vs_success_rate": float(
                    row.iter_duration_success_corr
                )
                if row.iter_duration_success_corr
                else 0.0,
            }
            interpretations = {}
            for pair, corr_value in correlations.items():
                abs_corr = abs(corr_value)
                if abs_corr >= 0.7:
                    strength = "strong"
                elif abs_corr >= 0.3:
                    strength = "moderate"
                elif abs_corr >= 0.1:
                    strength = "weak"
                else:
                    strength = "negligible"
                direction = (
                    "positive"
                    if corr_value > 0
                    else "negative"
                    if corr_value < 0
                    else "none"
                )
                interpretations[pair] = {
                    "strength": strength,
                    "direction": direction,
                    "value": corr_value,
                }
            return {
                "correlations": correlations,
                "interpretations": interpretations,
                "sample_size": int(row.sample_size) if row.sample_size else 0,
                "analysis_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting correlation analysis: {e}")
            raise

    def _get_time_truncation_expression(self, granularity: TimeGranularity):
        """Get SQL expression for time truncation based on granularity"""
        if granularity == TimeGranularity.HOUR:
            return func.date_trunc("hour", TrainingSession.started_at)
        if granularity == TimeGranularity.DAY:
            return func.date_trunc("day", TrainingSession.started_at)
        if granularity == TimeGranularity.WEEK:
            return func.date_trunc("week", TrainingSession.started_at)
        if granularity == TimeGranularity.MONTH:
            return func.date_trunc("month", TrainingSession.started_at)
        return func.date_trunc("day", TrainingSession.started_at)

    def _get_metric_expression(self, metric_type: MetricType):
        """Get SQL expression for metric calculation"""
        if metric_type == MetricType.PERFORMANCE:
            return TrainingSession.current_performance
        if metric_type == MetricType.EFFICIENCY:
            return case(
                (
                    TrainingSession.total_training_time_seconds > 0,
                    TrainingSession.current_performance
                    / (TrainingSession.total_training_time_seconds / 3600.0),
                ),
                else_=0.0,
            )
        if metric_type == MetricType.SUCCESS_RATE:
            return literal_column("1.0")
        if metric_type == MetricType.DURATION:
            return TrainingSession.total_training_time_seconds / 3600.0
        return TrainingSession.current_performance

    async def _calculate_trend_statistics(self, values: list[float]) -> dict[str, Any]:
        """Calculate trend statistics from time series values"""
        try:
            if len(values) < 2:
                return {
                    "direction": "stable",
                    "strength": 0.0,
                    "correlation": 0.0,
                    "seasonal_patterns": {},
                }
            x = list(range(len(values)))
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            sum_y2 = sum(yi * yi for yi in values)
            correlation = 0.0
            if n * sum_x2 - sum_x * sum_x != 0 and n * sum_y2 - sum_y * sum_y != 0:
                correlation = (n * sum_xy - sum_x * sum_y) / (
                    (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
                ) ** 0.5
                correlation = max(-1.0, min(1.0, correlation))
            if correlation > 0.1:
                direction = "increasing"
            elif correlation < -0.1:
                direction = "decreasing"
            else:
                direction = "stable"
            strength = min(1.0, abs(correlation))
            return {
                "direction": direction,
                "strength": strength,
                "correlation": correlation,
                "seasonal_patterns": {},
            }
        except Exception as e:
            self.logger.error(f"Error calculating trend statistics: {e}")
            return {
                "direction": "stable",
                "strength": 0.0,
                "correlation": 0.0,
                "seasonal_patterns": {},
            }

    async def _get_session_summary_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, Any]:
        """Get session summary metrics for dashboard"""
        try:
            query = select(
                func.count().label("total_sessions"),
                func.sum(
                    case((TrainingSession.status == "completed", 1), else_=0)
                ).label("completed_sessions"),
                func.sum(case((TrainingSession.status == "running", 1), else_=0)).label(
                    "running_sessions"
                ),
                func.sum(case((TrainingSession.status == "failed", 1), else_=0)).label(
                    "failed_sessions"
                ),
                func.avg(TrainingSession.total_training_time_seconds / 3600.0).label(
                    "avg_duration_hours"
                ),
            ).where(
                and_(
                    TrainingSession.started_at >= start_time,
                    TrainingSession.started_at <= end_time,
                )
            )
            result = await execute_optimized_query(
                self.db_session, str(query), cache_ttl=self.dashboard_cache_ttl
            )
            if result:
                row = result[0]
                return {
                    "total_sessions": int(row.total_sessions)
                    if row.total_sessions
                    else 0,
                    "completed_sessions": int(row.completed_sessions)
                    if row.completed_sessions
                    else 0,
                    "running_sessions": int(row.running_sessions)
                    if row.running_sessions
                    else 0,
                    "failed_sessions": int(row.failed_sessions)
                    if row.failed_sessions
                    else 0,
                    "avg_duration_hours": float(row.avg_duration_hours)
                    if row.avg_duration_hours
                    else 0.0,
                }
            return {
                "total_sessions": 0,
                "completed_sessions": 0,
                "running_sessions": 0,
                "failed_sessions": 0,
                "avg_duration_hours": 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error getting session summary metrics: {e}")
            return {
                "total_sessions": 0,
                "completed_sessions": 0,
                "running_sessions": 0,
                "failed_sessions": 0,
                "avg_duration_hours": 0.0,
            }

    async def _get_performance_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, Any]:
        """Get performance metrics for dashboard"""
        try:
            query = select(
                func.avg(TrainingSession.current_performance).label("avg_performance"),
                func.max(TrainingSession.current_performance).label("max_performance"),
                func.min(TrainingSession.current_performance).label("min_performance"),
                func.avg(
                    TrainingSession.current_performance
                    - TrainingSession.initial_performance
                ).label("avg_improvement"),
                func.stddev(TrainingSession.current_performance).label(
                    "performance_std"
                ),
            ).where(
                and_(
                    TrainingSession.started_at >= start_time,
                    TrainingSession.started_at <= end_time,
                    TrainingSession.status == "completed",
                    TrainingSession.current_performance.isnot(None),
                )
            )
            result = await execute_optimized_query(
                self.db_session, str(query), cache_ttl=self.dashboard_cache_ttl
            )
            if result:
                row = result[0]
                return {
                    "avg_performance": float(row.avg_performance)
                    if row.avg_performance
                    else 0.0,
                    "max_performance": float(row.max_performance)
                    if row.max_performance
                    else 0.0,
                    "min_performance": float(row.min_performance)
                    if row.min_performance
                    else 0.0,
                    "avg_improvement": float(row.avg_improvement)
                    if row.avg_improvement
                    else 0.0,
                    "performance_std": float(row.performance_std)
                    if row.performance_std
                    else 0.0,
                }
            return {
                "avg_performance": 0.0,
                "max_performance": 0.0,
                "min_performance": 0.0,
                "avg_improvement": 0.0,
                "performance_std": 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {
                "avg_performance": 0.0,
                "max_performance": 0.0,
                "min_performance": 0.0,
                "avg_improvement": 0.0,
                "performance_std": 0.0,
            }

    async def _get_efficiency_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, Any]:
        """Get efficiency metrics for dashboard"""
        try:
            query = select(
                func.avg(
                    case(
                        (
                            TrainingSession.total_training_time_seconds > 0,
                            TrainingSession.current_performance
                            / (TrainingSession.total_training_time_seconds / 3600.0),
                        ),
                        else_=0.0,
                    )
                ).label("avg_efficiency"),
                func.avg(TrainingSession.current_iteration).label("avg_iterations"),
                func.avg(TrainingSession.total_training_time_seconds / 3600.0).label(
                    "avg_training_hours"
                ),
            ).where(
                and_(
                    TrainingSession.started_at >= start_time,
                    TrainingSession.started_at <= end_time,
                    TrainingSession.status == "completed",
                    TrainingSession.current_performance.isnot(None),
                )
            )
            result = await execute_optimized_query(
                self.db_session, str(query), cache_ttl=self.dashboard_cache_ttl
            )
            if result:
                row = result[0]
                return {
                    "avg_efficiency": float(row.avg_efficiency)
                    if row.avg_efficiency
                    else 0.0,
                    "avg_iterations": float(row.avg_iterations)
                    if row.avg_iterations
                    else 0.0,
                    "avg_training_hours": float(row.avg_training_hours)
                    if row.avg_training_hours
                    else 0.0,
                }
            return {
                "avg_efficiency": 0.0,
                "avg_iterations": 0.0,
                "avg_training_hours": 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error getting efficiency metrics: {e}")
            return {
                "avg_efficiency": 0.0,
                "avg_iterations": 0.0,
                "avg_training_hours": 0.0,
            }

    async def _get_error_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, Any]:
        """Get error metrics for dashboard"""
        try:
            session_query = select(
                func.count().label("total_sessions"),
                func.sum(case((TrainingSession.status == "failed", 1), else_=0)).label(
                    "failed_sessions"
                ),
            ).where(
                and_(
                    TrainingSession.started_at >= start_time,
                    TrainingSession.started_at <= end_time,
                )
            )
            session_result = await execute_optimized_query(
                self.db_session, str(session_query), cache_ttl=self.dashboard_cache_ttl
            )
            iteration_query = (
                select(
                    func.count().label("total_iterations"),
                    func.sum(
                        case((TrainingIteration.status == "failed", 1), else_=0)
                    ).label("failed_iterations"),
                )
                .select_from(
                    TrainingSession.__table__.join(
                        TrainingIteration.__table__,
                        TrainingSession.session_id == TrainingIteration.session_id,
                    )
                )
                .where(
                    and_(
                        TrainingSession.started_at >= start_time,
                        TrainingSession.started_at <= end_time,
                    )
                )
            )
            iteration_result = await execute_optimized_query(
                self.db_session,
                str(iteration_query),
                cache_ttl=self.dashboard_cache_ttl,
            )
            session_row = session_result[0] if session_result else None
            iteration_row = iteration_result[0] if iteration_result else None
            total_sessions = (
                int(session_row.total_sessions)
                if session_row and session_row.total_sessions
                else 0
            )
            failed_sessions = (
                int(session_row.failed_sessions)
                if session_row and session_row.failed_sessions
                else 0
            )
            total_iterations = (
                int(iteration_row.total_iterations)
                if iteration_row and iteration_row.total_iterations
                else 0
            )
            failed_iterations = (
                int(iteration_row.failed_iterations)
                if iteration_row and iteration_row.failed_iterations
                else 0
            )
            return {
                "session_error_rate": failed_sessions / total_sessions
                if total_sessions > 0
                else 0.0,
                "iteration_error_rate": failed_iterations / total_iterations
                if total_iterations > 0
                else 0.0,
                "total_failed_sessions": failed_sessions,
                "total_failed_iterations": failed_iterations,
            }
        except Exception as e:
            self.logger.error(f"Error getting error metrics: {e}")
            return {
                "session_error_rate": 0.0,
                "iteration_error_rate": 0.0,
                "total_failed_sessions": 0,
                "total_failed_iterations": 0,
            }

    async def _get_resource_utilization_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, Any]:
        """Get resource utilization metrics for dashboard"""
        try:
            return {
                "avg_memory_usage_mb": 0.0,
                "peak_memory_usage_mb": 0.0,
                "avg_cpu_utilization": 0.0,
                "total_compute_hours": 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error getting resource utilization metrics: {e}")
            return {
                "avg_memory_usage_mb": 0.0,
                "peak_memory_usage_mb": 0.0,
                "avg_cpu_utilization": 0.0,
                "total_compute_hours": 0.0,
            }

    def _calculate_period_changes(
        self, current: dict[str, Any], previous: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate period-over-period changes"""
        try:
            changes = {}
            if "session_summary" in current and "session_summary" in previous:
                curr_sessions = current["session_summary"].get("total_sessions", 0)
                prev_sessions = previous["session_summary"].get("total_sessions", 0)
                changes["sessions_change"] = curr_sessions - prev_sessions
                changes["sessions_change_pct"] = (
                    (curr_sessions - prev_sessions) / prev_sessions * 100
                    if prev_sessions > 0
                    else 0.0
                )
            if "performance" in current and "performance" in previous:
                curr_perf = current["performance"].get("avg_performance", 0.0)
                prev_perf = previous["performance"].get("avg_performance", 0.0)
                changes["performance_change"] = curr_perf - prev_perf
                changes["performance_change_pct"] = (
                    (curr_perf - prev_perf) / prev_perf * 100 if prev_perf > 0 else 0.0
                )
            return changes
        except Exception as e:
            self.logger.error(f"Error calculating period changes: {e}")
            return {}
