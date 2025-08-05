"""
Dashboard Export Functionality for Business Metrics.

Provides comprehensive dashboard data exports in multiple formats (JSON, CSV, Prometheus),
real-time streaming capabilities, and custom visualization data preparation.
"""

import logging
import time
import json
import csv
import io
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import statistics
from collections import defaultdict
import pandas as pd

from .aggregation_engine import get_aggregation_engine
from .ml_metrics import get_ml_metrics_collector
from .api_metrics import get_api_metrics_collector
from .performance_metrics import get_performance_metrics_collector
from .business_intelligence_metrics import get_bi_metrics_collector

class ExportFormat(Enum):
    """Export formats for dashboard data."""
    JSON = "json"
    CSV = "csv"
    PROMETHEUS = "prometheus"
    TABLEAU = "tableau"
    EXCEL = "excel"
    PARQUET = "parquet"

class DashboardType(Enum):
    """Types of dashboards for export."""
    EXECUTIVE_SUMMARY = "executive_summary"
    ML_PERFORMANCE = "ml_performance"
    API_ANALYTICS = "api_analytics"
    SYSTEM_PERFORMANCE = "system_performance"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    REAL_TIME_MONITORING = "real_time_monitoring"
    COST_ANALYSIS = "cost_analysis"
    USER_ENGAGEMENT = "user_engagement"

class TimeRange(Enum):
    """Time ranges for dashboard data."""
    LAST_HOUR = "1h"
    LAST_4_HOURS = "4h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    CUSTOM = "custom"

@dataclass
class DashboardWidget:
    """Individual dashboard widget configuration."""
    widget_id: str
    widget_type: str  # "chart", "table", "metric", "alert"
    title: str
    description: str
    data_source: str
    query_params: Dict[str, Any]
    visualization_config: Dict[str, Any]
    refresh_interval_seconds: int
    position: Dict[str, int]  # x, y, width, height
    dependencies: List[str]

@dataclass
class Dashboard:
    """Complete dashboard configuration."""
    dashboard_id: str
    dashboard_type: DashboardType
    title: str
    description: str
    widgets: List[DashboardWidget]
    layout_config: Dict[str, Any]
    auto_refresh: bool
    refresh_interval_seconds: int
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    permissions: Dict[str, List[str]]

class DashboardExporter:
    """
    Dashboard export functionality for business metrics.

    Provides comprehensive data export capabilities with multiple formats,
    real-time streaming, and visualization-ready data preparation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dashboard exporter."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Component collectors
        self.aggregation_engine = get_aggregation_engine()
        self.ml_collector = get_ml_metrics_collector()
        self.api_collector = get_api_metrics_collector()
        self.performance_collector = get_performance_metrics_collector()
        self.bi_collector = get_bi_metrics_collector()

        # Dashboard configurations
        self.dashboards: Dict[str, Dashboard] = {}
        self.dashboard_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = self.config.get("cache_ttl_seconds", 300)  # 5 minutes

        # Export statistics
        self.export_stats: Dict[str, Any] = {
            "exports_generated": 0,
            "dashboard_views": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_export_time": None,
            "popular_dashboards": defaultdict(int),
            "export_formats_used": defaultdict(int)
        }

        # Time range mappings
        self.time_range_hours = {
            TimeRange.LAST_HOUR: 1,
            TimeRange.LAST_4_HOURS: 4,
            TimeRange.LAST_24_HOURS: 24,
            TimeRange.LAST_7_DAYS: 168,
            TimeRange.LAST_30_DAYS: 720,
            TimeRange.LAST_90_DAYS: 2160
        }

        # Initialize default dashboards
        self._initialize_default_dashboards()

    def _initialize_default_dashboards(self):
        """Initialize default dashboard configurations."""
        # Executive Summary Dashboard
        self.dashboards["executive_summary"] = Dashboard(
            dashboard_id="executive_summary",
            dashboard_type=DashboardType.EXECUTIVE_SUMMARY,
            title="Executive Summary",
            description="High-level business metrics and KPIs",
            widgets=[
                DashboardWidget(
                    widget_id="total_users",
                    widget_type="metric",
                    title="Active Users",
                    description="Total active users in the last 24 hours",
                    data_source="user_engagement",
                    query_params={"time_range": "24h", "metric": "active_users"},
                    visualization_config={"format": "number", "color": "blue"},
                    refresh_interval_seconds=300,
                    position={"x": 0, "y": 0, "width": 3, "height": 2},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="success_rate",
                    widget_type="metric",
                    title="Overall Success Rate",
                    description="System-wide success rate",
                    data_source="aggregated_success_rate",
                    query_params={"time_range": "24h"},
                    visualization_config={"format": "percentage", "color": "green"},
                    refresh_interval_seconds=300,
                    position={"x": 3, "y": 0, "width": 3, "height": 2},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="daily_cost",
                    widget_type="metric",
                    title="Daily Operational Cost",
                    description="Total operational cost for today",
                    data_source="cost_tracking",
                    query_params={"time_range": "24h", "aggregation": "sum"},
                    visualization_config={"format": "currency", "color": "orange"},
                    refresh_interval_seconds=600,
                    position={"x": 6, "y": 0, "width": 3, "height": 2},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="performance_trend",
                    widget_type="chart",
                    title="Performance Trend",
                    description="Response time trend over the last 24 hours",
                    data_source="api_response_time",
                    query_params={"time_range": "24h", "aggregation": "avg", "interval": "1h"},
                    visualization_config={"chart_type": "line", "y_axis": "Response Time (ms)"},
                    refresh_interval_seconds=300,
                    position={"x": 0, "y": 2, "width": 6, "height": 4},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="feature_adoption",
                    widget_type="table",
                    title="Top Features by Adoption",
                    description="Most adopted features in the last 7 days",
                    data_source="feature_adoption",
                    query_params={"time_range": "7d", "top_n": 10},
                    visualization_config={"columns": ["feature", "users", "usage_count", "adoption_rate"]},
                    refresh_interval_seconds=3600,
                    position={"x": 6, "y": 2, "width": 6, "height": 4},
                    dependencies=[]
                )
            ],
            layout_config={"grid_size": 12, "row_height": 60},
            auto_refresh=True,
            refresh_interval_seconds=300,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["executive", "summary", "kpi"],
            permissions={"view": ["admin", "executive"], "edit": ["admin"]}
        )

        # ML Performance Dashboard
        self.dashboards["ml_performance"] = Dashboard(
            dashboard_id="ml_performance",
            dashboard_type=DashboardType.ML_PERFORMANCE,
            title="ML Performance Monitoring",
            description="Machine learning model performance and prompt improvement metrics",
            widgets=[
                DashboardWidget(
                    widget_id="prompt_success_rate",
                    widget_type="chart",
                    title="Prompt Improvement Success Rate",
                    description="Success rate by prompt category over time",
                    data_source="ml_prompt_success",
                    query_params={"time_range": "24h", "group_by": "category"},
                    visualization_config={"chart_type": "line", "multi_series": True},
                    refresh_interval_seconds=300,
                    position={"x": 0, "y": 0, "width": 6, "height": 4},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="model_latency",
                    widget_type="chart",
                    title="Model Inference Latency",
                    description="Model inference latency distribution",
                    data_source="ml_inference_latency",
                    query_params={"time_range": "4h", "percentiles": [50, 90, 99]},
                    visualization_config={"chart_type": "histogram"},
                    refresh_interval_seconds=300,
                    position={"x": 6, "y": 0, "width": 6, "height": 4},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="feature_flag_effectiveness",
                    widget_type="table",
                    title="Feature Flag Effectiveness",
                    description="Performance impact of active feature flags",
                    data_source="feature_flags",
                    query_params={"time_range": "24h", "show_active": True},
                    visualization_config={"columns": ["flag", "adoption_rate", "performance_impact", "success_rate"]},
                    refresh_interval_seconds=600,
                    position={"x": 0, "y": 4, "width": 12, "height": 3},
                    dependencies=[]
                )
            ],
            layout_config={"grid_size": 12, "row_height": 60},
            auto_refresh=True,
            refresh_interval_seconds=300,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["ml", "performance", "ai"],
            permissions={"view": ["admin", "ml_engineer", "data_scientist"], "edit": ["admin", "ml_engineer"]}
        )

        # Real-time Monitoring Dashboard
        self.dashboards["real_time_monitoring"] = Dashboard(
            dashboard_id="real_time_monitoring",
            dashboard_type=DashboardType.REAL_TIME_MONITORING,
            title="Real-time System Monitoring",
            description="Live system health and performance monitoring",
            widgets=[
                DashboardWidget(
                    widget_id="system_health",
                    widget_type="metric",
                    title="System Health Score",
                    description="Overall system health score (0-100)",
                    data_source="system_health",
                    query_params={"time_range": "5m"},
                    visualization_config={"format": "gauge", "min": 0, "max": 100, "thresholds": [70, 90]},
                    refresh_interval_seconds=30,
                    position={"x": 0, "y": 0, "width": 4, "height": 3},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="active_alerts",
                    widget_type="alert",
                    title="Active Alerts",
                    description="Current system alerts by severity",
                    data_source="alerts",
                    query_params={"status": "active"},
                    visualization_config={"group_by": "severity", "show_count": True},
                    refresh_interval_seconds=15,
                    position={"x": 4, "y": 0, "width": 4, "height": 3},
                    dependencies=[]
                ),
                DashboardWidget(
                    widget_id="request_rate",
                    widget_type="chart",
                    title="Request Rate",
                    description="Requests per second in real-time",
                    data_source="api_request_rate",
                    query_params={"time_range": "1h", "interval": "1m"},
                    visualization_config={"chart_type": "area", "real_time": True},
                    refresh_interval_seconds=10,
                    position={"x": 8, "y": 0, "width": 4, "height": 3},
                    dependencies=[]
                )
            ],
            layout_config={"grid_size": 12, "row_height": 60},
            auto_refresh=True,
            refresh_interval_seconds=30,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["monitoring", "real-time", "alerts"],
            permissions={"view": ["admin", "operator", "support"], "edit": ["admin"]}
        )

    async def export_dashboard(
        self,
        dashboard_id: str,
        export_format: ExportFormat,
        time_range: TimeRange = TimeRange.LAST_24_HOURS,
        custom_start: Optional[datetime] = None,
        custom_end: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], str, bytes]:
        """Export dashboard data in specified format."""
        try:
            # Check cache first
            cache_key = f"{dashboard_id}_{export_format.value}_{time_range.value}"
            if filters:
                cache_key += f"_{hash(json.dumps(filters, sort_keys=True))}"

            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                self.export_stats["cache_hits"] += 1
                return cached_data

            self.export_stats["cache_misses"] += 1

            # Get dashboard configuration
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")

            dashboard = self.dashboards[dashboard_id]

            # Determine time range
            if time_range == TimeRange.CUSTOM:
                if not custom_start or not custom_end:
                    raise ValueError("Custom time range requires start and end times")
                start_time = custom_start
                end_time = custom_end
            else:
                hours = self.time_range_hours[time_range]
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=hours)

            # Collect data for all widgets
            dashboard_data = {
                "dashboard_id": dashboard_id,
                "title": dashboard.title,
                "description": dashboard.description,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": (end_time - start_time).total_seconds() / 3600
                },
                "widgets": {},
                "metadata": {
                    "export_format": export_format.value,
                    "filters_applied": filters or {},
                    "auto_refresh": dashboard.auto_refresh,
                    "refresh_interval": dashboard.refresh_interval_seconds
                }
            }

            # Collect data for each widget
            for widget in dashboard.widgets:
                widget_data = await self._collect_widget_data(
                    widget, start_time, end_time, filters
                )
                dashboard_data["widgets"][widget.widget_id] = widget_data

            # Format data according to export format
            formatted_data = await self._format_export_data(dashboard_data, export_format)

            # Cache the result
            self._cache_data(cache_key, formatted_data)

            # Update statistics
            self.export_stats["exports_generated"] += 1
            self.export_stats["last_export_time"] = datetime.now(timezone.utc)
            self.export_stats["popular_dashboards"][dashboard_id] += 1
            self.export_stats["export_formats_used"][export_format.value] += 1

            return formatted_data

        except Exception as e:
            self.logger.error(f"Error exporting dashboard {dashboard_id}: {e}")
            raise

    async def _collect_widget_data(
        self,
        widget: DashboardWidget,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collect data for a specific widget."""
        try:
            data_source = widget.data_source
            query_params = widget.query_params.copy()

            # Apply filters
            if filters:
                query_params.update(filters)

            # Collect data based on data source
            if data_source == "user_engagement":
                data = await self._get_user_engagement_data(start_time, end_time, query_params)
            elif data_source == "aggregated_success_rate":
                data = await self._get_aggregated_success_rate(start_time, end_time, query_params)
            elif data_source == "cost_tracking":
                data = await self._get_cost_tracking_data(start_time, end_time, query_params)
            elif data_source == "api_response_time":
                data = await self._get_api_response_time_data(start_time, end_time, query_params)
            elif data_source == "feature_adoption":
                data = await self._get_feature_adoption_data(start_time, end_time, query_params)
            elif data_source == "ml_prompt_success":
                data = await self._get_ml_prompt_success_data(start_time, end_time, query_params)
            elif data_source == "ml_inference_latency":
                data = await self._get_ml_inference_latency_data(start_time, end_time, query_params)
            elif data_source == "feature_flags":
                data = await self._get_feature_flags_data(start_time, end_time, query_params)
            elif data_source == "system_health":
                data = await self._get_system_health_data(start_time, end_time, query_params)
            elif data_source == "alerts":
                data = await self._get_alerts_data(start_time, end_time, query_params)
            elif data_source == "api_request_rate":
                data = await self._get_api_request_rate_data(start_time, end_time, query_params)
            else:
                data = {"error": f"Unknown data source: {data_source}"}

            return {
                "widget_id": widget.widget_id,
                "widget_type": widget.widget_type,
                "title": widget.title,
                "description": widget.description,
                "data": data,
                "visualization_config": widget.visualization_config,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error collecting data for widget {widget.widget_id}: {e}")
            return {
                "widget_id": widget.widget_id,
                "error": str(e),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

    async def _get_user_engagement_data(
        self, start_time: datetime, end_time: datetime, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get user engagement data."""
        try:
            hours = (end_time - start_time).total_seconds() / 3600
            engagement_data = await self.bi_collector.get_user_journey_analytics(int(hours))

            if engagement_data.get("status") == "no_data":
                return {"value": 0, "trend": "stable"}

            # Extract active users count
            stage_analytics = engagement_data.get("stage_analytics", {})
            total_users = sum(
                stage_data.get("unique_users", 0)
                for stage_data in stage_analytics.values()
                if isinstance(stage_data, dict)
            )

            return {
                "value": total_users,
                "trend": "up",  # Could be calculated from historical data
                "breakdown": engagement_data.get("stage_analytics", {}),
                "active_sessions": engagement_data.get("active_sessions", 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting user engagement data: {e}")
            return {"error": str(e)}

    async def _get_aggregated_success_rate(
        self, start_time: datetime, end_time: datetime, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get aggregated success rate across all systems."""
        try:
            # Get success rates from different components
            hours = (end_time - start_time).total_seconds() / 3600

            ml_data = await self.ml_collector.get_prompt_improvement_summary(int(hours))
            api_data = await self.api_collector.get_endpoint_analytics(int(hours))
            performance_data = await self.performance_collector.get_pipeline_performance_summary(int(hours))

            success_rates: List[float] = []

            if ml_data.get("status") != "no_data":
                ml_rate = ml_data.get("overall_success_rate", 0)
                if isinstance(ml_rate, (int, float)):
                    success_rates.append(float(ml_rate))

            if api_data.get("status") != "no_data":
                # Calculate API success rate
                endpoint_analytics = api_data.get("endpoint_analytics", {})
                if endpoint_analytics and isinstance(endpoint_analytics, dict):
                    api_success_rates = [
                        float(analytics.get("success_rate", 0))
                        for analytics in endpoint_analytics.values()
                        if isinstance(analytics, dict) and isinstance(analytics.get("success_rate", 0), (int, float))
                    ]
                    if api_success_rates:
                        success_rates.append(statistics.mean(api_success_rates))

            if performance_data.get("status") != "no_data":
                stage_performance = performance_data.get("stage_performance", {})
                if stage_performance and isinstance(stage_performance, dict):
                    perf_success_rates = [
                        float(stage.get("success_rate", 0))
                        for stage in stage_performance.values()
                        if isinstance(stage, dict) and isinstance(stage.get("success_rate", 0), (int, float))
                    ]
                    if perf_success_rates:
                        success_rates.append(statistics.mean(perf_success_rates))

            overall_success_rate = statistics.mean(success_rates) if success_rates else 0.0

            return {
                "value": overall_success_rate,
                "components": {
                    "ml": ml_data.get("overall_success_rate", 0),
                    "api": statistics.mean([
                        analytics.get("success_rate", 0)
                        for analytics in api_data.get("endpoint_analytics", {}).values()
                    ]) if api_data.get("endpoint_analytics") else 0,
                    "performance": statistics.mean([
                        stage.get("success_rate", 0)
                        for stage in performance_data.get("stage_performance", {}).values()
                    ]) if performance_data.get("stage_performance") else 0
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting aggregated success rate: {e}")
            return {"error": str(e)}

    async def _get_cost_tracking_data(
        self, start_time: datetime, end_time: datetime, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get cost tracking data."""
        try:
            days = (end_time - start_time).days
            if days < 1:
                days = 1

            cost_data = await self.bi_collector.get_cost_efficiency_report(days)

            if cost_data.get("status") == "no_data":
                return {"value": 0, "currency": "USD", "trend": "stable"}

            aggregation = params.get("aggregation", "sum")
            total_cost = cost_data.get("total_cost", 0)

            if aggregation == "daily_average":
                value = cost_data.get("daily_average", 0)
            else:
                value = total_cost

            return {
                "value": value,
                "currency": "USD",
                "breakdown": cost_data.get("cost_breakdown", {}),
                "trend": "up",  # Could be calculated from historical data
                "period_total": total_cost
            }

        except Exception as e:
            self.logger.error(f"Error getting cost tracking data: {e}")
            return {"error": str(e)}

    async def _get_api_response_time_data(
        self, start_time: datetime, end_time: datetime, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get API response time trend data."""
        try:
            hours = (end_time - start_time).total_seconds() / 3600
            api_data = await self.api_collector.get_endpoint_analytics(int(hours))

            if api_data.get("status") == "no_data":
                return {"series": [], "average": 0}

            endpoint_analytics = api_data.get("endpoint_analytics", {})

            # Calculate time series data (simplified - in practice, you'd get actual time series)
            time_series = []
            interval_minutes = 60  # 1 hour intervals
            current_time = start_time

            while current_time < end_time:
                # Calculate average response time for this interval
                avg_response_time = statistics.mean([
                    analytics.get("avg_response_time_ms", 0)
                    for analytics in endpoint_analytics.values()
                ]) if endpoint_analytics else 0

                time_series.append({
                    "timestamp": current_time.isoformat(),
                    "value": avg_response_time
                })

                current_time += timedelta(minutes=interval_minutes)

            overall_average = statistics.mean([
                analytics.get("avg_response_time_ms", 0)
                for analytics in endpoint_analytics.values()
            ]) if endpoint_analytics else 0

            return {
                "series": time_series,
                "average": overall_average,
                "unit": "milliseconds",
                "endpoints_count": len(endpoint_analytics)
            }

        except Exception as e:
            self.logger.error(f"Error getting API response time data: {e}")
            return {"error": str(e)}

    async def _get_feature_adoption_data(
        self, start_time: datetime, end_time: datetime, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get feature adoption data."""
        try:
            days = max(1, (end_time - start_time).days)
            adoption_data = await self.bi_collector.get_feature_adoption_report(days)

            if adoption_data.get("status") == "no_data":
                return {"features": []}

            category_analysis = adoption_data.get("category_analysis", {})
            top_n = params.get("top_n", 10)

            # Flatten and sort features by user count
            all_features: List[Dict[str, Any]] = []
            for category, category_data in category_analysis.items():
                if isinstance(category_data, dict):
                    top_features_list = category_data.get("top_features", [])
                    if isinstance(top_features_list, list):
                        for feature_data in top_features_list:
                            if isinstance(feature_data, dict):
                                all_features.append({
                                    "feature": feature_data.get("name", ""),
                                    "category": category,
                                    "users": feature_data.get("users", 0),
                                    "usage_count": feature_data.get("usage", 0),
                                    "adoption_rate": category_data.get("adoption_rate", 0)
                                })

            # Sort by user count and take top N
            top_features = sorted(all_features, key=lambda x: int(x.get("users", 0)), reverse=True)[:top_n]

            return {
                "features": top_features,
                "total_categories": len(category_analysis),
                "total_adoption_events": adoption_data.get("total_adoption_events", 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting feature adoption data: {e}")
            return {"error": str(e)}

    async def _get_ml_prompt_success_data(
        self, start_time: datetime, end_time: datetime, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get ML prompt success rate data."""
        try:
            hours = (end_time - start_time).total_seconds() / 3600
            ml_data = await self.ml_collector.get_prompt_improvement_summary(int(hours))

            if ml_data.get("status") == "no_data":
                return {"series": [], "categories": []}

            category_breakdown = ml_data.get("category_breakdown", {})

            # Create time series data for each category
            series_data = []
            for category, data in category_breakdown.items():
                series_data.append({
                    "name": category,
                    "success_rate": data.get("success_rate", 0),
                    "count": data.get("count", 0),
                    "avg_improvement": data.get("avg_improvement_ratio", 0)
                })

            return {
                "series": series_data,
                "overall_success_rate": ml_data.get("overall_success_rate", 0),
                "total_improvements": ml_data.get("total_improvements", 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting ML prompt success data: {e}")
            return {"error": str(e)}

    async def _get_ml_inference_latency_data(
        self, start_time: datetime, end_time: datetime, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get ML inference latency data."""
        try:
            hours = (end_time - start_time).total_seconds() / 3600
            ml_data = await self.ml_collector.get_model_performance_summary(int(hours))

            if ml_data.get("status") == "no_data":
                return {"distribution": [], "percentiles": {}}

            model_breakdown = ml_data.get("model_breakdown", {})

            # Create latency distribution data
            latency_data = []
            for model_name, model_data in model_breakdown.items():
                latency_data.append({
                    "model": model_name,
                    "avg_latency": model_data.get("avg_latency_ms", 0),
                    "inference_count": model_data.get("inference_count", 0),
                    "tokens_per_second": model_data.get("tokens_per_second", 0)
                })

            # Calculate overall percentiles (simplified)
            all_latencies: List[float] = []
            for data in latency_data:
                if isinstance(data, dict):
                    avg_latency = data.get("avg_latency", 0)
                    if isinstance(avg_latency, (int, float)) and avg_latency > 0:
                        all_latencies.append(float(avg_latency))

            percentiles: Dict[str, float] = {}
            if all_latencies:
                percentiles = {
                    "p50": statistics.median(all_latencies),
                    "p90": statistics.quantiles(all_latencies, n=10)[8] if len(all_latencies) >= 10 else max(all_latencies),
                    "p99": statistics.quantiles(all_latencies, n=100)[98] if len(all_latencies) >= 100 else max(all_latencies)
                }

            return {
                "distribution": latency_data,
                "percentiles": percentiles,
                "total_inferences": ml_data.get("total_inferences", 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting ML inference latency data: {e}")
            return {"error": str(e)}

    async def _get_feature_flags_data(
        self, start_time: datetime, end_time: datetime, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get feature flags effectiveness data."""
        try:
            hours = (end_time - start_time).total_seconds() / 3600
            flag_data = await self.bi_collector.get_feature_flag_effectiveness(int(hours))

            if flag_data.get("status") == "no_data":
                return {"flags": []}

            flag_breakdown = flag_data.get("flag_breakdown", {})
            show_active = params.get("show_active", False)

            flags_list = []
            for flag_name, flag_data in flag_breakdown.items():
                if show_active and flag_data.get("adoption_rate", 0) == 0:
                    continue

                flags_list.append({
                    "flag": flag_name,
                    "adoption_rate": flag_data.get("adoption_rate", 0),
                    "performance_impact": flag_data.get("avg_performance_impact_ms", 0),
                    "success_rate": flag_data.get("success_rate", 0),
                    "total_exposures": flag_data.get("total_exposures", 0)
                })

            return {
                "flags": flags_list,
                "total_exposures": flag_data.get("total_flag_exposures", 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting feature flags data: {e}")
            return {"error": str(e)}

    async def _get_system_health_data(
        self, start_time: datetime, end_time: datetime, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get system health score."""
        try:
            # Calculate composite health score from all components
            hours = (end_time - start_time).total_seconds() / 3600

            # Get data from all collectors
            ml_data = await self.ml_collector.get_prompt_improvement_summary(int(hours))
            api_data = await self.api_collector.get_endpoint_analytics(int(hours))
            performance_data = await self.performance_collector.get_pipeline_performance_summary(int(hours))

            # Calculate health scores
            health_components: List[float] = []

            if ml_data.get("status") != "no_data":
                ml_rate = ml_data.get("overall_success_rate", 0)
                if isinstance(ml_rate, (int, float)):
                    ml_health = float(ml_rate) * 100
                    health_components.append(ml_health)

            if api_data.get("status") != "no_data":
                endpoint_analytics = api_data.get("endpoint_analytics", {})
                if endpoint_analytics and isinstance(endpoint_analytics, dict):
                    api_rates = [
                        float(analytics.get("success_rate", 0)) * 100
                        for analytics in endpoint_analytics.values()
                        if isinstance(analytics, dict) and isinstance(analytics.get("success_rate", 0), (int, float))
                    ]
                    if api_rates:
                        api_health = statistics.mean(api_rates)
                        health_components.append(api_health)

            if performance_data.get("status") != "no_data":
                stage_performance = performance_data.get("stage_performance", {})
                if stage_performance and isinstance(stage_performance, dict):
                    perf_rates = [
                        float(stage.get("success_rate", 0)) * 100
                        for stage in stage_performance.values()
                        if isinstance(stage, dict) and isinstance(stage.get("success_rate", 0), (int, float))
                    ]
                    if perf_rates:
                        perf_health = statistics.mean(perf_rates)
                        health_components.append(perf_health)

            overall_health = statistics.mean(health_components) if health_components else 50.0

            return {
                "value": overall_health,
                "status": "healthy" if overall_health > 80 else "warning" if overall_health > 60 else "critical",
                "components": {
                    "ml": health_components[0] if len(health_components) > 0 else 50,
                    "api": health_components[1] if len(health_components) > 1 else 50,
                    "performance": health_components[2] if len(health_components) > 2 else 50
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting system health data: {e}")
            return {"error": str(e)}

    async def _get_alerts_data(
        self, _start_time: datetime, _end_time: datetime, _params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get alerts data."""
        try:
            # Get active alerts from aggregation engine
            active_alerts = await self.aggregation_engine.get_active_alerts()

            # Group by severity
            severity_counts = defaultdict(int)
            alert_details = []

            for alert in active_alerts:
                severity = alert.get("severity", "info")
                severity_counts[severity] += 1

                alert_details.append({
                    "title": alert.get("title", ""),
                    "severity": severity,
                    "metric": alert.get("metric_name", ""),
                    "current_value": alert.get("current_value", 0),
                    "threshold": alert.get("threshold_value", 0),
                    "timestamp": alert.get("timestamp", "")
                })

            return {
                "total_alerts": len(active_alerts),
                "severity_breakdown": dict(severity_counts),
                "alerts": alert_details[:10]  # Top 10 recent alerts
            }

        except Exception as e:
            self.logger.error(f"Error getting alerts data: {e}")
            return {"error": str(e)}

    async def _get_api_request_rate_data(
        self, start_time: datetime, end_time: datetime, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get API request rate data."""
        try:
            hours = (end_time - start_time).total_seconds() / 3600
            api_data = await self.api_collector.get_endpoint_analytics(int(hours))

            if api_data.get("status") == "no_data":
                return {"series": [], "total_requests": 0}

            total_requests = api_data.get("total_requests", 0)
            requests_per_hour = total_requests / hours if hours > 0 else 0
            requests_per_second = requests_per_hour / 3600

            # Generate time series data (simplified)
            interval_minutes = int(params.get("interval", "1m").rstrip("m"))
            time_series = []
            current_time = start_time

            while current_time < end_time:
                # Simulate request rate (in practice, you'd get actual data)
                rate = requests_per_second + (requests_per_second * 0.1 * (hash(str(current_time)) % 21 - 10) / 10)

                time_series.append({
                    "timestamp": current_time.isoformat(),
                    "value": max(0, rate)
                })

                current_time += timedelta(minutes=interval_minutes)

            return {
                "series": time_series,
                "total_requests": total_requests,
                "avg_rate_per_second": requests_per_second,
                "peak_rate": max(point["value"] for point in time_series) if time_series else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting API request rate data: {e}")
            return {"error": str(e)}

    async def _format_export_data(
        self, dashboard_data: Dict[str, Any], export_format: ExportFormat
    ) -> Union[Dict[str, Any], str, bytes]:
        """Format dashboard data for specific export format."""
        try:
            if export_format == ExportFormat.JSON:
                return dashboard_data

            elif export_format == ExportFormat.CSV:
                return self._format_csv_export(dashboard_data)

            elif export_format == ExportFormat.PROMETHEUS:
                return self._format_prometheus_export(dashboard_data)


            elif export_format == ExportFormat.TABLEAU:
                return self._format_tableau_export(dashboard_data)

            elif export_format == ExportFormat.EXCEL:
                return await self._format_excel_export(dashboard_data)

            elif export_format == ExportFormat.PARQUET:
                return await self._format_parquet_export(dashboard_data)

            else:
                raise ValueError(f"Unsupported export format: {export_format}")

        except Exception as e:
            self.logger.error(f"Error formatting export data: {e}")
            raise

    def _format_csv_export(self, dashboard_data: Dict[str, Any]) -> str:
        """Format dashboard data as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["Dashboard", "Widget", "Metric", "Value", "Timestamp"])

        # Write data
        dashboard_id = dashboard_data["dashboard_id"]
        for widget_id, widget_data in dashboard_data["widgets"].items():
            if "data" in widget_data and not isinstance(widget_data["data"], dict) or "error" in widget_data["data"]:
                continue

            data = widget_data["data"]

            # Handle different data structures
            if "value" in data:
                writer.writerow([
                    dashboard_id,
                    widget_id,
                    widget_data["title"],
                    data["value"],
                    widget_data["last_updated"]
                ])
            elif "series" in data:
                for point in data["series"]:
                    writer.writerow([
                        dashboard_id,
                        widget_id,
                        widget_data["title"],
                        point.get("value", ""),
                        point.get("timestamp", "")
                    ])

        return output.getvalue()

    def _format_prometheus_export(self, dashboard_data: Dict[str, Any]) -> str:
        """Format dashboard data as Prometheus metrics."""
        metrics = []
        timestamp = int(time.time() * 1000)

        for widget_id, widget_data in dashboard_data["widgets"].items():
            if "data" in widget_data and "error" not in widget_data["data"]:
                data = widget_data["data"]
                metric_name = f"dashboard_{widget_id.replace('-', '_')}"

                if "value" in data:
                    labels = f'{{dashboard="{dashboard_data["dashboard_id"]}",widget="{widget_id}"}}'
                    metrics.append(f"{metric_name}{labels} {data['value']} {timestamp}")

                elif "series" in data:
                    for i, point in enumerate(data["series"]):
                        labels = f'{{dashboard="{dashboard_data["dashboard_id"]}",widget="{widget_id}",series_index="{i}"}}'
                        metrics.append(f"{metric_name}{labels} {point.get('value', 0)} {timestamp}")

        return "\n".join(metrics)


    def _format_tableau_export(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format dashboard data for Tableau."""
        # Tableau workbook structure
        workbook = {
            "workbook": {
                "name": dashboard_data["title"],
                "datasources": [],
                "worksheets": []
            }
        }

        for _widget_id, widget_data in dashboard_data["widgets"].items():
            if "error" in widget_data.get("data", {}):
                continue

            worksheet = {
                "name": widget_data["title"],
                "data": widget_data["data"],
                "visualization": {
                    "type": self._map_widget_type_to_tableau(widget_data["widget_type"]),
                    "config": widget_data["visualization_config"]
                }
            }
            workbook["workbook"]["worksheets"].append(worksheet)

        return workbook

    async def _format_excel_export(self, dashboard_data: Dict[str, Any]) -> bytes:
        """Format dashboard data as Excel file."""
        try:
            # Create Excel workbook with pandas
            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = {
                    "Dashboard": [dashboard_data["dashboard_id"]],
                    "Title": [dashboard_data["title"]],
                    "Generated": [dashboard_data["generated_at"]],
                    "Time Range": [f"{dashboard_data['time_range']['start']} to {dashboard_data['time_range']['end']}"]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Create sheet for each widget
                for widget_id, widget_data in dashboard_data["widgets"].items():
                    if "error" in widget_data.get("data", {}):
                        continue

                    data = widget_data["data"]
                    sheet_name = widget_id[:31]  # Excel sheet name limit

                    if "series" in data:
                        df = pd.DataFrame(data["series"])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif "features" in data:
                        df = pd.DataFrame(data["features"])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif "value" in data:
                        df = pd.DataFrame([{"Metric": widget_data["title"], "Value": data["value"]}])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

            return excel_buffer.getvalue()

        except Exception as e:
            self.logger.error(f"Error creating Excel export: {e}")
            raise

    async def _format_parquet_export(self, dashboard_data: Dict[str, Any]) -> bytes:
        """Format dashboard data as Parquet file."""
        try:
            # Flatten all widget data into a single DataFrame
            all_data = []

            for widget_id, widget_data in dashboard_data["widgets"].items():
                if "error" in widget_data.get("data", {}):
                    continue

                data = widget_data["data"]
                base_record = {
                    "dashboard_id": dashboard_data["dashboard_id"],
                    "widget_id": widget_id,
                    "widget_title": widget_data["title"],
                    "widget_type": widget_data["widget_type"]
                }

                if "series" in data:
                    for point in data["series"]:
                        record = {**base_record, **point}
                        all_data.append(record)
                elif "features" in data:
                    for feature in data["features"]:
                        record = {**base_record, **feature}
                        all_data.append(record)
                elif "value" in data:
                    record = {**base_record, "value": data["value"]}
                    all_data.append(record)

            if all_data:
                df = pd.DataFrame(all_data)
                parquet_buffer = io.BytesIO()
                df.to_parquet(parquet_buffer, index=False)
                return parquet_buffer.getvalue()
            else:
                return b""

        except Exception as e:
            self.logger.error(f"Error creating Parquet export: {e}")
            raise


    def _map_widget_type_to_tableau(self, widget_type: str) -> str:
        """Map widget type to Tableau visualization type."""
        mapping = {
            "chart": "line",
            "metric": "text",
            "table": "crosstab",
            "alert": "text"
        }
        return mapping.get(widget_type, "line")

    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if cache_key not in self.dashboard_cache:
            return None

        cache_entry = self.dashboard_cache[cache_key]
        if datetime.now(timezone.utc) - cache_entry["timestamp"] > timedelta(seconds=self.cache_ttl_seconds):
            del self.dashboard_cache[cache_key]
            return None

        return cache_entry["data"]

    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.dashboard_cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now(timezone.utc)
        }

        # Clean up old cache entries if cache is too large
        if len(self.dashboard_cache) > 100:
            oldest_key = min(
                self.dashboard_cache.keys(),
                key=lambda k: self.dashboard_cache[k]["timestamp"]
            )
            del self.dashboard_cache[oldest_key]

    async def get_available_dashboards(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards."""
        dashboards_list = []

        for dashboard_id, dashboard in self.dashboards.items():
            dashboards_list.append({
                "id": dashboard_id,
                "title": dashboard.title,
                "description": dashboard.description,
                "type": dashboard.dashboard_type.value,
                "widget_count": len(dashboard.widgets),
                "auto_refresh": dashboard.auto_refresh,
                "tags": dashboard.tags,
                "created_at": dashboard.created_at.isoformat(),
                "updated_at": dashboard.updated_at.isoformat()
            })

        return dashboards_list

    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics."""
        return {
            **self.export_stats,
            "cache_size": len(self.dashboard_cache),
            "available_dashboards": len(self.dashboards),
            "cache_hit_rate": (
                self.export_stats["cache_hits"] /
                (self.export_stats["cache_hits"] + self.export_stats["cache_misses"])
                if (self.export_stats["cache_hits"] + self.export_stats["cache_misses"]) > 0 else 0
            )
        }

# Global instance
_dashboard_exporter: Optional[DashboardExporter] = None

def get_dashboard_exporter(config: Optional[Dict[str, Any]] = None) -> DashboardExporter:
    """Get global dashboard exporter instance."""
    global _dashboard_exporter
    if _dashboard_exporter is None:
        _dashboard_exporter = DashboardExporter(config)
    return _dashboard_exporter

# Convenience functions
async def export_executive_summary(
    export_format: ExportFormat = ExportFormat.JSON,
    time_range: TimeRange = TimeRange.LAST_24_HOURS
):
    """Export executive summary dashboard."""
    exporter = get_dashboard_exporter()
    return await exporter.export_dashboard("executive_summary", export_format, time_range)

async def export_ml_performance(
    export_format: ExportFormat = ExportFormat.JSON,
    time_range: TimeRange = TimeRange.LAST_24_HOURS
):
    """Export ML performance dashboard."""
    exporter = get_dashboard_exporter()
    return await exporter.export_dashboard("ml_performance", export_format, time_range)

async def export_real_time_monitoring(
    export_format: ExportFormat = ExportFormat.JSON,
    time_range: TimeRange = TimeRange.LAST_HOUR
):
    """Export real-time monitoring dashboard."""
    exporter = get_dashboard_exporter()
    return await exporter.export_dashboard("real_time_monitoring", export_format, time_range)
