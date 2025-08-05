"""Automated reporting and capacity planning for performance baselines."""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

# Plotting and visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Data processing
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Report generation
try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from .models import BaselineMetrics, PerformanceTrend, RegressionAlert, get_metric_definition
from .statistical_analyzer import StatisticalAnalyzer
from .baseline_collector import BaselineCollector
from .regression_detector import RegressionDetector

logger = logging.getLogger(__name__)

class ReportConfig:
    """Configuration for baseline reporting."""
    
    def __init__(
        self,
        output_directory: Optional[Path] = None,
        include_charts: bool = True,
        include_capacity_planning: bool = True,
        include_recommendations: bool = True,
        chart_format: str = 'png',
        chart_dpi: int = 150,
        report_formats: List[str] = None,  # ['html', 'json', 'text']
        capacity_forecast_days: int = 30,
        performance_sla_targets: Optional[Dict[str, float]] = None
    ):
        """Initialize report configuration.
        
        Args:
            output_directory: Directory for report outputs
            include_charts: Generate performance charts
            include_capacity_planning: Include capacity planning analysis
            include_recommendations: Include optimization recommendations
            chart_format: Format for charts ('png', 'svg', 'pdf')
            chart_dpi: Chart resolution (dots per inch)
            report_formats: Output formats for reports
            capacity_forecast_days: Days to forecast for capacity planning
            performance_sla_targets: SLA targets for different metrics
        """
        self.output_directory = output_directory or Path("./reports")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.include_charts = include_charts and MATPLOTLIB_AVAILABLE
        self.include_capacity_planning = include_capacity_planning
        self.include_recommendations = include_recommendations
        self.chart_format = chart_format
        self.chart_dpi = chart_dpi
        self.report_formats = report_formats or ['html', 'json']
        self.capacity_forecast_days = capacity_forecast_days
        
        # Default SLA targets
        self.performance_sla_targets = performance_sla_targets or {
            'response_time_p95': 200.0,  # ms
            'error_rate': 1.0,           # %
            'cpu_utilization': 70.0,     # %
            'memory_utilization': 80.0,  # %
            'availability': 99.9         # %
        }

class PerformanceReport:
    """Comprehensive performance report."""
    
    def __init__(
        self,
        report_id: str,
        report_type: str,
        generation_time: datetime,
        time_period: Tuple[datetime, datetime]
    ):
        self.report_id = report_id
        self.report_type = report_type
        self.generation_time = generation_time
        self.time_period = time_period
        
        # Report sections
        self.executive_summary: Dict[str, Any] = {}
        self.performance_overview: Dict[str, Any] = {}
        self.trend_analysis: Dict[str, Any] = {}
        self.capacity_planning: Dict[str, Any] = {}
        self.alerts_summary: Dict[str, Any] = {}
        self.recommendations: List[Dict[str, Any]] = []
        self.appendices: Dict[str, Any] = {}
        
        # Generated assets
        self.charts: Dict[str, str] = {}  # chart_name -> file_path
        self.data_files: Dict[str, str] = {}  # data_type -> file_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'generation_time': self.generation_time.isoformat(),
            'time_period': {
                'start': self.time_period[0].isoformat(),
                'end': self.time_period[1].isoformat()
            },
            'executive_summary': self.executive_summary,
            'performance_overview': self.performance_overview,
            'trend_analysis': self.trend_analysis,
            'capacity_planning': self.capacity_planning,
            'alerts_summary': self.alerts_summary,
            'recommendations': self.recommendations,
            'appendices': self.appendices,
            'charts': self.charts,
            'data_files': self.data_files
        }

class BaselineReporter:
    """Automated reporting and capacity planning system.
    
    Generates comprehensive performance reports with trend analysis,
    capacity planning, and actionable recommendations.
    """
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        collector: Optional[BaselineCollector] = None,
        analyzer: Optional[StatisticalAnalyzer] = None,
        detector: Optional[RegressionDetector] = None
    ):
        """Initialize baseline reporter.
        
        Args:
            config: Report configuration
            collector: Baseline collector for data access
            analyzer: Statistical analyzer for trend analysis
            detector: Regression detector for alert data
        """
        self.config = config or ReportConfig()
        self.collector = collector or BaselineCollector()
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.detector = detector or RegressionDetector()
        
        # Setup Jinja2 environment if available
        self.jinja_env = None
        if JINJA2_AVAILABLE:
            template_dir = Path(__file__).parent / "templates"
            if template_dir.exists():
                self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        logger.info(f"BaselineReporter initialized (charts: {self.config.include_charts})")

    async def generate_daily_report(self, report_date: Optional[datetime] = None) -> PerformanceReport:
        """Generate daily performance report.
        
        Args:
            report_date: Date for the report (defaults to yesterday)
            
        Returns:
            Generated performance report
        """
        if report_date is None:
            report_date = datetime.now(timezone.utc) - timedelta(days=1)
        
        # Define time period (24 hours)
        start_time = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        return await self._generate_report(
            report_type="daily",
            time_period=(start_time, end_time),
            comparison_period_days=7
        )

    async def generate_weekly_report(self, week_start: Optional[datetime] = None) -> PerformanceReport:
        """Generate weekly performance report.
        
        Args:
            week_start: Start of the week for the report
            
        Returns:
            Generated performance report
        """
        if week_start is None:
            # Start of last week (Monday)
            today = datetime.now(timezone.utc).date()
            days_since_monday = today.weekday()
            week_start = datetime.combine(
                today - timedelta(days=days_since_monday + 7),
                datetime.min.time()
            ).replace(tzinfo=timezone.utc)
        
        end_time = week_start + timedelta(days=7)
        
        return await self._generate_report(
            report_type="weekly",
            time_period=(week_start, end_time),
            comparison_period_days=28
        )

    async def generate_monthly_report(self, month_start: Optional[datetime] = None) -> PerformanceReport:
        """Generate monthly performance report.
        
        Args:
            month_start: Start of the month for the report
            
        Returns:
            Generated performance report
        """
        if month_start is None:
            # Start of last month
            today = datetime.now(timezone.utc).date()
            if today.month == 1:
                month_start = datetime(today.year - 1, 12, 1, tzinfo=timezone.utc)
            else:
                month_start = datetime(today.year, today.month - 1, 1, tzinfo=timezone.utc)
        
        # End of month
        if month_start.month == 12:
            end_time = datetime(month_start.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_time = datetime(month_start.year, month_start.month + 1, 1, tzinfo=timezone.utc)
        
        return await self._generate_report(
            report_type="monthly",
            time_period=(month_start, end_time),
            comparison_period_days=90
        )

    async def generate_capacity_planning_report(
        self,
        forecast_days: Optional[int] = None
    ) -> PerformanceReport:
        """Generate capacity planning report.
        
        Args:
            forecast_days: Days to forecast (defaults to config value)
            
        Returns:
            Generated capacity planning report
        """
        forecast_days = forecast_days or self.config.capacity_forecast_days
        
        # Use last 30 days of data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=30)
        
        report = await self._generate_report(
            report_type="capacity_planning",
            time_period=(start_time, end_time),
            comparison_period_days=60
        )
        
        # Add detailed capacity analysis
        await self._add_detailed_capacity_analysis(report, forecast_days)
        
        return report

    async def _generate_report(
        self,
        report_type: str,
        time_period: Tuple[datetime, datetime],
        comparison_period_days: int
    ) -> PerformanceReport:
        """Generate a performance report for the specified period."""
        report_id = str(uuid.uuid4())
        generation_time = datetime.now(timezone.utc)
        
        logger.info(f"Generating {report_type} report for period {time_period[0]} to {time_period[1]}")
        
        # Create report
        report = PerformanceReport(
            report_id=report_id,
            report_type=report_type,
            generation_time=generation_time,
            time_period=time_period
        )
        
        # Load baseline data for the period
        period_hours = int((time_period[1] - time_period[0]).total_seconds() / 3600)
        baselines = await self._load_baselines_for_period(time_period[0], period_hours)
        
        if not baselines:
            logger.warning(f"No baselines found for period {time_period}")
            report.executive_summary['status'] = 'insufficient_data'
            return report
        
        # Load comparison data
        comparison_start = time_period[0] - timedelta(days=comparison_period_days)
        comparison_baselines = await self._load_baselines_for_period(
            comparison_start, 
            int((time_period[0] - comparison_start).total_seconds() / 3600)
        )
        
        # Generate report sections
        await self._generate_executive_summary(report, baselines, comparison_baselines)
        await self._generate_performance_overview(report, baselines)
        await self._generate_trend_analysis(report, baselines, comparison_baselines)
        
        if self.config.include_capacity_planning:
            await self._generate_capacity_planning(report, baselines)
        
        await self._generate_alerts_summary(report, time_period)
        
        if self.config.include_recommendations:
            await self._generate_recommendations(report, baselines)
        
        # Generate charts
        if self.config.include_charts:
            await self._generate_charts(report, baselines)
        
        # Save report
        await self._save_report(report)
        
        logger.info(f"Generated {report_type} report: {report_id}")
        return report

    async def _load_baselines_for_period(
        self, 
        start_time: datetime, 
        hours: int
    ) -> List[BaselineMetrics]:
        """Load baselines for a specific time period."""
        # Implementation would depend on how baselines are stored
        # For now, use the collector's method
        return await self.collector.load_recent_baselines(hours)

    async def _generate_executive_summary(
        self,
        report: PerformanceReport,
        baselines: List[BaselineMetrics],
        comparison_baselines: List[BaselineMetrics]
    ) -> None:
        """Generate executive summary section."""
        if not baselines:
            report.executive_summary = {'status': 'no_data'}
            return
        
        # Calculate key metrics
        latest_baseline = baselines[-1]
        
        # Performance score
        try:
            performance_score = await self.analyzer.calculate_performance_score(latest_baseline)
            overall_grade = performance_score.get('grade', 'N/A')
            overall_score = performance_score.get('overall_score', 0)
        except Exception as e:
            logger.error(f"Failed to calculate performance score: {e}")
            overall_grade = 'N/A'
            overall_score = 0
        
        # SLA compliance
        sla_compliance = self._calculate_sla_compliance(baselines)
        
        # Key trends
        key_trends = await self._calculate_key_trends(baselines, comparison_baselines)
        
        # Active issues
        active_alerts = self.detector.get_active_alerts()
        critical_alerts = [alert for alert in active_alerts if alert.severity.value == 'critical']
        
        report.executive_summary = {
            'status': 'generated',
            'period_summary': {
                'total_baselines': len(baselines),
                'baseline_frequency_hours': self._calculate_baseline_frequency(baselines)
            },
            'performance_grade': overall_grade,
            'performance_score': overall_score,
            'sla_compliance': sla_compliance,
            'key_trends': key_trends,
            'active_issues': {
                'total_alerts': len(active_alerts),
                'critical_alerts': len(critical_alerts),
                'top_issues': [alert.metric_name for alert in critical_alerts[:3]]
            },
            'recommendations_count': 0  # Will be updated later
        }

    async def _generate_performance_overview(
        self,
        report: PerformanceReport,
        baselines: List[BaselineMetrics]
    ) -> None:
        """Generate performance overview section."""
        if not baselines:
            return
        
        # Aggregate metrics across all baselines
        aggregated_metrics = self._aggregate_baseline_metrics(baselines)
        
        # Calculate percentiles and statistics
        metrics_stats = {}
        for metric_name, values in aggregated_metrics.items():
            if values:
                metrics_stats[metric_name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': self._percentile(values, 95),
                    'p99': self._percentile(values, 99),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        # Resource utilization analysis
        resource_analysis = self._analyze_resource_utilization(baselines)
        
        # Performance distribution
        performance_distribution = self._analyze_performance_distribution(baselines)
        
        report.performance_overview = {
            'metrics_statistics': metrics_stats,
            'resource_analysis': resource_analysis,
            'performance_distribution': performance_distribution,
            'baseline_quality': {
                'completeness_score': self._calculate_completeness_score(baselines),
                'consistency_score': self._calculate_consistency_score(baselines)
            }
        }

    async def _generate_trend_analysis(
        self,
        report: PerformanceReport,
        baselines: List[BaselineMetrics],
        comparison_baselines: List[BaselineMetrics]
    ) -> None:
        """Generate trend analysis section."""
        trends = {}
        
        # Analyze trends for key metrics
        key_metrics = ['response_time', 'error_rate', 'throughput', 'cpu_utilization', 'memory_utilization']
        
        for metric_name in key_metrics:
            try:
                # Calculate trend over the report period
                trend = await self.analyzer.analyze_trend(
                    metric_name, 
                    baselines, 
                    timeframe_hours=len(baselines)
                )
                
                trends[metric_name] = {
                    'direction': trend.direction.value,
                    'magnitude': trend.magnitude,
                    'confidence_score': trend.confidence_score,
                    'is_significant': trend.is_significant(),
                    'sample_count': trend.sample_count,
                    'predicted_24h': trend.predicted_value_24h,
                    'predicted_7d': trend.predicted_value_7d
                }
            except Exception as e:
                logger.error(f"Failed to analyze trend for {metric_name}: {e}")
                trends[metric_name] = {'error': str(e)}
        
        # Identify most significant trends
        significant_trends = [
            (metric, data) for metric, data in trends.items()
            if isinstance(data, dict) and data.get('is_significant', False)
        ]
        
        # Sort by magnitude
        significant_trends.sort(key=lambda x: abs(x[1].get('magnitude', 0)), reverse=True)
        
        report.trend_analysis = {
            'all_trends': trends,
            'significant_trends': dict(significant_trends[:5]),  # Top 5
            'trend_summary': {
                'improving_metrics': len([t for t in trends.values() if isinstance(t, dict) and t.get('direction') == 'improving']),
                'degrading_metrics': len([t for t in trends.values() if isinstance(t, dict) and t.get('direction') == 'degrading']),
                'stable_metrics': len([t for t in trends.values() if isinstance(t, dict) and t.get('direction') == 'stable'])
            }
        }

    async def _generate_capacity_planning(
        self,
        report: PerformanceReport,
        baselines: List[BaselineMetrics]
    ) -> None:
        """Generate capacity planning section."""
        if not baselines:
            return
        
        # Analyze resource trends
        resource_forecasts = await self._forecast_resource_usage(baselines)
        
        # Calculate capacity recommendations
        capacity_recommendations = self._calculate_capacity_recommendations(baselines, resource_forecasts)
        
        # Growth analysis
        growth_analysis = self._analyze_growth_patterns(baselines)
        
        report.capacity_planning = {
            'resource_forecasts': resource_forecasts,
            'capacity_recommendations': capacity_recommendations,
            'growth_analysis': growth_analysis,
            'scaling_recommendations': self._generate_scaling_recommendations(resource_forecasts)
        }

    async def _generate_alerts_summary(
        self,
        report: PerformanceReport,
        time_period: Tuple[datetime, datetime]
    ) -> None:
        """Generate alerts summary section."""
        # Get alerts for the period
        active_alerts = self.detector.get_active_alerts()
        alert_stats = self.detector.get_alert_statistics()
        
        # Categorize alerts by severity
        alerts_by_severity = {
            'critical': 0,
            'warning': 0,
            'info': 0
        }
        
        alerts_by_metric = {}
        
        for alert in active_alerts:
            severity = alert.severity.value
            if severity in alerts_by_severity:
                alerts_by_severity[severity] += 1
            
            metric = alert.metric_name
            if metric not in alerts_by_metric:
                alerts_by_metric[metric] = 0
            alerts_by_metric[metric] += 1
        
        # Most problematic metrics
        top_problem_metrics = sorted(
            alerts_by_metric.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        report.alerts_summary = {
            'total_active_alerts': len(active_alerts),
            'alerts_by_severity': alerts_by_severity,
            'alerts_by_metric': alerts_by_metric,
            'top_problem_metrics': dict(top_problem_metrics),
            'alert_statistics': alert_stats
        }

    async def _generate_recommendations(
        self,
        report: PerformanceReport,
        baselines: List[BaselineMetrics]
    ) -> None:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Analyze current performance against SLA targets
        sla_violations = self._identify_sla_violations(baselines)
        
        for metric, violation_data in sla_violations.items():
            recommendation = {
                'id': str(uuid.uuid4()),
                'category': 'sla_compliance',
                'priority': 'high' if violation_data['severity'] > 0.5 else 'medium',
                'title': f"Address {metric.replace('_', ' ').title()} SLA Violations",
                'description': f"The {metric} metric is exceeding SLA targets by {violation_data['excess_percentage']:.1f}%",
                'action_items': self._generate_sla_action_items(metric, violation_data),
                'expected_impact': violation_data['impact_assessment'],
                'implementation_effort': 'medium'
            }
            recommendations.append(recommendation)
        
        # Capacity-based recommendations
        capacity_recs = self._generate_capacity_recommendations_detailed(baselines)
        recommendations.extend(capacity_recs)
        
        # Performance optimization recommendations
        optimization_recs = self._generate_optimization_recommendations(baselines)
        recommendations.extend(optimization_recs)
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        report.recommendations = recommendations
        
        # Update executive summary
        report.executive_summary['recommendations_count'] = len(recommendations)

    async def _generate_charts(self, report: PerformanceReport, baselines: List[BaselineMetrics]) -> None:
        """Generate performance charts."""
        if not MATPLOTLIB_AVAILABLE or not baselines:
            return
        
        charts_dir = self.config.output_directory / "charts" / report.report_id
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Time series data
        timestamps = [b.collection_timestamp for b in baselines]
        
        # Response time chart
        if any(b.response_times for b in baselines):
            chart_path = await self._create_time_series_chart(
                timestamps,
                [statistics.mean(b.response_times) if b.response_times else 0 for b in baselines],
                "Response Time Trend",
                "Response Time (ms)",
                str(charts_dir / f"response_time.{self.config.chart_format}")
            )
            report.charts['response_time'] = str(chart_path)
        
        # Resource utilization chart
        cpu_data = [statistics.mean(b.cpu_utilization) if b.cpu_utilization else 0 for b in baselines]
        memory_data = [statistics.mean(b.memory_utilization) if b.memory_utilization else 0 for b in baselines]
        
        if any(cpu_data) or any(memory_data):
            chart_path = await self._create_dual_line_chart(
                timestamps,
                cpu_data,
                memory_data,
                "Resource Utilization",
                "CPU Usage (%)",
                "Memory Usage (%)",
                str(charts_dir / f"resource_utilization.{self.config.chart_format}")
            )
            report.charts['resource_utilization'] = str(chart_path)
        
        # Error rate chart
        error_data = [statistics.mean(b.error_rates) if b.error_rates else 0 for b in baselines]
        if any(error_data):
            chart_path = await self._create_time_series_chart(
                timestamps,
                error_data,
                "Error Rate Trend",
                "Error Rate (%)",
                str(charts_dir / f"error_rate.{self.config.chart_format}")
            )
            report.charts['error_rate'] = str(chart_path)

    async def _create_time_series_chart(
        self,
        timestamps: List[datetime],
        values: List[float],
        title: str,
        ylabel: str,
        output_path: str
    ) -> Path:
        """Create a time series chart."""
        plt.figure(figsize=(12, 6), dpi=self.config.chart_dpi)
        plt.plot(timestamps, values, linewidth=2, marker='o', markersize=3)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel('Time', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, format=self.config.chart_format, dpi=self.config.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return Path(output_path)

    async def _create_dual_line_chart(
        self,
        timestamps: List[datetime],
        values1: List[float],
        values2: List[float],
        title: str,
        ylabel1: str,
        ylabel2: str,
        output_path: str
    ) -> Path:
        """Create a dual-line chart."""
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=self.config.chart_dpi)
        
        # First line
        color1 = 'tab:blue'
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel(ylabel1, color=color1, fontsize=12)
        ax1.plot(timestamps, values1, color=color1, linewidth=2, marker='o', markersize=3, label=ylabel1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Second line
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(ylabel2, color=color2, fontsize=12)
        ax2.plot(timestamps, values2, color=color2, linewidth=2, marker='s', markersize=3, label=ylabel2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, format=self.config.chart_format, dpi=self.config.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return Path(output_path)

    async def _save_report(self, report: PerformanceReport) -> None:
        """Save report in configured formats."""
        report_dir = self.config.output_directory / report.report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON version
        if 'json' in self.config.report_formats:
            json_path = report_dir / "report.json"
            with open(json_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            report.data_files['json'] = str(json_path)
        
        # Save HTML version
        if 'html' in self.config.report_formats:
            html_path = report_dir / "report.html"
            html_content = self._generate_html_report(report)
            with open(html_path, 'w') as f:
                f.write(html_content)
            report.data_files['html'] = str(html_path)
        
        # Save text version
        if 'text' in self.config.report_formats:
            text_path = report_dir / "report.txt"
            text_content = self._generate_text_report(report)
            with open(text_path, 'w') as f:
                f.write(text_content)
            report.data_files['text'] = str(text_path)

    def _generate_html_report(self, report: PerformanceReport) -> str:
        """Generate HTML version of the report."""
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template('performance_report.html')
                return template.render(report=report)
            except Exception as e:
                logger.warning(f"Failed to use Jinja template: {e}")
        
        # Fallback HTML generation
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Report - {report.report_type.title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .metric {{ margin: 10px 0; }}
        .alert {{ color: red; font-weight: bold; }}
        .good {{ color: green; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Performance Report - {report.report_type.title()}</h1>
    <p><strong>Report ID:</strong> {report.report_id}</p>
    <p><strong>Generated:</strong> {report.generation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    <p><strong>Period:</strong> {report.time_period[0].strftime('%Y-%m-%d %H:%M')} to {report.time_period[1].strftime('%Y-%m-%d %H:%M')}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric"><strong>Performance Grade:</strong> {report.executive_summary.get('performance_grade', 'N/A')}</div>
    <div class="metric"><strong>Active Issues:</strong> {report.executive_summary.get('active_issues', {}).get('total_alerts', 0)} alerts</div>
    
    <h2>Performance Overview</h2>
    <!-- Performance metrics would be displayed here -->
    
    <h2>Recommendations</h2>
    <ul>
    {''.join(f'<li><strong>{rec["title"]}:</strong> {rec["description"]}</li>' for rec in report.recommendations)}
    </ul>
</body>
</html>
"""
        return html

    def _generate_text_report(self, report: PerformanceReport) -> str:
        """Generate text version of the report."""
        lines = [
            f"PERFORMANCE REPORT - {report.report_type.upper()}",
            "=" * 60,
            f"Report ID: {report.report_id}",
            f"Generated: {report.generation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Period: {report.time_period[0].strftime('%Y-%m-%d %H:%M')} to {report.time_period[1].strftime('%Y-%m-%d %H:%M')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
            f"Performance Grade: {report.executive_summary.get('performance_grade', 'N/A')}",
            f"Active Issues: {report.executive_summary.get('active_issues', {}).get('total_alerts', 0)} alerts",
            "",
            "RECOMMENDATIONS",
            "-" * 15
        ]
        
        for i, rec in enumerate(report.recommendations, 1):
            lines.extend([
                f"{i}. {rec['title']}",
                f"   Priority: {rec['priority'].upper()}",
                f"   Description: {rec['description']}",
                ""
            ])
        
        return "\
".join(lines)

    # Helper methods for calculations and analysis
    
    def _aggregate_baseline_metrics(self, baselines: List[BaselineMetrics]) -> Dict[str, List[float]]:
        """Aggregate metrics from all baselines."""
        aggregated = {
            'response_time': [],
            'error_rate': [],
            'throughput': [],
            'cpu_utilization': [],
            'memory_utilization': []
        }
        
        for baseline in baselines:
            aggregated['response_time'].extend(baseline.response_times)
            aggregated['error_rate'].extend(baseline.error_rates)
            aggregated['throughput'].extend(baseline.throughput_values)
            aggregated['cpu_utilization'].extend(baseline.cpu_utilization)
            aggregated['memory_utilization'].extend(baseline.memory_utilization)
        
        return aggregated
    
    def _calculate_sla_compliance(self, baselines: List[BaselineMetrics]) -> Dict[str, Any]:
        """Calculate SLA compliance metrics."""
        compliance = {}
        aggregated = self._aggregate_baseline_metrics(baselines)
        
        for metric_name, target in self.config.performance_sla_targets.items():
            if metric_name.replace('_p95', '') in aggregated:
                base_metric = metric_name.replace('_p95', '')
                values = aggregated[base_metric]
                
                if values:
                    if metric_name.endswith('_p95'):
                        current_value = self._percentile(values, 95)
                    else:
                        current_value = statistics.mean(values)
                    
                    # Calculate compliance (percentage of time within SLA)
                    if metric_name in ['response_time_p95', 'error_rate', 'cpu_utilization', 'memory_utilization']:
                        compliant_values = [v for v in values if v <= target]
                    else:  # availability and other "higher is better" metrics
                        compliant_values = [v for v in values if v >= target]
                    
                    compliance_percentage = (len(compliant_values) / len(values)) * 100
                    
                    compliance[metric_name] = {
                        'target': target,
                        'current_value': current_value,
                        'compliance_percentage': compliance_percentage,
                        'is_compliant': compliance_percentage >= 95.0  # 95% SLA threshold
                    }
        
        return compliance
    
    async def _calculate_key_trends(self, baselines: List[BaselineMetrics], comparison_baselines: List[BaselineMetrics]) -> Dict[str, str]:
        """Calculate key performance trends."""
        trends = {}
        
        key_metrics = ['response_time', 'error_rate', 'cpu_utilization', 'memory_utilization']
        
        for metric in key_metrics:
            try:
                trend = await self.analyzer.analyze_trend(metric, baselines, len(baselines))
                trends[metric] = trend.direction.value
            except Exception:
                trends[metric] = 'unknown'
        
        return trends
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
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
    
    def _calculate_baseline_frequency(self, baselines: List[BaselineMetrics]) -> float:
        """Calculate average frequency of baselines in hours."""
        if len(baselines) < 2:
            return 0.0
        
        time_deltas = []
        for i in range(1, len(baselines)):
            delta = baselines[i].collection_timestamp - baselines[i-1].collection_timestamp
            time_deltas.append(delta.total_seconds() / 3600)  # Convert to hours
        
        return statistics.mean(time_deltas) if time_deltas else 0.0
    
    # Additional helper methods would be implemented here...
    # (For brevity, including placeholder implementations)
    
    def _analyze_resource_utilization(self, baselines: List[BaselineMetrics]) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        return {'placeholder': 'resource_analysis'}
    
    def _analyze_performance_distribution(self, baselines: List[BaselineMetrics]) -> Dict[str, Any]:
        """Analyze performance distribution."""
        return {'placeholder': 'performance_distribution'}
    
    def _calculate_completeness_score(self, baselines: List[BaselineMetrics]) -> float:
        """Calculate data completeness score."""
        return 95.0  # Placeholder
    
    def _calculate_consistency_score(self, baselines: List[BaselineMetrics]) -> float:
        """Calculate data consistency score."""
        return 90.0  # Placeholder
    
    async def _forecast_resource_usage(self, baselines: List[BaselineMetrics]) -> Dict[str, Any]:
        """Forecast future resource usage."""
        return {'placeholder': 'resource_forecasts'}
    
    def _calculate_capacity_recommendations(self, baselines: List[BaselineMetrics], forecasts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate capacity recommendations."""
        return []
    
    def _analyze_growth_patterns(self, baselines: List[BaselineMetrics]) -> Dict[str, Any]:
        """Analyze growth patterns."""
        return {'placeholder': 'growth_analysis'}
    
    def _generate_scaling_recommendations(self, forecasts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scaling recommendations."""
        return []
    
    def _identify_sla_violations(self, baselines: List[BaselineMetrics]) -> Dict[str, Any]:
        """Identify SLA violations."""
        return {}
    
    def _generate_sla_action_items(self, metric: str, violation_data: Dict[str, Any]) -> List[str]:
        """Generate action items for SLA violations."""
        return []
    
    def _generate_capacity_recommendations_detailed(self, baselines: List[BaselineMetrics]) -> List[Dict[str, Any]]:
        """Generate detailed capacity recommendations."""
        return []
    
    def _generate_optimization_recommendations(self, baselines: List[BaselineMetrics]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        return []
    
    async def _add_detailed_capacity_analysis(self, report: PerformanceReport, forecast_days: int) -> None:
        """Add detailed capacity analysis to the report."""
        pass

# Global reporter instance
_global_reporter: Optional[BaselineReporter] = None

def get_baseline_reporter() -> BaselineReporter:
    """Get the global baseline reporter instance."""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = BaselineReporter()
    return _global_reporter

def set_baseline_reporter(reporter: BaselineReporter) -> None:
    """Set the global baseline reporter instance."""
    global _global_reporter
    _global_reporter = reporter

# Convenience functions
async def generate_daily_performance_report(date: Optional[datetime] = None) -> PerformanceReport:
    """Generate a daily performance report."""
    reporter = get_baseline_reporter()
    return await reporter.generate_daily_report(date)

async def generate_weekly_performance_report(week_start: Optional[datetime] = None) -> PerformanceReport:
    """Generate a weekly performance report."""
    reporter = get_baseline_reporter()
    return await reporter.generate_weekly_report(week_start)

async def generate_capacity_planning_report(forecast_days: Optional[int] = None) -> PerformanceReport:
    """Generate a capacity planning report."""
    reporter = get_baseline_reporter()
    return await reporter.generate_capacity_planning_report(forecast_days)