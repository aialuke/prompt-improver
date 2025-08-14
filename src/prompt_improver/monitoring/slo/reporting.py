"""SLO/SLA Reporting and Dashboard Generation
=========================================

Implements comprehensive reporting capabilities for SLO compliance, SLA tracking,
dashboard generation, and executive-level reporting with business impact analysis.
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_improver.utils.datetime_utils import (
    format_compact_timestamp,
    format_date_only,
    format_display_date,
)
from prompt_improver.monitoring.slo.monitor import SLOMonitor

try:
    import jinja2

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats"""

    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportPeriod(Enum):
    """Standard reporting periods"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class SLOComplianceReport:
    """SLO compliance report data structure"""

    service_name: str
    slo_name: str
    report_period: ReportPeriod
    start_date: datetime
    end_date: datetime
    overall_compliance_percentage: float
    target_compliance_percentage: float
    compliance_status: str
    window_compliance: dict[str, dict[str, Any]]
    error_budget_consumed_percentage: float
    error_budget_remaining: float
    error_budget_status: str
    trend_direction: str
    trend_confidence: float
    incident_count: int
    alert_count: int
    mean_time_to_recovery: float
    recommendations: list[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "SLO Monitoring System"


@dataclass
class SLABreachReport:
    """SLA breach report for customer impact analysis"""

    service_name: str
    customer_id: str | None
    breach_start: datetime
    breach_end: datetime | None
    sla_target: str
    target_value: float
    actual_value: float
    breach_duration_minutes: float
    affected_requests: int
    estimated_revenue_impact: float
    customer_impact_level: str
    time_to_detection_minutes: float
    time_to_resolution_minutes: float | None
    root_cause: str | None
    remediation_actions: list[str]
    customer_notified: bool
    compensation_required: bool
    postmortem_required: bool


class SLOReporter:
    """Generate comprehensive SLO compliance reports"""

    def __init__(self, template_dir: str | None = None, output_dir: str | None = None):
        self.template_dir = (
            Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        )
        self.output_dir = Path(output_dir) if output_dir else Path("./reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if JINJA2_AVAILABLE:
            try:
                self.jinja_env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(str(self.template_dir)),
                    autoescape=jinja2.select_autoescape(["html", "xml"]),
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Jinja2 environment: {e}")
                self.jinja_env = None
        else:
            self.jinja_env = None

    async def generate_compliance_report(
        self,
        slo_monitor: SLOMonitor,
        period: ReportPeriod,
        end_date: datetime | None = None,
    ) -> SLOComplianceReport:
        """Generate comprehensive SLO compliance report"""
        end_date = end_date or datetime.now(UTC)
        start_date = self._calculate_period_start(end_date, period)
        evaluation_results = await slo_monitor.evaluate_slos()
        compliance_data = self._calculate_compliance_metrics(
            evaluation_results, start_date, end_date
        )
        trend_data = await self._analyze_compliance_trends(
            slo_monitor, start_date, end_date
        )
        incident_data = self._analyze_incidents_and_alerts(
            slo_monitor, start_date, end_date
        )
        recommendations = self._generate_recommendations(
            compliance_data, trend_data, incident_data
        )
        report = SLOComplianceReport(
            service_name=slo_monitor.slo_definition.service_name,
            slo_name=slo_monitor.slo_definition.name,
            report_period=period,
            start_date=start_date,
            end_date=end_date,
            overall_compliance_percentage=compliance_data["overall_compliance"],
            target_compliance_percentage=compliance_data["target_compliance"],
            compliance_status=compliance_data["status"],
            window_compliance=compliance_data["window_results"],
            error_budget_consumed_percentage=compliance_data["budget_consumed"],
            error_budget_remaining=compliance_data["budget_remaining"],
            error_budget_status=compliance_data["budget_status"],
            trend_direction=trend_data["direction"],
            trend_confidence=trend_data["confidence"],
            incident_count=incident_data["incident_count"],
            alert_count=incident_data["alert_count"],
            mean_time_to_recovery=incident_data["mttr"],
            recommendations=recommendations,
        )
        return report

    def _calculate_period_start(
        self, end_date: datetime, period: ReportPeriod
    ) -> datetime:
        """Calculate start date for reporting period"""
        if period == ReportPeriod.DAILY:
            return end_date - timedelta(days=1)
        if period == ReportPeriod.WEEKLY:
            return end_date - timedelta(weeks=1)
        if period == ReportPeriod.MONTHLY:
            return end_date - timedelta(days=30)
        if period == ReportPeriod.QUARTERLY:
            return end_date - timedelta(days=90)
        if period == ReportPeriod.YEARLY:
            return end_date - timedelta(days=365)
        return end_date - timedelta(days=1)

    def _calculate_compliance_metrics(
        self,
        evaluation_results: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Calculate compliance metrics from evaluation results"""
        slo_results = evaluation_results.get("slo_results", {})
        error_budget_status = evaluation_results.get("error_budget_status", {})
        compliance_ratios = []
        window_results = {}
        for target_name, target_results in slo_results.items():
            window_data = target_results.get("window_results", {})
            window_results[target_name] = window_data
            primary_window = window_data.get("24h") or window_data.get("7d")
            if primary_window:
                compliance_ratios.append(primary_window["compliance_ratio"])
        overall_compliance = (
            statistics.mean(compliance_ratios) * 100 if compliance_ratios else 0.0
        )
        if overall_compliance >= 99.0:
            status = "meeting"
        elif overall_compliance >= 95.0:
            status = "at_risk"
        else:
            status = "breaching"
        budgets = error_budget_status.get("budgets", {})
        budget_consumed = 0.0
        budget_remaining = 100.0
        if budgets:
            primary_budget = None
            for budget_key, budget_data in budgets.items():
                if "24h" in budget_key or "7d" in budget_key:
                    primary_budget = budget_data
                    break
            if primary_budget:
                budget_consumed = primary_budget.get("budget_percentage", 0.0)
                budget_remaining = 100.0 - budget_consumed
        budget_status = "healthy"
        if budget_consumed > 90:
            budget_status = "exhausted"
        elif budget_consumed > 75:
            budget_status = "critical"
        elif budget_consumed > 50:
            budget_status = "warning"
        return {
            "overall_compliance": overall_compliance,
            "target_compliance": 99.0,
            "status": status,
            "window_results": window_results,
            "budget_consumed": budget_consumed,
            "budget_remaining": budget_remaining,
            "budget_status": budget_status,
        }

    async def _analyze_compliance_trends(
        self, slo_monitor: SLOMonitor, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Analyze compliance trends over time"""
        if slo_monitor.calculators:
            calculator = next(iter(slo_monitor.calculators.values()))
            trends = calculator.analyze_trends(end_date)
            return {
                "direction": trends.get("trend_direction", "unknown"),
                "confidence": trends.get("trend_confidence", 0.0),
                "slope": trends.get("trend_slope", 0.0),
            }
        return {"direction": "unknown", "confidence": 0.0, "slope": 0.0}

    def _analyze_incidents_and_alerts(
        self, slo_monitor: SLOMonitor, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Analyze incidents and alerts for the period"""
        total_alerts = 0
        total_incidents = 0
        recovery_times = []
        for alerter in slo_monitor.burn_rate_alerts.values():
            active_alerts = alerter.get_active_alerts()
            total_alerts += len(active_alerts)
            for alert in active_alerts:
                if alert.severity.value in ["critical", "emergency"]:
                    total_incidents += 1
                    if alert.resolved_at:
                        recovery_time = (
                            alert.resolved_at - alert.started_at
                        ).total_seconds() / 60
                        recovery_times.append(recovery_time)
        mttr = statistics.mean(recovery_times) if recovery_times else 0.0
        return {
            "incident_count": total_incidents,
            "alert_count": total_alerts,
            "mttr": mttr,
        }

    def _generate_recommendations(
        self,
        compliance_data: dict[str, Any],
        trend_data: dict[str, Any],
        incident_data: dict[str, Any],
    ) -> list[str]:
        """Generate actionable recommendations based on report data"""
        recommendations = []
        if compliance_data["overall_compliance"] < 95.0:
            recommendations.append(
                "SLO compliance is below target. Consider reviewing error handling, improving monitoring, or adjusting SLO targets."
            )
        if compliance_data["budget_consumed"] > 75:
            recommendations.append(
                "Error budget consumption is high. Implement stricter change management and consider freezing non-essential deployments."
            )
        if trend_data["direction"] == "degrading" and trend_data["confidence"] > 0.7:
            recommendations.append(
                "Performance is showing a degrading trend. Investigate potential causes such as increased load, recent changes, or infrastructure issues."
            )
        if incident_data["incident_count"] > 3:
            recommendations.append(
                "High incident count detected. Review alerting thresholds, implement better monitoring, and conduct postmortem reviews."
            )
        if incident_data["mttr"] > 60:
            recommendations.append(
                "Mean time to recovery is high. Improve incident response procedures, automation, and on-call training."
            )
        if not recommendations:
            recommendations.append(
                "SLO performance is within acceptable parameters. Continue monitoring and maintain current operational practices."
            )
        return recommendations

    def export_report(
        self,
        report: SLOComplianceReport,
        format: ReportFormat,
        filename: str | None = None,
    ) -> str:
        """Export report in specified format"""
        if not filename:
            timestamp = format_compact_timestamp(report.generated_at)
            filename = f"slo_report_{report.service_name}_{timestamp}.{format.value}"
        filepath = self.output_dir / filename
        if format == ReportFormat.JSON:
            return self._export_json(report, filepath)
        if format == ReportFormat.HTML:
            return self._export_html(report, filepath)
        if format == ReportFormat.MARKDOWN:
            return self._export_markdown(report, filepath)
        if format == ReportFormat.CSV:
            return self._export_csv(report, filepath)
        raise ValueError(f"Unsupported report format: {format}")

    def _export_json(self, report: SLOComplianceReport, filepath: Path) -> str:
        """Export report as JSON"""
        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        return str(filepath)

    def _export_html(self, report: SLOComplianceReport, filepath: Path) -> str:
        """Export report as HTML"""
        if not self.jinja_env:
            html_content = self._generate_simple_html(report)
        else:
            try:
                template = self.jinja_env.get_template("slo_report.html")
                html_content = template.render(report=report)
            except Exception as e:
                logger.warning(f"Failed to use HTML template: {e}")
                html_content = self._generate_simple_html(report)
        with open(filepath, "w") as f:
            f.write(html_content)
        return str(filepath)

    def _export_markdown(self, report: SLOComplianceReport, filepath: Path) -> str:
        """Export report as Markdown"""
        md_content = f"# SLO Compliance Report\n\n## Service: {report.service_name}\n**Report Period:** {report.report_period.value}  \n**Period:** {format_date_only(report.start_date)} to {format_date_only(report.end_date)}  \n**Generated:** {format_display_date(report.generated_at)}\n\n## Summary\n- **Overall Compliance:** {report.overall_compliance_percentage:.2f}%\n- **Target Compliance:** {report.target_compliance_percentage:.2f}%\n- **Status:** {report.compliance_status}\n- **Error Budget Consumed:** {report.error_budget_consumed_percentage:.2f}%\n- **Error Budget Remaining:** {report.error_budget_remaining:.2f}%\n\n## Trends\n- **Direction:** {report.trend_direction}\n- **Confidence:** {report.trend_confidence:.2f}\n\n## Incidents & Alerts\n- **Incidents:** {report.incident_count}\n- **Alerts:** {report.alert_count}\n- **Mean Time to Recovery:** {report.mean_time_to_recovery:.1f} minutes\n\n## Recommendations\n"
        for i, rec in enumerate(report.recommendations, 1):
            md_content += f"{i}. {rec}\n"
        with open(filepath, "w") as f:
            f.write(md_content)
        return str(filepath)

    def _export_csv(self, report: SLOComplianceReport, filepath: Path) -> str:
        """Export report as CSV"""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value", "Unit", "Status"])
            writer.writerow(["Service", report.service_name, "", ""])
            writer.writerow(["SLO Name", report.slo_name, "", ""])
            writer.writerow([
                "Overall Compliance",
                f"{report.overall_compliance_percentage:.2f}",
                "%",
                report.compliance_status,
            ])
            writer.writerow([
                "Error Budget Consumed",
                f"{report.error_budget_consumed_percentage:.2f}",
                "%",
                report.error_budget_status,
            ])
            writer.writerow(["Incidents", str(report.incident_count), "count", ""])
            writer.writerow(["Alerts", str(report.alert_count), "count", ""])
            writer.writerow([
                "MTTR",
                f"{report.mean_time_to_recovery:.1f}",
                "minutes",
                "",
            ])
        return str(filepath)

    def _generate_simple_html(self, report: SLOComplianceReport) -> str:
        """Generate simple HTML report without templates"""
        return f"""\n<!DOCTYPE html>\n<html>\n<head>\n    <title>SLO Compliance Report - {report.service_name}</title>\n    <style>\n        body {{ font-family: Arial, sans-serif; margin: 40px; }}\n        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}\n        .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}\n        .status-meeting {{ border-left-color: #28a745; }}\n        .status-at-risk {{ border-left-color: #ffc107; }}\n        .status-breaching {{ border-left-color: #dc3545; }}\n        .recommendations {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; }}\n    </style>\n</head>\n<body>\n    <div class="header">\n        <h1>SLO Compliance Report</h1>\n        <h2>{report.service_name}</h2>\n        <p><strong>Period:</strong> {format_date_only(report.start_date)} to {format_date_only(report.end_date)}</p>\n        <p><strong>Generated:</strong> {format_display_date(report.generated_at)}</p>\n    </div>\n    \n    <div class="metric status-{report.compliance_status}">\n        <h3>Overall Compliance: {report.overall_compliance_percentage:.2f}%</h3>\n        <p>Status: {report.compliance_status.upper()}</p>\n    </div>\n    \n    <div class="metric">\n        <h3>Error Budget</h3>\n        <p>Consumed: {report.error_budget_consumed_percentage:.2f}%</p>\n        <p>Remaining: {report.error_budget_remaining:.2f}%</p>\n        <p>Status: {report.error_budget_status}</p>\n    </div>\n    \n    <div class="metric">\n        <h3>Incidents & Alerts</h3>\n        <p>Incidents: {report.incident_count}</p>\n        <p>Alerts: {report.alert_count}</p>\n        <p>Mean Time to Recovery: {report.mean_time_to_recovery:.1f} minutes</p>\n    </div>\n    \n    <div class="recommendations">\n        <h3>Recommendations</h3>\n        <ul>\n            {"".join(f"<li>{rec}</li>" for rec in report.recommendations)}\n        </ul>\n    </div>\n</body>\n</html>\n"""


class SLAReporter:
    """Generate SLA breach reports and customer impact analysis"""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./sla_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.breach_history: list[SLABreachReport] = []

    def record_sla_breach(
        self,
        service_name: str,
        sla_target: str,
        target_value: float,
        actual_value: float,
        breach_start: datetime,
        customer_id: str | None = None,
        affected_requests: int = 0,
        estimated_revenue_impact: float = 0.0,
    ) -> SLABreachReport:
        """Record a new SLA breach"""
        impact_level = self._calculate_impact_level(
            affected_requests, estimated_revenue_impact
        )
        breach_report = SLABreachReport(
            service_name=service_name,
            customer_id=customer_id,
            breach_start=breach_start,
            breach_end=None,
            sla_target=sla_target,
            target_value=target_value,
            actual_value=actual_value,
            breach_duration_minutes=0.0,
            affected_requests=affected_requests,
            estimated_revenue_impact=estimated_revenue_impact,
            customer_impact_level=impact_level,
            time_to_detection_minutes=0.0,
            time_to_resolution_minutes=None,
            root_cause=None,
            remediation_actions=[],
            customer_notified=False,
            compensation_required=impact_level in ["high", "critical"],
            postmortem_required=impact_level in ["high", "critical"],
        )
        self.breach_history.append(breach_report)
        logger.warning(f"SLA breach recorded: {service_name}/{sla_target}")
        return breach_report

    def resolve_sla_breach(
        self,
        breach_report: SLABreachReport,
        resolution_time: datetime,
        root_cause: str | None = None,
        remediation_actions: list[str] | None = None,
    ) -> None:
        """Resolve an SLA breach"""
        breach_report.breach_end = resolution_time
        breach_report.breach_duration_minutes = (
            resolution_time - breach_report.breach_start
        ).total_seconds() / 60
        if root_cause:
            breach_report.root_cause = root_cause
        if remediation_actions:
            breach_report.remediation_actions = remediation_actions
        breach_report.time_to_resolution_minutes = breach_report.breach_duration_minutes
        logger.info(
            f"SLA breach resolved: {breach_report.service_name}/{breach_report.sla_target}"
        )

    def _calculate_impact_level(
        self, affected_requests: int, revenue_impact: float
    ) -> str:
        """Calculate customer impact level"""
        if revenue_impact > 10000 or affected_requests > 100000:
            return "critical"
        if revenue_impact > 1000 or affected_requests > 10000:
            return "high"
        if revenue_impact > 100 or affected_requests > 1000:
            return "medium"
        return "low"

    def generate_breach_summary(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate SLA breach summary for period"""
        period_breaches = [
            breach
            for breach in self.breach_history
            if start_date <= breach.breach_start <= end_date
        ]
        if not period_breaches:
            return {
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "total_breaches": 0,
                "total_revenue_impact": 0.0,
                "breach_summary": {},
            }
        total_revenue_impact = sum(b.estimated_revenue_impact for b in period_breaches)
        impact_summary = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for breach in period_breaches:
            impact_summary[breach.customer_impact_level] += 1
        service_summary = {}
        for breach in period_breaches:
            if breach.service_name not in service_summary:
                service_summary[breach.service_name] = {
                    "breach_count": 0,
                    "revenue_impact": 0.0,
                    "affected_requests": 0,
                }
            service_summary[breach.service_name]["breach_count"] += 1
            service_summary[breach.service_name]["revenue_impact"] += (
                breach.estimated_revenue_impact
            )
            service_summary[breach.service_name]["affected_requests"] += (
                breach.affected_requests
            )
        resolved_breaches = [b for b in period_breaches if b.breach_end is not None]
        avg_resolution_time = 0.0
        if resolved_breaches:
            resolution_times = [
                b.time_to_resolution_minutes
                for b in resolved_breaches
                if b.time_to_resolution_minutes
            ]
            avg_resolution_time = (
                statistics.mean(resolution_times) if resolution_times else 0.0
            )
        return {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_breaches": len(period_breaches),
            "total_revenue_impact": total_revenue_impact,
            "impact_level_summary": impact_summary,
            "service_summary": service_summary,
            "average_resolution_time_minutes": avg_resolution_time,
            "resolution_rate": len(resolved_breaches) / len(period_breaches)
            if period_breaches
            else 0.0,
        }


class DashboardGenerator:
    """Generate interactive dashboards for SLO/SLA monitoring"""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./dashboards")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_slo_dashboard(
        self, slo_reports: list[SLOComplianceReport], title: str = "SLO Dashboard"
    ) -> str:
        """Generate comprehensive SLO dashboard"""
        dashboard_data = {
            "title": title,
            "generated_at": datetime.now(UTC).isoformat(),
            "services": [],
        }
        for report in slo_reports:
            service_data = {
                "name": report.service_name,
                "slo_name": report.slo_name,
                "compliance_percentage": report.overall_compliance_percentage,
                "compliance_status": report.compliance_status,
                "error_budget_consumed": report.error_budget_consumed_percentage,
                "trend_direction": report.trend_direction,
                "incident_count": report.incident_count,
                "alert_count": report.alert_count,
            }
            dashboard_data["services"].append(service_data)
        html_content = self._generate_dashboard_html(dashboard_data)
        timestamp = format_compact_timestamp(datetime.now(UTC))
        filename = f"slo_dashboard_{timestamp}.html"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(html_content)
        return str(filepath)

    def _generate_dashboard_html(self, data: dict[str, Any]) -> str:
        """Generate HTML dashboard"""
        services_html = ""
        for service in data["services"]:
            status_class = f"status-{service['compliance_status']}"
            trend_icon = {"improving": "↗️", "stable": "→", "degrading": "↘️"}.get(
                service["trend_direction"], "?"
            )
            services_html += f"""\n            <div class="service-card {status_class}">\n                <h3>{service["name"]}</h3>\n                <div class="metric-grid">\n                    <div class="metric">\n                        <div class="metric-value">{service["compliance_percentage"]:.1f}%</div>\n                        <div class="metric-label">Compliance</div>\n                    </div>\n                    <div class="metric">\n                        <div class="metric-value">{service["error_budget_consumed"]:.1f}%</div>\n                        <div class="metric-label">Budget Used</div>\n                    </div>\n                    <div class="metric">\n                        <div class="metric-value">{trend_icon}</div>\n                        <div class="metric-label">Trend</div>\n                    </div>\n                    <div class="metric">\n                        <div class="metric-value">{service["incident_count"]}</div>\n                        <div class="metric-label">Incidents</div>\n                    </div>\n                </div>\n                <div class="status-badge">{service["compliance_status"].upper()}</div>\n            </div>\n            """
        return f"""\n<!DOCTYPE html>\n<html>\n<head>\n    <title>{data["title"]}</title>\n    <meta charset="utf-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1">\n    <style>\n        * {{ margin: 0; padding: 0; box-sizing: border-box; }}\n        body {{ \n            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n            background-color: #f8f9fa;\n            color: #333;\n            line-height: 1.6;\n        }}\n        .header {{\n            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n            color: white;\n            padding: 2rem;\n            text-align: center;\n        }}\n        .header h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}\n        .header p {{ opacity: 0.9; }}\n        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}\n        .services-grid {{\n            display: grid;\n            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));\n            gap: 1.5rem;\n            margin-top: 2rem;\n        }}\n        .service-card {{\n            background: white;\n            border-radius: 12px;\n            padding: 1.5rem;\n            box-shadow: 0 4px 6px rgba(0,0,0,0.1);\n            border-left: 5px solid #ddd;\n            position: relative;\n            transition: transform 0.2s;\n        }}\n        .service-card:hover {{ transform: translateY(-2px); }}\n        .service-card.status-meeting {{ border-left-color: #28a745; }}\n        .service-card.status-at-risk {{ border-left-color: #ffc107; }}\n        .service-card.status-breaching {{ border-left-color: #dc3545; }}\n        .service-card h3 {{ margin-bottom: 1rem; font-size: 1.3rem; }}\n        .metric-grid {{\n            display: grid;\n            grid-template-columns: repeat(2, 1fr);\n            gap: 1rem;\n            margin-bottom: 1rem;\n        }}\n        .metric {{ text-align: center; }}\n        .metric-value {{\n            font-size: 1.5rem;\n            font-weight: bold;\n            color: #2c3e50;\n        }}\n        .metric-label {{\n            font-size: 0.85rem;\n            color: #6c757d;\n            margin-top: 0.25rem;\n        }}\n        .status-badge {{\n            position: absolute;\n            top: 1rem;\n            right: 1rem;\n            padding: 0.25rem 0.75rem;\n            border-radius: 20px;\n            font-size: 0.75rem;\n            font-weight: bold;\n            background-color: #e9ecef;\n            color: #495057;\n        }}\n        .status-meeting .status-badge {{ background-color: #d4edda; color: #155724; }}\n        .status-at-risk .status-badge {{ background-color: #fff3cd; color: #856404; }}\n        .status-breaching .status-badge {{ background-color: #f8d7da; color: #721c24; }}\n        .refresh-info {{\n            text-align: center;\n            margin-top: 2rem;\n            color: #6c757d;\n            font-size: 0.9rem;\n        }}\n    </style>\n</head>\n<body>\n    <div class="header">\n        <h1>{data["title"]}</h1>\n        <p>Real-time SLO monitoring and compliance tracking</p>\n    </div>\n    \n    <div class="container">\n        <div class="services-grid">\n            {services_html}\n        </div>\n        \n        <div class="refresh-info">\n            <p>Last updated: {data["generated_at"]}</p>\n            <p>Dashboard auto-refreshes every 5 minutes</p>\n        </div>\n    </div>\n    \n    <script>\n        // Auto-refresh every 5 minutes\n        setTimeout(() => {{ window.location.reload(); }}, 300000);\n    </script>\n</body>\n</html>\n"""


class ExecutiveReporter:
    """Generate executive-level reports with business impact analysis"""

    def __init__(self):
        self.revenue_per_request = 0.1
        self.sla_penalty_rate = 0.05

    def generate_executive_summary(
        self,
        slo_reports: list[SLOComplianceReport],
        sla_breaches: list[SLABreachReport],
        period: ReportPeriod,
    ) -> dict[str, Any]:
        """Generate executive summary with business metrics"""
        if slo_reports:
            avg_compliance = statistics.mean(
                r.overall_compliance_percentage for r in slo_reports
            )
            total_incidents = sum(r.incident_count for r in slo_reports)
            avg_mttr = statistics.mean(
                r.mean_time_to_recovery
                for r in slo_reports
                if r.mean_time_to_recovery > 0
            )
        else:
            avg_compliance = 0.0
            total_incidents = 0
            avg_mttr = 0.0
        total_revenue_impact = sum(b.estimated_revenue_impact for b in sla_breaches)
        total_affected_requests = sum(b.affected_requests for b in sla_breaches)
        availability_score = min(100, avg_compliance)
        satisfaction_impact = self._calculate_satisfaction_impact(sla_breaches)
        risk_level = self._assess_risk_level(
            avg_compliance, total_incidents, total_revenue_impact
        )
        return {
            "reporting_period": period.value,
            "executive_summary": {
                "overall_service_health": availability_score,
                "total_incidents": total_incidents,
                "average_resolution_time_hours": avg_mttr / 60,
                "revenue_impact_total": total_revenue_impact,
                "affected_customer_requests": total_affected_requests,
                "customer_satisfaction_impact": satisfaction_impact,
                "risk_level": risk_level,
            },
            "key_metrics": {
                "service_availability": f"{avg_compliance:.2f}%",
                "sla_breach_count": len(sla_breaches),
                "revenue_at_risk": f"${total_revenue_impact:,.2f}",
                "operational_excellence_score": self._calculate_operational_score(
                    slo_reports
                ),
            },
            "recommendations": self._generate_executive_recommendations(
                avg_compliance, total_incidents, total_revenue_impact, risk_level
            ),
            "next_period_projections": self._generate_projections(
                slo_reports, sla_breaches
            ),
        }

    def _calculate_satisfaction_impact(self, breaches: list[SLABreachReport]) -> str:
        """Calculate estimated customer satisfaction impact"""
        if not breaches:
            return "minimal"
        critical_breaches = sum(
            1 for b in breaches if b.customer_impact_level == "critical"
        )
        high_breaches = sum(1 for b in breaches if b.customer_impact_level == "high")
        if critical_breaches > 0:
            return "severe"
        if high_breaches > 2:
            return "significant"
        if len(breaches) > 5:
            return "moderate"
        return "low"

    def _assess_risk_level(
        self, compliance: float, incidents: int, revenue_impact: float
    ) -> str:
        """Assess overall business risk level"""
        risk_score = 0
        if compliance < 95:
            risk_score += 3
        elif compliance < 98:
            risk_score += 2
        elif compliance < 99.5:
            risk_score += 1
        if incidents > 10:
            risk_score += 3
        elif incidents > 5:
            risk_score += 2
        elif incidents > 2:
            risk_score += 1
        if revenue_impact > 50000:
            risk_score += 3
        elif revenue_impact > 10000:
            risk_score += 2
        elif revenue_impact > 1000:
            risk_score += 1
        if risk_score >= 7:
            return "critical"
        if risk_score >= 5:
            return "high"
        if risk_score >= 3:
            return "medium"
        return "low"

    def _calculate_operational_score(self, reports: list[SLOComplianceReport]) -> int:
        """Calculate operational excellence score (0-100)"""
        if not reports:
            return 0
        compliance_weight = 0.4
        budget_weight = 0.3
        incident_weight = 0.2
        trend_weight = 0.1
        compliance_score = statistics.mean(
            r.overall_compliance_percentage for r in reports
        )
        budget_scores = [100 - r.error_budget_consumed_percentage for r in reports]
        budget_score = statistics.mean(budget_scores)
        incident_scores = [max(0, 100 - r.incident_count * 10) for r in reports]
        incident_score = statistics.mean(incident_scores)
        improving_trends = sum(1 for r in reports if r.trend_direction == "improving")
        trend_score = improving_trends / len(reports) * 100
        total_score = (
            compliance_score * compliance_weight
            + budget_score * budget_weight
            + incident_score * incident_weight
            + trend_score * trend_weight
        )
        return int(total_score)

    def _generate_executive_recommendations(
        self, compliance: float, incidents: int, revenue_impact: float, risk_level: str
    ) -> list[str]:
        """Generate executive-level recommendations"""
        recommendations = []
        if risk_level in ["critical", "high"]:
            recommendations.append(
                "IMMEDIATE ACTION REQUIRED: Service reliability is below acceptable levels. Recommend emergency review of operational procedures and resource allocation."
            )
        if compliance < 95:
            recommendations.append(
                "Service availability is impacting customer experience. Recommend increased investment in infrastructure reliability and monitoring capabilities."
            )
        if revenue_impact > 10000:
            recommendations.append(
                f"Significant revenue impact (${revenue_impact:,.2f}) from service issues. Consider implementing stricter change management and improved testing procedures."
            )
        if incidents > 5:
            recommendations.append(
                "High incident count indicates systemic issues. Recommend comprehensive review of architecture and implementation of chaos engineering practices."
            )
        if not recommendations:
            recommendations.append(
                "Service performance is meeting business objectives. Continue current operational practices and consider optimization opportunities."
            )
        return recommendations

    def _generate_projections(
        self,
        slo_reports: list[SLOComplianceReport],
        sla_breaches: list[SLABreachReport],
    ) -> dict[str, Any]:
        """Generate projections for next period"""
        if not slo_reports:
            return {"status": "insufficient_data"}
        improving_services = sum(
            1 for r in slo_reports if r.trend_direction == "improving"
        )
        degrading_services = sum(
            1 for r in slo_reports if r.trend_direction == "degrading"
        )
        if improving_services > degrading_services:
            projected_change = "improvement"
        elif degrading_services > improving_services:
            projected_change = "degradation"
        else:
            projected_change = "stable"
        current_compliance = statistics.mean(
            r.overall_compliance_percentage for r in slo_reports
        )
        if projected_change == "improvement":
            projected_compliance = min(100, current_compliance + 2.0)
        elif projected_change == "degradation":
            projected_compliance = max(0, current_compliance - 3.0)
        else:
            projected_compliance = current_compliance
        current_revenue_impact = sum(b.estimated_revenue_impact for b in sla_breaches)
        if projected_change == "improvement":
            projected_revenue_impact = current_revenue_impact * 0.7
        elif projected_change == "degradation":
            projected_revenue_impact = current_revenue_impact * 1.5
        else:
            projected_revenue_impact = current_revenue_impact
        return {
            "trend_direction": projected_change,
            "projected_compliance": f"{projected_compliance:.1f}%",
            "projected_revenue_impact": f"${projected_revenue_impact:,.2f}",
            "confidence": "medium",
            "recommended_actions": [
                "Continue monitoring trends",
                "Review quarterly SLO targets",
                "Assess infrastructure capacity planning",
            ],
        }
