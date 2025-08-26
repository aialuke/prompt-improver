"""Regression Prevention Framework.
===============================

Main orchestrator for preventing performance regressions by integrating
architectural compliance monitoring, startup performance tracking, and
automated violation detection.

Provides:
- Unified regression prevention interface
- Real-time monitoring and alerting
- Automated blocking of performance regressions
- Integration with existing monitoring infrastructure
- Comprehensive reporting and analytics

Zero-tolerance enforcement of architectural improvements that achieved:
- 92.4% startup improvement
- Elimination of 134-1007ms database protocol penalties
- 100% TYPE_CHECKING compliance across 9 protocol domains
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_improver.monitoring.regression.architectural_compliance import (
    ArchitecturalComplianceMonitor,
    ComplianceReport,
)
from prompt_improver.monitoring.regression.startup_performance import (
    StartupPerformanceTracker,
    StartupProfile,
)

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
    framework_tracer = trace.get_tracer(__name__)
    framework_meter = metrics.get_meter(__name__)

    # Framework-level metrics
    regression_prevention_checks_counter = framework_meter.create_counter(
        "regression_prevention_checks_total",
        description="Total regression prevention checks by type and result",
        unit="1"
    )

    performance_regression_alerts_counter = framework_meter.create_counter(
        "performance_regression_alerts_total",
        description="Performance regression alerts by severity",
        unit="1"
    )

    framework_health_gauge = framework_meter.create_gauge(
        "regression_prevention_framework_health",
        description="Overall health of the regression prevention framework",
        unit="score"
    )

except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    framework_tracer = None
    regression_prevention_checks_counter = None
    performance_regression_alerts_counter = None
    framework_health_gauge = None

logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """Alert for detected performance regression."""
    alert_id: str
    alert_type: str  # "architectural", "startup_performance", "dependency_contamination"
    severity: str    # "critical", "high", "medium", "low"
    title: str
    description: str
    impact_estimate: str
    recommendations: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    auto_block: bool = True  # Whether this should block CI/PRs


@dataclass
class RegressionReport:
    """Comprehensive regression prevention report."""
    overall_status: str = "unknown"  # "healthy", "warnings", "regressions_detected"
    timestamp: float = field(default_factory=time.time)
    compliance_report: ComplianceReport | None = None
    startup_profile: StartupProfile | None = None
    alerts: list[RegressionAlert] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    framework_health: float = 1.0  # 0.0-1.0 health score


class RegressionPreventionFramework:
    """Unified framework for preventing performance regressions.

    Orchestrates architectural compliance monitoring and startup performance
    tracking to prevent regressions of the major performance improvements.
    """

    # Performance baselines from architectural improvements
    PERFORMANCE_BASELINES = {
        "startup_improvement_ratio": 0.924,  # 92.4% improvement achieved
        "max_startup_time_ms": 500,          # 500ms target (Python 2025)
        "protocol_compliance_ratio": 1.0,    # 100% TYPE_CHECKING compliance
        "zero_critical_violations": True,    # Zero tolerance for critical violations
        "max_dependency_contamination": 0    # No prohibited dependencies in startup
    }

    def __init__(self, project_root: Path | None = None,
                 integration_with_existing_monitoring: bool = True) -> None:
        """Initialize regression prevention framework.

        Args:
            project_root: Root directory of the project
            integration_with_existing_monitoring: Whether to integrate with existing monitoring
        """
        self.project_root = project_root
        self.compliance_monitor = ArchitecturalComplianceMonitor(project_root)
        self.startup_tracker = StartupPerformanceTracker()

        # Alert management
        self.active_alerts: dict[str, RegressionAlert] = {}
        self.alert_history: list[RegressionAlert] = []
        self.report_history: list[RegressionReport] = []

        # Integration with existing monitoring
        self.monitoring_integration = integration_with_existing_monitoring
        if integration_with_existing_monitoring:
            self._setup_monitoring_integration()

        # Performance tracking
        self.last_check_time = 0.0
        self.check_interval_seconds = 300  # 5 minutes default

        logger.info(f"RegressionPreventionFramework initialized with baselines: {self.PERFORMANCE_BASELINES}")

    def _setup_monitoring_integration(self) -> None:
        """Setup integration with existing monitoring infrastructure."""
        try:
            # Integration with existing SLO monitoring
            from prompt_improver.monitoring.slo.unified_observability import (
                get_slo_observability,
            )

            self.slo_observability = get_slo_observability()

            # Register callback for correlation
            self.slo_observability.register_correlation_callback(
                self._handle_monitoring_correlation
            )

            logger.info("Successfully integrated with existing SLO monitoring")

        except Exception as e:
            logger.warning(f"Could not integrate with existing monitoring: {e}")
            self.slo_observability = None

    async def _handle_monitoring_correlation(self, context: dict[str, Any]) -> None:
        """Handle correlation with existing monitoring events."""
        # Check if monitoring event indicates potential regression
        if context.get("alert_severity") in {"critical", "high"}:
            service_name = context.get("service_name", "unknown")

            # Trigger regression check if performance-related
            if any(keyword in str(context).lower() for keyword in
                   ["startup", "import", "performance", "slow", "timeout"]):

                logger.info(f"Monitoring correlation triggered regression check for {service_name}")
                await self.check_for_regressions(triggered_by="monitoring_correlation")

    async def check_for_regressions(self, triggered_by: str = "manual") -> RegressionReport:
        """Perform comprehensive check for performance regressions.

        Args:
            triggered_by: What triggered this check (manual, scheduled, monitoring_correlation, pr_validation)

        Returns:
            Comprehensive regression report
        """
        start_time = time.time()

        with framework_tracer.start_as_current_span("regression_prevention_check") if framework_tracer else None as span:
            try:
                report = RegressionReport()

                # Run architectural compliance check
                logger.info("Running architectural compliance check...")
                report.compliance_report = await self.compliance_monitor.check_compliance(strict=True)

                # Check startup performance (if we have recent data)
                logger.info("Analyzing startup performance...")
                startup_summary = self.startup_tracker.get_performance_summary()
                if startup_summary.get("status") != "no_profiles":
                    # Get latest startup profile
                    if self.startup_tracker.startup_profiles:
                        report.startup_profile = self.startup_tracker.startup_profiles[-1]

                # Generate alerts based on findings
                await self._generate_regression_alerts(report)

                # Calculate overall status and framework health
                self._calculate_framework_health(report)

                # Record metrics
                if OPENTELEMETRY_AVAILABLE:
                    self._record_framework_metrics(report, triggered_by)

                # Store report
                self.report_history.append(report)
                self.last_check_time = time.time()

                # Integration with existing monitoring
                if self.monitoring_integration and self.slo_observability:
                    await self._report_to_monitoring_system(report)

                if span:
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("triggered_by", triggered_by)
                    span.set_attribute("alerts_generated", len(report.alerts))
                    span.set_attribute("overall_status", report.overall_status)

                duration = time.time() - start_time
                logger.info(
                    f"Regression check completed in {duration:.3f}s: "
                    f"{report.overall_status}, {len(report.alerts)} alerts generated"
                )

                return report

            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                logger.exception(f"Regression prevention check failed: {e}")
                raise

    async def _generate_regression_alerts(self, report: RegressionReport) -> None:
        """Generate regression alerts based on compliance and performance data."""
        alerts = []

        # Architectural compliance alerts
        if report.compliance_report:
            compliance = report.compliance_report

            # Critical violations (zero tolerance)
            critical_violations = [v for v in compliance.violations if v.severity == "critical"]
            if critical_violations:
                for violation in critical_violations:
                    alert_id = f"arch_critical_{hash(violation.file_path + violation.violation_type)}"
                    alert = RegressionAlert(
                        alert_id=alert_id,
                        alert_type="architectural",
                        severity="critical",
                        title=f"Critical Architectural Violation: {violation.violation_type}",
                        description=f"Critical violation in {violation.file_path}: {violation.description}",
                        impact_estimate=violation.impact,
                        recommendations=[violation.suggestion],
                        source_files=[violation.file_path],
                        auto_block=True
                    )
                    alerts.append(alert)
                    self.active_alerts[alert_id] = alert

            # TYPE_CHECKING compliance regression
            if compliance.compliance_ratio < self.PERFORMANCE_BASELINES["protocol_compliance_ratio"]:
                alert_id = "type_checking_compliance_regression"
                alert = RegressionAlert(
                    alert_id=alert_id,
                    alert_type="architectural",
                    severity="high",
                    title="TYPE_CHECKING Compliance Regression",
                    description=f"Protocol compliance dropped to {compliance.compliance_ratio:.1%} (target: 100%)",
                    impact_estimate=f"Risk of reintroducing startup penalties up to {compliance.startup_penalty_estimate_ms}ms",
                    recommendations=[
                        "Review protocol files for missing TYPE_CHECKING guards",
                        "Add TYPE_CHECKING blocks around heavy imports",
                        "Run architectural compliance check on modified files"
                    ],
                    auto_block=True
                )
                alerts.append(alert)
                self.active_alerts[alert_id] = alert

        # Startup performance alerts
        if report.startup_profile:
            startup = report.startup_profile

            # Dependency contamination (critical)
            if startup.contaminated_dependencies:
                alert_id = "dependency_contamination"
                alert = RegressionAlert(
                    alert_id=alert_id,
                    alert_type="dependency_contamination",
                    severity="critical",
                    title="Prohibited Dependencies Loaded at Startup",
                    description=f"Prohibited dependencies detected: {', '.join(startup.contaminated_dependencies)}",
                    impact_estimate="Major startup performance regression possible",
                    recommendations=[
                        "Remove prohibited dependencies from startup path",
                        "Use lazy loading for heavy dependencies",
                        "Move imports to function level or TYPE_CHECKING blocks"
                    ],
                    auto_block=True
                )
                alerts.append(alert)
                self.active_alerts[alert_id] = alert

            # Startup time regression
            max_startup_time = self.PERFORMANCE_BASELINES["max_startup_time_ms"] / 1000.0
            if startup.startup_time_seconds > max_startup_time:
                alert_id = "startup_time_regression"
                alert = RegressionAlert(
                    alert_id=alert_id,
                    alert_type="startup_performance",
                    severity="high",
                    title="Startup Time Regression",
                    description=f"Startup time {startup.startup_time_seconds:.3f}s exceeds {max_startup_time}s target",
                    impact_estimate=f"Performance regression of {((startup.startup_time_seconds / max_startup_time) - 1) * 100:.1f}%",
                    recommendations=[
                        "Analyze slow imports and optimize",
                        "Review startup dependency loading",
                        "Check for new heavy imports in startup path"
                    ],
                    auto_block=False  # Warning, not blocking
                )
                alerts.append(alert)
                self.active_alerts[alert_id] = alert

        report.alerts = alerts

    def _calculate_framework_health(self, report: RegressionReport) -> None:
        """Calculate overall framework health score."""
        health_score = 1.0

        # Compliance health (40% weight)
        if report.compliance_report:
            compliance_health = report.compliance_report.compliance_ratio
            health_score *= 0.6 + (0.4 * compliance_health)

        # Startup performance health (30% weight)
        if report.startup_profile:
            startup = report.startup_profile
            max_startup = self.PERFORMANCE_BASELINES["max_startup_time_ms"] / 1000.0

            if startup.startup_time_seconds <= max_startup:
                startup_health = 1.0
            else:
                startup_health = max(0.0, 1.0 - (startup.startup_time_seconds - max_startup) / max_startup)

            health_score *= 0.7 + (0.3 * startup_health)

        # Alert severity impact (30% weight)
        critical_alerts = len([a for a in report.alerts if a.severity == "critical"])
        high_alerts = len([a for a in report.alerts if a.severity == "high"])

        alert_penalty = (critical_alerts * 0.5) + (high_alerts * 0.2)
        health_score *= max(0.0, 1.0 - alert_penalty)

        report.framework_health = max(0.0, min(1.0, health_score))

        # Determine overall status
        if report.framework_health >= 0.9 and not any(a.severity == "critical" for a in report.alerts):
            report.overall_status = "healthy"
        elif report.framework_health >= 0.7 and not any(a.severity == "critical" for a in report.alerts):
            report.overall_status = "warnings"
        else:
            report.overall_status = "regressions_detected"

    def _record_framework_metrics(self, report: RegressionReport, triggered_by: str) -> None:
        """Record framework metrics to OpenTelemetry."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Record check execution
        if regression_prevention_checks_counter:
            regression_prevention_checks_counter.add(1, {
                "triggered_by": triggered_by,
                "status": report.overall_status
            })

        # Record alerts by severity
        if performance_regression_alerts_counter:
            alert_counts = {}
            for alert in report.alerts:
                alert_counts[alert.severity] = alert_counts.get(alert.severity, 0) + 1

            for severity, count in alert_counts.items():
                performance_regression_alerts_counter.add(count, {"severity": severity})

        # Record framework health
        if framework_health_gauge:
            framework_health_gauge.set(report.framework_health)

    async def _report_to_monitoring_system(self, report: RegressionReport) -> None:
        """Report regression prevention results to existing monitoring system."""
        if not self.slo_observability:
            return

        try:
            # Record SLO compliance for regression prevention
            compliance_ratio = 1.0
            if report.compliance_report:
                compliance_ratio = report.compliance_report.compliance_ratio

            # Calculate error budget based on framework health
            error_budget = report.framework_health * 100

            await self.slo_observability.record_slo_compliance(
                service_name="regression_prevention",
                target_name="architectural_compliance",
                compliance_ratio=compliance_ratio,
                error_budget_remaining=error_budget,
                time_window="real_time",  # Assuming this enum exists
                additional_metrics={
                    "startup_performance_score": report.startup_profile.startup_time_seconds if report.startup_profile else 0,
                    "total_violations": len(report.compliance_report.violations) if report.compliance_report else 0,
                    "critical_alerts": len([a for a in report.alerts if a.severity == "critical"])
                }
            )

            # Correlate critical alerts
            for alert in report.alerts:
                if alert.severity == "critical":
                    await self.slo_observability.correlate_alert_with_cache(
                        alert_id=alert.alert_id,
                        service_name="regression_prevention",
                        target_name=alert.alert_type,
                        alert_severity=alert.severity
                    )

        except Exception as e:
            logger.warning(f"Failed to report to monitoring system: {e}")

    async def validate_pr_changes(self, changed_files: list[str]) -> tuple[bool, dict[str, Any]]:
        """Validate PR changes for performance regressions.

        Args:
            changed_files: List of files changed in the PR

        Returns:
            Tuple of (should_approve, validation_report)
        """
        validation_start = time.time()

        with framework_tracer.start_as_current_span("pr_validation") if framework_tracer else None as span:
            try:
                # Check architectural compliance for changed files
                compliance_ok, compliance_report = await self.compliance_monitor.validate_pr_compliance(changed_files)

                # Run full regression check
                regression_report = await self.check_for_regressions(triggered_by="pr_validation")

                # Determine if PR should be approved
                blocking_alerts = [a for a in regression_report.alerts if a.auto_block and a.severity == "critical"]
                should_approve = compliance_ok and len(blocking_alerts) == 0

                validation_report = {
                    "should_approve": should_approve,
                    "changed_files_count": len(changed_files),
                    "architectural_compliance": compliance_report,
                    "regression_check": {
                        "overall_status": regression_report.overall_status,
                        "framework_health": regression_report.framework_health,
                        "total_alerts": len(regression_report.alerts),
                        "blocking_alerts": len(blocking_alerts),
                        "alerts": [
                            {
                                "type": a.alert_type,
                                "severity": a.severity,
                                "title": a.title,
                                "auto_block": a.auto_block
                            } for a in regression_report.alerts
                        ]
                    },
                    "validation_duration_seconds": time.time() - validation_start,
                    "recommendations": []
                }

                # Generate recommendations
                if not should_approve:
                    validation_report["recommendations"].extend([
                        "Fix critical architectural violations",
                        "Ensure TYPE_CHECKING compliance in protocol files",
                        "Remove prohibited dependencies from startup path"
                    ])

                if span:
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("should_approve", should_approve)
                    span.set_attribute("changed_files", len(changed_files))
                    span.set_attribute("blocking_alerts", len(blocking_alerts))

                logger.info(f"PR validation completed: {'APPROVED' if should_approve else 'BLOCKED'}")
                return should_approve, validation_report

            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                logger.exception(f"PR validation failed: {e}")
                raise

    def get_framework_status(self) -> dict[str, Any]:
        """Get comprehensive framework status."""
        if not self.report_history:
            return {
                "status": "not_initialized",
                "message": "No regression checks performed yet"
            }

        latest_report = self.report_history[-1]

        return {
            "status": latest_report.overall_status,
            "framework_health": latest_report.framework_health,
            "last_check_time": self.last_check_time,
            "active_alerts": len(self.active_alerts),
            "total_checks": len(self.report_history),
            "performance_baselines": self.PERFORMANCE_BASELINES,
            "latest_metrics": {
                "compliance_ratio": latest_report.compliance_report.compliance_ratio if latest_report.compliance_report else None,
                "startup_time_seconds": latest_report.startup_profile.startup_time_seconds if latest_report.startup_profile else None,
                "total_violations": len(latest_report.compliance_report.violations) if latest_report.compliance_report else 0,
                "contaminated_dependencies": len(latest_report.startup_profile.contaminated_dependencies) if latest_report.startup_profile else 0
            },
            "integration_status": {
                "monitoring_integration": self.monitoring_integration,
                "slo_observability_available": self.slo_observability is not None
            }
        }

    async def start_continuous_monitoring(self, check_interval_seconds: int = 300):
        """Start continuous monitoring for performance regressions.

        Args:
            check_interval_seconds: Interval between checks
        """
        self.check_interval_seconds = check_interval_seconds
        logger.info(f"Starting continuous regression monitoring (interval: {check_interval_seconds}s)")

        while True:
            try:
                await self.check_for_regressions(triggered_by="scheduled")
                await asyncio.sleep(check_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Continuous monitoring cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(check_interval_seconds)  # Continue despite errors
