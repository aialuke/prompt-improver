"""VS Code Diagnostics Integration.
==============================

Real-time integration with VS Code via MCP IDE server for architectural
compliance monitoring and performance regression prevention.

Provides:
- Real-time diagnostics feedback in VS Code
- Immediate violation detection during development
- Integration with existing IDE diagnostics
- Live compliance status updates
- Performance regression alerts in editor

Prevents regressions by providing immediate feedback when developers
introduce architectural violations that could reintroduce performance penalties.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from prompt_improver.monitoring.regression.architectural_compliance import (
    ArchitecturalComplianceMonitor,
    ViolationDetail,
)
from prompt_improver.monitoring.regression.regression_prevention import (
    RegressionPreventionFramework,
)

try:
    # Import MCP IDE integration if available
    from prompt_improver.mcp_server.tools import getDiagnostics
    MCP_IDE_AVAILABLE = True
except ImportError:
    MCP_IDE_AVAILABLE = False
    getDiagnostics = None
except Exception:
    # Handle any other import issues (like syntax errors in dependencies)
    MCP_IDE_AVAILABLE = False
    getDiagnostics = None

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticInfo:
    """VS Code diagnostic information."""
    uri: str
    line: int
    column: int
    severity: str  # "Error", "Warning", "Information", "Hint"
    message: str
    source: str = "regression-prevention"
    code: str | None = None


@dataclass
class DiagnosticsReport:
    """Report of diagnostics integration status."""
    timestamp: float
    total_files_monitored: int
    diagnostics_found: int
    regression_diagnostics: int
    compliance_diagnostics: int
    performance_diagnostics: int
    diagnostics: list[DiagnosticInfo] = field(default_factory=list)


class VSCodeDiagnosticsMonitor:
    """Monitor VS Code diagnostics for architectural compliance and regressions."""

    # Diagnostic severity mapping
    SEVERITY_MAPPING = {
        "critical": "Error",
        "high": "Error",
        "medium": "Warning",
        "low": "Information"
    }

    # File patterns to monitor
    MONITORED_PATTERNS = [
        "src/prompt_improver/shared/interfaces/protocols/*.py",
        "src/prompt_improver/core/protocols/*.py",
        "src/prompt_improver/database/protocols/*.py",
        "src/prompt_improver/**/*.py"  # All Python files for import analysis
    ]

    def __init__(self) -> None:
        """Initialize VS Code diagnostics monitor."""
        self.compliance_monitor = ArchitecturalComplianceMonitor()
        self.regression_framework = RegressionPreventionFramework()

        # Track diagnostics state
        self.last_diagnostics: dict[str, list[DiagnosticInfo]] = {}
        self.monitoring_active = False

        # Integration status
        self.ide_available = MCP_IDE_AVAILABLE

        logger.info(f"VSCodeDiagnosticsMonitor initialized (IDE available: {self.ide_available})")

    async def get_current_diagnostics(self, uri: str | None = None) -> dict[str, Any]:
        """Get current VS Code diagnostics.

        Args:
            uri: Optional specific file URI to get diagnostics for

        Returns:
            Dictionary of diagnostics by file URI
        """
        if not self.ide_available:
            logger.warning("MCP IDE integration not available")
            return {}

        try:
            # Use MCP IDE integration to get diagnostics
            return await getDiagnostics(uri=uri)

        except Exception as e:
            logger.exception(f"Error getting VS Code diagnostics: {e}")
            return {}

    async def analyze_diagnostics_for_regressions(self, diagnostics: dict[str, Any]) -> DiagnosticsReport:
        """Analyze VS Code diagnostics for potential performance regressions.

        Args:
            diagnostics: VS Code diagnostics data

        Returns:
            Analysis report of regression-related diagnostics
        """
        report = DiagnosticsReport(
            timestamp=asyncio.get_event_loop().time(),
            total_files_monitored=0,
            diagnostics_found=0,
            regression_diagnostics=0,
            compliance_diagnostics=0,
            performance_diagnostics=0
        )

        if not diagnostics:
            return report

        # Analyze diagnostics for regression indicators
        for uri, file_diagnostics in diagnostics.items():
            report.total_files_monitored += 1

            if not isinstance(file_diagnostics, list):
                continue

            for diagnostic in file_diagnostics:
                report.diagnostics_found += 1

                # Check if diagnostic indicates potential regression
                is_regression_related = self._is_regression_related_diagnostic(diagnostic)

                if is_regression_related:
                    report.regression_diagnostics += 1

                    # Categorize the diagnostic
                    if self._is_compliance_diagnostic(diagnostic):
                        report.compliance_diagnostics += 1
                    elif self._is_performance_diagnostic(diagnostic):
                        report.performance_diagnostics += 1

                    # Convert to our diagnostic format
                    diag_info = self._convert_diagnostic(uri, diagnostic)
                    report.diagnostics.append(diag_info)

        return report

    def _is_regression_related_diagnostic(self, diagnostic: dict[str, Any]) -> bool:
        """Check if a diagnostic is related to potential performance regression."""
        message = diagnostic.get("message", "").lower()

        # Keywords that indicate potential regression issues
        regression_keywords = [
            "import", "type_checking", "circular", "protocol",
            "heavy dependency", "startup", "performance",
            "sqlalchemy", "asyncpg", "torch", "numpy", "pandas",
            "beartype", "coredis", "redis"
        ]

        return any(keyword in message for keyword in regression_keywords)

    def _is_compliance_diagnostic(self, diagnostic: dict[str, Any]) -> bool:
        """Check if diagnostic is related to architectural compliance."""
        message = diagnostic.get("message", "").lower()

        compliance_keywords = [
            "type_checking", "protocol", "import", "circular",
            "interface", "compliance"
        ]

        return any(keyword in message for keyword in compliance_keywords)

    def _is_performance_diagnostic(self, diagnostic: dict[str, Any]) -> bool:
        """Check if diagnostic is related to performance."""
        message = diagnostic.get("message", "").lower()

        performance_keywords = [
            "startup", "performance", "slow", "heavy", "dependency",
            "import time", "memory", "optimization"
        ]

        return any(keyword in message for keyword in performance_keywords)

    def _convert_diagnostic(self, uri: str, diagnostic: dict[str, Any]) -> DiagnosticInfo:
        """Convert VS Code diagnostic to our format."""
        # Extract position info
        range_info = diagnostic.get("range", {})
        start_pos = range_info.get("start", {})

        line = start_pos.get("line", 0)
        column = start_pos.get("character", 0)

        # Map severity
        vs_code_severity = diagnostic.get("severity", 1)  # 1=Error, 2=Warning, 3=Info, 4=Hint
        severity_map = {1: "Error", 2: "Warning", 3: "Information", 4: "Hint"}
        severity = severity_map.get(vs_code_severity, "Information")

        return DiagnosticInfo(
            uri=uri,
            line=line,
            column=column,
            severity=severity,
            message=diagnostic.get("message", ""),
            source=diagnostic.get("source", "regression-prevention"),
            code=diagnostic.get("code")
        )

    async def monitor_file_for_violations(self, file_path: str) -> list[ViolationDetail]:
        """Monitor a specific file for architectural violations.

        Args:
            file_path: Path to the file to monitor

        Returns:
            List of architectural violations found
        """
        from pathlib import Path

        try:
            file_path_obj = Path(file_path)

            # Check if this is a protocol file
            if "protocols" in str(file_path_obj):
                return await self.compliance_monitor._analyze_protocol_file(file_path_obj)

            return []

        except Exception as e:
            logger.exception(f"Error monitoring file {file_path}: {e}")
            return []

    async def provide_real_time_feedback(self, file_uri: str) -> dict[str, Any]:
        """Provide real-time feedback for a file being edited.

        Args:
            file_uri: URI of the file being edited

        Returns:
            Real-time feedback including violations and suggestions
        """
        feedback = {
            "uri": file_uri,
            "status": "clean",
            "violations": [],
            "suggestions": [],
            "performance_impact": "none"
        }

        try:
            # Convert URI to file path (simplified)
            file_path = file_uri.replace("file://", "")

            # Check for violations in the file
            violations = await self.monitor_file_for_violations(file_path)

            if violations:
                feedback["status"] = "violations"
                feedback["violations"] = [
                    {
                        "line": v.line_number,
                        "type": v.violation_type,
                        "severity": v.severity,
                        "message": v.description,
                        "suggestion": v.suggestion,
                        "impact": v.impact
                    }
                    for v in violations
                ]

                # Determine overall performance impact
                if any(v.severity == "critical" for v in violations):
                    feedback["performance_impact"] = "critical"
                elif any(v.severity == "high" for v in violations):
                    feedback["performance_impact"] = "high"
                else:
                    feedback["performance_impact"] = "medium"

                # Generate suggestions
                feedback["suggestions"] = self._generate_suggestions(violations)

            return feedback

        except Exception as e:
            logger.exception(f"Error providing real-time feedback for {file_uri}: {e}")
            feedback["status"] = "error"
            feedback["error"] = str(e)
            return feedback

    def _generate_suggestions(self, violations: list[ViolationDetail]) -> list[str]:
        """Generate actionable suggestions based on violations."""
        suggestions = []

        # Group violations by type
        violation_types = {}
        for v in violations:
            if v.violation_type not in violation_types:
                violation_types[v.violation_type] = []
            violation_types[v.violation_type].append(v)

        # Generate type-specific suggestions
        if "direct_heavy_import" in violation_types:
            suggestions.append(
                "Move heavy imports (sqlalchemy, asyncpg, torch, etc.) to TYPE_CHECKING blocks "
                "to prevent startup performance penalties"
            )

        if "missing_type_checking_import" in violation_types:
            suggestions.append(
                "Add 'from typing import TYPE_CHECKING' and wrap heavy imports in TYPE_CHECKING blocks"
            )

        if "circular_import" in violation_types:
            suggestions.append(
                "Refactor circular imports by moving shared types to separate modules "
                "or using TYPE_CHECKING blocks"
            )

        if "god_object" in violation_types:
            suggestions.append(
                "Split large classes (>500 lines) into smaller, focused classes "
                "following Single Responsibility Principle"
            )

        return suggestions

    async def start_real_time_monitoring(self, check_interval_seconds: float = 5.0):
        """Start real-time monitoring of VS Code diagnostics.

        Args:
            check_interval_seconds: How often to check for diagnostic changes
        """
        if not self.ide_available:
            logger.warning("Cannot start real-time monitoring: MCP IDE not available")
            return

        self.monitoring_active = True
        logger.info(f"Starting real-time diagnostics monitoring (interval: {check_interval_seconds}s)")

        while self.monitoring_active:
            try:
                # Get current diagnostics
                current_diagnostics = await self.get_current_diagnostics()

                # Analyze for regressions
                if current_diagnostics:
                    report = await self.analyze_diagnostics_for_regressions(current_diagnostics)

                    if report.regression_diagnostics > 0:
                        logger.info(
                            f"Found {report.regression_diagnostics} regression-related diagnostics "
                            f"across {report.total_files_monitored} files"
                        )

                        # You could integrate with alerting system here
                        await self._handle_regression_diagnostics(report)

                # Update tracking
                self.last_diagnostics = {
                    uri: [self._convert_diagnostic(uri, d) for d in diagnostics]
                    for uri, diagnostics in current_diagnostics.items()
                    if isinstance(diagnostics, list)
                }

                await asyncio.sleep(check_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Real-time monitoring cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in real-time monitoring: {e}")
                await asyncio.sleep(check_interval_seconds)

        self.monitoring_active = False

    async def _handle_regression_diagnostics(self, report: DiagnosticsReport) -> None:
        """Handle detected regression-related diagnostics."""
        # Check if we should trigger a full regression check
        critical_diagnostics = [d for d in report.diagnostics if d.severity == "Error"]

        if critical_diagnostics:
            logger.warning(f"Critical regression diagnostics detected: {len(critical_diagnostics)}")

            # Trigger regression prevention framework check
            try:
                regression_report = await self.regression_framework.check_for_regressions(
                    triggered_by="diagnostics_integration"
                )

                if regression_report.overall_status == "regressions_detected":
                    logger.error("Regression prevention framework confirmed regressions!")

                    # Could integrate with alerting/notification system here
                    await self._send_regression_alert(regression_report, report)

            except Exception as e:
                logger.exception(f"Error triggering regression check: {e}")

    async def _send_regression_alert(self, regression_report, diagnostics_report) -> None:
        """Send alert about detected regressions."""
        # This could integrate with existing alerting infrastructure
        logger.error(
            f"REGRESSION ALERT: Framework health {regression_report.framework_health:.1%}, "
            f"{len(regression_report.alerts)} alerts, "
            f"{diagnostics_report.regression_diagnostics} IDE diagnostics"
        )

        # Integration with existing monitoring system
        if hasattr(self.regression_framework, 'slo_observability') and self.regression_framework.slo_observability:
            try:
                for alert in regression_report.alerts:
                    if alert.severity == "critical":
                        await self.regression_framework.slo_observability.correlate_alert_with_cache(
                            alert_id=f"ide_{alert.alert_id}",
                            service_name="ide_diagnostics",
                            target_name="regression_prevention",
                            alert_severity=alert.severity
                        )
            except Exception as e:
                logger.warning(f"Could not correlate with monitoring system: {e}")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        logger.info("Stopped real-time diagnostics monitoring")

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status."""
        return {
            "monitoring_active": self.monitoring_active,
            "ide_available": self.ide_available,
            "last_diagnostics_count": sum(len(diags) for diags in self.last_diagnostics.values()),
            "monitored_files": list(self.last_diagnostics.keys()),
            "integration_health": "healthy" if self.ide_available else "degraded"
        }
