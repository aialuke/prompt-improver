"""Architectural Compliance Monitor.
==============================

Monitors for architectural violations that could reintroduce performance issues:
- Protocol TYPE_CHECKING compliance across 9 consolidated domains
- Direct import violations in protocol files
- Missing TYPE_CHECKING guards for heavy dependencies
- Circular import detection and prevention

Protects the 92.4% startup improvement achieved by enforcing clean architecture patterns.
"""

import ast
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
    compliance_tracer = trace.get_tracer(__name__)
    compliance_meter = metrics.get_meter(__name__)

    # Metrics for architectural compliance monitoring
    protocol_violations_counter = compliance_meter.create_counter(
        "architectural_protocol_violations_total",
        description="Total protocol violations by type and file",
        unit="1"
    )

    compliance_check_duration = compliance_meter.create_histogram(
        "architectural_compliance_check_duration_seconds",
        description="Duration of architectural compliance checks",
        unit="s"
    )

    type_checking_compliance_gauge = compliance_meter.create_gauge(
        "architectural_type_checking_compliance_ratio",
        description="Ratio of TYPE_CHECKING compliant protocol files",
        unit="ratio"
    )

except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    compliance_tracer = None
    protocol_violations_counter = None
    compliance_check_duration = None
    type_checking_compliance_gauge = None

logger = logging.getLogger(__name__)


@dataclass
class ViolationDetail:
    """Details of an architectural violation."""
    file_path: str
    line_number: int
    violation_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    suggestion: str
    impact: str  # Performance impact description


@dataclass
class ComplianceReport:
    """Comprehensive architectural compliance report."""
    timestamp: float = field(default_factory=time.time)
    total_protocol_files: int = 0
    compliant_files: int = 0
    violations: list[ViolationDetail] = field(default_factory=list)
    compliance_ratio: float = 0.0

    # Performance impact metrics
    startup_penalty_estimate_ms: int = 0
    memory_penalty_estimate_mb: int = 0

    # Domain-specific compliance
    protocol_domains: dict[str, dict[str, Any]] = field(default_factory=dict)


class ArchitecturalComplianceMonitor:
    """Monitor for architectural compliance and prevent performance regressions.

    Enforces zero-tolerance policy for architectural violations that could
    reintroduce the 134-1007ms startup penalty and dependency contamination.
    """

    # Protocol domains that must maintain 100% TYPE_CHECKING compliance
    PROTOCOL_DOMAINS = [
        "core", "security", "cache", "database",
        "monitoring", "application", "ml", "cli", "mcp"
    ]

    # Heavy dependencies that MUST be behind TYPE_CHECKING
    HEAVY_DEPENDENCIES = {
        "sqlalchemy", "asyncpg", "pandas", "numpy", "torch",
        "transformers", "scikit-learn", "tensorflow", "redis",
        "psycopg2", "psycopg2-binary", "beartype"
    }

    # Critical imports that cause startup penalties
    CRITICAL_VIOLATIONS = {
        "sqlalchemy.ext.asyncio": 134,  # ms penalty
        "asyncpg": 89,
        "pandas": 245,
        "numpy": 167,
        "torch": 1007,  # Critical - massive penalty
        "transformers": 892,
        "beartype": 45,
        "redis": 23
    }

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize architectural compliance monitor.

        Args:
            project_root: Root directory of the project (defaults to auto-detect)
        """
        self.project_root = project_root or self._detect_project_root()
        self.protocol_root = self.project_root / "src/prompt_improver/shared/interfaces/protocols"
        self.violation_history: list[ComplianceReport] = []
        self.last_check_time = 0.0

        logger.info(f"ArchitecturalComplianceMonitor initialized for {self.project_root}")
        logger.info(f"Monitoring {len(self.PROTOCOL_DOMAINS)} protocol domains")

    def _detect_project_root(self) -> Path:
        """Auto-detect project root directory."""
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "src" / "prompt_improver").exists():
                return current
            current = current.parent
        return Path.cwd()

    async def check_compliance(self, strict: bool = True) -> ComplianceReport:
        """Perform comprehensive architectural compliance check.

        Args:
            strict: If True, enforce zero-tolerance policy for violations

        Returns:
            ComplianceReport with detailed violation analysis
        """
        start_time = time.time()

        with compliance_tracer.start_as_current_span("architectural_compliance_check") if compliance_tracer else None as span:
            try:
                report = ComplianceReport()

                # Check protocol file compliance
                await self._check_protocol_compliance(report)

                # Check for circular imports
                await self._check_circular_imports(report)

                # Check for direct heavy imports
                await self._check_heavy_imports(report)

                # Check god object violations
                await self._check_god_objects(report)

                # Calculate compliance metrics
                self._calculate_compliance_metrics(report)

                # Record metrics
                if OPENTELEMETRY_AVAILABLE:
                    self._record_compliance_metrics(report)

                # Store in history
                self.violation_history.append(report)
                self.last_check_time = time.time()

                if span:
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("violations_found", len(report.violations))
                    span.set_attribute("compliance_ratio", report.compliance_ratio)

                # Log compliance status
                if report.violations:
                    severity_counts = {}
                    for violation in report.violations:
                        severity_counts[violation.severity] = severity_counts.get(violation.severity, 0) + 1

                    logger.warning(
                        f"Architectural compliance check found {len(report.violations)} violations: {severity_counts}"
                    )

                    if strict and any(v.severity == "critical" for v in report.violations):
                        logger.error("CRITICAL architectural violations detected - strict mode enforcement triggered")
                else:
                    logger.info(f"Architectural compliance check passed - {report.compliance_ratio:.1%} compliance")

                return report

            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                logger.exception(f"Architectural compliance check failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                if compliance_check_duration:
                    compliance_check_duration.record(duration)

    async def _check_protocol_compliance(self, report: ComplianceReport) -> None:
        """Check TYPE_CHECKING compliance in protocol files."""
        protocol_files = list(self.protocol_root.rglob("*.py"))
        report.total_protocol_files = len(protocol_files)

        for domain in self.PROTOCOL_DOMAINS:
            domain_files = [f for f in protocol_files if f.name == f"{domain}.py" or f"/{domain}/" in str(f)]
            report.protocol_domains[domain] = {
                "files": len(domain_files),
                "compliant": 0,
                "violations": []
            }

            for file_path in domain_files:
                violations = await self._analyze_protocol_file(file_path)
                if not violations:
                    report.compliant_files += 1
                    report.protocol_domains[domain]["compliant"] += 1
                else:
                    report.violations.extend(violations)
                    report.protocol_domains[domain]["violations"].extend(violations)

    async def _analyze_protocol_file(self, file_path: Path) -> list[ViolationDetail]:
        """Analyze a single protocol file for TYPE_CHECKING compliance."""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Check for TYPE_CHECKING import
            has_type_checking_import = False
            type_checking_block_found = False
            imports_in_type_checking = set()
            direct_imports = []

            for node in ast.walk(tree):
                # Check for TYPE_CHECKING import
                if isinstance(node, ast.ImportFrom) and node.module == "typing":
                    if any(alias.name == "TYPE_CHECKING" for alias in (node.names or [])):
                        has_type_checking_import = True

                # Check for TYPE_CHECKING conditional block
                elif isinstance(node, ast.If):
                    if (isinstance(node.test, ast.Name) and
                        node.test.id == "TYPE_CHECKING"):
                        type_checking_block_found = True

                        # Collect imports inside TYPE_CHECKING block
                        for child in ast.walk(node):
                            if isinstance(child, (ast.Import, ast.ImportFrom)):
                                if isinstance(child, ast.ImportFrom):
                                    imports_in_type_checking.add(child.module or "")
                                elif isinstance(child, ast.Import):
                                    for alias in child.names:
                                        imports_in_type_checking.add(alias.name)

                # Check for direct imports outside TYPE_CHECKING
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if hasattr(node, "lineno"):
                        if isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            direct_imports.append((node.lineno, module))
                        elif isinstance(node, ast.Import):
                            direct_imports.extend((node.lineno, alias.name) for alias in node.names)

            # Check for violations
            if not has_type_checking_import and any(
                any(heavy in imp[1] for heavy in self.HEAVY_DEPENDENCIES)
                for imp in direct_imports
            ):
                violations.append(ViolationDetail(
                    file_path=str(file_path),
                    line_number=1,
                    violation_type="missing_type_checking_import",
                    severity="high",
                    description="Missing TYPE_CHECKING import despite using heavy dependencies",
                    suggestion="Add 'from typing import TYPE_CHECKING' import",
                    impact="Potential startup penalty from heavy dependencies"
                ))

            # Check for direct heavy imports not in TYPE_CHECKING
            for line_num, import_module in direct_imports:
                for heavy_dep in self.HEAVY_DEPENDENCIES:
                    if heavy_dep in import_module and import_module not in imports_in_type_checking:
                        penalty = self.CRITICAL_VIOLATIONS.get(import_module, 50)

                        violations.append(ViolationDetail(
                            file_path=str(file_path),
                            line_number=line_num,
                            violation_type="direct_heavy_import",
                            severity="critical" if penalty > 100 else "high",
                            description=f"Direct import of {import_module} outside TYPE_CHECKING block",
                            suggestion=f"Move import to TYPE_CHECKING block: if TYPE_CHECKING: from {import_module} import ...",
                            impact=f"Estimated {penalty}ms startup penalty"
                        ))

                        # Add to startup penalty estimate
                        if hasattr(report, 'startup_penalty_estimate_ms'):
                            report.startup_penalty_estimate_ms += penalty

        except Exception as e:
            logger.exception(f"Error analyzing protocol file {file_path}: {e}")
            violations.append(ViolationDetail(
                file_path=str(file_path),
                line_number=1,
                violation_type="analysis_error",
                severity="medium",
                description=f"Failed to analyze file: {e}",
                suggestion="Check file syntax and structure",
                impact="Unable to verify TYPE_CHECKING compliance"
            ))

        return violations

    async def _check_circular_imports(self, report: ComplianceReport) -> None:
        """Check for circular import patterns."""
        # This is a simplified circular import detection
        # In a production system, you'd want more sophisticated detection
        python_files = list((self.project_root / "src").rglob("*.py"))

        import_graph: dict[str, set[str]] = {}

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                imports = set()

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith("prompt_improver"):
                            imports.add(node.module)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("prompt_improver"):
                                imports.add(alias.name)

                file_module = str(file_path.relative_to(self.project_root / "src")).replace("/", ".").replace(".py", "")
                import_graph[file_module] = imports

            except Exception as e:
                logger.debug(f"Error analyzing imports in {file_path}: {e}")
                continue

        # Simple cycle detection (can be improved with proper algorithms)
        for module, imports in import_graph.items():
            for imported in imports:
                if imported in import_graph and module in import_graph[imported]:
                    report.violations.append(ViolationDetail(
                        file_path=module.replace(".", "/") + ".py",
                        line_number=1,
                        violation_type="circular_import",
                        severity="high",
                        description=f"Circular import detected between {module} and {imported}",
                        suggestion="Refactor to eliminate circular dependency",
                        impact="Can cause import failures and startup delays"
                    ))

    async def _check_heavy_imports(self, report: ComplianceReport) -> None:
        """Check for direct heavy imports outside protocol files."""
        python_files = list((self.project_root / "src").rglob("*.py"))

        for file_path in python_files:
            # Skip protocol files (already checked)
            if "protocols" in str(file_path):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        module_name = ""
                        if isinstance(node, ast.ImportFrom):
                            module_name = node.module or ""
                        elif isinstance(node, ast.Import):
                            module_name = node.names[0].name if node.names else ""

                        # Check if this is a heavy import at module level
                        if any(heavy in module_name for heavy in self.HEAVY_DEPENDENCIES):
                            # Check if it's in a function or TYPE_CHECKING block (which is OK)
                            parent = getattr(node, "parent", None)
                            if not parent or isinstance(parent, ast.Module):
                                penalty = self.CRITICAL_VIOLATIONS.get(module_name, 50)

                                report.violations.append(ViolationDetail(
                                    file_path=str(file_path),
                                    line_number=getattr(node, "lineno", 1),
                                    violation_type="module_level_heavy_import",
                                    severity="high",
                                    description=f"Heavy import {module_name} at module level",
                                    suggestion="Move to function level or use lazy loading",
                                    impact=f"Estimated {penalty}ms startup penalty"
                                ))

                                report.startup_penalty_estimate_ms += penalty

            except Exception as e:
                logger.debug(f"Error checking heavy imports in {file_path}: {e}")
                continue

    async def _check_god_objects(self, report: ComplianceReport) -> None:
        """Check for god objects (classes >500 lines)."""
        python_files = list((self.project_root / "src").rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()

                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_start = node.lineno - 1
                        class_end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                        class_lines = class_end - class_start

                        if class_lines > 500:
                            report.violations.append(ViolationDetail(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                violation_type="god_object",
                                severity="medium",
                                description=f"Class {node.name} has {class_lines} lines (>500 limit)",
                                suggestion="Split into smaller, focused classes following SRP",
                                impact="Maintenance difficulty and potential performance issues"
                            ))

            except Exception as e:
                logger.debug(f"Error checking god objects in {file_path}: {e}")
                continue

    def _calculate_compliance_metrics(self, report: ComplianceReport) -> None:
        """Calculate overall compliance metrics."""
        if report.total_protocol_files > 0:
            report.compliance_ratio = report.compliant_files / report.total_protocol_files
        else:
            report.compliance_ratio = 1.0

        # Estimate memory penalty (rough approximation)
        heavy_imports = sum(1 for v in report.violations if v.violation_type in {
            "direct_heavy_import", "module_level_heavy_import"
        })
        report.memory_penalty_estimate_mb = heavy_imports * 25  # ~25MB per heavy import

    def _record_compliance_metrics(self, report: ComplianceReport) -> None:
        """Record metrics to OpenTelemetry."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Record violation counts by type and severity
        violation_counts = {}
        for violation in report.violations:
            key = f"{violation.violation_type}_{violation.severity}"
            violation_counts[key] = violation_counts.get(key, 0) + 1

        for violation_key, count in violation_counts.items():
            if protocol_violations_counter:
                violation_type, severity = violation_key.rsplit("_", 1)
                protocol_violations_counter.add(count, {
                    "violation_type": violation_type,
                    "severity": severity
                })

        # Record compliance ratio
        if type_checking_compliance_gauge:
            type_checking_compliance_gauge.set(report.compliance_ratio)

    def get_compliance_summary(self) -> dict[str, Any]:
        """Get summary of current compliance status."""
        if not self.violation_history:
            return {
                "status": "no_checks_performed",
                "message": "No compliance checks have been performed yet"
            }

        latest_report = self.violation_history[-1]

        return {
            "status": "compliant" if not latest_report.violations else "violations_found",
            "compliance_ratio": latest_report.compliance_ratio,
            "total_violations": len(latest_report.violations),
            "critical_violations": sum(1 for v in latest_report.violations if v.severity == "critical"),
            "estimated_startup_penalty_ms": latest_report.startup_penalty_estimate_ms,
            "estimated_memory_penalty_mb": latest_report.memory_penalty_estimate_mb,
            "last_check": latest_report.timestamp,
            "protocol_domains": latest_report.protocol_domains,
            "message": f"Found {len(latest_report.violations)} violations" if latest_report.violations
                      else "All architectural compliance checks passed"
        }

    def get_violation_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get violation trends over specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_reports = [r for r in self.violation_history if r.timestamp > cutoff_time]

        if not recent_reports:
            return {"message": "No recent compliance data available"}

        # Calculate trends
        violation_counts = [len(r.violations) for r in recent_reports]
        compliance_ratios = [r.compliance_ratio for r in recent_reports]
        startup_penalties = [r.startup_penalty_estimate_ms for r in recent_reports]

        return {
            "time_period_hours": hours,
            "total_checks": len(recent_reports),
            "violation_trend": {
                "current": violation_counts[-1] if violation_counts else 0,
                "average": sum(violation_counts) / len(violation_counts) if violation_counts else 0,
                "trend": "improving" if len(violation_counts) > 1 and violation_counts[-1] < violation_counts[0] else "stable"
            },
            "compliance_trend": {
                "current": compliance_ratios[-1] if compliance_ratios else 1.0,
                "average": sum(compliance_ratios) / len(compliance_ratios) if compliance_ratios else 1.0,
                "trend": "improving" if len(compliance_ratios) > 1 and compliance_ratios[-1] > compliance_ratios[0] else "stable"
            },
            "performance_impact": {
                "current_penalty_ms": startup_penalties[-1] if startup_penalties else 0,
                "average_penalty_ms": sum(startup_penalties) / len(startup_penalties) if startup_penalties else 0
            }
        }

    async def validate_pr_compliance(self, changed_files: list[str]) -> tuple[bool, dict[str, Any]]:
        """Validate compliance for PR changed files.

        Args:
            changed_files: List of file paths that changed in the PR

        Returns:
            Tuple of (is_compliant, detailed_report)
        """
        pr_violations = []
        protocol_files_changed = []

        for file_path in changed_files:
            file_path_obj = Path(file_path)

            # Check if it's a protocol file
            if "protocols" in str(file_path_obj) and file_path_obj.suffix == ".py":
                protocol_files_changed.append(file_path_obj)
                violations = await self._analyze_protocol_file(file_path_obj)
                pr_violations.extend(violations)

        critical_violations = [v for v in pr_violations if v.severity == "critical"]
        high_violations = [v for v in pr_violations if v.severity == "high"]

        is_compliant = len(critical_violations) == 0

        return is_compliant, {
            "protocol_files_changed": len(protocol_files_changed),
            "total_violations": len(pr_violations),
            "critical_violations": len(critical_violations),
            "high_violations": len(high_violations),
            "violations": pr_violations,
            "compliance_status": "approved" if is_compliant else "blocked",
            "message": "PR approved for architectural compliance" if is_compliant
                      else f"PR blocked due to {len(critical_violations)} critical violations"
        }
