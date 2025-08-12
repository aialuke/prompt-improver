"""Automated Architectural Compliance Testing Framework

Continuously monitors the codebase for architectural violations and
ensures compliance with defined patterns and principles.
"""

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from analyze_dependencies import ArchitecturalLayers, DependencyAnalyzer


@dataclass
class ComplianceRule:
    """Definition of an architectural compliance rule"""

    name: str
    description: str
    severity: str
    check_function: str
    parameters: dict[str, Any]


@dataclass
class ComplianceViolation:
    """A specific compliance violation"""

    rule_name: str
    severity: str
    message: str
    file_path: str
    line_number: int | None
    details: dict[str, Any]


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""

    timestamp: str
    total_files_analyzed: int
    total_violations: int
    violations_by_severity: dict[str, int]
    violations: list[ComplianceViolation]
    rules_checked: list[str]
    compliance_score: float


class ArchitecturalComplianceChecker:
    """Main compliance checking engine"""

    def __init__(self, src_path: Path, rules_config_path: Path | None = None):
        self.src_path = src_path
        self.dependency_analyzer = DependencyAnalyzer(src_path)
        self.violations: list[ComplianceViolation] = []
        if rules_config_path and rules_config_path.exists():
            with open(rules_config_path) as f:
                rules_data = json.load(f)
                self.rules = [ComplianceRule(**rule) for rule in rules_data["rules"]]
        else:
            self.rules = self._get_default_rules()

    def _get_default_rules(self) -> list[ComplianceRule]:
        """Get default architectural compliance rules"""
        return [
            ComplianceRule(
                name="no_circular_dependencies",
                description="Modules must not have circular dependencies",
                severity="error",
                check_function="check_circular_dependencies",
                parameters={},
            ),
            ComplianceRule(
                name="layer_dependency_direction",
                description="Dependencies must flow in correct architectural direction",
                severity="error",
                check_function="check_layer_dependencies",
                parameters={},
            ),
            ComplianceRule(
                name="high_coupling_limit",
                description="Modules should not have excessive dependencies",
                severity="warning",
                check_function="check_coupling_limits",
                parameters={"max_dependencies": 15},
            ),
            ComplianceRule(
                name="interface_protocol_usage",
                description="High-usage modules should have protocol interfaces",
                severity="info",
                check_function="check_interface_protocols",
                parameters={"min_dependents": 5},
            ),
            ComplianceRule(
                name="security_import_restrictions",
                description="Security modules should not depend on application layers",
                severity="error",
                check_function="check_security_isolation",
                parameters={},
            ),
            ComplianceRule(
                name="test_isolation",
                description="Test modules should not be imported by non-test code",
                severity="error",
                check_function="check_test_isolation",
                parameters={},
            ),
            ComplianceRule(
                name="utility_module_stability",
                description="Utility modules should have minimal external dependencies",
                severity="warning",
                check_function="check_utility_stability",
                parameters={"max_external_deps": 3},
            ),
        ]

    def check_circular_dependencies(
        self, parameters: dict[str, Any]
    ) -> list[ComplianceViolation]:
        """Check for circular dependencies"""
        violations = []
        cycles = self.dependency_analyzer.detect_circular_dependencies()
        for cycle in cycles:
            violations.append(
                ComplianceViolation(
                    rule_name="no_circular_dependencies",
                    severity="error",
                    message=f"Circular dependency detected: {' -> '.join(cycle)}",
                    file_path="",
                    line_number=None,
                    details={"cycle": cycle},
                )
            )
        return violations

    def check_layer_dependencies(
        self, parameters: dict[str, Any]
    ) -> list[ComplianceViolation]:
        """Check architectural layer dependency violations"""
        violations = []
        arch_violations = self.dependency_analyzer.find_architectural_violations()
        for violation in arch_violations:
            if violation["severity"] in ["critical", "error"]:
                violations.append(
                    ComplianceViolation(
                        rule_name="layer_dependency_direction",
                        severity="error",
                        message=f"Layer violation: {violation['from_layer']} -> {violation['to_layer']}",
                        file_path=violation["from_module"],
                        line_number=None,
                        details=violation,
                    )
                )
        return violations

    def check_coupling_limits(
        self, parameters: dict[str, Any]
    ) -> list[ComplianceViolation]:
        """Check for excessive coupling"""
        violations = []
        max_deps = parameters.get("max_dependencies", 15)
        high_coupling = self.dependency_analyzer.identify_high_coupling_modules(
            max_deps
        )
        for module, dep_count in high_coupling:
            violations.append(
                ComplianceViolation(
                    rule_name="high_coupling_limit",
                    severity="warning",
                    message=f"Module has {dep_count} dependencies (limit: {max_deps})",
                    file_path=module,
                    line_number=None,
                    details={"dependency_count": dep_count, "limit": max_deps},
                )
            )
        return violations

    def check_interface_protocols(
        self, parameters: dict[str, Any]
    ) -> list[ComplianceViolation]:
        """Check for missing protocol interfaces"""
        violations = []
        min_dependents = parameters.get("min_dependents", 5)
        for module, dependents in self.dependency_analyzer.reverse_dependencies.items():
            if len(dependents) >= min_dependents:
                protocol_path = self._find_protocol_for_module(module)
                if not protocol_path:
                    violations.append(
                        ComplianceViolation(
                            rule_name="interface_protocol_usage",
                            severity="info",
                            message=f"Module with {len(dependents)} dependents should have protocol interface",
                            file_path=module,
                            line_number=None,
                            details={
                                "dependent_count": len(dependents),
                                "dependents": list(dependents),
                            },
                        )
                    )
        return violations

    def check_security_isolation(
        self, parameters: dict[str, Any]
    ) -> list[ComplianceViolation]:
        """Check security module isolation"""
        violations = []
        for module, deps in self.dependency_analyzer.dependencies.items():
            if ".security." in module:
                module_layer = self.dependency_analyzer.module_to_layer.get(
                    module, "unknown"
                )
                for dep in deps:
                    dep_layer = self.dependency_analyzer.module_to_layer.get(
                        dep, "unknown"
                    )
                    if dep_layer in ["application", "interface", "external"]:
                        violations.append(
                            ComplianceViolation(
                                rule_name="security_import_restrictions",
                                severity="error",
                                message=f"Security module depends on {dep_layer} layer",
                                file_path=module,
                                line_number=None,
                                details={"dependency": dep, "dep_layer": dep_layer},
                            )
                        )
        return violations

    def check_test_isolation(
        self, parameters: dict[str, Any]
    ) -> list[ComplianceViolation]:
        """Check test module isolation"""
        violations = []
        for module, dependents in self.dependency_analyzer.reverse_dependencies.items():
            if ".test_" in module or "tests." in module:
                non_test_dependents = [
                    dep
                    for dep in dependents
                    if not (".test_" in dep or "tests." in dep)
                ]
                if non_test_dependents:
                    violations.append(
                        ComplianceViolation(
                            rule_name="test_isolation",
                            severity="error",
                            message="Test module imported by non-test code",
                            file_path=module,
                            line_number=None,
                            details={"non_test_dependents": non_test_dependents},
                        )
                    )
        return violations

    def check_utility_stability(
        self, parameters: dict[str, Any]
    ) -> list[ComplianceViolation]:
        """Check utility module stability"""
        violations = []
        max_external_deps = parameters.get("max_external_deps", 3)
        for module, deps in self.dependency_analyzer.dependencies.items():
            if ".utils." in module:
                external_deps = [
                    dep
                    for dep in deps
                    if not (
                        dep.startswith("prompt_improver.utils")
                        or dep.startswith("prompt_improver.core")
                    )
                ]
                if len(external_deps) > max_external_deps:
                    violations.append(
                        ComplianceViolation(
                            rule_name="utility_module_stability",
                            severity="warning",
                            message=f"Utility module has {len(external_deps)} external dependencies (limit: {max_external_deps})",
                            file_path=module,
                            line_number=None,
                            details={
                                "external_dependencies": external_deps,
                                "limit": max_external_deps,
                            },
                        )
                    )
        return violations

    def _find_protocol_for_module(self, module: str) -> Path | None:
        """Check if a protocol interface exists for a module"""
        protocol_patterns = [
            f"*{module.split('.')[-1]}_protocol.py",
            f"*{module.split('.')[-1]}Protocol.py",
        ]
        protocol_dir = self.src_path / "prompt_improver" / "core" / "protocols"
        if protocol_dir.exists():
            for pattern in protocol_patterns:
                matches = list(protocol_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None

    def run_all_checks(self) -> ComplianceReport:
        """Run all compliance checks and generate report"""
        self.violations = []
        for rule in self.rules:
            try:
                check_function = getattr(self, rule.check_function)
                rule_violations = check_function(rule.parameters)
                self.violations.extend(rule_violations)
            except AttributeError:
                print(f"Warning: Check function '{rule.check_function}' not found")
            except Exception as e:
                print(f"Error running check '{rule.name}': {e}")
        violations_by_severity = {}
        for violation in self.violations:
            violations_by_severity[violation.severity] = (
                violations_by_severity.get(violation.severity, 0) + 1
            )
        total_files = len(list(self.src_path.rglob("*.py")))
        error_violations = violations_by_severity.get("error", 0)
        warning_violations = violations_by_severity.get("warning", 0)
        penalty = error_violations * 10 + warning_violations * 3
        compliance_score = max(0, 100 - penalty)
        return ComplianceReport(
            timestamp=datetime.now().isoformat(),
            total_files_analyzed=total_files,
            total_violations=len(self.violations),
            violations_by_severity=violations_by_severity,
            violations=self.violations,
            rules_checked=[rule.name for rule in self.rules],
            compliance_score=compliance_score,
        )


def generate_rules_config(output_path: Path):
    """Generate default rules configuration file"""
    checker = ArchitecturalComplianceChecker(Path())
    rules_data = {
        "version": "1.0",
        "description": "Architectural compliance rules for prompt-improver",
        "rules": [asdict(rule) for rule in checker.rules],
    }
    with open(output_path, "w") as f:
        json.dump(rules_data, f, indent=2)
    print(f"Rules configuration generated: {output_path}")


def main():
    """Main entry point for compliance checking"""
    parser = argparse.ArgumentParser(description="Check architectural compliance")
    parser.add_argument(
        "--src-path",
        type=Path,
        default=Path("src/prompt_improver"),
        help="Path to source code directory",
    )
    parser.add_argument(
        "--rules-config", type=Path, help="Path to rules configuration file"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(), help="Output directory for reports"
    )
    parser.add_argument(
        "--format", choices=["json", "text"], default="text", help="Output format"
    )
    parser.add_argument(
        "--generate-rules",
        action="store_true",
        help="Generate default rules configuration",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with error code if violations found",
    )
    parser.add_argument(
        "--min-score", type=float, default=80.0, help="Minimum compliance score (0-100)"
    )
    args = parser.parse_args()
    if args.generate_rules:
        rules_path = args.output_dir / "architectural_rules.json"
        generate_rules_config(rules_path)
        return 0
    args.output_dir.mkdir(exist_ok=True)
    checker = ArchitecturalComplianceChecker(args.src_path, args.rules_config)
    report = checker.run_all_checks()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.format == "json":
        output_file = args.output_dir / f"compliance_report_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"JSON report saved to: {output_file}")
    else:
        output_file = args.output_dir / f"compliance_report_{timestamp}.txt"
        with open(output_file, "w") as f:
            f.write("ARCHITECTURAL COMPLIANCE REPORT\n")
            f.write(f"Generated: {report.timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"COMPLIANCE SCORE: {report.compliance_score:.1f}/100\n")
            f.write(f"Total files analyzed: {report.total_files_analyzed}\n")
            f.write(f"Total violations: {report.total_violations}\n\n")
            if report.violations_by_severity:
                f.write("VIOLATIONS BY SEVERITY:\n")
                for severity, count in report.violations_by_severity.items():
                    f.write(f"  {severity.upper()}: {count}\n")
                f.write("\n")
            if report.violations:
                f.write("DETAILED VIOLATIONS:\n")
                for violation in report.violations:
                    f.write(f"[{violation.severity.upper()}] {violation.rule_name}\n")
                    f.write(f"  File: {violation.file_path}\n")
                    f.write(f"  Message: {violation.message}\n")
                    if violation.details:
                        f.write(f"  Details: {violation.details}\n")
                    f.write("\n")
        print(f"Text report saved to: {output_file}")
    print("\nCOMPLIANCE SUMMARY:")
    print(f"- Compliance Score: {report.compliance_score:.1f}/100")
    print(f"- Total Violations: {report.total_violations}")
    for severity, count in report.violations_by_severity.items():
        print(f"  - {severity.title()}: {count}")
    exit_code = 0
    if args.fail_on_error and report.violations_by_severity.get("error", 0) > 0:
        print("FAILURE: Error-level violations found")
        exit_code = 1
    if report.compliance_score < args.min_score:
        print(
            f"FAILURE: Compliance score {report.compliance_score:.1f} below minimum {args.min_score}"
        )
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
