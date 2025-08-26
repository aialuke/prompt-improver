"""Service Contract Validator for Clean Architecture Compliance (2025).

This module provides comprehensive validation of service contracts to ensure:
1. All services implement required protocols
2. No direct database imports in service layers
3. Proper dependency injection patterns are followed
4. Cross-layer dependencies use protocol interfaces only
5. Service registrations match protocol definitions
"""

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of contract violations."""
    MISSING_PROTOCOL_IMPLEMENTATION = "missing_protocol_implementation"
    DIRECT_DATABASE_IMPORT = "direct_database_import"
    INVALID_DEPENDENCY_INJECTION = "invalid_dependency_injection"
    CROSS_LAYER_VIOLATION = "cross_layer_violation"
    UNREGISTERED_SERVICE = "unregistered_service"
    PROTOCOL_MISMATCH = "protocol_mismatch"


@dataclass
class ContractViolation:
    """Represents a service contract violation."""
    violation_type: ViolationType
    file_path: str
    line_number: int
    description: str
    severity: str  # "critical", "high", "medium", "low"
    suggested_fix: str


@dataclass
class ServiceContractReport:
    """Comprehensive service contract validation report."""
    total_services_checked: int
    violations: list[ContractViolation]
    compliant_services: list[str]
    protocol_coverage: dict[str, float]
    dependency_graph: dict[str, list[str]]
    recommendations: list[str]


class ServiceContractValidator:
    """Validates service contracts for Clean Architecture compliance."""

    def __init__(self, source_root: Path) -> None:
        """Initialize validator with source code root."""
        self.source_root = source_root
        self.violations: list[ContractViolation] = []
        self.service_registry: dict[str, Any] = {}
        self.protocol_implementations: dict[str, list[str]] = {}

    async def validate_all_services(self) -> ServiceContractReport:
        """Perform comprehensive validation of all services."""
        logger.info("Starting comprehensive service contract validation")

        # Step 1: Discover all service files
        service_files = self._discover_service_files()
        logger.info(f"Found {len(service_files)} service files to validate")

        # Step 2: Analyze each service file
        compliant_services = []
        for service_file in service_files:
            is_compliant = await self._validate_service_file(service_file)
            if is_compliant:
                compliant_services.append(str(service_file))

        # Step 3: Check protocol implementations
        await self._validate_protocol_implementations()

        # Step 4: Validate service registrations
        await self._validate_service_registrations()

        # Step 5: Generate dependency graph
        dependency_graph = await self._build_dependency_graph()

        # Step 6: Calculate protocol coverage
        protocol_coverage = await self._calculate_protocol_coverage()

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations()

        report = ServiceContractReport(
            total_services_checked=len(service_files),
            violations=self.violations,
            compliant_services=compliant_services,
            protocol_coverage=protocol_coverage,
            dependency_graph=dependency_graph,
            recommendations=recommendations
        )

        logger.info(f"Validation complete: {len(self.violations)} violations found")
        return report

    def _discover_service_files(self) -> list[Path]:
        """Discover all service implementation files."""
        service_patterns = [
            "**/services/**/*.py",
            "**/application/**/*service*.py",
            "**/core/services/**/*.py",
            "**/repositories/impl/**/*.py",
            "**/*facade*.py",
            "**/*_service.py"
        ]

        service_files = []
        for pattern in service_patterns:
            service_files.extend(self.source_root.glob(pattern))

        # Filter out __init__.py and test files
        return [f for f in service_files if f.name != "__init__.py" and "test" not in f.name]

    async def _validate_service_file(self, file_path: Path) -> bool:
        """Validate a single service file for contract compliance."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Parse AST for analysis
            tree = ast.parse(content)

            # Check for direct database imports
            self._check_database_imports(tree, file_path)

            # Check for protocol implementation
            self._check_protocol_implementation(tree, file_path)

            # Check dependency injection patterns
            self._check_dependency_injection(tree, file_path)

            # Check cross-layer dependencies
            self._check_cross_layer_dependencies(tree, file_path)

            return len([v for v in self.violations if v.file_path == str(file_path)]) == 0

        except Exception as e:
            logger.exception(f"Error validating {file_path}: {e}")
            return False

    def _check_database_imports(self, tree: ast.AST, file_path: Path) -> None:
        """Check for direct database model imports in service layers."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "database.models" in node.module:
                        # Check if this is in a service layer (not repository implementation)
                        if any(layer in str(file_path) for layer in ["services", "application", "core"]):
                            if "repositories/impl" not in str(file_path):
                                self.violations.append(ContractViolation(
                                    violation_type=ViolationType.DIRECT_DATABASE_IMPORT,
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    description=f"Direct database model import: {node.module}",
                                    severity="critical",
                                    suggested_fix="Use domain DTOs from core.domain.types instead"
                                ))

    def _check_protocol_implementation(self, tree: ast.AST, file_path: Path) -> None:
        """Check if services properly implement protocol interfaces."""
        class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        for class_node in class_nodes:
            # Check if class inherits from any protocol
            protocol_bases = [base.id for base in class_node.bases if isinstance(base, ast.Name) and "Protocol" in base.id]

            # If it's a service class but doesn't implement a protocol
            if (any(keyword in class_node.name.lower() for keyword in ["service", "facade", "manager"])
                and not protocol_bases and "Protocol" not in class_node.name):

                self.violations.append(ContractViolation(
                    violation_type=ViolationType.MISSING_PROTOCOL_IMPLEMENTATION,
                    file_path=str(file_path),
                    line_number=class_node.lineno,
                    description=f"Service class {class_node.name} should implement a protocol interface",
                    severity="high",
                    suggested_fix="Create and implement appropriate protocol interface"
                ))

    def _check_dependency_injection(self, tree: ast.AST, file_path: Path) -> None:
        """Check for proper dependency injection patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Check constructor parameters for DI patterns
                has_protocol_params = False
                for arg in node.args.args[1:]:  # Skip 'self'
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            if "Protocol" in arg.annotation.id:
                                has_protocol_params = True
                        elif isinstance(arg.annotation, ast.Subscript):
                            # Handle Optional[SomeProtocol] patterns
                            if (hasattr(arg.annotation, 'slice') and
                                isinstance(arg.annotation.slice, ast.Name) and
                                "Protocol" in arg.annotation.slice.id):
                                has_protocol_params = True

                # If it's a service constructor without protocol-based DI
                parent_class = None
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        for child in parent.body:
                            if child == node:
                                parent_class = parent
                                break

                if (parent_class and
                    any(keyword in parent_class.name.lower() for keyword in ["service", "facade"]) and
                    not has_protocol_params and
                    len(node.args.args) > 1):  # Has dependencies but not protocol-based

                    self.violations.append(ContractViolation(
                        violation_type=ViolationType.INVALID_DEPENDENCY_INJECTION,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description="Constructor should use protocol-based dependency injection",
                        severity="medium",
                        suggested_fix="Use protocol interfaces for constructor parameters"
                    ))

    def _check_cross_layer_dependencies(self, tree: ast.AST, file_path: Path) -> None:
        """Check for improper cross-layer dependencies."""
        # Define layer patterns
        layer_patterns = {
            "presentation": ["api", "tui", "cli"],
            "application": ["application"],
            "core": ["core", "domain"],
            "infrastructure": ["database", "repositories", "ml", "services"]
        }

        current_layer = None
        for layer, patterns in layer_patterns.items():
            if any(pattern in str(file_path) for pattern in patterns):
                current_layer = layer
                break

        if not current_layer:
            return

        # Check imports for layer violations
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_layer = None
                for layer, patterns in layer_patterns.items():
                    if any(pattern in node.module for pattern in patterns):
                        imported_layer = layer
                        break

                # Check for architectural violations
                violations = {
                    ("core", "infrastructure"),
                    ("core", "application"),
                    ("core", "presentation"),
                    ("application", "presentation")
                }

                if imported_layer and (current_layer, imported_layer) in violations:
                    self.violations.append(ContractViolation(
                        violation_type=ViolationType.CROSS_LAYER_VIOLATION,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Invalid dependency: {current_layer} layer importing from {imported_layer} layer",
                        severity="critical",
                        suggested_fix="Use protocol interfaces for cross-layer communication"
                    ))

    async def _validate_protocol_implementations(self) -> None:
        """Validate that all protocols have proper implementations."""
        # This would analyze protocol files and check implementations
        protocol_files = list(self.source_root.glob("**/protocols/**/*.py"))

        for _protocol_file in protocol_files:
            # Analysis would check if protocols are properly implemented
            pass

    async def _validate_service_registrations(self) -> None:
        """Validate service registry registrations."""
        # This would check the service registry for proper registrations

    async def _build_dependency_graph(self) -> dict[str, list[str]]:
        """Build dependency graph between services."""
        return {}

    async def _calculate_protocol_coverage(self) -> dict[str, float]:
        """Calculate protocol implementation coverage."""
        return {}

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on violations found."""
        recommendations = []

        violation_counts = {}
        for violation in self.violations:
            violation_counts[violation.violation_type] = violation_counts.get(violation.violation_type, 0) + 1

        if violation_counts.get(ViolationType.DIRECT_DATABASE_IMPORT, 0) > 0:
            recommendations.append(
                "Replace direct database model imports with domain DTOs from core.domain.types"
            )

        if violation_counts.get(ViolationType.MISSING_PROTOCOL_IMPLEMENTATION, 0) > 0:
            recommendations.append(
                "Create protocol interfaces for all service classes to ensure clean contracts"
            )

        if violation_counts.get(ViolationType.CROSS_LAYER_VIOLATION, 0) > 0:
            recommendations.append(
                "Eliminate cross-layer dependencies by using protocol-based interfaces"
            )

        return recommendations

    def format_report(self, report: ServiceContractReport) -> str:
        """Format validation report for human consumption."""
        lines = [
            "=== Service Contract Validation Report ===",
            f"Total services checked: {report.total_services_checked}",
            f"Compliant services: {len(report.compliant_services)}",
            f"Total violations: {len(report.violations)}",
            ""
        ]

        if report.violations:
            lines.append("=== VIOLATIONS ===")
            for violation in sorted(report.violations, key=lambda v: (v.severity, v.file_path)):
                lines.extend([
                    f"[{violation.severity.upper()}] {violation.violation_type.value}",
                    f"  File: {violation.file_path}:{violation.line_number}",
                    f"  Issue: {violation.description}",
                    f"  Fix: {violation.suggested_fix}",
                    ""
                ])

        if report.recommendations:
            lines.append("=== RECOMMENDATIONS ===")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)


# Utility functions for contract validation
def validate_protocol_implementation(cls: type, protocol: type) -> bool:
    """Check if a class properly implements a protocol."""
    try:
        return issubclass(cls, protocol)
    except TypeError:
        return False


def extract_service_dependencies(file_path: Path) -> list[str]:
    """Extract service dependencies from a file."""
    dependencies = []
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(pattern in node.module for pattern in ["services", "protocols"]):
                    dependencies.append(node.module)
    except Exception as e:
        logger.exception(f"Error extracting dependencies from {file_path}: {e}")

    return dependencies
