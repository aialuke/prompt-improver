#!/usr/bin/env python3
"""Comprehensive Architectural Validator.

Custom architectural validation script that enforces key architectural boundaries,
serving as a replacement for import-linter when it has internal parsing issues.

This script validates:
1. Circular Import Detection - Check for circular dependencies between modules
2. Clean Architecture Layers - Verify proper layering (Presentation → Application → Domain → Repository → Infrastructure)
3. Repository Pattern Enforcement - No direct database imports in service/presentation layers
4. Protocol Consolidation - No imports from old core.protocols locations
5. Domain Purity - Core/domain layers don't import heavy external libraries

Returns clear pass/fail status with specific file:line locations for violations.
"""

import ast
import asyncio
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of architectural violations."""
    CIRCULAR_IMPORT = "circular_import"
    LAYER_VIOLATION = "layer_violation"
    REPOSITORY_VIOLATION = "repository_violation"
    PROTOCOL_CONSOLIDATION = "protocol_consolidation"
    DOMAIN_PURITY = "domain_purity"
    DIRECT_DATABASE_IMPORT = "direct_database_import"
    HEAVY_IMPORT = "heavy_import"


class Severity(Enum):
    """Violation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Violation:
    """Represents an architectural violation."""
    file_path: str
    line_number: int
    violation_type: ViolationType
    severity: Severity
    message: str
    suggestion: str
    impact: str = ""


@dataclass
class ValidationResult:
    """Comprehensive validation results."""
    total_files_analyzed: int = 0
    violations: List[Violation] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    compliance_ratio: float = 1.0
    analysis_duration: float = 0.0
    
    @property
    def is_compliant(self) -> bool:
        """Returns True if no violations found."""
        return len(self.violations) == 0
    
    @property
    def critical_violations(self) -> List[Violation]:
        """Returns only critical violations."""
        return [v for v in self.violations if v.severity == Severity.CRITICAL]
    
    @property
    def violation_summary(self) -> Dict[str, int]:
        """Returns count of violations by type."""
        summary = defaultdict(int)
        for violation in self.violations:
            summary[violation.violation_type.value] += 1
        return dict(summary)


class ArchitecturalValidator:
    """Comprehensive architectural validator for the prompt improver project."""
    
    # Architectural layer definitions
    LAYER_HIERARCHY = {
        "presentation": {"api", "cli", "tui", "mcp_server"},
        "application": {"application"},
        "domain": {"core/domain", "rule_engine", "ml"},
        "infrastructure": {"database", "services", "monitoring", "performance", "security"},
        "shared": {"shared", "core/common", "core/types"},
        "repository": {"repositories"}
    }
    
    # Layer dependency rules (what each layer can depend on)
    ALLOWED_DEPENDENCIES = {
        "presentation": {"application", "shared", "core"},
        "application": {"domain", "repository", "shared", "core"},
        "domain": {"shared", "core"},
        "repository": {"infrastructure", "shared", "core"},
        "infrastructure": {"shared", "core"},
        "shared": {"core"},
        "core": set()
    }
    
    # Heavy dependencies that should not be imported directly in core/domain
    HEAVY_DEPENDENCIES = {
        "pandas", "numpy", "torch", "transformers", "scikit-learn", 
        "tensorflow", "sqlalchemy", "asyncpg", "psycopg2", "redis",
        "beartype"  # Even beartype can be heavy during startup
    }
    
    # Critical import penalties (in milliseconds)
    IMPORT_PENALTIES = {
        "torch": 1007,
        "transformers": 892,
        "pandas": 245,
        "numpy": 167,
        "sqlalchemy": 134,
        "asyncpg": 89,
        "beartype": 45,
        "redis": 23
    }
    
    # Database imports that violate repository pattern
    DATABASE_IMPORTS = {
        "prompt_improver.database",
        "prompt_improver.database.connection",
        "prompt_improver.database.services",
        "prompt_improver.database.models",
        "prompt_improver.database.query_utils"
    }
    
    def __init__(self, project_root: Path | None = None):
        """Initialize the architectural validator.
        
        Args:
            project_root: Root directory of the project (auto-detected if None)
        """
        self.project_root = project_root or self._detect_project_root()
        self.src_path = self.project_root / "src" / "prompt_improver"
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.file_to_module: Dict[Path, str] = {}
        self.module_to_layer: Dict[str, str] = {}
        
        logger.info(f"Initialized ArchitecturalValidator for {self.project_root}")
    
    def _detect_project_root(self) -> Path:
        """Auto-detect project root directory."""
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "src" / "prompt_improver").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    async def validate_architecture(self) -> ValidationResult:
        """Perform comprehensive architectural validation.
        
        Returns:
            ValidationResult with all violations found
        """
        start_time = time.time()
        logger.info("Starting comprehensive architectural validation...")
        
        result = ValidationResult()
        
        try:
            # Build dependency graph
            await self._build_dependency_graph(result)
            
            # Run all validation checks
            await self._check_circular_imports(result)
            await self._check_layer_violations(result)
            await self._check_repository_pattern(result)
            await self._check_protocol_consolidation(result)
            await self._check_domain_purity(result)
            
            # Calculate final metrics
            result.analysis_duration = time.time() - start_time
            result.compliance_ratio = self._calculate_compliance_ratio(result)
            
            # Log summary
            self._log_validation_summary(result)
            
        except Exception as e:
            logger.exception(f"Architectural validation failed: {e}")
            result.violations.append(Violation(
                file_path="<validation_system>",
                line_number=0,
                violation_type=ViolationType.CIRCULAR_IMPORT,  # Generic type
                severity=Severity.CRITICAL,
                message=f"Validation system error: {e}",
                suggestion="Check validator configuration and project structure"
            ))
        
        return result
    
    async def _build_dependency_graph(self, result: ValidationResult) -> None:
        """Build dependency graph for all Python modules."""
        logger.info("Building dependency graph...")
        
        python_files = list(self.src_path.rglob("*.py"))
        result.total_files_analyzed = len(python_files)
        
        for py_file in python_files:
            if py_file.name.startswith("__"):
                continue
                
            try:
                module_name = self._file_to_module_name(py_file)
                self.file_to_module[py_file] = module_name
                self.module_to_layer[module_name] = self._determine_layer(py_file)
                
                # Parse imports
                imports = await self._parse_file_imports(py_file)
                self.dependency_graph[module_name].update(imports)
                
            except Exception as e:
                logger.debug(f"Error processing {py_file}: {e}")
                continue
    
    async def _parse_file_imports(self, file_path: Path) -> Set[str]:
        """Parse imports from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
                    
        except Exception as e:
            logger.debug(f"Error parsing imports from {file_path}: {e}")
        
        return imports
    
    def _file_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        relative_path = file_path.relative_to(self.project_root / "src")
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        return ".".join(module_parts)
    
    def _determine_layer(self, file_path: Path) -> str:
        """Determine architectural layer for a file."""
        relative_path = str(file_path.relative_to(self.src_path))
        
        for layer, patterns in self.LAYER_HIERARCHY.items():
            for pattern in patterns:
                if relative_path.startswith(pattern):
                    return layer
        
        return "unknown"
    
    async def _check_circular_imports(self, result: ValidationResult) -> None:
        """Check for circular import dependencies."""
        logger.info("Checking for circular imports...")
        
        def find_cycles() -> List[List[str]]:
            """Find cycles in dependency graph using DFS."""
            cycles = []
            visited = set()
            rec_stack = set()
            
            def dfs(node: str, path: List[str]) -> None:
                if node in rec_stack:
                    # Found cycle
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:] + [node]
                    cycles.append(cycle)
                    return
                
                if node in visited:
                    return
                
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                
                for neighbor in self.dependency_graph.get(node, set()):
                    if neighbor.startswith("prompt_improver"):  # Only check internal imports
                        dfs(neighbor, path.copy())
                
                rec_stack.remove(node)
            
            for module in self.dependency_graph:
                if module not in visited:
                    dfs(module, [])
            
            return cycles
        
        cycles = find_cycles()
        result.circular_dependencies = cycles
        
        # Create violations for each cycle
        for cycle in cycles:
            if len(cycle) > 1:
                for i in range(len(cycle) - 1):
                    source_module = cycle[i]
                    target_module = cycle[i + 1]
                    
                    # Find file path for source module
                    source_file = None
                    for file_path, module_name in self.file_to_module.items():
                        if module_name == source_module:
                            source_file = str(file_path.relative_to(self.project_root))
                            break
                    
                    result.violations.append(Violation(
                        file_path=source_file or source_module,
                        line_number=1,
                        violation_type=ViolationType.CIRCULAR_IMPORT,
                        severity=Severity.HIGH,
                        message=f"Circular import: {' → '.join(cycle)}",
                        suggestion="Use dependency inversion or lazy imports to break the cycle",
                        impact="Can cause ImportError at runtime and slow startup"
                    ))
    
    async def _check_layer_violations(self, result: ValidationResult) -> None:
        """Check for clean architecture layer violations."""
        logger.info("Checking layer violations...")
        
        for source_module, dependencies in self.dependency_graph.items():
            source_layer = self.module_to_layer.get(source_module, "unknown")
            
            if source_layer == "unknown":
                continue
            
            allowed_layers = self.ALLOWED_DEPENDENCIES.get(source_layer, set())
            
            for dep_module in dependencies:
                if not dep_module.startswith("prompt_improver"):
                    continue
                
                target_layer = None
                # Find target layer by matching module against layer patterns
                for layer, patterns in self.LAYER_HIERARCHY.items():
                    for pattern in patterns:
                        if pattern in dep_module:
                            target_layer = layer
                            break
                    if target_layer:
                        break
                
                if target_layer and target_layer not in allowed_layers and source_layer != target_layer:
                    # Find file path
                    source_file = None
                    for file_path, module_name in self.file_to_module.items():
                        if module_name == source_module:
                            source_file = str(file_path.relative_to(self.project_root))
                            break
                    
                    result.violations.append(Violation(
                        file_path=source_file or source_module,
                        line_number=1,
                        violation_type=ViolationType.LAYER_VIOLATION,
                        severity=Severity.HIGH,
                        message=f"Layer violation: {source_layer} cannot depend on {target_layer}",
                        suggestion=f"Use dependency inversion or move to allowed layer: {allowed_layers}",
                        impact="Violates clean architecture principles"
                    ))
    
    async def _check_repository_pattern(self, result: ValidationResult) -> None:
        """Check for repository pattern violations (direct database imports)."""
        logger.info("Checking repository pattern violations...")
        
        # Find all files that import database modules directly
        for source_module, dependencies in self.dependency_graph.items():
            source_layer = self.module_to_layer.get(source_module, "unknown")
            
            # Only check presentation and application layers
            if source_layer not in {"presentation", "application"}:
                continue
            
            for dep_module in dependencies:
                if any(db_import in dep_module for db_import in self.DATABASE_IMPORTS):
                    # Find file path and line number
                    source_file = None
                    line_number = 1
                    
                    for file_path, module_name in self.file_to_module.items():
                        if module_name == source_module:
                            source_file = str(file_path.relative_to(self.project_root))
                            line_number = await self._find_import_line(file_path, dep_module)
                            break
                    
                    result.violations.append(Violation(
                        file_path=source_file or source_module,
                        line_number=line_number,
                        violation_type=ViolationType.REPOSITORY_VIOLATION,
                        severity=Severity.HIGH,
                        message=f"Direct database import in {source_layer} layer: {dep_module}",
                        suggestion="Use repository interfaces and dependency injection instead",
                        impact="Violates repository pattern and increases coupling"
                    ))
    
    async def _check_protocol_consolidation(self, result: ValidationResult) -> None:
        """Check for imports from old protocol locations."""
        logger.info("Checking protocol consolidation...")
        
        old_protocol_patterns = [
            "prompt_improver.core.protocols.cache_protocol",
            "prompt_improver.core.protocols.database_protocol",
            "prompt_improver.core.protocols.cache_service"
        ]
        
        for source_module, dependencies in self.dependency_graph.items():
            for dep_module in dependencies:
                for old_pattern in old_protocol_patterns:
                    if old_pattern in dep_module:
                        source_file = None
                        line_number = 1
                        
                        for file_path, module_name in self.file_to_module.items():
                            if module_name == source_module:
                                source_file = str(file_path.relative_to(self.project_root))
                                line_number = await self._find_import_line(file_path, dep_module)
                                break
                        
                        result.violations.append(Violation(
                            file_path=source_file or source_module,
                            line_number=line_number,
                            violation_type=ViolationType.PROTOCOL_CONSOLIDATION,
                            severity=Severity.MEDIUM,
                            message=f"Import from old protocol location: {dep_module}",
                            suggestion="Update import to use consolidated protocols in shared.interfaces.protocols",
                            impact="References deprecated protocol locations"
                        ))
    
    async def _check_domain_purity(self, result: ValidationResult) -> None:
        """Check that core/domain layers don't import heavy external libraries."""
        logger.info("Checking domain purity...")
        
        domain_modules = [mod for mod, layer in self.module_to_layer.items() 
                         if layer in {"domain", "shared"}]
        
        for module in domain_modules:
            dependencies = self.dependency_graph.get(module, set())
            
            for dep in dependencies:
                for heavy_dep in self.HEAVY_DEPENDENCIES:
                    if heavy_dep in dep and not dep.startswith("prompt_improver"):
                        penalty_ms = self.IMPORT_PENALTIES.get(heavy_dep, 50)
                        
                        source_file = None
                        line_number = 1
                        
                        for file_path, module_name in self.file_to_module.items():
                            if module_name == module:
                                source_file = str(file_path.relative_to(self.project_root))
                                line_number = await self._find_import_line(file_path, dep)
                                break
                        
                        severity = Severity.CRITICAL if penalty_ms > 500 else Severity.HIGH
                        
                        result.violations.append(Violation(
                            file_path=source_file or module,
                            line_number=line_number,
                            violation_type=ViolationType.DOMAIN_PURITY,
                            severity=severity,
                            message=f"Heavy import {dep} in domain layer",
                            suggestion="Use lazy loading or move import to infrastructure layer",
                            impact=f"Adds ~{penalty_ms}ms to startup time"
                        ))
    
    async def _find_import_line(self, file_path: Path, import_module: str) -> int:
        """Find line number of specific import in file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                if import_module in line and ('import' in line or 'from' in line):
                    return i
        except Exception:
            pass
        
        return 1
    
    def _calculate_compliance_ratio(self, result: ValidationResult) -> float:
        """Calculate overall compliance ratio."""
        if result.total_files_analyzed == 0:
            return 1.0
        
        files_with_violations = len(set(v.file_path for v in result.violations))
        compliant_files = result.total_files_analyzed - files_with_violations
        
        return compliant_files / result.total_files_analyzed
    
    def _log_validation_summary(self, result: ValidationResult) -> None:
        """Log validation summary."""
        logger.info(f"Validation completed in {result.analysis_duration:.2f}s")
        logger.info(f"Files analyzed: {result.total_files_analyzed}")
        logger.info(f"Compliance ratio: {result.compliance_ratio:.1%}")
        
        if result.violations:
            logger.warning(f"Found {len(result.violations)} violations:")
            for violation_type, count in result.violation_summary.items():
                logger.warning(f"  {violation_type}: {count}")
        else:
            logger.info("✅ No architectural violations found!")
    
    def generate_report(self, result: ValidationResult, format: str = "text") -> str:
        """Generate a formatted report of violations.
        
        Args:
            result: ValidationResult to report on
            format: Report format ("text", "json", "markdown")
        
        Returns:
            Formatted report string
        """
        if format == "text":
            return self._generate_text_report(result)
        elif format == "json":
            return self._generate_json_report(result)
        elif format == "markdown":
            return self._generate_markdown_report(result)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate text format report."""
        lines = [
            "🏗️  Architectural Validation Report",
            "=" * 40,
            f"Analysis Duration: {result.analysis_duration:.2f}s",
            f"Files Analyzed: {result.total_files_analyzed}",
            f"Compliance Ratio: {result.compliance_ratio:.1%}",
            ""
        ]
        
        if not result.violations:
            lines.extend([
                "✅ SUCCESS: No architectural violations found!",
                "",
                "All architectural boundaries are properly maintained:",
                "• Clean architecture layers ✓",
                "• Repository pattern enforcement ✓", 
                "• Domain purity ✓",
                "• No circular imports ✓",
                "• Protocol consolidation ✓"
            ])
            return "\n".join(lines)
        
        # Group violations by type
        by_type = defaultdict(list)
        for violation in result.violations:
            by_type[violation.violation_type].append(violation)
        
        lines.append(f"❌ FAILURE: Found {len(result.violations)} violations\n")
        
        # Critical violations first
        critical_violations = result.critical_violations
        if critical_violations:
            lines.extend([
                "🚨 CRITICAL VIOLATIONS (Must Fix):",
                "-" * 30
            ])
            for violation in critical_violations:
                lines.extend([
                    f"📁 {violation.file_path}:{violation.line_number}",
                    f"   {violation.message}",
                    f"   💡 {violation.suggestion}",
                    f"   ⚠️  {violation.impact}" if violation.impact else "",
                    ""
                ])
        
        # Other violations by type
        for vtype, violations in by_type.items():
            if vtype == ViolationType.DOMAIN_PURITY and any(v.severity == Severity.CRITICAL for v in violations):
                continue  # Already shown in critical section
            
            lines.extend([
                f"\n{vtype.value.upper().replace('_', ' ')} ({len(violations)} violations):",
                "-" * 30
            ])
            
            for violation in violations:
                lines.extend([
                    f"📁 {violation.file_path}:{violation.line_number}",
                    f"   {violation.message}",
                    f"   💡 {violation.suggestion}",
                    ""
                ])
        
        return "\n".join(lines)
    
    def _generate_json_report(self, result: ValidationResult) -> str:
        """Generate JSON format report."""
        import json
        
        data = {
            "summary": {
                "is_compliant": result.is_compliant,
                "total_violations": len(result.violations),
                "critical_violations": len(result.critical_violations),
                "compliance_ratio": result.compliance_ratio,
                "analysis_duration": result.analysis_duration,
                "files_analyzed": result.total_files_analyzed
            },
            "violations": [
                {
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "message": v.message,
                    "suggestion": v.suggestion,
                    "impact": v.impact
                }
                for v in result.violations
            ],
            "violation_summary": result.violation_summary,
            "circular_dependencies": result.circular_dependencies
        }
        
        return json.dumps(data, indent=2)
    
    def _generate_markdown_report(self, result: ValidationResult) -> str:
        """Generate Markdown format report."""
        lines = [
            "# 🏗️ Architectural Validation Report",
            "",
            f"**Analysis Duration:** {result.analysis_duration:.2f}s  ",
            f"**Files Analyzed:** {result.total_files_analyzed}  ",
            f"**Compliance Ratio:** {result.compliance_ratio:.1%}  ",
            ""
        ]
        
        if not result.violations:
            lines.extend([
                "## ✅ SUCCESS",
                "",
                "No architectural violations found! All boundaries are properly maintained.",
                ""
            ])
            return "\n".join(lines)
        
        lines.extend([
            f"## ❌ FAILURE: {len(result.violations)} Violations Found",
            ""
        ])
        
        # Critical violations first
        if result.critical_violations:
            lines.extend([
                "### 🚨 Critical Violations",
                ""
            ])
            for violation in result.critical_violations:
                lines.extend([
                    f"**File:** `{violation.file_path}:{violation.line_number}`  ",
                    f"**Message:** {violation.message}  ",
                    f"**Suggestion:** {violation.suggestion}  ",
                    f"**Impact:** {violation.impact}  " if violation.impact else "",
                    ""
                ])
        
        # Other violations by type
        by_type = defaultdict(list)
        for violation in result.violations:
            if violation.severity != Severity.CRITICAL:
                by_type[violation.violation_type].append(violation)
        
        for vtype, violations in by_type.items():
            title = vtype.value.replace('_', ' ').title()
            lines.extend([
                f"### {title} ({len(violations)} violations)",
                ""
            ])
            for violation in violations:
                lines.extend([
                    f"- **{violation.file_path}:{violation.line_number}** - {violation.message}",
                    ""
                ])
        
        return "\n".join(lines)


async def main():
    """Main entry point for the architectural validator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive architectural validation for prompt-improver project"
    )
    parser.add_argument(
        "--format", 
        choices=["text", "json", "markdown"], 
        default="text",
        help="Output format for the report"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit code 1 if any violations found"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = ArchitecturalValidator(args.project_root)
    
    # Run validation
    result = await validator.validate_architecture()
    
    # Generate and print report
    report = validator.generate_report(result, args.format)
    print(report)
    
    # Exit with appropriate code
    if args.strict and not result.is_compliant:
        logger.error("Validation failed in strict mode")
        sys.exit(1)
    elif result.critical_violations:
        logger.error("Critical violations found")
        sys.exit(1)
    else:
        logger.info("Validation completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())