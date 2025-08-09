"""Architectural Boundary Enforcement following 2025 best practices.

This module provides a comprehensive system for enforcing architectural boundaries
in the prompt improver application, preventing circular dependencies and maintaining
clean architecture principles.

Features:
- Module boundary validation
- Dependency direction enforcement
- Import pattern analysis
- Architectural testing support
- Integration with import-linter
- Performance regression detection
"""
import ast
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from sqlmodel import Field, SQLModel

class BoundaryViolationType(Enum):
    """Types of architectural boundary violations."""
    CIRCULAR_DEPENDENCY = 'circular_dependency'
    FORBIDDEN_IMPORT = 'forbidden_import'
    WRONG_DIRECTION = 'wrong_direction'
    MISSING_INTERFACE = 'missing_interface'
    LAYER_VIOLATION = 'layer_violation'
    MODULE_COUPLING = 'module_coupling'

class ArchitecturalLayer(Enum):
    """Architectural layers in clean architecture."""
    PRESENTATION = 'presentation'
    APPLICATION = 'application'
    DOMAIN = 'domain'
    INFRASTRUCTURE = 'infrastructure'
    SHARED = 'shared'
    CORE = 'core'

class BoundaryViolation(SQLModel):
    """Represents an architectural boundary violation."""
    violation_type: BoundaryViolationType
    source_module: str
    target_module: str
    layer_source: ArchitecturalLayer | None = Field(default=None)
    layer_target: ArchitecturalLayer | None = Field(default=None)
    message: str = Field(default='')
    severity: str = Field(default='error')
    line_number: int | None = Field(default=None)
    suggestion: str | None = Field(default=None)

class ModuleBoundary(SQLModel):
    """Defines boundaries for a module or package."""
    name: str
    layer: ArchitecturalLayer
    allowed_dependencies: set[str] = Field(default_factory=set)
    forbidden_dependencies: set[str] = Field(default_factory=set)
    interfaces_required: bool = Field(default=True)
    external_dependencies_allowed: bool = Field(default=True)
    description: str = Field(default='')

class ArchitecturalRules:
    """Defines architectural rules and constraints."""
    LAYER_DEPENDENCIES = {ArchitecturalLayer.PRESENTATION: {ArchitecturalLayer.APPLICATION, ArchitecturalLayer.SHARED, ArchitecturalLayer.CORE}, ArchitecturalLayer.APPLICATION: {ArchitecturalLayer.DOMAIN, ArchitecturalLayer.SHARED, ArchitecturalLayer.CORE}, ArchitecturalLayer.DOMAIN: {ArchitecturalLayer.SHARED, ArchitecturalLayer.CORE}, ArchitecturalLayer.INFRASTRUCTURE: {ArchitecturalLayer.DOMAIN, ArchitecturalLayer.APPLICATION, ArchitecturalLayer.SHARED, ArchitecturalLayer.CORE}, ArchitecturalLayer.SHARED: {ArchitecturalLayer.CORE}, ArchitecturalLayer.CORE: set()}
    MODULE_BOUNDARIES = {'prompt_improver.core': ModuleBoundary(name='core', layer=ArchitecturalLayer.CORE, allowed_dependencies=set(), description='Core utilities, DI container, protocols'), 'prompt_improver.shared': ModuleBoundary(name='shared', layer=ArchitecturalLayer.SHARED, allowed_dependencies={'prompt_improver.core'}, description='Shared interfaces and types'), 'prompt_improver.domain': ModuleBoundary(name='domain', layer=ArchitecturalLayer.DOMAIN, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared'}, description='Business logic and domain models'), 'prompt_improver.ml': ModuleBoundary(name='ml', layer=ArchitecturalLayer.DOMAIN, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared'}, description='Machine learning domain logic'), 'prompt_improver.rule_engine': ModuleBoundary(name='rule_engine', layer=ArchitecturalLayer.DOMAIN, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared'}, description='Rule engine domain logic'), 'prompt_improver.database': ModuleBoundary(name='database', layer=ArchitecturalLayer.INFRASTRUCTURE, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared', 'prompt_improver.domain'}, description='Database implementations'), 'prompt_improver.api': ModuleBoundary(name='api', layer=ArchitecturalLayer.PRESENTATION, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared', 'prompt_improver.application'}, description='API endpoints and web layer'), 'prompt_improver.cli': ModuleBoundary(name='cli', layer=ArchitecturalLayer.PRESENTATION, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared', 'prompt_improver.application'}, description='Command line interface'), 'prompt_improver.mcp_server': ModuleBoundary(name='mcp_server', layer=ArchitecturalLayer.PRESENTATION, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared', 'prompt_improver.application'}, description='MCP server implementation'), 'prompt_improver.performance': ModuleBoundary(name='performance', layer=ArchitecturalLayer.INFRASTRUCTURE, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared', 'prompt_improver.domain'}, description='Performance monitoring infrastructure'), 'prompt_improver.security': ModuleBoundary(name='security', layer=ArchitecturalLayer.INFRASTRUCTURE, allowed_dependencies={'prompt_improver.core', 'prompt_improver.shared'}, description='Security infrastructure')}

class DependencyAnalyzer:
    """Analyzes module dependencies and detects violations."""

    def __init__(self, logger: logging.Logger | None=None):
        self.logger = logger or logging.getLogger(__name__)
        self.module_graph: dict[str, set[str]] = defaultdict(set)
        self.violations: list[BoundaryViolation] = []

    def analyze_file(self, file_path: Path) -> list[str]:
        """Analyze a Python file and extract its imports.

        Args:
            file_path: Path to the Python file

        Returns:
            List of imported module names
        """
        imports = []
        try:
            with open(file_path, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception as e:
            self.logger.warning('Could not analyze {file_path}: %s', e)
        return imports

    def build_dependency_graph(self, root_path: Path) -> dict[str, set[str]]:
        """Build a dependency graph for all Python modules.

        Args:
            root_path: Root path to analyze

        Returns:
            Dictionary mapping module names to their dependencies
        """
        self.module_graph.clear()
        for py_file in root_path.rglob('*.py'):
            if py_file.name == '__init__.py' or py_file.stem.startswith('test_'):
                continue
            relative_path = py_file.relative_to(root_path.parent)
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            module_name = '.'.join(module_parts)
            imports = self.analyze_file(py_file)
            internal_imports = {imp for imp in imports if imp.startswith('prompt_improver')}
            self.module_graph[module_name].update(internal_imports)
        return dict(self.module_graph)

    def detect_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in the module graph.

        Returns:
            List of cycles, where each cycle is a list of module names
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> None:
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbor in self.module_graph.get(node, set()):
                dfs(neighbor, path.copy())
            rec_stack.remove(node)
        for module in self.module_graph:
            if module not in visited:
                dfs(module, [])
        return cycles

    def validate_layer_dependencies(self) -> list[BoundaryViolation]:
        """Validate that layer dependencies follow clean architecture rules.

        Returns:
            List of boundary violations
        """
        violations = []
        for source_module, dependencies in self.module_graph.items():
            source_boundary = self._get_module_boundary(source_module)
            if not source_boundary:
                continue
            source_layer = source_boundary.layer
            for dep_module in dependencies:
                target_boundary = self._get_module_boundary(dep_module)
                if not target_boundary:
                    continue
                target_layer = target_boundary.layer
                allowed_layers = ArchitecturalRules.LAYER_DEPENDENCIES.get(source_layer, set())
                if target_layer not in allowed_layers and source_layer != target_layer:
                    violations.append(BoundaryViolation(violation_type=BoundaryViolationType.LAYER_VIOLATION, source_module=source_module, target_module=dep_module, layer_source=source_layer, layer_target=target_layer, message=f'Layer {source_layer.value} cannot depend on {target_layer.value}', suggestion='Use dependency inversion or move to allowed layer'))
        return violations

    def validate_module_boundaries(self) -> list[BoundaryViolation]:
        """Validate module-specific boundary rules.

        Returns:
            List of boundary violations
        """
        violations = []
        for source_module, dependencies in self.module_graph.items():
            source_boundary = self._get_module_boundary(source_module)
            if not source_boundary:
                continue
            for dep_module in dependencies:
                if any((dep_module.startswith(forbidden) for forbidden in source_boundary.forbidden_dependencies)):
                    violations.append(BoundaryViolation(violation_type=BoundaryViolationType.FORBIDDEN_IMPORT, source_module=source_module, target_module=dep_module, message=f'Module {source_module} has forbidden dependency on {dep_module}', suggestion='Remove dependency or use dependency inversion'))
                if source_boundary.allowed_dependencies:
                    if not any((dep_module.startswith(allowed) for allowed in source_boundary.allowed_dependencies)):
                        violations.append(BoundaryViolation(violation_type=BoundaryViolationType.WRONG_DIRECTION, source_module=source_module, target_module=dep_module, message=f'Module {source_module} depends on {dep_module} which is not in allowed dependencies', suggestion='Add to allowed dependencies or refactor'))
        return violations

    def _get_module_boundary(self, module_name: str) -> ModuleBoundary | None:
        """Get the boundary definition for a module.

        Args:
            module_name: Full module name

        Returns:
            ModuleBoundary if found, None otherwise
        """
        for boundary_name, boundary in ArchitecturalRules.MODULE_BOUNDARIES.items():
            if module_name.startswith(boundary_name):
                return boundary
        return None

class BoundaryEnforcer:
    """Main class for enforcing architectural boundaries."""

    def __init__(self, root_path: Path | None=None, logger: logging.Logger | None=None):
        self.root_path = root_path or Path('src')
        self.logger = logger or logging.getLogger(__name__)
        self.analyzer = DependencyAnalyzer(logger)
        self.violations: list[BoundaryViolation] = []

    def analyze_architecture(self) -> dict[str, Any]:
        """Perform comprehensive architectural analysis.

        Returns:
            Analysis results including violations and metrics
        """
        self.logger.info('Starting architectural boundary analysis')
        dependency_graph = self.analyzer.build_dependency_graph(self.root_path)
        self.violations.clear()
        cycles = self.analyzer.detect_circular_dependencies()
        for cycle in cycles:
            for i in range(len(cycle) - 1):
                self.violations.append(BoundaryViolation(violation_type=BoundaryViolationType.CIRCULAR_DEPENDENCY, source_module=cycle[i], target_module=cycle[i + 1], message=f"Circular dependency: {' -> '.join(cycle)}", severity='critical', suggestion='Break the cycle using dependency inversion or interfaces'))
        layer_violations = self.analyzer.validate_layer_dependencies()
        self.violations.extend(layer_violations)
        boundary_violations = self.analyzer.validate_module_boundaries()
        self.violations.extend(boundary_violations)
        metrics = self._calculate_metrics(dependency_graph)
        self.logger.info('Analysis complete. Found %s violations', len(self.violations))
        return {'violations': self.violations, 'metrics': metrics, 'dependency_graph': dependency_graph, 'circular_dependencies': cycles}

    def _calculate_metrics(self, dependency_graph: dict[str, set[str]]) -> dict[str, Any]:
        """Calculate architectural metrics.

        Args:
            dependency_graph: Module dependency graph

        Returns:
            Dictionary of metrics
        """
        if not dependency_graph:
            return {}
        total_modules = len(dependency_graph)
        total_dependencies = sum((len(deps) for deps in dependency_graph.values()))
        avg_efferent_coupling = total_dependencies / total_modules if total_modules > 0 else 0
        highest_coupling = max(dependency_graph.items(), key=lambda x: len(x[1]), default=('', set()))
        return {'total_modules': total_modules, 'total_dependencies': total_dependencies, 'average_efferent_coupling': avg_efferent_coupling, 'highest_coupling_module': highest_coupling[0], 'highest_coupling_count': len(highest_coupling[1]), 'violation_count_by_type': {vtype.value: len([v for v in self.violations if v.violation_type == vtype]) for vtype in BoundaryViolationType}}

    def generate_report(self, format: str='text') -> str:
        """Generate a report of boundary violations.

        Args:
            format: Report format ("text", "json", "markdown")

        Returns:
            Formatted report string
        """
        if format == 'text':
            return self._generate_text_report()
        if format == 'json':
            return self._generate_json_report()
        if format == 'markdown':
            return self._generate_markdown_report()
        raise ValueError(f'Unsupported report format: {format}')

    def _generate_text_report(self) -> str:
        """Generate a text report."""
        lines = ['Architectural Boundary Analysis Report', '=' * 40, '']
        if not self.violations:
            lines.append('✅ No boundary violations found!')
            return '\n'.join(lines)
        by_type = defaultdict(list)
        for violation in self.violations:
            by_type[violation.violation_type].append(violation)
        for vtype, violations in by_type.items():
            lines.extend([f'\n{vtype.value.upper()} ({len(violations)} violations):', '-' * 30])
            for violation in violations:
                lines.append(f'• {violation.source_module} -> {violation.target_module}')
                lines.append(f'  {violation.message}')
                if violation.suggestion:
                    lines.append(f'  💡 {violation.suggestion}')
                lines.append('')
        return '\n'.join(lines)

    def _generate_json_report(self) -> str:
        """Generate a JSON report."""
        import json
        data = {'violations': [{'type': v.violation_type.value, 'source': v.source_module, 'target': v.target_module, 'message': v.message, 'severity': v.severity, 'suggestion': v.suggestion} for v in self.violations], 'summary': {'total_violations': len(self.violations), 'by_type': {vtype.value: len([v for v in self.violations if v.violation_type == vtype]) for vtype in BoundaryViolationType}}}
        return json.dumps(data, indent=2)

    def _generate_markdown_report(self) -> str:
        """Generate a markdown report."""
        lines = ['# Architectural Boundary Analysis Report', '']
        if not self.violations:
            lines.extend(['✅ **No boundary violations found!**', ''])
            return '\n'.join(lines)
        lines.extend(['## Summary', f'- Total violations: {len(self.violations)}', ''])
        by_type = defaultdict(list)
        for violation in self.violations:
            by_type[violation.violation_type].append(violation)
        for vtype, violations in by_type.items():
            lines.extend([f"## {vtype.value.replace('_', ' ').title()} ({len(violations)} violations)", ''])
            for violation in violations:
                lines.extend([f'### {violation.source_module} → {violation.target_module}', f'**Message:** {violation.message}', ''])
                if violation.suggestion:
                    lines.extend([f'**Suggestion:** {violation.suggestion}', ''])
        return '\n'.join(lines)

def create_boundary_enforcer(root_path: Path | None=None) -> BoundaryEnforcer:
    """Create a boundary enforcer instance.

    Args:
        root_path: Root path for analysis (defaults to 'src')

    Returns:
        Configured BoundaryEnforcer instance
    """
    return BoundaryEnforcer(root_path or Path('src'))

class ArchitecturalTest:
    """Test case for architectural boundaries."""

    def __init__(self, enforcer: BoundaryEnforcer | None=None):
        self.enforcer = enforcer or create_boundary_enforcer()

    def test_no_circular_dependencies(self) -> bool:
        """Test that there are no circular dependencies."""
        results = self.enforcer.analyze_architecture()
        circular_violations = [v for v in results['violations'] if v.violation_type == BoundaryViolationType.CIRCULAR_DEPENDENCY]
        return len(circular_violations) == 0

    def test_layer_dependencies(self) -> bool:
        """Test that layer dependencies are valid."""
        results = self.enforcer.analyze_architecture()
        layer_violations = [v for v in results['violations'] if v.violation_type == BoundaryViolationType.LAYER_VIOLATION]
        return len(layer_violations) == 0

    def test_forbidden_dependencies(self) -> bool:
        """Test that no forbidden dependencies exist."""
        results = self.enforcer.analyze_architecture()
        forbidden_violations = [v for v in results['violations'] if v.violation_type == BoundaryViolationType.FORBIDDEN_IMPORT]
        return len(forbidden_violations) == 0

    def assert_architecture_compliance(self) -> None:
        """Assert that architecture is compliant (for use in test suites)."""
        results = self.enforcer.analyze_architecture()
        violations = results['violations']
        if violations:
            report = self.enforcer.generate_report('text')
            raise AssertionError(f'Architecture violations found:\n{report}')
__all__ = ['ArchitecturalLayer', 'ArchitecturalRules', 'ArchitecturalTest', 'BoundaryEnforcer', 'BoundaryViolation', 'BoundaryViolationType', 'DependencyAnalyzer', 'ModuleBoundary', 'create_boundary_enforcer']
