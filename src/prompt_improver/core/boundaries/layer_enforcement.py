"""Clean Architecture Layer Enforcement and Circular Import Detection.

This module provides runtime validation and enforcement of clean architecture
boundaries to prevent circular import risks and architectural violations.

CRITICAL: This module prevents the introduction of new circular imports and
validates that all inter-layer communication follows clean architecture rules.
"""

import ast
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from prompt_improver.core.domain.enums import HealthStatus


class ArchitectureLayer(Enum):
    """Clean architecture layers."""
    PRESENTATION = "presentation"  # API, CLI, TUI
    APPLICATION = "application"    # Application Services, Workflows
    DOMAIN = "domain"             # Business Logic, Rules, Entities
    INFRASTRUCTURE = "infrastructure"  # Database, External Services


@dataclass
class LayerDefinition:
    """Definition of an architecture layer."""
    name: ArchitectureLayer
    allowed_dependencies: List[ArchitectureLayer]
    module_patterns: List[str]
    description: str


@dataclass
class ImportViolation:
    """Represents an architecture layer import violation."""
    source_module: str
    source_layer: ArchitectureLayer
    target_module: str
    target_layer: ArchitectureLayer
    violation_type: str
    description: str
    severity: str


@dataclass
class CircularImportPath:
    """Represents a circular import chain."""
    modules: List[str]
    import_chain: List[Tuple[str, str]]
    severity: str


class LayerBoundaryEnforcer:
    """Enforces clean architecture layer boundaries."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the layer boundary enforcer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self._layer_definitions = self._define_layers()
        self._module_layer_cache: Dict[str, ArchitectureLayer] = {}
        
    def _define_layers(self) -> Dict[ArchitectureLayer, LayerDefinition]:
        """Define the clean architecture layers and their rules."""
        return {
            ArchitectureLayer.PRESENTATION: LayerDefinition(
                name=ArchitectureLayer.PRESENTATION,
                allowed_dependencies=[ArchitectureLayer.APPLICATION],
                module_patterns=[
                    "*.api.*",
                    "*.cli.*", 
                    "*.tui.*",
                    "*.dashboard.*",
                    "*.endpoints.*",
                    "*websocket*",
                ],
                description="Presentation layer (API, CLI, TUI) - depends only on Application layer"
            ),
            ArchitectureLayer.APPLICATION: LayerDefinition(
                name=ArchitectureLayer.APPLICATION,
                allowed_dependencies=[ArchitectureLayer.DOMAIN],
                module_patterns=[
                    "*.application.*",
                    "*.workflows.*",
                    "*_application_service*",
                    "*application*",
                ],
                description="Application layer (Services, Workflows) - depends only on Domain layer"
            ),
            ArchitectureLayer.DOMAIN: LayerDefinition(
                name=ArchitectureLayer.DOMAIN,
                allowed_dependencies=[],  # Domain should have no external dependencies
                module_patterns=[
                    "*.core.*",
                    "*.rule_engine.*", 
                    "*.ml.*",
                    "*.analytics.*",
                    "*domain*",
                    "*business*",
                    "*rules*",
                ],
                description="Domain layer (Business Logic) - no external dependencies allowed"
            ),
            ArchitectureLayer.INFRASTRUCTURE: LayerDefinition(
                name=ArchitectureLayer.INFRASTRUCTURE,
                allowed_dependencies=[ArchitectureLayer.DOMAIN],  # Implements domain protocols
                module_patterns=[
                    "*.database.*",
                    "*.repositories.*",
                    "*.monitoring.*",
                    "*.security.*",
                    "*.cache.*",
                    "*.performance.*",
                    "*_repository*",
                    "*external*",
                    "*infrastructure*",
                ],
                description="Infrastructure layer - implements domain protocols"
            )
        }
    
    def get_module_layer(self, module_path: str) -> Optional[ArchitectureLayer]:
        """Determine which architecture layer a module belongs to.
        
        Args:
            module_path: Full module path
            
        Returns:
            Architecture layer or None if unclassified
        """
        if module_path in self._module_layer_cache:
            return self._module_layer_cache[module_path]
        
        # Check each layer's patterns
        for layer, definition in self._layer_definitions.items():
            for pattern in definition.module_patterns:
                if self._matches_pattern(module_path, pattern):
                    self._module_layer_cache[module_path] = layer
                    return layer
        
        return None
    
    def _matches_pattern(self, module_path: str, pattern: str) -> bool:
        """Check if module path matches a pattern.
        
        Args:
            module_path: Module path to check
            pattern: Pattern to match against
            
        Returns:
            Whether pattern matches
        """
        import fnmatch
        return fnmatch.fnmatch(module_path, pattern)
    
    def validate_import(
        self,
        source_module: str,
        target_module: str,
    ) -> Optional[ImportViolation]:
        """Validate a single import against clean architecture rules.
        
        Args:
            source_module: Module doing the import
            target_module: Module being imported
            
        Returns:
            ImportViolation if invalid, None if valid
        """
        source_layer = self.get_module_layer(source_module)
        target_layer = self.get_module_layer(target_module)
        
        # Skip validation if layers can't be determined
        if not source_layer or not target_layer:
            return None
        
        # Same layer imports are always allowed
        if source_layer == target_layer:
            return None
        
        # Check if dependency is allowed
        source_definition = self._layer_definitions[source_layer]
        if target_layer not in source_definition.allowed_dependencies:
            return ImportViolation(
                source_module=source_module,
                source_layer=source_layer,
                target_module=target_module,
                target_layer=target_layer,
                violation_type="INVALID_LAYER_DEPENDENCY",
                description=f"Layer {source_layer.value} cannot depend on {target_layer.value}",
                severity="ERROR"
            )
        
        return None
    
    def analyze_module_imports(
        self,
        module_path: str,
    ) -> List[ImportViolation]:
        """Analyze all imports in a module for violations.
        
        Args:
            module_path: Path to the Python module file
            
        Returns:
            List of import violations found
        """
        violations = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            module_name = self._get_module_name(module_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        violation = self.validate_import(module_name, alias.name)
                        if violation:
                            violations.append(violation)
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        violation = self.validate_import(module_name, node.module)
                        if violation:
                            violations.append(violation)
                            
        except Exception as e:
            # Log error but don't fail validation
            print(f"Error analyzing {module_path}: {e}")
        
        return violations
    
    def _get_module_name(self, module_path: str) -> str:
        """Convert file path to module name.
        
        Args:
            module_path: File path
            
        Returns:
            Module name
        """
        path = Path(module_path)
        if path.suffix == '.py':
            path = path.with_suffix('')
        
        # Convert to module notation
        parts = path.parts
        if 'src' in parts:
            src_index = parts.index('src')
            parts = parts[src_index + 1:]
        
        return '.'.join(parts)
    
    def scan_project_violations(
        self,
        src_path: Optional[Path] = None,
    ) -> List[ImportViolation]:
        """Scan entire project for architecture violations.
        
        Args:
            src_path: Source directory to scan
            
        Returns:
            List of all violations found
        """
        if src_path is None:
            src_path = self.project_root / 'src'
        
        violations = []
        
        for py_file in src_path.rglob('*.py'):
            if py_file.name.startswith('test_'):
                continue  # Skip test files
                
            file_violations = self.analyze_module_imports(str(py_file))
            violations.extend(file_violations)
        
        return violations
    
    def generate_violation_report(
        self,
        violations: List[ImportViolation]
    ) -> Dict[str, Any]:
        """Generate a detailed report of architecture violations.
        
        Args:
            violations: List of violations to report
            
        Returns:
            Detailed violation report
        """
        report = {
            "total_violations": len(violations),
            "violations_by_severity": {},
            "violations_by_layer": {},
            "violations_by_type": {},
            "detailed_violations": [],
            "recommendations": []
        }
        
        # Group violations
        for violation in violations:
            # By severity
            severity = violation.severity
            if severity not in report["violations_by_severity"]:
                report["violations_by_severity"][severity] = 0
            report["violations_by_severity"][severity] += 1
            
            # By source layer
            layer = violation.source_layer.value
            if layer not in report["violations_by_layer"]:
                report["violations_by_layer"][layer] = 0
            report["violations_by_layer"][layer] += 1
            
            # By type
            v_type = violation.violation_type
            if v_type not in report["violations_by_type"]:
                report["violations_by_type"][v_type] = 0
            report["violations_by_type"][v_type] += 1
            
            # Detailed info
            report["detailed_violations"].append({
                "source_module": violation.source_module,
                "source_layer": violation.source_layer.value,
                "target_module": violation.target_module,
                "target_layer": violation.target_layer.value,
                "violation_type": violation.violation_type,
                "description": violation.description,
                "severity": violation.severity
            })
        
        # Add recommendations
        if violations:
            report["recommendations"] = [
                "Create protocol boundaries between layers to eliminate direct dependencies",
                "Use dependency injection with protocols to invert dependencies",
                "Move shared types to domain layer to eliminate infrastructure dependencies",
                "Implement facade patterns to hide infrastructure complexity",
                "Consider creating adapter patterns for external service integrations"
            ]
        
        return report


class CircularImportDetector:
    """Detects circular import chains in the codebase."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize circular import detector.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self._import_graph: Dict[str, Set[str]] = {}
        
    def build_import_graph(
        self,
        src_path: Optional[Path] = None,
    ) -> Dict[str, Set[str]]:
        """Build a graph of all import relationships.
        
        Args:
            src_path: Source directory to scan
            
        Returns:
            Dictionary mapping modules to their imports
        """
        if src_path is None:
            src_path = self.project_root / 'src'
        
        self._import_graph = {}
        
        for py_file in src_path.rglob('*.py'):
            if py_file.name.startswith('test_'):
                continue
                
            module_name = self._get_module_name(str(py_file))
            imports = self._extract_imports(py_file)
            self._import_graph[module_name] = imports
        
        return self._import_graph
    
    def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Set of imported module names
        """
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        
        except Exception as e:
            print(f"Error extracting imports from {file_path}: {e}")
        
        return imports
    
    def _get_module_name(self, file_path: str) -> str:
        """Convert file path to module name."""
        path = Path(file_path)
        if path.suffix == '.py':
            path = path.with_suffix('')
        
        parts = path.parts
        if 'src' in parts:
            src_index = parts.index('src')
            parts = parts[src_index + 1:]
        
        return '.'.join(parts)
    
    def detect_circular_imports(
        self,
        import_graph: Optional[Dict[str, Set[str]]] = None,
    ) -> List[CircularImportPath]:
        """Detect circular import paths in the import graph.
        
        Args:
            import_graph: Import graph to analyze
            
        Returns:
            List of circular import paths found
        """
        if import_graph is None:
            import_graph = self._import_graph
        
        circular_paths = []
        visited = set()
        
        def dfs_detect_cycle(
            module: str,
            path: List[str],
            visiting: Set[str],
        ) -> None:
            if module in visiting:
                # Found a cycle
                cycle_start = path.index(module)
                cycle_modules = path[cycle_start:] + [module]
                
                # Create import chain
                import_chain = []
                for i in range(len(cycle_modules) - 1):
                    import_chain.append((cycle_modules[i], cycle_modules[i + 1]))
                
                circular_paths.append(CircularImportPath(
                    modules=cycle_modules,
                    import_chain=import_chain,
                    severity="ERROR"
                ))
                return
            
            if module in visited:
                return
            
            visiting.add(module)
            path.append(module)
            
            # Visit all imports of this module
            for imported_module in import_graph.get(module, set()):
                if imported_module in import_graph:  # Only consider internal modules
                    dfs_detect_cycle(imported_module, path, visiting)
            
            path.pop()
            visiting.remove(module)
            visited.add(module)
        
        # Check each module for cycles
        for module in import_graph:
            if module not in visited:
                dfs_detect_cycle(module, [], set())
        
        return circular_paths
    
    def generate_circular_import_report(
        self,
        circular_paths: List[CircularImportPath],
    ) -> Dict[str, Any]:
        """Generate a report of circular import issues.
        
        Args:
            circular_paths: List of circular import paths
            
        Returns:
            Detailed report of circular imports
        """
        report = {
            "total_circular_imports": len(circular_paths),
            "circular_import_details": [],
            "affected_modules": set(),
            "recommendations": []
        }
        
        for path in circular_paths:
            report["circular_import_details"].append({
                "modules": path.modules,
                "import_chain": [f"{src} -> {dst}" for src, dst in path.import_chain],
                "severity": path.severity,
                "cycle_length": len(path.modules) - 1
            })
            
            report["affected_modules"].update(path.modules)
        
        report["affected_modules"] = list(report["affected_modules"])
        
        if circular_paths:
            report["recommendations"] = [
                "Implement protocol boundaries to break circular dependencies",
                "Move shared types to a common domain module",
                "Use dependency injection to invert dependencies",
                "Create facade patterns to hide implementation details",
                "Consider splitting large modules into smaller, focused modules"
            ]
        
        return report


class DependencyValidator:
    """Validates dependencies follow clean architecture principles."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize dependency validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.layer_enforcer = LayerBoundaryEnforcer(project_root)
        self.circular_detector = CircularImportDetector(project_root)
    
    def validate_project_architecture(
        self,
        src_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Validate entire project architecture.
        
        Args:
            src_path: Source directory to validate
            
        Returns:
            Comprehensive validation report
        """
        # Check layer violations
        layer_violations = self.layer_enforcer.scan_project_violations(src_path)
        
        # Check circular imports
        import_graph = self.circular_detector.build_import_graph(src_path)
        circular_imports = self.circular_detector.detect_circular_imports(import_graph)
        
        # Generate reports
        layer_report = self.layer_enforcer.generate_violation_report(layer_violations)
        circular_report = self.circular_detector.generate_circular_import_report(circular_imports)
        
        # Combined report
        overall_health = HealthStatus.HEALTHY
        if layer_violations or circular_imports:
            overall_health = HealthStatus.UNHEALTHY
        
        return {
            "overall_health": overall_health.value,
            "layer_violations": layer_report,
            "circular_imports": circular_report,
            "summary": {
                "total_layer_violations": len(layer_violations),
                "total_circular_imports": len(circular_imports),
                "architecture_compliant": len(layer_violations) == 0 and len(circular_imports) == 0
            },
            "next_steps": self._generate_next_steps(layer_violations, circular_imports)
        }
    
    def _generate_next_steps(
        self,
        layer_violations: List[ImportViolation],
        circular_imports: List[CircularImportPath],
    ) -> List[str]:
        """Generate actionable next steps for fixing issues.
        
        Args:
            layer_violations: Layer boundary violations
            circular_imports: Circular import paths
            
        Returns:
            List of recommended actions
        """
        steps = []
        
        if layer_violations:
            steps.append("1. Fix layer boundary violations by implementing protocol boundaries")
            steps.append("2. Move shared types to domain layer to eliminate infrastructure dependencies") 
            steps.append("3. Use dependency injection with protocols to invert dependencies")
        
        if circular_imports:
            steps.append("4. Break circular import chains by introducing protocol interfaces")
            steps.append("5. Refactor god objects into smaller, focused services")
            steps.append("6. Implement facade patterns to hide internal dependencies")
        
        if not layer_violations and not circular_imports:
            steps.append("Architecture is clean! Consider adding runtime validation to prevent regressions.")
        
        return steps


def validate_architecture(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Entry point for validating project architecture.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Architecture validation report
    """
    validator = DependencyValidator(project_root)
    return validator.validate_project_architecture()