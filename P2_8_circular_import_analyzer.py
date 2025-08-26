#!/usr/bin/env python3
"""P2.8: Comprehensive Circular Import Analysis for Protocol Consolidation
Validates no circular dependencies introduced by protocol consolidation.
"""
import ast
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ImportAnalysis:
    file_path: str
    imports: list[str]
    from_imports: dict[str, list[str]]


@dataclass
class CircularDependency:
    cycle: list[str]
    cycle_length: int
    impact_level: str  # "low", "medium", "high", "critical"


@dataclass
class AnalysisReport:
    total_files_analyzed: int
    circular_dependencies_found: list[CircularDependency]
    protocol_consolidation_violations: list[str]
    clean_architecture_violations: list[str]
    analysis_status: str
    performance_metrics: dict[str, float]


class CircularImportAnalyzer:
    def __init__(self, root_path: str) -> None:
        self.root_path = Path(root_path)
        self.src_path = self.root_path / "src" / "prompt_improver"
        self.import_graph: dict[str, set[str]] = defaultdict(set)
        self.file_imports: dict[str, ImportAnalysis] = {}
        self.analyzed_files = 0

        # Protocol consolidation areas to validate
        self.protocol_areas = [
            "shared.interfaces.protocols.core",
            "shared.interfaces.protocols.database",
            "shared.interfaces.protocols.ml",
            "shared.interfaces.protocols.cache",
            "shared.interfaces.protocols.monitoring",
            "shared.interfaces.protocols.security",
            "shared.interfaces.protocols.application",
            "shared.interfaces.protocols.cli",
            "shared.interfaces.protocols.mcp"
        ]

        # Clean Architecture layers
        self.architecture_layers = {
            "presentation": ["api", "cli", "tui"],
            "application": ["application"],
            "domain": ["core.domain"],
            "repository": ["repositories"],
            "infrastructure": ["database", "ml", "services", "monitoring"]
        }

    def analyze_file(self, file_path: Path) -> ImportAnalysis | None:
        """Analyze a single Python file for import statements."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []
            from_imports = defaultdict(list)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    from_imports[node.module].extend(
                        alias.name for alias in node.names
                    )

            rel_path = file_path.relative_to(self.src_path)
            module_path = str(rel_path.with_suffix(''))

            analysis = ImportAnalysis(
                file_path=str(rel_path),
                imports=imports,
                from_imports=dict(from_imports)
            )

            self.file_imports[module_path] = analysis
            self.analyzed_files += 1
            return analysis

        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
            return None

    def build_import_graph(self) -> None:
        """Build the complete import dependency graph."""
        print("🔍 Building import dependency graph...")

        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py" and py_file.stat().st_size == 0:
                continue  # Skip empty __init__.py files

            analysis = self.analyze_file(py_file)
            if not analysis:
                continue

            rel_path = py_file.relative_to(self.src_path)
            module_name = str(rel_path.with_suffix('')).replace('/', '.')

            # Add direct imports
            for imp in analysis.imports:
                if imp.startswith('prompt_improver'):
                    clean_imp = imp.replace('prompt_improver.', '')
                    self.import_graph[module_name].add(clean_imp)

            # Add from imports
            for from_module in analysis.from_imports:
                if from_module and from_module.startswith('prompt_improver'):
                    clean_from = from_module.replace('prompt_improver.', '')
                    self.import_graph[module_name].add(clean_from)

    def detect_circular_dependencies(self) -> list[CircularDependency]:
        """Detect circular dependencies using DFS cycle detection."""
        print("🔄 Detecting circular dependencies...")

        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = [*path[cycle_start:], node]

                # Calculate impact level
                impact = self._calculate_impact_level(cycle)

                cycles.append(CircularDependency(
                    cycle=cycle,
                    cycle_length=len(cycle) - 1,
                    impact_level=impact
                ))
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.import_graph.get(node, []):
                dfs(neighbor)

            rec_stack.remove(node)
            path.pop()

        for module in self.import_graph:
            if module not in visited:
                dfs(module)

        return cycles

    def _calculate_impact_level(self, cycle: list[str]) -> str:
        """Calculate impact level of a circular dependency."""
        # Check if cycle involves protocol consolidation areas
        protocol_involved = any(
            any(area in module for area in self.protocol_areas)
            for module in cycle
        )

        # Check if cycle crosses architecture boundaries
        layers_involved = set()
        for module in cycle:
            for layer, modules in self.architecture_layers.items():
                if any(mod in module for mod in modules):
                    layers_involved.add(layer)

        if len(layers_involved) > 2:
            return "critical"
        if protocol_involved and len(layers_involved) > 1:
            return "high"
        if protocol_involved or len(layers_involved) > 1:
            return "medium"
        return "low"

    def validate_protocol_consolidation(self) -> list[str]:
        """Validate protocol consolidation hasn't introduced violations."""
        print("🔧 Validating protocol consolidation...")

        violations = []

        # Check for cross-dependencies between protocol areas
        for module, imports in self.import_graph.items():
            if "shared.interfaces.protocols" in module:
                protocol_area = None
                for area in self.protocol_areas:
                    if area in module:
                        protocol_area = area
                        break

                if protocol_area:
                    for imp in imports:
                        if "shared.interfaces.protocols" in imp:
                            # Check if importing from different protocol area
                            importing_area = None
                            for area in self.protocol_areas:
                                if area in imp and area != protocol_area:
                                    importing_area = area
                                    break

                            if importing_area:
                                violations.append(
                                    f"Protocol cross-dependency: {protocol_area} → {importing_area}"
                                )

        return violations

    def validate_clean_architecture(self) -> list[str]:
        """Validate Clean Architecture boundaries."""
        print("🏗️  Validating Clean Architecture boundaries...")

        violations = []

        # Define forbidden dependencies (reverse dependencies)
        forbidden = {
            "domain": ["application", "presentation", "repository", "infrastructure"],
            "application": ["presentation"],
            "repository": ["presentation", "application"],
            "infrastructure": ["presentation", "application", "domain", "repository"]
        }

        for module, imports in self.import_graph.items():
            module_layer = self._get_module_layer(module)
            if not module_layer:
                continue

            for imp in imports:
                import_layer = self._get_module_layer(imp)
                if not import_layer:
                    continue

                if import_layer in forbidden.get(module_layer, []):
                    violations.append(
                        f"Architecture violation: {module_layer}({module}) → {import_layer}({imp})"
                    )

        return violations

    def _get_module_layer(self, module: str) -> str | None:
        """Determine which architecture layer a module belongs to."""
        for layer, modules in self.architecture_layers.items():
            if any(mod in module for mod in modules):
                return layer
        return None

    def generate_report(self) -> AnalysisReport:
        """Generate comprehensive analysis report."""
        print("📊 Generating analysis report...")

        import time
        start_time = time.time()

        # Build import graph
        self.build_import_graph()
        graph_time = time.time()

        # Detect circular dependencies
        circular_deps = self.detect_circular_dependencies()
        circular_time = time.time()

        # Validate protocol consolidation
        protocol_violations = self.validate_protocol_consolidation()
        protocol_time = time.time()

        # Validate clean architecture
        architecture_violations = self.validate_clean_architecture()
        arch_time = time.time()

        # Determine overall status
        if circular_deps or protocol_violations or architecture_violations:
            status = "VIOLATIONS_FOUND"
        else:
            status = "CLEAN"

        performance_metrics = {
            "total_analysis_time": arch_time - start_time,
            "graph_build_time": graph_time - start_time,
            "circular_detection_time": circular_time - graph_time,
            "protocol_validation_time": protocol_time - circular_time,
            "architecture_validation_time": arch_time - protocol_time,
            "files_per_second": self.analyzed_files / (arch_time - start_time) if arch_time > start_time else 0
        }

        return AnalysisReport(
            total_files_analyzed=self.analyzed_files,
            circular_dependencies_found=circular_deps,
            protocol_consolidation_violations=protocol_violations,
            clean_architecture_violations=architecture_violations,
            analysis_status=status,
            performance_metrics=performance_metrics
        )


def main():
    print("=" * 70)
    print("P2.8: CIRCULAR IMPORT ANALYSIS FOR PROTOCOL CONSOLIDATION")
    print("=" * 70)

    root_path = "/Users/lukemckenzie/prompt-improver"
    analyzer = CircularImportAnalyzer(root_path)

    try:
        report = analyzer.generate_report()

        print("\n📈 ANALYSIS RESULTS:")
        print(f"   Total files analyzed: {report.total_files_analyzed}")
        print(f"   Analysis status: {report.analysis_status}")
        print(f"   Analysis time: {report.performance_metrics['total_analysis_time']:.2f}s")
        print(f"   Files/second: {report.performance_metrics['files_per_second']:.1f}")

        print(f"\n🔄 CIRCULAR DEPENDENCIES: {len(report.circular_dependencies_found)}")
        if report.circular_dependencies_found:
            for i, cycle in enumerate(report.circular_dependencies_found, 1):
                print(f"   {i}. {cycle.impact_level.upper()} impact ({cycle.cycle_length} modules)")
                print(f"      Cycle: {' → '.join(cycle.cycle)}")
        else:
            print("   ✅ No circular dependencies detected")

        print(f"\n🔧 PROTOCOL CONSOLIDATION VIOLATIONS: {len(report.protocol_consolidation_violations)}")
        if report.protocol_consolidation_violations:
            for violation in report.protocol_consolidation_violations:
                print(f"   ❌ {violation}")
        else:
            print("   ✅ Protocol consolidation integrity maintained")

        print(f"\n🏗️  CLEAN ARCHITECTURE VIOLATIONS: {len(report.clean_architecture_violations)}")
        if report.clean_architecture_violations:
            for violation in report.clean_architecture_violations:
                print(f"   ❌ {violation}")
        else:
            print("   ✅ Clean Architecture boundaries maintained")

        # Save detailed report
        report_file = f"{root_path}/P2_8_circular_import_analysis_report.json"
        with open(report_file, 'w', encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\n💾 Detailed report saved: {report_file}")

        print("\n" + "=" * 70)
        if report.analysis_status == "CLEAN":
            print("🎉 SUCCESS: No circular dependencies or architecture violations found!")
            print("✅ Protocol consolidation has maintained architectural integrity.")
            return 0
        print("❌ VIOLATIONS DETECTED: Review and fix the issues above.")
        return 1

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
