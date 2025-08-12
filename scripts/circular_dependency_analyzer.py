"""Circular Dependency Analyzer
Identifies circular import dependencies in the codebase and provides resolution strategies.
"""

import ast
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class CircularDependencyAnalyzer:
    """Analyzes Python files for circular import dependencies."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.src_path = self.root_path / "src"
        self.dependencies = defaultdict(set)
        self.modules = set()

    def extract_imports(self, file_path: Path) -> set[str]:
        """Extract all imports from a Python file."""
        imports = set()
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        for alias in node.names:
                            if alias.name != "*":
                                imports.add(f"{node.module}.{alias.name}")
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"⚠️  Could not parse {file_path}: {e}")
        return imports

    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(self.src_path)
            if relative_path.name == "__init__.py":
                module_parts = relative_path.parent.parts
            else:
                module_parts = relative_path.with_suffix("").parts
            return ".".join(module_parts)
        except ValueError:
            return str(file_path)

    def scan_directory(self) -> None:
        """Scan all Python files and build dependency graph."""
        print("🔍 Scanning Python files for imports...")
        python_files = list(self.src_path.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        for file_path in python_files:
            module_name = self.get_module_name(file_path)
            self.modules.add(module_name)
            imports = self.extract_imports(file_path)
            internal_imports = {
                imp for imp in imports if imp.startswith("prompt_improver")
            }
            self.dependencies[module_name] = internal_imports

    def find_circular_dependencies(self) -> list[list[str]]:
        """Find all circular dependencies using DFS."""
        print("🔄 Analyzing circular dependencies...")
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: list[str]) -> None:
            if node in rec_stack:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            if node in visited:
                return
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for dep in self.dependencies.get(node, set()):
                if dep in self.modules:
                    dfs(dep, path.copy())
            rec_stack.remove(node)

        for module in self.modules:
            if module not in visited:
                dfs(module, [])
        return cycles

    def analyze_import_patterns(self) -> dict[str, Any]:
        """Analyze import patterns to identify problematic areas."""
        analysis = {
            "total_modules": len(self.modules),
            "total_dependencies": sum(len(deps) for deps in self.dependencies.values()),
            "modules_with_most_imports": [],
            "most_imported_modules": [],
            "potential_circular_areas": [],
        }
        import_counts = [(mod, len(deps)) for mod, deps in self.dependencies.items()]
        import_counts.sort(key=lambda x: x[1], reverse=True)
        analysis["modules_with_most_imports"] = import_counts[:10]
        import_targets = defaultdict(int)
        for deps in self.dependencies.values():
            for dep in deps:
                if dep in self.modules:
                    import_targets[dep] += 1
        most_imported = sorted(import_targets.items(), key=lambda x: x[1], reverse=True)
        analysis["most_imported_modules"] = most_imported[:10]
        potential_circles = []
        for mod1, deps1 in self.dependencies.items():
            for mod2 in deps1:
                if mod2 in self.dependencies and mod1 in self.dependencies[mod2]:
                    potential_circles.append((mod1, mod2))
        analysis["potential_circular_areas"] = potential_circles
        return analysis

    def generate_resolution_strategy(self, cycles: list[list[str]]) -> dict[str, any]:
        """Generate strategies to resolve circular dependencies."""
        strategies = {
            "cycles_found": len(cycles),
            "affected_modules": set(),
            "resolution_steps": [],
            "common_utilities_to_extract": [],
            "interface_candidates": [],
        }
        for cycle in cycles:
            strategies["affected_modules"].update(cycle)
        module_frequency = defaultdict(int)
        for cycle in cycles:
            for module in cycle:
                module_frequency[module] += 1
        frequent_modules = [mod for mod, freq in module_frequency.items() if freq > 1]
        strategies["common_utilities_to_extract"] = frequent_modules
        if cycles:
            strategies["resolution_steps"] = [
                "1. Create prompt_improver.common package for shared utilities",
                "2. Move cross-cutting concerns to common package",
                "3. Extract interfaces for frequently imported modules",
                "4. Use dependency injection for complex dependencies",
                "5. Implement lazy imports where appropriate",
                "6. Update import statements to use absolute imports",
            ]
            for module in frequent_modules:
                if "manager" in module.lower():
                    strategies["interface_candidates"].append(
                        f"{module} -> Extract manager protocol"
                    )
                elif "service" in module.lower():
                    strategies["interface_candidates"].append(
                        f"{module} -> Extract service interface"
                    )
                elif "config" in module.lower():
                    strategies["interface_candidates"].append(
                        f"{module} -> Move to common.config"
                    )
        return strategies

    def save_analysis_report(
        self, cycles: list[list[str]], analysis: dict, strategies: dict
    ) -> str:
        """Save detailed analysis report."""
        report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "summary": {
                "total_modules_scanned": len(self.modules),
                "circular_dependencies_found": len(cycles),
                "affected_modules": len(strategies["affected_modules"]),
            },
            "circular_dependencies": cycles,
            "import_analysis": analysis,
            "resolution_strategies": strategies,
            "detailed_cycles": [],
        }
        for i, cycle in enumerate(cycles):
            cycle_detail = {
                "cycle_id": i + 1,
                "modules": cycle,
                "cycle_length": len(cycle) - 1,
                "severity": "HIGH"
                if len(cycle) > 4
                else "MEDIUM"
                if len(cycle) > 2
                else "LOW",
            }
            report["detailed_cycles"].append(cycle_detail)
        report_path = self.root_path / "circular_dependency_analysis.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return str(report_path)

    def print_analysis_summary(
        self, cycles: list[list[str]], analysis: dict, strategies: dict
    ) -> None:
        """Print human-readable analysis summary."""
        print("\n" + "=" * 60)
        print("🔄 CIRCULAR DEPENDENCY ANALYSIS SUMMARY")
        print("=" * 60)
        print("\n📊 SCAN RESULTS:")
        print(f"   Total modules scanned: {len(self.modules)}")
        print(f"   Total import relationships: {analysis['total_dependencies']}")
        print(f"   Circular dependencies found: {len(cycles)}")
        if cycles:
            print("\n❌ CIRCULAR DEPENDENCIES DETECTED:")
            for i, cycle in enumerate(cycles, 1):
                print(f"   Cycle {i}: {' -> '.join(cycle)}")
                severity = (
                    "HIGH" if len(cycle) > 4 else "MEDIUM" if len(cycle) > 2 else "LOW"
                )
                print(f"   Severity: {severity} (length: {len(cycle) - 1})")
                print()
        else:
            print("\n✅ NO CIRCULAR DEPENDENCIES FOUND!")
        print("\n📈 IMPORT PATTERNS:")
        print("   Modules with most imports:")
        for mod, count in analysis["modules_with_most_imports"][:5]:
            print(f"     {mod}: {count} imports")
        print("\n   Most imported modules:")
        for mod, count in analysis["most_imported_modules"][:5]:
            print(f"     {mod}: imported {count} times")
        if strategies["common_utilities_to_extract"]:
            print("\n🔧 RESOLUTION RECOMMENDATIONS:")
            print("   Modules to refactor (appear in multiple cycles):")
            for mod in strategies["common_utilities_to_extract"]:
                print(f"     - {mod}")
        if strategies["interface_candidates"]:
            print("\n   Interface extraction candidates:")
            for candidate in strategies["interface_candidates"]:
                print(f"     - {candidate}")
        if cycles:
            print("\n📋 RESOLUTION STEPS:")
            for step in strategies["resolution_steps"]:
                print(f"   {step}")


def main():
    """Main execution function."""
    root_path = Path(__file__).parent.parent
    analyzer = CircularDependencyAnalyzer(str(root_path))
    try:
        analyzer.scan_directory()
        cycles = analyzer.find_circular_dependencies()
        analysis = analyzer.analyze_import_patterns()
        strategies = analyzer.generate_resolution_strategy(cycles)
        report_path = analyzer.save_analysis_report(cycles, analysis, strategies)
        analyzer.print_analysis_summary(cycles, analysis, strategies)
        print(f"\n📄 Detailed report saved to: {report_path}")
        return len(cycles) == 0
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
