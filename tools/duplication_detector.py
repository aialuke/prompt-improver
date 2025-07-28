#!/usr/bin/env python3
"""
Automated code duplication detection tool.

Analyzes the codebase for various types of duplication and provides
refactoring recommendations.

Usage:
    python tools/duplication_detector.py --analyze
    python tools/duplication_detector.py --report
    python tools/duplication_detector.py --fix-common
"""

import ast
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import argparse
import json


@dataclass
class DuplicationReport:
    """Report of code duplication findings."""
    duplicate_imports: Dict[str, List[str]]
    duplicate_logger_inits: List[str]
    duplicate_config_patterns: List[str]
    duplicate_metrics_patterns: List[str]
    duplicate_error_patterns: List[str]
    similar_classes: Dict[str, List[str]]
    similar_functions: Dict[str, List[str]]
    total_files_analyzed: int
    recommendations: List[str]


class DuplicationDetector:
    """Detect various forms of code duplication."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.source_dir = self.project_root / "src" / "prompt_improver"

    def analyze_project(self) -> DuplicationReport:
        """Analyze the entire project for duplication."""
        print("🔍 Analyzing codebase for duplication patterns...")

        python_files = list(self.source_dir.rglob("*.py"))

        # Initialize report data
        duplicate_imports: Dict[str, List[str]] = defaultdict(list)
        duplicate_logger_inits: List[str] = []
        duplicate_config_patterns: List[str] = []
        duplicate_metrics_patterns: List[str] = []
        duplicate_error_patterns: List[str] = []
        similar_classes: Dict[str, List[str]] = defaultdict(list)
        similar_functions: Dict[str, List[str]] = defaultdict(list)

        import_patterns: Dict[str, List[str]] = defaultdict(list)
        logger_pattern = re.compile(r'logger\s*=\s*logging\.getLogger\(__name__\)')
        config_pattern = re.compile(r'try:\s*.*config.*=.*get_config|config.*=.*get_config.*except')
        metrics_pattern = re.compile(r'self\.metrics_registry\s*=\s*get_metrics_registry\(\)')

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Analyze imports
                self._analyze_imports(content, str(file_path), import_patterns)

                # Check for duplicate logger initializations
                if logger_pattern.search(content):
                    duplicate_logger_inits.append(str(file_path))

                # Check for duplicate config patterns
                if config_pattern.search(content, re.MULTILINE | re.DOTALL):
                    duplicate_config_patterns.append(str(file_path))

                # Check for duplicate metrics patterns
                if metrics_pattern.search(content):
                    duplicate_metrics_patterns.append(str(file_path))

                # Check for duplicate error patterns
                if self._has_duplicate_error_pattern(content):
                    duplicate_error_patterns.append(str(file_path))

                # Analyze AST for similar structures
                try:
                    tree = ast.parse(content, filename=str(file_path))
                    self._analyze_ast_similarities(tree, str(file_path), similar_classes, similar_functions)
                except SyntaxError:
                    print(f"⚠️  Syntax error in {file_path}, skipping AST analysis")

            except Exception as e:
                print(f"⚠️  Error analyzing {file_path}: {e}")

        # Find actual duplicates in imports
        for pattern, files in import_patterns.items():
            if len(files) > 5:  # Only report if used in many files
                duplicate_imports[pattern] = files

        # Generate recommendations
        recommendations = self._generate_recommendations(
            duplicate_imports, duplicate_logger_inits, duplicate_config_patterns,
            duplicate_metrics_patterns, duplicate_error_patterns,
            similar_classes, similar_functions
        )

        return DuplicationReport(
            duplicate_imports=dict(duplicate_imports),
            duplicate_logger_inits=duplicate_logger_inits,
            duplicate_config_patterns=duplicate_config_patterns,
            duplicate_metrics_patterns=duplicate_metrics_patterns,
            duplicate_error_patterns=duplicate_error_patterns,
            similar_classes=dict(similar_classes),
            similar_functions=dict(similar_functions),
            total_files_analyzed=len(python_files),
            recommendations=recommendations
        )

    def _analyze_imports(self, content: str, file_path: str, import_patterns: Dict[str, List[str]]) -> None:
        """Analyze import patterns in file."""
        import_lines = [line.strip() for line in content.split('\n')
                       if line.strip().startswith(('import ', 'from '))]

        for import_line in import_lines:
            # Normalize common patterns
            normalized = self._normalize_import(import_line)
            if normalized:
                import_patterns[normalized].append(file_path)

    def _normalize_import(self, import_line: str) -> Optional[str]:
        """Normalize import statement for pattern matching."""
        # Common patterns to track
        patterns = [
            r'import logging',
            r'from.*logging.*import.*getLogger',
            r'from.*metrics_registry.*import.*get_metrics_registry',
            r'from.*config.*import.*get_config',
            r'from pydantic import BaseModel',
            r'from typing import.*Optional.*Dict.*Any',
            r'import asyncio',
            r'from datetime import datetime',
            r'from dataclasses import dataclass'
        ]

        for pattern in patterns:
            if re.search(pattern, import_line):
                return pattern

        return None

    def _has_duplicate_error_pattern(self, content: str) -> bool:
        """Check for common error handling patterns."""
        error_patterns = [
            r'try:\s*.*except.*Exception.*:.*logger\.warning',
            r'try:\s*.*except.*:.*fallback',
            r'except.*Exception.*as.*e:.*logger\.error'
        ]

        for pattern in error_patterns:
            if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                return True

        return False

    def _analyze_ast_similarities(self, tree: ast.AST, file_path: str,
                                 similar_classes: Dict[str, List[str]],
                                 similar_functions: Dict[str, List[str]]):
        """Analyze AST for similar class and function structures."""

        class SimilarityAnalyzer(ast.NodeVisitor):
            def __init__(self) -> None:
                self.classes: List[tuple[str, str, str]] = []
                self.functions: List[tuple[str, str, str]] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                # Create signature for class structure
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                attrs = [n.targets[0].id for n in node.body
                        if isinstance(n, ast.Assign) and len(n.targets) == 1
                        and isinstance(n.targets[0], ast.Name)]

                signature = f"class_{len(methods)}_methods_{len(attrs)}_attrs"
                if methods:
                    signature += f"_{hashlib.md5('_'.join(sorted(methods)).encode()).hexdigest()[:8]}"

                self.classes.append((signature, node.name, file_path))
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # Create signature for function structure
                args_count = len(node.args.args)
                decorators_count = len(node.decorator_list)
                body_types = [type(n).__name__ for n in node.body[:3]]  # First 3 statements

                signature = f"func_{args_count}_args_{decorators_count}_decorators_{'_'.join(body_types)}"
                self.functions.append((signature, node.name, file_path))
                self.generic_visit(node)

        analyzer = SimilarityAnalyzer()
        analyzer.visit(tree)

        # Group similar structures
        for signature, name, path in analyzer.classes:
            similar_classes[signature].append(f"{name} in {path}")

        for signature, name, path in analyzer.functions:
            similar_functions[signature].append(f"{name} in {path}")

    def _generate_recommendations(self,
                                duplicate_imports: Dict[str, List[str]],
                                duplicate_logger_inits: List[str],
                                duplicate_config_patterns: List[str],
                                duplicate_metrics_patterns: List[str],
                                duplicate_error_patterns: List[str],
                                similar_classes: Dict[str, List[str]],
                                similar_functions: Dict[str, List[str]]) -> List[str]:
        """Generate refactoring recommendations."""
        # Note: similar_functions parameter reserved for future functionality
        _ = similar_functions  # Suppress unused parameter warning
        recommendations: List[str] = []

        if duplicate_logger_inits:
            recommendations.append(
                f"🔄 CRITICAL: Replace {len(duplicate_logger_inits)} duplicate logger initializations "
                f"with 'from prompt_improver.core.common import get_logger'"
            )

        if duplicate_config_patterns:
            recommendations.append(
                f"🔄 HIGH: Consolidate {len(duplicate_config_patterns)} duplicate config loading patterns "
                f"using ConfigMixin or get_config_safely()"
            )

        if duplicate_metrics_patterns:
            recommendations.append(
                f"🔄 HIGH: Consolidate {len(duplicate_metrics_patterns)} duplicate metrics patterns "
                f"using MetricsMixin"
            )

        if duplicate_error_patterns:
            recommendations.append(
                f"🔄 MEDIUM: Standardize {len(duplicate_error_patterns)} duplicate error handling patterns "
                f"using error_handling utilities"
            )

        # Find classes with many similar structures
        large_similar_groups = {k: v for k, v in similar_classes.items() if len(v) > 3}
        if large_similar_groups:
            recommendations.append(
                f"🔄 MEDIUM: Consider creating base classes for {len(large_similar_groups)} "
                f"groups of similar classes"
            )

        # Check for common imports that could be consolidated
        high_usage_imports = {k: v for k, v in duplicate_imports.items() if len(v) > 10}
        if high_usage_imports:
            recommendations.append(
                f"🔄 LOW: Consider creating import shortcuts for {len(high_usage_imports)} "
                f"frequently used import patterns"
            )

        return recommendations

    def generate_report(self, report: DuplicationReport) -> str:
        """Generate a formatted report."""
        lines = [
            "# Code Duplication Analysis Report",
            f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Files Analyzed:** {report.total_files_analyzed}",
            "",
            "## 📊 Summary",
            "",
            f"- **Duplicate Logger Initializations:** {len(report.duplicate_logger_inits)}",
            f"- **Duplicate Config Patterns:** {len(report.duplicate_config_patterns)}",
            f"- **Duplicate Metrics Patterns:** {len(report.duplicate_metrics_patterns)}",
            f"- **Duplicate Error Patterns:** {len(report.duplicate_error_patterns)}",
            f"- **Similar Class Groups:** {len([g for g in report.similar_classes.values() if len(g) > 2])}",
            f"- **Similar Function Groups:** {len([g for g in report.similar_functions.values() if len(g) > 2])}",
            "",
            "## 🎯 Recommendations",
            ""
        ]

        for rec in report.recommendations:
            lines.append(f"- {rec}")

        lines.extend([
            "",
            "## 📋 Detailed Findings",
            "",
            "### Logger Initialization Duplication",
            f"Found {len(report.duplicate_logger_inits)} files with duplicate `logger = logging.getLogger(__name__)` patterns:",
            ""
        ])

        for file_path in report.duplicate_logger_inits[:10]:  # Show first 10
            lines.append(f"- `{file_path}`")

        if len(report.duplicate_logger_inits) > 10:
            lines.append(f"- ... and {len(report.duplicate_logger_inits) - 10} more files")

        lines.extend([
            "",
            "### Configuration Loading Duplication",
            f"Found {len(report.duplicate_config_patterns)} files with duplicate config loading patterns:",
            ""
        ])

        for file_path in report.duplicate_config_patterns[:10]:
            lines.append(f"- `{file_path}`")

        if len(report.duplicate_config_patterns) > 10:
            lines.append(f"- ... and {len(report.duplicate_config_patterns) - 10} more files")

        lines.extend([
            "",
            "### Metrics Registry Duplication",
            f"Found {len(report.duplicate_metrics_patterns)} files with duplicate metrics registry patterns:",
            ""
        ])

        for file_path in report.duplicate_metrics_patterns[:10]:
            lines.append(f"- `{file_path}`")

        if len(report.duplicate_metrics_patterns) > 10:
            lines.append(f"- ... and {len(report.duplicate_metrics_patterns) - 10} more files")

        # Add similar classes section
        similar_groups = [g for g in report.similar_classes.values() if len(g) > 2]
        if similar_groups:
            lines.extend([
                "",
                "### Similar Class Structures",
                f"Found {len(similar_groups)} groups of classes with similar structures:",
                ""
            ])

            for i, group in enumerate(similar_groups[:5]):  # Show first 5 groups
                lines.append(f"**Group {i+1}:** {len(group)} similar classes")
                for class_info in group[:3]:  # Show first 3 in each group
                    lines.append(f"  - {class_info}")
                if len(group) > 3:
                    lines.append(f"  - ... and {len(group) - 3} more")
                lines.append("")

        return "\n".join(lines)

    def save_report(self, report: DuplicationReport, output_path: str):
        """Save report to file."""
        report_text = self.generate_report(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # Also save JSON version for automated processing
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'duplicate_imports': report.duplicate_imports,
                'duplicate_logger_inits': report.duplicate_logger_inits,
                'duplicate_config_patterns': report.duplicate_config_patterns,
                'duplicate_metrics_patterns': report.duplicate_metrics_patterns,
                'duplicate_error_patterns': report.duplicate_error_patterns,
                'similar_classes': report.similar_classes,
                'similar_functions': report.similar_functions,
                'total_files_analyzed': report.total_files_analyzed,
                'recommendations': report.recommendations
            }, f, indent=2)

        print(f"📄 Report saved to: {output_path}")
        print(f"📄 JSON data saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze code duplication in the project")
    parser.add_argument("--analyze", action="store_true", help="Analyze codebase for duplication")
    parser.add_argument("--report", action="store_true", help="Generate duplication report")
    parser.add_argument("--output", default="DUPLICATION_ANALYSIS.md", help="Output file for report")
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    detector = DuplicationDetector(args.project_root)

    if args.analyze or args.report:
        print("🚀 Starting duplication analysis...")
        report = detector.analyze_project()

        print("\n📊 Analysis Complete!")
        print(f"📁 Files analyzed: {report.total_files_analyzed}")
        print(f"🔄 Logger duplications: {len(report.duplicate_logger_inits)}")
        print(f"🔄 Config duplications: {len(report.duplicate_config_patterns)}")
        print(f"🔄 Metrics duplications: {len(report.duplicate_metrics_patterns)}")
        print(f"🔄 Error pattern duplications: {len(report.duplicate_error_patterns)}")

        if args.report:
            detector.save_report(report, args.output)
            print(f"\n✅ Full report generated: {args.output}")

        # Print top recommendations
        print("\n🎯 Top Recommendations:")
        for rec in report.recommendations[:3]:
            print(f"  {rec}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
