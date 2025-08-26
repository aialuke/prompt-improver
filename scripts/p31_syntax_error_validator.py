#!/usr/bin/env python3
"""P3.1 Syntax Error Validator
Validates the syntax error inventory and categorizes remediation priorities.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


class SyntaxError(NamedTuple):
    file_path: str
    line: int
    column: int
    error_type: str
    message: str
    category: str
    priority: int


def categorize_error(file_path: str, message: str, line_content: str = "") -> tuple[str, int]:
    """Categorize syntax error and assign priority."""
    # Category 1: ML Lazy Loading Import Malformation (CRITICAL)
    if "from get_" in line_content or "get_sklearn" in line_content or "get_numpy" in line_content:
        return "ML_LAZY_LOADING", 1

    # Category 2: Incomplete Try-Except Blocks (HIGH)
    if "expected 'except' or 'finally' block" in message:
        return "INCOMPLETE_TRY_EXCEPT", 2

    # Category 3: Mixed Import Block Structure (MEDIUM)
    if "invalid syntax" in message and ("import" in line_content or "from" in line_content):
        return "MIXED_IMPORT_STRUCTURE", 3

    # Category 4: Indentation and Structural Errors (LOW)
    if "unexpected indent" in message:
        return "INDENTATION_ERROR", 4

    # Default category
    return "OTHER_SYNTAX", 4


def validate_file_syntax(file_path: Path) -> list[SyntaxError]:
    """Validate single file and return syntax errors."""
    errors = []

    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Get lines for context
        lines = content.split('\n')

        # Try to parse with AST
        try:
            ast.parse(content, filename=str(file_path))
            return []  # No syntax errors
        except SyntaxError as e:
            line_content = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
            category, priority = categorize_error(str(file_path), str(e.msg), line_content)

            errors.append(SyntaxError(
                file_path=str(file_path),
                line=e.lineno or 0,
                column=e.offset or 0,
                error_type="SyntaxError",
                message=e.msg or "Unknown syntax error",
                category=category,
                priority=priority
            ))

    except Exception as e:
        errors.append(SyntaxError(
            file_path=str(file_path),
            line=1,
            column=1,
            error_type=type(e).__name__,
            message=str(e),
            category="FILE_READ_ERROR",
            priority=4
        ))

    return errors


def analyze_syntax_errors() -> dict[str, list[SyntaxError]]:
    """Analyze all Python files and categorize syntax errors."""
    print("🔍 P3.1: Starting comprehensive syntax error analysis...")

    all_errors = []
    src_path = Path("src")

    # Scan all Python files
    python_files = list(src_path.rglob("*.py"))
    print(f"📁 Scanning {len(python_files)} Python files...")

    for py_file in python_files:
        errors = validate_file_syntax(py_file)
        all_errors.extend(errors)

    # Categorize errors
    categorized = defaultdict(list)
    for error in all_errors:
        categorized[error.category].append(error)

    return dict(categorized)


def print_analysis_results(categorized_errors: dict[str, list[SyntaxError]]):
    """Print detailed analysis results."""
    total_errors = sum(len(errors) for errors in categorized_errors.values())
    total_files = len({error.file_path for errors in categorized_errors.values() for error in errors})

    print("\n🎯 P3.1 SYNTAX ERROR INVENTORY COMPLETE")
    print("═══════════════════════════════════════")
    print(f"Total Errors: {total_errors}")
    print(f"Affected Files: {total_files}")
    print(f"Categories: {len(categorized_errors)}")

    # Priority breakdown
    priority_counts = defaultdict(int)
    for errors in categorized_errors.values():
        for error in errors:
            priority_counts[error.priority] += 1

    print("\n📊 PRIORITY BREAKDOWN:")
    for priority in sorted(priority_counts.keys()):
        priority_name = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}[priority]
        print(f"Priority {priority} ({priority_name}): {priority_counts[priority]} errors")

    # Category details
    print("\n📋 CATEGORY BREAKDOWN:")
    for category, errors in sorted(categorized_errors.items()):
        print(f"\n{category} ({len(errors)} errors):")

        # Group by file for cleaner output
        by_file = defaultdict(list)
        for error in errors:
            by_file[error.file_path].append(error)

        for file_path, file_errors in sorted(by_file.items()):
            rel_path = str(Path(file_path).relative_to(Path.cwd()))
            print(f"  📄 {rel_path}")
            for error in file_errors[:3]:  # Show first 3 errors per file
                print(f"     L{error.line:>3}:{error.column:<2} {error.message}")
            if len(file_errors) > 3:
                print(f"     ... and {len(file_errors) - 3} more errors")


def validate_ml_lazy_loading_patterns():
    """Specific validation for ML lazy loading patterns."""
    print("\n🧪 VALIDATING ML LAZY LOADING PATTERNS:")
    print("═══════════════════════════════════════")

    # Search for problematic patterns
    patterns_to_check = [
        (r"from get_\w+\(\)\.\w+", "Direct get_*() import pattern"),
        (r"from get_\w+\(\)$", "Incomplete get_*() import"),
        (r"get_\w+\(\)\.\w+", "Direct get_*() usage in type hints"),
        (r"import.*get_", "Mixed get_ import patterns")
    ]

    issues_found = []
    src_path = Path("src")

    for py_file in src_path.rglob("*.py"):
        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()
                lines = content.split('\n')

            for line_no, line in enumerate(lines, 1):
                for pattern, description in patterns_to_check:
                    if re.search(pattern, line):
                        rel_path = str(py_file.relative_to(Path.cwd()))
                        issues_found.append(f"  📄 {rel_path}:{line_no} - {description}")

        except Exception as e:
            continue

    print(f"Found {len(issues_found)} lazy loading pattern issues:")
    for issue in sorted(set(issues_found))[:20]:  # Show first 20 unique issues
        print(issue)

    if len(issues_found) > 20:
        print(f"  ... and {len(issues_found) - 20} more issues")


def main():
    """Main analysis function."""
    # Core syntax error analysis
    categorized_errors = analyze_syntax_errors()
    print_analysis_results(categorized_errors)

    # ML-specific pattern validation
    validate_ml_lazy_loading_patterns()

    # Summary and next steps
    total_errors = sum(len(errors) for errors in categorized_errors.values())

    print("\n✅ P3.1 ANALYSIS COMPLETE")
    print("═════════════════════════")
    print(f"Status: {total_errors} syntax errors identified and categorized")
    print("Next Phase: P3.2 - ML Lazy Loading Import Remediation")
    print(f"Priority: Start with {len(categorized_errors.get('ML_LAZY_LOADING', []))} ML lazy loading errors")
    print("ETA: 4-6 hours for complete Phase 3 remediation")


if __name__ == "__main__":
    main()
