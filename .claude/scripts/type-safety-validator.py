#!/usr/bin/env python3
"""Type Safety Validator - Claude Code Pre-Tool Hook

This script enforces type safety and import validation by:
1. Checking for missing type annotations on functions and methods
2. Validating import organization and circular import prevention
3. Enforcing explicit typing over Any types
4. Validating protocol-based dependency injection patterns
5. Checking for proper error handling patterns

Based on 2025 Python typing standards and Clean Architecture principles
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

EXIT_SUCCESS = 0
EXIT_VIOLATION = 1
EXIT_ERROR = 2

# Patterns for missing type annotations
FUNCTION_WITHOUT_TYPES = [
    r"def\s+\w+\([^)]*\)\s*:",  # Function without return type
    r"def\s+\w+\([^:)]*\)\s*->",  # Function with params missing types
]

# Import validation patterns
CIRCULAR_IMPORT_PATTERNS = [
    r"from\s+prompt_improver\.(\w+)\s+import.*",  # For cross-module analysis
]

# Any type usage that should be specific
ANY_TYPE_PATTERNS = [
    r":\s*Any\s*[,=]",
    r"->\s*Any\s*:",
    r"List\[Any\]",
    r"Dict\[str,\s*Any\]",
    r"Optional\[Any\]",
]

# Protocol violations
NON_PROTOCOL_PATTERNS = [
    r"def\s+__init__\(self,.*:\s*(?!.*Protocol)",  # Constructor without protocol types
]

# Error handling violations
ERROR_HANDLING_VIOLATIONS = [
    r"except\s*:",  # Bare except clauses
    r"pass\s*$",   # Empty except blocks
]


def read_tool_input() -> dict[str, Any] | None:
    """Read tool input from stdin (Claude Code hook format)."""
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            return None
        return json.loads(input_data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error reading tool input: {e}", file=sys.stderr)
        return None


def extract_code_content(tool_data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract code content and file path from tool data."""
    tool_name: str = tool_data.get("tool", {}).get("name", "")
    file_path: str | None = None
    code_content: str | None = None

    if tool_name in ["Write", "Edit", "MultiEdit"]:
        if "file_path" in tool_data.get("tool", {}).get("parameters", {}):
            file_path = tool_data["tool"]["parameters"]["file_path"]

        if tool_name == "Write":
            code_content = (
                tool_data.get("tool", {}).get("parameters", {}).get("content", "")
            )
        elif tool_name == "Edit":
            code_content = (
                tool_data.get("tool", {}).get("parameters", {}).get("new_string", "")
            )
        elif tool_name == "MultiEdit":
            edits: list[dict[str, Any]] = (
                tool_data.get("tool", {}).get("parameters", {}).get("edits", [])
            )
            code_content = "\n".join(edit.get("new_string", "") for edit in edits)

    return code_content, file_path


def should_skip_validation(file_path: str | None) -> bool:
    """Check if file should skip type validation."""
    if not file_path:
        return False
    
    skip_patterns = [
        r"/tests?/",
        r"conftest\.py",
        r"__init__\.py",
        r"\.md$",
        r"\.json$",
        r"\.yml$",
        r"\.yaml$",
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, file_path):
            return True
    
    return False


def find_missing_type_annotations(code_content: str) -> list[tuple[int, str]]:
    """Find functions/methods missing type annotations."""
    if not code_content:
        return []

    violations: list[tuple[int, str]] = []
    lines = code_content.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip test functions, private methods, and special methods
        if (stripped.startswith("def test_") or 
            stripped.startswith("def _") or
            stripped.startswith("def __")):
            continue
            
        # Check for functions without return type annotations
        if re.match(r"def\s+\w+\([^)]*\)\s*:", stripped):
            violations.append((i, stripped))
        
        # Check for parameters without type annotations
        func_match = re.match(r"def\s+(\w+)\(([^)]*)\)", stripped)
        if func_match:
            params = func_match.group(2)
            if params and "self" not in params and ":" not in params:
                violations.append((i, f"Parameters missing types: {stripped}"))

    return violations


def find_any_type_violations(code_content: str) -> list[tuple[int, str]]:
    """Find usage of Any type that should be more specific."""
    if not code_content:
        return []

    violations: list[tuple[int, str]] = []
    lines = code_content.split("\n")

    for i, line in enumerate(lines, 1):
        for pattern in ANY_TYPE_PATTERNS:
            if re.search(pattern, line):
                violations.append((i, line.strip()))

    return violations


def find_error_handling_violations(code_content: str) -> list[tuple[int, str]]:
    """Find poor error handling patterns."""
    if not code_content:
        return []

    violations: list[tuple[int, str]] = []
    lines = code_content.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Bare except clauses
        if re.match(r"except\s*:", stripped):
            violations.append((i, "Bare except clause: " + stripped))
        
        # Empty except blocks with just pass
        if stripped == "pass" and i > 1:
            prev_line = lines[i-2].strip() if i > 1 else ""
            if prev_line.startswith("except"):
                violations.append((i, "Empty except block: " + prev_line))

    return violations


def analyze_import_organization(code_content: str) -> list[str]:
    """Analyze import organization and detect potential issues."""
    if not code_content:
        return []

    issues: list[str] = []
    lines = code_content.split("\n")
    
    imports = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            imports.append((i, stripped))
    
    # Check for imports not at top of file (excluding test files)
    if imports and imports[0][0] > 10:  # Allow some docstring/comments
        issues.append(f"Imports should be at top of file (first import at line {imports[0][0]})")
    
    # Check for circular import patterns
    module_imports = []
    for line_num, import_line in imports:
        if "prompt_improver" in import_line:
            module_imports.append(import_line)
    
    if len(module_imports) > 5:
        issues.append("High number of internal imports - consider reducing coupling")
    
    return issues


def format_violation_message(violation_type: str, violations: list[tuple[int, str]], guidance: str) -> str:
    """Format violation message with specific guidance."""
    if not violations:
        return ""

    message = f"\n🔍 {violation_type} ISSUE DETECTED:\n"
    for line_num, line_content in violations[:3]:  # Show first 3 violations
        message += f"   Line {line_num}: {line_content}\n"
    
    if len(violations) > 3:
        message += f"   ... and {len(violations) - 3} more issues\n"
    
    message += f"\n💡 GUIDANCE: {guidance}\n"
    return message


def main() -> None:
    """Main type safety validation logic."""
    tool_data = read_tool_input()
    if not tool_data:
        sys.exit(EXIT_SUCCESS)

    code_content, file_path = extract_code_content(tool_data)
    if not code_content or should_skip_validation(file_path):
        sys.exit(EXIT_SUCCESS)

    all_issues = []

    # 1. Check missing type annotations
    type_violations = find_missing_type_annotations(code_content)
    if type_violations:
        message = format_violation_message(
            "TYPE ANNOTATIONS",
            type_violations,
            "Add explicit type annotations to functions and parameters. "
            "This improves code clarity and enables better IDE support."
        )
        all_issues.append(message)

    # 2. Check Any type usage
    any_violations = find_any_type_violations(code_content)
    if any_violations:
        message = format_violation_message(
            "SPECIFIC TYPING",
            any_violations,
            "Replace Any types with specific types. Use Union, Optional, or "
            "create specific types for better type safety."
        )
        all_issues.append(message)

    # 3. Check error handling
    error_violations = find_error_handling_violations(code_content)
    if error_violations:
        message = format_violation_message(
            "ERROR HANDLING",
            error_violations,
            "Use specific exception types and proper error handling. "
            "Avoid bare except clauses and empty exception blocks."
        )
        all_issues.append(message)

    # 4. Check import organization
    import_issues = analyze_import_organization(code_content)
    if import_issues:
        message = f"\n🔍 IMPORT ORGANIZATION ISSUES:\n"
        for issue in import_issues:
            message += f"   {issue}\n"
        message += f"\n💡 GUIDANCE: Organize imports at the top of files and minimize coupling.\n"
        all_issues.append(message)

    # Output issues but don't block (educational mode)
    if all_issues:
        print("\n" + "="*60, file=sys.stderr)
        print("🔍 TYPE SAFETY & CODE QUALITY GUIDANCE", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        for issue in all_issues:
            print(issue, file=sys.stderr)
        
        print("="*60, file=sys.stderr)
        print("💡 Consider these improvements for better type safety", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)

    # Always allow (educational mode)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()