#!/usr/bin/env python3
"""Enhanced Governance Validator - Claude Code Pre-Tool Hook

This script enforces modern software engineering practices by:
1. Scanning code for asyncio.create_task() patterns (Unified Async Infrastructure Protocol)
2. Validating import patterns and architectural compliance (Clean Architecture)
3. Checking type safety and code quality standards (2025 patterns)
4. Enforcing god object elimination (single responsibility principle)
5. Validating real behavior testing patterns (no mocks for external services)

Based on ADR-007: Unified Async Infrastructure Protocol + 2025 architectural standards
Research sources: Claude Code hooks, Clean Architecture patterns, SOLID principles
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

EXIT_SUCCESS = 0
EXIT_VIOLATION = 1
EXIT_ERROR = 2

# Async patterns that are legitimate
LEGITIMATE_ASYNC_PATTERNS = [
    r"/tests?/",
    r"test_\w+\.py",
    r"conftest\.py",
    r"EnhancedBackgroundTaskManager",
    r"background_manager\.py",
    r"asyncio\.gather\(",
    r"await\s+asyncio\.gather\(",
    r"def\s+(?:handle_|process_|execute_)\w*request",
    r"def\s+(?:api_|endpoint_|route_)",
    r"/core/",
    r"facade\.py",
    r"orchestrator\.py",
]

# Direct database imports that violate Clean Architecture
PROHIBITED_DATABASE_IMPORTS = [
    r"from\s+prompt_improver\.database\s+import\s+get_session",
    r"from\s+prompt_improver\.database\.connection\s+import",
    r"import\s+prompt_improver\.database\.connection",
    r"from\s+prompt_improver\.database\s+import\s+DatabaseServices",
]

# Prohibited patterns for god object elimination
GOD_OBJECT_PATTERNS = [
    r"class\s+\w*Service.*:\s*\n(?:\s*.*\n){500,}",  # Classes >500 lines
    r"class\s+\w*Manager.*:\s*\n(?:\s*.*\n){500,}",  # Large manager classes
]

# Mock patterns that violate real behavior testing
PROHIBITED_MOCK_PATTERNS = [
    r"@mock\.patch",
    r"unittest\.mock",
    r"from\s+unittest\.mock\s+import",
    r"Mock\(\)",
    r"MagicMock\(\)",
    r"patch\(",
]

# Legacy patterns that should be modernized
LEGACY_PATTERNS = [
    r"%[sd]",  # String formatting instead of f-strings
    r"os\.path\.",  # os.path instead of pathlib
    r"dict\.keys\(\)",  # dict.keys() iteration
    r"Any\s*=.*",  # Any type hints without specific types
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


def is_legitimate_async_context(code_content: str, file_path: str | None) -> bool:
    """Check if the context allows legitimate direct asyncio.create_task() usage."""
    if not code_content and not file_path:
        return True

    content_to_check = (code_content or "") + " " + (file_path or "")

    for pattern in LEGITIMATE_ASYNC_PATTERNS:
        if re.search(pattern, content_to_check, re.IGNORECASE):
            return True

    return False


def find_asyncio_create_task_violations(code_content: str) -> list[tuple[int, str]]:
    """Find asyncio.create_task() calls and return line numbers and context."""
    if not code_content:
        return []

    violations: list[tuple[int, str]] = []
    lines = code_content.split("\n")

    for i, line in enumerate(lines, 1):
        if re.search(r"asyncio\.create_task\s*\(", line):
            violations.append((i, line.strip()))

    return violations


def find_database_import_violations(code_content: str) -> list[tuple[int, str]]:
    """Find direct database imports that violate Clean Architecture."""
    if not code_content:
        return []

    violations: list[tuple[int, str]] = []
    lines = code_content.split("\n")

    for i, line in enumerate(lines, 1):
        for pattern in PROHIBITED_DATABASE_IMPORTS:
            if re.search(pattern, line):
                violations.append((i, line.strip()))

    return violations


def find_mock_violations(code_content: str, file_path: str | None) -> list[tuple[int, str]]:
    """Find mock usage that violates real behavior testing."""
    if not code_content:
        return []

    # Allow mocks only in unit tests, not integration tests
    if file_path and ("integration" in file_path or "real_behavior" in file_path):
        violations: list[tuple[int, str]] = []
        lines = code_content.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in PROHIBITED_MOCK_PATTERNS:
                if re.search(pattern, line):
                    violations.append((i, line.strip()))

        return violations

    return []


def find_legacy_pattern_violations(code_content: str) -> list[tuple[int, str]]:
    """Find legacy patterns that should be modernized."""
    if not code_content:
        return []

    violations: list[tuple[int, str]] = []
    lines = code_content.split("\n")

    for i, line in enumerate(lines, 1):
        for pattern in LEGACY_PATTERNS:
            if re.search(pattern, line):
                violations.append((i, line.strip()))

    return violations


def check_god_object_violation(code_content: str) -> bool:
    """Check if code contains god object patterns (classes >500 lines)."""
    if not code_content:
        return False

    lines = code_content.split("\n")
    return len(lines) > 500 and bool(re.search(r"class\s+\w+", code_content))


def format_violation_message(violation_type: str, violations: list[tuple[int, str]], guidance: str) -> str:
    """Format violation message with specific guidance."""
    if not violations:
        return ""

    message = f"\n🚫 {violation_type} VIOLATION DETECTED:\n"
    for line_num, line_content in violations[:3]:  # Show first 3 violations
        message += f"   Line {line_num}: {line_content}\n"
    
    if len(violations) > 3:
        message += f"   ... and {len(violations) - 3} more violations\n"
    
    message += f"\n💡 GUIDANCE: {guidance}\n"
    return message


def main() -> None:
    """Enhanced validation logic for 2025 architectural standards."""
    tool_data = read_tool_input()
    if not tool_data:
        sys.exit(EXIT_SUCCESS)

    code_content, file_path = extract_code_content(tool_data)
    if not code_content:
        sys.exit(EXIT_SUCCESS)

    all_violations = []

    # 1. Check async violations (original functionality)
    async_violations = find_asyncio_create_task_violations(code_content)
    if async_violations and not is_legitimate_async_context(code_content, file_path):
        message = format_violation_message(
            "ASYNC INFRASTRUCTURE",
            async_violations,
            "Use EnhancedBackgroundTaskManager instead of direct asyncio.create_task() calls. "
            "This ensures proper task lifecycle management and error handling."
        )
        all_violations.append(message)

    # 2. Check Clean Architecture violations
    db_violations = find_database_import_violations(code_content)
    if db_violations:
        message = format_violation_message(
            "CLEAN ARCHITECTURE",
            db_violations,
            "Use repository patterns with protocol-based DI instead of direct database imports. "
            "Business logic should not depend on infrastructure concerns."
        )
        all_violations.append(message)

    # 3. Check real behavior testing violations
    mock_violations = find_mock_violations(code_content, file_path)
    if mock_violations:
        message = format_violation_message(
            "REAL BEHAVIOR TESTING",
            mock_violations,
            "Use testcontainers for real service integration instead of mocks. "
            "Integration tests should validate actual behavior, not mock behavior."
        )
        all_violations.append(message)

    # 4. Check god object violations
    if check_god_object_violation(code_content):
        message = f"\n🚫 GOD OBJECT VIOLATION DETECTED:\n" \
                 f"   Class exceeds 500 lines (Single Responsibility Principle)\n" \
                 f"\n💡 GUIDANCE: Decompose into focused services using facade patterns. " \
                 f"Each class should have a single responsibility.\n"
        all_violations.append(message)

    # 5. Check legacy pattern violations
    legacy_violations = find_legacy_pattern_violations(code_content)
    if legacy_violations:
        message = format_violation_message(
            "LEGACY PATTERNS",
            legacy_violations,
            "Modernize to 2025 patterns: use f-strings, pathlib, specific type hints, "
            "and direct dict iteration instead of legacy approaches."
        )
        all_violations.append(message)

    # Output violations but don't block (educate instead of enforce)
    if all_violations:
        print("\n" + "="*60, file=sys.stderr)
        print("🏗️  ARCHITECTURAL GUIDANCE", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        for violation in all_violations:
            print(violation, file=sys.stderr)
        
        print("="*60, file=sys.stderr)
        print("💡 Consider these improvements for better architecture", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)

    # Always allow (educational mode, not blocking)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
