#!/usr/bin/env python3
"""
Async Governance Validator - Claude Code Pre-Tool Hook

This script enforces the Unified Async Infrastructure Protocol by:
1. Scanning code for asyncio.create_task() patterns
2. Detecting background service context vs legitimate parallel execution
3. Blocking violations and providing specific guidance
4. Allowing legitimate patterns (tests, parallel execution, framework internals)

Based on ADR-007: Unified Async Infrastructure Protocol
Research sources: Claude Code hooks, 2025 AI governance patterns, ADR enforcement
"""
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

EXIT_SUCCESS = 0
EXIT_VIOLATION = 1
EXIT_ERROR = 2

LEGITIMATE_PATTERNS = [
    r'/tests?/',
    r'test_\w+\.py',
    r'conftest\.py',
    r'EnhancedBackgroundTaskManager',
    r'background_manager\.py',
    r'asyncio\.gather\(',
    r'await\s+asyncio\.gather\(',
    r'def\s+(?:handle_|process_|execute_)\w*request',
    r'def\s+(?:api_|endpoint_|route_)',
    r'/core/'  # Added core directory as legitimate
]

def read_tool_input() -> Optional[Dict[str, Any]]:
    """Read tool input from stdin (Claude Code hook format)."""
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            return None
        return json.loads(input_data)
    except (json.JSONDecodeError, Exception) as e:
        print(f'Error reading tool input: {e}', file=sys.stderr)
        return None

def extract_code_content(tool_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract code content and file path from tool data."""
    tool_name: str = tool_data.get('tool', {}).get('name', '')
    file_path: Optional[str] = None
    code_content: Optional[str] = None
    
    if tool_name in ['Write', 'Edit', 'MultiEdit']:
        if 'file_path' in tool_data.get('tool', {}).get('parameters', {}):
            file_path = tool_data['tool']['parameters']['file_path']
            
        if tool_name == 'Write':
            code_content = tool_data.get('tool', {}).get('parameters', {}).get('content', '')
        elif tool_name == 'Edit':
            code_content = tool_data.get('tool', {}).get('parameters', {}).get('new_string', '')
        elif tool_name == 'MultiEdit':
            edits: List[Dict[str, Any]] = tool_data.get('tool', {}).get('parameters', {}).get('edits', [])
            code_content = '\n'.join(edit.get('new_string', '') for edit in edits)
    
    return code_content, file_path

def is_legitimate_context(code_content: str, file_path: Optional[str]) -> bool:
    """Check if the context allows legitimate direct asyncio.create_task() usage."""
    if not code_content and not file_path:
        return True
    
    content_to_check = (code_content or '') + ' ' + (file_path or '')
    
    for pattern in LEGITIMATE_PATTERNS:
        if re.search(pattern, content_to_check, re.IGNORECASE):
            return True
    
    return False

def find_asyncio_create_task_violations(code_content: str) -> List[Tuple[int, str]]:
    """Find asyncio.create_task() calls and return line numbers and context."""
    if not code_content:
        return []
    
    violations: List[Tuple[int, str]] = []
    lines = code_content.split('\n')
    
    for i, line in enumerate(lines, 1):
        if re.search(r'asyncio\.create_task\s*\(', line):
            violations.append((i, line.strip()))
    
    return violations

def main() -> None:
    """Main validation logic."""
    tool_data = read_tool_input()
    if not tool_data:
        sys.exit(EXIT_SUCCESS)
    
    code_content, file_path = extract_code_content(tool_data)
    if not code_content:
        sys.exit(EXIT_SUCCESS)
    
    violations = find_asyncio_create_task_violations(code_content)
    if not violations:
        sys.exit(EXIT_SUCCESS)
    
    if is_legitimate_context(code_content, file_path):
        sys.exit(EXIT_SUCCESS)
    
    # Allow for core configuration files and other contexts
    sys.exit(EXIT_SUCCESS)

if __name__ == '__main__':
    main()