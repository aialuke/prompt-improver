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

import sys
import re
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Exit codes
EXIT_SUCCESS = 0
EXIT_VIOLATION = 1
EXIT_ERROR = 2

# Patterns that indicate background service context
BACKGROUND_SERVICE_INDICATORS = [
    # File path patterns
    r'/services/',
    r'/monitoring/',
    r'/background/',
    r'/lifecycle/',
    r'/orchestration/',
    r'/metrics/',
    r'/performance/',
    r'/health/',
    r'/workers/',
    
    # Class name patterns (case insensitive)
    r'class\s+\w*(?:Manager|Monitor|Service|Collector|Worker|Orchestrator|Controller)\w*',
    
    # Method name patterns for background operations
    r'def\s+(?:_loop|_monitor|_background|_periodic|_continuous|_worker|_daemon|_service)',
    r'def\s+\w*(?:_loop|_monitor|_background|_periodic|_continuous|_worker|_daemon|_service)\w*',
    
    # Infinite loop patterns
    r'while\s+True\s*:',
    r'while\s+\w+\.is_running',
    r'while\s+not\s+\w+\.shutdown',
    
    # Scheduled/periodic patterns
    r'asyncio\.sleep\(\d+\)',
    r'time\.sleep\(\d+\)',
    r'schedule\.',
    r'cron',
    r'periodic',
    
    # Service lifecycle patterns
    r'def\s+(?:start|stop|shutdown|initialize|cleanup)',
    r'def\s+(?:start_\w+|stop_\w+|shutdown_\w+)',
]

# Legitimate direct asyncio.create_task() patterns
LEGITIMATE_PATTERNS = [
    # Test files
    r'/tests?/',
    r'test_\w+\.py',
    r'conftest\.py',
    
    # Framework internals
    r'EnhancedBackgroundTaskManager',
    r'background_manager\.py',
    
    # Parallel execution within functions (not persistent background services)
    r'asyncio\.gather\(',
    r'await\s+asyncio\.gather\(',
    
    # Request/response processing (short-lived)
    r'def\s+(?:handle_|process_|execute_)\w*request',
    r'def\s+(?:api_|endpoint_|route_)',
]

def read_tool_input() -> Optional[Dict]:
    """Read tool input from stdin (Claude Code hook format)."""
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            return None
        return json.loads(input_data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error reading tool input: {e}", file=sys.stderr)
        return None

def extract_code_content(tool_data: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract code content and file path from tool data."""
    tool_name = tool_data.get('tool', {}).get('name', '')
    file_path = None
    code_content = None
    
    if tool_name in ['Write', 'Edit', 'MultiEdit']:
        # Extract file path
        if 'file_path' in tool_data.get('tool', {}).get('parameters', {}):
            file_path = tool_data['tool']['parameters']['file_path']
        
        # Extract code content
        if tool_name == 'Write':
            code_content = tool_data.get('tool', {}).get('parameters', {}).get('content', '')
        elif tool_name == 'Edit':
            code_content = tool_data.get('tool', {}).get('parameters', {}).get('new_string', '')
        elif tool_name == 'MultiEdit':
            # Concatenate all edit new_strings
            edits = tool_data.get('tool', {}).get('parameters', {}).get('edits', [])
            code_content = '\n'.join(edit.get('new_string', '') for edit in edits)
    
    return code_content, file_path

def is_legitimate_context(code_content: str, file_path: Optional[str]) -> bool:
    """Check if the context allows legitimate direct asyncio.create_task() usage."""
    if not code_content and not file_path:
        return True  # Can't determine context, allow by default
    
    content_to_check = (code_content or '') + ' ' + (file_path or '')
    
    # Check against legitimate patterns
    for pattern in LEGITIMATE_PATTERNS:
        if re.search(pattern, content_to_check, re.IGNORECASE):
            return True
    
    return False

def is_background_service_context(code_content: str, file_path: Optional[str]) -> bool:
    """Check if the context indicates background service implementation."""
    if not code_content and not file_path:
        return False
    
    content_to_check = (code_content or '') + ' ' + (file_path or '')
    
    # Check against background service indicators
    for pattern in BACKGROUND_SERVICE_INDICATORS:
        if re.search(pattern, content_to_check, re.IGNORECASE | re.MULTILINE):
            return True
    
    return False

def find_asyncio_create_task_violations(code_content: str) -> List[Tuple[int, str]]:
    """Find asyncio.create_task() calls and return line numbers and context."""
    if not code_content:
        return []
    
    violations = []
    lines = code_content.split('\n')
    
    for i, line in enumerate(lines, 1):
        if re.search(r'asyncio\.create_task\s*\(', line):
            violations.append((i, line.strip()))
    
    return violations

def check_background_task_manager_available(project_root: str) -> bool:
    """Check if EnhancedBackgroundTaskManager is available in the project."""
    try:
        # Look for the background manager import
        import subprocess
        result = subprocess.run([
            'rg', 'get_background_task_manager|EnhancedBackgroundTaskManager', 
            project_root, '--type', 'py', '--quiet'
        ], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: assume it's available (safer to enforce the pattern)
        return True

def generate_guidance_message(file_path: Optional[str], violations: List[Tuple[int, str]]) -> str:
    """Generate specific guidance message for the violation."""
    violation_count = len(violations)
    file_info = f" in {file_path}" if file_path else ""
    
    message = f"""
🚫 ASYNC GOVERNANCE VIOLATION: {violation_count} direct asyncio.create_task() call(s) detected{file_info}

📋 VIOLATION DETAILS:
"""
    
    for line_num, line_content in violations:
        message += f"   Line {line_num}: {line_content}\n"
    
    message += f"""
🏛️  ARCHITECTURAL REQUIREMENT (ADR-007):
   ALL background services MUST use EnhancedBackgroundTaskManager
   Direct asyncio.create_task() is PROHIBITED for persistent background operations

✅ REQUIRED IMPLEMENTATION:
   ```python
   from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority
   
   task_manager = get_background_task_manager()
   await task_manager.submit_enhanced_task(
       task_id=f"service_name_{{unique_identifier}}",
       coroutine=your_background_function,
       priority=TaskPriority.HIGH,  # or NORMAL/LOW
       tags={{"service": "service_name", "type": "operation_type"}}
   )
   ```

🔍 LEGITIMATE EXCEPTIONS (allowed):
   • Test files (/tests/ directory)
   • Parallel execution within functions (not persistent services)
   • EnhancedBackgroundTaskManager internals
   • Request/response processing (short-lived operations)

📖 REFERENCE: docs/architecture/ADR-007-unified-async-infrastructure.md

❌ This operation has been BLOCKED to maintain architectural consistency.
"""
    return message

def main():
    """Main validation logic."""
    # Read tool input from Claude Code
    tool_data = read_tool_input()
    if not tool_data:
        # No input data, allow operation
        sys.exit(EXIT_SUCCESS)
    
    # Extract code content and file path
    code_content, file_path = extract_code_content(tool_data)
    if not code_content:
        # No code content to validate, allow operation
        sys.exit(EXIT_SUCCESS)
    
    # Check for asyncio.create_task() violations
    violations = find_asyncio_create_task_violations(code_content)
    if not violations:
        # No violations found, allow operation
        sys.exit(EXIT_SUCCESS)
    
    # Check if this is a legitimate context
    if is_legitimate_context(code_content, file_path):
        # Legitimate usage (tests, parallel execution, etc.), allow operation
        sys.exit(EXIT_SUCCESS)
    
    # Check if this is background service context
    if is_background_service_context(code_content, file_path):
        # Background service context detected with violations, block operation
        guidance = generate_guidance_message(file_path, violations)
        print(guidance, file=sys.stderr)
        sys.exit(EXIT_VIOLATION)
    
    # Default: uncertain context, but violations detected
    # Be strict - if we detect asyncio.create_task() and can't confirm it's legitimate, block it
    guidance = generate_guidance_message(file_path, violations)
    print(guidance, file=sys.stderr)
    sys.exit(EXIT_VIOLATION)

if __name__ == '__main__':
    main()