#!/usr/bin/env python3
"""
Fix HealthResult constructor calls in ML orchestration health checkers.
"""

import re

def fix_health_results(file_path):
    """Fix all HealthResult calls in the file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match HealthResult calls with old parameters
    patterns = [
        # Fix name= to component=
        (r'(\s+)return HealthResult\(\s*\n(\s+)name=([^\n]+),\s*\n(\s+)status=([^\n]+),\s*\n(\s+)message=([^\n]+),\s*\n(\s+)details=([^\n]+),\s*\n(\s+)timestamp=([^\n]+),\s*\n(\s+)response_time=([^\n]+)\s*\n(\s+)\)',
         r'\1return HealthResult(\n\2status=\5,\n\2component=\3,\n\2message=\7,\n\2details=\9,\n\2timestamp=\11,\n\2response_time_ms=\13 * 1000\n\15)'),
        
        # Fix UNHEALTHY to FAILED
        (r'HealthStatus\.UNHEALTHY', 'HealthStatus.FAILED'),
        
        # Fix response_time to response_time_ms
        (r'response_time=([^*\n]+)(?!\s*\*)', r'response_time_ms=\1 * 1000'),
        (r'response_time=([^*\n]+)\s*\*\s*1000', r'response_time_ms=\1 * 1000'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed HealthResult calls in {file_path}")

if __name__ == "__main__":
    fix_health_results("/Users/lukemckenzie/prompt-improver/src/prompt_improver/performance/monitoring/health/ml_orchestration_checkers.py")