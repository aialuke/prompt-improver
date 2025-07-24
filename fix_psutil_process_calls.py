#!/usr/bin/env python3
"""
Fix psutil.Process() calls to use snake_case psutil.process()
"""

import re
from pathlib import Path


def fix_psutil_calls_in_file(file_path: Path) -> bool:
    """Fix psutil.Process() calls in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace psutil.Process() with psutil.process()
        content = re.sub(r'psutil\.Process\(', 'psutil.process(', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed psutil calls in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def fix_all_psutil_calls():
    """Fix psutil calls in all Python files."""
    files_to_fix = [
        "src/prompt_improver/ml/analysis/linguistic_analyzer.py",
        "src/prompt_improver/ml/optimization/algorithms/clustering_optimizer.py", 
        "src/prompt_improver/ml/optimization/batch/enhanced_batch_processor.py",
        "src/prompt_improver/ml/tests/test_ml_integration.py",
        "src/prompt_improver/ml/models/model_manager.py",
        "src/prompt_improver/ml/orchestration/monitoring/orchestrator_monitor.py",
        "src/prompt_improver/performance/optimization/memory_optimizer.py",
        "src/prompt_improver/performance/monitoring/monitoring.py",
    ]
    
    fixed_count = 0
    for file_path_str in files_to_fix:
        file_path = Path(file_path_str)
        if file_path.exists():
            if fix_psutil_calls_in_file(file_path):
                fixed_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n📊 Fixed {fixed_count} files")
    return fixed_count


if __name__ == "__main__":
    print("🔧 Fixing psutil.Process() calls to use snake_case...")
    fixed_count = fix_all_psutil_calls()
    
    if fixed_count > 0:
        print("✅ All psutil calls updated to snake_case")
    else:
        print("ℹ️  No files needed fixing")
