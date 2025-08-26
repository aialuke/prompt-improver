#!/usr/bin/env python3
"""
Systematic fix for ML import issues (P7.1)

Adds missing lazy ML loader imports to all files that use get_numpy(), get_torch(), etc.
but don't have the imports at the module level.
"""

import os
import re
from pathlib import Path

def find_missing_ml_imports(file_path: str) -> set[str]:
    """Find which ML lazy loader functions are used but not imported."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all get_* function calls
    ml_functions_used = set()
    patterns = [
        r'get_numpy\(\)',
        r'get_torch\(\)', 
        r'get_scipy\(\)',
        r'get_sklearn\(\)',
        r'get_scipy_stats\(\)',
        r'get_sklearn_utils\(\)',
        r'get_sklearn_metrics\(\)',
        r'get_transformers\(\)'
    ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            func_name = pattern.split('\\(')[0]
            ml_functions_used.add(func_name)
    
    # Find existing imports
    imported_functions = set()
    import_pattern = r'from\s+prompt_improver\.core\.utils\.lazy_ml_loader\s+import\s+(.*)'
    matches = re.findall(import_pattern, content)
    for match in matches:
        # Handle both single and multi-line imports
        imports = [imp.strip() for imp in match.split(',')]
        imported_functions.update(imports)
    
    # Return missing imports
    return ml_functions_used - imported_functions

def add_missing_imports(file_path: str, missing_imports: set[str]) -> bool:
    """Add missing ML imports to a file."""
    if not missing_imports:
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the best location to add imports
    # Look for existing lazy_ml_loader import
    existing_import_line = None
    last_import_line = 0
    
    for i, line in enumerate(lines):
        if 'from prompt_improver.core.utils.lazy_ml_loader import' in line:
            existing_import_line = i
            break
        elif line.startswith('import ') or line.startswith('from '):
            last_import_line = i
    
    if existing_import_line is not None:
        # Extend existing import
        current_import = lines[existing_import_line].strip()
        # Extract current imports
        match = re.search(r'import\s+(.*)', current_import)
        if match:
            current_imports = {imp.strip() for imp in match.group(1).split(',')}
            all_imports = current_imports | missing_imports
            new_import = f"from prompt_improver.core.utils.lazy_ml_loader import {', '.join(sorted(all_imports))}\n"
            lines[existing_import_line] = new_import
    else:
        # Add new import after last import
        new_import = f"from prompt_improver.core.utils.lazy_ml_loader import {', '.join(sorted(missing_imports))}\n"
        lines.insert(last_import_line + 1, new_import)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return True

def main():
    """Fix all ML import issues systematically."""
    print("🔧 Starting systematic ML import fix...")
    
    # Get all Python files that use ML functions
    project_root = Path(__file__).parent
    
    files_to_check = []
    for pattern in ['src/**/*.py', 'tests/**/*.py']:
        files_to_check.extend(project_root.glob(pattern))
    
    total_files = 0
    fixed_files = 0
    
    for file_path in files_to_check:
        # Skip __pycache__ and other generated files
        if '__pycache__' in str(file_path):
            continue
            
        try:
            missing_imports = find_missing_ml_imports(str(file_path))
            if missing_imports:
                print(f"📁 {file_path.relative_to(project_root)}: Adding {missing_imports}")
                if add_missing_imports(str(file_path), missing_imports):
                    fixed_files += 1
            total_files += 1
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"✅ Processed {total_files} files, fixed {fixed_files} files with missing imports")

if __name__ == '__main__':
    main()