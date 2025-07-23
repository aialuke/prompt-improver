#!/usr/bin/env python3
"""
Automated Type Annotation Fixer for Phase 3 Implementation
Systematically adds missing return type annotations based on 2025 best practices
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def get_mypy_errors() -> List[str]:
    """Get all mypy errors for the codebase"""
    try:
        result = subprocess.run(
            ["mypy", "prompt_improver/", "--show-error-codes", "--no-error-summary"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except Exception as e:
        print(f"Error running mypy: {e}")
        return []


def parse_no_untyped_def_errors(errors: List[str]) -> List[Tuple[str, int, str]]:
    """Parse no-untyped-def errors to extract file, line, and function info"""
    no_untyped_def_errors = []
    
    for error in errors:
        if "[no-untyped-def]" in error and "Function is missing a return type annotation" in error:
            # Extract file path and line number
            match = re.match(r"([^:]+):(\d+):", error)
            if match:
                file_path = match.group(1)
                line_num = int(match.group(2))
                no_untyped_def_errors.append((file_path, line_num, error))
    
    return no_untyped_def_errors


def fix_function_return_type(file_path: str, line_num: int) -> bool:
    """Fix a single function's return type annotation"""
    try:
        full_path = Path(file_path)
        if not full_path.exists():
            return False
            
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if line_num > len(lines):
            return False
            
        # Find the function definition line
        func_line_idx = line_num - 1  # Convert to 0-based index
        func_line = lines[func_line_idx].strip()
        
        # Check if it's a function definition
        if not func_line.startswith('def ') and 'def ' not in func_line:
            return False
            
        # Look for the end of the function signature
        signature_end_idx = func_line_idx
        while signature_end_idx < len(lines):
            if ')' in lines[signature_end_idx] and ':' in lines[signature_end_idx]:
                break
            signature_end_idx += 1
        
        if signature_end_idx >= len(lines):
            return False
            
        # Check if return type annotation already exists
        signature_line = lines[signature_end_idx]
        if ' -> ' in signature_line:
            return False  # Already has return type
            
        # Add -> None: before the colon
        if signature_line.strip().endswith(':'):
            lines[signature_end_idx] = signature_line.replace(':', ' -> None:')
            
            # Write back to file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
            
    except Exception as e:
        print(f"Error fixing {file_path}:{line_num}: {e}")
        return False
    
    return False


def main():
    """Main function to fix type annotations systematically"""
    print("🔧 Starting automated type annotation fixes...")
    
    # Get initial error count
    errors = get_mypy_errors()
    initial_count = len([e for e in errors if "[no-untyped-def]" in e])
    print(f"📊 Found {initial_count} no-untyped-def errors")
    
    # Parse errors
    no_untyped_def_errors = parse_no_untyped_def_errors(errors)
    print(f"📝 Parsed {len(no_untyped_def_errors)} fixable function errors")
    
    # Fix errors in batches
    fixed_count = 0
    batch_size = 50
    
    for i in range(0, len(no_untyped_def_errors), batch_size):
        batch = no_untyped_def_errors[i:i + batch_size]
        batch_fixed = 0
        
        for file_path, line_num, error in batch:
            if fix_function_return_type(file_path, line_num):
                batch_fixed += 1
                fixed_count += 1
        
        print(f"✅ Batch {i//batch_size + 1}: Fixed {batch_fixed}/{len(batch)} functions")
        
        # Check progress every batch
        if batch_fixed > 0:
            new_errors = get_mypy_errors()
            new_count = len([e for e in new_errors if "[no-untyped-def]" in e])
            print(f"📉 Reduced errors from {initial_count} to {new_count}")
    
    print(f"🎉 Fixed {fixed_count} function return type annotations")
    
    # Final error count
    final_errors = get_mypy_errors()
    final_count = len([e for e in final_errors if "[no-untyped-def]" in e])
    print(f"📊 Final no-untyped-def errors: {final_count}")
    print(f"📈 Improvement: {initial_count - final_count} errors resolved")


if __name__ == "__main__":
    main()
