#!/usr/bin/env python3
"""
Automated Refactoring Script for Naming Convention Fixes
Applies safe, validated naming convention fixes to the codebase.
"""

import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any


class SafeRefactorer:
    """Safely applies naming convention fixes with validation."""

    def __init__(self, fixes_file: str = "implementation_plan.json"):
        with open(fixes_file, 'r') as f:
            self.plan = json.load(f)
        self.fixes = self.plan['automated_fixes']
        self.backup_dir = Path("backup_before_refactoring")

    def create_backup(self) -> None:
        """Create backup of all files before refactoring."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)

        shutil.copytree("src", self.backup_dir / "src")
        print(f"Backup created in {self.backup_dir}")

    def apply_fix(self, fix: Dict[str, Any]) -> bool:
        """Apply a single fix with validation."""
        file_path = Path("src/prompt_improver") / fix['file_path']

        if not file_path.exists():
            print(f"Warning: File {file_path} not found")
            return False

        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Apply fix based on type
            if fix['fix_type'] == 'simple_rename':
                new_content = self._apply_simple_rename(content, fix)
            elif fix['fix_type'] == 'type_alias':
                new_content = self._apply_type_alias_fix(content, fix)
            else:
                print(f"Unknown fix type: {fix['fix_type']}")
                return False

            # Validate syntax
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                print(f"Syntax error after fix {fix['fix_id']}: {e}")
                return False

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"Applied fix {fix['fix_id']}: {fix['old_name']} -> {fix['new_name']}")
            return True

        except Exception as e:
            print(f"Error applying fix {fix['fix_id']}: {e}")
            return False

    def _apply_simple_rename(self, content: str, fix: Dict[str, Any]) -> str:
        """Apply simple variable/function rename."""
        old_name = fix['old_name']
        new_name = fix['new_name']

        # Use word boundaries to avoid partial matches
        pattern = rf'\b{re.escape(old_name)}\b'
        return re.sub(pattern, new_name, content)

    def _apply_type_alias_fix(self, content: str, fix: Dict[str, Any]) -> str:
        """Apply type alias fix with backward compatibility."""
        old_name = fix['old_name']
        new_name = fix['new_name']

        # Replace the type alias definition
        pattern = rf'^({re.escape(old_name)}\s*=.*)$'
        replacement = f'{new_name} = \1\n{old_name} = {new_name}  # Backward compatibility'

        return re.sub(pattern, replacement, content, flags=re.MULTILINE)

    def validate_changes(self) -> bool:
        """Validate all changes after applying fixes."""
        print("Validating changes...")

        # Run syntax validation
        result = subprocess.run([
            'python', '-c',
            'import sys; sys.path.insert(0, "src"); import prompt_improver'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Import validation failed: {result.stderr}")
            return False

        print("All validations passed!")
        return True

    def rollback(self) -> None:
        """Rollback changes using backup."""
        if not self.backup_dir.exists():
            print("No backup found for rollback")
            return

        shutil.rmtree("src")
        shutil.copytree(self.backup_dir / "src", "src")
        print("Rollback completed")

    def run_phase_2_fixes(self) -> None:
        """Run Phase 2: Low-risk variable renames."""
        print("=== Phase 2: Low-Risk Variable Renames ===")

        self.create_backup()

        # Apply only low-risk fixes
        low_risk_fixes = [f for f in self.fixes if f['fix_type'] == 'simple_rename']

        success_count = 0
        for fix in low_risk_fixes:
            if self.apply_fix(fix):
                success_count += 1

        print(f"Applied {success_count}/{len(low_risk_fixes)} fixes")

        if self.validate_changes():
            print("Phase 2 completed successfully!")
        else:
            print("Validation failed, rolling back...")
            self.rollback()


if __name__ == "__main__":
    refactorer = SafeRefactorer()
    refactorer.run_phase_2_fixes()
