#!/usr/bin/env python3
"""
Legacy Cleanup and Naming Convention Refactoring
Removes backward compatibility aliases and implements clean naming conventions.
"""

import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set


class LegacyCleanupRefactorer:
    """Removes legacy code and implements clean naming conventions."""
    
    def __init__(self):
        self.backup_dir = Path("backup_before_legacy_cleanup")
        self.legacy_patterns = self._define_legacy_patterns()
        self.naming_fixes = self._load_naming_fixes()
        self.removed_items = []
        
    def _define_legacy_patterns(self) -> Dict[str, List[str]]:
        """Define patterns for legacy code that should be removed."""
        return {
            'backward_compatibility_aliases': [
                r'# Backward compatibility.*\n.*=.*',
                r'# Deprecated:.*\n.*=.*',
                r'.*=.*# Backward compatibility',
                r'.*=.*# Deprecated',
            ],
            'legacy_imports': [
                r'from.*refactored_context_learner.*import',
                r'import.*refactored_context_learner',
            ],
            'legacy_class_names': [
                r'RefactoredContextLearner',
                r'RefactoredContextConfig',
                r'NewDomainFeatureExtractor',
            ],
            'legacy_comments': [
                r'# TODO.*legacy.*',
                r'# FIXME.*legacy.*',
                r'# XXX.*legacy.*',
                r'# Legacy.*',
                r'# Transitional.*',
            ],
            'deprecated_markers': [
                r'@pytest\.mark\.deprecated',
                r'pytest\.mark\.deprecated',
                r'# Deprecated.*',
            ]
        }
    
    def _load_naming_fixes(self) -> List[Dict[str, Any]]:
        """Load naming convention fixes from the analysis."""
        try:
            with open("implementation_plan.json", "r") as f:
                plan = json.load(f)
            return plan.get('automated_fixes', [])
        except FileNotFoundError:
            print("Warning: implementation_plan.json not found, generating fixes...")
            return []
    
    def create_backup(self) -> None:
        """Create backup before making changes."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        shutil.copytree("src", self.backup_dir / "src")
        print(f"✅ Backup created in {self.backup_dir}")
    
    def remove_legacy_code(self, file_path: Path) -> bool:
        """Remove legacy code patterns from a file."""
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            lines_removed = 0
            
            # Remove backward compatibility aliases
            for pattern in self.legacy_patterns['backward_compatibility_aliases']:
                matches = list(re.finditer(pattern, content, re.MULTILINE))
                for match in reversed(matches):  # Remove from end to preserve line numbers
                    removed_text = match.group(0)
                    content = content[:match.start()] + content[match.end():]
                    lines_removed += removed_text.count('\n') + 1
                    self.removed_items.append({
                        'file': str(file_path),
                        'type': 'backward_compatibility_alias',
                        'removed': removed_text.strip()
                    })
            
            # Remove legacy imports
            for pattern in self.legacy_patterns['legacy_imports']:
                content = re.sub(pattern + r'.*\n?', '', content, flags=re.MULTILINE)
            
            # Remove legacy comments
            for pattern in self.legacy_patterns['legacy_comments']:
                content = re.sub(pattern + r'.*\n?', '', content, flags=re.MULTILINE)
            
            # Remove deprecated test markers
            for pattern in self.legacy_patterns['deprecated_markers']:
                content = re.sub(pattern + r'.*\n?', '', content, flags=re.MULTILINE)
            
            # Clean up empty lines (more than 2 consecutive)
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"🧹 Cleaned legacy code from {file_path} ({lines_removed} lines removed)")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error cleaning {file_path}: {e}")
            return False
    
    def apply_naming_fix_clean(self, file_path: Path, fix: Dict[str, Any]) -> bool:
        """Apply naming fix without backward compatibility."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            old_name = fix['old_name']
            new_name = fix['new_name']
            
            # Apply clean rename (no backward compatibility)
            if fix['fix_type'] == 'type_alias':
                # For type aliases, do a clean replacement
                pattern = rf'^{re.escape(old_name)}\s*=(.*)$'
                replacement = f'{new_name} =\\1'
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            else:
                # For other fixes, use word boundary replacement
                pattern = rf'\b{re.escape(old_name)}\b'
                content = re.sub(pattern, new_name, content)
            
            # Validate syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                print(f"❌ Syntax error after fix {fix['fix_id']}: {e}")
                return False
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Applied clean fix: {old_name} → {new_name} in {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error applying fix to {file_path}: {e}")
            return False
    
    def remove_legacy_files(self) -> None:
        """Remove entire legacy files that are no longer needed."""
        legacy_files = [
            "src/prompt_improver/ml/learning/algorithms/refactored_context_learner.py",
            "tests/deprecated/",
            "archive/final_legacy_cleanup_verification.py",
        ]
        
        for file_pattern in legacy_files:
            file_path = Path(file_pattern)
            if file_path.exists():
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"🗑️  Removed legacy directory: {file_path}")
                else:
                    file_path.unlink()
                    print(f"🗑️  Removed legacy file: {file_path}")
                
                self.removed_items.append({
                    'file': str(file_path),
                    'type': 'legacy_file_removal',
                    'removed': f'Entire file/directory: {file_path}'
                })
    
    def update_imports_after_cleanup(self) -> None:
        """Update import statements after legacy cleanup."""
        src_dir = Path("src/prompt_improver")
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Update imports to use new naming conventions
                import_updates = {
                    'FloatArray': 'float_array',
                    'IntArray': 'int_array', 
                    'BoolArray': 'bool_array',
                    'FeatureArray': 'feature_array',
                    'LabelArray': 'label_array',
                    'MetricsDict': 'metrics_dict',
                    'ReducedFeatures': 'reduced_features',
                }
                
                for old_import, new_import in import_updates.items():
                    # Update from imports
                    pattern = rf'from\s+([^\s]+)\s+import\s+([^,\n]*\b{re.escape(old_import)}\b[^,\n]*)'
                    def replace_import(match):
                        module = match.group(1)
                        imports = match.group(2)
                        updated_imports = imports.replace(old_import, new_import)
                        return f'from {module} import {updated_imports}'
                    
                    content = re.sub(pattern, replace_import, content)
                
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"📦 Updated imports in {py_file}")
                    
            except Exception as e:
                print(f"❌ Error updating imports in {py_file}: {e}")
    
    def validate_cleanup(self) -> bool:
        """Validate that cleanup was successful."""
        print("\n🔍 Validating cleanup...")
        
        # Check that imports still work
        try:
            result = subprocess.run([
                'python', '-c', 
                'import sys; sys.path.insert(0, "src"); import prompt_improver'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"❌ Import validation failed: {result.stderr}")
                return False
            
            print("✅ Import validation passed")
            
        except subprocess.TimeoutExpired:
            print("❌ Import validation timed out")
            return False
        
        # Check for remaining legacy patterns
        legacy_found = False
        src_dir = Path("src/prompt_improver")
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for backward compatibility comments
                if re.search(r'# Backward compatibility|# Deprecated', content):
                    print(f"⚠️  Legacy compatibility code still found in {py_file}")
                    legacy_found = True
                
            except Exception:
                continue
        
        if not legacy_found:
            print("✅ No legacy compatibility code found")
        
        return not legacy_found
    
    def run_complete_cleanup(self) -> bool:
        """Run complete legacy cleanup and naming convention fixes."""
        print("🧹 Starting Complete Legacy Cleanup and Naming Convention Fix")
        print("=" * 70)
        
        # Create backup
        self.create_backup()
        
        # Remove legacy files first
        print("\n📁 Removing legacy files...")
        self.remove_legacy_files()
        
        # Process all Python files
        print("\n🔧 Processing Python files...")
        src_dir = Path("src/prompt_improver")
        processed_files = 0
        
        for py_file in src_dir.rglob("*.py"):
            # Remove legacy code patterns
            if self.remove_legacy_code(py_file):
                processed_files += 1
            
            # Apply naming fixes (clean, no backward compatibility)
            for fix in self.naming_fixes:
                if fix['file_path'] in str(py_file):
                    self.apply_naming_fix_clean(py_file, fix)
        
        print(f"\n📊 Processed {processed_files} files")
        
        # Update imports
        print("\n📦 Updating imports...")
        self.update_imports_after_cleanup()
        
        # Validate cleanup
        print("\n✅ Validating cleanup...")
        success = self.validate_cleanup()
        
        # Generate cleanup report
        self.generate_cleanup_report()
        
        if success:
            print("\n🎉 Legacy cleanup completed successfully!")
            print(f"📋 Removed {len(self.removed_items)} legacy items")
            print("📄 See cleanup_report.json for details")
        else:
            print("\n❌ Cleanup validation failed - check errors above")
        
        return success
    
    def generate_cleanup_report(self) -> None:
        """Generate a report of what was cleaned up."""
        report = {
            'cleanup_summary': {
                'total_items_removed': len(self.removed_items),
                'files_processed': len(set(item['file'] for item in self.removed_items)),
                'types_of_cleanup': list(set(item['type'] for item in self.removed_items))
            },
            'removed_items': self.removed_items,
            'validation_status': 'completed'
        }
        
        with open("cleanup_report.json", "w") as f:
            json.dump(report, f, indent=2)
    
    def rollback(self) -> None:
        """Rollback changes if needed."""
        if not self.backup_dir.exists():
            print("❌ No backup found for rollback")
            return
        
        if Path("src").exists():
            shutil.rmtree("src")
        
        shutil.copytree(self.backup_dir / "src", "src")
        print("🔄 Rollback completed - restored from backup")


if __name__ == "__main__":
    refactorer = LegacyCleanupRefactorer()
    success = refactorer.run_complete_cleanup()
    
    if not success:
        print("\n🤔 Would you like to rollback? (y/n)")
        # In automated mode, we'll continue without rollback
        print("Continuing without rollback for automated execution...")
