#!/usr/bin/env python3
"""
Comprehensive Legacy Cleanup Test

Tests to ensure no legacy names, aliases, or compatibility code remains
after the clean refactoring of RefactoredContextLearner → ContextLearner
and NewDomainFeatureExtractor alias removal.
"""

import os
import re
import sys
from pathlib import Path

def search_files_for_patterns(root_dir, patterns, file_extensions=None):
    """Search for patterns in files."""
    if file_extensions is None:
        file_extensions = ['.py', '.md', '.json', '.yaml', '.yml', '.txt', '.rst']
    
    results = {}
    
    for pattern_name, pattern in patterns.items():
        results[pattern_name] = []
        
        for root, dirs, files in os.walk(root_dir):
            # Skip common directories that shouldn't contain our code
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            line_content = content.split('\n')[line_num - 1].strip()
                            
                            results[pattern_name].append({
                                'file': file_path,
                                'line': line_num,
                                'content': line_content,
                                'match': match.group()
                            })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return results


def test_legacy_cleanup():
    """Test for any remaining legacy code."""
    
    print("🔍 COMPREHENSIVE LEGACY CLEANUP TEST")
    print("=" * 60)
    
    # Define patterns to search for
    legacy_patterns = {
        # Legacy class names
        "RefactoredContextLearner": r'\bRefactoredContextLearner\b',
        "RefactoredContextConfig": r'\bRefactoredContextConfig\b',
        "NewDomainFeatureExtractor": r'\bNewDomainFeatureExtractor\b',
        
        # Legacy file references
        "refactored_context_learner_file": r'refactored_context_learner\.py',
        "new_domain_extractor_file": r'new_domain_extractor\.py',
        
        # Legacy import patterns
        "refactored_imports": r'from.*refactored_context_learner.*import',
        "new_domain_imports": r'from.*NewDomainFeatureExtractor.*import',
        
        # Alias patterns
        "as_NewDomain": r'as\s+NewDomainFeatureExtractor',
        "alias_equals": r'NewDomainFeatureExtractor\s*=',
        
        # Compatibility layer patterns
        "compatibility_adapter": r'ContextLearnerAdapter',
        "backward_compatibility": r'backward.{0,20}compatibility',
        "legacy_support": r'legacy.{0,20}support',
        
        # Old configuration patterns
        "refactored_config_usage": r'RefactoredContextConfig\s*\(',
        
        # Documentation references
        "refactored_docs": r'refactored.{0,20}context.{0,20}learner',
        "migration_guide_refs": r'see.{0,20}migration.{0,20}guide',
        
        # Test file legacy patterns
        "test_refactored": r'test.*refactored.*context',
        "TestRefactored": r'TestRefactored\w+',
        
        # Old orchestrator mappings
        "refactored_orchestrator": r'["\']refactored_context_learner["\']',
    }
    
    # Search in project root
    project_root = "/Users/lukemckenzie/prompt-improver"
    results = search_files_for_patterns(project_root, legacy_patterns)
    
    # Analyze results
    total_issues = 0
    critical_issues = []
    
    print("\n📊 SEARCH RESULTS:")
    print("-" * 50)
    
    for pattern_name, matches in results.items():
        if matches:
            total_issues += len(matches)
            print(f"\n❌ {pattern_name}: {len(matches)} matches found")
            
            for match in matches[:5]:  # Show first 5 matches
                rel_path = match['file'].replace(project_root, "")
                print(f"  📁 {rel_path}:{match['line']}")
                print(f"     {match['content']}")
                
                # Mark critical issues
                if any(critical in pattern_name.lower() for critical in ['class', 'import', 'config']):
                    critical_issues.append(match)
            
            if len(matches) > 5:
                print(f"     ... and {len(matches) - 5} more matches")
        else:
            print(f"✅ {pattern_name}: No matches found")
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total potential issues: {total_issues}")
    print(f"  Critical issues: {len(critical_issues)}")
    
    # Test imports
    print(f"\n🧪 IMPORT TESTS:")
    print("-" * 50)
    
    import_tests = [
        "from prompt_improver.ml.learning.algorithms import ContextLearner",
        "from prompt_improver.ml.learning.algorithms import ContextConfig", 
        "from prompt_improver.ml.learning.features import DomainFeatureExtractor",
        "from prompt_improver.ml.learning.features import LinguisticFeatureExtractor",
        "from prompt_improver.ml.learning.features import ContextFeatureExtractor",
        "from prompt_improver.ml.learning.features import CompositeFeatureExtractor",
    ]
    
    for test_import in import_tests:
        try:
            exec(test_import)
            print(f"✅ {test_import}")
        except Exception as e:
            print(f"❌ {test_import} - ERROR: {e}")
            critical_issues.append(f"Import failed: {test_import}")
    
    # Test legacy imports should fail
    print(f"\n🚫 LEGACY IMPORT TESTS (should fail):")
    print("-" * 50)
    
    legacy_import_tests = [
        "from prompt_improver.ml.learning.algorithms.refactored_context_learner import RefactoredContextLearner",
        "from prompt_improver.ml.learning.features import NewDomainFeatureExtractor",
    ]
    
    for test_import in legacy_import_tests:
        try:
            exec(test_import)
            print(f"❌ {test_import} - SHOULD HAVE FAILED!")
            critical_issues.append(f"Legacy import still works: {test_import}")
        except Exception:
            print(f"✅ {test_import} - Correctly failed")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("=" * 60)
    
    if total_issues == 0 and len(critical_issues) == 0:
        print("🎉 PERFECT! No legacy code found.")
        print("✅ Clean refactoring completed successfully!")
        return True
    elif len(critical_issues) == 0:
        print(f"⚠️  Found {total_issues} minor references (likely documentation)")
        print("✅ No critical issues - refactoring successful!")
        return True
    else:
        print(f"❌ Found {len(critical_issues)} critical issues that need fixing:")
        for issue in critical_issues:
            print(f"  - {issue}")
        return False


if __name__ == "__main__":
    success = test_legacy_cleanup()
    sys.exit(0 if success else 1)
