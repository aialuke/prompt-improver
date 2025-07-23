#!/usr/bin/env python3
"""
Final Legacy Cleanup Verification

Verifies that all legacy names have been successfully removed from the codebase.
"""

import os
import re
import sys
from pathlib import Path

def search_critical_patterns(root_dir):
    """Search for critical legacy patterns that must be removed."""
    
    # Critical patterns that should NOT exist in production code
    critical_patterns = {
        "RefactoredContextLearner_class": r'\bRefactoredContextLearner\b',
        "RefactoredContextConfig_class": r'\bRefactoredContextConfig\b', 
        "NewDomainFeatureExtractor_class": r'\bNewDomainFeatureExtractor\b',
        "refactored_context_learner_import": r'from.*refactored_context_learner.*import',
        "refactored_file_reference": r'refactored_context_learner\.py',
    }
    
    # Files to exclude from search (test scripts, documentation that mentions migration)
    exclude_patterns = [
        r'comprehensive_legacy_cleanup_test\.py',
        r'final_legacy_cleanup_verification\.py',
        r'complexity_reduction_migration_guide\.md',  # Migration guide can mention old names
    ]
    
    results = {}
    
    for pattern_name, pattern in critical_patterns.items():
        results[pattern_name] = []
        
        for root, dirs, files in os.walk(root_dir):
            # Skip hidden directories and common build/cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith(('.py', '.md', '.json', '.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    
                    # Skip excluded files
                    if any(re.search(exclude, file_path) for exclude in exclude_patterns):
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            line_content = content.split('\n')[line_num - 1].strip()
                            
                            results[pattern_name].append({
                                'file': file_path.replace(root_dir, ""),
                                'line': line_num,
                                'content': line_content,
                                'match': match.group()
                            })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return results


def test_imports():
    """Test that new imports work and old imports fail."""
    
    print("🧪 TESTING IMPORTS:")
    print("-" * 40)
    
    # Test new imports work
    new_imports = [
        "from prompt_improver.ml.learning.algorithms import ContextLearner",
        "from prompt_improver.ml.learning.algorithms import ContextConfig", 
        "from prompt_improver.ml.learning.features import DomainFeatureExtractor",
    ]
    
    import_success = True
    
    for test_import in new_imports:
        try:
            exec(test_import)
            print(f"✅ {test_import}")
        except Exception as e:
            print(f"❌ {test_import} - ERROR: {e}")
            import_success = False
    
    # Test legacy imports fail
    legacy_imports = [
        "from prompt_improver.ml.learning.algorithms.refactored_context_learner import RefactoredContextLearner",
        "from prompt_improver.ml.learning.features import NewDomainFeatureExtractor",
    ]
    
    for test_import in legacy_imports:
        try:
            exec(test_import)
            print(f"❌ {test_import} - SHOULD HAVE FAILED!")
            import_success = False
        except Exception:
            print(f"✅ {test_import} - Correctly failed")
    
    return import_success


def main():
    """Run the final verification."""
    
    print("🔍 FINAL LEGACY CLEANUP VERIFICATION")
    print("=" * 50)
    
    project_root = "/Users/lukemckenzie/prompt-improver"
    results = search_critical_patterns(project_root)
    
    # Check for critical issues
    critical_issues = []
    
    print("\n📊 CRITICAL PATTERN SEARCH:")
    print("-" * 40)
    
    for pattern_name, matches in results.items():
        if matches:
            print(f"❌ {pattern_name}: {len(matches)} matches found")
            for match in matches[:3]:  # Show first 3
                print(f"  📁 {match['file']}:{match['line']}")
                print(f"     {match['content']}")
            critical_issues.extend(matches)
        else:
            print(f"✅ {pattern_name}: Clean")
    
    # Test imports
    print(f"\n")
    import_success = test_imports()
    
    # Final assessment
    print(f"\n🎯 FINAL VERIFICATION RESULT:")
    print("=" * 50)
    
    if len(critical_issues) == 0 and import_success:
        print("🎉 SUCCESS! Legacy cleanup is COMPLETE!")
        print("✅ No critical legacy references found")
        print("✅ All new imports working correctly")
        print("✅ All legacy imports correctly disabled")
        print("\n🚀 The codebase is clean and ready for production!")
        return True
    else:
        print(f"❌ ISSUES FOUND:")
        print(f"  Critical legacy references: {len(critical_issues)}")
        print(f"  Import issues: {not import_success}")
        
        if critical_issues:
            print(f"\n📋 Issues to fix:")
            for issue in critical_issues[:5]:  # Show first 5
                print(f"  - {issue['file']}:{issue['line']} - {issue['match']}")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
