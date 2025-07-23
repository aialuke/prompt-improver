#!/usr/bin/env python3
"""
Mock Usage Scanner - Step 2 of broader plan

For every collected test file:
1. Read source text
2. Detect mock data usage via simple heuristics:
   - `from unittest.mock import` or `import unittest.mock`
   - `patch(`, `MagicMock(`, `AsyncMock(`, `Mock(`
   - Pytest fixtures whose names start with `mock_` or contain the word `mock` / `fake` / `dummy`
   - Direct creation of hard-coded sample data dictionaries used as substitutes for real data

Record: file path, line number, and the exact line where first mock indicator appears.
"""

import re
import os
import json
from pathlib import Path

def scan_file_for_mock_indicators(file_path):
    """Scan a single file for mock usage indicators."""
    mock_indicators = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Skip empty lines and comments
        if not line_stripped or line_stripped.startswith('#'):
            continue
        
        # Check for unittest.mock imports
        if re.search(r'from unittest\.mock import|import unittest\.mock', line):
            mock_indicators.append({
                "line": line_num,
                "kind": "unittest_mock_import",
                "snippet": line.strip()
            })
        
        # Check for mock function usage
        mock_patterns = [
            (r'patch\s*\(', 'patch'),
            (r'MagicMock\s*\(', 'MagicMock'),
            (r'AsyncMock\s*\(', 'AsyncMock'),
            (r'Mock\s*\(', 'Mock'),
            (r'@patch\s*\(', 'patch_decorator'),
            (r'@patch\.object\s*\(', 'patch_object_decorator'),
        ]
        
        for pattern, kind in mock_patterns:
            if re.search(pattern, line):
                mock_indicators.append({
                    "line": line_num,
                    "kind": kind,
                    "snippet": line.strip()
                })
        
        # Check for mock/fake/dummy fixtures
        fixture_patterns = [
            r'@pytest\.fixture.*def\s+(mock_\w+)',
            r'@pytest\.fixture.*def\s+(\w*mock\w*)',
            r'@pytest\.fixture.*def\s+(fake_\w+)',
            r'@pytest\.fixture.*def\s+(\w*fake\w*)',
            r'@pytest\.fixture.*def\s+(dummy_\w+)',
            r'@pytest\.fixture.*def\s+(\w*dummy\w*)',
        ]
        
        for pattern in fixture_patterns:
            match = re.search(pattern, line)
            if match:
                mock_indicators.append({
                    "line": line_num,
                    "kind": "mock_fixture",
                    "snippet": line.strip()
                })
        
        # Check for simple fixture definitions without decorator on same line
        if re.search(r'def\s+(mock_\w+|fake_\w+|dummy_\w+|\w*mock\w*|\w*fake\w*|\w*dummy\w*)\s*\(', line):
            # Check if previous line has @pytest.fixture
            if line_num > 1 and '@pytest.fixture' in lines[line_num-2]:
                mock_indicators.append({
                    "line": line_num,
                    "kind": "mock_fixture",
                    "snippet": line.strip()
                })
        
        # Check for hard-coded sample data dictionaries
        # Look for dictionary assignments that might be mock data
        if re.search(r'=\s*\{[^}]*["\'](?:test|sample|mock|fake|dummy)["\']', line):
            mock_indicators.append({
                "line": line_num,
                "kind": "sample_data_dict",
                "snippet": line.strip()
            })
        
        # Check for fakeredis usage
        if re.search(r'fakeredis|FakeRedis', line):
            mock_indicators.append({
                "line": line_num,
                "kind": "fakeredis",
                "snippet": line.strip()
            })
        
        # Check for mock class definitions
        if re.search(r'class\s+Mock\w+|class\s+\w*Mock\w*|class\s+Fake\w+|class\s+\w*Fake\w*', line):
            mock_indicators.append({
                "line": line_num,
                "kind": "mock_class",
                "snippet": line.strip()
            })
    
    return mock_indicators

def scan_all_test_files():
    """Scan all test files for mock usage indicators."""
    test_files = []
    
    # Find all test files, excluding virtual environment
    test_patterns = [
        "tests/**/test_*.py",
        "tests/**/*.py", 
        "**/test_*.py",
        "**/*_test.py",
        "**/test*.py"
    ]
    
    for pattern in test_patterns:
        found_files = Path('.').glob(pattern)
        # Filter out virtual environment files
        filtered_files = [f for f in found_files if '.venv' not in str(f) and 'venv' not in str(f) and 'env' not in str(f)]
        test_files.extend(filtered_files)
    
    # Remove duplicates and sort
    test_files = sorted(list(set(test_files)))
    
    print(f"Found {len(test_files)} test files to scan")
    
    results = {}
    
    for test_file in test_files:
        file_path = str(test_file)
        print(f"Scanning {file_path}...")
        
        indicators = scan_file_for_mock_indicators(file_path)
        
        if indicators:
            results[file_path] = indicators
    
    return results

def main():
    """Main function to run the mock scanner."""
    print("Mock Usage Scanner - Step 2")
    print("=" * 50)
    
    results = scan_all_test_files()
    
    print(f"\nScan complete. Found mock usage in {len(results)} files.")
    
    # Display results
    for file_path, indicators in results.items():
        print(f"\n{file_path}:")
        for indicator in indicators:
            print(f"  Line {indicator['line']}: {indicator['kind']} - {indicator['snippet'][:80]}...")
    
    # Save results to JSON file
    with open('mock_usage_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to mock_usage_results.json")
    
    # Print summary statistics
    total_indicators = sum(len(indicators) for indicators in results.values())
    print(f"\nSummary Statistics:")
    print(f"- Total files with mock usage: {len(results)}")
    print(f"- Total mock indicators found: {total_indicators}")
    
    # Count by type
    type_counts = {}
    for indicators in results.values():
        for indicator in indicators:
            kind = indicator['kind']
            type_counts[kind] = type_counts.get(kind, 0) + 1
    
    print(f"\nMock usage by type:")
    for kind, count in sorted(type_counts.items()):
        print(f"  {kind}: {count}")

if __name__ == "__main__":
    main()
