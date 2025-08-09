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
import json
import os
from pathlib import Path
import re

def scan_file_for_mock_indicators(file_path):
    """Scan a single file for mock usage indicators."""
    mock_indicators = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f'Error reading {file_path}: {e}')
        return []
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            continue
        if re.search('from unittest\\.mock import|import unittest\\.mock', line):
            mock_indicators.append({'line': line_num, 'kind': 'unittest_mock_import', 'snippet': line.strip()})
        mock_patterns = [('patch\\s*\\(', 'patch'), ('MagicMock\\s*\\(', 'MagicMock'), ('AsyncMock\\s*\\(', 'AsyncMock'), ('Mock\\s*\\(', 'Mock'), ('@patch\\s*\\(', 'patch_decorator'), ('@patch\\.object\\s*\\(', 'patch_object_decorator')]
        for pattern, kind in mock_patterns:
            if re.search(pattern, line):
                mock_indicators.append({'line': line_num, 'kind': kind, 'snippet': line.strip()})
        fixture_patterns = ['@pytest\\.fixture.*def\\s+(mock_\\w+)', '@pytest\\.fixture.*def\\s+(\\w*mock\\w*)', '@pytest\\.fixture.*def\\s+(fake_\\w+)', '@pytest\\.fixture.*def\\s+(\\w*fake\\w*)', '@pytest\\.fixture.*def\\s+(dummy_\\w+)', '@pytest\\.fixture.*def\\s+(\\w*dummy\\w*)']
        for pattern in fixture_patterns:
            match = re.search(pattern, line)
            if match:
                mock_indicators.append({'line': line_num, 'kind': 'mock_fixture', 'snippet': line.strip()})
        if re.search('def\\s+(mock_\\w+|fake_\\w+|dummy_\\w+|\\w*mock\\w*|\\w*fake\\w*|\\w*dummy\\w*)\\s*\\(', line):
            if line_num > 1 and '@pytest.fixture' in lines[line_num - 2]:
                mock_indicators.append({'line': line_num, 'kind': 'mock_fixture', 'snippet': line.strip()})
        if re.search('=\\s*\\{[^}]*["\\\'](?:test|sample|mock|fake|dummy)["\\\']', line):
            mock_indicators.append({'line': line_num, 'kind': 'sample_data_dict', 'snippet': line.strip()})
        if re.search('fakeredis|FakeRedis', line):
            mock_indicators.append({'line': line_num, 'kind': 'fakeredis', 'snippet': line.strip()})
        if re.search('class\\s+Mock\\w+|class\\s+\\w*Mock\\w*|class\\s+Fake\\w+|class\\s+\\w*Fake\\w*', line):
            mock_indicators.append({'line': line_num, 'kind': 'mock_class', 'snippet': line.strip()})
    return mock_indicators

def scan_all_test_files():
    """Scan all test files for mock usage indicators."""
    test_files = []
    test_patterns = ['tests/**/test_*.py', 'tests/**/*.py', '**/test_*.py', '**/*_test.py', '**/test*.py']
    for pattern in test_patterns:
        found_files = Path('.').glob(pattern)
        filtered_files = [f for f in found_files if '.venv' not in str(f) and 'venv' not in str(f) and ('env' not in str(f))]
        test_files.extend(filtered_files)
    test_files = sorted(list(set(test_files)))
    print(f'Found {len(test_files)} test files to scan')
    results = {}
    for test_file in test_files:
        file_path = str(test_file)
        print(f'Scanning {file_path}...')
        indicators = scan_file_for_mock_indicators(file_path)
        if indicators:
            results[file_path] = indicators
    return results

def main():
    """Main function to run the mock scanner."""
    print('Mock Usage Scanner - Step 2')
    print('=' * 50)
    results = scan_all_test_files()
    print(f'\nScan complete. Found mock usage in {len(results)} files.')
    for file_path, indicators in results.items():
        print(f'\n{file_path}:')
        for indicator in indicators:
            print(f"  Line {indicator['line']}: {indicator['kind']} - {indicator['snippet'][:80]}...")
    with open('mock_usage_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to mock_usage_results.json')
    total_indicators = sum((len(indicators) for indicators in results.values()))
    print(f'\nSummary Statistics:')
    print(f'- Total files with mock usage: {len(results)}')
    print(f'- Total mock indicators found: {total_indicators}')
    type_counts = {}
    for indicators in results.values():
        for indicator in indicators:
            kind = indicator['kind']
            type_counts[kind] = type_counts.get(kind, 0) + 1
    print(f'\nMock usage by type:')
    for kind, count in sorted(type_counts.items()):
        print(f'  {kind}: {count}')
if __name__ == '__main__':
    main()
