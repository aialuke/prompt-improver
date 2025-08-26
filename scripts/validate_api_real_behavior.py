#!/usr/bin/env python3
"""API Real Behavior Validation Script.

Validates that all API service mocks have been eliminated and replaced with real service calls.
Ensures 100% real behavior testing for API endpoints and service integration.

Usage:
    python scripts/validate_api_real_behavior.py

Returns:
    0 - All API mocks eliminated, real behavior testing active
    1 - API mocks still present or validation failed
"""

import ast
import re
import sys
from pathlib import Path

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


class APIRealBehaviorValidator:
    """Validates elimination of API service mocks and real behavior implementation."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.src_dir = project_root / "src"

        # Patterns that indicate mocking (should be eliminated from API tests)
        self.mock_patterns = [
            r'unittest\.mock',
            r'@patch\s*\(',
            r'@patch\s*\.\s*object',
            r'Mock\(',
            r'MagicMock\(',
            r'AsyncMock\(',
            r'mock_.*api',
            r'mock_.*service',
            r'fake_.*api',
            r'patch.*api',
            r'patch.*service',
            r'\.patch\(',
            r'mock\.patch',
        ]

        # Patterns that indicate real behavior (should be present)
        self.real_behavior_patterns = [
            r'TestClient\(',
            r'real_api_client',
            r'api_helpers',
            r'api_performance_monitor',
            r'real_.*_client',
            r'FastAPI\(',
            r'create_test_app',
            r'api_test_data',
        ]

        self.validation_results = {
            'mocks_eliminated': [],
            'real_behavior_implemented': [],
            'remaining_mocks': [],
            'missing_real_behavior': [],
            'file_analysis': {},
        }

    def scan_file_for_patterns(self, file_path: Path) -> dict[str, list[tuple[int, str]]]:
        """Scan a file for mock patterns and real behavior patterns."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            mock_matches = []
            real_behavior_matches = []

            lines = content.split('\n')

            # Check for mock patterns
            for pattern in self.mock_patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        mock_matches.append((line_num, line.strip()))

            # Check for real behavior patterns
            for pattern in self.real_behavior_patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        real_behavior_matches.append((line_num, line.strip()))

            return {
                'mock_matches': mock_matches,
                'real_behavior_matches': real_behavior_matches,
                'file_size': len(lines),
                'imports': self._extract_imports(content),
            }

        except Exception as e:
            print(f"{RED}Error scanning {file_path}: {e}{RESET}")
            return {'mock_matches': [], 'real_behavior_matches': [], 'file_size': 0, 'imports': []}

    def _extract_imports(self, content: str) -> list[str]:
        """Extract import statements from file content."""
        try:
            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(name.name for name in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    imports.extend(f"{module}.{name.name}" for name in node.names)

            return imports
        except:
            return []

    def validate_api_test_files(self) -> None:
        """Validate API test files for mock elimination and real behavior."""
        print(f"{BLUE}{BOLD}🔍 Validating API Test Files{RESET}")
        print("=" * 50)

        api_test_patterns = [
            "tests/api/**/*.py",
            "tests/integration/**/test_*api*.py",
            "tests/integration/**/test_*service*.py",
            "tests/integration/**/test_*endpoint*.py",
        ]

        api_test_files = []
        for pattern in api_test_patterns:
            api_test_files.extend(self.project_root.glob(pattern))

        for test_file in api_test_files:
            if test_file.name == "__init__.py":
                continue

            analysis = self.scan_file_for_patterns(test_file)
            self.validation_results['file_analysis'][str(test_file)] = analysis

            relative_path = test_file.relative_to(self.project_root)

            # Check for remaining mocks
            if analysis['mock_matches']:
                self.validation_results['remaining_mocks'].append({
                    'file': str(relative_path),
                    'matches': analysis['mock_matches']
                })
                print(f"{RED}❌ {relative_path}: {len(analysis['mock_matches'])} mock patterns found{RESET}")
                for line_num, line in analysis['mock_matches'][:3]:  # Show first 3
                    print(f"   Line {line_num}: {line}")
                if len(analysis['mock_matches']) > 3:
                    print(f"   ... and {len(analysis['mock_matches']) - 3} more")
            else:
                self.validation_results['mocks_eliminated'].append(str(relative_path))
                print(f"{GREEN}✅ {relative_path}: No mock patterns found{RESET}")

            # Check for real behavior implementation
            if analysis['real_behavior_matches']:
                self.validation_results['real_behavior_implemented'].append({
                    'file': str(relative_path),
                    'matches': analysis['real_behavior_matches']
                })
                print(f"{GREEN}✅ {relative_path}: {len(analysis['real_behavior_matches'])} real behavior patterns found{RESET}")
            else:
                self.validation_results['missing_real_behavior'].append(str(relative_path))
                print(f"{YELLOW}⚠️  {relative_path}: No real behavior patterns found{RESET}")

    def validate_new_test_files(self) -> None:
        """Validate that new real behavior test files exist."""
        print(f"\n{BLUE}{BOLD}🔍 Validating New Real Behavior Test Files{RESET}")
        print("=" * 50)

        expected_real_test_files = [
            "tests/api/test_analytics_endpoints_real.py",
            "tests/api/test_health_endpoints_real.py",
            "tests/api/test_apriori_endpoints_real.py",
            "tests/api/conftest.py",
            "tests/integration/api/test_service_integration_real.py",
        ]

        for expected_file in expected_real_test_files:
            file_path = self.project_root / expected_file
            if file_path.exists():
                analysis = self.scan_file_for_patterns(file_path)

                # Check that real test files have real behavior patterns
                if analysis['real_behavior_matches']:
                    print(f"{GREEN}✅ {expected_file}: Real behavior test file exists and configured{RESET}")
                else:
                    print(f"{YELLOW}⚠️  {expected_file}: Exists but missing real behavior patterns{RESET}")
            else:
                print(f"{RED}❌ {expected_file}: Missing real behavior test file{RESET}")

    def validate_app_structure(self) -> None:
        """Validate that API app structure supports real testing."""
        print(f"\n{BLUE}{BOLD}🔍 Validating API App Structure{RESET}")
        print("=" * 50)

        # Check for FastAPI app factory
        app_file = self.project_root / "src/prompt_improver/api/app.py"
        if app_file.exists():
            analysis = self.scan_file_for_patterns(app_file)
            if any("FastAPI" in line for _, line in analysis['real_behavior_matches']):
                print(f"{GREEN}✅ FastAPI app factory exists{RESET}")
            else:
                print(f"{YELLOW}⚠️  app.py exists but may not have FastAPI factory{RESET}")
        else:
            print(f"{RED}❌ Missing src/prompt_improver/api/app.py{RESET}")

        # Check for API routers
        api_files = [
            "src/prompt_improver/api/analytics_endpoints.py",
            "src/prompt_improver/api/health.py",
            "src/prompt_improver/api/apriori_endpoints.py",
        ]

        for api_file in api_files:
            file_path = self.project_root / api_file
            if file_path.exists():
                print(f"{GREEN}✅ {api_file}: API router exists{RESET}")
            else:
                print(f"{YELLOW}⚠️  {api_file}: API router missing{RESET}")

    def generate_summary_report(self) -> bool:
        """Generate summary report and return success status."""
        print(f"\n{BLUE}{BOLD}📊 API Real Behavior Validation Summary{RESET}")
        print("=" * 60)

        total_files = len(self.validation_results['file_analysis'])
        mocks_eliminated_count = len(self.validation_results['mocks_eliminated'])
        remaining_mocks_count = len(self.validation_results['remaining_mocks'])
        real_behavior_count = len(self.validation_results['real_behavior_implemented'])

        print(f"Total API test files analyzed: {total_files}")
        print(f"Files with mocks eliminated: {GREEN}{mocks_eliminated_count}{RESET}")
        print(f"Files with remaining mocks: {RED}{remaining_mocks_count}{RESET}")
        print(f"Files with real behavior: {GREEN}{real_behavior_count}{RESET}")

        # Calculate success metrics
        mock_elimination_rate = (mocks_eliminated_count / total_files * 100) if total_files > 0 else 0
        real_behavior_rate = (real_behavior_count / total_files * 100) if total_files > 0 else 0

        print(f"\n{BOLD}Success Metrics:{RESET}")
        print(f"Mock elimination rate: {GREEN if mock_elimination_rate >= 90 else YELLOW if mock_elimination_rate >= 70 else RED}{mock_elimination_rate:.1f}%{RESET}")
        print(f"Real behavior implementation rate: {GREEN if real_behavior_rate >= 80 else YELLOW if real_behavior_rate >= 60 else RED}{real_behavior_rate:.1f}%{RESET}")

        # Determine overall success
        success = (
            remaining_mocks_count == 0 and  # No mocks remaining
            real_behavior_count >= total_files * 0.8 and  # 80% real behavior implementation
            total_files > 0  # At least some files analyzed
        )

        if success:
            print(f"\n{GREEN}{BOLD}✅ SUCCESS: API real behavior testing implementation complete!{RESET}")
            print(f"{GREEN}All API service mocks have been eliminated.{RESET}")
            print(f"{GREEN}Real service integration testing is active.{RESET}")
        else:
            print(f"\n{RED}{BOLD}❌ INCOMPLETE: API real behavior testing needs work{RESET}")
            if remaining_mocks_count > 0:
                print(f"{RED}• {remaining_mocks_count} files still have API service mocks{RESET}")
            if real_behavior_count < total_files * 0.8:
                print(f"{RED}• Real behavior implementation below 80% threshold{RESET}")

        return success

    def generate_detailed_report(self) -> None:
        """Generate detailed validation report."""
        print(f"\n{BLUE}{BOLD}📋 Detailed Validation Report{RESET}")
        print("=" * 60)

        if self.validation_results['remaining_mocks']:
            print(f"\n{RED}{BOLD}Files with Remaining Mocks:{RESET}")
            for file_info in self.validation_results['remaining_mocks']:
                print(f"  • {file_info['file']}")
                for line_num, line in file_info['matches'][:2]:
                    print(f"    Line {line_num}: {line}")

        if self.validation_results['missing_real_behavior']:
            print(f"\n{YELLOW}{BOLD}Files Missing Real Behavior Patterns:{RESET}")
            for file_path in self.validation_results['missing_real_behavior']:
                print(f"  • {file_path}")

        print(f"\n{GREEN}{BOLD}Files with Successful Mock Elimination:{RESET}")
        for file_path in self.validation_results['mocks_eliminated'][:10]:  # Show first 10
            print(f"  • {file_path}")
        if len(self.validation_results['mocks_eliminated']) > 10:
            print(f"  ... and {len(self.validation_results['mocks_eliminated']) - 10} more")

    def run_validation(self) -> bool:
        """Run complete API real behavior validation."""
        print(f"{BOLD}🚀 API Real Behavior Validation Starting...{RESET}")
        print(f"Project root: {self.project_root}")
        print()

        # Run validation steps
        self.validate_api_test_files()
        self.validate_new_test_files()
        self.validate_app_structure()

        # Generate reports
        success = self.generate_summary_report()
        self.generate_detailed_report()

        return success


def main():
    """Main validation function."""
    # Find project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    if not (project_root / "pyproject.toml").exists():
        print(f"{RED}Error: Could not find project root (no pyproject.toml found){RESET}")
        return 1

    # Run validation
    validator = APIRealBehaviorValidator(project_root)
    success = validator.run_validation()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
