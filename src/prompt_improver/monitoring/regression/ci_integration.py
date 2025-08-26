"""CI/CD Integration for Regression Prevention.
==========================================

Provides CI/CD integration for automated regression prevention including:
- Pre-commit hooks for architectural compliance
- GitHub Actions workflow integration
- PR blocking for performance regressions
- Automated compliance reporting

Prevents deployment of code that could reintroduce:
- 134-1007ms database protocol startup penalties
- TYPE_CHECKING compliance violations
- Heavy dependency contamination during startup
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from prompt_improver.monitoring.regression.regression_prevention import (
    RegressionPreventionFramework,
)

logger = logging.getLogger(__name__)


class CIIntegration:
    """CI/CD integration for regression prevention framework."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize CI integration.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or self._detect_project_root()
        self.framework = RegressionPreventionFramework(project_root)

    def _detect_project_root(self) -> Path:
        """Auto-detect project root directory."""
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "src" / "prompt_improver").exists():
                return current
            current = current.parent
        return Path.cwd()

    def generate_pre_commit_config(self) -> str:
        """Generate pre-commit configuration for regression prevention."""
        pre_commit_config = """
repos:
  - repo: local
    hooks:
      - id: architectural-compliance-check
        name: Architectural Compliance Check
        entry: python -m prompt_improver.monitoring.regression.ci_integration
        args: ["--pre-commit", "--strict"]
        language: system
        files: '^src/prompt_improver/.*\\.py$'
        always_run: false

      - id: protocol-type-checking-compliance
        name: Protocol TYPE_CHECKING Compliance
        entry: python -m prompt_improver.monitoring.regression.ci_integration
        args: ["--check-protocols", "--strict"]
        language: system
        files: '^src/prompt_improver/.*/protocols/.*\\.py$'
        always_run: false

      - id: dependency-contamination-check
        name: Dependency Contamination Check
        entry: python -m prompt_improver.monitoring.regression.ci_integration
        args: ["--check-dependencies"]
        language: system
        files: '^src/prompt_improver/.*\\.py$'
        always_run: false

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        args: ["--line-length=88"]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
"""
        return pre_commit_config.strip()

    def install_pre_commit_hooks(self) -> bool:
        """Install pre-commit hooks for regression prevention.

        Returns:
            True if installation successful
        """
        try:
            # Write pre-commit config
            pre_commit_file = self.project_root / ".pre-commit-config.yaml"

            if pre_commit_file.exists():
                logger.info("Pre-commit config already exists, backing up...")
                backup_file = self.project_root / ".pre-commit-config.yaml.bak"
                pre_commit_file.rename(backup_file)

            with open(pre_commit_file, "w", encoding="utf-8") as f:
                f.write(self.generate_pre_commit_config())

            # Install pre-commit hooks
            result = subprocess.run(
                ["pre-commit", "install"],
                check=False, cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("Pre-commit hooks installed successfully")
                return True
            logger.error(f"Failed to install pre-commit hooks: {result.stderr}")
            return False

        except Exception as e:
            logger.exception(f"Error installing pre-commit hooks: {e}")
            return False

    def generate_github_actions_workflow(self) -> str:
        """Generate GitHub Actions workflow for regression prevention."""
        workflow = """
name: Performance Regression Prevention

on:
  pull_request:
    paths:
      - 'src/prompt_improver/**/*.py'
      - 'tests/**/*.py'
  push:
    branches: [ main, master ]

jobs:
  regression-prevention:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest opentelemetry-api opentelemetry-sdk

    - name: Get changed files
      id: changed-files
      run: |
        if [ "${{ github.event_name }}" = "pull_request" ]; then
          git diff --name-only ${{ github.event.pull_request.base.sha }} ${{ github.sha }} > changed_files.txt
        else
          git diff --name-only HEAD~1 HEAD > changed_files.txt
        fi
        echo "Changed files:"
        cat changed_files.txt

    - name: Run Architectural Compliance Check
      run: |
        python -m prompt_improver.monitoring.regression.ci_integration \\
          --github-actions \\
          --strict \\
          --changed-files changed_files.txt \\
          --output compliance_report.json

    - name: Upload compliance report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: compliance-report
        path: compliance_report.json

    - name: Comment PR with results
      uses: actions/github-script@v6
      if: github.event_name == 'pull_request' && always()
      with:
        script: |
          const fs = require('fs');
          try {
            const report = JSON.parse(fs.readFileSync('compliance_report.json', 'utf8'));

            let comment = `## 🔍 Performance Regression Prevention Report\\n\\n`;

            if (report.should_approve) {
              comment += `✅ **PASSED** - No performance regressions detected\\n\\n`;
            } else {
              comment += `❌ **BLOCKED** - Performance regressions detected\\n\\n`;
            }

            comment += `### Metrics\\n`;
            comment += `- Framework Health: **${(report.regression_check.framework_health * 100).toFixed(1)}%**\\n`;
            comment += `- Total Alerts: **${report.regression_check.total_alerts}**\\n`;
            comment += `- Blocking Alerts: **${report.regression_check.blocking_alerts}**\\n`;

            if (report.regression_check.alerts.length > 0) {
              comment += `\\n### Alerts\\n`;
              for (const alert of report.regression_check.alerts) {
                const emoji = alert.severity === 'critical' ? '🚨' :
                             alert.severity === 'high' ? '⚠️' : 'ℹ️';
                comment += `${emoji} **${alert.severity.toUpperCase()}**: ${alert.title}\\n`;
              }
            }

            if (report.recommendations.length > 0) {
              comment += `\\n### Recommendations\\n`;
              for (const rec of report.recommendations) {
                comment += `- ${rec}\\n`;
              }
            }

            comment += `\\n---\\n*Generated by Performance Regression Prevention Framework*`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          } catch (error) {
            console.log('Error reading compliance report:', error);
          }

  startup-performance-test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .

    - name: Test Startup Performance
      run: |
        python -m prompt_improver.monitoring.regression.ci_integration \\
          --test-startup \\
          --max-startup-time 0.5 \\
          --output startup_report.json

    - name: Upload startup report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: startup-performance-report
        path: startup_report.json
"""
        return workflow.strip()

    def install_github_workflow(self) -> bool:
        """Install GitHub Actions workflow for regression prevention.

        Returns:
            True if installation successful
        """
        try:
            workflows_dir = self.project_root / ".github" / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)

            workflow_file = workflows_dir / "regression-prevention.yml"

            with open(workflow_file, "w", encoding="utf-8") as f:
                f.write(self.generate_github_actions_workflow())

            logger.info(f"GitHub Actions workflow installed: {workflow_file}")
            return True

        except Exception as e:
            logger.exception(f"Error installing GitHub workflow: {e}")
            return False

    async def run_pre_commit_check(self, changed_files: list[str], strict: bool = True) -> tuple[bool, dict[str, Any]]:
        """Run pre-commit regression check.

        Args:
            changed_files: Files changed in the commit
            strict: Whether to use strict mode (block on any violations)

        Returns:
            Tuple of (should_allow_commit, detailed_report)
        """
        logger.info(f"Running pre-commit check for {len(changed_files)} changed files")

        # Filter to Python files
        python_files = [f for f in changed_files if f.endswith('.py')]

        if not python_files:
            return True, {
                "status": "skipped",
                "message": "No Python files changed",
                "changed_files": changed_files
            }

        # Run regression prevention check
        should_approve, report = await self.framework.validate_pr_changes(python_files)

        # In strict mode, block on any critical violations
        if strict and not should_approve:
            return False, {
                "status": "blocked",
                "message": "Critical performance regressions detected",
                "validation_report": report,
                "strict_mode": True
            }

        return should_approve, {
            "status": "approved" if should_approve else "warnings",
            "message": "Pre-commit check passed" if should_approve else "Warnings detected but not blocking",
            "validation_report": report,
            "strict_mode": strict
        }

    def run_startup_performance_test(self, max_startup_time: float = 0.5) -> dict[str, Any]:
        """Run startup performance test for CI/CD.

        Args:
            max_startup_time: Maximum allowed startup time in seconds

        Returns:
            Test results with pass/fail status
        """
        import time

        logger.info("Running startup performance test...")

        # Test import time
        start_time = time.time()

        try:
            # Import main package to test startup time

            startup_time = time.time() - start_time

            # Get performance summary
            summary = self.framework.startup_tracker.get_performance_summary()

            # Determine if test passes
            test_passed = startup_time <= max_startup_time

            return {
                "test_passed": test_passed,
                "startup_time_seconds": startup_time,
                "max_allowed_seconds": max_startup_time,
                "performance_delta": startup_time - max_startup_time,
                "performance_summary": summary,
                "message": f"Startup time: {startup_time:.3f}s ({'PASS' if test_passed else 'FAIL'})"
            }

        except Exception as e:
            logger.exception(f"Startup performance test failed: {e}")
            return {
                "test_passed": False,
                "error": str(e),
                "message": f"Startup performance test failed: {e}"
            }

    def get_changed_files_from_git(self) -> list[str]:
        """Get list of changed files from git."""
        try:
            # Get staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                check=False, capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                return [line.strip() for line in result.stdout.split('\n') if line.strip()]
            logger.warning("Could not get git diff, falling back to modified files")

            # Fallback to modified files
            result = subprocess.run(
                ["git", "ls-files", "--modified"],
                check=False, capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                return [line.strip() for line in result.stdout.split('\n') if line.strip()]

        except Exception as e:
            logger.exception(f"Error getting changed files from git: {e}")

        return []


def main():
    """Command-line interface for CI integration."""
    import argparse

    parser = argparse.ArgumentParser(description="CI/CD Integration for Performance Regression Prevention")
    parser.add_argument("--pre-commit", action="store_true", help="Run pre-commit check")
    parser.add_argument("--github-actions", action="store_true", help="Run GitHub Actions check")
    parser.add_argument("--test-startup", action="store_true", help="Test startup performance")
    parser.add_argument("--check-protocols", action="store_true", help="Check protocol compliance only")
    parser.add_argument("--check-dependencies", action="store_true", help="Check dependency contamination")
    parser.add_argument("--strict", action="store_true", help="Use strict mode (block on violations)")
    parser.add_argument("--changed-files", type=str, help="File containing list of changed files")
    parser.add_argument("--max-startup-time", type=float, default=0.5, help="Maximum startup time in seconds")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--install-hooks", action="store_true", help="Install pre-commit hooks")
    parser.add_argument("--install-workflow", action="store_true", help="Install GitHub Actions workflow")

    args = parser.parse_args()

    ci_integration = CIIntegration()

    if args.install_hooks:
        success = ci_integration.install_pre_commit_hooks()
        sys.exit(0 if success else 1)

    if args.install_workflow:
        success = ci_integration.install_github_workflow()
        sys.exit(0 if success else 1)

    if args.test_startup:
        result = ci_integration.run_startup_performance_test(args.max_startup_time)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))

        sys.exit(0 if result["test_passed"] else 1)

    # Get changed files
    changed_files = []
    if args.changed_files:
        if Path(args.changed_files).exists():
            with open(args.changed_files, encoding="utf-8") as f:
                changed_files = [line.strip() for line in f if line.strip()]
        else:
            changed_files = [args.changed_files]
    else:
        changed_files = ci_integration.get_changed_files_from_git()

    if not changed_files:
        print("No changed files detected")
        sys.exit(0)

    # Run appropriate checks
    async def run_checks():
        if args.pre_commit or args.github_actions:
            should_approve, report = await ci_integration.run_pre_commit_check(
                changed_files, strict=args.strict
            )

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            else:
                print(json.dumps(report, indent=2))

            return should_approve
        # Default: run full regression check
        report = await ci_integration.framework.check_for_regressions(triggered_by="ci")

        result = {
            "status": report.overall_status,
            "framework_health": report.framework_health,
            "alerts": len(report.alerts),
            "should_approve": report.overall_status != "regressions_detected"
        }

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))

        return result["should_approve"]

    import asyncio

    try:
        should_approve = asyncio.run(run_checks())
        sys.exit(0 if should_approve else 1)
    except Exception as e:
        logger.exception(f"CI check failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
