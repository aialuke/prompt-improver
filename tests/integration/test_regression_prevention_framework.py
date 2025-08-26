"""Integration tests for Performance Regression Prevention Framework.

Tests the complete framework integration including:
- Architectural compliance monitoring
- Startup performance tracking
- Regression prevention orchestration
- CI/CD integration capabilities
- VS Code diagnostics integration
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from prompt_improver.monitoring.regression import (
    ArchitecturalComplianceMonitor,
    CIIntegration,
    RegressionPreventionFramework,
    StartupPerformanceTracker,
    VSCodeDiagnosticsMonitor,
)


class TestArchitecturalComplianceMonitor:
    """Test architectural compliance monitoring."""

    @pytest.mark.asyncio
    async def test_compliance_check(self):
        """Test basic compliance checking."""
        monitor = ArchitecturalComplianceMonitor()

        # Run compliance check
        report = await monitor.check_compliance(strict=False)

        # Validate report structure
        assert hasattr(report, 'compliance_ratio')
        assert hasattr(report, 'violations')
        assert hasattr(report, 'protocol_domains')
        assert isinstance(report.violations, list)
        assert 0.0 <= report.compliance_ratio <= 1.0

    @pytest.mark.asyncio
    async def test_protocol_file_analysis(self):
        """Test protocol file analysis."""
        monitor = ArchitecturalComplianceMonitor()

        # Create a test protocol file with violations
        with tempfile.NamedTemporaryFile(encoding='utf-8', mode='w', suffix='.py', delete=False) as f:
            f.write("""
# Test protocol file with violations
import sqlalchemy  # This should be in TYPE_CHECKING
from asyncpg import Connection  # This should be in TYPE_CHECKING

from typing import Protocol

class TestProtocol(Protocol):
    def test_method(self) -> None: ...
""")
            f.flush()
            temp_file = Path(f.name)

        try:
            # Analyze the file
            violations = await monitor._analyze_protocol_file(temp_file)

            # Should detect heavy imports outside TYPE_CHECKING
            assert len(violations) > 0

            # Check for direct heavy import violations
            heavy_import_violations = [v for v in violations if v.violation_type == "direct_heavy_import"]
            assert len(heavy_import_violations) >= 1  # Should detect sqlalchemy and/or asyncpg

        finally:
            temp_file.unlink()  # Clean up

    def test_compliance_summary(self):
        """Test compliance summary generation."""
        monitor = ArchitecturalComplianceMonitor()

        # Get compliance summary (should handle empty state gracefully)
        summary = monitor.get_compliance_summary()

        assert 'status' in summary
        assert 'message' in summary


class TestStartupPerformanceTracker:
    """Test startup performance tracking."""

    def test_performance_tracking(self):
        """Test basic performance tracking."""
        tracker = StartupPerformanceTracker()

        # Test monitoring context
        with tracker.monitor_startup("test_component") as profile:
            # Simulate some work
            time.sleep(0.01)  # 10ms

            # Profile should be available during monitoring
            assert profile is not None
            assert profile.startup_time_seconds == 0  # Not calculated yet

        # Check that profile was updated
        assert profile.startup_time_seconds > 0

        # Get performance summary
        summary = tracker.get_performance_summary()
        assert summary['status'] != 'no_profiles'
        assert 'latest_startup' in summary

    def test_slow_import_detection(self):
        """Test slow import detection."""
        tracker = StartupPerformanceTracker()

        # Get slow imports (should handle empty state)
        slow_imports = tracker.get_slow_imports(threshold_seconds=0.01)
        assert isinstance(slow_imports, list)

    @pytest.mark.asyncio
    async def test_startup_validation(self):
        """Test startup performance validation."""
        tracker = StartupPerformanceTracker()

        # Test with no profiles
        meets_target, analysis = await tracker.validate_startup_performance(target_duration_seconds=0.5)
        assert not meets_target  # No profiles should fail
        assert 'error' in analysis

        # Add a profile by running monitoring
        with tracker.monitor_startup("validation_test") as profile:
            time.sleep(0.01)  # Small delay

        # Now test validation with profile
        meets_target, analysis = await tracker.validate_startup_performance(target_duration_seconds=1.0)
        assert isinstance(meets_target, bool)
        assert 'meets_target' in analysis
        assert 'actual_duration_seconds' in analysis


class TestRegressionPreventionFramework:
    """Test the unified regression prevention framework."""

    @pytest.mark.asyncio
    async def test_regression_check(self):
        """Test comprehensive regression check."""
        framework = RegressionPreventionFramework(integration_with_existing_monitoring=False)

        # Run regression check
        report = await framework.check_for_regressions(triggered_by="test")

        # Validate report structure
        assert hasattr(report, 'overall_status')
        assert hasattr(report, 'framework_health')
        assert hasattr(report, 'alerts')
        assert isinstance(report.alerts, list)
        assert 0.0 <= report.framework_health <= 1.0
        assert report.overall_status in {'healthy', 'warnings', 'regressions_detected'}

    @pytest.mark.asyncio
    async def test_pr_validation(self):
        """Test PR change validation."""
        framework = RegressionPreventionFramework(integration_with_existing_monitoring=False)

        # Test with empty file list
        should_approve, validation = await framework.validate_pr_changes([])
        assert isinstance(should_approve, bool)
        assert isinstance(validation, dict)
        assert 'should_approve' in validation

    def test_framework_status(self):
        """Test framework status reporting."""
        framework = RegressionPreventionFramework(integration_with_existing_monitoring=False)

        # Get status (should handle no checks gracefully)
        status = framework.get_framework_status()

        assert 'status' in status
        assert 'performance_baselines' in status
        assert 'integration_status' in status


class TestCIIntegration:
    """Test CI/CD integration functionality."""

    def test_pre_commit_config_generation(self):
        """Test pre-commit configuration generation."""
        ci = CIIntegration()

        config = ci.generate_pre_commit_config()

        # Should contain our custom hooks
        assert 'architectural-compliance-check' in config
        assert 'protocol-type-checking-compliance' in config
        assert 'dependency-contamination-check' in config

    def test_github_workflow_generation(self):
        """Test GitHub Actions workflow generation."""
        ci = CIIntegration()

        workflow = ci.generate_github_actions_workflow()

        # Should contain key workflow elements
        assert 'Performance Regression Prevention' in workflow
        assert 'regression-prevention' in workflow
        assert 'architectural-compliance-check' in workflow

    def test_startup_performance_test(self):
        """Test CI startup performance testing."""
        ci = CIIntegration()

        # Run startup test
        result = ci.run_startup_performance_test(max_startup_time=2.0)  # Generous timeout for CI

        assert 'test_passed' in result
        assert 'startup_time_seconds' in result or 'error' in result
        assert 'message' in result

    @pytest.mark.asyncio
    async def test_pre_commit_check(self):
        """Test pre-commit check functionality."""
        ci = CIIntegration()

        # Test with empty file list
        should_allow, report = await ci.run_pre_commit_check([], strict=True)

        assert isinstance(should_allow, bool)
        assert isinstance(report, dict)
        assert 'status' in report


class TestVSCodeDiagnosticsMonitor:
    """Test VS Code diagnostics integration."""

    @pytest.mark.asyncio
    async def test_diagnostics_analysis(self):
        """Test diagnostics analysis functionality."""
        monitor = VSCodeDiagnosticsMonitor()

        # Test with empty diagnostics
        report = await monitor.analyze_diagnostics_for_regressions({})

        assert report.total_files_monitored == 0
        assert report.diagnostics_found == 0
        assert report.regression_diagnostics == 0

    @pytest.mark.asyncio
    async def test_file_monitoring(self):
        """Test file-level violation monitoring."""
        monitor = VSCodeDiagnosticsMonitor()

        # Create a test file with potential violations
        with tempfile.NamedTemporaryFile(encoding='utf-8', mode='w', suffix='.py', delete=False) as f:
            f.write("""
# Test file
import os
import sys

def test_function():
    pass
""")
            f.flush()
            temp_file = f.name

        try:
            # Monitor the file (should not find violations in this simple file)
            violations = await monitor.monitor_file_for_violations(temp_file)
            assert isinstance(violations, list)

        finally:
            Path(temp_file).unlink()  # Clean up

    def test_monitoring_status(self):
        """Test monitoring status reporting."""
        monitor = VSCodeDiagnosticsMonitor()

        status = monitor.get_monitoring_status()

        assert 'monitoring_active' in status
        assert 'ide_available' in status
        assert 'integration_health' in status


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete regression prevention workflow."""
        # Initialize all components
        compliance_monitor = ArchitecturalComplianceMonitor()
        startup_tracker = StartupPerformanceTracker()
        framework = RegressionPreventionFramework(integration_with_existing_monitoring=False)

        # 1. Run compliance check
        compliance_report = await compliance_monitor.check_compliance(strict=False)
        assert compliance_report is not None

        # 2. Test startup monitoring
        with startup_tracker.monitor_startup("integration_test") as profile:
            time.sleep(0.01)  # Minimal work

        startup_summary = startup_tracker.get_performance_summary()
        assert startup_summary['status'] != 'no_profiles'

        # 3. Run full regression check
        regression_report = await framework.check_for_regressions(triggered_by="integration_test")
        assert regression_report is not None
        assert hasattr(regression_report, 'overall_status')

        # 4. Test framework status
        framework_status = framework.get_framework_status()
        assert framework_status['status'] != 'not_initialized'

    @pytest.mark.asyncio
    async def test_performance_baseline_protection(self):
        """Test that performance baselines are properly protected."""
        framework = RegressionPreventionFramework(integration_with_existing_monitoring=False)

        # Get performance baselines
        status = framework.get_framework_status()
        baselines = status['performance_baselines']

        # Verify critical baselines are defined
        assert 'startup_improvement_ratio' in baselines
        assert 'max_startup_time_ms' in baselines
        assert 'protocol_compliance_ratio' in baselines
        assert 'zero_critical_violations' in baselines

        # Verify baseline values match expectations
        assert baselines['startup_improvement_ratio'] == 0.924  # 92.4%
        assert baselines['max_startup_time_ms'] == 500  # 500ms target
        assert baselines['protocol_compliance_ratio'] == 1.0  # 100%
        assert baselines['zero_critical_violations']


@pytest.mark.integration
class TestMonitoringIntegration:
    """Test integration with existing monitoring infrastructure."""

    @pytest.mark.asyncio
    async def test_opentelemetry_integration(self):
        """Test OpenTelemetry metrics integration."""
        # This test would verify OpenTelemetry metrics are properly recorded
        # when the framework runs. For now, we just test that the framework
        # handles missing OpenTelemetry gracefully.

        with patch('prompt_improver.monitoring.regression.architectural_compliance.OPENTELEMETRY_AVAILABLE', False):
            monitor = ArchitecturalComplianceMonitor()
            report = await monitor.check_compliance(strict=False)

            # Should still work without OpenTelemetry
            assert report is not None

    @pytest.mark.asyncio
    async def test_slo_monitoring_integration(self):
        """Test SLO monitoring system integration."""
        # Test that framework can work with or without SLO monitoring

        framework = RegressionPreventionFramework(integration_with_existing_monitoring=False)

        # Should work without SLO integration
        report = await framework.check_for_regressions(triggered_by="test")
        assert report is not None

        # Test framework status includes integration info
        status = framework.get_framework_status()
        assert 'integration_status' in status
