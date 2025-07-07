#!/usr/bin/env python3
"""
Phase 4 Regression Tests for Complex CLI Functions.
These tests ensure that refactored functions maintain the same behavior.
"""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from prompt_improver.cli import app


class TestLogsRegression:
    """Regression tests for the logs function (complexity 30, 35 branches)."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / ".local" / "share" / "apes" / "data" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create sample log file
        self.log_file = self.log_dir / "apes.log"
        sample_logs = [
            "2024-01-15 10:00:00 INFO Starting application...",
            "2024-01-15 10:00:01 DEBUG Database connection established",
            "2024-01-15 10:00:02 WARNING High memory usage detected: 85%",
            "2024-01-15 10:00:03 ERROR Failed to process request: timeout",
            "2024-01-15 10:00:04 INFO Request processed successfully",
        ] * 20  # Create 100 log entries

        with open(self.log_file, 'w', encoding="utf-8") as f:
            for log in sample_logs:
                f.write(log + '\n')

    def test_logs_basic_functionality(self):
        """Test basic logs command without follow mode."""
        with patch('pathlib.Path.home', return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ['logs', '--lines', '10'])

            # Should find and display logs
            assert result.exit_code == 0
            assert "Viewing logs:" in result.stdout

    def test_logs_level_filtering(self):
        """Test log level filtering functionality."""
        with patch('pathlib.Path.home', return_value=Path(self.temp_dir)):
            # Test INFO level filtering
            result = self.runner.invoke(app, ['logs', '--level', 'INFO', '--lines', '50'])
            assert result.exit_code == 0

            # Test ERROR level filtering
            result = self.runner.invoke(app, ['logs', '--level', 'ERROR', '--lines', '50'])
            assert result.exit_code == 0

    def test_logs_component_filtering(self):
        """Test component-specific log filtering."""
        # Create component-specific log file
        component_log = self.log_dir / "mcp.log"
        with open(component_log, 'w', encoding="utf-8") as f:
            f.write("2024-01-15 10:00:00 INFO MCP service started\n")

        with patch('pathlib.Path.home', return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ['logs', '--component', 'mcp'])
            assert result.exit_code == 0
            assert "Viewing logs:" in result.stdout

    def test_logs_nonexistent_directory(self):
        """Test behavior when log directory doesn't exist."""
        with patch('pathlib.Path.home', return_value=Path("/nonexistent")):
            result = self.runner.invoke(app, ['logs'])
            assert result.exit_code == 1
            assert "Log directory not found" in result.stdout

    def test_logs_nonexistent_component(self):
        """Test behavior when component log file doesn't exist."""
        with patch('pathlib.Path.home', return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ['logs', '--component', 'nonexistent'])
            assert result.exit_code == 1
            assert "Log file not found" in result.stdout
            assert "Available log files:" in result.stdout

    def test_logs_lines_parameter(self):
        """Test lines parameter functionality."""
        with patch('pathlib.Path.home', return_value=Path(self.temp_dir)):
            # Test with small number of lines
            result = self.runner.invoke(app, ['logs', '--lines', '5'])
            assert result.exit_code == 0

            # Test with large number of lines (more than available)
            result = self.runner.invoke(app, ['logs', '--lines', '1000'])
            assert result.exit_code == 0


class TestHealthRegression:
    """Regression tests for the health function (complexity 14)."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()

    @pytest.mark.asyncio
    async def test_health_basic_functionality(self):
        """Test basic health check functionality."""
        # Mock the health monitor to avoid actual system checks
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_health_results = {
                "overall_status": "healthy",
                "checks": {
                    "database": {"status": "healthy", "response_time_ms": 50.0, "message": "Connected"},
                    "mcp": {"status": "healthy", "response_time_ms": 25.0, "message": "Running"},
                    "system_resources": {"status": "healthy", "memory_usage_percent": 45.0, "cpu_usage_percent": 30.0}
                },
                "warning_checks": [],
                "failed_checks": []
            }
            mock_run.return_value = mock_health_results

            result = self.runner.invoke(app, ['health'])
            assert result.exit_code == 0
            assert "Running APES Health Check" in result.stdout

    def test_health_json_output(self):
        """Test health check with JSON output format."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_health_results = {
                "overall_status": "healthy",
                "checks": {},
                "warning_checks": [],
                "failed_checks": []
            }
            mock_run.return_value = mock_health_results

            result = self.runner.invoke(app, ['health', '--json'])
            assert result.exit_code == 0

    def test_health_detailed_mode(self):
        """Test health check with detailed diagnostics."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_health_results = {
                "overall_status": "warning",
                "checks": {
                    "system_resources": {
                        "status": "warning",
                        "memory_usage_percent": 85.0,
                        "cpu_usage_percent": 75.0,
                        "disk_usage_percent": 60.0
                    }
                },
                "warning_checks": ["system_resources"],
                "failed_checks": []
            }
            mock_run.return_value = mock_health_results

            result = self.runner.invoke(app, ['health', '--detailed'])
            assert result.exit_code == 0

    def test_health_error_handling(self):
        """Test health check error handling."""
        with patch('prompt_improver.cli.asyncio.run', side_effect=Exception("Health check failed")):
            result = self.runner.invoke(app, ['health'])
            assert result.exit_code == 1
            assert "Health check failed" in result.stdout


class TestAlertsRegression:
    """Regression tests for the alerts function (complexity 13)."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()

    def test_alerts_basic_functionality(self):
        """Test basic alerts command functionality."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_summary = {
                "alert_summary": {
                    "total_alerts": 5,
                    "critical_alerts": 2,
                    "warning_alerts": 3,
                    "most_common_alert": "High Memory Usage"
                },
                "current_performance": {
                    "avg_response_time_ms": 150.0,
                    "memory_usage_mb": 180.0,
                    "database_connections": 10
                },
                "health_status": "warning"
            }
            mock_run.return_value = mock_summary

            result = self.runner.invoke(app, ['alerts'])
            assert result.exit_code == 0
            assert "APES Alert Status" in result.stdout

    def test_alerts_severity_filtering(self):
        """Test alerts with severity filtering."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_summary = {
                "alert_summary": {
                    "total_alerts": 3,
                    "critical_alerts": 1,
                    "warning_alerts": 2,
                    "most_common_alert": "Database Connection Issue"
                },
                "current_performance": {},
                "health_status": "critical"
            }
            mock_run.return_value = mock_summary

            # Test critical severity filter
            result = self.runner.invoke(app, ['alerts', '--severity', 'critical'])
            assert result.exit_code == 0

            # Test warning severity filter
            result = self.runner.invoke(app, ['alerts', '--severity', 'warning'])
            assert result.exit_code == 0

    def test_alerts_time_period(self):
        """Test alerts with different time periods."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_summary = {
                "alert_summary": {"total_alerts": 0},
                "current_performance": {},
                "health_status": "healthy"
            }
            mock_run.return_value = mock_summary

            # Test with custom hours parameter
            result = self.runner.invoke(app, ['alerts', '--hours', '48'])
            assert result.exit_code == 0

    def test_alerts_no_alerts_found(self):
        """Test alerts command when no alerts are found."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_summary = {
                "alert_summary": {"total_alerts": 0},
                "current_performance": {},
                "health_status": "healthy"
            }
            mock_run.return_value = mock_summary

            result = self.runner.invoke(app, ['alerts'])
            assert result.exit_code == 0
            assert "No alerts found" in result.stdout

    def test_alerts_recommendations(self):
        """Test alerts command shows recommendations for unhealthy systems."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_summary = {
                "alert_summary": {
                    "total_alerts": 1,
                    "critical_alerts": 1,
                    "warning_alerts": 0
                },
                "current_performance": {
                    "avg_response_time_ms": 250.0,  # > 200
                    "memory_usage_mb": 220.0,       # > 200
                    "database_connections": 18      # > 15
                },
                "health_status": "critical"
            }
            mock_run.return_value = mock_summary

            result = self.runner.invoke(app, ['alerts'])
            assert result.exit_code == 0
            assert "Recommendations:" in result.stdout

    def test_alerts_error_handling(self):
        """Test alerts command error handling."""
        with patch('prompt_improver.cli.asyncio.run', side_effect=Exception("Monitoring service unavailable")):
            result = self.runner.invoke(app, ['alerts'])
            assert result.exit_code == 1
            assert "Failed to retrieve alerts" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
