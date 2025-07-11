#!/usr/bin/env python3
"""
Phase 4 Regression Tests for Complex CLI Functions.
These tests ensure that refactored functions maintain the same behavior.
Enhanced with comprehensive stdio testing following pytest best practices.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from io import StringIO
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
        self.log_dir = (
            Path(self.temp_dir) / ".local" / "share" / "apes" / "data" / "logs"
        )
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

        with open(self.log_file, "w", encoding="utf-8") as f:
            for log in sample_logs:
                f.write(log + "\n")

    def test_logs_basic_functionality(self):
        """Test basic logs command without follow mode."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ["logs", "--lines", "10"])

            # Should find and display logs
            assert result.exit_code == 0
            assert "Viewing logs:" in result.stdout

    def test_logs_level_filtering(self):
        """Test log level filtering functionality."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Test INFO level filtering
            result = self.runner.invoke(
                app, ["logs", "--level", "INFO", "--lines", "50"]
            )
            assert result.exit_code == 0

            # Test ERROR level filtering
            result = self.runner.invoke(
                app, ["logs", "--level", "ERROR", "--lines", "50"]
            )
            assert result.exit_code == 0

    def test_logs_component_filtering(self):
        """Test component-specific log filtering."""
        # Create component-specific log file
        component_log = self.log_dir / "mcp.log"
        with open(component_log, "w", encoding="utf-8") as f:
            f.write("2024-01-15 10:00:00 INFO MCP service started\n")

        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ["logs", "--component", "mcp"])
            assert result.exit_code == 0
            assert "Viewing logs:" in result.stdout

    def test_logs_nonexistent_directory(self):
        """Test behavior when log directory doesn't exist."""
        with patch("pathlib.Path.home", return_value=Path("/nonexistent")):
            result = self.runner.invoke(app, ["logs"])
            assert result.exit_code == 1
            assert "Log directory not found" in result.stdout

    def test_logs_nonexistent_component(self):
        """Test behavior when component log file doesn't exist."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ["logs", "--component", "nonexistent"])
            assert result.exit_code == 1
            assert "Log file not found" in result.stdout
            assert "Available log files:" in result.stdout

    def test_logs_lines_parameter(self):
        """Test lines parameter functionality."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Test with small number of lines
            result = self.runner.invoke(app, ["logs", "--lines", "5"])
            assert result.exit_code == 0

            # Test with large number of lines (more than available)
            result = self.runner.invoke(app, ["logs", "--lines", "1000"])
            assert result.exit_code == 0

    # =========================
    # Enhanced stdio Testing
    # =========================

    def test_logs_stdio_comprehensive(self, capsys):
        """Test logs command with comprehensive stdio handling."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Test with real stdio capture
            result = self.runner.invoke(app, ["logs", "--lines", "10"])
            
            # Capture any additional stdio output
            captured = capsys.readouterr()
            
            # Verify CLI runner output
            assert result.exit_code == 0
            assert "Viewing logs:" in result.stdout
            
            # Verify no stderr leakage
            assert captured.err == ""

    def test_logs_with_system_output(self, capfd):
        """Test logs with system-level output capture."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Test with file descriptor level capture for system calls
            result = self.runner.invoke(app, ["logs", "--lines", "10"])
            captured = capfd.readouterr()
            
            assert result.exit_code == 0
            # Verify system-level output handling
            assert "Viewing logs:" in result.stdout

    def test_logs_large_output_streaming(self, capsys):
        """Test logs command with large output streams."""
        # Create large log file
        large_logs = ["2024-01-15 10:00:00 INFO Large log entry"] * 1000
        with open(self.log_file, "w", encoding="utf-8") as f:
            for log in large_logs:
                f.write(log + "\n")
        
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ["logs", "--lines", "100"])
            
            # Test streaming behavior
            assert result.exit_code == 0
            assert "Viewing logs:" in result.stdout
            # Verify reasonable output size
            assert len(result.stdout.split('\n')) <= 105  # lines + headers

    def test_logs_with_interactive_input(self, monkeypatch):
        """Test logs command with simulated interactive input."""
        # Simulate user input for follow mode
        monkeypatch.setattr('sys.stdin', StringIO('q\n'))
        
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Test without follow mode first (safer)
            result = self.runner.invoke(app, ["logs", "--lines", "10"])
            
            # Should handle interactive scenarios gracefully
            assert result.exit_code == 0

    def test_logs_binary_output_handling(self, capsys):
        """Test logs with binary output handling."""
        # Create log with mixed binary content
        with open(self.log_file, "wb") as f:
            f.write(b"2024-01-15 10:00:00 INFO Binary log entry\n")
            f.write(b"\x00\x01\x02 Non-UTF8 content\n")
            f.write(b"2024-01-15 10:00:01 INFO Normal log entry\n")
        
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ["logs", "--lines", "10"])
            
            # Should handle binary content gracefully
            assert result.exit_code == 0
            # Should contain readable content
            assert "Binary log entry" in result.stdout or "Normal log entry" in result.stdout

    @pytest.mark.parametrize("line_count", [10, 100, 500])
    def test_logs_performance_with_output_capture(self, line_count, capsys):
        """Test logs performance with various output sizes."""
        # Create logs of varying sizes
        logs = [f"2024-01-15 10:00:00 INFO Log entry {i}" for i in range(line_count)]
        with open(self.log_file, "w", encoding="utf-8") as f:
            for log in logs:
                f.write(log + "\n")
        
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            start_time = time.time()
            result = self.runner.invoke(app, ["logs", "--lines", str(min(line_count, 100))])
            end_time = time.time()
            
            # Performance assertions
            assert result.exit_code == 0
            assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
            
            # Verify output completeness
            assert "Viewing logs:" in result.stdout
            # Should have reasonable amount of content
            output_lines = result.stdout.count('\n')
            assert output_lines >= 5  # At least some content

    def test_logs_with_output_redirection(self, tmp_path, capsys):
        """Test logs command with output redirection simulation."""
        output_file = tmp_path / "logs_output.txt"
        
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Simulate output redirection by capturing and writing
            with capsys.disabled():
                result = self.runner.invoke(app, ["logs", "--lines", "10"])
                
                # Write to file as if redirected
                output_file.write_text(result.stdout)
            
            # Verify redirection worked
            assert result.exit_code == 0
            assert output_file.exists()
            content = output_file.read_text()
            assert "Viewing logs:" in content

    def test_logs_stderr_handling(self, capsys):
        """Test logs command stderr handling."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Test error condition that might produce stderr
            result = self.runner.invoke(app, ["logs", "--component", "nonexistent"])
            captured = capsys.readouterr()
            
            # Should handle errors gracefully
            assert result.exit_code == 1
            assert "Log file not found" in result.stdout
            # Verify stderr handling
            assert captured.err == ""  # No stderr leakage

    def test_logs_output_formatting(self, capsys):
        """Test logs output formatting and structure."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            result = self.runner.invoke(app, ["logs", "--lines", "5"])
            
            assert result.exit_code == 0
            
            # Verify output structure
            output_lines = result.stdout.split('\n')
            # Should have header and content
            assert len(output_lines) >= 2
            
            # Check for expected formatting patterns
            assert any("Viewing logs:" in line for line in output_lines)

    def test_logs_concurrent_output(self, capsys):
        """Test logs command with concurrent stdout/stderr scenarios."""
        with patch("pathlib.Path.home", return_value=Path(self.temp_dir)):
            # Test multiple rapid invocations
            results = []
            for i in range(3):
                result = self.runner.invoke(app, ["logs", "--lines", "5"])
                results.append(result)
            
            # All should succeed
            for result in results:
                assert result.exit_code == 0
                assert "Viewing logs:" in result.stdout
            
            # Verify no output interference
            captured = capsys.readouterr()
            assert captured.err == ""


class TestHealthRegression:
    """Regression tests for the health function (complexity 14)."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()

    @pytest.mark.asyncio
    async def test_health_basic_functionality(self):
        """Test basic health check functionality."""
        # Mock the health monitor to avoid actual system checks
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_health_results = {
                "overall_status": "healthy",
                "checks": {
                    "database": {
                        "status": "healthy",
                        "response_time_ms": 50.0,
                        "message": "Connected",
                    },
                    "mcp": {
                        "status": "healthy",
                        "response_time_ms": 25.0,
                        "message": "Running",
                    },
                    "system_resources": {
                        "status": "healthy",
                        "memory_usage_percent": 45.0,
                        "cpu_usage_percent": 30.0,
                    },
                },
                "warning_checks": [],
                "failed_checks": [],
            }
            mock_run.return_value = mock_health_results

            result = self.runner.invoke(app, ["health"])
            assert result.exit_code == 0
            assert "Running APES Health Check" in result.stdout

    def test_health_json_output(self):
        """Test health check with JSON output format."""
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_health_results = {
                "overall_status": "healthy",
                "checks": {},
                "warning_checks": [],
                "failed_checks": [],
            }
            mock_run.return_value = mock_health_results

            result = self.runner.invoke(app, ["health", "--json"])
            assert result.exit_code == 0

    def test_health_detailed_mode(self):
        """Test health check with detailed diagnostics."""
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_health_results = {
                "overall_status": "warning",
                "checks": {
                    "system_resources": {
                        "status": "warning",
                        "memory_usage_percent": 85.0,
                        "cpu_usage_percent": 75.0,
                        "disk_usage_percent": 60.0,
                    }
                },
                "warning_checks": ["system_resources"],
                "failed_checks": [],
            }
            mock_run.return_value = mock_health_results

            result = self.runner.invoke(app, ["health", "--detailed"])
            assert result.exit_code == 0

    def test_health_error_handling(self):
        """Test health check error handling."""
        with patch(
            "prompt_improver.cli.asyncio.run",
            side_effect=Exception("Health check failed"),
        ):
            result = self.runner.invoke(app, ["health"])
            assert result.exit_code == 1
            assert "Health check failed" in result.stdout


class TestAlertsRegression:
    """Regression tests for the alerts function (complexity 13)."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()

    def test_alerts_basic_functionality(self):
        """Test basic alerts command functionality."""
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_summary = {
                "alert_summary": {
                    "total_alerts": 5,
                    "critical_alerts": 2,
                    "warning_alerts": 3,
                    "most_common_alert": "High Memory Usage",
                },
                "current_performance": {
                    "avg_response_time_ms": 150.0,
                    "memory_usage_mb": 180.0,
                    "database_connections": 10,
                },
                "health_status": "warning",
            }
            mock_run.return_value = mock_summary

            result = self.runner.invoke(app, ["alerts"])
            assert result.exit_code == 0
            assert "APES Alert Status" in result.stdout

    def test_alerts_severity_filtering(self):
        """Test alerts with severity filtering."""
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_summary = {
                "alert_summary": {
                    "total_alerts": 3,
                    "critical_alerts": 1,
                    "warning_alerts": 2,
                    "most_common_alert": "Database Connection Issue",
                },
                "current_performance": {},
                "health_status": "critical",
            }
            mock_run.return_value = mock_summary

            # Test critical severity filter
            result = self.runner.invoke(app, ["alerts", "--severity", "critical"])
            assert result.exit_code == 0

            # Test warning severity filter
            result = self.runner.invoke(app, ["alerts", "--severity", "warning"])
            assert result.exit_code == 0

    def test_alerts_time_period(self):
        """Test alerts with different time periods."""
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_summary = {
                "alert_summary": {"total_alerts": 0},
                "current_performance": {},
                "health_status": "healthy",
            }
            mock_run.return_value = mock_summary

            # Test with custom hours parameter
            result = self.runner.invoke(app, ["alerts", "--hours", "48"])
            assert result.exit_code == 0

    def test_alerts_no_alerts_found(self):
        """Test alerts command when no alerts are found."""
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_summary = {
                "alert_summary": {"total_alerts": 0},
                "current_performance": {},
                "health_status": "healthy",
            }
            mock_run.return_value = mock_summary

            result = self.runner.invoke(app, ["alerts"])
            assert result.exit_code == 0
            assert "No alerts found" in result.stdout

    def test_alerts_recommendations(self):
        """Test alerts command shows recommendations for unhealthy systems."""
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_summary = {
                "alert_summary": {
                    "total_alerts": 1,
                    "critical_alerts": 1,
                    "warning_alerts": 0,
                },
                "current_performance": {
                    "avg_response_time_ms": 250.0,  # > 200
                    "memory_usage_mb": 220.0,  # > 200
                    "database_connections": 18,  # > 15
                },
                "health_status": "critical",
            }
            mock_run.return_value = mock_summary

            result = self.runner.invoke(app, ["alerts"])
            assert result.exit_code == 0
            assert "Recommendations:" in result.stdout

    def test_alerts_error_handling(self):
        """Test alerts command error handling."""
        with patch(
            "prompt_improver.cli.asyncio.run",
            side_effect=Exception("Monitoring service unavailable"),
        ):
            result = self.runner.invoke(app, ["alerts"])
            assert result.exit_code == 1
            assert "Failed to retrieve alerts" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
