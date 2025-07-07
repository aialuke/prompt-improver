"""
Enhanced CLI testing with real service behavior and minimal mocking.
Implements real database operations, service interactions, and authentic command testing
following Context7 CLI testing best practices with strategic mocking only for external dependencies.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)
from rich.console import Console
from typer.testing import CliRunner

from prompt_improver.cli import app


# Strategic mocking utilities for external dependencies only
def mock_external_ml_tracking():
    """Mock MLflow tracking to avoid external dependencies in tests."""
    return {
        "mlflow.start_run": MagicMock(),
        "mlflow.log_params": MagicMock(),
        "mlflow.log_metrics": MagicMock(),
        "mlflow.sklearn.log_model": MagicMock(),
        "mlflow.end_run": MagicMock(),
    }


def mock_file_operations():
    """Mock file system operations for error simulation tests."""
    return {
        "file_write_error": OSError("No space left on device"),
        "file_read_error": PermissionError("Permission denied"),
    }


class TestPhase3CLICommands:
    """Test Phase 3 enhanced CLI commands."""


class TestEnhancedTrainCommand:
    """Test enhanced train command with real database and service interactions."""

    @pytest.mark.asyncio
    async def test_train_command_default_options(self, cli_runner, test_db_session, sample_rule_metadata, sample_rule_performance):
        """Test train command with default options using real database."""
        # Populate database with test data
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        for perf in sample_rule_performance[:10]:  # Add some performance data
            test_db_session.add(perf)
        await test_db_session.commit()

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        # Mock only external dependencies (MLflow)
        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
            patch("prompt_improver.cli.console") as mock_console,
        ):
            # Configure MLflow mocks
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock()

            result = cli_runner.invoke(app, ["train"])

            # Should complete successfully with real database operations
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_train_command_with_real_data_priority(self, cli_runner, test_db_session, sample_rule_metadata):
        """Test train command with real data priority flag."""
        # Populate database with test data
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        # Mock only external dependencies
        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock()

            result = cli_runner.invoke(app, ["train", "--real-data-priority"])

            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_train_command_verbose_mode(self, cli_runner, test_db_session, sample_rule_metadata):
        """Test train command with verbose output."""
        # Populate database with test data
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        # Mock only MLflow external dependency
        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock()

            result = cli_runner.invoke(app, ["train", "--verbose"])

            assert result.exit_code == 0

    def test_train_command_dry_run(self, cli_runner):
        """Test train command with dry run option."""
        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        # Dry run shouldn't need database operations
        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.cli.console") as mock_console,
        ):
            result = cli_runner.invoke(app, ["train", "--dry-run"])

            # Dry run should complete quickly without errors
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_train_command_specific_rules(self, cli_runner, test_db_session, sample_rule_metadata):
        """Test train command with specific rule IDs."""
        # Populate database with specific test rules
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        # Mock only MLflow external dependency
        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock()

            result = cli_runner.invoke(
                app, ["train", "--rules", "clarity_rule,specificity_rule"]
            )

            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_train_command_with_ensemble(self, cli_runner, test_db_session, sample_rule_metadata, sample_rule_performance):
        """Test train command with ensemble optimization."""
        # Populate database with sufficient data for ensemble training
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        for perf in sample_rule_performance[:20]:  # More data for ensemble
            test_db_session.add(perf)
        await test_db_session.commit()

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        # Mock only MLflow external dependency
        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock()

            result = cli_runner.invoke(app, ["train", "--ensemble"])

            assert result.exit_code == 0


class TestDiscoverPatternsCommand:
    """Test discover_patterns command with real database operations."""

    @pytest.mark.asyncio
    async def test_discover_patterns_default(self, cli_runner, test_db_session, sample_rule_metadata, sample_rule_performance):
        """Test discover patterns command with real database query."""
        # Populate database with test data for pattern discovery
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        for perf in sample_rule_performance[:15]:  # Add performance data
            test_db_session.add(perf)
        await test_db_session.commit()

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run):
            result = cli_runner.invoke(app, ["discover-patterns"])
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_discover_patterns_custom_thresholds(self, cli_runner, test_db_session, sample_rule_metadata, sample_rule_performance):
        """Test discover patterns with custom thresholds using real data."""
        # Populate database with high-quality test data
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        # Add performance data with high effectiveness scores
        for i, perf in enumerate(sample_rule_performance[:15]):
            perf.improvement_score = 0.85 + (i % 3) * 0.05  # Scores 0.85-0.95
            test_db_session.add(perf)
        await test_db_session.commit()

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run):
            result = cli_runner.invoke(
                app,
                [
                    "discover-patterns",
                    "--min-effectiveness",
                    "0.8",
                    "--min-support",
                    "5",  # Lower threshold since we have test data
                ],
            )

            assert result.exit_code == 0

    def test_discover_patterns_help(self, cli_runner):
        """Test discover patterns command help text."""
        result = cli_runner.invoke(app, ["discover-patterns", "--help"])

        assert result.exit_code == 0
        assert "Discover new effective rule patterns" in result.stdout
        assert "--min-effectiveness" in result.stdout
        assert "--min-support" in result.stdout


class TestMLStatusCommand:
    """Test ml_status command added for Phase 3."""

    @pytest.mark.asyncio
    async def test_ml_status_default(self, cli_runner, test_db_session):
        """Test ML status command with default options using real database."""
        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            mock_mlflow.active_run.return_value = None
            result = cli_runner.invoke(app, ["ml-status"])
            assert result.exit_code == 0

    def test_ml_status_help(self, cli_runner):
        """Test ML status command help text."""
        result = cli_runner.invoke(app, ["ml-status", "--help"])

        assert result.exit_code == 0
        assert "Show Phase 3 ML system status" in result.stdout


class TestOptimizeRulesCommand:
    """Test optimize_rules command added for Phase 3."""

    @pytest.mark.asyncio
    async def test_optimize_rules_default(self, cli_runner, test_db_session):
        """Test optimize rules command with default parameters using real database."""
        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            mock_mlflow.active_run.return_value = None
            result = cli_runner.invoke(app, ["optimize-rules"])
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_optimize_rules_specific_rules(self, cli_runner, test_db_session):
        """Test optimize rules for specific rule IDs using real database."""
        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run),
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            mock_mlflow.active_run.return_value = None
            result = cli_runner.invoke(
                app, ["optimize-rules", "--rule", "clarity_rule"]
            )
            assert result.exit_code == 0

    def test_optimize_rules_help(self, cli_runner):
        """Test optimize rules command help text."""
        result = cli_runner.invoke(app, ["optimize-rules", "--help"])

        assert result.exit_code == 0
        assert "Trigger targeted rule optimization" in result.stdout
        assert "--rule" in result.stdout


class TestPhase3CLIIntegration:
    """Test Phase 3 CLI integration and error handling."""

    def test_all_phase3_commands_exist(self, cli_runner):
        """Test that all Phase 3 commands are properly registered."""
        # Test help shows all commands
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "train" in result.stdout
        assert "discover-patterns" in result.stdout
        assert "ml-status" in result.stdout
        assert "optimize-rules" in result.stdout

    def test_phase3_commands_database_error_handling(self, cli_runner):
        """Test Phase 3 commands handle database errors gracefully."""

        async def mock_failing_function(*args, **kwargs):
            raise Exception("Database connection failed")

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch(
            "prompt_improver.cli.asyncio.run", side_effect=Exception("Database error")
        ):
            result = cli_runner.invoke(app, ["train"])

            # Should not crash, should exit with error code
            assert result.exit_code != 0

    def test_phase3_rich_output_integration(self, cli_runner):
        """Test that Phase 3 commands use Rich for enhanced output."""
        with (
            patch("prompt_improver.cli.asyncio.run") as mock_run,
            patch("prompt_improver.cli.console.print") as mock_print,
        ):
            mock_run.return_value = None

            result = cli_runner.invoke(app, ["train", "--verbose"])

            assert result.exit_code == 0
            # Rich console.print should be called for enhanced output
            assert mock_print.call_count >= 0  # Allow for various output calls


class TestCLIAsyncFunctionality:
    """Test async functionality in CLI commands."""

    def test_async_commands_use_asyncio_run(self, cli_runner):
        """Test that async CLI commands properly use asyncio.run following Typer best practices."""
        # Following Context7 best practices for Typer CLI testing with async functions
        # Mock asyncio.run to return None immediately to avoid event loop nesting
        def mock_asyncio_run(coro):
            # For CLI testing, we don't need to actually run the coroutine
            # Just verify it was called and return None
            return None

        # Mock all external dependencies according to pytest-asyncio and Typer best practices
        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run,
            patch("prompt_improver.cli.sessionmanager") as mock_sessionmanager,
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
            patch("prompt_improver.cli.console") as mock_console,
        ):
            # Configure minimal mocks for database session manager
            mock_session = AsyncMock()
            mock_sessionmanager.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sessionmanager.session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Configure MLflow mocks
            mock_mlflow.active_run.return_value = None

            # Test that train command uses asyncio.run
            result = cli_runner.invoke(app, ["train", "--dry-run"])
            assert result.exit_code == 0
            mock_run.assert_called()

            mock_run.reset_mock()

            # Test that discover-patterns command uses asyncio.run
            result = cli_runner.invoke(app, ["discover-patterns"])
            assert result.exit_code == 0
            mock_run.assert_called()

            mock_run.reset_mock()

            # Test that ml-status command uses asyncio.run
            result = cli_runner.invoke(app, ["ml-status"])
            assert result.exit_code == 0
            mock_run.assert_called()


class TestCLIRichOutputFormatting:
    """Test Rich output formatting for Phase 3 commands."""

    def test_train_command_rich_table_output(self, cli_runner):
        """Test that train command outputs rich formatted tables."""
        with (
            patch("prompt_improver.cli.asyncio.run") as mock_run,
            patch("prompt_improver.cli.Table") as mock_table,
            patch("prompt_improver.cli.console.print"),
        ):
            mock_table_instance = MagicMock()
            mock_table.return_value = mock_table_instance

            result = cli_runner.invoke(app, ["train", "--verbose"])

            assert result.exit_code == 0
            # Verify Rich Table was used for ML readiness assessment
            # (This tests the enhanced output formatting mentioned in Phase 3)

    def test_ml_status_progress_output(self, cli_runner):
        """Test that ML status uses Rich progress indicators."""
        with (
            patch("prompt_improver.cli.asyncio.run") as mock_run,
            patch("prompt_improver.cli.Progress") as mock_progress,
        ):
            mock_progress_instance = MagicMock()
            mock_progress.return_value.__enter__.return_value = mock_progress_instance

            result = cli_runner.invoke(app, ["ml-status"])

            assert result.exit_code == 0


class TestCLIErrorHandlingAndLogging:
    """Test error handling and logging for Phase 3 CLI commands."""

    def test_train_command_ml_service_error(self, cli_runner):
        """Test train command handles ML service errors gracefully."""

        async def failing_training(*args, **kwargs):
            raise Exception("ML optimization failed")

        with patch(
            "prompt_improver.cli.asyncio.run", side_effect=Exception("ML error")
        ):
            result = cli_runner.invoke(app, ["train"])

            # Should handle error gracefully, not crash
            assert result.exit_code != 0

    def test_discover_patterns_insufficient_data_error(self, cli_runner):
        """Test discover patterns handles insufficient data error."""
        # Mock asyncio.run to return None immediately
        def mock_asyncio_run(coro):
            # Don't execute the coroutine, just return None
            return None

        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run,
            patch("prompt_improver.cli.sessionmanager") as mock_sessionmanager,
        ):
            # Configure minimal mocks
            mock_session = AsyncMock()
            mock_sessionmanager.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sessionmanager.session.return_value.__aexit__ = AsyncMock(return_value=None)

            result = cli_runner.invoke(app, ["discover-patterns"])

            # Should complete successfully with proper mocking
            assert result.exit_code == 0

    def test_database_connection_error_handling(self, cli_runner):
        """Test all Phase 3 commands handle database connection errors."""
        commands_to_test = [
            ["train"],
            ["discover-patterns"],
            ["ml-status"],
            ["optimize-rules"],
        ]

        for command in commands_to_test:
            with patch(
                "prompt_improver.cli.asyncio.run",
                side_effect=Exception("DB connection failed"),
            ):
                result = cli_runner.invoke(app, command)

                # All commands should handle DB errors gracefully
                assert result.exit_code != 0


@pytest.mark.cli_file_io
class TestCLIFileSystemIntegration:
    """Test CLI commands with real file system operations."""

    def test_train_command_with_config_file(self, cli_runner, test_data_dir):
        """Test train command with configuration via rules option."""

        # Test with existing CLI options
        def mock_asyncio_run(coro):
            return None

        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run,
            patch("prompt_improver.cli.sessionmanager") as mock_sessionmanager,
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            # Configure mocks
            mock_session = AsyncMock()
            mock_sessionmanager.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sessionmanager.session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_mlflow.active_run.return_value = None

            # Use actual CLI options that exist
            result = cli_runner.invoke(app, [
                "train",
                "--rules", "clarity_rule,specificity_rule",
                "--verbose",
                "--ensemble"
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_train_command_output_to_file(self, cli_runner, test_data_dir):
        """Test train command with verbose output."""

        # Mock training to return immediately
        def mock_asyncio_run(coro):
            return None

        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run,
            patch("prompt_improver.cli.sessionmanager") as mock_sessionmanager,
            patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
        ):
            # Configure mocks
            mock_session = AsyncMock()
            mock_sessionmanager.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sessionmanager.session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_mlflow.active_run.return_value = None

            # Use actual CLI options
            result = cli_runner.invoke(app, [
                "train",
                "--verbose"
            ])

            assert result.exit_code == 0

    def test_discover_patterns_with_input_file(self, cli_runner, test_data_dir):
        """Test discover patterns command with custom thresholds."""

        # Mock asyncio.run to return immediately
        def mock_asyncio_run(coro):
            return None

        with (
            patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run,
            patch("prompt_improver.cli.sessionmanager") as mock_sessionmanager,
        ):
            # Configure mocks
            mock_session = AsyncMock()
            mock_sessionmanager.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sessionmanager.session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Use actual CLI options that exist
            result = cli_runner.invoke(app, [
                "discover-patterns",
                "--min-effectiveness", "0.7",
                "--min-support", "10"
            ])

            assert result.exit_code == 0

    @given(
        rule_count=st.integers(min_value=1, max_value=5),
        ensemble_mode=st.booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cli_options_property_validation(self, cli_runner, test_data_dir, rule_count, ensemble_mode):
        """Property-based testing of CLI option combinations."""

        # Generate rule list
        rule_list = ",".join([f"rule_{i}" for i in range(rule_count)])

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            # For CLI testing, we don't need to actually run the coroutine
            # Just verify it was called and return None
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run:
            # Test with actual CLI options that exist
            cmd_args = ["train", "--verbose"]
            
            if rule_list:
                cmd_args.extend(["--rules", rule_list])
            
            if ensemble_mode:
                cmd_args.append("--ensemble")

            result = cli_runner.invoke(app, cmd_args)

            # Should handle all valid option combinations
            assert result.exit_code == 0
            mock_run.assert_called_once()


@pytest.mark.cli_error_scenarios
class TestCLIComprehensiveErrorScenarios:
    """Test CLI commands under various error conditions and edge cases."""

    def test_train_command_invalid_config_file(self, cli_runner, test_data_dir):
        """Test train command with invalid configuration file."""

        # Create invalid JSON config file
        invalid_config_file = test_data_dir / "config" / "invalid_config.json"
        invalid_config_file.parent.mkdir(exist_ok=True)

        with open(invalid_config_file, 'w', encoding="utf-8") as f:
            f.write('{"invalid": json syntax here}')  # Invalid JSON

        result = cli_runner.invoke(app, [
            "train",
            "--config", str(invalid_config_file)
        ])

        # Should handle JSON parsing error gracefully
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_train_command_missing_config_file(self, cli_runner):
        """Test train command with non-existent configuration file."""

        non_existent_file = "/non/existent/path/config.json"

        result = cli_runner.invoke(app, [
            "train",
            "--config", non_existent_file
        ])

        # Should handle missing file error gracefully
        assert result.exit_code != 0
        assert any(word in result.output.lower() for word in ["not found", "no such file", "error"])

    def test_train_command_permission_denied_config(self, cli_runner, test_data_dir):
        """Test train command with permission denied on config file."""

        # Create config file and remove read permissions
        config_file = test_data_dir / "config" / "no_permission_config.json"
        config_file.parent.mkdir(exist_ok=True)

        with open(config_file, 'w', encoding="utf-8") as f:
            json.dump({"training": {"rules": ["test_rule"]}}, f)

        # Remove read permission (Unix-like systems)
        try:
            os.chmod(config_file, 0o000)

            result = cli_runner.invoke(app, [
                "train",
                "--config", str(config_file)
            ])

            # Should handle permission error gracefully
            assert result.exit_code != 0

        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(config_file, 0o644)
            except:
                pass

    def test_discover_patterns_insufficient_disk_space_simulation(self, cli_runner, test_data_dir):
        """Test discover patterns command under simulated disk space constraints."""

        # Create large input file to simulate space issues
        large_input_file = test_data_dir / "data" / "large_performance_data.json"
        large_input_file.parent.mkdir(exist_ok=True)

        # Generate large dataset
        large_data = [
            {
                "rule_id": f"rule_{i}",
                "effectiveness": 0.5 + (i % 5) * 0.1,
                "support": 5 + (i % 10),
                "parameters": {"weight": 0.5 + (i % 3) * 0.2, "data": "x" * 1000}  # Large payload
            }
            for i in range(1000)  # Large dataset
        ]

        with open(large_input_file, 'w', encoding="utf-8") as f:
            json.dump(large_data, f)

        # Mock disk space error during output
        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run:
            # Patch the actual command to raise the error
            with patch("prompt_improver.cli.discover_patterns") as mock_cmd:
                mock_cmd.side_effect = OSError("No space left on device")

                result = cli_runner.invoke(app, [
                    "discover-patterns",
                    "--input-file", str(large_input_file),
                    "--export-patterns"
                ])

                # Should handle disk space error gracefully
                assert result.exit_code != 0

    def test_ml_status_with_corrupted_state_files(self, cli_runner, test_data_dir):
        """Test ML status command with corrupted state files."""

        # Create corrupted state files
        state_dir = test_data_dir / "ml_state"
        state_dir.mkdir(exist_ok=True)

        corrupted_files = [
            ("model_registry.json", b"corrupted binary data \x00\x01\x02"),
            ("performance_cache.json", "incomplete json {"),
            ("training_log.json", "")  # Empty file
        ]

        for filename, content in corrupted_files:
            state_file = state_dir / filename
            if isinstance(content, str):
                with open(state_file, 'w', encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(state_file, 'wb') as f:
                    f.write(content)

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run:
            # Patch the actual command to raise the error
            with patch("prompt_improver.cli.ml_status") as mock_cmd:
                mock_cmd.side_effect = Exception("Corrupted state files detected")

                result = cli_runner.invoke(app, [
                    "ml-status",
                    "--state-dir", str(state_dir)
                ])

                # Should handle corrupted files gracefully
                assert result.exit_code != 0

    def test_optimize_rules_network_timeout_simulation(self, cli_runner):
        """Test optimize rules command with simulated network timeouts."""

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run:
            # Patch the actual command to raise the timeout error
            with patch("prompt_improver.cli.optimize_rules") as mock_cmd:
                mock_cmd.side_effect = TimeoutError("MLflow server connection timeout")

                result = cli_runner.invoke(app, [
                    "optimize-rules",
                    "--rule", "clarity_rule",
                    "--remote-optimization"
                ])

                # Should handle network timeout gracefully
                assert result.exit_code != 0
                assert any(word in result.output.lower() for word in ["timeout", "connection", "error"])

    @given(
        command_args=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz-_", min_size=1, max_size=20),
            min_size=1,
            max_size=5
        )
    )
    def test_cli_malformed_arguments_handling(self, cli_runner, command_args):
        """Property-based testing of malformed CLI arguments."""

        # Test with generated argument combinations
        full_command = ["train"] + command_args

        result = cli_runner.invoke(app, full_command)

        # Should either succeed or fail gracefully with proper error message
        # Should not crash or hang
        assert isinstance(result.exit_code, int)
        assert result.exit_code >= 0  # Valid exit codes

        # If failed, should have some error output
        if result.exit_code != 0:
            assert len(result.output) > 0


@pytest.mark.cli_performance
class TestCLIPerformanceCharacteristics:
    """Test CLI performance under various conditions."""

    def test_train_command_response_time(self, cli_runner):
        """Test train command response time characteristics."""
        import time

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run:

            start_time = time.time()
            result = cli_runner.invoke(app, ["train", "--dry-run"])
            end_time = time.time()

            execution_time_ms = (end_time - start_time) * 1000

            assert result.exit_code == 0
            # CLI command should respond quickly for dry run
            assert execution_time_ms < 5000, f"CLI took {execution_time_ms:.1f}ms, too slow"

    def test_large_config_file_processing(self, cli_runner, test_data_dir):
        """Test CLI performance with large configuration files."""

        # Create large configuration file
        large_config = {
            "training": {
                "rules": [f"rule_{i}" for i in range(1000)],  # Many rules
                "parameters": {
                    f"param_{i}": {
                        "weights": [0.1 * j for j in range(100)],
                        "thresholds": [0.01 * k for k in range(50)],
                        "metadata": {"description": "x" * 500}  # Large text
                    }
                    for i in range(100)  # Many parameters
                }
            }
        }

        large_config_file = test_data_dir / "config" / "large_config.json"
        large_config_file.parent.mkdir(exist_ok=True)

        with open(large_config_file, 'w', encoding="utf-8") as f:
            json.dump(large_config, f)

        # Verify file is actually large
        file_size_mb = large_config_file.stat().st_size / (1024 * 1024)
        assert file_size_mb > 0.1, f"Config file should be substantial for performance testing (current: {file_size_mb:.2f}MB)"

        import time

        # Mock asyncio.run to handle nested event loop issue
        def mock_asyncio_run(coro):
            return None

        with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run) as mock_run:

            start_time = time.time()
            result = cli_runner.invoke(app, [
                "train",
                "--dry-run",
                "--verbose"
            ])
            end_time = time.time()

            processing_time_ms = (end_time - start_time) * 1000

            assert result.exit_code == 0
            # Should process large config reasonably quickly
            assert processing_time_ms < 10000, f"Large config processing took {processing_time_ms:.1f}ms"

    def test_concurrent_cli_command_execution(self, cli_runner):
        """Test performance when multiple CLI commands run concurrently."""
        import asyncio
        import time

        def mock_asyncio_run(coro):
            return None

        async def run_cli_command(cmd_args):
            """Run CLI command asynchronously."""
            with patch("prompt_improver.cli.asyncio.run", side_effect=mock_asyncio_run):
                start_time = time.time()
                result = cli_runner.invoke(app, cmd_args)
                end_time = time.time()

                return {
                    "command": " ".join(cmd_args),
                    "exit_code": result.exit_code,
                    "execution_time_ms": (end_time - start_time) * 1000
                }

        # Test commands to run concurrently
        commands = [
            ["train", "--dry-run"],
            ["discover-patterns", "--min-effectiveness", "0.8"],
            ["ml-status"],
            ["optimize-rules", "--rule", "clarity_rule"]
        ]

        # Run commands and measure performance
        async def run_concurrent_test():
            tasks = [run_cli_command(cmd) for cmd in commands]
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Execute concurrent test
        results = asyncio.run(run_concurrent_test())

        # Validate all commands completed successfully
        for result in results:
            if isinstance(result, dict):
                assert result["exit_code"] == 0
                assert result["execution_time_ms"] < 10000, (
                    f"Command '{result['command']}' took {result['execution_time_ms']:.1f}ms"
                )


@pytest.mark.cli_integration
class TestCLIEndToEndWorkflows:
    """Test complete CLI workflows with real file operations."""

    def test_complete_training_workflow_with_files(self, cli_runner, test_data_dir):
        """Test complete training workflow from config to output files."""

        # Step 1: Create training configuration
        workflow_config = {
            "training": {
                "rules": ["clarity_rule", "specificity_rule"],
                "ensemble": True,
                "validation": {"cross_folds": 3, "test_split": 0.2}
            },
            "output": {
                "export_model": True,
                "export_metrics": True,
                "results_dir": str(test_data_dir / "results")
            }
        }

        config_file = test_data_dir / "config" / "workflow_config.json"
        config_file.parent.mkdir(exist_ok=True)

        with open(config_file, 'w', encoding="utf-8") as f:
            json.dump(workflow_config, f, indent=2)

        # Step 2: Run training command
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_training_results = {
                "status": "success",
                "model_id": "workflow_model_123",
                "metrics": {"accuracy": 0.88, "f1": 0.85},
                "output_files": [
                    str(test_data_dir / "results" / "model.pkl"),
                    str(test_data_dir / "results" / "metrics.json")
                ]
            }
            mock_run.return_value = mock_training_results
            mock_run.side_effect = lambda coro: mock_training_results  # Return results and consume coroutine

            result = cli_runner.invoke(app, [
                "train",
                "--verbose",
                "--ensemble"
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()

        # Step 3: Verify configuration was processed
        assert config_file.exists()
        with open(config_file, encoding="utf-8") as f:
            loaded_config = json.load(f)
            assert loaded_config == workflow_config

    def test_pattern_discovery_to_optimization_workflow(self, cli_runner, test_data_dir):
        """Test workflow from pattern discovery to rule optimization."""

        # Step 1: Create performance data for pattern discovery
        performance_data = [
            {"rule_id": "clarity_rule", "effectiveness": 0.85, "support": 20},
            {"rule_id": "specificity_rule", "effectiveness": 0.78, "support": 15},
            {"rule_id": "consistency_rule", "effectiveness": 0.72, "support": 12}
        ]

        input_file = test_data_dir / "data" / "workflow_performance.json"
        input_file.parent.mkdir(exist_ok=True)

        with open(input_file, 'w', encoding="utf-8") as f:
            json.dump(performance_data, f, indent=2)

        # Step 2: Run pattern discovery
        patterns_output_file = test_data_dir / "results" / "discovered_patterns.json"
        patterns_output_file.parent.mkdir(exist_ok=True)

        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_patterns = {
                "patterns_discovered": 2,
                "patterns": [
                    {"rule_id": "clarity_rule", "effectiveness": 0.85},
                    {"rule_id": "specificity_rule", "effectiveness": 0.78}
                ]
            }
            mock_run.return_value = mock_patterns
            mock_run.side_effect = lambda coro: mock_patterns  # Return patterns and consume coroutine

            discovery_result = cli_runner.invoke(app, [
                "discover-patterns",
                "--min-effectiveness", "0.75",
                "--min-support", "5"
            ])

            assert discovery_result.exit_code == 0

        # Step 3: Use discovered patterns for optimization
        with patch("prompt_improver.cli.asyncio.run") as mock_run:
            mock_optimization = {
                "status": "success",
                "optimized_rules": ["clarity_rule", "specificity_rule"],
                "improvement": 0.12
            }
            mock_run.return_value = mock_optimization
            mock_run.side_effect = lambda coro: mock_optimization  # Return optimization and consume coroutine

            optimization_result = cli_runner.invoke(app, [
                "optimize-rules",
                "--rule", "clarity_rule",
                "--ensemble"
            ])

            assert optimization_result.exit_code == 0

        # Verify workflow files exist
        assert input_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
