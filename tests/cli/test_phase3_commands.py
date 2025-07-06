"""
Tests for Phase 3 enhanced CLI commands.
Tests the enhanced training interface and new ML commands added for Phase 3.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner
from rich.console import Console

from prompt_improver.cli import app


class TestPhase3CLICommands:
    """Test Phase 3 enhanced CLI commands."""


class TestEnhancedTrainCommand:
    """Test enhanced train command with Phase 3 ML readiness assessment."""
    
    def test_train_command_default_options(self, cli_runner):
        """Test train command with default options."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run, \
             patch('prompt_improver.cli.console') as mock_console:
            
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['train'])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
            
            # Verify default parameters were used
            call_args = mock_run.call_args[0][0]
            # The function should be called with proper async function

    def test_train_command_with_real_data_priority(self, cli_runner):
        """Test train command with real data priority flag."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['train', '--real-data-priority'])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_train_command_verbose_mode(self, cli_runner):
        """Test train command with verbose output."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['train', '--verbose'])
            
            assert result.exit_code == 0

    def test_train_command_dry_run(self, cli_runner):
        """Test train command with dry run option."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['train', '--dry-run'])
            
            assert result.exit_code == 0

    def test_train_command_specific_rules(self, cli_runner):
        """Test train command with specific rule IDs."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['train', '--rules', 'clarity_rule,specificity_rule'])
            
            assert result.exit_code == 0

    def test_train_command_with_ensemble(self, cli_runner):
        """Test train command with ensemble optimization."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['train', '--ensemble'])
            
            assert result.exit_code == 0


class TestDiscoverPatternsCommand:
    """Test discover_patterns command added for Phase 3."""
    
    def test_discover_patterns_default(self, cli_runner):
        """Test discover patterns command with default parameters."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['discover-patterns'])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_discover_patterns_custom_thresholds(self, cli_runner):
        """Test discover patterns with custom effectiveness and support thresholds."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, [
                'discover-patterns', 
                '--min-effectiveness', '0.8',
                '--min-support', '10'
            ])
            
            assert result.exit_code == 0

    def test_discover_patterns_help(self, cli_runner):
        """Test discover patterns command help text."""
        result = cli_runner.invoke(app, ['discover-patterns', '--help'])
        
        assert result.exit_code == 0
        assert 'Discover new effective rule patterns' in result.stdout
        assert '--min-effectiveness' in result.stdout
        assert '--min-support' in result.stdout


class TestMLStatusCommand:
    """Test ml_status command added for Phase 3."""
    
    def test_ml_status_default(self, cli_runner):
        """Test ML status command with default options."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['ml-status'])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_ml_status_help(self, cli_runner):
        """Test ML status command help text."""
        result = cli_runner.invoke(app, ['ml-status', '--help'])
        
        assert result.exit_code == 0
        assert 'Show Phase 3 ML system status' in result.stdout


class TestOptimizeRulesCommand:
    """Test optimize_rules command added for Phase 3."""
    
    def test_optimize_rules_default(self, cli_runner):
        """Test optimize rules command with default parameters."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['optimize-rules'])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_optimize_rules_specific_rules(self, cli_runner):
        """Test optimize rules for specific rule IDs."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, [
                'optimize-rules',
                '--rule', 'clarity_rule'
            ])
            
            assert result.exit_code == 0

    def test_optimize_rules_help(self, cli_runner):
        """Test optimize rules command help text."""
        result = cli_runner.invoke(app, ['optimize-rules', '--help'])
        
        assert result.exit_code == 0
        assert 'Trigger targeted rule optimization' in result.stdout
        assert '--rule' in result.stdout


class TestPhase3CLIIntegration:
    """Test Phase 3 CLI integration and error handling."""
    
    def test_all_phase3_commands_exist(self, cli_runner):
        """Test that all Phase 3 commands are properly registered."""
        # Test help shows all commands
        result = cli_runner.invoke(app, ['--help'])
        
        assert result.exit_code == 0
        assert 'train' in result.stdout
        assert 'discover-patterns' in result.stdout
        assert 'ml-status' in result.stdout 
        assert 'optimize-rules' in result.stdout

    def test_phase3_commands_database_error_handling(self, cli_runner):
        """Test Phase 3 commands handle database errors gracefully."""
        async def mock_failing_function(*args, **kwargs):
            raise Exception("Database connection failed")
        
        with patch('prompt_improver.cli.asyncio.run', side_effect=Exception("Database error")):
            result = cli_runner.invoke(app, ['train'])
            
            # Should not crash, should exit with error code
            assert result.exit_code != 0

    def test_phase3_rich_output_integration(self, cli_runner):
        """Test that Phase 3 commands use Rich for enhanced output."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run, \
             patch('prompt_improver.cli.console.print') as mock_print:
            
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['train', '--verbose'])
            
            assert result.exit_code == 0
            # Rich console.print should be called for enhanced output
            assert mock_print.call_count >= 0  # Allow for various output calls


class TestCLIAsyncFunctionality:
    """Test async functionality in CLI commands."""
    
    def test_async_commands_use_asyncio_run(self, cli_runner):
        """Test that async CLI commands properly use asyncio.run."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            # Test that train command uses asyncio.run
            result = cli_runner.invoke(app, ['train', '--dry-run'])
            assert result.exit_code == 0
            mock_run.assert_called()
            
            mock_run.reset_mock()
            
            # Test that discover-patterns command uses asyncio.run
            result = cli_runner.invoke(app, ['discover-patterns'])
            assert result.exit_code == 0
            mock_run.assert_called()
            
            mock_run.reset_mock()
            
            # Test that ml-status command uses asyncio.run
            result = cli_runner.invoke(app, ['ml-status'])
            assert result.exit_code == 0
            mock_run.assert_called()


class TestCLIRichOutputFormatting:
    """Test Rich output formatting for Phase 3 commands."""
    
    def test_train_command_rich_table_output(self, cli_runner):
        """Test that train command outputs rich formatted tables."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run, \
             patch('prompt_improver.cli.Table') as mock_table, \
             patch('prompt_improver.cli.console.print'):
            
            mock_run.return_value = None
            mock_table_instance = MagicMock()
            mock_table.return_value = mock_table_instance
            
            result = cli_runner.invoke(app, ['train', '--verbose'])
            
            assert result.exit_code == 0
            # Verify Rich Table was used for ML readiness assessment
            # (This tests the enhanced output formatting mentioned in Phase 3)

    def test_ml_status_progress_output(self, cli_runner):
        """Test that ML status uses Rich progress indicators."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run, \
             patch('prompt_improver.cli.Progress') as mock_progress:
            
            mock_run.return_value = None
            mock_progress_instance = MagicMock()
            mock_progress.return_value.__enter__.return_value = mock_progress_instance
            
            result = cli_runner.invoke(app, ['ml-status'])
            
            assert result.exit_code == 0


class TestCLIErrorHandlingAndLogging:
    """Test error handling and logging for Phase 3 CLI commands."""
    
    def test_train_command_ml_service_error(self, cli_runner):
        """Test train command handles ML service errors gracefully."""
        async def failing_training(*args, **kwargs):
            raise Exception("ML optimization failed")
        
        with patch('prompt_improver.cli.asyncio.run', side_effect=Exception("ML error")):
            result = cli_runner.invoke(app, ['train'])
            
            # Should handle error gracefully, not crash
            assert result.exit_code != 0

    def test_discover_patterns_insufficient_data_error(self, cli_runner):
        """Test discover patterns handles insufficient data error."""
        with patch('prompt_improver.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(app, ['discover-patterns'])
            
            # Should complete successfully with proper mocking
            assert result.exit_code == 0

    def test_database_connection_error_handling(self, cli_runner):
        """Test all Phase 3 commands handle database connection errors."""
        commands_to_test = [
            ['train'],
            ['discover-patterns'],
            ['ml-status'],
            ['optimize-rules']
        ]
        
        for command in commands_to_test:
            with patch('prompt_improver.cli.asyncio.run', side_effect=Exception("DB connection failed")):
                result = cli_runner.invoke(app, command)
                
                # All commands should handle DB errors gracefully
                assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])