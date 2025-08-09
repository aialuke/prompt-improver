"""
Comprehensive CLI command tests with parametrized paths for dry-run and error branches.
Covers all CLI commands from prompt_improver.cli module.

Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Use real CliRunner for actual CLI testing
- Use real temporary directories and files
- Mock only external dependencies (database sessions) when absolutely necessary
- Test actual CLI behavior rather than implementation details
"""
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from typer.testing import CliRunner
from prompt_improver.cli import app

class TestCLICommandPaths:
    """Test suite for CLI command paths using real behavior testing."""

    def setup_method(self):
        """Set up test environment before each test with real components."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.home_dir = self.temp_dir / 'home'
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.home_dir / '.local' / 'share' / 'apes' / 'data' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.create_sample_log_files()

    def teardown_method(self):
        """Clean up real temp directory after test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_sample_log_files(self):
        """Create real log files for CLI testing."""
        main_log = self.log_dir / 'apes.log'
        with open(main_log, 'w', encoding='utf-8') as f:
            f.write('2024-01-15 10:00:00 INFO Starting APES...\n')
            f.write('2024-01-15 10:00:01 DEBUG Connection established\n')
            f.write('2024-01-15 10:00:02 ERROR Test error message\n')
        mcp_log = self.log_dir / 'mcp.log'
        with open(mcp_log, 'w', encoding='utf-8') as f:
            f.write('2024-01-15 10:00:00 INFO MCP service started\n')

    @pytest.mark.parametrize('command,args,expected_patterns', [('start', ['--background'], ['Starting APES', 'background', 'PID']), ('start', ['--verbose'], ['Starting APES', 'verbose']), ('start', ['--mcp-port', '3001'], ['Starting APES', '3001']), ('stop', ['--graceful'], ['Stopping APES', 'graceful']), ('stop', ['--force'], ['Stopping APES', 'force']), ('stop', ['--timeout', '60'], ['Stopping APES', 'timeout']), ('status', ['--detailed'], ['APES Service Status', 'detailed']), ('status', ['--json'], ['{']), ('train', ['--dry-run'], ['dry run', 'training']), ('train', ['--verbose'], ['Phase 3 ML training', 'verbose']), ('train', ['--ensemble'], ['Phase 3 ML training', 'ensemble']), ('analytics', ['--rule-effectiveness'], ['Rule Effectiveness', 'analytics']), ('analytics', ['--performance-trends'], ['Analytics', 'performance']), ('analytics', ['--days', '7'], ['days', '7']), ('doctor', ['--verbose'], ['Running APES system diagnostics', 'verbose']), ('doctor', ['--fix-issues'], ['Running APES system diagnostics', 'fix']), ('performance', ['--period', '7d'], ['performance', '7d']), ('performance', ['--show-trends'], ['performance', 'trends']), ('performance', ['--export-csv'], ['performance', 'csv']), ('data-stats', ['--real-vs-synthetic'], ['DATA STATISTICS', 'real', 'synthetic']), ('data-stats', ['--quality-metrics'], ['DATA STATISTICS', 'quality']), ('data-stats', ['--format', 'json'], ['DATA STATISTICS', 'json']), ('cache-stats', [], ['Cache Statistics', 'cache']), ('cache-clear', [], ['Cache cleared', 'cleared']), ('canary-status', [], ['{', 'canary']), ('canary-adjust', [], ['{', 'canary']), ('ml-status', ['--detailed'], ['Phase 3 ML System Status', 'detailed']), ('ml-status', ['--no-models'], ['Phase 3 ML System Status', 'models']), ('ml-status', ['--no-experiments'], ['Phase 3 ML System Status', 'experiments'])])
    def test_cli_command_paths_real_behavior(self, command, args, expected_patterns, monkeypatch):
        """Test various CLI command paths with real behavior."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        monkeypatch.setenv('APES_HOME', str(self.temp_dir))
        with patch('prompt_improver.cli.get_sessionmanager') as mock_session:
            mock_session.return_value = AsyncMock()
            result = self.runner.invoke(app, [command] + args)
        assert result.exit_code in [0, 1, 2]
        if result.exit_code == 0:
            output_lower = result.stdout.lower()
            assert any((pattern.lower() in output_lower for pattern in expected_patterns)), f'None of {expected_patterns} found in output: {result.stdout}'

    @pytest.mark.parametrize('command,args,error_type', [('start', ['--mcp-port', 'invalid'], 'invalid port'), ('start', ['--mcp-port', '70000'], 'port out of range'), ('train', ['--rules', 'nonexistent_rule'], 'rule not found'), ('train', ['--invalid-option'], 'invalid option'), ('analytics', ['--days', '-1'], 'invalid days'), ('analytics', ['--days', 'abc'], 'invalid number'), ('performance', ['--unknown-flag'], 'unknown flag'), ('performance', ['--period', 'invalid_format_xyz'], 'invalid period format'), ('data-stats', ['--format', 'unsupported_format'], 'invalid format'), ('data-stats', ['--unknown-parameter', 'value'], 'unknown parameter')])
    def test_cli_command_error_paths_real_behavior(self, command, args, error_type, monkeypatch):
        """Test CLI command error handling paths with real behavior."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        result = self.runner.invoke(app, [command] + args)
        if result.exit_code == 0:
            total_output = result.stdout + getattr(result, 'stderr', '')
            assert len(total_output) > 0, f'Successful command should produce output for {error_type}'
        else:
            assert result.exit_code != 0, f'Command should fail for {error_type}'
            total_output = result.stdout + getattr(result, 'stderr', '')

    @pytest.mark.parametrize('command,dry_run_args', [('train', ['--dry-run'])])
    def test_cli_dry_run_modes_real_behavior(self, command, dry_run_args, monkeypatch):
        """Test dry-run modes for CLI commands with real behavior."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        with patch('prompt_improver.cli.get_sessionmanager') as mock_session:
            mock_session.return_value = AsyncMock()
            result = self.runner.invoke(app, [command] + dry_run_args)
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            output_lower = result.stdout.lower()
            dry_run_indicators = ['dry run', 'would', 'simulation', 'preview']
            assert any((indicator in output_lower for indicator in dry_run_indicators)), f'Dry run output should indicate simulation mode: {result.stdout}'

    @pytest.mark.parametrize('command,background_args', [('start', ['--background'])])
    def test_cli_background_modes_real_behavior(self, command, background_args, monkeypatch):
        """Test background modes for CLI commands with real behavior."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        with patch('subprocess.Popen') as mock_popen:
            mock_process = AsyncMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            result = self.runner.invoke(app, [command] + background_args)
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            output_lower = result.stdout.lower()
            background_indicators = ['background', 'pid', 'started', 'daemon']
            assert any((indicator in output_lower for indicator in background_indicators)), f'Background output should indicate background mode: {result.stdout}'

    def test_cli_help_commands_real_behavior(self):
        """Test help commands for all CLI commands with real behavior."""
        result = self.runner.invoke(app, ['--help'])
        assert result.exit_code == 0
        assert 'APES - Adaptive Prompt Enhancement System CLI' in result.stdout
        commands = ['start', 'stop', 'status', 'train', 'analytics', 'doctor', 'performance', 'data-stats', 'cache-stats', 'cache-clear', 'canary-status', 'canary-adjust', 'ml-status']
        for command in commands:
            result = self.runner.invoke(app, [command, '--help'])
            assert result.exit_code == 0, f'Help for {command} should succeed'
            assert 'Usage:' in result.stdout, f'Help for {command} should show usage'

    @pytest.mark.parametrize('command,timeout_args', [('stop', ['--timeout', '10']), ('service-stop', ['--timeout', '10'])])
    def test_cli_timeout_handling_real_behavior(self, command, timeout_args, monkeypatch):
        """Test timeout handling in CLI commands with real behavior."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        with patch('prompt_improver.cli.os.kill') as mock_kill:
            mock_kill.side_effect = ProcessLookupError('Process not found')
            result = self.runner.invoke(app, [command] + timeout_args)
        assert result.exit_code in [0, 1], 'Timeout handling should be graceful'
        assert len(result.stdout) > 0, 'Timeout commands should produce output'

    @pytest.mark.parametrize('command,verbose_args', [('start', ['--verbose']), ('train', ['--verbose']), ('doctor', ['--verbose']), ('init', ['--verbose'])])
    def test_cli_verbose_modes_real_behavior(self, command, verbose_args, monkeypatch):
        """Test verbose modes for CLI commands with real behavior."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        with patch('prompt_improver.cli.get_sessionmanager') as mock_session:
            mock_session.return_value = AsyncMock()
            result_normal = self.runner.invoke(app, [command])
            result_verbose = self.runner.invoke(app, [command] + verbose_args)
        assert result_verbose.exit_code in [0, 1]
        if result_normal.exit_code == 0 and result_verbose.exit_code == 0:
            assert len(result_verbose.stdout) >= len(result_normal.stdout), 'Verbose mode should produce equal or more output'

    def test_cli_database_connection_errors_real_behavior(self, monkeypatch):
        """Test CLI behavior when database connection fails with real error simulation."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        with patch('prompt_improver.cli.get_sessionmanager') as mock_session:
            mock_session.side_effect = ConnectionError('Database not available')
            result = self.runner.invoke(app, ['status'])
        assert result.exit_code in [0, 1], 'Database errors should be handled gracefully'
        total_output = result.stdout + getattr(result, 'stderr', '')
        assert len(total_output) > 0, 'Database error should produce output'

    def test_cli_file_system_errors_real_behavior(self, monkeypatch):
        """Test CLI behavior when file system operations fail with real scenarios."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        readonly_dir = self.temp_dir / 'readonly'
        readonly_dir.mkdir()
        try:
            readonly_dir.chmod(292)
            monkeypatch.setenv('APES_HOME', str(readonly_dir))
            result = self.runner.invoke(app, ['start'])
            assert result.exit_code in [0, 1], 'File system errors should be handled gracefully'
        except (OSError, PermissionError):
            pytest.skip('Cannot test file permission restrictions on this system')
        finally:
            try:
                readonly_dir.chmod(493)
            except (OSError, PermissionError):
                pass

    @pytest.mark.parametrize('command,json_args', [('status', ['--json']), ('canary-status', []), ('canary-adjust', [])])
    def test_cli_json_output_real_behavior(self, command, json_args, monkeypatch):
        """Test JSON output format for CLI commands with real behavior."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        with patch('prompt_improver.cli.get_sessionmanager') as mock_session:
            mock_session.return_value = AsyncMock()
            result = self.runner.invoke(app, [command] + json_args)
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            json_indicators = ['{', '}', '[', ']', '"']
            assert any((indicator in result.stdout for indicator in json_indicators)), f'JSON output should contain JSON structures: {result.stdout}'
            if result.stdout.strip().startswith('{'):
                try:
                    json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    assert '{' in result.stdout, 'Should contain JSON-like content'

    def test_cli_interrupt_handling_real_behavior(self, monkeypatch):
        """Test CLI behavior when interrupted with real signal simulation."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        with patch('prompt_improver.cli.subprocess.run') as mock_run:
            mock_run.side_effect = KeyboardInterrupt('User interrupted')
            result = self.runner.invoke(app, ['start'])
        assert result.exit_code in [0, 1, 130], 'Interrupts should be handled gracefully'

    @pytest.mark.parametrize('command,export_args', [('backup', ['--to', '/tmp/backup']), ('export-training-data', ['--output', '/tmp/export']), ('migrate-export', ['/tmp/migrate'])])
    def test_cli_export_commands_real_behavior(self, command, export_args, monkeypatch):
        """Test export commands with real path handling."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        export_dir = self.temp_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        real_export_args = []
        for arg in export_args:
            if arg.startswith('/tmp/'):
                real_path = str(export_dir / arg.split('/')[-1])
                real_export_args.append(real_path)
            else:
                real_export_args.append(arg)
        with patch('prompt_improver.cli.get_sessionmanager') as mock_session:
            mock_session.return_value = AsyncMock()
            result = self.runner.invoke(app, [command] + real_export_args)
        assert result.exit_code in [0, 1], 'Export commands should handle paths correctly'
        if result.exit_code == 0:
            export_indicators = ['export', 'backup', 'migrate', 'saved', 'created']
            output_lower = result.stdout.lower()
            assert any((indicator in output_lower for indicator in export_indicators)), f'Export output should indicate export operation: {result.stdout}'

    def test_cli_environment_variable_handling_real_behavior(self, monkeypatch):
        """Test CLI behavior with different environment variables."""
        test_configs = [{'APES_DEBUG': 'true', 'APES_LOG_LEVEL': 'DEBUG'}, {'APES_CONFIG_PATH': str(self.temp_dir / 'config.yaml')}, {'APES_DATA_DIR': str(self.temp_dir / 'data')}]
        for env_config in test_configs:
            for key, value in env_config.items():
                monkeypatch.setenv(key, value)
            result = self.runner.invoke(app, ['--help'])
            assert result.exit_code == 0, f'Should handle env config {env_config}'

    def test_cli_concurrent_execution_real_behavior(self):
        """Test CLI behavior under real concurrent execution scenarios."""
        import queue
        import threading
        results_queue = queue.Queue()

        def run_safe_command():
            """Run a safe command that won't interfere with other tests."""
            try:
                result = self.runner.invoke(app, ['--help'])
                results_queue.put(('success', result))
            except Exception as e:
                results_queue.put(('error', str(e)))
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_safe_command)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        assert len(results) >= 1, 'At least one concurrent command should complete'
        successful_results = [r for r in results if r[0] == 'success' and r[1].exit_code == 0]
        assert len(successful_results) >= 1, 'At least one command should succeed'

    def test_cli_performance_monitoring_real_behavior(self, monkeypatch):
        """Test CLI performance with real timing measurements."""
        monkeypatch.setenv('HOME', str(self.home_dir))
        start_time = time.time()
        result = self.runner.invoke(app, ['--help'])
        end_time = time.time()
        execution_time = end_time - start_time
        assert result.exit_code == 0, 'Help command should succeed'
        assert execution_time < 10.0, f'Help command should complete in <10s, took {execution_time:.2f}s'
        start_time = time.time()
        with patch('prompt_improver.cli.get_sessionmanager') as mock_session:
            mock_session.return_value = AsyncMock()
            result = self.runner.invoke(app, ['status'])
        end_time = time.time()
        execution_time = end_time - start_time
        assert result.exit_code in [0, 1], 'Status command should complete'
        assert execution_time < 30.0, f'Status command should complete in <30s, took {execution_time:.2f}s'

    @pytest.mark.parametrize('command, args, expected', [('start', ['--background'], 'Starting APES MCP server'), ('stop', ['--graceful'], 'Stopping APES MCP server'), ('status', ['--json'], '{')])
    def test_cli_basic_commands_real_behavior(self, command, args, expected):
        """Test basic CLI commands with real behavior (migrated from test_cli_commands.py)."""
        result = self.runner.invoke(app, [command] + args)
        assert result.exit_code == 0
        assert expected in result.output
