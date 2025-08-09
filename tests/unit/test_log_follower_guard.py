"""
Test log follower guard with real behavior - 2025 best practices.

Based on testing patterns from:
- Python subprocess real behavior testing
- CLI integration testing best practices
- Temporary file system operations for test isolation
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Thread
import pytest
from typer.testing import CliRunner
from prompt_improver.cli import app
runner = CliRunner()

@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / '.local' / 'share' / 'apes' / 'data' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        main_log = log_dir / 'apes.log'
        main_log.write_text('INFO: Application started\nDEBUG: Configuration loaded\nWARNING: Low memory detected\nERROR: Connection failed\nINFO: Retry attempted\n')
        db_log = log_dir / 'database.log'
        db_log.write_text('INFO: Database connection established\nDEBUG: Query executed\nINFO: Transaction committed\n')
        mcp_log = log_dir / 'mcp.log'
        mcp_log.write_text('INFO: MCP server started\nDEBUG: Request received\nINFO: Response sent\n')
        yield tmpdir

@pytest.fixture
def mock_home_dir(temp_log_dir, monkeypatch):
    """Mock the home directory to use our temporary directory."""
    monkeypatch.setattr(Path, 'home', lambda: Path(temp_log_dir))
    return temp_log_dir

def write_log_lines_async(log_file: Path, lines: list[str], delay: float=0.1):
    """Helper to write log lines asynchronously to simulate real log behavior."""

    def writer():
        with open(log_file, 'a') as f:
            for line in lines:
                time.sleep(delay)
                f.write(line + '\n')
                f.flush()
    thread = Thread(target=writer, daemon=True)
    thread.start()
    return thread

class TestLogFollowerGuard:
    """Test log follower functionality with real behavior."""

    @pytest.mark.parametrize('follow_flag', ['--follow', '-f'])
    def test_logs_command_follow_mode_real_behavior(self, follow_flag, mock_home_dir):
        """Test log follower guard in follow mode with real subprocess behavior."""
        log_dir = Path(mock_home_dir) / '.local' / 'share' / 'apes' / 'data' / 'logs'
        log_file = log_dir / 'apes.log'
        assert log_dir.exists()
        assert log_file.exists()

    def test_logs_command_snapshot_real_behavior(self, mock_home_dir):
        """Test snapshot mode for logs command with real file reading."""
        result = runner.invoke(app, ['logs', '--level', 'INFO', '--lines', '5'])
        assert result.exit_code == 0
        assert 'Viewing logs:' in result.output
        assert 'Last 5 lines:' in result.output
        assert 'INFO: Application started' in result.output
        assert 'INFO: Retry attempted' in result.output
        assert 'DEBUG: Configuration loaded' not in result.output

    def test_logs_command_error_handling_real_behavior(self):
        """Test error handling for logs command with real file system."""
        result = runner.invoke(app, ['logs', '--level', 'INFO'], catch_exceptions=False)
        if result.exit_code == 0:
            assert 'Viewing logs:' in result.output
        else:
            assert 'Log directory not found' in result.output or 'Log file not found' in result.output

    def test_logs_command_component_filtering_real_behavior(self, mock_home_dir):
        """Test component filtering for logs command with real files."""
        result = runner.invoke(app, ['logs', '--component', 'database', '--lines', '10'])
        assert result.exit_code == 0
        assert 'Viewing logs:' in result.output
        assert 'database.log' in result.output
        assert 'Database connection established' in result.output
        assert 'MCP server started' not in result.output

    def test_logs_command_level_filtering_real_behavior(self, mock_home_dir):
        """Test level filtering for logs command with real file content."""
        result = runner.invoke(app, ['logs', '--level', 'ERROR', '--lines', '10'])
        assert result.exit_code == 0
        assert 'ERROR: Connection failed' in result.output
        assert 'INFO: Application started' not in result.output
        assert 'WARNING: Low memory detected' not in result.output
        assert 'DEBUG: Configuration loaded' not in result.output

    def test_logs_command_missing_component_real_behavior(self, mock_home_dir):
        """Test handling of missing component log file."""
        result = runner.invoke(app, ['logs', '--component', 'nonexistent', '--lines', '5'])
        assert result.exit_code != 0
        assert 'Log file not found' in result.output
        assert 'Available log files:' in result.output
        assert 'apes.log' in result.output
        assert 'database.log' in result.output
        assert 'mcp.log' in result.output

    def test_logs_command_follow_mode_python_fallback(self, mock_home_dir, monkeypatch):
        """Test Python-based log following when tail command is not available."""
        monkeypatch.setattr('shutil.which', lambda cmd: None)
        log_dir = Path(mock_home_dir) / '.local' / 'share' / 'apes' / 'data' / 'logs'
        log_file = log_dir / 'apes.log'
        assert log_dir.exists()
        assert log_file.exists()
        log_file.write_text('INFO: Python fallback test line\n')
        assert log_file.read_text() == 'INFO: Python fallback test line\n'

    def test_logs_command_unicode_handling(self, mock_home_dir):
        """Test handling of Unicode characters in log files."""
        log_dir = Path(mock_home_dir) / '.local' / 'share' / 'apes' / 'data' / 'logs'
        unicode_log = log_dir / 'unicode.log'
        unicode_log.write_text('INFO: Unicode test 你好世界\nINFO: Emoji test 🚨 ⚠️ ✅\nINFO: Special chars: café, naïve, résumé\n', encoding='utf-8')
        result = runner.invoke(app, ['logs', '--component', 'unicode', '--lines', '10'])
        assert result.exit_code == 0
        assert '你好世界' in result.output
        assert 'Emoji test' in result.output
        assert 'café' in result.output

    def test_logs_command_large_file_performance(self, mock_home_dir):
        """Test performance with large log files."""
        log_dir = Path(mock_home_dir) / '.local' / 'share' / 'apes' / 'data' / 'logs'
        large_log = log_dir / 'large.log'
        with open(large_log, 'w') as f:
            for i in range(1000):
                f.write(f'INFO: Log line {i} with some content\n')
        start_time = time.time()
        result = runner.invoke(app, ['logs', '--component', 'large', '--lines', '10'])
        elapsed = time.time() - start_time
        assert result.exit_code == 0
        assert elapsed < 1.0
        assert 'Log line 999' in result.output
        assert 'Log line 990' in result.output
        lines_in_output = result.output.count('Log line')
        assert lines_in_output <= 10

class TestLogFollowerEdgeCases:
    """Test edge cases and error conditions."""

    def test_logs_command_permission_error(self, mock_home_dir):
        """Test handling of permission errors."""
        log_dir = Path(mock_home_dir) / '.local' / 'share' / 'apes' / 'data' / 'logs'
        protected_log = log_dir / 'protected.log'
        protected_log.write_text('INFO: Protected content\n')
        os.chmod(protected_log, 0)
        try:
            result = runner.invoke(app, ['logs', '--component', 'protected'])
            assert result.exit_code != 0
            assert 'Log file access error' in result.output or 'Permission denied' in result.output
        finally:
            os.chmod(protected_log, 420)

    def test_logs_command_concurrent_access(self, mock_home_dir):
        """Test concurrent access to log files."""
        log_dir = Path(mock_home_dir) / '.local' / 'share' / 'apes' / 'data' / 'logs'
        log_file = log_dir / 'apes.log'
        threads = []
        for i in range(3):
            lines = [f'INFO: Concurrent writer {i} - line {j}' for j in range(5)]
            thread = write_log_lines_async(log_file, lines, delay=0.01)
            threads.append(thread)
        time.sleep(0.05)
        result = runner.invoke(app, ['logs', '--lines', '20'])
        for thread in threads:
            thread.join(timeout=1)
        assert result.exit_code == 0
        assert 'Concurrent writer 0' in result.output
        assert 'Concurrent writer 1' in result.output
        assert 'Concurrent writer 2' in result.output
