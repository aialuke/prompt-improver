"""
Tests for Week 9 Enhanced PID Manager Implementation
Tests atomic PID operations, stale detection, multi-session coordination with real behavior testing.
"""

import asyncio
import json
import os
import pytest
import signal
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.cli.core.pid_manager import (
    PIDManager, ProcessState, PIDFileStatus, ProcessInfo, PIDFileInfo
)


class TestPIDManager:
    """Test enhanced PID manager functionality."""

    @pytest.fixture
    def temp_pid_dir(self):
        """Create temporary PID directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def pid_manager(self, temp_pid_dir):
        """Create PID manager for testing."""
        return PIDManager(pid_dir=temp_pid_dir)

    @pytest.fixture
    def mock_pid_data(self):
        """Create mock PID file data."""
        return {
            "pid": 12345,
            "session_id": "test_session",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "process_name": "python",
            "command_line": ["python", "test.py"],
            "working_directory": "/tmp",
            "owner_uid": os.getuid(),
            "owner_gid": os.getgid(),
            "python_version": "3.13",
            "hostname": "test-host",
            "additional_info": {}
        }

    def test_pid_manager_initialization(self, temp_pid_dir):
        """Test PID manager initialization and directory setup."""
        manager = PIDManager(pid_dir=temp_pid_dir)
        
        # Verify initialization
        assert manager.pid_dir == temp_pid_dir
        assert manager.pid_dir.exists()
        assert len(manager.active_sessions) == 0
        assert len(manager.pid_files) == 0
        assert manager.lock_timeout == 5.0

    def test_process_state_enum(self):
        """Test process state enumeration."""
        assert ProcessState.RUNNING.value == "running"
        assert ProcessState.STOPPED.value == "stopped"
        assert ProcessState.ZOMBIE.value == "zombie"
        assert ProcessState.NOT_FOUND.value == "not_found"

    def test_pid_file_status_enum(self):
        """Test PID file status enumeration."""
        assert PIDFileStatus.VALID.value == "valid"
        assert PIDFileStatus.STALE.value == "stale"
        assert PIDFileStatus.CORRUPTED.value == "corrupted"
        assert PIDFileStatus.LOCKED.value == "locked"
        assert PIDFileStatus.NOT_FOUND.value == "not_found"

    @pytest.mark.asyncio
    async def test_create_pid_file_success(self, pid_manager):
        """Test successful PID file creation."""
        session_id = "test_session_create"
        
        # Create PID file
        success, message = await pid_manager.create_pid_file(session_id)
        
        # Verify creation
        assert success is True
        assert "successfully" in message.lower()
        
        # Verify file exists
        pid_file_path = pid_manager.pid_dir / f"{session_id}.pid"
        assert pid_file_path.exists()
        
        # Verify file content
        with open(pid_file_path, 'r') as f:
            pid_data = json.load(f)
        
        assert pid_data["pid"] == os.getpid()
        assert pid_data["session_id"] == session_id
        assert "created_at" in pid_data
        
        # Verify internal tracking
        assert session_id in pid_manager.active_sessions
        assert session_id in pid_manager.pid_files

    @pytest.mark.asyncio
    async def test_create_pid_file_duplicate(self, pid_manager):
        """Test PID file creation when file already exists."""
        session_id = "test_session_duplicate"
        
        # Create first PID file
        success1, message1 = await pid_manager.create_pid_file(session_id)
        assert success1 is True
        
        # Try to create duplicate
        success2, message2 = await pid_manager.create_pid_file(session_id)
        assert success2 is False
        assert "already exists" in message2.lower()

    @pytest.mark.asyncio
    async def test_validate_pid_file_valid(self, pid_manager, mock_pid_data, temp_pid_dir):
        """Test validation of valid PID file."""
        # Create valid PID file
        pid_file_path = temp_pid_dir / "valid_session.pid"
        mock_pid_data["pid"] = os.getpid()  # Use current process PID
        
        with open(pid_file_path, 'w') as f:
            json.dump(mock_pid_data, f)
        
        # Validate PID file
        pid_info = await pid_manager._validate_pid_file(pid_file_path)
        
        # Verify validation results
        assert pid_info.status == PIDFileStatus.VALID
        assert pid_info.pid == os.getpid()
        assert pid_info.session_id == "test_session"
        assert len(pid_info.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_validate_pid_file_not_found(self, pid_manager, temp_pid_dir):
        """Test validation of non-existent PID file."""
        pid_file_path = temp_pid_dir / "nonexistent.pid"
        
        pid_info = await pid_manager._validate_pid_file(pid_file_path)
        
        assert pid_info.status == PIDFileStatus.NOT_FOUND
        assert pid_info.pid is None
        assert "does not exist" in pid_info.validation_errors[0]

    @pytest.mark.asyncio
    async def test_validate_pid_file_corrupted(self, pid_manager, temp_pid_dir):
        """Test validation of corrupted PID file."""
        # Create corrupted PID file
        pid_file_path = temp_pid_dir / "corrupted_session.pid"
        pid_file_path.write_text("invalid json content")
        
        pid_info = await pid_manager._validate_pid_file(pid_file_path)
        
        assert pid_info.status == PIDFileStatus.CORRUPTED
        assert any("JSON format" in error for error in pid_info.validation_errors)

    @pytest.mark.asyncio
    async def test_validate_pid_file_stale(self, pid_manager, mock_pid_data, temp_pid_dir):
        """Test validation of stale PID file."""
        # Create stale PID file with non-existent PID
        pid_file_path = temp_pid_dir / "stale_session.pid"
        mock_pid_data["pid"] = 99999  # Non-existent PID
        
        with open(pid_file_path, 'w') as f:
            json.dump(mock_pid_data, f)
        
        pid_info = await pid_manager._validate_pid_file(pid_file_path)
        
        assert pid_info.status == PIDFileStatus.STALE
        assert any("no longer exists" in error for error in pid_info.validation_errors)

    @pytest.mark.asyncio
    async def test_remove_pid_file_success(self, pid_manager):
        """Test successful PID file removal."""
        session_id = "test_session_remove"
        
        # Create PID file first
        success, _ = await pid_manager.create_pid_file(session_id)
        assert success is True
        
        # Remove PID file
        success, message = await pid_manager.remove_pid_file(session_id)
        
        # Verify removal
        assert success is True
        assert "successfully" in message.lower()
        
        # Verify file is gone
        pid_file_path = pid_manager.pid_dir / f"{session_id}.pid"
        assert not pid_file_path.exists()
        
        # Verify internal tracking updated
        assert session_id not in pid_manager.active_sessions
        assert session_id not in pid_manager.pid_files

    @pytest.mark.asyncio
    async def test_remove_pid_file_not_found(self, pid_manager):
        """Test removal of non-existent PID file."""
        session_id = "nonexistent_session"
        
        success, message = await pid_manager.remove_pid_file(session_id)
        
        assert success is True  # Not an error if file doesn't exist
        assert "does not exist" in message.lower()

    @pytest.mark.asyncio
    async def test_remove_pid_file_force(self, pid_manager, mock_pid_data, temp_pid_dir):
        """Test forced removal of PID file."""
        session_id = "test_session_force"
        
        # Create PID file with different PID
        pid_file_path = temp_pid_dir / f"{session_id}.pid"
        mock_pid_data["pid"] = 99999  # Non-existent PID
        
        with open(pid_file_path, 'w') as f:
            json.dump(mock_pid_data, f)
        
        # Force remove
        success, message = await pid_manager.remove_pid_file(session_id, force=True)
        
        assert success is True
        assert not pid_file_path.exists()

    @pytest.mark.asyncio
    async def test_scan_all_pid_files(self, pid_manager, temp_pid_dir):
        """Test scanning all PID files in directory."""
        # Create multiple PID files
        session_ids = ["session1", "session2", "session3"]
        
        for session_id in session_ids:
            await pid_manager.create_pid_file(session_id)
        
        # Create a corrupted file
        corrupted_path = temp_pid_dir / "corrupted.pid"
        corrupted_path.write_text("invalid json")
        
        # Scan all files
        pid_files = await pid_manager.scan_all_pid_files()
        
        # Verify scan results
        assert len(pid_files) == 4  # 3 valid + 1 corrupted
        
        for session_id in session_ids:
            assert session_id in pid_files
            assert pid_files[session_id].status == PIDFileStatus.VALID
        
        assert "corrupted" in pid_files
        assert pid_files["corrupted"].status == PIDFileStatus.CORRUPTED

    @pytest.mark.asyncio
    async def test_cleanup_stale_pid_files_dry_run(self, pid_manager, mock_pid_data, temp_pid_dir):
        """Test dry run cleanup of stale PID files."""
        # Create stale PID file
        stale_path = temp_pid_dir / "stale_session.pid"
        mock_pid_data["pid"] = 99999  # Non-existent PID
        
        with open(stale_path, 'w') as f:
            json.dump(mock_pid_data, f)
        
        # Create corrupted PID file
        corrupted_path = temp_pid_dir / "corrupted_session.pid"
        corrupted_path.write_text("invalid json")
        
        # Dry run cleanup
        report = await pid_manager.cleanup_stale_pid_files(dry_run=True)
        
        # Verify dry run results
        assert report["scanned"] == 2
        assert len(report["stale_removed"]) == 1
        assert len(report["corrupted_removed"]) == 1
        assert report["total_removed"] == 0  # No actual removal in dry run
        
        # Verify files still exist
        assert stale_path.exists()
        assert corrupted_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_stale_pid_files_actual(self, pid_manager, mock_pid_data, temp_pid_dir):
        """Test actual cleanup of stale PID files."""
        # Create stale PID file
        stale_path = temp_pid_dir / "stale_session.pid"
        mock_pid_data["pid"] = 99999  # Non-existent PID
        
        with open(stale_path, 'w') as f:
            json.dump(mock_pid_data, f)
        
        # Actual cleanup
        report = await pid_manager.cleanup_stale_pid_files(dry_run=False)
        
        # Verify cleanup results
        assert report["scanned"] == 1
        assert len(report["stale_removed"]) == 1
        assert report["total_removed"] == 1
        
        # Verify file is removed
        assert not stale_path.exists()

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, pid_manager):
        """Test getting active session information."""
        # Create active session
        session_id = "active_session"
        success, _ = await pid_manager.create_pid_file(session_id)
        assert success is True
        
        # Get active sessions
        active_sessions = await pid_manager.get_active_sessions()
        
        # Verify results
        assert session_id in active_sessions
        process_info = active_sessions[session_id]
        assert isinstance(process_info, ProcessInfo)
        assert process_info.pid == os.getpid()
        assert process_info.session_id == session_id
        assert process_info.state in [ProcessState.RUNNING, ProcessState.SLEEPING]

    @pytest.mark.asyncio
    async def test_send_signal_to_session_success(self, pid_manager):
        """Test sending signal to session process."""
        session_id = "signal_session"
        
        # Create session
        success, _ = await pid_manager.create_pid_file(session_id)
        assert success is True
        
        # Send harmless signal (SIGUSR1)
        success, message = await pid_manager.send_signal_to_session(session_id, signal.SIGUSR1)
        
        # Verify signal sent
        assert success is True
        assert "SIGUSR1" in message or "USR1" in message
        assert str(os.getpid()) in message

    @pytest.mark.asyncio
    async def test_send_signal_to_session_not_found(self, pid_manager):
        """Test sending signal to non-existent session."""
        session_id = "nonexistent_session"
        
        success, message = await pid_manager.send_signal_to_session(session_id, signal.SIGUSR1)
        
        assert success is False
        assert "not_found" in message.lower() or "does not exist" in message.lower()

    @pytest.mark.asyncio
    async def test_get_session_health(self, pid_manager):
        """Test getting session health information."""
        session_id = "health_session"
        
        # Create session
        success, _ = await pid_manager.create_pid_file(session_id)
        assert success is True
        
        # Get health report
        health_report = await pid_manager.get_session_health(session_id)
        
        # Verify health report
        assert health_report["session_id"] == session_id
        assert health_report["pid_file_status"] == "valid"
        assert health_report["process_status"] in ["running", "sleeping"]
        assert health_report["health_score"] > 0.5
        assert "process_metrics" in health_report

    @pytest.mark.asyncio
    async def test_get_session_health_not_found(self, pid_manager):
        """Test getting health for non-existent session."""
        session_id = "nonexistent_health_session"
        
        health_report = await pid_manager.get_session_health(session_id)
        
        assert health_report["session_id"] == session_id
        assert health_report["pid_file_status"] == "not_found"
        assert health_report["process_status"] == "not_found"
        assert health_report["health_score"] == 0.0

    @pytest.mark.asyncio
    async def test_coordinate_multi_session_cleanup(self, pid_manager):
        """Test multi-session cleanup coordination."""
        # Create multiple sessions
        session_ids = ["multi1", "multi2", "multi3"]
        
        for session_id in session_ids:
            success, _ = await pid_manager.create_pid_file(session_id)
            assert success is True
        
        # Coordinate cleanup
        results = await pid_manager.coordinate_multi_session_operation("cleanup", force=True)
        
        # Verify results
        assert results["operation"] == "cleanup"
        assert results["summary"]["total"] == 3
        assert results["summary"]["successful"] == 3
        assert results["summary"]["failed"] == 0
        
        # Verify all sessions cleaned up
        for session_id in session_ids:
            pid_file_path = pid_manager.pid_dir / f"{session_id}.pid"
            assert not pid_file_path.exists()

    @pytest.mark.asyncio
    async def test_coordinate_multi_session_health_check(self, pid_manager):
        """Test multi-session health check coordination."""
        # Create sessions
        session_ids = ["health1", "health2"]
        
        for session_id in session_ids:
            success, _ = await pid_manager.create_pid_file(session_id)
            assert success is True
        
        # Coordinate health check
        results = await pid_manager.coordinate_multi_session_operation("health_check")
        
        # Verify results
        assert results["operation"] == "health_check"
        assert results["summary"]["total"] == 2
        assert results["summary"]["successful"] == 2
        
        # Verify health reports
        for session_id in session_ids:
            assert session_id in results["results"]
            health_report = results["results"][session_id]
            assert health_report["health_score"] > 0.5

    @pytest.mark.asyncio
    async def test_coordinate_multi_session_signal(self, pid_manager):
        """Test multi-session signal coordination."""
        # Create sessions
        session_ids = ["signal1", "signal2"]
        
        for session_id in session_ids:
            success, _ = await pid_manager.create_pid_file(session_id)
            assert success is True
        
        # Coordinate signal sending
        results = await pid_manager.coordinate_multi_session_operation("signal", signal=signal.SIGUSR1)
        
        # Verify results
        assert results["operation"] == "signal"
        assert results["summary"]["total"] == 2
        assert results["summary"]["successful"] == 2
        
        # Verify signal results
        for session_id in session_ids:
            assert session_id in results["results"]
            signal_result = results["results"][session_id]
            assert signal_result["success"] is True

    @pytest.mark.asyncio
    async def test_file_locking_behavior(self, pid_manager, temp_pid_dir):
        """Test file locking behavior during concurrent operations."""
        session_id = "lock_test_session"
        
        # Create PID file
        success, _ = await pid_manager.create_pid_file(session_id)
        assert success is True
        
        # Test that we can check lock status
        pid_file_path = temp_pid_dir / f"{session_id}.pid"
        is_locked = await pid_manager._check_file_lock(pid_file_path)
        
        # File should not be locked when not in use
        assert is_locked is False

    @pytest.mark.asyncio
    async def test_process_info_gathering(self, pid_manager):
        """Test comprehensive process information gathering."""
        current_pid = os.getpid()
        session_id = "process_info_test"
        
        # Gather process info
        process_info = await pid_manager._gather_process_info(current_pid, session_id)
        
        # Verify process info
        assert process_info.pid == current_pid
        assert process_info.session_id == session_id
        assert process_info.state in [ProcessState.RUNNING, ProcessState.SLEEPING]
        assert process_info.owner_uid == os.getuid()
        assert process_info.owner_gid == os.getgid()
        assert isinstance(process_info.command_line, list)
        assert len(process_info.command_line) > 0
