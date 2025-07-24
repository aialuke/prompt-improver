"""
Tests for Week 9 Emergency Operations Implementation
Tests emergency operations triggered by signals with real behavior testing.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.cli.core.emergency_operations import EmergencyOperationsManager
from prompt_improver.cli.core.signal_handler import SignalContext, SignalOperation
import signal


class TestEmergencyOperationsManager:
    """Test emergency operations manager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def emergency_ops(self, temp_backup_dir):
        """Create emergency operations manager for testing."""
        return EmergencyOperationsManager(backup_dir=temp_backup_dir)

    @pytest.fixture
    def signal_context(self):
        """Create signal context for testing."""
        return SignalContext(
            operation=SignalOperation.CHECKPOINT,
            signal_name="SIGUSR1",
            signal_number=signal.SIGUSR1,
            triggered_at=datetime.now(timezone.utc)
        )

    @pytest.mark.asyncio
    async def test_emergency_checkpoint_creation(self, emergency_ops, signal_context, temp_backup_dir):
        """Test emergency checkpoint creation with real file operations."""
        # Create emergency checkpoint
        result = await emergency_ops.create_emergency_checkpoint(signal_context)

        # Verify checkpoint was created successfully
        assert result["status"] == "success"
        assert result["checkpoint_id"] is not None
        assert "emergency_" in result["checkpoint_id"]
        
        # Verify checkpoint file exists
        checkpoint_path = Path(result["path"])
        assert checkpoint_path.exists()
        assert checkpoint_path.parent == temp_backup_dir

        # Verify checkpoint content
        checkpoint_data = json.loads(checkpoint_path.read_text())
        assert checkpoint_data["emergency"] is True
        assert checkpoint_data["trigger"]["signal"] == "SIGUSR1"
        assert checkpoint_data["trigger"]["operation"] == "checkpoint"
        assert "system_state" in checkpoint_data
        assert "training_state" in checkpoint_data

        # Verify operation history tracking
        assert len(emergency_ops.operation_history["checkpoints"]) == 1
        assert emergency_ops.operation_history["checkpoints"][0]["checkpoint_id"] == result["checkpoint_id"]

    @pytest.mark.asyncio
    async def test_status_report_generation(self, emergency_ops, temp_backup_dir):
        """Test status report generation with real system data."""
        # Create signal context for status report
        context = SignalContext(
            operation=SignalOperation.STATUS_REPORT,
            signal_name="SIGUSR2",
            signal_number=signal.SIGUSR2,
            triggered_at=datetime.now(timezone.utc)
        )

        # Generate status report
        result = await emergency_ops.generate_status_report(context)

        # Verify report was generated successfully
        assert result["report_id"] is not None
        assert "status_" in result["report_id"]
        assert result["trigger"]["signal"] == "SIGUSR2"
        assert result["trigger"]["operation"] == "status_report"

        # Verify report content
        assert "system" in result
        assert "training" in result
        assert "resources" in result
        assert "emergency_operations" in result

        # Verify system information is present
        system_info = result["system"]
        assert "pid" in system_info
        assert "memory_usage_mb" in system_info

        # Verify resource information
        resources = result["resources"]
        assert "memory" in resources
        assert "cpu" in resources
        assert "connections" in resources

        # Verify operation history tracking
        assert len(emergency_ops.operation_history["status_reports"]) == 1

    @pytest.mark.asyncio
    async def test_configuration_reload(self, emergency_ops, temp_backup_dir):
        """Test configuration reload with real behavior."""
        # Create signal context for config reload
        context = SignalContext(
            operation=SignalOperation.CONFIG_RELOAD,
            signal_name="SIGHUP",
            signal_number=signal.SIGHUP,
            triggered_at=datetime.now(timezone.utc)
        )

        # Perform configuration reload
        result = await emergency_ops.reload_configuration(context)

        # Verify reload was performed successfully
        assert result["reload_id"] is not None
        assert "config_reload_" in result["reload_id"]
        assert result["trigger"]["signal"] == "SIGHUP"
        assert result["trigger"]["operation"] == "config_reload"

        # Verify reload content
        assert "old_config" in result
        assert "new_config" in result
        assert "changes" in result
        assert "status" in result

        # Verify operation history tracking
        assert len(emergency_ops.operation_history["config_reloads"]) == 1

    @pytest.mark.asyncio
    async def test_system_state_gathering(self, emergency_ops):
        """Test system state gathering with real system data."""
        system_state = await emergency_ops._gather_system_state()

        # Verify system state contains expected information
        assert "pid" in system_state
        assert "timestamp" in system_state
        assert "memory_usage_mb" in system_state
        assert "cpu_percent" in system_state
        assert "disk_usage" in system_state

        # Verify disk usage information
        disk_usage = system_state["disk_usage"]
        assert "total_gb" in disk_usage
        assert "used_gb" in disk_usage
        assert "free_gb" in disk_usage

        # Verify data types
        assert isinstance(system_state["pid"], int)
        assert isinstance(system_state["memory_usage_mb"], (int, float))
        assert isinstance(system_state["cpu_percent"], (int, float))

    @pytest.mark.asyncio
    async def test_resource_status_gathering(self, emergency_ops):
        """Test resource status gathering including CoreDis awareness."""
        resource_status = await emergency_ops._gather_resource_status()

        # Verify resource status contains expected information
        assert "memory" in resource_status
        assert "cpu" in resource_status
        assert "connections" in resource_status

        # Verify memory information
        memory_info = resource_status["memory"]
        assert "total_mb" in memory_info
        assert "available_mb" in memory_info
        assert "percent" in memory_info

        # Verify CPU information
        cpu_info = resource_status["cpu"]
        assert "percent" in cpu_info
        assert "count" in cpu_info

        # Verify connection status
        connections = resource_status["connections"]
        assert "database" in connections
        assert "coredis" in connections

    @pytest.mark.asyncio
    async def test_multiple_emergency_operations(self, emergency_ops, temp_backup_dir):
        """Test multiple emergency operations in sequence."""
        # Create different signal contexts
        checkpoint_context = SignalContext(
            operation=SignalOperation.CHECKPOINT,
            signal_name="SIGUSR1",
            signal_number=signal.SIGUSR1,
            triggered_at=datetime.now(timezone.utc)
        )

        status_context = SignalContext(
            operation=SignalOperation.STATUS_REPORT,
            signal_name="SIGUSR2",
            signal_number=signal.SIGUSR2,
            triggered_at=datetime.now(timezone.utc)
        )

        config_context = SignalContext(
            operation=SignalOperation.CONFIG_RELOAD,
            signal_name="SIGHUP",
            signal_number=signal.SIGHUP,
            triggered_at=datetime.now(timezone.utc)
        )

        # Perform multiple operations
        checkpoint_result = await emergency_ops.create_emergency_checkpoint(checkpoint_context)
        status_result = await emergency_ops.generate_status_report(status_context)
        config_result = await emergency_ops.reload_configuration(config_context)

        # Verify all operations succeeded
        assert checkpoint_result["status"] == "success"
        assert status_result["report_id"] is not None
        assert config_result["reload_id"] is not None

        # Verify operation history tracking
        assert len(emergency_ops.operation_history["checkpoints"]) == 1
        assert len(emergency_ops.operation_history["status_reports"]) == 1
        assert len(emergency_ops.operation_history["config_reloads"]) == 1

        # Verify files were created
        checkpoint_files = list(temp_backup_dir.glob("emergency_*.json"))
        status_files = list(temp_backup_dir.glob("status_*.json"))
        config_files = list(temp_backup_dir.glob("config_reload_*.json"))

        assert len(checkpoint_files) == 1
        assert len(status_files) == 1
        assert len(config_files) == 1

    @pytest.mark.asyncio
    async def test_emergency_operations_error_handling(self, emergency_ops):
        """Test error handling in emergency operations."""
        # Create signal context
        context = SignalContext(
            operation=SignalOperation.CHECKPOINT,
            signal_name="SIGUSR1",
            signal_number=signal.SIGUSR1,
            triggered_at=datetime.now(timezone.utc)
        )

        # Mock system state gathering to raise an exception
        with patch.object(emergency_ops, '_gather_system_state', side_effect=Exception("Test error")):
            result = await emergency_ops.create_emergency_checkpoint(context)

            # Verify error handling
            assert result["status"] == "error"
            assert "Test error" in result["error"]
            assert result["checkpoint_id"] is None

    @pytest.mark.asyncio
    async def test_training_state_gathering_with_database(self, emergency_ops):
        """Test training state gathering with database integration."""
        # This test requires database setup, so we'll mock the database interaction
        with patch('prompt_improver.cli.core.emergency_operations.get_session_context') as mock_session:
            # Mock database session and query results
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock query result
            mock_result = MagicMock()
            mock_session_obj = MagicMock()
            mock_session_obj.session_id = "test_session_123"
            mock_session_obj.started_at = datetime.now(timezone.utc)
            mock_session_obj.total_iterations = 5
            mock_session_obj.status = "running"
            
            mock_result.scalars.return_value.all.return_value = [mock_session_obj]
            mock_db_session.execute.return_value = mock_result

            # Test training state gathering
            training_state = await emergency_ops._gather_training_state()

            # Verify training state contains expected information
            assert "active_sessions" in training_state
            assert "sessions" in training_state
            assert training_state["active_sessions"] == 1
            assert len(training_state["sessions"]) == 1
            
            session_data = training_state["sessions"][0]
            assert session_data["session_id"] == "test_session_123"
            assert session_data["status"] == "running"
            assert session_data["total_iterations"] == 5


class TestEmergencyOperationsIntegration:
    """Test integration of emergency operations with signal handling."""

    @pytest.mark.asyncio
    async def test_emergency_operations_file_persistence(self):
        """Test that emergency operations create persistent files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir)
            emergency_ops = EmergencyOperationsManager(backup_dir=backup_dir)

            # Create checkpoint
            context = SignalContext(
                operation=SignalOperation.CHECKPOINT,
                signal_name="SIGUSR1",
                signal_number=signal.SIGUSR1,
                triggered_at=datetime.now(timezone.utc)
            )

            result = await emergency_ops.create_emergency_checkpoint(context)

            # Verify file persistence
            checkpoint_path = Path(result["path"])
            assert checkpoint_path.exists()
            
            # Verify file content is valid JSON
            checkpoint_data = json.loads(checkpoint_path.read_text())
            assert checkpoint_data["emergency"] is True
            assert checkpoint_data["checkpoint_id"] == result["checkpoint_id"]
