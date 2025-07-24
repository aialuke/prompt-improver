"""
Tests for Week 9 Emergency Save Manager Implementation
Tests emergency save procedures with real behavior testing.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.cli.core.emergency_save import (
    EmergencySaveManager, EmergencySaveContext, EmergencySaveResult
)


class TestEmergencySaveManager:
    """Test emergency save manager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def emergency_save_manager(self, temp_backup_dir):
        """Create emergency save manager for testing."""
        return EmergencySaveManager(backup_dir=temp_backup_dir)

    @pytest.fixture
    def save_context(self):
        """Create emergency save context for testing."""
        return EmergencySaveContext(
            save_id=f"test_save_{int(datetime.now().timestamp())}",
            triggered_at=datetime.now(timezone.utc),
            trigger_type="manual",
            trigger_details={"test": True},
            priority=1,
            atomic=True,
            validate=True
        )

    @pytest.mark.asyncio
    async def test_emergency_save_manager_initialization(self, temp_backup_dir):
        """Test emergency save manager initialization."""
        manager = EmergencySaveManager(backup_dir=temp_backup_dir)
        
        # Verify initialization
        assert manager.backup_dir == temp_backup_dir
        assert manager.backup_dir.exists()
        assert manager.temp_dir.exists()
        assert len(manager.active_saves) == 0
        assert len(manager.save_history) == 0

    @pytest.mark.asyncio
    async def test_atomic_emergency_save_success(self, emergency_save_manager, save_context, temp_backup_dir):
        """Test successful atomic emergency save operation."""
        # Perform emergency save
        result = await emergency_save_manager.perform_emergency_save(
            context=save_context,
            components=["system_state", "configuration"]
        )

        # Verify save result
        assert result.status == "success"
        assert result.save_id == save_context.save_id
        assert len(result.saved_components) == 2
        assert len(result.failed_components) == 0
        assert result.total_size_bytes > 0
        assert result.duration_seconds > 0

        # Verify files were created
        save_dir = temp_backup_dir / save_context.save_id
        assert save_dir.exists()
        
        system_file = save_dir / "system_state.json"
        config_file = save_dir / "configuration.json"
        assert system_file.exists()
        assert config_file.exists()

        # Verify file content
        system_data = json.loads(system_file.read_text())
        assert system_data["component"] == "system_state"
        assert system_data["save_id"] == save_context.save_id
        assert "data" in system_data

        config_data = json.loads(config_file.read_text())
        assert config_data["component"] == "configuration"
        assert config_data["save_id"] == save_context.save_id
        assert "data" in config_data

        # Verify save history
        assert len(emergency_save_manager.save_history) == 1
        assert emergency_save_manager.save_history[0].save_id == save_context.save_id

    @pytest.mark.asyncio
    async def test_emergency_save_with_validation(self, emergency_save_manager, save_context):
        """Test emergency save with validation enabled."""
        # Enable validation
        save_context.validate = True

        # Perform emergency save
        result = await emergency_save_manager.perform_emergency_save(
            context=save_context,
            components=["system_state"]
        )

        # Verify validation was performed
        assert result.status == "success"
        assert "validation_time" in result.validation_results
        assert "components_validated" in result.validation_results
        assert "validation_errors" in result.validation_results
        assert "integrity_checks" in result.validation_results

        # Verify validation results
        validation = result.validation_results
        assert "system_state" in validation["components_validated"]
        assert validation["integrity_checks"]["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_emergency_save_component_gathering(self, emergency_save_manager):
        """Test individual component gathering methods."""
        # Test system state gathering
        system_state = await emergency_save_manager._gather_system_state()
        assert "pid" in system_state
        assert "memory_mb" in system_state
        assert "gathered_at" in system_state

        # Test configuration gathering
        config = await emergency_save_manager._gather_configuration()
        assert "environment" in config
        assert "gathered_at" in config

        # Test progress snapshots gathering
        snapshots = await emergency_save_manager._gather_progress_snapshots()
        assert "active_snapshots" in snapshots
        assert "snapshots" in snapshots
        assert "gathered_at" in snapshots

    @pytest.mark.asyncio
    async def test_emergency_save_atomic_rollback(self, emergency_save_manager, save_context, temp_backup_dir):
        """Test atomic rollback on component failure."""
        # Enable atomic mode
        save_context.atomic = True

        # Mock a component to fail
        with patch.object(emergency_save_manager, '_gather_system_state', side_effect=Exception("Test error")):
            result = await emergency_save_manager.perform_emergency_save(
                context=save_context,
                components=["system_state"]
            )

            # Verify rollback occurred
            assert result.status == "error"
            assert "Test error" in result.error_message
            assert len(result.saved_components) == 0
            assert len(result.failed_components) == 1

            # Verify no files were left behind
            save_dir = temp_backup_dir / save_context.save_id
            assert not save_dir.exists()

    @pytest.mark.asyncio
    async def test_emergency_save_partial_success(self, emergency_save_manager, save_context):
        """Test partial success when atomic mode is disabled."""
        # Disable atomic mode
        save_context.atomic = False

        # Mock one component to fail
        with patch.object(emergency_save_manager, '_gather_system_state', side_effect=Exception("Test error")):
            result = await emergency_save_manager.perform_emergency_save(
                context=save_context,
                components=["system_state", "configuration"]
            )

            # Verify partial success
            assert result.status == "partial"
            assert len(result.saved_components) == 1
            assert len(result.failed_components) == 1
            assert "configuration" in result.saved_components
            assert "system_state" in result.failed_components

    @pytest.mark.asyncio
    async def test_emergency_save_training_sessions_gathering(self, emergency_save_manager):
        """Test training sessions gathering with database mocking."""
        # Mock database session and query results
        with patch('prompt_improver.cli.core.emergency_save.get_session_context') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock query result
            mock_result = MagicMock()
            mock_session_obj = MagicMock()
            mock_session_obj.session_id = "test_session_123"
            mock_session_obj.status = "running"
            mock_session_obj.started_at = datetime.now(timezone.utc)
            mock_session_obj.completed_at = None
            mock_session_obj.total_iterations = 5
            mock_session_obj.continuous_mode = True
            mock_session_obj.improvement_threshold = 0.02
            
            mock_result.scalars.return_value.all.return_value = [mock_session_obj]
            mock_db_session.execute.return_value = mock_result

            # Test training sessions gathering
            training_data = await emergency_save_manager._gather_training_sessions()

            # Verify training data
            assert "total_sessions" in training_data
            assert "sessions" in training_data
            assert training_data["total_sessions"] == 1
            assert len(training_data["sessions"]) == 1
            
            session_data = training_data["sessions"][0]
            assert session_data["session_id"] == "test_session_123"
            assert session_data["status"] == "running"
            assert session_data["total_iterations"] == 5

    @pytest.mark.asyncio
    async def test_emergency_save_concurrent_operations(self, emergency_save_manager):
        """Test concurrent emergency save operations."""
        # Create multiple save contexts
        context1 = EmergencySaveContext(
            save_id="concurrent_save_1",
            triggered_at=datetime.now(timezone.utc),
            trigger_type="manual",
            trigger_details={"test": 1}
        )
        
        context2 = EmergencySaveContext(
            save_id="concurrent_save_2",
            triggered_at=datetime.now(timezone.utc),
            trigger_type="manual",
            trigger_details={"test": 2}
        )

        # Run concurrent saves
        results = await asyncio.gather(
            emergency_save_manager.perform_emergency_save(context1, ["system_state"]),
            emergency_save_manager.perform_emergency_save(context2, ["configuration"]),
            return_exceptions=True
        )

        # Verify both saves completed
        assert len(results) == 2
        assert all(isinstance(r, EmergencySaveResult) for r in results)
        assert all(r.status == "success" for r in results)

        # Verify save history
        assert len(emergency_save_manager.save_history) == 2

    @pytest.mark.asyncio
    async def test_emergency_save_validation_errors(self, emergency_save_manager, save_context, temp_backup_dir):
        """Test validation error detection."""
        # Perform save first
        result = await emergency_save_manager.perform_emergency_save(
            context=save_context,
            components=["system_state"]
        )
        
        # Manually corrupt the saved file
        save_dir = temp_backup_dir / save_context.save_id
        system_file = save_dir / "system_state.json"
        system_file.write_text("invalid json content")

        # Run validation
        validation_results = await emergency_save_manager._validate_emergency_save(
            save_dir, ["system_state"], save_context
        )

        # Verify validation detected errors
        assert len(validation_results["validation_errors"]) > 0
        assert validation_results["integrity_checks"]["success_rate"] < 1.0

    @pytest.mark.asyncio
    async def test_emergency_save_all_components(self, emergency_save_manager, save_context):
        """Test emergency save with all default components."""
        # Perform save with all components
        result = await emergency_save_manager.perform_emergency_save(
            context=save_context,
            components=None  # Should use all default components
        )

        # Verify all components were attempted
        expected_components = [
            "training_sessions",
            "database_state", 
            "system_state",
            "configuration",
            "progress_snapshots"
        ]
        
        total_components = len(result.saved_components) + len(result.failed_components)
        assert total_components == len(expected_components)

    @pytest.mark.asyncio
    async def test_emergency_save_file_atomicity(self, emergency_save_manager, save_context, temp_backup_dir):
        """Test file atomicity during save operations."""
        # Perform emergency save
        result = await emergency_save_manager.perform_emergency_save(
            context=save_context,
            components=["system_state"]
        )

        # Verify no temporary files remain
        temp_files = list(temp_backup_dir.rglob("*.tmp"))
        assert len(temp_files) == 0

        # Verify final files exist
        save_dir = temp_backup_dir / save_context.save_id
        system_file = save_dir / "system_state.json"
        assert system_file.exists()

        # Verify file content is valid JSON
        data = json.loads(system_file.read_text())
        assert data["component"] == "system_state"
