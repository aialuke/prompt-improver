"""
Tests for Week 8 Progress Preservation System Implementation
Tests comprehensive progress preservation, checkpoint creation/restoration, and resource cleanup.
"""

import asyncio
import pytest
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Import directly to avoid CLI module import issues
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from prompt_improver.cli.core.progress_preservation import (
    ProgressPreservationManager,
    ProgressSnapshot
)


class TestProgressPreservationSystem:
    """Test comprehensive progress preservation system for Week 8."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def progress_manager(self, temp_backup_dir):
        """Create progress preservation manager with temporary directory."""
        return ProgressPreservationManager(backup_dir=temp_backup_dir)

    @pytest.fixture
    def sample_snapshot(self):
        """Create sample progress snapshot for testing."""
        return ProgressSnapshot(
            session_id="test_session_123",
            iteration=5,
            timestamp=datetime.now(timezone.utc),
            performance_metrics={
                "model_accuracy": 0.85,
                "rule_effectiveness": 0.78,
                "pattern_coverage": 0.82
            },
            rule_optimizations={
                "clarity_rule": {
                    "rule_name": "Clarity Enhancement Rule",
                    "improvement_score": 0.15,
                    "optimized_parameters": {"threshold": 0.8},
                    "before_metrics": {"clarity_score": 0.65},
                    "after_metrics": {"clarity_score": 0.80}
                }
            },
            synthetic_data_generated=150,
            workflow_state={
                "workflow_id": "workflow_123",
                "duration": 45.5,
                "discovered_patterns": {
                    "pattern_1": {
                        "effectiveness_score": 0.75,
                        "parameters": {"complexity": "medium"}
                    }
                }
            },
            model_checkpoints=["model_checkpoint_1.pkl", "model_checkpoint_2.pkl"],
            improvement_score=0.12
        )

    @pytest.mark.asyncio
    async def test_save_training_progress_file_backup(self, progress_manager, sample_snapshot, temp_backup_dir):
        """Test saving training progress to file backup."""
        # Test saving progress
        result = await progress_manager.save_training_progress(
            session_id=sample_snapshot.session_id,
            iteration=sample_snapshot.iteration,
            performance_metrics=sample_snapshot.performance_metrics,
            rule_optimizations=sample_snapshot.rule_optimizations,
            workflow_state=sample_snapshot.workflow_state,
            synthetic_data_generated=sample_snapshot.synthetic_data_generated,
            model_checkpoints=sample_snapshot.model_checkpoints,
            improvement_score=sample_snapshot.improvement_score
        )

        assert result is True

        # Verify backup file was created
        backup_file = temp_backup_dir / f"{sample_snapshot.session_id}_progress.json"
        assert backup_file.exists()

        # Verify backup file content
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        assert "snapshots" in backup_data
        assert len(backup_data["snapshots"]) == 1

        snapshot_data = backup_data["snapshots"][0]
        assert snapshot_data["session_id"] == sample_snapshot.session_id
        assert snapshot_data["iteration"] == sample_snapshot.iteration
        assert snapshot_data["improvement_score"] == sample_snapshot.improvement_score

    @pytest.mark.asyncio
    async def test_create_checkpoint_comprehensive(self, progress_manager, temp_backup_dir):
        """Test comprehensive checkpoint creation."""
        session_id = "test_session_checkpoint"

        # Mock database session and TrainingSession
        mock_session = MagicMock()
        mock_training_session = MagicMock(spec=TrainingSession)
        mock_training_session.session_id = session_id
        mock_training_session.current_iteration = 10
        mock_training_session.current_performance = 0.85
        mock_training_session.best_performance = 0.87
        mock_training_session.performance_history = [0.75, 0.80, 0.85, 0.87]
        mock_training_session.status = "running"
        mock_training_session.total_training_time_seconds = 3600.0
        mock_training_session.models_trained = 3
        mock_training_session.rules_optimized = 5
        mock_training_session.patterns_discovered = 2
        mock_training_session.active_workflow_id = "workflow_123"
        mock_training_session.workflow_history = ["workflow_1", "workflow_2"]
        mock_training_session.error_count = 0
        mock_training_session.retry_count = 1
        mock_training_session.last_error = None
        mock_training_session.continuous_mode = True
        mock_training_session.max_iterations = None
        mock_training_session.improvement_threshold = 0.02
        mock_training_session.timeout_seconds = 3600
        mock_training_session.auto_init_enabled = True
        mock_training_session.checkpoint_data = None
        mock_training_session.last_checkpoint_at = None
        mock_training_session.started_at = datetime.now(timezone.utc)
        mock_training_session.completed_at = None
        mock_training_session.last_activity_at = datetime.now(timezone.utc)

        # Mock database operations
        with patch('prompt_improver.cli.core.progress_preservation.get_session_context') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session

            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = mock_training_session
            mock_db_session.execute.return_value = mock_result

            # Test checkpoint creation
            checkpoint_id = await progress_manager.create_checkpoint(session_id)

            assert checkpoint_id is not None
            assert checkpoint_id.startswith(f"{session_id}_checkpoint_")

            # Verify checkpoint file was created
            checkpoint_file = temp_backup_dir / f"{checkpoint_id}.json"
            assert checkpoint_file.exists()

            # Verify checkpoint content
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            assert checkpoint_data["checkpoint_id"] == checkpoint_id
            assert checkpoint_data["session_id"] == session_id
            assert "session_data" in checkpoint_data
            assert checkpoint_data["session_data"]["current_iteration"] == 10
            assert checkpoint_data["session_data"]["status"] == "running"

    def test_pid_file_management(self, progress_manager, temp_backup_dir):
        """Test PID file creation, checking, and cleanup."""
        session_id = "test_session_pid"

        # Test PID file creation
        result = progress_manager.create_pid_file(session_id)
        assert result is True

        # Verify PID file exists
        pid_file = temp_backup_dir / f"{session_id}.pid"
        assert pid_file.exists()

        # Verify PID file content
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        assert pid == os.getpid()

        # Test duplicate PID file creation (should fail)
        result = progress_manager.create_pid_file(session_id)
        assert result is False

        # Test PID file removal
        result = progress_manager.remove_pid_file(session_id)
        assert result is True
        assert not pid_file.exists()

    def test_orphaned_session_detection(self, progress_manager, temp_backup_dir):
        """Test detection and cleanup of orphaned sessions."""
        # Create fake PID files with non-existent PIDs
        orphaned_session_1 = "orphaned_session_1"
        orphaned_session_2 = "orphaned_session_2"

        # Create PID files with fake PIDs
        fake_pid_1 = 999999  # Very unlikely to exist
        fake_pid_2 = 999998

        pid_file_1 = temp_backup_dir / f"{orphaned_session_1}.pid"
        pid_file_2 = temp_backup_dir / f"{orphaned_session_2}.pid"

        with open(pid_file_1, 'w') as f:
            f.write(str(fake_pid_1))
        with open(pid_file_2, 'w') as f:
            f.write(str(fake_pid_2))

        # Test orphaned session detection
        orphaned_sessions = progress_manager.check_orphaned_sessions()

        assert len(orphaned_sessions) >= 2
        assert orphaned_session_1 in orphaned_sessions
        assert orphaned_session_2 in orphaned_sessions

    @pytest.mark.asyncio
    async def test_resource_cleanup_comprehensive(self, progress_manager):
        """Test comprehensive resource cleanup."""
        session_id = "test_session_cleanup"

        # Add session to active sessions
        progress_manager.active_sessions[session_id] = MagicMock()

        # Test resource cleanup
        result = await progress_manager.cleanup_resources(session_id)
        assert result is True

        # Verify session was removed from active sessions
        assert session_id not in progress_manager.active_sessions

    @pytest.mark.asyncio
    async def test_session_recovery_from_backup(self, progress_manager, sample_snapshot, temp_backup_dir):
        """Test session recovery from backup files."""
        # First save a snapshot
        await progress_manager.save_training_progress(
            session_id=sample_snapshot.session_id,
            iteration=sample_snapshot.iteration,
            performance_metrics=sample_snapshot.performance_metrics,
            rule_optimizations=sample_snapshot.rule_optimizations,
            workflow_state=sample_snapshot.workflow_state,
            synthetic_data_generated=sample_snapshot.synthetic_data_generated,
            model_checkpoints=sample_snapshot.model_checkpoints,
            improvement_score=sample_snapshot.improvement_score
        )

        # Test recovery
        recovered_snapshot = await progress_manager.recover_session_progress(sample_snapshot.session_id)

        assert recovered_snapshot is not None
        assert recovered_snapshot.session_id == sample_snapshot.session_id
        assert recovered_snapshot.iteration == sample_snapshot.iteration
        assert recovered_snapshot.improvement_score == sample_snapshot.improvement_score

    def test_backup_file_rotation(self, progress_manager, temp_backup_dir):
        """Test backup file rotation to prevent bloat."""
        session_id = "test_session_rotation"

        # Create a backup file with many snapshots
        backup_file = temp_backup_dir / f"{session_id}_progress.json"

        # Create 60 fake snapshots (more than the 50 limit)
        snapshots = []
        for i in range(60):
            snapshot = {
                "session_id": session_id,
                "iteration": i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "improvement_score": 0.5 + (i * 0.01)
            }
            snapshots.append(snapshot)

        backup_data = {"snapshots": snapshots}
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f)

        # Create a new snapshot to trigger rotation
        test_snapshot = ProgressSnapshot(
            session_id=session_id,
            iteration=61,
            timestamp=datetime.now(timezone.utc),
            performance_metrics={"test": 0.9},
            rule_optimizations={},
            synthetic_data_generated=0,
            workflow_state={},
            model_checkpoints=[],
            improvement_score=0.95
        )

        # Save the snapshot (should trigger rotation)
        progress_manager._save_to_backup_file(test_snapshot)

        # Verify rotation occurred
        with open(backup_file, 'r') as f:
            rotated_data = json.load(f)

        # Should have exactly 50 snapshots (49 old + 1 new)
        assert len(rotated_data["snapshots"]) == 50

        # The newest snapshot should be the last one
        newest_snapshot = rotated_data["snapshots"][-1]
        assert newest_snapshot["iteration"] == 61
        assert newest_snapshot["improvement_score"] == 0.95
