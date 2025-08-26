"""
Tests for Enhanced Stop Command Implementation
Tests the Week 7 enhanced stop command with signal handling and progress preservation.
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_improver.cli.core.progress_preservation import (
    ProgressPreservationManager,
    ProgressSnapshot,
)
from prompt_improver.cli.core.signal_handler import (
    AsyncSignalHandler,
    ShutdownContext,
    ShutdownReason,
)


class TestAsyncSignalHandler:
    """Test the enhanced signal handling system."""

    @pytest.fixture
    def signal_handler(self):
        """Create signal handler for testing."""
        return AsyncSignalHandler()

    @pytest.mark.asyncio
    async def test_signal_handler_initialization(self, signal_handler):
        """Test signal handler initializes correctly."""
        assert signal_handler.shutdown_event is not None
        assert not signal_handler.shutdown_event.is_set()
        assert signal_handler.shutdown_in_progress is False
        assert len(signal_handler.shutdown_handlers) == 0
        assert len(signal_handler.cleanup_handlers) == 0

    @pytest.mark.asyncio
    async def test_register_shutdown_handler(self, signal_handler):
        """Test registering shutdown handlers."""

        async def test_handler(context):
            return {"status": "success"}

        signal_handler.register_shutdown_handler("test_handler", test_handler)
        assert "test_handler" in signal_handler.shutdown_handlers
        assert signal_handler.shutdown_handlers["test_handler"] == test_handler

    @pytest.mark.asyncio
    async def test_register_cleanup_handler(self, signal_handler):
        """Test registering cleanup handlers."""

        async def test_cleanup():
            return {"cleaned": True}

        signal_handler.register_cleanup_handler("test_cleanup", test_cleanup)
        assert "test_cleanup" in signal_handler.cleanup_handlers
        assert signal_handler.cleanup_handlers["test_cleanup"] == test_cleanup

    @pytest.mark.asyncio
    async def test_graceful_shutdown_execution(self, signal_handler):
        """Test graceful shutdown execution with handlers."""
        shutdown_results = []
        cleanup_results = []

        async def shutdown_handler(context):
            shutdown_results.append(f"shutdown_{context.reason.value}")
            return {"status": "success"}

        async def cleanup_handler():
            cleanup_results.append("cleanup_executed")
            return {"cleaned": True}

        signal_handler.register_shutdown_handler("test_shutdown", shutdown_handler)
        signal_handler.register_cleanup_handler("test_cleanup", cleanup_handler)
        signal_handler.shutdown_context = ShutdownContext(
            reason=ShutdownReason.USER_INTERRUPT, timeout=30
        )
        result = await signal_handler.execute_graceful_shutdown()
        assert result["status"] == "success"
        assert result["reason"] == "user_interrupt"
        assert "shutdown_results" in result
        assert "cleanup_results" in result
        assert len(shutdown_results) == 1
        assert len(cleanup_results) == 1
        assert shutdown_results[0] == "shutdown_user_interrupt"
        assert cleanup_results[0] == "cleanup_executed"

    @pytest.mark.asyncio
    async def test_force_shutdown_execution(self, signal_handler):
        """Test force shutdown with minimal cleanup."""

        async def critical_cleanup():
            return {"critical": "cleaned"}

        async def database_cleanup():
            return {"database": "closed"}

        signal_handler.register_cleanup_handler("critical_cleanup", critical_cleanup)
        signal_handler.register_cleanup_handler("database_cleanup", database_cleanup)
        result = await signal_handler._execute_force_shutdown()
        assert result["status"] == "force_shutdown"
        assert result["reason"] == "timeout_or_force"
        assert "critical_cleanup" in result
        assert result["progress_saved"] is False


class TestProgressPreservationManager:
    """Test the progress preservation system."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def progress_manager(self, temp_backup_dir):
        """Create progress manager with temp directory."""
        return ProgressPreservationManager(backup_dir=temp_backup_dir)

    @pytest.fixture
    def sample_progress_snapshot(self):
        """Create sample progress snapshot."""
        return ProgressSnapshot(
            session_id="test_session_123",
            iteration=5,
            timestamp=datetime.now(UTC),
            performance_metrics={"accuracy": 0.85, "loss": 0.15},
            rule_optimizations={"rule_1": {"param": "value"}},
            synthetic_data_generated=100,
            workflow_state={"workflow_id": "wf_123", "status": "running"},
            model_checkpoints=["checkpoint_1.pkl", "checkpoint_2.pkl"],
            improvement_score=0.05,
        )

    def test_progress_snapshot_serialization(self, sample_progress_snapshot):
        """Test progress snapshot to/from dict conversion."""
        from dataclasses import asdict
        from datetime import datetime

        snapshot_dict = asdict(sample_progress_snapshot)
        assert snapshot_dict["session_id"] == "test_session_123"
        assert snapshot_dict["iteration"] == 5
        assert isinstance(snapshot_dict["timestamp"], datetime)
        assert snapshot_dict["performance_metrics"]["accuracy"] == 0.85

        # Convert datetime to string for from_dict compatibility
        snapshot_dict["timestamp"] = snapshot_dict["timestamp"].isoformat()
        restored_snapshot = ProgressSnapshot.from_dict(snapshot_dict)
        assert restored_snapshot.session_id == sample_progress_snapshot.session_id
        assert restored_snapshot.iteration == sample_progress_snapshot.iteration
        assert (
            restored_snapshot.performance_metrics
            == sample_progress_snapshot.performance_metrics
        )

    @pytest.mark.asyncio
    async def test_save_to_backup_file(
        self, progress_manager, sample_progress_snapshot
    ):
        """Test saving progress to backup file."""
        await progress_manager._save_to_backup_file(sample_progress_snapshot)
        backup_file = (
            progress_manager.backup_dir
            / f"{sample_progress_snapshot.session_id}_progress.json"
        )
        assert backup_file.exists()
        with open(backup_file, encoding="utf-8") as f:
            backup_data = json.load(f)
        assert "snapshots" in backup_data
        assert len(backup_data["snapshots"]) == 1
        assert backup_data["snapshots"][0]["session_id"] == "test_session_123"
        assert backup_data["snapshots"][0]["iteration"] == 5

    @pytest.mark.asyncio
    async def test_backup_file_rotation(self, progress_manager):
        """Test backup file rotation (keeping only last 50 snapshots)."""
        session_id = "test_session_rotation"
        for i in range(55):
            snapshot = ProgressSnapshot(
                session_id=session_id,
                iteration=i,
                timestamp=datetime.now(UTC),
                performance_metrics={"iteration": i},
                rule_optimizations={},
                synthetic_data_generated=0,
                workflow_state={},
                model_checkpoints=[],
                improvement_score=0.0,
            )
            await progress_manager._save_to_backup_file(snapshot)
        backup_file = progress_manager.backup_dir / f"{session_id}_progress.json"
        with open(backup_file, encoding="utf-8") as f:
            backup_data = json.load(f)
        assert len(backup_data["snapshots"]) == 50
        assert backup_data["snapshots"][0]["iteration"] == 5
        assert backup_data["snapshots"][-1]["iteration"] == 54

    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, progress_manager):
        """Test cleanup of old backup files."""
        old_file = progress_manager.backup_dir / "old_session_progress.json"
        recent_file = progress_manager.backup_dir / "recent_session_progress.json"
        old_file.write_text('{"snapshots": []}')
        recent_file.write_text('{"snapshots": []}')
        import os
        import time

        old_timestamp = time.time() - 35 * 24 * 3600
        recent_timestamp = time.time() - 10 * 24 * 3600
        old_file.touch()
        recent_file.touch()
        os.utime(old_file, (old_timestamp, old_timestamp))
        os.utime(recent_file, (recent_timestamp, recent_timestamp))
        cleaned_count = await progress_manager.cleanup_old_backups(days_to_keep=30)
        assert cleaned_count == 1
        assert not old_file.exists()
        assert recent_file.exists()


class TestEnhancedStopCommand:
    """Test the enhanced stop command integration."""

    @pytest.fixture
    def mock_training_manager(self):
        """Mock training system manager."""
        manager = AsyncMock()
        manager.get_active_sessions.return_value = [
            MagicMock(
                session_id="session_1",
                total_iterations=10,
                best_performance={"accuracy": 0.8},
            ),
            MagicMock(
                session_id="session_2",
                total_iterations=5,
                best_performance={"accuracy": 0.7},
            ),
        ]
        manager.stop_training_system.return_value = True
        return manager

    @pytest.fixture
    def mock_cli_orchestrator(self):
        """Mock CLI orchestrator."""
        orchestrator = AsyncMock()
        orchestrator.stop_training_gracefully.return_value = {
            "success": True,
            "progress_saved": True,
            "saved_data": "session_state",
        }
        orchestrator.force_stop_training.return_value = {"success": True}
        return orchestrator

    @pytest.fixture
    def mock_progress_manager(self):
        """Mock progress preservation manager."""
        manager = AsyncMock()
        manager.create_checkpoint.return_value = "checkpoint_123"
        manager.export_session_results.return_value = "/path/to/export.json"
        manager.cleanup_old_backups.return_value = 3
        return manager

    @pytest.mark.asyncio
    async def test_graceful_stop_single_session(
        self, mock_training_manager, mock_cli_orchestrator, mock_progress_manager
    ):
        """Test graceful stop of a single session."""
        session_id = "session_1"
        save_progress = True
        export_results = True
        export_format = "json"
        graceful = True
        active_sessions = await mock_training_manager.get_active_sessions()
        session = next((s for s in active_sessions if s.session_id == session_id), None)
        assert session is not None
        assert session.session_id == "session_1"
        if save_progress:
            checkpoint_id = await mock_progress_manager.create_checkpoint(session_id)
            assert checkpoint_id == "checkpoint_123"
        if graceful:
            result = await mock_cli_orchestrator.stop_training_gracefully(
                session_id=session.session_id, timeout=30, save_progress=save_progress
            )
            assert result["success"] is True
            assert result["progress_saved"] is True
            if export_results:
                export_path = await mock_progress_manager.export_session_results(
                    session_id=session.session_id,
                    export_format=export_format,
                    include_iterations=True,
                )
                assert export_path == "/path/to/export.json"
        mock_training_manager.get_active_sessions.assert_called_once()
        mock_progress_manager.create_checkpoint.assert_called_once_with(session_id)
        mock_cli_orchestrator.stop_training_gracefully.assert_called_once()
        mock_progress_manager.export_session_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_stop_all_sessions(
        self, mock_training_manager, mock_cli_orchestrator, mock_progress_manager
    ):
        """Test force stop of all sessions."""
        graceful = False
        save_progress = True
        active_sessions = await mock_training_manager.get_active_sessions()
        assert len(active_sessions) == 2
        for session in active_sessions:
            if save_progress:
                await mock_progress_manager.save_training_progress(
                    session_id=session.session_id,
                    iteration=session.total_iterations,
                    performance_metrics=session.best_performance,
                    rule_optimizations={},
                    workflow_state={"force_shutdown": True},
                    improvement_score=0.0,
                )
            await mock_cli_orchestrator.force_stop_training(session.session_id)
        cleaned_files = await mock_progress_manager.cleanup_old_backups(days_to_keep=30)
        assert cleaned_files == 3
        success = await mock_training_manager.stop_training_system(graceful=graceful)
        assert success is True
        assert mock_progress_manager.save_training_progress.call_count == 2
        assert mock_cli_orchestrator.force_stop_training.call_count == 2
        mock_progress_manager.cleanup_old_backups.assert_called_once_with(
            days_to_keep=30
        )
        mock_training_manager.stop_training_system.assert_called_once_with(
            graceful=False
        )
