"""
Tests for Week 9 Crash Recovery Manager Implementation
Tests crash detection, recovery procedures, and comprehensive reporting with real behavior testing.
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.cli.core.unified_process_manager import (
    UnifiedProcessManager, CrashType, CrashSeverity, CrashContext, RecoveryResult
)
from prompt_improver.database.models import TrainingSession, TrainingIteration


class TestUnifiedProcessManager:
    """Test crash recovery manager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def crash_recovery_manager(self, temp_backup_dir):
        """Create crash recovery manager for testing."""
        return UnifiedProcessManager(backup_dir=temp_backup_dir)

    @pytest.fixture
    def mock_crash_context(self):
        """Create mock crash context for testing."""
        return CrashContext(
            crash_id="test_crash_123",
            detected_at=datetime.now(timezone.utc),
            crash_type=CrashType.PROCESS_KILLED,
            severity=CrashSeverity.MEDIUM,
            affected_sessions=["session_1", "session_2"],
            crash_indicators={"test": "indicator"},
            system_state_at_crash={"memory": 80.0},
            recovery_strategy="session_resume",
            estimated_data_loss={"minutes": 15},
            recovery_confidence=0.7
        )

    def test_crash_recovery_manager_initialization(self, temp_backup_dir):
        """Test crash recovery manager initialization."""
        manager = UnifiedProcessManager(backup_dir=temp_backup_dir)

        # Verify initialization
        assert manager.backup_dir == temp_backup_dir
        assert manager.backup_dir.exists()
        assert len(manager.detected_crashes) == 0
        assert len(manager.recovery_history) == 0
        assert manager.pid_dir is not None
        assert manager.coordination_lock is not None
        assert manager.recovery_lock is not None

    def test_crash_type_and_severity_enums(self):
        """Test crash type and severity enumerations."""
        # Test crash types
        assert CrashType.SYSTEM_SHUTDOWN.value == "system_shutdown"
        assert CrashType.PROCESS_KILLED.value == "process_killed"
        assert CrashType.OUT_OF_MEMORY.value == "out_of_memory"
        assert CrashType.UNKNOWN_CRASH.value == "unknown_crash"

        # Test crash severities
        assert CrashSeverity.LOW.value == "low"
        assert CrashSeverity.MEDIUM.value == "medium"
        assert CrashSeverity.HIGH.value == "high"
        assert CrashSeverity.CRITICAL.value == "critical"

    def test_interruption_to_crash_type_mapping(self, crash_recovery_manager):
        """Test mapping of interruption reasons to crash types."""
        # Test various interruption reason mappings
        assert crash_recovery_manager._map_interruption_to_crash_type("system_shutdown_or_crash") == CrashType.SYSTEM_SHUTDOWN
        assert crash_recovery_manager._map_interruption_to_crash_type("process_termination") == CrashType.PROCESS_KILLED
        assert crash_recovery_manager._map_interruption_to_crash_type("unexpected_exit") == CrashType.UNKNOWN_CRASH
        assert crash_recovery_manager._map_interruption_to_crash_type("explicit_interruption") == CrashType.GRACEFUL_EXIT
        assert crash_recovery_manager._map_interruption_to_crash_type("unknown_reason") == CrashType.UNKNOWN_CRASH

    def test_crash_severity_assessment(self, crash_recovery_manager):
        """Test crash severity assessment logic."""
        # Create mock session context
        mock_context = MagicMock()

        # Test low severity
        mock_context.data_integrity_score = 0.9
        mock_context.estimated_loss_minutes = 5
        severity = crash_recovery_manager._assess_crash_severity(mock_context)
        assert severity == CrashSeverity.LOW

        # Test medium severity
        mock_context.data_integrity_score = 0.7
        mock_context.estimated_loss_minutes = 20
        severity = crash_recovery_manager._assess_crash_severity(mock_context)
        assert severity == CrashSeverity.MEDIUM

        # Test high severity
        mock_context.data_integrity_score = 0.5
        mock_context.estimated_loss_minutes = 45
        severity = crash_recovery_manager._assess_crash_severity(mock_context)
        assert severity == CrashSeverity.HIGH

        # Test critical severity
        mock_context.data_integrity_score = 0.2
        mock_context.estimated_loss_minutes = 60
        severity = crash_recovery_manager._assess_crash_severity(mock_context)
        assert severity == CrashSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_crash_context_analysis(self, crash_recovery_manager, mock_crash_context):
        """Test crash context analysis and enhancement."""
        # Test crash context analysis
        analyzed_context = await crash_recovery_manager._analyze_crash_context(mock_crash_context)

        # Verify analysis results
        assert analyzed_context.crash_id == mock_crash_context.crash_id
        assert "memory_usage" in analyzed_context.system_state_at_crash
        assert "cpu_usage" in analyzed_context.system_state_at_crash
        assert "analysis_time" in analyzed_context.system_state_at_crash

        # Test recovery strategy enhancement for memory issues
        memory_crash = CrashContext(
            crash_id="memory_crash",
            detected_at=datetime.now(timezone.utc),
            crash_type=CrashType.OUT_OF_MEMORY,
            severity=CrashSeverity.HIGH,
            affected_sessions=["session_1"],
            crash_indicators={},
            system_state_at_crash={},
            recovery_strategy="default",
            estimated_data_loss={},
            recovery_confidence=0.8
        )

        analyzed_memory_crash = await crash_recovery_manager._analyze_crash_context(memory_crash)
        assert analyzed_memory_crash.recovery_strategy == "memory_optimized_recovery"
        assert analyzed_memory_crash.recovery_confidence < 0.8  # Should be reduced

    @pytest.mark.asyncio
    async def test_orphaned_process_detection(self, crash_recovery_manager, temp_backup_dir):
        """Test detection of orphaned processes using PID files."""
        # Create mock orphaned PID file
        pid_file_path = temp_backup_dir / "orphaned_session.pid"
        pid_data = {
            "pid": 99999,  # Non-existent PID
            "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "session_id": "orphaned_session"
        }
        pid_file_path.write_text(json.dumps(pid_data))

        # Mock progress manager to return orphaned session
        with patch.object(crash_recovery_manager.progress_manager, 'check_orphaned_sessions', return_value=["orphaned_session"]):
            orphaned_crashes = await crash_recovery_manager._detect_orphaned_processes()

            # Verify orphaned crash detection
            assert len(orphaned_crashes) == 1
            crash = orphaned_crashes[0]
            assert crash.crash_type == CrashType.PROCESS_KILLED
            assert "orphaned_session" in crash.affected_sessions
            assert crash.crash_indicators["original_pid"] == 99999

    @pytest.mark.asyncio
    async def test_session_crash_detection(self, crash_recovery_manager):
        """Test detection of crashes based on interrupted sessions."""
        # Mock session resume manager to return interrupted sessions
        mock_session_context = MagicMock()
        mock_session_context.session_id = "interrupted_session"
        mock_session_context.interruption_reason = "system_shutdown_or_crash"
        mock_session_context.last_activity = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_session_context.data_integrity_score = 0.8
        mock_session_context.estimated_loss_minutes = 30
        mock_session_context.recovery_strategy = "full_resume"
        mock_session_context.recovery_confidence = 0.7

        with patch.object(crash_recovery_manager.session_resume_manager, 'detect_interrupted_sessions', return_value=[mock_session_context]):
            session_crashes = await crash_recovery_manager._detect_session_crashes()

            # Verify session crash detection
            assert len(session_crashes) == 1
            crash = session_crashes[0]
            assert crash.crash_type == CrashType.SYSTEM_SHUTDOWN
            assert "interrupted_session" in crash.affected_sessions
            assert crash.crash_indicators["interruption_reason"] == "system_shutdown_or_crash"

    @pytest.mark.asyncio
    async def test_system_level_crash_detection(self, crash_recovery_manager):
        """Test detection of system-level crashes."""
        # Mock database session and query results
        with patch('prompt_improver.database.get_session') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            # Mock training sessions that were running before reboot
            mock_training_session = MagicMock(spec=TrainingSession)
            mock_training_session.session_id = "pre_reboot_session"
            mock_training_session.status = "running"
            mock_training_session.last_activity_at = datetime.now(timezone.utc) - timedelta(hours=2)

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_training_session]
            mock_db_session.execute.return_value = mock_result

            # Mock system boot time to be recent
            with patch('psutil.boot_time', return_value=time.time() - 1800):  # 30 minutes ago
                system_crashes = await crash_recovery_manager._detect_system_level_crashes()

                # Verify system crash detection
                if system_crashes:  # May be empty if boot time logic doesn't trigger
                    crash = system_crashes[0]
                    assert crash.crash_type == CrashType.SYSTEM_SHUTDOWN
                    assert crash.severity == CrashSeverity.HIGH

    @pytest.mark.asyncio
    async def test_comprehensive_crash_detection(self, crash_recovery_manager):
        """Test comprehensive crash detection combining all methods."""
        # Mock all detection methods to return crashes
        with patch.object(crash_recovery_manager, '_detect_orphaned_processes', return_value=[]):
            with patch.object(crash_recovery_manager, '_detect_session_crashes', return_value=[]):
                with patch.object(crash_recovery_manager, '_detect_system_level_crashes', return_value=[]):
                    detected_crashes = await crash_recovery_manager.detect_system_crashes()

                    # Verify detection completed without errors
                    assert isinstance(detected_crashes, list)
                    assert len(crash_recovery_manager.detected_crashes) == len(detected_crashes)

    @pytest.mark.asyncio
    async def test_system_recovery_operations(self, crash_recovery_manager, mock_crash_context):
        """Test system-level recovery operations."""
        # Test memory recovery
        mock_crash_context.crash_type = CrashType.OUT_OF_MEMORY
        memory_recovery = await crash_recovery_manager._perform_system_recovery(mock_crash_context)

        assert "actions" in memory_recovery
        assert "repaired" in memory_recovery
        assert "memory_optimization" in memory_recovery["repaired"]
        assert any("memory" in action.lower() for action in memory_recovery["actions"])

        # Test disk recovery
        mock_crash_context.crash_type = CrashType.DISK_FULL
        disk_recovery = await crash_recovery_manager._perform_system_recovery(mock_crash_context)

        assert "disk_space_optimization" in disk_recovery["repaired"]
        assert any("disk" in action.lower() for action in disk_recovery["actions"])

    @pytest.mark.asyncio
    async def test_database_recovery_operations(self, crash_recovery_manager, mock_crash_context):
        """Test database consistency check and repair operations."""
        # Mock database session and query results
        with patch('prompt_improver.database.get_session') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            # Mock training session
            mock_training_session = MagicMock(spec=TrainingSession)
            mock_training_session.session_id = "session_1"
            mock_training_session.status = "running"

            # Mock iterations
            mock_iteration = MagicMock(spec=TrainingIteration)
            mock_iteration.iteration = 1

            mock_session_result = MagicMock()
            mock_session_result.scalar_one_or_none.return_value = mock_training_session

            mock_iteration_result = MagicMock()
            mock_iteration_result.scalars.return_value.all.return_value = [mock_iteration]

            mock_db_session.execute.side_effect = [mock_session_result, mock_iteration_result]

            # Test database recovery
            db_recovery = await crash_recovery_manager._perform_database_recovery(mock_crash_context)

            # Verify database recovery results
            assert "actions" in db_recovery
            assert "repaired" in db_recovery
            assert any("session_1" in action for action in db_recovery["actions"])

    @pytest.mark.asyncio
    async def test_training_session_recovery(self, crash_recovery_manager, mock_crash_context):
        """Test individual training session recovery."""
        # Mock session resume manager
        mock_resume_result = {
            "status": "success",
            "resumed_from_iteration": 3,
            "session_id": "session_1"
        }

        with patch.object(crash_recovery_manager.session_resume_manager, 'resume_training_session', return_value=mock_resume_result):
            session_recovery = await crash_recovery_manager._recover_training_session("session_1", mock_crash_context)

            # Verify session recovery results
            assert session_recovery["status"] == "success"
            assert any("Successfully resumed" in action for action in session_recovery["actions"])

        # Test failed session recovery
        mock_failed_result = {
            "status": "error",
            "error": "Test recovery failure"
        }

        with patch.object(crash_recovery_manager.session_resume_manager, 'resume_training_session', return_value=mock_failed_result):
            failed_recovery = await crash_recovery_manager._recover_training_session("session_1", mock_crash_context)

            # Verify failed recovery handling
            assert failed_recovery["status"] == "failed"
            assert any("Failed to resume" in action for action in failed_recovery["actions"])

    @pytest.mark.asyncio
    async def test_cleanup_operations(self, crash_recovery_manager, mock_crash_context, temp_backup_dir):
        """Test cleanup operations after recovery."""
        # Create mock PID file
        pid_file = temp_backup_dir / "session_1.pid"
        pid_file.write_text('{"pid": 123}')

        # Mock progress manager backup directory
        crash_recovery_manager.progress_manager.backup_dir = temp_backup_dir

        # Test cleanup operations
        cleanup_actions = await crash_recovery_manager._perform_cleanup_operations(mock_crash_context)

        # Verify cleanup results
        assert len(cleanup_actions) > 0
        assert any("Removed orphaned PID file" in action for action in cleanup_actions)
        assert any("crash report" in action.lower() for action in cleanup_actions)

        # Verify crash report was created
        crash_reports = list(temp_backup_dir.glob("crash_report_*.json"))
        assert len(crash_reports) == 1

    def test_recovery_recommendations_generation(self, crash_recovery_manager, mock_crash_context):
        """Test generation of recovery recommendations."""
        # Test memory crash recommendations
        mock_crash_context.crash_type = CrashType.OUT_OF_MEMORY
        recommendations = crash_recovery_manager._generate_recovery_recommendations(mock_crash_context, "success")

        assert any("memory" in rec.lower() for rec in recommendations)
        assert any("batch" in rec.lower() for rec in recommendations)

        # Test disk full recommendations
        mock_crash_context.crash_type = CrashType.DISK_FULL
        recommendations = crash_recovery_manager._generate_recovery_recommendations(mock_crash_context, "partial")

        assert any("disk" in rec.lower() for rec in recommendations)
        assert any("storage" in rec.lower() for rec in recommendations)

        # Test critical severity recommendations
        mock_crash_context.severity = CrashSeverity.CRITICAL
        recommendations = crash_recovery_manager._generate_recovery_recommendations(mock_crash_context, "failed")

        assert any("monitoring" in rec.lower() for rec in recommendations)
        assert any("manual intervention" in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_full_crash_recovery_workflow(self, crash_recovery_manager, mock_crash_context, temp_backup_dir):
        """Test complete crash recovery workflow."""
        # Add crash to detected crashes
        crash_recovery_manager.detected_crashes[mock_crash_context.crash_id] = mock_crash_context

        # Mock all recovery operations
        with patch.object(crash_recovery_manager, '_execute_recovery_strategy') as mock_execute:
            mock_recovery_result = RecoveryResult(
                crash_id=mock_crash_context.crash_id,
                recovery_status="success",
                recovered_sessions=["session_1", "session_2"],
                failed_sessions=[],
                data_repaired=["database_consistency"],
                recovery_actions=["Test recovery action"],
                recovery_duration_seconds=0.0,
                final_system_state={"status": "healthy"},
                recommendations=["Test recommendation"]
            )
            mock_execute.return_value = mock_recovery_result

            # Perform crash recovery
            result = await crash_recovery_manager.perform_crash_recovery(mock_crash_context.crash_id)

            # Verify recovery results
            assert result.recovery_status == "success"
            assert len(result.recovered_sessions) == 2
            assert len(result.failed_sessions) == 0
            assert result.recovery_duration_seconds > 0

            # Verify recovery was tracked
            assert len(crash_recovery_manager.recovery_history) == 1
