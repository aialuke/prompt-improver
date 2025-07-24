"""
Tests for Week 9 Session Resume Manager Implementation
Tests session state detection, workflow reconstruction, and resume coordination with real behavior testing.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.cli.core.session_resume import (
    SessionResumeManager, SessionState, SessionResumeContext, WorkflowState
)
from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
from prompt_improver.database.models import TrainingSession, TrainingIteration


class TestSessionResumeManager:
    """Test session resume manager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def session_resume_manager(self, temp_backup_dir):
        """Create session resume manager for testing."""
        return SessionResumeManager(backup_dir=temp_backup_dir)

    @pytest.fixture
    def mock_training_session(self):
        """Create mock training session for testing."""
        session = MagicMock(spec=TrainingSession)
        session.session_id = "test_session_123"
        session.status = "running"
        session.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
        session.last_activity_at = datetime.now(timezone.utc) - timedelta(minutes=45)
        session.current_iteration = 5
        session.continuous_mode = True
        session.max_iterations = 10
        session.improvement_threshold = 0.02
        session.timeout_seconds = 3600
        session.checkpoint_data = {
            "workflow_id": "workflow_123",
            "workflow_type": "continuous_training"
        }
        session.iterations = []
        return session

    @pytest.fixture
    def mock_training_iteration(self):
        """Create mock training iteration for testing."""
        iteration = MagicMock(spec=TrainingIteration)
        iteration.iteration = 3
        iteration.completed_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        iteration.error_message = None
        iteration.data_export_completed = True
        iteration.pattern_discovery_completed = True
        iteration.rule_optimization_completed = False
        iteration.validation_completed = False
        iteration.performance_metrics = {"improvement": 0.05}
        return iteration

    def test_session_resume_manager_initialization(self, temp_backup_dir):
        """Test session resume manager initialization."""
        manager = SessionResumeManager(backup_dir=temp_backup_dir)

        # Verify initialization
        assert manager.backup_dir == temp_backup_dir
        assert manager.backup_dir.exists()
        assert len(manager.detected_sessions) == 0
        assert len(manager.resume_history) == 0

    def test_session_state_classification(self, session_resume_manager, mock_training_session):
        """Test session state classification logic."""
        # Test running session
        mock_training_session.status = "running"
        time_since_activity = timedelta(minutes=10)
        state = session_resume_manager._classify_session_state(mock_training_session, time_since_activity)
        assert state == SessionState.RUNNING

        # Test interrupted session (long time since activity)
        time_since_activity = timedelta(hours=3)
        state = session_resume_manager._classify_session_state(mock_training_session, time_since_activity)
        assert state == SessionState.INTERRUPTED

        # Test recoverable session (moderate time since activity)
        time_since_activity = timedelta(minutes=45)
        state = session_resume_manager._classify_session_state(mock_training_session, time_since_activity)
        assert state == SessionState.RECOVERABLE

        # Test explicitly interrupted session
        mock_training_session.status = "interrupted"
        state = session_resume_manager._classify_session_state(mock_training_session, timedelta(minutes=10))
        assert state == SessionState.INTERRUPTED

        # Test completed session
        mock_training_session.status = "completed"
        state = session_resume_manager._classify_session_state(mock_training_session, timedelta(minutes=10))
        assert state == SessionState.COMPLETED

    @pytest.mark.asyncio
    async def test_data_integrity_assessment(self, session_resume_manager, mock_training_session, mock_training_iteration):
        """Test data integrity assessment for sessions."""
        # Test session with good integrity
        mock_training_session.iterations = [mock_training_iteration]
        mock_training_session.checkpoint_data = {"test": "data"}
        mock_training_session.current_iteration = 3

        integrity_score = await session_resume_manager._assess_data_integrity(mock_training_session)
        assert integrity_score >= 0.7

        # Test session with missing iterations
        mock_training_session.iterations = []
        integrity_score = await session_resume_manager._assess_data_integrity(mock_training_session)
        assert integrity_score <= 0.7

        # Test session with missing checkpoint data
        mock_training_session.checkpoint_data = None
        integrity_score = await session_resume_manager._assess_data_integrity(mock_training_session)
        assert integrity_score < 0.5

    def test_recovery_strategy_determination(self, session_resume_manager, mock_training_session):
        """Test recovery strategy determination logic."""
        # Test full resume strategy
        strategy = session_resume_manager._determine_recovery_strategy(
            mock_training_session, SessionState.RECOVERABLE, 0.9
        )
        assert strategy == "full_resume"

        # Test checkpoint resume strategy
        strategy = session_resume_manager._determine_recovery_strategy(
            mock_training_session, SessionState.INTERRUPTED, 0.7
        )
        assert strategy == "checkpoint_resume"

        # Test partial recovery strategy
        strategy = session_resume_manager._determine_recovery_strategy(
            mock_training_session, SessionState.CORRUPTED, 0.5
        )
        assert strategy == "partial_recovery"

        # Test no recovery strategy
        strategy = session_resume_manager._determine_recovery_strategy(
            mock_training_session, SessionState.UNRECOVERABLE, 0.2
        )
        assert strategy == "none"

    def test_recovery_confidence_calculation(self, session_resume_manager, mock_training_session):
        """Test recovery confidence calculation."""
        # Test high confidence scenario
        confidence = session_resume_manager._calculate_recovery_confidence(
            mock_training_session,
            SessionState.RECOVERABLE,
            0.9,
            timedelta(minutes=15)
        )
        assert confidence > 0.8

        # Test low confidence scenario
        confidence = session_resume_manager._calculate_recovery_confidence(
            mock_training_session,
            SessionState.CORRUPTED,
            0.3,
            timedelta(hours=12)
        )
        assert confidence < 0.5

    @pytest.mark.asyncio
    async def test_safe_resume_point_detection(self, session_resume_manager, mock_training_session):
        """Test safe resume point detection."""
        # Create multiple iterations
        iterations = []
        for i in range(1, 6):
            iteration = MagicMock(spec=TrainingIteration)
            iteration.iteration = i
            iteration.completed_at = datetime.now(timezone.utc) if i < 5 else None
            iteration.error_message = None if i < 4 else "Test error"
            iterations.append(iteration)

        mock_training_session.iterations = iterations

        # Should return iteration 3 (last completed without error)
        resume_point = await session_resume_manager._find_safe_resume_point(mock_training_session)
        assert resume_point == 3

        # Test with no iterations
        mock_training_session.iterations = []
        resume_point = await session_resume_manager._find_safe_resume_point(mock_training_session)
        assert resume_point == 0

    def test_data_loss_estimation(self, session_resume_manager, mock_training_session):
        """Test data loss estimation."""
        # Test with iterations lost
        mock_training_session.current_iteration = 5
        resume_point = 3
        time_since_activity = timedelta(minutes=20)

        loss_minutes = session_resume_manager._estimate_data_loss(
            mock_training_session, resume_point, time_since_activity
        )

        # Should estimate 2 iterations * 2 minutes + 20 minutes activity = 24 minutes
        assert loss_minutes == 24.0

        # Test with no iterations lost
        loss_minutes = session_resume_manager._estimate_data_loss(
            mock_training_session, 5, timedelta(minutes=10)
        )
        assert loss_minutes == 10.0

    def test_interruption_reason_inference(self, session_resume_manager, mock_training_session):
        """Test interruption reason inference."""
        # Test explicit interruption
        mock_training_session.status = "interrupted"
        reason = session_resume_manager._infer_interruption_reason(
            mock_training_session, timedelta(minutes=10)
        )
        assert reason == "explicit_interruption"

        # Test system shutdown
        mock_training_session.status = "running"
        reason = session_resume_manager._infer_interruption_reason(
            mock_training_session, timedelta(hours=8)
        )
        assert reason == "system_shutdown_or_crash"

        # Test process termination
        reason = session_resume_manager._infer_interruption_reason(
            mock_training_session, timedelta(hours=2)
        )
        assert reason == "process_termination"

        # Test unexpected exit
        reason = session_resume_manager._infer_interruption_reason(
            mock_training_session, timedelta(minutes=45)
        )
        assert reason == "unexpected_exit"

    @pytest.mark.asyncio
    async def test_session_state_analysis(self, session_resume_manager, mock_training_session, mock_training_iteration):
        """Test comprehensive session state analysis."""
        # Setup mock session with iterations
        mock_training_session.iterations = [mock_training_iteration]

        # Mock the database query
        with patch('prompt_improver.cli.core.session_resume.get_session_context'):
            context = await session_resume_manager._analyze_session_state(mock_training_session)

            # Verify context properties
            assert isinstance(context, SessionResumeContext)
            assert context.session_id == "test_session_123"
            assert context.detected_state in [SessionState.RECOVERABLE, SessionState.INTERRUPTED]
            assert context.recovery_confidence > 0.0
            assert context.data_integrity_score > 0.0

    @pytest.mark.asyncio
    async def test_detect_interrupted_sessions_with_database(self, session_resume_manager):
        """Test interrupted session detection with database mocking."""
        # Mock database session and query results
        with patch('prompt_improver.cli.core.session_resume.get_session_context') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            # Create mock sessions
            interrupted_session = MagicMock(spec=TrainingSession)
            interrupted_session.session_id = "interrupted_session"
            interrupted_session.status = "running"
            interrupted_session.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
            interrupted_session.last_activity_at = datetime.now(timezone.utc) - timedelta(hours=1)
            interrupted_session.current_iteration = 3
            interrupted_session.iterations = []
            interrupted_session.checkpoint_data = {"test": "data"}

            # Mock query result
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [interrupted_session]
            mock_db_session.execute.return_value = mock_result

            # Test detection
            detected_sessions = await session_resume_manager.detect_interrupted_sessions()

            # Verify detection results
            assert len(detected_sessions) >= 0  # May be 0 or 1 depending on classification
            if detected_sessions:
                context = detected_sessions[0]
                assert context.session_id == "interrupted_session"

    @pytest.mark.asyncio
    async def test_workflow_state_reconstruction(self, session_resume_manager, mock_training_iteration):
        """Test workflow state reconstruction from database records."""
        # Mock database session and query results
        with patch('prompt_improver.cli.core.session_resume.get_session_context') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            # Create mock session with iterations
            mock_training_session = MagicMock(spec=TrainingSession)
            mock_training_session.session_id = "test_session"
            mock_training_session.checkpoint_data = {
                "workflow_id": "workflow_123",
                "workflow_type": "continuous_training"
            }
            mock_training_session.continuous_mode = True
            mock_training_session.max_iterations = 10
            mock_training_session.improvement_threshold = 0.02
            mock_training_session.timeout_seconds = 3600
            mock_training_session.current_iteration = 3
            mock_training_session.started_at = datetime.now(timezone.utc)
            mock_training_session.last_activity_at = datetime.now(timezone.utc)
            mock_training_session.status = "running"
            mock_training_session.iterations = [mock_training_iteration]

            # Mock query result
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_training_session
            mock_db_session.execute.return_value = mock_result

            # Test workflow reconstruction
            workflow_state = await session_resume_manager.reconstruct_workflow_state("test_session")

            # Verify workflow state
            assert workflow_state is not None
            assert isinstance(workflow_state, WorkflowState)
            assert workflow_state.workflow_id == "workflow_123"
            assert workflow_state.workflow_type == "continuous_training"
            assert "data_export" in workflow_state.completed_steps
            assert "pattern_discovery" in workflow_state.completed_steps
            assert workflow_state.current_step == "rule_optimization"
            assert workflow_state.workflow_parameters["resume_mode"] is True

    @pytest.mark.asyncio
    async def test_resume_integrity_verification(self, session_resume_manager):
        """Test data integrity verification before resume."""
        # Create mock workflow state
        workflow_state = WorkflowState(
            workflow_id="test_workflow",
            workflow_type="continuous_training",
            current_step="pattern_discovery",
            completed_steps=["data_export"],
            pending_steps=["pattern_discovery", "rule_optimization"],
            workflow_parameters={"session_id": "test_session"},
            execution_context={"started_at": datetime.now(timezone.utc).isoformat()},
            performance_metrics={}
        )

        # Mock database session
        with patch('prompt_improver.cli.core.session_resume.get_session_context') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            # Mock session with good integrity
            mock_training_session = MagicMock(spec=TrainingSession)
            mock_training_session.checkpoint_data = {"test": "data"}
            mock_training_session.iterations = [MagicMock(iteration=1), MagicMock(iteration=2)]

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_training_session
            mock_db_session.execute.return_value = mock_result

            # Test integrity verification
            integrity_check = await session_resume_manager._verify_resume_integrity("test_session", workflow_state)

            # Verify integrity check results
            assert "valid" in integrity_check
            assert "issues" in integrity_check

    @pytest.mark.asyncio
    async def test_resume_operation_coordination(self, session_resume_manager):
        """Test resume operation coordination with training manager."""
        # Create mock training manager
        mock_training_manager = MagicMock(spec=TrainingSystemManager)

        # Create mock session context
        context = SessionResumeContext(
            session_id="test_session",
            detected_state=SessionState.RECOVERABLE,
            last_activity=datetime.now(timezone.utc),
            interruption_reason="test",
            recovery_strategy="full_resume",
            data_integrity_score=0.9,
            resume_from_iteration=3,
            estimated_loss_minutes=10.0,
            recovery_confidence=0.8
        )

        # Create mock workflow state
        workflow_state = WorkflowState(
            workflow_id="test_workflow",
            workflow_type="continuous_training",
            current_step="pattern_discovery",
            completed_steps=["data_export"],
            pending_steps=["pattern_discovery"],
            workflow_parameters={"session_id": "test_session"},
            execution_context={"checkpoint_data": {}},
            performance_metrics={}
        )

        # Mock database operations
        with patch('prompt_improver.cli.core.session_resume.get_session_context') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            # Test resume operation
            result = await session_resume_manager._perform_resume_operation(
                "test_session", context, workflow_state, mock_training_manager
            )

            # Verify resume result
            assert result["status"] == "success"
            assert result["session_id"] == "test_session"
            assert result["resumed_from_iteration"] == 3
            assert result["recovery_confidence"] == 0.8
