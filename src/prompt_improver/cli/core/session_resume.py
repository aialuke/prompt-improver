"""Session Resume Manager for Training Session Recovery
Implements session state detection, workflow reconstruction, and resume coordination.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import select, text, update

from prompt_improver.cli.core.progress_preservation import ProgressService
from prompt_improver.cli.services.training_orchestrator import (
    TrainingOrchestrator as TrainingService,
)
from prompt_improver.database.models import TrainingSession


class SessionState(Enum):
    """Enumeration of training session states for resume detection."""

    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    RECOVERABLE = "recoverable"
    UNRECOVERABLE = "unrecoverable"


@dataclass
class SessionResumeContext:
    """Context information for session resume operations."""

    session_id: str
    detected_state: SessionState
    last_activity: datetime
    interruption_reason: str | None
    recovery_strategy: str
    data_integrity_score: float
    resume_from_iteration: int
    estimated_loss_minutes: float
    recovery_confidence: float


@dataclass
class WorkflowState:
    """Reconstructed workflow state for resume operations."""

    workflow_id: str | None
    workflow_type: str
    current_step: str
    completed_steps: list[str]
    pending_steps: list[str]
    workflow_parameters: dict[str, Any]
    execution_context: dict[str, Any]
    performance_metrics: dict[str, Any]


class SessionService:
    """Session service implementing Clean Architecture patterns for training session resume capabilities with state detection and workflow reconstruction.

    Features:
    - Automatic detection of interrupted training sessions
    - Session state classification and recovery assessment
    - Workflow state reconstruction from database records
    - Data integrity verification for safe resume operations
    - Integration with TrainingService and ProgressService
    - Comprehensive resume coordination and monitoring
    """

    def __init__(self, backup_dir: Path | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("./session_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.progress_manager = ProgressService(backup_dir=self.backup_dir)
        self.detected_sessions: dict[str, SessionResumeContext] = {}
        self.resume_history: list[dict[str, Any]] = []
        self.resume_lock = asyncio.Lock()
        self.detection_lock = asyncio.Lock()

    async def detect_interrupted_sessions(self) -> list[SessionResumeContext]:
        """Detect interrupted training sessions that can be resumed.

        Returns:
            List of session resume contexts for interrupted sessions
        """
        async with self.detection_lock:
            self.logger.info("Detecting interrupted training sessions")
            try:
                async with self.session_manager.session_context() as db_session:
                    result = await db_session.execute(select(TrainingSession))
                    all_sessions = result.scalars().all()
                    cutoff_time = datetime.now(UTC) - timedelta(minutes=30)
                    sessions = [session for session in all_sessions if session.status in {"running", "interrupted"} or (
                            session.status == "running"
                            and session.last_activity_at < cutoff_time
                        )]
                interrupted_sessions = []
                for session in sessions:
                    context = await self._analyze_session_state(session)
                    if context.detected_state in {
                        SessionState.INTERRUPTED,
                        SessionState.RECOVERABLE,
                    }:
                        interrupted_sessions.append(context)
                        self.detected_sessions[session.session_id] = context
                        self.logger.info(
                            f"Detected interrupted session: {session.session_id} (state: {context.detected_state.value}, confidence: {context.recovery_confidence:.2f})"
                        )
                self.logger.info(
                    f"Found {len(interrupted_sessions)} interrupted sessions"
                )
                return interrupted_sessions
            except Exception as e:
                self.logger.exception(f"Failed to detect interrupted sessions: {e}")
                return []

    async def _analyze_session_state(
        self, session: TrainingSession
    ) -> SessionResumeContext:
        """Analyze session state to determine if it can be resumed."""
        try:
            last_activity = session.last_activity_at or session.started_at
            time_since_activity = datetime.now(UTC) - last_activity
            detected_state = self._classify_session_state(session, time_since_activity)
            integrity_score = await self._assess_data_integrity(session)
            recovery_strategy = self._determine_recovery_strategy(
                session, detected_state, integrity_score
            )
            recovery_confidence = self._calculate_recovery_confidence(
                session, detected_state, integrity_score, time_since_activity
            )
            resume_from_iteration = await self._find_safe_resume_point(session)
            estimated_loss = self._estimate_data_loss(
                session, resume_from_iteration, time_since_activity
            )
            return SessionResumeContext(
                session_id=session.session_id,
                detected_state=detected_state,
                last_activity=last_activity,
                interruption_reason=self._infer_interruption_reason(
                    session, time_since_activity
                ),
                recovery_strategy=recovery_strategy,
                data_integrity_score=integrity_score,
                resume_from_iteration=resume_from_iteration,
                estimated_loss_minutes=estimated_loss,
                recovery_confidence=recovery_confidence,
            )
        except Exception as e:
            self.logger.exception(f"Failed to analyze session {session.session_id}: {e}")
            return SessionResumeContext(
                session_id=session.session_id,
                detected_state=SessionState.UNRECOVERABLE,
                last_activity=datetime.now(UTC),
                interruption_reason=f"Analysis failed: {e}",
                recovery_strategy="none",
                data_integrity_score=0.0,
                resume_from_iteration=0,
                estimated_loss_minutes=0.0,
                recovery_confidence=0.0,
            )

    def _classify_session_state(
        self, session: TrainingSession, time_since_activity: timedelta
    ) -> SessionState:
        """Classify session state based on database information and timing."""
        if session.status == "completed":
            return SessionState.COMPLETED
        if session.status == "failed":
            return SessionState.FAILED
        if session.status == "interrupted":
            return SessionState.INTERRUPTED
        if session.status == "running":
            if time_since_activity > timedelta(hours=2):
                return SessionState.INTERRUPTED
            if time_since_activity > timedelta(minutes=30):
                return SessionState.RECOVERABLE
            return SessionState.RUNNING
        return SessionState.CORRUPTED

    async def _assess_data_integrity(self, session: TrainingSession) -> float:
        """Assess data integrity for the session."""
        try:
            integrity_score = 1.0
            if not session.iterations:
                integrity_score -= 0.3
            if not session.checkpoint_data:
                integrity_score -= 0.2
            if session.iterations:
                expected_iterations = session.current_iteration
                actual_iterations = len(session.iterations)
                if actual_iterations < expected_iterations * 0.8:
                    integrity_score -= 0.3
            if session.last_activity_at:
                time_since_activity = datetime.now(UTC) - session.last_activity_at
                if time_since_activity > timedelta(hours=24):
                    integrity_score -= 0.2
            return max(0.0, integrity_score)
        except Exception as e:
            self.logger.warning(
                f"Failed to assess data integrity for {session.session_id}: {e}"
            )
            return 0.5

    def _determine_recovery_strategy(
        self, session: TrainingSession, state: SessionState, integrity: float
    ) -> str:
        """Determine the best recovery strategy for the session."""
        session_age = datetime.now(UTC) - session.started_at
        if state == SessionState.UNRECOVERABLE or integrity < 0.3:
            return "none"
        if state == SessionState.CORRUPTED or integrity < 0.6:
            return "partial_recovery"
        if state in {SessionState.INTERRUPTED, SessionState.RECOVERABLE}:
            if session_age.total_seconds() > 86400:
                return "partial_recovery"
            if integrity > 0.8:
                return "full_resume"
            return "checkpoint_resume"
        return "none"

    def _calculate_recovery_confidence(
        self,
        session: TrainingSession,
        state: SessionState,
        integrity: float,
        time_since_activity: timedelta,
    ) -> float:
        """Calculate confidence score for successful recovery."""
        state_confidence = {
            SessionState.RECOVERABLE: 0.9,
            SessionState.INTERRUPTED: 0.7,
            SessionState.RUNNING: 0.95,
            SessionState.CORRUPTED: 0.3,
            SessionState.FAILED: 0.2,
            SessionState.UNRECOVERABLE: 0.0,
        }.get(state, 0.1)
        integrity_confidence = integrity
        hours_since_activity = time_since_activity.total_seconds() / 3600
        time_confidence = max(0.1, 1.0 - hours_since_activity / 24)
        checkpoint_confidence = 0.8 if session.checkpoint_data else 0.5
        confidence = (
            state_confidence * 0.3
            + integrity_confidence * 0.3
            + time_confidence * 0.2
            + checkpoint_confidence * 0.2
        )
        return min(1.0, max(0.0, confidence))

    async def _find_safe_resume_point(self, session: TrainingSession) -> int:
        """Find the safest iteration to resume from."""
        try:
            if not session.iterations:
                return 0
            valid_iterations = [
                iteration
                for iteration in session.iterations
                if iteration.completed_at is not None
                and iteration.error_message is None
            ]
            if valid_iterations:
                valid_iterations.sort(key=lambda x: x.iteration)
                return valid_iterations[-1].iteration
            return 0
        except Exception as e:
            self.logger.warning(
                f"Failed to find safe resume point for {session.session_id}: {e}"
            )
            return 0

    def _estimate_data_loss(
        self,
        session: TrainingSession,
        resume_point: int,
        time_since_activity: timedelta,
    ) -> float:
        """Estimate potential data loss in minutes."""
        try:
            iterations_lost = max(0, session.current_iteration - resume_point)
            estimated_iteration_time = 2.0
            iteration_loss_minutes = iterations_lost * estimated_iteration_time
            activity_loss_minutes = min(30.0, time_since_activity.total_seconds() / 60)
            return iteration_loss_minutes + activity_loss_minutes
        except Exception as e:
            self.logger.warning(
                f"Failed to estimate data loss for {session.session_id}: {e}"
            )
            return 0.0

    def _infer_interruption_reason(
        self, session: TrainingSession, time_since_activity: timedelta
    ) -> str | None:
        """Infer the likely reason for session interruption."""
        if session.status == "interrupted":
            return "explicit_interruption"
        if time_since_activity > timedelta(hours=6):
            return "system_shutdown_or_crash"
        if time_since_activity > timedelta(hours=1):
            return "process_termination"
        if time_since_activity > timedelta(minutes=30):
            return "unexpected_exit"
        return "recent_activity"

    async def reconstruct_workflow_state(self, session_id: str) -> WorkflowState | None:
        """Reconstruct workflow state from database records for resume operations.

        Args:
            session_id: Training session identifier

        Returns:
            Reconstructed workflow state or None if reconstruction fails
        """
        try:
            self.logger.info(f"Reconstructing workflow state for session: {session_id}")
            async with self.session_manager.session_context() as db_session:
                result = await db_session.execute(
                    select(TrainingSession).where(text(f"session_id = '{session_id}'"))
                )
                session = result.scalar_one_or_none()
                if session:
                    from prompt_improver.database.models import TrainingIteration

                    iterations_result = await db_session.execute(
                        select(TrainingIteration).where(
                            text(f"session_id = '{session_id}'")
                        )
                    )
                    session.iterations = list(iterations_result.scalars().all())
                if not session:
                    self.logger.error(f"Session not found: {session_id}")
                    return None
                checkpoint_data = session.checkpoint_data or {}
                workflow_id = checkpoint_data.get("workflow_id")
                workflow_type = checkpoint_data.get(
                    "workflow_type", "continuous_training"
                )
                completed_steps = []
                current_step = "initialization"
                pending_steps = [
                    "data_export",
                    "pattern_discovery",
                    "rule_optimization",
                    "validation",
                ]
                if session.iterations:
                    latest_iteration = max(
                        session.iterations, key=lambda x: x.iteration
                    )
                    if latest_iteration.synthetic_data_generated > 0:
                        completed_steps.append("data_export")
                        current_step = "pattern_discovery"
                        if "data_export" in pending_steps:
                            pending_steps.remove("data_export")
                    if latest_iteration.discovered_patterns:
                        completed_steps.append("pattern_discovery")
                        current_step = "rule_optimization"
                        if "pattern_discovery" in pending_steps:
                            pending_steps.remove("pattern_discovery")
                    if latest_iteration.rule_optimizations:
                        completed_steps.append("rule_optimization")
                        current_step = "validation"
                        if "rule_optimization" in pending_steps:
                            pending_steps.remove("rule_optimization")
                    if latest_iteration.completed_at:
                        completed_steps.append("validation")
                        current_step = "completed"
                        if "validation" in pending_steps:
                            pending_steps.remove("validation")
                workflow_parameters = {
                    "session_id": session_id,
                    "continuous_mode": session.continuous_mode,
                    "max_iterations": session.max_iterations,
                    "improvement_threshold": session.improvement_threshold,
                    "timeout_seconds": session.timeout_seconds,
                    "current_iteration": session.current_iteration,
                    "resume_mode": True,
                }
                execution_context = {
                    "started_at": session.started_at.isoformat(),
                    "last_activity_at": session.last_activity_at.isoformat()
                    if session.last_activity_at
                    else None,
                    "total_iterations_completed": len(session.iterations),
                    "checkpoint_data": checkpoint_data,
                    "session_status": session.status,
                }
                performance_metrics = {}
                if session.iterations:
                    performance_values = [
                        iter.performance_metrics.get("improvement", 0.0)
                        for iter in session.iterations
                        if iter.performance_metrics
                    ]
                    if performance_values:
                        performance_metrics = {
                            "average_improvement": sum(performance_values)
                            / len(performance_values),
                            "best_improvement": max(performance_values),
                            "total_iterations": len(performance_values),
                            "performance_trend": performance_values[-5:]
                            if len(performance_values) >= 5
                            else performance_values,
                        }
                workflow_state = WorkflowState(
                    workflow_id=workflow_id,
                    workflow_type=workflow_type,
                    current_step=current_step,
                    completed_steps=completed_steps,
                    pending_steps=pending_steps,
                    workflow_parameters=workflow_parameters,
                    execution_context=execution_context,
                    performance_metrics=performance_metrics,
                )
                self.logger.info(
                    f"Workflow state reconstructed for {session_id}: {current_step}"
                )
                return workflow_state
        except Exception as e:
            self.logger.exception(
                f"Failed to reconstruct workflow state for {session_id}: {e}"
            )
            return None

    async def resume_training_session(
        self,
        session_id: str,
        training_manager: TrainingService,
        force_resume: bool = False,
    ) -> dict[str, Any]:
        """Resume interrupted training session with full workflow reconstruction.

        Args:
            session_id: Training session identifier
            training_manager: Training system manager instance
            force_resume: Force resume even with low confidence

        Returns:
            Resume operation results
        """
        async with self.resume_lock:
            self.logger.info(f"Attempting to resume training session: {session_id}")
            try:
                if session_id not in self.detected_sessions:
                    await self.detect_interrupted_sessions()
                if session_id not in self.detected_sessions:
                    return {
                        "status": "error",
                        "error": f"Session {session_id} not found or not resumable",
                        "session_id": session_id,
                    }
                context = self.detected_sessions[session_id]
                if not force_resume and context.recovery_confidence < 0.6:
                    return {
                        "status": "error",
                        "error": f"Resume confidence too low: {context.recovery_confidence:.2f}",
                        "session_id": session_id,
                        "context": asdict(context),
                    }
                workflow_state = await self.reconstruct_workflow_state(session_id)
                if not workflow_state:
                    return {
                        "status": "error",
                        "error": "Failed to reconstruct workflow state",
                        "session_id": session_id,
                    }
                integrity_check = await self._verify_resume_integrity(
                    session_id, workflow_state
                )
                if not integrity_check["valid"]:
                    return {
                        "status": "error",
                        "error": f"Data integrity check failed: {integrity_check['reason']}",
                        "session_id": session_id,
                        "integrity_check": integrity_check,
                    }
                async with self.session_manager.session_context() as db_session:
                    await db_session.execute(
                        update(TrainingSession)
                        .where(text(f"session_id = '{session_id}'"))
                        .values(
                            current_iteration=context.resume_from_iteration,
                            last_activity_at=datetime.now(UTC),
                            status="resuming",
                        )
                    )
                    await db_session.commit()
                resume_result = await self._perform_resume_operation(
                    session_id, context, workflow_state, training_manager
                )
                self.resume_history.append({
                    "session_id": session_id,
                    "resumed_at": datetime.now(UTC).isoformat(),
                    "context": asdict(context),
                    "workflow_state": asdict(workflow_state),
                    "result": resume_result,
                })
                self.logger.info(
                    f"Resume operation completed for {session_id}: {resume_result['status']}"
                )
                return resume_result
            except Exception as e:
                error_result = {
                    "status": "error",
                    "error": str(e),
                    "session_id": session_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                self.resume_history.append(error_result)
                self.logger.exception(f"Resume operation failed for {session_id}: {e}")
                return error_result

    async def _verify_resume_integrity(
        self, session_id: str, workflow_state: WorkflowState
    ) -> dict[str, Any]:
        """Verify data integrity before resuming session."""
        try:
            integrity_issues = []
            async with self.session_manager.session_context() as db_session:
                result = await db_session.execute(
                    select(TrainingSession).where(text(f"session_id = '{session_id}'"))
                )
                session = result.scalar_one_or_none()
                if not session:
                    integrity_issues.append("Session not found in database")
                if session and session.iterations:
                    iteration_numbers = [iter.iteration for iter in session.iterations]
                    expected_range = list(range(1, max(iteration_numbers) + 1))
                    missing_iterations = set(expected_range) - set(iteration_numbers)
                    if missing_iterations:
                        integrity_issues.append(
                            f"Missing iterations: {sorted(missing_iterations)}"
                        )
                if session and (not session.checkpoint_data):
                    integrity_issues.append("Missing checkpoint data")
            if not workflow_state.workflow_parameters:
                integrity_issues.append("Missing workflow parameters")
            if not workflow_state.execution_context:
                integrity_issues.append("Missing execution context")
            return {
                "valid": len(integrity_issues) == 0,
                "issues": integrity_issues,
                "reason": "; ".join(integrity_issues) if integrity_issues else None,
            }
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Integrity check failed: {e}"],
                "reason": str(e),
            }

    async def _perform_resume_operation(
        self,
        session_id: str,
        context: SessionResumeContext,
        workflow_state: WorkflowState,
        training_manager: TrainingService,
    ) -> dict[str, Any]:
        """Perform the actual resume operation."""
        try:
            async with self.session_manager.session_context() as db_session:
                await db_session.execute(
                    update(TrainingSession)
                    .where(text(f"session_id = '{session_id}'"))
                    .values(
                        status="running",
                        last_activity_at=datetime.now(UTC),
                        checkpoint_data={
                            **workflow_state.execution_context.get(
                                "checkpoint_data", {}
                            ),
                            "resumed_at": datetime.now(UTC).isoformat(),
                            "resume_context": asdict(context),
                        },
                    )
                )
                await db_session.commit()
            training_manager._training_session_id = session_id
            resume_config = {
                **workflow_state.workflow_parameters,
                "resume_from_iteration": context.resume_from_iteration,
                "workflow_state": asdict(workflow_state),
            }
            return {
                "status": "success",
                "session_id": session_id,
                "resumed_from_iteration": context.resume_from_iteration,
                "recovery_confidence": context.recovery_confidence,
                "estimated_loss_minutes": context.estimated_loss_minutes,
                "workflow_state": asdict(workflow_state),
                "resume_config": resume_config,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
