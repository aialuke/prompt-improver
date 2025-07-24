"""
Session Resume Manager for Training Session Recovery
Implements session state detection, workflow reconstruction, and resume coordination.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .progress_preservation import ProgressPreservationManager, ProgressSnapshot
from .training_system_manager import TrainingSystemManager
from ...database import get_session_context
from ...database.models import TrainingSession, TrainingIteration
from sqlalchemy import select, update, and_, or_
from sqlalchemy.orm import selectinload


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
    interruption_reason: Optional[str]
    recovery_strategy: str
    data_integrity_score: float
    resume_from_iteration: int
    estimated_loss_minutes: float
    recovery_confidence: float  # 0.0 to 1.0


@dataclass
class WorkflowState:
    """Reconstructed workflow state for resume operations."""
    workflow_id: Optional[str]
    workflow_type: str
    current_step: str
    completed_steps: List[str]
    pending_steps: List[str]
    workflow_parameters: Dict[str, Any]
    execution_context: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class SessionResumeManager:
    """
    Manages training session resume capabilities with state detection and workflow reconstruction.

    Features:
    - Automatic detection of interrupted training sessions
    - Session state classification and recovery assessment
    - Workflow state reconstruction from database records
    - Data integrity verification for safe resume operations
    - Integration with TrainingSystemManager and ProgressPreservationManager
    - Comprehensive resume coordination and monitoring
    """

    def __init__(self, backup_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("./session_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress preservation manager
        self.progress_manager = ProgressPreservationManager(backup_dir=self.backup_dir)

        # Session resume tracking
        self.detected_sessions: Dict[str, SessionResumeContext] = {}
        self.resume_history: List[Dict[str, Any]] = []

        # Resume operation locks
        self.resume_lock = asyncio.Lock()
        self.detection_lock = asyncio.Lock()

    async def detect_interrupted_sessions(self) -> List[SessionResumeContext]:
        """
        Detect interrupted training sessions that can be resumed.

        Returns:
            List of session resume contexts for interrupted sessions
        """
        async with self.detection_lock:
            self.logger.info("Detecting interrupted training sessions")

            try:
                # Get all sessions from database
                async with get_session_context() as db_session:
                    result = await db_session.execute(
                        select(TrainingSession)
                        .options(selectinload(TrainingSession.iterations))
                        .where(
                            or_(
                                TrainingSession.status == "running",
                                TrainingSession.status == "interrupted",
                                and_(
                                    TrainingSession.status == "running",
                                    TrainingSession.last_activity_at < datetime.now(timezone.utc) - timedelta(minutes=30)
                                )
                            )
                        )
                    )
                    sessions = result.scalars().all()

                interrupted_sessions = []

                for session in sessions:
                    # Analyze session state
                    context = await self._analyze_session_state(session)

                    if context.detected_state in [SessionState.INTERRUPTED, SessionState.RECOVERABLE]:
                        interrupted_sessions.append(context)
                        self.detected_sessions[session.session_id] = context

                        self.logger.info(
                            f"Detected interrupted session: {session.session_id} "
                            f"(state: {context.detected_state.value}, confidence: {context.recovery_confidence:.2f})"
                        )

                self.logger.info(f"Found {len(interrupted_sessions)} interrupted sessions")
                return interrupted_sessions

            except Exception as e:
                self.logger.error(f"Failed to detect interrupted sessions: {e}")
                return []

    async def _analyze_session_state(self, session: TrainingSession) -> SessionResumeContext:
        """Analyze session state to determine if it can be resumed."""
        try:
            # Calculate time since last activity
            last_activity = session.last_activity_at or session.started_at
            time_since_activity = datetime.now(timezone.utc) - last_activity

            # Determine session state
            detected_state = self._classify_session_state(session, time_since_activity)

            # Assess data integrity
            integrity_score = await self._assess_data_integrity(session)

            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(session, detected_state, integrity_score)

            # Calculate recovery confidence
            recovery_confidence = self._calculate_recovery_confidence(
                session, detected_state, integrity_score, time_since_activity
            )

            # Determine resume point
            resume_from_iteration = await self._find_safe_resume_point(session)

            # Estimate data loss
            estimated_loss = self._estimate_data_loss(session, resume_from_iteration, time_since_activity)

            return SessionResumeContext(
                session_id=session.session_id,
                detected_state=detected_state,
                last_activity=last_activity,
                interruption_reason=self._infer_interruption_reason(session, time_since_activity),
                recovery_strategy=recovery_strategy,
                data_integrity_score=integrity_score,
                resume_from_iteration=resume_from_iteration,
                estimated_loss_minutes=estimated_loss,
                recovery_confidence=recovery_confidence
            )

        except Exception as e:
            self.logger.error(f"Failed to analyze session {session.session_id}: {e}")
            return SessionResumeContext(
                session_id=session.session_id,
                detected_state=SessionState.UNRECOVERABLE,
                last_activity=datetime.now(timezone.utc),
                interruption_reason=f"Analysis failed: {e}",
                recovery_strategy="none",
                data_integrity_score=0.0,
                resume_from_iteration=0,
                estimated_loss_minutes=0.0,
                recovery_confidence=0.0
            )

    def _classify_session_state(self, session: TrainingSession, time_since_activity: timedelta) -> SessionState:
        """Classify session state based on database information and timing."""
        # Check explicit status
        if session.status == "completed":
            return SessionState.COMPLETED
        elif session.status == "failed":
            return SessionState.FAILED
        elif session.status == "interrupted":
            return SessionState.INTERRUPTED

        # Check for running sessions that appear interrupted
        if session.status == "running":
            if time_since_activity > timedelta(hours=2):
                return SessionState.INTERRUPTED
            elif time_since_activity > timedelta(minutes=30):
                return SessionState.RECOVERABLE
            else:
                return SessionState.RUNNING

        return SessionState.CORRUPTED

    async def _assess_data_integrity(self, session: TrainingSession) -> float:
        """Assess data integrity for the session."""
        try:
            integrity_score = 1.0

            # Check if session has iterations
            if not session.iterations:
                integrity_score -= 0.3

            # Check for checkpoint data
            if not session.checkpoint_data:
                integrity_score -= 0.2

            # Check iteration consistency
            if session.iterations:
                expected_iterations = session.current_iteration
                actual_iterations = len(session.iterations)

                if actual_iterations < expected_iterations * 0.8:
                    integrity_score -= 0.3

            # Check for recent activity
            if session.last_activity_at:
                time_since_activity = datetime.now(timezone.utc) - session.last_activity_at
                if time_since_activity > timedelta(hours=24):
                    integrity_score -= 0.2

            return max(0.0, integrity_score)

        except Exception as e:
            self.logger.warning(f"Failed to assess data integrity for {session.session_id}: {e}")
            return 0.5

    def _determine_recovery_strategy(self, session: TrainingSession, state: SessionState, integrity: float) -> str:
        """Determine the best recovery strategy for the session."""
        if state == SessionState.UNRECOVERABLE or integrity < 0.3:
            return "none"
        elif state == SessionState.CORRUPTED or integrity < 0.6:
            return "partial_recovery"
        elif state in [SessionState.INTERRUPTED, SessionState.RECOVERABLE]:
            if integrity > 0.8:
                return "full_resume"
            else:
                return "checkpoint_resume"
        else:
            return "none"

    def _calculate_recovery_confidence(
        self,
        session: TrainingSession,
        state: SessionState,
        integrity: float,
        time_since_activity: timedelta
    ) -> float:
        """Calculate confidence score for successful recovery."""
        base_confidence = 0.5

        # State-based confidence
        state_confidence = {
            SessionState.RECOVERABLE: 0.9,
            SessionState.INTERRUPTED: 0.7,
            SessionState.RUNNING: 0.95,
            SessionState.CORRUPTED: 0.3,
            SessionState.FAILED: 0.2,
            SessionState.UNRECOVERABLE: 0.0
        }.get(state, 0.1)

        # Integrity-based confidence
        integrity_confidence = integrity

        # Time-based confidence (fresher is better)
        hours_since_activity = time_since_activity.total_seconds() / 3600
        time_confidence = max(0.1, 1.0 - (hours_since_activity / 24))

        # Checkpoint availability
        checkpoint_confidence = 0.8 if session.checkpoint_data else 0.5

        # Weighted average
        confidence = (
            state_confidence * 0.3 +
            integrity_confidence * 0.3 +
            time_confidence * 0.2 +
            checkpoint_confidence * 0.2
        )

        return min(1.0, max(0.0, confidence))

    async def _find_safe_resume_point(self, session: TrainingSession) -> int:
        """Find the safest iteration to resume from."""
        try:
            if not session.iterations:
                return 0

            # Find the last completed iteration with valid data
            valid_iterations = [
                iteration for iteration in session.iterations
                if iteration.completed_at is not None and iteration.error_message is None
            ]

            if valid_iterations:
                # Sort by iteration number and return the last valid one
                valid_iterations.sort(key=lambda x: x.iteration)
                return valid_iterations[-1].iteration
            else:
                return 0

        except Exception as e:
            self.logger.warning(f"Failed to find safe resume point for {session.session_id}: {e}")
            return 0

    def _estimate_data_loss(self, session: TrainingSession, resume_point: int, time_since_activity: timedelta) -> float:
        """Estimate potential data loss in minutes."""
        try:
            # Calculate based on iterations lost
            iterations_lost = max(0, session.current_iteration - resume_point)

            # Estimate time per iteration (assume 2 minutes average)
            estimated_iteration_time = 2.0
            iteration_loss_minutes = iterations_lost * estimated_iteration_time

            # Add time since last activity (potential work in progress)
            activity_loss_minutes = min(30.0, time_since_activity.total_seconds() / 60)

            return iteration_loss_minutes + activity_loss_minutes

        except Exception as e:
            self.logger.warning(f"Failed to estimate data loss for {session.session_id}: {e}")
            return 0.0

    def _infer_interruption_reason(self, session: TrainingSession, time_since_activity: timedelta) -> Optional[str]:
        """Infer the likely reason for session interruption."""
        if session.status == "interrupted":
            return "explicit_interruption"
        elif time_since_activity > timedelta(hours=6):
            return "system_shutdown_or_crash"
        elif time_since_activity > timedelta(hours=1):
            return "process_termination"
        elif time_since_activity > timedelta(minutes=30):
            return "unexpected_exit"
        else:
            return "recent_activity"

    async def reconstruct_workflow_state(self, session_id: str) -> Optional[WorkflowState]:
        """
        Reconstruct workflow state from database records for resume operations.

        Args:
            session_id: Training session identifier

        Returns:
            Reconstructed workflow state or None if reconstruction fails
        """
        try:
            self.logger.info(f"Reconstructing workflow state for session: {session_id}")

            async with get_session_context() as db_session:
                # Get session with iterations
                result = await db_session.execute(
                    select(TrainingSession)
                    .options(selectinload(TrainingSession.iterations))
                    .where(TrainingSession.session_id == session_id)
                )
                session = result.scalar_one_or_none()

                if not session:
                    self.logger.error(f"Session not found: {session_id}")
                    return None

                # Extract workflow information from checkpoint data
                checkpoint_data = session.checkpoint_data or {}
                workflow_id = checkpoint_data.get("workflow_id")
                workflow_type = checkpoint_data.get("workflow_type", "continuous_training")

                # Reconstruct execution state from iterations
                completed_steps = []
                current_step = "initialization"
                pending_steps = ["data_export", "pattern_discovery", "rule_optimization", "validation"]

                if session.iterations:
                    # Analyze completed iterations to determine workflow progress
                    latest_iteration = max(session.iterations, key=lambda x: x.iteration)

                    if latest_iteration.data_export_completed:
                        completed_steps.append("data_export")
                        current_step = "pattern_discovery"
                        pending_steps.remove("data_export")

                    if latest_iteration.pattern_discovery_completed:
                        completed_steps.append("pattern_discovery")
                        current_step = "rule_optimization"
                        if "pattern_discovery" in pending_steps:
                            pending_steps.remove("pattern_discovery")

                    if latest_iteration.rule_optimization_completed:
                        completed_steps.append("rule_optimization")
                        current_step = "validation"
                        if "rule_optimization" in pending_steps:
                            pending_steps.remove("rule_optimization")

                    if latest_iteration.validation_completed:
                        completed_steps.append("validation")
                        current_step = "completed"
                        if "validation" in pending_steps:
                            pending_steps.remove("validation")

                # Reconstruct workflow parameters
                workflow_parameters = {
                    "session_id": session_id,
                    "continuous_mode": session.continuous_mode,
                    "max_iterations": session.max_iterations,
                    "improvement_threshold": session.improvement_threshold,
                    "timeout_seconds": session.timeout_seconds,
                    "current_iteration": session.current_iteration,
                    "resume_mode": True
                }

                # Reconstruct execution context
                execution_context = {
                    "started_at": session.started_at.isoformat(),
                    "last_activity_at": session.last_activity_at.isoformat() if session.last_activity_at else None,
                    "total_iterations_completed": len(session.iterations),
                    "checkpoint_data": checkpoint_data,
                    "session_status": session.status
                }

                # Aggregate performance metrics
                performance_metrics = {}
                if session.iterations:
                    # Calculate aggregate metrics from iterations
                    performance_values = [
                        iter.performance_metrics.get("improvement", 0.0)
                        for iter in session.iterations
                        if iter.performance_metrics
                    ]

                    if performance_values:
                        performance_metrics = {
                            "average_improvement": sum(performance_values) / len(performance_values),
                            "best_improvement": max(performance_values),
                            "total_iterations": len(performance_values),
                            "performance_trend": performance_values[-5:] if len(performance_values) >= 5 else performance_values
                        }

                workflow_state = WorkflowState(
                    workflow_id=workflow_id,
                    workflow_type=workflow_type,
                    current_step=current_step,
                    completed_steps=completed_steps,
                    pending_steps=pending_steps,
                    workflow_parameters=workflow_parameters,
                    execution_context=execution_context,
                    performance_metrics=performance_metrics
                )

                self.logger.info(f"Workflow state reconstructed for {session_id}: {current_step}")
                return workflow_state

        except Exception as e:
            self.logger.error(f"Failed to reconstruct workflow state for {session_id}: {e}")
            return None

    async def resume_training_session(
        self,
        session_id: str,
        training_manager: TrainingSystemManager,
        force_resume: bool = False
    ) -> Dict[str, Any]:
        """
        Resume interrupted training session with full workflow reconstruction.

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
                # Check if session is in detected sessions
                if session_id not in self.detected_sessions:
                    # Detect session state
                    await self.detect_interrupted_sessions()

                if session_id not in self.detected_sessions:
                    return {
                        "status": "error",
                        "error": f"Session {session_id} not found or not resumable",
                        "session_id": session_id
                    }

                context = self.detected_sessions[session_id]

                # Check resume confidence
                if not force_resume and context.recovery_confidence < 0.6:
                    return {
                        "status": "error",
                        "error": f"Resume confidence too low: {context.recovery_confidence:.2f}",
                        "session_id": session_id,
                        "context": asdict(context)
                    }

                # Reconstruct workflow state
                workflow_state = await self.reconstruct_workflow_state(session_id)
                if not workflow_state:
                    return {
                        "status": "error",
                        "error": "Failed to reconstruct workflow state",
                        "session_id": session_id
                    }

                # Verify data integrity before resume
                integrity_check = await self._verify_resume_integrity(session_id, workflow_state)
                if not integrity_check["valid"]:
                    return {
                        "status": "error",
                        "error": f"Data integrity check failed: {integrity_check['reason']}",
                        "session_id": session_id,
                        "integrity_check": integrity_check
                    }

                # Perform actual resume operation
                resume_result = await self._perform_resume_operation(
                    session_id, context, workflow_state, training_manager
                )

                # Track resume operation
                self.resume_history.append({
                    "session_id": session_id,
                    "resumed_at": datetime.now(timezone.utc).isoformat(),
                    "context": asdict(context),
                    "workflow_state": asdict(workflow_state),
                    "result": resume_result
                })

                self.logger.info(f"Resume operation completed for {session_id}: {resume_result['status']}")
                return resume_result

            except Exception as e:
                error_result = {
                    "status": "error",
                    "error": str(e),
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                self.resume_history.append(error_result)
                self.logger.error(f"Resume operation failed for {session_id}: {e}")
                return error_result

    async def _verify_resume_integrity(self, session_id: str, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Verify data integrity before resuming session."""
        try:
            integrity_issues = []

            # Check database consistency
            async with get_session_context() as db_session:
                result = await db_session.execute(
                    select(TrainingSession).where(TrainingSession.session_id == session_id)
                )
                session = result.scalar_one_or_none()

                if not session:
                    integrity_issues.append("Session not found in database")

                # Check iteration consistency
                if session and session.iterations:
                    iteration_numbers = [iter.iteration for iter in session.iterations]
                    expected_range = list(range(1, max(iteration_numbers) + 1))
                    missing_iterations = set(expected_range) - set(iteration_numbers)

                    if missing_iterations:
                        integrity_issues.append(f"Missing iterations: {sorted(missing_iterations)}")

                # Check checkpoint data consistency
                if session and not session.checkpoint_data:
                    integrity_issues.append("Missing checkpoint data")

            # Check workflow state consistency
            if not workflow_state.workflow_parameters:
                integrity_issues.append("Missing workflow parameters")

            if not workflow_state.execution_context:
                integrity_issues.append("Missing execution context")

            return {
                "valid": len(integrity_issues) == 0,
                "issues": integrity_issues,
                "reason": "; ".join(integrity_issues) if integrity_issues else None
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Integrity check failed: {e}"],
                "reason": str(e)
            }

    async def _perform_resume_operation(
        self,
        session_id: str,
        context: SessionResumeContext,
        workflow_state: WorkflowState,
        training_manager: TrainingSystemManager
    ) -> Dict[str, Any]:
        """Perform the actual resume operation."""
        try:
            # Update session status to resuming
            async with get_session_context() as db_session:
                await db_session.execute(
                    update(TrainingSession)
                    .where(TrainingSession.session_id == session_id)
                    .values(
                        status="running",
                        last_activity_at=datetime.now(timezone.utc),
                        checkpoint_data={
                            **workflow_state.execution_context.get("checkpoint_data", {}),
                            "resumed_at": datetime.now(timezone.utc).isoformat(),
                            "resume_context": asdict(context)
                        }
                    )
                )
                await db_session.commit()

            # Set training manager session
            training_manager._training_session_id = session_id

            # Resume workflow execution
            resume_config = {
                **workflow_state.workflow_parameters,
                "resume_from_iteration": context.resume_from_iteration,
                "workflow_state": asdict(workflow_state)
            }

            return {
                "status": "success",
                "session_id": session_id,
                "resumed_from_iteration": context.resume_from_iteration,
                "recovery_confidence": context.recovery_confidence,
                "estimated_loss_minutes": context.estimated_loss_minutes,
                "workflow_state": asdict(workflow_state),
                "resume_config": resume_config,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
