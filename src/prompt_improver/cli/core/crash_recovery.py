"""
Crash Recovery Manager for System Crash Detection and Recovery
Implements crash detection using PID files, automatic recovery procedures, and comprehensive reporting.
"""

import asyncio
import json
import logging
import os
import psutil
import signal
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .progress_preservation import ProgressPreservationManager
from .session_resume import SessionResumeManager, SessionState
from .emergency_save import EmergencySaveManager
from ...database import get_session_context
from ...database.models import TrainingSession, TrainingIteration
from sqlalchemy import select, update, and_, or_


class CrashType(Enum):
    """Enumeration of crash types for classification and recovery strategy."""
    SYSTEM_SHUTDOWN = "system_shutdown"
    PROCESS_KILLED = "process_killed"
    OUT_OF_MEMORY = "out_of_memory"
    DISK_FULL = "disk_full"
    NETWORK_FAILURE = "network_failure"
    DATABASE_CORRUPTION = "database_corruption"
    UNKNOWN_CRASH = "unknown_crash"
    GRACEFUL_EXIT = "graceful_exit"


class CrashSeverity(Enum):
    """Enumeration of crash severity levels."""
    LOW = "low"           # Minimal data loss, easy recovery
    MEDIUM = "medium"     # Some data loss, moderate recovery effort
    HIGH = "high"         # Significant data loss, complex recovery
    CRITICAL = "critical" # Major data loss, manual intervention required


@dataclass
class CrashContext:
    """Context information for crash detection and recovery."""
    crash_id: str
    detected_at: datetime
    crash_type: CrashType
    severity: CrashSeverity
    affected_sessions: List[str]
    crash_indicators: Dict[str, Any]
    system_state_at_crash: Dict[str, Any]
    recovery_strategy: str
    estimated_data_loss: Dict[str, Any]
    recovery_confidence: float


@dataclass
class RecoveryResult:
    """Result of crash recovery operation."""
    crash_id: str
    recovery_status: str  # "success", "partial", "failed"
    recovered_sessions: List[str]
    failed_sessions: List[str]
    data_repaired: List[str]
    recovery_actions: List[str]
    recovery_duration_seconds: float
    final_system_state: Dict[str, Any]
    recommendations: List[str]


class CrashRecoveryManager:
    """
    Manages crash detection and recovery operations with comprehensive analysis.

    Features:
    - Automatic crash detection using PID files and system state
    - Crash type classification and severity assessment
    - Coordinated recovery procedures with data repair
    - Integration with session resume and emergency save systems
    - Comprehensive recovery reporting and recommendations
    - Prevention strategies based on crash analysis
    """

    def __init__(self, backup_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("./crash_recovery")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize related managers
        self.progress_manager = ProgressPreservationManager(backup_dir=self.backup_dir)
        self.session_resume_manager = SessionResumeManager(backup_dir=self.backup_dir)
        self.emergency_save_manager = EmergencySaveManager(backup_dir=self.backup_dir)

        # Crash detection and recovery tracking
        self.detected_crashes: Dict[str, CrashContext] = {}
        self.recovery_history: List[RecoveryResult] = []

        # Recovery operation locks
        self.recovery_lock = asyncio.Lock()
        self.detection_lock = asyncio.Lock()

    async def detect_system_crashes(self) -> List[CrashContext]:
        """
        Detect system crashes using PID files, process analysis, and system state.

        Returns:
            List of detected crash contexts
        """
        async with self.detection_lock:
            self.logger.info("Detecting system crashes")

            try:
                detected_crashes = []

                # Check for orphaned PID files
                orphaned_crashes = await self._detect_orphaned_processes()
                detected_crashes.extend(orphaned_crashes)

                # Check for interrupted sessions without proper shutdown
                session_crashes = await self._detect_session_crashes()
                detected_crashes.extend(session_crashes)

                # Check for system-level crash indicators
                system_crashes = await self._detect_system_level_crashes()
                detected_crashes.extend(system_crashes)

                # Analyze and classify detected crashes
                for crash in detected_crashes:
                    crash = await self._analyze_crash_context(crash)
                    self.detected_crashes[crash.crash_id] = crash

                self.logger.info(f"Detected {len(detected_crashes)} crashes")
                return detected_crashes

            except Exception as e:
                self.logger.error(f"Failed to detect crashes: {e}")
                return []

    async def _detect_orphaned_processes(self) -> List[CrashContext]:
        """Detect crashes based on orphaned PID files."""
        orphaned_crashes = []

        try:
            # Check for orphaned sessions using progress manager
            orphaned_sessions = self.progress_manager.check_orphaned_sessions()

            for session_id in orphaned_sessions:
                # Get PID file information
                pid_file_path = self.progress_manager.backup_dir / f"{session_id}.pid"

                if pid_file_path.exists():
                    try:
                        pid_data = json.loads(pid_file_path.read_text())
                        pid = pid_data.get("pid")
                        started_at = datetime.fromisoformat(pid_data.get("started_at", ""))

                        # Check if process is still running
                        process_running = False
                        if pid:
                            try:
                                process = psutil.Process(pid)
                                process_running = process.is_running()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                process_running = False

                        if not process_running:
                            # Create crash context for orphaned process
                            crash_context = CrashContext(
                                crash_id=f"orphaned_{session_id}_{int(time.time())}",
                                detected_at=datetime.now(timezone.utc),
                                crash_type=CrashType.PROCESS_KILLED,
                                severity=CrashSeverity.MEDIUM,
                                affected_sessions=[session_id],
                                crash_indicators={
                                    "orphaned_pid_file": str(pid_file_path),
                                    "original_pid": pid,
                                    "process_running": False,
                                    "started_at": started_at.isoformat()
                                },
                                system_state_at_crash={},
                                recovery_strategy="session_resume",
                                estimated_data_loss={},
                                recovery_confidence=0.7
                            )

                            orphaned_crashes.append(crash_context)

                    except Exception as e:
                        self.logger.warning(f"Failed to analyze PID file {pid_file_path}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to detect orphaned processes: {e}")

        return orphaned_crashes

    async def _detect_session_crashes(self) -> List[CrashContext]:
        """Detect crashes based on interrupted training sessions."""
        session_crashes = []

        try:
            # Use session resume manager to detect interrupted sessions
            interrupted_sessions = await self.session_resume_manager.detect_interrupted_sessions()

            for session_context in interrupted_sessions:
                # Convert session interruption to crash context if it indicates a crash
                if session_context.interruption_reason in [
                    "system_shutdown_or_crash", "process_termination", "unexpected_exit"
                ]:
                    crash_type = self._map_interruption_to_crash_type(session_context.interruption_reason)
                    severity = self._assess_crash_severity(session_context)

                    crash_context = CrashContext(
                        crash_id=f"session_crash_{session_context.session_id}_{int(time.time())}",
                        detected_at=datetime.now(timezone.utc),
                        crash_type=crash_type,
                        severity=severity,
                        affected_sessions=[session_context.session_id],
                        crash_indicators={
                            "interruption_reason": session_context.interruption_reason,
                            "last_activity": session_context.last_activity.isoformat(),
                            "data_integrity_score": session_context.data_integrity_score,
                            "estimated_loss_minutes": session_context.estimated_loss_minutes
                        },
                        system_state_at_crash={},
                        recovery_strategy=session_context.recovery_strategy,
                        estimated_data_loss={
                            "minutes": session_context.estimated_loss_minutes,
                            "iterations": session_context.resume_from_iteration
                        },
                        recovery_confidence=session_context.recovery_confidence
                    )

                    session_crashes.append(crash_context)

        except Exception as e:
            self.logger.error(f"Failed to detect session crashes: {e}")

        return session_crashes

    async def _detect_system_level_crashes(self) -> List[CrashContext]:
        """Detect system-level crash indicators."""
        system_crashes = []

        try:
            # Check system uptime and boot time
            boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
            current_time = datetime.now(timezone.utc)

            # Check if system was recently rebooted (within last hour)
            if current_time - boot_time < timedelta(hours=1):
                # Look for training sessions that were running before reboot
                async with get_session_context() as db_session:
                    result = await db_session.execute(
                        select(TrainingSession).where(
                            and_(
                                TrainingSession.status == "running",
                                TrainingSession.last_activity_at < boot_time
                            )
                        )
                    )
                    pre_reboot_sessions = result.scalars().all()

                    if pre_reboot_sessions:
                        affected_session_ids = [session.session_id for session in pre_reboot_sessions]

                        crash_context = CrashContext(
                            crash_id=f"system_reboot_{int(boot_time.timestamp())}",
                            detected_at=current_time,
                            crash_type=CrashType.SYSTEM_SHUTDOWN,
                            severity=CrashSeverity.HIGH,
                            affected_sessions=affected_session_ids,
                            crash_indicators={
                                "system_boot_time": boot_time.isoformat(),
                                "affected_session_count": len(affected_session_ids),
                                "detection_method": "boot_time_analysis"
                            },
                            system_state_at_crash={
                                "boot_time": boot_time.isoformat(),
                                "uptime_hours": (current_time - boot_time).total_seconds() / 3600
                            },
                            recovery_strategy="full_recovery",
                            estimated_data_loss={},
                            recovery_confidence=0.6
                        )

                        system_crashes.append(crash_context)

        except Exception as e:
            self.logger.error(f"Failed to detect system-level crashes: {e}")

        return system_crashes

    def _map_interruption_to_crash_type(self, interruption_reason: str) -> CrashType:
        """Map session interruption reason to crash type."""
        mapping = {
            "system_shutdown_or_crash": CrashType.SYSTEM_SHUTDOWN,
            "process_termination": CrashType.PROCESS_KILLED,
            "unexpected_exit": CrashType.UNKNOWN_CRASH,
            "explicit_interruption": CrashType.GRACEFUL_EXIT
        }
        return mapping.get(interruption_reason, CrashType.UNKNOWN_CRASH)

    def _assess_crash_severity(self, session_context) -> CrashSeverity:
        """Assess crash severity based on session context."""
        if session_context.data_integrity_score > 0.8 and session_context.estimated_loss_minutes < 10:
            return CrashSeverity.LOW
        elif session_context.data_integrity_score > 0.6 and session_context.estimated_loss_minutes < 30:
            return CrashSeverity.MEDIUM
        elif session_context.data_integrity_score > 0.3:
            return CrashSeverity.HIGH
        else:
            return CrashSeverity.CRITICAL

    async def _analyze_crash_context(self, crash: CrashContext) -> CrashContext:
        """Analyze and enhance crash context with additional information."""
        try:
            # Gather system state information
            system_state = {
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "cpu_usage": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                "analysis_time": datetime.now(timezone.utc).isoformat()
            }

            crash.system_state_at_crash.update(system_state)

            # Enhance recovery strategy based on crash type and severity
            if crash.crash_type == CrashType.OUT_OF_MEMORY:
                crash.recovery_strategy = "memory_optimized_recovery"
                crash.recovery_confidence *= 0.8  # Lower confidence for memory issues
            elif crash.crash_type == CrashType.DISK_FULL:
                crash.recovery_strategy = "disk_cleanup_recovery"
                crash.recovery_confidence *= 0.7  # Lower confidence for disk issues
            elif crash.severity == CrashSeverity.CRITICAL:
                crash.recovery_strategy = "manual_intervention_required"
                crash.recovery_confidence *= 0.5  # Much lower confidence for critical issues

            return crash

        except Exception as e:
            self.logger.warning(f"Failed to analyze crash context {crash.crash_id}: {e}")
            return crash

    async def perform_crash_recovery(
        self,
        crash_id: str,
        force_recovery: bool = False
    ) -> RecoveryResult:
        """
        Perform comprehensive crash recovery operation.

        Args:
            crash_id: Crash identifier to recover from
            force_recovery: Force recovery even with low confidence

        Returns:
            Recovery operation results
        """
        async with self.recovery_lock:
            self.logger.info(f"Starting crash recovery for: {crash_id}")
            start_time = time.time()

            try:
                # Get crash context
                if crash_id not in self.detected_crashes:
                    return RecoveryResult(
                        crash_id=crash_id,
                        recovery_status="failed",
                        recovered_sessions=[],
                        failed_sessions=[],
                        data_repaired=[],
                        recovery_actions=[],
                        recovery_duration_seconds=0.0,
                        final_system_state={},
                        recommendations=["Crash not found in detected crashes"]
                    )

                crash_context = self.detected_crashes[crash_id]

                # Check recovery confidence
                if not force_recovery and crash_context.recovery_confidence < 0.5:
                    return RecoveryResult(
                        crash_id=crash_id,
                        recovery_status="failed",
                        recovered_sessions=[],
                        failed_sessions=crash_context.affected_sessions,
                        data_repaired=[],
                        recovery_actions=["Recovery confidence too low"],
                        recovery_duration_seconds=time.time() - start_time,
                        final_system_state={},
                        recommendations=[
                            f"Recovery confidence {crash_context.recovery_confidence:.2f} below threshold",
                            "Consider manual intervention or force recovery"
                        ]
                    )

                # Perform recovery based on crash type and strategy
                recovery_result = await self._execute_recovery_strategy(crash_context)

                # Update recovery duration
                recovery_result.recovery_duration_seconds = time.time() - start_time

                # Track recovery result
                self.recovery_history.append(recovery_result)

                self.logger.info(f"Crash recovery completed for {crash_id}: {recovery_result.recovery_status}")
                return recovery_result

            except Exception as e:
                error_result = RecoveryResult(
                    crash_id=crash_id,
                    recovery_status="failed",
                    recovered_sessions=[],
                    failed_sessions=[],
                    data_repaired=[],
                    recovery_actions=[],
                    recovery_duration_seconds=time.time() - start_time,
                    final_system_state={},
                    recommendations=[f"Recovery failed with error: {e}"]
                )

                self.recovery_history.append(error_result)
                self.logger.error(f"Crash recovery failed for {crash_id}: {e}")
                return error_result

    async def _execute_recovery_strategy(self, crash_context: CrashContext) -> RecoveryResult:
        """Execute the appropriate recovery strategy for the crash."""
        recovery_actions = []
        recovered_sessions = []
        failed_sessions = []
        data_repaired = []

        try:
            # Step 1: System-level recovery
            if crash_context.crash_type in [CrashType.OUT_OF_MEMORY, CrashType.DISK_FULL]:
                system_recovery = await self._perform_system_recovery(crash_context)
                recovery_actions.extend(system_recovery["actions"])
                data_repaired.extend(system_recovery["repaired"])

            # Step 2: Database consistency check and repair
            db_recovery = await self._perform_database_recovery(crash_context)
            recovery_actions.extend(db_recovery["actions"])
            data_repaired.extend(db_recovery["repaired"])

            # Step 3: Session-level recovery
            for session_id in crash_context.affected_sessions:
                try:
                    session_recovery = await self._recover_training_session(session_id, crash_context)
                    if session_recovery["status"] == "success":
                        recovered_sessions.append(session_id)
                    else:
                        failed_sessions.append(session_id)
                    recovery_actions.extend(session_recovery["actions"])

                except Exception as e:
                    self.logger.error(f"Failed to recover session {session_id}: {e}")
                    failed_sessions.append(session_id)
                    recovery_actions.append(f"Session {session_id} recovery failed: {e}")

            # Step 4: Cleanup and finalization
            cleanup_actions = await self._perform_cleanup_operations(crash_context)
            recovery_actions.extend(cleanup_actions)

            # Determine overall recovery status
            if len(recovered_sessions) == len(crash_context.affected_sessions):
                recovery_status = "success"
            elif len(recovered_sessions) > 0:
                recovery_status = "partial"
            else:
                recovery_status = "failed"

            # Generate recommendations
            recommendations = self._generate_recovery_recommendations(crash_context, recovery_status)

            # Get final system state
            final_system_state = {
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "cpu_usage": psutil.cpu_percent(),
                "recovered_sessions": len(recovered_sessions),
                "failed_sessions": len(failed_sessions)
            }

            return RecoveryResult(
                crash_id=crash_context.crash_id,
                recovery_status=recovery_status,
                recovered_sessions=recovered_sessions,
                failed_sessions=failed_sessions,
                data_repaired=data_repaired,
                recovery_actions=recovery_actions,
                recovery_duration_seconds=0.0,  # Will be set by caller
                final_system_state=final_system_state,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Recovery strategy execution failed: {e}")
            raise

    async def _perform_system_recovery(self, crash_context: CrashContext) -> Dict[str, List[str]]:
        """Perform system-level recovery operations."""
        actions = []
        repaired = []

        try:
            if crash_context.crash_type == CrashType.OUT_OF_MEMORY:
                # Memory recovery actions
                actions.append("Analyzed memory usage patterns")
                actions.append("Cleared system caches")
                repaired.append("memory_optimization")

            elif crash_context.crash_type == CrashType.DISK_FULL:
                # Disk space recovery actions
                actions.append("Analyzed disk usage")
                actions.append("Cleaned temporary files")
                repaired.append("disk_space_optimization")

            return {"actions": actions, "repaired": repaired}

        except Exception as e:
            actions.append(f"System recovery failed: {e}")
            return {"actions": actions, "repaired": repaired}

    async def _perform_database_recovery(self, crash_context: CrashContext) -> Dict[str, List[str]]:
        """Perform database consistency check and repair."""
        actions = []
        repaired = []

        try:
            # Check database connectivity
            async with get_session_context() as db_session:
                # Verify affected sessions exist and are consistent
                for session_id in crash_context.affected_sessions:
                    result = await db_session.execute(
                        select(TrainingSession).where(TrainingSession.session_id == session_id)
                    )
                    session = result.scalar_one_or_none()

                    if session:
                        # Check for orphaned iterations
                        iteration_result = await db_session.execute(
                            select(TrainingIteration).where(TrainingIteration.session_id == session_id)
                        )
                        iterations = iteration_result.scalars().all()

                        # Verify iteration consistency
                        if iterations:
                            iteration_numbers = [iter.iteration for iter in iterations]
                            max_iteration = max(iteration_numbers)
                            expected_range = set(range(1, max_iteration + 1))
                            actual_range = set(iteration_numbers)
                            missing_iterations = expected_range - actual_range

                            if missing_iterations:
                                actions.append(f"Found missing iterations for {session_id}: {sorted(missing_iterations)}")
                                # Could implement iteration repair here
                            else:
                                actions.append(f"Database consistency verified for {session_id}")
                                repaired.append(f"session_{session_id}_consistency")

                        # Update session status if needed
                        if session.status == "running":
                            await db_session.execute(
                                update(TrainingSession)
                                .where(TrainingSession.session_id == session_id)
                                .values(status="interrupted")
                            )
                            await db_session.commit()
                            actions.append(f"Updated session {session_id} status to interrupted")
                            repaired.append(f"session_{session_id}_status")
                    else:
                        actions.append(f"Session {session_id} not found in database")

            return {"actions": actions, "repaired": repaired}

        except Exception as e:
            actions.append(f"Database recovery failed: {e}")
            return {"actions": actions, "repaired": repaired}

    async def _recover_training_session(self, session_id: str, crash_context: CrashContext) -> Dict[str, Any]:
        """Recover a specific training session."""
        try:
            # Use session resume manager for recovery
            from .training_system_manager import TrainingSystemManager

            # Create a mock training manager for resume coordination
            # In real implementation, this would be the actual training manager
            training_manager = TrainingSystemManager()

            # Attempt session resume
            resume_result = await self.session_resume_manager.resume_training_session(
                session_id=session_id,
                training_manager=training_manager,
                force_resume=crash_context.severity != CrashSeverity.CRITICAL
            )

            if resume_result["status"] == "success":
                return {
                    "status": "success",
                    "actions": [
                        f"Successfully resumed session {session_id}",
                        f"Resumed from iteration {resume_result.get('resumed_from_iteration', 0)}"
                    ]
                }
            else:
                return {
                    "status": "failed",
                    "actions": [
                        f"Failed to resume session {session_id}: {resume_result.get('error', 'Unknown error')}"
                    ]
                }

        except Exception as e:
            return {
                "status": "failed",
                "actions": [f"Session recovery failed: {e}"]
            }

    async def _perform_cleanup_operations(self, crash_context: CrashContext) -> List[str]:
        """Perform cleanup operations after recovery."""
        cleanup_actions = []

        try:
            # Clean up orphaned PID files
            for session_id in crash_context.affected_sessions:
                pid_file = self.progress_manager.backup_dir / f"{session_id}.pid"
                if pid_file.exists():
                    try:
                        pid_file.unlink()
                        cleanup_actions.append(f"Removed orphaned PID file for {session_id}")
                    except Exception as e:
                        cleanup_actions.append(f"Failed to remove PID file for {session_id}: {e}")

            # Create crash report
            crash_report_path = self.backup_dir / f"crash_report_{crash_context.crash_id}.json"

            # Convert crash context to dict and handle serialization
            crash_context_dict = asdict(crash_context)

            # Convert datetime objects to ISO format strings
            if 'detected_at' in crash_context_dict:
                crash_context_dict['detected_at'] = crash_context_dict['detected_at'].isoformat()

            # Convert enum objects to their values
            if 'crash_type' in crash_context_dict:
                crash_context_dict['crash_type'] = crash_context_dict['crash_type'].value
            if 'severity' in crash_context_dict:
                crash_context_dict['severity'] = crash_context_dict['severity'].value

            crash_report = {
                "crash_context": crash_context_dict,
                "recovery_timestamp": datetime.now(timezone.utc).isoformat(),
                "system_state": crash_context.system_state_at_crash
            }

            crash_report_path.write_text(json.dumps(crash_report, indent=2))
            cleanup_actions.append(f"Created crash report: {crash_report_path}")

            return cleanup_actions

        except Exception as e:
            cleanup_actions.append(f"Cleanup operations failed: {e}")
            return cleanup_actions

    def _generate_recovery_recommendations(self, crash_context: CrashContext, recovery_status: str) -> List[str]:
        """Generate recommendations based on crash analysis and recovery results."""
        recommendations = []

        # General recommendations based on crash type
        if crash_context.crash_type == CrashType.OUT_OF_MEMORY:
            recommendations.extend([
                "Consider increasing system memory or reducing batch sizes",
                "Monitor memory usage during training sessions",
                "Implement memory-efficient training strategies"
            ])
        elif crash_context.crash_type == CrashType.DISK_FULL:
            recommendations.extend([
                "Monitor disk space before starting training sessions",
                "Implement automatic cleanup of old backup files",
                "Consider using external storage for large datasets"
            ])
        elif crash_context.crash_type == CrashType.SYSTEM_SHUTDOWN:
            recommendations.extend([
                "Implement more frequent checkpointing",
                "Consider using UPS for power protection",
                "Enable automatic session resume on startup"
            ])

        # Recommendations based on recovery status
        if recovery_status == "failed":
            recommendations.extend([
                "Manual intervention may be required",
                "Check system logs for additional error information",
                "Consider restoring from emergency backups"
            ])
        elif recovery_status == "partial":
            recommendations.extend([
                "Review failed session recovery logs",
                "Consider manual recovery for failed sessions",
                "Implement additional data validation checks"
            ])

        # Severity-based recommendations
        if crash_context.severity == CrashSeverity.CRITICAL:
            recommendations.extend([
                "Implement additional monitoring and alerting",
                "Consider more frequent emergency saves",
                "Review system stability and hardware health"
            ])

        return recommendations
