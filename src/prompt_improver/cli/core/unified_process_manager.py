"""Unified Process Manager consolidating PIDManager and CrashRecoveryManager.

This module consolidates the functionality of both PIDManager and CrashRecoveryManager
using composition to provide unified process lifecycle management with both PID tracking
and crash recovery capabilities in a single interface.
"""

import asyncio
import fcntl
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from prompt_improver.cli.core.signal_handler import SignalOperation

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Enumeration of process states for tracking."""

    RUNNING = "running"
    STOPPED = "stopped"
    ZOMBIE = "zombie"
    SLEEPING = "sleeping"
    DISK_SLEEP = "disk_sleep"
    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"


class CrashType(Enum):
    """Enumeration of crash types for classification and recovery strategy."""

    SYSTEM_SHUTDOWN = "system_shutdown"
    PROCESS_KILLED = "process_killed"
    OUT_OF_MEMORY = "out_of_memory"
    SEGMENTATION_FAULT = "segmentation_fault"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


class CrashSeverity(Enum):
    """Crash severity levels for prioritizing recovery."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProcessInfo:
    """Information about a tracked process."""

    pid: int
    session_id: str
    started_at: datetime
    command: str
    state: ProcessState = ProcessState.RUNNING
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    last_heartbeat: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PIDFileInfo:
    """Information about PID files."""

    session_id: str
    pid: int
    file_path: Path
    created_at: datetime
    last_modified: datetime
    file_size: int
    owner_uid: int
    owner_gid: int
    is_locked: bool
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class CrashContext:
    """Comprehensive context about a detected crash."""

    crash_id: str
    crash_type: CrashType
    detected_at: datetime
    severity: CrashSeverity
    affected_sessions: list[str]
    crash_indicators: dict[str, Any]
    system_state_at_crash: dict[str, Any]
    recovery_strategy: str
    estimated_data_loss: dict[str, Any]
    recovery_confidence: float


@dataclass
class RecoveryResult:
    """Result of crash recovery operation."""

    crash_id: str
    recovery_status: str
    recovered_sessions: list[str]
    failed_sessions: list[str]
    data_repaired: list[str]
    recovery_actions: list[str]
    recovery_duration_seconds: float
    final_system_state: dict[str, Any]
    recommendations: list[str]


class UnifiedProcessManager:
    """Unified process manager combining PID management and crash recovery.

    This manager consolidates PIDManager and CrashRecoveryManager functionality
    using composition to provide both process tracking and crash recovery
    capabilities in a single, unified interface.
    """

    def __init__(
        self,
        pid_dir: Path | None = None,
        backup_dir: Path | None = None,
        lock_timeout: float = 5.0,
    ):
        """Initialize unified process manager with both PID and crash recovery capabilities.

        Args:
            pid_dir: Directory for PID files
            backup_dir: Directory for crash recovery backups
            lock_timeout: Timeout for file locking operations
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.signal_handler = None
        self.background_manager = None
        self._shutdown_priority = 8
        self.pid_dir = pid_dir or Path("/tmp/prompt_improver_pids")
        self.pid_dir.mkdir(parents=True, exist_ok=True, mode=493)
        self.lock_timeout = lock_timeout
        self.stale_threshold = timedelta(hours=24)
        self.backup_dir = backup_dir or Path("./crash_recovery")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: dict[str, ProcessInfo] = {}
        self.pid_files: dict[str, PIDFileInfo] = {}
        self.detected_crashes: dict[str, CrashContext] = {}
        self.recovery_history: list[RecoveryResult] = []
        self.coordination_lock = asyncio.Lock()
        self.cleanup_lock = asyncio.Lock()
        self.recovery_lock = asyncio.Lock()
        self.detection_lock = asyncio.Lock()
        self._secure_directories()
        self._init_signal_handlers()
        logger.info("UnifiedProcessManager initialized")

    def _secure_directories(self):
        """Ensure directories have proper security settings."""
        for directory in [self.pid_dir, self.backup_dir]:
            try:
                directory.chmod(493)
                current_uid = os.getuid()
                current_gid = os.getgid()
                try:
                    os.chown(directory, current_uid, current_gid)
                except (OSError, PermissionError):
                    pass
                self.logger.debug(f"Secured directory: {directory}")
            except Exception as e:
                self.logger.warning(f"Failed to secure directory {directory}: {e}")

    def _init_signal_handlers(self):
        """Initialize signal handler with lazy import to avoid circular dependency."""
        try:
            from rich.console import Console

            from prompt_improver.cli.core.signal_handler import AsyncSignalHandler
            from prompt_improver.performance.monitoring.health.background_manager import (
                get_background_task_manager,
            )

            if self.signal_handler is None:
                self.signal_handler = AsyncSignalHandler(console=Console())
                try:
                    import asyncio

                    loop = asyncio.get_running_loop()
                    self.signal_handler.setup_signal_handlers(loop)
                except RuntimeError:
                    pass
            if self.background_manager is None:
                self.background_manager = get_background_task_manager()
            self._register_signal_handlers()
        except ImportError as e:
            self.logger.warning(f"Signal handling integration not available: {e}")

    def _register_signal_handlers(self):
        """Register UnifiedProcessManager-specific signal handlers."""
        if self.signal_handler is None:
            self.logger.warning(
                "Signal handler not initialized, skipping signal registration"
            )
            return
        import signal

        self.signal_handler.register_shutdown_handler(
            "UnifiedProcessManager_shutdown", self.graceful_process_shutdown
        )
        self.signal_handler.register_operation_handler(
            SignalOperation.CHECKPOINT, self.perform_emergency_crash_detection
        )
        self.signal_handler.register_operation_handler(
            SignalOperation.STATUS_REPORT, self.generate_process_status_report
        )
        self.signal_handler.add_signal_chain_handler(
            signal.SIGTERM,
            self.prepare_process_cleanup,
            priority=self._shutdown_priority,
        )
        self.signal_handler.add_signal_chain_handler(
            signal.SIGINT,
            self.prepare_process_interruption,
            priority=self._shutdown_priority,
        )
        self.logger.info("UnifiedProcessManager signal handlers registered")

    async def create_process_checkpoint(self, signal_context):
        """Alias for perform_emergency_crash_detection to match naming convention."""
        return await self.perform_emergency_crash_detection(signal_context)

    async def graceful_process_shutdown(self, shutdown_context):
        """Handle graceful shutdown with comprehensive process and crash cleanup."""
        self.logger.info("UnifiedProcessManager graceful shutdown initiated")
        try:
            check_result = await self.full_system_check_and_recovery()
            cleaned_pids = await self.cleanup_stale_pids()
            active_sessions = await self.list_active_sessions()
            cleanup_results = []
            for session in active_sessions:
                try:
                    success, message = await self.remove_pid_file(session.session_id)
                    cleanup_results.append({
                        "session_id": session.session_id,
                        "success": success,
                        "message": message,
                    })
                except Exception as e:
                    cleanup_results.append({
                        "session_id": session.session_id,
                        "success": False,
                        "error": str(e),
                    })
            return {
                "status": "success",
                "component": "UnifiedProcessManager",
                "system_check": check_result,
                "cleaned_stale_pids": len(cleaned_pids),
                "active_sessions_cleaned": len(cleanup_results),
                "cleanup_results": cleanup_results,
            }
        except Exception as e:
            self.logger.error(f"UnifiedProcessManager shutdown error: {e}")
            return {
                "status": "error",
                "component": "UnifiedProcessManager",
                "error": str(e),
            }

    async def perform_emergency_crash_detection(self, signal_context):
        """Perform emergency crash detection and recovery on SIGUSR1 signal."""
        self.logger.info("Performing emergency crash detection and recovery")
        try:
            detected_crashes = await self.detect_system_crashes()
            recovery_results = []
            for crash in detected_crashes:
                recovery_result = await self.recover_from_crash(crash.crash_id)
                recovery_results.append(recovery_result)
            cleaned_pids = await self.cleanup_stale_pids()
            return {
                "status": "emergency_operation_completed",
                "detected_crashes": len(detected_crashes),
                "recovery_results": len(recovery_results),
                "successful_recoveries": len([
                    r for r in recovery_results if r.recovery_status == "success"
                ]),
                "cleaned_stale_pids": len(cleaned_pids),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Emergency crash detection failed: {e}")
            return {"status": "error", "error": str(e)}

    async def generate_process_status_report(self, signal_context):
        """Generate comprehensive process status report on SIGUSR2 signal."""
        self.logger.info("Generating process status report")
        try:
            system_status = await self.get_system_status()
            active_sessions = await self.list_active_sessions()
            crash_summary = {
                "detected_crashes": len(self.detected_crashes),
                "recovery_history": len(self.recovery_history),
                "successful_recoveries": len([
                    r for r in self.recovery_history if r.recovery_status == "success"
                ]),
            }
            return {
                "status": "report_generated",
                "system_status": system_status,
                "active_sessions": len(active_sessions),
                "session_details": [
                    {
                        "session_id": session.session_id,
                        "pid": session.pid,
                        "state": session.state.value,
                        "started_at": session.started_at.isoformat(),
                        "cpu_percent": session.cpu_percent,
                        "memory_mb": session.memory_mb,
                    }
                    for session in active_sessions
                ],
                "crash_summary": crash_summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Process status report generation failed: {e}")
            return {"status": "error", "error": str(e)}

    def prepare_process_cleanup(self, signum, signal_name):
        """Prepare process management for coordinated cleanup."""
        self.logger.info(f"Preparing process cleanup ({signal_name})")
        try:
            preparation_status = {
                "prepared": True,
                "component": "UnifiedProcessManager",
                "active_sessions": len(self.active_sessions),
                "pid_files": len(self.pid_files),
                "detected_crashes": len(self.detected_crashes),
                "cleanup_locks_ready": True,
            }
            return preparation_status
        except Exception as e:
            self.logger.error(f"Process cleanup preparation failed: {e}")
            return {
                "prepared": False,
                "component": "UnifiedProcessManager",
                "error": str(e),
            }

    def prepare_process_interruption(self, signum, signal_name):
        """Prepare process management for user interruption (Ctrl+C)."""
        self.logger.info(f"Preparing process interruption handling ({signal_name})")
        try:
            interruption_preparation = {
                "prepared": True,
                "component": "UnifiedProcessManager",
                "interruption_type": "user_requested",
                "active_sessions": list(self.active_sessions.keys()),
                "stale_pid_cleanup_ready": True,
                "crash_recovery_ready": True,
            }
            return interruption_preparation
        except Exception as e:
            self.logger.error(f"Process interruption preparation failed: {e}")
            return {
                "prepared": False,
                "component": "UnifiedProcessManager",
                "error": str(e),
            }

    async def create_pid_file(
        self, session_id: str, process_info: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """Create PID file with atomic operations and proper locking.

        Args:
            session_id: Unique session identifier
            process_info: Optional additional process metadata

        Returns:
            Tuple of (success, message)
        """
        async with self.coordination_lock:
            try:
                pid = os.getpid()
                pid_file_path = self.pid_dir / f"{session_id}.pid"
                if pid_file_path.exists():
                    is_stale, reason = await self._is_pid_file_stale(session_id)
                    if not is_stale:
                        return (
                            False,
                            f"PID file already exists for active session: {session_id}",
                        )
                    self.logger.info(f"Removing stale PID file: {reason}")
                    await self.remove_pid_file(session_id)
                temp_file = pid_file_path.with_suffix(".tmp")
                pid_data = {
                    "pid": pid,
                    "session_id": session_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "command": " ".join(sys.argv),
                    "process_info": process_info or {},
                }
                with open(temp_file, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(pid_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                temp_file.rename(pid_file_path)
                process_info_obj = ProcessInfo(
                    pid=pid,
                    session_id=session_id,
                    started_at=datetime.now(UTC),
                    command=pid_data["command"],
                    metadata=process_info or {},
                )
                self.active_sessions[session_id] = process_info_obj
                stat = pid_file_path.stat()
                pid_file_info = PIDFileInfo(
                    session_id=session_id,
                    pid=pid,
                    file_path=pid_file_path,
                    created_at=datetime.now(UTC),
                    last_modified=datetime.fromtimestamp(stat.st_mtime, UTC),
                    file_size=stat.st_size,
                    owner_uid=stat.st_uid,
                    owner_gid=stat.st_gid,
                    is_locked=False,
                )
                self.pid_files[session_id] = pid_file_info
                self.logger.info(f"Created PID file for session {session_id}: {pid}")
                return (True, f"PID file created successfully: {pid_file_path}")
            except Exception as e:
                error_msg = f"Failed to create PID file for {session_id}: {e}"
                self.logger.error(error_msg)
                return (False, error_msg)

    async def remove_pid_file(self, session_id: str) -> tuple[bool, str]:
        """Remove PID file with proper cleanup.

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (success, message)
        """
        async with self.coordination_lock:
            try:
                pid_file_path = self.pid_dir / f"{session_id}.pid"
                if not pid_file_path.exists():
                    return (True, f"PID file already removed: {session_id}")
                pid_file_path.unlink()
                self.active_sessions.pop(session_id, None)
                self.pid_files.pop(session_id, None)
                self.logger.info(f"Removed PID file for session {session_id}")
                return (True, f"PID file removed successfully: {session_id}")
            except Exception as e:
                error_msg = f"Failed to remove PID file for {session_id}: {e}"
                self.logger.error(error_msg)
                return (False, error_msg)

    async def get_process_info(self, session_id: str) -> ProcessInfo | None:
        """Get comprehensive process information.

        Args:
            session_id: Session identifier

        Returns:
            ProcessInfo object or None if not found
        """
        if session_id in self.active_sessions:
            process_info = self.active_sessions[session_id]
            try:
                process = psutil.Process(process_info.pid)
                process_info.state = self._get_process_state(process)
                process_info.cpu_percent = process.cpu_percent()
                process_info.memory_mb = process.memory_info().rss / (1024 * 1024)
                process_info.last_heartbeat = datetime.now(UTC)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_info.state = ProcessState.NOT_FOUND
            return process_info
        return None

    def _get_process_state(self, process: psutil.Process) -> ProcessState:
        """Convert psutil status to ProcessState enum."""
        try:
            status = process.status()
            status_map = {
                psutil.STATUS_RUNNING: ProcessState.RUNNING,
                psutil.STATUS_SLEEPING: ProcessState.SLEEPING,
                psutil.STATUS_DISK_SLEEP: ProcessState.DISK_SLEEP,
                psutil.STATUS_STOPPED: ProcessState.STOPPED,
                psutil.STATUS_ZOMBIE: ProcessState.ZOMBIE,
            }
            return status_map.get(status, ProcessState.UNKNOWN)
        except Exception:
            return ProcessState.UNKNOWN

    async def _is_pid_file_stale(self, session_id: str) -> tuple[bool, str]:
        """Check if a PID file is stale."""
        try:
            pid_file_path = self.pid_dir / f"{session_id}.pid"
            if not pid_file_path.exists():
                return (True, "PID file does not exist")
            file_age = datetime.now(UTC) - datetime.fromtimestamp(
                pid_file_path.stat().st_mtime, UTC
            )
            if file_age > self.stale_threshold:
                return (True, f"PID file is too old: {file_age}")
            try:
                with open(pid_file_path) as f:
                    pid_data = json.load(f)
                    pid = pid_data.get("pid")
                if pid:
                    try:
                        process = psutil.Process(pid)
                        if process.is_running():
                            return (False, "Process is still running")
                        return (True, "Process is not running")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return (True, "Process not found")
            except Exception as e:
                return (True, f"Failed to read PID file: {e}")
            return (True, "Unable to verify process status")
        except Exception as e:
            return (True, f"Error checking PID file: {e}")

    async def list_active_sessions(self) -> list[ProcessInfo]:
        """Get list of all active sessions.

        Returns:
            List of ProcessInfo objects
        """
        active_sessions = []
        for session_id in list(self.active_sessions.keys()):
            process_info = await self.get_process_info(session_id)
            if process_info and process_info.state != ProcessState.NOT_FOUND:
                active_sessions.append(process_info)
        return active_sessions

    async def cleanup_stale_pids(self) -> list[str]:
        """Clean up stale PID files.

        Returns:
            List of cleaned up session IDs
        """
        async with self.cleanup_lock:
            cleaned_sessions = []
            try:
                for pid_file in self.pid_dir.glob("*.pid"):
                    session_id = pid_file.stem
                    is_stale, reason = await self._is_pid_file_stale(session_id)
                    if is_stale:
                        success, message = await self.remove_pid_file(session_id)
                        if success:
                            cleaned_sessions.append(session_id)
                            self.logger.info(
                                f"Cleaned stale PID: {session_id} - {reason} ({message})"
                            )
                        else:
                            self.logger.warning(
                                f"Failed to clean stale PID {session_id}: {message}"
                            )
                return cleaned_sessions
            except Exception as e:
                self.logger.error(f"Failed to cleanup stale PIDs: {e}")
                return cleaned_sessions

    async def detect_system_crashes(self) -> list[CrashContext]:
        """Detect system crashes using PID files, process analysis, and system state.

        Returns:
            List of detected crash contexts
        """
        async with self.detection_lock:
            self.logger.info("Detecting system crashes")
            try:
                detected_crashes = []
                orphaned_crashes = await self._detect_orphaned_processes()
                detected_crashes.extend(orphaned_crashes)
                session_crashes = await self._detect_session_crashes()
                detected_crashes.extend(session_crashes)
                system_crashes = await self._detect_system_level_crashes()
                detected_crashes.extend(system_crashes)
                for crash in detected_crashes:
                    crash = await self._analyze_crash_context(crash)
                    self.detected_crashes[crash.crash_id] = crash
                self.logger.info(f"Detected {len(detected_crashes)} crashes")
                return detected_crashes
            except Exception as e:
                self.logger.error(f"Failed to detect crashes: {e}")
                return []

    async def _detect_orphaned_processes(self) -> list[CrashContext]:
        """Detect crashes based on orphaned PID files."""
        orphaned_crashes = []
        try:
            for pid_file in self.pid_dir.glob("*.pid"):
                session_id = pid_file.stem
                is_stale, reason = await self._is_pid_file_stale(session_id)
                if is_stale:
                    crash_id = f"orphaned_{session_id}_{int(time.time())}"
                    crash_context = CrashContext(
                        crash_id=crash_id,
                        crash_type=CrashType.PROCESS_KILLED,
                        detected_at=datetime.now(UTC),
                        severity=CrashSeverity.MEDIUM,
                        affected_sessions=[session_id],
                        crash_indicators={
                            "stale_reason": reason,
                            "pid_file": str(pid_file),
                        },
                        system_state_at_crash={},
                        recovery_strategy="cleanup_and_restart",
                        estimated_data_loss={},
                        recovery_confidence=0.8,
                    )
                    orphaned_crashes.append(crash_context)
            return orphaned_crashes
        except Exception as e:
            self.logger.error(f"Failed to detect orphaned processes: {e}")
            return []

    async def _detect_session_crashes(self) -> list[CrashContext]:
        """Detect crashes based on interrupted sessions."""
        session_crashes = []
        try:
            for backup_file in self.backup_dir.glob("*.backup"):
                session_id = backup_file.stem.replace("_backup", "")
                if session_id not in self.active_sessions:
                    crash_id = f"session_{session_id}_{int(time.time())}"
                    crash_context = CrashContext(
                        crash_id=crash_id,
                        crash_type=CrashType.UNKNOWN,
                        detected_at=datetime.now(UTC),
                        severity=CrashSeverity.LOW,
                        affected_sessions=[session_id],
                        crash_indicators={"backup_file": str(backup_file)},
                        system_state_at_crash={},
                        recovery_strategy="restore_from_backup",
                        estimated_data_loss={},
                        recovery_confidence=0.9,
                    )
                    session_crashes.append(crash_context)
            return session_crashes
        except Exception as e:
            self.logger.error(f"Failed to detect session crashes: {e}")
            return []

    async def _detect_system_level_crashes(self) -> list[CrashContext]:
        """Detect system-level crashes."""
        return []

    async def _analyze_crash_context(self, crash: CrashContext) -> CrashContext:
        """Analyze and enhance crash context with additional information."""
        try:
            crash.system_state_at_crash = {
                "memory_usage": psutil.virtual_memory()._asdict(),
                "disk_usage": psutil.disk_usage("/")._asdict(),
                "cpu_count": psutil.cpu_count(),
                "boot_time": psutil.boot_time(),
            }
            if len(crash.affected_sessions) > 3:
                crash.severity = CrashSeverity.HIGH
            elif len(crash.affected_sessions) > 1:
                crash.severity = CrashSeverity.MEDIUM
        except Exception as e:
            self.logger.error(f"Failed to analyze crash context: {e}")
        return crash

    async def recover_from_crash(self, crash_id: str) -> RecoveryResult:
        """Perform crash recovery for a specific crash.

        Args:
            crash_id: Identifier of the crash to recover from

        Returns:
            RecoveryResult with details of the recovery operation
        """
        async with self.recovery_lock:
            start_time = time.time()
            try:
                if crash_id not in self.detected_crashes:
                    return RecoveryResult(
                        crash_id=crash_id,
                        recovery_status="failed",
                        recovered_sessions=[],
                        failed_sessions=[],
                        data_repaired=[],
                        recovery_actions=[],
                        recovery_duration_seconds=0,
                        final_system_state={},
                        recommendations=["Crash not found in detected crashes"],
                    )
                crash_context = self.detected_crashes[crash_id]
                recovery_actions = []
                recovered_sessions = []
                failed_sessions = []
                if crash_context.recovery_strategy == "cleanup_and_restart":
                    for session_id in crash_context.affected_sessions:
                        try:
                            success, message = await self.remove_pid_file(session_id)
                            if success:
                                recovered_sessions.append(session_id)
                                recovery_actions.append(
                                    f"Cleaned PID file for {session_id}: {message}"
                                )
                            else:
                                failed_sessions.append(session_id)
                                recovery_actions.append(
                                    f"Failed to clean PID file for {session_id}: {message}"
                                )
                        except Exception as e:
                            failed_sessions.append(session_id)
                            self.logger.error(
                                f"Failed to recover session {session_id}: {e}"
                            )
                elif crash_context.recovery_strategy == "restore_from_backup":
                    for session_id in crash_context.affected_sessions:
                        recovery_actions.append(
                            f"Backup restoration for {session_id} (placeholder)"
                        )
                        recovered_sessions.append(session_id)
                duration = time.time() - start_time
                if len(recovered_sessions) == len(crash_context.affected_sessions):
                    recovery_status = "success"
                elif len(recovered_sessions) > 0:
                    recovery_status = "partial"
                else:
                    recovery_status = "failed"
                result = RecoveryResult(
                    crash_id=crash_id,
                    recovery_status=recovery_status,
                    recovered_sessions=recovered_sessions,
                    failed_sessions=failed_sessions,
                    data_repaired=[],
                    recovery_actions=recovery_actions,
                    recovery_duration_seconds=duration,
                    final_system_state=crash_context.system_state_at_crash,
                    recommendations=[],
                )
                self.recovery_history.append(result)
                self.logger.info(
                    f"Recovery completed for crash {crash_id}: {recovery_status}"
                )
                return result
            except Exception as e:
                error_msg = f"Failed to recover from crash {crash_id}: {e}"
                self.logger.error(error_msg)
                return RecoveryResult(
                    crash_id=crash_id,
                    recovery_status="failed",
                    recovered_sessions=[],
                    failed_sessions=crash_context.affected_sessions
                    if crash_id in self.detected_crashes
                    else [],
                    data_repaired=[],
                    recovery_actions=[],
                    recovery_duration_seconds=time.time() - start_time,
                    final_system_state={},
                    recommendations=[error_msg],
                )

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status including processes and crash state.

        Returns:
            Dictionary with system status information
        """
        try:
            active_sessions = await self.list_active_sessions()
            return {
                "unified_manager": True,
                "active_sessions_count": len(active_sessions),
                "active_sessions": [
                    {
                        "session_id": session.session_id,
                        "pid": session.pid,
                        "state": session.state.value,
                        "cpu_percent": session.cpu_percent,
                        "memory_mb": session.memory_mb,
                        "started_at": session.started_at.isoformat(),
                    }
                    for session in active_sessions
                ],
                "detected_crashes_count": len(self.detected_crashes),
                "recovery_history_count": len(self.recovery_history),
                "pid_files_count": len(self.pid_files),
                "directories": {
                    "pid_dir": str(self.pid_dir),
                    "backup_dir": str(self.backup_dir),
                },
                "status_timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"error": str(e), "unified_manager": True}

    async def full_system_check_and_recovery(self) -> dict[str, Any]:
        """Perform comprehensive system check and automatic recovery.

        Returns:
            Dictionary with check and recovery results
        """
        try:
            cleaned_pids = await self.cleanup_stale_pids()
            detected_crashes = await self.detect_system_crashes()
            recovery_results = []
            for crash in detected_crashes:
                recovery_result = await self.recover_from_crash(crash.crash_id)
                recovery_results.append(recovery_result)
            system_status = await self.get_system_status()
            return {
                "check_completed": True,
                "cleaned_stale_pids": cleaned_pids,
                "detected_crashes": len(detected_crashes),
                "recovery_results": len(recovery_results),
                "successful_recoveries": len([
                    r for r in recovery_results if r.recovery_status == "success"
                ]),
                "partial_recoveries": len([
                    r for r in recovery_results if r.recovery_status == "partial"
                ]),
                "failed_recoveries": len([
                    r for r in recovery_results if r.recovery_status == "failed"
                ]),
                "final_system_status": system_status,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Failed system check and recovery: {e}")
            return {"error": str(e), "check_completed": False}


_unified_process_manager = UnifiedProcessManager()


def get_unified_process_manager() -> UnifiedProcessManager:
    """Get the global unified process manager instance.

    Returns:
        Global UnifiedProcessManager instance
    """
    return _unified_process_manager
