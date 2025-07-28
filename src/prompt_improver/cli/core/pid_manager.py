"""
Enhanced PID Manager for Multi-Session Process Tracking
Implements 2025 best practices for PID file management with atomic operations,
stale detection, and comprehensive process coordination.
"""

import asyncio
import fcntl
import json
import logging
import os
import psutil
import signal

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import tempfile

class ProcessState(Enum):
    """Enumeration of process states for tracking."""
    RUNNING = "running"
    STOPPED = "stopped"
    ZOMBIE = "zombie"
    SLEEPING = "sleeping"
    DISK_SLEEP = "disk_sleep"
    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"

class PIDFileStatus(Enum):
    """Enumeration of PID file status."""
    VALID = "valid"
    STALE = "stale"
    CORRUPTED = "corrupted"
    LOCKED = "locked"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"

@dataclass
class ProcessInfo:
    """Comprehensive process information."""
    pid: int
    session_id: str
    process_name: str
    command_line: List[str]
    state: ProcessState
    cpu_percent: float
    memory_percent: float
    create_time: datetime
    owner_uid: int
    owner_gid: int
    working_directory: str
    environment: Dict[str, str]
    parent_pid: Optional[int]
    children_pids: List[int]

@dataclass
class PIDFileInfo:
    """PID file metadata and validation information."""
    file_path: Path
    status: PIDFileStatus
    pid: Optional[int]
    session_id: Optional[str]
    created_at: Optional[datetime]
    last_modified: Optional[datetime]
    file_size: int
    permissions: str
    owner_uid: int
    owner_gid: int
    is_locked: bool
    validation_errors: List[str]

class PIDManager:
    """
    Enhanced PID Manager implementing 2025 best practices for process tracking.

    Features:
    - Atomic PID file operations using fcntl locking
    - Intelligent stale PID detection and cleanup
    - Multi-session process coordination
    - Comprehensive process health monitoring
    - Secure file operations with proper permissions
    - Cross-platform compatibility (Unix/Linux)
    - Integration with modern init systems
    """

    def __init__(self, pid_dir: Optional[Path] = None, lock_timeout: float = 5.0):
        self.logger = logging.getLogger(__name__)

        # PID directory setup with secure permissions using centralized config
        if pid_dir is None:
            try:
                from ...core.config import get_config
                config = get_config()
                self.pid_dir = config.directory_paths.pid_dir
                self.secure_permissions = config.directory_paths.secure_temp_permissions
            except ImportError:
                # Fallback if config not available
                self.pid_dir = Path("/tmp/prompt_improver_pids")
                self.secure_permissions = 0o755
        else:
            self.pid_dir = pid_dir
            self.secure_permissions = 0o755
            
        self.pid_dir.mkdir(parents=True, exist_ok=True, mode=self.secure_permissions)

        # Ensure proper ownership and permissions
        self._secure_pid_directory()

        # Configuration
        self.lock_timeout = lock_timeout
        self.stale_threshold = timedelta(hours=24)  # Consider PID stale after 24 hours

        # Process tracking
        self.active_sessions: Dict[str, ProcessInfo] = {}
        self.pid_files: Dict[str, PIDFileInfo] = {}

        # Coordination locks
        self.coordination_lock = asyncio.Lock()
        self.cleanup_lock = asyncio.Lock()

    def _secure_pid_directory(self):
        """Ensure PID directory has proper security settings."""
        try:
            # Set directory permissions (rwxr-xr-x)
            self.pid_dir.chmod(0o755)

            # Try to set ownership to current user (may fail without privileges)
            try:
                current_uid = os.getuid()
                current_gid = os.getgid()
                os.chown(self.pid_dir, current_uid, current_gid)
            except (OSError, PermissionError):
                # Not critical if we can't change ownership
                pass

            self.logger.debug(f"Secured PID directory: {self.pid_dir}")

        except Exception as e:
            self.logger.warning(f"Failed to secure PID directory: {e}")

    async def create_pid_file(
        self,
        session_id: str,
        process_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Create PID file with atomic operations and proper locking.

        Args:
            session_id: Unique session identifier
            process_info: Additional process metadata

        Returns:
            Tuple of (success, message)
        """
        pid_file_path = self.pid_dir / f"{session_id}.pid"

        try:
            # Get current process information
            current_pid = os.getpid()
            current_process = psutil.Process(current_pid)

            # Prepare PID file data
            pid_data = {
                "pid": current_pid,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "process_name": current_process.name(),
                "command_line": current_process.cmdline(),
                "working_directory": str(Path.cwd()),
                "owner_uid": os.getuid(),
                "owner_gid": os.getgid(),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "hostname": os.uname().nodename,
                "additional_info": process_info or {}
            }

            # Atomic PID file creation with exclusive locking
            success, message = await self._atomic_pid_file_write(pid_file_path, pid_data)

            if success:
                # Update internal tracking
                process_info_obj = await self._gather_process_info(current_pid, session_id)
                self.active_sessions[session_id] = process_info_obj

                # Create PID file info
                pid_file_info = PIDFileInfo(
                    file_path=pid_file_path,
                    status=PIDFileStatus.VALID,
                    pid=current_pid,
                    session_id=session_id,
                    created_at=datetime.now(timezone.utc),
                    last_modified=datetime.now(timezone.utc),
                    file_size=pid_file_path.stat().st_size,
                    permissions=oct(pid_file_path.stat().st_mode)[-3:],
                    owner_uid=os.getuid(),
                    owner_gid=os.getgid(),
                    is_locked=False,
                    validation_errors=[]
                )
                self.pid_files[session_id] = pid_file_info

                self.logger.info(f"Created PID file for session {session_id}: {pid_file_path}")

            return success, message

        except Exception as e:
            error_msg = f"Failed to create PID file for {session_id}: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    async def _atomic_pid_file_write(self, pid_file_path: Path, pid_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Perform atomic PID file write with exclusive locking."""
        temp_file = None
        lock_fd = None

        try:
            # Create temporary file in same directory for atomic operation
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                dir=pid_file_path.parent,
                prefix=f".{pid_file_path.name}.",
                suffix=".tmp",
                delete=False
            )

            # Acquire exclusive lock on temporary file
            lock_fd = temp_file.fileno()
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Check if target PID file already exists and is valid
            if pid_file_path.exists():
                existing_status = await self._validate_pid_file(pid_file_path)
                if existing_status.status == PIDFileStatus.VALID:
                    return False, f"Valid PID file already exists for session"
                elif existing_status.status == PIDFileStatus.LOCKED:
                    return False, f"PID file is locked by another process"

            # Write PID data to temporary file
            json.dump(pid_data, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())

            # Set proper permissions (rw-r--r--)
            os.chmod(temp_file.name, 0o644)

            # Atomic move to final location
            os.rename(temp_file.name, pid_file_path)

            return True, "PID file created successfully"

        except (OSError, IOError) as e:
            if e.errno == 11:  # EAGAIN - would block
                return False, "PID file is locked by another process"
            else:
                return False, f"Failed to create PID file: {e}"
        except Exception as e:
            return False, f"Unexpected error creating PID file: {e}"
        finally:
            # Cleanup
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except:
                    pass

            if temp_file is not None:
                temp_file.close()
                # Remove temp file if atomic move failed
                temp_path = Path(temp_file.name)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass

    async def _gather_process_info(self, pid: int, session_id: str) -> ProcessInfo:
        """Gather comprehensive process information."""
        try:
            process = psutil.Process(pid)

            # Map psutil status to our enum
            status_map = {
                psutil.STATUS_RUNNING: ProcessState.RUNNING,
                psutil.STATUS_SLEEPING: ProcessState.SLEEPING,
                psutil.STATUS_DISK_SLEEP: ProcessState.DISK_SLEEP,
                psutil.STATUS_STOPPED: ProcessState.STOPPED,
                psutil.STATUS_ZOMBIE: ProcessState.ZOMBIE,
            }

            process_state = status_map.get(process.status(), ProcessState.UNKNOWN)

            # Get process metrics
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()

            # Get process details
            create_time = datetime.fromtimestamp(process.create_time(), tz=timezone.utc)

            # Get owner information
            owner_uid = process.uids().real
            owner_gid = process.gids().real

            # Get working directory (may fail for some processes)
            try:
                working_dir = process.cwd()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                working_dir = "unknown"

            # Get environment (may fail for some processes)
            try:
                environment = process.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                environment = {}

            # Get parent and children
            try:
                parent_pid = process.ppid()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                parent_pid = None

            try:
                children_pids = [child.pid for child in process.children()]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                children_pids = []

            return ProcessInfo(
                pid=pid,
                session_id=session_id,
                process_name=process.name(),
                command_line=process.cmdline(),
                state=process_state,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                create_time=create_time,
                owner_uid=owner_uid,
                owner_gid=owner_gid,
                working_directory=working_dir,
                environment=environment,
                parent_pid=parent_pid,
                children_pids=children_pids
            )

        except psutil.NoSuchProcess:
            # Process no longer exists
            return ProcessInfo(
                pid=pid,
                session_id=session_id,
                process_name="unknown",
                command_line=[],
                state=ProcessState.NOT_FOUND,
                cpu_percent=0.0,
                memory_percent=0.0,
                create_time=datetime.now(timezone.utc),
                owner_uid=os.getuid(),
                owner_gid=os.getgid(),
                working_directory="unknown",
                environment={},
                parent_pid=None,
                children_pids=[]
            )
        except Exception as e:
            self.logger.warning(f"Failed to gather process info for PID {pid}: {e}")
            raise

    async def _validate_pid_file(self, pid_file_path: Path) -> PIDFileInfo:
        """Validate PID file and return comprehensive status information."""
        validation_errors = []

        try:
            # Check file existence
            if not pid_file_path.exists():
                return PIDFileInfo(
                    file_path=pid_file_path,
                    status=PIDFileStatus.NOT_FOUND,
                    pid=None,
                    session_id=None,
                    created_at=None,
                    last_modified=None,
                    file_size=0,
                    permissions="000",
                    owner_uid=0,
                    owner_gid=0,
                    is_locked=False,
                    validation_errors=["File does not exist"]
                )

            # Get file stats
            file_stat = pid_file_path.stat()
            file_size = file_stat.st_size
            permissions = oct(file_stat.st_mode)[-3:]
            owner_uid = file_stat.st_uid
            owner_gid = file_stat.st_gid
            last_modified = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)

            # Check file permissions
            if not os.access(pid_file_path, os.R_OK):
                validation_errors.append("File is not readable")
                return PIDFileInfo(
                    file_path=pid_file_path,
                    status=PIDFileStatus.PERMISSION_DENIED,
                    pid=None,
                    session_id=None,
                    created_at=None,
                    last_modified=last_modified,
                    file_size=file_size,
                    permissions=permissions,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                    is_locked=False,
                    validation_errors=validation_errors
                )

            # Check if file is locked
            is_locked = await self._check_file_lock(pid_file_path)

            # Read and parse PID file content
            try:
                with open(pid_file_path, 'r') as f:
                    # Try to acquire shared lock for reading
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                        pid_data = json.load(f)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (OSError, IOError) as e:
                        if e.errno == 11:  # EAGAIN - would block
                            is_locked = True
                        raise

            except json.JSONDecodeError as e:
                validation_errors.append(f"Invalid JSON format: {e}")
                return PIDFileInfo(
                    file_path=pid_file_path,
                    status=PIDFileStatus.CORRUPTED,
                    pid=None,
                    session_id=None,
                    created_at=None,
                    last_modified=last_modified,
                    file_size=file_size,
                    permissions=permissions,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                    is_locked=is_locked,
                    validation_errors=validation_errors
                )
            except Exception as e:
                validation_errors.append(f"Failed to read file: {e}")
                status = PIDFileStatus.LOCKED if is_locked else PIDFileStatus.CORRUPTED
                return PIDFileInfo(
                    file_path=pid_file_path,
                    status=status,
                    pid=None,
                    session_id=None,
                    created_at=None,
                    last_modified=last_modified,
                    file_size=file_size,
                    permissions=permissions,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                    is_locked=is_locked,
                    validation_errors=validation_errors
                )

            # Extract PID and session information
            pid = pid_data.get("pid")
            session_id = pid_data.get("session_id")
            created_at_str = pid_data.get("created_at")

            # Validate required fields
            if not isinstance(pid, int) or pid <= 0:
                validation_errors.append("Invalid or missing PID")

            if not session_id:
                validation_errors.append("Missing session ID")

            # Parse creation time
            created_at = None
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except ValueError:
                    validation_errors.append("Invalid creation timestamp format")

            # Check if process exists and is valid
            process_valid = False
            if pid and isinstance(pid, int):
                try:
                    process = psutil.Process(pid)
                    process_valid = process.is_running()

                    # Additional validation: check if process matches expected characteristics
                    if process_valid:
                        # Verify process owner matches PID file owner
                        try:
                            process_uid = process.uids().real
                            if process_uid != owner_uid:
                                validation_errors.append(f"Process owner mismatch: {process_uid} vs {owner_uid}")
                                process_valid = False
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            pass

                        # Check process age vs PID file age
                        if created_at:
                            try:
                                process_create_time = datetime.fromtimestamp(process.create_time(), tz=timezone.utc)
                                time_diff = abs((process_create_time - created_at).total_seconds())
                                if time_diff > 60:  # Allow 1 minute tolerance
                                    validation_errors.append(f"Process creation time mismatch: {time_diff}s difference")
                                    process_valid = False
                            except (psutil.AccessDenied, psutil.NoSuchProcess):
                                pass

                except psutil.NoSuchProcess:
                    validation_errors.append("Process no longer exists")
                except psutil.AccessDenied:
                    validation_errors.append("Access denied to process information")
                except Exception as e:
                    validation_errors.append(f"Error checking process: {e}")

            # Check for stale PID file
            if created_at and datetime.now(timezone.utc) - created_at > self.stale_threshold:
                validation_errors.append("PID file is stale (older than threshold)")

            # Determine final status
            if validation_errors:
                if not process_valid:
                    status = PIDFileStatus.STALE
                else:
                    status = PIDFileStatus.CORRUPTED
            else:
                status = PIDFileStatus.VALID

            return PIDFileInfo(
                file_path=pid_file_path,
                status=status,
                pid=pid,
                session_id=session_id,
                created_at=created_at,
                last_modified=last_modified,
                file_size=file_size,
                permissions=permissions,
                owner_uid=owner_uid,
                owner_gid=owner_gid,
                is_locked=is_locked,
                validation_errors=validation_errors
            )

        except Exception as e:
            self.logger.error(f"Failed to validate PID file {pid_file_path}: {e}")
            return PIDFileInfo(
                file_path=pid_file_path,
                status=PIDFileStatus.CORRUPTED,
                pid=None,
                session_id=None,
                created_at=None,
                last_modified=None,
                file_size=0,
                permissions="000",
                owner_uid=0,
                owner_gid=0,
                is_locked=False,
                validation_errors=[f"Validation failed: {e}"]
            )

    async def _check_file_lock(self, file_path: Path) -> bool:
        """Check if file is currently locked by another process."""
        try:
            with open(file_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return False
        except (OSError, IOError) as e:
            if e.errno == 11:  # EAGAIN - would block
                return True
            return False
        except Exception:
            return False

    async def remove_pid_file(self, session_id: str, force: bool = False) -> Tuple[bool, str]:
        """
        Remove PID file with proper validation and cleanup.

        Args:
            session_id: Session identifier
            force: Force removal even if process is still running

        Returns:
            Tuple of (success, message)
        """
        pid_file_path = self.pid_dir / f"{session_id}.pid"

        try:
            # Validate PID file first
            pid_info = await self._validate_pid_file(pid_file_path)

            if pid_info.status == PIDFileStatus.NOT_FOUND:
                return True, "PID file does not exist"

            # Check if we should remove the file
            should_remove = False
            reason = ""

            if force:
                should_remove = True
                reason = "Forced removal requested"
            elif pid_info.status in [PIDFileStatus.STALE, PIDFileStatus.CORRUPTED]:
                should_remove = True
                reason = f"PID file is {pid_info.status.value}"
            elif pid_info.status == PIDFileStatus.VALID:
                # Check if current process owns this PID file
                current_pid = os.getpid()
                if pid_info.pid == current_pid:
                    should_remove = True
                    reason = "Current process owns PID file"
                else:
                    return False, f"Cannot remove PID file: process {pid_info.pid} is still running"
            elif pid_info.status == PIDFileStatus.LOCKED:
                return False, "Cannot remove PID file: file is locked by another process"
            elif pid_info.status == PIDFileStatus.PERMISSION_DENIED:
                return False, "Cannot remove PID file: permission denied"

            if should_remove:
                # Perform atomic removal with locking
                success, message = await self._atomic_pid_file_remove(pid_file_path)

                if success:
                    # Update internal tracking
                    self.active_sessions.pop(session_id, None)
                    self.pid_files.pop(session_id, None)

                    self.logger.info(f"Removed PID file for session {session_id}: {reason}")

                return success, message
            else:
                return False, f"PID file removal not allowed: {pid_info.status.value}"

        except Exception as e:
            error_msg = f"Failed to remove PID file for {session_id}: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    async def _atomic_pid_file_remove(self, pid_file_path: Path) -> Tuple[bool, str]:
        """Perform atomic PID file removal with proper locking."""
        try:
            # Try to acquire exclusive lock before removal
            with open(pid_file_path, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Double-check file content before removal
                f.seek(0)
                try:
                    pid_data = json.load(f)
                    current_pid = os.getpid()

                    # Only allow removal if current process owns the PID file
                    # or if the referenced process no longer exists
                    file_pid = pid_data.get("pid")
                    if file_pid == current_pid:
                        # Current process owns it - safe to remove
                        pass
                    elif file_pid:
                        # Check if referenced process still exists
                        try:
                            psutil.Process(file_pid)
                            # Process still exists - don't remove unless forced
                            return False, f"Process {file_pid} is still running"
                        except psutil.NoSuchProcess:
                            # Process no longer exists - safe to remove
                            pass

                except json.JSONDecodeError:
                    # Corrupted file - safe to remove
                    pass

                # Release lock before unlinking
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Remove the file
            pid_file_path.unlink()
            return True, "PID file removed successfully"

        except (OSError, IOError) as e:
            if e.errno == 11:  # EAGAIN - would block
                return False, "PID file is locked by another process"
            else:
                return False, f"Failed to remove PID file: {e}"
        except Exception as e:
            return False, f"Unexpected error removing PID file: {e}"

    async def scan_all_pid_files(self) -> Dict[str, PIDFileInfo]:
        """
        Scan all PID files in the directory and return their status.

        Returns:
            Dictionary mapping session IDs to PID file information
        """
        async with self.coordination_lock:
            self.logger.debug("Scanning all PID files")

            try:
                pid_files = {}

                # Find all .pid files in the directory
                for pid_file_path in self.pid_dir.glob("*.pid"):
                    try:
                        # Extract session ID from filename
                        session_id = pid_file_path.stem

                        # Validate the PID file
                        pid_info = await self._validate_pid_file(pid_file_path)
                        pid_files[session_id] = pid_info

                        # Update internal tracking
                        self.pid_files[session_id] = pid_info

                    except Exception as e:
                        self.logger.warning(f"Failed to process PID file {pid_file_path}: {e}")
                        # Create error entry
                        pid_files[pid_file_path.stem] = PIDFileInfo(
                            file_path=pid_file_path,
                            status=PIDFileStatus.CORRUPTED,
                            pid=None,
                            session_id=None,
                            created_at=None,
                            last_modified=None,
                            file_size=0,
                            permissions="000",
                            owner_uid=0,
                            owner_gid=0,
                            is_locked=False,
                            validation_errors=[f"Scan error: {e}"]
                        )

                self.logger.info(f"Scanned {len(pid_files)} PID files")
                return pid_files

            except Exception as e:
                self.logger.error(f"Failed to scan PID files: {e}")
                return {}

    async def cleanup_stale_pid_files(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up stale and corrupted PID files.

        Args:
            dry_run: If True, only report what would be cleaned up

        Returns:
            Cleanup report with details of actions taken
        """
        async with self.cleanup_lock:
            self.logger.info(f"Starting PID file cleanup (dry_run={dry_run})")

            cleanup_report = {
                "scanned": 0,
                "stale_removed": [],
                "corrupted_removed": [],
                "locked_skipped": [],
                "permission_denied": [],
                "errors": [],
                "total_removed": 0
            }

            try:
                # Scan all PID files
                pid_files = await self.scan_all_pid_files()
                cleanup_report["scanned"] = len(pid_files)

                for session_id, pid_info in pid_files.items():
                    try:
                        should_remove = False
                        reason = ""

                        if pid_info.status == PIDFileStatus.STALE:
                            should_remove = True
                            reason = "stale"
                        elif pid_info.status == PIDFileStatus.CORRUPTED:
                            should_remove = True
                            reason = "corrupted"
                        elif pid_info.status == PIDFileStatus.PERMISSION_DENIED:
                            cleanup_report["permission_denied"].append({
                                "session_id": session_id,
                                "path": str(pid_info.file_path),
                                "errors": pid_info.validation_errors
                            })
                        elif pid_info.status == PIDFileStatus.LOCKED:
                            cleanup_report["locked_skipped"].append({
                                "session_id": session_id,
                                "path": str(pid_info.file_path),
                                "pid": pid_info.pid
                            })

                        if should_remove:
                            if dry_run:
                                # Just report what would be removed
                                if reason == "stale":
                                    cleanup_report["stale_removed"].append({
                                        "session_id": session_id,
                                        "path": str(pid_info.file_path),
                                        "pid": pid_info.pid,
                                        "errors": pid_info.validation_errors
                                    })
                                else:
                                    cleanup_report["corrupted_removed"].append({
                                        "session_id": session_id,
                                        "path": str(pid_info.file_path),
                                        "errors": pid_info.validation_errors
                                    })
                            else:
                                # Actually remove the file
                                success, message = await self.remove_pid_file(session_id, force=True)

                                if success:
                                    if reason == "stale":
                                        cleanup_report["stale_removed"].append({
                                            "session_id": session_id,
                                            "path": str(pid_info.file_path),
                                            "pid": pid_info.pid,
                                            "message": message
                                        })
                                    else:
                                        cleanup_report["corrupted_removed"].append({
                                            "session_id": session_id,
                                            "path": str(pid_info.file_path),
                                            "message": message
                                        })
                                    cleanup_report["total_removed"] += 1
                                else:
                                    cleanup_report["errors"].append({
                                        "session_id": session_id,
                                        "path": str(pid_info.file_path),
                                        "error": message
                                    })

                    except Exception as e:
                        cleanup_report["errors"].append({
                            "session_id": session_id,
                            "error": str(e)
                        })

                action = "Would remove" if dry_run else "Removed"
                total_would_remove = len(cleanup_report["stale_removed"]) + len(cleanup_report["corrupted_removed"])
                self.logger.info(f"Cleanup complete: {action} {total_would_remove} PID files")

                return cleanup_report

            except Exception as e:
                cleanup_report["errors"].append({"error": f"Cleanup failed: {e}"})
                self.logger.error(f"PID file cleanup failed: {e}")
                return cleanup_report

    async def get_active_sessions(self) -> Dict[str, ProcessInfo]:
        """
        Get information about all active sessions.

        Returns:
            Dictionary mapping session IDs to process information
        """
        try:
            # Refresh session information
            await self.scan_all_pid_files()

            active_sessions = {}

            for session_id, pid_info in self.pid_files.items():
                if pid_info.status == PIDFileStatus.VALID and pid_info.pid:
                    try:
                        # Gather current process information
                        process_info = await self._gather_process_info(pid_info.pid, session_id)
                        if process_info.state != ProcessState.NOT_FOUND:
                            active_sessions[session_id] = process_info
                            self.active_sessions[session_id] = process_info
                    except Exception as e:
                        self.logger.warning(f"Failed to get process info for session {session_id}: {e}")

            return active_sessions

        except Exception as e:
            self.logger.error(f"Failed to get active sessions: {e}")
            return {}

    async def send_signal_to_session(self, session_id: str, signal_num: int) -> Tuple[bool, str]:
        """
        Send signal to a specific session process.

        Args:
            session_id: Session identifier
            signal_num: Signal number to send

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get PID file information
            pid_file_path = self.pid_dir / f"{session_id}.pid"
            pid_info = await self._validate_pid_file(pid_file_path)

            if pid_info.status != PIDFileStatus.VALID:
                return False, f"Cannot send signal: PID file is {pid_info.status.value}"

            if not pid_info.pid:
                return False, "No PID found in PID file"

            # Verify process exists
            try:
                process = psutil.Process(pid_info.pid)
                if not process.is_running():
                    return False, f"Process {pid_info.pid} is not running"
            except psutil.NoSuchProcess:
                return False, f"Process {pid_info.pid} does not exist"

            # Send signal
            try:
                os.kill(pid_info.pid, signal_num)
                signal_name = signal.Signals(signal_num).name
                return True, f"Sent signal {signal_name} ({signal_num}) to process {pid_info.pid}"
            except ProcessLookupError:
                return False, f"Process {pid_info.pid} no longer exists"
            except PermissionError:
                return False, f"Permission denied to send signal to process {pid_info.pid}"
            except Exception as e:
                return False, f"Failed to send signal: {e}"

        except Exception as e:
            return False, f"Failed to send signal to session {session_id}: {e}"

    async def get_session_health(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive health information for a session.

        Args:
            session_id: Session identifier

        Returns:
            Health report dictionary
        """
        try:
            health_report = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pid_file_status": "not_found",
                "process_status": "not_found",
                "health_score": 0.0,
                "issues": [],
                "recommendations": []
            }

            # Check PID file
            pid_file_path = self.pid_dir / f"{session_id}.pid"
            pid_info = await self._validate_pid_file(pid_file_path)
            health_report["pid_file_status"] = pid_info.status.value

            if pid_info.validation_errors:
                health_report["issues"].extend(pid_info.validation_errors)

            # Check process if PID is available
            if pid_info.pid:
                try:
                    process_info = await self._gather_process_info(pid_info.pid, session_id)
                    health_report["process_status"] = process_info.state.value

                    # Calculate health score
                    health_score = 0.0

                    # PID file health (40% of score)
                    if pid_info.status == PIDFileStatus.VALID:
                        health_score += 0.4
                    elif pid_info.status in [PIDFileStatus.STALE, PIDFileStatus.CORRUPTED]:
                        health_score += 0.1

                    # Process health (60% of score)
                    if process_info.state == ProcessState.RUNNING:
                        health_score += 0.6
                    elif process_info.state in [ProcessState.SLEEPING, ProcessState.DISK_SLEEP]:
                        health_score += 0.4
                    elif process_info.state == ProcessState.STOPPED:
                        health_score += 0.2

                    health_report["health_score"] = health_score

                    # Add process metrics
                    health_report["process_metrics"] = {
                        "cpu_percent": process_info.cpu_percent,
                        "memory_percent": process_info.memory_percent,
                        "uptime_seconds": (datetime.now(timezone.utc) - process_info.create_time).total_seconds(),
                        "children_count": len(process_info.children_pids)
                    }

                    # Generate recommendations
                    if health_score < 0.5:
                        health_report["recommendations"].append("Session appears unhealthy - consider restart")
                    if process_info.cpu_percent > 90:
                        health_report["recommendations"].append("High CPU usage detected")
                    if process_info.memory_percent > 90:
                        health_report["recommendations"].append("High memory usage detected")

                except Exception as e:
                    health_report["issues"].append(f"Failed to gather process info: {e}")

            return health_report

        except Exception as e:
            return {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "health_score": 0.0
            }

    async def coordinate_multi_session_operation(
        self,
        operation: str,
        session_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Coordinate operations across multiple sessions.

        Args:
            operation: Operation to perform ('cleanup', 'signal', 'health_check')
            session_ids: Specific sessions to target (None for all)
            **kwargs: Operation-specific parameters

        Returns:
            Operation results
        """
        async with self.coordination_lock:
            self.logger.info(f"Coordinating multi-session operation: {operation}")

            try:
                # Get target sessions
                if session_ids is None:
                    # Get all sessions
                    all_pid_files = await self.scan_all_pid_files()
                    target_sessions = list(all_pid_files.keys())
                else:
                    target_sessions = session_ids

                results = {
                    "operation": operation,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "target_sessions": target_sessions,
                    "results": {},
                    "summary": {
                        "total": len(target_sessions),
                        "successful": 0,
                        "failed": 0,
                        "errors": []
                    }
                }

                # Execute operation on each session
                for session_id in target_sessions:
                    try:
                        if operation == "cleanup":
                            # Individual session cleanup
                            success, message = await self.remove_pid_file(session_id, force=kwargs.get("force", False))
                            results["results"][session_id] = {
                                "success": success,
                                "message": message
                            }
                        elif operation == "signal":
                            # Send signal to session
                            signal_num = kwargs.get("signal", signal.SIGTERM)
                            success, message = await self.send_signal_to_session(session_id, signal_num)
                            results["results"][session_id] = {
                                "success": success,
                                "message": message
                            }
                        elif operation == "health_check":
                            # Get session health
                            health_report = await self.get_session_health(session_id)
                            results["results"][session_id] = health_report
                            success = health_report.get("health_score", 0) > 0.5
                        else:
                            results["results"][session_id] = {
                                "success": False,
                                "message": f"Unknown operation: {operation}"
                            }
                            success = False

                        if success:
                            results["summary"]["successful"] += 1
                        else:
                            results["summary"]["failed"] += 1

                    except Exception as e:
                        error_msg = f"Operation failed for {session_id}: {e}"
                        results["results"][session_id] = {
                            "success": False,
                            "error": error_msg
                        }
                        results["summary"]["failed"] += 1
                        results["summary"]["errors"].append(error_msg)

                self.logger.info(f"Multi-session operation complete: {results['summary']['successful']}/{results['summary']['total']} successful")
                return results

            except Exception as e:
                error_msg = f"Multi-session operation failed: {e}"
                self.logger.error(error_msg)
                return {
                    "operation": operation,
                    "error": error_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
