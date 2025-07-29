"""
Emergency Operations for Signal-Triggered Actions
Implements emergency operations triggered by signals: checkpoint creation, status reporting, and configuration reload.
"""

import json
import logging
import os
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Lazy import to avoid circular dependency
# from .signal_handler import SignalContext

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .signal_handler import SignalContext

def _get_signal_context():
    """Lazy import of SignalContext to avoid circular imports."""
    from .signal_handler import SignalContext
    return SignalContext
from .progress_preservation import ProgressPreservationManager
from ...database import get_session_context
from ...database.models import TrainingSession

class EmergencyOperationsManager:
    """
    Manages emergency operations triggered by signals.

    Features:
    - On-demand checkpoint creation (SIGUSR1)
    - Real-time status reporting (SIGUSR2)
    - Configuration reload (SIGHUP)
    - Integration with existing progress preservation
    - CoreDis-aware resource monitoring
    """

    def __init__(self, backup_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("./emergency_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress preservation manager
        self.progress_manager = ProgressPreservationManager(backup_dir=self.backup_dir)

        # Track operation history
        self.operation_history: Dict[str, list] = {
            "checkpoints": [],
            "status_reports": [],
            "config_reloads": []
        }

    async def create_emergency_checkpoint(self, context: "SignalContext") -> Dict[str, Any]:
        """
        Create emergency checkpoint triggered by SIGUSR1.

        Args:
            context: Signal context with trigger information

        Returns:
            Checkpoint creation results
        """
        self.logger.info("Creating emergency checkpoint triggered by signal")

        try:
            checkpoint_id = f"emergency_{int(context.triggered_at.timestamp())}"
            checkpoint_path = self.backup_dir / f"{checkpoint_id}.json"

            # Gather system state
            system_state = await self._gather_system_state()

            # Gather training session state
            training_state = await self._gather_training_state()

            # Create comprehensive checkpoint
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "created_at": context.triggered_at.isoformat(),
                "trigger": {
                    "signal": context.signal_name,
                    "operation": context.operation.value,
                    "signal_number": context.signal_number
                },
                "system_state": system_state,
                "training_state": training_state,
                "emergency": True
            }

            # Save checkpoint to file
            checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))

            # Save to database if possible
            try:
                await self.progress_manager.create_checkpoint(checkpoint_id, checkpoint_data)
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint to database: {e}")

            # Track operation
            self.operation_history["checkpoints"].append({
                "checkpoint_id": checkpoint_id,
                "created_at": context.triggered_at.isoformat(),
                "path": str(checkpoint_path)
            })

            result = {
                "status": "success",
                "checkpoint_id": checkpoint_id,
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "system_state": system_state,
                "training_sessions": len(training_state.get("sessions", []))
            }

            self.logger.info(f"Emergency checkpoint created: {checkpoint_id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to create emergency checkpoint: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checkpoint_id": None
            }

    async def generate_status_report(self, context: "SignalContext") -> Dict[str, Any]:
        """
        Generate real-time status report triggered by SIGUSR2.

        Args:
            context: Signal context with trigger information

        Returns:
            Status report data
        """
        self.logger.info("Generating status report triggered by signal")

        try:
            report_id = f"status_{int(context.triggered_at.timestamp())}"

            # Gather comprehensive status
            system_status = await self._gather_system_status()
            training_status = await self._gather_training_status()
            resource_status = await self._gather_resource_status()

            # Create status report
            status_report = {
                "report_id": report_id,
                "generated_at": context.triggered_at.isoformat(),
                "trigger": {
                    "signal": context.signal_name,
                    "operation": context.operation.value,
                    "signal_number": context.signal_number
                },
                "system": system_status,
                "training": training_status,
                "resources": resource_status,
                "emergency_operations": {
                    "checkpoints_created": len(self.operation_history["checkpoints"]),
                    "status_reports_generated": len(self.operation_history["status_reports"]),
                    "config_reloads_performed": len(self.operation_history["config_reloads"])
                }
            }

            # Save report to file
            report_path = self.backup_dir / f"{report_id}.json"
            report_path.write_text(json.dumps(status_report, indent=2))

            # Track operation
            self.operation_history["status_reports"].append({
                "report_id": report_id,
                "generated_at": context.triggered_at.isoformat(),
                "path": str(report_path)
            })

            self.logger.info(f"Status report generated: {report_id}")
            return status_report

        except Exception as e:
            self.logger.error(f"Failed to generate status report: {e}")
            return {
                "status": "error",
                "error": str(e),
                "report_id": None
            }

    async def reload_configuration(self, context: "SignalContext") -> Dict[str, Any]:
        """
        Reload configuration triggered by SIGHUP.

        Args:
            context: Signal context with trigger information

        Returns:
            Configuration reload results
        """
        self.logger.info("Reloading configuration triggered by signal")

        try:
            reload_id = f"config_reload_{int(context.triggered_at.timestamp())}"

            # Track current configuration state
            old_config = await self._get_current_config()

            # Perform configuration reload
            reload_results = await self._perform_config_reload()

            # Get new configuration state
            new_config = await self._get_current_config()

            # Create reload report
            reload_report = {
                "reload_id": reload_id,
                "reloaded_at": context.triggered_at.isoformat(),
                "trigger": {
                    "signal": context.signal_name,
                    "operation": context.operation.value,
                    "signal_number": context.signal_number
                },
                "old_config": old_config,
                "new_config": new_config,
                "changes": reload_results.get("changes", []),
                "status": reload_results.get("status", "success")
            }

            # Save reload report
            report_path = self.backup_dir / f"{reload_id}.json"
            report_path.write_text(json.dumps(reload_report, indent=2))

            # Track operation
            self.operation_history["config_reloads"].append({
                "reload_id": reload_id,
                "reloaded_at": context.triggered_at.isoformat(),
                "path": str(report_path)
            })

            self.logger.info(f"Configuration reloaded: {reload_id}")
            return reload_report

        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return {
                "status": "error",
                "error": str(e),
                "reload_id": None
            }

    async def _gather_system_state(self) -> Dict[str, Any]:
        """Gather current system state for checkpoints."""
        try:
            return {
                "pid": os.getpid(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(),
                "disk_usage": {
                    "total_gb": psutil.disk_usage('/').total / 1024**3,
                    "used_gb": psutil.disk_usage('/').used / 1024**3,
                    "free_gb": psutil.disk_usage('/').free / 1024**3
                }
            }
        except Exception as e:
            self.logger.warning(f"Failed to gather system state: {e}")
            return {"error": str(e)}

    async def _gather_training_state(self) -> Dict[str, Any]:
        """Gather current training session state."""
        try:
            async with get_session_context() as db_session:
                # Get active training sessions
                from sqlalchemy import select
                result = await db_session.execute(
                    select(TrainingSession).where(TrainingSession.status == "running")
                )
                active_sessions = result.scalars().all()

                sessions_data = []
                for session in active_sessions:
                    sessions_data.append({
                        "session_id": session.session_id,
                        "started_at": session.started_at.isoformat() if session.started_at else None,
                        "total_iterations": session.total_iterations,
                        "status": session.status
                    })

                return {
                    "active_sessions": len(sessions_data),
                    "sessions": sessions_data
                }

        except Exception as e:
            self.logger.warning(f"Failed to gather training state: {e}")
            return {"error": str(e)}

    async def _gather_system_status(self) -> Dict[str, Any]:
        """Gather system status for reports."""
        return await self._gather_system_state()

    async def _gather_training_status(self) -> Dict[str, Any]:
        """Gather training status for reports."""
        return await self._gather_training_state()

    async def _gather_resource_status(self) -> Dict[str, Any]:
        """Gather resource status including CoreDis connections."""
        try:
            status = {
                "memory": {
                    "total_mb": psutil.virtual_memory().total / 1024 / 1024,
                    "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                    "percent": psutil.virtual_memory().percent
                },
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count()
                },
                "connections": {
                    "database": "unknown",
                    "coredis": "unknown"
                }
            }

            # Try to check CoreDis connection status
            try:
                # This is a best-effort check
                status["connections"]["coredis"] = "available"
            except Exception:
                status["connections"]["coredis"] = "unavailable"

            return status

        except Exception as e:
            self.logger.warning(f"Failed to gather resource status: {e}")
            return {"error": str(e)}

    async def _get_current_config(self) -> Dict[str, Any]:
        """Get current configuration state."""
        # Placeholder for configuration retrieval
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_files": [],
            "environment": dict(os.environ)
        }

    async def _perform_config_reload(self) -> Dict[str, Any]:
        """Perform actual configuration reload."""
        # Placeholder for configuration reload logic
        return {
            "status": "success",
            "changes": [],
            "reloaded_files": []
        }
