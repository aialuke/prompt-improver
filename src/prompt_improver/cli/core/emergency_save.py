"""Emergency Save Manager for Atomic Save Operations
Implements emergency save procedures with atomic operations, validation, and integration with existing systems.
"""

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_improver.cli.core.progress_preservation import ProgressPreservationManager
from prompt_improver.database import get_session_context
from prompt_improver.database.models import TrainingSession


@dataclass
class EmergencySaveContext:
    """Context information for emergency save operations."""

    save_id: str
    triggered_at: datetime
    trigger_type: str
    trigger_details: dict[str, Any]
    priority: int = 1
    atomic: bool = True
    validate: bool = True


@dataclass
class EmergencySaveResult:
    """Result of emergency save operation."""

    save_id: str
    status: str
    saved_components: list[str]
    failed_components: list[str]
    total_size_bytes: int
    duration_seconds: float
    validation_results: dict[str, Any]
    error_message: str | None = None


class EmergencySaveManager:
    """Manages emergency save operations with atomic guarantees and validation.

    Features:
    - Atomic save operations with rollback capability
    - Integration with existing ProgressPreservationManager
    - CoreDis-aware database connection handling
    - Multi-level validation and verification
    - Emergency save triggers and monitoring
    - Comprehensive error handling and recovery
    """

    def __init__(self, backup_dir: Path | None = None):
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("./emergency_saves")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.progress_manager = ProgressPreservationManager(backup_dir=self.backup_dir)
        self.active_saves: dict[str, EmergencySaveContext] = {}
        self.save_history: list[EmergencySaveResult] = []
        self.temp_dir = self.backup_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.save_lock = asyncio.Lock()
        self.validation_lock = asyncio.Lock()

    async def perform_emergency_save(
        self, context: EmergencySaveContext, components: list[str] | None = None
    ) -> EmergencySaveResult:
        """Perform atomic emergency save operation.

        Args:
            context: Emergency save context
            components: List of components to save (None = all)

        Returns:
            Emergency save result
        """
        start_time = time.time()
        self.logger.info(f"Starting emergency save: {context.save_id}")
        if components is None:
            components = [
                "training_sessions",
                "database_state",
                "system_state",
                "configuration",
                "progress_snapshots",
            ]
        async with self.save_lock:
            try:
                self.active_saves[context.save_id] = context
                temp_save_dir = self.temp_dir / context.save_id
                temp_save_dir.mkdir(exist_ok=True)
                saved_components = []
                failed_components = []
                total_size = 0
                for component in components:
                    try:
                        component_result = await self._save_component_atomic(
                            component, temp_save_dir, context
                        )
                        saved_components.append(component)
                        total_size += component_result.get("size_bytes", 0)
                    except Exception as e:
                        self.logger.error(f"Failed to save component {component}: {e}")
                        failed_components.append(component)
                        if context.atomic and failed_components:
                            await self._rollback_save(temp_save_dir, context.save_id)
                            raise Exception(
                                f"Atomic save failed on component {component}: {e}"
                            )
                validation_results = {}
                if context.validate:
                    validation_results = await self._validate_emergency_save(
                        temp_save_dir, saved_components, context
                    )
                final_save_dir = self.backup_dir / context.save_id
                if temp_save_dir.exists():
                    if final_save_dir.exists():
                        shutil.rmtree(final_save_dir)
                    shutil.move(str(temp_save_dir), str(final_save_dir))
                duration = time.time() - start_time
                result = EmergencySaveResult(
                    save_id=context.save_id,
                    status="success" if not failed_components else "partial",
                    saved_components=saved_components,
                    failed_components=failed_components,
                    total_size_bytes=total_size,
                    duration_seconds=duration,
                    validation_results=validation_results,
                )
                self.save_history.append(result)
                self.logger.info(
                    f"Emergency save completed: {context.save_id} ({duration:.1f}s)"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                error_result = EmergencySaveResult(
                    save_id=context.save_id,
                    status="error",
                    saved_components=[],
                    failed_components=components,
                    total_size_bytes=0,
                    duration_seconds=duration,
                    validation_results={},
                    error_message=str(e),
                )
                self.save_history.append(error_result)
                self.logger.error(f"Emergency save failed: {context.save_id}: {e}")
                return error_result
            finally:
                self.active_saves.pop(context.save_id, None)

    async def _save_component_atomic(
        self, component: str, temp_dir: Path, context: EmergencySaveContext
    ) -> dict[str, Any]:
        """Save a single component atomically."""
        component_file = temp_dir / f"{component}.json"
        if component == "training_sessions":
            data = await self._gather_training_sessions()
        elif component == "database_state":
            data = await self._gather_database_state()
        elif component == "system_state":
            data = await self._gather_system_state()
        elif component == "configuration":
            data = await self._gather_configuration()
        elif component == "progress_snapshots":
            data = await self._gather_progress_snapshots()
        else:
            raise ValueError(f"Unknown component: {component}")
        component_data = {
            "component": component,
            "saved_at": context.triggered_at.isoformat(),
            "save_id": context.save_id,
            "trigger": context.trigger_type,
            "data": data,
        }
        temp_file = component_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(component_data, indent=2))
        temp_file.rename(component_file)
        size_bytes = component_file.stat().st_size
        self.logger.debug(f"Saved component {component}: {size_bytes} bytes")
        return {"size_bytes": size_bytes, "path": str(component_file)}

    async def _validate_emergency_save(
        self, save_dir: Path, components: list[str], context: EmergencySaveContext
    ) -> dict[str, Any]:
        """Validate emergency save data."""
        async with self.validation_lock:
            validation_results = {
                "validation_time": datetime.now(UTC).isoformat(),
                "save_id": context.save_id,
                "trigger_type": context.trigger_type,
                "components_validated": [],
                "validation_errors": [],
                "integrity_checks": {},
            }
            for component in components:
                component_file = save_dir / f"{component}.json"
                try:
                    if not component_file.exists():
                        validation_results["validation_errors"].append(
                            f"Component file missing: {component}"
                        )
                        continue
                    data = json.loads(component_file.read_text())
                    required_fields = ["component", "saved_at", "save_id", "data"]
                    for field in required_fields:
                        if field not in data:
                            validation_results["validation_errors"].append(
                                f"Missing required field '{field}' in {component}"
                            )
                    if component == "training_sessions":
                        await self._validate_training_sessions(
                            data["data"], validation_results
                        )
                    elif component == "database_state":
                        await self._validate_database_state(
                            data["data"], validation_results
                        )
                    validation_results["components_validated"].append(component)
                except Exception as e:
                    validation_results["validation_errors"].append(
                        f"Validation error for {component}: {e!s}"
                    )
            validation_results["integrity_checks"]["total_components"] = len(components)
            validation_results["integrity_checks"]["validated_components"] = len(
                validation_results["components_validated"]
            )
            validation_results["integrity_checks"]["error_count"] = len(
                validation_results["validation_errors"]
            )
            validation_results["integrity_checks"]["success_rate"] = (
                len(validation_results["components_validated"]) / len(components)
                if components
                else 0
            )
            return validation_results

    async def _rollback_save(self, temp_dir: Path, save_id: str) -> None:
        """Rollback failed atomic save operation."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            self.logger.info(f"Rolled back failed save: {save_id}")
        except Exception as e:
            self.logger.error(f"Failed to rollback save {save_id}: {e}")

    async def _gather_training_sessions(self) -> dict[str, Any]:
        """Gather training session data for emergency save."""
        try:
            async with get_session_context() as db_session:
                from sqlalchemy import select

                result = await db_session.execute(select(TrainingSession))
                sessions = result.scalars().all()
                sessions_data = []
                for session in sessions:
                    session_dict = {
                        "session_id": session.session_id,
                        "status": session.status,
                        "started_at": session.started_at.isoformat()
                        if session.started_at
                        else None,
                        "completed_at": session.completed_at.isoformat()
                        if session.completed_at
                        else None,
                        "max_iterations": session.max_iterations,
                        "current_iteration": session.current_iteration,
                        "continuous_mode": session.continuous_mode,
                        "improvement_threshold": session.improvement_threshold,
                    }
                    sessions_data.append(session_dict)
                return {
                    "total_sessions": len(sessions_data),
                    "sessions": sessions_data,
                    "gathered_at": datetime.now(UTC).isoformat(),
                }
        except Exception as e:
            self.logger.warning(f"Failed to gather training sessions: {e}")
            return {"error": str(e), "sessions": []}

    async def _gather_database_state(self) -> dict[str, Any]:
        """Gather database state for emergency save."""
        return {
            "connection_status": "unknown",
            "gathered_at": datetime.now(UTC).isoformat(),
        }

    async def _gather_system_state(self) -> dict[str, Any]:
        """Gather system state for emergency save."""
        try:
            import psutil

            return {
                "pid": os.getpid(),
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(),
                "gathered_at": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}

    async def _gather_configuration(self) -> dict[str, Any]:
        """Gather configuration for emergency save."""
        return {
            "environment": dict(os.environ),
            "gathered_at": datetime.now(UTC).isoformat(),
        }

    async def _gather_progress_snapshots(self) -> dict[str, Any]:
        """Gather progress snapshots for emergency save."""
        try:
            snapshots = []
            for session_id, snapshot in self.progress_manager.active_sessions.items():
                snapshot_dict = asdict(snapshot)
                snapshot_dict["session_id"] = session_id
                snapshots.append(snapshot_dict)
            return {
                "active_snapshots": len(snapshots),
                "snapshots": snapshots,
                "gathered_at": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {"error": str(e), "snapshots": []}

    async def _validate_training_sessions(
        self, data: dict[str, Any], results: dict[str, Any]
    ) -> None:
        """Validate training sessions data."""
        if "sessions" not in data:
            results["validation_errors"].append(
                "Training sessions data missing 'sessions' field"
            )
            return
        for i, session in enumerate(data["sessions"]):
            if "session_id" not in session:
                results["validation_errors"].append(f"Session {i} missing session_id")

    async def _validate_database_state(
        self, data: dict[str, Any], results: dict[str, Any]
    ) -> None:
        """Validate database state data."""
        if "sessions" not in data:
            results["validation_errors"].append("Database state missing 'sessions' key")
            return
        if not isinstance(data["sessions"], list):
            results["validation_errors"].append(
                "Database state 'sessions' is not a list"
            )
            return
        required_fields = ["session_id", "status"]
        for i, session in enumerate(data["sessions"]):
            for field in required_fields:
                if field not in session:
                    results["validation_errors"].append(
                        f"Session {i} missing required field '{field}'"
                    )
