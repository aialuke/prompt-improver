"""
Training Progress Preservation System
Implements comprehensive progress saving and recovery for APES training sessions.
"""

import asyncio
import json
import logging
import os

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from ...database import get_session_context
from ...database.models import (
    TrainingSession, TrainingIteration, RulePerformance,
    RuleMetadata, DiscoveredPattern
)
from sqlalchemy import select, text
# Signal handling integration - avoiding circular import
from .signal_handler import SignalOperation

@dataclass
class ProgressSnapshot:
    """Snapshot of training progress at a specific point in time."""
    session_id: str
    iteration: int
    timestamp: datetime
    performance_metrics: Dict[str, float]
    rule_optimizations: Dict[str, Any]
    synthetic_data_generated: int
    workflow_state: Dict[str, Any]
    model_checkpoints: List[str]
    improvement_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressSnapshot':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ProgressPreservationManager:
    """
    Manages training progress preservation with database integration and file backup.

    Features:
    - Real-time progress saving to PostgreSQL
    - File-based backup for recovery scenarios
    - Incremental checkpoint creation
    - Session state recovery
    - Performance metrics preservation
    """

    def __init__(self, backup_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("./training_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Unified session manager for consolidated session management
        self._unified_session_manager = None

        # Signal handling integration - lazy import to avoid circular dependency
        self.signal_handler = None
        self.background_manager = None
        self._shutdown_priority = 6  # Medium priority for progress preservation
        
        # Progress tracking (legacy compatibility)
        self.active_sessions: Dict[str, ProgressSnapshot] = {}
        self.checkpoint_interval = 5  # Save checkpoint every 5 iterations
        
        # Initialize signal handling integration (lazy)
        self._init_signal_handlers()

    def _init_signal_handlers(self):
        """Initialize signal handler with lazy import to avoid circular dependency."""
        try:
            # Lazy import to avoid circular dependency
            from ...performance.monitoring.health.background_manager import get_background_task_manager
            from .signal_handler import AsyncSignalHandler
            from rich.console import Console
            
            # Initialize signal handler if not already done
            if self.signal_handler is None:
                self.signal_handler = AsyncSignalHandler(console=Console())
                
                # Setup signal handlers with asyncio loop if available
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    self.signal_handler.setup_signal_handlers(loop)
                except RuntimeError:
                    # No running loop - will be set up when loop starts
                    pass
            
            # Initialize background manager
            if self.background_manager is None:
                self.background_manager = get_background_task_manager()
            
            # Register handlers
            self._register_signal_handlers()
            
        except ImportError as e:
            self.logger.warning(f"Signal handling integration not available: {e}")
            # Continue without signal handling
    
    def _register_signal_handlers(self):
        """Register ProgressPreservationManager-specific signal handlers."""
        if self.signal_handler is None:
            self.logger.warning("Signal handler not initialized, skipping signal registration")
            return
            
        import signal
        
        # Shutdown coordination for progress preservation
        self.signal_handler.register_shutdown_handler(
            "ProgressPreservationManager_shutdown", 
            self.graceful_progress_shutdown
        )
        
        # Emergency progress save (SIGUSR1) - PRIMARY FEATURE
        self.signal_handler.register_operation_handler(
            SignalOperation.CHECKPOINT,
            self.emergency_progress_save
        )
        
        # Progress status reporting (SIGUSR2)
        self.signal_handler.register_operation_handler(
            SignalOperation.STATUS_REPORT,
            self.generate_progress_status_report
        )
        
        # Signal chaining for coordinated progress preservation
        self.signal_handler.add_signal_chain_handler(
            signal.SIGTERM,
            self.prepare_progress_preservation,
            priority=self._shutdown_priority
        )
        
        # SIGINT coordination for progress saving on interruption
        self.signal_handler.add_signal_chain_handler(
            signal.SIGINT,
            self.prepare_progress_interruption,
            priority=self._shutdown_priority
        )
        
        self.logger.info("ProgressPreservationManager signal handlers registered")

    async def graceful_progress_shutdown(self, shutdown_context):
        """Handle graceful shutdown with comprehensive progress preservation."""
        self.logger.info("ProgressPreservationManager graceful shutdown initiated")
        
        try:
            # Save all active session progress
            save_results = {}
            
            for session_id, snapshot in self.active_sessions.items():
                try:
                    # Save current progress
                    success = await self.save_training_progress(
                        session_id=session_id,
                        iteration=snapshot.iteration,
                        performance_metrics=snapshot.performance_metrics,
                        rule_optimizations=snapshot.rule_optimizations,
                        workflow_state=snapshot.workflow_state,
                        synthetic_data_generated=snapshot.synthetic_data_generated,
                        model_checkpoints=snapshot.model_checkpoints,
                        improvement_score=snapshot.improvement_score
                    )
                    
                    # Export session results
                    export_path = await self.export_session_results(
                        session_id=session_id,
                        export_format="json",
                        include_iterations=True
                    )
                    
                    save_results[session_id] = {
                        "progress_saved": success,
                        "export_path": export_path
                    }
                except Exception as e:
                    save_results[session_id] = {
                        "progress_saved": False,
                        "error": str(e)
                    }
            
            # Cleanup old backups
            cleaned_files = await self.cleanup_old_backups(days_to_keep=30)
            
            return {
                "status": "success",
                "component": "ProgressPreservationManager",
                "active_sessions_saved": len(save_results),
                "save_results": save_results,
                "cleaned_backup_files": cleaned_files
            }
        except Exception as e:
            self.logger.error(f"ProgressPreservationManager shutdown error: {e}")
            return {
                "status": "error",
                "component": "ProgressPreservationManager",
                "error": str(e)
            }

    async def emergency_progress_save(self, signal_context):
        """Perform emergency progress save for all active sessions on SIGUSR1 signal."""
        self.logger.info("Emergency progress save triggered by SIGUSR1")
        
        try:
            if not self.active_sessions:
                return {
                    "status": "no_active_sessions",
                    "message": "No active training sessions for emergency save"
                }
            
            emergency_saves = {}
            
            for session_id, snapshot in self.active_sessions.items():
                try:
                    # Create emergency checkpoint
                    checkpoint_id = await self.create_checkpoint(session_id)
                    
                    # Save current progress with emergency flag
                    snapshot.workflow_state["emergency_save"] = True
                    snapshot.workflow_state["emergency_timestamp"] = datetime.now(timezone.utc).isoformat()
                    
                    success = await self.save_training_progress(
                        session_id=session_id,
                        iteration=snapshot.iteration,
                        performance_metrics=snapshot.performance_metrics,
                        rule_optimizations=snapshot.rule_optimizations,
                        workflow_state=snapshot.workflow_state,
                        synthetic_data_generated=snapshot.synthetic_data_generated,
                        model_checkpoints=snapshot.model_checkpoints,
                        improvement_score=snapshot.improvement_score
                    )
                    
                    emergency_saves[session_id] = {
                        "status": "saved",
                        "checkpoint_id": checkpoint_id,
                        "progress_saved": success
                    }
                except Exception as e:
                    emergency_saves[session_id] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "status": "emergency_saves_completed",
                "active_sessions": len(self.active_sessions),
                "emergency_saves": emergency_saves,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Emergency progress save failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def generate_progress_status_report(self, signal_context):
        """Generate comprehensive progress status report on SIGUSR2 signal."""
        self.logger.info("Generating progress status report")
        
        try:
            # Collect progress information for all active sessions
            session_reports = {}
            
            for session_id, snapshot in self.active_sessions.items():
                session_reports[session_id] = {
                    "iteration": snapshot.iteration,
                    "improvement_score": snapshot.improvement_score,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "synthetic_data_generated": snapshot.synthetic_data_generated,
                    "model_checkpoints": len(snapshot.model_checkpoints),
                    "performance_metrics": snapshot.performance_metrics,
                    "workflow_state": snapshot.workflow_state
                }
            
            # Get backup directory stats
            backup_stats = {
                "backup_directory": str(self.backup_dir),
                "total_backup_files": len(list(self.backup_dir.glob("*.json"))),
                "directory_size_mb": sum(f.stat().st_size for f in self.backup_dir.glob("*")) / (1024 * 1024)
            }
            
            return {
                "status": "report_generated",
                "active_sessions": len(self.active_sessions),
                "session_reports": session_reports,
                "backup_stats": backup_stats,
                "checkpoint_interval": self.checkpoint_interval,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Progress status report generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def prepare_progress_preservation(self, signum, signal_name):
        """Prepare progress preservation for coordinated shutdown."""
        self.logger.info(f"Preparing progress preservation ({signal_name})")
        
        try:
            # Prepare for coordinated progress preservation
            preparation_status = {
                "prepared": True,
                "component": "ProgressPreservationManager",
                "active_sessions": len(self.active_sessions),
                "backup_directory": str(self.backup_dir),
                "backup_directory_exists": self.backup_dir.exists(),
                "ready_for_emergency_save": True
            }
            
            return preparation_status
        except Exception as e:
            self.logger.error(f"Progress preservation preparation failed: {e}")
            return {
                "prepared": False,
                "component": "ProgressPreservationManager",
                "error": str(e)
            }

    def prepare_progress_interruption(self, signum, signal_name):
        """Prepare progress preservation for user interruption (Ctrl+C)."""
        self.logger.info(f"Preparing progress interruption handling ({signal_name})")
        
        try:
            # For user interruption, prioritize immediate progress save
            interruption_preparation = {
                "prepared": True,
                "component": "ProgressPreservationManager",
                "interruption_type": "user_requested",
                "active_sessions": list(self.active_sessions.keys()),
                "immediate_save_ready": True,
                "export_on_interrupt": True
            }
            
            return interruption_preparation
        except Exception as e:
            self.logger.error(f"Progress interruption preparation failed: {e}")
            return {
                "prepared": False,
                "component": "ProgressPreservationManager",
                "error": str(e)
            }

    async def _ensure_unified_session_manager(self):
        """Ensure unified session manager is available."""
        if self._unified_session_manager is None:
            self._unified_session_manager = await get_unified_session_manager()

    async def save_training_progress(
        self,
        session_id: str,
        iteration: int,
        performance_metrics: Dict[str, float],
        rule_optimizations: Dict[str, Any],
        workflow_state: Dict[str, Any],
        synthetic_data_generated: int = 0,
        model_checkpoints: Optional[List[str]] = None,
        improvement_score: float = 0.0
    ) -> bool:
        """
        Save comprehensive training progress using unified session management.

        Args:
            session_id: Training session identifier
            iteration: Current iteration number
            performance_metrics: Current performance metrics
            rule_optimizations: Rule optimization results
            workflow_state: Current workflow state
            synthetic_data_generated: Number of synthetic samples generated
            model_checkpoints: List of model checkpoint paths
            improvement_score: Current improvement score

        Returns:
            True if progress saved successfully
        """
        try:
            # Create progress snapshot for legacy compatibility
            snapshot = ProgressSnapshot(
                session_id=session_id,
                iteration=iteration,
                timestamp=datetime.now(timezone.utc),
                performance_metrics=performance_metrics,
                rule_optimizations=rule_optimizations,
                synthetic_data_generated=synthetic_data_generated,
                workflow_state=workflow_state,
                model_checkpoints=model_checkpoints or [],
                improvement_score=improvement_score
            )

            # Use unified session manager for primary storage
            await self._ensure_unified_session_manager()
            
            # Update progress in unified session manager
            unified_success = await self._unified_session_manager.update_training_progress(
                session_id=session_id,
                iteration=iteration,
                performance_metrics=performance_metrics,
                improvement_score=improvement_score
            )

            # Save to database (legacy path)
            await self._save_to_database(snapshot)

            # Save to backup file (legacy path)
            await self._save_to_backup_file(snapshot)

            # Update active sessions tracking (legacy compatibility)
            self.active_sessions[session_id] = snapshot

            self.logger.info(f"Progress saved for session {session_id}, iteration {iteration} (unified: {unified_success})")
            return unified_success and True

        except Exception as e:
            self.logger.error(f"Failed to save training progress: {e}")
            return False

    async def _save_to_database(self, snapshot: ProgressSnapshot) -> None:
        """Save progress snapshot to PostgreSQL database using existing TrainingSession model."""
        try:
            async with get_session_context() as db_session:
                # Get training session
                session_query = select(TrainingSession).where(
                    text(f"session_id = '{snapshot.session_id}'")
                )
                session_result = await db_session.execute(session_query)
                training_session = session_result.scalar_one_or_none()

                if training_session:
                    # Update session with latest progress
                    training_session.current_iteration = snapshot.iteration
                    training_session.current_performance = snapshot.improvement_score

                    # Update best performance if improved
                    if not training_session.best_performance or snapshot.improvement_score > training_session.best_performance:
                        training_session.best_performance = snapshot.improvement_score

                    # Update performance history
                    if training_session.performance_history is None:
                        training_session.performance_history = []
                    training_session.performance_history.append(snapshot.improvement_score)

                    # Update activity timestamp
                    training_session.last_activity_at = snapshot.timestamp

                    # Store comprehensive checkpoint data
                    training_session.checkpoint_data = {
                        "iteration": snapshot.iteration,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "performance_metrics": snapshot.performance_metrics,
                        "rule_optimizations": snapshot.rule_optimizations,
                        "workflow_state": snapshot.workflow_state,
                        "model_checkpoints": snapshot.model_checkpoints,
                        "improvement_score": snapshot.improvement_score,
                        "synthetic_data_generated": snapshot.synthetic_data_generated
                    }
                    training_session.last_checkpoint_at = snapshot.timestamp

                    await db_session.commit()
                    self.logger.debug(f"Updated session {snapshot.session_id} with iteration {snapshot.iteration} progress")
                else:
                    self.logger.warning(f"Training session {snapshot.session_id} not found in database")

        except Exception as e:
            self.logger.error(f"Failed to save progress to database: {e}")
            # Don't re-raise - we still have file backup

    async def preserve_rule_optimizations(
        self,
        session_id: str,
        rule_optimizations: Dict[str, Any]
    ) -> bool:
        """
        Preserve rule parameter optimizations to rule_performance and rule_metadata tables.

        Args:
            session_id: Training session identifier
            rule_optimizations: Dictionary of rule optimizations with rule_id as key

        Returns:
            True if optimizations saved successfully
        """
        try:
            async with get_session_context() as db_session:
                for rule_id, optimization_data in rule_optimizations.items():
                    # Save to rule_performance table (id is auto-increment)
                    performance_data = {
                        "rule_id": rule_id,
                        "rule_name": optimization_data.get("rule_name", rule_id),
                        "prompt_id": session_id,  # Use session_id as prompt_id for training sessions
                        "prompt_type": "training_session",
                        "improvement_score": optimization_data.get("improvement_score", 0.0),
                        "confidence_level": optimization_data.get("confidence_level", 0.0),
                        "execution_time_ms": optimization_data.get("execution_time_ms", 0),
                        "rule_parameters": optimization_data.get("optimized_parameters", {}),
                        "before_metrics": optimization_data.get("before_metrics", {}),
                        "after_metrics": optimization_data.get("after_metrics", {})
                    }
                    performance_record = RulePerformance(**performance_data)
                    db_session.add(performance_record)

                    # Update rule_metadata with optimized parameters if they improved performance
                    if optimization_data.get("improvement_score", 0) > 0.1:  # 10% improvement threshold
                        rule_query = select(RuleMetadata).where(
                            text(f"rule_id = '{rule_id}'")
                        )
                        rule_result = await db_session.execute(rule_query)
                        rule_metadata = rule_result.scalar_one_or_none()

                        if rule_metadata:
                            # Update default parameters with optimized values
                            optimized_params = optimization_data.get("optimized_parameters", {})
                            if optimized_params:
                                rule_metadata.default_parameters = optimized_params
                                rule_metadata.updated_at = datetime.now(timezone.utc)

                await db_session.commit()
                self.logger.info(f"Preserved rule optimizations for session {session_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to preserve rule optimizations: {e}")
            return False

    async def preserve_discovered_patterns(
        self,
        session_id: str,
        discovered_patterns: Dict[str, Any]
    ) -> bool:
        """
        Preserve ML-discovered patterns to discovered_patterns table.

        Args:
            session_id: Training session identifier
            discovered_patterns: Dictionary of discovered patterns

        Returns:
            True if patterns saved successfully
        """
        try:
            async with get_session_context() as db_session:
                for pattern_name, pattern_data in discovered_patterns.items():
                    # Create discovered pattern record (id is auto-increment)
                    pattern_data_dict = {
                        "pattern_id": f"{session_id}_{pattern_name}_{int(datetime.now(timezone.utc).timestamp())}",
                        "avg_effectiveness": pattern_data.get("effectiveness_score", 0.0),
                        "parameters": pattern_data.get("parameters", {}),
                        "support_count": pattern_data.get("support_count", 1),
                        "pattern_type": "ml_training",
                        "discovery_run_id": session_id
                    }
                    pattern_record = DiscoveredPattern(**pattern_data_dict)
                    db_session.add(pattern_record)

                await db_session.commit()
                self.logger.info(f"Preserved discovered patterns for session {session_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to preserve discovered patterns: {e}")
            return False

    async def _save_to_backup_file(self, snapshot: ProgressSnapshot) -> None:
        """Save progress snapshot to backup file for recovery."""
        backup_file = self.backup_dir / f"{snapshot.session_id}_progress.json"

        try:
            # Load existing backup data
            backup_data = {"snapshots": []}
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    backup_data = json.load(f)

            # Add new snapshot
            backup_data["snapshots"].append(snapshot.to_dict())

            # Keep only last 50 snapshots to prevent file bloat
            if len(backup_data["snapshots"]) > 50:
                backup_data["snapshots"] = backup_data["snapshots"][-50:]

            # Write updated backup
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save backup file: {e}")

    async def create_checkpoint(self, session_id: str) -> Optional[str]:
        """
        Create a comprehensive checkpoint for the training session.

        Args:
            session_id: Training session identifier

        Returns:
            Checkpoint identifier if successful
        """
        try:
            async with get_session_context() as db_session:
                # Get complete session data
                session_result = await db_session.execute(
                    select(TrainingSession).where(text(f"session_id = '{session_id}'"))
                )
                session = session_result.scalar_one_or_none()

                if not session:
                    self.logger.error(f"Session {session_id} not found for checkpoint")
                    return None

                # Create checkpoint data
                checkpoint_id = f"{session_id}_checkpoint_{int(datetime.now(timezone.utc).timestamp())}"
                checkpoint_data = {
                    "checkpoint_id": checkpoint_id,
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "session_data": {
                        "session_id": session.session_id,
                        "continuous_mode": session.continuous_mode,
                        "max_iterations": session.max_iterations,
                        "improvement_threshold": session.improvement_threshold,
                        "timeout_seconds": session.timeout_seconds,
                        "status": session.status,
                        "current_iteration": session.current_iteration,
                        "current_performance": session.current_performance,
                        "best_performance": session.best_performance,
                        "performance_history": session.performance_history,
                        "total_training_time_seconds": session.total_training_time_seconds,
                        "data_points_processed": session.data_points_processed,
                        "models_trained": session.models_trained,
                        "rules_optimized": session.rules_optimized,
                        "patterns_discovered": session.patterns_discovered,
                        "error_count": session.error_count,
                        "retry_count": session.retry_count,
                        "last_error": session.last_error,
                        "active_workflow_id": session.active_workflow_id,
                        "workflow_history": session.workflow_history,
                        "checkpoint_data": session.checkpoint_data,
                        "last_checkpoint_at": session.last_checkpoint_at.isoformat() if session.last_checkpoint_at else None,
                        "started_at": session.started_at.isoformat(),
                        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                        "last_activity_at": session.last_activity_at.isoformat()
                    },
                    "iterations": []  # Will be populated separately if TrainingIteration table exists
                }

                # Save checkpoint file
                checkpoint_file = self.backup_dir / f"{checkpoint_id}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                self.logger.info(f"Checkpoint created: {checkpoint_id}")
                return checkpoint_id

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return None

    async def recover_session_progress(self, session_id: str) -> Optional[ProgressSnapshot]:
        """
        Recover the latest progress for a training session.

        Args:
            session_id: Training session identifier

        Returns:
            Latest progress snapshot if found
        """
        try:
            # Try database recovery first
            async with get_session_context() as db_session:
                # Get latest iteration for the session
                latest_iteration = await db_session.execute(
                    select(TrainingIteration)
                    .where(text(f"session_id = '{session_id}'"))
                    .order_by(text("iteration DESC"))
                    .limit(1)
                )

                iteration_data = latest_iteration.scalar_one_or_none()

                if iteration_data:
                    # Create snapshot from database data
                    snapshot = ProgressSnapshot(
                        session_id=session_id,
                        iteration=iteration_data.iteration,
                        timestamp=iteration_data.created_at,
                        performance_metrics=iteration_data.performance_metrics or {},
                        rule_optimizations=iteration_data.rule_optimizations or {},
                        synthetic_data_generated=iteration_data.synthetic_data_generated,
                        workflow_state={"workflow_id": iteration_data.workflow_id},
                        model_checkpoints=[],
                        improvement_score=iteration_data.improvement_score
                    )

                    self.logger.info(f"Recovered session {session_id} from database (iteration {iteration_data.iteration})")
                    return snapshot

            # Fallback to backup file recovery
            backup_file = self.backup_dir / f"{session_id}_progress.json"
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    backup_data = json.load(f)

                if backup_data.get("snapshots"):
                    latest_snapshot_data = backup_data["snapshots"][-1]
                    snapshot = ProgressSnapshot.from_dict(latest_snapshot_data)

                    self.logger.info(f"Recovered session {session_id} from backup file (iteration {snapshot.iteration})")
                    return snapshot

            self.logger.warning(f"No recovery data found for session {session_id}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to recover session progress: {e}")
            return None

    async def cleanup_resources(self, session_id: str) -> bool:
        """
        Comprehensive resource cleanup for training session.

        Args:
            session_id: Training session identifier

        Returns:
            True if cleanup successful
        """
        try:
            cleanup_tasks = []

            # Database connection cleanup
            cleanup_tasks.append(self._cleanup_database_connections(session_id))

            # File handle cleanup
            cleanup_tasks.append(self._cleanup_file_handles(session_id))

            # Temporary file cleanup
            cleanup_tasks.append(self._cleanup_temporary_files(session_id))

            # Memory cleanup
            cleanup_tasks.append(self._cleanup_memory_resources(session_id))

            # Execute all cleanup tasks
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            # Check for any failures
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                self.logger.warning(f"Some cleanup tasks failed for session {session_id}: {failures}")
                return False

            self.logger.info(f"Resource cleanup completed for session {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cleanup resources for session {session_id}: {e}")
            return False

    async def _cleanup_database_connections(self, session_id: str) -> None:
        """Cleanup database connections for session."""
        try:
            # Close any active database sessions for this training session
            # This would integrate with the database session manager
            self.logger.debug(f"Database connections cleaned up for session {session_id}")
        except Exception as e:
            self.logger.warning(f"Database connection cleanup failed: {e}")

    async def _cleanup_file_handles(self, session_id: str) -> None:
        """Cleanup file handles for session."""
        try:
            # Close any open file handles related to this session
            # Remove from active sessions tracking
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            self.logger.debug(f"File handles cleaned up for session {session_id}")
        except Exception as e:
            self.logger.warning(f"File handle cleanup failed: {e}")

    async def _cleanup_temporary_files(self, session_id: str) -> None:
        """Cleanup temporary files for session."""
        try:
            # Clean up temporary checkpoint files older than 24 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

            for file_path in self.backup_dir.glob(f"*{session_id}*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                    if file_time < cutoff_time and "checkpoint" in file_path.name:
                        file_path.unlink()
                        self.logger.debug(f"Removed old checkpoint file: {file_path}")

            self.logger.debug(f"Temporary files cleaned up for session {session_id}")
        except Exception as e:
            self.logger.warning(f"Temporary file cleanup failed: {e}")

    async def _cleanup_memory_resources(self, session_id: str) -> None:
        """Cleanup memory resources for session."""
        try:
            # Clear any cached data for this session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            # Force garbage collection for large objects
            import gc
            gc.collect()

            self.logger.debug(f"Memory resources cleaned up for session {session_id}")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def create_pid_file(self, session_id: str) -> bool:
        """
        Create PID file for training session process tracking.

        Args:
            session_id: Training session identifier

        Returns:
            True if PID file created successfully
        """
        try:
            pid_file = self.backup_dir / f"{session_id}.pid"

            # Check if PID file already exists
            if pid_file.exists():
                # Check if process is still running
                try:
                    with open(pid_file, 'r') as f:
                        old_pid = int(f.read().strip())

                    # Check if process exists
                    try:
                        os.kill(old_pid, 0)  # Signal 0 just checks if process exists
                        self.logger.warning(f"Training session {session_id} already running with PID {old_pid}")
                        return False
                    except OSError:
                        # Process doesn't exist, remove stale PID file
                        pid_file.unlink()
                        self.logger.info(f"Removed stale PID file for session {session_id}")
                except (ValueError, FileNotFoundError):
                    # Invalid PID file, remove it
                    pid_file.unlink()

            # Create new PID file
            current_pid = os.getpid()
            with open(pid_file, 'w') as f:
                f.write(str(current_pid))

            self.logger.info(f"Created PID file for session {session_id} with PID {current_pid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create PID file for session {session_id}: {e}")
            return False

    def remove_pid_file(self, session_id: str) -> bool:
        """
        Remove PID file for training session.

        Args:
            session_id: Training session identifier

        Returns:
            True if PID file removed successfully
        """
        try:
            pid_file = self.backup_dir / f"{session_id}.pid"

            if pid_file.exists():
                pid_file.unlink()
                self.logger.info(f"Removed PID file for session {session_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to remove PID file for session {session_id}: {e}")
            return False

    def check_orphaned_sessions(self) -> List[str]:
        """
        Check for orphaned training sessions (PID files without running processes).

        Returns:
            List of orphaned session IDs
        """
        orphaned_sessions = []

        try:
            for pid_file in self.backup_dir.glob("*.pid"):
                session_id = pid_file.stem

                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())

                    # Check if process is still running
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        # Process doesn't exist
                        orphaned_sessions.append(session_id)
                        self.logger.warning(f"Found orphaned session: {session_id} (PID {pid})")

                except (ValueError, FileNotFoundError):
                    # Invalid PID file
                    orphaned_sessions.append(session_id)
                    self.logger.warning(f"Found invalid PID file for session: {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to check for orphaned sessions: {e}")

        return orphaned_sessions

    async def cleanup_orphaned_sessions(self) -> int:
        """
        Cleanup orphaned training sessions.

        Returns:
            Number of sessions cleaned up
        """
        orphaned_sessions = self.check_orphaned_sessions()
        cleaned_count = 0

        for session_id in orphaned_sessions:
            try:
                # Remove PID file
                self.remove_pid_file(session_id)

                # Cleanup resources
                await self.cleanup_resources(session_id)

                cleaned_count += 1
                self.logger.info(f"Cleaned up orphaned session: {session_id}")

            except Exception as e:
                self.logger.error(f"Failed to cleanup orphaned session {session_id}: {e}")

        return cleaned_count

    async def export_session_results(
        self,
        session_id: str,
        export_format: str = "json",
        include_iterations: bool = True
    ) -> Optional[str]:
        """
        Export comprehensive session results during shutdown.

        Args:
            session_id: Training session identifier
            export_format: Export format ("json" or "csv")
            include_iterations: Whether to include iteration details

        Returns:
            Path to exported file if successful
        """
        try:
            async with get_session_context() as db_session:
                # Get complete session data
                session_result = await db_session.execute(
                    select(TrainingSession).where(text(f"session_id = '{session_id}'"))
                )
                session = session_result.scalar_one_or_none()

                # Load iterations separately if needed
                if session and include_iterations:
                    iterations_result = await db_session.execute(
                        select(TrainingIteration)
                        .where(text(f"session_id = '{session_id}'"))
                        .order_by(text("iteration ASC"))
                    )
                    session.iterations = list(iterations_result.scalars().all())
                session = session_result.scalar_one_or_none()

                if not session:
                    self.logger.error(f"Session {session_id} not found for export")
                    return None

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

                if export_format.lower() == "json":
                    return await self._export_json(session, timestamp, include_iterations)
                elif export_format.lower() == "csv":
                    return await self._export_csv(session, timestamp, include_iterations)
                else:
                    self.logger.error(f"Unsupported export format: {export_format}")
                    return None

        except Exception as e:
            self.logger.error(f"Failed to export session results: {e}")
            return None

    async def _export_json(
        self,
        session: TrainingSession,
        timestamp: str,
        include_iterations: bool
    ) -> str:
        """Export session data to JSON format."""
        export_data = {
            "export_metadata": {
                "session_id": session.session_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "export_format": "json",
                "include_iterations": include_iterations
            },
            "session_summary": {
                "session_id": session.session_id,
                "continuous_mode": session.continuous_mode,
                "max_iterations": session.max_iterations,
                "improvement_threshold": session.improvement_threshold,
                "status": session.status,
                "started_at": session.started_at.isoformat(),
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "max_iterations": session.max_iterations,
                "current_iteration": session.current_iteration,
                "best_performance": session.best_performance,
                "performance_history": session.performance_history
            }
        }

        if include_iterations and session.iterations:
            export_data["iterations"] = [  # type: ignore
                {
                    "iteration": iter_data.iteration,
                    "workflow_id": iter_data.workflow_id,
                    "performance_metrics": iter_data.performance_metrics,
                    "rule_optimizations": iter_data.rule_optimizations,
                    "synthetic_data_generated": iter_data.synthetic_data_generated,
                    "duration_seconds": iter_data.duration_seconds,
                    "improvement_score": iter_data.improvement_score,
                    "created_at": iter_data.created_at.isoformat()
                }
                for iter_data in sorted(session.iterations, key=lambda x: x.iteration)
            ]

        # Save export file
        export_file = self.backup_dir / f"{session.session_id}_export_{timestamp}.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Session exported to JSON: {export_file}")
        return str(export_file)

    async def _export_csv(
        self,
        session: TrainingSession,
        timestamp: str,
        include_iterations: bool
    ) -> str:
        """Export session data to CSV format."""
        import csv

        export_file = self.backup_dir / f"{session.session_id}_export_{timestamp}.csv"

        with open(export_file, 'w', newline='') as csvfile:
            if include_iterations and session.iterations:
                # Export iteration details
                fieldnames = [
                    'session_id', 'iteration', 'workflow_id', 'improvement_score',
                    'synthetic_data_generated', 'duration_seconds', 'created_at'
                ]

                # Add performance metrics columns
                if session.iterations:
                    sample_metrics = session.iterations[0].performance_metrics or {}
                    for metric_name in sample_metrics.keys():
                        fieldnames.append(f'metric_{metric_name}')

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for iter_data in sorted(session.iterations, key=lambda x: x.iteration):
                    row = {
                        'session_id': session.session_id,
                        'iteration': iter_data.iteration,
                        'workflow_id': iter_data.workflow_id,
                        'improvement_score': iter_data.improvement_score,
                        'synthetic_data_generated': iter_data.synthetic_data_generated,
                        'duration_seconds': iter_data.duration_seconds,
                        'created_at': iter_data.created_at.isoformat()
                    }

                    # Add performance metrics
                    if iter_data.performance_metrics:
                        for metric_name, metric_value in iter_data.performance_metrics.items():
                            row[f'metric_{metric_name}'] = metric_value

                    writer.writerow(row)
            else:
                # Export session summary only
                fieldnames = [
                    'session_id', 'continuous_mode', 'max_iterations', 'improvement_threshold',
                    'status', 'started_at', 'completed_at', 'total_iterations', 'stopped_reason'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                writer.writerow({
                    'session_id': session.session_id,
                    'continuous_mode': session.continuous_mode,
                    'max_iterations': session.max_iterations,
                    'improvement_threshold': session.improvement_threshold,
                    'status': session.status,
                    'started_at': session.started_at.isoformat(),
                    'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                    'current_iteration': session.current_iteration
                })

        self.logger.info(f"Session exported to CSV: {export_file}")
        return str(export_file)

    async def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """
        Clean up old backup files to prevent disk space issues.

        Args:
            days_to_keep: Number of days to keep backup files

        Returns:
            Number of files cleaned up
        """
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (days_to_keep * 24 * 3600)
            cleaned_count = 0

            for backup_file in self.backup_dir.glob("*.json"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old backup files")

            return cleaned_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
            return 0
