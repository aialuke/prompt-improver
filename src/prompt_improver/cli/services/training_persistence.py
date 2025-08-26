"""Training Persistence Service - Clean Architecture Implementation.

Implements database operations, session persistence, and recovery functionality.
Extracted from training_system_manager.py (2109 lines) as part of decomposition.
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from prompt_improver.cli.services.training_protocols import TrainingPersistenceProtocol
from prompt_improver.database import ManagerMode, get_database_services


class TrainingPersistence(TrainingPersistenceProtocol):
    """Training persistence service implementing Clean Architecture patterns.

    Responsibilities:
    - Database operations for training sessions
    - Progress persistence and recovery
    - Training data management
    - Session state tracking
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("apes.training_persistence")
        self._unified_session_manager = None

    async def _ensure_database_services(self) -> None:
        """Ensure database services are available for persistence operations."""
        if self._unified_session_manager is None:
            self._unified_session_manager = await get_database_services(
                ManagerMode.MCP_SERVER
            )

    async def create_training_session(self, training_config: dict[str, Any]) -> str:
        """Create training session using unified connection management.

        Args:
            training_config: Training configuration dictionary

        Returns:
            Created training session ID
        """
        try:
            await self._ensure_database_services()

            # Generate unique session ID
            session_id = f"training_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # Store session data in database using unified connection manager
            async with self._unified_session_manager.get_async_session() as db_session:
                # Insert training session record
                query = text("""
                    INSERT INTO training_sessions (session_id, config, status, created_at)
                    VALUES (:session_id, :config, 'initializing', :created_at)
                """)

                await db_session.execute(
                    query,
                    {
                        "session_id": session_id,
                        "config": json.dumps(training_config),
                        "created_at": time.time(),
                    },
                )
                await db_session.commit()

            self.logger.info(f"Created training session: {session_id}")
            return session_id

        except Exception as e:
            self.logger.exception(f"Failed to create training session: {e}")
            raise

    async def update_training_progress(
        self,
        session_id: str,
        iteration: int,
        performance_metrics: dict[str, float],
        improvement_score: float = 0.0,
    ) -> bool:
        """Update training progress using unified connection management.

        Args:
            session_id: Training session ID
            iteration: Current iteration number
            performance_metrics: Performance metrics dictionary
            improvement_score: Improvement score for this iteration

        Returns:
            True if updated successfully, False otherwise
        """
        if not session_id:
            self.logger.warning("No session ID provided for progress update")
            return False

        try:
            await self._ensure_database_services()

            # Update training session progress in database
            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    UPDATE training_sessions
                    SET current_iteration = :iteration,
                        performance_metrics = :metrics,
                        improvement_score = :score,
                        updated_at = :updated_at,
                        status = 'running'
                    WHERE session_id = :session_id
                """)

                result = await db_session.execute(
                    query,
                    {
                        "iteration": iteration,
                        "metrics": json.dumps(performance_metrics),
                        "score": improvement_score,
                        "updated_at": time.time(),
                        "session_id": session_id,
                    },
                )
                await db_session.commit()

                # Check if update was successful
                if result.rowcount == 0:
                    self.logger.warning(f"No training session found with ID: {session_id}")
                    return False

            self.logger.debug(f"Updated training progress for session {session_id}: iteration {iteration}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to update training progress: {e}")
            return False

    async def get_training_session_context(self, session_id: str) -> dict[str, Any] | None:
        """Get training session context from database.

        Args:
            session_id: Training session ID

        Returns:
            Training session context if available, None otherwise
        """
        if not session_id:
            self.logger.warning("No session ID provided for context retrieval")
            return None

        try:
            await self._ensure_database_services()

            # Get session data from database
            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    SELECT session_id, config, status, current_iteration,
                           performance_metrics, improvement_score, created_at, updated_at
                    FROM training_sessions
                    WHERE session_id = :session_id
                """)

                result = await db_session.execute(
                    query, {"session_id": session_id}
                )
                row = result.first()

                if row:
                    return {
                        "session_id": row[0],
                        "config": json.loads(row[1]) if row[1] else {},
                        "status": row[2],
                        "current_iteration": row[3] or 0,
                        "performance_metrics": json.loads(row[4]) if row[4] else {},
                        "improvement_score": row[5] or 0.0,
                        "created_at": row[6],
                        "updated_at": row[7],
                    }
                return None

        except Exception as e:
            self.logger.exception(f"Failed to get training session context: {e}")
            return None

    async def save_training_progress(self, session_id: str) -> bool:
        """Save current training progress to database.

        Args:
            session_id: Training session ID to save progress for

        Returns:
            True if progress saved successfully, False otherwise
        """
        if not session_id:
            self.logger.warning("No session ID provided for progress saving")
            return False

        try:
            await self._ensure_database_services()

            # Update session status to indicate progress save
            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    UPDATE training_sessions
                    SET status = 'paused',
                        updated_at = :updated_at,
                        progress_saved_at = :progress_saved_at
                    WHERE session_id = :session_id
                """)

                result = await db_session.execute(
                    query,
                    {
                        "updated_at": time.time(),
                        "progress_saved_at": time.time(),
                        "session_id": session_id,
                    },
                )
                await db_session.commit()

                if result.rowcount == 0:
                    self.logger.warning(f"No training session found for progress save: {session_id}")
                    return False

            self.logger.info(f"Saved training progress for session {session_id}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to save training progress: {e}")
            return False

    async def create_checkpoint(self, session_id: str) -> str:
        """Create emergency training checkpoint.

        Args:
            session_id: Training session ID to create checkpoint for

        Returns:
            Checkpoint ID if created successfully
        """
        try:
            await self._ensure_database_services()

            checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # Get current session state
            session_context = await self.get_training_session_context(session_id)
            if not session_context:
                raise ValueError(f"Training session {session_id} not found")

            # Create checkpoint record
            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    INSERT INTO training_checkpoints
                    (checkpoint_id, session_id, session_state, created_at)
                    VALUES (:checkpoint_id, :session_id, :session_state, :created_at)
                """)

                await db_session.execute(
                    query,
                    {
                        "checkpoint_id": checkpoint_id,
                        "session_id": session_id,
                        "session_state": json.dumps(session_context),
                        "created_at": time.time(),
                    },
                )
                await db_session.commit()

            self.logger.info(f"Created checkpoint {checkpoint_id} for session {session_id}")
            return checkpoint_id

        except Exception as e:
            self.logger.exception(f"Failed to create checkpoint: {e}")
            raise

    async def restore_from_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Restore training session from checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to restore from

        Returns:
            Restored session context if successful, None otherwise
        """
        try:
            await self._ensure_database_services()

            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    SELECT session_id, session_state, created_at
                    FROM training_checkpoints
                    WHERE checkpoint_id = :checkpoint_id
                """)

                result = await db_session.execute(
                    query, {"checkpoint_id": checkpoint_id}
                )
                row = result.first()

                if not row:
                    self.logger.warning(f"Checkpoint {checkpoint_id} not found")
                    return None

                session_state = json.loads(row[1])

                # Restore session to database
                await self._restore_session_state(session_state)

                self.logger.info(f"Restored session from checkpoint {checkpoint_id}")
                return session_state

        except Exception as e:
            self.logger.exception(f"Failed to restore from checkpoint: {e}")
            return None

    async def get_active_sessions(self) -> list[dict[str, Any]]:
        """Get all active training sessions.

        Returns:
            List of active training session dictionaries
        """
        try:
            await self._ensure_database_services()

            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    SELECT session_id, config, status, current_iteration,
                           performance_metrics, improvement_score, created_at, updated_at
                    FROM training_sessions
                    WHERE status IN ('initializing', 'running', 'paused')
                    ORDER BY created_at DESC
                """)

                result = await db_session.execute(query)
                rows = result.fetchall()

                active_sessions = []
                for row in rows:
                    session_data = {
                        "session_id": row[0],
                        "config": json.loads(row[1]) if row[1] else {},
                        "status": row[2],
                        "current_iteration": row[3] or 0,
                        "performance_metrics": json.loads(row[4]) if row[4] else {},
                        "improvement_score": row[5] or 0.0,
                        "created_at": row[6],
                        "updated_at": row[7],
                    }
                    active_sessions.append(session_data)

                return active_sessions

        except Exception as e:
            self.logger.exception(f"Failed to get active sessions: {e}")
            return []

    async def terminate_session(self, session_id: str, reason: str = "manual_termination") -> bool:
        """Terminate a training session.

        Args:
            session_id: Session ID to terminate
            reason: Reason for termination

        Returns:
            True if terminated successfully, False otherwise
        """
        try:
            await self._ensure_database_services()

            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    UPDATE training_sessions
                    SET status = 'terminated',
                        termination_reason = :reason,
                        terminated_at = :terminated_at,
                        updated_at = :updated_at
                    WHERE session_id = :session_id
                """)

                result = await db_session.execute(
                    query,
                    {
                        "reason": reason,
                        "terminated_at": time.time(),
                        "updated_at": time.time(),
                        "session_id": session_id,
                    },
                )
                await db_session.commit()

                if result.rowcount == 0:
                    self.logger.warning(f"No training session found to terminate: {session_id}")
                    return False

            self.logger.info(f"Terminated training session {session_id}: {reason}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to terminate session: {e}")
            return False

    async def get_session_history(
        self,
        session_id: str | None = None,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get training session history.

        Args:
            session_id: Specific session ID, or None for all sessions
            limit: Maximum number of records to return

        Returns:
            List of training session records
        """
        try:
            await self._ensure_database_services()

            async with self._unified_session_manager.get_async_session() as db_session:
                if session_id:
                    query = text("""
                        SELECT session_id, config, status, current_iteration,
                               performance_metrics, improvement_score, created_at,
                               updated_at, terminated_at, termination_reason
                        FROM training_sessions
                        WHERE session_id = :session_id
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """)
                    result = await db_session.execute(
                        query, {"session_id": session_id, "limit": limit}
                    )
                else:
                    query = text("""
                        SELECT session_id, config, status, current_iteration,
                               performance_metrics, improvement_score, created_at,
                               updated_at, terminated_at, termination_reason
                        FROM training_sessions
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """)
                    result = await db_session.execute(query, {"limit": limit})

                rows = result.fetchall()

                history = []
                for row in rows:
                    session_data = {
                        "session_id": row[0],
                        "config": json.loads(row[1]) if row[1] else {},
                        "status": row[2],
                        "current_iteration": row[3] or 0,
                        "performance_metrics": json.loads(row[4]) if row[4] else {},
                        "improvement_score": row[5] or 0.0,
                        "created_at": row[6],
                        "updated_at": row[7],
                        "terminated_at": row[8],
                        "termination_reason": row[9],
                    }
                    history.append(session_data)

                return history

        except Exception as e:
            self.logger.exception(f"Failed to get session history: {e}")
            return []

    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old completed training sessions.

        Args:
            days_old: Sessions older than this many days will be cleaned up

        Returns:
            Number of sessions cleaned up
        """
        try:
            await self._ensure_database_services()

            cutoff_time = time.time() - (days_old * 24 * 60 * 60)

            async with self._unified_session_manager.get_async_session() as db_session:
                # First, get count of sessions to be deleted
                count_query = text("""
                    SELECT COUNT(*) FROM training_sessions
                    WHERE status IN ('completed', 'terminated', 'failed')
                    AND created_at < :cutoff_time
                """)
                result = await db_session.execute(count_query, {"cutoff_time": cutoff_time})
                count = result.scalar() or 0

                if count == 0:
                    return 0

                # Delete old sessions
                delete_query = text("""
                    DELETE FROM training_sessions
                    WHERE status IN ('completed', 'terminated', 'failed')
                    AND created_at < :cutoff_time
                """)
                await db_session.execute(delete_query, {"cutoff_time": cutoff_time})
                await db_session.commit()

            self.logger.info(f"Cleaned up {count} old training sessions")
            return count

        except Exception as e:
            self.logger.exception(f"Failed to cleanup old sessions: {e}")
            return 0

    async def store_training_iteration_data(
        self,
        session_id: str,
        iteration: int,
        iteration_data: dict[str, Any]
    ) -> bool:
        """Store detailed training iteration data.

        Args:
            session_id: Training session ID
            iteration: Iteration number
            iteration_data: Detailed iteration data

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            await self._ensure_database_services()

            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    INSERT INTO training_iterations
                    (session_id, iteration, iteration_data, created_at)
                    VALUES (:session_id, :iteration, :iteration_data, :created_at)
                    ON CONFLICT (session_id, iteration)
                    DO UPDATE SET
                        iteration_data = EXCLUDED.iteration_data,
                        updated_at = :created_at
                """)

                await db_session.execute(
                    query,
                    {
                        "session_id": session_id,
                        "iteration": iteration,
                        "iteration_data": json.dumps(iteration_data),
                        "created_at": time.time(),
                    },
                )
                await db_session.commit()

            return True

        except Exception as e:
            self.logger.exception(f"Failed to store training iteration data: {e}")
            return False

    async def get_training_iteration_data(
        self,
        session_id: str,
        iteration: int | None = None
    ) -> list[dict[str, Any]]:
        """Get training iteration data for analysis.

        Args:
            session_id: Training session ID
            iteration: Specific iteration number, or None for all iterations

        Returns:
            List of iteration data records
        """
        try:
            await self._ensure_database_services()

            async with self._unified_session_manager.get_async_session() as db_session:
                if iteration is not None:
                    query = text("""
                        SELECT iteration, iteration_data, created_at, updated_at
                        FROM training_iterations
                        WHERE session_id = :session_id AND iteration = :iteration
                    """)
                    result = await db_session.execute(
                        query, {"session_id": session_id, "iteration": iteration}
                    )
                else:
                    query = text("""
                        SELECT iteration, iteration_data, created_at, updated_at
                        FROM training_iterations
                        WHERE session_id = :session_id
                        ORDER BY iteration ASC
                    """)
                    result = await db_session.execute(query, {"session_id": session_id})

                rows = result.fetchall()

                iteration_data = []
                for row in rows:
                    data = {
                        "iteration": row[0],
                        "data": json.loads(row[1]) if row[1] else {},
                        "created_at": row[2],
                        "updated_at": row[3],
                    }
                    iteration_data.append(data)

                return iteration_data

        except Exception as e:
            self.logger.exception(f"Failed to get training iteration data: {e}")
            return []

    async def _restore_session_state(self, session_state: dict[str, Any]) -> None:
        """Restore session state to database.

        Args:
            session_state: Session state to restore
        """
        try:
            session_id = session_state["session_id"]

            async with self._unified_session_manager.get_async_session() as db_session:
                query = text("""
                    UPDATE training_sessions
                    SET config = :config,
                        current_iteration = :current_iteration,
                        performance_metrics = :performance_metrics,
                        improvement_score = :improvement_score,
                        status = 'restored',
                        updated_at = :updated_at
                    WHERE session_id = :session_id
                """)

                await db_session.execute(
                    query,
                    {
                        "config": json.dumps(session_state.get("config", {})),
                        "current_iteration": session_state.get("current_iteration", 0),
                        "performance_metrics": json.dumps(session_state.get("performance_metrics", {})),
                        "improvement_score": session_state.get("improvement_score", 0.0),
                        "updated_at": time.time(),
                        "session_id": session_id,
                    },
                )
                await db_session.commit()

        except Exception as e:
            self.logger.exception(f"Failed to restore session state: {e}")
            raise

    # Utility methods
    async def get_database_health(self) -> dict[str, Any]:
        """Check database connectivity and health.

        Returns:
            Database health status
        """
        try:
            await self._ensure_database_services()

            async with self._unified_session_manager.get_async_session() as db_session:
                # Test basic connectivity
                await db_session.execute(text("SELECT 1"))

                # Get session counts
                result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_sessions")
                )
                total_sessions = result.scalar() or 0

                result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_sessions WHERE status IN ('running', 'paused')")
                )
                active_sessions = result.scalar() or 0

                return {
                    "status": "healthy",
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
