"""Training repository implementation for training sessions and iterations.

Handles training session lifecycle, iteration management, and training data operations
following repository pattern with protocol-based dependency injection.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, select, update

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    TrainingIteration,
    TrainingPrompt,
    TrainingSession,
    TrainingSessionCreate,
    TrainingSessionUpdate,
)
from prompt_improver.repositories.base_repository import BaseRepository
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    TrainingMetrics,
    TrainingSessionFilter,
)

logger = logging.getLogger(__name__)


class TrainingRepository(BaseRepository[TrainingSession]):
    """Repository for training session and iteration management."""

    def __init__(self, connection_manager: DatabaseServices) -> None:
        super().__init__(
            model_class=TrainingSession,
            connection_manager=connection_manager,
            create_model_class=TrainingSessionCreate,
            update_model_class=TrainingSessionUpdate,
        )
        self.connection_manager = connection_manager
        logger.info("Training repository initialized")

    # Training Session Management

    async def create_training_session(
        self, session_data: TrainingSessionCreate
    ) -> TrainingSession:
        """Create a new training session."""
        return await self.create(session_data)

    async def get_training_sessions(
        self,
        filters: TrainingSessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TrainingSession]:
        """Retrieve training sessions with filtering."""
        async with self.get_session() as session:
            try:
                query = select(TrainingSession)

                # Apply filters
                if filters:
                    conditions = []
                    if filters.status:
                        conditions.append(TrainingSession.status == filters.status)
                    if filters.continuous_mode is not None:
                        conditions.append(
                            TrainingSession.continuous_mode == filters.continuous_mode
                        )
                    if filters.min_performance is not None:
                        conditions.append(
                            TrainingSession.current_performance
                            >= filters.min_performance
                        )
                    if filters.max_performance is not None:
                        conditions.append(
                            TrainingSession.current_performance
                            <= filters.max_performance
                        )
                    if filters.date_from:
                        conditions.append(
                            TrainingSession.created_at >= filters.date_from
                        )
                    if filters.date_to:
                        conditions.append(TrainingSession.created_at <= filters.date_to)
                    if filters.active_workflow_id:
                        conditions.append(
                            TrainingSession.active_workflow_id
                            == filters.active_workflow_id
                        )

                    if conditions:
                        query = query.where(and_(*conditions))

                # Apply sorting
                if hasattr(TrainingSession, sort_by):
                    sort_field = getattr(TrainingSession, sort_by)
                    if sort_desc:
                        query = query.order_by(desc(sort_field))
                    else:
                        query = query.order_by(sort_field)

                # Apply pagination
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.exception(f"Error getting training sessions: {e}")
                raise

    async def get_training_session_by_id(
        self, session_id: str
    ) -> TrainingSession | None:
        """Get training session by ID."""
        async with self.get_session() as session:
            try:
                query = select(TrainingSession).where(TrainingSession.id == session_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except Exception as e:
                logger.exception(f"Error getting training session by ID: {e}")
                raise

    async def update_training_session(
        self,
        session_id: str,
        update_data: TrainingSessionUpdate,
    ) -> TrainingSession | None:
        """Update training session."""
        async with self.get_session() as session:
            try:
                update_dict = update_data.model_dump(exclude_unset=True)
                query = (
                    update(TrainingSession)
                    .where(TrainingSession.id == session_id)
                    .values(**update_dict)
                )
                result = await session.execute(query)

                if result.rowcount == 0:
                    return None

                await session.commit()
                return await self.get_training_session_by_id(session_id)
            except Exception as e:
                logger.exception(f"Error updating training session: {e}")
                raise

    async def get_active_training_sessions(self) -> list[TrainingSession]:
        """Get all currently active training sessions."""
        async with self.get_session() as session:
            try:
                query = select(TrainingSession).where(
                    TrainingSession.status.in_(["running", "paused"])
                )
                result = await session.execute(query)
                return list(result.scalars().all())
            except Exception as e:
                logger.exception(f"Error getting active training sessions: {e}")
                raise

    async def get_training_session_metrics(
        self,
        session_id: str,
    ) -> TrainingMetrics | None:
        """Get comprehensive metrics for training session."""
        async with self.get_session() as session:
            try:
                # Get training session
                training_session = await self.get_training_session_by_id(session_id)
                if not training_session:
                    return None

                # Get iteration count
                iterations_query = select(func.count(TrainingIteration.id)).where(
                    TrainingIteration.session_id == session_id
                )
                iterations_result = await session.execute(iterations_query)
                total_iterations = iterations_result.scalar() or 0

                # Get latest iteration for performance metrics
                latest_iteration_query = (
                    select(TrainingIteration)
                    .where(TrainingIteration.session_id == session_id)
                    .order_by(desc(TrainingIteration.iteration_number))
                    .limit(1)
                )
                latest_result = await session.execute(latest_iteration_query)
                latest_iteration = latest_result.scalar_one_or_none()

                # Calculate improvement rate
                improvement_rate = None
                if total_iterations > 1:
                    first_iteration_query = (
                        select(TrainingIteration)
                        .where(TrainingIteration.session_id == session_id)
                        .order_by(TrainingIteration.iteration_number)
                        .limit(1)
                    )
                    first_result = await session.execute(first_iteration_query)
                    first_iteration = first_result.scalar_one_or_none()

                    if (
                        first_iteration
                        and latest_iteration
                        and first_iteration.performance_score
                        and latest_iteration.performance_score
                    ):
                        improvement_rate = (
                            latest_iteration.performance_score
                            - first_iteration.performance_score
                        ) / total_iterations

                return TrainingMetrics(
                    session_id=session_id,
                    total_iterations=total_iterations,
                    current_performance=training_session.current_performance,
                    best_performance=training_session.best_performance,
                    improvement_rate=improvement_rate,
                    efficiency_score=training_session.efficiency_score,
                    resource_utilization=training_session.resource_utilization or {},
                    status_summary={
                        "status": training_session.status,
                        "continuous_mode": training_session.continuous_mode,
                        "created_at": training_session.created_at.isoformat(),
                        "updated_at": training_session.updated_at.isoformat(),
                    },
                )

            except Exception as e:
                logger.exception(f"Error getting training session metrics: {e}")
                raise

    # Training Iteration Management

    async def create_training_iteration(
        self,
        iteration_data: dict[str, Any],
    ) -> TrainingIteration:
        """Create training iteration record."""
        async with self.get_session() as session:
            try:
                iteration = TrainingIteration(**iteration_data)
                session.add(iteration)
                await session.commit()
                await session.refresh(iteration)
                logger.info(f"Created training iteration {iteration.id}")
                return iteration
            except Exception as e:
                logger.exception(f"Error creating training iteration: {e}")
                raise

    async def get_training_iterations(
        self,
        session_id: str,
        start_iteration: int | None = None,
        end_iteration: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TrainingIteration]:
        """Get training iterations for session."""
        async with self.get_session() as session:
            try:
                query = select(TrainingIteration).where(
                    TrainingIteration.session_id == session_id
                )

                conditions = []
                if start_iteration is not None:
                    conditions.append(
                        TrainingIteration.iteration_number >= start_iteration
                    )
                if end_iteration is not None:
                    conditions.append(
                        TrainingIteration.iteration_number <= end_iteration
                    )

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(TrainingIteration.iteration_number)
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.exception(f"Error getting training iterations: {e}")
                raise

    async def get_latest_iteration(
        self,
        session_id: str,
    ) -> TrainingIteration | None:
        """Get latest iteration for training session."""
        async with self.get_session() as session:
            try:
                query = (
                    select(TrainingIteration)
                    .where(TrainingIteration.session_id == session_id)
                    .order_by(desc(TrainingIteration.iteration_number))
                    .limit(1)
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except Exception as e:
                logger.exception(f"Error getting latest iteration: {e}")
                raise

    async def get_iteration_performance_trend(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Get performance trend across iterations."""
        async with self.get_session() as session:
            try:
                query = (
                    select(TrainingIteration)
                    .where(
                        and_(
                            TrainingIteration.session_id == session_id,
                            TrainingIteration.performance_score.is_not(None),
                        )
                    )
                    .order_by(TrainingIteration.iteration_number)
                )
                result = await session.execute(query)
                iterations = result.scalars().all()

                return [
                    {
                        "iteration_number": iteration.iteration_number,
                        "performance_score": iteration.performance_score,
                        "timestamp": iteration.created_at.isoformat(),
                        "duration_seconds": iteration.duration_seconds,
                    }
                    for iteration in iterations
                ]

            except Exception as e:
                logger.exception(f"Error getting iteration performance trend: {e}")
                raise

    # Training Data Management - Core Methods Only

    async def get_training_prompts_count(
        self,
        session_id: str | None = None,
        is_active: bool = True,
    ) -> int:
        """Get count of training prompts for session."""
        async with self.get_session() as session:
            try:
                query = select(func.count(TrainingPrompt.id))
                conditions = []

                if session_id:
                    conditions.append(TrainingPrompt.session_id == session_id)
                if is_active:
                    conditions.append(TrainingPrompt.is_active == is_active)

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                return result.scalar() or 0
            except Exception as e:
                logger.exception(f"Error getting training prompts count: {e}")
                raise

    # Cleanup and Maintenance

    async def cleanup_old_iterations(
        self,
        session_id: str,
        keep_latest: int = 100,
    ) -> int:
        """Clean up old iterations, keeping latest N."""
        async with self.get_session() as session:
            try:
                # Get iteration IDs to keep
                keep_query = (
                    select(TrainingIteration.id)
                    .where(TrainingIteration.session_id == session_id)
                    .order_by(desc(TrainingIteration.iteration_number))
                    .limit(keep_latest)
                )
                keep_result = await session.execute(keep_query)
                keep_ids = [row[0] for row in keep_result.all()]

                if not keep_ids:
                    return 0

                # Delete iterations not in keep list
                from sqlalchemy import delete

                delete_query = delete(TrainingIteration).where(
                    and_(
                        TrainingIteration.session_id == session_id,
                        TrainingIteration.id.not_in(keep_ids),
                    )
                )
                result = await session.execute(delete_query)
                await session.commit()

                deleted_count = result.rowcount
                logger.info(
                    f"Cleaned up {deleted_count} old iterations for session {session_id}"
                )
                return deleted_count

            except Exception as e:
                logger.exception(f"Error cleaning up old iterations: {e}")
                raise

    async def cleanup_failed_sessions(
        self,
        days_old: int = 7,
    ) -> int:
        """Clean up failed training sessions older than specified days."""
        async with self.get_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_old)

                from sqlalchemy import delete

                delete_query = delete(TrainingSession).where(
                    and_(
                        TrainingSession.status == "failed",
                        TrainingSession.created_at < cutoff_date,
                    )
                )
                result = await session.execute(delete_query)
                await session.commit()

                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} failed training sessions")
                return deleted_count

            except Exception as e:
                logger.exception(f"Error cleaning up failed sessions: {e}")
                raise

    async def archive_completed_sessions(
        self,
        days_old: int = 30,
    ) -> int:
        """Archive completed sessions older than specified days."""
        async with self.get_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_old)

                query = (
                    update(TrainingSession)
                    .where(
                        and_(
                            TrainingSession.status == "completed",
                            TrainingSession.created_at < cutoff_date,
                            TrainingSession.status != "archived",
                        )
                    )
                    .values(status="archived")
                )
                result = await session.execute(query)
                await session.commit()

                archived_count = result.rowcount
                logger.info(f"Archived {archived_count} completed training sessions")
                return archived_count

            except Exception as e:
                logger.exception(f"Error archiving completed sessions: {e}")
                raise
