"""Clean Persistence Service - Database persistence operations using dependency injection.

Refactored service that follows clean architecture principles:
- Uses repository interfaces instead of direct database imports
- No coupling to infrastructure layers
- Pure domain logic with injected dependencies

Handles all database operations for:
- Improvement sessions storage
- Performance metrics tracking 
- User feedback management
- ML optimization results storage
- A/B experiment management
- Pattern discovery results
"""

import logging
from datetime import datetime
from typing import Any

from prompt_improver.repositories.protocols import (
    PersistenceRepositoryProtocol,
)
from prompt_improver.repositories.protocols.persistence_repository_protocol import (
    SessionData,
    RulePerformanceData,
    FeedbackData,
    ModelPerformanceData,
    ExperimentData,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class CleanPersistenceService:
    """Clean persistence service using dependency injection."""

    def __init__(self, persistence_repository: PersistenceRepositoryProtocol):
        """Initialize service with injected repository dependency."""
        self._repository = persistence_repository

    async def store_session(
        self,
        session_id: str,
        original_prompt: str,
        final_prompt: str,
        rules_applied: list[dict[str, Any]],
        user_context: dict[str, Any] | None = None,
        performance_metrics: dict[str, float] | None = None,
    ) -> bool:
        """Store improvement session data."""
        try:
            session_data = SessionData(
                session_id=session_id,
                original_prompt=original_prompt,
                final_prompt=final_prompt,
                rules_applied=rules_applied,
                user_context=user_context,
                performance_metrics=performance_metrics,
                created_at=aware_utc_now(),
            )
            
            success = await self._repository.store_session(session_data)
            if success:
                logger.info(f"Successfully stored session {session_id}")
            else:
                logger.error(f"Failed to store session {session_id}")
            
            return success

        except Exception as e:
            logger.error(f"Error storing session {session_id}: {e}")
            return False

    async def store_rule_performance(
        self,
        session_id: str,
        rule_id: str,
        improvement_score: float,
        confidence_level: float,
        execution_time_ms: int,
        prompt_type: str | None = None,
        prompt_category: str | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> bool:
        """Store rule performance metrics."""
        try:
            performance_data = RulePerformanceData(
                rule_id=rule_id,
                session_id=session_id,
                improvement_score=improvement_score,
                confidence_level=confidence_level,
                execution_time_ms=execution_time_ms,
                prompt_type=prompt_type,
                prompt_category=prompt_category,
                context_data=context_data,
                created_at=aware_utc_now(),
            )

            success = await self._repository.store_rule_performance(performance_data)
            if success:
                logger.debug(f"Stored rule performance for {rule_id} in session {session_id}")
            else:
                logger.error(f"Failed to store rule performance for {rule_id}")
                
            return success

        except Exception as e:
            logger.error(f"Error storing rule performance: {e}")
            return False

    async def store_feedback(
        self,
        session_id: str,
        rating: int,
        feedback_text: str | None = None,
        improvement_areas: list[str] | None = None,
        user_id: str | None = None,
        model_id: str | None = None,
    ) -> bool:
        """Store user feedback."""
        try:
            feedback_data = FeedbackData(
                session_id=session_id,
                user_id=user_id,
                rating=rating,
                feedback_text=feedback_text,
                improvement_areas=improvement_areas,
                model_id=model_id,
                is_processed=False,
                created_at=aware_utc_now(),
            )

            success = await self._repository.store_feedback(feedback_data)
            if success:
                logger.info(f"Stored feedback for session {session_id}")
            else:
                logger.error(f"Failed to store feedback for session {session_id}")
                
            return success

        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            return False

    async def store_model_performance(
        self,
        model_id: str,
        model_type: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        training_data_size: int,
        evaluation_data_size: int,
        hyperparameters: dict[str, Any] | None = None,
    ) -> bool:
        """Store ML model performance metrics."""
        try:
            performance_data = ModelPerformanceData(
                model_id=model_id,
                model_type=model_type,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                training_data_size=training_data_size,
                evaluation_data_size=evaluation_data_size,
                hyperparameters=hyperparameters,
                evaluation_date=aware_utc_now(),
                created_at=aware_utc_now(),
            )

            success = await self._repository.store_model_performance(performance_data)
            if success:
                logger.info(f"Stored model performance for {model_id}")
            else:
                logger.error(f"Failed to store model performance for {model_id}")
                
            return success

        except Exception as e:
            logger.error(f"Error storing model performance: {e}")
            return False

    async def store_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        experiment_type: str,
        configuration: dict[str, Any],
        description: str | None = None,
    ) -> bool:
        """Store A/B experiment configuration."""
        try:
            experiment_data = ExperimentData(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                description=description,
                experiment_type=experiment_type,
                status="active",
                configuration=configuration,
                start_date=aware_utc_now(),
                created_at=aware_utc_now(),
            )

            success = await self._repository.store_experiment(experiment_data)
            if success:
                logger.info(f"Stored experiment {experiment_id}")
            else:
                logger.error(f"Failed to store experiment {experiment_id}")
                
            return success

        except Exception as e:
            logger.error(f"Error storing experiment: {e}")
            return False

    async def get_session(self, session_id: str) -> SessionData | None:
        """Retrieve session by ID."""
        try:
            return await self._repository.get_session(session_id)
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None

    async def get_recent_sessions(
        self, 
        limit: int = 100,
        user_context_filter: dict[str, Any] | None = None
    ) -> list[SessionData]:
        """Get recent sessions with optional filtering."""
        try:
            return await self._repository.get_recent_sessions(limit, user_context_filter)
        except Exception as e:
            logger.error(f"Error retrieving recent sessions: {e}")
            return []

    async def get_rule_performance_history(
        self,
        rule_id: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 1000
    ) -> list[RulePerformanceData]:
        """Get performance history for a rule."""
        try:
            return await self._repository.get_rule_performance_history(
                rule_id, date_from, date_to, limit
            )
        except Exception as e:
            logger.error(f"Error retrieving rule performance history: {e}")
            return []

    async def get_feedback_by_session(self, session_id: str) -> list[FeedbackData]:
        """Get feedback for a session."""
        try:
            return await self._repository.get_feedback_by_session(session_id)
        except Exception as e:
            logger.error(f"Error retrieving feedback for session {session_id}: {e}")
            return []

    async def get_session_analytics(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> dict[str, Any]:
        """Get session analytics for date range."""
        try:
            return await self._repository.get_session_analytics(date_from, date_to)
        except Exception as e:
            logger.error(f"Error retrieving session analytics: {e}")
            return {}

    async def cleanup_old_data(self, days_old: int = 90) -> tuple[int, int]:
        """Clean up old sessions and performance data."""
        try:
            sessions_deleted = await self._repository.cleanup_old_sessions(days_old)
            performance_deleted = await self._repository.cleanup_old_performance_data(days_old)
            
            logger.info(f"Cleaned up {sessions_deleted} sessions and {performance_deleted} performance records")
            return sessions_deleted, performance_deleted
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0, 0