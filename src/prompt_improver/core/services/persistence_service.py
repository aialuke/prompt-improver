"""Persistence Service - Database persistence operations.

Handles all database operations for:
- Improvement sessions storage
- Performance metrics tracking
- User feedback management
- ML optimization results storage
- A/B experiment management
- Pattern discovery results

Follows single responsibility principle for data persistence concerns.
"""

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select as sqlmodel_select

from prompt_improver.utils.datetime_utils import aware_utc_now

if TYPE_CHECKING:
    from prompt_improver.database.models import (
        UserFeedback,
    )

logger = logging.getLogger(__name__)


class PersistenceService:
    """Service focused on database persistence operations."""

    async def store_session(
        self,
        session_id: str,
        original_prompt: str,
        final_prompt: str,
        rules_applied: list[dict[str, Any]],
        user_context: dict[str, Any] | None,
        db_session: AsyncSession,
    ) -> None:
        """Store improvement session in database."""
        try:
            rule_ids = (
                [rule.get("rule_id", "unknown") for rule in rules_applied]
                if rules_applied
                else None
            )
            from prompt_improver.database.models import (
                ImprovementSession,
                ImprovementSessionCreate,
            )

            session_data = ImprovementSessionCreate(
                session_id=session_id,
                original_prompt=original_prompt,
                final_prompt=final_prompt,
                rules_applied=rule_ids,
                user_context=user_context,
            )
            db_session.add(ImprovementSession(**session_data.model_dump()))
            await db_session.commit()
        except Exception as e:
            logger.exception(f"Error storing session: {e}")
            await db_session.rollback()

    async def store_performance_metrics(
        self, performance_data: list[dict[str, Any]], db_session: AsyncSession
    ) -> None:
        """Store rule performance metrics."""
        try:
            from prompt_improver.database.models import (
                RulePerformance,
                RulePerformanceCreate,
            )

            for data in performance_data:
                perf_record = RulePerformanceCreate(
                    session_id=data.get("session_id", "unknown"),
                    rule_id=data["rule_id"],
                    improvement_score=data["improvement_score"],
                    execution_time_ms=data["execution_time_ms"],
                    confidence_level=data["confidence"],
                    parameters_used=data.get("parameters_used"),
                )
                db_session.add(RulePerformance(**perf_record.model_dump()))
            await db_session.commit()
        except Exception as e:
            logger.exception(f"Error storing performance metrics: {e}")
            await db_session.rollback()

    async def store_user_feedback(
        self,
        session_id: str,
        rating: int,
        feedback_text: str | None,
        improvement_areas: list[str] | None,
        db_session: AsyncSession,
    ) -> "UserFeedback":
        """Store user feedback."""
        try:
            from prompt_improver.database.models import (
                ImprovementSession,
                UserFeedback,
                UserFeedbackCreate,
            )

            # Verify session exists
            session_query = sqlmodel_select(ImprovementSession).where(
                ImprovementSession.session_id == session_id
            )
            session_result = await db_session.execute(session_query)
            session_record = session_result.scalar_one_or_none()
            if not session_record:
                raise ValueError(f"Session {session_id} not found")

            feedback_data = UserFeedbackCreate(
                session_id=session_id,
                rating=rating,
                feedback_text=feedback_text,
                improvement_areas=improvement_areas,
            )
            feedback = UserFeedback(**feedback_data.model_dump())
            db_session.add(feedback)
            await db_session.commit()
            await db_session.refresh(feedback)
            return feedback
        except Exception as e:
            logger.exception(f"Error storing user feedback: {e}")
            await db_session.rollback()
            raise

    async def store_optimization_trigger(
        self,
        db_session: AsyncSession,
        feedback_id: int,
        model_id: str | None,
    ) -> None:
        """Store optimization trigger event for tracking."""
        try:
            from prompt_improver.database.models import UserFeedback

            stmt = sqlmodel_select(UserFeedback).where(UserFeedback.id == feedback_id)
            result = await db_session.execute(stmt)
            feedback = result.scalar_one_or_none()
            if feedback:
                feedback.ml_optimized = True
                feedback.model_id = model_id
                db_session.add(feedback)
                await db_session.commit()
        except Exception as e:
            logger.exception(f"Failed to store optimization trigger: {e}")
            await db_session.rollback()

    async def store_ml_optimization_results(
        self,
        db_session: AsyncSession,
        optimization_result: dict[str, Any],
        training_samples: int,
    ) -> None:
        """Store ML optimization results for analytics."""
        try:
            from prompt_improver.database.models import MLModelPerformance

            performance_record = MLModelPerformance(
                model_id=optimization_result.get("model_id", "unknown"),
                performance_score=optimization_result.get("best_score", 0),
                accuracy=optimization_result.get("accuracy", 0),
                precision=optimization_result.get("precision", 0),
                recall=optimization_result.get("recall", 0),
                training_samples=training_samples,
                created_at=aware_utc_now(),
            )
            db_session.add(performance_record)
            await db_session.commit()
        except Exception as e:
            logger.exception(f"Failed to store ML optimization results: {e}")
            await db_session.rollback()

    async def create_ab_experiments_from_patterns(
        self, db_session: AsyncSession, patterns: list[dict[str, Any]]
    ) -> None:
        """Create A/B experiments from discovered patterns."""
        try:
            from prompt_improver.database.models import ABExperiment

            for i, pattern in enumerate(patterns):
                experiment_name = f"Pattern_{i + 1}_Effectiveness_{pattern.get('avg_effectiveness', 0):.2f}"
                existing_stmt = sqlmodel_select(ABExperiment).where(
                    ABExperiment.experiment_name.like(f"Pattern_{i + 1}_%"),
                    ABExperiment.status == "running",
                )
                result = await db_session.execute(existing_stmt)
                existing = result.scalar_one_or_none()
                if not existing:
                    experiment = ABExperiment(
                        experiment_name=experiment_name,
                        control_rules={"baseline": "current_rules"},
                        treatment_rules={"optimized": pattern.get("parameters", {})},
                        status="running",
                        started_at=aware_utc_now(),
                    )
                    db_session.add(experiment)
            await db_session.commit()
        except Exception as e:
            logger.exception(f"Failed to create A/B experiments: {e}")
            await db_session.rollback()

    async def store_pattern_discovery_results(
        self, db_session: AsyncSession, discovery_result: dict[str, Any]
    ) -> None:
        """Store pattern discovery results for future reference."""
        try:
            logger.info(
                f"Pattern discovery completed with {discovery_result.get('patterns_discovered')} patterns"
            )
            # Could extend this to store patterns in a dedicated table if needed
        except Exception as e:
            logger.exception(f"Failed to store pattern discovery results: {e}")
            await db_session.rollback()

    async def get_feedback_by_id(
        self, feedback_id: int, db_session: AsyncSession
    ) -> "UserFeedback | None":
        """Get user feedback by ID."""
        try:
            from prompt_improver.database.models import UserFeedback

            stmt = sqlmodel_select(UserFeedback).where(UserFeedback.id == feedback_id)
            result = await db_session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.exception(f"Error retrieving feedback {feedback_id}: {e}")
            return None

    async def get_performance_data_for_optimization(
        self, db_session: AsyncSession
    ) -> list[tuple[Any, ...]]:
        """Get performance data for ML optimization."""
        try:
            from sqlalchemy import select

            from prompt_improver.database.models import RuleMetadata, RulePerformance

            perf_stmt = (
                select(RulePerformance, RuleMetadata.default_parameters)
                .join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id)
                .where(RulePerformance.created_at.isnot(None))
            )
            result = await db_session.execute(perf_stmt)
            return result.fetchall()
        except Exception as e:
            logger.exception(f"Error getting performance data: {e}")
            return []

    async def bulk_store_performance_metrics(
        self,
        performance_data: list[dict[str, Any]],
        db_session: AsyncSession,
        batch_size: int = 100,
    ) -> None:
        """Store performance metrics in batches for better performance."""
        try:
            from prompt_improver.database.models import (
                RulePerformance,
                RulePerformanceCreate,
            )

            for i in range(0, len(performance_data), batch_size):
                batch = performance_data[i : i + batch_size]
                records = []
                for data in batch:
                    perf_record = RulePerformanceCreate(
                        session_id=data.get("session_id", "unknown"),
                        rule_id=data["rule_id"],
                        improvement_score=data["improvement_score"],
                        execution_time_ms=data["execution_time_ms"],
                        confidence_level=data["confidence"],
                        parameters_used=data.get("parameters_used"),
                    )
                    records.append(RulePerformance(**perf_record.model_dump()))

                db_session.add_all(records)
                await db_session.flush()

            await db_session.commit()
        except Exception as e:
            logger.exception(f"Error in bulk storing performance metrics: {e}")
            await db_session.rollback()

    async def cleanup_old_sessions(
        self, db_session: AsyncSession, days_old: int = 90
    ) -> int:
        """Clean up old improvement sessions."""
        try:
            from datetime import timedelta

            from sqlalchemy import delete

            from prompt_improver.database.models import ImprovementSession

            cutoff_date = aware_utc_now() - timedelta(days=days_old)
            stmt = delete(ImprovementSession).where(
                ImprovementSession.created_at < cutoff_date
            )
            result = await db_session.execute(stmt)
            await db_session.commit()

            deleted_count = result.rowcount
            logger.info(
                f"Cleaned up {deleted_count} old sessions (older than {days_old} days)"
            )
            return deleted_count
        except Exception as e:
            logger.exception(f"Error cleaning up old sessions: {e}")
            await db_session.rollback()
            return 0
