"""Concrete implementation of the prompt repository protocol.

Provides database operations for prompt improvement sessions, A/B experiments,
discovered patterns, and cross-domain analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlmodel import select as sqlmodel_select

from prompt_improver.database.models import (
    ABExperiment,
    DiscoveredPattern,
    ImprovementSession,
    ImprovementSessionCreate,
    RuleMetadata,
    RulePerformance,
    UserFeedback,
)
from prompt_improver.repositories.protocols.prompt_repository_protocol import (
    ExperimentFilter,
    OptimizationData,
    PatternFilter,
    PromptRepositoryProtocol,
    SessionAnalytics,
    SessionFilter,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class PromptRepository:
    """Concrete implementation of PromptRepositoryProtocol."""

    def __init__(self, session_factory):
        """Initialize repository with session factory."""
        self._session_factory = session_factory

    async def _get_session(self) -> AsyncSession:
        """Get database session from factory."""
        return self._session_factory()

    # Session Management
    async def create_session(
        self, session_data: ImprovementSessionCreate
    ) -> ImprovementSession:
        """Create a new improvement session."""
        async with await self._get_session() as session:
            db_session = ImprovementSession(**session_data.model_dump())
            session.add(db_session)
            await session.commit()
            await session.refresh(db_session)
            return db_session

    async def get_session(self, session_id: str) -> ImprovementSession | None:
        """Get session by session ID."""
        async with await self._get_session() as session:
            query = select(ImprovementSession).where(
                ImprovementSession.session_id == session_id
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def get_sessions(
        self,
        filters: SessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ImprovementSession]:
        """Get sessions with filtering and pagination."""
        async with await self._get_session() as session:
            query = select(ImprovementSession)

            if filters:
                conditions = []
                if filters.session_id:
                    conditions.append(
                        ImprovementSession.session_id == filters.session_id
                    )
                if filters.min_quality_score is not None:
                    conditions.append(
                        ImprovementSession.quality_score >= filters.min_quality_score
                    )
                if filters.min_improvement_score is not None:
                    conditions.append(
                        ImprovementSession.improvement_score
                        >= filters.min_improvement_score
                    )
                if filters.min_confidence_level is not None:
                    conditions.append(
                        ImprovementSession.confidence_level
                        >= filters.min_confidence_level
                    )
                if filters.date_from:
                    conditions.append(
                        ImprovementSession.created_at >= filters.date_from
                    )
                if filters.date_to:
                    conditions.append(ImprovementSession.created_at <= filters.date_to)
                if filters.user_context_contains:
                    conditions.append(
                        ImprovementSession.user_context.contains(
                            filters.user_context_contains
                        )
                    )
                if filters.has_user_feedback is not None:
                    if filters.has_user_feedback:
                        query = query.join(UserFeedback)
                    else:
                        query = query.outerjoin(UserFeedback).where(
                            UserFeedback.id.is_(None)
                        )

                if conditions:
                    query = query.where(and_(*conditions))

            # Apply sorting
            sort_column = getattr(
                ImprovementSession, sort_by, ImprovementSession.created_at
            )
            if sort_desc:
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(sort_column)

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return result.scalars().all()

    async def update_session(
        self, session_id: str, update_data: dict[str, Any]
    ) -> ImprovementSession | None:
        """Update session data."""
        async with await self._get_session() as session:
            query = select(ImprovementSession).where(
                ImprovementSession.session_id == session_id
            )
            result = await session.execute(query)
            db_session = result.scalar_one_or_none()

            if not db_session:
                return None

            for key, value in update_data.items():
                if hasattr(db_session, key):
                    setattr(db_session, key, value)

            db_session.updated_at = aware_utc_now()
            await session.commit()
            await session.refresh(db_session)
            return db_session

    async def get_session_analytics(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        filters: SessionFilter | None = None,
    ) -> SessionAnalytics:
        """Get comprehensive session analytics."""
        async with await self._get_session() as session:
            base_query = select(ImprovementSession)

            conditions = []
            if date_from:
                conditions.append(ImprovementSession.created_at >= date_from)
            if date_to:
                conditions.append(ImprovementSession.created_at <= date_to)
            if conditions:
                base_query = base_query.where(and_(*conditions))

            # Total sessions
            count_result = await session.execute(
                select(func.count(ImprovementSession.id)).select_from(
                    base_query.subquery()
                )
            )
            total_sessions = count_result.scalar() or 0

            # Average scores
            avg_query = select(
                func.avg(ImprovementSession.improvement_score),
                func.avg(ImprovementSession.confidence_level),
                func.count(UserFeedback.id),
            ).select_from(
                base_query.subquery().outerjoin(
                    UserFeedback,
                    text("improvement_session.session_id = user_feedback.session_id"),
                )
            )

            avg_result = await session.execute(avg_query)
            avg_improvement, avg_confidence, feedback_count = avg_result.first()

            return SessionAnalytics(
                total_sessions=total_sessions,
                avg_improvement_score=float(avg_improvement or 0),
                avg_confidence_level=float(avg_confidence or 0),
                sessions_with_feedback=feedback_count or 0,
                most_common_rules_applied=[],  # Would need additional complex query
                improvement_trends={},  # Would need time-series analysis
                user_satisfaction_correlation={},  # Would need correlation analysis
            )

    async def get_recent_sessions(
        self, hours_back: int = 24, limit: int = 50
    ) -> list[ImprovementSession]:
        """Get recent improvement sessions."""
        cutoff_time = aware_utc_now() - timedelta(hours=hours_back)

        async with await self._get_session() as session:
            query = (
                select(ImprovementSession)
                .where(ImprovementSession.created_at >= cutoff_time)
                .order_by(desc(ImprovementSession.created_at))
                .limit(limit)
            )
            result = await session.execute(query)
            return result.scalars().all()

    # A/B Experiment Management
    async def create_experiment(self, experiment_data: dict[str, Any]) -> ABExperiment:
        """Create a new A/B experiment."""
        async with await self._get_session() as session:
            experiment = ABExperiment(**experiment_data)
            session.add(experiment)
            await session.commit()
            await session.refresh(experiment)
            return experiment

    async def get_experiment(self, experiment_id: str) -> ABExperiment | None:
        """Get experiment by ID."""
        async with await self._get_session() as session:
            query = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def get_experiments(
        self,
        filters: ExperimentFilter | None = None,
        sort_by: str = "started_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ABExperiment]:
        """Get experiments with filtering."""
        async with await self._get_session() as session:
            query = select(ABExperiment)

            if filters:
                conditions = []
                if filters.status:
                    conditions.append(ABExperiment.status == filters.status)
                if filters.target_metric:
                    conditions.append(
                        ABExperiment.target_metric == filters.target_metric
                    )
                if filters.min_sample_size is not None:
                    conditions.append(
                        ABExperiment.current_sample_size >= filters.min_sample_size
                    )
                if filters.started_after:
                    conditions.append(ABExperiment.started_at >= filters.started_after)
                if filters.completed_before:
                    conditions.append(
                        ABExperiment.completed_at <= filters.completed_before
                    )

                if conditions:
                    query = query.where(and_(*conditions))

            # Apply sorting
            sort_column = getattr(ABExperiment, sort_by, ABExperiment.started_at)
            if sort_desc:
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(sort_column)

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return result.scalars().all()

    async def update_experiment(
        self, experiment_id: str, update_data: dict[str, Any]
    ) -> ABExperiment | None:
        """Update experiment data."""
        async with await self._get_session() as session:
            query = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id
            )
            result = await session.execute(query)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return None

            for key, value in update_data.items():
                if hasattr(experiment, key):
                    setattr(experiment, key, value)

            await session.commit()
            await session.refresh(experiment)
            return experiment

    async def get_active_experiments(
        self, target_metric: str | None = None
    ) -> list[ABExperiment]:
        """Get currently running experiments."""
        async with await self._get_session() as session:
            query = select(ABExperiment).where(ABExperiment.status == "running")

            if target_metric:
                query = query.where(ABExperiment.target_metric == target_metric)

            result = await session.execute(query)
            return result.scalars().all()

    async def complete_experiment(
        self, experiment_id: str, results: dict[str, Any]
    ) -> ABExperiment | None:
        """Mark experiment as completed with results."""
        async with await self._get_session() as session:
            query = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id
            )
            result = await session.execute(query)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return None

            experiment.status = "completed"
            experiment.completed_at = aware_utc_now()
            experiment.results = results

            await session.commit()
            await session.refresh(experiment)
            return experiment

    # Discovered Pattern Management
    async def create_pattern(self, pattern_data: dict[str, Any]) -> DiscoveredPattern:
        """Store a discovered pattern."""
        async with await self._get_session() as session:
            pattern = DiscoveredPattern(**pattern_data)
            session.add(pattern)
            await session.commit()
            await session.refresh(pattern)
            return pattern

    async def get_pattern(self, pattern_id: str) -> DiscoveredPattern | None:
        """Get pattern by ID."""
        async with await self._get_session() as session:
            query = select(DiscoveredPattern).where(
                DiscoveredPattern.pattern_id == pattern_id
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def get_patterns(
        self,
        filters: PatternFilter | None = None,
        sort_by: str = "avg_effectiveness",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DiscoveredPattern]:
        """Get patterns with filtering."""
        async with await self._get_session() as session:
            query = select(DiscoveredPattern)

            if filters:
                conditions = []
                if filters.pattern_type:
                    conditions.append(
                        DiscoveredPattern.pattern_type == filters.pattern_type
                    )
                if filters.min_effectiveness is not None:
                    conditions.append(
                        DiscoveredPattern.avg_effectiveness >= filters.min_effectiveness
                    )
                if filters.min_support_count is not None:
                    conditions.append(
                        DiscoveredPattern.support_count >= filters.min_support_count
                    )
                if filters.discovery_run_id:
                    conditions.append(
                        DiscoveredPattern.discovery_run_id == filters.discovery_run_id
                    )
                if filters.date_from:
                    conditions.append(DiscoveredPattern.created_at >= filters.date_from)
                if filters.date_to:
                    conditions.append(DiscoveredPattern.created_at <= filters.date_to)

                if conditions:
                    query = query.where(and_(*conditions))

            # Apply sorting
            sort_column = getattr(
                DiscoveredPattern, sort_by, DiscoveredPattern.avg_effectiveness
            )
            if sort_desc:
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(sort_column)

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return result.scalars().all()

    async def get_effective_patterns(
        self, min_effectiveness: float = 0.7, min_support: int = 5, limit: int = 20
    ) -> list[DiscoveredPattern]:
        """Get most effective discovered patterns."""
        async with await self._get_session() as session:
            query = (
                select(DiscoveredPattern)
                .where(
                    and_(
                        DiscoveredPattern.avg_effectiveness >= min_effectiveness,
                        DiscoveredPattern.support_count >= min_support,
                    )
                )
                .order_by(desc(DiscoveredPattern.avg_effectiveness))
                .limit(limit)
            )
            result = await session.execute(query)
            return result.scalars().all()

    async def update_pattern(
        self, pattern_id: str, update_data: dict[str, Any]
    ) -> DiscoveredPattern | None:
        """Update pattern data."""
        async with await self._get_session() as session:
            query = select(DiscoveredPattern).where(
                DiscoveredPattern.pattern_id == pattern_id
            )
            result = await session.execute(query)
            pattern = result.scalar_one_or_none()

            if not pattern:
                return None

            for key, value in update_data.items():
                if hasattr(pattern, key):
                    setattr(pattern, key, value)

            pattern.updated_at = aware_utc_now()
            await session.commit()
            await session.refresh(pattern)
            return pattern

    # Cross-Domain Analytics for Optimization
    async def get_optimization_training_data(
        self,
        rule_ids: list[str] | None = None,
        min_samples: int = 20,
        lookback_days: int = 30,
        include_synthetic: bool = True,
        synthetic_ratio: float = 0.3,
    ) -> OptimizationData:
        """Get training data for ML optimization."""
        async with await self._get_session() as session:
            cutoff_date = aware_utc_now() - timedelta(days=lookback_days)

            # Build query for rule performance data
            query = (
                select(RulePerformance, RuleMetadata.default_parameters)
                .join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id)
                .where(RulePerformance.created_at >= cutoff_date)
            )

            if rule_ids:
                query = query.where(RulePerformance.rule_id.in_(rule_ids))

            result = await session.execute(query)
            performance_data = result.fetchall()

            if len(performance_data) < min_samples:
                logger.warning(
                    f"Insufficient real data: {len(performance_data)} samples, need {min_samples}"
                )
                # In a real implementation, we would generate synthetic data here
                # For now, return minimal structure
                return OptimizationData(
                    features=[],
                    effectiveness_scores=[],
                    rule_ids=rule_ids,
                    metadata={
                        "total_samples": len(performance_data),
                        "real_samples": len(performance_data),
                        "synthetic_samples": 0,
                        "synthetic_ratio": 0.0,
                    },
                )

            # Extract features and scores
            features = []
            scores = []

            for row in performance_data:
                rule_perf = row.RulePerformance
                params = row.default_parameters or {}

                # Extract numeric features
                feature_vector = [
                    rule_perf.improvement_score or 0.0,
                    rule_perf.execution_time_ms or 0.0,
                    params.get("weight", 1.0),
                    params.get("priority", 5),
                    len(params),
                    1.0 if params.get("active", True) else 0.0,
                ]

                features.append(feature_vector)
                scores.append(rule_perf.confidence_level or 0.8)

            return OptimizationData(
                features=features,
                effectiveness_scores=scores,
                rule_ids=rule_ids,
                metadata={
                    "total_samples": len(features),
                    "real_samples": len(features),
                    "synthetic_samples": 0,
                    "synthetic_ratio": 0.0,
                },
            )

    async def get_rule_performance_with_metadata(
        self,
        rule_ids: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        include_parameters: bool = True,
    ) -> list[dict[str, Any]]:
        """Get rule performance data with metadata for optimization."""
        async with await self._get_session() as session:
            query = select(RulePerformance, RuleMetadata).join(
                RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id
            )

            conditions = []
            if rule_ids:
                conditions.append(RulePerformance.rule_id.in_(rule_ids))
            if date_from:
                conditions.append(RulePerformance.created_at >= date_from)
            if date_to:
                conditions.append(RulePerformance.created_at <= date_to)

            if conditions:
                query = query.where(and_(*conditions))

            result = await session.execute(query)
            performance_data = result.fetchall()

            results = []
            for row in performance_data:
                rule_perf = row.RulePerformance
                rule_meta = row.RuleMetadata

                data = {
                    "rule_id": rule_perf.rule_id,
                    "rule_name": rule_perf.rule_name,
                    "improvement_score": rule_perf.improvement_score,
                    "confidence_level": rule_perf.confidence_level,
                    "execution_time_ms": rule_perf.execution_time_ms,
                    "created_at": rule_perf.created_at,
                    "rule_enabled": rule_meta.enabled,
                    "rule_priority": rule_meta.priority,
                }

                if include_parameters:
                    data["rule_parameters"] = rule_perf.rule_parameters
                    data["default_parameters"] = rule_meta.default_parameters
                    data["prompt_characteristics"] = rule_perf.prompt_characteristics

                results.append(data)

            return results

    async def analyze_session_effectiveness(
        self,
        session_ids: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze overall session effectiveness for optimization."""
        async with await self._get_session() as session:
            query = select(ImprovementSession)

            conditions = []
            if session_ids:
                conditions.append(ImprovementSession.session_id.in_(session_ids))
            if date_from:
                conditions.append(ImprovementSession.created_at >= date_from)
            if date_to:
                conditions.append(ImprovementSession.created_at <= date_to)

            if conditions:
                query = query.where(and_(*conditions))

            result = await session.execute(query)
            sessions = result.scalars().all()

            if not sessions:
                return {"total_sessions": 0, "analysis": "No sessions found"}

            # Calculate basic statistics
            improvement_scores = [
                s.improvement_score for s in sessions if s.improvement_score
            ]
            confidence_scores = [
                s.confidence_level for s in sessions if s.confidence_level
            ]

            analysis = {
                "total_sessions": len(sessions),
                "sessions_with_improvement_score": len(improvement_scores),
                "avg_improvement_score": sum(improvement_scores)
                / len(improvement_scores)
                if improvement_scores
                else 0,
                "avg_confidence_level": sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0,
                "sessions_with_user_context": sum(
                    1 for s in sessions if s.user_context
                ),
            }

            return analysis

    async def get_pattern_experiment_candidates(
        self,
        min_effectiveness: float = 0.75,
        max_existing_experiments: int = 3,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get patterns that are candidates for A/B experiments."""
        async with await self._get_session() as session:
            # Get effective patterns
            pattern_query = (
                select(DiscoveredPattern)
                .where(DiscoveredPattern.avg_effectiveness >= min_effectiveness)
                .order_by(desc(DiscoveredPattern.avg_effectiveness))
            )

            pattern_result = await session.execute(pattern_query)
            patterns = pattern_result.scalars().all()

            candidates = []
            for pattern in patterns:
                # Check how many experiments already exist for this pattern
                experiment_query = select(func.count(ABExperiment.id)).where(
                    ABExperiment.experiment_name.like(f"%{pattern.pattern_id}%")
                )
                experiment_result = await session.execute(experiment_query)
                experiment_count = experiment_result.scalar() or 0

                if experiment_count < max_existing_experiments:
                    candidates.append({
                        "pattern_id": pattern.pattern_id,
                        "avg_effectiveness": pattern.avg_effectiveness,
                        "support_count": pattern.support_count,
                        "existing_experiments": experiment_count,
                        "parameters": pattern.parameters,
                    })

                if len(candidates) >= limit:
                    break

            return candidates

    # Cross-Domain Queries for Prompt Improvement Service
    async def get_sessions_with_feedback_and_performance(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_rating: int = 3,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get sessions with both user feedback and rule performance data."""
        async with await self._get_session() as session:
            # This would be a complex query joining multiple tables
            # For now, return a placeholder implementation
            query = (
                select(ImprovementSession, UserFeedback)
                .join(
                    UserFeedback,
                    ImprovementSession.session_id == UserFeedback.session_id,
                )
                .where(UserFeedback.rating >= min_rating)
            )

            conditions = []
            if date_from:
                conditions.append(ImprovementSession.created_at >= date_from)
            if date_to:
                conditions.append(ImprovementSession.created_at <= date_to)

            if conditions:
                query = query.where(and_(*conditions))

            query = query.limit(limit)
            result = await session.execute(query)

            results = []
            for row in result:
                session_data = row.ImprovementSession
                feedback_data = row.UserFeedback

                results.append({
                    "session_id": session_data.session_id,
                    "improvement_score": session_data.improvement_score,
                    "confidence_level": session_data.confidence_level,
                    "user_rating": feedback_data.rating,
                    "feedback_text": feedback_data.feedback_text,
                    "created_at": session_data.created_at,
                })

            return results

    async def get_rule_effectiveness_by_session_context(
        self,
        context_filters: dict[str, Any] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Analyze rule effectiveness by session context."""
        # This would require complex analysis of user_context JSONB fields
        # Return placeholder for now
        return {
            "analysis_note": "Rule effectiveness by context analysis would require complex JSONB queries",
            "context_filters": context_filters,
            "date_range": {"from": date_from, "to": date_to},
        }

    # Bulk Operations and Maintenance
    async def archive_old_sessions(
        self, days_old: int = 90, keep_successful_sessions: bool = True
    ) -> int:
        """Archive old sessions."""
        # Implementation would move old sessions to archive tables
        cutoff_date = aware_utc_now() - timedelta(days=days_old)

        async with await self._get_session() as session:
            query = select(func.count(ImprovementSession.id)).where(
                ImprovementSession.created_at < cutoff_date
            )

            if keep_successful_sessions:
                query = query.where(
                    or_(
                        ImprovementSession.improvement_score < 0.7,
                        ImprovementSession.improvement_score.is_(None),
                    )
                )

            result = await session.execute(query)
            count = result.scalar() or 0

            logger.info(f"Would archive {count} sessions older than {days_old} days")
            return count

    async def cleanup_incomplete_sessions(self, hours_old: int = 24) -> int:
        """Clean up incomplete/orphaned sessions."""
        cutoff_time = aware_utc_now() - timedelta(hours=hours_old)

        async with await self._get_session() as session:
            # Find sessions without improvement scores that are old
            query = select(func.count(ImprovementSession.id)).where(
                and_(
                    ImprovementSession.created_at < cutoff_time,
                    ImprovementSession.improvement_score.is_(None),
                )
            )

            result = await session.execute(query)
            count = result.scalar() or 0

            logger.info(
                f"Would cleanup {count} incomplete sessions older than {hours_old} hours"
            )
            return count

    async def batch_update_session_scores(
        self, score_updates: dict[str, dict[str, float]]
    ) -> int:
        """Batch update session quality/improvement scores."""
        if not score_updates:
            return 0

        updated_count = 0
        async with await self._get_session() as session:
            for session_id, scores in score_updates.items():
                query = select(ImprovementSession).where(
                    ImprovementSession.session_id == session_id
                )
                result = await session.execute(query)
                db_session = result.scalar_one_or_none()

                if db_session:
                    for score_type, value in scores.items():
                        if hasattr(db_session, score_type):
                            setattr(db_session, score_type, value)
                    db_session.updated_at = aware_utc_now()
                    updated_count += 1

            await session.commit()

        return updated_count

    # Health and Diagnostics
    async def health_check(self) -> dict[str, Any]:
        """Perform repository health check."""
        try:
            async with await self._get_session() as session:
                # Test basic connectivity
                result = await session.execute(
                    select(func.count(ImprovementSession.id))
                )
                session_count = result.scalar()

                return {
                    "status": "healthy",
                    "session_count": session_count,
                    "timestamp": aware_utc_now().isoformat(),
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": aware_utc_now().isoformat(),
            }

    async def get_connection_info(self) -> dict[str, Any]:
        """Get database connection information."""
        return {
            "repository_type": "PromptRepository",
            "session_factory": str(type(self._session_factory)),
            "supported_operations": [
                "session_management",
                "experiment_management",
                "pattern_management",
                "cross_domain_analytics",
            ],
        }
