"""ML Repository Facade providing unified access to specialized ML repositories.

Implements the original MLRepositoryProtocol interface while delegating to specialized
repositories following the facade pattern and maintaining backwards compatibility.
"""

import logging
from datetime import datetime
from typing import Any

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    GenerationBatch,
    GenerationMethodPerformance,
    GenerationSession,
    MLModelPerformance,
    SyntheticDataSample,
    TrainingIteration,
    TrainingPrompt,
    TrainingSession,
    TrainingSessionCreate,
    TrainingSessionUpdate,
)
from prompt_improver.repositories.impl.ml_repository_service.experiment_repository import (
    ExperimentRepository,
)
from prompt_improver.repositories.impl.ml_repository_service.inference_repository import (
    InferenceRepository,
)
from prompt_improver.repositories.impl.ml_repository_service.metrics_repository import (
    MetricsRepository,
)
from prompt_improver.repositories.impl.ml_repository_service.model_repository import (
    ModelRepository,
)
from prompt_improver.repositories.impl.ml_repository_service.training_repository import (
    TrainingRepository,
)
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    GenerationSessionFilter,
    MLRepositoryProtocol,
    ModelPerformanceFilter,
    ModelVersionInfo,
    SyntheticDataMetrics,
    TrainingMetrics,
    TrainingSessionFilter,
)

logger = logging.getLogger(__name__)


class MLRepositoryFacade(MLRepositoryProtocol):
    """Unified ML repository facade coordinating specialized repositories."""

    def __init__(self, connection_manager: DatabaseServices) -> None:
        self.connection_manager = connection_manager

        # Initialize specialized repositories
        self.model_repository = ModelRepository(connection_manager)
        self.training_repository = TrainingRepository(connection_manager)
        self.metrics_repository = MetricsRepository(connection_manager)
        self.experiment_repository = ExperimentRepository(connection_manager)
        self.inference_repository = InferenceRepository(connection_manager)

        logger.info("ML Repository Facade initialized with specialized repositories")

    # Training Session Management - Delegate to TrainingRepository

    async def create_training_session(
        self, session_data: TrainingSessionCreate
    ) -> TrainingSession:
        """Create a new training session."""
        return await self.training_repository.create_training_session(session_data)

    async def get_training_sessions(
        self,
        filters: TrainingSessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TrainingSession]:
        """Retrieve training sessions with filtering."""
        return await self.training_repository.get_training_sessions(
            filters, sort_by, sort_desc, limit, offset
        )

    async def get_training_session_by_id(
        self, session_id: str
    ) -> TrainingSession | None:
        """Get training session by ID."""
        return await self.training_repository.get_training_session_by_id(session_id)

    async def update_training_session(
        self,
        session_id: str,
        update_data: TrainingSessionUpdate,
    ) -> TrainingSession | None:
        """Update training session."""
        return await self.training_repository.update_training_session(
            session_id, update_data
        )

    async def get_active_training_sessions(self) -> list[TrainingSession]:
        """Get all currently active training sessions."""
        return await self.training_repository.get_active_training_sessions()

    async def get_training_session_metrics(
        self,
        session_id: str,
    ) -> TrainingMetrics | None:
        """Get comprehensive metrics for training session."""
        return await self.training_repository.get_training_session_metrics(session_id)

    # Training Iteration Management - Delegate to TrainingRepository

    async def create_training_iteration(
        self,
        iteration_data: dict[str, Any],
    ) -> TrainingIteration:
        """Create training iteration record."""
        return await self.training_repository.create_training_iteration(iteration_data)

    async def get_training_iterations(
        self,
        session_id: str,
        start_iteration: int | None = None,
        end_iteration: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TrainingIteration]:
        """Get training iterations for session."""
        return await self.training_repository.get_training_iterations(
            session_id, start_iteration, end_iteration, limit, offset
        )

    async def get_latest_iteration(
        self,
        session_id: str,
    ) -> TrainingIteration | None:
        """Get latest iteration for training session."""
        return await self.training_repository.get_latest_iteration(session_id)

    async def get_iteration_performance_trend(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Get performance trend across iterations."""
        return await self.training_repository.get_iteration_performance_trend(session_id)

    # Model Performance Management - Delegate to ModelRepository

    async def create_model_performance(
        self,
        performance_data: dict[str, Any],
    ) -> MLModelPerformance:
        """Record model performance metrics."""
        return await self.model_repository.create_model_performance(performance_data)

    async def get_model_performances(
        self,
        filters: ModelPerformanceFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MLModelPerformance]:
        """Get model performance records."""
        return await self.model_repository.get_model_performances(
            filters, sort_by, sort_desc, limit, offset
        )

    async def get_model_performance_by_id(
        self,
        model_id: str,
    ) -> list[MLModelPerformance]:
        """Get performance history for specific model."""
        return await self.model_repository.get_model_performance_by_id(model_id)

    async def get_best_performing_models(
        self,
        metric: str = "accuracy",
        model_type: str | None = None,
        limit: int = 10,
    ) -> list[MLModelPerformance]:
        """Get top performing models by metric."""
        return await self.model_repository.get_best_performing_models(
            metric, model_type, limit
        )

    async def get_model_version_history(
        self,
        model_id: str,
    ) -> list[ModelVersionInfo]:
        """Get version history for model."""
        return await self.model_repository.get_model_version_history(model_id)

    # Training Data Management - Delegate to TrainingRepository (core methods)

    async def create_training_prompt(
        self,
        prompt_data: dict[str, Any],
    ) -> TrainingPrompt:
        """Create training prompt record."""
        # This method would need to be implemented in TrainingRepository
        # For now, we'll implement it directly here
        from prompt_improver.repositories.base_repository import BaseRepository

        base_repo = BaseRepository(
            model_class=TrainingPrompt,
            connection_manager=self.connection_manager,
        )

        async with base_repo.get_session() as session:
            try:
                prompt = TrainingPrompt(**prompt_data)
                session.add(prompt)
                await session.commit()
                await session.refresh(prompt)
                logger.info(f"Created training prompt {prompt.id}")
                return prompt
            except Exception as e:
                logger.exception(f"Error creating training prompt: {e}")
                raise

    async def get_training_prompts(
        self,
        data_source: str | None = None,
        is_active: bool = True,
        min_priority: int | None = None,
        session_id: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[TrainingPrompt]:
        """Get training prompts with filters."""
        # Implement basic training prompts retrieval
        from sqlalchemy import and_, desc, select

        from prompt_improver.repositories.base_repository import BaseRepository

        base_repo = BaseRepository(
            model_class=TrainingPrompt,
            connection_manager=self.connection_manager,
        )

        async with base_repo.get_session() as session:
            try:
                query = select(TrainingPrompt)
                conditions = []

                if data_source:
                    conditions.append(TrainingPrompt.data_source == data_source)
                if is_active:
                    conditions.append(TrainingPrompt.is_active == is_active)
                if min_priority is not None:
                    conditions.append(TrainingPrompt.priority >= min_priority)
                if session_id:
                    conditions.append(TrainingPrompt.session_id == session_id)

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(desc(TrainingPrompt.priority))
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.exception(f"Error getting training prompts: {e}")
                raise

    async def update_training_prompt(
        self,
        prompt_id: int,
        update_data: dict[str, Any],
    ) -> TrainingPrompt | None:
        """Update training prompt."""
        # Implement basic training prompt update
        from sqlalchemy import select, update

        from prompt_improver.repositories.base_repository import BaseRepository

        base_repo = BaseRepository(
            model_class=TrainingPrompt,
            connection_manager=self.connection_manager,
        )

        async with base_repo.get_session() as session:
            try:
                query = (
                    update(TrainingPrompt)
                    .where(TrainingPrompt.id == prompt_id)
                    .values(**update_data)
                )
                result = await session.execute(query)

                if result.rowcount == 0:
                    return None

                await session.commit()

                # Get updated prompt
                get_query = select(TrainingPrompt).where(TrainingPrompt.id == prompt_id)
                get_result = await session.execute(get_query)
                return get_result.scalar_one_or_none()
            except Exception as e:
                logger.exception(f"Error updating training prompt: {e}")
                raise

    async def deactivate_training_prompts(
        self,
        prompt_ids: list[int],
    ) -> int:
        """Deactivate training prompts, returns count updated."""
        from sqlalchemy import update

        from prompt_improver.repositories.base_repository import BaseRepository

        base_repo = BaseRepository(
            model_class=TrainingPrompt,
            connection_manager=self.connection_manager,
        )

        async with base_repo.get_session() as session:
            try:
                query = (
                    update(TrainingPrompt)
                    .where(TrainingPrompt.id.in_(prompt_ids))
                    .values(is_active=False)
                )
                result = await session.execute(query)
                await session.commit()
                logger.info(f"Deactivated {result.rowcount} training prompts")
                return result.rowcount
            except Exception as e:
                logger.exception(f"Error deactivating training prompts: {e}")
                raise

    # Generation Session Management - Delegate to ExperimentRepository

    async def create_generation_session(
        self,
        session_data: dict[str, Any],
    ) -> GenerationSession:
        """Create synthetic data generation session."""
        return await self.experiment_repository.create_generation_session(session_data)

    async def get_generation_sessions(
        self,
        filters: GenerationSessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GenerationSession]:
        """Get generation sessions with filters."""
        return await self.experiment_repository.get_generation_sessions(
            filters, sort_by, sort_desc, limit, offset
        )

    async def get_generation_session_by_id(
        self,
        session_id: str,
    ) -> GenerationSession | None:
        """Get generation session by ID."""
        return await self.experiment_repository.get_generation_session_by_id(session_id)

    async def update_generation_session(
        self,
        session_id: str,
        update_data: dict[str, Any],
    ) -> GenerationSession | None:
        """Update generation session."""
        return await self.experiment_repository.update_generation_session(
            session_id, update_data
        )

    # Generation Batch Management - Delegate to ExperimentRepository

    async def create_generation_batch(
        self,
        batch_data: dict[str, Any],
    ) -> GenerationBatch:
        """Create generation batch record."""
        return await self.experiment_repository.create_generation_batch(batch_data)

    async def get_generation_batches(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GenerationBatch]:
        """Get generation batches for session."""
        return await self.experiment_repository.get_generation_batches(
            session_id, limit, offset
        )

    async def update_generation_batch(
        self,
        batch_id: str,
        update_data: dict[str, Any],
    ) -> GenerationBatch | None:
        """Update generation batch."""
        return await self.experiment_repository.update_generation_batch(
            batch_id, update_data
        )

    # Synthetic Data Sample Management - Delegate to InferenceRepository

    async def create_synthetic_data_samples(
        self,
        samples_data: list[dict[str, Any]],
    ) -> list[SyntheticDataSample]:
        """Create multiple synthetic data samples."""
        return await self.inference_repository.create_synthetic_data_samples(samples_data)

    async def get_synthetic_data_samples(
        self,
        session_id: str | None = None,
        batch_id: str | None = None,
        min_quality_score: float | None = None,
        domain_category: str | None = None,
        status: str = "active",
        limit: int = 1000,
        offset: int = 0,
    ) -> list[SyntheticDataSample]:
        """Get synthetic data samples with filters."""
        return await self.inference_repository.get_synthetic_data_samples(
            session_id, batch_id, min_quality_score, domain_category, status, limit, offset
        )

    async def update_synthetic_data_sample(
        self,
        sample_id: str,
        update_data: dict[str, Any],
    ) -> SyntheticDataSample | None:
        """Update synthetic data sample."""
        return await self.inference_repository.update_synthetic_data_sample(
            sample_id, update_data
        )

    async def archive_synthetic_samples(
        self,
        sample_ids: list[str],
    ) -> int:
        """Archive synthetic data samples, returns count updated."""
        return await self.inference_repository.archive_synthetic_samples(sample_ids)

    # Analytics and Insights - Delegate to MetricsRepository

    async def get_synthetic_data_metrics(
        self,
        session_id: str,
    ) -> SyntheticDataMetrics | None:
        """Get comprehensive synthetic data metrics."""
        return await self.metrics_repository.get_synthetic_data_metrics(session_id)

    async def get_training_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Get training analytics for date range."""
        return await self.metrics_repository.get_training_analytics(start_date, end_date)

    async def get_generation_method_performance(
        self,
        session_id: str | None = None,
        method_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[GenerationMethodPerformance]:
        """Get generation method performance data."""
        return await self.experiment_repository.get_generation_method_performance(
            session_id, method_name, date_from, date_to
        )

    # Cleanup and Maintenance - Delegate to appropriate repositories

    async def cleanup_old_iterations(
        self,
        session_id: str,
        keep_latest: int = 100,
    ) -> int:
        """Clean up old iterations, keeping latest N."""
        return await self.training_repository.cleanup_old_iterations(session_id, keep_latest)

    async def cleanup_failed_sessions(
        self,
        days_old: int = 7,
    ) -> int:
        """Clean up failed training sessions older than specified days."""
        return await self.training_repository.cleanup_failed_sessions(days_old)

    async def archive_completed_sessions(
        self,
        days_old: int = 30,
    ) -> int:
        """Archive completed sessions older than specified days."""
        return await self.training_repository.archive_completed_sessions(days_old)

    # Intelligence Processing - Delegate to InferenceRepository

    async def get_prompt_characteristics_batch(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get batch of prompt characteristics for ML processing."""
        return await self.inference_repository.get_prompt_characteristics_batch(batch_size)

    async def get_rule_performance_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule performance data for intelligence processing."""
        return await self.metrics_repository.get_rule_performance_data(batch_size)

    async def cache_rule_intelligence(
        self, intelligence_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule intelligence results with upsert logic."""
        return await self.inference_repository.cache_rule_intelligence(intelligence_data)

    async def get_rule_combinations_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule combination data for analysis."""
        return await self.metrics_repository.get_rule_combinations_data(batch_size)

    async def cache_combination_intelligence(
        self, combination_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule combination intelligence results."""
        return await self.inference_repository.cache_combination_intelligence(combination_data)

    async def cache_pattern_discovery(
        self, pattern_data: dict[str, Any]
    ) -> None:
        """Cache pattern discovery results."""
        return await self.inference_repository.cache_pattern_discovery(pattern_data)

    async def cleanup_expired_cache(self) -> dict[str, Any]:
        """Clean up expired intelligence cache entries."""
        return await self.inference_repository.cleanup_expired_cache()

    async def process_ml_predictions_batch(
        self, batch_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process ML predictions for batch of data."""
        return await self.inference_repository.process_ml_predictions_batch(batch_data)

    # Additional facade-specific methods for coordination

    async def get_repository_health_status(self) -> dict[str, Any]:
        """Get health status of all specialized repositories."""
        try:
            # Test basic connectivity to each repository
            health_status = {
                "facade_status": "healthy",
                "repositories": {},
                "total_repositories": 5,
                "healthy_repositories": 0,
            }

            # Test model repository
            try:
                await self.model_repository.get_best_performing_models(limit=1)
                health_status["repositories"]["model_repository"] = "healthy"
                health_status["healthy_repositories"] += 1
            except Exception as e:
                health_status["repositories"]["model_repository"] = f"unhealthy: {e!s}"

            # Test training repository
            try:
                await self.training_repository.get_active_training_sessions()
                health_status["repositories"]["training_repository"] = "healthy"
                health_status["healthy_repositories"] += 1
            except Exception as e:
                health_status["repositories"]["training_repository"] = f"unhealthy: {e!s}"

            # Test metrics repository
            try:
                await self.metrics_repository.get_rule_performance_data(batch_size=1)
                health_status["repositories"]["metrics_repository"] = "healthy"
                health_status["healthy_repositories"] += 1
            except Exception as e:
                health_status["repositories"]["metrics_repository"] = f"unhealthy: {e!s}"

            # Test experiment repository
            try:
                await self.experiment_repository.get_active_generation_sessions()
                health_status["repositories"]["experiment_repository"] = "healthy"
                health_status["healthy_repositories"] += 1
            except Exception as e:
                health_status["repositories"]["experiment_repository"] = f"unhealthy: {e!s}"

            # Test inference repository
            try:
                await self.inference_repository.cleanup_expired_cache()
                health_status["repositories"]["inference_repository"] = "healthy"
                health_status["healthy_repositories"] += 1
            except Exception as e:
                health_status["repositories"]["inference_repository"] = f"unhealthy: {e!s}"

            # Update overall status
            if health_status["healthy_repositories"] == health_status["total_repositories"]:
                health_status["facade_status"] = "healthy"
            elif health_status["healthy_repositories"] > 0:
                health_status["facade_status"] = "degraded"
            else:
                health_status["facade_status"] = "unhealthy"

            return health_status

        except Exception as e:
            logger.exception(f"Error checking repository health status: {e}")
            return {
                "facade_status": "unhealthy",
                "error": str(e),
                "repositories": {},
                "total_repositories": 5,
                "healthy_repositories": 0,
            }
