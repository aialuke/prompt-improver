"""Model repository implementation for model performance and versioning.

Handles model performance tracking, versioning, comparison, and deployment status
following repository pattern with protocol-based dependency injection.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import MLModelPerformance
from prompt_improver.repositories.base_repository import BaseRepository
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    ModelPerformanceFilter,
    ModelVersionInfo,
)

logger = logging.getLogger(__name__)


class ModelRepository(BaseRepository[MLModelPerformance]):
    """Repository for model performance and versioning management."""

    def __init__(self, connection_manager: DatabaseServices):
        super().__init__(
            model_class=MLModelPerformance,
            connection_manager=connection_manager,
        )
        self.connection_manager = connection_manager
        logger.info("Model repository initialized")

    # Model Performance Management

    async def create_model_performance(
        self,
        performance_data: dict[str, Any],
    ) -> MLModelPerformance:
        """Record model performance metrics."""
        async with self.get_session() as session:
            try:
                performance = MLModelPerformance(**performance_data)
                session.add(performance)
                await session.commit()
                await session.refresh(performance)
                logger.info(f"Created model performance record {performance.id}")
                return performance
            except Exception as e:
                logger.error(f"Error creating model performance: {e}")
                raise

    async def get_model_performances(
        self,
        filters: ModelPerformanceFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MLModelPerformance]:
        """Get model performance records."""
        async with self.get_session() as session:
            try:
                query = select(MLModelPerformance)

                if filters:
                    conditions = []
                    if filters.model_type:
                        conditions.append(
                            MLModelPerformance.model_type == filters.model_type
                        )
                    if filters.min_accuracy is not None:
                        conditions.append(
                            MLModelPerformance.accuracy >= filters.min_accuracy
                        )
                    if filters.min_precision is not None:
                        conditions.append(
                            MLModelPerformance.precision >= filters.min_precision
                        )
                    if filters.min_recall is not None:
                        conditions.append(
                            MLModelPerformance.recall >= filters.min_recall
                        )
                    if filters.min_training_samples is not None:
                        conditions.append(
                            MLModelPerformance.training_samples
                            >= filters.min_training_samples
                        )
                    if filters.date_from:
                        conditions.append(
                            MLModelPerformance.created_at >= filters.date_from
                        )
                    if filters.date_to:
                        conditions.append(
                            MLModelPerformance.created_at <= filters.date_to
                        )

                    if conditions:
                        query = query.where(and_(*conditions))

                # Apply sorting
                if hasattr(MLModelPerformance, sort_by):
                    sort_field = getattr(MLModelPerformance, sort_by)
                    if sort_desc:
                        query = query.order_by(desc(sort_field))
                    else:
                        query = query.order_by(sort_field)

                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting model performances: {e}")
                raise

    async def get_model_performance_by_id(
        self,
        model_id: str,
    ) -> list[MLModelPerformance]:
        """Get performance history for specific model."""
        async with self.get_session() as session:
            try:
                query = (
                    select(MLModelPerformance)
                    .where(MLModelPerformance.model_id == model_id)
                    .order_by(desc(MLModelPerformance.created_at))
                )
                result = await session.execute(query)
                return list(result.scalars().all())
            except Exception as e:
                logger.error(f"Error getting model performance by ID: {e}")
                raise

    async def get_best_performing_models(
        self,
        metric: str = "accuracy",
        model_type: str | None = None,
        limit: int = 10,
    ) -> list[MLModelPerformance]:
        """Get top performing models by metric."""
        async with self.get_session() as session:
            try:
                query = select(MLModelPerformance)

                if model_type:
                    query = query.where(MLModelPerformance.model_type == model_type)

                # Apply metric-based sorting
                if hasattr(MLModelPerformance, metric):
                    metric_field = getattr(MLModelPerformance, metric)
                    query = query.where(metric_field.is_not(None))
                    query = query.order_by(desc(metric_field))

                query = query.limit(limit)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting best performing models: {e}")
                raise

    async def get_latest_model_version(
        self,
        model_type: str,
    ) -> str | None:
        """Get latest model version for a type."""
        async with self.get_session() as session:
            try:
                query = (
                    select(MLModelPerformance.model_version)
                    .where(MLModelPerformance.model_type == model_type)
                    .order_by(desc(MLModelPerformance.created_at))
                    .limit(1)
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting latest model version: {e}")
                raise

    async def get_model_versions(
        self,
        model_type: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get model versions for a type."""
        async with self.get_session() as session:
            try:
                query = (
                    select(MLModelPerformance)
                    .where(MLModelPerformance.model_type == model_type)
                    .order_by(desc(MLModelPerformance.created_at))
                    .limit(limit)
                )
                result = await session.execute(query)
                performances = result.scalars().all()

                versions = []
                for perf in performances:
                    version_info = {
                        "model_id": perf.model_id,
                        "version": perf.model_version or "v1.0",
                        "performance_metrics": {
                            "accuracy": perf.accuracy or 0.0,
                            "precision": perf.precision or 0.0,
                            "recall": perf.recall or 0.0,
                            "f1_score": perf.f1_score or 0.0,
                        },
                        "training_samples": perf.training_samples,
                        "created_at": perf.created_at,
                        "metadata": perf.metadata or {},
                    }
                    versions.append(version_info)

                return versions

            except Exception as e:
                logger.error(f"Error getting model versions: {e}")
                raise

    async def compare_model_performance(
        self,
        model_id1: str,
        model_id2: str,
    ) -> dict[str, Any]:
        """Compare performance of two models."""
        async with self.get_session() as session:
            try:
                # Get latest performance for each model
                query1 = (
                    select(MLModelPerformance)
                    .where(MLModelPerformance.model_id == model_id1)
                    .order_by(desc(MLModelPerformance.created_at))
                    .limit(1)
                )
                query2 = (
                    select(MLModelPerformance)
                    .where(MLModelPerformance.model_id == model_id2)
                    .order_by(desc(MLModelPerformance.created_at))
                    .limit(1)
                )

                result1 = await session.execute(query1)
                result2 = await session.execute(query2)

                model1 = result1.scalar_one_or_none()
                model2 = result2.scalar_one_or_none()

                if not model1 or not model2:
                    return {"error": "One or both models not found"}

                metrics = ["accuracy", "precision", "recall", "f1_score"]
                comparison = {
                    "model1": {
                        "id": model_id1,
                        "metrics": {
                            metric: getattr(model1, metric, 0.0) or 0.0
                            for metric in metrics
                        },
                    },
                    "model2": {
                        "id": model_id2,
                        "metrics": {
                            metric: getattr(model2, metric, 0.0) or 0.0
                            for metric in metrics
                        },
                    },
                    "differences": {},
                    "winner": {},
                }

                # Calculate differences and winners
                for metric in metrics:
                    val1 = getattr(model1, metric, 0.0) or 0.0
                    val2 = getattr(model2, metric, 0.0) or 0.0
                    diff = val1 - val2
                    comparison["differences"][metric] = diff
                    comparison["winner"][metric] = model_id1 if diff > 0 else model_id2

                return comparison

            except Exception as e:
                logger.error(f"Error comparing model performance: {e}")
                raise

    async def update_model_metrics(
        self,
        model_id: str,
        metrics: dict[str, float],
    ) -> bool:
        """Update model metrics."""
        async with self.get_session() as session:
            try:
                # Get latest performance record
                query = (
                    select(MLModelPerformance)
                    .where(MLModelPerformance.model_id == model_id)
                    .order_by(desc(MLModelPerformance.created_at))
                    .limit(1)
                )
                result = await session.execute(query)
                performance = result.scalar_one_or_none()

                if not performance:
                    return False

                # Update metrics
                update_query = (
                    update(MLModelPerformance)
                    .where(MLModelPerformance.id == performance.id)
                    .values(**metrics)
                )
                await session.execute(update_query)
                await session.commit()

                logger.info(f"Updated metrics for model {model_id}")
                return True

            except Exception as e:
                logger.error(f"Error updating model metrics: {e}")
                return False

    async def get_model_deployment_status(
        self,
        model_id: str,
    ) -> dict[str, Any]:
        """Get model deployment status."""
        async with self.get_session() as session:
            try:
                query = (
                    select(MLModelPerformance)
                    .where(MLModelPerformance.model_id == model_id)
                    .order_by(desc(MLModelPerformance.created_at))
                    .limit(1)
                )
                result = await session.execute(query)
                performance = result.scalar_one_or_none()

                if not performance:
                    return {"status": "not_found", "model_id": model_id}

                # Extract deployment status from metadata
                metadata = performance.metadata or {}
                deployment_status = metadata.get("deployment_status", "unknown")

                return {
                    "model_id": model_id,
                    "status": deployment_status,
                    "version": performance.model_version,
                    "last_updated": performance.created_at.isoformat(),
                    "performance_summary": {
                        "accuracy": performance.accuracy,
                        "precision": performance.precision,
                        "recall": performance.recall,
                        "f1_score": performance.f1_score,
                    },
                    "metadata": metadata,
                }

            except Exception as e:
                logger.error(f"Error getting model deployment status: {e}")
                return {"status": "error", "error": str(e)}

    async def get_model_version_history(
        self,
        model_id: str,
    ) -> list[ModelVersionInfo]:
        """Get version history for model."""
        try:
            performances = await self.get_model_performance_by_id(model_id)

            return [
                ModelVersionInfo(
                    model_id=model_id,
                    version=f"v{i + 1}.0",
                    performance_metrics={
                        "accuracy": perf.accuracy or 0,
                        "precision": perf.precision or 0,
                        "recall": perf.recall or 0,
                    },
                    deployment_status="active" if i == 0 else "retired",
                    created_at=perf.created_at,
                    metadata=perf.metadata or {},
                )
                for i, perf in enumerate(performances[:10])  # Latest 10 versions
            ]

        except Exception as e:
            logger.error(f"Error getting model version history: {e}")
            raise

    async def archive_old_models(
        self,
        days_old: int = 90,
    ) -> int:
        """Archive old model versions."""
        async with self.get_session() as session:
            try:
                from datetime import datetime, timedelta

                cutoff_date = datetime.now() - timedelta(days=days_old)

                # Update old models to archived status in metadata
                query = select(MLModelPerformance).where(
                    MLModelPerformance.created_at < cutoff_date
                )
                result = await session.execute(query)
                old_models = result.scalars().all()

                archived_count = 0
                for model in old_models:
                    metadata = model.metadata or {}
                    if metadata.get("status") != "archived":
                        metadata["status"] = "archived"
                        metadata["archived_at"] = datetime.now().isoformat()

                        update_query = (
                            update(MLModelPerformance)
                            .where(MLModelPerformance.id == model.id)
                            .values(metadata=metadata)
                        )
                        await session.execute(update_query)
                        archived_count += 1

                await session.commit()
                logger.info(f"Archived {archived_count} old model versions")
                return archived_count

            except Exception as e:
                logger.error(f"Error archiving old models: {e}")
                raise