"""Experiment repository implementation for generation sessions and A/B testing.

Handles synthetic data generation sessions, generation batches, method performance,
and experiment tracking following repository pattern with protocol-based dependency injection.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    GenerationBatch,
    GenerationMethodPerformance,
    GenerationSession,
)
from prompt_improver.repositories.base_repository import BaseRepository
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    GenerationSessionFilter,
)

logger = logging.getLogger(__name__)


class ExperimentRepository(BaseRepository[GenerationSession]):
    """Repository for experiment tracking and A/B testing operations."""

    def __init__(self, connection_manager: DatabaseServices):
        super().__init__(
            model_class=GenerationSession,
            connection_manager=connection_manager,
        )
        self.connection_manager = connection_manager
        logger.info("Experiment repository initialized")

    # Generation Session Management

    async def create_generation_session(
        self,
        session_data: dict[str, Any],
    ) -> GenerationSession:
        """Create synthetic data generation session."""
        async with self.get_session() as session:
            try:
                generation_session = GenerationSession(**session_data)
                session.add(generation_session)
                await session.commit()
                await session.refresh(generation_session)
                logger.info(f"Created generation session {generation_session.id}")
                return generation_session
            except Exception as e:
                logger.error(f"Error creating generation session: {e}")
                raise

    async def get_generation_sessions(
        self,
        filters: GenerationSessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GenerationSession]:
        """Get generation sessions with filters."""
        async with self.get_session() as session:
            try:
                query = select(GenerationSession)

                if filters:
                    conditions = []
                    if filters.session_type:
                        conditions.append(
                            GenerationSession.session_type == filters.session_type
                        )
                    if filters.generation_method:
                        conditions.append(
                            GenerationSession.generation_method
                            == filters.generation_method
                        )
                    if filters.status:
                        conditions.append(GenerationSession.status == filters.status)
                    if filters.min_quality_threshold is not None:
                        conditions.append(
                            GenerationSession.quality_threshold
                            >= filters.min_quality_threshold
                        )
                    if filters.training_session_id:
                        conditions.append(
                            GenerationSession.training_session_id
                            == filters.training_session_id
                        )
                    if filters.date_from:
                        conditions.append(
                            GenerationSession.created_at >= filters.date_from
                        )
                    if filters.date_to:
                        conditions.append(
                            GenerationSession.created_at <= filters.date_to
                        )

                    if conditions:
                        query = query.where(and_(*conditions))

                # Apply sorting
                if hasattr(GenerationSession, sort_by):
                    sort_field = getattr(GenerationSession, sort_by)
                    if sort_desc:
                        query = query.order_by(desc(sort_field))
                    else:
                        query = query.order_by(sort_field)

                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting generation sessions: {e}")
                raise

    async def get_generation_session_by_id(
        self,
        session_id: str,
    ) -> GenerationSession | None:
        """Get generation session by ID."""
        async with self.get_session() as session:
            try:
                query = select(GenerationSession).where(
                    GenerationSession.id == session_id
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting generation session by ID: {e}")
                raise

    async def update_generation_session(
        self,
        session_id: str,
        update_data: dict[str, Any],
    ) -> GenerationSession | None:
        """Update generation session."""
        async with self.get_session() as session:
            try:
                query = (
                    update(GenerationSession)
                    .where(GenerationSession.id == session_id)
                    .values(**update_data)
                )
                result = await session.execute(query)

                if result.rowcount == 0:
                    return None

                await session.commit()
                return await self.get_generation_session_by_id(session_id)
            except Exception as e:
                logger.error(f"Error updating generation session: {e}")
                raise

    async def get_active_generation_sessions(self) -> list[GenerationSession]:
        """Get all currently active generation sessions."""
        async with self.get_session() as session:
            try:
                query = select(GenerationSession).where(
                    GenerationSession.status.in_(["running", "paused"])
                )
                result = await session.execute(query)
                return list(result.scalars().all())
            except Exception as e:
                logger.error(f"Error getting active generation sessions: {e}")
                raise

    # Generation Batch Management

    async def create_generation_batch(
        self,
        batch_data: dict[str, Any],
    ) -> GenerationBatch:
        """Create generation batch record."""
        async with self.get_session() as session:
            try:
                batch = GenerationBatch(**batch_data)
                session.add(batch)
                await session.commit()
                await session.refresh(batch)
                logger.info(f"Created generation batch {batch.id}")
                return batch
            except Exception as e:
                logger.error(f"Error creating generation batch: {e}")
                raise

    async def get_generation_batches(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GenerationBatch]:
        """Get generation batches for session."""
        async with self.get_session() as session:
            try:
                query = (
                    select(GenerationBatch)
                    .where(GenerationBatch.session_id == session_id)
                    .order_by(desc(GenerationBatch.created_at))
                    .limit(limit)
                    .offset(offset)
                )
                result = await session.execute(query)
                return list(result.scalars().all())
            except Exception as e:
                logger.error(f"Error getting generation batches: {e}")
                raise

    async def update_generation_batch(
        self,
        batch_id: str,
        update_data: dict[str, Any],
    ) -> GenerationBatch | None:
        """Update generation batch."""
        async with self.get_session() as session:
            try:
                query = (
                    update(GenerationBatch)
                    .where(GenerationBatch.id == batch_id)
                    .values(**update_data)
                )
                result = await session.execute(query)

                if result.rowcount == 0:
                    return None

                await session.commit()

                # Get updated batch
                get_query = select(GenerationBatch).where(
                    GenerationBatch.id == batch_id
                )
                get_result = await session.execute(get_query)
                return get_result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error updating generation batch: {e}")
                raise

    async def get_batch_completion_status(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Get completion status for all batches in a session."""
        async with self.get_session() as session:
            try:
                query = select(
                    func.count(GenerationBatch.id).label("total_batches"),
                    func.count(GenerationBatch.id)
                    .filter(GenerationBatch.status == "completed")
                    .label("completed_batches"),
                    func.count(GenerationBatch.id)
                    .filter(GenerationBatch.status == "failed")
                    .label("failed_batches"),
                    func.avg(GenerationBatch.generation_time_ms).label("avg_generation_time"),
                ).where(GenerationBatch.session_id == session_id)

                result = await session.execute(query)
                stats = result.first()

                total = stats.total_batches or 0
                completed = stats.completed_batches or 0
                failed = stats.failed_batches or 0
                
                completion_rate = completed / total if total > 0 else 0.0
                failure_rate = failed / total if total > 0 else 0.0

                return {
                    "session_id": session_id,
                    "total_batches": total,
                    "completed_batches": completed,
                    "failed_batches": failed,
                    "in_progress_batches": total - completed - failed,
                    "completion_rate": completion_rate,
                    "failure_rate": failure_rate,
                    "avg_generation_time_ms": float(stats.avg_generation_time or 0),
                }

            except Exception as e:
                logger.error(f"Error getting batch completion status: {e}")
                raise

    # Generation Method Performance

    async def get_generation_method_performance(
        self,
        session_id: str | None = None,
        method_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[GenerationMethodPerformance]:
        """Get generation method performance data."""
        async with self.get_session() as session:
            try:
                query = select(GenerationMethodPerformance)
                conditions = []

                if session_id:
                    conditions.append(
                        GenerationMethodPerformance.session_id == session_id
                    )
                if method_name:
                    conditions.append(
                        GenerationMethodPerformance.method_name == method_name
                    )
                if date_from:
                    conditions.append(
                        GenerationMethodPerformance.created_at >= date_from
                    )
                if date_to:
                    conditions.append(GenerationMethodPerformance.created_at <= date_to)

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting generation method performance: {e}")
                raise

    async def create_method_performance_record(
        self,
        performance_data: dict[str, Any],
    ) -> GenerationMethodPerformance:
        """Create method performance record."""
        async with self.get_session() as session:
            try:
                performance = GenerationMethodPerformance(**performance_data)
                session.add(performance)
                await session.commit()
                await session.refresh(performance)
                logger.info(f"Created method performance record {performance.id}")
                return performance
            except Exception as e:
                logger.error(f"Error creating method performance record: {e}")
                raise

    async def get_method_performance_comparison(
        self,
        method_names: list[str],
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Compare performance across different generation methods."""
        async with self.get_session() as session:
            try:
                comparison_data = {}
                
                for method_name in method_names:
                    query = select(
                        func.avg(GenerationMethodPerformance.avg_quality_score).label("avg_quality"),
                        func.avg(GenerationMethodPerformance.generation_time_ms).label("avg_time"),
                        func.avg(GenerationMethodPerformance.success_rate).label("avg_success_rate"),
                        func.count(GenerationMethodPerformance.id).label("sample_count"),
                    ).where(GenerationMethodPerformance.method_name == method_name)
                    
                    if date_from:
                        query = query.where(GenerationMethodPerformance.created_at >= date_from)
                    if date_to:
                        query = query.where(GenerationMethodPerformance.created_at <= date_to)
                    
                    result = await session.execute(query)
                    stats = result.first()
                    
                    comparison_data[method_name] = {
                        "avg_quality_score": float(stats.avg_quality or 0),
                        "avg_generation_time_ms": float(stats.avg_time or 0),
                        "avg_success_rate": float(stats.avg_success_rate or 0),
                        "sample_count": stats.sample_count or 0,
                        "efficiency_score": (
                            stats.avg_quality / (stats.avg_time / 1000) 
                            if stats.avg_quality and stats.avg_time 
                            else 0.0
                        ),
                    }
                
                # Determine best method by efficiency
                best_method = max(
                    comparison_data.items(),
                    key=lambda x: x[1]["efficiency_score"],
                    default=(None, None)
                )[0]
                
                return {
                    "method_comparison": comparison_data,
                    "best_method": best_method,
                    "comparison_period": {
                        "from": date_from.isoformat() if date_from else None,
                        "to": date_to.isoformat() if date_to else None,
                    },
                }
                
            except Exception as e:
                logger.error(f"Error getting method performance comparison: {e}")
                raise

    # A/B Testing and Experiment Management

    async def create_ab_test_session(
        self,
        test_config: dict[str, Any],
    ) -> GenerationSession:
        """Create A/B test generation session."""
        async with self.get_session() as session:
            try:
                # Add A/B test specific metadata
                session_data = test_config.copy()
                session_data["session_type"] = "ab_test"
                session_data["metadata"] = session_data.get("metadata", {})
                session_data["metadata"]["test_type"] = "ab_test"
                session_data["metadata"]["test_variants"] = test_config.get("variants", [])
                
                generation_session = GenerationSession(**session_data)
                session.add(generation_session)
                await session.commit()
                await session.refresh(generation_session)
                logger.info(f"Created A/B test session {generation_session.id}")
                return generation_session
            except Exception as e:
                logger.error(f"Error creating A/B test session: {e}")
                raise

    async def get_ab_test_results(
        self,
        test_session_id: str,
    ) -> dict[str, Any]:
        """Get A/B test results for a test session."""
        async with self.get_session() as session:
            try:
                # Get test session
                test_session = await self.get_generation_session_by_id(test_session_id)
                if not test_session or test_session.session_type != "ab_test":
                    return {"error": "A/B test session not found"}
                
                # Get batches for this test
                batches = await self.get_generation_batches(test_session_id)
                
                # Group results by variant
                variant_results = {}
                for batch in batches:
                    batch_metadata = batch.metadata or {}
                    variant = batch_metadata.get("test_variant", "default")
                    
                    if variant not in variant_results:
                        variant_results[variant] = {
                            "total_samples": 0,
                            "avg_quality": 0.0,
                            "avg_generation_time": 0.0,
                            "success_count": 0,
                            "batch_count": 0,
                        }
                    
                    variant_data = variant_results[variant]
                    variant_data["total_samples"] += batch.samples_generated or 0
                    variant_data["batch_count"] += 1
                    
                    if batch.avg_quality_score:
                        variant_data["avg_quality"] = (
                            (variant_data["avg_quality"] * (variant_data["batch_count"] - 1) + 
                             batch.avg_quality_score) / variant_data["batch_count"]
                        )
                    
                    if batch.generation_time_ms:
                        variant_data["avg_generation_time"] = (
                            (variant_data["avg_generation_time"] * (variant_data["batch_count"] - 1) + 
                             batch.generation_time_ms) / variant_data["batch_count"]
                        )
                    
                    if batch.status == "completed":
                        variant_data["success_count"] += 1
                
                # Calculate statistical significance (simplified)
                total_variants = len(variant_results)
                if total_variants >= 2:
                    best_variant = max(
                        variant_results.items(),
                        key=lambda x: x[1]["avg_quality"]
                    )[0]
                    
                    return {
                        "test_session_id": test_session_id,
                        "variant_results": variant_results,
                        "best_variant": best_variant,
                        "total_variants": total_variants,
                        "test_status": test_session.status,
                        "statistical_significance": "moderate",  # Placeholder
                    }
                
                return {
                    "test_session_id": test_session_id,
                    "variant_results": variant_results,
                    "total_variants": total_variants,
                    "test_status": test_session.status,
                }
                
            except Exception as e:
                logger.error(f"Error getting A/B test results: {e}")
                raise

    async def get_experiment_summary(
        self,
        experiment_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Get summary of experiments by type and date range."""
        async with self.get_session() as session:
            try:
                query = select(
                    GenerationSession.session_type,
                    func.count(GenerationSession.id).label("total_experiments"),
                    func.count(GenerationSession.id)
                    .filter(GenerationSession.status == "completed")
                    .label("completed_experiments"),
                    func.avg(GenerationSession.quality_threshold).label("avg_quality_threshold"),
                ).group_by(GenerationSession.session_type)
                
                conditions = []
                if experiment_type:
                    conditions.append(GenerationSession.session_type == experiment_type)
                if date_from:
                    conditions.append(GenerationSession.created_at >= date_from)
                if date_to:
                    conditions.append(GenerationSession.created_at <= date_to)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                result = await session.execute(query)
                summary_data = {}
                
                for row in result:
                    summary_data[row.session_type or "default"] = {
                        "total_experiments": row.total_experiments or 0,
                        "completed_experiments": row.completed_experiments or 0,
                        "completion_rate": (
                            row.completed_experiments / row.total_experiments
                            if row.total_experiments else 0.0
                        ),
                        "avg_quality_threshold": float(row.avg_quality_threshold or 0),
                    }
                
                return {
                    "experiment_summary": summary_data,
                    "filter_criteria": {
                        "experiment_type": experiment_type,
                        "date_from": date_from.isoformat() if date_from else None,
                        "date_to": date_to.isoformat() if date_to else None,
                    },
                }
                
            except Exception as e:
                logger.error(f"Error getting experiment summary: {e}")
                raise