"""Database service for synthetic data generation tracking (Week 6)

Provides comprehensive database operations for generation metadata,
performance tracking, and analytics with bulk operations support.
"""

import logging
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    GenerationAnalytics,
    GenerationBatch,
    GenerationMethodPerformance,
    GenerationQualityAssessment,
    GenerationSession,
    SyntheticDataSample,
)
from ...utils.datetime_utils import naive_utc_now

logger = logging.getLogger(__name__)

class GenerationDatabaseService:
    """Database service for generation metadata and analytics"""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    # ===== SESSION MANAGEMENT =====

    async def create_generation_session(
        self,
        generation_method: str,
        target_samples: int,
        session_type: str = "synthetic_data",
        training_session_id: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        performance_gaps: Optional[Dict[str, float]] = None,
        focus_areas: Optional[List[str]] = None,
        quality_threshold: float = 0.7
    ) -> GenerationSession:
        """Create a new generation session"""

        session = GenerationSession(
            session_id=str(uuid.uuid4()),
            session_type=session_type,
            training_session_id=training_session_id,
            generation_method=generation_method,
            target_samples=target_samples,
            quality_threshold=quality_threshold,
            performance_gaps=performance_gaps,
            focus_areas=focus_areas,
            configuration=configuration,
            status="running"
        )

        self.db_session.add(session)
        await self.db_session.commit()
        await self.db_session.refresh(session)

        logger.info(f"Created generation session {session.session_id} for {target_samples} samples")
        return session

    async def update_session_status(
        self,
        session_id: str,
        status: str,
        error_message: Optional[str] = None,
        final_sample_count: Optional[int] = None
    ) -> None:
        """Update session status and completion info"""

        result = await self.db_session.execute(
            select(GenerationSession).where(GenerationSession.session_id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Generation session {session_id} not found")

        session.status = status
        session.updated_at = naive_utc_now()

        if status in ["completed", "failed", "cancelled"]:
            session.completed_at = naive_utc_now()
            if session.started_at:
                duration = (session.completed_at - session.started_at).total_seconds()
                session.total_duration_seconds = duration

        if error_message:
            session.error_message = error_message

        if final_sample_count is not None:
            session.final_sample_count = final_sample_count

        await self.db_session.commit()
        logger.info(f"Updated session {session_id} status to {status}")

    # ===== BATCH MANAGEMENT =====

    async def create_generation_batch(
        self,
        session_id: str,
        batch_number: int,
        batch_size: int,
        generation_method: str,
        samples_requested: int
    ) -> GenerationBatch:
        """Create a new generation batch"""

        batch = GenerationBatch(
            batch_id=str(uuid.uuid4()),
            session_id=session_id,
            batch_number=batch_number,
            batch_size=batch_size,
            generation_method=generation_method,
            samples_requested=samples_requested
        )

        self.db_session.add(batch)
        await self.db_session.commit()
        await self.db_session.refresh(batch)

        return batch

    async def update_batch_performance(
        self,
        batch_id: str,
        processing_time_seconds: float,
        samples_generated: int,
        samples_filtered: int = 0,
        error_count: int = 0,
        memory_usage_mb: Optional[float] = None,
        memory_peak_mb: Optional[float] = None,
        efficiency_score: Optional[float] = None,
        average_quality_score: Optional[float] = None,
        diversity_score: Optional[float] = None,
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update batch performance metrics"""

        result = await self.db_session.execute(
            select(GenerationBatch).where(GenerationBatch.batch_id == batch_id)
        )
        batch = result.scalar_one_or_none()

        if not batch:
            raise ValueError(f"Generation batch {batch_id} not found")

        # Update performance metrics
        batch.processing_time_seconds = processing_time_seconds
        batch.samples_generated = samples_generated
        batch.samples_filtered = samples_filtered
        batch.error_count = error_count
        batch.memory_usage_mb = memory_usage_mb
        batch.memory_peak_mb = memory_peak_mb
        batch.efficiency_score = efficiency_score
        batch.average_quality_score = average_quality_score
        batch.diversity_score = diversity_score
        batch.batch_metadata = batch_metadata

        # Calculate derived metrics
        total_samples = samples_generated + error_count
        if total_samples > 0:
            batch.success_rate = samples_generated / total_samples

        if processing_time_seconds > 0:
            batch.throughput_samples_per_sec = samples_generated / processing_time_seconds

        await self.db_session.commit()

    # ===== SAMPLE MANAGEMENT =====

    async def bulk_insert_samples(
        self,
        session_id: str,
        batch_id: str,
        samples_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Bulk insert synthetic data samples"""

        sample_ids = []
        samples = []

        for sample_data in samples_data:
            sample_id = str(uuid.uuid4())
            sample_ids.append(sample_id)

            sample = SyntheticDataSample(
                sample_id=sample_id,
                session_id=session_id,
                batch_id=batch_id,
                feature_vector=sample_data.get("feature_vector", {}),
                effectiveness_score=sample_data.get("effectiveness_score"),
                quality_score=sample_data.get("quality_score"),
                domain_category=sample_data.get("domain_category"),
                difficulty_level=sample_data.get("difficulty_level"),
                focus_areas=sample_data.get("focus_areas"),
                generation_method=sample_data.get("generation_method"),
                generation_strategy=sample_data.get("generation_strategy"),
                targeting_info=sample_data.get("targeting_info")
            )
            samples.append(sample)

        # Bulk insert
        self.db_session.add_all(samples)
        await self.db_session.commit()

        logger.info(f"Bulk inserted {len(samples)} samples for session {session_id}")
        return sample_ids

    async def archive_old_samples(
        self,
        days_old: int = 30,
        batch_size: int = 1000
    ) -> int:
        """Archive old samples to manage storage (data lifecycle management)"""

        cutoff_date = naive_utc_now() - timedelta(days=days_old)

        # Update samples in batches
        total_archived = 0
        while True:
            result = await self.db_session.execute(
                select(SyntheticDataSample.id)
                .where(
                    and_(
                        SyntheticDataSample.created_at < cutoff_date,
                        SyntheticDataSample.status == "active"
                    )
                )
                .limit(batch_size)
            )
            sample_ids = [row[0] for row in result.fetchall()]

            if not sample_ids:
                break

            # Update status to archived
            await self.db_session.execute(
                text("""
                    UPDATE synthetic_data_samples
                    SET status = 'archived', archived_at = NOW()
                    WHERE id = ANY(:sample_ids)
                """),
                {"sample_ids": sample_ids}
            )

            total_archived += len(sample_ids)
            await self.db_session.commit()

        logger.info(f"Archived {total_archived} old samples")
        return total_archived

    # ===== PERFORMANCE TRACKING =====

    async def record_method_performance(
        self,
        session_id: str,
        method_name: str,
        generation_time_seconds: float,
        quality_score: float,
        diversity_score: float,
        memory_usage_mb: float,
        success_rate: float,
        samples_generated: int,
        performance_gaps_addressed: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
        configuration: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record performance metrics for a generation method"""

        performance = GenerationMethodPerformance(
            method_name=method_name,
            session_id=session_id,
            generation_time_seconds=generation_time_seconds,
            quality_score=quality_score,
            diversity_score=diversity_score,
            memory_usage_mb=memory_usage_mb,
            success_rate=success_rate,
            samples_generated=samples_generated,
            performance_gaps_addressed=performance_gaps_addressed,
            batch_size=batch_size,
            configuration=configuration
        )

        self.db_session.add(performance)
        await self.db_session.commit()

    async def get_method_performance_history(
        self,
        method_name: str,
        days_back: int = 30,
        limit: int = 100
    ) -> List[GenerationMethodPerformance]:
        """Get recent performance history for a method"""

        cutoff_date = naive_utc_now() - timedelta(days=days_back)

        result = await self.db_session.execute(
            select(GenerationMethodPerformance)
            .where(
                and_(
                    GenerationMethodPerformance.method_name == method_name,
                    GenerationMethodPerformance.recorded_at >= cutoff_date
                )
            )
            .order_by(desc(GenerationMethodPerformance.recorded_at))
            .limit(limit)
        )

        return result.scalars().all()

    # ===== QUALITY ASSESSMENT =====

    async def record_quality_assessment(
        self,
        session_id: str,
        assessment_type: str,
        overall_quality_score: float,
        samples_assessed: int,
        samples_passed: int,
        samples_failed: int,
        batch_id: Optional[str] = None,
        assessment_results: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None
    ) -> GenerationQualityAssessment:
        """Record quality assessment results"""

        assessment = GenerationQualityAssessment(
            assessment_id=str(uuid.uuid4()),
            session_id=session_id,
            batch_id=batch_id,
            assessment_type=assessment_type,
            overall_quality_score=overall_quality_score,
            samples_assessed=samples_assessed,
            samples_passed=samples_passed,
            samples_failed=samples_failed,
            assessment_results=assessment_results,
            recommendations=recommendations
        )

        self.db_session.add(assessment)
        await self.db_session.commit()
        await self.db_session.refresh(assessment)

        return assessment

    # ===== ANALYTICS =====

    async def get_generation_statistics(
        self,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get comprehensive generation statistics"""

        cutoff_date = naive_utc_now() - timedelta(days=days_back)

        # Session statistics
        session_stats = await self.db_session.execute(
            select(
                func.count(GenerationSession.id).label("total_sessions"),
                func.sum(GenerationSession.samples_generated).label("total_samples"),
                func.avg(GenerationSession.average_quality_score).label("avg_quality"),
                func.avg(GenerationSession.generation_efficiency).label("avg_efficiency")
            )
            .where(GenerationSession.started_at >= cutoff_date)
        )
        stats = session_stats.first()

        # Method performance
        method_stats = await self.db_session.execute(
            select(
                GenerationMethodPerformance.method_name,
                func.count().label("executions"),
                func.avg(GenerationMethodPerformance.quality_score).label("avg_quality"),
                func.avg(GenerationMethodPerformance.success_rate).label("avg_success_rate")
            )
            .where(GenerationMethodPerformance.recorded_at >= cutoff_date)
            .group_by(GenerationMethodPerformance.method_name)
        )
        method_performance = {row[0]: {"executions": row[1], "avg_quality": row[2], "avg_success_rate": row[3]}
                            for row in method_stats.fetchall()}

        return {
            "period_days": days_back,
            "total_sessions": stats[0] or 0,
            "total_samples_generated": stats[1] or 0,
            "average_quality_score": float(stats[2]) if stats[2] else 0.0,
            "average_efficiency": float(stats[3]) if stats[3] else 0.0,
            "method_performance": method_performance,
            "generated_at": naive_utc_now().isoformat()
        }

    async def cleanup_old_data(
        self,
        days_to_keep: int = 90
    ) -> Dict[str, int]:
        """Clean up old generation data (data lifecycle management)"""

        cutoff_date = naive_utc_now() - timedelta(days=days_to_keep)
        cleanup_stats = {}

        # Delete old quality assessments
        result = await self.db_session.execute(
            text("DELETE FROM generation_quality_assessments WHERE assessed_at < :cutoff_date"),
            {"cutoff_date": cutoff_date}
        )
        cleanup_stats["quality_assessments_deleted"] = result.rowcount

        # Delete old method performance records
        result = await self.db_session.execute(
            text("DELETE FROM generation_method_performance WHERE recorded_at < :cutoff_date"),
            {"cutoff_date": cutoff_date}
        )
        cleanup_stats["method_performance_deleted"] = result.rowcount

        # Archive old samples instead of deleting
        archived_count = await self.archive_old_samples(days_to_keep)
        cleanup_stats["samples_archived"] = archived_count

        await self.db_session.commit()

        logger.info(f"Cleaned up old generation data: {cleanup_stats}")
        return cleanup_stats
