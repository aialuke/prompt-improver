"""Metrics repository implementation for ML analytics and performance metrics.

Handles analytics, insights, comprehensive metrics tracking, and data analysis
following repository pattern with protocol-based dependency injection.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    GenerationAnalytics,
    MLModelPerformance,
    RulePerformance,
    TrainingIteration,
    TrainingSession,
)
from prompt_improver.repositories.base_repository import BaseRepository
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    SyntheticDataMetrics,
    TrainingMetrics,
)

logger = logging.getLogger(__name__)


class MetricsRepository(BaseRepository[TrainingSession]):
    """Repository for ML analytics and performance metrics."""

    def __init__(self, connection_manager: DatabaseServices):
        super().__init__(
            model_class=TrainingSession,
            connection_manager=connection_manager,
        )
        self.connection_manager = connection_manager
        logger.info("Metrics repository initialized")

    # Analytics and Insights Implementation

    async def get_synthetic_data_metrics(
        self,
        session_id: str,
    ) -> SyntheticDataMetrics | None:
        """Get comprehensive synthetic data metrics."""
        async with self.get_session() as session:
            try:
                from prompt_improver.database.models import GenerationSession, SyntheticDataSample

                # Get session info
                generation_session_query = select(GenerationSession).where(
                    GenerationSession.id == session_id
                )
                generation_result = await session.execute(generation_session_query)
                generation_session = generation_result.scalar_one_or_none()
                
                if not generation_session:
                    return None

                # Get sample statistics
                samples_query = select(
                    func.count(SyntheticDataSample.id).label("total_samples"),
                    func.avg(SyntheticDataSample.quality_score).label("avg_quality"),
                    func.avg(SyntheticDataSample.generation_time_ms).label("avg_time"),
                ).where(SyntheticDataSample.session_id == session_id)
                samples_result = await session.execute(samples_query)
                stats = samples_result.first()

                # Calculate generation efficiency
                generation_efficiency = 1.0  # Placeholder calculation
                if stats.avg_time:
                    generation_efficiency = min(1000 / stats.avg_time, 1.0)

                # Get method performance (placeholder)
                method_performance = {
                    generation_session.generation_method or "default": {
                        "avg_quality": float(stats.avg_quality or 0),
                        "avg_time_ms": float(stats.avg_time or 0),
                        "success_rate": 0.95,
                    }
                }

                # Quality distribution
                quality_distribution = {"high": 0, "medium": 0, "low": 0}  # Simplified

                return SyntheticDataMetrics(
                    session_id=session_id,
                    total_samples=stats.total_samples or 0,
                    avg_quality_score=float(stats.avg_quality or 0),
                    generation_efficiency=generation_efficiency,
                    method_performance=method_performance,
                    quality_distribution=quality_distribution,
                )

            except Exception as e:
                logger.error(f"Error getting synthetic data metrics: {e}")
                raise

    async def get_training_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Get training analytics for date range."""
        async with self.get_session() as session:
            try:
                # Get session statistics
                sessions_query = select(
                    func.count(TrainingSession.id).label("total_sessions"),
                    func.avg(TrainingSession.current_performance).label(
                        "avg_performance"
                    ),
                    func.count(TrainingSession.id)
                    .filter(TrainingSession.status == "completed")
                    .label("completed_sessions"),
                ).where(
                    and_(
                        TrainingSession.created_at >= start_date,
                        TrainingSession.created_at <= end_date,
                    )
                )
                sessions_result = await session.execute(sessions_query)
                session_stats = sessions_result.first()

                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                    "training_sessions": {
                        "total": session_stats.total_sessions or 0,
                        "completed": session_stats.completed_sessions or 0,
                        "avg_performance": float(session_stats.avg_performance or 0),
                    },
                    "generated_at": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error(f"Error getting training analytics: {e}")
                raise

    async def get_rule_performance_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule performance data for intelligence processing."""
        async with self.get_session() as session:
            try:
                query = (
                    select(
                        RulePerformance.rule_id,
                        RulePerformance.rule_name,
                        func.count(RulePerformance.id).label("usage_count"),
                        func.avg(RulePerformance.improvement_score).label("avg_improvement"),
                        func.avg(RulePerformance.confidence_level).label("avg_confidence"),
                        func.count(
                            RulePerformance.id.filter(
                                RulePerformance.improvement_score >= 0.7
                            )
                        ).label("success_count"),
                        func.max(RulePerformance.created_at).label("last_used"),
                    )
                    .where(RulePerformance.created_at > datetime.now() - timedelta(days=90))
                    .group_by(RulePerformance.rule_id, RulePerformance.rule_name)
                    .order_by(desc("usage_count"))
                    .limit(batch_size)
                )
                
                result = await session.execute(query)
                performance_data = []
                
                for row in result:
                    usage_count = row.usage_count or 0
                    success_count = row.success_count or 0
                    effectiveness_ratio = success_count / usage_count if usage_count > 0 else 0.0
                    
                    performance_data.append({
                        "rule_id": row.rule_id,
                        "rule_name": row.rule_name,
                        "usage_count": usage_count,
                        "success_count": success_count,
                        "effectiveness_ratio": effectiveness_ratio,
                        "avg_improvement": row.avg_improvement or 0.0,
                        "confidence_score": row.avg_confidence or 0.0,
                        "last_used": row.last_used,
                    })
                
                logger.info(f"Retrieved {len(performance_data)} rule performance records")
                return performance_data
                
            except Exception as e:
                logger.error(f"Error getting rule performance data: {e}")
                raise

    async def get_rule_combinations_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule combination data for analysis."""
        async with self.get_session() as session:
            try:
                # Get rule combinations that appear together frequently
                query = text("""
                    WITH rule_sessions AS (
                        SELECT 
                            prompt_id,
                            array_agg(DISTINCT rule_id ORDER BY rule_id) as rule_combination,
                            avg(improvement_score) as avg_improvement,
                            avg(confidence_level) as avg_quality,
                            count(*) as usage_count,
                            max(created_at) as last_used
                        FROM rule_performance 
                        WHERE prompt_id IS NOT NULL 
                            AND created_at > NOW() - INTERVAL '90 days'
                        GROUP BY prompt_id
                        HAVING count(DISTINCT rule_id) >= 2
                    )
                    SELECT 
                        rule_combination,
                        avg(avg_improvement) as avg_improvement,
                        avg(avg_quality) as avg_quality,
                        sum(usage_count) as usage_count,
                        max(last_used) as last_used,
                        count(*) as session_count
                    FROM rule_sessions
                    GROUP BY rule_combination
                    ORDER BY usage_count DESC
                    LIMIT :batch_size
                """)
                
                result = await session.execute(query, {"batch_size": batch_size})
                combination_data = []
                
                for row in result:
                    combination_data.append({
                        "rule_combination": row.rule_combination,
                        "avg_improvement": row.avg_improvement or 0.0,
                        "avg_quality": row.avg_quality or 0.0,
                        "usage_count": row.usage_count or 0,
                        "last_used": row.last_used,
                        "session_count": row.session_count or 0,
                    })
                
                logger.info(f"Retrieved {len(combination_data)} rule combinations")
                return combination_data
                
            except Exception as e:
                logger.error(f"Error getting rule combinations data: {e}")
                # Return empty list on error to allow processing to continue
                return []

    async def get_performance_trends(
        self,
        model_type: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get performance trends over time."""
        async with self.get_session() as session:
            try:
                start_date = datetime.now() - timedelta(days=days)
                
                query = select(
                    func.date_trunc('day', MLModelPerformance.created_at).label('date'),
                    func.avg(MLModelPerformance.accuracy).label('avg_accuracy'),
                    func.avg(MLModelPerformance.precision).label('avg_precision'),
                    func.avg(MLModelPerformance.recall).label('avg_recall'),
                    func.count(MLModelPerformance.id).label('model_count'),
                ).where(MLModelPerformance.created_at >= start_date)
                
                if model_type:
                    query = query.where(MLModelPerformance.model_type == model_type)
                
                query = query.group_by(func.date_trunc('day', MLModelPerformance.created_at))
                query = query.order_by('date')
                
                result = await session.execute(query)
                trends = []
                
                for row in result:
                    trends.append({
                        "date": row.date.isoformat() if row.date else None,
                        "avg_accuracy": float(row.avg_accuracy or 0),
                        "avg_precision": float(row.avg_precision or 0),
                        "avg_recall": float(row.avg_recall or 0),
                        "model_count": row.model_count or 0,
                    })
                
                return {
                    "model_type": model_type,
                    "period_days": days,
                    "trends": trends,
                }
                
            except Exception as e:
                logger.error(f"Error getting performance trends: {e}")
                raise

    async def get_training_efficiency_metrics(
        self,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Get training efficiency metrics."""
        async with self.get_session() as session:
            try:
                query = select(
                    func.avg(TrainingIteration.duration_seconds).label('avg_duration'),
                    func.avg(TrainingIteration.performance_score).label('avg_performance'),
                    func.count(TrainingIteration.id).label('total_iterations'),
                    func.min(TrainingIteration.performance_score).label('min_performance'),
                    func.max(TrainingIteration.performance_score).label('max_performance'),
                )
                
                if session_id:
                    query = query.where(TrainingIteration.session_id == session_id)
                
                result = await session.execute(query)
                stats = result.first()
                
                # Calculate efficiency score
                efficiency_score = 0.0
                if stats.avg_duration and stats.avg_performance:
                    # Simple efficiency: performance per second
                    efficiency_score = stats.avg_performance / stats.avg_duration
                
                return {
                    "session_id": session_id,
                    "avg_iteration_duration": float(stats.avg_duration or 0),
                    "avg_performance_score": float(stats.avg_performance or 0),
                    "total_iterations": stats.total_iterations or 0,
                    "performance_range": {
                        "min": float(stats.min_performance or 0),
                        "max": float(stats.max_performance or 0),
                    },
                    "efficiency_score": efficiency_score,
                }
                
            except Exception as e:
                logger.error(f"Error getting training efficiency metrics: {e}")
                raise

    async def get_model_comparison_metrics(
        self,
        model_ids: list[str],
    ) -> dict[str, Any]:
        """Get comparative metrics for multiple models."""
        async with self.get_session() as session:
            try:
                comparisons = {}
                
                for model_id in model_ids:
                    query = (
                        select(MLModelPerformance)
                        .where(MLModelPerformance.model_id == model_id)
                        .order_by(desc(MLModelPerformance.created_at))
                        .limit(1)
                    )
                    result = await session.execute(query)
                    performance = result.scalar_one_or_none()
                    
                    if performance:
                        comparisons[model_id] = {
                            "accuracy": performance.accuracy or 0.0,
                            "precision": performance.precision or 0.0,
                            "recall": performance.recall or 0.0,
                            "f1_score": performance.f1_score or 0.0,
                            "training_samples": performance.training_samples or 0,
                            "model_type": performance.model_type,
                            "last_updated": performance.created_at.isoformat(),
                        }
                
                # Calculate relative performance
                if len(comparisons) > 1:
                    metrics = ["accuracy", "precision", "recall", "f1_score"]
                    best_in_metric = {}
                    
                    for metric in metrics:
                        best_value = max(
                            comp[metric] for comp in comparisons.values()
                        )
                        best_models = [
                            model_id for model_id, comp in comparisons.items()
                            if comp[metric] == best_value
                        ]
                        best_in_metric[metric] = best_models
                    
                    return {
                        "comparisons": comparisons,
                        "best_in_metric": best_in_metric,
                        "total_models": len(comparisons),
                    }
                
                return {"comparisons": comparisons, "total_models": len(comparisons)}
                
            except Exception as e:
                logger.error(f"Error getting model comparison metrics: {e}")
                raise

    async def get_resource_utilization_metrics(
        self,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """Get resource utilization metrics."""
        async with self.get_session() as session:
            try:
                start_time = datetime.now() - timedelta(hours=time_range_hours)
                
                # Get training sessions in time range
                query = select(TrainingSession).where(
                    TrainingSession.created_at >= start_time
                )
                result = await session.execute(query)
                sessions = result.scalars().all()
                
                total_sessions = len(sessions)
                active_sessions = len([s for s in sessions if s.status in ["running", "paused"]])
                
                # Calculate average resource utilization
                resource_utilizations = [
                    s.resource_utilization for s in sessions 
                    if s.resource_utilization
                ]
                
                avg_cpu = 0.0
                avg_memory = 0.0
                avg_gpu = 0.0
                
                if resource_utilizations:
                    cpu_values = [r.get("cpu_percent", 0) for r in resource_utilizations]
                    memory_values = [r.get("memory_percent", 0) for r in resource_utilizations]
                    gpu_values = [r.get("gpu_percent", 0) for r in resource_utilizations]
                    
                    avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
                    avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0.0
                    avg_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else 0.0
                
                return {
                    "time_range_hours": time_range_hours,
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "avg_resource_utilization": {
                        "cpu_percent": avg_cpu,
                        "memory_percent": avg_memory,
                        "gpu_percent": avg_gpu,
                    },
                    "utilization_efficiency": min(1.0, (avg_cpu + avg_memory + avg_gpu) / 3 / 100),
                }
                
            except Exception as e:
                logger.error(f"Error getting resource utilization metrics: {e}")
                raise