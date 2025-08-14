"""Inference repository implementation for predictions and synthetic data operations.

Handles synthetic data samples, batch inference, predictions storage, and data quality
following repository pattern with protocol-based dependency injection.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    AdvancedPatternResults,
    RuleIntelligenceCache,
    SyntheticDataSample,
)
from prompt_improver.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class InferenceRepository(BaseRepository[SyntheticDataSample]):
    """Repository for inference operations and synthetic data management."""

    def __init__(self, connection_manager: DatabaseServices):
        super().__init__(
            model_class=SyntheticDataSample,
            connection_manager=connection_manager,
        )
        self.connection_manager = connection_manager
        logger.info("Inference repository initialized")

    # Synthetic Data Sample Management

    async def create_synthetic_data_samples(
        self,
        samples_data: list[dict[str, Any]],
    ) -> list[SyntheticDataSample]:
        """Create multiple synthetic data samples."""
        async with self.get_session() as session:
            try:
                samples = [SyntheticDataSample(**data) for data in samples_data]
                session.add_all(samples)
                await session.commit()

                for sample in samples:
                    await session.refresh(sample)

                logger.info(f"Created {len(samples)} synthetic data samples")
                return samples
            except Exception as e:
                logger.error(f"Error creating synthetic data samples: {e}")
                raise

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
        async with self.get_session() as session:
            try:
                query = select(SyntheticDataSample)
                conditions = []

                if session_id:
                    conditions.append(SyntheticDataSample.session_id == session_id)
                if batch_id:
                    conditions.append(SyntheticDataSample.batch_id == batch_id)
                if min_quality_score is not None:
                    conditions.append(
                        SyntheticDataSample.quality_score >= min_quality_score
                    )
                if domain_category:
                    conditions.append(
                        SyntheticDataSample.domain_category == domain_category
                    )
                if status:
                    conditions.append(SyntheticDataSample.status == status)

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(desc(SyntheticDataSample.quality_score))
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.error(f"Error getting synthetic data samples: {e}")
                raise

    async def update_synthetic_data_sample(
        self,
        sample_id: str,
        update_data: dict[str, Any],
    ) -> SyntheticDataSample | None:
        """Update synthetic data sample."""
        async with self.get_session() as session:
            try:
                query = (
                    update(SyntheticDataSample)
                    .where(SyntheticDataSample.id == sample_id)
                    .values(**update_data)
                )
                result = await session.execute(query)

                if result.rowcount == 0:
                    return None

                await session.commit()

                # Get updated sample
                get_query = select(SyntheticDataSample).where(
                    SyntheticDataSample.id == sample_id
                )
                get_result = await session.execute(get_query)
                return get_result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error updating synthetic data sample: {e}")
                raise

    async def archive_synthetic_samples(
        self,
        sample_ids: list[str],
    ) -> int:
        """Archive synthetic data samples, returns count updated."""
        async with self.get_session() as session:
            try:
                query = (
                    update(SyntheticDataSample)
                    .where(SyntheticDataSample.id.in_(sample_ids))
                    .values(status="archived")
                )
                result = await session.execute(query)
                await session.commit()
                logger.info(f"Archived {result.rowcount} synthetic samples")
                return result.rowcount
            except Exception as e:
                logger.error(f"Error archiving synthetic samples: {e}")
                raise

    async def get_sample_quality_distribution(
        self,
        session_id: str | None = None,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Get quality distribution of synthetic samples."""
        async with self.get_session() as session:
            try:
                query = select(
                    func.count(SyntheticDataSample.id).label("total_samples"),
                    func.avg(SyntheticDataSample.quality_score).label("avg_quality"),
                    func.min(SyntheticDataSample.quality_score).label("min_quality"),
                    func.max(SyntheticDataSample.quality_score).label("max_quality"),
                    func.count(SyntheticDataSample.id)
                    .filter(SyntheticDataSample.quality_score >= 0.8)
                    .label("high_quality_count"),
                    func.count(SyntheticDataSample.id)
                    .filter(
                        and_(
                            SyntheticDataSample.quality_score >= 0.5,
                            SyntheticDataSample.quality_score < 0.8,
                        )
                    )
                    .label("medium_quality_count"),
                    func.count(SyntheticDataSample.id)
                    .filter(SyntheticDataSample.quality_score < 0.5)
                    .label("low_quality_count"),
                )

                conditions = []
                if session_id:
                    conditions.append(SyntheticDataSample.session_id == session_id)
                if batch_id:
                    conditions.append(SyntheticDataSample.batch_id == batch_id)

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                stats = result.first()

                total = stats.total_samples or 0
                
                return {
                    "session_id": session_id,
                    "batch_id": batch_id,
                    "total_samples": total,
                    "quality_statistics": {
                        "avg_quality": float(stats.avg_quality or 0),
                        "min_quality": float(stats.min_quality or 0),
                        "max_quality": float(stats.max_quality or 0),
                    },
                    "quality_distribution": {
                        "high_quality": {
                            "count": stats.high_quality_count or 0,
                            "percentage": (stats.high_quality_count / total * 100) if total > 0 else 0,
                        },
                        "medium_quality": {
                            "count": stats.medium_quality_count or 0,
                            "percentage": (stats.medium_quality_count / total * 100) if total > 0 else 0,
                        },
                        "low_quality": {
                            "count": stats.low_quality_count or 0,
                            "percentage": (stats.low_quality_count / total * 100) if total > 0 else 0,
                        },
                    },
                }

            except Exception as e:
                logger.error(f"Error getting sample quality distribution: {e}")
                raise

    # Batch Inference Operations

    async def process_ml_predictions_batch(
        self, batch_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process ML predictions for batch of data."""
        try:
            predictions = []
            
            for item in batch_data:
                # Generate prediction based on rule performance data
                effectiveness_ratio = item.get("effectiveness_ratio", 0.0)
                confidence_score = item.get("confidence_score", 0.0)
                usage_count = item.get("usage_count", 0)
                
                # Simple prediction algorithm (would be replaced with actual ML model)
                predicted_effectiveness = min(1.0, effectiveness_ratio * 1.1)
                prediction_confidence = min(1.0, confidence_score + (usage_count / 100.0) * 0.1)
                
                prediction = {
                    "rule_id": item.get("rule_id"),
                    "predicted_effectiveness": predicted_effectiveness,
                    "prediction_confidence": prediction_confidence,
                    "model_version": "simple_heuristic_v1.0",
                    "features_used": ["effectiveness_ratio", "confidence_score", "usage_count"],
                    "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                predictions.append(prediction)
            
            logger.info(f"Generated {len(predictions)} ML predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing ML predictions batch: {e}")
            return []

    async def get_batch_inference_status(
        self,
        batch_id: str,
    ) -> dict[str, Any]:
        """Get status of batch inference operation."""
        async with self.get_session() as session:
            try:
                query = select(
                    func.count(SyntheticDataSample.id).label("total_samples"),
                    func.count(SyntheticDataSample.id)
                    .filter(SyntheticDataSample.status == "processed")
                    .label("processed_samples"),
                    func.avg(SyntheticDataSample.generation_time_ms).label("avg_processing_time"),
                ).where(SyntheticDataSample.batch_id == batch_id)

                result = await session.execute(query)
                stats = result.first()

                total = stats.total_samples or 0
                processed = stats.processed_samples or 0
                
                return {
                    "batch_id": batch_id,
                    "total_samples": total,
                    "processed_samples": processed,
                    "pending_samples": total - processed,
                    "completion_rate": processed / total if total > 0 else 0.0,
                    "avg_processing_time_ms": float(stats.avg_processing_time or 0),
                    "status": "completed" if processed == total else "in_progress",
                }

            except Exception as e:
                logger.error(f"Error getting batch inference status: {e}")
                raise

    # Intelligence Processing and Caching

    async def get_prompt_characteristics_batch(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get batch of prompt characteristics for ML processing."""
        async with self.get_session() as session:
            try:
                from prompt_improver.database.models import PromptSession
                
                query = (
                    select(PromptSession)
                    .where(PromptSession.user_context.is_not(None))
                    .order_by(desc(PromptSession.created_at))
                    .limit(batch_size)
                )
                
                result = await session.execute(query)
                sessions = result.scalars().all()
                
                # Convert to dict format expected by intelligence processor
                characteristics_data = []
                for session in sessions:
                    characteristics_data.append({
                        "session_id": session.session_id,
                        "original_prompt": session.original_prompt,
                        "improved_prompt": session.improved_prompt,
                        "improvement_score": session.improvement_score or 0.0,
                        "quality_score": session.quality_score or 0.0,
                        "confidence_level": session.confidence_level or 0.0,
                        "user_context": session.user_context or {},
                        "created_at": session.created_at,
                    })
                
                logger.info(f"Retrieved {len(characteristics_data)} prompt characteristics")
                return characteristics_data
                
            except Exception as e:
                logger.error(f"Error getting prompt characteristics batch: {e}")
                raise

    async def cache_rule_intelligence(
        self, intelligence_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule intelligence results with upsert logic."""
        if not intelligence_data:
            return
            
        async with self.get_session() as session:
            try:
                cached_count = 0
                for intel_item in intelligence_data:
                    # Generate cache key
                    rule_id = intel_item["rule_id"]
                    cache_key = f"{rule_id}_general_analysis"
                    
                    # Convert intelligence data to cache format
                    intelligence_dict = intel_item.get("intelligence_data", {})
                    
                    # Create or update cache entry
                    cache_entry = RuleIntelligenceCache(
                        cache_key=cache_key,
                        rule_id=rule_id,
                        rule_name=intel_item.get("rule_name", ""),
                        effectiveness_score=intelligence_dict.get("effectiveness_ratio", 0.0),
                        characteristic_match_score=0.75,  # Placeholder
                        historical_performance_score=intel_item.get("effectiveness_prediction", 0.0),
                        ml_prediction_score=intel_item.get("confidence_score", 0.0),
                        recency_score=0.8,  # Placeholder
                        total_score=intel_item.get("effectiveness_prediction", 0.0),
                        confidence_level=intel_item.get("confidence_score", 0.0),
                        sample_size=intelligence_dict.get("usage_count", 0),
                        pattern_insights=intel_item.get("pattern_insights", {}),
                        optimization_recommendations=intel_item.get("usage_recommendations", []),
                        performance_trend="stable",
                        prompt_characteristics_hash="general",
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=12),
                    )
                    
                    # Upsert logic
                    existing_query = select(RuleIntelligenceCache).where(
                        RuleIntelligenceCache.cache_key == cache_key
                    )
                    existing_result = await session.execute(existing_query)
                    existing_entry = existing_result.scalar_one_or_none()
                    
                    if existing_entry:
                        # Update existing entry
                        for field, value in cache_entry.__dict__.items():
                            if not field.startswith('_') and field != 'id':
                                setattr(existing_entry, field, value)
                        existing_entry.updated_at = datetime.now(timezone.utc)
                    else:
                        # Create new entry
                        session.add(cache_entry)
                    
                    cached_count += 1
                
                await session.commit()
                logger.info(f"Cached {cached_count} rule intelligence entries")
                
            except Exception as e:
                logger.error(f"Error caching rule intelligence: {e}")
                raise

    async def cache_combination_intelligence(
        self, combination_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule combination intelligence results."""
        if not combination_data:
            return
            
        async with self.get_session() as session:
            try:
                # For now, log the combination intelligence
                # In a full implementation, this would store to a dedicated table
                logger.info(f"Caching {len(combination_data)} rule combination intelligence entries")
                
                for combo in combination_data:
                    rule_combo = combo.get("rule_combination", [])
                    synergy_score = combo.get("synergy_score", 0.0)
                    logger.debug(f"Cached combination {rule_combo} with synergy {synergy_score}")
                
            except Exception as e:
                logger.error(f"Error caching combination intelligence: {e}")
                raise

    async def cache_pattern_discovery(
        self, pattern_data: dict[str, Any]
    ) -> None:
        """Cache pattern discovery results."""
        async with self.get_session() as session:
            try:
                # Create pattern discovery cache entry
                pattern_entry = AdvancedPatternResults(
                    discovery_run_id=str(uuid.uuid4()),
                    discovery_method="general_analysis",
                    pattern_type=pattern_data.get("pattern_type", "general"),
                    min_effectiveness=0.7,
                    min_support=3,
                    parameter_patterns=pattern_data.get("discovery_data", {}).get("frequent_patterns", []),
                    sequence_patterns=[],
                    performance_patterns={},
                    semantic_patterns={},
                    apriori_patterns={},
                    ensemble_analysis=pattern_data.get("insights_summary", {}),
                    cross_validation={},
                    confidence_level=pattern_data.get("confidence_level", 0.8),
                    patterns_found=len(pattern_data.get("discovery_data", {}).get("frequent_patterns", [])),
                    insights_generated=len(pattern_data.get("actionable_recommendations", [])),
                    performance_impact={},
                    validation_results={},
                    actionable_recommendations=pattern_data.get("actionable_recommendations", []),
                )
                
                session.add(pattern_entry)
                await session.commit()
                
                logger.info(f"Cached pattern discovery results: {pattern_entry.discovery_run_id}")
                
            except Exception as e:
                logger.error(f"Error caching pattern discovery: {e}")
                raise

    async def cleanup_expired_cache(self) -> dict[str, Any]:
        """Clean up expired intelligence cache entries."""
        async with self.get_session() as session:
            try:
                from sqlalchemy import delete
                
                # Delete expired cache entries
                delete_query = delete(RuleIntelligenceCache).where(
                    RuleIntelligenceCache.expires_at < datetime.now(timezone.utc)
                )
                
                result = await session.execute(delete_query)
                await session.commit()
                
                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} expired cache entries")
                
                return {"cache_cleaned": cleaned_count}
                
            except Exception as e:
                logger.error(f"Error cleaning up expired cache: {e}")
                return {"cache_cleaned": 0}

    # Data Quality and Validation

    async def validate_sample_quality(
        self,
        sample_id: str,
        quality_thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Validate synthetic sample quality against thresholds."""
        async with self.get_session() as session:
            try:
                query = select(SyntheticDataSample).where(
                    SyntheticDataSample.id == sample_id
                )
                result = await session.execute(query)
                sample = result.scalar_one_or_none()
                
                if not sample:
                    return {"error": "Sample not found", "sample_id": sample_id}
                
                # Default quality thresholds
                thresholds = quality_thresholds or {
                    "min_quality_score": 0.6,
                    "max_generation_time_ms": 10000,
                    "min_content_length": 10,
                }
                
                validation_results = {
                    "sample_id": sample_id,
                    "quality_score": sample.quality_score or 0.0,
                    "generation_time_ms": sample.generation_time_ms or 0,
                    "content_length": len(sample.synthetic_data or ""),
                    "validations": {},
                    "overall_valid": True,
                }
                
                # Quality score validation
                quality_valid = (sample.quality_score or 0) >= thresholds["min_quality_score"]
                validation_results["validations"]["quality_score"] = {
                    "valid": quality_valid,
                    "threshold": thresholds["min_quality_score"],
                    "actual": sample.quality_score or 0.0,
                }
                
                # Generation time validation
                time_valid = (sample.generation_time_ms or 0) <= thresholds["max_generation_time_ms"]
                validation_results["validations"]["generation_time"] = {
                    "valid": time_valid,
                    "threshold": thresholds["max_generation_time_ms"],
                    "actual": sample.generation_time_ms or 0,
                }
                
                # Content length validation
                content_length = len(sample.synthetic_data or "")
                length_valid = content_length >= thresholds["min_content_length"]
                validation_results["validations"]["content_length"] = {
                    "valid": length_valid,
                    "threshold": thresholds["min_content_length"],
                    "actual": content_length,
                }
                
                # Overall validation
                validation_results["overall_valid"] = all([
                    quality_valid, time_valid, length_valid
                ])
                
                return validation_results
                
            except Exception as e:
                logger.error(f"Error validating sample quality: {e}")
                raise

    async def get_data_quality_report(
        self,
        session_id: str | None = None,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive data quality report."""
        async with self.get_session() as session:
            try:
                query = select(SyntheticDataSample)
                conditions = []
                
                if session_id:
                    conditions.append(SyntheticDataSample.session_id == session_id)
                if batch_id:
                    conditions.append(SyntheticDataSample.batch_id == batch_id)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                result = await session.execute(query)
                samples = result.scalars().all()
                
                if not samples:
                    return {"error": "No samples found", "session_id": session_id, "batch_id": batch_id}
                
                # Calculate quality metrics
                quality_scores = [s.quality_score for s in samples if s.quality_score is not None]
                generation_times = [s.generation_time_ms for s in samples if s.generation_time_ms is not None]
                content_lengths = [len(s.synthetic_data or "") for s in samples]
                
                report = {
                    "session_id": session_id,
                    "batch_id": batch_id,
                    "total_samples": len(samples),
                    "quality_metrics": {
                        "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                        "min_quality_score": min(quality_scores) if quality_scores else 0.0,
                        "max_quality_score": max(quality_scores) if quality_scores else 0.0,
                        "samples_with_quality": len(quality_scores),
                    },
                    "performance_metrics": {
                        "avg_generation_time_ms": sum(generation_times) / len(generation_times) if generation_times else 0.0,
                        "min_generation_time_ms": min(generation_times) if generation_times else 0.0,
                        "max_generation_time_ms": max(generation_times) if generation_times else 0.0,
                        "samples_with_timing": len(generation_times),
                    },
                    "content_metrics": {
                        "avg_content_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0.0,
                        "min_content_length": min(content_lengths) if content_lengths else 0.0,
                        "max_content_length": max(content_lengths) if content_lengths else 0.0,
                    },
                    "status_distribution": {},
                }
                
                # Status distribution
                status_counts = {}
                for sample in samples:
                    status = sample.status or "unknown"
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                report["status_distribution"] = status_counts
                
                return report
                
            except Exception as e:
                logger.error(f"Error getting data quality report: {e}")
                raise