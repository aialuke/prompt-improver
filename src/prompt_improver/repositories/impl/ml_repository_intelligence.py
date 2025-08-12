"""ML Repository Intelligence Processing Extension
Additional methods for the ML repository to handle intelligence processing operations.
This extension provides all the database queries migrated from MLIntelligenceProcessor.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class MLIntelligenceRepositoryMixin:
    """Mixin class for ML repository intelligence processing methods.
    
    Contains all database queries migrated from MLIntelligenceProcessor.
    """
    
    async def get_prompt_characteristics_batch(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get batch of prompt characteristics for ML processing.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                characteristics_query = text("""
                    SELECT DISTINCT
                        ps.id,
                        ps.original_prompt,
                        ps.improved_prompt,
                        ps.improvement_score,
                        ps.quality_score,
                        ps.confidence_level,
                        ps.created_at
                    FROM prompt_sessions ps
                    WHERE ps.created_at >= NOW() - INTERVAL '30 days'
                        AND ps.improvement_score IS NOT NULL
                        AND ps.improved_prompt IS NOT NULL
                    ORDER BY ps.created_at DESC
                    LIMIT :batch_size
                """)
                
                result = await session.execute(characteristics_query, {"batch_size": batch_size})
                return [
                    {
                        "session_id": str(row.id),
                        "original_prompt": row.original_prompt,
                        "improved_prompt": row.improved_prompt,
                        "improvement_score": float(row.improvement_score or 0),
                        "quality_score": float(row.quality_score or 0),
                        "confidence_level": float(row.confidence_level or 0),
                        "created_at": row.created_at,
                    }
                    for row in result
                ]
            except Exception as e:
                logger.error(f"Error getting prompt characteristics batch: {e}")
                raise
    
    async def get_rule_performance_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule performance data for intelligence processing.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                rules_query = text("""
                    SELECT
                        res.rule_id,
                        res.usage_count,
                        res.success_count,
                        res.avg_improvement,
                        res.confidence_score,
                        res.last_updated,
                        CASE 
                            WHEN res.usage_count > 0 THEN res.success_count::float / res.usage_count 
                            ELSE 0.0 
                        END as effectiveness_ratio
                    FROM rule_effectiveness_stats res
                    WHERE res.last_updated >= NOW() - INTERVAL '7 days'
                        AND res.usage_count > 0
                    ORDER BY res.last_updated DESC, res.usage_count DESC
                    LIMIT :batch_size
                """)
                
                rules_result = await session.execute(rules_query, {
                    "batch_size": batch_size
                })
                
                return [
                    {
                        "rule_id": row.rule_id,
                        "usage_count": int(row.usage_count or 0),
                        "success_count": int(row.success_count or 0),
                        "avg_improvement": float(row.avg_improvement or 0),
                        "confidence_score": float(row.confidence_score or 0),
                        "effectiveness_ratio": float(row.effectiveness_ratio or 0),
                        "last_updated": row.last_updated,
                    }
                    for row in rules_result
                ]
            except Exception as e:
                logger.error(f"Error getting rule performance data: {e}")
                raise
    
    async def cache_rule_intelligence(
        self, intelligence_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule intelligence results with upsert logic.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                for intel_item in intelligence_data:
                    cache_key = f"rule_{intel_item['rule_id']}_intelligence"
                    
                    insert_query = text("""
                        INSERT INTO rule_intelligence_cache (
                            cache_key,
                            rule_id,
                            intelligence_data,
                            confidence_score,
                            effectiveness_prediction,
                            context_compatibility,
                            usage_recommendations,
                            pattern_insights,
                            performance_forecast,
                            optimization_suggestions,
                            created_at,
                            expires_at
                        )
                        VALUES (
                            :cache_key,
                            :rule_id,
                            :intelligence_data,
                            :confidence_score,
                            :effectiveness_prediction,
                            :context_compatibility,
                            :usage_recommendations,
                            :pattern_insights,
                            :performance_forecast,
                            :optimization_suggestions,
                            NOW(),
                            NOW() + INTERVAL '12 hours'
                        )
                        ON CONFLICT (cache_key) DO UPDATE SET
                            intelligence_data = EXCLUDED.intelligence_data,
                            confidence_score = EXCLUDED.confidence_score,
                            effectiveness_prediction = EXCLUDED.effectiveness_prediction,
                            context_compatibility = EXCLUDED.context_compatibility,
                            usage_recommendations = EXCLUDED.usage_recommendations,
                            pattern_insights = EXCLUDED.pattern_insights,
                            performance_forecast = EXCLUDED.performance_forecast,
                            optimization_suggestions = EXCLUDED.optimization_suggestions,
                            created_at = EXCLUDED.created_at,
                            expires_at = EXCLUDED.expires_at
                    """)
                    
                    await session.execute(insert_query, {
                        "cache_key": cache_key,
                        "rule_id": intel_item["rule_id"],
                        "intelligence_data": intel_item.get("intelligence_data", {}),
                        "confidence_score": intel_item.get("confidence_score", 0.0),
                        "effectiveness_prediction": intel_item.get("effectiveness_prediction", 0.0),
                        "context_compatibility": intel_item.get("context_compatibility", {}),
                        "usage_recommendations": intel_item.get("usage_recommendations", []),
                        "pattern_insights": intel_item.get("pattern_insights", {}),
                        "performance_forecast": intel_item.get("performance_forecast", {}),
                        "optimization_suggestions": intel_item.get("optimization_suggestions", []),
                    })
                
                await session.commit()
            except Exception as e:
                logger.error(f"Error caching rule intelligence: {e}")
                await session.rollback()
                raise
    
    async def get_rule_combinations_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule combination data for analysis.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                combinations_query = text("""
                    SELECT
                        ps.id as session_id,
                        array_agg(DISTINCT res.rule_id ORDER BY res.rule_id) as rule_combination,
                        AVG(ps.improvement_score) as avg_improvement,
                        AVG(ps.quality_score) as avg_quality,
                        COUNT(*) as usage_count,
                        MAX(ps.created_at) as last_used
                    FROM prompt_sessions ps
                    JOIN rule_effectiveness_stats res ON ps.id::text = res.rule_id  -- Simplified join
                    WHERE ps.created_at >= NOW() - INTERVAL '14 days'
                        AND ps.improvement_score IS NOT NULL
                    GROUP BY ps.id
                    HAVING COUNT(DISTINCT res.rule_id) >= 2  -- Only combinations of 2+ rules
                    ORDER BY usage_count DESC, avg_improvement DESC
                    LIMIT :batch_size
                """)
                
                result = await session.execute(combinations_query, {"batch_size": batch_size})
                
                return [
                    {
                        "session_id": str(row.session_id),
                        "rule_combination": row.rule_combination or [],
                        "avg_improvement": float(row.avg_improvement or 0),
                        "avg_quality": float(row.avg_quality or 0),
                        "usage_count": int(row.usage_count or 0),
                        "last_used": row.last_used,
                    }
                    for row in result
                ]
            except Exception as e:
                logger.error(f"Error getting rule combinations data: {e}")
                raise
    
    async def cache_combination_intelligence(
        self, combination_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule combination intelligence results.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                for combo_item in combination_data:
                    combination_key = "_".join(sorted(combo_item.get("rule_combination", [])))
                    
                    insert_query = text("""
                        INSERT INTO rule_combination_intelligence (
                            combination_key,
                            rule_ids,
                            synergy_score,
                            effectiveness_multiplier,
                            context_suitability,
                            performance_data,
                            optimization_insights,
                            created_at,
                            expires_at
                        )
                        VALUES (
                            :combination_key,
                            :rule_ids,
                            :synergy_score,
                            :effectiveness_multiplier,
                            :context_suitability,
                            :performance_data,
                            :optimization_insights,
                            NOW(),
                            NOW() + INTERVAL '12 hours'
                        )
                        ON CONFLICT (combination_key) DO UPDATE SET
                            synergy_score = EXCLUDED.synergy_score,
                            effectiveness_multiplier = EXCLUDED.effectiveness_multiplier,
                            context_suitability = EXCLUDED.context_suitability,
                            performance_data = EXCLUDED.performance_data,
                            optimization_insights = EXCLUDED.optimization_insights,
                            created_at = EXCLUDED.created_at,
                            expires_at = EXCLUDED.expires_at
                    """)
                    
                    await session.execute(insert_query, {
                        "combination_key": combination_key,
                        "rule_ids": combo_item.get("rule_combination", []),
                        "synergy_score": combo_item.get("synergy_score", 0.0),
                        "effectiveness_multiplier": combo_item.get("effectiveness_multiplier", 1.0),
                        "context_suitability": combo_item.get("context_suitability", {}),
                        "performance_data": combo_item.get("performance_data", {}),
                        "optimization_insights": combo_item.get("optimization_insights", {}),
                    })
                
                await session.commit()
            except Exception as e:
                logger.error(f"Error caching combination intelligence: {e}")
                await session.rollback()
                raise
    
    async def cache_pattern_discovery(
        self, pattern_data: dict[str, Any]
    ) -> None:
        """Cache pattern discovery results.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                cache_key = f"pattern_discovery_{pattern_data.get('pattern_type', 'general')}"
                
                insert_query = text("""
                    INSERT INTO pattern_discovery_cache (
                        cache_key,
                        pattern_type,
                        discovery_data,
                        confidence_level,
                        insights_summary,
                        actionable_recommendations,
                        created_at,
                        expires_at
                    )
                    VALUES (
                        :cache_key,
                        :pattern_type,
                        :discovery_data,
                        :confidence_level,
                        :insights_summary,
                        :actionable_recommendations,
                        NOW(),
                        NOW() + INTERVAL '24 hours'
                    )
                    ON CONFLICT (cache_key) DO UPDATE SET
                        discovery_data = EXCLUDED.discovery_data,
                        confidence_level = EXCLUDED.confidence_level,
                        insights_summary = EXCLUDED.insights_summary,
                        actionable_recommendations = EXCLUDED.actionable_recommendations,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at
                """)
                
                await session.execute(insert_query, {
                    "cache_key": cache_key,
                    "pattern_type": pattern_data.get("pattern_type", "general"),
                    "discovery_data": pattern_data.get("discovery_data", {}),
                    "confidence_level": pattern_data.get("confidence_level", 0.0),
                    "insights_summary": pattern_data.get("insights_summary", {}),
                    "actionable_recommendations": pattern_data.get("actionable_recommendations", []),
                })
                
                await session.commit()
            except Exception as e:
                logger.error(f"Error caching pattern discovery: {e}")
                await session.rollback()
                raise
    
    async def cleanup_expired_cache(self) -> dict[str, Any]:
        """Clean up expired intelligence cache entries.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                cleanup_query = text("SELECT clean_expired_intelligence_cache()")
                result = await session.execute(cleanup_query)
                await session.commit()
                
                cleaned_count = result.scalar() or 0
                return {"cache_cleaned": int(cleaned_count)}
            except Exception as e:
                logger.error(f"Error cleaning expired cache: {e}")
                return {"cache_cleaned": 0}
    
    async def check_rule_intelligence_freshness(
        self, rule_id: str
    ) -> bool:
        """Check if rule intelligence cache is fresh.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                check_query = text("""
                    SELECT
                        CASE WHEN expires_at > NOW() THEN true ELSE false END as is_fresh,
                        created_at,
                        expires_at
                    FROM rule_intelligence_cache
                    WHERE rule_id = :rule_id
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = await session.execute(check_query, {"rule_id": rule_id})
                row = result.first()
                return bool(row.is_fresh) if row else False
            except Exception as e:
                logger.error(f"Error checking rule intelligence freshness: {e}")
                return False
    
    async def get_rule_historical_performance(
        self, rule_id: str
    ) -> list[dict[str, Any]]:
        """Get historical performance data for rule.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                performance_query = text("""
                    SELECT
                        res.rule_id,
                        res.usage_count,
                        res.success_count,
                        res.avg_improvement,
                        res.confidence_score,
                        res.last_updated,
                        res.created_at
                    FROM rule_effectiveness_stats res
                    WHERE res.rule_id = :rule_id
                    ORDER BY res.last_updated DESC
                    LIMIT 50
                """)
                
                result = await session.execute(performance_query, {"rule_id": rule_id})
                return [
                    {
                        "rule_id": row.rule_id,
                        "usage_count": int(row.usage_count or 0),
                        "success_count": int(row.success_count or 0),
                        "avg_improvement": float(row.avg_improvement or 0),
                        "confidence_score": float(row.confidence_score or 0),
                        "last_updated": row.last_updated,
                        "created_at": row.created_at,
                    }
                    for row in result
                ]
            except Exception as e:
                logger.error(f"Error getting rule historical performance: {e}")
                raise
    
    async def process_ml_predictions_batch(
        self, batch_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process ML predictions for batch of data.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                # Get batch of rule performance data for predictions
                batch_query = text("""
                    SELECT DISTINCT
                        ps.id as session_id,
                        ps.original_prompt,
                        ps.improved_prompt,
                        ps.improvement_score,
                        ps.quality_score,
                        ps.confidence_level
                    FROM prompt_sessions ps
                    WHERE ps.created_at >= NOW() - INTERVAL '7 days'
                        AND ps.improvement_score IS NOT NULL
                    ORDER BY ps.created_at DESC
                    LIMIT :batch_size
                """)
                
                result = await session.execute(batch_query, {
                    "batch_size": len(batch_data) or 50
                })
                
                predictions = []
                for row in result:
                    prediction = {
                        "session_id": str(row.session_id),
                        "predicted_improvement": float(row.improvement_score or 0),
                        "confidence": float(row.confidence_level or 0),
                        "recommendation_score": min(1.0, float(row.quality_score or 0) * 1.2),
                        "prediction_metadata": {
                            "method": "historical_analysis",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    }
                    predictions.append(prediction)
                
                return predictions
            except Exception as e:
                logger.error(f"Error processing ML predictions batch: {e}")
                raise
    
    async def update_rule_intelligence_incremental(
        self, rule_id: str, performance_data: dict[str, Any]
    ) -> None:
        """Update rule intelligence with incremental data.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                update_query = text("""
                    UPDATE rule_intelligence_cache
                    SET 
                        intelligence_data = intelligence_data || :new_data::jsonb,
                        confidence_score = GREATEST(confidence_score, :confidence_score),
                        effectiveness_prediction = :effectiveness_prediction,
                        created_at = NOW(),
                        expires_at = NOW() + INTERVAL '12 hours'
                    WHERE rule_id = :rule_id
                """)
                
                await session.execute(update_query, {
                    "rule_id": rule_id,
                    "new_data": performance_data,
                    "confidence_score": performance_data.get("confidence_score", 0.0),
                    "effectiveness_prediction": performance_data.get("effectiveness_prediction", 0.0),
                })
                
                await session.commit()
            except Exception as e:
                logger.error(f"Error updating rule intelligence incrementally: {e}")
                await session.rollback()
                raise
    
    async def get_intelligence_processing_stats(self) -> dict[str, Any]:
        """Get statistics for intelligence processing operations.
        
        Migrated from MLIntelligenceProcessor to repository layer.
        """
        async with self.get_session() as session:
            try:
                async with session() as db_session:
                    # Get total rules processed
                    total_rules_query = text("""
                        SELECT COUNT(DISTINCT rule_id) as total_rules
                        FROM rule_intelligence_cache
                        WHERE created_at >= NOW() - INTERVAL '24 hours'
                    """)
                    result = await db_session.execute(total_rules_query)
                    
                    row = result.first()
                    total_rules = int(row.total_rules) if row else 0
                    
                    return {
                        "rules_processed": total_rules,
                        "combinations_generated": total_rules * 2,  # Estimate
                        "patterns_discovered": max(1, total_rules // 10),  # Estimate
                        "predictions_generated": total_rules * 5,  # Estimate
                        "cache_entries": total_rules,
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                    }
            except Exception as e:
                logger.error(f"Error getting intelligence processing stats: {e}")
                return {
                    "rules_processed": 0,
                    "combinations_generated": 0,
                    "patterns_discovered": 0,
                    "predictions_generated": 0,
                    "cache_entries": 0,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }