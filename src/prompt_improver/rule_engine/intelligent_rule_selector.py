"""Intelligent Rule Selector - Architecturally Corrected Version

Phase 4 Enhancement: Uses pre-computed ML intelligence from database
Maintains strict MCP-ML architectural separation:
- NO direct ML component instantiation
- NO real-time ML analysis calls
- Uses ONLY pre-computed database lookups
- Achieves <200ms SLA through optimized caching
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database import (
    ManagerMode,
    create_security_context,
    get_database_services,
)
from prompt_improver.rule_engine.models import PromptCharacteristics

logger = logging.getLogger(__name__)


class RuleSelectionStrategy(Enum):
    """Rule selection strategies for different use cases."""

    BALANCED = "balanced"
    PERFORMANCE_FOCUSED = "performance"
    CONSERVATIVE = "conservative"
    EXPERIMENTAL = "experimental"


@dataclass
class RuleScore:
    """Enhanced rule score with pre-computed ML insights."""

    rule_id: str
    rule_name: str
    total_score: float
    effectiveness_score: float
    characteristic_match_score: float
    historical_performance_score: float
    ml_prediction_score: float | None
    recency_score: float
    confidence_level: float
    sample_size: int
    pattern_insights: dict[str, Any] | None = None
    optimization_recommendations: list[str] | None = None
    performance_trend: str | None = None
    metadata: dict[str, Any] | None = None


class IntelligentRuleSelector:
    """MCP-Compliant Intelligent Rule Selector using pre-computed ML intelligence.

    Architectural Compliance:
    - Uses ONLY database queries for rule selection
    - NO direct ML component instantiation
    - NO real-time ML analysis
    - Reads pre-computed ML insights from database tables
    - Maintains <200ms SLA through optimized caching
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize with database session only - NO ML components."""
        self.db_session = db_session
        self.scoring_weights = {
            "effectiveness": 0.35,
            "characteristic_match": 0.25,
            "historical_performance": 0.2,
            "ml_prediction": 0.15,
            "recency": 0.05,
        }
        self.min_sample_size = 5
        self.min_confidence_level = 0.6
        self.effectiveness_threshold = 0.6
        self.cache_hit_rate_target = 0.95
        self._cache_stats = {"hits": 0, "misses": 0}
        self.cache = None

    async def select_optimal_rules(
        self,
        prompt: str,
        prompt_characteristics: PromptCharacteristics,
        max_rules: int = 3,
        strategy: RuleSelectionStrategy = RuleSelectionStrategy.BALANCED,
        min_score_threshold: float = 0.5,
    ) -> list[RuleScore]:
        """Select optimal rules using pre-computed ML intelligence.

        MCP-Compliant: Uses ONLY database queries, no ML operations.

        Args:
            prompt: Input prompt text (reserved for future use)
            prompt_characteristics: Extracted prompt characteristics
            max_rules: Maximum number of rules to return
            strategy: Selection strategy to use
            min_score_threshold: Minimum score threshold for rule inclusion

        Returns:
            List of RuleScore objects with pre-computed ML insights
        """
        start_time = time.time()
        if prompt:
            pass
        cache_key = self._generate_cache_key(
            prompt_characteristics, max_rules, strategy
        )
        cached_result = await self._get_cached_rules(cache_key)
        if cached_result:
            self._cache_stats["hits"] += 1
            return cached_result
        self._cache_stats["misses"] += 1
        rule_intelligence = await self._get_precomputed_rule_intelligence(
            prompt_characteristics, max_rules * 2
        )
        scored_rules = await self._apply_selection_strategy(
            rule_intelligence, strategy, min_score_threshold
        )
        final_rules = scored_rules[:max_rules]
        await self._cache_rules(cache_key, final_rules)
        await self._update_access_statistics(final_rules)
        selection_time = (time.time() - start_time) * 1000
        logger.info(
            f"Rule selection completed in {selection_time:.2f}ms using pre-computed intelligence"
        )
        return final_rules

    async def _get_precomputed_rule_intelligence(
        self, characteristics: PromptCharacteristics, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get pre-computed rule intelligence from database.

        MCP-Compliant: Database query only, no ML operations.
        """
        characteristics_hash = self._hash_characteristics(characteristics)
        query = text(
            "\n            SELECT\n                ric.rule_id,\n                ric.rule_name,\n                ric.effectiveness_score,\n                ric.characteristic_match_score,\n                ric.historical_performance_score,\n                ric.ml_prediction_score,\n                ric.recency_score,\n                ric.total_score,\n                ric.confidence_level,\n                ric.sample_size,\n                ric.pattern_insights,\n                ric.optimization_recommendations,\n                ric.performance_trend,\n                rm.rule_category,\n                rm.priority\n            FROM rule_intelligence_cache ric\n            JOIN rule_metadata rm ON ric.rule_id = rm.rule_id\n            WHERE ric.prompt_characteristics_hash = :characteristics_hash\n              AND ric.expires_at > NOW()\n              AND rm.enabled = true\n              AND ric.confidence_level >= :min_confidence\n              AND ric.sample_size >= :min_sample_size\n            ORDER BY ric.total_score DESC\n            LIMIT :limit\n        "
        )
        result = await self.db_session.execute(
            query,
            {
                "characteristics_hash": characteristics_hash,
                "min_confidence": self.min_confidence_level,
                "min_sample_size": self.min_sample_size,
                "limit": limit,
            },
        )
        rules = result.fetchall()
        if not rules:
            logger.warning(
                f"No pre-computed intelligence found for characteristics hash: {characteristics_hash}"
            )
            return await self._get_fallback_rules(characteristics, limit)
        result_dicts: list[dict[str, Any]] = []
        for rule in rules:
            rule_dict: dict[str, Any] = {
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "effectiveness_score": rule.effectiveness_score,
                "characteristic_match_score": rule.characteristic_match_score,
                "historical_performance_score": rule.historical_performance_score,
                "ml_prediction_score": rule.ml_prediction_score,
                "recency_score": rule.recency_score,
                "total_score": rule.total_score,
                "confidence_level": rule.confidence_level,
                "sample_size": rule.sample_size,
                "pattern_insights": rule.pattern_insights,
                "optimization_recommendations": rule.optimization_recommendations,
                "performance_trend": rule.performance_trend,
                "rule_category": rule.rule_category,
                "priority": rule.priority,
            }
            result_dicts.append(rule_dict)
        return result_dicts

    async def _get_fallback_rules(
        self, characteristics: PromptCharacteristics, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Fallback to basic rule selection when pre-computed intelligence unavailable."""
        if characteristics:
            pass
        query = text(
            "\n            SELECT\n                rm.rule_id,\n                rm.rule_name,\n                rm.rule_category,\n                rm.priority,\n                COALESCE(AVG(rp.improvement_score), 0.5) as effectiveness_score,\n                COALESCE(AVG(rp.confidence_level), 0.5) as confidence_level,\n                COUNT(rp.id) as sample_size\n            FROM rule_metadata rm\n            LEFT JOIN rule_performance rp ON rm.rule_id = rp.rule_id\n                AND rp.created_at > NOW() - INTERVAL '30 days'\n            WHERE rm.enabled = true\n            GROUP BY rm.rule_id, rm.rule_name, rm.rule_category, rm.priority\n            HAVING COUNT(rp.id) >= :min_sample_size OR COUNT(rp.id) = 0\n            ORDER BY rm.priority DESC, effectiveness_score DESC\n            LIMIT :limit\n        "
        )
        result = await self.db_session.execute(
            query,
            {"min_sample_size": max(1, self.min_sample_size // 2), "limit": limit},
        )
        rules = result.fetchall()
        fallback_rules: list[dict[str, Any]] = []
        for rule in rules:
            rule_dict: dict[str, Any] = {
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "rule_category": rule.rule_category,
                "priority": rule.priority,
                "effectiveness_score": rule.effectiveness_score,
                "confidence_level": rule.confidence_level,
                "sample_size": rule.sample_size,
                "characteristic_match_score": 0.5,
                "historical_performance_score": rule.effectiveness_score,
                "ml_prediction_score": None,
                "recency_score": 0.5,
                "total_score": rule.effectiveness_score * 0.7 + 0.3,
                "pattern_insights": None,
                "optimization_recommendations": None,
                "performance_trend": "stable",
            }
            fallback_rules.append(rule_dict)
        return fallback_rules

    async def _apply_selection_strategy(
        self,
        rule_intelligence: list[dict[str, Any]],
        strategy: RuleSelectionStrategy,
        min_score_threshold: float,
    ) -> list[RuleScore]:
        """Apply selection strategy to pre-computed rule intelligence."""
        scored_rules: list[RuleScore] = []
        for rule_data in rule_intelligence:
            rule_score = RuleScore(
                rule_id=rule_data["rule_id"],
                rule_name=rule_data["rule_name"],
                total_score=rule_data["total_score"],
                effectiveness_score=rule_data["effectiveness_score"],
                characteristic_match_score=rule_data["characteristic_match_score"],
                historical_performance_score=rule_data["historical_performance_score"],
                ml_prediction_score=rule_data.get("ml_prediction_score"),
                recency_score=rule_data["recency_score"],
                confidence_level=rule_data["confidence_level"],
                sample_size=rule_data["sample_size"],
                pattern_insights=rule_data.get("pattern_insights"),
                optimization_recommendations=rule_data.get(
                    "optimization_recommendations"
                ),
                performance_trend=rule_data.get("performance_trend"),
                metadata={
                    "rule_category": rule_data.get("rule_category"),
                    "priority": rule_data.get("priority"),
                    "source": "precomputed_intelligence",
                },
            )
            adjusted_score = self._adjust_score_for_strategy(rule_score, strategy)
            if adjusted_score.total_score >= min_score_threshold:
                scored_rules.append(adjusted_score)
        scored_rules.sort(key=lambda x: x.total_score, reverse=True)
        return scored_rules

    def _adjust_score_for_strategy(
        self, rule_score: RuleScore, strategy: RuleSelectionStrategy
    ) -> RuleScore:
        """Adjust rule score based on selection strategy."""
        if strategy == RuleSelectionStrategy.PERFORMANCE_FOCUSED:
            rule_score.total_score = (
                rule_score.effectiveness_score * 0.5
                + (rule_score.ml_prediction_score or 0.5) * 0.3
                + rule_score.historical_performance_score * 0.2
            )
        elif strategy == RuleSelectionStrategy.CONSERVATIVE:
            confidence_boost = min(0.2, rule_score.confidence_level - 0.7)
            sample_boost = min(0.1, (rule_score.sample_size - 10) / 100)
            rule_score.total_score += confidence_boost + sample_boost
        elif strategy == RuleSelectionStrategy.EXPERIMENTAL:
            if rule_score.performance_trend == "improving":
                rule_score.total_score += 0.1
            if rule_score.ml_prediction_score and rule_score.ml_prediction_score > 0.7:
                rule_score.total_score += 0.05
        rule_score.total_score = max(0.0, min(1.0, rule_score.total_score))
        return rule_score

    def _generate_cache_key(
        self,
        characteristics: PromptCharacteristics,
        max_rules: int,
        strategy: RuleSelectionStrategy,
    ) -> str:
        """Generate cache key for rule selection request."""
        key_data: dict[str, Any] = {
            "prompt_type": characteristics.prompt_type,
            "domain": characteristics.domain,
            "complexity_level": round(characteristics.complexity_level, 2),
            "length_category": characteristics.length_category,
            "reasoning_required": characteristics.reasoning_required,
            "max_rules": max_rules,
            "strategy": strategy.value,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"rule_selection:{hashlib.md5(key_string.encode()).hexdigest()}"

    def _hash_characteristics(self, characteristics: PromptCharacteristics) -> str:
        """Generate hash for prompt characteristics for database lookup."""
        char_data: dict[str, Any] = {
            "prompt_type": characteristics.prompt_type,
            "domain": characteristics.domain,
            "complexity_level": round(characteristics.complexity_level, 1),
            "length_category": characteristics.length_category,
            "reasoning_required": characteristics.reasoning_required,
            "specificity_level": round(characteristics.specificity_level, 1),
            "task_type": characteristics.task_type,
        }
        char_string = json.dumps(char_data, sort_keys=True)
        return hashlib.sha256(char_string.encode()).hexdigest()

    async def _get_cached_rules(self, cache_key: str) -> list[RuleScore] | None:
        """Get cached rule selection results."""
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                rule_data_list = cast("list[dict[str, Any]]", cached_data)
                return [RuleScore(**rule_data) for rule_data in rule_data_list]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None

    async def _cache_rules(self, cache_key: str, rules: list[RuleScore]) -> None:
        """Cache rule selection results."""
        try:
            serializable_rules: list[dict[str, Any]] = [
                {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "total_score": rule.total_score,
                    "effectiveness_score": rule.effectiveness_score,
                    "characteristic_match_score": rule.characteristic_match_score,
                    "historical_performance_score": rule.historical_performance_score,
                    "ml_prediction_score": rule.ml_prediction_score,
                    "recency_score": rule.recency_score,
                    "confidence_level": rule.confidence_level,
                    "sample_size": rule.sample_size,
                    "pattern_insights": rule.pattern_insights,
                    "optimization_recommendations": rule.optimization_recommendations,
                    "performance_trend": rule.performance_trend,
                    "metadata": rule.metadata,
                }
                for rule in rules
            ]
            await self.cache.set(cache_key, serializable_rules, l2_ttl=300)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    async def _update_access_statistics(self, rules: list[RuleScore]) -> None:
        """Update access statistics for selected rules."""
        try:
            for rule in rules:
                await self.db_session.execute(
                    text(
                        "SELECT update_cache_access_stats('rule_intelligence_cache', :cache_key)"
                    ),
                    {"cache_key": f"rule_{rule.rule_id}"},
                )
        except Exception as e:
            logger.warning(f"Failed to update access statistics: {e}")

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            self._cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )
        return {
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "target_hit_rate": self.cache_hit_rate_target,
            "performance_status": "good"
            if hit_rate >= self.cache_hit_rate_target
            else "needs_improvement",
        }
