"""Clean Rule Selection Service - Rule discovery and selection using dependency injection.

Refactored service that follows clean architecture principles:
- Uses repository interfaces instead of direct database imports
- No coupling to infrastructure layers
- Pure domain logic with injected dependencies

Handles all rule-related operations including:
- Loading rules from repository
- Rule caching and management
- Bandit-based and traditional rule selection
- Rule metadata management
"""

import json
import logging
import time
from typing import Any

from prompt_improver.repositories.protocols import RulesRepositoryProtocol
from prompt_improver.repositories.protocols.rules_repository_protocol import RuleFilter
from prompt_improver.core.events.ml_event_bus import (
    MLEvent,
    MLEventType,
    get_ml_event_bus,
)
from prompt_improver.rule_engine.base import BasePromptRule

logger = logging.getLogger(__name__)


class CleanRuleSelectionService:
    """Clean rule selection service using dependency injection."""

    def __init__(self, rules_repository: RulesRepositoryProtocol):
        """Initialize service with injected repository dependency."""
        self._repository = rules_repository
        self.rule_cache = {}
        self.cache_ttl = 300
        self.enable_bandit_optimization = True
        self.bandit_experiment_id: str | None = None

    async def get_active_rules(self) -> dict[str, BasePromptRule]:
        """Get active rules with caching."""
        cache_key = "active_rules"
        current_time = time.time()

        # Check cache first
        if cache_key in self.rule_cache:
            cached_data, timestamp = self.rule_cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                logger.debug("Returning cached active rules")
                return cached_data

        try:
            # Load fresh rules from repository
            rule_filter = RuleFilter(enabled=True)
            rule_data_list = await self._repository.get_rules(
                filters=rule_filter, 
                sort_by="priority", 
                sort_desc=True
            )

            # Convert rule data to rule instances
            active_rules = {}
            for rule_data in rule_data_list:
                try:
                    rule_instance = await self._instantiate_rule(rule_data)
                    if rule_instance:
                        active_rules[rule_data["rule_id"]] = rule_instance
                except Exception as e:
                    logger.error(f"Failed to instantiate rule {rule_data.get('rule_id', 'unknown')}: {e}")

            # Update cache
            self.rule_cache[cache_key] = (active_rules, current_time)
            logger.info(f"Loaded {len(active_rules)} active rules")
            return active_rules

        except Exception as e:
            logger.error(f"Failed to load active rules: {e}")
            # Return cached data if available, otherwise empty dict
            if cache_key in self.rule_cache:
                cached_data, _ = self.rule_cache[cache_key]
                logger.warning("Returning stale cached rules due to load failure")
                return cached_data
            return {}

    async def get_rules_by_category(self, category: str) -> dict[str, BasePromptRule]:
        """Get active rules filtered by category."""
        try:
            rule_data_list = await self._repository.get_rules_by_category(
                category=category, 
                enabled_only=True
            )

            category_rules = {}
            for rule_data in rule_data_list:
                try:
                    rule_instance = await self._instantiate_rule(rule_data)
                    if rule_instance:
                        category_rules[rule_data["rule_id"]] = rule_instance
                except Exception as e:
                    logger.error(f"Failed to instantiate rule {rule_data.get('rule_id', 'unknown')}: {e}")

            logger.info(f"Loaded {len(category_rules)} rules for category '{category}'")
            return category_rules

        except Exception as e:
            logger.error(f"Failed to load rules for category '{category}': {e}")
            return {}

    async def get_rule_by_id(self, rule_id: str) -> BasePromptRule | None:
        """Get specific rule by ID."""
        try:
            rule_data = await self._repository.get_rule_by_id(rule_id)
            if not rule_data:
                return None

            return await self._instantiate_rule(rule_data)

        except Exception as e:
            logger.error(f"Failed to get rule by ID {rule_id}: {e}")
            return None

    async def select_optimal_rules(
        self,
        prompt_characteristics: dict[str, Any],
        max_rules: int = 5,
        use_bandit: bool = True,
    ) -> list[dict[str, Any]]:
        """Select optimal rules based on characteristics and performance."""
        try:
            if use_bandit and self.enable_bandit_optimization:
                return await self._bandit_rule_selection(
                    prompt_characteristics, max_rules
                )
            else:
                return await self._traditional_rule_selection(
                    prompt_characteristics, max_rules
                )

        except Exception as e:
            logger.error(f"Failed to select optimal rules: {e}")
            # Fallback to basic rule selection
            return await self._fallback_rule_selection(max_rules)

    async def update_rule_performance(
        self,
        rule_id: str,
        session_id: str,
        improvement_score: float,
        confidence_level: float,
        execution_time_ms: int,
        context_data: dict[str, Any] | None = None,
    ) -> bool:
        """Update rule performance metrics."""
        try:
            performance_data = {
                "rule_id": rule_id,
                "session_id": session_id,
                "improvement_score": improvement_score,
                "confidence_level": confidence_level,
                "execution_time_ms": execution_time_ms,
                "prompt_type": context_data.get("prompt_type") if context_data else None,
                "prompt_category": context_data.get("prompt_category") if context_data else None,
                "context_data": context_data,
            }

            performance_record = await self._repository.create_rule_performance(performance_data)
            success = performance_record is not None
            
            if success:
                logger.debug(f"Updated performance for rule {rule_id}")
                # Emit ML event for performance tracking
                await self._emit_performance_event(rule_id, improvement_score)
            else:
                logger.error(f"Failed to update performance for rule {rule_id}")

            return success

        except Exception as e:
            logger.error(f"Error updating rule performance: {e}")
            return False

    async def get_rule_effectiveness_analysis(
        self, 
        rule_id: str,
        days_back: int = 30
    ) -> dict[str, Any]:
        """Get comprehensive effectiveness analysis for rule."""
        try:
            from datetime import datetime, timedelta
            date_from = datetime.now() - timedelta(days=days_back)

            analysis = await self._repository.get_rule_effectiveness_analysis(
                rule_id, date_from=date_from
            )

            if analysis:
                return {
                    "rule_id": analysis.rule_id,
                    "rule_name": analysis.rule_name,
                    "total_applications": analysis.total_applications,
                    "avg_improvement_score": analysis.avg_improvement_score,
                    "success_rate": analysis.success_rate,
                    "avg_execution_time_ms": analysis.avg_execution_time_ms,
                    "recommendations": analysis.recommendations,
                    "performance_by_category": analysis.performance_by_category,
                }
            else:
                return {"rule_id": rule_id, "error": "No analysis data available"}

        except Exception as e:
            logger.error(f"Error getting rule effectiveness analysis: {e}")
            return {"rule_id": rule_id, "error": str(e)}

    async def invalidate_cache(self, cache_key: str | None = None) -> None:
        """Invalidate rule cache."""
        if cache_key:
            self.rule_cache.pop(cache_key, None)
            logger.debug(f"Invalidated cache key: {cache_key}")
        else:
            self.rule_cache.clear()
            logger.info("Invalidated all rule cache")

    async def _instantiate_rule(self, rule_data: dict[str, Any]) -> BasePromptRule | None:
        """Instantiate a rule from rule metadata."""
        try:
            # This would normally involve dynamic module loading
            # For now, return a mock rule instance
            rule_class_name = rule_data.get("rule_class", "UnknownRule")
            rule_config = rule_data.get("configuration", {})
            
            # TODO: Implement proper rule instantiation based on rule_class_name
            # This would involve importing and instantiating the actual rule class
            logger.debug(f"Would instantiate {rule_class_name} with config {rule_config}")
            
            # Placeholder - return None to indicate rule instantiation is not implemented
            return None

        except Exception as e:
            logger.error(f"Error instantiating rule: {e}")
            return None

    async def _bandit_rule_selection(
        self,
        prompt_characteristics: dict[str, Any],
        max_rules: int
    ) -> list[dict[str, Any]]:
        """Select rules using multi-armed bandit optimization."""
        try:
            # Get top performing rules based on recent performance
            top_rules = await self._repository.get_top_performing_rules(
                metric="improvement_score",
                min_applications=5,
                limit=max_rules * 2  # Get more candidates for bandit selection
            )

            # Apply bandit algorithm (simplified Thompson sampling)
            selected_rules = []
            for rule_data in top_rules[:max_rules]:
                # Calculate confidence intervals and selection probability
                avg_score = rule_data.get("avg_improvement_score", 0.5)
                applications = rule_data.get("total_applications", 1)
                
                # Simple exploration vs exploitation
                exploration_bonus = 0.1 / (applications ** 0.5)
                selection_score = avg_score + exploration_bonus
                
                selected_rules.append({
                    "rule_id": rule_data["rule_id"],
                    "selection_score": selection_score,
                    "reason": "bandit_optimization",
                    "confidence": avg_score,
                })

            # Sort by selection score
            selected_rules.sort(key=lambda x: x["selection_score"], reverse=True)
            logger.info(f"Bandit selected {len(selected_rules)} rules")
            return selected_rules

        except Exception as e:
            logger.error(f"Error in bandit rule selection: {e}")
            return await self._traditional_rule_selection(prompt_characteristics, max_rules)

    async def _traditional_rule_selection(
        self,
        prompt_characteristics: dict[str, Any],
        max_rules: int
    ) -> list[dict[str, Any]]:
        """Select rules using traditional priority-based selection."""
        try:
            rule_filter = RuleFilter(enabled=True)
            rules = await self._repository.get_rules(
                filters=rule_filter,
                sort_by="priority",
                sort_desc=True,
                limit=max_rules
            )

            selected_rules = []
            for rule_data in rules:
                selected_rules.append({
                    "rule_id": rule_data["rule_id"],
                    "selection_score": rule_data.get("priority", 1.0),
                    "reason": "priority_based",
                    "confidence": 0.8,  # Default confidence for priority-based selection
                })

            logger.info(f"Traditional selection chose {len(selected_rules)} rules")
            return selected_rules

        except Exception as e:
            logger.error(f"Error in traditional rule selection: {e}")
            return await self._fallback_rule_selection(max_rules)

    async def _fallback_rule_selection(self, max_rules: int) -> list[dict[str, Any]]:
        """Fallback rule selection when other methods fail."""
        try:
            # Get any available rules as fallback
            rules = await self._repository.get_rules(
                limit=max_rules,
                sort_by="created_at",
                sort_desc=True
            )

            fallback_rules = []
            for rule_data in rules[:max_rules]:
                fallback_rules.append({
                    "rule_id": rule_data["rule_id"],
                    "selection_score": 0.5,  # Neutral score
                    "reason": "fallback",
                    "confidence": 0.3,  # Low confidence for fallback
                })

            logger.warning(f"Using fallback selection with {len(fallback_rules)} rules")
            return fallback_rules

        except Exception as e:
            logger.error(f"Error in fallback rule selection: {e}")
            return []

    async def _emit_performance_event(self, rule_id: str, improvement_score: float) -> None:
        """Emit ML event for rule performance tracking."""
        try:
            event_bus = get_ml_event_bus()
            event = MLEvent(
                event_type=MLEventType.RULE_PERFORMANCE_UPDATED,
                data={
                    "rule_id": rule_id,
                    "improvement_score": improvement_score,
                    "timestamp": time.time(),
                },
                source="rule_selection_service",
            )
            await event_bus.emit(event)
        except Exception as e:
            logger.debug(f"Failed to emit performance event: {e}")  # Non-critical failure