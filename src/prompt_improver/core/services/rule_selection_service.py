"""Rule Selection Service - Rule discovery and selection logic

Handles all rule-related operations including:
- Loading rules from database
- Rule caching and management
- Bandit-based and traditional rule selection
- Rule metadata management

Follows single responsibility principle for rule selection concerns.
"""

import json
import logging
import time
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.core.events.ml_event_bus import (
    MLEvent,
    MLEventType,
    get_ml_event_bus,
)
from prompt_improver.rule_engine.base import BasePromptRule

logger = logging.getLogger(__name__)


def _get_database_session():
    """Lazy import of database session to avoid circular imports."""
    from prompt_improver.database import get_session_context

    return get_session_context


class RuleSelectionService:
    """Service focused on rule discovery and selection operations"""

    def __init__(self, enable_bandit_optimization: bool = True):
        self.rule_cache = {}
        self.cache_ttl = 300
        self.enable_bandit_optimization = enable_bandit_optimization
        self.bandit_experiment_id: str | None = None

    async def get_active_rules(
        self, db_session: AsyncSession | None = None
    ) -> dict[str, BasePromptRule]:
        """Get active rules with caching"""
        cache_key = "active_rules"
        if cache_key in self.rule_cache:
            cached_rules, timestamp = self.rule_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_rules

        if db_session:
            rules = await self._load_rules_from_database(db_session)
        else:
            get_session_context = _get_database_session()
            async with get_session_context() as session:
                rules = await self._load_rules_from_database(session)

        self.rule_cache[cache_key] = (rules, time.time())
        return rules

    async def _load_rules_from_database(
        self, db_session: AsyncSession
    ) -> dict[str, BasePromptRule]:
        """Load and instantiate rules from database configuration"""
        try:
            from prompt_improver.database.models import RuleMetadata

            query = (
                select(RuleMetadata)
                .where(RuleMetadata.enabled)
                .order_by(desc(RuleMetadata.priority))
            )
            result = await db_session.execute(query)
            rule_configs = result.scalars().all()
            rules = {}

            for config in rule_configs:
                rule_class = self._get_rule_class(config.rule_id)
                if rule_class:
                    rule_instance = rule_class()
                    if hasattr(rule_instance, "rule_id"):
                        rule_instance.rule_id = config.rule_id
                    if hasattr(rule_instance, "priority"):
                        rule_instance.priority = config.priority
                    if config.default_parameters:
                        if isinstance(config.default_parameters, str):
                            params = json.loads(config.default_parameters)
                        else:
                            params = config.default_parameters
                        if hasattr(rule_instance, "configure"):
                            rule_instance.configure(params)
                    rules[config.rule_id] = rule_instance
            return rules
        except Exception as e:
            logger.error(f"Failed to load rules from database: {e}")
            return self._get_fallback_rules()

    def _get_rule_class(self, rule_id: str) -> type[BasePromptRule] | None:
        """Map rule_id to actual rule class"""
        rule_mapping = {
            "clarity_enhancement": self._import_clarity_rule,
            "chain_of_thought": self._import_chain_of_thought_rule,
            "few_shot_examples": self._import_few_shot_rule,
            "role_based_prompting": self._import_role_based_rule,
            "xml_structure_enhancement": self._import_xml_structure_rule,
            "specificity_enhancement": self._import_specificity_rule,
        }
        import_func = rule_mapping.get(rule_id)
        if import_func:
            return import_func()
        return None

    def _import_clarity_rule(self):
        """Import clarity rule class"""
        try:
            from prompt_improver.rule_engine.rules.clarity import ClarityRule

            return ClarityRule
        except ImportError:
            logger.warning("ClarityRule not found, using base rule")
            return BasePromptRule

    def _import_chain_of_thought_rule(self):
        """Import chain of thought rule class"""
        try:
            from prompt_improver.rule_engine.rules.chain_of_thought import (
                ChainOfThoughtRule,
            )

            return ChainOfThoughtRule
        except ImportError:
            logger.warning("ChainOfThoughtRule not found, using base rule")
            return BasePromptRule

    def _import_few_shot_rule(self):
        """Import few shot example rule class"""
        try:
            from prompt_improver.rule_engine.rules.few_shot_examples import (
                FewShotExampleRule,
            )

            return FewShotExampleRule
        except ImportError:
            logger.warning("FewShotExampleRule not found, using base rule")
            return BasePromptRule

    def _import_role_based_rule(self):
        """Import role based prompting rule class"""
        try:
            from prompt_improver.rule_engine.rules.role_based_prompting import (
                RoleBasedPromptingRule,
            )

            return RoleBasedPromptingRule
        except ImportError:
            logger.warning("RoleBasedPromptingRule not found, using base rule")
            return BasePromptRule

    def _import_xml_structure_rule(self):
        """Import XML structure rule class"""
        try:
            from prompt_improver.rule_engine.rules.xml_structure_enhancement import (
                XMLStructureRule,
            )

            return XMLStructureRule
        except ImportError:
            logger.warning("XMLStructureRule not found, using base rule")
            return BasePromptRule

    def _import_specificity_rule(self):
        """Import specificity rule class"""
        try:
            from prompt_improver.rule_engine.rules.specificity import (
                SpecificityRule,
            )

            return SpecificityRule
        except ImportError:
            logger.warning("SpecificityRule not found, using base rule")
            return BasePromptRule

    def _get_fallback_rules(self) -> dict[str, BasePromptRule]:
        """Fallback rules when database loading fails"""
        try:
            from prompt_improver.rule_engine.rules.clarity import ClarityRule
            from prompt_improver.rule_engine.rules.specificity import (
                SpecificityRule,
            )

            rules = {
                "clarity_enhancement": ClarityRule(),
                "specificity_enhancement": SpecificityRule(),
            }
        except ImportError:
            rules = {}
        return rules

    async def get_optimal_rules(
        self,
        prompt_characteristics: dict[str, Any],
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get optimal rules using multi-armed bandit or traditional selection"""
        if self.enable_bandit_optimization and self.bandit_experiment_id is None:
            await self._initialize_bandit_experiment(db_session)

        if self.enable_bandit_optimization and self.bandit_experiment_id:
            return await self._get_bandit_optimized_rules(
                prompt_characteristics, preferred_rules, db_session, limit
            )
        return await self._get_traditional_optimal_rules(
            preferred_rules, db_session, limit
        )

    async def _initialize_bandit_experiment(
        self, db_session: AsyncSession | None = None
    ) -> str | None:
        """Initialize bandit experiment via event bus communication"""
        if not self.enable_bandit_optimization:
            return None
        try:
            if db_session:
                from prompt_improver.database.models import RuleMetadata

                query = select(RuleMetadata).where(RuleMetadata.enabled)
                result = await db_session.execute(query)
                rules_metadata = result.scalars().all()
                available_rules = [rule.rule_id for rule in rules_metadata]
            else:
                available_rules = [
                    "clarity_enhancement",
                    "specificity_enhancement",
                    "chain_of_thought",
                    "few_shot_examples",
                    "role_based_prompting",
                    "xml_structure_enhancement",
                ]

            if len(available_rules) >= 2:
                event_bus = await get_ml_event_bus()
                experiment_event = MLEvent(
                    event_type=MLEventType.TRAINING_REQUEST,
                    source="rule_selection_service",
                    data={
                        "operation": "initialize_bandit_experiment",
                        "experiment_name": "rule_optimization",
                        "arms": available_rules,
                        "algorithm": "thompson_sampling",
                    },
                )
                await event_bus.publish(experiment_event)
                import uuid

                self.bandit_experiment_id = f"bandit_exp_{uuid.uuid4().hex[:8]}"
                logger.info(
                    f"Requested bandit experiment initialization for {len(available_rules)} rules"
                )
                return self.bandit_experiment_id
            else:
                logger.warning(
                    "Insufficient rules for bandit optimization, using traditional selection"
                )
                return None
        except Exception as e:
            logger.error(f"Failed to request bandit experiment initialization: {e}")
            self.enable_bandit_optimization = False
            return None

    async def _get_bandit_optimized_rules(
        self,
        prompt_characteristics: dict[str, Any],
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get rules using event-based bandit optimization"""
        if not self.enable_bandit_optimization or self.bandit_experiment_id is None:
            return await self._get_traditional_optimal_rules(
                preferred_rules, db_session, limit
            )
        try:
            event_bus = await get_ml_event_bus()
            selection_event = MLEvent(
                event_type=MLEventType.ANALYSIS_REQUEST,
                source="rule_selection_service",
                data={
                    "operation": "select_bandit_rules",
                    "experiment_id": self.bandit_experiment_id,
                    "context": self._prepare_bandit_context(prompt_characteristics),
                    "limit": limit,
                    "preferred_rules": preferred_rules or [],
                },
            )
            await event_bus.publish(selection_event)

            # Default rules for fallback
            default_rules = [
                "clarity_enhancement",
                "specificity_enhancement",
                "chain_of_thought",
                "few_shot_examples",
                "role_based_prompting",
            ]
            selected_rules = (
                preferred_rules[:limit] if preferred_rules else default_rules[:limit]
            )
            rule_confidence_map = dict.fromkeys(selected_rules, 0.8)
            optimal_rules = []

            if db_session:
                from prompt_improver.database.models import RuleMetadata

                for rule_id in selected_rules:
                    query = select(RuleMetadata).where(RuleMetadata.rule_id == rule_id)
                    result = await db_session.execute(query)
                    rule_metadata = result.scalar_one_or_none()
                    if rule_metadata:
                        optimal_rules.append({
                            "rule_id": rule_metadata.rule_id,
                            "rule_name": rule_metadata.rule_name,
                            "priority": rule_metadata.priority,
                            "enabled": rule_metadata.enabled,
                            "bandit_confidence": rule_confidence_map[rule_id],
                            "selection_method": "bandit",
                        })
            else:
                rule_name_map = {
                    "clarity_enhancement": "Clarity Enhancement Rule",
                    "specificity_enhancement": "Specificity Enhancement Rule",
                    "chain_of_thought": "Chain of Thought Rule",
                    "few_shot_examples": "Few Shot Examples Rule",
                    "role_based_prompting": "Role-Based Prompting Rule",
                    "xml_structure_enhancement": "XML Structure Enhancement Rule",
                }
                for rule_id in selected_rules:
                    optimal_rules.append({
                        "rule_id": rule_id,
                        "rule_name": rule_name_map.get(rule_id, rule_id),
                        "priority": 1.0,
                        "enabled": True,
                        "bandit_confidence": rule_confidence_map[rule_id],
                        "selection_method": "bandit",
                    })

            # Handle preferred rules priority
            if preferred_rules:
                preferred_set = set(preferred_rules)
                optimal_rules.sort(
                    key=lambda x: (
                        x["rule_id"] in preferred_set,
                        x.get("bandit_confidence", 0),
                        x.get("priority", 0),
                    ),
                    reverse=True,
                )

            logger.debug(
                f"Bandit selected {len(optimal_rules)} rules with confidences: {[(r['rule_id'], r.get('bandit_confidence', 0)) for r in optimal_rules]}"
            )
            return optimal_rules[:limit]
        except Exception as e:
            logger.error(f"Error in bandit rule selection: {e}")
            return await self._get_traditional_optimal_rules(
                preferred_rules, db_session, limit
            )

    async def _get_traditional_optimal_rules(
        self,
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get optimal rules using traditional database-based selection"""
        if not db_session:
            return [
                {
                    "rule_id": "clarity_enhancement",
                    "rule_name": "Clarity Enhancement Rule",
                    "selection_method": "default",
                },
                {
                    "rule_id": "specificity_enhancement",
                    "rule_name": "Specificity Enhancement Rule",
                    "selection_method": "default",
                },
            ]

        try:
            from prompt_improver.database.models import RuleMetadata

            query = (
                select(RuleMetadata)
                .where(RuleMetadata.enabled)
                .order_by(desc(RuleMetadata.priority))
                .limit(limit)
            )
            result = await db_session.execute(query)
            rules_metadata = result.scalars().all()
            optimal_rules = []

            for rule in rules_metadata:
                optimal_rules.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "selection_method": "traditional",
                })

            if preferred_rules:
                preferred_set = set(preferred_rules)
                optimal_rules.sort(
                    key=lambda x: (x["rule_id"] in preferred_set, x["priority"]),
                    reverse=True,
                )
            return optimal_rules
        except Exception as e:
            logger.error(f"Error in traditional rule selection: {e}")
            return []

    def _prepare_bandit_context(
        self, prompt_characteristics: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare context for contextual bandit from prompt characteristics"""
        context = {}
        if "length" in prompt_characteristics:
            context["prompt_length"] = min(
                1.0, prompt_characteristics["length"] / 1000.0
            )
        if "complexity" in prompt_characteristics:
            complexity_map = {"low": 0.1, "medium": 0.5, "high": 0.9}
            context["complexity"] = complexity_map.get(
                prompt_characteristics["complexity"], 0.5
            )
        if "readability_score" in prompt_characteristics:
            context["readability"] = min(
                1.0, max(0.0, prompt_characteristics["readability_score"] / 100.0)
            )
        if "sentiment" in prompt_characteristics:
            sentiment_map = {"negative": 0.0, "neutral": 0.5, "positive": 1.0}
            context["sentiment"] = sentiment_map.get(
                prompt_characteristics["sentiment"], 0.5
            )

        context["has_examples"] = float(
            prompt_characteristics.get("has_examples", False)
        )
        context["has_instructions"] = float(
            prompt_characteristics.get("has_instructions", False)
        )
        context["has_context"] = float(prompt_characteristics.get("has_context", False))

        domain = prompt_characteristics.get("domain", "general")
        domain_features = {
            "technical": 0.9,
            "creative": 0.7,
            "analytical": 0.8,
            "conversational": 0.3,
            "general": 0.5,
        }
        context["domain_score"] = domain_features.get(domain, 0.5)
        return context

    async def get_rules_metadata(
        self, enabled_only: bool = True, db_session: AsyncSession | None = None
    ) -> list[dict[str, Any]]:
        """Get rules metadata"""
        if not db_session:
            return []
        try:
            from prompt_improver.database.models import RuleMetadata

            query = select(RuleMetadata)
            if enabled_only:
                query = query.where(RuleMetadata.enabled == True)
            result = await db_session.execute(query)
            rules = result.scalars().all()
            return [
                {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "rule_category": rule.category,
                    "rule_description": rule.description,
                    "enabled": rule.enabled,
                    "priority": rule.priority,
                }
                for rule in rules
            ]
        except Exception as e:
            logger.error(f"Error getting rules metadata: {e}")
            return []

    def get_bandit_experiment_id(self) -> str | None:
        """Get current bandit experiment ID"""
        return self.bandit_experiment_id

    def set_bandit_experiment_id(self, experiment_id: str) -> None:
        """Set bandit experiment ID"""
        self.bandit_experiment_id = experiment_id
