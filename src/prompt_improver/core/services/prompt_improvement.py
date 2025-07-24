"""Prompt Improvement Service
Modern implementation with database integration and ML optimization
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select as sqlmodel_select

from prompt_improver.utils.datetime_utils import aware_utc_now

from ...ml.automl.orchestrator import AutoMLConfig, AutoMLMode, AutoMLOrchestrator
from ...database import get_session_context
from ...database.models import (
    ImprovementSession,
    ImprovementSessionCreate,
    RuleMetadata,
    RulePerformance,
    RulePerformanceCreate,
    UserFeedback,
    UserFeedbackCreate,
)
from ...ml.optimization.algorithms.multi_armed_bandit import (
    BanditAlgorithm,
    BanditConfig,
    MultiarmedBanditFramework,
    create_rule_optimization_bandit,
    intelligent_rule_selection,
)
from ...rule_engine.base import BasePromptRule
from ...performance.testing.ab_testing_service import ABTestingService
from ...ml.core.ml_integration import MLModelService

logger = logging.getLogger(__name__)

class PromptImprovementService:
    """Service for prompt improvement with database integration
    Following 2025 best practices for async operations
    """

    def __init__(
        self, enable_bandit_optimization: bool = True, enable_automl: bool = True
    ):
        self.rules = {}
        self.rule_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.ml_service = MLModelService()

        # Multi-armed bandit for intelligent rule selection
        self.enable_bandit_optimization = enable_bandit_optimization
        self.bandit_framework: MultiarmedBanditFramework | None = None
        self.bandit_experiment_id: str | None = None
        self.ab_testing_service: ABTestingService | None = None

        # AutoML orchestration following 2025 best practices
        self.enable_automl = enable_automl
        self.automl_orchestrator: AutoMLOrchestrator | None = None

        if enable_bandit_optimization:
            # Initialize bandit framework for rule optimization
            bandit_config = BanditConfig(
                algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
                epsilon=0.15,  # Slightly higher exploration for rule selection
                epsilon_decay=0.995,
                min_epsilon=0.05,
                prior_alpha=1.0,
                prior_beta=1.0,
                min_samples_per_arm=10,
                warmup_trials=50,
                enable_regret_tracking=True,
                confidence_level=0.95,
            )
            self.bandit_framework = MultiarmedBanditFramework(bandit_config)
            self.ab_testing_service = ABTestingService()

    async def initialize_automl(self, db_manager) -> None:
        """Initialize AutoML orchestrator with existing components"""
        if not self.enable_automl:
            return

        try:
            # Import here to avoid circular dependencies
            from ..evaluation.experiment_orchestrator import ExperimentOrchestrator
            from ..optimization.rule_optimizer import RuleOptimizer
            from ...performance.analytics.real_time_analytics import RealTimeAnalyticsService
            from ..ml.models.model_manager import ModelManager

            # Initialize AutoML configuration following 2025 best practices
            automl_config = AutoMLConfig(
                study_name="prompt_improver_automl_v2",
                optimization_mode=AutoMLMode.HYPERPARAMETER_OPTIMIZATION,
                enable_real_time_feedback=True,
                enable_early_stopping=True,
                enable_artifact_storage=True,
                enable_drift_detection=True,
                n_trials=50,  # Moderate for production use
                timeout=1800,  # 30 minutes
            )

            # Initialize supporting services
            analytics_service = RealTimeAnalyticsService()
            experiment_orchestrator = ExperimentOrchestrator()
            model_manager = ModelManager()
            rule_optimizer = RuleOptimizer()

            # Create AutoML orchestrator with integrated components
            self.automl_orchestrator = AutoMLOrchestrator(
                config=automl_config,
                db_manager=db_manager,
                rule_optimizer=rule_optimizer,
                experiment_orchestrator=experiment_orchestrator,
                analytics_service=analytics_service,
                model_manager=model_manager,
            )

            logger.info(
                "AutoML orchestrator initialized with 2025 integration patterns"
            )

        except Exception as e:
            logger.error(f"Failed to initialize AutoML orchestrator: {e}")
            self.automl_orchestrator = None

    async def _load_rules_from_database(
        self, db_session: AsyncSession
    ) -> dict[str, BasePromptRule]:
        """Load and instantiate rules from database configuration"""
        try:
            # Get enabled rules from database
            query = (
                select(RuleMetadata)
                .where(RuleMetadata.enabled == True)
                .order_by(RuleMetadata.priority.desc())
            )

            result = await db_session.execute(query)
            rule_configs = result.scalars().all()

            rules = {}
            for config in rule_configs:
                # Dynamic rule instantiation based on rule_id
                rule_class = self._get_rule_class(config.rule_id)
                if rule_class:
                    rule_instance = rule_class()
                    # Set basic attributes if they exist
                    if hasattr(rule_instance, "rule_id"):
                        rule_instance.rule_id = config.rule_id
                    if hasattr(rule_instance, "priority"):
                        rule_instance.priority = config.priority

                    # Apply database-stored parameters
                    if config.default_parameters:
                        # Handle both string and dict types for compatibility
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
            from ..rule_engine.rules.clarity import ClarityRule

            return ClarityRule
        except ImportError:
            logger.warning("ClarityRule not found, using base rule")
            return BasePromptRule

    def _import_chain_of_thought_rule(self):
        """Import chain of thought rule class"""
        try:
            from ..rule_engine.rules.chain_of_thought import ChainOfThoughtRule

            return ChainOfThoughtRule
        except ImportError:
            logger.warning("ChainOfThoughtRule not found, using base rule")
            return BasePromptRule

    def _import_few_shot_rule(self):
        """Import few shot example rule class"""
        try:
            from ..rule_engine.rules.few_shot_examples import FewShotExampleRule

            return FewShotExampleRule
        except ImportError:
            logger.warning("FewShotExampleRule not found, using base rule")
            return BasePromptRule

    def _import_role_based_rule(self):
        """Import role based prompting rule class"""
        try:
            from ..rule_engine.rules.role_based_prompting import RoleBasedPromptingRule

            return RoleBasedPromptingRule
        except ImportError:
            logger.warning("RoleBasedPromptingRule not found, using base rule")
            return BasePromptRule

    def _import_xml_structure_rule(self):
        """Import XML structure rule class"""
        try:
            from ..rule_engine.rules.xml_structure_enhancement import XMLStructureRule

            return XMLStructureRule
        except ImportError:
            logger.warning("XMLStructureRule not found, using base rule")
            return BasePromptRule

    def _import_specificity_rule(self):
        """Import specificity rule class"""
        try:
            from ..rule_engine.rules.specificity import SpecificityRule

            return SpecificityRule
        except ImportError:
            logger.warning("SpecificityRule not found, using base rule")
            return BasePromptRule

    def _get_fallback_rules(self) -> dict[str, BasePromptRule]:
        """Fallback rules when database loading fails"""
        try:
            from ..rule_engine.rules.clarity import ClarityRule
            from ..rule_engine.rules.specificity import SpecificityRule

            rules = {
                "clarity_enhancement": ClarityRule(),
                "specificity_enhancement": SpecificityRule(),
            }
        except ImportError:
            # Ultimate fallback - empty rule set
            rules = {}
        return rules

    async def get_active_rules(
        self, db_session: AsyncSession | None = None
    ) -> dict[str, BasePromptRule]:
        """Get active rules with caching"""
        cache_key = "active_rules"

        # Check cache
        if cache_key in self.rule_cache:
            cached_rules, timestamp = self.rule_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_rules

        # Load from database using provided session or create new one
        if db_session:
            rules = await self._load_rules_from_database(db_session)
        else:
            async with get_session_context() as session:
                rules = await self._load_rules_from_database(session)

        # Cache results
        self.rule_cache[cache_key] = (rules, time.time())
        return rules

    async def get_active_rules_standalone(self) -> dict[str, BasePromptRule]:
        """Get active rules without requiring external database session

        This method creates its own database session following 2025 best practices.
        Useful for initialization and testing scenarios.
        """
        return await self.get_active_rules()

    async def improve_prompt(
        self,
        prompt: str,
        user_context: dict[str, Any] | None = None,
        session_id: str | None = None,
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Improve a prompt using data-driven rule selection"""
        start_time = time.time()

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # 1. Analyze prompt characteristics
        prompt_characteristics = await self._analyze_prompt(prompt)

        # 2. Get optimal rules from database based on historical performance
        optimal_rules = await self._get_optimal_rules(
            prompt_characteristics=prompt_characteristics,
            preferred_rules=preferred_rules,
            db_session=db_session,
        )

        # 3. Apply rules in order of effectiveness
        improved_prompt = prompt
        applied_rules = []
        performance_data = []

        for rule_info in optimal_rules:
            rule_id = rule_info.get("rule_id", rule_info.get("id", "unknown"))

            if rule_id in self.rules:
                rule = self.rules[rule_id]

                # Check if rule applies (if the rule has a check method)
                if hasattr(rule, "check"):
                    check_result = rule.check(improved_prompt)
                    if not check_result.applies:
                        continue

                rule_start = time.time()

                # Apply rule (if the rule has an apply method)
                if hasattr(rule, "apply"):
                    try:
                        result = rule.apply(improved_prompt)
                        if hasattr(result, "success") and result.success:
                            # Calculate improvement metrics
                            before_metrics = self._calculate_metrics(improved_prompt)
                            after_metrics = self._calculate_metrics(
                                result.improved_prompt
                            )
                            improvement_score = self._calculate_improvement_score(
                                before_metrics, after_metrics
                            )

                            rule_execution_time = int((time.time() - rule_start) * 1000)
                            confidence = getattr(result, "confidence", 0.8)

                            # Store performance data
                            performance_data.append({
                                "rule_id": rule_id,
                                "rule_name": rule_info.get("rule_name", rule_id),
                                "improvement_score": improvement_score,
                                "confidence": confidence,
                                "execution_time_ms": rule_execution_time,
                                "before_metrics": before_metrics,
                                "after_metrics": after_metrics,
                                "prompt_characteristics": prompt_characteristics,
                            })

                            improved_prompt = result.improved_prompt
                            applied_rules.append({
                                "rule_id": rule_id,
                                "rule_name": rule_info.get("rule_name", rule_id),
                                "improvement_score": improvement_score,
                                "confidence": confidence,
                            })
                    except Exception as e:
                        print(f"Error applying rule {rule_id}: {e}")
                        continue

        total_time = int((time.time() - start_time) * 1000)

        # Update bandit with performance rewards
        if self.enable_bandit_optimization and applied_rules:
            await self.update_bandit_rewards(applied_rules, db_session)

        # Store session information
        if db_session:
            await self._store_session(
                session_id=session_id,
                original_prompt=prompt,
                final_prompt=improved_prompt,
                rules_applied=applied_rules,
                user_context=user_context,
                db_session=db_session,
            )

        return {
            "original_prompt": prompt,
            "improved_prompt": improved_prompt,
            "applied_rules": applied_rules,
            "processing_time_ms": total_time,
            "session_id": session_id,
            "improvement_summary": self._generate_improvement_summary(applied_rules),
            "confidence_score": self._calculate_overall_confidence(applied_rules),
            "performance_data": performance_data,
        }

    async def _get_optimal_rules(
        self,
        prompt_characteristics: dict[str, Any],
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get optimal rules using multi-armed bandit optimization"""
        # Initialize bandit experiment if not already done
        if self.enable_bandit_optimization and self.bandit_experiment_id is None:
            await self._initialize_bandit_experiment(db_session)

        # Use bandit for intelligent rule selection if enabled
        if self.enable_bandit_optimization and self.bandit_experiment_id:
            return await self._get_bandit_optimized_rules(
                prompt_characteristics, preferred_rules, db_session, limit
            )

        # Fallback to traditional rule selection
        return await self._get_traditional_optimal_rules(
            prompt_characteristics, preferred_rules, db_session, limit
        )

    async def _initialize_bandit_experiment(
        self, db_session: AsyncSession | None = None
    ) -> None:
        """Initialize bandit experiment for rule optimization"""
        if not self.enable_bandit_optimization or self.bandit_framework is None:
            return

        try:
            # Get all available rules from database
            if db_session:
                query = select(RuleMetadata).where(RuleMetadata.enabled == True)
                result = await db_session.execute(query)
                rules_metadata = result.scalars().all()
                available_rules = [rule.rule_id for rule in rules_metadata]
            else:
                # Fallback to default rules
                available_rules = [
                    "clarity_enhancement",
                    "specificity_enhancement",
                    "chain_of_thought",
                    "few_shot_examples",
                    "role_based_prompting",
                    "xml_structure_enhancement",
                ]

            if len(available_rules) >= 2:
                # Create bandit experiment for rule selection
                self.bandit_experiment_id = (
                    await self.bandit_framework.create_experiment(
                        experiment_name="rule_optimization",
                        arms=available_rules,
                        algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
                    )
                )

                logger.info(
                    f"Initialized bandit experiment for {len(available_rules)} rules"
                )
            else:
                logger.warning(
                    "Insufficient rules for bandit optimization, using traditional selection"
                )

        except Exception as e:
            logger.error(f"Failed to initialize bandit experiment: {e}")
            self.enable_bandit_optimization = False

    async def _get_bandit_optimized_rules(
        self,
        prompt_characteristics: dict[str, Any],
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get rules using multi-armed bandit optimization"""
        if (
            not self.enable_bandit_optimization
            or self.bandit_framework is None
            or self.bandit_experiment_id is None
        ):
            return await self._get_traditional_optimal_rules(
                prompt_characteristics, preferred_rules, db_session, limit
            )

        try:
            # Prepare context for contextual bandit
            context = self._prepare_bandit_context(prompt_characteristics)

            # Select rules using bandit algorithm
            selected_rules = []
            rule_confidence_map = {}

            # Select up to 'limit' rules, avoiding duplicates
            for _ in range(min(limit, 10)):  # Max 10 iterations to avoid infinite loop
                # Select arm using bandit
                arm_result = await self.bandit_framework.select_arm(
                    self.bandit_experiment_id, context
                )

                rule_id = arm_result.arm_id
                if rule_id not in rule_confidence_map:
                    rule_confidence_map[rule_id] = arm_result.confidence
                    selected_rules.append(rule_id)

                # Stop if we have enough unique rules
                if len(selected_rules) >= limit:
                    break

            # Get rule metadata and create results
            optimal_rules = []
            if db_session:
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
                # Fallback without database
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

            # Honor preferred rules if specified
            if preferred_rules:
                # Ensure preferred rules are included, even if bandit didn't select them
                preferred_set = set(preferred_rules)
                selected_set = set(rule["rule_id"] for rule in optimal_rules)

                for preferred_rule in preferred_rules:
                    if (
                        preferred_rule not in selected_set
                        and len(optimal_rules) < limit
                    ):
                        # Add preferred rule that wasn't selected by bandit
                        if db_session:
                            query = select(RuleMetadata).where(
                                RuleMetadata.rule_id == preferred_rule
                            )
                            result = await db_session.execute(query)
                            rule_metadata = result.scalar_one_or_none()

                            if rule_metadata:
                                optimal_rules.append({
                                    "rule_id": rule_metadata.rule_id,
                                    "rule_name": rule_metadata.rule_name,
                                    "priority": rule_metadata.priority,
                                    "enabled": rule_metadata.enabled,
                                    "bandit_confidence": 0.5,  # Default confidence for preferred rules
                                    "selection_method": "preferred",
                                })

                # Sort to prioritize preferred rules
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
            # Fallback to traditional method
            return await self._get_traditional_optimal_rules(
                prompt_characteristics, preferred_rules, db_session, limit
            )

    async def _get_traditional_optimal_rules(
        self,
        prompt_characteristics: dict[str, Any],
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get optimal rules using traditional database-based selection"""
        if not db_session:
            # Fallback to default rules if no database session
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
            # Query for high-performing rules
            query = (
                select(RuleMetadata)
                .where(RuleMetadata.enabled == True)
                .order_by(RuleMetadata.priority.desc())
                .limit(limit)
            )

            result = await db_session.execute(query)
            rules_metadata = result.scalars().all()

            # Convert to dict format
            optimal_rules = []
            for rule in rules_metadata:
                optimal_rules.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "selection_method": "traditional",
                })

            # If preferred rules are specified, prioritize them
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

        # Extract numeric features for bandit context
        if "length" in prompt_characteristics:
            context["prompt_length"] = min(
                1.0, prompt_characteristics["length"] / 1000.0
            )  # Normalize to 0-1

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
            # Convert sentiment to numeric value
            sentiment_map = {"negative": 0.0, "neutral": 0.5, "positive": 1.0}
            context["sentiment"] = sentiment_map.get(
                prompt_characteristics["sentiment"], 0.5
            )

        # Boolean features
        context["has_examples"] = float(
            prompt_characteristics.get("has_examples", False)
        )
        context["has_instructions"] = float(
            prompt_characteristics.get("has_instructions", False)
        )
        context["has_context"] = float(prompt_characteristics.get("has_context", False))

        # Domain features (simple mapping)
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

    async def update_bandit_rewards(
        self,
        applied_rules: list[dict[str, Any]],
        db_session: AsyncSession | None = None,
    ) -> None:
        """Update bandit with rewards from rule application results"""
        if (
            not self.enable_bandit_optimization
            or self.bandit_framework is None
            or not self.bandit_experiment_id
        ):
            return

        try:
            for rule_info in applied_rules:
                rule_id = rule_info.get("rule_id")
                improvement_score = rule_info.get("improvement_score", 0.0)
                confidence = rule_info.get("confidence", 0.0)

                if rule_id and improvement_score is not None:
                    # Normalize improvement score to 0-1 range for bandit
                    # Typical improvement scores might range from -0.5 to 1.0
                    normalized_reward = max(
                        0.0, min(1.0, (improvement_score + 0.5) / 1.5)
                    )

                    # Weight by confidence if available
                    if confidence > 0:
                        weighted_reward = normalized_reward * confidence
                    else:
                        weighted_reward = normalized_reward * 0.8  # Default confidence

                    # Update bandit with reward - create proper ArmResult
                    from ..optimization.multi_armed_bandit import ArmResult

                    arm_result = ArmResult(
                        arm_id=rule_id,
                        reward=weighted_reward,
                        context=None,
                        metadata={
                            "improvement_score": improvement_score,
                            "confidence": confidence,
                        },
                    )

                    await self.bandit_framework.update_reward(
                        self.bandit_experiment_id, arm_result, weighted_reward
                    )

                    logger.debug(
                        f"Updated bandit reward for {rule_id}: {weighted_reward:.3f} (from improvement: {improvement_score:.3f})"
                    )

        except Exception as e:
            logger.error(f"Error updating bandit rewards: {e}")

    async def get_bandit_performance_summary(self) -> dict[str, Any]:
        """Get performance summary of the bandit optimization"""
        if (
            not self.enable_bandit_optimization
            or self.bandit_framework is None
            or not self.bandit_experiment_id
        ):
            return {"status": "disabled", "message": "Bandit optimization not enabled"}

        try:
            summary = self.bandit_framework.get_experiment_summary(
                self.bandit_experiment_id
            )

            # Add interpretation
            total_trials = summary.get("total_trials", 0)
            best_arm = summary.get("best_arm", "unknown")
            average_reward = summary.get("average_reward", 0.0)

            interpretation = []

            if total_trials < 50:
                interpretation.append(
                    "Bandit is in exploration phase - collecting data on rule performance"
                )
            elif total_trials < 200:
                interpretation.append(
                    "Bandit is learning rule preferences - some patterns emerging"
                )
            else:
                interpretation.append(
                    f"Bandit has converged - {best_arm} is the preferred rule"
                )

            if average_reward > 0.7:
                interpretation.append("High average reward - rules are performing well")
            elif average_reward > 0.5:
                interpretation.append(
                    "Moderate average reward - some room for improvement"
                )
            else:
                interpretation.append("Low average reward - may need rule tuning")

            return {
                "status": "active",
                "summary": summary,
                "interpretation": interpretation,
                "recommendations": [
                    f"Continue using {best_arm} for similar prompts",
                    "Monitor performance trends for optimization opportunities",
                ],
            }

        except Exception as e:
            logger.error(f"Error getting bandit performance: {e}")
            return {"status": "error", "error": str(e)}

    async def _analyze_prompt(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt characteristics for rule selection"""
        return {
            "type": self._classify_prompt_type(prompt),
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "complexity": self._calculate_complexity(prompt),
            "clarity_score": self._assess_clarity(prompt),
            "specificity_score": self._assess_specificity(prompt),
            "has_questions": "?" in prompt,
            "has_examples": "example" in prompt.lower()
            or "for instance" in prompt.lower(),
        }

    def _classify_prompt_type(self, prompt: str) -> str:
        """Classify the type of prompt"""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["explain", "describe", "what is"]):
            return "explanation"
        if any(word in prompt_lower for word in ["create", "generate", "write"]):
            return "creation"
        if any(word in prompt_lower for word in ["analyze", "evaluate", "compare"]):
            return "analysis"
        if any(word in prompt_lower for word in ["help", "how to", "guide"]):
            return "instruction"
        return "general"

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score (0-1)"""
        words = prompt.split()

        # Factors: length, vocabulary complexity, sentence structure
        length_score = min(len(words) / 50, 1.0)  # Normalize to 50 words
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        vocab_score = min(avg_word_length / 8, 1.0)  # Normalize to 8 chars

        return (length_score + vocab_score) / 2

    def _assess_clarity(self, prompt: str) -> float:
        """Assess prompt clarity (0-1, higher is clearer)"""
        # Simple heuristics for clarity
        clarity_score = 1.0

        # Penalize very short or very long prompts
        word_count = len(prompt.split())
        if word_count < 5:
            clarity_score -= 0.3
        elif word_count > 100:
            clarity_score -= 0.2

        # Reward specific language
        if any(word in prompt.lower() for word in ["specific", "detailed", "exactly"]):
            clarity_score += 0.1

        # Penalize vague language
        if any(
            word in prompt.lower() for word in ["something", "anything", "whatever"]
        ):
            clarity_score -= 0.2

        return max(0.0, min(1.0, clarity_score))

    def _assess_specificity(self, prompt: str) -> float:
        """Assess prompt specificity (0-1, higher is more specific)"""
        specificity_score = 0.5  # Base score

        # Look for specific indicators
        specific_indicators = ["when", "where", "how", "why", "which", "what kind"]
        for indicator in specific_indicators:
            if indicator in prompt.lower():
                specificity_score += 0.1

        # Look for examples or constraints
        if any(
            word in prompt.lower() for word in ["example", "constraint", "requirement"]
        ):
            specificity_score += 0.2

        # Penalize vague language
        vague_words = ["maybe", "possibly", "might", "could be", "general"]
        for word in vague_words:
            if word in prompt.lower():
                specificity_score -= 0.1

        return max(0.0, min(1.0, specificity_score))

    def _calculate_metrics(self, prompt: str) -> dict[str, float]:
        """Calculate metrics for a prompt"""
        return {
            "clarity": self._assess_clarity(prompt),
            "specificity": self._assess_specificity(prompt),
            "completeness": self._assess_completeness(prompt),
            "structure": self._assess_structure(prompt),
        }

    def _assess_completeness(self, prompt: str) -> float:
        """Assess if prompt provides complete information"""
        # Simple heuristic: longer prompts tend to be more complete
        word_count = len(prompt.split())
        return min(word_count / 30, 1.0)  # Normalize to 30 words

    def _assess_structure(self, prompt: str) -> float:
        """Assess prompt structure quality"""
        structure_score = 0.5

        # Reward punctuation use
        if "." in prompt:
            structure_score += 0.2
        if "," in prompt:
            structure_score += 0.1

        # Reward proper capitalization
        if prompt and prompt[0].isupper():
            structure_score += 0.1

        # Penalize all caps
        if prompt.isupper():
            structure_score -= 0.3

        return max(0.0, min(1.0, structure_score))

    def _calculate_improvement_score(
        self, before: dict[str, float], after: dict[str, float]
    ) -> float:
        """Calculate improvement score based on metrics"""
        weights = {
            "clarity": 0.3,
            "specificity": 0.3,
            "completeness": 0.2,
            "structure": 0.2,
        }

        total_improvement = 0
        for metric, weight in weights.items():
            before_score = before.get(metric, 0)
            after_score = after.get(metric, 0)
            if before_score > 0:
                improvement = (after_score - before_score) / before_score
            else:
                improvement = after_score
            total_improvement += improvement * weight

        return max(0, min(1, total_improvement))

    def _generate_improvement_summary(
        self, applied_rules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate a summary of improvements made"""
        if not applied_rules:
            return {
                "total_rules_applied": 0,
                "average_confidence": 0,
                "improvement_areas": [],
            }

        total_confidence = sum(rule.get("confidence", 0) for rule in applied_rules)
        avg_confidence = total_confidence / len(applied_rules)

        improvement_areas = [rule["rule_name"] for rule in applied_rules]

        return {
            "total_rules_applied": len(applied_rules),
            "average_confidence": avg_confidence,
            "improvement_areas": improvement_areas,
            "estimated_improvement": sum(
                rule.get("improvement_score", 0) for rule in applied_rules
            ),
        }

    def _calculate_overall_confidence(
        self, applied_rules: list[dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score"""
        if not applied_rules:
            return 0.0

        confidences = [rule.get("confidence", 0) for rule in applied_rules]
        return sum(confidences) / len(confidences)

    async def _store_session(
        self,
        session_id: str,
        original_prompt: str,
        final_prompt: str,
        rules_applied: list[dict[str, Any]],
        user_context: dict[str, Any] | None,
        db_session: AsyncSession,
    ):
        """Store improvement session in database"""
        try:
            # Extract rule IDs from rules_applied for the model
            rule_ids = (
                [rule.get("rule_id", "unknown") for rule in rules_applied]
                if rules_applied
                else None
            )

            session_data = ImprovementSessionCreate(
                session_id=session_id,
                original_prompt=original_prompt,
                final_prompt=final_prompt,
                rules_applied=rule_ids,
                user_context=user_context,
            )

            db_session.add(ImprovementSession(**session_data.model_dump()))
            await db_session.commit()

        except Exception as e:
            print(f"Error storing session: {e}")
            await db_session.rollback()

    async def store_performance_metrics(
        self, performance_data: list[dict[str, Any]], db_session: AsyncSession
    ):
        """Store rule performance metrics"""
        try:
            for data in performance_data:
                perf_record = RulePerformanceCreate(
                    session_id=data.get("session_id", "unknown"),
                    rule_id=data["rule_id"],
                    improvement_score=data["improvement_score"],
                    execution_time_ms=data["execution_time_ms"],
                    confidence_level=data["confidence"],
                    parameters_used=data.get("parameters_used"),
                )

                db_session.add(RulePerformance(**perf_record.dict()))

            await db_session.commit()

        except Exception as e:
            print(f"Error storing performance metrics: {e}")
            await db_session.rollback()

    async def store_user_feedback(
        self,
        session_id: str,
        rating: int,
        feedback_text: str | None,
        improvement_areas: list[str] | None,
        db_session: AsyncSession,
    ) -> UserFeedback:
        """Store user feedback"""
        try:
            # Get session information to link feedback
            session_query = select(ImprovementSession).where(
                ImprovementSession.session_id == session_id
            )
            session_result = await db_session.execute(session_query)
            session_record = session_result.scalar_one_or_none()

            if not session_record:
                raise ValueError(f"Session {session_id} not found")

            feedback_data = UserFeedbackCreate(
                session_id=session_id,
                rating=rating,
                feedback_text=feedback_text,
                improvement_areas=improvement_areas,
            )

            feedback = UserFeedback(**feedback_data.dict())
            db_session.add(feedback)
            await db_session.commit()
            await db_session.refresh(feedback)

            return feedback

        except Exception as e:
            print(f"Error storing user feedback: {e}")
            await db_session.rollback()
            raise

    async def get_rules_metadata(
        self, enabled_only: bool = True, db_session: AsyncSession | None = None
    ) -> list[dict[str, Any]]:
        """Get rules metadata"""
        if not db_session:
            return []

        try:
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
            print(f"Error getting rules metadata: {e}")
            return []

    async def trigger_optimization(self, feedback_id: int, db_session: AsyncSession):
        """Trigger ML optimization based on feedback"""
        from ..ml.core.ml_integration import get_ml_service

        # Get feedback data and related performance metrics
        stmt = sqlmodel_select(UserFeedback).where(UserFeedback.id == feedback_id)
        result = await db_session.execute(stmt)
        feedback = result.scalar_one_or_none()

        if not feedback:
            logger.warning(f"Feedback {feedback_id} not found")
            return {"status": "error", "message": f"Feedback {feedback_id} not found"}

        # Get related performance data for training
        perf_stmt = (
            select(RulePerformance, RuleMetadata.default_parameters)
            .join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id)
            .where(RulePerformance.created_at.isnot(None))
        )  # Remove session_id filter since it may not exist

        result = await db_session.execute(perf_stmt)
        performance_data = result.fetchall()

        if not performance_data:
            logger.warning(
                f"No performance data found for session {feedback.session_id}"
            )
            return {"status": "error", "message": "No performance data available"}

        # Prepare training data
        features = []
        effectiveness_scores = []

        for row in performance_data:
            # Extract features from rule parameters and performance
            # row is a Row object with named access
            rule_perf = row.RulePerformance  # RulePerformance object
            params = row.default_parameters or {}  # default_parameters dict
            features.append([
                rule_perf.improvement_score or 0,
                rule_perf.execution_time_ms or 0,
                params.get("weight", 1.0),
                params.get("priority", 5),
                len(params),  # Number of parameters
                1.0 if params.get("active", True) else 0.0,  # Active status
            ])
            effectiveness_scores.append(
                rule_perf.confidence_level or (feedback.rating / 5.0)
            )

        # Run optimization if we have enough data
        if len(features) >= 10:
            ml_service = await get_ml_service()
            training_data = {
                "features": features,
                "effectiveness_scores": effectiveness_scores,
            }

            optimization_result = await ml_service.optimize_rules(
                training_data,
                db_session,
                rule_ids=None,  # Optimize all rules
            )

            if optimization_result.get("status") == "success":
                logger.info(
                    f"ML optimization completed successfully: {optimization_result.get('best_score', 'N/A')}"
                )

                # Store optimization trigger in database
                await self._store_optimization_trigger(
                    db_session,
                    feedback_id,
                    optimization_result.get("model_id"),
                    len(features),
                    optimization_result.get("best_score", 0),
                )

                return {
                    "status": "success",
                    "message": "ML optimization completed",
                    "performance_score": optimization_result.get("best_score"),
                    "training_samples": len(features),
                    "model_id": optimization_result.get("model_id"),
                }
            logger.error(f"ML optimization failed: {optimization_result.get('error')}")
            return {
                "status": "error",
                "message": f"Optimization failed: {optimization_result.get('error')}",
            }
        logger.warning(
            f"Insufficient data for optimization: {len(features)} samples (minimum: 10)"
        )
        return {
            "status": "insufficient_data",
            "message": f"Need at least 10 samples for optimization, found {len(features)}",
        }

    async def run_ml_optimization(
        self, rule_ids: list[str] | None, db_session: AsyncSession
    ):
        """Run ML optimization for specified rules with automatic real+synthetic data"""
        from ..ml.core.ml_integration import get_ml_service
        from ..ml.core.training_data_loader import TrainingDataLoader

        # Use the new unified training data loader
        data_loader = TrainingDataLoader(
            real_data_priority=True,  # Prioritize real data
            min_samples=20,
            lookback_days=30,
            synthetic_ratio=0.3,  # Allow up to 30% synthetic data
        )

        # Load training data automatically combining real and synthetic
        training_data = await data_loader.load_training_data(
            db_session=db_session,
            rule_ids=rule_ids
        )

        # Check if we have sufficient data
        if not training_data["validation"]["is_valid"]:
            logger.warning(
                f"Insufficient training data: {training_data['metadata']['total_samples']} samples "
                f"({training_data['metadata']['real_samples']} real, "
                f"{training_data['metadata']['synthetic_samples']} synthetic)"
            )
            return {
                "status": "insufficient_data",
                "message": f"Need at least 20 samples for optimization",
                "samples_found": training_data['metadata']['total_samples'],
                "real_samples": training_data['metadata']['real_samples'],
                "synthetic_samples": training_data['metadata']['synthetic_samples'],
                "warnings": training_data["validation"]["warnings"],
            }

        # Extract features and labels from the unified training data
        features = training_data["features"]
        effectiveness_scores = training_data["labels"]

        logger.info(
            f"Using {len(features)} training samples for ML optimization: "
            f"{training_data['metadata']['real_samples']} real, "
            f"{training_data['metadata']['synthetic_samples']} synthetic "
            f"(ratio: {training_data['metadata']['synthetic_ratio']:.1%} synthetic)"
        )

        # Run ML optimization
        ml_service = await get_ml_service()
        ml_training_data = {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
            "metadata": training_data["metadata"],
        }

        optimization_result = await ml_service.optimize_rules(
            ml_training_data, db_session, rule_ids=rule_ids
        )

        if optimization_result.get("status") == "success":
            # Also try ensemble optimization if we have enough data
            if len(features) >= 50:
                ensemble_result = await ml_service.optimize_ensemble_rules(
                    ml_training_data, db_session
                )

                if ensemble_result.get("status") == "success":
                    logger.info(
                        f"Ensemble optimization completed: {ensemble_result.get('ensemble_score', 'N/A')}"
                    )
                    optimization_result["ensemble"] = ensemble_result

            # Store optimization results
            await self._store_ml_optimization_results(
                db_session, rule_ids or [], optimization_result, len(features)
            )

            logger.info(
                f"ML optimization completed successfully: {optimization_result.get('best_score', 'N/A')}"
            )
            return optimization_result
        logger.error(f"ML optimization failed: {optimization_result.get('error')}")
        return optimization_result

    async def discover_patterns(
        self, min_effectiveness: float, min_support: int, db_session: AsyncSession
    ):
        """Discover new patterns from successful improvements"""
        from ..ml.core.ml_integration import get_ml_service

        # Use ML service for pattern discovery
        ml_service = await get_ml_service()
        result = await ml_service.discover_patterns(
            db_session=db_session,
            min_effectiveness=min_effectiveness,
            min_support=min_support,
        )

        if (
            result.get("status") == "success"
            and result.get("patterns_discovered", 0) > 0
        ):
            logger.info(
                f"Pattern discovery completed: {result['patterns_discovered']} patterns found"
            )

            # Create A/B experiments for top patterns
            await self._create_ab_experiments_from_patterns(
                db_session,
                result.get("patterns", [])[:3],  # Top 3 patterns
            )

            # Store pattern discovery results
            await self._store_pattern_discovery_results(db_session, result)

            return {
                "status": "success",
                "patterns_discovered": result.get("patterns_discovered", 0),
                "patterns": result.get("patterns", []),
                "total_analyzed": result.get("total_analyzed", 0),
                "processing_time_ms": result.get("processing_time_ms", 0),
            }
        if result.get("status") == "insufficient_data":
            logger.warning(
                f"Insufficient data for pattern discovery: {result.get('message')}"
            )
            return result
        logger.error(f"Pattern discovery failed: {result.get('error')}")
        return result

    # Helper methods for Phase 3 implementation
    async def _store_optimization_trigger(
        self,
        db_session: AsyncSession,
        feedback_id: int,
        model_id: str | None,
        training_samples: int,
        performance_score: float,
    ):
        """Store optimization trigger event for tracking."""
        try:
            # Store in optimization log or update feedback record
            stmt = sqlmodel_select(UserFeedback).where(UserFeedback.id == feedback_id)
            result = await db_session.execute(stmt)
            feedback = result.scalar_one_or_none()

            if feedback:
                # Update feedback with optimization info
                feedback.ml_optimized = True
                feedback.model_id = model_id
                db_session.add(feedback)
                await db_session.commit()

        except Exception as e:
            logger.error(f"Failed to store optimization trigger: {e}")
            await db_session.rollback()

    async def _store_ml_optimization_results(
        self,
        db_session: AsyncSession,
        rule_ids: list[str],
        optimization_result: dict[str, Any],
        training_samples: int,
    ):
        """Store ML optimization results for analytics."""
        try:
            from ..database.models import MLModelPerformance

            # Store model performance record
            performance_record = MLModelPerformance(
                model_id=optimization_result.get("model_id", "unknown"),
                performance_score=optimization_result.get("best_score", 0),
                accuracy=optimization_result.get("accuracy", 0),
                precision=optimization_result.get("precision", 0),
                recall=optimization_result.get("recall", 0),
                training_samples=training_samples,
                created_at=aware_utc_now(),
            )

            db_session.add(performance_record)
            await db_session.commit()

        except Exception as e:
            logger.error(f"Failed to store ML optimization results: {e}")
            await db_session.rollback()

    async def _create_ab_experiments_from_patterns(
        self, db_session: AsyncSession, patterns: list[dict[str, Any]]
    ):
        """Create A/B experiments from discovered patterns."""
        try:
            from ..database.models import ABExperiment

            for i, pattern in enumerate(patterns):
                # Create experiment name
                experiment_name = f"Pattern_{i + 1}_Effectiveness_{pattern.get('avg_effectiveness', 0):.2f}"

                # Check if similar experiment already exists
                existing_stmt = sqlmodel_select(ABExperiment).where(
                    ABExperiment.experiment_name.like(f"Pattern_{i + 1}_%"),
                    ABExperiment.status == "running",
                )
                result = await db_session.execute(existing_stmt)
                existing = result.scalar_one_or_none()

                if not existing:
                    # Create new A/B experiment
                    experiment = ABExperiment(
                        experiment_name=experiment_name,
                        control_rules={"baseline": "current_rules"},
                        treatment_rules={"optimized": pattern.get("parameters", {})},
                        status="running",
                        started_at=aware_utc_now(),
                    )

                    db_session.add(experiment)

            await db_session.commit()

        except Exception as e:
            logger.error(f"Failed to create A/B experiments: {e}")
            await db_session.rollback()

    async def _store_pattern_discovery_results(
        self, db_session: AsyncSession, discovery_result: dict[str, Any]
    ):
        """Store pattern discovery results for future reference."""
        try:
            # Note: DiscoveredPattern model doesn't exist in current schema
            # This is a placeholder for future implementation
            logger.info(
                f"Pattern discovery completed with {discovery_result.get('patterns_discovered', 0)} patterns"
            )

        except Exception as e:
            logger.error(f"Failed to store pattern discovery results: {e}")
            await db_session.rollback()

    async def start_automl_optimization(
        self,
        optimization_target: str = "rule_effectiveness",
        experiment_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start AutoML optimization integrating Optuna with existing A/B testing

        Args:
            optimization_target: What to optimize ('rule_effectiveness', 'user_satisfaction', etc.)
            experiment_config: Configuration for A/B testing integration

        Returns:
            Dictionary with optimization results and metadata
        """
        if not self.automl_orchestrator:
            return {
                "error": "AutoML orchestrator not initialized",
                "suggestion": "Call initialize_automl() first",
            }

        try:
            logger.info(f"Starting AutoML optimization for {optimization_target}")

            # Use the orchestrator to start optimization
            result = await self.automl_orchestrator.start_optimization(
                optimization_target=optimization_target,
                experiment_config=experiment_config or {},
            )

            # If optimization succeeded, create A/B test for best configuration
            if result.get("status") == "completed" and result.get("best_params"):
                if self.ab_testing_service:
                    # Create A/B test to validate optimized parameters
                    ab_config = {
                        "experiment_name": f"automl_validation_{optimization_target}",
                        "treatment_config": result["best_params"],
                        "sample_size": 200,
                        "confidence_level": 0.95,
                    }

                    ab_result = await self.ab_testing_service.create_experiment(
                        **ab_config
                    )
                    result["ab_test_id"] = ab_result.get("experiment_id")
                    logger.info(
                        f"Created A/B test {ab_result.get('experiment_id')} for AutoML validation"
                    )

            return result

        except Exception as e:
            logger.error(f"AutoML optimization failed: {e}")
            return {"error": str(e), "optimization_target": optimization_target}

    async def get_automl_status(self) -> dict[str, Any]:
        """Get current AutoML optimization status"""
        if not self.automl_orchestrator:
            return {"status": "not_initialized"}

        return await self.automl_orchestrator.get_optimization_status()

    async def stop_automl_optimization(self) -> dict[str, Any]:
        """Stop current AutoML optimization"""
        if not self.automl_orchestrator:
            return {"status": "not_initialized"}

        return await self.automl_orchestrator.stop_optimization()
