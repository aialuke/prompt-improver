"""Prompt Improvement Service
Modern implementation with database integration and ML optimization
"""

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select as sqlmodel_select

from ..database.models import (
    ImprovementSession,
    ImprovementSessionCreate,
    RuleMetadata,
    RulePerformance,
    RulePerformanceCreate,
    UserFeedback,
    UserFeedbackCreate,
)
from ..rule_engine.base import BasePromptRule
from .ml_integration import MLModelService

logger = logging.getLogger(__name__)


class PromptImprovementService:
    """Service for prompt improvement with database integration
    Following 2025 best practices for async operations
    """

    def __init__(self):
        self.rules = self._load_rules()
        self.ml_service = MLModelService()

    def _load_rules(self) -> dict[str, BasePromptRule]:
        """Load available rules from rule registry"""
        # For now, load the basic rules we have
        # In production, this would dynamically load from the database
        try:
            from ..rule_engine.rules.clarity import ClarityRule
            from ..rule_engine.rules.specificity import SpecificityRule

            rules = {
                "clarity_rule": ClarityRule(),
                "specificity_rule": SpecificityRule(),
            }
        except ImportError:
            # Fallback if rule modules don't exist yet
            rules = {}

        return rules

    async def improve_prompt(
        self,
        prompt: str,
        user_context: dict[str, Any] | None = None,
        session_id: str | None = None,
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession = None,
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
        db_session: AsyncSession = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get optimal rules based on historical performance"""
        if not db_session:
            # Fallback to default rules if no database session
            return [
                {"rule_id": "clarity_rule", "rule_name": "Clarity Enhancement Rule"},
                {
                    "rule_id": "specificity_rule",
                    "rule_name": "Specificity Enhancement Rule",
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
            print(f"Error getting optimal rules: {e}")
            # Fallback to default rules
            return [
                {"rule_id": "clarity_rule", "rule_name": "Clarity Enhancement Rule"},
                {
                    "rule_id": "specificity_rule",
                    "rule_name": "Specificity Enhancement Rule",
                },
            ]

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
            session_data = ImprovementSessionCreate(
                session_id=session_id,
                original_prompt=original_prompt,
                final_prompt=final_prompt,
                rules_applied=rules_applied,
                iteration_count=1,
                session_metadata=user_context,
                status="completed",
            )

            db_session.add(ImprovementSession(**session_data.dict()))
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
                    rule_id=data["rule_id"],
                    rule_name=data["rule_name"],
                    improvement_score=data["improvement_score"],
                    confidence_level=data["confidence"],
                    execution_time_ms=data["execution_time_ms"],
                    prompt_characteristics=data["prompt_characteristics"],
                    before_metrics=data["before_metrics"],
                    after_metrics=data["after_metrics"],
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
                original_prompt=session_record.original_prompt,
                improved_prompt=session_record.final_prompt
                or session_record.original_prompt,
                user_rating=rating,
                applied_rules=session_record.rules_applied or {},
                improvement_areas={"areas": improvement_areas}
                if improvement_areas
                else None,
                user_notes=feedback_text,
                session_id=session_id,
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
        self, enabled_only: bool = True, db_session: AsyncSession = None
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
                    "rule_category": rule.rule_category,
                    "rule_description": rule.rule_description,
                    "enabled": rule.enabled,
                    "priority": rule.priority,
                    "rule_version": rule.rule_version,
                }
                for rule in rules
            ]

        except Exception as e:
            print(f"Error getting rules metadata: {e}")
            return []

    async def trigger_optimization(self, feedback_id: int, db_session: AsyncSession):
        """Trigger ML optimization based on feedback"""
        from .ml_integration import get_ml_service

        # Get feedback data and related performance metrics
        stmt = sqlmodel_select(UserFeedback).where(UserFeedback.id == feedback_id)
        result = await db_session.execute(stmt)
        feedback = result.scalar_one_or_none()

        if not feedback:
            logger.warning(f"Feedback {feedback_id} not found")
            return {"status": "error", "message": f"Feedback {feedback_id} not found"}

        # Get related performance data for training
        perf_stmt = (
            select(
                RulePerformance,
                RuleMetadata.default_parameters,
            )
            .join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id)
            .where(RulePerformance.created_at.isnot(None))  # Remove session_id filter since it may not exist
        )

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
            # row is a tuple: (RulePerformance, default_parameters)
            rule_perf = row[0]  # RulePerformance object
            params = row[1] or {}  # default_parameters dict
            features.append([
                rule_perf.improvement_score or 0,
                rule_perf.execution_time_ms or 0,
                params.get("weight", 1.0),
                params.get("priority", 5),
                len(params),  # Number of parameters
                1.0 if params.get("active", True) else 0.0,  # Active status
            ])
            effectiveness_scores.append(
                rule_perf.confidence_level or (feedback.user_rating / 5.0)
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
        """Run ML optimization for specified rules"""
        from .ml_integration import get_ml_service

        # Get all performance data for training
        perf_stmt = (
            select(
                RulePerformance,
                RuleMetadata.default_parameters,
                RuleMetadata.priority,
            )
            .join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id)
            .where(RuleMetadata.enabled == True)
        )

        # Filter by specific rule IDs if provided
        if rule_ids:
            perf_stmt = perf_stmt.where(RulePerformance.rule_id.in_(rule_ids))

        # Only include recent data (last 30 days) for relevance
        recent_date = datetime.utcnow() - timedelta(days=30)
        perf_stmt = perf_stmt.where(RulePerformance.created_at >= recent_date)

        result = await db_session.execute(perf_stmt)
        performance_data = result.fetchall()

        if len(performance_data) < 20:
            logger.warning(
                f"Insufficient data for ML optimization: {len(performance_data)} samples (minimum: 20)"
            )
            return {
                "status": "insufficient_data",
                "message": f"Need at least 20 samples for optimization, found {len(performance_data)}",
                "samples_found": len(performance_data),
            }

        # Prepare training data with enhanced features
        features = []
        effectiveness_scores = []

        for row in performance_data:
            # Access the fields from the tuple
            # row is a tuple: (RulePerformance, default_parameters, priority)
            rule_perf = row[0]  # RulePerformance object
            params = row[1] or {}  # default_parameters dict
            priority = row[2] or 5  # priority value

            # Enhanced feature engineering
            features.append([
                rule_perf.improvement_score or 0,  # Core improvement metric
                rule_perf.execution_time_ms or 0,  # Performance metric
                params.get("weight", 1.0),  # Rule weight
                priority,  # Rule priority
                rule_perf.confidence_level or 0,  # Confidence level
                len(params),  # Parameter complexity
                params.get("confidence_threshold", 0.5),  # Confidence threshold
                1.0 if params.get("enabled", True) else 0.0,  # Enabled status
                params.get("min_length", 0) / 100.0,  # Normalized min length
                params.get("max_length", 1000) / 1000.0,  # Normalized max length
            ])

            # Target effectiveness score (0-1 scale)
            effectiveness = min(1.0, max(0.0, (rule_perf.improvement_score or 0) / 100.0))
            effectiveness_scores.append(effectiveness)

        # Run ML optimization
        ml_service = await get_ml_service()
        training_data = {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
        }

        optimization_result = await ml_service.optimize_rules(
            training_data, db_session, rule_ids=rule_ids
        )

        if optimization_result.get("status") == "success":
            # Also try ensemble optimization if we have enough data
            if len(features) >= 50:
                ensemble_result = await ml_service.optimize_ensemble_rules(
                    training_data, db_session
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
        from .ml_integration import get_ml_service

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
                model_id=optimization_result.get("model_id"),
                performance_score=optimization_result.get("best_score", 0),
                accuracy=optimization_result.get("accuracy", 0),
                precision=optimization_result.get("precision", 0),
                recall=optimization_result.get("recall", 0),
                training_data_size=training_samples,
                created_at=datetime.utcnow(),
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
                        started_at=datetime.utcnow(),
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
            from ..database.models import DiscoveredPattern

            patterns = discovery_result.get("patterns", [])
            for pattern in patterns:
                # Store each discovered pattern
                pattern_record = DiscoveredPattern(
                    pattern_name=f"Auto_Pattern_{pattern.get('avg_effectiveness', 0):.2f}",
                    parameters=pattern.get("parameters", {}),
                    effectiveness_score=pattern.get("avg_effectiveness", 0),
                    support_count=pattern.get("support_count", 0),
                    confidence_score=0.8,  # Default confidence
                    created_at=datetime.utcnow(),
                )

                db_session.add(pattern_record)

            await db_session.commit()

        except Exception as e:
            logger.error(f"Failed to store pattern discovery results: {e}")
            await db_session.rollback()
