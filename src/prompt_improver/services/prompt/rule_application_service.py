"""Rule Application Service - Focused service for rule execution and validation.

This service handles:
- Rule application and execution
- Rule compatibility validation
- Rule chain execution
- Rule performance metrics

Part of the PromptServiceFacade decomposition following Clean Architecture principles.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from prompt_improver.rule_engine.base import BasePromptRule
from prompt_improver.shared.interfaces.protocols.application import (
    RuleApplicationServiceProtocol,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class RuleApplicationService(RuleApplicationServiceProtocol):
    """Service for rule execution and validation."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    async def _get_ml_event_bus(self):
        """Lazy load ML event bus to avoid torch dependencies."""
        try:
            from prompt_improver.core.events.ml_event_bus import get_ml_event_bus
            return await get_ml_event_bus()
        except ImportError:
            self.logger.info("ML event bus not available (torch not installed)")
            return None

    def _create_ml_event(self, event_type: str, source: str, data: dict[str, Any]):
        """Lazy create ML event to avoid torch dependencies."""
        try:
            from prompt_improver.core.events.ml_event_bus import MLEvent, MLEventType
            return MLEvent(
                event_type=getattr(MLEventType, event_type),
                source=source,
                data=data
            )
        except ImportError:
            self.logger.info("ML events not available (torch not installed)")
            return None

    async def apply_rules(
        self,
        prompt: str,
        rules: list[BasePromptRule],
        session_id: UUID | None = None,
        config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Apply a set of rules to a prompt."""
        try:
            start_time = datetime.now()
            results = {
                "original_prompt": prompt,
                "session_id": str(session_id) if session_id else None,
                "applied_rules": [],
                "final_prompt": prompt,
                "execution_summary": {},
                "timestamp": aware_utc_now().isoformat()
            }

            current_prompt = prompt
            total_improvement_score = 0.0
            successful_applications = 0

            for i, rule in enumerate(rules):
                try:
                    rule_start_time = datetime.now()

                    # Apply rule
                    rule_result = await self._execute_single_rule(
                        current_prompt, rule, config
                    )

                    rule_execution_time = (datetime.now() - rule_start_time).total_seconds() * 1000

                    if rule_result.get("success", False):
                        current_prompt = rule_result.get("improved_prompt", current_prompt)
                        improvement_score = rule_result.get("improvement_score", 0.0)
                        total_improvement_score += improvement_score
                        successful_applications += 1

                        rule_info = {
                            "rule_id": getattr(rule, 'rule_id', f"rule_{i}"),
                            "rule_name": getattr(rule, 'name', rule.__class__.__name__),
                            "execution_order": i + 1,
                            "success": True,
                            "improvement_score": improvement_score,
                            "execution_time_ms": rule_execution_time,
                            "changes_made": rule_result.get("changes_made", []),
                            "confidence": rule_result.get("confidence", 0.0)
                        }
                    else:
                        rule_info = {
                            "rule_id": getattr(rule, 'rule_id', f"rule_{i}"),
                            "rule_name": getattr(rule, 'name', rule.__class__.__name__),
                            "execution_order": i + 1,
                            "success": False,
                            "error": rule_result.get("error", "Unknown error"),
                            "execution_time_ms": rule_execution_time
                        }

                    results["applied_rules"].append(rule_info)

                    # Update performance stats
                    await self._update_rule_performance_stats(
                        getattr(rule, 'rule_id', rule.__class__.__name__),
                        rule_execution_time,
                        rule_result.get("success", False),
                        improvement_score if rule_result.get("success") else 0.0
                    )

                except Exception as rule_error:
                    logger.exception(f"Error applying rule {rule.__class__.__name__}: {rule_error}")
                    results["applied_rules"].append({
                        "rule_id": getattr(rule, 'rule_id', f"rule_{i}"),
                        "rule_name": getattr(rule, 'name', rule.__class__.__name__),
                        "execution_order": i + 1,
                        "success": False,
                        "error": str(rule_error),
                        "execution_time_ms": 0
                    })

            results["final_prompt"] = current_prompt
            total_execution_time = (datetime.now() - start_time).total_seconds() * 1000

            results["execution_summary"] = {
                "total_rules": len(rules),
                "successful_applications": successful_applications,
                "failed_applications": len(rules) - successful_applications,
                "total_improvement_score": total_improvement_score,
                "average_improvement": total_improvement_score / max(successful_applications, 1),
                "total_execution_time_ms": total_execution_time,
                "success_rate": successful_applications / len(rules) if rules else 0.0
            }

            # Publish execution event
            await self._publish_execution_event(results, session_id)

            return results

        except Exception as e:
            logger.exception(f"Error applying rules: {e}")
            raise

    async def validate_rule_compatibility(
        self,
        rules: list[BasePromptRule],
        prompt_type: str | None = None
    ) -> dict[str, bool]:
        """Validate if rules are compatible with each other."""
        try:

            compatibility_matrix = {}
            warnings = []

            for i, rule1 in enumerate(rules):
                rule1_id = getattr(rule1, 'rule_id', rule1.__class__.__name__)

                for j, rule2 in enumerate(rules):
                    if i >= j:  # Avoid duplicate checks
                        continue

                    rule2_id = getattr(rule2, 'rule_id', rule2.__class__.__name__)
                    pair_key = f"{rule1_id}+{rule2_id}"

                    # Check compatibility
                    is_compatible = await self._check_rule_pair_compatibility(
                        rule1, rule2, prompt_type
                    )

                    compatibility_matrix[pair_key] = is_compatible

                    if not is_compatible:
                        warnings.append({
                            "rule1": rule1_id,
                            "rule2": rule2_id,
                            "issue": "Potential conflict detected",
                            "recommendation": "Consider applying in different order"
                        })

            # Overall compatibility
            all_compatible = all(compatibility_matrix.values())

            return {
                "overall_compatible": all_compatible,
                "compatibility_matrix": compatibility_matrix,
                "warnings": warnings,
                "recommendation": "safe" if all_compatible else "review_order"
            }

        except Exception as e:
            logger.exception(f"Error validating rule compatibility: {e}")
            return {"overall_compatible": False, "error": str(e)}

    async def execute_rule_chain(
        self,
        prompt: str,
        rule_chain: list[BasePromptRule],
        stop_on_error: bool = False
    ) -> list[dict[str, Any]]:
        """Execute a chain of rules in sequence."""
        try:
            chain_results = []
            current_prompt = prompt

            for i, rule in enumerate(rule_chain):
                try:
                    rule_start_time = datetime.now()

                    rule_result = await self._execute_single_rule(current_prompt, rule)
                    rule_execution_time = (datetime.now() - rule_start_time).total_seconds() * 1000

                    result_entry = {
                        "step": i + 1,
                        "rule_id": getattr(rule, 'rule_id', rule.__class__.__name__),
                        "rule_name": getattr(rule, 'name', rule.__class__.__name__),
                        "input_prompt": current_prompt,
                        "output_prompt": rule_result.get("improved_prompt", current_prompt),
                        "success": rule_result.get("success", False),
                        "improvement_score": rule_result.get("improvement_score", 0.0),
                        "execution_time_ms": rule_execution_time,
                        "changes_made": rule_result.get("changes_made", []),
                        "confidence": rule_result.get("confidence", 0.0)
                    }

                    if rule_result.get("success"):
                        current_prompt = rule_result.get("improved_prompt", current_prompt)
                    else:
                        result_entry["error"] = rule_result.get("error", "Rule execution failed")
                        if stop_on_error:
                            result_entry["chain_stopped"] = True
                            chain_results.append(result_entry)
                            break

                    chain_results.append(result_entry)

                except Exception as rule_error:
                    error_entry = {
                        "step": i + 1,
                        "rule_id": getattr(rule, 'rule_id', rule.__class__.__name__),
                        "rule_name": getattr(rule, 'name', rule.__class__.__name__),
                        "input_prompt": current_prompt,
                        "success": False,
                        "error": str(rule_error),
                        "execution_time_ms": 0
                    }

                    chain_results.append(error_entry)

                    if stop_on_error:
                        error_entry["chain_stopped"] = True
                        break

            return chain_results

        except Exception as e:
            logger.exception(f"Error executing rule chain: {e}")
            raise

    async def get_rule_performance_metrics(
        self,
        rule_id: str,
        time_range: tuple[datetime, datetime] | None = None
    ) -> dict[str, Any]:
        """Get performance metrics for a specific rule."""
        try:
            # Note: Performance metrics now handled at database level
            # Return basic metrics structure for compatibility
            return {
                "rule_id": rule_id,
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time_ms": 0.0,
                "average_improvement_score": 0.0,
                "last_execution": None,
                "note": "Performance metrics moved to database level for unified caching"
            }

        except Exception as e:
            logger.exception(f"Error getting rule performance metrics for {rule_id}: {e}")
            return {"rule_id": rule_id, "error": str(e)}

    async def _execute_single_rule(
        self,
        prompt: str,
        rule: BasePromptRule,
        config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a single rule on a prompt."""
        try:
            # Check if rule has required methods
            if not hasattr(rule, 'apply'):
                return {
                    "success": False,
                    "error": "Rule does not implement apply method"
                }

            # Apply rule with configuration
            if config:
                rule_config = config.get(getattr(rule, 'rule_id', 'default'), {})
                if hasattr(rule, 'configure'):
                    rule.configure(rule_config)

            # Execute rule
            result = await rule.apply(prompt, {})

            return {
                "success": True,
                "improved_prompt": result.get("improved_prompt", prompt),
                "improvement_score": result.get("improvement_score", 0.0),
                "confidence": result.get("confidence", 0.5),
                "changes_made": result.get("changes_made", []),
                "metadata": result.get("metadata", {})
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _check_rule_pair_compatibility(
        self,
        rule1: BasePromptRule,
        rule2: BasePromptRule,
        prompt_type: str | None = None
    ) -> bool:
        """Check if two rules are compatible."""
        try:
            # Get rule categories/types
            rule1_type = getattr(rule1, 'rule_type', 'general')
            rule2_type = getattr(rule2, 'rule_type', 'general')

            # Known incompatible combinations
            incompatible_pairs = {
                ('structure', 'structure'),  # Multiple structure rules may conflict
                ('format', 'format'),        # Multiple format rules may conflict
            }

            if (rule1_type, rule2_type) in incompatible_pairs:
                return False

            # Check for conflicting requirements
            rule1_requirements = getattr(rule1, 'requirements', set())
            rule2_requirements = getattr(rule2, 'requirements', set())

            # Rules that both require exclusive control are incompatible
            return not ('exclusive_control' in rule1_requirements and 'exclusive_control' in rule2_requirements)

        except Exception as e:
            logger.warning(f"Error checking rule compatibility: {e}")
            return True  # Default to compatible if check fails

    async def _update_rule_performance_stats(
        self,
        rule_id: str,
        execution_time_ms: float,
        success: bool,
        improvement_score: float
    ) -> None:
        """Update performance statistics for a rule.

        Note: Performance stats are now handled at the database level
        through repositories for unified caching and persistence.
        """
        # Performance tracking moved to database level
        # This method kept for API compatibility but no longer caches locally

    async def _publish_execution_event(
        self,
        results: dict[str, Any],
        session_id: UUID | None
    ) -> None:
        """Publish rule execution event for monitoring."""
        try:
            event_bus = await self._get_ml_event_bus()
            if event_bus:
                execution_event = self._create_ml_event(
                    "TRAINING_DATA",
                    "rule_application_service",
                    {
                        "operation": "rule_execution_completed",
                        "session_id": str(session_id) if session_id else None,
                        "execution_summary": results["execution_summary"],
                        "timestamp": results["timestamp"]
                    }
                )
                if execution_event:
                    await event_bus.publish(execution_event)
        except Exception as e:
            logger.warning(f"Failed to publish execution event: {e}")
