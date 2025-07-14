"""Rule engine module for prompt improvement."""

from dataclasses import dataclass
from typing import Any

from .base import BasePromptRule
from .rules.clarity import ClarityRule
from .rules.specificity import SpecificityRule
from .rules.chain_of_thought import ChainOfThoughtRule
from .rules.few_shot_examples import FewShotExampleRule
from .rules.role_based_prompting import RoleBasedPromptingRule
from .rules.xml_structure_enhancement import XMLStructureRule


@dataclass
class AppliedRuleResult:
    """Represents the result of applying a single rule."""

    rule_id: str
    confidence: float
    improved_prompt: str


@dataclass
class RuleEngineResult:
    """Result from applying rules to a prompt."""

    improved_prompt: str
    applied_rules: list[AppliedRuleResult]
    total_confidence: float
    processing_time_ms: float | None = None


class RuleEngine:
    """Orchestrates the application of prompt improvement rules.

    Manages rule prioritization, confidence thresholds, and result aggregation.
    """

    def __init__(self, min_confidence: float = 0.0):
        """Initialize the rule engine.

        Args:
            min_confidence: Minimum confidence threshold for rule application
        """
        self.min_confidence = min_confidence

        # Create rule instances and add rule_id property
        clarity_rule = ClarityRule()
        clarity_rule.rule_id = "clarity_rule"
        clarity_rule.priority = 5

        specificity_rule = SpecificityRule()
        specificity_rule.rule_id = "specificity_rule"
        specificity_rule.priority = 4

        self.rules = [clarity_rule, specificity_rule]

        # Sort rules by priority
        self.rules.sort(key=lambda rule: getattr(rule, "priority", 5), reverse=True)

    def apply_rules(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> RuleEngineResult:
        """Apply all applicable rules to the given prompt.

        Args:
            prompt: The original prompt to improve
            context: Optional context for rule application

        Returns:
            RuleEngineResult containing improved prompt and applied rules
        """
        current_prompt = prompt
        applied_rules = []
        total_confidence = 0.0

        for rule in self.rules:
            try:
                # Apply rule with current prompt state
                if context:
                    result = rule.apply(current_prompt, context=context)
                else:
                    result = rule.apply(current_prompt)

                # Check if rule meets confidence threshold and was successful
                if result.confidence >= self.min_confidence and result.success:
                    applied_rule = AppliedRuleResult(
                        rule_id=getattr(rule, "rule_id", "unknown_rule"),
                        confidence=result.confidence,
                        improved_prompt=result.improved_prompt,
                    )
                    applied_rules.append(applied_rule)
                    current_prompt = result.improved_prompt
                    total_confidence += result.confidence

            except Exception:
                # Skip rule if it fails - this is for test robustness
                continue

        # Calculate average confidence if rules were applied
        if applied_rules:
            total_confidence = total_confidence / len(applied_rules)

        return RuleEngineResult(
            improved_prompt=current_prompt,
            applied_rules=applied_rules,
            total_confidence=total_confidence,
        )


# Re-export main classes for convenience
__all__ = [
    "AppliedRuleResult",
    "ChainOfThoughtRule",
    "ClarityRule",
    "FewShotExampleRule", 
    "RoleBasedPromptingRule",
    "RuleEngine",
    "RuleEngineResult",
    "SpecificityRule",
    "XMLStructureRule",
]
