"""SpecificityRule: A rule to improve the specificity and detail of a prompt."""

import asyncio

from ...services.llm_transformer import LLMTransformerService
from ..base import (
    BasePromptRule,
    RuleCheckResult,
    TransformationResult,
)


class SpecificityRule(BasePromptRule):
    """This rule checks for and improves prompts that lack specific instructions,
    examples, or constraints.
    """

    def __init__(self):
        self.llm_transformer = LLMTransformerService()

    @property
    def metadata(self):
        """Provides metadata for the SpecificityRule."""
        return {
            "name": "SpecificityRule",
            "type": "Core",
            "description": "Adds specific constraints, examples, and detailed instructions.",
            "source": "Anthropic Best Practices",
        }

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Checks if the prompt lacks specific details."""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        # Check for lack of specific indicators
        specific_indicators = ["when", "where", "how", "why", "which", "what kind"]
        has_indicators = any(
            indicator in prompt_lower for indicator in specific_indicators
        )

        # Check for examples or constraints
        has_examples = any(
            word in prompt_lower for word in ["example", "for instance", "such as"]
        )
        has_constraints = any(
            word in prompt_lower
            for word in ["must", "should", "requirement", "constraint"]
        )

        # Short prompts often lack specificity
        is_short = word_count < 10

        # Calculate specificity score
        specificity_score = 0
        if has_indicators:
            specificity_score += 0.3
        if has_examples:
            specificity_score += 0.3
        if has_constraints:
            specificity_score += 0.3
        if not is_short:
            specificity_score += 0.1

        # Rule applies if specificity score is low
        needs_improvement = specificity_score < 0.5 or is_short

        return RuleCheckResult(
            applies=needs_improvement,
            confidence=0.8 if needs_improvement else 0.9,
            metadata={
                "has_indicators": has_indicators,
                "has_examples": has_examples,
                "has_constraints": has_constraints,
                "is_short": is_short,
                "word_count": word_count,
                "specificity_score": specificity_score,
            },
        )

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Applies LLM-based transformation to improve specificity."""
        # Check if specificity improvements are needed
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(
                success=True,
                improved_prompt=prompt,
                confidence=1.0,
                transformations=[],
            )

        # Use LLM transformer for intelligent enhancement
        try:
            # Check if we're in an async context
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # If we have a running loop, we're in async context - skip LLM for now
                # and use fallback (proper async support would require async rule methods)
                raise RuntimeError("In async context, use fallback")
            except RuntimeError:
                # No running loop, safe to create one
                enhancement_result = asyncio.run(
                    self.llm_transformer.enhance_specificity(prompt, context)
                )

            return TransformationResult(
                success=True,
                improved_prompt=enhancement_result["enhanced_prompt"],
                confidence=enhancement_result["confidence"],
                transformations=enhancement_result["transformations"],
            )

        except Exception as e:
            # Fallback to simple improvement if LLM enhancement fails
            fallback_improvement = self._fallback_specificity_improvement(
                prompt, check_result.metadata
            )
            return TransformationResult(
                success=True,
                improved_prompt=fallback_improvement["improved_prompt"],
                confidence=0.6,
                transformations=[
                    {
                        "type": "fallback_specificity",
                        "message": "Applied basic specificity improvements",
                        "error": str(e),
                    }
                ],
            )

    def to_llm_instruction(self) -> str:
        """Generates an LLM instruction for applying this rule."""
        return """
Review the following prompt and make it more specific and detailed.
Add concrete examples, clear constraints, and explicit requirements.
Specify the desired format, length, style, and any other relevant parameters.
For example, instead of 'write a summary', use 'write a 3-paragraph summary in bullet points for a technical audience'.
Include specific context about the intended use case or audience when relevant.
"""

    def _fallback_specificity_improvement(self, prompt: str, metadata: dict) -> dict:
        """Fallback method for basic specificity improvements"""
        improved_prompt = prompt
        additions = []

        # Add format specification if missing
        if not metadata.get("has_constraints"):
            additions.append(
                "Format: Provide your response in a clear, structured format."
            )

        # Add examples if missing
        if not metadata.get("has_examples") and metadata.get("word_count", 0) > 5:
            additions.append(
                "Example: Include specific examples to illustrate your points."
            )

        # Add length constraint for short prompts
        if metadata.get("is_short"):
            additions.append(
                "Detail: Please provide comprehensive information with specific details."
            )

        # Add the improvements
        if additions:
            improved_prompt = prompt + "\n\n" + "\n".join(additions)

        return {
            "improved_prompt": improved_prompt,
            "transformations": [
                {
                    "type": "basic_specificity_improvement",
                    "additions": additions,
                    "metadata": metadata,
                }
            ],
        }
