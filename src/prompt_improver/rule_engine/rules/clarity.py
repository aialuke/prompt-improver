"""
ClarityRule: A rule to improve the clarity and specificity of a prompt.
"""

import string  # Import the string module
import asyncio

from ..base import (
    BasePromptRule,
    LLMInstruction,
    RuleCheckResult,
    TransformationResult,
)
from ...services.llm_transformer import LLMTransformerService

# A simple list of vague words to check for.
# In a real implementation, this could be more sophisticated (e.g., using a thesaurus or word embeddings).
VAGUE_WORDS = [
    "thing",
    "stuff",
    "it",
    "they",
    "something",
    "better",
    "good",
    "nice",
    "analyze",
    "summarize",  # Vague without a specified length or format
]


class ClarityRule(BasePromptRule):
    """
    This rule checks for and suggests improvements for vague language in a prompt.

    It encourages specificity by:
    - Identifying generic or ambiguous words.
    - Suggesting the addition of explicit constraints, formats, or examples.
    """
    
    def __init__(self):
        self.llm_transformer = LLMTransformerService()

    @property
    def metadata(self):
        """
        Provides metadata for the ClarityRule.
        """
        return {
            "name": "ClarityRule",
            "type": "Core",
            "description": "Eliminates vague terms and encourages adding specific requirements.",
            "source": "Anthropic Best Practices",
        }

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """
        Checks if the prompt contains vague words.
        """
        # Clean the prompt by removing punctuation and splitting into words
        translator = str.maketrans("", "", string.punctuation)
        cleaned_prompt = prompt.lower().translate(translator)
        prompt_words = cleaned_prompt.split()

        found_vague_words = [word for word in VAGUE_WORDS if word in prompt_words]
        
        return RuleCheckResult(
            applies=len(found_vague_words) > 0,
            confidence=0.8 if found_vague_words else 1.0,
            metadata={
                "vague_words": found_vague_words,
                "total_words": len(prompt_words),
                "vague_word_ratio": len(found_vague_words) / len(prompt_words) if prompt_words else 0
            }
        )

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """
        Applies LLM-based transformation to improve clarity.
        """
        # Check for vague words first
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(
                success=True,
                improved_prompt=prompt,
                confidence=1.0,
                transformations=[],
            )
        
        # Extract vague words from check result
        vague_words = check_result.metadata.get("vague_words", [])
        
        # Use LLM transformer for intelligent enhancement
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                enhancement_result = loop.run_until_complete(
                    self.llm_transformer.enhance_clarity(prompt, vague_words, context)
                )
            finally:
                loop.close()
            
            return TransformationResult(
                success=True,
                improved_prompt=enhancement_result["enhanced_prompt"],
                confidence=enhancement_result["confidence"],
                transformations=enhancement_result["transformations"],
            )
            
        except Exception as e:
            # Fallback to simple improvement if LLM enhancement fails
            fallback_improvement = self._fallback_clarity_improvement(prompt, vague_words)
            return TransformationResult(
                success=True,
                improved_prompt=fallback_improvement["improved_prompt"],
                confidence=0.6,
                transformations=[{
                    "type": "fallback_clarity",
                    "message": "Applied basic clarity improvements",
                    "error": str(e)
                }],
            )

    def to_llm_instruction(self) -> str:
        """
        Generates an LLM instruction for applying this rule.
        """
        return """
Review the following prompt and identify any vague or ambiguous language.
Rewrite the prompt to be more specific, direct, and clear.
For example, replace 'summarize the text' with 'summarize the text into three key bullet points'.
Replace 'make it better' with 'rewrite the text to be more persuasive for a marketing audience'.
Focus on adding concrete details, constraints, and explicit instructions.
"""

    def _fallback_clarity_improvement(self, prompt: str, vague_words: list) -> dict:
        """Fallback method for basic clarity improvements"""
        improved_prompt = prompt
        
        # Simple replacements for common vague words
        simple_replacements = {
            "thing": "specific item",
            "stuff": "relevant information",
            "analyze": "examine systematically",
            "summarize": "provide a concise summary"
        }
        
        for vague_word in vague_words:
            if vague_word in simple_replacements:
                improved_prompt = improved_prompt.replace(
                    vague_word, simple_replacements[vague_word]
                )
        
        # If no replacements made, add guidance
        if improved_prompt == prompt:
            guidance = f"\n\nFor clarity, consider specifying: {', '.join(vague_words[:3])}"
            improved_prompt = prompt + guidance
        
        return {
            "improved_prompt": improved_prompt,
            "transformations": [{
                "type": "basic_clarity_improvement",
                "vague_words_addressed": vague_words
            }]
        }
