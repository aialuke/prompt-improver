"""ClarityRule: Enhanced rule to improve prompt clarity using research-validated patterns.

Based on research synthesis from:
- Anthropic Claude Documentation (XML structure optimization)
- OpenAI Best Practices (specificity patterns)
- AWS Prompt Engineering Guide (success criteria guidelines)
"""

import asyncio
import re
import string
from typing import Any, Dict, List

from ...ml.preprocessing.llm_transformer import LLMTransformerService
from ..base import BasePromptRule, RuleCheckResult, TransformationResult

# Enhanced vague words list based on research
VAGUE_WORDS = [
    "thing",
    "stuff",
    "it",
    "this",
    "that",
    "they",
    "something",
    "anything",
    "better",
    "good",
    "nice",
    "bad",
    "awesome",
    "great",
    "fine",
    "okay",
    "analyze",
    "summarize",
    "improve",
    "enhance",
    "optimize",
    "fix",
    "some",
    "many",
    "few",
    "several",
    "various",
    "different",
    "certain",
    "appropriate",
    "suitable",
    "relevant",
    "important",
    "significant",
]

# Hedge words that reduce specificity
HEDGE_WORDS = [
    "perhaps",
    "maybe",
    "might",
    "could",
    "should",
    "probably",
    "possibly",
    "somewhat",
    "rather",
    "quite",
    "fairly",
    "pretty",
    "kind of",
    "sort of",
]

# Success criteria patterns
SUCCESS_CRITERIA_PATTERNS = [
    "The response should include",
    "Success criteria:",
    "Expected outcomes:",
    "The final output must",
    "Requirements:",
]


class ClarityRule(BasePromptRule):
    """Enhanced clarity rule using research-validated patterns from Anthropic and OpenAI.

    Features:
    - XML structure enhancement (Anthropic patterns)
    - Specificity enhancement (OpenAI patterns)
    - Success criteria addition (AWS guidelines)
    - Configurable parameters
    """

    def __init__(self):
        self.llm_transformer = LLMTransformerService()

        # Research-validated default parameters
        self.config = {
            "min_clarity_score": 0.7,
            "sentence_complexity_threshold": 20,
            "use_structured_xml": True,
            "apply_specificity_patterns": True,
            "add_success_criteria": True,
            "context_placement_priority": "before_examples",
            "vague_word_detection": True,
            "measurable_outcome_enforcement": True,
        }

        # Attributes for dynamic loading system
        self.rule_id = "clarity_enhancement"
        self.priority = 10

    def configure(self, params: dict[str, Any]):
        """Configure rule parameters from database"""
        self.config.update(params)

    @property
    def metadata(self):
        """Enhanced metadata with research foundation"""
        return {
            "name": "Clarity Enhancement Rule",
            "type": "Fundamental",
            "description": "Improves prompt clarity using research-validated patterns from Anthropic and OpenAI documentation",
            "category": "fundamental",
            "research_foundation": [
                "Anthropic XML structure optimization",
                "OpenAI specificity patterns",
                "AWS success criteria guidelines",
            ],
            "version": "2.0.0",
            "priority": self.priority,
            "source": "Research Synthesis 2025",
        }

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Enhanced check using research-validated clarity metrics"""
        # Calculate comprehensive clarity score
        clarity_metrics = self._calculate_clarity_metrics(prompt)

        clarity_score = clarity_metrics["overall_score"]
        applies = clarity_score < self.config["min_clarity_score"]

        return RuleCheckResult(
            applies=applies,
            confidence=0.9 if applies else 0.95,
            metadata={
                "clarity_score": clarity_score,
                "vague_words": clarity_metrics["vague_words"],
                "hedge_words": clarity_metrics["hedge_words"],
                "sentence_complexity": clarity_metrics["sentence_complexity"],
                "has_success_criteria": clarity_metrics["has_success_criteria"],
                "specificity_score": clarity_metrics["specificity_score"],
                "recommendations": clarity_metrics["recommendations"],
            },
        )

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Apply research-validated clarity enhancements"""
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(
                success=True, improved_prompt=prompt, confidence=1.0, transformations=[]
            )

        improved_prompt = prompt
        transformations = []

        # Apply enhancements based on research patterns
        if self.config["use_structured_xml"]:
            improved_prompt, xml_transformations = self._apply_xml_structure(
                improved_prompt
            )
            transformations.extend(xml_transformations)

        if self.config["apply_specificity_patterns"]:
            improved_prompt, specificity_transformations = self._enhance_specificity(
                improved_prompt
            )
            transformations.extend(specificity_transformations)

        if self.config["add_success_criteria"]:
            improved_prompt, criteria_transformations = self._add_success_criteria(
                improved_prompt
            )
            transformations.extend(criteria_transformations)

        if self.config["vague_word_detection"]:
            improved_prompt, vague_transformations = self._replace_vague_language(
                improved_prompt
            )
            transformations.extend(vague_transformations)

        # Calculate final confidence based on improvements made
        final_score = self._calculate_clarity_metrics(improved_prompt)["overall_score"]
        confidence = min(0.95, 0.6 + (final_score * 0.3))

        return TransformationResult(
            success=True,
            improved_prompt=improved_prompt,
            confidence=confidence,
            transformations=transformations,
        )

    def to_llm_instruction(self) -> str:
        """Generate research-based LLM instruction"""
        return """
<instruction>
Enhance the prompt's clarity using research-validated patterns:

1. SPECIFICITY (OpenAI Research):
   - Replace vague terms with concrete, specific language
   - Add measurable constraints and explicit requirements
   - Include who, what, when, where, why details

2. XML STRUCTURE (Anthropic Guidelines):
   - Organize complex prompts with XML tags
   - Use <context>, <instruction>, <examples>, <output_format> tags
   - Place critical context before instructions

3. SUCCESS CRITERIA (AWS Best Practices):
   - Define what constitutes successful completion
   - Include measurable outcomes and evaluation criteria
   - Specify format, length, and quality requirements

4. HEDGE WORD REMOVAL:
   - Remove uncertainty words (maybe, perhaps, might)
   - Use direct, authoritative language
   - Make instructions clear and actionable

Focus on making every instruction explicit, measurable, and unambiguous.
</instruction>
"""

    def _calculate_clarity_metrics(self, prompt: str) -> dict[str, Any]:
        """Calculate comprehensive clarity metrics based on research"""
        # Basic text analysis - strip punctuation for accurate word matching
        words = re.sub(r'[^\w\s]', '', prompt.lower()).split()
        sentences = re.split(r"[.!?]+", prompt)

        # Vague word detection
        vague_words = [word for word in VAGUE_WORDS if word in words]
        vague_ratio = len(vague_words) / len(words) if words else 0

        # Hedge word detection
        hedge_words = [word for word in HEDGE_WORDS if word in words]
        hedge_ratio = len(hedge_words) / len(words) if words else 0

        # Sentence complexity (average words per sentence)
        valid_sentences = [s for s in sentences if s.strip()]
        avg_sentence_length = (
            len(words) / len(valid_sentences) if valid_sentences else 0
        )
        complexity_score = min(
            1.0, avg_sentence_length / self.config["sentence_complexity_threshold"]
        )

        # Success criteria detection
        has_success_criteria = any(
            pattern.lower() in prompt.lower() for pattern in SUCCESS_CRITERIA_PATTERNS
        )

        # Specificity score (inverse of vague + hedge ratios)
        specificity_score = max(0, 1.0 - (vague_ratio + hedge_ratio))

        # Overall clarity score (weighted combination)
        overall_score = (
            specificity_score * 0.4
            + (1.0 - complexity_score) * 0.2
            + (1.0 if has_success_criteria else 0.5) * 0.2
            + (1.0 if len(words) > 10 else 0.7) * 0.2  # Length adequacy
        )

        # Generate recommendations
        recommendations = []
        if vague_ratio > 0.1:
            recommendations.append("Replace vague language with specific terms")
        if hedge_ratio > 0.05:
            recommendations.append("Remove hedge words for more direct instructions")
        if not has_success_criteria:
            recommendations.append("Add explicit success criteria")
        if avg_sentence_length > self.config["sentence_complexity_threshold"]:
            recommendations.append("Break down complex sentences")

        return {
            "overall_score": overall_score,
            "vague_words": vague_words,
            "hedge_words": hedge_words,
            "vague_ratio": vague_ratio,
            "hedge_ratio": hedge_ratio,
            "sentence_complexity": avg_sentence_length,
            "has_success_criteria": has_success_criteria,
            "specificity_score": specificity_score,
            "recommendations": recommendations,
        }

    def _apply_xml_structure(self, prompt: str) -> tuple[str, list[dict]]:
        """Apply Anthropic XML structure patterns"""
        if len(prompt.split()) < 20:  # Only for complex prompts
            return prompt, []

        # Extract context and instructions
        context_parts = []
        instruction_parts = []

        # Simple heuristic: first sentences often contain context
        sentences = re.split(r"[.!?]+", prompt)
        if len(sentences) > 2:
            context_parts = sentences[: len(sentences) // 2]
            instruction_parts = sentences[len(sentences) // 2 :]
        else:
            instruction_parts = sentences

        structured_prompt = ""

        if context_parts:
            context_text = (
                ". ".join([s.strip() for s in context_parts if s.strip()]) + "."
            )
            structured_prompt += f"<context>\n{context_text}\n</context>\n\n"

        if instruction_parts:
            instruction_text = (
                ". ".join([s.strip() for s in instruction_parts if s.strip()]) + "."
            )
            structured_prompt += f"<instruction>\n{instruction_text}\n</instruction>"

        if structured_prompt and structured_prompt != prompt:
            return structured_prompt, [
                {
                    "type": "xml_structure_enhancement",
                    "description": "Applied Anthropic XML organization patterns",
                }
            ]

        return prompt, []

    def _enhance_specificity(self, prompt: str) -> tuple[str, list[dict]]:
        """Apply OpenAI specificity patterns"""
        improvements = []
        enhanced_prompt = prompt

        # Replace vague quantifiers
        vague_quantifiers = {
            "some": "approximately 3-5",
            "many": "more than 10",
            "few": "2-3",
            "several": "4-6",
        }

        for vague, specific in vague_quantifiers.items():
            if vague in enhanced_prompt.lower():
                enhanced_prompt = re.sub(
                    r"\b" + vague + r"\b",
                    specific,
                    enhanced_prompt,
                    flags=re.IGNORECASE,
                )
                improvements.append({
                    "type": "specificity_enhancement",
                    "description": f"Replaced '{vague}' with '{specific}'",
                })

        return enhanced_prompt, improvements

    def _add_success_criteria(self, prompt: str) -> tuple[str, list[dict]]:
        """Add success criteria using AWS best practices"""
        if any(
            pattern.lower() in prompt.lower() for pattern in SUCCESS_CRITERIA_PATTERNS
        ):
            return prompt, []  # Already has success criteria

        # Generate success criteria based on prompt content
        success_criteria = "\n\nSuccess criteria:\n- Response should be clear and actionable\n- Include specific examples where applicable\n- Address all aspects of the request"

        enhanced_prompt = prompt + success_criteria

        return enhanced_prompt, [
            {
                "type": "success_criteria_addition",
                "description": "Added explicit success criteria (AWS best practices)",
            }
        ]

    def _replace_vague_language(self, prompt: str) -> tuple[str, list[dict]]:
        """Replace vague language with specific alternatives"""
        improvements = []
        enhanced_prompt = prompt

        # Enhanced replacements based on research
        specific_replacements = {
            "thing": "specific item or concept",
            "stuff": "relevant information or materials",
            "analyze": "examine systematically and provide detailed insights on",
            "summarize": "provide a concise summary with key points including",
            "improve": "enhance by implementing specific changes to",
            "better": "more effective, efficient, or suitable for the intended purpose",
            "good": "high-quality, well-structured, and meeting specified criteria",
        }

        for vague, specific in specific_replacements.items():
            if re.search(r"\b" + vague + r"\b", enhanced_prompt, re.IGNORECASE):
                enhanced_prompt = re.sub(
                    r"\b" + vague + r"\b",
                    specific,
                    enhanced_prompt,
                    flags=re.IGNORECASE,
                )
                improvements.append({
                    "type": "vague_language_replacement",
                    "description": f"Replaced '{vague}' with '{specific}'",
                })

        return enhanced_prompt, improvements

    def _fallback_clarity_improvement(self, prompt: str, vague_words: list) -> dict:
        """Enhanced fallback method with research patterns"""
        improved_prompt = prompt
        transformations = []

        # Apply basic improvements
        simple_replacements = {
            "thing": "specific item",
            "this": "the specific item mentioned",
            "that": "the specific element referenced",
            "stuff": "relevant information",
            "analyze": "examine systematically",
            "summarize": "provide a concise summary with key points",
        }

        for vague_word in vague_words:
            if vague_word in simple_replacements:
                improved_prompt = improved_prompt.replace(
                    vague_word, simple_replacements[vague_word]
                )
                transformations.append({
                    "type": "basic_replacement",
                    "original": vague_word,
                    "replacement": simple_replacements[vague_word],
                })

        # Add specificity guidance if minimal improvements
        if len(transformations) < 2:
            guidance = f"\n\nFor enhanced clarity, consider specifying: {', '.join(vague_words[:3])}"
            improved_prompt = prompt + guidance
            transformations.append({
                "type": "specificity_guidance",
                "guidance_added": True,
            })

        return {"improved_prompt": improved_prompt, "transformations": transformations}
