"""SpecificityRule: Enhanced rule to reduce vague language and increase prompt specificity.

Based on research synthesis from:
- Multi-source specificity research across LLM optimization studies
- Vague language detection and replacement methodologies
- Measurable outcome frameworks for prompt effectiveness
- Quantifiable metrics and concrete language patterns
"""

import re
from typing import Any

from ...ml.preprocessing.llm_transformer import LLMTransformerService
from ..base import BasePromptRule, RuleCheckResult, TransformationResult

# Vague language patterns that reduce specificity
VAGUE_LANGUAGE_PATTERNS = {
    "quantifiers": [
        "some",
        "many",
        "few",
        "several",
        "various",
        "different",
        "numerous",
        "multiple",
    ],
    "qualifiers": [
        "good",
        "bad",
        "nice",
        "fine",
        "okay",
        "great",
        "awesome",
        "terrible",
        "decent",
    ],
    "hedge_words": [
        "maybe",
        "perhaps",
        "might",
        "could",
        "should",
        "probably",
        "possibly",
        "somewhat",
    ],
    "vague_nouns": [
        "thing",
        "stuff",
        "item",
        "element",
        "aspect",
        "factor",
        "feature",
        "component",
    ],
    "vague_verbs": ["do", "make", "get", "have", "use", "go", "come", "take", "give"],
    "vague_adjectives": [
        "appropriate",
        "suitable",
        "relevant",
        "important",
        "significant",
        "interesting",
    ],
}

# Specificity improvement patterns
SPECIFICITY_PATTERNS = {
    "who_what_when_where": {
        "who": [
            "target audience",
            "specific user type",
            "professional role",
            "stakeholder group",
        ],
        "what": [
            "specific deliverable",
            "exact outcome",
            "concrete result",
            "measurable output",
        ],
        "when": ["timeframe", "deadline", "schedule", "timing constraints"],
        "where": ["context", "environment", "platform", "setting"],
    },
    "concrete_examples": ["such as", "for instance", "including", "like", "e.g."],
    "quantifiable_metrics": [
        "number",
        "percentage",
        "count",
        "length",
        "duration",
        "size",
        "amount",
    ],
}

# Measurable outcome indicators
MEASURABLE_INDICATORS = [
    "specific",
    "measurable",
    "quantifiable",
    "concrete",
    "precise",
    "exact",
    "numerical",
    "detailed",
    "explicit",
    "clear-cut",
    "well-defined",
]

# Success criteria patterns
SUCCESS_CRITERIA_KEYWORDS = [
    "criteria",
    "requirements",
    "must",
    "should",
    "needs to",
    "has to",
    "expected",
    "desired",
    "target",
    "goal",
    "objective",
    "outcome",
]

class SpecificityRule(BasePromptRule):
    """Enhanced specificity rule using research-validated patterns for vague language detection and replacement.

    features:
    - Comprehensive vague language detection and replacement
    - Who/what/when/where specificity patterns
    - Concrete examples and quantifiable metrics integration
    - Measurable outcome enforcement
    - Configurable specificity thresholds
    """

    def __init__(self):
        self.llm_transformer = LLMTransformerService()

        # Research-validated default parameters
        self.config = {
            "vague_language_threshold": 0.3,
            "require_specific_outcomes": True,
            "include_success_criteria": True,
            "enforce_measurable_goals": True,
            "specificity_patterns": [
                "who_what_when_where",
                "concrete_examples",
                "quantifiable_metrics",
            ],
            "avoid_hedge_words": True,
            "concrete_noun_preference": True,
            "action_verb_specificity": True,
        }

        # Attributes for dynamic loading system
        self.rule_id = "specificity_enhancement"
        self.priority = 9

    def configure(self, params: dict[str, Any]):
        """Configure rule parameters from database"""
        self.config.update(params)

    @property
    def metadata(self):
        """Enhanced metadata with research foundation"""
        return {
            "name": "Specificity and Detail Rule",
            "type": "Fundamental",
            "description": "Reduces vague language and increases prompt specificity using multi-source research patterns",
            "category": "fundamental",
            "research_foundation": [
                "Multi-source specificity research",
                "Vague language detection studies",
                "Measurable outcome frameworks",
            ],
            "version": "2.0.0",
            "priority": self.priority,
            "source": "Research Synthesis 2025",
        }

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Enhanced check using research-validated specificity metrics"""
        # Calculate comprehensive specificity metrics
        specificity_metrics = self._calculate_specificity_metrics(prompt)

        vague_language_ratio = specificity_metrics["vague_language_ratio"]
        applies = vague_language_ratio > self.config["vague_language_threshold"]

        # Also apply if missing key specificity elements
        if not applies:
            applies = (
                not specificity_metrics["has_measurable_outcomes"]
                or not specificity_metrics["has_concrete_examples"]
                or specificity_metrics["specificity_score"] < 0.6
            )

        return RuleCheckResult(
            applies=applies,
            confidence=0.9 if applies else 0.95,
            metadata={
                "vague_language_ratio": vague_language_ratio,
                "vague_words": specificity_metrics["vague_words"],
                "hedge_words": specificity_metrics["hedge_words"],
                "specificity_score": specificity_metrics["specificity_score"],
                "has_measurable_outcomes": specificity_metrics[
                    "has_measurable_outcomes"
                ],
                "has_concrete_examples": specificity_metrics["has_concrete_examples"],
                "has_success_criteria": specificity_metrics["has_success_criteria"],
                "missing_elements": specificity_metrics["missing_elements"],
                "recommendations": specificity_metrics["recommendations"],
            },
        )

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Apply research-validated specificity enhancements"""
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(
                success=True, improved_prompt=prompt, confidence=1.0, transformations=[]
            )

        specificity_metrics = check_result.metadata
        improved_prompt = prompt
        transformations = []

        # Apply specificity enhancements based on research patterns
        if self.config["avoid_hedge_words"]:
            improved_prompt, hedge_transformations = self._remove_hedge_words(
                improved_prompt
            )
            transformations.extend(hedge_transformations)

        if "who_what_when_where" in self.config["specificity_patterns"]:
            improved_prompt, wwww_transformations = self._apply_who_what_when_where(
                improved_prompt
            )
            transformations.extend(wwww_transformations)

        if "concrete_examples" in self.config["specificity_patterns"]:
            improved_prompt, example_transformations = self._add_concrete_examples(
                improved_prompt
            )
            transformations.extend(example_transformations)

        if "quantifiable_metrics" in self.config["specificity_patterns"]:
            improved_prompt, metric_transformations = self._add_quantifiable_metrics(
                improved_prompt
            )
            transformations.extend(metric_transformations)

        if self.config["require_specific_outcomes"]:
            improved_prompt, outcome_transformations = self._enforce_specific_outcomes(
                improved_prompt
            )
            transformations.extend(outcome_transformations)

        if self.config["include_success_criteria"]:
            improved_prompt, criteria_transformations = self._add_success_criteria(
                improved_prompt
            )
            transformations.extend(criteria_transformations)

        # Replace vague language with specific alternatives
        improved_prompt, vague_transformations = self._replace_vague_language(
            improved_prompt
        )
        transformations.extend(vague_transformations)

        # Calculate final confidence based on improvements
        final_metrics = self._calculate_specificity_metrics(improved_prompt)
        improvement_score = (
            final_metrics["specificity_score"]
            - specificity_metrics["specificity_score"]
        )
        confidence = min(0.95, 0.7 + (improvement_score * 0.3))

        return TransformationResult(
            success=True,
            improved_prompt=improved_prompt,
            confidence=confidence,
            transformations=transformations,
        )

    def to_llm_instruction(self) -> str:
        """Generate research-based LLM instruction for specificity enhancement"""
        return """
<instruction>
Enhance prompt specificity using research-validated patterns:

1. VAGUE LANGUAGE REMOVAL:
   - Replace hedge words (maybe, perhaps, might) with direct statements
   - Convert vague quantifiers (some, many) to specific numbers
   - Replace vague nouns (thing, stuff) with concrete terms

2. WHO/WHAT/WHEN/WHERE SPECIFICITY:
   - WHO: Specify target audience, user type, or stakeholder
   - WHAT: Define exact deliverable, outcome, or result
   - WHEN: Include timeframe, deadline, or timing constraints
   - WHERE: Clarify context, environment, or setting

3. CONCRETE EXAMPLES:
   - Add specific instances using "such as", "for example"
   - Include real-world scenarios and use cases
   - Provide sample inputs and expected outputs

4. QUANTIFIABLE METRICS:
   - Specify numbers, percentages, counts, or measurements
   - Define length, duration, size constraints
   - Include performance or quality metrics

5. MEASURABLE OUTCOMES:
   - Define success criteria explicitly
   - Use action verbs with specific results
   - Include evaluation methods or standards

Focus on making every element concrete, measurable, and unambiguous.
</instruction>
"""

    def _calculate_specificity_metrics(self, prompt: str) -> dict[str, Any]:
        """Calculate comprehensive specificity metrics based on research"""
        words = prompt.lower().split()
        total_words = len(words)

        # Detect vague language
        vague_words = []
        hedge_words = []

        for category, word_list in VAGUE_LANGUAGE_PATTERNS.items():
            for word in word_list:
                if word in words:
                    if category == "hedge_words":
                        hedge_words.append(word)
                    else:
                        vague_words.append(word)

        vague_language_ratio = (len(vague_words) + len(hedge_words)) / max(
            total_words, 1
        )

        # Check for measurable outcomes
        has_measurable_outcomes = any(
            indicator in prompt.lower() for indicator in MEASURABLE_INDICATORS
        )

        # Check for concrete examples
        has_concrete_examples = any(
            example_word in prompt.lower()
            for example_word in SPECIFICITY_PATTERNS["concrete_examples"]
        )

        # Check for success criteria
        has_success_criteria = any(
            keyword in prompt.lower() for keyword in SUCCESS_CRITERIA_KEYWORDS
        )

        # Calculate overall specificity score
        specificity_score = self._calculate_overall_specificity_score(
            prompt, vague_language_ratio
        )

        # Identify missing elements
        missing_elements = []
        if not has_measurable_outcomes:
            missing_elements.append("measurable_outcomes")
        if not has_concrete_examples:
            missing_elements.append("concrete_examples")
        if not has_success_criteria:
            missing_elements.append("success_criteria")
        if vague_language_ratio > self.config["vague_language_threshold"]:
            missing_elements.append("specific_language")

        # Generate recommendations
        recommendations = []
        if vague_language_ratio > 0.2:
            recommendations.append("Replace vague language with specific terms")
        if not has_measurable_outcomes:
            recommendations.append("Add measurable success criteria")
        if not has_concrete_examples:
            recommendations.append("Include specific examples")
        if len(words) < 15:
            recommendations.append("Add more detailed specifications")

        return {
            "vague_language_ratio": vague_language_ratio,
            "vague_words": vague_words,
            "hedge_words": hedge_words,
            "specificity_score": specificity_score,
            "has_measurable_outcomes": has_measurable_outcomes,
            "has_concrete_examples": has_concrete_examples,
            "has_success_criteria": has_success_criteria,
            "missing_elements": missing_elements,
            "recommendations": recommendations,
        }

    def _calculate_overall_specificity_score(
        self, prompt: str, vague_ratio: float
    ) -> float:
        """Calculate overall specificity score using multiple factors"""
        # Base score from vague language (inverted)
        vague_score = max(0, 1.0 - vague_ratio * 2)

        # Length adequacy (more specific prompts tend to be longer)
        words = prompt.split()
        length_score = min(1.0, len(words) / 20)

        # Specific indicator presence
        specific_indicators = [
            "specific",
            "exactly",
            "precisely",
            "must",
            "should",
            "will",
        ]
        indicator_count = sum(
            1 for word in words if word.lower() in specific_indicators
        )
        indicator_score = min(1.0, indicator_count / max(len(words), 1) * 10)

        # Quantifiable elements presence
        number_pattern = r"\b\d+\b"
        has_numbers = bool(re.search(number_pattern, prompt))
        number_score = 0.2 if has_numbers else 0

        # Weighted combination
        overall_score = (
            vague_score * 0.4
            + length_score * 0.2
            + indicator_score * 0.3
            + number_score * 0.1
        )

        return min(1.0, overall_score)

    def _remove_hedge_words(self, prompt: str) -> tuple[str, list[dict]]:
        """Remove hedge words that reduce specificity"""
        improved_prompt = prompt
        transformations = []

        hedge_replacements = {
            "maybe": "will",
            "perhaps": "will",
            "might": "will",
            "could": "should",
            "probably": "will",
            "possibly": "can",
            "somewhat": "",
            "rather": "",
            "quite": "",
            "fairly": "",
            "pretty": "",
        }

        for hedge, replacement in hedge_replacements.items():
            if hedge in improved_prompt.lower():
                pattern = r"\b" + re.escape(hedge) + r"\b"
                if replacement:
                    improved_prompt = re.sub(
                        pattern, replacement, improved_prompt, flags=re.IGNORECASE
                    )
                    transformations.append({
                        "type": "hedge_word_replacement",
                        "description": f"Replaced '{hedge}' with '{replacement}' for directness",
                        "original": hedge,
                        "replacement": replacement,
                    })
                else:
                    improved_prompt = re.sub(
                        pattern, "", improved_prompt, flags=re.IGNORECASE
                    )
                    improved_prompt = re.sub(r"\s+", " ", improved_prompt).strip()
                    transformations.append({
                        "type": "hedge_word_removal",
                        "description": f"Removed unnecessary hedge word '{hedge}'",
                        "original": hedge,
                    })

        return improved_prompt, transformations

    def _apply_who_what_when_where(self, prompt: str) -> tuple[str, list[dict]]:
        """Apply who/what/when/where specificity pattern"""
        if any(word in prompt.lower() for word in ["who", "what", "when", "where"]):
            return prompt, []  # Already has WWWW elements

        # Add WWWW guidance
        guidance = "\n\nPlease specify:\n- WHO: Target audience or user type\n- WHAT: Specific deliverable or outcome\n- WHEN: Timeframe or constraints\n- WHERE: Context or environment"

        enhanced_prompt = prompt + guidance

        return enhanced_prompt, [
            {
                "type": "who_what_when_where_specificity",
                "description": "Added who/what/when/where specificity guidance",
                "research_basis": "Specificity pattern research",
            }
        ]

    def _add_concrete_examples(self, prompt: str) -> tuple[str, list[dict]]:
        """Add concrete examples to improve specificity"""
        if any(
            example_word in prompt.lower()
            for example_word in SPECIFICITY_PATTERNS["concrete_examples"]
        ):
            return prompt, []  # Already has examples

        example_addition = "\n\nInclude specific examples such as real-world scenarios, sample inputs, or concrete use cases."
        enhanced_prompt = prompt + example_addition

        return enhanced_prompt, [
            {
                "type": "concrete_examples_addition",
                "description": "Added guidance for concrete examples",
                "research_basis": "Example-driven specificity improvement",
            }
        ]

    def _add_quantifiable_metrics(self, prompt: str) -> tuple[str, list[dict]]:
        """Add quantifiable metrics and measurements"""
        if any(
            metric in prompt.lower()
            for metric in SPECIFICITY_PATTERNS["quantifiable_metrics"]
        ):
            return prompt, []  # Already has metrics

        metrics_addition = "\n\nSpecify quantifiable metrics such as numbers, percentages, sizes, or timeframes where applicable."
        enhanced_prompt = prompt + metrics_addition

        return enhanced_prompt, [
            {
                "type": "quantifiable_metrics_addition",
                "description": "Added guidance for quantifiable metrics",
                "research_basis": "Measurable outcome frameworks",
            }
        ]

    def _enforce_specific_outcomes(self, prompt: str) -> tuple[str, list[dict]]:
        """Enforce specific, measurable outcomes"""
        if any(indicator in prompt.lower() for indicator in MEASURABLE_INDICATORS):
            return prompt, []  # Already has specific outcomes

        outcome_enforcement = "\n\nDefine specific, measurable outcomes and success criteria for this task."
        enhanced_prompt = prompt + outcome_enforcement

        return enhanced_prompt, [
            {
                "type": "specific_outcomes_enforcement",
                "description": "Added specific outcome requirements",
                "research_basis": "Measurable outcome research",
            }
        ]

    def _add_success_criteria(self, prompt: str) -> tuple[str, list[dict]]:
        """Add explicit success criteria"""
        if any(keyword in prompt.lower() for keyword in SUCCESS_CRITERIA_KEYWORDS):
            return prompt, []  # Already has success criteria

        criteria_addition = "\n\nSuccess criteria: Define what constitutes successful completion of this task, including quality standards and evaluation methods."
        enhanced_prompt = prompt + criteria_addition

        return enhanced_prompt, [
            {
                "type": "success_criteria_addition",
                "description": "Added explicit success criteria framework",
                "research_basis": "Success criteria definition research",
            }
        ]

    def _replace_vague_language(self, prompt: str) -> tuple[str, list[dict]]:
        """Replace vague language with specific alternatives"""
        improved_prompt = prompt
        transformations = []

        # Specific replacements based on research
        specific_replacements = {
            "some": "approximately 3-5",
            "many": "more than 10",
            "few": "2-3",
            "several": "4-6",
            "thing": "specific item or element",
            "stuff": "relevant information or materials",
            "good": "high-quality and well-structured",
            "bad": "low-quality or inappropriate",
            "nice": "well-designed and effective",
            "appropriate": "suitable for the specific context",
            "relevant": "directly applicable to the requirements",
            "important": "critical for achieving the objectives",
        }

        for vague, specific in specific_replacements.items():
            if re.search(r"\b" + vague + r"\b", improved_prompt, re.IGNORECASE):
                improved_prompt = re.sub(
                    r"\b" + vague + r"\b",
                    specific,
                    improved_prompt,
                    flags=re.IGNORECASE,
                )
                transformations.append({
                    "type": "vague_language_replacement",
                    "description": f"Replaced vague term '{vague}' with specific alternative '{specific}'",
                    "original": vague,
                    "replacement": specific,
                })

        return improved_prompt, transformations

    def _fallback_specificity_improvement(self, prompt: str, metadata: dict) -> dict:
        """Enhanced fallback method with research patterns"""
        improved_prompt = prompt
        transformations = []

        # Apply basic improvements
        additions = []

        # Add format specification if missing constraints
        if not metadata.get("has_success_criteria"):
            additions.append(
                "Success criteria: Provide clear, measurable outcomes with specific quality standards."
            )

        # Add examples if missing and prompt is substantial
        if (
            not metadata.get("has_concrete_examples")
            and metadata.get("word_count", 0) > 10
        ):
            additions.append(
                "Examples: Include specific, real-world examples to illustrate requirements."
            )

        # Add metrics for vague prompts
        if metadata.get("vague_language_ratio", 0) > 0.3:
            additions.append(
                "Specificity: Replace vague terms with concrete, measurable language."
            )

        # Add comprehensive detail for short prompts
        if metadata.get("word_count", 0) < 15:
            additions.append(
                "Details: Provide comprehensive information including context, constraints, and expected outcomes."
            )

        # Apply improvements
        if additions:
            improved_prompt = prompt + "\n\n" + "\n".join(additions)
            transformations.append({
                "type": "basic_specificity_improvement",
                "additions": additions,
                "metadata": metadata,
            })

        return {"improved_prompt": improved_prompt, "transformations": transformations}
