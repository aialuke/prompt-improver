"""Linguistic Quality Rule for advanced prompt analysis.

This rule integrates the LinguisticAnalyzer to assess prompts based on
readability, syntactic complexity, entity richness, and structural clarity.
"""

import logging
from typing import Any, Dict, List

from ...ml.analysis.linguistic_analyzer import LinguisticAnalyzer, LinguisticConfig
from ..base import BasePromptRule, RuleCheckResult, TransformationResult


class LinguisticQualityRule(BasePromptRule):
    """Rule that evaluates prompts using advanced linguistic analysis.

    This rule assesses:
    - Readability and complexity
    - Named entity richness
    - Syntactic structure quality
    - Prompt component clarity
    - Technical term appropriateness
    """

    def __init__(self):
        """Initialize the linguistic quality rule."""
        self.logger = logging.getLogger(__name__)

        # Initialize linguistic analyzer
        config = LinguisticConfig(
            enable_ner=True,
            enable_dependency_parsing=True,
            enable_readability=True,
            enable_complexity_metrics=True,
            enable_prompt_segmentation=True,
            use_transformers_ner=False,  # Use NLTK for faster analysis
            enable_caching=True,
        )

        self.linguistic_analyzer = LinguisticAnalyzer(config)

        # Quality thresholds
        self.thresholds = {
            "min_readability": 0.4,  # Minimum readability score
            "max_complexity": 0.8,  # Maximum syntactic complexity
            "min_instruction_clarity": 0.3,  # Minimum instruction clarity
            "min_entity_diversity": 0.1,  # Minimum entity diversity
            "optimal_sentence_length": 20,  # Optimal average sentence length
            "max_sentence_length": 40,  # Maximum average sentence length
        }

    @property
    def metadata(self):
        """Rule metadata."""
        return {
            "name": "Linguistic Quality Rule",
            "description": "Evaluates prompts using advanced linguistic analysis including NER, dependency parsing, and readability metrics",
            "category": "quality",
            "priority": 7,
            "version": "1.0.0",
        }

    def check(self, prompt: str, context: dict[str, Any] = None) -> RuleCheckResult:
        """Check if linguistic analysis should be applied to this prompt.

        Args:
            prompt: The prompt text to check
            context: Additional context for evaluation

        Returns:
            RuleCheckResult indicating if rule should be applied
        """
        if not prompt.strip():
            return RuleCheckResult(applies=False, confidence=0.0, metadata={})

        # Apply to all non-empty prompts since linguistic analysis is always valuable
        return RuleCheckResult(
            applies=True,
            confidence=0.9,
            metadata={"analysis_type": "linguistic_quality"},
        )

    def apply(
        self, prompt: str, context: dict[str, Any] = None
    ) -> TransformationResult:
        """Apply linguistic analysis and generate improvement suggestions.

        Args:
            prompt: The prompt text to improve
            context: Additional context for evaluation

        Returns:
            TransformationResult containing analysis and suggestions
        """
        if not prompt.strip():
            return TransformationResult(
                success=False,
                improved_prompt=prompt,
                confidence=0.0,
                transformations=[],
            )

        try:
            # Perform linguistic analysis
            features = self.linguistic_analyzer.analyze(prompt)

            # Calculate component scores
            readability_score = self._assess_readability(features)
            complexity_score = self._assess_complexity(features)
            structure_score = self._assess_structure(features)
            entity_score = self._assess_entity_richness(features)
            clarity_score = self._assess_clarity(features)

            # Calculate overall score
            overall_score = self._calculate_overall_score({
                "readability": readability_score,
                "complexity": complexity_score,
                "structure": structure_score,
                "entity_richness": entity_score,
                "clarity": clarity_score,
            })

            # Generate suggestions
            suggestions = self._generate_suggestions(
                features,
                {
                    "readability": readability_score,
                    "complexity": complexity_score,
                    "structure": structure_score,
                    "entity_richness": entity_score,
                    "clarity": clarity_score,
                },
            )

            # Create improved prompt with suggestions
            improved_prompt = self._create_improved_prompt(prompt, suggestions)

            # Create transformations list
            transformations = []
            for suggestion in suggestions:
                transformations.append({
                    "type": "linguistic_improvement",
                    "description": suggestion,
                    "confidence": 0.8,
                })

            return TransformationResult(
                success=True,
                improved_prompt=improved_prompt,
                confidence=features.confidence,
                transformations=transformations,
            )

        except Exception as e:
            self.logger.error(f"Linguistic quality analysis failed: {e}")
            return TransformationResult(
                success=False,
                improved_prompt=prompt,
                confidence=0.0,
                transformations=[],
            )

    def to_llm_instruction(self) -> str:
        """Generate LLM instruction for applying linguistic improvements.

        Returns:
            String instruction for LLM to apply linguistic improvements
        """
        return """Analyze this prompt for linguistic quality and improve it by:
1. Enhancing readability and clarity
2. Improving sentence structure and flow
3. Adding specific technical terms where appropriate
4. Ensuring clear instructions and examples
5. Maintaining appropriate complexity level

Focus on making the prompt more precise, readable, and effective while preserving its original intent."""

    def _assess_readability(self, features) -> float:
        """Assess readability quality."""
        if features.readability_score == 0:
            return 0.5  # Neutral score if analysis failed

        readability_score = features.readability_score

        # Penalize if too low or too high (too simple can lack detail)
        if readability_score < self.thresholds["min_readability"]:
            return readability_score * 2  # Boost low scores
        if readability_score > 0.9:
            return 0.9  # Cap very high scores
        return readability_score

    def _assess_complexity(self, features) -> float:
        """Assess syntactic complexity (balanced complexity is good)."""
        if features.syntactic_complexity == 0:
            return 0.7  # Assume reasonable complexity if no analysis

        complexity = features.syntactic_complexity

        # Optimal complexity is moderate (not too simple, not too complex)
        if complexity < 0.3:
            return 0.6  # Too simple
        if complexity > self.thresholds["max_complexity"]:
            return 1.0 - complexity  # Too complex
        return 0.8 + (0.2 * (1 - abs(complexity - 0.5) * 2))  # Optimal range

    def _assess_structure(self, features) -> float:
        """Assess structural quality."""
        structure_score = features.sentence_structure_quality

        # Adjust based on sentence length
        if features.avg_sentence_length > 0:
            length_penalty = 0
            if features.avg_sentence_length > self.thresholds["max_sentence_length"]:
                length_penalty = 0.2
            elif features.avg_sentence_length < 5:
                length_penalty = 0.1  # Too short

            structure_score = max(0, structure_score - length_penalty)

        return structure_score

    def _assess_entity_richness(self, features) -> float:
        """Assess entity richness and diversity."""
        entity_count = len(features.entities)
        entity_density = features.entity_density
        technical_term_count = len(features.technical_terms)

        # Calculate richness score
        richness_score = 0.0

        # Entity diversity component
        if entity_density >= self.thresholds["min_entity_diversity"]:
            richness_score += 0.4

        # Technical relevance component
        if technical_term_count > 0:
            richness_score += min(technical_term_count * 0.1, 0.3)

        # Entity variety component
        if len(features.entity_types) > 1:
            richness_score += 0.3

        return min(richness_score, 1.0)

    def _assess_clarity(self, features) -> float:
        """Assess instruction clarity."""
        clarity_score = features.instruction_clarity_score

        # Bonus for good prompt structure
        structure_bonus = 0
        if features.has_clear_instructions:
            structure_bonus += 0.1
        if features.has_examples:
            structure_bonus += 0.1
        if features.has_context:
            structure_bonus += 0.05

        return min(clarity_score + structure_bonus, 1.0)

    def _calculate_overall_score(self, component_scores: dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = {
            "readability": 0.25,
            "syntactic_complexity": 0.20,
            "structure_quality": 0.20,
            "entity_richness": 0.15,
            "instruction_clarity": 0.20,
        }

        total_score = sum(
            component_scores.get(component, 0.0) * weight
            for component, weight in weights.items()
        )

        return round(total_score, 3)

    def _generate_suggestions(
        self, features, component_scores: dict[str, float]
    ) -> list[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        # Readability suggestions
        if component_scores["readability"] < 0.6:
            if features.flesch_reading_ease < 30:
                suggestions.append(
                    "Consider simplifying sentence structure and using shorter sentences for better readability."
                )
            if features.avg_sentence_length > self.thresholds["max_sentence_length"]:
                suggestions.append(
                    "Break down long sentences into shorter, clearer statements."
                )

        # Complexity suggestions
        if component_scores.get("syntactic_complexity", 0) < 0.6:
            if features.syntactic_complexity > self.thresholds["max_complexity"]:
                suggestions.append(
                    "Reduce syntactic complexity by using simpler grammatical structures."
                )
            elif features.syntactic_complexity < 0.2:
                suggestions.append(
                    "Add more detail and structure to make the prompt more informative."
                )

        # Structure suggestions
        if component_scores.get("structure_quality", 0) < 0.6:
            suggestions.append(
                "Improve sentence structure with clearer subject-verb-object relationships."
            )

        # Entity richness suggestions
        if component_scores.get("entity_richness", 0) < 0.5:
            suggestions.append(
                "Include more specific terms, examples, or domain-relevant concepts."
            )

        # Clarity suggestions
        if component_scores.get("instruction_clarity", 0) < 0.6:
            if not features.has_clear_instructions:
                suggestions.append(
                    "Add clear action words (write, create, analyze, etc.) to specify the task."
                )
            if not features.has_examples:
                suggestions.append(
                    "Include examples to illustrate the desired output or approach."
                )
            if not features.has_context:
                suggestions.append(
                    "Provide context or background information to clarify the task scope."
                )

        return suggestions

    def _create_improved_prompt(
        self, original_prompt: str, suggestions: list[str]
    ) -> str:
        """Create an improved version of the prompt with suggestions."""
        if not suggestions:
            return original_prompt

        # For now, return original prompt with suggestions as comments
        # In a more sophisticated implementation, this could use LLM to actually improve the prompt
        improved_prompt = original_prompt

        if suggestions:
            improved_prompt += "\n\n# Linguistic Quality Suggestions:\n"
            for i, suggestion in enumerate(suggestions, 1):
                improved_prompt += f"# {i}. {suggestion}\n"

        return improved_prompt

    def evaluate(self, prompt: str, context: dict[str, Any] = None) -> dict[str, Any]:
        """Evaluate prompt using real linguistic analysis (for testing compatibility).

        Args:
            prompt: The prompt text to evaluate
            context: Additional context for evaluation

        Returns:
            Dictionary containing evaluation results matching test expectations
        """
        if not prompt.strip():
            return {
                "score": 0.0,
                "confidence": 0.0,
                "passed": False,
                "component_scores": {},
                "linguistic_features": {},
                "suggestions": [],
                "explanation": "Empty prompt cannot be evaluated",
                "metadata": {"rule_name": "Linguistic Quality Rule", "error": False}
            }

        try:
            # Perform linguistic analysis
            features = self.linguistic_analyzer.analyze(prompt)

            # Calculate component scores
            readability_score = self._assess_readability(features)
            complexity_score = self._assess_complexity(features)
            structure_score = self._assess_structure(features)
            entity_score = self._assess_entity_richness(features)
            clarity_score = self._assess_clarity(features)

            component_scores = {
                "readability": readability_score,
                "syntactic_complexity": complexity_score,
                "structure_quality": structure_score,
                "entity_richness": entity_score,
                "instruction_clarity": clarity_score,
            }

            # Calculate overall score
            overall_score = self._calculate_overall_score(component_scores)

            # Generate suggestions
            suggestions = self._generate_suggestions(features, component_scores)

            # Create linguistic features dict for test compatibility
            linguistic_features = {
                "flesch_reading_ease": features.flesch_reading_ease,
                "avg_sentence_length": features.avg_sentence_length,
                "lexical_diversity": features.lexical_diversity,
                "entity_count": len(features.entities),
                "technical_terms": features.technical_terms,
                "has_clear_instructions": features.has_clear_instructions,
                "has_examples": features.has_examples,
                "has_context": features.has_context,
            }

            # Generate explanation
            explanation = self._generate_explanation(overall_score, suggestions)

            # Determine if passed
            passed = overall_score >= 0.5 and len(suggestions) <= 3

            return {
                "score": overall_score,
                "confidence": features.confidence,
                "passed": passed,
                "component_scores": component_scores,
                "linguistic_features": linguistic_features,
                "suggestions": suggestions,
                "explanation": explanation,
                "metadata": {
                    "rule_name": "Linguistic Quality Rule",
                    "analysis_method": "advanced_linguistic",
                    "features_analyzed": list(linguistic_features.keys()),
                    "error": False
                }
            }

        except Exception as e:
            self.logger.error(f"Linguistic quality evaluation failed: {e}")
            return {
                "score": 0.0,
                "confidence": 0.0,
                "passed": False,
                "component_scores": {},
                "linguistic_features": {},
                "suggestions": [],
                "explanation": f"Error during linguistic analysis: {e!s}",
                "metadata": {"rule_name": "Linguistic Quality Rule", "error": True}
            }

    def _generate_explanation(self, score: float, suggestions: list[str]) -> str:
        """Generate explanation based on score and suggestions."""
        if score >= 0.8:
            return "Excellent linguistic quality with clear structure and good readability."
        if score >= 0.6:
            return "Good linguistic quality with minor areas for improvement."
        if score >= 0.4:
            return "Moderate linguistic quality that needs improvement in several areas."
        return "Poor linguistic quality requiring significant improvement in clarity, structure, and readability."
