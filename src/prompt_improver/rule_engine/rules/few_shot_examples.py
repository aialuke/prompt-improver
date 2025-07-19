"""FewShotExampleRule: Enhanced rule for optimal few-shot example integration.

Based on research synthesis from:
- Brown et al. (2020) few-shot learning research
- PromptHub example optimization studies
- IBM research on example diversity and effectiveness
- Recency bias optimization techniques
"""

import random
import re
from typing import Any, Dict, List, Tuple

from ..base import BasePromptRule, RuleCheckResult, TransformationResult

# Task type patterns for example selection
TASK_TYPE_PATTERNS = {
    "classification": [
        r"\b(classify|categorize|label|identify|determine)\b",
        r"\b(what (type|kind|category)|which (class|group))\b",
    ],
    "generation": [
        r"\b(write|create|generate|compose|draft)\b",
        r"\b(make|build|develop|produce)\b",
    ],
    "transformation": [
        r"\b(rewrite|rephrase|translate|convert|transform)\b",
        r"\b(change|modify|adapt|adjust)\b",
    ],
    "analysis": [
        r"\b(analyze|evaluate|assess|examine|review)\b",
        r"\b(what does|what is|explain|describe)\b",
    ],
    "extraction": [
        r"\b(extract|find|identify|locate|get)\b",
        r"\b(list|enumerate|name|mention)\b",
    ],
    "comparison": [
        r"\b(compare|contrast|difference|similarity)\b",
        r"\b(versus|vs|better|worse)\b",
    ],
}

# Example quality indicators
QUALITY_INDICATORS = [
    "clear_format",
    "diverse_content",
    "appropriate_length",
    "relevant_domain",
    "balanced_examples",
    "progressive_difficulty",
]


class FewShotExampleRule(BasePromptRule):
    """Enhanced few-shot example rule using research-validated optimization patterns.

    Features:
    - 2-5 optimal examples based on Brown et al. research
    - Diverse examples with balanced positive/negative cases
    - XML delimiters for clear structure
    - Recency bias optimization (strongest example last)
    - Domain-specific example generation
    """

    def __init__(self):
        # Research-validated default parameters
        self.config = {
            "optimal_example_count": 3,
            "require_diverse_examples": True,
            "include_negative_examples": True,
            "use_xml_delimiters": True,
            "example_placement": "after_context",
            "recency_bias_optimization": True,
            "domain_specific_examples": True,
            "format_consistency_check": True,
        }

        # Attributes for dynamic loading system
        self.rule_id = "few_shot_examples"
        self.priority = 7

    def configure(self, params: dict[str, Any]):
        """Configure rule parameters from database"""
        self.config.update(params)

    @property
    def metadata(self):
        """Enhanced metadata with research foundation"""
        return {
            "name": "Few-Shot Example Integration Rule",
            "type": "Examples",
            "description": "Incorporates 2-5 optimal examples based on research from PromptHub and OpenAI documentation",
            "category": "examples",
            "research_foundation": [
                "Brown et al. (2020) few-shot learning",
                "PromptHub example optimization",
                "IBM research on example diversity",
                "Recency bias studies",
            ],
            "version": "2.0.0",
            "priority": self.priority,
            "source": "Research Synthesis 2025",
        }

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Check if prompt would benefit from few-shot examples"""
        example_metrics = self._analyze_example_requirements(prompt)

        # Determine if few-shot examples should be added
        applies = (
            example_metrics["task_benefits_from_examples"]
            and not example_metrics["already_has_examples"]
            and example_metrics["task_complexity"] > 0.3
        )

        confidence = 0.9 if applies else 0.85

        return RuleCheckResult(
            applies=applies,
            confidence=confidence,
            metadata={
                "task_type": example_metrics["task_type"],
                "task_complexity": example_metrics["task_complexity"],
                "task_benefits_from_examples": example_metrics[
                    "task_benefits_from_examples"
                ],
                "already_has_examples": example_metrics["already_has_examples"],
                "recommended_example_count": example_metrics[
                    "recommended_example_count"
                ],
                "domain": example_metrics["domain"],
                "format_requirements": example_metrics["format_requirements"],
            },
        )

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Apply few-shot example enhancement"""
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(
                success=True, improved_prompt=prompt, confidence=1.0, transformations=[]
            )

        example_metrics = check_result.metadata
        improved_prompt = prompt
        transformations = []

        # Generate appropriate examples
        task_type = example_metrics.get("task_type", "general")
        domain = example_metrics.get("domain", "general")
        example_count = min(
            self.config["optimal_example_count"],
            example_metrics.get("recommended_example_count", 3),
        )

        # Create examples based on task type and domain
        examples = self._generate_examples(task_type, domain, example_count)

        if examples:
            # Apply examples with optimal placement and formatting
            improved_prompt, example_transformations = self._integrate_examples(
                improved_prompt, examples, example_metrics
            )
            transformations.extend(example_transformations)

        # Calculate confidence based on example quality
        confidence = min(0.95, 0.7 + (len(examples) * 0.05))

        return TransformationResult(
            success=True,
            improved_prompt=improved_prompt,
            confidence=confidence,
            transformations=transformations,
        )

    def to_llm_instruction(self) -> str:
        """Generate research-based LLM instruction for few-shot examples"""
        return """
<instruction>
Add few-shot examples using research-validated patterns:

1. OPTIMAL COUNT (Brown et al. 2020):
   - Use 2-5 examples (research shows diminishing returns beyond 5)
   - Balance example count with prompt length constraints
   - Prioritize quality over quantity

2. EXAMPLE DIVERSITY (IBM Research):
   - Include varied scenarios and edge cases
   - Mix positive and negative examples when appropriate
   - Cover different aspects of the task

3. RECENCY BIAS OPTIMIZATION:
   - Place strongest, most representative example last
   - Order examples from simple to complex
   - End with the most relevant case

4. CLEAR FORMATTING:
   - Use XML delimiters for structure (<example1>, <example2>)
   - Maintain consistent format across examples
   - Include input-output pairs where applicable

Focus on examples that clearly demonstrate the expected format and quality.
</instruction>
"""

    def _analyze_example_requirements(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt to determine few-shot example requirements"""
        words = prompt.lower().split()

        # Determine task type
        task_type = self._identify_task_type(prompt)

        # Determine domain
        domain = self._identify_domain(prompt)

        # Check if examples would benefit this task type
        example_beneficial_tasks = [
            "classification",
            "generation",
            "transformation",
            "extraction",
        ]
        task_benefits_from_examples = task_type in example_beneficial_tasks

        # Check if prompt already has examples
        example_indicators = [
            "example",
            "for instance",
            "such as",
            "like",
            "e.g.",
            "i.e.",
        ]
        already_has_examples = any(
            indicator in prompt.lower() for indicator in example_indicators
        )

        # Calculate task complexity
        complexity_indicators = len([
            word
            for word in words
            if word
            in [
                "complex",
                "detailed",
                "comprehensive",
                "specific",
                "nuanced",
                "sophisticated",
                "advanced",
                "technical",
                "specialized",
            ]
        ])
        task_complexity = min(1.0, complexity_indicators / max(len(words), 1) * 10)

        # Add base complexity for certain task types
        if task_type in ["generation", "transformation", "analysis"]:
            task_complexity = max(task_complexity, 0.5)

        # Determine recommended example count based on complexity
        if task_complexity > 0.7:
            recommended_count = 5
        elif task_complexity > 0.5:
            recommended_count = 4
        elif task_complexity > 0.3:
            recommended_count = 3
        else:
            recommended_count = 2

        # Check for format requirements
        format_requirements = self._identify_format_requirements(prompt)

        return {
            "task_type": task_type,
            "domain": domain,
            "task_complexity": task_complexity,
            "task_benefits_from_examples": task_benefits_from_examples,
            "already_has_examples": already_has_examples,
            "recommended_example_count": recommended_count,
            "format_requirements": format_requirements,
        }

    def _identify_task_type(self, prompt: str) -> str:
        """Identify the primary task type from the prompt"""
        prompt_lower = prompt.lower()

        for task_type, patterns in TASK_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return task_type

        return "general"

    def _identify_domain(self, prompt: str) -> str:
        """Identify the domain or subject area of the prompt"""
        prompt_lower = prompt.lower()

        domain_keywords = {
            "business": [
                "business",
                "marketing",
                "sales",
                "finance",
                "corporate",
                "company",
            ],
            "technical": [
                "code",
                "programming",
                "software",
                "algorithm",
                "technical",
                "system",
            ],
            "academic": [
                "research",
                "study",
                "academic",
                "scholarly",
                "paper",
                "thesis",
            ],
            "creative": ["creative", "story", "poem", "art", "design", "writing"],
            "legal": ["legal", "law", "contract", "regulation", "compliance", "policy"],
            "medical": [
                "medical",
                "health",
                "patient",
                "diagnosis",
                "treatment",
                "clinical",
            ],
            "education": [
                "teach",
                "learn",
                "student",
                "education",
                "training",
                "course",
            ],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return domain

        return "general"

    def _identify_format_requirements(self, prompt: str) -> dict[str, Any]:
        """Identify specific format requirements from the prompt"""
        format_requirements = {
            "structured": False,
            "json": False,
            "xml": False,
            "bullet_points": False,
            "numbered_list": False,
            "table": False,
        }

        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["json", "object", "key-value"]):
            format_requirements["json"] = True
        if any(word in prompt_lower for word in ["xml", "tag", "markup"]):
            format_requirements["xml"] = True
        if any(word in prompt_lower for word in ["bullet", "points", "•", "*"]):
            format_requirements["bullet_points"] = True
        if any(word in prompt_lower for word in ["numbered", "list", "1.", "2."]):
            format_requirements["numbered_list"] = True
        if any(word in prompt_lower for word in ["table", "column", "row"]):
            format_requirements["table"] = True
        if any(word in prompt_lower for word in ["format", "structure", "template"]):
            format_requirements["structured"] = True

        return format_requirements

    def _generate_examples(
        self, task_type: str, domain: str, count: int
    ) -> list[dict[str, str]]:
        """Generate appropriate examples based on task type and domain"""
        example_templates = self._get_example_templates(task_type, domain)

        if not example_templates:
            return []

        # Select diverse examples
        selected_examples = []

        # Ensure we have enough templates or generate variations
        if len(example_templates) >= count:
            # Use research-based selection (diverse + recency bias optimization)
            selected_examples = self._select_diverse_examples(example_templates, count)
        else:
            # Use all available and generate variations if needed
            selected_examples = example_templates
            while len(selected_examples) < count and len(example_templates) > 0:
                # Add variations of existing examples
                variation = self._create_example_variation(
                    example_templates[0], len(selected_examples)
                )
                selected_examples.append(variation)

        # Apply recency bias optimization (strongest example last)
        if self.config["recency_bias_optimization"]:
            selected_examples = self._optimize_example_order(selected_examples)

        return selected_examples[:count]

    def _get_example_templates(
        self, task_type: str, domain: str
    ) -> list[dict[str, str]]:
        """Get example templates based on task type and domain"""
        # Base templates by task type
        base_templates = {
            "classification": [
                {
                    "input": "This product review is very positive and enthusiastic.",
                    "output": "Sentiment: Positive",
                },
                {
                    "input": "The service was disappointing and slow.",
                    "output": "Sentiment: Negative",
                },
                {
                    "input": "The experience was okay, nothing special.",
                    "output": "Sentiment: Neutral",
                },
            ],
            "generation": [
                {
                    "input": "Write a professional email requesting a meeting.",
                    "output": "Subject: Meeting Request\n\nDear [Name],\n\nI hope this email finds you well. I would like to schedule a meeting to discuss [topic]. Please let me know your availability for next week.\n\nBest regards,\n[Your name]",
                },
                {
                    "input": "Create a brief product description for wireless headphones.",
                    "output": "Premium wireless headphones featuring noise cancellation, 30-hour battery life, and crystal-clear audio quality. Perfect for music lovers and professionals alike.",
                },
            ],
            "transformation": [
                {
                    "input": 'Make this formal: "Hey, can you send me the report?"',
                    "output": "Could you please provide me with the report at your earliest convenience?",
                },
                {
                    "input": 'Simplify: "The implementation of the aforementioned protocol requires substantial consideration."',
                    "output": "Setting up this protocol needs careful thought.",
                },
            ],
            "extraction": [
                {
                    "input": 'Extract the main points from: "Our sales increased 25% this quarter due to new marketing strategies and improved customer service."',
                    "output": "• Sales increased 25% this quarter\n• Improvement due to new marketing strategies\n• Enhanced customer service contributed to growth",
                }
            ],
        }

        # Get base templates for task type
        templates = base_templates.get(task_type, [])

        # Customize for domain if possible
        if domain != "general":
            templates = self._customize_for_domain(templates, domain)

        return templates

    def _customize_for_domain(
        self, templates: list[dict[str, str]], domain: str
    ) -> list[dict[str, str]]:
        """Customize examples for specific domain"""
        # This is a simplified implementation - in production, you'd have comprehensive domain-specific examples
        domain_customizations = {
            "business": {
                "keywords": ["revenue", "strategy", "market", "client", "ROI"],
                "context": "business environment",
            },
            "technical": {
                "keywords": ["code", "function", "algorithm", "system", "API"],
                "context": "technical implementation",
            },
            "academic": {
                "keywords": [
                    "research",
                    "hypothesis",
                    "methodology",
                    "analysis",
                    "findings",
                ],
                "context": "academic research",
            },
        }

        customization = domain_customizations.get(domain, {})

        # Apply light customization (in production, this would be more sophisticated)
        if customization:
            # This is a placeholder - real implementation would have domain-specific example databases
            pass

        return templates

    def _select_diverse_examples(
        self, templates: list[dict[str, str]], count: int
    ) -> list[dict[str, str]]:
        """Select diverse examples using research-based criteria"""
        if len(templates) <= count:
            return templates

        # Simple diversity selection (in production, would use more sophisticated metrics)
        selected = []
        remaining = templates.copy()

        # First, select the most representative example
        representative = remaining[0]  # Simplified selection
        selected.append(representative)
        remaining.remove(representative)

        # Then select diverse examples
        while len(selected) < count and remaining:
            # Simple diversity metric: different input length and content
            best_candidate = None
            best_diversity_score = -1

            for candidate in remaining:
                diversity_score = self._calculate_diversity_score(candidate, selected)
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def _calculate_diversity_score(
        self, candidate: dict[str, str], selected: list[dict[str, str]]
    ) -> float:
        """Calculate diversity score for example selection"""
        if not selected:
            return 1.0

        # Simple diversity metrics
        candidate_length = len(candidate.get("input", ""))
        candidate_words = set(candidate.get("input", "").lower().split())

        diversity_scores = []
        for existing in selected:
            existing_length = len(existing.get("input", ""))
            existing_words = set(existing.get("input", "").lower().split())

            # Length diversity (normalized difference)
            length_diversity = abs(candidate_length - existing_length) / max(
                candidate_length, existing_length, 1
            )

            # Content diversity (word overlap)
            word_overlap = (
                len(candidate_words & existing_words)
                / len(candidate_words | existing_words)
                if candidate_words | existing_words
                else 0
            )
            content_diversity = 1.0 - word_overlap

            # Combined diversity score
            diversity = (length_diversity + content_diversity) / 2
            diversity_scores.append(diversity)

        # Return average diversity from all selected examples
        return sum(diversity_scores) / len(diversity_scores)

    def _optimize_example_order(
        self, examples: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Optimize example order for recency bias (strongest example last)"""
        if len(examples) <= 1:
            return examples

        # Simple ordering: shortest to longest (complexity proxy)
        # In production, would use more sophisticated quality metrics
        ordered = sorted(examples, key=lambda x: len(x.get("input", "")))

        return ordered

    def _create_example_variation(
        self, template: dict[str, str], variation_index: int
    ) -> dict[str, str]:
        """Create a variation of an existing example"""
        # Simple variation (in production, would be more sophisticated)
        input_text = template.get("input", "")
        output_text = template.get("output", "")

        # Add slight variation to input
        variations = [
            f"Alternative scenario: {input_text}",
            f"Similar case: {input_text}",
            f"Related example: {input_text}",
        ]

        return {
            "input": variations[variation_index % len(variations)],
            "output": output_text,
        }

    def _integrate_examples(
        self, prompt: str, examples: list[dict[str, str]], metrics: dict
    ) -> tuple[str, list[dict]]:
        """Integrate examples into the prompt with optimal formatting"""
        if not examples:
            return prompt, []

        transformations = []

        # Format examples with XML delimiters if enabled
        if self.config["use_xml_delimiters"]:
            formatted_examples = []
            for i, example in enumerate(examples, 1):
                if "input" in example and "output" in example:
                    formatted_example = f"<example{i}>\nInput: {example['input']}\nOutput: {example['output']}\n</example{i}>"
                else:
                    formatted_example = (
                        f"<example{i}>\n{example.get('input', '')}\n</example{i}>"
                    )
                formatted_examples.append(formatted_example)

            examples_text = "\n\n".join(formatted_examples)
        else:
            # Simple formatting
            formatted_examples = []
            for i, example in enumerate(examples, 1):
                if "input" in example and "output" in example:
                    formatted_example = f"Example {i}:\nInput: {example['input']}\nOutput: {example['output']}"
                else:
                    formatted_example = f"Example {i}: {example.get('input', '')}"
                formatted_examples.append(formatted_example)

            examples_text = "\n\n".join(formatted_examples)

        # Integrate examples based on placement configuration
        placement = self.config["example_placement"]

        if placement == "before_prompt":
            enhanced_prompt = f"Here are some examples:\n\n{examples_text}\n\n{prompt}"
        elif placement == "after_context":
            # Try to identify context vs instruction
            if len(prompt.split(".")) > 2:
                sentences = prompt.split(".")
                context_part = ".".join(sentences[: len(sentences) // 2]) + "."
                instruction_part = ".".join(sentences[len(sentences) // 2 :])
                enhanced_prompt = f"{context_part}\n\nExamples:\n{examples_text}\n\n{instruction_part}"
            else:
                enhanced_prompt = f"{prompt}\n\nExamples:\n{examples_text}"
        else:  # default: after prompt
            enhanced_prompt = f"{prompt}\n\nExamples:\n{examples_text}"

        transformations.append({
            "type": "few_shot_examples",
            "description": f"Added {len(examples)} diverse examples with XML formatting",
            "example_count": len(examples),
            "placement": placement,
            "research_basis": "Brown et al. 2-5 optimal examples + diversity optimization",
        })

        return enhanced_prompt, transformations
