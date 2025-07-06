"""LLM-based prompt transformation service.
Provides intelligent prompt enhancement using language models.
"""

import re
from typing import Any, Dict, List, Optional

# For now, we'll use a simple rule-based approach with structured improvements
# In production, this would integrate with actual LLM APIs like OpenAI, Anthropic, etc.


class LLMTransformerService:
    """Service for LLM-based prompt transformations"""

    def __init__(self):
        """Initialize the transformer service"""
        self.transformation_patterns = self._load_transformation_patterns()

    async def enhance_clarity(
        self, prompt: str, vague_words: list[str], context: dict | None = None
    ) -> dict[str, Any]:
        """Enhance prompt clarity by replacing vague terms with specific alternatives.

        Args:
            prompt: The original prompt
            vague_words: List of vague words detected
            context: Optional context for enhancement

        Returns:
            Enhanced prompt with transformation details
        """
        enhanced_prompt = prompt
        transformations = []
        confidence = 0.8

        # Apply intelligent transformations for each vague word
        for vague_word in vague_words:
            if vague_word in self.transformation_patterns["clarity"]:
                pattern_info = self.transformation_patterns["clarity"][vague_word]

                # Find context-appropriate replacement
                replacement = self._find_best_replacement(
                    vague_word, prompt, pattern_info, context
                )

                if replacement:
                    # Replace the vague word with specific alternative
                    enhanced_prompt = self._replace_with_context(
                        enhanced_prompt, vague_word, replacement
                    )

                    transformations.append({
                        "type": "clarity_enhancement",
                        "original_word": vague_word,
                        "replacement": replacement,
                        "reason": pattern_info.get("reason", "Improved specificity"),
                    })

        # If no direct replacements, add contextual guidance
        if not transformations:
            guidance = self._generate_clarity_guidance(prompt, vague_words)
            if guidance:
                enhanced_prompt = f"{prompt}\n\n{guidance}"
                transformations.append({
                    "type": "clarity_guidance",
                    "guidance": guidance,
                    "reason": "Added specific instructions for clarity",
                })

        return {
            "enhanced_prompt": enhanced_prompt,
            "transformations": transformations,
            "confidence": confidence,
            "improvement_type": "clarity",
        }

    async def enhance_specificity(
        self, prompt: str, context: dict | None = None
    ) -> dict[str, Any]:
        """Enhance prompt specificity by adding constraints and examples.

        Args:
            prompt: The original prompt
            context: Optional context for enhancement

        Returns:
            Enhanced prompt with transformation details
        """
        enhanced_prompt = prompt
        transformations = []
        confidence = 0.75

        # Analyze prompt structure
        analysis = self._analyze_prompt_structure(prompt)

        # Add specificity enhancements based on analysis
        if analysis["lacks_format_specification"]:
            format_spec = self._suggest_format_specification(prompt, context)
            if format_spec:
                enhanced_prompt += f"\n\n{format_spec}"
                transformations.append({
                    "type": "format_specification",
                    "addition": format_spec,
                    "reason": "Added output format requirements",
                })

        if analysis["lacks_constraints"]:
            constraints = self._suggest_constraints(prompt, context)
            if constraints:
                enhanced_prompt += f"\n\n{constraints}"
                transformations.append({
                    "type": "constraint_addition",
                    "addition": constraints,
                    "reason": "Added specific constraints",
                })

        if analysis["needs_examples"] and not analysis["has_examples"]:
            examples = self._generate_examples(prompt, context)
            if examples:
                enhanced_prompt += f"\n\n{examples}"
                transformations.append({
                    "type": "example_addition",
                    "addition": examples,
                    "reason": "Added clarifying examples",
                })

        return {
            "enhanced_prompt": enhanced_prompt,
            "transformations": transformations,
            "confidence": confidence,
            "improvement_type": "specificity",
        }

    def _load_transformation_patterns(self) -> dict[str, Any]:
        """Load transformation patterns for different rule types"""
        return {
            "clarity": {
                "thing": {
                    "replacements": {
                        "document": ["document", "file", "text", "article"],
                        "data": ["dataset", "information", "data points", "records"],
                        "analysis": [
                            "analysis report",
                            "detailed analysis",
                            "comprehensive review",
                        ],
                        "default": [
                            "specific item",
                            "particular element",
                            "concrete example",
                        ],
                    },
                    "reason": "Replace 'thing' with specific noun",
                },
                "stuff": {
                    "replacements": {
                        "content": ["content", "material", "information"],
                        "data": ["data", "information", "details"],
                        "analysis": ["findings", "results", "insights"],
                        "default": [
                            "specific items",
                            "particular elements",
                            "relevant details",
                        ],
                    },
                    "reason": "Replace 'stuff' with specific noun",
                },
                "analyze": {
                    "replacements": {
                        "business": [
                            "examine",
                            "evaluate",
                            "assess",
                            "review systematically",
                        ],
                        "data": ["examine", "interpret", "evaluate patterns in"],
                        "text": ["examine", "review", "evaluate the content of"],
                        "default": [
                            "examine thoroughly",
                            "evaluate systematically",
                            "assess in detail",
                        ],
                    },
                    "reason": "Replace generic 'analyze' with specific action",
                },
                "summarize": {
                    "replacements": {
                        "long": [
                            "create a concise summary of",
                            "provide key points from",
                        ],
                        "complex": [
                            "distill the main ideas from",
                            "extract essential points from",
                        ],
                        "default": [
                            "provide a clear summary of",
                            "outline the key points of",
                        ],
                    },
                    "reason": "Replace generic 'summarize' with specific instruction",
                },
            }
        }

    def _find_best_replacement(
        self, vague_word: str, prompt: str, pattern_info: dict, context: dict | None
    ) -> str | None:
        """Find the best replacement for a vague word based on context"""
        replacements = pattern_info.get("replacements", {})

        # Analyze prompt context to choose best replacement
        prompt_lower = prompt.lower()

        # Check for domain-specific context
        for domain, replacement_list in replacements.items():
            if domain == "default":
                continue

            # Look for domain keywords in prompt
            if domain in prompt_lower:
                return replacement_list[0]  # Return first (most common) replacement

        # Check context dictionary if provided
        if context and context.get("domain"):
            domain = context["domain"].lower()
            if domain in replacements:
                return replacements[domain][0]

        # Fall back to default replacement
        default_replacements = replacements.get("default", [])
        return default_replacements[0] if default_replacements else None

    def _replace_with_context(
        self, prompt: str, vague_word: str, replacement: str
    ) -> str:
        """Replace vague word with replacement, preserving context"""
        # Simple word boundary replacement
        pattern = r"\b" + re.escape(vague_word) + r"\b"
        return re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)

    def _generate_clarity_guidance(
        self, prompt: str, vague_words: list[str]
    ) -> str | None:
        """Generate guidance for improving clarity"""
        if not vague_words:
            return None

        guidance = "For improved clarity, consider specifying:"
        suggestions = []

        for word in vague_words[:3]:  # Limit to top 3
            if word.lower() == "thing":
                suggestions.append(
                    "- What specific item or concept you're referring to"
                )
            elif word.lower() == "stuff":
                suggestions.append("- What specific content or materials you mean")
            elif word.lower() == "analyze":
                suggestions.append(
                    "- What type of analysis you need (summary, comparison, evaluation)"
                )
            elif word.lower() == "summarize":
                suggestions.append("- What format and length you want for the summary")
            else:
                suggestions.append(f"- What you mean by '{word}' in this context")

        if suggestions:
            return guidance + "\n" + "\n".join(suggestions)

        return None

    def _analyze_prompt_structure(self, prompt: str) -> dict[str, bool]:
        """Analyze prompt structure to identify areas for improvement"""
        prompt_lower = prompt.lower()

        # Check for format specifications
        format_indicators = ["format", "structure", "organize", "layout", "present"]
        has_format = any(indicator in prompt_lower for indicator in format_indicators)

        # Check for constraints
        constraint_indicators = [
            "limit",
            "maximum",
            "minimum",
            "no more than",
            "at least",
            "exactly",
        ]
        has_constraints = any(
            indicator in prompt_lower for indicator in constraint_indicators
        )

        # Check for examples
        example_indicators = ["example", "for instance", "such as", "like", "e.g."]
        has_examples = any(
            indicator in prompt_lower for indicator in example_indicators
        )

        # Check if examples would be helpful
        instruction_words = ["explain", "describe", "show", "demonstrate", "illustrate"]
        needs_examples = any(word in prompt_lower for word in instruction_words)

        # Check for very short, vague prompts
        is_very_short = len(prompt.split()) < 5
        has_vague_words = any(
            word in prompt_lower
            for word in ["better", "good", "nice", "fix", "improve", "help"]
        )

        return {
            "lacks_format_specification": not has_format
            and (len(prompt.split()) > 10 or is_very_short),
            "lacks_constraints": not has_constraints
            and (
                any(word in prompt_lower for word in ["list", "write", "create"])
                or is_very_short
            ),
            "has_examples": has_examples,
            "needs_examples": (needs_examples and not has_examples)
            or (is_very_short and has_vague_words),
        }

    def _suggest_format_specification(
        self, prompt: str, context: dict | None
    ) -> str | None:
        """Suggest format specifications based on prompt analysis"""
        prompt_lower = prompt.lower()

        if "list" in prompt_lower:
            return "Format: Please provide your response as a numbered list with clear bullet points."
        if "summary" in prompt_lower or "summarize" in prompt_lower:
            return "Format: Please structure your response with key points in bullet format, limiting to 3-5 main ideas."
        if "explain" in prompt_lower or "describe" in prompt_lower:
            return "Format: Please organize your explanation with clear headings and provide concrete examples for each main point."
        if "compare" in prompt_lower:
            return "Format: Please structure your comparison in a clear table or side-by-side format with specific criteria."
        if any(word in prompt_lower for word in ["better", "improve", "fix", "help"]):
            return "Format: Please specify what you want improved and provide clear criteria for success."
        return "Format: Please structure your response with clear sections and specific details for each point."

    def _suggest_constraints(self, prompt: str, context: dict | None) -> str | None:
        """Suggest appropriate constraints based on prompt type"""
        prompt_lower = prompt.lower()

        constraints = []

        if "write" in prompt_lower or "create" in prompt_lower:
            constraints.append("Length: Aim for 200-300 words")

        if "list" in prompt_lower:
            constraints.append("Limit: Include 5-7 main items")

        if "explain" in prompt_lower:
            constraints.append("Depth: Focus on practical, actionable information")

        # Add constraints for vague prompts
        if any(word in prompt_lower for word in ["better", "improve", "fix", "help"]):
            constraints.append("Scope: Specify exactly what needs improvement")
            constraints.append(
                "Context: Provide background information and current situation"
            )
            constraints.append(
                "Success criteria: Define what 'better' means in measurable terms"
            )

        if constraints:
            return "Constraints:\n- " + "\n- ".join(constraints)

        return None

    def _generate_examples(self, prompt: str, context: dict | None) -> str | None:
        """Generate helpful examples based on prompt content"""
        prompt_lower = prompt.lower()

        if "explain" in prompt_lower and "concept" in prompt_lower:
            return "Example: If explaining 'machine learning', include both a simple definition and a real-world application like email spam filtering."
        if "list" in prompt_lower and "benefit" in prompt_lower:
            return "Example: Instead of just 'saves time', specify 'reduces processing time from 2 hours to 15 minutes'."
        if "compare" in prompt_lower:
            return "Example: Compare specific features, costs, and outcomes rather than general statements."
        if "analyze" in prompt_lower:
            return "Example: Include specific data points, trends, and measurable impacts in your analysis."

        return None
