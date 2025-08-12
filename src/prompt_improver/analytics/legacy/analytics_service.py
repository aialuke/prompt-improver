"""Analytics Service - Analytics and metrics operations

Handles all analytics and metrics operations including:
- Prompt analysis and classification
- Metrics calculation (clarity, specificity, completeness, structure)
- Improvement scoring and confidence calculation
- Summary generation and reporting
- Bandit context preparation

Follows single responsibility principle for analytics concerns.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service focused on analytics and metrics operations"""

    async def analyze_prompt(self, prompt: str) -> dict[str, Any]:
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
            "has_instructions": any(
                word in prompt.lower()
                for word in ["please", "should", "must", "need to", "required"]
            ),
            "has_context": len(prompt.split()) > 20,
            "domain": self._detect_domain(prompt),
            "readability_score": self._calculate_readability(prompt),
            "sentiment": self._detect_sentiment(prompt),
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
        if not words:
            return 0.0

        length_score = min(len(words) / 50, 1.0)
        avg_word_length = sum(len(word) for word in words) / len(words)
        vocab_score = min(avg_word_length / 8, 1.0)

        # Factor in sentence complexity
        sentences = prompt.split(".")
        sentence_complexity = (
            min(len(sentences) / 10, 1.0) if len(sentences) > 1 else 0.5
        )

        return (length_score + vocab_score + sentence_complexity) / 3

    def _assess_clarity(self, prompt: str) -> float:
        """Assess prompt clarity (0-1, higher is clearer)"""
        clarity_score = 1.0
        word_count = len(prompt.split())

        if word_count < 5:
            clarity_score -= 0.3
        elif word_count > 100:
            clarity_score -= 0.2

        if any(word in prompt.lower() for word in ["specific", "detailed", "exactly"]):
            clarity_score += 0.1

        if any(
            word in prompt.lower() for word in ["something", "anything", "whatever"]
        ):
            clarity_score -= 0.2

        # Check for clear structure
        if ":" in prompt or prompt.count("\n") > 0:
            clarity_score += 0.1

        return max(0.0, min(1.0, clarity_score))

    def _assess_specificity(self, prompt: str) -> float:
        """Assess prompt specificity (0-1, higher is more specific)"""
        specificity_score = 0.5

        specific_indicators = ["when", "where", "how", "why", "which", "what kind"]
        for indicator in specific_indicators:
            if indicator in prompt.lower():
                specificity_score += 0.1

        if any(
            word in prompt.lower() for word in ["example", "constraint", "requirement"]
        ):
            specificity_score += 0.2

        vague_words = ["maybe", "possibly", "might", "could be", "general"]
        for word in vague_words:
            if word in prompt.lower():
                specificity_score -= 0.1

        # Check for specific measurements, numbers, or concrete terms
        import re

        if re.search(r"\d+", prompt):
            specificity_score += 0.1

        return max(0.0, min(1.0, specificity_score))

    def _detect_domain(self, prompt: str) -> str:
        """Detect the domain/category of the prompt"""
        prompt_lower = prompt.lower()

        # Technical domain indicators
        if any(
            word in prompt_lower
            for word in [
                "code",
                "programming",
                "software",
                "algorithm",
                "database",
                "api",
                "function",
                "variable",
                "class",
                "method",
                "debug",
                "technical",
            ]
        ):
            return "technical"

        # Creative domain indicators
        if any(
            word in prompt_lower
            for word in [
                "story",
                "creative",
                "imagine",
                "narrative",
                "character",
                "plot",
                "artistic",
                "design",
                "write",
                "compose",
                "poetry",
            ]
        ):
            return "creative"

        # Analytical domain indicators
        if any(
            word in prompt_lower
            for word in [
                "analyze",
                "data",
                "statistics",
                "research",
                "study",
                "evaluate",
                "assessment",
                "comparison",
                "metrics",
                "performance",
            ]
        ):
            return "analytical"

        # Conversational domain indicators
        if any(
            word in prompt_lower
            for word in [
                "chat",
                "talk",
                "conversation",
                "discuss",
                "opinion",
                "feel",
                "think",
                "casual",
                "friendly",
            ]
        ):
            return "conversational"

        return "general"

    def _calculate_readability(self, prompt: str) -> float:
        """Calculate readability score (0-100, higher is more readable)"""
        words = prompt.split()
        if not words:
            return 0.0

        # Simple readability approximation
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = prompt.count(".") + prompt.count("!") + prompt.count("?")
        if sentence_count == 0:
            sentence_count = 1

        avg_sentence_length = len(words) / sentence_count

        # Inverse relationship: shorter words and sentences = higher readability
        readability = max(0, 100 - (avg_word_length * 10) - (avg_sentence_length * 2))
        return min(100, readability)

    def _detect_sentiment(self, prompt: str) -> str:
        """Detect sentiment of the prompt"""
        prompt_lower = prompt.lower()

        positive_words = [
            "good",
            "great",
            "excellent",
            "please",
            "thanks",
            "help",
            "wonderful",
            "amazing",
            "perfect",
            "love",
            "like",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "problem",
            "issue",
            "wrong",
            "error",
            "fail",
            "difficult",
            "hard",
        ]

        positive_count = sum(1 for word in positive_words if word in prompt_lower)
        negative_count = sum(1 for word in negative_words if word in prompt_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def calculate_metrics(self, prompt: str) -> dict[str, float]:
        """Calculate metrics for a prompt"""
        return {
            "clarity": self._assess_clarity(prompt),
            "specificity": self._assess_specificity(prompt),
            "completeness": self._assess_completeness(prompt),
            "structure": self._assess_structure(prompt),
        }

    def _assess_completeness(self, prompt: str) -> float:
        """Assess if prompt provides complete information"""
        word_count = len(prompt.split())
        base_score = min(word_count / 30, 1.0)

        # Boost for complete sentences
        if prompt.endswith(".") or prompt.endswith("!") or prompt.endswith("?"):
            base_score += 0.1

        # Boost for context provision
        if any(word in prompt.lower() for word in ["context", "background", "because"]):
            base_score += 0.1

        return min(1.0, base_score)

    def _assess_structure(self, prompt: str) -> float:
        """Assess prompt structure quality"""
        structure_score = 0.5

        if "." in prompt:
            structure_score += 0.2
        if "," in prompt:
            structure_score += 0.1
        if prompt and prompt[0].isupper():
            structure_score += 0.1
        if prompt.isupper():
            structure_score -= 0.3

        # Check for organized structure (lists, sections, etc.)
        if any(marker in prompt for marker in ["\n", ":", ";", "-"]):
            structure_score += 0.1

        return max(0.0, min(1.0, structure_score))

    def calculate_improvement_score(
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

    def generate_improvement_summary(
        self, applied_rules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate a summary of improvements made"""
        if not applied_rules:
            return {
                "total_rules_applied": 0,
                "average_confidence": 0,
                "improvement_areas": [],
                "estimated_improvement": 0,
                "quality_indicators": {
                    "high_confidence_rules": 0,
                    "medium_confidence_rules": 0,
                    "low_confidence_rules": 0,
                },
            }

        total_confidence = sum(rule.get("confidence", 0) for rule in applied_rules)
        avg_confidence = total_confidence / len(applied_rules)
        improvement_areas = [rule["rule_name"] for rule in applied_rules]
        estimated_improvement = sum(
            rule.get("improvement_score", 0) for rule in applied_rules
        )

        # Categorize rules by confidence
        quality_indicators = {
            "high_confidence_rules": sum(
                1 for rule in applied_rules if rule.get("confidence", 0) >= 0.8
            ),
            "medium_confidence_rules": sum(
                1 for rule in applied_rules if 0.5 <= rule.get("confidence", 0) < 0.8
            ),
            "low_confidence_rules": sum(
                1 for rule in applied_rules if rule.get("confidence", 0) < 0.5
            ),
        }

        return {
            "total_rules_applied": len(applied_rules),
            "average_confidence": avg_confidence,
            "improvement_areas": improvement_areas,
            "estimated_improvement": estimated_improvement,
            "quality_indicators": quality_indicators,
            "recommendation": self._generate_recommendation(
                applied_rules, avg_confidence
            ),
        }

    def _generate_recommendation(
        self, applied_rules: list[dict[str, Any]], avg_confidence: float
    ) -> str:
        """Generate a recommendation based on applied rules and confidence"""
        if avg_confidence >= 0.8:
            return (
                "High confidence improvements applied. Results should be significant."
            )
        elif avg_confidence >= 0.6:
            return "Moderate improvements applied. Consider additional refinement."
        elif avg_confidence >= 0.4:
            return "Basic improvements applied. Manual review recommended."
        else:
            return "Low confidence improvements. Manual revision strongly recommended."

    def calculate_overall_confidence(
        self, applied_rules: list[dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score"""
        if not applied_rules:
            return 0.0

        confidences = [rule.get("confidence", 0) for rule in applied_rules]

        # Weight recent rules more heavily if there are many
        if len(confidences) > 3:
            weights = [1.0 + (i * 0.1) for i in range(len(confidences))]
            weighted_sum = sum(c * w for c, w in zip(confidences, weights))
            weight_sum = sum(weights)
            return weighted_sum / weight_sum
        else:
            return sum(confidences) / len(confidences)

    def prepare_bandit_context(
        self, prompt_characteristics: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare context for contextual bandit from prompt characteristics"""
        context = {}

        if "length" in prompt_characteristics:
            context["prompt_length"] = min(
                1.0, prompt_characteristics["length"] / 1000.0
            )

        if "complexity" in prompt_characteristics:
            if isinstance(prompt_characteristics["complexity"], (int, float)):
                context["complexity"] = min(1.0, prompt_characteristics["complexity"])
            else:
                complexity_map = {"low": 0.1, "medium": 0.5, "high": 0.9}
                context["complexity"] = complexity_map.get(
                    prompt_characteristics["complexity"], 0.5
                )

        if "readability_score" in prompt_characteristics:
            context["readability"] = min(
                1.0, max(0.0, prompt_characteristics["readability_score"] / 100.0)
            )

        if "sentiment" in prompt_characteristics:
            sentiment_map = {"negative": 0.0, "neutral": 0.5, "positive": 1.0}
            context["sentiment"] = sentiment_map.get(
                prompt_characteristics["sentiment"], 0.5
            )

        context["has_examples"] = float(
            prompt_characteristics.get("has_examples", False)
        )
        context["has_instructions"] = float(
            prompt_characteristics.get("has_instructions", False)
        )
        context["has_context"] = float(prompt_characteristics.get("has_context", False))

        domain = prompt_characteristics.get("domain", "general")
        domain_features = {
            "technical": 0.9,
            "creative": 0.7,
            "analytical": 0.8,
            "conversational": 0.3,
            "general": 0.5,
        }
        context["domain_score"] = domain_features.get(domain, 0.5)

        # Add prompt type context
        prompt_type = prompt_characteristics.get("type", "general")
        type_features = {
            "explanation": 0.7,
            "creation": 0.8,
            "analysis": 0.9,
            "instruction": 0.6,
            "general": 0.5,
        }
        context["type_score"] = type_features.get(prompt_type, 0.5)

        return context

    def calculate_rule_effectiveness_metrics(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate aggregate metrics for rule effectiveness analysis"""
        if not performance_data:
            return {
                "average_improvement_score": 0.0,
                "average_confidence": 0.0,
                "average_execution_time": 0.0,
                "success_rate": 0.0,
                "total_applications": 0,
            }

        improvement_scores = [
            data.get("improvement_score", 0) for data in performance_data
        ]
        confidences = [data.get("confidence", 0) for data in performance_data]
        execution_times = [
            data.get("execution_time_ms", 0) for data in performance_data
        ]
        successful_applications = sum(
            1 for data in performance_data if data.get("improvement_score", 0) > 0
        )

        return {
            "average_improvement_score": sum(improvement_scores)
            / len(improvement_scores),
            "average_confidence": sum(confidences) / len(confidences),
            "average_execution_time": sum(execution_times) / len(execution_times),
            "success_rate": successful_applications / len(performance_data),
            "total_applications": len(performance_data),
            "rule_breakdown": self._calculate_rule_breakdown(performance_data),
        }

    def _calculate_rule_breakdown(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Calculate metrics breakdown by individual rules"""
        rule_stats = {}

        for data in performance_data:
            rule_id = data.get("rule_id", "unknown")
            if rule_id not in rule_stats:
                rule_stats[rule_id] = {
                    "applications": 0,
                    "total_improvement": 0.0,
                    "total_confidence": 0.0,
                    "total_execution_time": 0.0,
                    "successful_applications": 0,
                }

            stats = rule_stats[rule_id]
            stats["applications"] += 1
            stats["total_improvement"] += data.get("improvement_score", 0)
            stats["total_confidence"] += data.get("confidence", 0)
            stats["total_execution_time"] += data.get("execution_time_ms", 0)

            if data.get("improvement_score", 0) > 0:
                stats["successful_applications"] += 1

        # Calculate averages
        for rule_id, stats in rule_stats.items():
            if stats["applications"] > 0:
                stats["average_improvement"] = (
                    stats["total_improvement"] / stats["applications"]
                )
                stats["average_confidence"] = (
                    stats["total_confidence"] / stats["applications"]
                )
                stats["average_execution_time"] = (
                    stats["total_execution_time"] / stats["applications"]
                )
                stats["success_rate"] = (
                    stats["successful_applications"] / stats["applications"]
                )

        return rule_stats
