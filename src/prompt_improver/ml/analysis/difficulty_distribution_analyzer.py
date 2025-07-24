"""
Difficulty Distribution Analyzer - 2025 Best Practices Implementation
Advanced difficulty distribution algorithms and focus area targeting for adaptive data generation.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .performance_gap_analyzer import PerformanceGap


class DifficultyLevel(Enum):
    """Difficulty levels for synthetic data generation."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class DifficultyProfile:
    """Profile for difficulty distribution."""
    distribution_weights: Dict[str, float]
    hardness_threshold: float
    focus_areas: List[str]
    complexity_factors: Dict[str, float]
    adaptive_parameters: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class FocusAreaTarget:
    """Target specification for focus areas."""
    area_name: str
    priority: float
    target_improvement: float
    current_performance: float
    difficulty_emphasis: str  # "easy", "medium", "hard", "adaptive"
    sample_allocation: float
    metadata: Dict[str, Any]


class DifficultyDistributionAnalyzer:
    """
    2025 best practices difficulty distribution analyzer.

    Implements advanced algorithms for:
    - Adaptive difficulty distribution based on performance gaps
    - Focus area targeting with intelligent sample allocation
    - Hardness characterization and complexity modeling
    - Dynamic difficulty adjustment based on learning progress
    """

    def __init__(self):
        self.logger = logging.getLogger("apes.difficulty_distribution_analyzer")

        # 2025 best practice configuration
        self.config = {
            "default_distribution": {"easy": 0.33, "medium": 0.34, "hard": 0.33},
            "adaptive_learning_rate": 0.1,
            "hardness_percentiles": {"easy": 30, "medium": 70, "hard": 90},
            "focus_area_weights": {
                "clarity": 1.0,
                "specificity": 1.0,
                "effectiveness": 1.2,
                "consistency": 1.1,
                "coverage": 0.9
            },
            "complexity_factors": {
                "semantic_complexity": 0.3,
                "syntactic_complexity": 0.2,
                "domain_complexity": 0.25,
                "rule_complexity": 0.25
            },
            "min_samples_per_level": 20,
            "max_extreme_ratio": 0.1  # Maximum 10% extreme difficulty
        }

    async def analyze_optimal_difficulty_distribution(
        self,
        performance_gaps: List[PerformanceGap],
        hardness_analysis: Dict[str, Any],
        focus_areas: Optional[List[str]] = None,
        current_performance: Optional[Dict[str, float]] = None
    ) -> DifficultyProfile:
        """
        Analyze optimal difficulty distribution for targeted generation (2025 best practice).

        Args:
            performance_gaps: List of identified performance gaps
            hardness_analysis: Results from hardness analysis
            focus_areas: Specific areas to focus on
            current_performance: Current performance metrics

        Returns:
            Optimal difficulty distribution profile
        """
        self.logger.info("Analyzing optimal difficulty distribution")

        try:
            # 1. Analyze gap-based difficulty requirements
            gap_difficulty_requirements = self._analyze_gap_difficulty_requirements(performance_gaps)

            # 2. Incorporate hardness analysis
            hardness_insights = self._extract_hardness_insights(hardness_analysis)

            # 3. Calculate focus area priorities
            focus_priorities = await self._calculate_focus_area_priorities(
                performance_gaps, focus_areas, current_performance
            )

            # 4. Determine adaptive distribution
            adaptive_distribution = self._calculate_adaptive_distribution(
                gap_difficulty_requirements, hardness_insights, focus_priorities
            )

            # 5. Apply complexity factors
            complexity_adjusted_distribution = self._apply_complexity_factors(
                adaptive_distribution, performance_gaps
            )

            # 6. Validate and normalize distribution
            final_distribution = self._validate_and_normalize_distribution(
                complexity_adjusted_distribution
            )

            # 7. Calculate hardness threshold
            optimal_threshold = self._calculate_optimal_hardness_threshold(
                hardness_analysis, performance_gaps
            )

            # 8. Generate adaptive parameters
            adaptive_params = self._generate_adaptive_parameters(
                performance_gaps, hardness_insights, focus_priorities
            )

            profile = DifficultyProfile(
                distribution_weights=final_distribution,
                hardness_threshold=optimal_threshold,
                focus_areas=focus_areas or [],
                complexity_factors=self.config["complexity_factors"].copy(),
                adaptive_parameters=adaptive_params,
                metadata={
                    "analysis_timestamp": datetime.now(timezone.utc),
                    "gap_count": len(performance_gaps),
                    "hardness_distribution": hardness_insights,
                    "focus_priorities": focus_priorities,
                    "distribution_reasoning": self._generate_distribution_reasoning(
                        final_distribution, gap_difficulty_requirements, hardness_insights
                    )
                }
            )

            self.logger.info(f"Difficulty distribution analysis completed: {final_distribution}")
            return profile

        except Exception as e:
            self.logger.error(f"Error in difficulty distribution analysis: {e}")
            raise

    async def generate_focus_area_targets(
        self,
        performance_gaps: List[PerformanceGap],
        focus_areas: List[str],
        current_performance: Dict[str, float],
        target_samples: int
    ) -> List[FocusAreaTarget]:
        """
        Generate detailed focus area targets for adaptive generation.

        Args:
            performance_gaps: List of performance gaps
            focus_areas: Areas to focus on
            current_performance: Current performance metrics
            target_samples: Total samples to generate

        Returns:
            List of focus area targets with sample allocation
        """
        self.logger.info(f"Generating focus area targets for {len(focus_areas)} areas")

        try:
            targets = []

            # Calculate total priority weight
            total_priority = 0.0
            area_priorities = {}

            for area in focus_areas:
                priority = self._calculate_area_priority(area, performance_gaps, current_performance)
                area_priorities[area] = priority
                total_priority += priority

            # Generate targets for each focus area
            for area in focus_areas:
                priority = area_priorities[area]

                # Calculate sample allocation
                sample_allocation = (priority / total_priority) if total_priority > 0 else 1.0 / len(focus_areas)

                # Determine difficulty emphasis
                difficulty_emphasis = self._determine_area_difficulty_emphasis(area, performance_gaps)

                # Calculate target improvement
                current_perf = current_performance.get(f"{area}_effectiveness", 0.5)
                target_improvement = self._calculate_target_improvement(area, current_perf, performance_gaps)

                target = FocusAreaTarget(
                    area_name=area,
                    priority=priority,
                    target_improvement=target_improvement,
                    current_performance=current_perf,
                    difficulty_emphasis=difficulty_emphasis,
                    sample_allocation=sample_allocation,
                    metadata={
                        "estimated_samples": int(target_samples * sample_allocation),
                        "gap_relevance": self._calculate_gap_relevance(area, performance_gaps),
                        "complexity_score": self._calculate_area_complexity(area, performance_gaps)
                    }
                )

                targets.append(target)

            # Sort by priority
            targets.sort(key=lambda x: x.priority, reverse=True)

            self.logger.info(f"Generated {len(targets)} focus area targets")
            return targets

        except Exception as e:
            self.logger.error(f"Error generating focus area targets: {e}")
            raise

    def _analyze_gap_difficulty_requirements(self, performance_gaps: List[PerformanceGap]) -> Dict[str, float]:
        """Analyze difficulty requirements based on performance gaps."""
        if not performance_gaps:
            return self.config["default_distribution"].copy()

        # Calculate severity-based requirements
        severities = [gap.severity for gap in performance_gaps]
        critical_ratio = sum(1 for s in severities if s >= 0.7) / len(severities)
        moderate_ratio = sum(1 for s in severities if 0.3 <= s < 0.7) / len(severities)
        minor_ratio = 1.0 - critical_ratio - moderate_ratio

        # Map severity to difficulty requirements
        if critical_ratio > 0.5:
            # Many critical gaps - need more hard examples
            return {"easy": 0.15, "medium": 0.25, "hard": 0.60}
        elif critical_ratio > 0.3:
            # Some critical gaps - balanced with hard emphasis
            return {"easy": 0.20, "medium": 0.35, "hard": 0.45}
        elif moderate_ratio > 0.6:
            # Many moderate gaps - balanced approach
            return {"easy": 0.25, "medium": 0.50, "hard": 0.25}
        else:
            # Few gaps - standard distribution
            return {"easy": 0.33, "medium": 0.34, "hard": 0.33}

    def _extract_hardness_insights(self, hardness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actionable insights from hardness analysis."""
        distribution = hardness_analysis.get("distribution", {})

        insights = {
            "hard_examples_ratio": distribution.get("hard_examples_ratio", 0.3),
            "hardness_variance": distribution.get("std", 0.2),
            "median_hardness": distribution.get("median", 0.5),
            "hardness_skew": self._calculate_hardness_skew(distribution),
            "requires_extreme_examples": distribution.get("hard_examples_ratio", 0.3) > 0.6
        }

        return insights

    def _calculate_hardness_skew(self, distribution: Dict[str, float]) -> str:
        """Calculate hardness distribution skew."""
        mean = distribution.get("mean", 0.5)
        median = distribution.get("median", 0.5)

        if mean > median + 0.1:
            return "right_skewed"  # More hard examples
        elif mean < median - 0.1:
            return "left_skewed"   # More easy examples
        else:
            return "symmetric"

    async def _calculate_focus_area_priorities(
        self,
        performance_gaps: List[PerformanceGap],
        focus_areas: Optional[List[str]],
        current_performance: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate priority scores for focus areas."""
        if not focus_areas:
            return {}

        priorities = {}

        for area in focus_areas:
            # Base priority from configuration
            base_priority = self.config["focus_area_weights"].get(area, 1.0)

            # Adjust based on performance gaps
            gap_adjustment = self._calculate_gap_adjustment(area, performance_gaps)

            # Adjust based on current performance
            performance_adjustment = 1.0
            if current_performance:
                current_perf = current_performance.get(f"{area}_effectiveness", 0.5)
                # Lower performance = higher priority
                performance_adjustment = 1.0 + (1.0 - current_perf) * 0.5

            final_priority = base_priority * gap_adjustment * performance_adjustment
            priorities[area] = final_priority

        return priorities

    def _calculate_gap_adjustment(self, area: str, performance_gaps: List[PerformanceGap]) -> float:
        """Calculate priority adjustment based on relevant gaps."""
        relevant_gaps = []

        for gap in performance_gaps:
            # Check if gap is relevant to this area
            if (area.lower() in gap.gap_type.lower() or
                area.lower() in str(gap.metadata).lower()):
                relevant_gaps.append(gap)

        if not relevant_gaps:
            return 1.0

        # Calculate adjustment based on gap severity and improvement potential
        total_impact = sum(gap.severity * gap.improvement_potential for gap in relevant_gaps)
        average_impact = total_impact / len(relevant_gaps)

        # Scale to reasonable adjustment range (0.5 to 2.0)
        adjustment = 0.5 + (average_impact * 1.5)
        return min(2.0, max(0.5, adjustment))

    def _calculate_adaptive_distribution(
        self,
        gap_requirements: Dict[str, float],
        hardness_insights: Dict[str, Any],
        focus_priorities: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate adaptive difficulty distribution."""

        # Start with gap-based requirements
        distribution = gap_requirements.copy()

        # Adjust based on hardness insights
        hard_ratio = hardness_insights["hard_examples_ratio"]
        if hard_ratio > 0.5:
            # Many hard examples in data - increase hard generation
            distribution["hard"] = min(0.7, distribution["hard"] * 1.3)
            distribution["easy"] = max(0.1, distribution["easy"] * 0.7)
        elif hard_ratio < 0.2:
            # Few hard examples - increase easy/medium
            distribution["easy"] = min(0.5, distribution["easy"] * 1.2)
            distribution["medium"] = min(0.5, distribution["medium"] * 1.1)

        # Adjust based on focus area priorities
        if focus_priorities:
            max_priority = max(focus_priorities.values())
            if max_priority > 1.5:
                # High priority areas - increase hard examples
                distribution["hard"] = min(0.6, distribution["hard"] * 1.2)

        # Normalize to ensure sum = 1.0
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        return distribution

    def _apply_complexity_factors(
        self,
        distribution: Dict[str, float],
        performance_gaps: List[PerformanceGap]
    ) -> Dict[str, float]:
        """Apply complexity factors to difficulty distribution."""

        # Calculate overall complexity score
        complexity_score = self._calculate_overall_complexity(performance_gaps)

        adjusted_distribution = distribution.copy()

        if complexity_score > 0.7:
            # High complexity - shift toward harder examples
            shift_amount = (complexity_score - 0.7) * 0.3
            adjusted_distribution["hard"] += shift_amount
            adjusted_distribution["easy"] = max(0.1, adjusted_distribution["easy"] - shift_amount * 0.5)
            adjusted_distribution["medium"] = max(0.1, adjusted_distribution["medium"] - shift_amount * 0.5)
        elif complexity_score < 0.3:
            # Low complexity - shift toward easier examples
            shift_amount = (0.3 - complexity_score) * 0.2
            adjusted_distribution["easy"] += shift_amount
            adjusted_distribution["hard"] = max(0.1, adjusted_distribution["hard"] - shift_amount)

        return adjusted_distribution

    def _calculate_overall_complexity(self, performance_gaps: List[PerformanceGap]) -> float:
        """Calculate overall complexity score from performance gaps."""
        if not performance_gaps:
            return 0.5

        # Complexity based on gap characteristics
        severity_complexity = np.mean([gap.severity for gap in performance_gaps])
        confidence_complexity = 1.0 - np.mean([gap.confidence for gap in performance_gaps])
        type_complexity = len(set(gap.gap_type for gap in performance_gaps)) / 3.0  # Normalize

        overall_complexity = (
            severity_complexity * 0.4 +
            confidence_complexity * 0.3 +
            type_complexity * 0.3
        )

        return min(1.0, max(0.0, overall_complexity))

    def _validate_and_normalize_distribution(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize difficulty distribution."""

        # Ensure minimum samples per level
        min_ratio = self.config["min_samples_per_level"] / 1000.0  # Assume 1000 total samples

        validated_distribution = {}
        for level in ["easy", "medium", "hard"]:
            validated_distribution[level] = max(min_ratio, distribution.get(level, 0.33))

        # Add extreme level if warranted
        if any(gap.severity > 0.9 for gap in [] if hasattr(self, '_current_gaps')):
            extreme_ratio = min(self.config["max_extreme_ratio"], 0.05)
            validated_distribution["extreme"] = extreme_ratio

        # Normalize to sum to 1.0
        total = sum(validated_distribution.values())
        if total > 0:
            validated_distribution = {k: v / total for k, v in validated_distribution.items()}

        return validated_distribution

    def _calculate_optimal_hardness_threshold(
        self,
        hardness_analysis: Dict[str, Any],
        performance_gaps: List[PerformanceGap]
    ) -> float:
        """Calculate optimal hardness threshold for example classification."""

        # Start with hardness analysis recommendation
        base_threshold = hardness_analysis.get("optimal_threshold", 0.7)

        # Adjust based on gap severity
        if performance_gaps:
            avg_severity = np.mean([gap.severity for gap in performance_gaps])
            if avg_severity > 0.7:
                # High severity gaps - lower threshold to catch more hard examples
                base_threshold = max(0.5, base_threshold - 0.1)
            elif avg_severity < 0.3:
                # Low severity gaps - raise threshold
                base_threshold = min(0.8, base_threshold + 0.1)

        return base_threshold

    def _generate_adaptive_parameters(
        self,
        performance_gaps: List[PerformanceGap],
        hardness_insights: Dict[str, Any],
        focus_priorities: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate adaptive parameters for dynamic difficulty adjustment."""

        return {
            "learning_rate": self.config["adaptive_learning_rate"],
            "adjustment_frequency": 100,  # Adjust every 100 samples
            "performance_window": 50,     # Look at last 50 samples for adjustment
            "min_improvement_threshold": 0.02,
            "max_difficulty_shift": 0.1,
            "focus_area_boost": 1.2 if focus_priorities else 1.0,
            "hardness_sensitivity": hardness_insights.get("hardness_variance", 0.2),
            "plateau_detection": {
                "window_size": 20,
                "threshold": 0.01
            }
        }

    def _generate_distribution_reasoning(
        self,
        distribution: Dict[str, float],
        gap_requirements: Dict[str, float],
        hardness_insights: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for distribution choice."""

        hard_ratio = distribution.get("hard", 0.33)
        easy_ratio = distribution.get("easy", 0.33)

        if hard_ratio > 0.5:
            return f"Hard-focused distribution ({hard_ratio:.1%} hard) due to critical performance gaps and {hardness_insights['hard_examples_ratio']:.1%} hard examples in data"
        elif easy_ratio > 0.4:
            return f"Easy-focused distribution ({easy_ratio:.1%} easy) suitable for current gap profile with low complexity requirements"
        else:
            return f"Balanced distribution ({hard_ratio:.1%} hard, {easy_ratio:.1%} easy) appropriate for moderate gap severity and complexity"

    # Focus area targeting helper methods

    def _calculate_area_priority(
        self,
        area: str,
        performance_gaps: List[PerformanceGap],
        current_performance: Dict[str, float]
    ) -> float:
        """Calculate priority score for a focus area."""

        # Base priority from configuration
        base_priority = self.config["focus_area_weights"].get(area, 1.0)

        # Gap-based adjustment
        relevant_gaps = [gap for gap in performance_gaps
                        if area.lower() in gap.gap_type.lower() or
                           area.lower() in str(gap.metadata).lower()]

        if relevant_gaps:
            gap_impact = np.mean([gap.severity * gap.improvement_potential for gap in relevant_gaps])
            gap_adjustment = 1.0 + gap_impact
        else:
            gap_adjustment = 1.0

        # Performance-based adjustment
        current_perf = current_performance.get(f"{area}_effectiveness", 0.5)
        performance_adjustment = 2.0 - current_perf  # Lower performance = higher priority

        return base_priority * gap_adjustment * performance_adjustment

    def _determine_area_difficulty_emphasis(self, area: str, performance_gaps: List[PerformanceGap]) -> str:
        """Determine difficulty emphasis for a focus area."""

        relevant_gaps = [gap for gap in performance_gaps
                        if area.lower() in gap.gap_type.lower()]

        if not relevant_gaps:
            return "adaptive"

        avg_severity = np.mean([gap.severity for gap in relevant_gaps])

        if avg_severity >= 0.7:
            return "hard"
        elif avg_severity >= 0.4:
            return "medium"
        else:
            return "easy"

    def _calculate_target_improvement(
        self,
        area: str,
        current_performance: float,
        performance_gaps: List[PerformanceGap]
    ) -> float:
        """Calculate target improvement for a focus area."""

        # Find relevant gaps
        relevant_gaps = [gap for gap in performance_gaps
                        if area.lower() in gap.gap_type.lower()]

        if relevant_gaps:
            # Target improvement based on gap improvement potential
            max_improvement = max(gap.improvement_potential for gap in relevant_gaps)
            return min(0.3, max_improvement)  # Cap at 30% improvement
        else:
            # Default improvement target
            return min(0.2, 1.0 - current_performance)

    def _calculate_gap_relevance(self, area: str, performance_gaps: List[PerformanceGap]) -> float:
        """Calculate how relevant performance gaps are to this area."""

        relevant_gaps = [gap for gap in performance_gaps
                        if area.lower() in gap.gap_type.lower() or
                           area.lower() in str(gap.metadata).lower()]

        if not relevant_gaps:
            return 0.0

        # Relevance based on gap count and severity
        relevance_score = len(relevant_gaps) / len(performance_gaps)
        severity_weight = np.mean([gap.severity for gap in relevant_gaps])

        return min(1.0, relevance_score * severity_weight * 2.0)

    def _calculate_area_complexity(self, area: str, performance_gaps: List[PerformanceGap]) -> float:
        """Calculate complexity score for a focus area."""

        relevant_gaps = [gap for gap in performance_gaps
                        if area.lower() in gap.gap_type.lower()]

        if not relevant_gaps:
            return 0.5  # Default complexity

        # Complexity based on gap characteristics
        severity_complexity = np.mean([gap.severity for gap in relevant_gaps])
        confidence_complexity = 1.0 - np.mean([gap.confidence for gap in relevant_gaps])

        return (severity_complexity + confidence_complexity) / 2.0
