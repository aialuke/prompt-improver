"""
Generation Strategy Analyzer - 2025 Best Practices Implementation
Intelligent strategy determination for adaptive synthetic data generation.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .performance_gap_analyzer import PerformanceGap, GapAnalysisResult


class GenerationStrategy(Enum):
    """Available generation strategies with 2025 best practices."""
    STATISTICAL = "statistical"
    NEURAL_ENHANCED = "neural_enhanced"
    RULE_FOCUSED = "rule_focused"
    DIVERSITY_ENHANCED = "diversity_enhanced"
    HYBRID = "hybrid"
    DIFFUSION = "diffusion"


@dataclass
class StrategyRecommendation:
    """Strategy recommendation with detailed analysis."""
    primary_strategy: GenerationStrategy
    secondary_strategy: GenerationStrategy
    confidence: float
    reasoning: str
    expected_improvement: float
    estimated_samples: int
    difficulty_distribution: Dict[str, float]
    focus_areas: List[str]
    metadata: Dict[str, Any]


class GenerationStrategyAnalyzer:
    """
    2025 best practices generation strategy analyzer.

    Analyzes performance gaps and determines optimal generation strategies
    based on gap characteristics, hardness analysis, and improvement potential.
    """

    def __init__(self):
        self.logger = logging.getLogger("apes.generation_strategy_analyzer")

        # 2025 strategy selection thresholds
        self.config = {
            "neural_threshold": 0.7,        # Severity threshold for neural methods
            "complexity_threshold": 0.6,    # Complexity threshold for advanced methods
            "diversity_threshold": 0.5,     # Coverage gap threshold for diversity
            "hybrid_threshold": 0.8,        # Multiple high-severity gaps for hybrid
            "confidence_threshold": 0.8,    # Minimum confidence for recommendations
            "sample_efficiency": {
                "statistical": 1.0,
                "neural_enhanced": 1.5,
                "rule_focused": 1.2,
                "diversity_enhanced": 0.9,
                "hybrid": 2.0,
                "diffusion": 2.5
            }
        }

    async def analyze_optimal_strategy(
        self,
        gap_analysis: GapAnalysisResult,
        hardness_analysis: Dict[str, Any],
        focus_areas: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> StrategyRecommendation:
        """
        Analyze and recommend optimal generation strategy (2025 best practice).

        Args:
            gap_analysis: Results from performance gap analysis
            hardness_analysis: Results from example hardness analysis
            focus_areas: Specific areas to focus on
            constraints: Resource or time constraints

        Returns:
            Comprehensive strategy recommendation
        """
        self.logger.info("Analyzing optimal generation strategy")

        try:
            # 1. Analyze gap characteristics
            gap_profile = self._analyze_gap_profile(gap_analysis)

            # 2. Assess generation complexity
            complexity_assessment = self._assess_generation_complexity(
                gap_analysis, hardness_analysis
            )

            # 3. Evaluate strategy candidates
            strategy_scores = await self._evaluate_strategy_candidates(
                gap_profile, complexity_assessment, hardness_analysis
            )

            # 4. Apply constraints and preferences
            if constraints:
                strategy_scores = self._apply_constraints(strategy_scores, constraints)

            # 5. Select optimal strategies
            primary_strategy, secondary_strategy = self._select_strategies(strategy_scores)

            # 6. Calculate recommendation confidence
            confidence = self._calculate_recommendation_confidence(
                strategy_scores, gap_profile, complexity_assessment
            )

            # 7. Generate detailed recommendation
            recommendation = self._generate_strategy_recommendation(
                primary_strategy,
                secondary_strategy,
                confidence,
                gap_profile,
                complexity_assessment,
                hardness_analysis,
                focus_areas or []
            )

            self.logger.info(f"Strategy analysis completed: {primary_strategy.value} "
                           f"(confidence: {confidence:.2f})")

            return recommendation

        except Exception as e:
            self.logger.error(f"Error in strategy analysis: {e}")
            raise

    def _analyze_gap_profile(self, gap_analysis: GapAnalysisResult) -> Dict[str, Any]:
        """Analyze the profile of performance gaps."""
        all_gaps = gap_analysis.critical_gaps + gap_analysis.improvement_opportunities

        if not all_gaps:
            return {
                "total_gaps": 0,
                "severity_distribution": {},
                "gap_types": {},
                "complexity_score": 0.0
            }

        # Analyze severity distribution
        severities = [gap.severity for gap in all_gaps]
        severity_distribution = {
            "critical": sum(1 for s in severities if s >= 0.7) / len(severities),
            "moderate": sum(1 for s in severities if 0.3 <= s < 0.7) / len(severities),
            "minor": sum(1 for s in severities if s < 0.3) / len(severities),
            "mean": np.mean(severities),
            "std": np.std(severities)
        }

        # Analyze gap types
        gap_types = {}
        for gap in all_gaps:
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1

        # Calculate overall complexity
        complexity_score = (
            severity_distribution["critical"] * 1.0 +
            severity_distribution["moderate"] * 0.6 +
            severity_distribution["minor"] * 0.2
        )

        return {
            "total_gaps": len(all_gaps),
            "severity_distribution": severity_distribution,
            "gap_types": gap_types,
            "complexity_score": complexity_score,
            "improvement_potential": np.mean([gap.improvement_potential for gap in all_gaps])
        }

    def _assess_generation_complexity(
        self,
        gap_analysis: GapAnalysisResult,
        hardness_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the complexity of data generation requirements."""

        # Hardness-based complexity
        hard_ratio = hardness_analysis.get("distribution", {}).get("hard_examples_ratio", 0.3)
        hardness_variance = hardness_analysis.get("distribution", {}).get("std", 0.2)

        # Gap-based complexity
        all_gaps = gap_analysis.critical_gaps + gap_analysis.improvement_opportunities
        gap_complexity = np.mean([gap.severity * (1 - gap.confidence) for gap in all_gaps]) if all_gaps else 0.0

        # Pattern complexity (based on gap types)
        unique_gap_types = len(set(gap.gap_type for gap in all_gaps))
        pattern_complexity = min(unique_gap_types / 3.0, 1.0)  # Normalize to 0-1

        # Overall complexity score
        overall_complexity = (
            hard_ratio * 0.4 +
            gap_complexity * 0.3 +
            pattern_complexity * 0.2 +
            hardness_variance * 0.1
        )

        return {
            "overall_complexity": overall_complexity,
            "hardness_complexity": hard_ratio,
            "gap_complexity": gap_complexity,
            "pattern_complexity": pattern_complexity,
            "requires_advanced_methods": overall_complexity > self.config["complexity_threshold"]
        }

    async def _evaluate_strategy_candidates(
        self,
        gap_profile: Dict[str, Any],
        complexity_assessment: Dict[str, Any],
        hardness_analysis: Dict[str, Any]
    ) -> Dict[GenerationStrategy, float]:
        """Evaluate all strategy candidates and assign scores."""

        strategy_scores = {}

        # Statistical strategy scoring
        statistical_score = self._score_statistical_strategy(gap_profile, complexity_assessment)
        strategy_scores[GenerationStrategy.STATISTICAL] = statistical_score

        # Neural enhanced strategy scoring
        neural_score = self._score_neural_strategy(gap_profile, complexity_assessment, hardness_analysis)
        strategy_scores[GenerationStrategy.NEURAL_ENHANCED] = neural_score

        # Rule focused strategy scoring
        rule_score = self._score_rule_focused_strategy(gap_profile, complexity_assessment)
        strategy_scores[GenerationStrategy.RULE_FOCUSED] = rule_score

        # Diversity enhanced strategy scoring
        diversity_score = self._score_diversity_strategy(gap_profile, complexity_assessment)
        strategy_scores[GenerationStrategy.DIVERSITY_ENHANCED] = diversity_score

        # Hybrid strategy scoring
        hybrid_score = self._score_hybrid_strategy(gap_profile, complexity_assessment)
        strategy_scores[GenerationStrategy.HYBRID] = hybrid_score

        # Diffusion strategy scoring (experimental)
        diffusion_score = self._score_diffusion_strategy(gap_profile, complexity_assessment)
        strategy_scores[GenerationStrategy.DIFFUSION] = diffusion_score

        return strategy_scores

    def _score_statistical_strategy(self, gap_profile: Dict[str, Any], complexity: Dict[str, Any]) -> float:
        """Score statistical generation strategy."""
        base_score = 0.7  # Good baseline strategy

        # Penalty for high complexity
        if complexity["overall_complexity"] > 0.6:
            base_score *= 0.7

        # Bonus for simple gaps
        if gap_profile["complexity_score"] < 0.3:
            base_score *= 1.2

        # Penalty for critical gaps
        critical_ratio = gap_profile["severity_distribution"].get("critical", 0)
        if critical_ratio > 0.5:
            base_score *= 0.6

        return min(1.0, base_score)

    def _score_neural_strategy(self, gap_profile: Dict[str, Any], complexity: Dict[str, Any], hardness: Dict[str, Any]) -> float:
        """Score neural enhanced generation strategy."""
        base_score = 0.5  # Requires justification

        # Bonus for high complexity
        if complexity["overall_complexity"] > self.config["neural_threshold"]:
            base_score += 0.4

        # Bonus for hard examples
        hard_ratio = hardness.get("distribution", {}).get("hard_examples_ratio", 0.3)
        if hard_ratio > 0.4:
            base_score += 0.3

        # Bonus for critical gaps
        critical_ratio = gap_profile["severity_distribution"].get("critical", 0)
        if critical_ratio > 0.3:
            base_score += 0.2

        # Bonus for effectiveness gaps
        effectiveness_gaps = gap_profile["gap_types"].get("effectiveness", 0)
        if effectiveness_gaps > 0:
            base_score += 0.1

        return min(1.0, base_score)

    def _score_rule_focused_strategy(self, gap_profile: Dict[str, Any], complexity: Dict[str, Any]) -> float:
        """Score rule-focused generation strategy."""
        base_score = 0.4

        # Strong bonus for consistency gaps
        consistency_gaps = gap_profile["gap_types"].get("consistency", 0)
        if consistency_gaps > 0:
            base_score += 0.5

        # Bonus for moderate complexity
        if 0.3 <= complexity["overall_complexity"] <= 0.7:
            base_score += 0.2

        # Bonus for pattern complexity
        if complexity["pattern_complexity"] > 0.5:
            base_score += 0.2

        return min(1.0, base_score)

    def _score_diversity_strategy(self, gap_profile: Dict[str, Any], complexity: Dict[str, Any]) -> float:
        """Score diversity enhanced generation strategy."""
        base_score = 0.5

        # Bonus for coverage gaps
        coverage_gaps = gap_profile["gap_types"].get("coverage", 0)
        if coverage_gaps > 0:
            base_score += 0.4

        # Bonus for low pattern complexity
        if complexity["pattern_complexity"] < 0.4:
            base_score += 0.2

        # Bonus for moderate overall complexity
        if 0.2 <= complexity["overall_complexity"] <= 0.6:
            base_score += 0.2

        return min(1.0, base_score)

    def _score_hybrid_strategy(self, gap_profile: Dict[str, Any], complexity: Dict[str, Any]) -> float:
        """Score hybrid generation strategy."""
        base_score = 0.3  # Complex strategy, needs strong justification

        # Bonus for multiple gap types
        gap_type_count = len(gap_profile["gap_types"])
        if gap_type_count >= 2:
            base_score += 0.3

        # Bonus for very high complexity
        if complexity["overall_complexity"] > self.config["hybrid_threshold"]:
            base_score += 0.4

        # Bonus for high improvement potential
        if gap_profile["improvement_potential"] > 0.6:
            base_score += 0.2

        return min(1.0, base_score)

    def _score_diffusion_strategy(self, gap_profile: Dict[str, Any], complexity: Dict[str, Any]) -> float:
        """Score diffusion generation strategy (experimental)."""
        base_score = 0.2  # Experimental, conservative scoring

        # Bonus for very high complexity
        if complexity["overall_complexity"] > 0.8:
            base_score += 0.3

        # Bonus for very high critical gap ratio
        critical_ratio = gap_profile["severity_distribution"].get("critical", 0)
        if critical_ratio > 0.7:
            base_score += 0.2

        return min(1.0, base_score)

    def _apply_constraints(self, strategy_scores: Dict[GenerationStrategy, float], constraints: Dict[str, Any]) -> Dict[GenerationStrategy, float]:
        """Apply resource and time constraints to strategy scores."""
        constrained_scores = strategy_scores.copy()

        # Time constraints
        if constraints.get("time_limit", "medium") == "low":
            # Penalize complex strategies
            constrained_scores[GenerationStrategy.NEURAL_ENHANCED] *= 0.5
            constrained_scores[GenerationStrategy.HYBRID] *= 0.3
            constrained_scores[GenerationStrategy.DIFFUSION] *= 0.2

        # Resource constraints
        if constraints.get("compute_resources", "medium") == "low":
            # Penalize compute-intensive strategies
            constrained_scores[GenerationStrategy.NEURAL_ENHANCED] *= 0.6
            constrained_scores[GenerationStrategy.DIFFUSION] *= 0.3

        # Sample size constraints
        max_samples = constraints.get("max_samples", 2000)
        if max_samples < 500:
            # Favor efficient strategies
            constrained_scores[GenerationStrategy.STATISTICAL] *= 1.2
            constrained_scores[GenerationStrategy.DIVERSITY_ENHANCED] *= 1.1

        return constrained_scores

    def _select_strategies(self, strategy_scores: Dict[GenerationStrategy, float]) -> Tuple[GenerationStrategy, GenerationStrategy]:
        """Select primary and secondary strategies."""
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)

        primary_strategy = sorted_strategies[0][0]
        secondary_strategy = sorted_strategies[1][0] if len(sorted_strategies) > 1 else primary_strategy

        return primary_strategy, secondary_strategy

    def _calculate_recommendation_confidence(
        self,
        strategy_scores: Dict[GenerationStrategy, float],
        gap_profile: Dict[str, Any],
        complexity_assessment: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the strategy recommendation."""
        sorted_scores = sorted(strategy_scores.values(), reverse=True)

        if len(sorted_scores) < 2:
            return 0.5

        # Confidence based on score separation
        score_separation = sorted_scores[0] - sorted_scores[1]
        separation_confidence = min(score_separation * 2, 1.0)

        # Confidence based on gap analysis quality
        total_gaps = gap_profile["total_gaps"]
        gap_confidence = min(total_gaps / 10.0, 1.0) if total_gaps > 0 else 0.3

        # Confidence based on complexity assessment
        complexity_confidence = 1.0 - abs(complexity_assessment["overall_complexity"] - 0.5)

        # Combined confidence
        overall_confidence = (
            separation_confidence * 0.4 +
            gap_confidence * 0.3 +
            complexity_confidence * 0.3
        )

        return max(0.1, min(1.0, overall_confidence))

    def _generate_strategy_recommendation(
        self,
        primary_strategy: GenerationStrategy,
        secondary_strategy: GenerationStrategy,
        confidence: float,
        gap_profile: Dict[str, Any],
        complexity_assessment: Dict[str, Any],
        hardness_analysis: Dict[str, Any],
        focus_areas: List[str]
    ) -> StrategyRecommendation:
        """Generate comprehensive strategy recommendation."""

        # Calculate expected improvement
        expected_improvement = self._calculate_expected_improvement(
            primary_strategy, gap_profile, complexity_assessment
        )

        # Estimate sample requirements
        estimated_samples = self._estimate_sample_requirements(
            primary_strategy, gap_profile, complexity_assessment
        )

        # Determine difficulty distribution
        difficulty_distribution = self._determine_difficulty_distribution(
            primary_strategy, gap_profile, hardness_analysis
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            primary_strategy, gap_profile, complexity_assessment, hardness_analysis
        )

        return StrategyRecommendation(
            primary_strategy=primary_strategy,
            secondary_strategy=secondary_strategy,
            confidence=confidence,
            reasoning=reasoning,
            expected_improvement=expected_improvement,
            estimated_samples=estimated_samples,
            difficulty_distribution=difficulty_distribution,
            focus_areas=focus_areas,
            metadata={
                "analysis_timestamp": datetime.now(timezone.utc),
                "gap_profile": gap_profile,
                "complexity_assessment": complexity_assessment,
                "hardness_analysis": hardness_analysis
            }
        )

    def _calculate_expected_improvement(
        self,
        strategy: GenerationStrategy,
        gap_profile: Dict[str, Any],
        complexity_assessment: Dict[str, Any]
    ) -> float:
        """Calculate expected improvement from strategy."""
        base_improvement = gap_profile["improvement_potential"]

        # Strategy effectiveness multipliers
        effectiveness_multipliers = {
            GenerationStrategy.STATISTICAL: 0.7,
            GenerationStrategy.NEURAL_ENHANCED: 0.9,
            GenerationStrategy.RULE_FOCUSED: 0.8,
            GenerationStrategy.DIVERSITY_ENHANCED: 0.75,
            GenerationStrategy.HYBRID: 0.95,
            GenerationStrategy.DIFFUSION: 0.85
        }

        multiplier = effectiveness_multipliers.get(strategy, 0.7)

        # Adjust for complexity match
        if complexity_assessment["requires_advanced_methods"] and strategy in [
            GenerationStrategy.NEURAL_ENHANCED, GenerationStrategy.HYBRID, GenerationStrategy.DIFFUSION
        ]:
            multiplier *= 1.1

        return min(1.0, base_improvement * multiplier)

    def _estimate_sample_requirements(
        self,
        strategy: GenerationStrategy,
        gap_profile: Dict[str, Any],
        complexity_assessment: Dict[str, Any]
    ) -> int:
        """Estimate sample requirements for strategy."""
        base_samples = max(200, gap_profile["total_gaps"] * 100)

        # Apply strategy efficiency multiplier
        efficiency = self.config["sample_efficiency"].get(strategy.value, 1.0)
        adjusted_samples = int(base_samples * efficiency)

        # Adjust for complexity
        if complexity_assessment["overall_complexity"] > 0.7:
            adjusted_samples = int(adjusted_samples * 1.3)

        return min(2000, max(100, adjusted_samples))

    def _determine_difficulty_distribution(
        self,
        strategy: GenerationStrategy,
        gap_profile: Dict[str, Any],
        hardness_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Determine optimal difficulty distribution for strategy."""
        critical_ratio = gap_profile["severity_distribution"].get("critical", 0)
        hard_ratio = hardness_analysis.get("distribution", {}).get("hard_examples_ratio", 0.3)

        if strategy == GenerationStrategy.NEURAL_ENHANCED or critical_ratio > 0.5:
            return {"easy": 0.15, "medium": 0.25, "hard": 0.60}
        elif strategy == GenerationStrategy.RULE_FOCUSED:
            return {"easy": 0.20, "medium": 0.50, "hard": 0.30}
        elif strategy == GenerationStrategy.DIVERSITY_ENHANCED:
            return {"easy": 0.40, "medium": 0.40, "hard": 0.20}
        else:
            return {"easy": 0.33, "medium": 0.34, "hard": 0.33}

    def _generate_reasoning(
        self,
        strategy: GenerationStrategy,
        gap_profile: Dict[str, Any],
        complexity_assessment: Dict[str, Any],
        hardness_analysis: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for strategy selection."""
        critical_ratio = gap_profile["severity_distribution"].get("critical", 0)
        complexity = complexity_assessment["overall_complexity"]
        hard_ratio = hardness_analysis.get("distribution", {}).get("hard_examples_ratio", 0.3)

        if strategy == GenerationStrategy.NEURAL_ENHANCED:
            return f"Neural enhancement recommended: {critical_ratio:.1%} critical gaps, {complexity:.1f} complexity score, {hard_ratio:.1%} hard examples require advanced pattern learning"
        elif strategy == GenerationStrategy.RULE_FOCUSED:
            consistency_gaps = gap_profile["gap_types"].get("consistency", 0)
            return f"Rule-focused generation recommended: {consistency_gaps} consistency gaps require targeted rule improvement"
        elif strategy == GenerationStrategy.DIVERSITY_ENHANCED:
            coverage_gaps = gap_profile["gap_types"].get("coverage", 0)
            return f"Diversity enhancement recommended: {coverage_gaps} coverage gaps require broader pattern exploration"
        elif strategy == GenerationStrategy.HYBRID:
            gap_types = len(gap_profile["gap_types"])
            return f"Hybrid approach recommended: {gap_types} different gap types and {complexity:.1f} complexity require multi-method approach"
        else:
            return f"Statistical generation sufficient: {complexity:.1f} complexity score and {critical_ratio:.1%} critical gaps manageable with standard methods"
