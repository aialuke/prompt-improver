"""Rule Combination Optimization System.

Implements intelligent rule combination selection using historical effectiveness data
and statistical analysis to optimize multi-rule applications.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, stdev

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .intelligent_rule_selector import RuleScore
from .models import PromptCharacteristics

logger = logging.getLogger(__name__)

@dataclass
class RuleCombination:
    """Represents a combination of rules with effectiveness metrics."""
    rule_ids: List[str]
    rule_names: List[str]
    combined_effectiveness: float
    individual_scores: Dict[str, float]
    sample_size: int
    statistical_confidence: float
    synergy_score: float
    conflict_score: float
    metadata: Dict[str, Any]

@dataclass
class CombinationAnalysis:
    """Analysis results for rule combinations."""
    recommended_combinations: List[RuleCombination]
    synergy_pairs: List[Tuple[str, str, float]]
    conflict_pairs: List[Tuple[str, str, float]]
    optimal_combination: Optional[RuleCombination]
    analysis_confidence: float

class RuleCombinationOptimizer:
    """Optimizes rule combinations using historical effectiveness data.

    Features:
    - Historical combination analysis from rule_combinations table
    - Synergy detection between complementary rules
    - Conflict detection between competing rules
    - Statistical significance validation
    - Dynamic combination generation and testing
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize rule combination optimizer.

        Args:
            db_session: Database session for queries
        """
        self.db_session = db_session

        # Optimization thresholds
        self.effectiveness_threshold = 0.6
        self.min_sample_size = 5
        self.min_confidence = 0.7
        self.max_combination_size = 4

        # Synergy and conflict detection
        self.synergy_threshold = 0.15  # 15% improvement over individual rules
        self.conflict_threshold = -0.10  # 10% degradation indicates conflict

        # Performance tracking
        self._optimization_cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def optimize_rule_combinations(
        self,
        selected_rules: List[RuleScore],
        characteristics: PromptCharacteristics,
        max_combinations: int = 5
    ) -> CombinationAnalysis:
        """Optimize rule combinations for given prompt characteristics.

        Args:
            selected_rules: Pre-selected rules to combine
            characteristics: Prompt characteristics for context
            max_combinations: Maximum combinations to return

        Returns:
            CombinationAnalysis with optimization results
        """
        if len(selected_rules) < 2:
            return CombinationAnalysis(
                recommended_combinations=[],
                synergy_pairs=[],
                conflict_pairs=[],
                optimal_combination=None,
                analysis_confidence=0.0
            )

        # Get historical combination data
        historical_combinations = await self._get_historical_combinations(selected_rules)

        # Analyze synergies and conflicts
        synergy_pairs = await self._detect_synergies(selected_rules, historical_combinations)
        conflict_pairs = await self._detect_conflicts(selected_rules, historical_combinations)

        # Generate and evaluate combinations
        candidate_combinations = await self._generate_candidate_combinations(
            selected_rules, characteristics
        )

        # Score and rank combinations
        scored_combinations = await self._score_combinations(
            candidate_combinations, synergy_pairs, conflict_pairs
        )

        # Select top combinations
        recommended_combinations = scored_combinations[:max_combinations]

        # Determine optimal combination
        optimal_combination = None
        if recommended_combinations:
            optimal_combination = max(
                recommended_combinations,
                key=lambda c: c.combined_effectiveness * c.statistical_confidence
            )

        # Calculate analysis confidence
        analysis_confidence = self._calculate_analysis_confidence(
            historical_combinations, scored_combinations
        )

        logger.info(
            f"Optimized {len(candidate_combinations)} combinations, "
            f"found {len(synergy_pairs)} synergies, {len(conflict_pairs)} conflicts"
        )

        return CombinationAnalysis(
            recommended_combinations=recommended_combinations,
            synergy_pairs=synergy_pairs,
            conflict_pairs=conflict_pairs,
            optimal_combination=optimal_combination,
            analysis_confidence=analysis_confidence
        )

    async def _get_historical_combinations(
        self, selected_rules: List[RuleScore]
    ) -> List[Dict[str, Any]]:
        """Get historical combination data from database.

        Args:
            selected_rules: Rules to find combinations for

        Returns:
            List of historical combination records
        """
        rule_ids = [rule.rule_id for rule in selected_rules]

        # Query for combinations involving any of the selected rules
        query = text("""
            SELECT
                combination_id,
                rule_set,
                prompt_type,
                combined_effectiveness,
                individual_scores,
                sample_size,
                statistical_confidence,
                created_at,
                updated_at
            FROM rule_combinations
            WHERE
                combined_effectiveness >= :effectiveness_threshold
                AND sample_size >= :min_sample_size
                AND statistical_confidence >= :min_confidence
                AND (
                    rule_set ?| :rule_ids  -- Contains any of the rule IDs
                )
            ORDER BY combined_effectiveness DESC, sample_size DESC
            LIMIT 100
        """)

        result = await self.db_session.execute(query, {
            "effectiveness_threshold": self.effectiveness_threshold,
            "min_sample_size": self.min_sample_size,
            "min_confidence": self.min_confidence,
            "rule_ids": rule_ids
        })

        return [dict(row._mapping) for row in result.fetchall()]

    async def _detect_synergies(
        self,
        selected_rules: List[RuleScore],
        historical_combinations: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, float]]:
        """Detect synergistic rule pairs.

        Args:
            selected_rules: Selected rules
            historical_combinations: Historical combination data

        Returns:
            List of synergistic pairs with synergy scores
        """
        synergy_pairs = []
        rule_effectiveness = {rule.rule_id: rule.effectiveness_score for rule in selected_rules}

        # Analyze pairwise combinations
        for combo in historical_combinations:
            rule_set = combo["rule_set"]
            if not isinstance(rule_set, list) or len(rule_set) != 2:
                continue

            rule1, rule2 = rule_set
            if rule1 not in rule_effectiveness or rule2 not in rule_effectiveness:
                continue

            # Calculate expected effectiveness (average of individual rules)
            expected_effectiveness = (
                rule_effectiveness[rule1] + rule_effectiveness[rule2]
            ) / 2.0

            # Calculate actual combined effectiveness
            actual_effectiveness = combo["combined_effectiveness"]

            # Calculate synergy score
            synergy_score = actual_effectiveness - expected_effectiveness

            if synergy_score >= self.synergy_threshold:
                synergy_pairs.append((rule1, rule2, synergy_score))

        # Sort by synergy score
        synergy_pairs.sort(key=lambda x: x[2], reverse=True)

        return synergy_pairs

    async def _detect_conflicts(
        self,
        selected_rules: List[RuleScore],
        historical_combinations: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, float]]:
        """Detect conflicting rule pairs.

        Args:
            selected_rules: Selected rules
            historical_combinations: Historical combination data

        Returns:
            List of conflicting pairs with conflict scores
        """
        conflict_pairs = []
        rule_effectiveness = {rule.rule_id: rule.effectiveness_score for rule in selected_rules}

        # Analyze pairwise combinations for conflicts
        for combo in historical_combinations:
            rule_set = combo["rule_set"]
            if not isinstance(rule_set, list) or len(rule_set) != 2:
                continue

            rule1, rule2 = rule_set
            if rule1 not in rule_effectiveness or rule2 not in rule_effectiveness:
                continue

            # Calculate expected effectiveness
            expected_effectiveness = (
                rule_effectiveness[rule1] + rule_effectiveness[rule2]
            ) / 2.0

            # Calculate actual combined effectiveness
            actual_effectiveness = combo["combined_effectiveness"]

            # Calculate conflict score (negative synergy)
            conflict_score = actual_effectiveness - expected_effectiveness

            if conflict_score <= self.conflict_threshold:
                conflict_pairs.append((rule1, rule2, abs(conflict_score)))

        # Sort by conflict severity
        conflict_pairs.sort(key=lambda x: x[2], reverse=True)

        return conflict_pairs

    async def _generate_candidate_combinations(
        self,
        selected_rules: List[RuleScore],
        characteristics: PromptCharacteristics
    ) -> List[RuleCombination]:
        """Generate candidate rule combinations.

        Args:
            selected_rules: Selected rules
            characteristics: Prompt characteristics

        Returns:
            List of candidate combinations
        """
        candidates = []
        rule_ids = [rule.rule_id for rule in selected_rules]

        # Generate combinations of different sizes
        for size in range(2, min(len(selected_rules) + 1, self.max_combination_size + 1)):
            for rule_combo in combinations(selected_rules, size):
                combo_rule_ids = [rule.rule_id for rule in rule_combo]
                combo_rule_names = [rule.rule_name for rule in rule_combo]

                # Calculate baseline effectiveness (average of individual rules)
                baseline_effectiveness = mean([rule.effectiveness_score for rule in rule_combo])

                # Create candidate combination
                candidate = RuleCombination(
                    rule_ids=combo_rule_ids,
                    rule_names=combo_rule_names,
                    combined_effectiveness=baseline_effectiveness,
                    individual_scores={
                        rule.rule_id: rule.effectiveness_score for rule in rule_combo
                    },
                    sample_size=min([rule.sample_size for rule in rule_combo]),
                    statistical_confidence=mean([rule.confidence_level for rule in rule_combo]),
                    synergy_score=0.0,
                    conflict_score=0.0,
                    metadata={
                        "prompt_type": characteristics.prompt_type,
                        "domain": characteristics.domain,
                        "combination_size": size,
                        "generated_at": time.time()
                    }
                )

                candidates.append(candidate)

        return candidates

    async def _score_combinations(
        self,
        candidate_combinations: List[RuleCombination],
        synergy_pairs: List[Tuple[str, str, float]],
        conflict_pairs: List[Tuple[str, str, float]]
    ) -> List[RuleCombination]:
        """Score and rank rule combinations.

        Args:
            candidate_combinations: Candidate combinations
            synergy_pairs: Detected synergies
            conflict_pairs: Detected conflicts

        Returns:
            Scored and ranked combinations
        """
        # Create lookup dictionaries for synergies and conflicts
        synergy_dict = {(pair[0], pair[1]): pair[2] for pair in synergy_pairs}
        synergy_dict.update({(pair[1], pair[0]): pair[2] for pair in synergy_pairs})

        conflict_dict = {(pair[0], pair[1]): pair[2] for pair in conflict_pairs}
        conflict_dict.update({(pair[1], pair[0]): pair[2] for pair in conflict_pairs})

        # Score each combination
        for combination in candidate_combinations:
            synergy_score = 0.0
            conflict_score = 0.0

            # Calculate synergy and conflict scores for all pairs in combination
            for i, rule1 in enumerate(combination.rule_ids):
                for j, rule2 in enumerate(combination.rule_ids[i+1:], i+1):
                    # Check for synergy
                    if (rule1, rule2) in synergy_dict:
                        synergy_score += synergy_dict[(rule1, rule2)]

                    # Check for conflict
                    if (rule1, rule2) in conflict_dict:
                        conflict_score += conflict_dict[(rule1, rule2)]

            # Update combination scores
            combination.synergy_score = synergy_score
            combination.conflict_score = conflict_score

            # Adjust combined effectiveness based on synergies and conflicts
            adjustment = synergy_score - conflict_score
            combination.combined_effectiveness = max(
                0.0, combination.combined_effectiveness + adjustment
            )

        # Sort by adjusted effectiveness and confidence
        candidate_combinations.sort(
            key=lambda c: c.combined_effectiveness * c.statistical_confidence,
            reverse=True
        )

        return candidate_combinations

    def _calculate_analysis_confidence(
        self,
        historical_combinations: List[Dict[str, Any]],
        scored_combinations: List[RuleCombination]
    ) -> float:
        """Calculate confidence in the analysis results.

        Args:
            historical_combinations: Historical data used
            scored_combinations: Generated combinations

        Returns:
            Analysis confidence score (0.0 to 1.0)
        """
        factors = []

        # Historical data availability
        data_factor = min(1.0, len(historical_combinations) / 20.0)
        factors.append(data_factor * 0.4)

        # Sample size adequacy
        if historical_combinations:
            avg_sample_size = mean([combo["sample_size"] for combo in historical_combinations])
            sample_factor = min(1.0, avg_sample_size / 50.0)
            factors.append(sample_factor * 0.3)
        else:
            factors.append(0.0)

        # Statistical confidence
        if historical_combinations:
            avg_confidence = mean([combo["statistical_confidence"] for combo in historical_combinations])
            factors.append(avg_confidence * 0.2)
        else:
            factors.append(0.0)

        # Combination diversity
        if scored_combinations:
            effectiveness_values = [c.combined_effectiveness for c in scored_combinations]
            if len(effectiveness_values) > 1:
                diversity_factor = min(1.0, stdev(effectiveness_values) / 0.2)
                factors.append(diversity_factor * 0.1)
            else:
                factors.append(0.5 * 0.1)
        else:
            factors.append(0.0)

        return sum(factors)

    async def get_combination_recommendations(
        self,
        rule_ids: List[str],
        prompt_type: str = None
    ) -> List[Dict[str, Any]]:
        """Get specific combination recommendations from historical data.

        Args:
            rule_ids: Rule IDs to find combinations for
            prompt_type: Optional prompt type filter

        Returns:
            List of combination recommendations
        """
        query = text("""
            SELECT
                rule_set,
                combined_effectiveness,
                individual_scores,
                sample_size,
                statistical_confidence,
                prompt_type
            FROM rule_combinations
            WHERE
                combined_effectiveness >= :effectiveness_threshold
                AND sample_size >= :min_sample_size
                AND rule_set @> :rule_ids  -- Contains all specified rule IDs
                AND (:prompt_type IS NULL OR prompt_type = :prompt_type)
            ORDER BY combined_effectiveness DESC, sample_size DESC
            LIMIT 10
        """)

        result = await self.db_session.execute(query, {
            "effectiveness_threshold": self.effectiveness_threshold,
            "min_sample_size": self.min_sample_size,
            "rule_ids": rule_ids,
            "prompt_type": prompt_type
        })

        return [dict(row._mapping) for row in result.fetchall()]

    async def record_combination_result(
        self,
        rule_ids: List[str],
        effectiveness_score: float,
        individual_scores: Dict[str, float],
        prompt_type: str,
        confidence_level: float
    ) -> None:
        """Record a new combination result for future optimization.

        Args:
            rule_ids: Rule IDs in the combination
            effectiveness_score: Combined effectiveness score
            individual_scores: Individual rule scores
            prompt_type: Type of prompt used
            confidence_level: Confidence in the result
        """
        # Check if combination already exists
        query = text("""
            SELECT id, sample_size, combined_effectiveness
            FROM rule_combinations
            WHERE rule_set = :rule_ids AND prompt_type = :prompt_type
        """)

        result = await self.db_session.execute(query, {
            "rule_ids": rule_ids,
            "prompt_type": prompt_type
        })

        existing = result.fetchone()

        if existing:
            # Update existing combination
            new_sample_size = existing.sample_size + 1
            # Running average of effectiveness
            new_effectiveness = (
                (existing.combined_effectiveness * existing.sample_size + effectiveness_score) /
                new_sample_size
            )

            update_query = text("""
                UPDATE rule_combinations
                SET
                    combined_effectiveness = :new_effectiveness,
                    sample_size = :new_sample_size,
                    statistical_confidence = :confidence_level,
                    updated_at = NOW()
                WHERE id = :id
            """)

            await self.db_session.execute(update_query, {
                "new_effectiveness": new_effectiveness,
                "new_sample_size": new_sample_size,
                "confidence_level": confidence_level,
                "id": existing.id
            })
        else:
            # Insert new combination
            insert_query = text("""
                INSERT INTO rule_combinations (
                    rule_set, prompt_type, combined_effectiveness,
                    individual_scores, sample_size, statistical_confidence
                ) VALUES (
                    :rule_ids, :prompt_type, :effectiveness_score,
                    :individual_scores, 1, :confidence_level
                )
            """)

            await self.db_session.execute(insert_query, {
                "rule_ids": rule_ids,
                "prompt_type": prompt_type,
                "effectiveness_score": effectiveness_score,
                "individual_scores": individual_scores,
                "confidence_level": confidence_level
            })

        await self.db_session.commit()

        logger.info(f"Recorded combination result for {len(rule_ids)} rules: {effectiveness_score:.3f}")
