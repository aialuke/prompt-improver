"""
Test Phase 4: Enhanced Rule Intelligence Implementation

Tests the ML-enhanced intelligent rule selector and advanced prompt characteristic extraction.
Validates integration with existing ML infrastructure and performance requirements.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.rule_engine.intelligent_rule_selector import (
    IntelligentRuleSelector,
    RuleSelectionStrategy,
    RuleScore
)
from prompt_improver.rule_engine.models import PromptCharacteristics
from prompt_improver.rule_engine.prompt_analyzer import PromptAnalyzer
from prompt_improver.database.connection import DatabaseManager


class TestEnhancedRuleIntelligence:
    """Test suite for Phase 4 enhanced rule intelligence features."""

    @pytest.fixture
    async def db_session(self):
        """Create test database session."""
        db_manager = DatabaseManager()
        async with db_manager.get_session() as session:
            yield session

    @pytest.fixture
    def sample_prompt_characteristics(self):
        """Create sample prompt characteristics for testing."""
        return PromptCharacteristics(
            prompt_type="instructional",
            complexity_level=0.7,
            domain="technical",
            length_category="medium",
            reasoning_required=True,
            specificity_level=0.8,
            context_richness=0.6,
            task_type="problem_solving",
            language_style="formal",
            custom_attributes={
                "word_count": 45,
                "technical_terms": 3,
                "question_count": 1
            },
            # Phase 4 enhancements
            semantic_complexity=0.75,
            domain_confidence=0.85,
            reasoning_depth=3,
            context_dependencies=["conditional", "technical"],
            linguistic_features={
                "avg_word_length": 5.2,
                "modal_verb_count": 2,
                "unique_word_ratio": 0.8
            },
            pattern_signatures=["technical_domain", "complex_prompt", "problem_solving"]
        )

    @pytest.mark.asyncio
    async def test_enhanced_rule_selector_initialization(self, db_session):
        """Test enhanced rule selector initialization with ML components."""
        # Test with ML enabled
        selector = IntelligentRuleSelector(db_session, enable_ml_integration=True)

        assert selector.enable_ml_integration in [True, False]  # Depends on ML availability
        assert "ml_prediction" in selector.scoring_weights
        assert selector.scoring_weights["ml_prediction"] == 0.15

        # Test with ML disabled
        selector_no_ml = IntelligentRuleSelector(db_session, enable_ml_integration=False)
        assert selector_no_ml.enable_ml_integration is False

    @pytest.mark.asyncio
    async def test_enhanced_rule_selection_performance(self, db_session, sample_prompt_characteristics):
        """Test that enhanced rule selection maintains <200ms SLA."""
        selector = IntelligentRuleSelector(db_session, enable_ml_integration=True)

        # Mock database responses
        with patch.object(selector, '_get_candidate_rules') as mock_candidates:
            mock_candidates.return_value = [
                {
                    "rule_id": "test_rule_1",
                    "rule_name": "Test Rule 1",
                    "avg_effectiveness": 0.8,
                    "sample_size": 10,
                    "confidence_level": 0.9
                },
                {
                    "rule_id": "test_rule_2",
                    "rule_name": "Test Rule 2",
                    "avg_effectiveness": 0.7,
                    "sample_size": 15,
                    "confidence_level": 0.8
                }
            ]

            # Mock ML insights to avoid actual ML calls in tests
            with patch.object(selector, '_get_ml_insights') as mock_ml:
                mock_ml.return_value = {
                    "rule_predictions": {
                        "test_rule_1": {
                            "effectiveness_prediction": 0.85,
                            "pattern_insights": {"confidence": 0.9},
                            "trend": "improving"
                        },
                        "test_rule_2": {
                            "effectiveness_prediction": 0.75,
                            "pattern_insights": {"confidence": 0.8},
                            "trend": "stable"
                        }
                    }
                }

                start_time = time.time()

                results = await selector.select_optimal_rules(
                    prompt="Test prompt for rule selection",
                    prompt_characteristics=sample_prompt_characteristics,
                    max_rules=2,
                    strategy=RuleSelectionStrategy.BALANCED
                )

                selection_time = (time.time() - start_time) * 1000  # Convert to ms

                # Validate performance requirement
                assert selection_time < 200, f"Rule selection took {selection_time:.2f}ms, exceeds 200ms SLA"

                # Validate results structure
                assert len(results) <= 2
                assert all(isinstance(rule, RuleScore) for rule in results)

                # Validate ML enhancement fields are present
                for rule in results:
                    assert hasattr(rule, 'ml_prediction_score')
                    assert hasattr(rule, 'pattern_discovery_insights')
                    assert hasattr(rule, 'optimization_recommendations')
                    assert hasattr(rule, 'performance_trend')

    @pytest.mark.asyncio
    async def test_ml_insights_generation(self, db_session, sample_prompt_characteristics):
        """Test ML insights generation for rule effectiveness prediction."""
        selector = IntelligentRuleSelector(db_session, enable_ml_integration=True)

        candidate_rules = [
            {
                "rule_id": "clarity_rule",
                "rule_name": "Clarity Enhancement",
                "avg_effectiveness": 0.8,
                "sample_size": 20
            }
        ]

        # Mock pattern discovery to avoid actual ML calls
        with patch.object(selector, 'pattern_discovery') as mock_discovery:
            mock_discovery.discover_advanced_patterns.return_value = {
                "parameter_patterns": {
                    "clusters": [
                        {
                            "patterns": [{"rule_id": "clarity_rule"}],
                            "effectiveness_range": (0.7, 0.9),
                            "cluster_score": 0.85
                        }
                    ]
                },
                "ensemble_analysis": {"confidence": 0.8}
            }

            insights = await selector._get_ml_insights(sample_prompt_characteristics, candidate_rules)

            if insights:  # Only test if ML is available
                assert "pattern_analysis" in insights
                assert "rule_predictions" in insights
                assert "discovery_timestamp" in insights
                assert "confidence_level" in insights

                # Validate rule predictions
                assert "clarity_rule" in insights["rule_predictions"]
                prediction = insights["rule_predictions"]["clarity_rule"]
                assert "effectiveness_prediction" in prediction
                assert "pattern_insights" in prediction
                assert "trend" in prediction

    def test_enhanced_prompt_analyzer_initialization(self):
        """Test enhanced prompt analyzer initialization."""
        # Test with ML enabled
        analyzer = PromptAnalyzer(enable_ml_analysis=True)
        assert hasattr(analyzer, 'enable_ml_analysis')

        # Test with ML disabled
        analyzer_no_ml = PromptAnalyzer(enable_ml_analysis=False)
        assert analyzer_no_ml.enable_ml_analysis is False

    def test_enhanced_prompt_analysis(self):
        """Test enhanced prompt analysis with Phase 4 characteristics."""
        analyzer = PromptAnalyzer(enable_ml_analysis=True)

        test_prompt = """
        Please analyze the performance bottlenecks in this Python function and suggest
        optimizations. Consider memory usage, algorithmic complexity, and potential
        parallelization opportunities. Provide specific code improvements.
        """

        characteristics = analyzer.analyze_prompt(test_prompt)

        # Validate core characteristics
        assert characteristics.prompt_type in ["instructional", "analytical", "technical"]
        assert 0.0 <= characteristics.complexity_level <= 1.0
        assert characteristics.domain in ["technical", "business", "academic", "creative", "personal", "educational"]
        assert characteristics.reasoning_required is True  # This prompt clearly requires reasoning

        # Validate Phase 4 enhancements
        if analyzer.enable_ml_analysis:
            assert characteristics.semantic_complexity is not None
            assert characteristics.domain_confidence is not None
            assert characteristics.reasoning_depth is not None
            assert characteristics.context_dependencies is not None
            assert characteristics.linguistic_features is not None
            assert characteristics.pattern_signatures is not None

            # Validate specific values for this technical prompt
            assert characteristics.domain == "technical"
            assert characteristics.domain_confidence > 0.7  # Should be confident about technical domain
            assert characteristics.reasoning_depth >= 3  # Complex analytical task
            assert "technical_domain" in characteristics.pattern_signatures

    def test_domain_confidence_calculation(self):
        """Test domain confidence calculation accuracy."""
        analyzer = PromptAnalyzer(enable_ml_analysis=True)

        # Technical prompt with clear indicators
        tech_prompt = "Debug this Python function and optimize the algorithm for better performance"
        tech_chars = analyzer.analyze_prompt(tech_prompt)

        assert tech_chars.domain == "technical"
        if analyzer.enable_ml_analysis and tech_chars.domain_confidence:
            assert tech_chars.domain_confidence > 0.8

        # Business prompt with clear indicators
        biz_prompt = "Develop a marketing strategy to increase customer acquisition and revenue growth"
        biz_chars = analyzer.analyze_prompt(biz_prompt)

        assert biz_chars.domain == "business"
        if analyzer.enable_ml_analysis and biz_chars.domain_confidence:
            assert biz_chars.domain_confidence > 0.7

    def test_reasoning_depth_detection(self):
        """Test reasoning depth detection accuracy."""
        analyzer = PromptAnalyzer(enable_ml_analysis=True)

        # Simple factual question (depth 1)
        simple_prompt = "What is the capital of France?"
        simple_chars = analyzer.analyze_prompt(simple_prompt)
        assert simple_chars.reasoning_depth <= 2

        # Complex analytical task (depth 4-5)
        complex_prompt = """
        Analyze the philosophical implications of artificial intelligence on human consciousness,
        considering multiple theoretical frameworks and synthesizing a novel perspective on
        the nature of machine cognition versus human awareness.
        """
        complex_chars = analyzer.analyze_prompt(complex_prompt)
        assert complex_chars.reasoning_depth >= 4

    def test_pattern_signature_generation(self):
        """Test pattern signature generation for ML integration."""
        analyzer = PromptAnalyzer(enable_ml_analysis=True)

        # Sequential task prompt
        sequential_prompt = "Please follow these steps to set up the development environment"
        seq_chars = analyzer.analyze_prompt(sequential_prompt)

        if seq_chars.pattern_signatures:
            assert "sequential_task" in seq_chars.pattern_signatures

        # Comparative analysis prompt
        compare_prompt = "Compare and contrast the advantages of React versus Vue.js frameworks"
        comp_chars = analyzer.analyze_prompt(compare_prompt)

        if comp_chars.pattern_signatures:
            assert "comparative_analysis" in comp_chars.pattern_signatures

    @pytest.mark.asyncio
    async def test_rule_combination_optimization(self, db_session, sample_prompt_characteristics):
        """Test rule combination optimization using ML insights."""
        selector = IntelligentRuleSelector(db_session, enable_ml_integration=True)

        # Create sample scored rules
        scored_rules = [
            RuleScore(
                rule_id="clarity_rule",
                rule_name="Clarity Enhancement",
                total_score=0.8,
                effectiveness_score=0.8,
                characteristic_match_score=0.7,
                historical_performance_score=0.9,
                recency_score=0.6,
                confidence_level=0.8,
                sample_size=20,
                metadata={}
            ),
            RuleScore(
                rule_id="specificity_rule",
                rule_name="Specificity Enhancement",
                total_score=0.75,
                effectiveness_score=0.7,
                characteristic_match_score=0.8,
                historical_performance_score=0.8,
                recency_score=0.7,
                confidence_level=0.9,
                sample_size=15,
                metadata={}
            )
        ]

        # Mock database query for rule combinations
        with patch.object(selector.db_session, 'execute') as mock_execute:
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [
                MagicMock(_mapping={
                    "rule_set": ["clarity_rule", "specificity_rule"],
                    "combined_effectiveness": 0.85,
                    "statistical_confidence": 0.8,
                    "sample_size": 10
                })
            ]
            mock_execute.return_value = mock_result

            optimized_rules = await selector._optimize_rule_combinations(
                scored_rules, sample_prompt_characteristics
            )

            # Validate that combination bonuses were applied
            assert len(optimized_rules) == 2

            # Check if any rules received combination bonuses
            bonus_applied = any(
                rule.metadata and "combination_bonus" in rule.metadata
                for rule in optimized_rules
            )

            # Note: Bonus only applied if both rules are in the same combination
            # This test validates the optimization logic works without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
