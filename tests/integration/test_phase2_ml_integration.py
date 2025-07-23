"""Phase 2 ML Integration Tests

Tests causal inference and pattern discovery enhancements to ensure proper
training data integration and false-positive prevention.

Phase 2 Components:
- CausalInferenceAnalyzer with training data integration
- AdvancedPatternDiscovery with training data pattern mining
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.models import RuleMetadata, RulePerformance, TrainingPrompt
from prompt_improver.ml.core.training_data_loader import TrainingDataLoader
from prompt_improver.ml.evaluation.causal_inference_analyzer import (
    CausalInferenceAnalyzer,
    TreatmentAssignment,
    CausalMethod
)
from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
    AdvancedPatternDiscovery
)
from prompt_improver.utils.datetime_utils import aware_utc_now


class TestPhase2MLIntegration:
    """Test suite for Phase 2 ML Training Data Integration components."""

    @pytest.fixture
    async def training_data_loader(self):
        """Create training data loader for Phase 2 testing."""
        return TrainingDataLoader(
            real_data_priority=True,
            min_samples=10,  # Lower minimum for testing
            lookback_days=30,
            synthetic_ratio=0.2
        )

    @pytest.fixture
    async def causal_analyzer(self, training_data_loader):
        """Create causal inference analyzer with training data integration."""
        return CausalInferenceAnalyzer(
            significance_level=0.05,
            minimum_effect_size=0.1,
            bootstrap_samples=100,  # Reduced for faster testing
            enable_sensitivity_analysis=True,
            training_loader=training_data_loader
        )

    @pytest.fixture
    async def pattern_discovery(self, training_data_loader):
        """Create advanced pattern discovery with training data integration."""
        return AdvancedPatternDiscovery(
            db_manager=None  # Will use lazy initialization
        )

    @pytest.fixture
    async def phase2_training_data(self, test_db_session: AsyncSession):
        """Create enhanced training data for Phase 2 testing."""
        # Create rule metadata
        rule_metadata = RuleMetadata(
            rule_id="phase2_test_rule",
            rule_name="Phase 2 Test Rule",
            category="causal_test",
            priority=5,
            rule_version="2.0.0",
            default_parameters={"causal_param": "test_value"}
        )
        test_db_session.add(rule_metadata)
        
        # Create diverse rule performance data for causal analysis
        base_time = aware_utc_now() - timedelta(days=5)
        performances = []
        
        # Create treatment and control groups
        for i in range(50):
            # Treatment group (higher effectiveness)
            if i < 25:
                effectiveness = 0.7 + (i % 5) * 0.05  # 0.7-0.9 range
                rule_id = "phase2_test_rule"
                treatment_group = 1
            else:
                # Control group (lower effectiveness) 
                effectiveness = 0.3 + (i % 5) * 0.04  # 0.3-0.5 range
                rule_id = "phase2_test_rule"
                treatment_group = 0
            
            perf = RulePerformance(
                rule_id=rule_id,
                rule_name="Phase 2 Test Rule",
                prompt_id=f"phase2_session_{i}",
                improvement_score=effectiveness,
                execution_time_ms=100 + i * 5,
                created_at=base_time + timedelta(hours=i),
                rule_parameters={
                    "treatment_group": treatment_group,
                    "param_value": i * 0.1,
                    "test_feature": np.random.random()
                }
            )
            performances.append(perf)
            test_db_session.add(perf)
        
        # Create training prompts with causal patterns
        for i in range(20):
            effectiveness = 0.6 + (i % 2) * 0.3  # Alternating high/low
            training_prompt = TrainingPrompt(
                prompt_text=f"Phase 2 training prompt {i}",
                enhancement_result={
                    "original_text": f"Phase 2 training prompt {i}",
                    "improved_text": f"Enhanced phase 2 training prompt {i}",
                    "improvement_score": effectiveness,
                    "rule_applications": ["phase2_test_rule"],
                    "causal_features": {
                        "treatment_indicator": i % 2,
                        "context_complexity": i * 0.05,
                        "domain_specificity": np.random.random()
                    }
                },
                created_at=base_time + timedelta(hours=i * 3),
                data_source="real"
            )
            test_db_session.add(training_prompt)
        
        await test_db_session.commit()
        return {
            "rule_performances": performances,
            "training_prompts": 20,
            "causal_groups": {"treatment": 25, "control": 25}
        }

    async def test_causal_analyzer_training_data_integration(
        self,
        test_db_session: AsyncSession,
        phase2_training_data,
        causal_analyzer: CausalInferenceAnalyzer
    ):
        """Test causal inference analyzer training data integration."""
        
        # Test training data causal analysis
        result = await causal_analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="phase2_test_rule",
            outcome_metric="improvement_score",
            treatment_variable="treatment_group"
        )
        
        # Verify integration success
        assert result.analysis_id.startswith("training_data_causality"), "Should generate training data analysis ID"
        assert result.average_treatment_effect is not None, "Should estimate treatment effect"
        assert result.treatment_assignment == TreatmentAssignment.QUASI_EXPERIMENTAL, "Should use quasi-experimental assignment"
        
        # Verify training data insights
        training_metadata = result.average_treatment_effect.metadata
        assert "training_samples" in training_metadata, "Should include training sample count"
        assert "real_samples" in training_metadata, "Should include real sample count"
        assert "synthetic_samples" in training_metadata, "Should include synthetic sample count"
        
        # Verify business recommendations include training insights
        recommendations = result.business_recommendations
        training_recs = [rec for rec in recommendations if "Training" in rec or "training" in rec]
        assert len(training_recs) > 0, "Should include training-specific recommendations"

    async def test_causal_analyzer_insufficient_data_handling(
        self,
        test_db_session: AsyncSession,
        causal_analyzer: CausalInferenceAnalyzer
    ):
        """Test causal analyzer handles insufficient training data gracefully."""
        
        # Create analyzer with high minimum sample requirement
        high_threshold_loader = TrainingDataLoader(
            min_samples=1000,  # Very high threshold
            real_data_priority=True
        )
        high_threshold_analyzer = CausalInferenceAnalyzer(
            training_loader=high_threshold_loader
        )
        
        # Test with insufficient data
        result = await high_threshold_analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="nonexistent_rule"
        )
        
        # Verify insufficient data handling
        assert "insufficient_data" in result.analysis_id, "Should detect insufficient data"
        assert result.average_treatment_effect.effect_name == "Insufficient Data", "Should indicate insufficient data"
        assert not result.average_treatment_effect.statistical_significance, "Should not claim significance"
        assert not result.average_treatment_effect.practical_significance, "Should not claim practical significance"
        
        # Verify appropriate warnings
        assert len(result.statistical_warnings) > 0, "Should include warnings"
        assert any("Sample size" in warning for warning in result.statistical_warnings), "Should warn about sample size"

    async def test_causal_analyzer_rule_effectiveness_analysis(
        self,
        test_db_session: AsyncSession,
        phase2_training_data,
        causal_analyzer: CausalInferenceAnalyzer
    ):
        """Test rule effectiveness causal analysis."""
        
        intervention_rules = ["phase2_test_rule"]
        control_rules = ["control_rule_1", "control_rule_2"]
        
        result = await causal_analyzer.analyze_rule_effectiveness_causality(
            test_db_session,
            intervention_rules,
            control_rules
        )
        
        # Verify rule effectiveness analysis
        assert result.analysis_id.startswith("rule_effectiveness_causality"), "Should generate rule effectiveness analysis ID"
        assert result.average_treatment_effect is not None, "Should estimate rule effectiveness effect"
        
        # Verify rule-specific metadata
        rule_metadata = result.average_treatment_effect.metadata
        assert "intervention_rules" in rule_metadata, "Should include intervention rules"
        assert "control_rules" in rule_metadata, "Should include control rules"
        assert rule_metadata["n_intervention_rules"] == 1, "Should count intervention rules"
        assert rule_metadata["n_control_rules"] == 2, "Should count control rules"

    async def test_pattern_discovery_training_data_integration(
        self,
        test_db_session: AsyncSession,
        phase2_training_data,
        pattern_discovery: AdvancedPatternDiscovery
    ):
        """Test advanced pattern discovery training data integration."""
        
        # Test training data pattern discovery
        result = await pattern_discovery.discover_training_data_patterns(
            test_db_session,
            pattern_types=["clustering", "feature_importance", "effectiveness"],
            min_effectiveness=0.7,
            use_clustering=True,
            include_feature_patterns=True
        )
        
        # Verify integration success
        assert result["status"] == "success", f"Pattern discovery should succeed, got: {result}"
        assert "training_metadata" in result, "Should include training metadata"
        assert "patterns" in result, "Should include discovered patterns"
        assert "insights" in result, "Should include pattern insights"
        assert "recommendations" in result, "Should include recommendations"
        
        # Verify pattern types
        patterns = result["patterns"]
        expected_patterns = ["clustering", "feature_importance", "effectiveness"]
        for pattern_type in expected_patterns:
            assert pattern_type in patterns, f"Should include {pattern_type} patterns"
        
        # Verify clustering patterns if successful
        if patterns["clustering"]["status"] == "success":
            assert "n_clusters" in patterns["clustering"], "Should report number of clusters"
            assert "cluster_analysis" in patterns["clustering"], "Should include cluster analysis"
        
        # Verify feature importance patterns
        if patterns["feature_importance"]["status"] == "success":
            assert "top_features" in patterns["feature_importance"], "Should identify top features"
            assert "max_correlation" in patterns["feature_importance"], "Should report max correlation"
        
        # Verify effectiveness patterns
        if patterns["effectiveness"]["status"] == "success":
            assert "effectiveness_stats" in patterns["effectiveness"], "Should include effectiveness statistics"

    async def test_pattern_discovery_insufficient_data_handling(
        self,
        test_db_session: AsyncSession,
        pattern_discovery: AdvancedPatternDiscovery
    ):
        """Test pattern discovery handles insufficient training data gracefully."""
        
        # Create pattern discovery with high threshold loader
        high_threshold_loader = TrainingDataLoader(
            min_samples=1000,  # Very high threshold
            real_data_priority=True
        )
        high_threshold_discovery = AdvancedPatternDiscovery()
        
        # Test with insufficient data
        result = await high_threshold_discovery.discover_training_data_patterns(
            test_db_session,
            pattern_types=["clustering"]
        )
        
        # Verify insufficient data handling
        assert result["status"] == "insufficient_data", "Should detect insufficient data"
        assert "samples" in result, "Should report sample count"
        assert "message" in result, "Should include descriptive message"

    async def test_phase2_false_positive_prevention(
        self,
        test_db_session: AsyncSession,
        causal_analyzer: CausalInferenceAnalyzer,
        pattern_discovery: AdvancedPatternDiscovery
    ):
        """Test Phase 2 components prevent false positives with empty data."""
        
        # Create components with very high thresholds to ensure insufficient data
        empty_loader = TrainingDataLoader(min_samples=1000)
        
        empty_causal_analyzer = CausalInferenceAnalyzer(
            training_loader=empty_loader
        )
        empty_pattern_discovery = AdvancedPatternDiscovery()
        
        # Test causal analyzer false positive prevention
        causal_result = await empty_causal_analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="nonexistent_rule"
        )
        
        # Should not claim false positive results
        assert not causal_result.average_treatment_effect.statistical_significance, "Should not claim statistical significance"
        assert not causal_result.average_treatment_effect.practical_significance, "Should not claim practical significance"
        assert causal_result.robustness_score == 0.0, "Should report zero robustness"
        assert not causal_result.overall_assumptions_satisfied, "Should not claim assumptions satisfied"
        
        # Test pattern discovery false positive prevention
        pattern_result = await empty_pattern_discovery.discover_training_data_patterns(
            test_db_session
        )
        
        # Should not claim false positive discoveries
        assert pattern_result["status"] == "insufficient_data", "Should detect insufficient data"
        assert "error" not in pattern_result or pattern_result.get("status") != "success", "Should not claim success"

    async def test_phase2_error_handling(
        self,
        test_db_session: AsyncSession,
        causal_analyzer: CausalInferenceAnalyzer,
        pattern_discovery: AdvancedPatternDiscovery
    ):
        """Test Phase 2 components handle errors gracefully."""
        
        # Test causal analyzer error handling with broken loader
        broken_loader = TrainingDataLoader()
        
        async def broken_load(*args, **kwargs):
            raise Exception("Simulated database failure")
        
        broken_loader.load_training_data = broken_load
        broken_analyzer = CausalInferenceAnalyzer(training_loader=broken_loader)
        
        # Test error handling
        result = await broken_analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="test_rule"
        )
        
        # Should handle errors gracefully
        assert "error" in result.analysis_id, "Should generate error result"
        assert result.causal_interpretation.startswith("Error in causal analysis"), "Should indicate error"
        assert len(result.statistical_warnings) > 0, "Should include error warnings"

    async def test_phase2_integration_with_real_data(
        self,
        test_db_session: AsyncSession,
        phase2_training_data,
        causal_analyzer: CausalInferenceAnalyzer,
        pattern_discovery: AdvancedPatternDiscovery
    ):
        """Test Phase 2 components integration with realistic data scenarios."""
        
        # Test parameter optimization causal analysis
        param_result = await causal_analyzer.analyze_parameter_optimization_causality(
            test_db_session,
            parameter_name="param_value",
            threshold_value=2.5
        )
        
        # Verify parameter analysis
        assert param_result.analysis_id.startswith("parameter_optimization_causality"), "Should analyze parameter optimization"
        
        # Verify parameter-specific metadata
        param_metadata = param_result.average_treatment_effect.metadata
        assert "parameter_name" in param_metadata, "Should include parameter name"
        assert "threshold_value" in param_metadata, "Should include threshold value"
        
        # Test comprehensive pattern discovery
        comprehensive_result = await pattern_discovery.discover_training_data_patterns(
            test_db_session,
            pattern_types=["clustering", "feature_importance", "effectiveness"],
            min_effectiveness=0.6,
            use_clustering=True,
            include_feature_patterns=True
        )
        
        # Verify comprehensive analysis
        assert comprehensive_result["status"] == "success", "Should succeed with real data"
        assert len(comprehensive_result["insights"]) > 0, "Should generate insights"
        assert len(comprehensive_result["recommendations"]) > 0, "Should generate recommendations"
        
        # Verify training metadata integration
        training_meta = comprehensive_result["training_metadata"]
        assert training_meta["total_samples"] > 0, "Should have training samples"
        assert "real_samples" in training_meta, "Should track real samples"
        assert "synthetic_samples" in training_meta, "Should track synthetic samples"

    async def test_phase2_performance_bounds(
        self,
        test_db_session: AsyncSession,
        phase2_training_data,
        causal_analyzer: CausalInferenceAnalyzer,
        pattern_discovery: AdvancedPatternDiscovery
    ):
        """Test Phase 2 components complete within reasonable time bounds."""
        import time
        
        # Test causal analyzer performance
        start_time = time.time()
        
        causal_result = await causal_analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="phase2_test_rule"
        )
        
        causal_elapsed = time.time() - start_time
        
        # Should complete within reasonable time (15 seconds for test data)
        assert causal_elapsed < 15.0, f"Causal analysis took too long: {causal_elapsed:.2f}s"
        
        # Test pattern discovery performance
        start_time = time.time()
        
        pattern_result = await pattern_discovery.discover_training_data_patterns(
            test_db_session,
            pattern_types=["clustering", "feature_importance"]
        )
        
        pattern_elapsed = time.time() - start_time
        
        # Should complete within reasonable time (10 seconds for test data)
        assert pattern_elapsed < 10.0, f"Pattern discovery took too long: {pattern_elapsed:.2f}s"
        
        # Verify both operations succeeded
        if causal_result.analysis_id.endswith("insufficient_data"):
            # This is acceptable for test data
            assert not causal_result.average_treatment_effect.statistical_significance
        else:
            assert causal_result.analysis_id.startswith("training_data_causality")
        
        assert pattern_result["status"] in ["success", "insufficient_data"], "Pattern discovery should complete"


# Mark tests that require ML dependencies
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not all([
            __import__('sklearn', fromlist=['']).__version__,
            __import__('numpy', fromlist=['']).__version__,
            __import__('scipy', fromlist=['']).__version__
        ]),
        reason="Phase 2 ML dependencies required"
    )
]