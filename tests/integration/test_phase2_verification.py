"""Phase 2 Component Verification Tests

Comprehensive tests to verify the 2 newly integrated Phase 2 components
are working correctly and not providing false-positive outputs.

Components Under Test:
1. CausalInferenceAnalyzer - Training data integration
2. AdvancedPatternDiscovery - Enhanced training data pattern mining
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.models import RuleMetadata, RulePerformance, TrainingPrompt
from prompt_improver.ml.core.training_data_loader import TrainingDataLoader
from prompt_improver.utils.datetime_utils import aware_utc_now


class TestPhase2ComponentVerification:
    """Verification tests for Phase 2 newly integrated components."""

    @pytest.fixture
    async def verification_training_data(self, test_db_session: AsyncSession):
        """Create comprehensive test data for verification."""
        # Create rule metadata
        rule_metadata = RuleMetadata(
            rule_id="verification_rule",
            rule_name="Verification Test Rule",
            category="verification",
            priority=5,
            rule_version="1.0.0",
            default_parameters={"test_param": "verification"}
        )
        test_db_session.add(rule_metadata)
        
        # Create sufficient performance data for analysis
        base_time = aware_utc_now() - timedelta(days=10)
        performances = []
        
        # Create 100 data points with clear patterns for testing
        for i in range(100):
            # Create treatment/control groups with measurable differences
            if i < 50:
                # Treatment group - higher effectiveness
                effectiveness = 0.8 + np.random.normal(0, 0.1)
                treatment_group = 1
            else:
                # Control group - lower effectiveness
                effectiveness = 0.4 + np.random.normal(0, 0.1)
                treatment_group = 0
            
            # Ensure effectiveness stays in valid range
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            perf = RulePerformance(
                rule_id="verification_rule",
                rule_name="Verification Test Rule",
                prompt_id=f"verification_session_{i}",
                improvement_score=effectiveness,
                execution_time_ms=100 + i,
                created_at=base_time + timedelta(hours=i),
                rule_parameters={
                    "treatment_group": treatment_group,
                    "feature_1": i * 0.01,
                    "feature_2": np.random.random(),
                    "feature_3": (i % 10) * 0.1
                }
            )
            performances.append(perf)
            test_db_session.add(perf)
        
        # Create training prompts with diverse patterns
        for i in range(50):
            effectiveness = 0.5 + (i % 5) * 0.1 + np.random.normal(0, 0.05)
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            training_prompt = TrainingPrompt(
                prompt_text=f"Verification training prompt {i}",
                enhancement_result={
                    "original_text": f"Original prompt {i}",
                    "improved_text": f"Improved prompt {i}",
                    "improvement_score": effectiveness,
                    "rule_applications": ["verification_rule"],
                    "metadata": {
                        "domain": "verification",
                        "complexity": i % 3,
                        "feature_vector": [i * 0.02, np.random.random(), (i % 7) * 0.1]
                    }
                },
                created_at=base_time + timedelta(hours=i * 2),
                data_source="real"
            )
            test_db_session.add(training_prompt)
        
        await test_db_session.commit()
        return {
            "rule_performances": 100,
            "training_prompts": 50,
            "total_samples": 150
        }

    @pytest.fixture
    async def empty_database_session(self, test_db_session: AsyncSession):
        """Create empty database session for false-positive testing."""
        # Ensure the database is truly empty for this test
        return test_db_session

    async def test_causal_analyzer_functionality_verification(
        self,
        test_db_session: AsyncSession,
        verification_training_data
    ):
        """Verify CausalInferenceAnalyzer works correctly with real data."""
        
        # Import here to avoid dependency issues
        from prompt_improver.ml.evaluation.causal_inference_analyzer import (
            CausalInferenceAnalyzer,
            TreatmentAssignment
        )
        
        # Create analyzer with proper training data integration
        training_loader = TrainingDataLoader(
            real_data_priority=True,
            min_samples=20,
            lookback_days=30,
            synthetic_ratio=0.2
        )
        
        analyzer = CausalInferenceAnalyzer(
            significance_level=0.05,
            minimum_effect_size=0.1,
            bootstrap_samples=100,
            enable_sensitivity_analysis=True,
            training_loader=training_loader
        )
        
        # Test 1: Training data causal analysis
        result = await analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="verification_rule",
            outcome_metric="improvement_score",
            treatment_variable="treatment_group"
        )
        
        # Verify proper functionality
        assert result is not None, "Should return a valid result"
        assert result.analysis_id.startswith("training_data_causality"), "Should have correct analysis ID"
        assert result.average_treatment_effect is not None, "Should estimate treatment effect"
        assert result.treatment_assignment == TreatmentAssignment.QUASI_EXPERIMENTAL, "Should use correct assignment"
        
        # Verify training data integration
        metadata = result.average_treatment_effect.metadata
        assert "training_samples" in metadata, "Should include training sample count"
        assert "real_samples" in metadata, "Should include real sample count"
        assert "synthetic_samples" in metadata, "Should include synthetic sample count"
        assert metadata["training_samples"] > 0, "Should have processed training samples"
        
        # Verify business recommendations include training insights
        recommendations = result.business_recommendations
        training_recs = [rec for rec in recommendations if "Training" in rec or "training" in rec or "📊" in rec or "💡" in rec]
        assert len(training_recs) > 0, "Should include training-specific recommendations"
        
        # Test 2: Rule effectiveness causal analysis
        effectiveness_result = await analyzer.analyze_rule_effectiveness_causality(
            test_db_session,
            intervention_rules=["verification_rule"],
            control_rules=["control_rule"]
        )
        
        # Verify rule effectiveness analysis
        assert effectiveness_result is not None, "Should return effectiveness result"
        assert effectiveness_result.analysis_id.startswith("rule_effectiveness_causality"), "Should have correct analysis ID"
        
        rule_metadata = effectiveness_result.average_treatment_effect.metadata
        assert "intervention_rules" in rule_metadata, "Should include intervention rules"
        assert "control_rules" in rule_metadata, "Should include control rules"
        
        # Test 3: Parameter optimization causal analysis
        param_result = await analyzer.analyze_parameter_optimization_causality(
            test_db_session,
            parameter_name="feature_1",
            threshold_value=0.5
        )
        
        # Verify parameter analysis
        assert param_result is not None, "Should return parameter result"
        assert param_result.analysis_id.startswith("parameter_optimization_causality"), "Should have correct analysis ID"
        
        param_metadata = param_result.average_treatment_effect.metadata
        assert "parameter_name" in param_metadata, "Should include parameter name"
        assert "threshold_value" in param_metadata, "Should include threshold value"

    async def test_causal_analyzer_false_positive_prevention(
        self,
        test_db_session: AsyncSession
    ):
        """Verify CausalInferenceAnalyzer prevents false positives."""
        
        from prompt_improver.ml.evaluation.causal_inference_analyzer import (
            CausalInferenceAnalyzer
        )
        
        # Test 1: High threshold training loader (insufficient data)
        high_threshold_loader = TrainingDataLoader(
            min_samples=10000,  # Impossibly high threshold
            real_data_priority=True
        )
        
        analyzer = CausalInferenceAnalyzer(
            training_loader=high_threshold_loader
        )
        
        # Test with insufficient data
        result = await analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="nonexistent_rule"
        )
        
        # Verify false positive prevention
        assert result.analysis_id.endswith("insufficient_data"), "Should detect insufficient data"
        assert result.average_treatment_effect.effect_name == "Insufficient Data", "Should indicate insufficient data"
        assert not result.average_treatment_effect.statistical_significance, "Should not claim statistical significance"
        assert not result.average_treatment_effect.practical_significance, "Should not claim practical significance"
        assert result.robustness_score == 0.0, "Should report zero robustness"
        assert not result.overall_assumptions_satisfied, "Should not claim assumptions satisfied"
        
        # Test 2: Non-existent rule
        normal_loader = TrainingDataLoader(min_samples=10)
        normal_analyzer = CausalInferenceAnalyzer(training_loader=normal_loader)
        
        no_data_result = await normal_analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="completely_nonexistent_rule"
        )
        
        # Should handle gracefully without false positives
        assert not no_data_result.average_treatment_effect.statistical_significance, "Should not claim significance with no data"
        assert not no_data_result.average_treatment_effect.practical_significance, "Should not claim practical significance with no data"

    async def test_pattern_discovery_functionality_verification(
        self,
        test_db_session: AsyncSession,
        verification_training_data
    ):
        """Verify AdvancedPatternDiscovery enhanced integration works correctly."""
        
        from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
            AdvancedPatternDiscovery
        )
        
        # Create pattern discovery with training data integration
        training_loader = TrainingDataLoader(
            real_data_priority=True,
            min_samples=20,
            lookback_days=30,
            synthetic_ratio=0.2
        )
        
        discovery = AdvancedPatternDiscovery(
            db_manager=None,  # Use lazy initialization
            training_loader=training_loader
        )
        
        # Test 1: Training data pattern discovery
        result = await discovery.discover_training_data_patterns(
            test_db_session,
            pattern_types=["clustering", "feature_importance", "effectiveness"],
            min_effectiveness=0.7,
            use_clustering=True,
            include_feature_patterns=True
        )
        
        # Verify proper functionality
        assert result["status"] == "success", f"Pattern discovery should succeed, got: {result}"
        assert "training_metadata" in result, "Should include training metadata"
        assert "patterns" in result, "Should include discovered patterns"
        assert "insights" in result, "Should include pattern insights"
        assert "recommendations" in result, "Should include recommendations"
        
        # Verify training metadata
        training_meta = result["training_metadata"]
        assert training_meta["total_samples"] > 0, "Should have training samples"
        assert "real_samples" in training_meta, "Should track real samples"
        assert "synthetic_samples" in training_meta, "Should track synthetic samples"
        assert "synthetic_ratio" in training_meta, "Should track synthetic ratio"
        
        # Verify pattern types
        patterns = result["patterns"]
        expected_patterns = ["clustering", "feature_importance", "effectiveness"]
        for pattern_type in expected_patterns:
            assert pattern_type in patterns, f"Should include {pattern_type} patterns"
            pattern_result = patterns[pattern_type]
            assert "status" in pattern_result, f"{pattern_type} should have status"
        
        # Verify insights and recommendations
        assert len(result["insights"]) > 0, "Should generate insights from training data"
        assert len(result["recommendations"]) > 0, "Should generate actionable recommendations"
        
        # Verify recommendations include training-specific content
        recommendations = result["recommendations"]
        training_recs = [rec for rec in recommendations if any(indicator in rec for indicator in ["🔍", "📊", "✅", "📈", "🤖"])]
        assert len(training_recs) > 0, "Should include training-specific recommendations"

    async def test_pattern_discovery_false_positive_prevention(
        self,
        test_db_session: AsyncSession
    ):
        """Verify AdvancedPatternDiscovery prevents false positives."""
        
        from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
            AdvancedPatternDiscovery
        )
        
        # Test 1: High threshold training loader (insufficient data)
        high_threshold_loader = TrainingDataLoader(
            min_samples=10000,  # Impossibly high threshold
            real_data_priority=True
        )
        
        discovery = AdvancedPatternDiscovery(
            training_loader=high_threshold_loader
        )
        
        # Test with insufficient data
        result = await discovery.discover_training_data_patterns(
            test_db_session,
            pattern_types=["clustering", "feature_importance"]
        )
        
        # Verify false positive prevention
        assert result["status"] == "insufficient_data", "Should detect insufficient data"
        assert "samples" in result, "Should report sample count"
        assert "message" in result, "Should include descriptive message"
        assert result["samples"] >= 0, "Sample count should be non-negative"
        
        # Should not claim success or provide false discoveries
        assert "patterns" not in result or not result.get("patterns"), "Should not provide patterns with insufficient data"
        assert "insights" not in result or len(result.get("insights", [])) == 0, "Should not provide insights with insufficient data"

    async def test_edge_case_handling_verification(
        self,
        test_db_session: AsyncSession
    ):
        """Verify both components handle edge cases without false positives."""
        
        from prompt_improver.ml.evaluation.causal_inference_analyzer import (
            CausalInferenceAnalyzer
        )
        from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
            AdvancedPatternDiscovery
        )
        
        # Create components
        training_loader = TrainingDataLoader(min_samples=5)
        analyzer = CausalInferenceAnalyzer(training_loader=training_loader)
        discovery = AdvancedPatternDiscovery(training_loader=training_loader)
        
        # Test 1: Empty database
        causal_result = await analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="empty_test"
        )
        
        pattern_result = await discovery.discover_training_data_patterns(
            test_db_session
        )
        
        # Both should handle empty database gracefully
        assert not causal_result.average_treatment_effect.statistical_significance, "Causal: Should not claim significance with empty data"
        assert pattern_result["status"] in ["insufficient_data", "error"], "Pattern: Should not claim success with empty data"
        
        # Test 2: Broken training loader
        broken_loader = TrainingDataLoader()
        
        async def broken_load(*args, **kwargs):
            raise Exception("Simulated training data failure")
        
        broken_loader.load_training_data = broken_load
        
        broken_analyzer = CausalInferenceAnalyzer(training_loader=broken_loader)
        broken_discovery = AdvancedPatternDiscovery(training_loader=broken_loader)
        
        # Test error handling
        error_causal_result = await broken_analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="error_test"
        )
        
        error_pattern_result = await broken_discovery.discover_training_data_patterns(
            test_db_session
        )
        
        # Should handle errors gracefully without false positives
        assert error_causal_result.analysis_id.endswith("error"), "Causal: Should generate error result"
        assert not error_causal_result.average_treatment_effect.statistical_significance, "Causal: Should not claim significance on error"
        
        assert error_pattern_result["status"] == "error", "Pattern: Should report error status"
        assert "error" in error_pattern_result, "Pattern: Should include error message"

    async def test_integration_data_flow_verification(
        self,
        test_db_session: AsyncSession,
        verification_training_data
    ):
        """Verify proper data flow through training data integration."""
        
        from prompt_improver.ml.evaluation.causal_inference_analyzer import (
            CausalInferenceAnalyzer
        )
        from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
            AdvancedPatternDiscovery
        )
        
        # Create components with same training loader
        training_loader = TrainingDataLoader(
            real_data_priority=True,
            min_samples=10,
            lookback_days=30,
            synthetic_ratio=0.3
        )
        
        analyzer = CausalInferenceAnalyzer(training_loader=training_loader)
        discovery = AdvancedPatternDiscovery(training_loader=training_loader)
        
        # Test data flow consistency
        causal_result = await analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="verification_rule"
        )
        
        pattern_result = await discovery.discover_training_data_patterns(
            test_db_session
        )
        
        # Verify consistent training data metadata
        if causal_result.analysis_id.startswith("training_data_causality"):
            causal_metadata = causal_result.average_treatment_effect.metadata
            pattern_metadata = pattern_result.get("training_metadata", {})
            
            # Both should report similar training data sizes
            causal_samples = causal_metadata.get("training_samples", 0)
            pattern_samples = pattern_metadata.get("total_samples", 0)
            
            # Allow for some variance due to different processing, but should be similar
            if causal_samples > 0 and pattern_samples > 0:
                ratio = min(causal_samples, pattern_samples) / max(causal_samples, pattern_samples)
                assert ratio > 0.5, f"Training sample counts should be similar: causal={causal_samples}, pattern={pattern_samples}"
            
            # Both should track real vs synthetic data
            assert "real_samples" in causal_metadata, "Causal: Should track real samples"
            assert "synthetic_samples" in causal_metadata, "Causal: Should track synthetic samples"
            assert "real_samples" in pattern_metadata, "Pattern: Should track real samples"
            assert "synthetic_samples" in pattern_metadata, "Pattern: Should track synthetic samples"

    async def test_performance_bounds_verification(
        self,
        test_db_session: AsyncSession,
        verification_training_data
    ):
        """Verify both components complete within reasonable time bounds."""
        import time
        
        from prompt_improver.ml.evaluation.causal_inference_analyzer import (
            CausalInferenceAnalyzer
        )
        from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
            AdvancedPatternDiscovery
        )
        
        # Create components
        training_loader = TrainingDataLoader(min_samples=10)
        analyzer = CausalInferenceAnalyzer(
            training_loader=training_loader,
            bootstrap_samples=50  # Reduced for performance testing
        )
        discovery = AdvancedPatternDiscovery(training_loader=training_loader)
        
        # Test causal analyzer performance
        start_time = time.time()
        causal_result = await analyzer.analyze_training_data_causality(
            test_db_session,
            rule_id="verification_rule"
        )
        causal_elapsed = time.time() - start_time
        
        # Test pattern discovery performance
        start_time = time.time()
        pattern_result = await discovery.discover_training_data_patterns(
            test_db_session,
            pattern_types=["clustering", "feature_importance"]
        )
        pattern_elapsed = time.time() - start_time
        
        # Verify reasonable performance bounds
        assert causal_elapsed < 30.0, f"Causal analysis took too long: {causal_elapsed:.2f}s"
        assert pattern_elapsed < 15.0, f"Pattern discovery took too long: {pattern_elapsed:.2f}s"
        
        # Verify both operations completed successfully or handled errors appropriately
        assert causal_result is not None, "Causal analysis should return a result"
        assert pattern_result is not None, "Pattern discovery should return a result"
        assert pattern_result["status"] in ["success", "insufficient_data"], "Pattern discovery should complete with valid status"


# Mark tests that require ML dependencies
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not all([
            __import__('sklearn', fromlist=['']),
            __import__('numpy', fromlist=['']),
            __import__('scipy', fromlist=[''])
        ]),
        reason="Phase 2 verification requires ML dependencies"
    )
]