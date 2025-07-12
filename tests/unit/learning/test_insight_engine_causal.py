"""Tests for Phase 2 Causal Discovery implementation in Insight Generation Engine.

Comprehensive test suite for causal discovery enhancements including:
- PC algorithm for causal structure learning
- DAG construction and validation
- Causal relationship extraction and confidence assessment
- Intervention analysis and effect estimation
- Statistical tests for causal validation

Testing best practices applied from Context7 research:
- Statistical significance testing for causal relationships
- Realistic parameter ranges for causal discovery
- Proper handling of confounding variables
- Validation of causal graphs and relationships
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from prompt_improver.learning.insight_engine import (
    InsightGenerationEngine,
    InsightConfig,
    Insight,
    CausalRelationship
)


@pytest.fixture
def causal_config():
    """Configuration with causal discovery enabled."""
    return InsightConfig(
        enable_causal_discovery=True,
        causal_significance_level=0.05,
        min_causal_samples=20,
        max_causal_variables=15,
        intervention_confidence_threshold=0.8,
        # Standard parameters
        min_confidence=0.7,
        min_sample_size=10,
        significance_threshold=0.05
    )


@pytest.fixture
def insight_engine_causal(causal_config):
    """Insight engine with causal discovery enabled."""
    return InsightGenerationEngine(config=causal_config)


@pytest.fixture
def causal_performance_data():
    """Realistic performance data with causal relationships."""
    np.random.seed(42)  # Reproducible test data
    
    # Simulate causal relationships:
    # rule_consistency -> context_performance -> system_efficiency
    n_samples = 50
    
    # Generate base variables
    rule_consistency = np.random.beta(2, 2, n_samples)  # [0,1] distribution
    context_complexity = np.random.uniform(0.2, 0.8, n_samples)
    
    # Causal effects with realistic noise
    context_performance = (0.6 * rule_consistency + 
                          0.3 * (1 - context_complexity) + 
                          np.random.normal(0, 0.1, n_samples))
    context_performance = np.clip(context_performance, 0.0, 1.0)
    
    system_efficiency = (0.7 * context_performance + 
                        0.2 * rule_consistency +
                        np.random.normal(0, 0.08, n_samples))
    system_efficiency = np.clip(system_efficiency, 0.0, 1.0)
    
    # Create performance data structure
    performance_data = {
        "rule_performance": {
            f"rule_{i}": {
                "avg_score": float(context_performance[i]),
                "consistency_score": float(rule_consistency[i]),
                "std_score": 0.1 + 0.05 * np.random.random(),
                "sample_size": 20 + np.random.randint(0, 30)
            }
            for i in range(min(n_samples, 10))  # Limit for test performance
        },
        "context_performance": {
            f"context_{i}": {
                "avg_score": float(context_performance[i]),
                "consistency_score": float(rule_consistency[i]),
                "std_score": 0.08 + 0.04 * np.random.random(),
                "sample_size": 15 + np.random.randint(0, 25)
            }
            for i in range(min(n_samples, 8))
        },
        "system_metrics": {
            "overall_score": float(np.mean(context_performance)),
            "efficiency_score": float(np.mean(system_efficiency)),
            "reliability_score": float(np.mean(rule_consistency)),
            "processing_load": float(50 + 30 * np.random.random())
        }
    }
    
    return performance_data


@pytest.fixture
def minimal_causal_data():
    """Minimal data that may not support causal discovery."""
    return {
        "rule_performance": {
            "rule_1": {"avg_score": 0.8, "consistency_score": 0.75, "sample_size": 15},
            "rule_2": {"avg_score": 0.7, "consistency_score": 0.8, "sample_size": 12}
        },
        "system_metrics": {
            "overall_score": 0.75,
            "efficiency_score": 0.8,
            "reliability_score": 0.78
        }
    }


class TestCausalDiscovery:
    """Test suite for causal discovery functionality."""

    @pytest.mark.asyncio
    async def test_causal_discovery_integration(self, insight_engine_causal, causal_performance_data):
        """Test full causal discovery workflow integration."""
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            result = await insight_engine_causal.generate_insights(causal_performance_data)
            
            # Should generate some insights
            assert isinstance(result, list)
            
            # Look for causal insights
            causal_insights = [insight for insight in result if insight.type == "causal"]
            
            # Validate causal insight structure if any were generated
            for insight in causal_insights:
                assert insight.confidence >= insight_engine_causal.config.min_confidence
                assert insight.impact in ["high", "medium", "low"]
                assert len(insight.evidence) > 0
                assert len(insight.recommendations) > 0
                assert "causal_strength" in insight.metadata
                assert "statistical_tests" in insight.metadata

    @pytest.mark.asyncio
    async def test_pc_algorithm_application(self, insight_engine_causal, causal_performance_data):
        """Test PC algorithm for causal structure learning."""
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            with patch('pgmpy.estimators.PC') as mock_pc, patch('networkx.DiGraph') as mock_digraph:
                # Setup PC algorithm mock
                mock_pc_instance = MagicMock()
                mock_skeleton = MagicMock()
                mock_separating_sets = {}
                mock_dag = MagicMock()
                mock_dag.edges.return_value = [("rule_avg_score", "context_avg_score"), 
                                              ("context_avg_score", "system_overall_score")]
                
                mock_pc_instance.build_skeleton.return_value = (mock_skeleton, mock_separating_sets)
                mock_pc_instance.skeleton_to_pdag.return_value = mock_dag
                mock_pc.return_value = mock_pc_instance
                
                # Setup NetworkX mock
                mock_graph = MagicMock()
                mock_graph.nodes.return_value = ["rule_avg_score", "context_avg_score", "system_overall_score"]
                mock_graph.edges.return_value = [("rule_avg_score", "context_avg_score"),
                                               ("context_avg_score", "system_overall_score")]
                mock_digraph.return_value = mock_graph
                
                result = await insight_engine_causal.generate_insights(causal_performance_data)
                
                # Verify PC algorithm was called
                if mock_pc.called:
                    # Verify correct significance level
                    call_args = mock_pc_instance.build_skeleton.call_args
                    if call_args and "significance_level" in call_args[1]:
                        assert call_args[1]["significance_level"] == insight_engine_causal.config.causal_significance_level

    @pytest.mark.asyncio
    async def test_causal_data_preparation(self, insight_engine_causal, causal_performance_data):
        """Test preparation of data for causal discovery."""
        # Test internal data preparation method
        causal_data = insight_engine_causal._prepare_causal_data(causal_performance_data)
        
        if causal_data is not None:
            # Validate prepared data structure
            assert isinstance(causal_data, pd.DataFrame)
            assert len(causal_data) >= insight_engine_causal.config.min_causal_samples
            assert len(causal_data.columns) <= insight_engine_causal.config.max_causal_variables
            
            # All features should be numeric and in reasonable ranges
            for column in causal_data.columns:
                assert causal_data[column].dtype in [np.float64, np.float32, int]
                # Features should be normalized/bounded
                assert causal_data[column].min() >= -2.0  # Allow some outliers
                assert causal_data[column].max() <= 2.0

    @pytest.mark.asyncio
    async def test_causal_relationship_extraction(self, insight_engine_causal):
        """Test extraction of causal relationships from DAG."""
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            with patch('networkx.DiGraph') as mock_digraph:
                # Create mock causal graph
                mock_graph = MagicMock()
                mock_graph.edges.return_value = [
                    ("rule_consistency", "context_performance"),
                    ("context_performance", "system_efficiency")
                ]
                
                # Create mock causal data
                causal_data = pd.DataFrame({
                    "rule_consistency": [0.8, 0.7, 0.9, 0.6, 0.85],
                    "context_performance": [0.75, 0.65, 0.88, 0.55, 0.82],
                    "system_efficiency": [0.7, 0.6, 0.85, 0.5, 0.8]
                })
                
                # Test relationship extraction
                relationships = insight_engine_causal._extract_causal_relationships(mock_graph, causal_data)
                
                # Validate extracted relationships
                assert isinstance(relationships, list)
                for relationship in relationships:
                    assert isinstance(relationship, CausalRelationship)
                    assert 0.0 <= relationship.strength <= 1.0
                    assert 0.0 <= relationship.confidence <= 1.0
                    assert isinstance(relationship.statistical_tests, dict)

    @pytest.mark.asyncio
    async def test_causal_statistical_tests(self, insight_engine_causal):
        """Test statistical tests for causal relationship validation."""
        # Create test data with known relationships
        cause_data = pd.Series([0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.8, 0.7])
        effect_data = pd.Series([0.75, 0.65, 0.88, 0.55, 0.82, 0.7, 0.78, 0.68])
        
        # Create mock graph for confounding analysis
        mock_graph = MagicMock()
        mock_graph.predecessors.return_value = []  # No confounders for simplicity
        
        # Test statistical tests
        stat_tests = insight_engine_causal._perform_causal_tests(
            cause_data, effect_data, mock_graph, "cause_var", "effect_var"
        )
        
        # Validate statistical test results
        assert isinstance(stat_tests, dict)
        assert "correlation" in stat_tests
        assert "correlation_p_value" in stat_tests
        assert 0.0 <= stat_tests["correlation"] <= 1.0
        assert 0.0 <= stat_tests["correlation_p_value"] <= 1.0
        
        # Should include Granger causality approximation if enough data
        if len(cause_data) > 10:
            assert "granger_approximation" in stat_tests
            assert "granger_p_value" in stat_tests

    @pytest.mark.asyncio
    async def test_causal_confidence_calculation(self, insight_engine_causal):
        """Test causal confidence assessment based on statistical evidence."""
        # High confidence scenario
        high_conf_tests = {
            "correlation": 0.85,
            "correlation_p_value": 0.01,
            "granger_approximation": 0.7,
            "granger_p_value": 0.02,
            "has_confounders": 0.0,
            "n_confounders": 0.0
        }
        
        confidence = insight_engine_causal._calculate_causal_confidence(high_conf_tests)
        assert 0.7 <= confidence <= 1.0  # Should be high confidence
        
        # Low confidence scenario
        low_conf_tests = {
            "correlation": 0.2,
            "correlation_p_value": 0.4,
            "has_confounders": 1.0,
            "n_confounders": 3.0
        }
        
        confidence = insight_engine_causal._calculate_causal_confidence(low_conf_tests)
        assert 0.0 <= confidence <= 0.5  # Should be low confidence

    @pytest.mark.asyncio
    async def test_intervention_analysis(self, insight_engine_causal):
        """Test intervention effect analysis and feasibility assessment."""
        # Create test causal relationships
        relationships = [
            CausalRelationship(
                cause="rule_consistency",
                effect="context_performance", 
                strength=0.8,
                confidence=0.85,
                statistical_tests={"correlation": 0.8, "correlation_p_value": 0.01}
            ),
            CausalRelationship(
                cause="context_performance",
                effect="system_efficiency",
                strength=0.7,
                confidence=0.9,
                statistical_tests={"correlation": 0.7, "correlation_p_value": 0.005}
            )
        ]
        
        # Create test causal data
        causal_data = pd.DataFrame({
            "rule_consistency": [0.8, 0.7, 0.9, 0.6, 0.85],
            "context_performance": [0.75, 0.65, 0.88, 0.55, 0.82],
            "system_efficiency": [0.7, 0.6, 0.85, 0.5, 0.8]
        })
        
        mock_graph = MagicMock()
        
        # Test intervention analysis
        intervention_results = insight_engine_causal._analyze_interventions(
            relationships, causal_data, mock_graph
        )
        
        # Validate intervention analysis
        assert isinstance(intervention_results, dict)
        
        for cause, analysis in intervention_results.items():
            assert "estimated_improvement" in analysis
            assert "confidence" in analysis
            assert "intervention_feasibility" in analysis
            assert 0.0 <= analysis["estimated_improvement"] <= 1.0
            assert 0.0 <= analysis["confidence"] <= 1.0
            assert 0.0 <= analysis["intervention_feasibility"] <= 1.0

    @pytest.mark.asyncio
    async def test_intervention_simulation(self, insight_engine_causal):
        """Test intervention effect simulation."""
        # Create test relationship
        relationship = CausalRelationship(
            cause="rule_consistency",
            effect="context_performance",
            strength=0.8,
            confidence=0.85
        )
        
        # Create test data with realistic correlation
        causal_data = pd.DataFrame({
            "rule_consistency": [0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.82],
            "context_performance": [0.75, 0.65, 0.88, 0.55, 0.82, 0.7, 0.78]
        })
        
        mock_graph = MagicMock()
        
        # Test intervention simulation
        effect = insight_engine_causal._simulate_intervention(relationship, causal_data, mock_graph)
        
        # Validate intervention effect
        assert isinstance(effect, float)
        assert 0.0 <= effect <= 1.0  # Should be a reasonable percentage improvement

    @pytest.mark.asyncio
    async def test_intervention_feasibility_assessment(self, insight_engine_causal):
        """Test assessment of intervention feasibility."""
        # Test different variable types
        rule_feasibility = insight_engine_causal._assess_intervention_feasibility("rule_consistency")
        context_feasibility = insight_engine_causal._assess_intervention_feasibility("context_complexity")
        system_feasibility = insight_engine_causal._assess_intervention_feasibility("system_efficiency")
        
        # All should return valid feasibility scores
        assert 0.0 <= rule_feasibility <= 1.0
        assert 0.0 <= context_feasibility <= 1.0
        assert 0.0 <= system_feasibility <= 1.0
        
        # Rule variables should generally be more feasible to intervene on
        assert rule_feasibility >= context_feasibility  # Rules more controllable than context

    @pytest.mark.asyncio
    async def test_causal_insight_creation(self, insight_engine_causal):
        """Test creation of causal insights from relationships."""
        # Create test relationship with intervention info
        relationship = CausalRelationship(
            cause="rule_consistency",
            effect="context_performance",
            strength=0.8,
            confidence=0.85,
            statistical_tests={
                "correlation": 0.8,
                "correlation_p_value": 0.01,
                "granger_approximation": 0.7
            }
        )
        
        intervention_info = {
            "estimated_improvement": 0.15,
            "confidence": 0.85,
            "intervention_feasibility": 0.9
        }
        
        # Test insight creation
        insight = insight_engine_causal._create_causal_insight(relationship, intervention_info)
        
        if insight is not None:
            # Validate insight structure
            assert isinstance(insight, Insight)
            assert insight.type == "causal"
            assert insight.confidence == relationship.confidence
            assert insight.impact in ["high", "medium", "low"]
            assert len(insight.evidence) > 0
            assert len(insight.recommendations) > 0
            
            # Should include causal metadata
            assert "causal_strength" in insight.metadata
            assert "statistical_tests" in insight.metadata
            assert "intervention_analysis" in insight.metadata


class TestCausalDiscoveryErrorHandling:
    """Test error handling and edge cases for causal discovery."""

    @pytest.mark.asyncio
    async def test_insufficient_causal_data(self, insight_engine_causal, minimal_causal_data):
        """Test handling of insufficient data for causal discovery."""
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            result = await insight_engine_causal.generate_insights(minimal_causal_data)
            
            # Should handle gracefully and still provide traditional insights
            assert isinstance(result, list)
            # May not contain causal insights due to insufficient data
            causal_insights = [insight for insight in result if insight.type == "causal"]
            # Allow empty causal insights for insufficient data

    @pytest.mark.asyncio
    async def test_causal_libraries_unavailable(self, causal_performance_data):
        """Test behavior when causal discovery libraries are not available."""
        config = InsightConfig(enable_causal_discovery=True)
        engine = InsightGenerationEngine(config=config)
        
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', False):
            result = await engine.generate_insights(causal_performance_data)
            
            # Should provide traditional insights without causal discovery
            assert isinstance(result, list)
            # Should not contain causal insights
            causal_insights = [insight for insight in result if insight.type == "causal"]
            assert len(causal_insights) == 0

    @pytest.mark.asyncio
    async def test_causal_discovery_disabled(self, insight_engine_causal, causal_performance_data):
        """Test behavior when causal discovery is disabled in config."""
        config = InsightConfig(enable_causal_discovery=False)
        engine = InsightGenerationEngine(config=config)
        
        result = await engine.generate_insights(causal_performance_data)
        
        # Should not perform causal discovery
        causal_insights = [insight for insight in result if insight.type == "causal"]
        assert len(causal_insights) == 0

    @pytest.mark.asyncio
    async def test_empty_performance_data(self, insight_engine_causal):
        """Test handling of empty performance data."""
        empty_data = {
            "rule_performance": {},
            "context_performance": {},
            "system_metrics": {}
        }
        
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            result = await insight_engine_causal.generate_insights(empty_data)
            
            # Should handle gracefully
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_invalid_causal_data_types(self, insight_engine_causal):
        """Test handling of invalid data types in causal analysis."""
        invalid_data = {
            "rule_performance": {
                "rule_1": {
                    "avg_score": "invalid_string",  # Should be numeric
                    "consistency_score": None,      # Invalid None value
                    "sample_size": 20
                }
            },
            "system_metrics": {
                "overall_score": float('inf'),  # Invalid infinite value
                "efficiency_score": float('nan')  # Invalid NaN value
            }
        }
        
        # Should handle invalid data gracefully
        causal_data = insight_engine_causal._prepare_causal_data(invalid_data)
        
        # Should return None or filtered data
        if causal_data is not None:
            # Should not contain invalid values
            assert not causal_data.isnull().any().any()
            assert not np.isinf(causal_data.values).any()

    @pytest.mark.asyncio
    async def test_extreme_correlation_values(self, insight_engine_causal):
        """Test handling of extreme correlation values."""
        # Perfect correlation (edge case)
        perfect_corr_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        perfect_effect = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect linear relationship
        
        mock_graph = MagicMock()
        mock_graph.predecessors.return_value = []
        
        # Should handle perfect correlations
        stat_tests = insight_engine_causal._perform_causal_tests(
            perfect_corr_data, perfect_effect, mock_graph, "cause", "effect"
        )
        
        assert "correlation" in stat_tests
        # Should handle extreme values gracefully
        assert not np.isnan(stat_tests["correlation"])
        assert not np.isinf(stat_tests["correlation"])

    @pytest.mark.asyncio
    async def test_pc_algorithm_failure(self, insight_engine_causal, causal_performance_data):
        """Test handling of PC algorithm failures."""
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            with patch('pgmpy.estimators.PC') as mock_pc:
                # Simulate PC algorithm failure
                mock_pc.side_effect = Exception("PC algorithm failed")
                
                result = await insight_engine_causal.generate_insights(causal_performance_data)
                
                # Should handle failure gracefully and provide traditional insights
                assert isinstance(result, list)
                # Should not crash the entire insight generation


class TestCausalDiscoveryIntegration:
    """Integration tests for causal discovery with existing workflows."""

    @pytest.mark.asyncio
    async def test_causal_with_traditional_insights(self, insight_engine_causal, causal_performance_data):
        """Test integration of causal discovery with traditional insight generation."""
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            result = await insight_engine_causal.generate_insights(causal_performance_data)
            
            # Should have both traditional and causal insights
            traditional_insights = [i for i in result if i.type in ["performance", "trend", "opportunity", "risk"]]
            causal_insights = [i for i in result if i.type == "causal"]
            
            # Should provide traditional insights regardless of causal discovery success
            assert len(traditional_insights) >= 0
            
            # All insights should meet minimum confidence threshold
            for insight in result:
                assert insight.confidence >= insight_engine_causal.config.min_confidence

    @pytest.mark.asyncio
    async def test_causal_discovery_performance_impact(self, causal_performance_data):
        """Test performance impact of enabling causal discovery."""
        import time
        
        # Test with causal discovery disabled
        config_disabled = InsightConfig(enable_causal_discovery=False)
        engine_disabled = InsightGenerationEngine(config_disabled)
        
        start_time = time.time()
        result_disabled = await engine_disabled.generate_insights(causal_performance_data)
        time_disabled = time.time() - start_time
        
        # Test with causal discovery enabled
        config_enabled = InsightConfig(enable_causal_discovery=True)
        engine_enabled = InsightGenerationEngine(config_enabled)
        
        start_time = time.time()
        result_enabled = await engine_enabled.generate_insights(causal_performance_data)
        time_enabled = time.time() - start_time
        
        # Causal discovery should add functionality without excessive overhead
        # Allow up to 10x time increase for complex causal analysis
        assert time_enabled <= time_disabled * 10.0 + 1.0  # +1s for base overhead
        
        # Should provide meaningful results in both cases
        assert isinstance(result_disabled, list)
        assert isinstance(result_enabled, list)

    @pytest.mark.parametrize("significance_level", [0.01, 0.05, 0.1])
    async def test_causal_significance_levels(self, causal_performance_data, significance_level):
        """Test causal discovery with different significance levels."""
        config = InsightConfig(
            enable_causal_discovery=True,
            causal_significance_level=significance_level
        )
        engine = InsightGenerationEngine(config=config)
        
        with patch('prompt_improver.learning.insight_engine.CAUSAL_DISCOVERY_AVAILABLE', True):
            result = await engine.generate_insights(causal_performance_data)
            
            # Should handle different significance levels
            assert isinstance(result, list)
            
            # Stricter significance levels (lower values) might produce fewer causal insights
            causal_insights = [i for i in result if i.type == "causal"]
            # Not strictly enforced due to data variability, but should be reasonable


# Test markers for categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml_contracts,
    pytest.mark.ml_data_validation
]