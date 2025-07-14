"""
Unit tests for Pattern Significance Analyzer
Tests pattern recognition and significance testing for A/B experiments
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, Mock

from src.prompt_improver.evaluation.pattern_significance_analyzer import (
    PatternSignificanceAnalyzer,
    PatternType,
    SignificanceMethod,
    PatternTestResult,
    PatternSignificanceReport,
    quick_pattern_analysis
)


class TestPatternSignificanceAnalyzer:
    """Test suite for PatternSignificanceAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return PatternSignificanceAnalyzer(
            alpha=0.05,
            min_sample_size=30,
            effect_size_threshold=0.1,
            apply_multiple_testing_correction=True
        )
    
    @pytest.fixture
    def categorical_pattern_data(self):
        """Create sample categorical pattern data"""
        patterns_data = {
            'pattern_1': {'type': 'category_preference', 'description': 'User prefers category A'},
            'pattern_2': {'type': 'click_behavior', 'description': 'High click-through rate'},
            'pattern_3': {'type': 'completion_rate', 'description': 'Task completion patterns'}
        }
        
        control_data = {
            'pattern_1': {'category_A': 45, 'category_B': 55, 'category_C': 30},
            'pattern_2': {'high_ctr': 25, 'medium_ctr': 60, 'low_ctr': 45},
            'pattern_3': {'completed': 85, 'partial': 30, 'abandoned': 15}
        }
        
        treatment_data = {
            'pattern_1': {'category_A': 65, 'category_B': 40, 'category_C': 25},
            'pattern_2': {'high_ctr': 45, 'medium_ctr': 55, 'low_ctr': 30},
            'pattern_3': {'completed': 95, 'partial': 25, 'abandoned': 10}
        }
        
        pattern_types = {
            'pattern_1': PatternType.CATEGORICAL,
            'pattern_2': PatternType.BEHAVIORAL,
            'pattern_3': PatternType.PERFORMANCE
        }
        
        return patterns_data, control_data, treatment_data, pattern_types
    
    @pytest.fixture
    def sequential_pattern_data(self):
        """Create sample sequential pattern data"""
        patterns_data = {
            'sequence_1': {'type': 'navigation_sequence', 'description': 'User navigation patterns'}
        }
        
        control_data = {
            'sequence_1': {
                'sequences': ['A->B->C', 'A->C->B', 'B->A->C'] * 20,
                'total_observations': 600
            }
        }
        
        treatment_data = {
            'sequence_1': {
                'sequences': ['A->B->C', 'A->C->B', 'B->A->C'] * 25,
                'total_observations': 500
            }
        }
        
        pattern_types = {
            'sequence_1': PatternType.SEQUENTIAL
        }
        
        return patterns_data, control_data, treatment_data, pattern_types
    
    @pytest.fixture
    def temporal_pattern_data(self):
        """Create sample temporal pattern data"""
        np.random.seed(42)
        
        patterns_data = {
            'temporal_1': {'type': 'response_time', 'description': 'User response timing patterns'}
        }
        
        control_data = {
            'temporal_1': {
                'timestamps': np.random.exponential(2.0, 50).tolist()
            }
        }
        
        treatment_data = {
            'temporal_1': {
                'timestamps': np.random.exponential(1.5, 45).tolist()
            }
        }
        
        pattern_types = {
            'temporal_1': PatternType.TEMPORAL
        }
        
        return patterns_data, control_data, treatment_data, pattern_types
    
    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = PatternSignificanceAnalyzer(
            alpha=0.01,
            min_sample_size=50,
            effect_size_threshold=0.2,
            apply_multiple_testing_correction=False
        )
        
        assert analyzer.alpha == 0.01
        assert analyzer.min_sample_size == 50
        assert analyzer.effect_size_threshold == 0.2
        assert not analyzer.apply_multiple_testing_correction
    
    def test_pattern_data_validation(self, analyzer, categorical_pattern_data):
        """Test pattern data validation"""
        patterns_data, control_data, treatment_data, _ = categorical_pattern_data
        
        validated = analyzer._validate_pattern_data(patterns_data, control_data, treatment_data)
        
        assert isinstance(validated, dict)
        assert len(validated) == 3  # All patterns should be valid
        assert 'pattern_1' in validated
        assert 'pattern_2' in validated
        assert 'pattern_3' in validated
    
    def test_pattern_data_validation_insufficient_sample(self, analyzer):
        """Test validation with insufficient sample size"""
        patterns_data = {'small_pattern': {'type': 'test'}}
        control_data = {'small_pattern': {'a': 5, 'b': 10}}  # Total 15 < min_sample_size
        treatment_data = {'small_pattern': {'a': 8, 'b': 12}}  # Total 20 < min_sample_size
        
        validated = analyzer._validate_pattern_data(patterns_data, control_data, treatment_data)
        
        assert len(validated) == 0  # Should filter out insufficient data
    
    def test_categorical_pattern_testing(self, analyzer, categorical_pattern_data):
        """Test categorical pattern significance testing"""
        _, control_data, treatment_data, pattern_types = categorical_pattern_data
        
        result = analyzer._test_categorical_pattern(
            'pattern_1', 
            control_data['pattern_1'], 
            treatment_data['pattern_1']
        )
        
        assert isinstance(result, PatternTestResult)
        assert result.pattern_id == 'pattern_1'
        assert result.pattern_type == PatternType.CATEGORICAL
        assert result.test_method in [SignificanceMethod.CHI_SQUARE, SignificanceMethod.FISHER_EXACT]
        assert result.statistic is not None
        assert result.p_value is not None
        assert result.effect_size is not None
        assert result.confidence_interval is not None
        assert result.sample_size > 0
        assert isinstance(result.interpretation, str)
        assert isinstance(result.recommendations, list)
    
    def test_sequential_pattern_testing(self, analyzer, sequential_pattern_data):
        """Test sequential pattern significance testing"""
        _, control_data, treatment_data, pattern_types = sequential_pattern_data
        
        result = analyzer._test_sequential_pattern(
            'sequence_1',
            control_data['sequence_1'],
            treatment_data['sequence_1']
        )
        
        assert isinstance(result, PatternTestResult)
        assert result.pattern_id == 'sequence_1'
        assert result.pattern_type == PatternType.SEQUENTIAL
        assert result.test_method == SignificanceMethod.PROPORTION_Z_TEST
        assert result.statistic is not None
        assert result.p_value is not None
        assert result.effect_size is not None
        assert 'control_rate' in result.metadata
        assert 'treatment_rate' in result.metadata
    
    def test_temporal_pattern_testing(self, analyzer, temporal_pattern_data):
        """Test temporal pattern significance testing"""
        _, control_data, treatment_data, pattern_types = temporal_pattern_data
        
        result = analyzer._test_temporal_pattern(
            'temporal_1',
            control_data['temporal_1'],
            treatment_data['temporal_1']
        )
        
        assert isinstance(result, PatternTestResult)
        assert result.pattern_id == 'temporal_1'
        assert result.pattern_type == PatternType.TEMPORAL
        assert result.test_method == SignificanceMethod.SEQUENCE_TEST
        assert result.statistic is not None
        assert result.p_value is not None
        assert result.effect_size is not None
        assert 'control_median' in result.metadata
        assert 'treatment_median' in result.metadata
    
    def test_performance_pattern_testing(self, analyzer):
        """Test performance pattern significance testing"""
        np.random.seed(42)
        
        # Create performance data
        control_performance = {
            'values': np.random.normal(0.7, 0.1, 50).tolist()
        }
        treatment_performance = {
            'values': np.random.normal(0.75, 0.1, 50).tolist()
        }
        
        result = analyzer._test_performance_pattern(
            'performance_1',
            control_performance,
            treatment_performance
        )
        
        assert isinstance(result, PatternTestResult)
        assert result.pattern_id == 'performance_1'
        assert result.pattern_type == PatternType.PERFORMANCE
        assert result.statistic is not None
        assert result.p_value is not None
        assert result.effect_size is not None
        assert 'control_mean' in result.metadata
        assert 'treatment_mean' in result.metadata
        assert 'mean_difference' in result.metadata
    
    def test_cramers_v_calculation(self, analyzer):
        """Test Cramér's V effect size calculation"""
        # Create a 2x2 contingency table
        contingency_table = np.array([[20, 30], [40, 10]])
        
        cramers_v = analyzer._calculate_cramers_v(contingency_table)
        
        assert 0 <= cramers_v <= 1
        assert isinstance(cramers_v, float)
    
    def test_comprehensive_pattern_analysis(self, analyzer, categorical_pattern_data):
        """Test complete pattern significance analysis"""
        patterns_data, control_data, treatment_data, pattern_types = categorical_pattern_data
        
        report = analyzer.analyze_pattern_significance(
            patterns_data=patterns_data,
            control_data=control_data,
            treatment_data=treatment_data,
            pattern_types=pattern_types
        )
        
        assert isinstance(report, PatternSignificanceReport)
        assert report.analysis_id is not None
        assert report.timestamp is not None
        assert report.total_patterns_tested == 3
        assert isinstance(report.significant_patterns, list)
        assert isinstance(report.non_significant_patterns, list)
        assert isinstance(report.pattern_categories, dict)
        assert 0 <= report.overall_significance_rate <= 1
        assert 0 <= report.false_discovery_rate <= 1
        assert isinstance(report.business_insights, list)
        assert 0 <= report.quality_score <= 1
    
    def test_multiple_testing_correction(self, analyzer):
        """Test multiple testing correction"""
        # Create mock test results
        test_results = [
            Mock(p_value=0.01, effect_size=0.5),
            Mock(p_value=0.03, effect_size=0.3),
            Mock(p_value=0.045, effect_size=0.2),
            Mock(p_value=0.08, effect_size=0.1)
        ]
        
        correction = analyzer._apply_multiple_testing_correction(test_results)
        
        assert isinstance(correction, dict)
        assert 'method' in correction
        assert 'original_p_values' in correction
        assert 'corrected_p_values' in correction
        assert 'rejected' in correction
        assert len(correction['corrected_p_values']) == len(test_results)
        assert len(correction['rejected']) == len(test_results)
    
    def test_pattern_categorization(self, analyzer):
        """Test pattern categorization"""
        # Create mock test results with various patterns
        test_results = [
            Mock(pattern_type=PatternType.CATEGORICAL, p_value=0.01, effect_size=0.8),
            Mock(pattern_type=PatternType.CATEGORICAL, p_value=0.06, effect_size=0.3),
            Mock(pattern_type=PatternType.SEQUENTIAL, p_value=0.02, effect_size=0.5),
            Mock(pattern_type=PatternType.TEMPORAL, p_value=0.001, effect_size=0.1)
        ]
        
        categories = analyzer._categorize_patterns(test_results)
        
        assert isinstance(categories, dict)
        assert 'categorical_total' in categories
        assert 'categorical_significant' in categories
        assert 'sequential_total' in categories
        assert 'temporal_total' in categories
        assert 'large_effect' in categories
        assert 'medium_effect' in categories
        assert 'small_effect' in categories
    
    def test_false_discovery_rate_calculation(self, analyzer):
        """Test false discovery rate calculation"""
        # Create mock significant patterns
        significant_patterns = [
            Mock(p_value=0.01),
            Mock(p_value=0.03),
            Mock(p_value=0.045)
        ]
        
        fdr = analyzer._calculate_false_discovery_rate(significant_patterns)
        
        assert 0 <= fdr <= 1
        assert isinstance(fdr, float)
        
        # Test with empty list
        fdr_empty = analyzer._calculate_false_discovery_rate([])
        assert fdr_empty == 0.0
    
    def test_business_insights_generation(self, analyzer):
        """Test business insights generation"""
        # Create mock significant patterns
        significant_patterns = [
            Mock(effect_size=0.8, pattern_type=PatternType.CATEGORICAL),
            Mock(effect_size=0.6, pattern_type=PatternType.PERFORMANCE),
            Mock(effect_size=0.3, pattern_type=PatternType.BEHAVIORAL)
        ]
        
        pattern_interactions = {
            'synergistic_patterns': [('pattern1', 'pattern2')],
            'antagonistic_patterns': []
        }
        
        insights = analyzer._generate_business_insights(significant_patterns, pattern_interactions)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert any("high-impact" in insight.lower() for insight in insights)
    
    def test_quality_score_calculation(self, analyzer):
        """Test quality score calculation"""
        # Create mock test results
        test_results = [
            Mock(sample_size=100, effect_size=0.5, p_value=0.01, pattern_type=PatternType.CATEGORICAL),
            Mock(sample_size=80, effect_size=0.3, p_value=0.03, pattern_type=PatternType.SEQUENTIAL),
            Mock(sample_size=120, effect_size=0.2, p_value=0.08, pattern_type=PatternType.TEMPORAL)
        ]
        
        # Mock multiple testing correction
        multiple_testing_correction = {'method': 'benjamini_hochberg'}
        
        quality_score = analyzer._calculate_quality_score(test_results, multiple_testing_correction)
        
        assert 0 <= quality_score <= 1
        assert isinstance(quality_score, float)
    
    def test_pattern_interactions_analysis(self, analyzer):
        """Test pattern interactions analysis"""
        # Create mock significant patterns
        significant_patterns = [
            Mock(pattern_id='pattern1', effect_size=0.5),
            Mock(pattern_id='pattern2', effect_size=0.6),
            Mock(pattern_id='pattern3', effect_size=0.4)
        ]
        
        # Mock data (interactions analysis is simplified in current implementation)
        control_data = {}
        treatment_data = {}
        
        interactions = analyzer._analyze_pattern_interactions(
            significant_patterns, control_data, treatment_data
        )
        
        assert isinstance(interactions, dict)
        assert 'pattern_correlations' in interactions
        assert 'synergistic_patterns' in interactions
        assert 'antagonistic_patterns' in interactions
    
    def test_interpretation_generation(self, analyzer):
        """Test interpretation generation for different pattern types"""
        # Test categorical interpretation
        interpretation = analyzer._interpret_categorical_result(
            0.01, 0.5, SignificanceMethod.CHI_SQUARE
        )
        assert isinstance(interpretation, str)
        assert "association" in interpretation.lower()
        
        # Test sequential interpretation
        interpretation = analyzer._interpret_sequential_result(0.02, 0.15)
        assert isinstance(interpretation, str)
        assert "sequential" in interpretation.lower()
        
        # Test temporal interpretation
        interpretation = analyzer._interpret_temporal_result(0.03, 0.4)
        assert isinstance(interpretation, str)
        assert "temporal" in interpretation.lower()
        
        # Test performance interpretation
        interpretation = analyzer._interpret_performance_result(0.01, 0.6)
        assert isinstance(interpretation, str)
        assert "performance" in interpretation.lower()
    
    def test_recommendations_generation(self, analyzer):
        """Test recommendations generation"""
        # Test categorical recommendations
        recommendations = analyzer._generate_categorical_recommendations(0.01, 0.8, 100)
        assert isinstance(recommendations, list)
        assert any("IMPLEMENT" in rec for rec in recommendations)
        
        # Test sequential recommendations
        recommendations = analyzer._generate_sequential_recommendations(0.02, 0.15)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Test temporal recommendations
        recommendations = analyzer._generate_temporal_recommendations(0.03, 0.4)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Test performance recommendations
        recommendations = analyzer._generate_performance_recommendations(0.01, 0.6)
        assert isinstance(recommendations, list)
        assert any("DEPLOY" in rec for rec in recommendations)
    
    def test_quick_pattern_analysis_utility(self, categorical_pattern_data):
        """Test quick pattern analysis utility function"""
        patterns_data, control_data, treatment_data, pattern_types = categorical_pattern_data
        
        result = quick_pattern_analysis(
            patterns_data=patterns_data,
            control_data=control_data,
            treatment_data=treatment_data,
            alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'total_patterns' in result
        assert 'significant_patterns' in result
        assert 'significance_rate' in result
        assert 'false_discovery_rate' in result
        assert 'top_patterns' in result
        assert 'business_insights' in result
        assert 'quality_score' in result
        
        assert isinstance(result['top_patterns'], list)
        assert 0 <= result['significance_rate'] <= 1
        assert 0 <= result['quality_score'] <= 1
    
    def test_error_handling(self, analyzer):
        """Test error handling in pattern analysis"""
        # Test with empty patterns data
        result = analyzer.analyze_pattern_significance(
            patterns_data={},
            control_data={},
            treatment_data={}
        )
        
        # Should handle empty data gracefully
        assert result.total_patterns_tested == 0
        assert len(result.significant_patterns) == 0
        assert len(result.non_significant_patterns) == 0
        assert "No patterns could be analyzed" in result.business_insights
        
        # Test quick analysis with error
        error_result = quick_pattern_analysis(
            patterns_data={},
            control_data={},
            treatment_data={}
        )
        
        assert error_result['total_patterns'] == 0
    
    def test_edge_cases(self, analyzer):
        """Test edge cases in pattern analysis"""
        # Test with single pattern with sufficient sample size
        single_pattern_data = {
            'single': {'type': 'test'}
        }
        single_control = {
            'single': {'a': 50, 'b': 50}  # Total = 100, sufficient
        }
        single_treatment = {
            'single': {'a': 60, 'b': 40}  # Total = 100, sufficient
        }
        
        result = analyzer.analyze_pattern_significance(
            patterns_data=single_pattern_data,
            control_data=single_control,
            treatment_data=single_treatment
        )
        
        # Should handle single pattern (no multiple testing correction needed)
        assert result.total_patterns_tested == 1
        assert result.multiple_testing_correction is None
        
        # Test with identical patterns (no effect)
        identical_control = {'pattern': {'a': 50, 'b': 50}}
        identical_treatment = {'pattern': {'a': 50, 'b': 50}}
        
        result_identical = analyzer.analyze_pattern_significance(
            patterns_data={'pattern': {'type': 'test'}},
            control_data=identical_control,
            treatment_data=identical_treatment
        )
        
        # Should detect no significant differences
        assert len(result_identical.significant_patterns) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])