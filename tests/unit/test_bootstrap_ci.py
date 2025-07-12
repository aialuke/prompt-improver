"""
Tests for Bootstrap Confidence Intervals Implementation

Tests the enhanced Statistical Analyzer bootstrap CI functionality with BCa method,
Hedges' g effect sizes, and adaptive FDR correction.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock

from src.prompt_improver.evaluation.statistical_analyzer import (
    StatisticalAnalyzer, 
    StatisticalConfig,
    DescriptiveStats
)


class TestBootstrapConfidenceIntervals:
    """Test suite for bootstrap confidence intervals with BCa method"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a statistical analyzer instance for testing"""
        config = StatisticalConfig(
            significance_level=0.05,
            confidence_level=0.95,
            minimum_sample_size=5,
            recommended_sample_size=30
        )
        return StatisticalAnalyzer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)  # For reproducible tests
        return {
            'normal_data': np.random.normal(50, 10, 100).tolist(),
            'skewed_data': np.random.exponential(2, 100).tolist(),
            'small_sample': np.random.normal(50, 10, 15).tolist(),
            'very_small_sample': [45, 48, 52, 55]
        }
    
    def test_bootstrap_ci_basic_functionality(self, analyzer, sample_data):
        """Test basic bootstrap confidence interval calculation"""
        values = sample_data['normal_data']
        
        # Test BCa method
        ci_bca = analyzer._calculate_bootstrap_ci(values, n_bootstrap=1000, method='bca')
        
        assert len(ci_bca) == 2
        assert ci_bca[0] < ci_bca[1]  # Lower bound < upper bound
        assert isinstance(ci_bca[0], float)
        assert isinstance(ci_bca[1], float)
        
        # Test percentile method for comparison
        ci_percentile = analyzer._calculate_bootstrap_ci(values, n_bootstrap=1000, method='percentile')
        
        assert len(ci_percentile) == 2
        assert ci_percentile[0] < ci_percentile[1]
    
    def test_bootstrap_ci_with_small_samples(self, analyzer, sample_data):
        """Test bootstrap CI with small sample sizes"""
        small_values = sample_data['small_sample']
        very_small_values = sample_data['very_small_sample']
        
        # Small sample (n=15)
        ci_small = analyzer._calculate_bootstrap_ci(small_values, n_bootstrap=1000)
        assert len(ci_small) == 2
        assert ci_small[0] < ci_small[1]
        
        # Very small sample (n=4)
        ci_very_small = analyzer._calculate_bootstrap_ci(very_small_values, n_bootstrap=1000)
        assert len(ci_very_small) == 2
        assert ci_very_small[0] < ci_very_small[1]
    
    def test_bootstrap_ci_insufficient_data(self, analyzer):
        """Test bootstrap CI with insufficient data"""
        # Empty data
        ci_empty = analyzer._calculate_bootstrap_ci([])
        assert ci_empty == (0.0, 0.0)
        
        # Single value
        ci_single = analyzer._calculate_bootstrap_ci([50.0])
        assert ci_single == (0.0, 0.0)
        
        # Two values
        ci_two = analyzer._calculate_bootstrap_ci([45.0, 55.0])
        assert ci_two == (0.0, 0.0)
    
    def test_bootstrap_ci_edge_cases(self, analyzer):
        """Test bootstrap CI with edge cases"""
        # All same values
        ci_constant = analyzer._calculate_bootstrap_ci([50.0] * 20, n_bootstrap=1000)
        assert len(ci_constant) == 2
        # For constant values, CI should be very narrow
        assert abs(ci_constant[1] - ci_constant[0]) < 1.0
        
        # Very large values
        large_values = [1e6, 1e6 + 100, 1e6 + 200, 1e6 + 300, 1e6 + 400]
        ci_large = analyzer._calculate_bootstrap_ci(large_values, n_bootstrap=1000)
        assert len(ci_large) == 2
        assert ci_large[0] < ci_large[1]
    
    def test_bootstrap_ci_with_error_handling(self, analyzer):
        """Test bootstrap CI error handling"""
        # Non-numeric data should be handled gracefully
        with patch('numpy.array') as mock_array:
            mock_array.side_effect = ValueError("Invalid data")
            ci_error = analyzer._calculate_bootstrap_ci([1, 2, 3, 4, 5])
            # Should fallback to simple percentile method
            assert len(ci_error) == 2
    
    @pytest.mark.asyncio
    async def test_bootstrap_ci_integration_with_confidence_intervals(self, analyzer, sample_data):
        """Test bootstrap CI integration with main confidence interval calculation"""
        # Create test results data
        test_results = [
            {
                "overallScore": score,
                "clarity": score + np.random.normal(0, 5),
                "completeness": score + np.random.normal(0, 5),
                "actionability": score + np.random.normal(0, 5),
                "effectiveness": score + np.random.normal(0, 5)
            }
            for score in sample_data['normal_data'][:50]
        ]
        
        # Perform statistical analysis
        analysis = await analyzer.perform_statistical_analysis(test_results)
        
        # Check that bootstrap CIs are included
        assert "confidence_intervals" in analysis
        ci_data = analysis["confidence_intervals"]
        
        for metric in ["overall_scores", "clarity_scores", "completeness_scores"]:
            if metric in ci_data:
                metric_ci = ci_data[metric]
                assert "bootstrap_lower" in metric_ci
                assert "bootstrap_upper" in metric_ci
                assert "bootstrap_method" in metric_ci
                assert metric_ci["bootstrap_method"] == "BCa"
                assert metric_ci["bootstrap_lower"] < metric_ci["bootstrap_upper"]


class TestHedgesGEffectSize:
    """Test suite for Hedges' g effect size calculation"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a statistical analyzer instance for testing"""
        return StatisticalAnalyzer()
    
    def test_hedges_g_basic_calculation(self, analyzer):
        """Test basic Hedges' g calculation"""
        # Create two groups with known effect size
        np.random.seed(42)
        control = np.random.normal(50, 10, 30).tolist()
        treatment = np.random.normal(55, 10, 30).tolist()  # 0.5 effect size
        
        effect_sizes = analyzer._calculate_effect_sizes(control, treatment)
        
        assert "cohens_d" in effect_sizes
        assert "hedges_g" in effect_sizes
        assert "glass_delta" in effect_sizes
        assert "recommended_measure" in effect_sizes
        assert "interpretation" in effect_sizes
        assert "bias_correction_factor" in effect_sizes
        
        # Hedges' g should be slightly smaller than Cohen's d (bias correction)
        assert effect_sizes["hedges_g"] < effect_sizes["cohens_d"]
        assert abs(effect_sizes["hedges_g"] - effect_sizes["cohens_d"]) < 0.1
    
    def test_hedges_g_small_samples(self, analyzer):
        """Test Hedges' g with small samples (where it's preferred)"""
        # Small samples (n=15 each)
        np.random.seed(42)
        control_small = np.random.normal(50, 10, 15).tolist()
        treatment_small = np.random.normal(55, 10, 15).tolist()
        
        effect_sizes = analyzer._calculate_effect_sizes(control_small, treatment_small)
        
        # For small samples, Hedges' g should be recommended
        assert effect_sizes["recommended_measure"] == "hedges_g"
        
        # Bias correction should be more pronounced with smaller samples
        bias_correction = effect_sizes["bias_correction_factor"]
        assert 0.9 < bias_correction < 1.0  # Should be less than 1 for bias correction
    
    def test_hedges_g_large_samples(self, analyzer):
        """Test Hedges' g with large samples (where Cohen's d is acceptable)"""
        # Large samples (n=100 each)
        np.random.seed(42)
        control_large = np.random.normal(50, 10, 100).tolist()
        treatment_large = np.random.normal(55, 10, 100).tolist()
        
        effect_sizes = analyzer._calculate_effect_sizes(control_large, treatment_large)
        
        # For large samples, Cohen's d should be recommended
        assert effect_sizes["recommended_measure"] == "cohens_d"
        
        # Bias correction should be minimal with larger samples
        bias_correction = effect_sizes["bias_correction_factor"]
        assert 0.98 < bias_correction < 1.0
    
    def test_hedges_g_effect_size_interpretation(self, analyzer):
        """Test effect size interpretation with field-specific guidelines"""
        # Test different effect sizes
        control = [50] * 30
        
        # Negligible effect
        treatment_negligible = [50.1] * 30
        effect_negligible = analyzer._calculate_effect_sizes(control, treatment_negligible)
        assert effect_negligible["interpretation"] == "negligible"
        
        # Small effect  
        treatment_small = [52] * 30
        effect_small = analyzer._calculate_effect_sizes(control, treatment_small)
        assert effect_small["interpretation"] == "small"
        
        # Medium effect
        treatment_medium = [56] * 30  
        effect_medium = analyzer._calculate_effect_sizes(control, treatment_medium)
        assert effect_medium["interpretation"] == "medium"
        
        # Large effect
        treatment_large = [60] * 30
        effect_large = analyzer._calculate_effect_sizes(control, treatment_large)
        assert effect_large["interpretation"] == "large"
    
    def test_hedges_g_insufficient_data(self, analyzer):
        """Test Hedges' g with insufficient data"""
        # Single value in each group
        result = analyzer._calculate_effect_sizes([50], [55])
        assert result["error"] == "Insufficient sample size"
        
        # Empty groups
        result = analyzer._calculate_effect_sizes([], [55, 60])
        assert result["error"] == "Insufficient sample size"
    
    def test_hedges_g_zero_variance(self, analyzer):
        """Test Hedges' g when variance is zero"""
        # Same values in both groups (zero pooled std)
        control = [50] * 20
        treatment = [50] * 20
        
        effect_sizes = analyzer._calculate_effect_sizes(control, treatment)
        
        assert effect_sizes["cohens_d"] == 0
        assert effect_sizes["hedges_g"] == 0
        assert effect_sizes["recommended_measure"] == "none"


class TestAdaptiveFDRCorrection:
    """Test suite for adaptive FDR correction (Benjamini-Krieger-Yekutieli)"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a statistical analyzer instance for testing"""
        return StatisticalAnalyzer()
    
    def test_adaptive_fdr_basic_functionality(self, analyzer):
        """Test basic adaptive FDR correction"""
        # Create p-values with some significant results
        p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
        
        fdr_results = analyzer._apply_multiple_testing_correction(
            p_values, fdr_level=0.05, method='adaptive'
        )
        
        assert "original_p_values" in fdr_results
        assert "adjusted_p_values" in fdr_results
        assert "rejected_hypotheses" in fdr_results
        assert "num_discoveries" in fdr_results
        assert "expected_fdr" in fdr_results
        assert "method_used" in fdr_results
        assert "interpretation" in fdr_results
        
        # Should have some discoveries
        assert fdr_results["num_discoveries"] >= 2
        
        # Adjusted p-values should be >= original p-values
        for orig, adj in zip(fdr_results["original_p_values"], fdr_results["adjusted_p_values"]):
            assert adj >= orig
    
    def test_adaptive_fdr_vs_standard_bh(self, analyzer):
        """Compare adaptive FDR with standard Benjamini-Hochberg"""
        p_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.08, 0.1, 0.3, 0.5, 0.9]
        
        # Adaptive method
        fdr_adaptive = analyzer._apply_multiple_testing_correction(
            p_values, fdr_level=0.05, method='adaptive'
        )
        
        # Standard BH method
        fdr_standard = analyzer._apply_multiple_testing_correction(
            p_values, fdr_level=0.05, method='standard'
        )
        
        # Both should work and return valid results
        assert isinstance(fdr_adaptive["num_discoveries"], int)
        assert isinstance(fdr_standard["num_discoveries"], int)
        
        # Adaptive method often has more power (more discoveries)
        # But this is not guaranteed for all datasets
        assert fdr_adaptive["method_used"] in ["fdr_by_adaptive", "fdr_bh_fallback"]
        assert fdr_standard["method_used"] == "fdr_bh_standard"
    
    def test_fdr_correction_no_significant_results(self, analyzer):
        """Test FDR correction when no results are significant"""
        # All p-values > 0.05
        p_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        
        fdr_results = analyzer._apply_multiple_testing_correction(p_values, fdr_level=0.05)
        
        assert fdr_results["num_discoveries"] == 0
        assert "No significant results" in fdr_results["interpretation"]
    
    def test_fdr_correction_all_significant(self, analyzer):
        """Test FDR correction when all results are highly significant"""
        # All p-values very small
        p_values = [0.001, 0.002, 0.003, 0.004, 0.005]
        
        fdr_results = analyzer._apply_multiple_testing_correction(p_values, fdr_level=0.05)
        
        assert fdr_results["num_discoveries"] == len(p_values)
        assert "results are reliable" in fdr_results["interpretation"]
    
    def test_fdr_correction_edge_cases(self, analyzer):
        """Test FDR correction with edge cases"""
        # Empty p-values
        fdr_empty = analyzer._apply_multiple_testing_correction([])
        assert "error" in fdr_empty
        
        # Single p-value
        fdr_single = analyzer._apply_multiple_testing_correction([0.01])
        assert fdr_single["num_discoveries"] == 1
        
        # Very large number of tests
        p_values_large = [0.01 + 0.001 * i for i in range(100)]
        fdr_large = analyzer._apply_multiple_testing_correction(p_values_large)
        assert isinstance(fdr_large["num_discoveries"], int)
    
    def test_fdr_interpretation_quality(self, analyzer):
        """Test quality of FDR result interpretations"""
        # Test different scenarios
        scenarios = [
            ([0.001, 0.002, 0.003], "reliable"),  # Very significant
            ([0.04, 0.045, 0.049], "reliable"),   # Borderline significant
            ([0.1, 0.2, 0.3], "No significant"), # No significant results
        ]
        
        for p_values, expected_keyword in scenarios:
            fdr_results = analyzer._apply_multiple_testing_correction(p_values, fdr_level=0.05)
            interpretation = fdr_results["interpretation"]
            assert expected_keyword in interpretation
    
    def test_fdr_correction_different_levels(self, analyzer):
        """Test FDR correction with different significance levels"""
        p_values = [0.001, 0.01, 0.03, 0.05, 0.1]
        
        # Strict level (1%)
        fdr_strict = analyzer._apply_multiple_testing_correction(p_values, fdr_level=0.01)
        
        # Standard level (5%)
        fdr_standard = analyzer._apply_multiple_testing_correction(p_values, fdr_level=0.05)
        
        # Lenient level (10%)
        fdr_lenient = analyzer._apply_multiple_testing_correction(p_values, fdr_level=0.10)
        
        # More lenient levels should have more discoveries
        assert fdr_strict["num_discoveries"] <= fdr_standard["num_discoveries"]
        assert fdr_standard["num_discoveries"] <= fdr_lenient["num_discoveries"]


@pytest.mark.integration
class TestBootstrapCIIntegration:
    """Integration tests for bootstrap CI with other statistical methods"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a statistical analyzer instance for integration testing"""
        return StatisticalAnalyzer()
    
    @pytest.mark.asyncio
    async def test_full_statistical_analysis_with_bootstrap(self, analyzer):
        """Test complete statistical analysis including bootstrap methods"""
        # Create comprehensive test dataset
        np.random.seed(42)
        test_results = []
        
        for i in range(50):
            base_score = 60 + np.random.normal(0, 10)
            result = {
                "overallScore": base_score,
                "clarity": base_score + np.random.normal(0, 5),
                "completeness": base_score + np.random.normal(2, 5),
                "actionability": base_score + np.random.normal(-1, 5),
                "effectiveness": base_score + np.random.normal(1, 5),
                "strategy": "test_strategy",
                "model": "test_model",
                "complexity": "medium"
            }
            test_results.append(result)
        
        # Perform comprehensive analysis
        analysis = await analyzer.perform_statistical_analysis(test_results)
        
        # Verify all components are present
        assert "descriptive_stats" in analysis
        assert "confidence_intervals" in analysis
        assert "correlation_analysis" in analysis
        
        # Verify bootstrap CIs are calculated
        ci_data = analysis["confidence_intervals"]
        for metric in ["overall_scores", "clarity_scores", "completeness_scores"]:
            if metric in ci_data:
                assert "bootstrap_lower" in ci_data[metric]
                assert "bootstrap_upper" in ci_data[metric]
                assert "bootstrap_method" in ci_data[metric]
        
        # Verify summary includes new insights
        assert "summary" in analysis
        assert "key_insights" in analysis
        
        # Verify recommendations include statistical advice
        assert "recommendations" in analysis
        recommendations = analysis["recommendations"]
        assert isinstance(recommendations, list)
    
    def test_bootstrap_ci_performance(self, analyzer):
        """Test bootstrap CI performance with large datasets"""
        import time
        
        # Large dataset
        np.random.seed(42)
        large_values = np.random.normal(50, 10, 1000).tolist()
        
        start_time = time.time()
        ci_result = analyzer._calculate_bootstrap_ci(large_values, n_bootstrap=1000)
        end_time = time.time()
        
        # Should complete in reasonable time (< 5 seconds)
        assert (end_time - start_time) < 5.0
        assert len(ci_result) == 2
        assert ci_result[0] < ci_result[1]
    
    @pytest.mark.parametrize("sample_size,expected_recommendation", [
        (10, "hedges_g"),
        (30, "hedges_g"), 
        (60, "cohens_d"),
        (100, "cohens_d")
    ])
    def test_effect_size_recommendation_by_sample_size(self, analyzer, sample_size, expected_recommendation):
        """Test that effect size recommendations change based on sample size"""
        np.random.seed(42)
        control = np.random.normal(50, 10, sample_size).tolist()
        treatment = np.random.normal(52, 10, sample_size).tolist()
        
        effect_sizes = analyzer._calculate_effect_sizes(control, treatment)
        assert effect_sizes["recommended_measure"] == expected_recommendation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])