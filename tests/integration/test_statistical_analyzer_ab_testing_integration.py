"""
Integration Tests for Statistical Analyzer and A/B Testing Framework

Tests the integration between the Statistical Analyzer and A/B Testing Framework to ensure
they work together correctly for comprehensive experiment analysis and validation.

Follows integration testing best practices:
- Database integration testing with realistic data
- Cross-component workflow validation  
- End-to-end statistical analysis pipelines
- Performance validation under realistic loads
- Error handling and resilience testing
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from scipy import stats
import warnings

from src.prompt_improver.evaluation.statistical_analyzer import (
    StatisticalAnalyzer, 
    StatisticalConfig,
    DescriptiveStats
)
from src.prompt_improver.services.ab_testing import (
    ABTestingService,
    ExperimentResult
)


@pytest.mark.integration
class TestStatisticalAnalyzerABTestingIntegration:
    """Integration test suite for Statistical Analyzer and A/B Testing Framework"""
    
    @pytest.fixture(scope="class")
    def statistical_analyzer(self):
        """Create a statistical analyzer instance for integration testing"""
        config = StatisticalConfig(
            significance_level=0.05,
            confidence_level=0.95,
            minimum_sample_size=30,
            recommended_sample_size=100
        )
        return StatisticalAnalyzer(config)
    
    @pytest.fixture(scope="class") 
    def ab_testing_service(self):
        """Create an A/B testing service instance for integration testing"""
        return ABTestingService()
    
    @pytest.fixture(scope="function")
    def random_seed(self):
        """Set reproducible random seed for each test"""
        np.random.seed(42)
        return 42
    
    @pytest.fixture
    def experiment_data_factory(self, random_seed):
        """Factory fixture for generating realistic experiment data"""
        def _create_experiment_data(n_control=100, n_treatment=100, effect_size=0.3, metric_type='improvement_score'):
            """Create realistic A/B experiment data with statistical properties"""
            
            # Generate baseline control data
            if metric_type == 'improvement_score':
                # Improvement scores (0-1 scale, higher is better)
                control_scores = np.random.beta(2, 3, n_control) * 0.8 + 0.1  # 0.1-0.9 range
                treatment_scores = np.random.beta(2, 3, n_treatment) * 0.8 + 0.1 + effect_size
                treatment_scores = np.clip(treatment_scores, 0, 1)
            elif metric_type == 'response_time_ms':
                # Response times (ms, lower is better)
                control_scores = np.random.lognormal(np.log(200), 0.3, n_control)
                treatment_scores = np.random.lognormal(np.log(200 * (1 - effect_size)), 0.3, n_treatment)
            else:
                # Generic continuous metric
                control_scores = np.random.normal(0.5, 0.15, n_control)
                treatment_scores = np.random.normal(0.5 + effect_size, 0.15, n_treatment)
            
            # Create detailed test results for statistical analysis
            control_results = []
            for i, score in enumerate(control_scores):
                result = {
                    "overallScore": float(score),
                    "clarity": float(score + np.random.normal(0, 0.05)),
                    "completeness": float(score + np.random.normal(0, 0.05)), 
                    "actionability": float(score + np.random.normal(0, 0.05)),
                    "effectiveness": float(score + np.random.normal(0, 0.05)),
                    "strategy": "control_strategy",
                    "model": "baseline_model",
                    "complexity": np.random.choice(["low", "medium", "high"]),
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                    "experiment_group": "control",
                    "participant_id": f"control_{i:03d}"
                }
                control_results.append(result)
            
            treatment_results = []
            for i, score in enumerate(treatment_scores):
                result = {
                    "overallScore": float(score),
                    "clarity": float(score + np.random.normal(0, 0.05)),
                    "completeness": float(score + np.random.normal(0, 0.05)),
                    "actionability": float(score + np.random.normal(0, 0.05)), 
                    "effectiveness": float(score + np.random.normal(0, 0.05)),
                    "strategy": "treatment_strategy",
                    "model": "optimized_model",
                    "complexity": np.random.choice(["low", "medium", "high"]),
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                    "experiment_group": "treatment",
                    "participant_id": f"treatment_{i:03d}"
                }
                treatment_results.append(result)
            
            # Create A/B testing compatible data
            ab_control_data = list(control_scores)
            ab_treatment_data = list(treatment_scores)
            
            return {
                'control_results': control_results,
                'treatment_results': treatment_results,
                'ab_control_data': ab_control_data,
                'ab_treatment_data': ab_treatment_data,
                'true_effect_size': effect_size,
                'metric_type': metric_type
            }
        return _create_experiment_data
    
    @pytest.mark.asyncio
    async def test_end_to_end_experiment_analysis_workflow(self, statistical_analyzer, ab_testing_service, experiment_data_factory):
        """Test complete end-to-end experiment analysis workflow"""
        # Generate experiment data with meaningful effect
        experiment_data = experiment_data_factory(
            n_control=150,
            n_treatment=150, 
            effect_size=0.25,
            metric_type='improvement_score'
        )
        
        # Step 1: Perform statistical analysis on the results
        all_results = experiment_data['control_results'] + experiment_data['treatment_results']
        statistical_analysis = await statistical_analyzer.perform_statistical_analysis(all_results)
        
        # Verify statistical analysis results
        assert "descriptive_stats" in statistical_analysis
        assert "confidence_intervals" in statistical_analysis
        assert "correlation_analysis" in statistical_analysis
        
        # Step 2: Perform A/B testing analysis
        ab_analysis = ab_testing_service._perform_statistical_analysis(
            experiment_data['ab_control_data'],
            experiment_data['ab_treatment_data']
        )
        
        # Verify A/B testing results
        assert isinstance(ab_analysis, ExperimentResult)
        assert ab_analysis.statistical_significance
        assert ab_analysis.practical_significance
        assert ab_analysis.effect_size > 0.15  # Should detect meaningful effect
        
        # Step 3: Cross-validate results between analyzers
        # Statistical analyzer overall scores
        control_overall = [r["overallScore"] for r in experiment_data['control_results']]
        treatment_overall = [r["overallScore"] for r in experiment_data['treatment_results']]
        
        # Compare means (should be consistent)
        stat_control_mean = np.mean(control_overall)
        stat_treatment_mean = np.mean(treatment_overall)
        
        assert abs(ab_analysis.control_mean - stat_control_mean) < 0.01, "Control means should match"
        assert abs(ab_analysis.treatment_mean - stat_treatment_mean) < 0.01, "Treatment means should match"
        
        # Step 4: Verify bootstrap CIs are calculated by statistical analyzer
        ci_data = statistical_analysis["confidence_intervals"]
        if "overall_scores" in ci_data:
            assert "bootstrap_lower" in ci_data["overall_scores"]
            assert "bootstrap_upper" in ci_data["overall_scores"]
            assert "bootstrap_method" in ci_data["overall_scores"]
            assert ci_data["overall_scores"]["bootstrap_method"] == "BCa"
        
        # Step 5: Verify CUPED analysis if available
        if hasattr(ab_testing_service, '_apply_cuped_analysis'):
            # Create mock pre-experiment data
            control_data_cuped = {
                'outcome': experiment_data['ab_control_data'],
                'pre_value': np.random.normal(0.45, 0.1, len(experiment_data['ab_control_data']))
            }
            treatment_data_cuped = {
                'outcome': experiment_data['ab_treatment_data'],
                'pre_value': np.random.normal(0.45, 0.1, len(experiment_data['ab_treatment_data']))
            }
            
            cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data_cuped, control_data_cuped)
            if cuped_results:  # If CUPED analysis succeeded
                assert 'variance_reduction_percent' in cuped_results
                assert 'treatment_effect_cuped' in cuped_results
    
    @pytest.mark.asyncio
    async def test_statistical_analyzer_bootstrap_ci_integration(self, statistical_analyzer, experiment_data_factory):
        """Test Statistical Analyzer bootstrap confidence interval integration"""
        # Generate data for bootstrap CI testing
        experiment_data = experiment_data_factory(n_control=80, n_treatment=80, effect_size=0.3)
        
        # Perform statistical analysis
        all_results = experiment_data['control_results'] + experiment_data['treatment_results']
        analysis = await statistical_analyzer.perform_statistical_analysis(all_results)
        
        # Verify bootstrap CIs are calculated
        ci_data = analysis["confidence_intervals"]
        
        # Check for bootstrap confidence intervals in key metrics
        expected_metrics = ["overall_scores", "clarity_scores", "completeness_scores"]
        bootstrap_methods_found = 0
        
        for metric in expected_metrics:
            if metric in ci_data:
                metric_ci = ci_data[metric]
                if "bootstrap_lower" in metric_ci and "bootstrap_upper" in metric_ci:
                    bootstrap_methods_found += 1
                    
                    # Verify bootstrap CI properties
                    assert metric_ci["bootstrap_lower"] < metric_ci["bootstrap_upper"]
                    assert "bootstrap_method" in metric_ci
                    assert metric_ci["bootstrap_method"] in ["BCa", "percentile"]
                    
                    # Compare with traditional CI
                    if "lower" in metric_ci and "upper" in metric_ci:
                        traditional_width = metric_ci["upper"] - metric_ci["lower"]
                        bootstrap_width = metric_ci["bootstrap_upper"] - metric_ci["bootstrap_lower"]
                        
                        # Bootstrap CIs should be reasonably similar to traditional CIs
                        width_ratio = bootstrap_width / traditional_width if traditional_width > 0 else 1
                        assert 0.7 <= width_ratio <= 1.3, f"Bootstrap CI width ratio {width_ratio} outside reasonable bounds"
        
        assert bootstrap_methods_found > 0, "Should find bootstrap CIs for at least one metric"
    
    @pytest.mark.asyncio
    async def test_effect_size_calculation_consistency(self, statistical_analyzer, ab_testing_service, experiment_data_factory):
        """Test effect size calculation consistency between components"""
        # Generate data with known effect size
        true_effect_size = 0.4
        experiment_data = experiment_data_factory(
            n_control=100,
            n_treatment=100,
            effect_size=true_effect_size
        )
        
        # Get effect sizes from both analyzers
        ab_analysis = ab_testing_service._perform_statistical_analysis(
            experiment_data['ab_control_data'],
            experiment_data['ab_treatment_data']
        )
        
        # Calculate effect size manually for comparison
        control_mean = np.mean(experiment_data['ab_control_data'])
        treatment_mean = np.mean(experiment_data['ab_treatment_data'])
        pooled_std = np.sqrt((np.var(experiment_data['ab_control_data'], ddof=1) + 
                             np.var(experiment_data['ab_treatment_data'], ddof=1)) / 2)
        manual_effect_size = (treatment_mean - control_mean) / pooled_std
        
        # Verify effect size calculations are consistent
        assert abs(ab_analysis.effect_size - manual_effect_size) < 0.01, "A/B testing effect size should match manual calculation"
        
        # Verify effect size is reasonable given true effect size
        assert abs(ab_analysis.effect_size - true_effect_size) < 0.2, "Effect size should be close to true effect size"
        
        # Test Statistical Analyzer effect size calculation if available
        all_results = experiment_data['control_results'] + experiment_data['treatment_results']
        statistical_analysis = await statistical_analyzer.perform_statistical_analysis(all_results)
        
        # Check if effect sizes are reported in statistical analysis
        if "effect_sizes" in statistical_analysis:
            stat_effect_sizes = statistical_analysis["effect_sizes"]
            if "cohens_d" in stat_effect_sizes:
                cohens_d = stat_effect_sizes["cohens_d"]
                assert abs(cohens_d - ab_analysis.effect_size) < 0.1, "Cohen's d should be similar to A/B testing effect size"
    
    @pytest.mark.asyncio
    async def test_multiple_testing_correction_integration(self, statistical_analyzer, experiment_data_factory):
        """Test multiple testing correction integration in Statistical Analyzer"""
        # Generate multiple experiment results for multiple testing scenario
        all_results = []
        
        # Create 5 different experimental conditions
        for condition in range(5):
            experiment_data = experiment_data_factory(
                n_control=60,
                n_treatment=60, 
                effect_size=0.2 + condition * 0.1
            )
            
            # Add condition identifier to results
            for result in experiment_data['control_results']:
                result['condition'] = f'condition_{condition}'
            for result in experiment_data['treatment_results']:
                result['condition'] = f'condition_{condition}'
            
            all_results.extend(experiment_data['control_results'])
            all_results.extend(experiment_data['treatment_results'])
        
        # Perform statistical analysis
        analysis = await statistical_analyzer.perform_statistical_analysis(all_results)
        
        # Check for multiple testing correction if available
        if "multiple_testing_correction" in analysis:
            mtc_results = analysis["multiple_testing_correction"]
            
            # Verify FDR correction structure
            assert "method_used" in mtc_results
            assert "adjusted_p_values" in mtc_results
            assert "num_discoveries" in mtc_results
            
            # Verify adaptive FDR method was used
            assert mtc_results["method_used"] in ["fdr_by_adaptive", "fdr_bh_standard", "fdr_bh_fallback"]
            
            # Verify p-values are properly adjusted
            original_p = mtc_results.get("original_p_values", [])
            adjusted_p = mtc_results.get("adjusted_p_values", [])
            
            if original_p and adjusted_p:
                assert len(original_p) == len(adjusted_p)
                # Adjusted p-values should be >= original (except for very small corrections)
                for orig, adj in zip(original_p, adjusted_p):
                    assert adj >= orig - 0.001, "Adjusted p-values should be >= original"
    
    @pytest.mark.asyncio
    async def test_integration_performance_under_load(self, statistical_analyzer, ab_testing_service, experiment_data_factory):
        """Test integration performance under realistic load"""
        import time
        
        # Generate large-scale experiment data
        experiment_data = experiment_data_factory(
            n_control=500,
            n_treatment=500,
            effect_size=0.2
        )
        
        # Measure end-to-end analysis performance
        start_time = time.time()
        
        # Statistical analysis
        all_results = experiment_data['control_results'] + experiment_data['treatment_results']
        statistical_analysis = await statistical_analyzer.perform_statistical_analysis(all_results)
        
        # A/B testing analysis
        ab_analysis = ab_testing_service._perform_statistical_analysis(
            experiment_data['ab_control_data'],
            experiment_data['ab_treatment_data']
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance requirements
        assert total_time < 10.0, f"End-to-end analysis took {total_time:.2f}s, should be < 10s"
        
        # Verify results are still accurate under load
        assert ab_analysis.statistical_significance, "Should maintain statistical significance under load"
        assert "descriptive_stats" in statistical_analysis, "Should maintain statistical analysis quality under load"
        
        # Memory usage should be reasonable (basic check)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 1000, f"Memory usage {memory_mb:.1f}MB exceeds reasonable limit"
    
    @pytest.mark.asyncio 
    async def test_error_handling_and_resilience(self, statistical_analyzer, ab_testing_service, experiment_data_factory):
        """Test error handling and resilience in integration scenarios"""
        
        # Test 1: Handle missing data gracefully
        incomplete_experiment_data = experiment_data_factory(n_control=5, n_treatment=5, effect_size=0.1)
        
        # Statistical analyzer should handle small samples gracefully
        small_results = incomplete_experiment_data['control_results'] + incomplete_experiment_data['treatment_results']
        analysis = await statistical_analyzer.perform_statistical_analysis(small_results)
        
        assert "descriptive_stats" in analysis, "Should provide basic stats even with small samples"
        
        # A/B testing should indicate insufficient sample size
        ab_analysis = ab_testing_service._perform_statistical_analysis(
            incomplete_experiment_data['ab_control_data'],
            incomplete_experiment_data['ab_treatment_data']
        )
        
        # Should complete but may not have statistical power
        assert isinstance(ab_analysis, ExperimentResult)
        
        # Test 2: Handle corrupted data
        corrupted_data = experiment_data_factory(n_control=50, n_treatment=50, effect_size=0.3)
        
        # Introduce NaN values
        corrupted_results = corrupted_data['control_results'].copy()
        corrupted_results[0]["overallScore"] = float('nan')
        corrupted_results[1]["clarity"] = None
        
        # Should handle gracefully
        analysis_corrupted = await statistical_analyzer.perform_statistical_analysis(corrupted_results)
        assert "descriptive_stats" in analysis_corrupted, "Should handle corrupted data gracefully"
        
        # Test 3: Handle extreme values
        extreme_data = experiment_data_factory(n_control=30, n_treatment=30, effect_size=0.2)
        extreme_results = extreme_data['control_results'].copy()
        
        # Add extreme outliers
        extreme_results[0]["overallScore"] = 999.0
        extreme_results[1]["overallScore"] = -999.0
        
        analysis_extreme = await statistical_analyzer.perform_statistical_analysis(extreme_results)
        assert "descriptive_stats" in analysis_extreme, "Should handle extreme values"
    
    @pytest.mark.asyncio
    async def test_cross_component_data_consistency(self, statistical_analyzer, ab_testing_service, experiment_data_factory):
        """Test data consistency across components with different data formats"""
        
        # Generate consistent experiment data
        experiment_data = experiment_data_factory(n_control=100, n_treatment=100, effect_size=0.25)
        
        # Extract overall scores for comparison
        control_overall = [r["overallScore"] for r in experiment_data['control_results']]
        treatment_overall = [r["overallScore"] for r in experiment_data['treatment_results']]
        
        # Verify data consistency
        assert len(control_overall) == len(experiment_data['ab_control_data'])
        assert len(treatment_overall) == len(experiment_data['ab_treatment_data'])
        
        # Statistical properties should match
        np.testing.assert_allclose(control_overall, experiment_data['ab_control_data'], rtol=1e-10)
        np.testing.assert_allclose(treatment_overall, experiment_data['ab_treatment_data'], rtol=1e-10)
        
        # Perform both analyses
        all_results = experiment_data['control_results'] + experiment_data['treatment_results']
        statistical_analysis = await statistical_analyzer.perform_statistical_analysis(all_results)
        
        ab_analysis = ab_testing_service._perform_statistical_analysis(
            experiment_data['ab_control_data'],
            experiment_data['ab_treatment_data']
        )
        
        # Cross-verify descriptive statistics
        if "descriptive_stats" in statistical_analysis:
            overall_stats = statistical_analysis["descriptive_stats"].get("overall_scores")
            if overall_stats:
                # Check means are consistent
                combined_mean = (ab_analysis.control_mean * len(control_overall) + 
                               ab_analysis.treatment_mean * len(treatment_overall)) / (len(control_overall) + len(treatment_overall))
                
                stat_mean = overall_stats.get("mean")
                if stat_mean:
                    assert abs(combined_mean - stat_mean) < 0.01, "Combined mean should match statistical analysis mean"
        
        # Verify statistical significance consistency
        # Both should detect the same level of significance
        t_stat, p_val = stats.ttest_ind(treatment_overall, control_overall)
        
        assert ab_analysis.statistical_significance == (p_val < 0.05), "Statistical significance should be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])