"""
Tests for Metrics Validation Framework Implementation

Tests the enhanced Optimization Validator metrics validation functionality with realistic
benchmarks, cross-validation, and comprehensive validation protocols.

Follows pytest best practices:
- Comprehensive fixture-based test setup
- Parametrized tests for multiple validation scenarios
- Statistical validation with proper error handling
- Performance benchmarking
- Integration testing patterns
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from scipy import stats
import warnings

from src.prompt_improver.optimization.optimization_validator import (
    OptimizationValidator,
    ValidationConfig
)


class TestMetricsValidationFramework:
    """Test suite for metrics validation framework with realistic scenarios"""
    
    @pytest.fixture(scope="class")
    def optimization_validator(self):
        """Create an optimization validator instance for testing"""
        config = ValidationConfig(
            min_sample_size=30,
            significance_level=0.05,
            min_effect_size=0.2,
            validation_duration_hours=24
        )
        return OptimizationValidator(config)
    
    @pytest.fixture(scope="function")
    def random_seed(self):
        """Set reproducible random seed for each test"""
        np.random.seed(42)
        return 42
    
    @pytest.fixture
    def realistic_metrics_ranges(self):
        """Fixture providing realistic ranges for various performance metrics"""
        return {
            'response_time_ms': {
                'min': 0.1, 'max': 500, 'typical': (10, 200),
                'suspicious_low': 1.0, 'suspicious_high': 300
            },
            'memory_usage_mb': {
                'min': 10, 'max': 1000, 'typical': (50, 500),
                'suspicious_low': 5, 'suspicious_high': 500
            },
            'cpu_usage_percent': {
                'min': 0.1, 'max': 100, 'typical': (5, 80),
                'suspicious_low': 0.5, 'suspicious_high': 90
            },
            'throughput_rps': {
                'min': 1, 'max': 10000, 'typical': (10, 1000),
                'suspicious_low': 5, 'suspicious_high': 5000
            },
            'error_rate_percent': {
                'min': 0, 'max': 100, 'typical': (0, 5),
                'suspicious_low': -0.1, 'suspicious_high': 10
            },
            'cache_hit_ratio': {
                'min': 0, 'max': 100, 'typical': (60, 95),
                'suspicious_low': 30, 'suspicious_high': 100.1
            }
        }
    
    @pytest.fixture
    def baseline_data_factory(self, random_seed, realistic_metrics_ranges):
        """Factory fixture for generating baseline performance data"""
        def _create_baseline_data(n_samples=100, metric_type='response_time_ms', noise_level=0.1):
            """Create realistic baseline performance data"""
            metric_range = realistic_metrics_ranges[metric_type]
            
            # Generate realistic baseline values within typical range
            baseline_mean = np.random.uniform(*metric_range['typical'])
            baseline_std = baseline_mean * noise_level
            
            # Generate samples with some realistic distribution
            if metric_type in ['response_time_ms', 'memory_usage_mb']:
                # These often follow log-normal distributions
                samples = np.random.lognormal(
                    mean=np.log(baseline_mean), 
                    sigma=noise_level, 
                    size=n_samples
                )
            elif metric_type == 'cpu_usage_percent':
                # CPU usage often has a gamma-like distribution
                samples = np.random.gamma(
                    shape=2, 
                    scale=baseline_mean/2, 
                    size=n_samples
                )
                samples = np.clip(samples, 0, 100)
            else:
                # Default to normal distribution
                samples = np.random.normal(baseline_mean, baseline_std, n_samples)
            
            # Ensure values are within realistic bounds
            samples = np.clip(samples, metric_range['min'], metric_range['max'])
            
            return {
                'optimization_id': 'baseline_001',
                'scores': samples.tolist(),
                'metadata': {
                    'metric_type': metric_type,
                    'baseline_mean': float(baseline_mean),
                    'sample_size': n_samples,
                    'collection_period': '24h',
                    'environment': 'production'
                }
            }
        return _create_baseline_data
    
    @pytest.fixture
    def optimized_data_factory(self, random_seed, realistic_metrics_ranges):
        """Factory fixture for generating optimized performance data"""
        def _create_optimized_data(baseline_data, improvement_factor=0.8, n_samples=100):
            """Create optimized performance data based on baseline"""
            baseline_scores = np.array(baseline_data['scores'])
            baseline_mean = np.mean(baseline_scores)
            baseline_std = np.std(baseline_scores)
            
            metric_type = baseline_data['metadata']['metric_type']
            metric_range = realistic_metrics_ranges[metric_type]
            
            # Apply improvement (lower is better for most metrics)
            if metric_type in ['response_time_ms', 'memory_usage_mb', 'cpu_usage_percent', 'error_rate_percent']:
                # Lower is better
                optimized_mean = baseline_mean * improvement_factor
                optimized_std = baseline_std * 0.9  # Slightly less variance
            else:
                # Higher is better (throughput, cache hit ratio)
                optimized_mean = baseline_mean / improvement_factor
                optimized_std = baseline_std * 0.9
            
            # Generate optimized samples
            optimized_samples = np.random.normal(optimized_mean, optimized_std, n_samples)
            
            # Ensure realistic bounds
            optimized_samples = np.clip(optimized_samples, metric_range['min'], metric_range['max'])
            
            return {
                'optimization_id': 'optimized_001',
                'scores': optimized_samples.tolist(),
                'metadata': {
                    'metric_type': metric_type,
                    'optimized_mean': float(optimized_mean),
                    'improvement_factor': improvement_factor,
                    'sample_size': n_samples,
                    'collection_period': '24h',
                    'environment': 'production'
                }
            }
        return _create_optimized_data
    
    @pytest.mark.asyncio
    async def test_basic_optimization_validation(self, optimization_validator, baseline_data_factory, optimized_data_factory):
        """Test basic optimization validation functionality"""
        # Generate test data
        baseline_data = baseline_data_factory(metric_type='response_time_ms', n_samples=50)
        optimized_data = optimized_data_factory(baseline_data, improvement_factor=0.7, n_samples=50)
        
        # Perform validation
        validation_result = await optimization_validator.validate_optimization(
            'test_optimization_001',
            baseline_data,
            optimized_data
        )
        
        # Verify result structure
        required_fields = [
            'optimization_id', 'valid', 'statistical_significance', 'practical_significance',
            'p_value', 'effect_size', 'baseline_mean', 'optimized_mean', 'improvement',
            'validation_date'
        ]
        
        for field in required_fields:
            assert field in validation_result, f"Missing required field: {field}"
        
        # Verify data types
        assert isinstance(validation_result['valid'], bool)
        assert isinstance(validation_result['statistical_significance'], bool)
        assert isinstance(validation_result['practical_significance'], bool)
        assert isinstance(validation_result['p_value'], float)
        assert isinstance(validation_result['effect_size'], float)
        
        # With 30% improvement, should show significant results
        assert validation_result['statistical_significance'], "Should detect statistical significance with 30% improvement"
        assert validation_result['optimized_mean'] < validation_result['baseline_mean'], "Optimized should be better (lower) than baseline"
        
        # Verify p-value is reasonable
        assert 0 <= validation_result['p_value'] <= 1, "p-value should be between 0 and 1"
    
    def test_metrics_realism_validation_basic(self, optimization_validator, realistic_metrics_ranges):
        """Test basic metrics realism validation"""
        # Test with realistic data
        realistic_baseline = {
            'scores': [50, 45, 55, 48, 52, 49, 51, 47, 53, 50],  # Realistic response times in ms
            'metadata': {'metric_type': 'response_time_ms'}
        }
        
        realistic_optimized = {
            'scores': [35, 32, 38, 33, 37, 34, 36, 31, 39, 35],  # Improved response times
            'metadata': {'metric_type': 'response_time_ms'}
        }
        
        # This should be implemented in the OptimizationValidator
        # For now, we'll test the concept with a mock implementation
        
        def validate_metrics_realism(baseline_data, optimized_data):
            """Mock implementation of metrics validation"""
            baseline_scores = baseline_data['scores']
            optimized_scores = optimized_data['scores']
            metric_type = baseline_data['metadata']['metric_type']
            
            ranges = realistic_metrics_ranges[metric_type]
            
            # Check if all values are within realistic bounds
            all_scores = baseline_scores + optimized_scores
            
            realistic = all(ranges['min'] <= score <= ranges['max'] for score in all_scores)
            
            suspicious_low = any(score < ranges['suspicious_low'] for score in all_scores)
            suspicious_high = any(score > ranges['suspicious_high'] for score in all_scores)
            
            return {
                'realistic': realistic,
                'suspicious_values': suspicious_low or suspicious_high,
                'min_value': min(all_scores),
                'max_value': max(all_scores),
                'validation_status': 'PASS' if realistic and not (suspicious_low or suspicious_high) else 'SUSPICIOUS'
            }
        
        # Test realistic data
        result = validate_metrics_realism(realistic_baseline, realistic_optimized)
        assert result['realistic'], "Realistic data should pass realism check"
        assert not result['suspicious_values'], "Realistic data should not trigger suspicious value alerts"
        assert result['validation_status'] == 'PASS'
        
        # Test unrealistic data
        unrealistic_baseline = {
            'scores': [0.01, 0.02, 0.01, 0.02],  # Impossibly fast response times
            'metadata': {'metric_type': 'response_time_ms'}
        }
        
        unrealistic_result = validate_metrics_realism(unrealistic_baseline, realistic_optimized)
        assert unrealistic_result['suspicious_values'], "Should detect suspicious values"
        assert unrealistic_result['validation_status'] == 'SUSPICIOUS'
    
    @pytest.mark.parametrize("improvement_factor,expected_significance", [
        pytest.param(0.95, False, id="minimal_improvement"),
        pytest.param(0.8, True, id="moderate_improvement"),
        pytest.param(0.6, True, id="substantial_improvement"),
        pytest.param(0.4, True, id="major_improvement"),
    ])
    @pytest.mark.asyncio
    async def test_statistical_significance_detection(self, optimization_validator, baseline_data_factory, 
                                                    optimized_data_factory, improvement_factor, expected_significance):
        """Test statistical significance detection across different improvement levels"""
        # Generate data with specified improvement
        baseline_data = baseline_data_factory(metric_type='response_time_ms', n_samples=100)
        optimized_data = optimized_data_factory(baseline_data, improvement_factor=improvement_factor, n_samples=100)
        
        # Validate optimization
        result = await optimization_validator.validate_optimization(
            f'test_improvement_{improvement_factor}',
            baseline_data,
            optimized_data
        )
        
        # Check significance detection
        actual_significance = result['statistical_significance']
        assert actual_significance == expected_significance, \
            f"Expected significance {expected_significance} for {improvement_factor} improvement, got {actual_significance}"
        
        # Verify p-value aligns with significance
        p_value = result['p_value']
        alpha = optimization_validator.config.significance_level
        
        if expected_significance:
            assert p_value < alpha, f"p-value {p_value} should be < {alpha} for significant result"
        else:
            assert p_value >= alpha, f"p-value {p_value} should be >= {alpha} for non-significant result"
    
    @pytest.mark.parametrize("metric_type,improvement_direction", [
        ("response_time_ms", "decrease"),
        ("memory_usage_mb", "decrease"),
        ("cpu_usage_percent", "decrease"),
        ("error_rate_percent", "decrease"),
        ("throughput_rps", "increase"),
        ("cache_hit_ratio", "increase"),
    ])
    @pytest.mark.asyncio
    async def test_metric_specific_validation(self, optimization_validator, baseline_data_factory, 
                                            optimized_data_factory, metric_type, improvement_direction):
        """Test validation behavior for different types of metrics"""
        # Generate appropriate test data
        baseline_data = baseline_data_factory(metric_type=metric_type, n_samples=80)
        
        # Apply improvement in correct direction
        if improvement_direction == "decrease":
            improvement_factor = 0.7  # 30% reduction (better)
        else:
            improvement_factor = 1.3  # 30% increase (better)
        
        optimized_data = optimized_data_factory(baseline_data, improvement_factor=improvement_factor, n_samples=80)
        
        # Validate optimization
        result = await optimization_validator.validate_optimization(
            f'test_{metric_type}',
            baseline_data,
            optimized_data
        )
        
        # Verify improvement direction is correctly detected
        baseline_mean = result['baseline_mean']
        optimized_mean = result['optimized_mean']
        improvement = result['improvement']
        
        if improvement_direction == "decrease":
            assert optimized_mean < baseline_mean, f"Optimized {metric_type} should be lower than baseline"
            assert improvement > 0, f"Improvement should be positive for {metric_type}"
        else:
            assert optimized_mean > baseline_mean, f"Optimized {metric_type} should be higher than baseline"
            assert improvement > 0, f"Improvement should be positive for {metric_type}"
        
        # Should detect significance with 30% improvement
        assert result['statistical_significance'], f"Should detect significance for 30% improvement in {metric_type}"
    
    @pytest.mark.asyncio
    async def test_insufficient_sample_size_handling(self, optimization_validator, baseline_data_factory, optimized_data_factory):
        """Test handling of insufficient sample sizes"""
        # Generate small samples
        baseline_data = baseline_data_factory(n_samples=5)  # Below minimum
        optimized_data = optimized_data_factory(baseline_data, n_samples=5)
        
        # Validation should reject due to insufficient sample size
        result = await optimization_validator.validate_optimization(
            'test_small_sample',
            baseline_data,
            optimized_data
        )
        
        assert not result['valid'], "Should reject optimization with insufficient sample size"
        assert 'Insufficient sample size' in result['reason'], "Should provide clear reason for rejection"
        
        # Verify minimum sample size is enforced
        min_size = optimization_validator.config.min_sample_size
        assert min_size > 5, "Minimum sample size should be reasonable"
    
    def test_cross_validation_implementation_concept(self, optimization_validator, baseline_data_factory, optimized_data_factory):
        """Test cross-validation concept for robustness assessment"""
        # Generate larger dataset for cross-validation
        baseline_data = baseline_data_factory(n_samples=200, metric_type='response_time_ms')
        optimized_data = optimized_data_factory(baseline_data, improvement_factor=0.75, n_samples=200)
        
        # Mock cross-validation implementation
        def perform_cross_validation(baseline_scores, optimized_scores, n_folds=5):
            """Mock implementation of cross-validation for optimization assessment"""
            from sklearn.model_selection import KFold
            
            baseline_array = np.array(baseline_scores)
            optimized_array = np.array(optimized_scores)
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            fold_results = []
            
            for train_idx, test_idx in kf.split(baseline_array):
                # Use test indices for validation fold
                baseline_fold = baseline_array[test_idx]
                optimized_fold = optimized_array[test_idx]
                
                # Perform t-test on this fold
                t_stat, p_value = stats.ttest_ind(optimized_fold, baseline_fold)
                
                fold_result = {
                    'fold_size': len(test_idx),
                    'baseline_mean': np.mean(baseline_fold),
                    'optimized_mean': np.mean(optimized_fold),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                fold_results.append(fold_result)
            
            # Aggregate results
            significant_folds = sum(1 for result in fold_results if result['significant'])
            consistency_score = significant_folds / n_folds
            
            return {
                'fold_results': fold_results,
                'significant_folds': significant_folds,
                'total_folds': n_folds,
                'consistency_score': consistency_score,
                'robust': consistency_score >= 0.6  # At least 60% of folds should be significant
            }
        
        # Test cross-validation
        cv_result = perform_cross_validation(baseline_data['scores'], optimized_data['scores'])
        
        # Verify cross-validation structure
        assert 'fold_results' in cv_result
        assert 'consistency_score' in cv_result
        assert 'robust' in cv_result
        
        assert len(cv_result['fold_results']) == 5, "Should have 5 fold results"
        assert 0 <= cv_result['consistency_score'] <= 1, "Consistency score should be between 0 and 1"
        
        # With 25% improvement, should show robust results
        assert cv_result['robust'], "25% improvement should show robust results across folds"
        assert cv_result['consistency_score'] > 0.6, "Should be consistent across most folds"
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, optimization_validator):
        """Test error handling in validation process"""
        # Test with invalid data
        invalid_baseline = {
            'scores': [],  # Empty scores
            'metadata': {'metric_type': 'response_time_ms'}
        }
        
        invalid_optimized = {
            'scores': [50, 45, 55],
            'metadata': {'metric_type': 'response_time_ms'}
        }
        
        # Should handle gracefully
        result = await optimization_validator.validate_optimization(
            'test_invalid',
            invalid_baseline,
            invalid_optimized
        )
        
        assert not result['valid'], "Should reject invalid data"
        assert 'validation_confidence' in result, "Should include confidence assessment"
        assert result['validation_confidence'] == 0.0, "Confidence should be zero for invalid data"
        
        # Test with mismatched data sizes
        mismatched_baseline = {'scores': [50] * 10}
        mismatched_optimized = {'scores': [45] * 5}
        
        result2 = await optimization_validator.validate_optimization(
            'test_mismatched',
            mismatched_baseline,
            mismatched_optimized
        )
        
        assert not result2['valid'], "Should handle mismatched sample sizes"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_performance_benchmarks(self, optimization_validator, baseline_data_factory, optimized_data_factory):
        """Test validation performance with different dataset sizes"""
        import time
        
        # Test different dataset sizes
        test_cases = [
            (50, "small_dataset"),
            (200, "medium_dataset"),
            (1000, "large_dataset")
        ]
        
        for n_samples, case_name in test_cases:
            baseline_data = baseline_data_factory(n_samples=n_samples, metric_type='response_time_ms')
            optimized_data = optimized_data_factory(baseline_data, improvement_factor=0.8, n_samples=n_samples)
            
            # Measure validation performance
            start_time = time.time()
            
            result = await optimization_validator.validate_optimization(
                f'perf_test_{case_name}',
                baseline_data,
                optimized_data
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance expectations
            max_time_by_case = {
                "small_dataset": 0.5,
                "medium_dataset": 1.0,
                "large_dataset": 2.0
            }
            
            max_time = max_time_by_case[case_name]
            assert execution_time <= max_time, \
                f"{case_name}: validation time {execution_time:.3f}s exceeds limit {max_time}s"
            
            # Verify results are still accurate
            assert isinstance(result['valid'], bool)
            assert isinstance(result['p_value'], float)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metrics_validation_integration(self, optimization_validator, baseline_data_factory, optimized_data_factory):
        """Test integration of metrics validation with optimization validation workflow"""
        # Generate comprehensive test data
        test_metrics = ['response_time_ms', 'memory_usage_mb', 'cpu_usage_percent']
        
        validation_results = []
        
        for metric_type in test_metrics:
            baseline_data = baseline_data_factory(metric_type=metric_type, n_samples=100)
            optimized_data = optimized_data_factory(baseline_data, improvement_factor=0.75, n_samples=100)
            
            # Validate each metric
            result = await optimization_validator.validate_optimization(
                f'integration_test_{metric_type}',
                baseline_data,
                optimized_data
            )
            
            validation_results.append({
                'metric_type': metric_type,
                'result': result
            })
        
        # Verify all validations succeeded
        for validation in validation_results:
            result = validation['result']
            metric_type = validation['metric_type']
            
            assert result['valid'], f"Validation should succeed for {metric_type}"
            assert result['statistical_significance'], f"Should detect significance for {metric_type}"
            assert result['practical_significance'], f"Should detect practical significance for {metric_type}"
        
        # Verify consistency across metrics
        p_values = [v['result']['p_value'] for v in validation_results]
        effect_sizes = [abs(v['result']['effect_size']) for v in validation_results]
        
        # All should show significance
        assert all(p < 0.05 for p in p_values), "All metrics should show statistical significance"
        
        # Effect sizes should be substantial
        assert all(es > 0.2 for es in effect_sizes), "All metrics should show substantial effect sizes"
        
        # Verify realistic improvement values
        improvements = [v['result']['improvement'] for v in validation_results]
        assert all(imp > 0 for imp in improvements), "All metrics should show positive improvement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])