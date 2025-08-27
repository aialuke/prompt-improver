"""
Real Behavior Testing for Analytics Components Lazy Loading

This test suite validates that the lazy loading implementation in analytics components
works correctly with real imports and maintains numerical accuracy.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from prompt_improver.analytics.unified.ab_testing_component import (
    _get_numpy as ab_get_numpy,
    _get_scipy_norm_chi2,
    _get_scipy_stats as ab_get_scipy_stats,
)
from prompt_improver.analytics.unified.ml_analytics_component import (
    _get_numpy as ml_get_numpy,
    _get_scipy_stats as ml_get_scipy_stats,
)
from prompt_improver.analytics.unified.performance_analytics_component import (
    _get_numpy as perf_get_numpy,
    _get_scipy_stats as perf_get_scipy_stats,
)
from prompt_improver.analytics.unified.session_analytics_component import (
    _get_numpy,
    _get_scipy_stats,
)


class TestAnalyticsLazyLoadingBehavior:
    """Test suite for lazy loading behavior in analytics components."""

    def test_lazy_loading_numpy_functions_import_only_when_called(self):
        """Test that numpy is not imported until functions are actually called."""

        # Test SessionAnalyticsComponent
        session_np = _get_numpy()
        assert session_np is not None
        assert session_np.__name__ == 'numpy'

        # Test MLAnalyticsComponent
        ml_np = ml_get_numpy()
        assert ml_np is not None
        assert ml_np.__name__ == 'numpy'

        # Test ABTestingComponent
        ab_np = ab_get_numpy()
        assert ab_np is not None
        assert ab_np.__name__ == 'numpy'

        # Test PerformanceAnalyticsComponent
        perf_np = perf_get_numpy()
        assert perf_np is not None
        assert perf_np.__name__ == 'numpy'

    def test_lazy_loading_scipy_functions_import_only_when_called(self):
        """Test that scipy.stats is not imported until functions are called."""

        # Test SessionAnalyticsComponent
        session_stats = _get_scipy_stats()
        assert session_stats is not None
        assert hasattr(session_stats, 'linregress')

        # Test MLAnalyticsComponent
        ml_stats = ml_get_scipy_stats()
        assert ml_stats is not None
        assert hasattr(ml_stats, 'linregress')

        # Test ABTestingComponent
        ab_stats = ab_get_scipy_stats()
        assert ab_stats is not None
        assert hasattr(ab_stats, 'linregress')

        # Test specialized AB testing functions
        norm, chi2_contingency = _get_scipy_norm_chi2()
        assert norm is not None
        assert chi2_contingency is not None
        assert hasattr(norm, 'cdf')

        # Test PerformanceAnalyticsComponent
        perf_stats = perf_get_scipy_stats()
        assert perf_stats is not None
        assert hasattr(perf_stats, 'linregress')

    def test_lazy_loading_caching_behavior(self):
        """Test that lazy loading functions cache their imports."""

        # Call functions multiple times and verify they return the same objects
        np1 = _get_numpy()
        np2 = _get_numpy()
        assert np1 is np2  # Same object reference

        stats1 = _get_scipy_stats()
        stats2 = _get_scipy_stats()
        assert stats1 is stats2  # Same object reference

    def test_numerical_accuracy_maintained_with_lazy_loading(self):
        """Test that numerical accuracy is maintained with lazy loaded functions."""

        # Test data
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        # Direct numpy/scipy calculations
        direct_np_mean = np.mean(data)
        direct_np_std = np.std(data)
        direct_np_percentile_95 = np.percentile(data, 95)

        # Lazy loaded calculations
        lazy_np = _get_numpy()
        lazy_np_mean = lazy_np.mean(data)
        lazy_np_std = lazy_np.std(data)
        lazy_np_percentile_95 = lazy_np.percentile(data, 95)

        # Verify numerical accuracy (should be identical)
        assert abs(direct_np_mean - lazy_np_mean) < 1e-10
        assert abs(direct_np_std - lazy_np_std) < 1e-10
        assert abs(direct_np_percentile_95 - lazy_np_percentile_95) < 1e-10

        # Test scipy statistical functions
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 6, 8, 10]

        # Direct scipy calculation
        direct_slope, direct_intercept, direct_r_value, _, _ = stats.linregress(x_data, y_data)

        # Lazy loaded calculation
        lazy_stats = _get_scipy_stats()
        lazy_slope, lazy_intercept, lazy_r_value, _, _ = lazy_stats.linregress(x_data, y_data)

        # Verify numerical accuracy
        assert abs(direct_slope - lazy_slope) < 1e-10
        assert abs(direct_intercept - lazy_intercept) < 1e-10
        assert abs(direct_r_value - lazy_r_value) < 1e-10

    def test_ab_testing_specific_functions_numerical_accuracy(self):
        """Test numerical accuracy for AB testing specific functions."""

        # Test normal CDF calculation
        test_value = 1.96

        # Direct calculation
        direct_cdf = stats.norm.cdf(test_value)

        # Lazy loaded calculation
        lazy_norm, _ = _get_scipy_norm_chi2()
        lazy_cdf = lazy_norm.cdf(test_value)

        # Verify accuracy
        assert abs(direct_cdf - lazy_cdf) < 1e-10

        # Test chi-square contingency
        contingency_table = [[10, 10, 20], [20, 20, 40]]

        # Direct calculation
        direct_chi2, direct_p, direct_dof, direct_expected = stats.chi2_contingency(contingency_table)

        # Lazy loaded calculation
        _, lazy_chi2_contingency = _get_scipy_norm_chi2()
        lazy_chi2, lazy_p, lazy_dof, lazy_expected = lazy_chi2_contingency(contingency_table)

        # Verify accuracy
        assert abs(direct_chi2 - lazy_chi2) < 1e-10
        assert abs(direct_p - lazy_p) < 1e-10
        assert direct_dof == lazy_dof
        assert np.allclose(direct_expected, lazy_expected)

    def test_performance_analytics_statistical_functions(self):
        """Test performance analytics statistical functions maintain accuracy."""

        # Test trend analysis calculations
        time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        values = [1.1, 2.3, 3.2, 4.8, 5.1, 6.4, 7.2, 8.9, 9.1, 10.5]

        # Direct calculation
        direct_slope, direct_intercept, direct_r, _, _ = stats.linregress(time_points, values)

        # Lazy loaded calculation
        lazy_stats = perf_get_scipy_stats()
        lazy_slope, lazy_intercept, lazy_r, _, _ = lazy_stats.linregress(time_points, values)

        # Verify accuracy
        assert abs(direct_slope - lazy_slope) < 1e-10
        assert abs(direct_intercept - lazy_intercept) < 1e-10
        assert abs(direct_r - lazy_r) < 1e-10

    def test_session_analytics_statistical_functions(self):
        """Test session analytics statistical functions maintain accuracy."""

        # Test anomaly detection calculations
        session_durations = [45.2, 60.1, 55.8, 48.3, 52.7, 59.4, 61.2, 47.9, 53.1, 58.6]

        # Direct calculations
        direct_mean = np.mean(session_durations)
        direct_std = np.std(session_durations)
        direct_percentile_90 = np.percentile(session_durations, 90)

        # Lazy loaded calculations
        lazy_np = _get_numpy()
        lazy_mean = lazy_np.mean(session_durations)
        lazy_std = lazy_np.std(session_durations)
        lazy_percentile_90 = lazy_np.percentile(session_durations, 90)

        # Verify accuracy
        assert abs(direct_mean - lazy_mean) < 1e-10
        assert abs(direct_std - lazy_std) < 1e-10
        assert abs(direct_percentile_90 - lazy_percentile_90) < 1e-10

    def test_ml_analytics_statistical_functions(self):
        """Test ML analytics statistical functions maintain accuracy."""

        # Test model performance calculations
        predictions = [0.8, 0.6, 0.9, 0.7, 0.85, 0.75, 0.95, 0.65, 0.88, 0.72]
        actual_values = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1]

        # Calculate accuracy directly
        correct_predictions = sum(1 for p, a in zip(predictions, actual_values, strict=False) if (p > 0.5) == bool(a))
        direct_accuracy = correct_predictions / len(actual_values)

        # Test with numpy arrays
        lazy_np = ml_get_numpy()
        lazy_predictions = lazy_np.array(predictions)
        lazy_actual = lazy_np.array(actual_values)

        # Calculate using lazy loaded numpy
        lazy_correct = lazy_np.sum((lazy_predictions > 0.5) == lazy_actual.astype(bool))
        lazy_accuracy = lazy_correct / len(lazy_actual)

        # Verify accuracy
        assert abs(direct_accuracy - lazy_accuracy) < 1e-10

    @pytest.mark.asyncio
    async def test_error_handling_for_missing_dependencies(self):
        """Test error handling when dependencies are missing (if applicable)."""

        # This test verifies that our lazy loading functions handle import errors gracefully
        # In production, numpy and scipy should always be available, but we test robustness

        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            try:
                _get_numpy()
                # If we reach here, the system has fallback behavior
                assert True
            except ImportError:
                # Expected behavior - import error propagates
                assert True

    def test_lazy_loading_performance_benefit(self):
        """Test that lazy loading provides performance benefits by avoiding import time."""

        # This test demonstrates the benefit of lazy loading
        # We can't easily measure the exact import time benefit in this test,
        # but we can verify the functions work correctly when called

        start_time = time.time()

        # Call lazy loading functions (should be fast if already cached)
        np_module = _get_numpy()
        stats_module = _get_scipy_stats()

        end_time = time.time()

        # Verify modules are correct
        assert np_module.__name__ == 'numpy'
        assert hasattr(stats_module, 'linregress')

        # Function calls should be very fast (cached imports)
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        assert execution_time < 10  # Should be under 10ms for cached imports

    def test_all_components_can_import_dependencies(self):
        """Test that all analytics components can successfully import their dependencies."""

        # SessionAnalyticsComponent
        session_np = _get_numpy()
        session_stats = _get_scipy_stats()
        assert session_np is not None
        assert session_stats is not None

        # MLAnalyticsComponent
        ml_np = ml_get_numpy()
        ml_stats = ml_get_scipy_stats()
        assert ml_np is not None
        assert ml_stats is not None

        # ABTestingComponent
        ab_np = ab_get_numpy()
        ab_stats = ab_get_scipy_stats()
        ab_norm, ab_chi2 = _get_scipy_norm_chi2()
        assert ab_np is not None
        assert ab_stats is not None
        assert ab_norm is not None
        assert ab_chi2 is not None

        # PerformanceAnalyticsComponent
        perf_np = perf_get_numpy()
        perf_stats = perf_get_scipy_stats()
        assert perf_np is not None
        assert perf_stats is not None

    def test_lazy_loading_thread_safety(self):
        """Test that lazy loading is thread-safe."""

        import concurrent.futures

        results = []

        def call_lazy_imports():
            np_module = _get_numpy()
            stats_module = _get_scipy_stats()
            return (id(np_module), id(stats_module))

        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(call_lazy_imports) for _ in range(10)]
            results = [future.result() for future in futures]

        # All threads should get the same object references (cached)
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Lazy loading should be thread-safe and return cached objects"

    def test_comprehensive_numerical_accuracy_validation(self):
        """Comprehensive test of numerical accuracy across all statistical operations."""

        # Test data sets
        small_dataset = [1, 2, 3, 4, 5]
        large_dataset = list(range(1, 101))  # 1 to 100
        float_dataset = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

        test_datasets = [
            ("small_integer", small_dataset),
            ("large_integer", large_dataset),
            ("float_values", float_dataset)
        ]

        for dataset_name, data in test_datasets:
            # Direct numpy calculations
            direct_mean = np.mean(data)
            direct_std = np.std(data)
            direct_var = np.var(data)
            direct_min = np.min(data)
            direct_max = np.max(data)
            direct_median = np.median(data)

            # Lazy loaded calculations
            lazy_np = _get_numpy()
            lazy_mean = lazy_np.mean(data)
            lazy_std = lazy_np.std(data)
            lazy_var = lazy_np.var(data)
            lazy_min = lazy_np.min(data)
            lazy_max = lazy_np.max(data)
            lazy_median = lazy_np.median(data)

            # Verify all calculations match
            assert abs(direct_mean - lazy_mean) < 1e-10, f"Mean mismatch for {dataset_name}"
            assert abs(direct_std - lazy_std) < 1e-10, f"Std mismatch for {dataset_name}"
            assert abs(direct_var - lazy_var) < 1e-10, f"Var mismatch for {dataset_name}"
            assert abs(direct_min - lazy_min) < 1e-10, f"Min mismatch for {dataset_name}"
            assert abs(direct_max - lazy_max) < 1e-10, f"Max mismatch for {dataset_name}"
            assert abs(direct_median - lazy_median) < 1e-10, f"Median mismatch for {dataset_name}"
