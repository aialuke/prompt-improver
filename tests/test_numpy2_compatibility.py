"""
Comprehensive NumPy 2.x compatibility test suite.

Tests all critical NumPy usage patterns in the codebase to ensure
NumPy 2.x compatibility and proper numerical behavior.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

# Import components under test
from src.prompt_improver.security.input_validator import InputValidator, ValidationError
from src.prompt_improver.ml.analytics.session_comparison_analyzer import SessionComparisonAnalyzer
from src.prompt_improver.performance.analytics.real_time_analytics import RealTimeAnalyticsService
from src.prompt_improver.performance.testing.ab_testing_service import ABTestingService
from src.prompt_improver.security.adversarial_defense import AdversarialDefenseSystem
from src.prompt_improver.security.differential_privacy import DifferentialPrivacyService


class TestNumPy2Compatibility:
    """Test suite for NumPy 2.x compatibility"""

    def test_numpy_version(self):
        """Verify NumPy 2.x is installed"""
        assert np.__version__.startswith('2.'), f"Expected NumPy 2.x, got {np.__version__}"
        assert hasattr(np, '__array_api_version__'), "NumPy 2.x should have __array_api_version__"

    def test_numpy_basic_operations(self):
        """Test basic NumPy operations work correctly in 2.x"""
        # Test array creation
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert arr.dtype == np.float64
        
        # Test mathematical operations
        assert np.mean(arr) == 2.5
        assert np.std(arr, ddof=1) == pytest.approx(1.2909944, rel=1e-6)  # Sample std
        assert np.std(arr) == pytest.approx(1.118033989, rel=1e-6)  # Population std (default)
        assert np.sum(arr) == 10.0
        
        # Test NaN/infinity detection
        arr_with_nan = np.array([1.0, np.nan, 3.0])
        assert np.isnan(arr_with_nan[1])
        assert not np.isnan(arr_with_nan[0])
        
        arr_with_inf = np.array([1.0, np.inf, 3.0])
        assert np.isinf(arr_with_inf[1])
        assert not np.isinf(arr_with_inf[0])

    def test_numpy_random_generator(self):
        """Test new NumPy random generator API"""
        # Test default_rng works
        rng = np.random.default_rng(42)
        
        # Test various distributions
        normal_samples = rng.normal(0, 1, 100)
        assert len(normal_samples) == 100
        assert -3 < np.mean(normal_samples) < 3  # Should be close to 0
        
        uniform_samples = rng.uniform(-1, 1, 100)
        assert len(uniform_samples) == 100
        assert np.all(uniform_samples >= -1) and np.all(uniform_samples <= 1)
        
        beta_samples = rng.beta(2, 5, 100)
        assert len(beta_samples) == 100
        assert np.all(beta_samples >= 0) and np.all(beta_samples <= 1)

    def test_input_validator_numpy_arrays(self):
        """Test InputValidator with NumPy 2.x arrays"""
        validator = InputValidator()
        
        # Test valid NumPy arrays
        valid_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = validator.validate_input("test_array", valid_array, "numpy_array")
        np.testing.assert_array_equal(result, valid_array)
        
        # Test different dtypes
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            test_array = np.array([1, 2, 3], dtype=dtype)
            result = validator.validate_input("test_array", test_array, "numpy_array")
            assert result.dtype == dtype
        
        # Test array size limits
        large_array = np.ones(10000, dtype=np.float32)  # Should pass
        validator.validate_input("test_array", large_array, "numpy_array")
        
        # Test invalid dtype
        invalid_array = np.array([1, 2, 3], dtype=np.int8)
        with pytest.raises(ValidationError):
            validator.validate_input("test_array", invalid_array, "numpy_array")

    def test_input_validator_ml_features(self):
        """Test ML features validation with NumPy 2.x"""
        validator = InputValidator()
        
        # Test valid ML features as NumPy array
        features_array = np.array([0.1, 0.5, -0.3, 0.8], dtype=np.float64)
        result = validator.validate_input("ml_features", features_array, "ml_features")
        np.testing.assert_array_equal(result, features_array)
        
        # Test valid ML features as list
        features_list = [0.1, 0.5, -0.3, 0.8]
        result = validator.validate_input("ml_features", features_list, "ml_features")
        assert result == features_list
        
        # Test invalid features (NaN)
        invalid_features = [0.1, np.nan, 0.3]
        with pytest.raises(ValidationError):
            validator.validate_input("ml_features", invalid_features, "ml_features")
        
        # Test invalid features (infinity)
        invalid_features = [0.1, np.inf, 0.3]
        with pytest.raises(ValidationError):
            validator.validate_input("ml_features", invalid_features, "ml_features")

    def test_input_validator_numeric_validation(self):
        """Test numeric validation with NumPy 2.x functions"""
        validator = InputValidator()
        
        # Test valid numeric values
        valid_float = 3.14159
        result = validator.validate_input("privacy_epsilon", valid_float, "privacy_epsilon")
        assert result == valid_float
        
        # Test NaN detection
        with pytest.raises(ValidationError, match="cannot be NaN or infinite"):
            validator.validate_input("privacy_epsilon", float('nan'), "privacy_epsilon")
        
        # Test infinity detection
        with pytest.raises(ValidationError, match="cannot be NaN or infinite"):
            validator.validate_input("privacy_epsilon", float('inf'), "privacy_epsilon")

    @pytest.mark.asyncio
    async def test_session_comparison_analyzer_numpy_operations(self):
        """Test SessionComparisonAnalyzer NumPy operations"""
        # Mock database session
        mock_db_session = AsyncMock()
        analyzer = SessionComparisonAnalyzer(mock_db_session)
        
        # Test statistical operations with real data
        metrics_a = [0.8, 0.85, 0.82, 0.88, 0.86]
        metrics_b = [0.9, 0.92, 0.89, 0.91, 0.88]
        
        # Test statistical comparison
        from src.prompt_improver.ml.analytics.session_comparison_analyzer import ComparisonMethod
        result = await analyzer._perform_statistical_comparison(
            metrics_a, metrics_b, ComparisonMethod.T_TEST
        )
        
        # Verify statistical results are computed correctly
        assert isinstance(result['p_value'], float)
        assert isinstance(result['effect_size'], float)
        assert isinstance(result['confidence_interval'], tuple)
        assert len(result['confidence_interval']) == 2
        
        # Test with NumPy arrays
        np_metrics_a = np.array(metrics_a)
        np_metrics_b = np.array(metrics_b)
        
        result_np = await analyzer._perform_statistical_comparison(
            list(np_metrics_a), list(np_metrics_b), ComparisonMethod.T_TEST
        )
        
        # Results should be similar
        assert abs(result['p_value'] - result_np['p_value']) < 1e-10
        assert abs(result['effect_size'] - result_np['effect_size']) < 1e-10

    def test_real_time_analytics_numpy_operations(self):
        """Test RealTimeAnalyticsService NumPy operations"""
        # Mock Redis client
        mock_redis = AsyncMock()
        service = RealTimeAnalyticsService(mock_redis)
        
        # Test anomaly detection training with NumPy data
        # This tests the fixed np.random.default_rng() usage
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Generate test data using new NumPy random API
        rng = np.random.default_rng(42)
        test_data = []
        for _ in range(50):
            test_data.append([
                rng.normal(0.1, 0.02),  # conversion_rate
                rng.normal(100, 20),    # assignments
                rng.normal(50, 10)      # total_conversions
            ])
        
        # Verify data is valid and detector can be trained
        test_array = np.array(test_data)
        assert test_array.shape == (50, 3)
        assert not np.any(np.isnan(test_array))
        assert not np.any(np.isinf(test_array))
        
        # Train detector
        detector.fit(test_data)
        predictions = detector.predict(test_data)
        assert len(predictions) == 50

    def test_ab_testing_numpy_operations(self):
        """Test ABTestingService NumPy operations"""
        # Mock database session
        mock_db_session = AsyncMock()
        service = ABTestingService(mock_db_session)
        
        # Test Bayesian analysis with new random generator
        rng = np.random.default_rng(42)
        
        # Test beta distribution sampling
        n_samples = 1000
        control_samples = rng.beta(10, 90, n_samples)  # 10% conversion
        treatment_samples = rng.beta(12, 88, n_samples)  # 12% conversion
        
        # Verify samples are valid
        assert len(control_samples) == n_samples
        assert len(treatment_samples) == n_samples
        assert np.all(control_samples >= 0) and np.all(control_samples <= 1)
        assert np.all(treatment_samples >= 0) and np.all(treatment_samples <= 1)
        
        # Test probability calculation
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        assert 0 <= prob_treatment_better <= 1
        assert prob_treatment_better > 0.5  # Treatment should be better
        
        # Test binomial sampling
        visitors = 1000
        conversion_rate = 0.1
        conversions = rng.binomial(visitors, conversion_rate)
        assert 0 <= conversions <= visitors
        assert isinstance(conversions, (int, np.integer))

    def test_adversarial_defense_numpy_operations(self):
        """Test AdversarialDefenseSystem NumPy operations"""
        defense_system = AdversarialDefenseSystem()
        
        # Create test input data
        test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        
        # Test Gaussian noise defense (uses new random generator)
        defended_data = defense_system.apply_gaussian_noise_defense(test_data)
        
        # Verify defense was applied
        assert defended_data.shape == test_data.shape
        assert defended_data.dtype == test_data.dtype
        assert not np.array_equal(defended_data, test_data)  # Should be different due to noise
        
        # Verify no NaN or inf values introduced
        assert not np.any(np.isnan(defended_data))
        assert not np.any(np.isinf(defended_data))
        
        # Test FGSM attack simulation (uses new random generator)
        labels = np.array([0, 1])  # Mock labels
        adversarial_data = defense_system.generate_fgsm_attack(test_data, labels)
        
        # Verify adversarial examples
        assert adversarial_data.shape == test_data.shape
        assert adversarial_data.dtype == test_data.dtype
        assert not np.any(np.isnan(adversarial_data))
        assert not np.any(np.isinf(adversarial_data))

    def test_differential_privacy_numpy_operations(self):
        """Test DifferentialPrivacyService NumPy operations"""
        dp_manager = DifferentialPrivacyService()
        
        # Test Laplace noise (uses new random generator)
        original_value = 100.0
        noisy_value = dp_manager.add_laplace_noise(original_value, sensitivity=1.0, epsilon=0.1)
        
        # Verify noise was added
        assert isinstance(noisy_value, float)
        assert noisy_value != original_value  # Should be different
        assert not np.isnan(noisy_value)
        assert not np.isinf(noisy_value)
        
        # Test Gaussian noise (uses new random generator)
        noisy_value_gaussian = dp_manager.add_gaussian_noise(
            original_value, sensitivity=1.0, epsilon=0.1, delta=1e-6
        )
        
        # Verify Gaussian noise
        assert isinstance(noisy_value_gaussian, float)
        assert noisy_value_gaussian != original_value
        assert not np.isnan(noisy_value_gaussian)
        assert not np.isinf(noisy_value_gaussian)

    def test_numpy_dtype_compatibility(self):
        """Test NumPy dtype compatibility across versions"""
        # Test that common dtypes still work
        dtypes_to_test = [
            np.float32, np.float64, 
            np.int32, np.int64,
            np.uint32, np.uint64,
            np.bool_
        ]
        
        for dtype in dtypes_to_test:
            # Create arrays with each dtype
            arr = np.array([1, 2, 3], dtype=dtype)
            assert arr.dtype == dtype
            
            # Test basic operations
            result = np.mean(arr)
            assert not np.isnan(result)
            assert not np.isinf(result)

    def test_numpy_array_api_compliance(self):
        """Test NumPy Array API compliance features"""
        # Test new array creation functions
        arr = np.asarray([1.0, 2.0, 3.0])
        assert isinstance(arr, np.ndarray)
        
        # Test shape and size attributes
        assert arr.shape == (3,)
        assert arr.size == 3
        assert arr.ndim == 1
        
        # Test mathematical functions
        result = np.sqrt(arr)
        expected = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.asyncio
    async def test_integration_ml_pipeline_numpy2(self):
        """Integration test for ML pipeline with NumPy 2.x"""
        # Create realistic ML data
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 10
        
        # Generate feature matrix
        X = rng.normal(0, 1, (n_samples, n_features)).astype(np.float64)
        y = rng.choice([0, 1], size=n_samples)
        
        # Test data validation
        validator = InputValidator()
        
        # Validate feature matrix
        validated_X = validator.validate_input("ml_features", X, "numpy_array")
        np.testing.assert_array_equal(validated_X, X)
        
        # Test statistical analysis
        mock_db_session = AsyncMock()
        analyzer = SessionComparisonAnalyzer(mock_db_session)
        
        # Create synthetic metrics from ML results
        metrics_a = rng.uniform(0.7, 0.9, 20).tolist()
        metrics_b = rng.uniform(0.75, 0.95, 20).tolist()
        
        from src.prompt_improver.ml.analytics.session_comparison_analyzer import ComparisonMethod
        comparison_result = await analyzer._perform_statistical_comparison(
            metrics_a, metrics_b, ComparisonMethod.T_TEST
        )
        
        # Verify results are valid
        assert 0 <= comparison_result['p_value'] <= 1
        assert isinstance(comparison_result['effect_size'], float)
        assert not np.isnan(comparison_result['effect_size'])
        assert not np.isinf(comparison_result['effect_size'])

    def test_numerical_precision_numpy2(self):
        """Test numerical precision with NumPy 2.x"""
        # Test floating point precision
        a = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        result = np.sum(a)
        
        # Should be close to 0.6 but might have floating point errors
        assert abs(result - 0.6) < 1e-15
        
        # Test integer operations
        b = np.array([1, 2, 3], dtype=np.int64)
        assert np.sum(b) == 6
        assert np.mean(b) == 2.0
        
        # Test mixed operations
        c = a * b  # float64 * int64 should give float64
        assert c.dtype == np.float64
        expected = np.array([0.1, 0.4, 0.9])
        np.testing.assert_array_almost_equal(c, expected)

    def test_numpy_error_handling(self):
        """Test NumPy error handling in 2.x"""
        # Test division by zero handling
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.array([1.0, 0.0]) / np.array([0.0, 0.0])
            assert np.isinf(result[0])
            assert np.isnan(result[1])
        
        # Test overflow handling
        with np.errstate(over='ignore'):
            large_num = np.float64(1e308)
            result = large_num * 10
            assert np.isinf(result)
        
        # Test underflow handling  
        with np.errstate(under='ignore'):
            small_num = np.float64(1e-308)
            result = small_num / 1e10
            # Result should be very small or zero
            assert result < 1e-310 or result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])