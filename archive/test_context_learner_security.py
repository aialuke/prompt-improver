"""Security tests for Context Learner implementation.

Tests input validation, memory safety, and authentication for ML operations.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.ml.learning.algorithms.context_learner import (
    ContextSpecificLearner,
    ContextConfig,
)
from prompt_improver.security import ValidationError, MemoryGuard


class TestContextLearnerSecurity:
    """Test security aspects of the context learner."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.fetch = AsyncMock(return_value=[])
        return session

    @pytest.fixture
    def context_learner(self):
        """Create context learner with test configuration."""
        config = ContextConfig(
            enable_semantic_clustering=False,
            enable_linguistic_features=False,
            enable_domain_features=False,
            enable_context_aware_weighting=False,
            privacy_preserving=True,
        )
        return ContextSpecificLearner(config=config)

    def test_input_validation_initialization(self, context_learner):
        """Test that security components are properly initialized."""
        assert hasattr(context_learner, 'input_validator')
        assert hasattr(context_learner, 'memory_guard')
        assert hasattr(context_learner, 'auth_service')
        assert context_learner.input_validator is not None
        assert context_learner.memory_guard is not None
        assert context_learner.auth_service is not None

    def test_session_id_validation_success(self, context_learner, mock_db_session):
        """Test successful session ID validation."""
        valid_session_id = "test_session_12345"
        
        # Mock the database query to return empty results
        mock_db_session.execute.return_value.fetchall = MagicMock(return_value=[])
        
        # This should not raise an exception
        try:
            # We can't directly call the async method in a sync test,
            # so we'll test the validation component directly
            validated = context_learner.input_validator.validate_input("session_id", valid_session_id)
            assert validated == valid_session_id
        except ValidationError:
            pytest.fail("Valid session ID should not raise ValidationError")

    def test_session_id_validation_failure(self, context_learner):
        """Test session ID validation with invalid inputs."""
        invalid_session_ids = [
            "",  # Empty string
            "ab",  # Too short
            "a" * 200,  # Too long
            "session<script>alert('xss')</script>",  # XSS attempt
            "session'; DROP TABLE users; --",  # SQL injection attempt
            "session && rm -rf /",  # Command injection attempt
            None,  # None value
            123,  # Wrong type
        ]
        
        for invalid_session in invalid_session_ids:
            with pytest.raises(ValidationError):
                context_learner.input_validator.validate_input("session_id", invalid_session)

    def test_context_data_validation_success(self, context_learner):
        """Test successful context data validation."""
        valid_context = {
            "domain": "web_development",
            "projectType": "web",
            "complexity": "medium",
            "teamSize": 5
        }
        
        validated = context_learner.input_validator.validate_input("context_data", valid_context)
        assert isinstance(validated, dict)
        assert "domain" in validated
        assert "projectType" in validated

    def test_context_data_validation_failure(self, context_learner):
        """Test context data validation with invalid inputs."""
        invalid_contexts = [
            {},  # Missing required fields
            {"domain": "x" * 200},  # Domain too long
            {"domain": "web", "projectType": "invalid_type"},  # Invalid project type
            {"domain": "<script>alert('xss')</script>"},  # XSS in domain
            None,  # None value
            "not_a_dict",  # Wrong type
        ]
        
        for invalid_context in invalid_contexts:
            with pytest.raises(ValidationError):
                context_learner.input_validator.validate_input("context_data", invalid_context)

    def test_numpy_array_validation_success(self, context_learner):
        """Test successful numpy array validation."""
        valid_arrays = [
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.zeros((100, 100), dtype=np.float32),
        ]
        
        for valid_array in valid_arrays:
            validated = context_learner.input_validator.validate_input("numpy_array", valid_array)
            assert isinstance(validated, np.ndarray)
            assert validated.shape == valid_array.shape

    def test_numpy_array_validation_failure(self, context_learner):
        """Test numpy array validation with invalid inputs."""
        # Array that's too large (exceeds memory limit)
        huge_array = np.ones((10000, 10000), dtype=np.float64)  # ~800MB
        
        with pytest.raises(ValidationError):
            context_learner.input_validator.validate_input("numpy_array", huge_array)
        
        # Array with NaN values
        nan_array = np.array([1.0, np.nan, 3.0])
        
        with pytest.raises(ValidationError):
            context_learner.input_validator.validate_input("numpy_array", nan_array)
        
        # Array with infinite values
        inf_array = np.array([1.0, np.inf, 3.0])
        
        with pytest.raises(ValidationError):
            context_learner.input_validator.validate_input("numpy_array", inf_array)

    def test_memory_bounds_checking(self, context_learner):
        """Test memory bounds checking for buffer operations."""
        # Test safe buffer size
        small_buffer = b"test data"
        assert context_learner.memory_guard.validate_buffer_size(small_buffer, "test")
        
        # Test oversized buffer (this should raise MemoryError)
        large_buffer = b"x" * (200 * 1024 * 1024)  # 200MB
        
        with pytest.raises(MemoryError):
            context_learner.memory_guard.validate_buffer_size(large_buffer, "test")

    def test_safe_frombuffer_operation(self, context_learner):
        """Test safe numpy frombuffer operation."""
        # Create valid buffer
        test_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        buffer = test_array.tobytes()
        
        # This should work safely
        result = context_learner.memory_guard.safe_frombuffer(buffer, np.float64)
        np.testing.assert_array_equal(result, test_array)

    def test_safe_frombuffer_invalid_size(self, context_learner):
        """Test safe frombuffer with invalid buffer size."""
        # Create buffer that doesn't align with dtype
        invalid_buffer = b"123"  # 3 bytes, not divisible by 8 (float64 size)
        
        with pytest.raises(ValueError):
            context_learner.memory_guard.safe_frombuffer(invalid_buffer, np.float64)

    def test_safe_tobytes_operation(self, context_learner):
        """Test safe numpy tobytes operation."""
        test_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        # This should work safely
        result = context_learner.memory_guard.safe_tobytes(test_array)
        assert isinstance(result, bytes)
        assert len(result) == test_array.nbytes

    def test_safe_tobytes_oversized_array(self, context_learner):
        """Test safe tobytes with oversized array."""
        # Create array that exceeds memory limits
        huge_array = np.ones((20000, 20000), dtype=np.float64)  # ~3.2GB
        
        with pytest.raises(MemoryError):
            context_learner.memory_guard.safe_tobytes(huge_array)

    def test_privacy_parameters_validation(self, context_learner):
        """Test differential privacy parameter validation."""
        # Valid epsilon values
        valid_epsilons = [0.1, 1.0, 5.0, 10.0]
        for epsilon in valid_epsilons:
            validated = context_learner.input_validator.validate_input("privacy_epsilon", epsilon)
            assert validated == epsilon
        
        # Invalid epsilon values
        invalid_epsilons = [-1.0, 0.0, 15.0, "not_a_number", None]
        for epsilon in invalid_epsilons:
            with pytest.raises(ValidationError):
                context_learner.input_validator.validate_input("privacy_epsilon", epsilon)

    def test_ml_features_validation(self, context_learner):
        """Test ML features validation."""
        # Valid features
        valid_features = [
            [1.0, 2.0, 3.0],
            np.array([0.5, 1.5, 2.5]),
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ]
        
        for features in valid_features:
            validated = context_learner.input_validator.validate_input("ml_features", features)
            assert validated is not None
        
        # Invalid features
        invalid_features = [
            [1.0, np.nan, 3.0],  # Contains NaN
            [1.0, np.inf, 3.0],  # Contains infinity
            [1.0, 2e10, 3.0],    # Values too large
            ["not", "numbers"],   # Wrong types
            None,                 # None value
        ]
        
        for features in invalid_features:
            with pytest.raises(ValidationError):
                context_learner.input_validator.validate_input("ml_features", features)

    @pytest.mark.asyncio
    async def test_authentication_decorator_integration(self, context_learner, mock_db_session):
        """Test that authentication decorators are properly applied."""
        # Test with valid session ID
        valid_session = "test_session_12345"
        
        # Mock the data fetching to avoid database dependencies
        with patch.object(context_learner.training_loader, 'load_data_for_context_learning', 
                         return_value=AsyncMock(return_value=[])):
            try:
                # This should trigger the authentication decorator
                await context_learner.update_from_new_data(mock_db_session, valid_session)
                # If we get here, authentication passed
                assert True
            except PermissionError:
                pytest.fail("Valid session should not trigger PermissionError")

    @pytest.mark.asyncio
    async def test_authentication_decorator_failure(self, context_learner, mock_db_session):
        """Test authentication decorator with invalid session."""
        # Test with invalid session ID
        invalid_session = "invalid<script>"
        
        with pytest.raises(PermissionError):
            await context_learner.update_from_new_data(mock_db_session, invalid_session)

    def test_context_key_generation_security(self, context_learner):
        """Test security of context key generation."""
        # Test with malicious context data
        malicious_contexts = [
            {"domain": "<script>alert('xss')</script>", "projectType": "web"},
            {"domain": "'; DROP TABLE users; --", "projectType": "web"},
            {"domain": "normal", "projectType": "$(rm -rf /)"},
        ]
        
        for malicious_context in malicious_contexts:
            # This should handle malicious input gracefully
            result = context_learner._generate_context_key(malicious_context)
            # The result should be sanitized or use safe defaults
            assert "<script>" not in result
            assert "DROP TABLE" not in result
            assert "rm -rf" not in result

    def test_memory_monitoring(self, context_learner):
        """Test memory usage monitoring."""
        memory_stats = context_learner.memory_guard.check_memory_usage()
        
        assert isinstance(memory_stats, dict)
        assert "current_mb" in memory_stats
        assert "peak_mb" in memory_stats
        assert "usage_percent" in memory_stats
        assert memory_stats["current_mb"] >= 0
        assert memory_stats["peak_mb"] >= 0
        assert 0 <= memory_stats["usage_percent"] <= 200  # Allow for some overhead

    def test_secure_hash_generation(self, context_learner):
        """Test secure hash ID generation."""
        test_inputs = ["user123", "session456", "context789"]
        
        hash_id = context_learner.input_validator.create_hash_id(*test_inputs)
        
        assert isinstance(hash_id, str)
        assert len(hash_id) == 16  # SHA256 truncated to 16 chars
        
        # Same inputs should produce same hash
        hash_id2 = context_learner.input_validator.create_hash_id(*test_inputs)
        assert hash_id == hash_id2
        
        # Different inputs should produce different hash
        hash_id3 = context_learner.input_validator.create_hash_id("different", "inputs")
        assert hash_id != hash_id3


class TestMemoryGuardStandalone:
    """Test the memory guard component in isolation."""

    def test_memory_guard_initialization(self):
        """Test memory guard initialization."""
        guard = MemoryGuard(max_memory_mb=100, max_buffer_size=10 * 1024 * 1024)
        
        assert guard.max_memory_mb == 100
        assert guard.max_buffer_size == 10 * 1024 * 1024
        assert guard.initial_memory > 0

    def test_buffer_size_validation(self):
        """Test buffer size validation."""
        guard = MemoryGuard(max_buffer_size=1024)  # 1KB limit
        
        # Small buffer should pass
        small_data = b"test"
        assert guard.validate_buffer_size(small_data, "test")
        
        # Large buffer should fail
        large_data = b"x" * 2048  # 2KB
        with pytest.raises(MemoryError):
            guard.validate_buffer_size(large_data, "test")

    def test_memory_monitoring_context(self):
        """Test memory monitoring context manager."""
        guard = MemoryGuard()
        
        with guard.monitor_operation("test_operation") as monitor:
            # Simulate some memory usage
            test_data = np.ones((1000, 1000))  # ~8MB
            assert test_data.size > 0  # Use the data to prevent optimization
        
        # Monitor should have tracked the operation
        assert guard.peak_memory >= guard.initial_memory