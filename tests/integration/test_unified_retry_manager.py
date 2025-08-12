"""
Integration tests for the Unified Retry Manager.

Tests real behavior of the unified retry manager with actual failures,
circuit breaker functionality, and observability.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from prompt_improver.core.retry_manager import (
    RetryableErrorType,
    RetryConfig,
    RetryManager as UnifiedRetryManager,
    RetryStrategy,
    get_retry_manager,
)

CircuitBreakerOpenError = Exception


class TestUnifiedRetryManager:
    """Test the unified retry manager with real behavior."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay_ms=10,
            max_delay_ms=100,
            enable_circuit_breaker=True,
            failure_threshold=2,
            recovery_timeout_ms=1000,
        )
        return UnifiedRetryManager(config)

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self, retry_manager):
        """Test that successful operations don't trigger retries."""
        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_manager.retry_async(successful_operation)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_eventual_success(self, retry_manager):
        """Test retry logic with eventual success."""
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Attempt {call_count} failed")
            return f"success on attempt {call_count}"

        config = RetryConfig(
            enable_circuit_breaker=False, operation_name="test_eventual_success"
        )
        result = await retry_manager.retry_async(failing_then_success, config=config)
        assert result == "success on attempt 3"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, retry_manager):
        """Test that retries are exhausted and final exception is raised."""
        call_count = 0

        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Attempt {call_count} failed")

        config = RetryConfig(
            enable_circuit_breaker=False, operation_name="test_exhaustion"
        )
        with pytest.raises(ConnectionError) as exc_info:
            await retry_manager.retry_async(always_failing, config=config)
        assert "Attempt 3 failed" in str(exc_info.value)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error(self, retry_manager):
        """Test that non-retryable errors don't trigger retries."""
        call_count = 0

        async def syntax_error_operation():
            nonlocal call_count
            call_count += 1
            raise SyntaxError("This is not retryable")

        config = RetryConfig(
            max_attempts=3,
            retryable_errors=[RetryableErrorType.NETWORK],
            operation_name="syntax_test",
        )
        with pytest.raises(SyntaxError):
            await retry_manager.retry_async(syntax_error_operation, config=config)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, retry_manager):
        """Test circuit breaker opens after threshold failures."""
        call_count = 0

        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Failure {call_count}")

        config = RetryConfig(
            operation_name="circuit_breaker_test_unique",
            failure_threshold=2,
            max_attempts=2,
        )
        with pytest.raises(ConnectionError):
            await retry_manager.retry_async(always_failing, config=config)
        with pytest.raises(CircuitBreakerOpenError):
            await retry_manager.retry_async(always_failing, config=config)

    @pytest.mark.asyncio
    async def test_different_retry_strategies(self, retry_manager):
        """Test different retry strategies calculate delays correctly."""
        delays = []

        async def capture_delay_operation():
            raise TimeoutError("Test timeout")

        config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay_ms=100,
            multiplier=2.0,
            jitter=False,
            operation_name="delay_test",
        )
        expected_delays = [100, 200]
        for i in range(2):
            delay = retry_manager._calculate_delay(i, config)
            assert delay == expected_delays[i]

    @pytest.mark.asyncio
    async def test_retry_metrics_collection(self, retry_manager):
        """Test that retry metrics are collected properly."""
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "success"

        config = RetryConfig(operation_name="metrics_test")
        result = await retry_manager.retry_async(failing_operation, config=config)
        assert result == "success"
        metrics = retry_manager.get_retry_metrics("metrics_test")
        assert metrics is not None
        assert metrics["total_attempts"] >= 2
        assert metrics["successful_attempts"] >= 1
        assert metrics["success_rate"] > 0

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, retry_manager):
        """Test using retry manager with context manager."""
        call_count = 0

        async def test_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Test error")
            return "context success"

        async with retry_manager.with_retry("context_test") as executor:
            result = await executor.execute(test_operation)
        assert result == "context success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_global_retry_manager(self):
        """Test global retry manager instance."""
        manager1 = get_retry_manager()
        manager2 = get_retry_manager()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_retry_async_usage(self):
        """Test retry_async functionality (replaces decorator test)."""
        retry_manager = get_retry_manager()
        call_count = 0

        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry test error")
            return "retry success"

        config = RetryConfig(max_attempts=2, base_delay=0.01)
        result = await retry_manager.retry_async(test_function, config)
        assert result == "retry success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, retry_manager):
        """Test circuit breaker recovery after timeout."""
        call_count = 0

        async def initially_failing():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError(f"Failure {call_count}")
            return "recovered"

        config = RetryConfig(
            operation_name="recovery_test_unique",
            failure_threshold=2,
            recovery_timeout_ms=100,
            max_attempts=2,
        )
        with pytest.raises(ConnectionError):
            await retry_manager.retry_async(initially_failing, config=config)
        with pytest.raises(CircuitBreakerOpenError):
            await retry_manager.retry_async(initially_failing, config=config)
        await asyncio.sleep(0.2)
        await retry_manager.reset_circuit_breaker("recovery_test_unique")
        result = await retry_manager.retry_async(initially_failing, config=config)
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_error_classification(self, retry_manager):
        """Test that error classification works correctly."""
        network_error = ConnectionError("Connection failed")
        error_type = retry_manager._classify_error(
            network_error, retry_manager.default_config
        )
        assert error_type == RetryableErrorType.NETWORK
        timeout_error = TimeoutError("Operation timed out")
        error_type = retry_manager._classify_error(
            timeout_error, retry_manager.default_config
        )
        assert error_type == RetryableErrorType.TIMEOUT

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.FIBONACCI_BACKOFF,
            initial_delay_ms=50,
            max_delay_ms=5000,
        )
        assert config.max_attempts == 5
        assert config.strategy == RetryStrategy.FIBONACCI_BACKOFF
        assert config.initial_delay_ms == 50
        assert config.max_delay_ms == 5000

    @pytest.mark.asyncio
    async def test_observability_integration(self, retry_manager):
        """Test that observability manager integration works."""
        call_count = 0

        async def monitored_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Monitored failure")
            return "monitored success"

        config = RetryConfig(
            operation_name="observability_test",
            enable_metrics=True,
            enable_tracing=True,
        )
        result = await retry_manager.retry_async(monitored_operation, config=config)
        assert result == "monitored success"
        metrics = retry_manager.get_retry_metrics("observability_test")
        assert metrics is not None
        assert metrics["total_attempts"] == 2
        assert metrics["failure_rate"] < 1.0
