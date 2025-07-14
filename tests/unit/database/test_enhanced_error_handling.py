"""
Unit tests for enhanced psycopg error handling implementation.
Tests the 2025 best practices for error classification, retry mechanisms, and metrics.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.prompt_improver.database.error_handling import (
    ErrorSeverity, ErrorCategory, ErrorContext, RetryConfig, CircuitBreakerConfig,
    CircuitBreaker, RetryManager, ErrorMetrics, DatabaseErrorClassifier,
    CircuitBreakerState
)
from src.prompt_improver.database.config import DatabaseConfig

# Mock psycopg errors for testing
class MockPsycopgError(Exception):
    def __init__(self, message: str, sqlstate: str = None):
        super().__init__(message)
        self.sqlstate = sqlstate


class MockOperationalError(MockPsycopgError):
    pass


class MockIntegrityError(MockPsycopgError):
    pass


class MockConnectionTimeout(MockPsycopgError):
    pass


@pytest.fixture
def error_context():
    """Create a test error context."""
    return ErrorContext(
        operation="test_operation",
        query="SELECT 1",
        params={"param1": "value1"},
        connection_id="test-conn-123"
    )


@pytest.fixture
def retry_config():
    """Create a test retry configuration."""
    return RetryConfig(
        max_attempts=3,
        base_delay_ms=50,
        max_delay_ms=1000,
        exponential_base=2.0,
        jitter=False  # Disable jitter for predictable testing
    )


@pytest.fixture
def circuit_breaker_config():
    """Create a test circuit breaker configuration."""
    return CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout_seconds=1,
        success_threshold=2,
        enabled=True
    )


class TestDatabaseErrorClassifier:
    """Test error classification functionality."""
    
    def test_classify_error_by_sqlstate(self):
        """Test error classification using SQLSTATE codes."""
        # Test connection error
        conn_error = MockPsycopgError("Connection failed", sqlstate="08006")
        category, severity = DatabaseErrorClassifier.classify_error(conn_error)
        assert category == ErrorCategory.CONNECTION
        assert severity == ErrorSeverity.HIGH
        
        # Test timeout error
        timeout_error = MockPsycopgError("Query canceled", sqlstate="57014")
        category, severity = DatabaseErrorClassifier.classify_error(timeout_error)
        assert category == ErrorCategory.TIMEOUT
        assert severity == ErrorSeverity.HIGH
        
        # Test integrity error
        integrity_error = MockPsycopgError("Unique violation", sqlstate="23505")
        category, severity = DatabaseErrorClassifier.classify_error(integrity_error)
        assert category == ErrorCategory.INTEGRITY
        assert severity == ErrorSeverity.MEDIUM
        
        # Test syntax error
        syntax_error = MockPsycopgError("Syntax error", sqlstate="42601")
        category, severity = DatabaseErrorClassifier.classify_error(syntax_error)
        assert category == ErrorCategory.SYNTAX
        assert severity == ErrorSeverity.LOW
    
    def test_classify_error_fallback(self):
        """Test error classification fallback for unknown errors."""
        unknown_error = Exception("Unknown error")
        category, severity = DatabaseErrorClassifier.classify_error(unknown_error)
        assert category == ErrorCategory.FATAL
        assert severity == ErrorSeverity.CRITICAL
    
    def test_is_retryable(self):
        """Test retry determination logic."""
        # Retryable errors
        conn_error = MockPsycopgError("Connection failed", sqlstate="08006")
        assert DatabaseErrorClassifier.is_retryable(conn_error)
        
        deadlock_error = MockPsycopgError("Deadlock detected", sqlstate="40P01")
        assert DatabaseErrorClassifier.is_retryable(deadlock_error)
        
        # Non-retryable errors
        syntax_error = MockPsycopgError("Syntax error", sqlstate="42601")
        assert not DatabaseErrorClassifier.is_retryable(syntax_error)
        
        integrity_error = MockPsycopgError("Unique violation", sqlstate="23505")
        assert not DatabaseErrorClassifier.is_retryable(integrity_error)


class TestRetryManager:
    """Test retry mechanism functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_operation(self, retry_config, error_context):
        """Test retry manager with successful operation."""
        retry_manager = RetryManager(retry_config)
        
        async def successful_operation():
            return "success"
        
        result = await retry_manager.retry_async(successful_operation, error_context)
        assert result == "success"
        assert error_context.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_retryable_error_with_eventual_success(self, retry_config, error_context):
        """Test retry manager with retryable error that eventually succeeds."""
        retry_manager = RetryManager(retry_config)
        call_count = 0
        
        async def failing_then_success_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate transient connection error
                raise MockPsycopgError("Connection failed", sqlstate="08006")
            return "success"
        
        result = await retry_manager.retry_async(failing_then_success_operation, error_context)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_non_retryable_error(self, retry_config, error_context):
        """Test retry manager with non-retryable error."""
        retry_manager = RetryManager(retry_config)
        
        async def non_retryable_operation():
            # Syntax error is not retryable
            raise MockPsycopgError("Syntax error", sqlstate="42601")
        
        with pytest.raises(MockPsycopgError):
            await retry_manager.retry_async(non_retryable_operation, error_context)
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, retry_config, error_context):
        """Test retry manager when max retries are exceeded."""
        retry_manager = RetryManager(retry_config)
        
        async def always_failing_operation():
            # Always raise retryable error
            raise MockPsycopgError("Connection failed", sqlstate="08006")
        
        with pytest.raises(MockPsycopgError):
            await retry_manager.retry_async(always_failing_operation, error_context)
    
    def test_calculate_delay(self, retry_config):
        """Test delay calculation with exponential backoff."""
        retry_manager = RetryManager(retry_config)
        
        # Test delay calculation
        delay1 = retry_manager._calculate_delay(1)
        delay2 = retry_manager._calculate_delay(2)
        delay3 = retry_manager._calculate_delay(3)
        
        assert delay1 == 50  # base_delay_ms
        assert delay2 == 100  # base_delay_ms * exponential_base^1
        assert delay3 == 200  # base_delay_ms * exponential_base^2


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_closed_state_successful_operation(self, circuit_breaker_config):
        """Test circuit breaker in closed state with successful operation."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)
        
        async def successful_operation():
            return "success"
        
        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_transition_to_open_state(self, circuit_breaker_config):
        """Test circuit breaker transition to open state after failures."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)
        
        async def failing_operation():
            raise Exception("Operation failed")
        
        # Fail enough times to open the circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker_config.failure_threshold
    
    @pytest.mark.asyncio
    async def test_open_state_blocks_requests(self, circuit_breaker_config):
        """Test that open circuit breaker blocks requests."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)
        circuit_breaker.state = CircuitBreakerState.OPEN
        
        async def any_operation():
            return "should not execute"
        
        # Should raise OperationalError when circuit is open
        with pytest.raises(Exception):  # MockOperationalError equivalent
            await circuit_breaker.call(any_operation)
    
    @pytest.mark.asyncio
    async def test_half_open_to_closed_transition(self, circuit_breaker_config):
        """Test circuit breaker transition from half-open to closed state."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        
        async def successful_operation():
            return "success"
        
        # Execute successful operations to meet success threshold
        for _ in range(circuit_breaker_config.success_threshold):
            result = await circuit_breaker.call(successful_operation)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0


class TestErrorMetrics:
    """Test error metrics collection and reporting."""
    
    def test_record_error(self, error_context):
        """Test error recording functionality."""
        metrics = ErrorMetrics()
        error = MockPsycopgError("Connection failed", sqlstate="08006")
        
        metrics.record_error(error_context, error)
        
        assert ErrorCategory.CONNECTION in metrics.error_counts
        assert metrics.error_counts[ErrorCategory.CONNECTION] == 1
        assert len(metrics.last_errors) == 1
        assert metrics.last_errors[0].operation == "test_operation"
    
    def test_error_rate_calculation(self, error_context):
        """Test error rate calculation."""
        metrics = ErrorMetrics()
        error = MockPsycopgError("Connection failed", sqlstate="08006")
        
        # Record multiple errors
        for _ in range(5):
            metrics.record_error(error_context, error)
        
        # Check error rate (errors per minute)
        rate = metrics.get_error_rate(ErrorCategory.CONNECTION, window_minutes=60)
        assert rate == 5 / 60  # 5 errors in 60 minutes
    
    def test_metrics_summary(self, error_context):
        """Test comprehensive metrics summary."""
        metrics = ErrorMetrics()
        
        # Record different types of errors
        conn_error = MockPsycopgError("Connection failed", sqlstate="08006")
        timeout_error = MockPsycopgError("Query timeout", sqlstate="57014")
        
        metrics.record_error(error_context, conn_error)
        metrics.record_error(error_context, timeout_error)
        
        summary = metrics.get_metrics_summary()
        
        assert summary["total_errors"] == 2
        assert ErrorCategory.CONNECTION.value in summary["error_counts_by_category"]
        assert ErrorCategory.TIMEOUT.value in summary["error_counts_by_category"]
        assert len(summary["last_errors"]) == 2


@pytest.mark.asyncio
async def test_enhanced_error_context_decorator():
    """Test the error context enhancement decorator."""
    from src.prompt_improver.database.error_handling import enhance_error_context
    
    class MockClient:
        def __init__(self):
            self.error_metrics = ErrorMetrics()
    
    @enhance_error_context
    async def test_method(self, query: str, params: dict = None):
        if "fail" in query:
            raise MockPsycopgError("Test error", sqlstate="08006")
        return "success"
    
    client = MockClient()
    
    # Test successful operation
    result = await test_method(client, "SELECT 1")
    assert result == "success"
    
    # Test failed operation
    with pytest.raises(MockPsycopgError):
        await test_method(client, "SELECT fail")
    
    # Check that error was recorded in metrics
    assert len(client.error_metrics.last_errors) == 1
    assert client.error_metrics.last_errors[0].operation == "MockClient.test_method"


class TestIntegration:
    """Integration tests for the complete error handling system."""
    
    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self, retry_config, circuit_breaker_config, error_context):
        """Test integration of retry manager with circuit breaker."""
        retry_manager = RetryManager(retry_config)
        circuit_breaker = CircuitBreaker(circuit_breaker_config)
        
        call_count = 0
        
        async def intermittent_failure():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise MockPsycopgError("Connection failed", sqlstate="08006")
            return f"success after {call_count} attempts"
        
        async def operation_with_retry():
            return await retry_manager.retry_async(intermittent_failure, error_context)
        
        result = await circuit_breaker.call(operation_with_retry)
        assert "success" in result
        assert call_count == 2
        assert circuit_breaker.state == CircuitBreakerState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__]) 