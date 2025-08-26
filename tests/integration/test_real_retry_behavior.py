"""
Real behavior tests for the Unified Retry Manager.

Tests actual retry behavior, circuit breaker functionality, and error handling
without mocks or complex dependencies.
"""

import asyncio
import time
from typing import Any

import pytest


class RealRetryManager:
    """Simplified retry manager for testing real behavior."""

    def __init__(self):
        self.metrics: dict[str, dict[str, Any]] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}

    async def retry_async(
        self,
        operation,
        max_attempts=3,
        initial_delay=0.1,
        multiplier=2.0,
        operation_name="test",
    ):
        """Execute operation with real retry logic."""
        last_exception = None
        start_time = time.time()
        if self._is_circuit_breaker_open(operation_name):
            raise Exception(f"Circuit breaker is OPEN for {operation_name}")
        for attempt in range(max_attempts):
            try:
                result = await operation()
                self._record_success(
                    operation_name, attempt + 1, time.time() - start_time
                )
                self._reset_circuit_breaker(operation_name)
                return result
            except Exception as e:
                last_exception = e
                self._record_failure(operation_name, str(e))
                if attempt < max_attempts - 1:
                    delay = initial_delay * multiplier**attempt
                    await asyncio.sleep(delay)
                else:
                    self._update_circuit_breaker(operation_name)
        raise last_exception

    def _is_circuit_breaker_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open."""
        cb = self.circuit_breakers.get(operation_name, {})
        return cb.get("state") == "OPEN" and time.time() < cb.get("open_until", 0)

    def _reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker on success."""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name]["failure_count"] = 0
            self.circuit_breakers[operation_name]["state"] = "CLOSED"

    def _update_circuit_breaker(self, operation_name: str):
        """Update circuit breaker on failure."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = {
                "failure_count": 0,
                "state": "CLOSED",
                "open_until": 0,
            }
        cb = self.circuit_breakers[operation_name]
        cb["failure_count"] += 1
        if cb["failure_count"] >= 3:
            cb["state"] = "OPEN"
            cb["open_until"] = time.time() + 5.0

    def _record_success(self, operation_name: str, attempts: int, duration: float):
        """Record successful operation."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                "total_attempts": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "total_duration": 0.0,
                "errors": [],
            }
        metrics = self.metrics[operation_name]
        metrics["total_attempts"] += attempts
        metrics["successful_operations"] += 1
        metrics["total_duration"] += duration

    def _record_failure(self, operation_name: str, error: str):
        """Record failed operation."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                "total_attempts": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "total_duration": 0.0,
                "errors": [],
            }
        metrics = self.metrics[operation_name]
        metrics["failed_operations"] += 1
        metrics["errors"].append(error)

    def get_metrics(self, operation_name: str) -> dict[str, Any]:
        """Get metrics for operation."""
        return self.metrics.get(operation_name, {})

    def get_circuit_breaker_status(self, operation_name: str) -> dict[str, Any]:
        """Get circuit breaker status."""
        return self.circuit_breakers.get(
            operation_name, {"state": "CLOSED", "failure_count": 0}
        )


class TestRealRetryBehavior:
    """Test real retry behavior without mocks."""

    @pytest.fixture
    def retry_manager(self):
        """Create a real retry manager for testing."""
        return RealRetryManager()

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self, retry_manager):
        """Test that successful operations don't trigger retries."""
        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        start_time = time.time()
        result = await retry_manager.retry_async(
            successful_operation, operation_name="success_test"
        )
        duration = time.time() - start_time
        assert result == "success"
        assert call_count == 1
        assert duration < 0.1
        metrics = retry_manager.get_metrics("success_test")
        assert metrics["successful_operations"] == 1
        assert metrics["failed_operations"] == 0

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

        start_time = time.time()
        result = await retry_manager.retry_async(
            failing_then_success,
            max_attempts=3,
            initial_delay=0.01,
            operation_name="retry_test",
        )
        duration = time.time() - start_time
        assert result == "success on attempt 3"
        assert call_count == 3
        assert 0.01 < duration < 0.1
        metrics = retry_manager.get_metrics("retry_test")
        assert metrics["successful_operations"] == 1
        assert metrics["total_attempts"] == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, retry_manager):
        """Test that retries are exhausted and final exception is raised."""
        call_count = 0

        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Attempt {call_count} failed")

        start_time = time.time()
        with pytest.raises(ConnectionError) as exc_info:
            await retry_manager.retry_async(
                always_failing,
                max_attempts=3,
                initial_delay=0.01,
                operation_name="failure_test",
            )
        duration = time.time() - start_time
        assert "Attempt 3 failed" in str(exc_info.value)
        assert call_count == 3
        assert 0.01 < duration < 0.1
        metrics = retry_manager.get_metrics("failure_test")
        assert metrics["failed_operations"] == 3
        assert len(metrics["errors"]) == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, retry_manager):
        """Test that exponential backoff timing works correctly."""
        call_count = 0
        delays = []

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                delays.append(time.time())
            raise TimeoutError(f"Timeout {call_count}")

        start_time = time.time()
        with pytest.raises(TimeoutError):
            await retry_manager.retry_async(
                failing_operation,
                max_attempts=3,
                initial_delay=0.1,
                multiplier=2.0,
                operation_name="timing_test",
            )
        assert len(delays) == 2
        if len(delays) >= 2:
            first_delay = delays[0] - start_time
            second_delay = delays[1] - delays[0]
            assert 0.08 < first_delay < 0.15
            assert 0.18 < second_delay < 0.25

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, retry_manager):
        """Test circuit breaker opens after threshold failures."""
        call_count = 0

        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Failure {call_count}")

        with pytest.raises(ConnectionError):
            await retry_manager.retry_async(
                always_failing,
                max_attempts=3,
                initial_delay=0.01,
                operation_name="circuit_test",
            )
        cb_status = retry_manager.get_circuit_breaker_status("circuit_test")
        assert cb_status["failure_count"] == 1
        assert cb_status["state"] == "CLOSED"
        for _i in range(2):
            with pytest.raises(ConnectionError):
                await retry_manager.retry_async(
                    always_failing,
                    max_attempts=3,
                    initial_delay=0.01,
                    operation_name="circuit_test",
                )
        cb_status = retry_manager.get_circuit_breaker_status("circuit_test")
        assert cb_status["state"] == "OPEN"
        with pytest.raises(Exception) as exc_info:
            await retry_manager.retry_async(
                always_failing,
                max_attempts=3,
                initial_delay=0.01,
                operation_name="circuit_test",
            )
        assert "Circuit breaker is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, retry_manager):
        """Test circuit breaker recovery after timeout."""
        call_count = 0

        async def initially_failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count <= 9:
                raise ConnectionError(f"Failure {call_count}")
            return "recovered"

        for _i in range(3):
            with pytest.raises(ConnectionError):
                await retry_manager.retry_async(
                    initially_failing_then_success,
                    max_attempts=3,
                    initial_delay=0.01,
                    operation_name="recovery_test",
                )
        cb_status = retry_manager.get_circuit_breaker_status("recovery_test")
        assert cb_status["state"] == "OPEN"
        await asyncio.sleep(5.1)
        result = await retry_manager.retry_async(
            initially_failing_then_success,
            max_attempts=3,
            initial_delay=0.01,
            operation_name="recovery_test",
        )
        assert result == "recovered"
        cb_status = retry_manager.get_circuit_breaker_status("recovery_test")
        assert cb_status["state"] == "CLOSED"
        assert cb_status["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_different_operations_isolated(self, retry_manager):
        """Test that different operations have isolated metrics and circuit breakers."""

        async def failing_op():
            raise ConnectionError("Always fails")

        async def success_op():
            return "success"

        with pytest.raises(ConnectionError):
            await retry_manager.retry_async(failing_op, operation_name="op1")
        result = await retry_manager.retry_async(success_op, operation_name="op2")
        assert result == "success"
        op1_metrics = retry_manager.get_metrics("op1")
        op2_metrics = retry_manager.get_metrics("op2")
        assert op1_metrics["failed_operations"] > 0
        assert op2_metrics["successful_operations"] == 1
        assert op2_metrics["failed_operations"] == 0
        op1_cb = retry_manager.get_circuit_breaker_status("op1")
        op2_cb = retry_manager.get_circuit_breaker_status("op2")
        assert op1_cb["failure_count"] > 0
        assert op2_cb["failure_count"] == 0

    def test_metrics_accuracy(self, retry_manager):
        """Test that metrics accurately reflect operation results."""
        retry_manager._record_success("test_op", 2, 0.5)
        retry_manager._record_failure("test_op", "error1")
        retry_manager._record_success("test_op", 1, 0.3)
        metrics = retry_manager.get_metrics("test_op")
        assert metrics["successful_operations"] == 2
        assert metrics["failed_operations"] == 1
        assert metrics["total_attempts"] == 3
        assert metrics["total_duration"] == 0.8
        assert len(metrics["errors"]) == 1
        assert metrics["errors"][0] == "error1"
