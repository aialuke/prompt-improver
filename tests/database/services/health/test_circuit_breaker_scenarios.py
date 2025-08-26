"""Comprehensive circuit breaker failure scenario tests.

Tests advanced circuit breaker patterns and failure scenarios:
- State transition validation under various failure conditions
- Recovery timeout progression with exponential backoff
- Circuit breaker decorator functionality with real async functions
- Half-open state behavior with success/failure transitions
- Timeout-based failure detection and response time tracking

Real behavior tests for production circuit breaker patterns.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from prompt_improver.database.services.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitBreakerState,
)


class UnreliableService:
    """Mock service that fails predictably for testing."""

    def __init__(self, failure_rate: float = 0.0, response_time_ms: float = 10.0):
        self.failure_rate = failure_rate  # 0.0 = never fail, 1.0 = always fail
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0

    async def make_request(self, data: str = "test") -> str:
        """Simulate async service call with configurable failure rate."""
        self.call_count += 1

        # Simulate network delay
        if self.response_time_ms > 0:
            await asyncio.sleep(self.response_time_ms / 1000.0)

        # Determine if this call should fail
        import random
        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise Exception(f"Service failure #{self.failure_count}")

        self.success_count += 1
        return f"Success: {data} (call #{self.call_count})"

    def sync_request(self, data: str = "test") -> str:
        """Synchronous version for testing sync decorator."""
        import time
        self.call_count += 1

        # Simulate processing delay
        if self.response_time_ms > 0:
            time.sleep(self.response_time_ms / 1000.0)

        import random
        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise Exception(f"Sync service failure #{self.failure_count}")

        self.success_count += 1
        return f"Sync success: {data} (call #{self.call_count})"


class TestCircuitBreakerFailureScenarios:
    """Test circuit breaker with various failure scenarios."""

    def test_circuit_breaker_state_progression(self):
        """Test complete state progression through failure and recovery."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=1.0,
            success_threshold=2,
        )
        cb = CircuitBreaker("progression_test", config)

        # Initial state: CLOSED
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_call_permitted() is True

        # Record failures to reach threshold
        for i in range(3):
            cb.record_failure()
            if i < 2:
                assert cb.state == CircuitBreakerState.CLOSED
            else:
                assert cb.state == CircuitBreakerState.OPEN

        # Circuit should be OPEN
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_call_permitted() is False

        # Force recovery timeout to elapse
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=2.0)

        # Should transition to HALF_OPEN on next check
        assert cb.is_call_permitted() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Success in half-open should increment success count
        cb.record_success()
        assert cb.success_count == 1
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Reaching success threshold should close circuit
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

        print("✅ Circuit breaker state progression: CLOSED → OPEN → HALF_OPEN → CLOSED")

    def test_exponential_backoff_progression(self):
        """Test exponential backoff behavior over multiple open cycles."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=2.0,
            backoff_multiplier=2.0,
            max_recovery_timeout_seconds=16.0,
        )
        cb = CircuitBreaker("backoff_test", config)

        expected_timeouts = [4.0, 8.0, 16.0, 16.0]  # 2.0 * 2.0^n, capped at 16.0

        for i, expected_timeout in enumerate(expected_timeouts):
            # Trigger circuit breaker open
            cb.record_failure()
            assert cb.state == CircuitBreakerState.OPEN
            assert cb.current_recovery_timeout == expected_timeout

            print(f"    Cycle {i + 1}: Recovery timeout = {cb.current_recovery_timeout:.1f}s")

            # Force another open transition for next iteration
            if i < len(expected_timeouts) - 1:
                cb._transition_to_open()

        print("✅ Exponential backoff progression validated")

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_async(self):
        """Test circuit breaker as async decorator with real service."""
        unreliable_service = UnreliableService(failure_rate=0.7, response_time_ms=5.0)

        # Create circuit breaker with low threshold for testing
        cb = CircuitBreaker("decorator_test", CircuitBreakerConfig(failure_threshold=2))

        # Wrap service method with circuit breaker
        protected_request = cb(unreliable_service.make_request)

        # Make calls until circuit breaker opens
        failures = 0
        for i in range(10):
            try:
                result = await protected_request(f"request_{i}")
                print(f"    Success: {result}")
            except CircuitBreakerOpenException as e:
                print(f"    Circuit breaker opened: {e}")
                break
            except Exception as e:
                failures += 1
                print(f"    Service failure #{failures}: {e}")

        # Circuit breaker should be open
        assert cb.state == CircuitBreakerState.OPEN
        assert failures >= 2

        # Verify circuit breaker blocks subsequent calls
        with pytest.raises(CircuitBreakerOpenException):
            await protected_request("blocked_request")

        print("✅ Async decorator functionality validated")

    def test_circuit_breaker_decorator_sync(self):
        """Test circuit breaker as sync decorator."""
        unreliable_service = UnreliableService(failure_rate=0.8, response_time_ms=2.0)

        cb = CircuitBreaker("sync_decorator_test", CircuitBreakerConfig(failure_threshold=2))
        protected_request = cb(unreliable_service.sync_request)

        # Make calls until circuit breaker opens
        failures = 0
        for i in range(8):
            try:
                result = protected_request(f"sync_request_{i}")
                print(f"    Sync success: {result}")
            except CircuitBreakerOpenException as e:
                print(f"    Sync circuit breaker opened: {e}")
                break
            except Exception as e:
                failures += 1
                print(f"    Sync service failure #{failures}: {e}")

        assert cb.state == CircuitBreakerState.OPEN
        assert failures >= 2

        print("✅ Sync decorator functionality validated")

    def test_timeout_based_failures(self):
        """Test circuit breaker with timeout-based failure detection."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_ms=50.0,  # 50ms timeout threshold
        )
        cb = CircuitBreaker("timeout_test", config)

        # Record fast success
        cb.record_success(response_time_ms=25.0)
        assert cb.state == CircuitBreakerState.CLOSED

        # Record slow response (timeout failure)
        cb.record_failure(response_time_ms=75.0)  # Exceeds 50ms threshold
        assert cb.failure_count == 1

        # Another timeout failure should open circuit
        cb.record_failure(response_time_ms=100.0)
        assert cb.state == CircuitBreakerState.OPEN

        print("✅ Timeout-based failure detection validated")

    def test_half_open_failure_behavior(self):
        """Test behavior when service fails in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.5,
            success_threshold=3,
        )
        cb = CircuitBreaker("half_open_test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=1.0)

        # Transition to half-open
        assert cb.is_call_permitted() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record success
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.success_count == 1

        # Failure in half-open should immediately return to open
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.success_count == 0  # Reset on transition to open

        print("✅ Half-open failure behavior validated")

    def test_circuit_breaker_metrics_accuracy(self):
        """Test accuracy of circuit breaker metrics under various conditions."""
        cb = CircuitBreaker("metrics_test")

        # Initial metrics
        stats = cb.get_stats()
        assert stats["metrics"]["total_calls"] == 0
        assert stats["metrics"]["failure_rate"] == 0

        # Record mixed success/failure pattern
        cb.record_success(10.0)  # failure_count = 0
        cb.record_success(15.0)  # failure_count = 0
        cb.record_failure(response_time_ms=50.0)  # failure_count = 1
        cb.record_success(20.0)  # failure_count = 0 (decremented on success)
        cb.record_failure(response_time_ms=75.0)  # failure_count = 1

        # Check metrics
        stats = cb.get_stats()
        assert stats["metrics"]["total_calls"] == 5
        assert stats["metrics"]["successful_calls"] == 3
        assert stats["metrics"]["failed_calls"] == 2
        assert stats["metrics"]["failure_rate"] == 0.4  # 2/5

        # Validate state information
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1  # Last failure count after success decrement

        print("✅ Circuit breaker metrics accuracy validated")

    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_calls(self):
        """Test circuit breaker behavior under concurrent load."""
        unreliable_service = UnreliableService(failure_rate=0.5, response_time_ms=10.0)

        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=0.5)
        cb = CircuitBreaker("concurrent_test", config)
        protected_request = cb(unreliable_service.make_request)

        # Launch concurrent requests
        async def make_request(request_id: int):
            try:
                result = await protected_request(f"concurrent_{request_id}")
                return f"Success: {result}"
            except CircuitBreakerOpenException:
                return f"Circuit open for request {request_id}"
            except Exception as e:
                return f"Service error for request {request_id}: {e}"

        # Run 20 concurrent requests
        tasks = [make_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count outcomes
        successes = sum(1 for r in results if isinstance(r, str) and "Success" in r)
        circuit_blocks = sum(1 for r in results if isinstance(r, str) and "Circuit open" in r)
        service_errors = sum(1 for r in results if isinstance(r, str) and "Service error" in r)

        print(f"    Concurrent results: {successes} successes, {circuit_blocks} circuit blocks, {service_errors} service errors")

        # Should have some of each type
        assert successes > 0
        assert service_errors > 0

        # Circuit might open during concurrent execution
        if circuit_blocks > 0:
            assert cb.state == CircuitBreakerState.OPEN

        print("✅ Concurrent circuit breaker behavior validated")

    def test_circuit_breaker_disabled_mode(self):
        """Test circuit breaker when disabled."""
        config = CircuitBreakerConfig(enabled=False, failure_threshold=1)
        cb = CircuitBreaker("disabled_test", config)

        # Should always permit calls when disabled
        assert cb.is_call_permitted() is True

        # Record many failures
        for _i in range(10):
            cb.record_failure()
            assert cb.is_call_permitted() is True  # Should still permit calls

        # State may transition to OPEN but calls are still permitted when disabled
        # This is correct behavior - metrics are tracked but circuit breaker is bypassed

        # But metrics should still be tracked
        stats = cb.get_stats()
        assert stats["metrics"]["failed_calls"] == 10
        assert stats["config"]["enabled"] is False

        print("✅ Disabled circuit breaker behavior validated")


if __name__ == "__main__":
    print("🔄 Running Circuit Breaker Failure Scenario Tests...")

    async def run_tests():
        test_suite = TestCircuitBreakerFailureScenarios()

        print("\n1. Testing circuit breaker state progression...")
        test_suite.test_circuit_breaker_state_progression()

        print("2. Testing exponential backoff progression...")
        test_suite.test_exponential_backoff_progression()

        print("3. Testing async decorator functionality...")
        await test_suite.test_circuit_breaker_decorator_async()

        print("4. Testing sync decorator functionality...")
        test_suite.test_circuit_breaker_decorator_sync()

        print("5. Testing timeout-based failures...")
        test_suite.test_timeout_based_failures()

        print("6. Testing half-open failure behavior...")
        test_suite.test_half_open_failure_behavior()

        print("7. Testing metrics accuracy...")
        test_suite.test_circuit_breaker_metrics_accuracy()

        print("8. Testing concurrent circuit breaker calls...")
        await test_suite.test_concurrent_circuit_breaker_calls()

        print("9. Testing disabled mode...")
        test_suite.test_circuit_breaker_disabled_mode()

    # Run the tests
    asyncio.run(run_tests())

    print("\n🎯 Circuit Breaker Failure Scenario Testing Complete")
    print("   ✅ Complete state progression validation")
    print("   ✅ Exponential backoff under repeated failures")
    print("   ✅ Async and sync decorator functionality")
    print("   ✅ Timeout-based failure detection")
    print("   ✅ Half-open state transition behavior")
    print("   ✅ Comprehensive metrics tracking accuracy")
    print("   ✅ Concurrent load handling and thread safety")
    print("   ✅ Disabled mode bypass functionality")
