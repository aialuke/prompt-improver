"""
Test Circuit Breaker Implementation with Real Behavior
No mocks - testing actual circuit breaker behavior
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock

from prompt_improver.performance.monitoring.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpen,
    CircuitBreakerRegistry
)


class TestCircuitBreaker:
    """Test circuit breaker with real behavior"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_failures(self):
        """Test that circuit breaker opens after failure threshold"""
        # Real configuration
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,  # 1 second for faster testing
            response_time_threshold_ms=100
        )
        
        breaker = CircuitBreaker("test_service", config)
        
        # Define a real failing function
        call_count = 0
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Intentional failure {call_count}")
        
        # Circuit should be closed initially
        assert breaker.state == CircuitState.CLOSED
        
        # Make calls that fail
        for i in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_function)
        
        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN
        assert breaker._failure_count == config.failure_threshold
        
        # Next call should be rejected without executing function
        initial_call_count = call_count
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            await breaker.call(failing_function)
        
        # Verify function was not called
        assert call_count == initial_call_count
        assert "Circuit breaker 'test_service' is OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_behavior(self):
        """Test circuit breaker recovery from open to half-open to closed"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,  # 500ms for faster testing
            half_open_max_calls=2
        )
        
        breaker = CircuitBreaker("recovery_test", config)
        
        # Function that can be controlled to fail or succeed
        should_fail = True
        async def controllable_function():
            if should_fail:
                raise RuntimeError("Controlled failure")
            return "success"
        
        # Open the circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(controllable_function)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(config.recovery_timeout + 0.1)
        
        # Circuit should transition to half-open on next access
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Make function succeed now
        should_fail = False
        
        # Need successful calls in half-open to close circuit
        for _ in range(config.half_open_max_calls):
            result = await breaker.call(controllable_function)
            assert result == "success"
        
        # Circuit should be closed now
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
    
    @pytest.mark.asyncio
    async def test_slow_response_counts_as_failure(self):
        """Test that slow responses are treated as failures"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            response_time_threshold_ms=50  # 50ms threshold
        )
        
        breaker = CircuitBreaker("slow_test", config)
        
        async def slow_function():
            await asyncio.sleep(0.1)  # 100ms - exceeds threshold
            return "slow_success"
        
        # Execute slow calls
        for i in range(config.failure_threshold):
            result = await breaker.call(slow_function)
            assert result == "slow_success"  # Call succeeds but is recorded as slow
        
        # Circuit should open due to slow responses
        assert breaker.state == CircuitState.OPEN
        
        # Verify metrics show the calls
        metrics = breaker.get_metrics()
        assert metrics["successful_calls"] == config.failure_threshold
        assert metrics["failed_calls"] == 0  # No actual failures
        assert metrics["failure_count"] == config.failure_threshold  # But counted as failures
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_accuracy(self):
        """Test that metrics are accurately tracked"""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("metrics_test", config)
        
        success_count = 0
        failure_count = 0
        
        async def mixed_function(should_fail: bool):
            if should_fail:
                raise Exception("Planned failure")
            return "success"
        
        # Mix of successes and failures
        test_pattern = [False, False, True, False, True, True, False]
        
        for should_fail in test_pattern:
            try:
                await breaker.call(mixed_function, should_fail)
                success_count += 1
            except Exception:
                failure_count += 1
        
        metrics = breaker.get_metrics()
        
        # Verify accurate tracking
        assert metrics["total_calls"] == len(test_pattern)
        assert metrics["successful_calls"] == success_count
        assert metrics["failed_calls"] == failure_count
        assert metrics["rejected_calls"] == 0  # Circuit didn't open
        
        # Verify success rate calculation
        expected_rate = success_count / len(test_pattern)
        assert abs(metrics["success_rate"] - expected_rate) < 0.01
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_state_transitions(self):
        """Test all possible state transitions"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.3,
            half_open_max_calls=1
        )
        
        breaker = CircuitBreaker("transition_test", config)
        
        # Track state changes
        state_changes = []
        def record_state_change(name, new_state):
            state_changes.append((name, new_state))
        
        breaker.on_state_change = record_state_change
        
        async def failing_func():
            raise Exception("fail")
        
        async def success_func():
            return "ok"
        
        # CLOSED -> OPEN (via failures)
        assert breaker.state == CircuitState.CLOSED
        
        for _ in range(config.failure_threshold):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        assert ("transition_test", CircuitState.OPEN) in state_changes
        
        # OPEN -> HALF_OPEN (via timeout)
        await asyncio.sleep(config.recovery_timeout + 0.1)
        _ = breaker.state  # Trigger state check
        
        assert breaker.state == CircuitState.HALF_OPEN
        assert ("transition_test", CircuitState.HALF_OPEN) in state_changes
        
        # HALF_OPEN -> OPEN (via failure)
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # OPEN -> HALF_OPEN -> CLOSED (via success)
        await asyncio.sleep(config.recovery_timeout + 0.1)
        _ = breaker.state  # Trigger transition
        assert breaker.state == CircuitState.HALF_OPEN
        
        result = await breaker.call(success_func)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED
        assert ("transition_test", CircuitState.CLOSED) in state_changes


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry behavior"""
    
    def test_registry_creates_unique_breakers(self):
        """Test that registry creates and reuses breakers correctly"""
        registry = CircuitBreakerRegistry()
        
        # Create first breaker
        breaker1 = registry.get_or_create("service1")
        assert breaker1.name == "service1"
        
        # Get same breaker
        breaker2 = registry.get_or_create("service1")
        assert breaker1 is breaker2  # Same instance
        
        # Create different breaker
        breaker3 = registry.get_or_create("service2")
        assert breaker3.name == "service2"
        assert breaker3 is not breaker1
    
    def test_registry_metrics_aggregation(self):
        """Test that registry correctly aggregates metrics"""
        registry = CircuitBreakerRegistry()
        
        # Create multiple breakers
        breaker1 = registry.get_or_create("api_service")
        breaker2 = registry.get_or_create("db_service")
        
        # Simulate some activity
        breaker1._call_metrics["total_calls"] = 100
        breaker1._call_metrics["successful_calls"] = 95
        
        breaker2._call_metrics["total_calls"] = 50
        breaker2._call_metrics["successful_calls"] = 48
        
        # Get all metrics
        all_metrics = registry.get_all_metrics()
        
        assert "api_service" in all_metrics
        assert "db_service" in all_metrics
        assert all_metrics["api_service"]["total_calls"] == 100
        assert all_metrics["db_service"]["total_calls"] == 50
    
    @pytest.mark.asyncio
    async def test_registry_reset_all(self):
        """Test that reset_all properly resets all breakers"""
        registry = CircuitBreakerRegistry()
        
        # Create and open multiple breakers
        for i in range(3):
            breaker = registry.get_or_create(f"service_{i}")
            breaker._state = CircuitState.OPEN
            breaker._failure_count = 5
        
        # Reset all
        registry.reset_all()
        
        # Verify all are reset
        for i in range(3):
            breaker = registry.get_or_create(f"service_{i}")
            assert breaker.state == CircuitState.CLOSED
            assert breaker._failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_thread_safety():
    """Test circuit breaker behavior under concurrent access"""
    config = CircuitBreakerConfig(
        failure_threshold=10,
        response_time_threshold_ms=50
    )
    
    breaker = CircuitBreaker("concurrent_test", config)
    
    call_results = {"success": 0, "failure": 0, "rejected": 0}
    
    async def concurrent_task(task_id: int):
        for i in range(20):
            try:
                # Alternate between fast and slow calls
                async def work():
                    if i % 3 == 0:
                        await asyncio.sleep(0.06)  # Slow call
                    else:
                        await asyncio.sleep(0.01)  # Fast call
                    
                    if i % 5 == 0:
                        raise Exception(f"Task {task_id} iteration {i} failed")
                    return f"success_{task_id}_{i}"
                
                result = await breaker.call(work)
                call_results["success"] += 1
                
            except CircuitBreakerOpen:
                call_results["rejected"] += 1
            except Exception:
                call_results["failure"] += 1
            
            # Small delay between calls
            await asyncio.sleep(0.01)
    
    # Run multiple concurrent tasks
    tasks = [concurrent_task(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    # Verify results make sense
    total_calls = sum(call_results.values())
    assert total_calls == 100  # 5 tasks * 20 calls each
    
    # Check metrics consistency
    metrics = breaker.get_metrics()
    assert metrics["total_calls"] + metrics["rejected_calls"] == total_calls
    
    # Circuit should have opened at some point due to failures
    assert call_results["rejected"] > 0