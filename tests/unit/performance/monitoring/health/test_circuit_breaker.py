"""
Test Circuit Breaker Implementation with Real Behavior
No mocks - testing actual circuit breaker behavior
"""

import asyncio

import pytest

from prompt_improver.performance.monitoring.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitBreaker:
    """Test circuit breaker with real behavior"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_failures(self):
        """Test that circuit breaker opens after failure threshold"""
        config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=1, response_time_threshold_ms=100
        )
        breaker = CircuitBreaker("test_service", config)
        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Intentional failure {call_count}")

        assert breaker.state == CircuitState.CLOSED
        for _i in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_function)
        assert breaker.state == CircuitState.OPEN
        assert breaker._failure_count == config.failure_threshold
        initial_call_count = call_count
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            await breaker.call(failing_function)
        assert call_count == initial_call_count
        assert "Circuit breaker 'test_service' is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_behavior(self):
        """Test circuit breaker recovery from open to half-open to closed"""
        config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.5, half_open_max_calls=2
        )
        breaker = CircuitBreaker("recovery_test", config)
        should_fail = True

        async def controllable_function():
            if should_fail:
                raise RuntimeError("Controlled failure")
            return "success"

        for _ in range(config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(controllable_function)
        assert breaker.state == CircuitState.OPEN
        await asyncio.sleep(config.recovery_timeout + 0.1)
        assert breaker.state == CircuitState.HALF_OPEN
        should_fail = False
        for _ in range(config.half_open_max_calls):
            result = await breaker.call(controllable_function)
            assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_slow_response_counts_as_failure(self):
        """Test that slow responses are treated as failures"""
        config = CircuitBreakerConfig(
            failure_threshold=3, response_time_threshold_ms=50
        )
        breaker = CircuitBreaker("slow_test", config)

        async def slow_function():
            await asyncio.sleep(0.1)
            return "slow_success"

        for _i in range(config.failure_threshold):
            result = await breaker.call(slow_function)
            assert result == "slow_success"
        assert breaker.state == CircuitState.OPEN
        metrics = breaker.get_metrics()
        assert metrics["successful_calls"] == config.failure_threshold
        assert metrics["failed_calls"] == 0
        assert metrics["failure_count"] == config.failure_threshold

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

        test_pattern = [False, False, True, False, True, True, False]
        for should_fail in test_pattern:
            try:
                await breaker.call(mixed_function, should_fail)
                success_count += 1
            except Exception:
                failure_count += 1
        metrics = breaker.get_metrics()
        assert metrics["total_calls"] == len(test_pattern)
        assert metrics["successful_calls"] == success_count
        assert metrics["failed_calls"] == failure_count
        assert metrics["rejected_calls"] == 0
        expected_rate = success_count / len(test_pattern)
        assert abs(metrics["success_rate"] - expected_rate) < 0.01

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_transitions(self):
        """Test all possible state transitions"""
        config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.3, half_open_max_calls=1
        )
        breaker = CircuitBreaker("transition_test", config)
        state_changes = []

        def record_state_change(name, new_state):
            state_changes.append((name, new_state))

        breaker.on_state_change = record_state_change

        async def failing_func():
            raise Exception("fail")

        async def success_func():
            return "ok"

        assert breaker.state == CircuitState.CLOSED
        for _ in range(config.failure_threshold):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        assert breaker.state == CircuitState.OPEN
        assert ("transition_test", CircuitState.OPEN) in state_changes
        await asyncio.sleep(config.recovery_timeout + 0.1)
        _ = breaker.state
        assert breaker.state == CircuitState.HALF_OPEN
        assert ("transition_test", CircuitState.HALF_OPEN) in state_changes
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        assert breaker.state == CircuitState.OPEN
        await asyncio.sleep(config.recovery_timeout + 0.1)
        _ = breaker.state
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
        breaker1 = registry.get_or_create("service1")
        assert breaker1.name == "service1"
        breaker2 = registry.get_or_create("service1")
        assert breaker1 is breaker2
        breaker3 = registry.get_or_create("service2")
        assert breaker3.name == "service2"
        assert breaker3 is not breaker1

    def test_registry_metrics_aggregation(self):
        """Test that registry correctly aggregates metrics"""
        registry = CircuitBreakerRegistry()
        breaker1 = registry.get_or_create("api_service")
        breaker2 = registry.get_or_create("db_service")
        breaker1._call_metrics["total_calls"] = 100
        breaker1._call_metrics["successful_calls"] = 95
        breaker2._call_metrics["total_calls"] = 50
        breaker2._call_metrics["successful_calls"] = 48
        all_metrics = registry.get_all_metrics()
        assert "api_service" in all_metrics
        assert "db_service" in all_metrics
        assert all_metrics["api_service"]["total_calls"] == 100
        assert all_metrics["db_service"]["total_calls"] == 50

    @pytest.mark.asyncio
    async def test_registry_reset_all(self):
        """Test that reset_all properly resets all breakers"""
        registry = CircuitBreakerRegistry()
        for i in range(3):
            breaker = registry.get_or_create(f"service_{i}")
            breaker._state = CircuitState.OPEN
            breaker._failure_count = 5
        registry.reset_all()
        for i in range(3):
            breaker = registry.get_or_create(f"service_{i}")
            assert breaker.state == CircuitState.CLOSED
            assert breaker._failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_thread_safety():
    """Test circuit breaker behavior under concurrent access"""
    config = CircuitBreakerConfig(failure_threshold=10, response_time_threshold_ms=50)
    breaker = CircuitBreaker("concurrent_test", config)
    call_results = {"success": 0, "failure": 0, "rejected": 0}

    async def concurrent_task(task_id: int):
        for i in range(20):
            try:

                async def work():
                    if i % 3 == 0:
                        await asyncio.sleep(0.06)
                    else:
                        await asyncio.sleep(0.01)
                    if i % 5 == 0:
                        raise Exception(f"Task {task_id} iteration {i} failed")
                    return f"success_{task_id}_{i}"

                result = await breaker.call(work)
                call_results["success"] += 1
            except CircuitBreakerOpen:
                call_results["rejected"] += 1
            except Exception:
                call_results["failure"] += 1
            await asyncio.sleep(0.01)

    tasks = [concurrent_task(i) for i in range(5)]
    await asyncio.gather(*tasks)
    total_calls = sum(call_results.values())
    assert total_calls == 100
    metrics = breaker.get_metrics()
    assert metrics["total_calls"] + metrics["rejected_calls"] == total_calls
    assert call_results["rejected"] > 0
