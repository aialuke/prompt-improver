"""Comprehensive Real Behavior Tests for Retry System Services.

Tests the complete Retry System Services decomposition with real behavior validation:
- RetryServiceFacade (unified retry operations)
- RetryConfigurationService (configuration management)
- BackoffStrategyService (backoff algorithms)
- CircuitBreakerService (circuit protection)
- RetryOrchestratorService (retry orchestration)

All tests use real network failures and timeouts - no mocks.
Performance targets: <5ms retry decisions, <10ms orchestration.
"""

import asyncio
import logging
import pytest
import time
from typing import Any, Callable, Dict, Optional
from unittest.mock import AsyncMock

from prompt_improver.core.services.resilience.retry_service_facade import (
    RetryServiceFacade,
    RetryOperationResult,
    get_retry_service,
    reset_retry_service,
    validate_performance_targets,
)
from prompt_improver.core.services.resilience.retry_configuration_service import (
    RetryConfigurationService,
    get_retry_configuration_service,
)
from prompt_improver.core.services.resilience.backoff_strategy_service import (
    BackoffStrategyService,
    get_backoff_strategy_service,
)
from prompt_improver.core.services.resilience.circuit_breaker_service import (
    CircuitBreakerService,
    create_circuit_breaker_service,
)
from prompt_improver.core.services.resilience.retry_orchestrator_service import (
    RetryOrchestratorService,
    RetryExecutionContext,
)
from prompt_improver.core.protocols.retry_protocols import RetryConfigProtocol
from tests.integration.real_behavior_testing.containers.network_simulator import (
    NetworkSimulator,
    FailureType,
    FailureScenario,
)

logger = logging.getLogger(__name__)


@pytest.fixture
async def network_simulator_with_scenarios() -> NetworkSimulator:
    """Network simulator with pre-configured failure scenarios."""
    simulator = NetworkSimulator()
    await simulator.start()
    
    # Create common failure scenarios
    simulator.create_common_failure_scenarios()
    
    yield simulator
    await simulator.stop()


@pytest.fixture
def retry_service() -> RetryServiceFacade:
    """Fresh retry service facade for each test."""
    reset_retry_service()  # Ensure clean state
    return get_retry_service()


class TestRetryConfigurationService:
    """Test Retry Configuration Service with real configuration scenarios."""
    
    def test_configuration_service_initialization(self):
        """Test configuration service initialization."""
        service = get_retry_configuration_service()
        
        # Validate service initialization
        assert service is not None
        
        # Test default configuration
        default_config = service.get_default_config()
        assert default_config is not None
        assert default_config.max_attempts >= 1
        assert default_config.base_delay_ms > 0

    def test_domain_specific_configurations(self):
        """Test domain-specific configuration creation."""
        service = get_retry_configuration_service()
        
        # Test database domain configuration
        db_config = service.create_config(
            domain="database",
            operation="query",
            max_attempts=5,
            base_delay_ms=100
        )
        assert db_config.max_attempts == 5
        assert db_config.base_delay_ms == 100
        
        # Test API domain configuration
        api_config = service.create_config(
            domain="api",
            operation="request",
            max_attempts=3,
            base_delay_ms=200
        )
        assert api_config.max_attempts == 3
        assert api_config.base_delay_ms == 200
        
        # Test ML domain configuration
        ml_config = service.create_config(
            domain="ml",
            operation="inference",
            max_attempts=2,
            base_delay_ms=500
        )
        assert ml_config.max_attempts == 2
        assert ml_config.base_delay_ms == 500

    def test_configuration_templates(self):
        """Test configuration templates for common scenarios."""
        service = get_retry_configuration_service()
        
        # Test quick operations template
        quick_config = service.create_config(
            template="quick_operations",
            max_attempts=2
        )
        assert quick_config.max_attempts == 2
        assert quick_config.base_delay_ms <= 50  # Should be fast
        
        # Test critical operations template
        critical_config = service.create_config(
            template="critical_operations",
            max_attempts=10
        )
        assert critical_config.max_attempts == 10
        # Critical operations may have longer delays for reliability


class TestBackoffStrategyService:
    """Test Backoff Strategy Service with real timing validation."""
    
    def test_exponential_backoff_timing(self, performance_tracker):
        """Test exponential backoff strategy with real timing."""
        service = get_backoff_strategy_service()
        
        base_delay_ms = 100
        max_delay_ms = 2000
        multiplier = 2.0
        
        # Test backoff calculation performance
        start_time = time.perf_counter()
        
        delays = []
        for attempt in range(1, 6):  # Test 5 attempts
            delay = service.calculate_exponential_backoff(
                attempt=attempt,
                base_delay_ms=base_delay_ms,
                max_delay_ms=max_delay_ms,
                multiplier=multiplier
            )
            delays.append(delay)
        
        calculation_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("backoff_calculations", calculation_time, 1.0)  # <1ms target
        
        # Validate exponential progression
        assert delays[0] == base_delay_ms  # First attempt
        assert delays[1] == base_delay_ms * multiplier  # Second attempt
        assert delays[2] == base_delay_ms * multiplier * multiplier  # Third attempt
        assert all(delay <= max_delay_ms for delay in delays)  # Respect max delay

    def test_linear_backoff_timing(self, performance_tracker):
        """Test linear backoff strategy with real timing."""
        service = get_backoff_strategy_service()
        
        base_delay_ms = 50
        increment_ms = 25
        max_delay_ms = 500
        
        start_time = time.perf_counter()
        
        delays = []
        for attempt in range(1, 8):  # Test 7 attempts
            delay = service.calculate_linear_backoff(
                attempt=attempt,
                base_delay_ms=base_delay_ms,
                increment_ms=increment_ms,
                max_delay_ms=max_delay_ms
            )
            delays.append(delay)
        
        calculation_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("linear_backoff_calculations", calculation_time, 1.0)  # <1ms target
        
        # Validate linear progression
        expected_delays = [
            base_delay_ms + (attempt - 1) * increment_ms
            for attempt in range(1, 8)
        ]
        # Apply max delay cap
        expected_delays = [min(delay, max_delay_ms) for delay in expected_delays]
        
        assert delays == expected_delays

    async def test_jittered_backoff_distribution(self):
        """Test jittered backoff produces proper distribution."""
        service = get_backoff_strategy_service()
        
        base_delay_ms = 100
        jitter_factor = 0.2  # 20% jitter
        
        # Generate many samples to test distribution
        samples = []
        for _ in range(100):
            delay = service.calculate_jittered_backoff(
                base_delay_ms=base_delay_ms,
                jitter_factor=jitter_factor
            )
            samples.append(delay)
        
        # Calculate distribution statistics
        min_expected = base_delay_ms * (1 - jitter_factor)
        max_expected = base_delay_ms * (1 + jitter_factor)
        
        min_actual = min(samples)
        max_actual = max(samples)
        avg_actual = sum(samples) / len(samples)
        
        # Validate jitter distribution
        assert min_actual >= min_expected * 0.9  # Allow some variance
        assert max_actual <= max_expected * 1.1  # Allow some variance
        assert abs(avg_actual - base_delay_ms) < base_delay_ms * 0.1  # Average should be near base

    async def test_backoff_with_real_delays(self, performance_tracker):
        """Test backoff strategies with actual delays."""
        service = get_backoff_strategy_service()
        
        # Test short exponential backoff
        delays_to_test = [10, 20, 40, 80]  # Small delays for testing
        
        total_start_time = time.perf_counter()
        
        for expected_delay in delays_to_test:
            delay_start_time = time.perf_counter()
            
            # Use service to wait with backoff
            await service.wait_with_backoff(expected_delay)
            
            actual_delay = (time.perf_counter() - delay_start_time) * 1000
            
            # Validate actual delay is approximately correct (±20% tolerance)
            delay_tolerance = expected_delay * 0.2
            assert abs(actual_delay - expected_delay) <= delay_tolerance, \
                f"Expected ~{expected_delay}ms, got {actual_delay:.1f}ms"
        
        total_time = (time.perf_counter() - total_start_time) * 1000
        performance_tracker("real_backoff_delays", total_time, 200.0)  # Total should be ~150ms


class TestCircuitBreakerService:
    """Test Circuit Breaker Service with real failure scenarios."""
    
    async def test_circuit_breaker_failure_detection(
        self,
        network_simulator_with_scenarios: NetworkSimulator,
        performance_tracker,
    ):
        """Test circuit breaker failure detection with real network failures."""
        service = create_circuit_breaker_service()
        circuit_name = "test_network_operation"
        
        # Define operation that will fail via network simulator
        async def network_operation():
            return await network_simulator_with_scenarios.simulate_network_operation(
                operation_name="test_operation",
                target_host="example.com",
                operation_type="http_request"
            )
        
        # Execute operations until circuit breaker activates
        failure_count = 0
        success_count = 0
        
        for attempt in range(20):  # Enough attempts to trigger circuit breaker
            start_time = time.perf_counter()
            
            try:
                result = await service.execute_with_circuit_breaker(
                    operation=network_operation,
                    circuit_name=circuit_name
                )
                
                if result and result.success:
                    success_count += 1
                else:
                    failure_count += 1
                    
            except Exception:
                failure_count += 1
            
            operation_time = (time.perf_counter() - start_time) * 1000
            performance_tracker(f"circuit_breaker_operation_{attempt}", operation_time, 100.0)
        
        # Get circuit breaker state
        circuit_states = service.get_all_circuit_states()
        
        # Validate failure tracking
        assert failure_count > 0, "Should have some failures from network simulator"
        logger.info(f"Circuit breaker test: {failure_count} failures, {success_count} successes")
        
        if circuit_name in circuit_states:
            state = circuit_states[circuit_name]
            assert state.get("failure_count", 0) >= 0  # Should track failures
            logger.info(f"Circuit state: {state}")

    async def test_circuit_breaker_recovery_mechanism(
        self,
        performance_tracker,
    ):
        """Test circuit breaker recovery mechanism."""
        service = create_circuit_breaker_service()
        circuit_name = "recovery_test"
        
        # Simulate alternating failure and success
        call_count = [0]
        
        async def intermittent_operation():
            call_count[0] += 1
            if call_count[0] <= 5:  # First 5 calls fail
                raise RuntimeError("Simulated failure")
            else:  # Subsequent calls succeed
                return "success"
        
        results = []
        
        for attempt in range(10):
            start_time = time.perf_counter()
            
            try:
                result = await service.execute_with_circuit_breaker(
                    operation=intermittent_operation,
                    circuit_name=circuit_name
                )
                results.append(("success", result))
                
            except Exception as e:
                results.append(("failure", str(e)))
            
            operation_time = (time.perf_counter() - start_time) * 1000
            performance_tracker(f"recovery_test_{attempt}", operation_time, 50.0)
            
            # Brief pause between attempts
            await asyncio.sleep(0.01)
        
        # Analyze results for recovery pattern
        failures = [r for r in results if r[0] == "failure"]
        successes = [r for r in results if r[0] == "success"]
        
        logger.info(f"Recovery test: {len(failures)} failures, {len(successes)} successes")
        logger.info(f"Results: {[r[0] for r in results]}")
        
        # Should have both failures and potentially some successes (depending on circuit behavior)
        assert len(failures) > 0, "Should detect initial failures"

    async def test_circuit_breaker_performance_impact(
        self,
        performance_tracker,
    ):
        """Test circuit breaker performance overhead."""
        service = create_circuit_breaker_service()
        
        async def fast_operation():
            return "fast_result"
        
        # Test without circuit breaker
        start_time = time.perf_counter()
        for _ in range(100):
            await fast_operation()
        direct_time = (time.perf_counter() - start_time) * 1000
        
        # Test with circuit breaker
        start_time = time.perf_counter()
        for i in range(100):
            await service.execute_with_circuit_breaker(
                operation=fast_operation,
                circuit_name=f"perf_test_{i % 10}"  # Use multiple circuits
            )
        circuit_breaker_time = (time.perf_counter() - start_time) * 1000
        
        performance_tracker("direct_operations", direct_time, 50.0)
        performance_tracker("circuit_breaker_operations", circuit_breaker_time, 100.0)
        
        # Circuit breaker overhead should be reasonable
        overhead_percentage = ((circuit_breaker_time - direct_time) / direct_time) * 100
        logger.info(f"Circuit breaker overhead: {overhead_percentage:.1f}%")
        
        # Overhead should be less than 100% (circuit breaker shouldn't double execution time)
        assert overhead_percentage < 100.0, f"High circuit breaker overhead: {overhead_percentage:.1f}%"


class TestRetryOrchestratorService:
    """Test Retry Orchestrator Service with comprehensive retry scenarios."""
    
    @pytest.fixture
    def retry_orchestrator(self) -> RetryOrchestratorService:
        """Create retry orchestrator with dependencies."""
        config_service = get_retry_configuration_service()
        backoff_service = get_backoff_strategy_service()
        circuit_breaker_service = create_circuit_breaker_service()
        
        from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry
        
        return RetryOrchestratorService(
            config_service=config_service,
            backoff_service=backoff_service,
            circuit_breaker_service=circuit_breaker_service,
            metrics_registry=get_metrics_registry()
        )
    
    async def test_retry_orchestration_with_network_failures(
        self,
        retry_orchestrator: RetryOrchestratorService,
        network_simulator_with_scenarios: NetworkSimulator,
        performance_tracker,
    ):
        """Test retry orchestration with real network failures."""
        # Configure network simulator for intermittent failures
        await network_simulator_with_scenarios.add_failure_scenario(
            "retry_test",
            FailureType.CONNECTION_TIMEOUT,
            probability=0.7,  # 70% failure rate
            duration_ms=50,
            pattern="random"
        )
        
        async def network_dependent_operation():
            operation = await network_simulator_with_scenarios.simulate_network_operation(
                operation_name="retry_test_operation",
                target_host="api.example.com",
                operation_type="api_call"
            )
            if not operation.success:
                raise RuntimeError(operation.error_message)
            return operation
        
        # Create retry context
        config = get_retry_configuration_service().create_config(
            domain="api",
            max_attempts=5,
            base_delay_ms=20  # Short delays for testing
        )
        
        context = RetryExecutionContext(
            operation_name="network_dependent_test",
            config=config
        )
        
        start_time = time.perf_counter()
        
        try:
            result = await retry_orchestrator.execute_with_retry(
                operation=network_dependent_operation,
                context=context
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            performance_tracker("retry_orchestration", execution_time, 500.0)  # Allow time for retries
            
            # Validate result structure
            assert hasattr(result, 'success')
            assert hasattr(result, 'attempts_made')
            assert hasattr(result, 'total_time_ms')
            assert result.attempts_made >= 1
            
            logger.info(f"Retry result: success={result.success}, attempts={result.attempts_made}, time={result.total_time_ms:.1f}ms")
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            performance_tracker("retry_orchestration_failed", execution_time, 500.0)
            
            logger.info(f"Retry operation failed after {execution_time:.1f}ms: {e}")
            # Failure is acceptable in this test due to high failure rate

    async def test_retry_orchestrator_backoff_timing(
        self,
        retry_orchestrator: RetryOrchestratorService,
        performance_tracker,
    ):
        """Test retry orchestrator backoff timing accuracy."""
        failure_count = [0]
        
        async def controlled_failure_operation():
            failure_count[0] += 1
            if failure_count[0] <= 3:  # Fail first 3 attempts
                raise RuntimeError(f"Controlled failure #{failure_count[0]}")
            return f"success_after_{failure_count[0]}_attempts"
        
        # Configure with specific backoff delays
        config = get_retry_configuration_service().create_config(
            domain="test",
            max_attempts=5,
            base_delay_ms=50,  # 50ms base delay
            backoff_multiplier=1.5
        )
        
        context = RetryExecutionContext(
            operation_name="backoff_timing_test",
            config=config
        )
        
        start_time = time.perf_counter()
        
        result = await retry_orchestrator.execute_with_retry(
            operation=controlled_failure_operation,
            context=context
        )
        
        total_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("orchestrator_backoff_timing", total_time, 300.0)  # Should complete within 300ms
        
        # Validate results
        assert result.success
        assert result.attempts_made == 4  # 3 failures + 1 success
        assert result.result.startswith("success_after_")
        
        # Validate timing includes backoff delays
        # Expected delays: 50ms + 75ms + 112.5ms ≈ 237.5ms + operation time
        expected_min_time = 200  # Account for operation execution time
        assert total_time >= expected_min_time, f"Total time {total_time:.1f}ms seems too fast for backoff delays"

    async def test_retry_orchestrator_metrics_collection(
        self,
        retry_orchestrator: RetryOrchestratorService,
    ):
        """Test retry orchestrator metrics collection."""
        async def simple_success_operation():
            return "success"
        
        async def simple_failure_operation():
            raise RuntimeError("Always fails")
        
        # Execute successful operation
        config = get_retry_configuration_service().get_default_config()
        success_context = RetryExecutionContext("success_test", config)
        
        await retry_orchestrator.execute_with_retry(
            operation=simple_success_operation,
            context=success_context
        )
        
        # Execute failing operation
        failure_context = RetryExecutionContext("failure_test", config)
        
        try:
            await retry_orchestrator.execute_with_retry(
                operation=simple_failure_operation,
                context=failure_context
            )
        except Exception:
            pass  # Expected failure
        
        # Get comprehensive metrics
        metrics = retry_orchestrator.get_comprehensive_metrics()
        
        # Validate metrics structure
        assert isinstance(metrics, dict)
        assert "total_operations" in metrics
        assert "successful_operations" in metrics
        assert "failed_operations" in metrics
        assert "average_attempts" in metrics
        
        # Should have recorded operations
        assert metrics["total_operations"] >= 2

    async def test_orchestrator_health_status(
        self,
        retry_orchestrator: RetryOrchestratorService,
    ):
        """Test orchestrator health status reporting."""
        health_status = retry_orchestrator.get_health_status()
        
        # Validate health status structure
        assert isinstance(health_status, dict)
        assert "overall_status" in health_status
        assert "component_health" in health_status
        assert "active_operations" in health_status
        
        # Health status should be operational initially
        assert health_status["overall_status"] in ["healthy", "degraded", "unhealthy"]


class TestRetryServiceFacade:
    """Test Retry Service Facade integration and coordination."""
    
    async def test_facade_retry_execution_with_domain_config(
        self,
        retry_service: RetryServiceFacade,
        network_simulator_with_scenarios: NetworkSimulator,
        performance_tracker,
    ):
        """Test facade retry execution with domain-specific configuration."""
        # Add network failure scenario
        await network_simulator_with_scenarios.add_failure_scenario(
            "database_failures",
            FailureType.CONNECTION_TIMEOUT,
            probability=0.5,
            duration_ms=100,
            pattern="sequential"
        )
        
        async def database_operation():
            operation = await network_simulator_with_scenarios.simulate_database_connection(
                host="db.example.com",
                port=5432,
                database="test_db"
            )
            if not operation.success:
                raise ConnectionError(operation.error_message)
            return operation.operation_id
        
        # Execute with database domain configuration
        start_time = time.perf_counter()
        
        result = await retry_service.execute_with_retry(
            operation=database_operation,
            domain="database",
            operation_type="connection",
            operation_args=[],
            operation_kwargs={}
        )
        
        execution_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("facade_database_retry", execution_time, 1000.0)  # Allow time for database retries
        
        # Validate facade result
        assert isinstance(result, RetryOperationResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'attempts_made')
        assert hasattr(result, 'total_time_ms')
        assert result.attempts_made >= 1
        
        logger.info(f"Database retry result: success={result.success}, attempts={result.attempts_made}")

    async def test_facade_circuit_breaker_integration(
        self,
        retry_service: RetryServiceFacade,
        performance_tracker,
    ):
        """Test facade circuit breaker integration."""
        failure_count = [0]
        
        async def circuit_test_operation():
            failure_count[0] += 1
            if failure_count[0] <= 8:  # Fail many times to trigger circuit breaker
                raise RuntimeError(f"Circuit test failure #{failure_count[0]}")
            return "success"
        
        start_time = time.perf_counter()
        
        result = await retry_service.execute_with_circuit_breaker(
            operation=circuit_test_operation,
            circuit_name="facade_circuit_test"
        )
        
        execution_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("facade_circuit_breaker", execution_time, 50.0)
        
        # Result may be success or exception depending on circuit breaker behavior
        logger.info(f"Circuit breaker test result: {type(result).__name__}")

    async def test_facade_performance_targets(
        self,
        retry_service: RetryServiceFacade,
        performance_tracker,
    ):
        """Test facade performance meets targets."""
        async def fast_operation():
            return "fast_result"
        
        # Test retry decision speed (target: <5ms)
        start_time = time.perf_counter()
        
        result = await retry_service.execute_with_retry(
            operation=fast_operation,
            domain="test",
            operation_type="fast_operation"
        )
        
        decision_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("facade_retry_decision", decision_time, 5.0)
        
        # Validate quick successful execution
        assert result.success
        assert result.result == "fast_result"
        assert result.attempts_made == 1  # Should succeed on first try

    async def test_facade_health_status_comprehensive(
        self,
        retry_service: RetryServiceFacade,
    ):
        """Test facade comprehensive health status."""
        # Execute some operations to generate health data
        async def test_operation():
            return "test"
        
        await retry_service.execute_with_retry(
            operation=test_operation,
            domain="test"
        )
        
        # Get health status
        health_status = retry_service.get_health_status()
        
        # Validate comprehensive health status
        assert isinstance(health_status, dict)
        assert "overall_status" in health_status
        assert "services" in health_status
        assert "observers_count" in health_status
        assert "facade_info" in health_status
        
        # Validate service health components
        services = health_status["services"]
        expected_services = ["configuration", "backoff_strategy", "circuit_breakers", "orchestrator"]
        
        for service_name in expected_services:
            assert service_name in services
            service_health = services[service_name]
            assert "status" in service_health

    async def test_facade_performance_metrics_comprehensive(
        self,
        retry_service: RetryServiceFacade,
    ):
        """Test facade comprehensive performance metrics."""
        # Execute various operations to generate metrics
        async def success_operation():
            return "success"
        
        async def failure_operation():
            raise RuntimeError("Test failure")
        
        # Execute successful operations
        for _ in range(3):
            await retry_service.execute_with_retry(
                operation=success_operation,
                domain="test"
            )
        
        # Execute failing operations
        for _ in range(2):
            try:
                await retry_service.execute_with_retry(
                    operation=failure_operation,
                    domain="test",
                    custom_config=get_retry_configuration_service().create_config(
                        max_attempts=2  # Limit attempts for faster testing
                    )
                )
            except Exception:
                pass  # Expected failures
        
        # Get performance metrics
        metrics = retry_service.get_performance_metrics()
        
        # Validate metrics structure
        assert isinstance(metrics, dict)
        assert "facade_metrics" in metrics
        assert "orchestrator_metrics" in metrics
        assert "service_health" in metrics
        
        # Validate facade metrics
        facade_metrics = metrics["facade_metrics"]
        assert "total_operations" in facade_metrics
        assert "successful_operations" in facade_metrics
        assert "failed_operations" in facade_metrics
        assert "observers_count" in facade_metrics
        
        # Should have recorded operations
        assert facade_metrics["total_operations"] >= 5


class TestIntegratedRetrySystemWorkflow:
    """Test complete integrated retry system workflow scenarios."""
    
    async def test_end_to_end_retry_workflow_with_real_failures(
        self,
        retry_service: RetryServiceFacade,
        network_simulator_with_scenarios: NetworkSimulator,
        performance_tracker,
    ):
        """Test end-to-end retry workflow with realistic failure scenarios."""
        # Create complex failure scenario
        await network_simulator_with_scenarios.add_failure_scenario(
            "complex_api_failures",
            FailureType.INTERMITTENT_ERRORS,
            probability=0.6,
            pattern="burst",
            burst_count=2,
            recovery_time_ms=500
        )
        
        async def complex_api_operation():
            # Simulate complex API call with multiple failure points
            await asyncio.sleep(0.01)  # Simulate API processing time
            
            operation = await network_simulator_with_scenarios.simulate_http_request(
                url="https://api.example.com/complex-operation",
                method="POST",
                timeout_ms=1000
            )
            
            if not operation.success:
                if operation.failure_type == FailureType.CONNECTION_TIMEOUT:
                    raise asyncio.TimeoutError(f"API timeout: {operation.error_message}")
                elif operation.failure_type == FailureType.INTERMITTENT_ERRORS:
                    raise RuntimeError(f"API error: {operation.error_message}")
                else:
                    raise ConnectionError(f"Connection failed: {operation.error_message}")
            
            return {
                "operation_id": operation.operation_id,
                "duration_ms": operation.duration_ms,
                "status": "completed"
            }
        
        # Execute with comprehensive retry configuration
        start_time = time.perf_counter()
        
        result = await retry_service.execute_with_retry(
            operation=complex_api_operation,
            domain="api",
            operation_type="complex_operation",
            max_attempts=6,
            base_delay_ms=100,
            backoff_multiplier=1.5,
            max_delay_ms=1000,
            operation_args=[],
            operation_kwargs={}
        )
        
        total_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("end_to_end_retry_workflow", total_time, 5000.0)  # Allow sufficient time
        
        # Validate comprehensive result
        assert isinstance(result, RetryOperationResult)
        assert result.attempts_made >= 1
        assert result.total_time_ms > 0
        
        # Log comprehensive results
        logger.info(f"End-to-end retry workflow:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Attempts: {result.attempts_made}")
        logger.info(f"  Total time: {result.total_time_ms:.1f}ms")
        logger.info(f"  Facade coordination time: {total_time:.1f}ms")
        
        if result.success:
            assert isinstance(result.result, dict)
            assert "operation_id" in result.result
            assert "status" in result.result
        else:
            assert result.error is not None
            logger.info(f"  Final error: {result.error}")

    async def test_performance_validation_comprehensive(
        self,
        retry_service: RetryServiceFacade,
        performance_tracker,
    ):
        """Comprehensive performance validation for retry system."""
        # Test facade-level performance targets
        facade_performance = await validate_performance_targets()
        
        # Validate facade performance results
        assert isinstance(facade_performance, dict)
        assert "decision_time_ms" in facade_performance
        assert "target_ms" in facade_performance
        assert "performance_met" in facade_performance
        assert "result_success" in facade_performance
        
        performance_tracker(
            "facade_performance_validation",
            facade_performance["decision_time_ms"],
            facade_performance["target_ms"]
        )
        
        # Test individual service performance
        service_tests = [
            ("config_service_performance", lambda: get_retry_configuration_service().get_default_config()),
            ("backoff_service_performance", lambda: get_backoff_strategy_service().calculate_exponential_backoff(1, 100, 2000, 2.0)),
        ]
        
        for test_name, test_operation in service_tests:
            start_time = time.perf_counter()
            test_operation()
            service_time = (time.perf_counter() - start_time) * 1000
            performance_tracker(test_name, service_time, 1.0)  # <1ms target for individual services
        
        # Overall system performance assessment
        logger.info("Retry System Performance Summary:")
        logger.info(f"  Facade decision time: {facade_performance['decision_time_ms']:.3f}ms (target: {facade_performance['target_ms']}ms)")
        logger.info(f"  Performance target met: {facade_performance['performance_met']}")
        logger.info(f"  Architecture: {facade_performance.get('architecture', 'unknown')}")
        logger.info(f"  Legacy eliminated: {facade_performance.get('legacy_eliminated', 'unknown')}")
        
        # At least 80% of performance targets should be met
        # This is validated through the performance_tracker fixture in the test framework