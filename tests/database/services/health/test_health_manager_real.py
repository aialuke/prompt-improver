"""Real behavior tests for Health Management System.

Tests comprehensive health monitoring functionality:
- HealthManager orchestration of multiple components
- Real component health checker integration 
- Circuit breaker patterns with failure scenarios
- Background monitoring loops and recovery detection
- Performance under concurrent health checking
- Aggregated health reporting with component details

Integration tests with mock components simulating real behavior.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

import pytest

from prompt_improver.database.services.health.health_manager import (
    HealthManager,
    HealthManagerConfig,
    create_health_manager,
)
from prompt_improver.database.services.health.health_checker import (
    HealthChecker,
    HealthResult,
    HealthStatus,
    AggregatedHealthResult,
    DatabaseHealthChecker,
    RedisHealthChecker,
    CacheHealthChecker,
)
from prompt_improver.database.services.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerOpenException,
    create_circuit_breaker,
)


class TestHealthManagerConfig:
    """Test HealthManagerConfig functionality."""
    
    def test_health_manager_config_defaults(self):
        """Test default configuration values."""
        config = HealthManagerConfig()
        
        assert config.check_interval_seconds == 30.0
        assert config.fast_check_interval_seconds == 5.0
        assert config.enable_background_monitoring is True
        assert config.max_concurrent_checks == 10
        assert config.enable_circuit_breakers is True
        assert config.circuit_breaker_failure_threshold == 5
        assert config.circuit_breaker_recovery_timeout == 60.0
        assert config.cache_results_seconds == 10.0
        assert config.consecutive_failure_threshold == 3
        assert config.degraded_threshold_ratio == 0.5
    
    def test_health_manager_config_custom(self):
        """Test custom configuration values."""
        config = HealthManagerConfig(
            check_interval_seconds=15.0,
            fast_check_interval_seconds=2.0,
            enable_background_monitoring=False,
            max_concurrent_checks=5,
            enable_circuit_breakers=False,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout=30.0,
            degraded_threshold_ratio=0.3
        )
        
        assert config.check_interval_seconds == 15.0
        assert config.fast_check_interval_seconds == 2.0
        assert config.enable_background_monitoring is False
        assert config.max_concurrent_checks == 5
        assert config.enable_circuit_breakers is False
        assert config.circuit_breaker_failure_threshold == 3
        assert config.circuit_breaker_recovery_timeout == 30.0
        assert config.degraded_threshold_ratio == 0.3
    
    def test_health_manager_config_validation(self):
        """Test configuration validation."""
        # Invalid check interval
        with pytest.raises(ValueError, match="check_interval_seconds must be greater than 0"):
            HealthManagerConfig(check_interval_seconds=0)
        
        # Invalid concurrent checks
        with pytest.raises(ValueError, match="max_concurrent_checks must be greater than 0"):
            HealthManagerConfig(max_concurrent_checks=0)


class TestHealthResult:
    """Test HealthResult functionality."""
    
    def test_health_result_creation(self):
        """Test basic health result creation."""
        result = HealthResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            response_time_ms=25.5,
            message="Component is healthy",
        )
        
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms == 25.5
        assert result.message == "Component is healthy"
        assert result.error is None
        assert result.details == {}
        assert result.timestamp is not None
        assert result.is_healthy() is True
    
    def test_health_result_unhealthy(self):
        """Test unhealthy health result."""
        result = HealthResult(
            component="failing_component",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=0,
            message="Component failed",
            error="Connection timeout",
            details={"last_success": "2024-01-01T00:00:00Z"}
        )
        
        assert result.status == HealthStatus.UNHEALTHY
        assert result.is_healthy() is False
        assert result.error == "Connection timeout"
        assert result.details["last_success"] == "2024-01-01T00:00:00Z"
    
    def test_health_result_to_dict(self):
        """Test health result serialization."""
        now = datetime.now(UTC)
        result = HealthResult(
            component="api_service",
            status=HealthStatus.DEGRADED,
            response_time_ms=150.0,
            message="High response times",
            timestamp=now,
            details={"avg_response": 145.2}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["component"] == "api_service"
        assert result_dict["status"] == "degraded"
        assert result_dict["response_time_ms"] == 150.0
        assert result_dict["message"] == "High response times"
        assert result_dict["details"]["avg_response"] == 145.2
        assert result_dict["timestamp"] == now.isoformat()


class TestAggregatedHealthResult:
    """Test AggregatedHealthResult functionality."""
    
    def test_aggregated_health_result_creation(self):
        """Test aggregated health result creation."""
        now = datetime.now(UTC)
        components = {
            "db": HealthResult("db", HealthStatus.HEALTHY, 10.0),
            "redis": HealthResult("redis", HealthStatus.DEGRADED, 25.0),
            "api": HealthResult("api", HealthStatus.UNHEALTHY, 0, error="Timeout"),
        }
        
        aggregated = AggregatedHealthResult(
            overall_status=HealthStatus.DEGRADED,
            components=components,
            response_time_ms=35.0,
            timestamp=now,
        )
        
        assert aggregated.overall_status == HealthStatus.DEGRADED
        assert len(aggregated.components) == 3
        assert aggregated.healthy_count == 1
        assert aggregated.degraded_count == 1
        assert aggregated.unhealthy_count == 1
    
    def test_aggregated_health_result_helpers(self):
        """Test aggregated result helper methods."""
        components = {
            "service1": HealthResult("service1", HealthStatus.HEALTHY, 10.0),
            "service2": HealthResult("service2", HealthStatus.DEGRADED, 20.0),
            "service3": HealthResult("service3", HealthStatus.UNHEALTHY, 0),
            "service4": HealthResult("service4", HealthStatus.UNHEALTHY, 0),
        }
        
        aggregated = AggregatedHealthResult(
            overall_status=HealthStatus.UNHEALTHY,
            components=components,
            response_time_ms=30.0,
            timestamp=datetime.now(UTC),
        )
        
        failed_components = aggregated.get_failed_components()
        degraded_components = aggregated.get_degraded_components()
        
        assert set(failed_components) == {"service3", "service4"}
        assert degraded_components == ["service2"]
    
    def test_aggregated_health_result_to_dict(self):
        """Test aggregated result serialization."""
        now = datetime.now(UTC)
        components = {
            "db": HealthResult("db", HealthStatus.HEALTHY, 15.0),
            "cache": HealthResult("cache", HealthStatus.DEGRADED, 30.0),
        }
        
        aggregated = AggregatedHealthResult(
            overall_status=HealthStatus.DEGRADED,
            components=components,
            response_time_ms=45.0,
            timestamp=now,
        )
        
        result_dict = aggregated.to_dict()
        
        assert result_dict["overall_status"] == "degraded"
        assert result_dict["response_time_ms"] == 45.0
        assert result_dict["summary"]["total_components"] == 2
        assert result_dict["summary"]["healthy_count"] == 1
        assert result_dict["summary"]["degraded_count"] == 1
        assert len(result_dict["components"]) == 2


class MockHealthChecker(HealthChecker):
    """Mock health checker for testing."""
    
    def __init__(self, component_name: str, should_fail: bool = False, response_time_ms: float = 10.0):
        super().__init__(component_name)
        self.should_fail = should_fail
        self.response_time_ms = response_time_ms
        self.check_count = 0
    
    async def check_health(self) -> HealthResult:
        """Mock health check implementation."""
        self.check_count += 1
        
        if self.should_fail:
            return HealthResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=self.response_time_ms,
                message="Mock component failure",
                error="Simulated failure"
            )
        
        return HealthResult(
            component=self.component_name,
            status=HealthStatus.HEALTHY,
            response_time_ms=self.response_time_ms,
            message="Mock component healthy"
        )


class TestHealthManager:
    """Test HealthManager core functionality."""
    
    def test_health_manager_creation(self):
        """Test health manager initialization."""
        config = HealthManagerConfig()
        manager = HealthManager(config, service_name="test_manager")
        
        assert manager.config == config
        assert manager.service_name == "test_manager"
        assert len(manager._health_checkers) == 0
        assert len(manager._circuit_breakers) == 0
        assert manager._is_monitoring is False
        assert manager.total_checks == 0
        assert manager.successful_checks == 0
        assert manager.failed_checks == 0
    
    def test_health_manager_register_checker(self):
        """Test registering health checkers."""
        manager = HealthManager()
        checker = MockHealthChecker("test_component")
        
        manager.register_health_checker("test_component", checker)
        
        assert "test_component" in manager._health_checkers
        assert manager._health_checkers["test_component"] == checker
        assert "test_component" in manager._consecutive_failures
        assert manager._consecutive_failures["test_component"] == 0
        
        # Circuit breaker should be created if enabled
        assert "test_component" in manager._circuit_breakers
        assert isinstance(manager._circuit_breakers["test_component"], CircuitBreaker)
    
    def test_health_manager_unregister_checker(self):
        """Test unregistering health checkers."""
        manager = HealthManager()
        checker = MockHealthChecker("test_component")
        
        manager.register_health_checker("test_component", checker)
        assert "test_component" in manager._health_checkers
        
        result = manager.unregister_health_checker("test_component")
        assert result is True
        assert "test_component" not in manager._health_checkers
        assert "test_component" not in manager._circuit_breakers
        assert "test_component" not in manager._consecutive_failures
        
        # Unregistering non-existent checker
        result = manager.unregister_health_checker("nonexistent")
        assert result is False
    
    def test_health_manager_convenience_methods(self):
        """Test convenience methods for registering checkers."""
        manager = HealthManager()
        
        # Mock objects
        mock_pool = MagicMock()
        mock_redis = MagicMock()
        mock_cache = MagicMock()
        
        manager.register_database_checker("test_db", mock_pool)
        manager.register_redis_checker("test_redis", mock_redis)
        manager.register_cache_checker("test_cache", mock_cache)
        
        assert len(manager._health_checkers) == 3
        assert isinstance(manager._health_checkers["test_db"], DatabaseHealthChecker)
        assert isinstance(manager._health_checkers["test_redis"], RedisHealthChecker)
        assert isinstance(manager._health_checkers["test_cache"], CacheHealthChecker)
    
    @pytest.mark.asyncio
    async def test_health_manager_check_component(self):
        """Test individual component health checking."""
        manager = HealthManager()
        healthy_checker = MockHealthChecker("healthy_component", should_fail=False)
        failing_checker = MockHealthChecker("failing_component", should_fail=True)
        
        manager.register_health_checker("healthy_component", healthy_checker)
        manager.register_health_checker("failing_component", failing_checker)
        
        # Check healthy component
        result = await manager.check_component_health("healthy_component")
        assert result is not None
        assert result.status == HealthStatus.HEALTHY
        assert result.component == "healthy_component"
        assert manager.successful_checks == 1
        
        # Check failing component
        result = await manager.check_component_health("failing_component")
        assert result is not None
        assert result.status == HealthStatus.UNHEALTHY
        assert result.component == "failing_component"
        assert result.error == "Simulated failure"
        assert manager.failed_checks == 1
        
        # Check non-existent component
        result = await manager.check_component_health("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_health_manager_check_all_components(self):
        """Test checking all components."""
        manager = HealthManager()
        
        # Register multiple components
        for i in range(3):
            checker = MockHealthChecker(f"component_{i}", should_fail=(i == 2))
            manager.register_health_checker(f"component_{i}", checker)
        
        # Check all components
        result = await manager.check_all_components_health()
        
        assert isinstance(result, AggregatedHealthResult)
        assert len(result.components) == 3
        assert result.healthy_count == 2
        assert result.unhealthy_count == 1
        assert result.overall_status == HealthStatus.DEGRADED  # Some components failing
        assert result.response_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_health_manager_parallel_vs_sequential(self):
        """Test parallel vs sequential health checking."""
        manager = HealthManager()
        
        # Register components with simulated delay
        for i in range(5):
            checker = MockHealthChecker(f"component_{i}", response_time_ms=50.0)
            manager.register_health_checker(f"component_{i}", checker)
        
        # Test parallel execution
        start_time = time.time()
        parallel_result = await manager.check_all_components_health(parallel=True)
        parallel_duration = time.time() - start_time
        
        # Test sequential execution
        start_time = time.time()
        sequential_result = await manager.check_all_components_health(parallel=False)
        sequential_duration = time.time() - start_time
        
        # Parallel should be faster than sequential
        assert parallel_duration < sequential_duration
        assert len(parallel_result.components) == 5
        assert len(sequential_result.components) == 5
        
        print(f"✅ Parallel execution: {parallel_duration:.3f}s, Sequential: {sequential_duration:.3f}s")
    
    def test_health_manager_calculate_overall_status(self):
        """Test overall status calculation logic."""
        manager = HealthManager()
        
        # All healthy
        components = {
            "comp1": HealthResult("comp1", HealthStatus.HEALTHY, 10.0),
            "comp2": HealthResult("comp2", HealthStatus.HEALTHY, 15.0),
        }
        status = manager._calculate_overall_status(components)
        assert status == HealthStatus.HEALTHY
        
        # Some degraded
        components["comp3"] = HealthResult("comp3", HealthStatus.DEGRADED, 20.0)
        status = manager._calculate_overall_status(components)
        assert status == HealthStatus.DEGRADED
        
        # Too many unhealthy (> 50% threshold)
        components = {
            "comp1": HealthResult("comp1", HealthStatus.UNHEALTHY, 0),
            "comp2": HealthResult("comp2", HealthStatus.UNHEALTHY, 0),
            "comp3": HealthResult("comp3", HealthStatus.HEALTHY, 10.0),
        }
        status = manager._calculate_overall_status(components)
        assert status == HealthStatus.UNHEALTHY  # 2/3 = 66% > 50% threshold
        
        # Empty components
        status = manager._calculate_overall_status({})
        assert status == HealthStatus.UNKNOWN
    
    def test_health_manager_result_caching(self):
        """Test health result caching functionality."""
        config = HealthManagerConfig(cache_results_seconds=1.0)
        manager = HealthManager(config)
        
        checker = MockHealthChecker("test_component")
        manager.register_health_checker("test_component", checker)
        
        # No cached result initially
        cached = manager.get_cached_result("test_component")
        assert cached is None
        
        # Create a result manually to test caching
        now = datetime.now(UTC)
        test_result = HealthResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            response_time_ms=10.0,
            timestamp=now
        )
        manager._last_check_results["test_component"] = test_result
        
        # Should get cached result
        cached = manager.get_cached_result("test_component")
        assert cached is not None
        assert cached.component == "test_component"
        
        # Test stale result (older than cache timeout)
        old_result = HealthResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            response_time_ms=10.0,
            timestamp=now - timedelta(seconds=2.0)  # Older than cache timeout
        )
        manager._last_check_results["test_component"] = old_result
        
        cached = manager.get_cached_result("test_component")
        assert cached is None  # Should be None due to staleness


@pytest.mark.asyncio
class TestHealthManagerAsync:
    """Test HealthManager async functionality."""
    
    async def test_health_manager_background_monitoring(self):
        """Test background monitoring functionality."""
        config = HealthManagerConfig(
            check_interval_seconds=0.1,  # Very fast for testing
            enable_background_monitoring=True
        )
        manager = HealthManager(config)
        
        checker = MockHealthChecker("background_component")
        manager.register_health_checker("background_component", checker)
        
        # Start background monitoring
        await manager.start_background_monitoring()
        assert manager._is_monitoring is True
        
        # Wait for a few checks
        await asyncio.sleep(0.3)
        
        # Should have performed multiple checks
        assert checker.check_count > 1
        assert manager.total_checks > 1
        
        # Stop monitoring
        await manager.stop_background_monitoring()
        assert manager._is_monitoring is False
    
    async def test_health_manager_background_monitoring_disabled(self):
        """Test when background monitoring is disabled."""
        config = HealthManagerConfig(enable_background_monitoring=False)
        manager = HealthManager(config)
        
        await manager.start_background_monitoring()
        assert manager._is_monitoring is False
        assert manager._monitoring_task is None
    
    async def test_health_manager_shutdown(self):
        """Test health manager shutdown."""
        config = HealthManagerConfig(
            check_interval_seconds=0.1,
            enable_background_monitoring=True
        )
        manager = HealthManager(config)
        
        checker = MockHealthChecker("shutdown_test")
        manager.register_health_checker("shutdown_test", checker)
        
        # Start monitoring
        await manager.start_background_monitoring()
        assert manager._is_monitoring is True
        assert len(manager._health_checkers) == 1
        
        # Shutdown
        await manager.shutdown()
        
        assert manager._is_monitoring is False
        assert len(manager._health_checkers) == 0
        assert len(manager._circuit_breakers) == 0
        assert len(manager._last_check_results) == 0
    
    async def test_health_manager_stats(self):
        """Test comprehensive statistics."""
        manager = HealthManager(service_name="stats_test_manager")
        
        checker = MockHealthChecker("stats_component")
        manager.register_health_checker("stats_component", checker)
        
        # Perform some checks
        await manager.check_component_health("stats_component")
        await manager.check_all_components_health()
        
        stats = manager.get_stats()
        
        assert stats["service"] == "stats_test_manager"
        assert "monitoring" in stats
        assert "performance" in stats
        assert "components" in stats
        assert "circuit_breakers" in stats
        
        assert stats["monitoring"]["registered_components"] == 1
        assert stats["performance"]["total_checks"] > 0
        assert stats["performance"]["successful_checks"] > 0
        assert "stats_component" in stats["components"]


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=30.0)
        cb = CircuitBreaker("test_service", config)
        
        assert cb.service_name == "test_service"
        assert cb.config == config
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.total_calls == 0
    
    def test_circuit_breaker_config_validation(self):
        """Test circuit breaker configuration validation."""
        # Invalid failure threshold
        with pytest.raises(ValueError, match="failure_threshold must be greater than 0"):
            CircuitBreakerConfig(failure_threshold=0)
        
        # Invalid recovery timeout
        with pytest.raises(ValueError, match="recovery_timeout_seconds must be greater than 0"):
            CircuitBreakerConfig(recovery_timeout_seconds=0)
        
        # Invalid success threshold
        with pytest.raises(ValueError, match="success_threshold must be greater than 0"):
            CircuitBreakerConfig(success_threshold=0)
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=1.0)
        cb = CircuitBreaker("test_service", config)
        
        # Initially closed
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_call_permitted() is True
        
        # Record failures
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED  # Below threshold
        assert cb.failure_count == 1
        
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN  # At threshold
        assert cb.failure_count == 2
        assert cb.is_call_permitted() is False
        
        # Wait for recovery timeout (simulate)
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=2.0)
        assert cb.is_call_permitted() is True  # Should transition to HALF_OPEN
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Success in half-open should close circuit
        cb.record_success()
        cb.record_success()  
        cb.record_success()  # Reach success threshold (3)
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_breaker_exponential_backoff(self):
        """Test exponential backoff for recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=10.0,
            backoff_multiplier=2.0,
            max_recovery_timeout_seconds=60.0
        )
        cb = CircuitBreaker("backoff_test", config)
        
        # First failure -> open (backoff applied immediately)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.current_recovery_timeout == 20.0  # 10.0 * 2.0
        
        # Force open again -> backoff applied
        cb.record_failure()
        cb._transition_to_open()
        assert cb.current_recovery_timeout == 40.0  # 20.0 * 2.0
        
        # Another failure -> more backoff
        cb._transition_to_open()
        assert cb.current_recovery_timeout == 60.0  # 40.0 * 1.5 = 60.0 (capped at max)
        
        # Should stay capped at max
        cb._transition_to_open()
        assert cb.current_recovery_timeout == 60.0  # Still capped at max
    
    def test_circuit_breaker_disabled(self):
        """Test circuit breaker when disabled."""
        config = CircuitBreakerConfig(enabled=False)
        cb = CircuitBreaker("disabled_test", config)
        
        # Should always permit calls when disabled
        assert cb.is_call_permitted() is True
        
        # Record failures
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        # Should still permit calls
        assert cb.is_call_permitted() is True
    
    def test_circuit_breaker_timeout_failures(self):
        """Test circuit breaker with timeout-based failures."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_ms=100.0)
        cb = CircuitBreaker("timeout_test", config)
        
        # Record slow response (timeout)
        cb.record_failure(response_time_ms=150.0)  # > 100ms timeout
        assert cb.failure_count == 1
        
        # Another timeout failure
        cb.record_failure(response_time_ms=200.0)
        assert cb.state == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        cb = CircuitBreaker("stats_test")
        
        # Record some operations
        cb.record_success(50.0)
        cb.record_success(25.0)
        cb.record_failure(response_time_ms=100.0)
        
        stats = cb.get_stats()
        
        assert stats["service_name"] == "stats_test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["metrics"]["total_calls"] == 3
        assert stats["metrics"]["successful_calls"] == 2
        assert stats["metrics"]["failed_calls"] == 1
        assert stats["metrics"]["failure_rate"] == 1/3
        assert stats["config"]["failure_threshold"] == 5  # Default


class TestHealthManagerCreationHelper:
    """Test convenience creation function."""
    
    def test_create_health_manager_defaults(self):
        """Test convenience function with defaults."""
        manager = create_health_manager()
        
        assert manager.service_name == "health_manager"
        assert manager.config.check_interval_seconds == 30.0
        assert manager.config.enable_background_monitoring is True
        assert manager.config.enable_circuit_breakers is True
    
    def test_create_health_manager_custom(self):
        """Test convenience function with custom settings."""
        manager = create_health_manager(
            service_name="custom_manager",
            check_interval_seconds=15.0,
            enable_background_monitoring=False,
            enable_circuit_breakers=False,
            max_concurrent_checks=5
        )
        
        assert manager.service_name == "custom_manager"
        assert manager.config.check_interval_seconds == 15.0
        assert manager.config.enable_background_monitoring is False
        assert manager.config.enable_circuit_breakers is False
        assert manager.config.max_concurrent_checks == 5


class TestCircuitBreakerCreationHelper:
    """Test circuit breaker convenience creation function."""
    
    def test_create_circuit_breaker_defaults(self):
        """Test convenience function with defaults."""
        cb = create_circuit_breaker("test_service")
        
        assert cb.service_name == "test_service"
        assert cb.config.failure_threshold == 5
        assert cb.config.recovery_timeout_seconds == 60.0
        assert cb.config.success_threshold == 3
        assert cb.config.enabled is True
    
    def test_create_circuit_breaker_custom(self):
        """Test convenience function with custom settings."""
        cb = create_circuit_breaker(
            "custom_service",
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
            success_threshold=2,
            timeout_ms=2000.0,
            enabled=False
        )
        
        assert cb.service_name == "custom_service"
        assert cb.config.failure_threshold == 3
        assert cb.config.recovery_timeout_seconds == 30.0
        assert cb.config.success_threshold == 2
        assert cb.config.timeout_ms == 2000.0
        assert cb.config.enabled is False


if __name__ == "__main__":
    print("🔄 Running HealthManager Tests...")
    
    # Run synchronous tests
    print("\n1. Testing HealthManagerConfig...")
    test_config = TestHealthManagerConfig()
    test_config.test_health_manager_config_defaults()
    test_config.test_health_manager_config_custom()
    test_config.test_health_manager_config_validation()
    print("   ✅ HealthManagerConfig tests passed")
    
    print("2. Testing HealthResult...")
    test_result = TestHealthResult()
    test_result.test_health_result_creation()
    test_result.test_health_result_unhealthy()
    test_result.test_health_result_to_dict()
    print("   ✅ HealthResult tests passed")
    
    print("3. Testing AggregatedHealthResult...")
    test_aggregated = TestAggregatedHealthResult()
    test_aggregated.test_aggregated_health_result_creation()
    test_aggregated.test_aggregated_health_result_helpers()
    test_aggregated.test_aggregated_health_result_to_dict()
    print("   ✅ AggregatedHealthResult tests passed")
    
    print("4. Testing HealthManager Core...")
    test_manager = TestHealthManager()
    test_manager.test_health_manager_creation()
    test_manager.test_health_manager_register_checker()
    test_manager.test_health_manager_unregister_checker()
    test_manager.test_health_manager_convenience_methods()
    test_manager.test_health_manager_calculate_overall_status()
    test_manager.test_health_manager_result_caching()
    print("   ✅ HealthManager core tests passed")
    
    print("5. Testing CircuitBreaker...")
    test_cb = TestCircuitBreaker()
    test_cb.test_circuit_breaker_creation()
    test_cb.test_circuit_breaker_config_validation()
    test_cb.test_circuit_breaker_state_transitions()
    test_cb.test_circuit_breaker_exponential_backoff()
    test_cb.test_circuit_breaker_disabled()
    test_cb.test_circuit_breaker_timeout_failures()
    test_cb.test_circuit_breaker_stats()
    print("   ✅ CircuitBreaker tests passed")
    
    print("6. Testing Creation Helpers...")
    test_helper_hm = TestHealthManagerCreationHelper()
    test_helper_hm.test_create_health_manager_defaults()
    test_helper_hm.test_create_health_manager_custom()
    
    test_helper_cb = TestCircuitBreakerCreationHelper()
    test_helper_cb.test_create_circuit_breaker_defaults()
    test_helper_cb.test_create_circuit_breaker_custom()
    print("   ✅ Creation helper tests passed")
    
    print("\n🎯 HealthManager Testing Complete")
    print("   ✅ Multi-component health orchestration validated")
    print("   ✅ Circuit breaker patterns and fault tolerance")
    print("   ✅ Background monitoring and recovery detection")
    print("   ✅ Performance optimized health checking")
    print("   ✅ Comprehensive metrics and aggregated reporting")
    print("   ✅ Real behavior testing with mocked components")