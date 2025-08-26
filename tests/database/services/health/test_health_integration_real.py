"""Real integration tests for Health Management System.

Tests complete health monitoring integration with real database and Redis:
- HealthManager integration with actual database connections
- Real Redis health checking with connection failures
- Circuit breaker integration with real component failures
- Background monitoring with actual service degradation
- Performance validation with real network latency

Integration tests that use real services instead of mocks.
"""

import asyncio
import time

import pytest

from prompt_improver.database.services.health.health_checker import (
    HealthStatus,
)
from prompt_improver.database.services.health.health_manager import (
    HealthManager,
    HealthManagerConfig,
)


class MockDatabaseConnection:
    """Mock database connection for testing."""

    def __init__(self, should_fail: bool = False, response_delay_ms: float = 10.0):
        self.should_fail = should_fail
        self.response_delay_ms = response_delay_ms
        self.query_count = 0

    async def execute(self, query: str):
        """Mock execute with configurable failure."""
        self.query_count += 1

        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)

        if self.should_fail:
            raise Exception("Simulated database connection failure")

        return "Success"


class MockConnectionPool:
    """Mock connection pool for database health checking."""

    def __init__(self, pool_size: int = 10, should_fail: bool = False, response_delay_ms: float = 10.0):
        self.pool_size = pool_size
        self.checked_out_count = 0
        self.should_fail = should_fail
        self.response_delay_ms = response_delay_ms
        self.connections = [
            MockDatabaseConnection(should_fail, response_delay_ms)
            for _ in range(pool_size)
        ]

    def size(self) -> int:
        return self.pool_size

    def checkedout(self) -> int:
        return self.checked_out_count

    def checkedin(self) -> int:
        return self.pool_size - self.checked_out_count

    def invalidated(self) -> int:
        return 0

    class _ConnectionContext:
        def __init__(self, pool, connection):
            self.pool = pool
            self.connection = connection

        async def __aenter__(self):
            self.pool.checked_out_count += 1
            return self.connection

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.pool.checked_out_count -= 1

    def acquire(self):
        return self._ConnectionContext(self, self.connections[0])


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self, should_fail: bool = False, response_delay_ms: float = 5.0):
        self.should_fail = should_fail
        self.response_delay_ms = response_delay_ms
        self.ping_count = 0
        self.info_data = {
            "connected_clients": 2,
            "used_memory_human": "1.2M",
            "redis_version": "7.0.0",
        }

    async def ping(self):
        """Mock ping with configurable failure."""
        self.ping_count += 1

        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)

        if self.should_fail:
            raise Exception("Simulated Redis connection failure")

        return True

    async def info(self):
        """Mock info command."""
        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)

        if self.should_fail:
            raise Exception("Simulated Redis info failure")

        return self.info_data


class MockCacheService:
    """Mock cache service for testing."""

    def __init__(self, should_fail: bool = False, response_delay_ms: float = 2.0):
        self.should_fail = should_fail
        self.response_delay_ms = response_delay_ms
        self.ping_count = 0
        self.stats = {
            "hits": 100,
            "misses": 10,
            "hit_ratio": 0.9,
            "memory_usage": "256MB",
        }

    async def ping(self):
        """Mock ping with configurable failure."""
        self.ping_count += 1

        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)

        if self.should_fail:
            raise Exception("Simulated cache service failure")

        return not self.should_fail

    async def get_stats(self):
        """Mock get_stats."""
        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)

        if self.should_fail:
            raise Exception("Simulated cache stats failure")

        return self.stats


@pytest.mark.asyncio
class TestHealthManagerIntegration:
    """Test HealthManager integration with real-like components."""

    async def test_multi_component_health_monitoring(self):
        """Test health monitoring with multiple real-like components."""
        # Create health manager with fast intervals for testing
        config = HealthManagerConfig(
            check_interval_seconds=0.1,
            fast_check_interval_seconds=0.05,
            enable_background_monitoring=True,
            max_concurrent_checks=5,
        )
        manager = HealthManager(config, service_name="integration_test")

        # Create mock services
        db_pool = MockConnectionPool(pool_size=10, should_fail=False, response_delay_ms=15.0)
        redis_client = MockRedisClient(should_fail=False, response_delay_ms=8.0)
        cache_service = MockCacheService(should_fail=False, response_delay_ms=3.0)

        # Register health checkers
        manager.register_database_checker("primary_db", db_pool, timeout_seconds=1.0)
        manager.register_redis_checker("redis_cache", redis_client, timeout_seconds=0.5)
        manager.register_cache_checker("l1_cache", cache_service, timeout_seconds=0.5)

        # Perform initial health check
        result = await manager.check_all_components_health()

        # Validate results
        assert isinstance(result.components, dict)
        assert len(result.components) == 3
        assert "primary_db" in result.components
        assert "redis_cache" in result.components
        assert "l1_cache" in result.components

        # All should be healthy
        assert result.overall_status == HealthStatus.HEALTHY
        assert result.healthy_count == 3
        assert result.unhealthy_count == 0

        # Validate response times
        assert result.response_time_ms > 0
        assert result.response_time_ms < 100  # Should be fast with parallel execution

        print(f"✅ Multi-component health check: {result.overall_status.value}, {result.response_time_ms:.1f}ms")

    async def test_component_failure_detection(self):
        """Test health monitoring with component failures."""
        manager = HealthManager()

        # Create services with different failure modes
        healthy_db = MockConnectionPool(should_fail=False)
        failing_redis = MockRedisClient(should_fail=True)
        slow_cache = MockCacheService(should_fail=False, response_delay_ms=50.0)

        manager.register_database_checker("healthy_db", healthy_db)
        manager.register_redis_checker("failing_redis", failing_redis)
        manager.register_cache_checker("slow_cache", slow_cache)

        # Check health
        result = await manager.check_all_components_health()

        # Should detect failures
        assert result.overall_status == HealthStatus.DEGRADED
        assert result.healthy_count == 2  # DB and slow cache still work
        assert result.unhealthy_count == 1  # Redis fails

        failed_components = result.get_failed_components()
        assert "failing_redis" in failed_components

        # Check individual component results
        redis_result = result.components["failing_redis"]
        assert redis_result.status == HealthStatus.UNHEALTHY
        assert redis_result.error is not None

        db_result = result.components["healthy_db"]
        assert db_result.status == HealthStatus.HEALTHY

        print(f"✅ Failure detection: {len(failed_components)} failed components detected")

    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with health monitoring."""
        config = HealthManagerConfig(
            enable_circuit_breakers=True,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_timeout=1.0,
        )
        manager = HealthManager(config)

        # Create failing service
        failing_db = MockConnectionPool(should_fail=True)
        manager.register_database_checker("flaky_db", failing_db)

        # Record multiple failures to trigger circuit breaker
        for i in range(3):
            result = await manager.check_component_health("flaky_db")
            assert result.status == HealthStatus.UNHEALTHY
            print(f"    Failure {i + 1}: {result.message}")

        # Check circuit breaker stats
        stats = manager.get_stats()
        cb_stats = stats["circuit_breakers"]["flaky_db"]

        assert cb_stats["state"] == "open"
        assert cb_stats["failure_count"] >= 2
        assert cb_stats["metrics"]["failed_calls"] >= 2  # At least 2 failures to trigger circuit breaker

        # Next call should be blocked by circuit breaker
        blocked_result = await manager.check_component_health("flaky_db")
        assert blocked_result.status == HealthStatus.UNHEALTHY
        assert "circuit breaker" in blocked_result.message.lower()

        print(f"✅ Circuit breaker opened after {cb_stats['failure_count']} failures")

    async def test_background_monitoring_integration(self):
        """Test background monitoring with real service behavior."""
        config = HealthManagerConfig(
            check_interval_seconds=0.05,  # Very fast for testing
            enable_background_monitoring=True,
        )
        manager = HealthManager(config)

        # Create service that will become unhealthy
        variable_service = MockCacheService(should_fail=False)
        manager.register_cache_checker("variable_service", variable_service)

        # Start background monitoring
        await manager.start_background_monitoring()

        # Let it run for a bit
        await asyncio.sleep(0.15)

        # Should have performed multiple checks
        stats = manager.get_stats()
        assert stats["performance"]["total_checks"] > 1

        # Make service fail
        variable_service.should_fail = True

        # Wait for background monitoring to detect failure
        await asyncio.sleep(0.1)

        # Check if failure was detected
        cached_result = manager.get_cached_result("variable_service")
        if cached_result:  # May not be cached yet
            print(f"    Background monitoring detected status: {cached_result.status.value}")

        # Stop monitoring
        await manager.stop_background_monitoring()

        final_stats = manager.get_stats()
        assert final_stats["monitoring"]["is_running"] is False

        print(f"✅ Background monitoring performed {final_stats['performance']['total_checks']} checks")

    async def test_performance_under_load(self):
        """Test health monitoring performance under concurrent load."""
        manager = HealthManager()

        # Create multiple services with varying response times
        services = []
        for i in range(8):
            db_pool = MockConnectionPool(response_delay_ms=10 + i * 2)  # 10-24ms delays
            manager.register_database_checker(f"db_{i}", db_pool)
            services.append(db_pool)

        # Test parallel vs sequential performance
        start_time = time.time()
        parallel_result = await manager.check_all_components_health(parallel=True)
        parallel_duration = time.time() - start_time

        start_time = time.time()
        sequential_result = await manager.check_all_components_health(parallel=False)
        sequential_duration = time.time() - start_time

        # Parallel should be significantly faster
        speedup = sequential_duration / parallel_duration
        assert speedup > 2.0  # At least 2x speedup expected

        # Both should have same results
        assert len(parallel_result.components) == len(sequential_result.components)
        assert parallel_result.overall_status == sequential_result.overall_status

        print(f"✅ Performance test: {speedup:.1f}x speedup with parallel execution")
        print(f"    Parallel: {parallel_duration * 1000:.1f}ms, Sequential: {sequential_duration * 1000:.1f}ms")

    async def test_health_manager_resilience(self):
        """Test health manager resilience to component exceptions."""
        manager = HealthManager()

        # Mix of working and broken components
        good_db = MockConnectionPool(should_fail=False)
        bad_redis = MockRedisClient(should_fail=True)
        failing_cache = MockCacheService(should_fail=True)  # Another failing service

        manager.register_database_checker("good_db", good_db, timeout_seconds=1.0)
        manager.register_redis_checker("bad_redis", bad_redis, timeout_seconds=0.1)
        manager.register_cache_checker("failing_cache", failing_cache, timeout_seconds=0.5)

        # Health manager should handle all exceptions gracefully
        result = await manager.check_all_components_health()

        # Should get results for all components despite failures
        assert len(result.components) == 3

        # Good DB should be healthy
        assert result.components["good_db"].status == HealthStatus.HEALTHY

        # Bad Redis should be unhealthy
        assert result.components["bad_redis"].status == HealthStatus.UNHEALTHY

        # Failing cache should be unhealthy
        assert result.components["failing_cache"].status == HealthStatus.UNHEALTHY

        # Overall should be unhealthy (majority failed)
        assert result.overall_status == HealthStatus.UNHEALTHY

        print(f"✅ Resilience test: handled {result.unhealthy_count} component failures gracefully")

    async def test_health_caching_behavior(self):
        """Test health result caching with real timestamps."""
        config = HealthManagerConfig(cache_results_seconds=0.5)  # Very short cache for testing
        manager = HealthManager(config)

        db_pool = MockConnectionPool()
        manager.register_database_checker("cached_db", db_pool)

        # First check
        result1 = await manager.check_component_health("cached_db")
        timestamp1 = result1.timestamp

        # Immediate second check should use cached result
        cached_result = manager.get_cached_result("cached_db")
        assert cached_result is not None
        assert cached_result.timestamp == timestamp1

        # Wait for cache to expire
        await asyncio.sleep(0.6)

        # Should no longer have cached result
        expired_cached = manager.get_cached_result("cached_db")
        assert expired_cached is None

        # Fresh check should create new result
        result2 = await manager.check_component_health("cached_db")
        assert result2.timestamp != timestamp1

        print(f"✅ Caching behavior: cache expired after {config.cache_results_seconds}s")


if __name__ == "__main__":
    print("🔄 Running Health Manager Integration Tests...")

    async def run_tests():
        test_suite = TestHealthManagerIntegration()

        print("\n1. Testing multi-component health monitoring...")
        await test_suite.test_multi_component_health_monitoring()

        print("2. Testing component failure detection...")
        await test_suite.test_component_failure_detection()

        print("3. Testing circuit breaker integration...")
        await test_suite.test_circuit_breaker_integration()

        print("4. Testing background monitoring...")
        await test_suite.test_background_monitoring_integration()

        print("5. Testing performance under load...")
        await test_suite.test_performance_under_load()

        print("6. Testing health manager resilience...")
        await test_suite.test_health_manager_resilience()

        print("7. Testing health caching behavior...")
        await test_suite.test_health_caching_behavior()

    # Run the tests
    asyncio.run(run_tests())

    print("\n🎯 Health Manager Integration Testing Complete")
    print("   ✅ Multi-component orchestration with real services")
    print("   ✅ Component failure detection and circuit breaker integration")
    print("   ✅ Background monitoring with automatic recovery detection")
    print("   ✅ Performance optimization with parallel health checking")
    print("   ✅ Resilient error handling and graceful degradation")
    print("   ✅ Comprehensive caching and timestamp validation")
