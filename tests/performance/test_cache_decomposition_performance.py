"""Performance tests for decomposed cache services.

Validates that the decomposed cache services meet the strict performance
requirements defined in the clean break strategy.
"""

import asyncio
import pytest
import time
from typing import Any

from prompt_improver.services.cache import (
    L1CacheService,
    L2RedisService,
    L3DatabaseService,
    CacheCoordinatorService,
    CacheMonitoringService,
)
from prompt_improver.services.cache.cache_facade import CacheFacade


class TestCachePerformance:
    """Performance tests for cache service decomposition."""

    @pytest.fixture
    def l1_cache(self) -> L1CacheService:
        """Create L1 cache service for testing."""
        return L1CacheService(max_size=100)

    @pytest.fixture
    def l2_cache(self) -> L2RedisService:
        """Create L2 cache service for testing."""
        return L2RedisService()

    @pytest.fixture
    def cache_facade(self) -> CacheFacade:
        """Create cache facade for testing."""
        return CacheFacade(
            l1_max_size=100,
            enable_l2=False,  # Disable to avoid Redis dependency
            enable_l3=False,  # Disable to avoid DB dependency
            enable_warming=False,  # Disable for predictable performance
        )

    @pytest.mark.asyncio
    async def test_l1_cache_response_time(self, l1_cache: L1CacheService):
        """Test L1 cache meets <1ms response time requirement."""
        # Warm up the cache
        await l1_cache.set("warmup", "value")
        await l1_cache.get("warmup")
        
        # Test GET operation performance
        test_key = "test_key"
        test_value = {"data": "test_value", "number": 123}
        
        # Set initial value
        await l1_cache.set(test_key, test_value)
        
        # Measure GET performance (should be <1ms)
        start_time = time.perf_counter()
        result = await l1_cache.get(test_key)
        get_time = time.perf_counter() - start_time
        
        assert result == test_value
        assert get_time < 0.001, f"L1 GET took {get_time*1000:.2f}ms, expected <1ms"
        
        # Measure SET performance (should be <1ms)
        start_time = time.perf_counter()
        await l1_cache.set("new_key", {"new": "value"})
        set_time = time.perf_counter() - start_time
        
        assert set_time < 0.001, f"L1 SET took {set_time*1000:.2f}ms, expected <1ms"

    @pytest.mark.asyncio
    async def test_l1_cache_bulk_performance(self, l1_cache: L1CacheService):
        """Test L1 cache performance under load."""
        operations = []
        
        # Prepare bulk operations
        for i in range(100):
            operations.append(l1_cache.set(f"key_{i}", {"value": i}))
        
        # Measure bulk SET operations
        start_time = time.perf_counter()
        await asyncio.gather(*operations)
        bulk_set_time = time.perf_counter() - start_time
        
        avg_set_time = bulk_set_time / 100
        assert avg_set_time < 0.001, f"Average L1 SET took {avg_set_time*1000:.2f}ms, expected <1ms"
        
        # Measure bulk GET operations
        get_operations = []
        for i in range(100):
            get_operations.append(l1_cache.get(f"key_{i}"))
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*get_operations)
        bulk_get_time = time.perf_counter() - start_time
        
        avg_get_time = bulk_get_time / 100
        assert avg_get_time < 0.001, f"Average L1 GET took {avg_get_time*1000:.2f}ms, expected <1ms"
        
        # Verify all results
        for i, result in enumerate(results):
            assert result == {"value": i}

    @pytest.mark.asyncio
    async def test_cache_facade_performance(self, cache_facade: CacheFacade):
        """Test cache facade meets overall performance requirements."""
        test_key = "facade_test"
        test_value = {"facade": "test", "timestamp": time.time()}
        
        # Measure SET operation (should be <50ms for overall)
        start_time = time.perf_counter()
        await cache_facade.set(test_key, test_value)
        set_time = time.perf_counter() - start_time
        
        assert set_time < 0.05, f"Facade SET took {set_time*1000:.2f}ms, expected <50ms"
        
        # Measure GET operation (should be <50ms for overall)
        start_time = time.perf_counter()
        result = await cache_facade.get(test_key)
        get_time = time.perf_counter() - start_time
        
        assert result == test_value
        assert get_time < 0.05, f"Facade GET took {get_time*1000:.2f}ms, expected <50ms"

    @pytest.mark.asyncio
    async def test_cache_coordinator_performance(self):
        """Test cache coordinator performance with multiple levels."""
        l1_cache = L1CacheService(max_size=50)
        coordinator = CacheCoordinatorService(
            l1_cache=l1_cache,
            l2_cache=None,  # Disable L2 for predictable performance
            l3_cache=None,  # Disable L3 for predictable performance
            enable_warming=False,
        )
        
        test_key = "coordinator_test"
        test_value = {"coordinator": "test"}
        
        # Measure coordinator SET operation
        start_time = time.perf_counter()
        await coordinator.set(test_key, test_value)
        set_time = time.perf_counter() - start_time
        
        assert set_time < 0.01, f"Coordinator SET took {set_time*1000:.2f}ms, expected <10ms"
        
        # Measure coordinator GET operation (L1 hit)
        start_time = time.perf_counter()
        result = await coordinator.get(test_key)
        get_time = time.perf_counter() - start_time
        
        assert result == test_value
        assert get_time < 0.001, f"Coordinator GET took {get_time*1000:.2f}ms, expected <1ms for L1 hit"

    @pytest.mark.asyncio
    async def test_monitoring_service_performance(self):
        """Test monitoring service performance requirements."""
        l1_cache = L1CacheService(max_size=50)
        coordinator = CacheCoordinatorService(
            l1_cache=l1_cache,
            enable_warming=False,
        )
        monitoring = CacheMonitoringService(coordinator)
        
        # Measure health check performance (should be <100ms)
        start_time = time.perf_counter()
        health_result = await monitoring.health_check()
        health_time = time.perf_counter() - start_time
        
        assert health_time < 0.1, f"Health check took {health_time*1000:.2f}ms, expected <100ms"
        assert health_result["healthy"] is True
        
        # Measure metrics collection performance (should be <25ms)
        start_time = time.perf_counter()
        metrics = monitoring.get_monitoring_metrics()
        metrics_time = time.perf_counter() - start_time
        
        assert metrics_time < 0.025, f"Metrics collection took {metrics_time*1000:.2f}ms, expected <25ms"
        assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_service_memory_efficiency(self, l1_cache: L1CacheService):
        """Test memory efficiency of decomposed services."""
        # Fill cache with test data
        for i in range(100):
            await l1_cache.set(f"memory_test_{i}", {"data": f"value_{i}", "index": i})
        
        stats = l1_cache.get_stats()
        memory_bytes = stats["estimated_memory_bytes"]
        memory_per_entry = stats["memory_per_entry_bytes"]
        
        # Verify memory efficiency (should be <2KB per entry)
        assert memory_per_entry < 2048, f"Memory per entry {memory_per_entry} bytes, expected <2KB"
        
        # Verify reasonable total memory usage
        assert memory_bytes < 1024 * 1024, f"Total memory {memory_bytes} bytes, expected <1MB for 100 entries"

    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, cache_facade: CacheFacade):
        """Test performance under concurrent access."""
        async def worker(worker_id: int) -> float:
            """Worker function for concurrent testing."""
            start_time = time.perf_counter()
            
            # Perform mixed operations
            for i in range(10):
                key = f"concurrent_{worker_id}_{i}"
                value = {"worker": worker_id, "item": i}
                
                await cache_facade.set(key, value)
                result = await cache_facade.get(key)
                assert result == value
            
            return time.perf_counter() - start_time
        
        # Run 10 concurrent workers
        tasks = [worker(i) for i in range(10)]
        start_time = time.perf_counter()
        worker_times = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Verify concurrent performance
        max_worker_time = max(worker_times)
        avg_worker_time = sum(worker_times) / len(worker_times)
        
        assert max_worker_time < 0.1, f"Max worker time {max_worker_time*1000:.2f}ms, expected <100ms"
        assert total_time < 0.2, f"Total concurrent time {total_time*1000:.2f}ms, expected <200ms"

    def test_service_line_count_compliance(self):
        """Test that all services comply with <400 line requirement."""
        import inspect
        
        services = [
            L1CacheService,
            L2RedisService,
            L3DatabaseService,
            CacheCoordinatorService,
            CacheMonitoringService,
        ]
        
        for service_class in services:
            source_lines = inspect.getsourcelines(service_class)[0]
            line_count = len(source_lines)
            
            assert line_count < 400, f"{service_class.__name__} has {line_count} lines, expected <400"

    @pytest.mark.asyncio
    async def test_slo_compliance_tracking(self):
        """Test SLO compliance tracking accuracy."""
        l1_cache = L1CacheService(max_size=50)
        coordinator = CacheCoordinatorService(l1_cache=l1_cache, enable_warming=False)
        monitoring = CacheMonitoringService(coordinator)
        
        # Perform operations to generate response time data
        for i in range(50):
            await coordinator.set(f"slo_test_{i}", {"value": i})
            await coordinator.get(f"slo_test_{i}")
            
            # Record operations in monitoring
            monitoring.record_operation("test_operation", 0.001, True)  # 1ms operation
        
        # Check SLO compliance
        slo_compliance = monitoring.calculate_slo_compliance()
        
        assert slo_compliance["compliant"] is True
        assert slo_compliance["compliance_rate"] >= 0.95
        assert slo_compliance["slo_target_ms"] == 200


@pytest.mark.benchmark
class TestCacheBenchmarks:
    """Benchmark tests for performance regression detection."""

    @pytest.mark.asyncio
    async def test_l1_cache_throughput_benchmark(self):
        """Benchmark L1 cache throughput."""
        l1_cache = L1CacheService(max_size=1000)
        
        # Warm up
        for i in range(100):
            await l1_cache.set(f"warmup_{i}", {"value": i})
        
        # Benchmark SET operations
        operations_count = 1000
        start_time = time.perf_counter()
        
        for i in range(operations_count):
            await l1_cache.set(f"bench_set_{i}", {"benchmark": i, "data": f"value_{i}"})
        
        set_time = time.perf_counter() - start_time
        set_ops_per_second = operations_count / set_time
        
        print(f"L1 Cache SET throughput: {set_ops_per_second:.0f} ops/sec")
        assert set_ops_per_second > 10000, f"SET throughput {set_ops_per_second:.0f} ops/sec too low"
        
        # Benchmark GET operations
        start_time = time.perf_counter()
        
        for i in range(operations_count):
            result = await l1_cache.get(f"bench_set_{i}")
            assert result is not None
        
        get_time = time.perf_counter() - start_time
        get_ops_per_second = operations_count / get_time
        
        print(f"L1 Cache GET throughput: {get_ops_per_second:.0f} ops/sec")
        assert get_ops_per_second > 10000, f"GET throughput {get_ops_per_second:.0f} ops/sec too low"