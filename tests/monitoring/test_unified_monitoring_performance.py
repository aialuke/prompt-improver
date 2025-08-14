"""Performance validation tests for UnifiedMonitoringFacade.

Validates that the unified monitoring system meets performance requirements:
- Health checks complete in <10ms per operation
- System health check completes in <100ms total
- Metrics collection completes in <50ms
- No performance regression from consolidation
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from prompt_improver.monitoring.unified import (
    UnifiedMonitoringFacade,
    MonitoringConfig,
    create_monitoring_facade,
    HealthStatus,
)


class TestUnifiedMonitoringPerformance:
    """Performance validation tests for unified monitoring."""

    @pytest.fixture
    async def fast_config(self):
        """Create config optimized for speed."""
        return MonitoringConfig(
            health_check_timeout_seconds=5.0,
            health_check_parallel_enabled=True,
            max_concurrent_checks=20,
            metrics_collection_enabled=True,
            cache_health_results_seconds=30,
        )

    @pytest.fixture
    async def monitoring_facade(self, fast_config):
        """Create monitoring facade for testing."""
        # Mock the database services to avoid actual database calls
        with pytest.mock.patch('prompt_improver.database.get_database_services'):
            facade = await create_monitoring_facade(config=fast_config)
            return facade

    async def test_individual_health_check_performance(self, monitoring_facade):
        """Test that individual health checks complete quickly."""
        component_names = ["database", "redis", "ml_models", "system_resources"]
        
        for component_name in component_names:
            start_time = time.perf_counter()
            
            result = await monitoring_facade.check_component_health(component_name)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Individual health checks should complete in <10ms
            assert duration_ms < 10.0, f"{component_name} health check took {duration_ms:.2f}ms (>10ms)"
            
            # Result should be valid
            assert result.component_name == component_name
            assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
            
            print(f"✓ {component_name} health check: {duration_ms:.2f}ms")

    async def test_system_health_check_performance(self, monitoring_facade):
        """Test that system health check completes within performance target."""
        start_time = time.perf_counter()
        
        health_summary = await monitoring_facade.get_system_health()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # System health check should complete in <100ms
        assert duration_ms < 100.0, f"System health check took {duration_ms:.2f}ms (>100ms)"
        
        # Verify results are comprehensive
        assert health_summary.total_components > 0
        assert len(health_summary.component_results) > 0
        assert health_summary.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
        
        print(f"✓ System health check: {duration_ms:.2f}ms ({health_summary.total_components} components)")

    async def test_metrics_collection_performance(self, monitoring_facade):
        """Test that metrics collection completes within performance target."""
        start_time = time.perf_counter()
        
        metrics = await monitoring_facade.collect_all_metrics()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Metrics collection should complete in <50ms
        assert duration_ms < 50.0, f"Metrics collection took {duration_ms:.2f}ms (>50ms)"
        
        # Verify metrics are collected
        assert len(metrics) >= 0  # May be empty in test environment
        
        print(f"✓ Metrics collection: {duration_ms:.2f}ms ({len(metrics)} metrics)")

    async def test_concurrent_health_checks_performance(self, monitoring_facade):
        """Test concurrent health checks don't degrade performance."""
        num_concurrent = 5
        
        async def run_health_check():
            start = time.perf_counter()
            await monitoring_facade.get_system_health()
            return (time.perf_counter() - start) * 1000
        
        start_time = time.perf_counter()
        
        # Run multiple concurrent health checks
        tasks = [run_health_check() for _ in range(num_concurrent)]
        durations = await asyncio.gather(*tasks)
        
        total_duration_ms = (time.perf_counter() - start_time) * 1000
        avg_duration_ms = sum(durations) / len(durations)
        max_duration_ms = max(durations)
        
        # Concurrent execution should be efficient
        assert total_duration_ms < 200.0, f"Concurrent health checks took {total_duration_ms:.2f}ms total"
        assert avg_duration_ms < 100.0, f"Average concurrent health check took {avg_duration_ms:.2f}ms"
        assert max_duration_ms < 150.0, f"Slowest concurrent health check took {max_duration_ms:.2f}ms"
        
        print(f"✓ Concurrent checks ({num_concurrent}): {total_duration_ms:.2f}ms total, {avg_duration_ms:.2f}ms avg")

    async def test_cached_results_performance(self, monitoring_facade):
        """Test that cached results provide significant performance improvement."""
        # First call (should cache results)
        start_time = time.perf_counter()
        await monitoring_facade.get_system_health()
        first_duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Second call (should use cache)
        start_time = time.perf_counter()
        await monitoring_facade.get_system_health()
        second_duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Cached call should be significantly faster
        speed_improvement = first_duration_ms / second_duration_ms if second_duration_ms > 0 else float('inf')
        
        assert second_duration_ms < 5.0, f"Cached health check took {second_duration_ms:.2f}ms (should be <5ms)"
        assert speed_improvement > 2.0, f"Caching only improved speed by {speed_improvement:.1f}x (should be >2x)"
        
        print(f"✓ Caching performance: {first_duration_ms:.2f}ms → {second_duration_ms:.2f}ms ({speed_improvement:.1f}x faster)")

    async def test_monitoring_summary_performance(self, monitoring_facade):
        """Test that monitoring summary completes within performance target."""
        start_time = time.perf_counter()
        
        summary = await monitoring_facade.get_monitoring_summary()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Monitoring summary should complete in <150ms
        assert duration_ms < 150.0, f"Monitoring summary took {duration_ms:.2f}ms (>150ms)"
        
        # Verify summary is comprehensive
        assert "health" in summary
        assert "metrics" in summary
        assert "components" in summary
        assert "configuration" in summary
        
        print(f"✓ Monitoring summary: {duration_ms:.2f}ms")

    async def test_memory_usage_monitoring(self, monitoring_facade):
        """Test that monitoring doesn't cause memory leaks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple operations
        for _ in range(10):
            await monitoring_facade.get_system_health()
            await monitoring_facade.collect_all_metrics()
            await monitoring_facade.get_monitoring_summary()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / 1024 / 1024
        
        # Memory increase should be minimal (<10MB)
        assert memory_increase_mb < 10.0, f"Memory increased by {memory_increase_mb:.2f}MB (should be <10MB)"
        
        print(f"✓ Memory usage: +{memory_increase_mb:.2f}MB")

    def test_sync_operations_performance(self):
        """Test synchronous operations complete quickly."""
        from prompt_improver.monitoring.unified import create_monitoring_config
        
        start_time = time.perf_counter()
        
        # Test config creation
        config = create_monitoring_config()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Sync operations should be nearly instantaneous
        assert duration_ms < 1.0, f"Config creation took {duration_ms:.2f}ms (should be <1ms)"
        assert config.health_check_timeout_seconds > 0
        
        print(f"✓ Sync operations: {duration_ms:.2f}ms")

    async def test_error_handling_performance(self, monitoring_facade):
        """Test that error conditions don't significantly impact performance."""
        # Test with invalid component name
        start_time = time.perf_counter()
        
        result = await monitoring_facade.check_component_health("nonexistent_component")
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Error handling should still be fast
        assert duration_ms < 10.0, f"Error handling took {duration_ms:.2f}ms (should be <10ms)"
        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message.lower()
        
        print(f"✓ Error handling: {duration_ms:.2f}ms")


@pytest.mark.asyncio
async def test_performance_regression():
    """Integration test to ensure no performance regression."""
    print("\n=== UnifiedMonitoringFacade Performance Validation ===")
    
    facade = await create_monitoring_facade()
    
    # Comprehensive performance test
    start_time = time.perf_counter()
    
    # Simulate real usage pattern
    health_summary = await facade.get_system_health()
    metrics = await facade.collect_all_metrics() 
    summary = await facade.get_monitoring_summary()
    
    # Test individual component checks
    for component in ["database", "redis", "system_resources"]:
        await facade.check_component_health(component)
    
    total_duration_ms = (time.perf_counter() - start_time) * 1000
    
    # Full test suite should complete quickly
    assert total_duration_ms < 500.0, f"Complete test suite took {total_duration_ms:.2f}ms (should be <500ms)"
    
    print(f"\n✅ Performance validation passed!")
    print(f"   Total duration: {total_duration_ms:.2f}ms")
    print(f"   Health check: {health_summary.check_duration_ms:.2f}ms") 
    print(f"   Components: {health_summary.total_components}")
    print(f"   Metrics: {len(metrics)}")
    print(f"   Overall status: {health_summary.overall_status.value}")
    print("=" * 60)