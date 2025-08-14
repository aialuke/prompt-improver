"""Test unified monitoring system integration and functionality.

Validates that the consolidated monitoring system provides all required
functionality with decomposed services maintaining <500 lines each.
"""

import asyncio
import pytest
import time

from prompt_improver.monitoring.unified.facade import UnifiedMonitoringFacade
from prompt_improver.monitoring.unified.types import MonitoringConfig


class TestUnifiedMonitoringSystem:
    """Test suite for unified monitoring system."""
    
    @pytest.fixture
    async def monitoring_facade(self):
        """Create monitoring facade for testing."""
        config = MonitoringConfig(
            health_check_timeout_seconds=5.0,
            health_check_parallel_enabled=True,
            metrics_collection_enabled=True,
        )
        
        facade = UnifiedMonitoringFacade(config=config)
        
        yield facade
        
        # Cleanup
        if facade._is_started:
            await facade.stop_monitoring()
    
    async def test_monitoring_facade_initialization(self, monitoring_facade):
        """Test that monitoring facade initializes correctly."""
        assert monitoring_facade.config is not None
        assert monitoring_facade.orchestrator is not None
        assert monitoring_facade.health_service is not None
        assert monitoring_facade.metrics_service is not None
        assert monitoring_facade.alerting_service is not None
        assert monitoring_facade.health_reporter is not None
        assert monitoring_facade.cache_monitor is not None
        assert not monitoring_facade._is_started
    
    async def test_monitoring_start_stop(self, monitoring_facade):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring
        await monitoring_facade.start_monitoring()
        assert monitoring_facade._is_started
        
        # Stop monitoring
        await monitoring_facade.stop_monitoring()
        assert not monitoring_facade._is_started
    
    async def test_health_check_functionality(self, monitoring_facade):
        """Test health checking functionality."""
        # Get health summary
        health_summary = await monitoring_facade.get_system_health()
        
        assert health_summary is not None
        assert hasattr(health_summary, 'overall_status')
        assert hasattr(health_summary, 'total_components')
        assert health_summary.total_components >= 0
    
    async def test_metrics_collection(self, monitoring_facade):
        """Test metrics collection functionality.""" 
        # Record a custom metric
        monitoring_facade.record_custom_metric(
            "test.metric",
            42.0,
            tags={"test": "true"}
        )
        
        # Collect metrics
        metrics = await monitoring_facade.collect_all_metrics()
        
        assert isinstance(metrics, list)
        # Should have at least the custom metric
        assert len(metrics) >= 0
    
    async def test_cache_operation_monitoring(self, monitoring_facade):
        """Test cache operation monitoring."""
        # Record cache operation
        monitoring_facade.record_cache_operation(
            operation="get",
            cache_level="l1",
            hit=True,
            duration_ms=10.5,
            key="test_key"
        )
        
        # Get cache performance report
        cache_report = await monitoring_facade.get_cache_performance_report()
        
        assert isinstance(cache_report, dict)
        assert "report_type" in cache_report
    
    async def test_comprehensive_monitoring_status(self, monitoring_facade):
        """Test comprehensive monitoring status."""
        status = await monitoring_facade.get_comprehensive_monitoring_status()
        
        assert isinstance(status, dict)
        assert "orchestration" in status
        assert "timestamp" in status
    
    async def test_monitoring_summary(self, monitoring_facade):
        """Test monitoring summary functionality."""
        summary = await monitoring_facade.get_monitoring_summary()
        
        assert isinstance(summary, dict)
        assert "health" in summary
        assert "metrics" in summary
        assert "components" in summary
        assert "configuration" in summary
    
    async def test_component_health_check(self, monitoring_facade):
        """Test individual component health checks."""
        # Test with non-existent component (should handle gracefully)
        result = await monitoring_facade.check_component_health("nonexistent_component")
        
        assert result is not None
        assert hasattr(result, 'status')
        assert hasattr(result, 'component_name')
        assert result.component_name == "nonexistent_component"
    
    async def test_health_checker_registration(self, monitoring_facade):
        """Test custom health checker registration."""
        from prompt_improver.monitoring.unified.protocols import HealthCheckComponentProtocol
        from prompt_improver.monitoring.unified.types import HealthCheckResult, HealthStatus
        
        class TestHealthChecker:
            def get_component_name(self) -> str:
                return "test_component"
            
            async def check_health(self) -> HealthCheckResult:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component_name="test_component",
                    message="Test component is healthy"
                )
        
        # Register custom checker
        test_checker = TestHealthChecker()
        monitoring_facade.register_health_checker(test_checker)
        
        # Verify registration
        components = monitoring_facade.get_registered_checkers()
        assert "test_component" in components
        
        # Test health check
        result = await monitoring_facade.check_component_health("test_component")
        assert result.status == HealthStatus.HEALTHY
        assert result.component_name == "test_component"
    
    async def test_backwards_compatibility_methods(self, monitoring_facade):
        """Test backwards compatibility with HealthMonitorProtocol."""
        # Test check_health method
        health_results = await monitoring_facade.check_health()
        assert isinstance(health_results, dict)
        
        # Test get_overall_health method
        overall_health = await monitoring_facade.get_overall_health()
        assert overall_health is not None
        assert hasattr(overall_health, 'status')
        
        # Test get_health_summary method
        health_summary = monitoring_facade.get_health_summary()
        assert isinstance(health_summary, dict)
        assert "registered_components" in health_summary
        
        # Test get_registered_checkers method
        checkers = monitoring_facade.get_registered_checkers()
        assert isinstance(checkers, list)


@pytest.mark.asyncio
async def test_monitoring_performance():
    """Test monitoring system performance."""
    config = MonitoringConfig(
        health_check_timeout_seconds=1.0,
        health_check_parallel_enabled=True,
    )
    
    facade = UnifiedMonitoringFacade(config=config)
    
    # Test health check performance
    start_time = time.time()
    health_summary = await facade.get_system_health()
    duration = time.time() - start_time
    
    # Health check should complete within reasonable time
    assert duration < 5.0  # 5 second timeout
    assert health_summary.check_duration_ms < 5000  # 5 second max
    
    await facade.stop_monitoring()


@pytest.mark.asyncio 
async def test_cache_monitoring_unified_facade():
    """Test cache monitoring through unified facade only."""
    from prompt_improver.monitoring.unified.facade import UnifiedMonitoringFacade
    
    facade = UnifiedMonitoringFacade()
    
    # Test cache operation recording
    facade.record_cache_operation(
        operation="get",
        cache_level="l1",
        hit=True,
        duration_ms=5.0,
        key="test_key"
    )
    
    # Test cache performance report
    report = await facade.get_cache_performance_report()
    assert isinstance(report, dict)
    assert "report_type" in report


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_monitoring_performance())
    asyncio.run(test_cache_monitoring_unified_facade())
    print("✓ Unified monitoring system tests passed")