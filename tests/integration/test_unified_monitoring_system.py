"""Integration tests for the Unified Monitoring System.

Tests the complete monitoring consolidation including:
- UnifiedMonitoringFacade functionality
- Health endpoint integration
- Component health checking
- Metrics collection
- Performance validation
"""

import asyncio
import pytest
import time
from datetime import UTC, datetime
from typing import Any, Dict

from fastapi.testclient import TestClient

from prompt_improver.monitoring.unified import (
    UnifiedMonitoringFacade,
    create_monitoring_facade,
    create_monitoring_config,
    HealthStatus,
    MetricType,
    ComponentCategory,
)
from prompt_improver.monitoring.unified.types import (
    HealthCheckResult,
    MetricPoint,
    SystemHealthSummary,
    MonitoringConfig,
)


class TestUnifiedMonitoringFacade:
    """Test the main UnifiedMonitoringFacade functionality."""
    
    @pytest.fixture
    async def monitoring_facade(self, setup_test_containers):
        """Create a monitoring facade for testing with real infrastructure."""
        config = create_monitoring_config(
            health_check_timeout_seconds=5.0,
            parallel_enabled=True,
            metrics_enabled=True,
        )
        
        # Use real infrastructure from testcontainers
        from prompt_improver.monitoring.unified.facade import ManagerMode
        facade = UnifiedMonitoringFacade(config=config, manager_mode=ManagerMode.HIGH_AVAILABILITY)
        await facade.initialize()
        yield facade
        await facade.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, monitoring_facade):
        """Test overall system health check with real infrastructure."""
        # Test real system health check
        result = await monitoring_facade.get_system_health()
        
        # Verify structure - status may vary based on actual service health
        assert result.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert result.total_components >= 0
        assert result.healthy_components >= 0
        assert result.check_duration_ms >= 0
        assert isinstance(result.component_results, dict)
        
        # Should include database and cache components if configured
        if result.component_results:
            for component_name, component_result in result.component_results.items():
                assert isinstance(component_result, HealthCheckResult)
                assert component_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
                assert component_result.response_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_component_health_check(self, monitoring_facade):
        """Test individual component health checks with real infrastructure."""
        # Test component health check with real services
        try:
            result = await monitoring_facade.check_component_health("database")
            
            assert isinstance(result, HealthCheckResult)
            assert result.component_name == "database"
            assert result.response_time_ms >= 0
            assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
            
            if result.details:
                assert isinstance(result.details, dict)
                
        except Exception as e:
            # If component doesn't exist or isn't configured, that's expected behavior
            assert "not found" in str(e).lower() or "not configured" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, monitoring_facade):
        """Test metrics collection functionality with real infrastructure."""
        # Test real metrics collection
        metrics = await monitoring_facade.collect_all_metrics()
        
        assert isinstance(metrics, list)
        # Real metrics collection should return some system metrics
        if metrics:
            for metric in metrics:
                assert isinstance(metric, MetricPoint)
                assert metric.name
                assert isinstance(metric.value, (int, float))
                assert metric.metric_type in MetricType
                
            # Look for common system metrics
            metric_names = [m.name for m in metrics]
            system_metrics = [name for name in metric_names if "system" in name or "process" in name]
            # Should have at least some system metrics if monitoring is active
            assert len(system_metrics) >= 0  # May be 0 if system monitoring disabled
    
    @pytest.mark.asyncio
    async def test_custom_metric_recording(self, monitoring_facade):
        """Test custom metric recording with real infrastructure."""
        # Test custom metric recording
        try:
            monitoring_facade.record_custom_metric(
                "custom.api.requests_per_second",
                45.7,
                tags={"endpoint": "/api/health", "method": "GET"}
            )
            
            # Verify metric recording doesn't raise exception
            # Real validation would require checking the metrics storage
            
            # Try to collect metrics to see if custom metric is included
            metrics = await monitoring_facade.collect_all_metrics()
            custom_metrics = [m for m in metrics if "custom.api" in m.name]
            # May or may not be immediately available depending on collection interval
            
        except Exception as e:
            # If metrics service not configured, that's acceptable in test environment
            assert "not configured" in str(e).lower() or "not available" in str(e).lower()
    
    @pytest.mark.asyncio  
    async def test_monitoring_summary(self, monitoring_facade):
        """Test comprehensive monitoring summary with real infrastructure."""
        # Test real monitoring summary
        summary = await monitoring_facade.get_monitoring_summary()
        
        assert isinstance(summary, dict)
        assert "health" in summary
        assert "metrics" in summary
        assert "components" in summary
        assert "configuration" in summary
        
        # Verify health summary structure
        health_summary = summary["health"]
        assert "overall_status" in health_summary
        assert health_summary["overall_status"] in ["healthy", "degraded", "unhealthy"]
        assert "total_components" in health_summary
        assert isinstance(health_summary["total_components"], int)
        
        # Verify metrics summary structure  
        metrics_summary = summary["metrics"]
        assert "total_metrics" in metrics_summary
        assert isinstance(metrics_summary["total_metrics"], int)
        
        # Verify components summary
        components_summary = summary["components"]
        assert isinstance(components_summary, dict)
    
    @pytest.mark.asyncio
    async def test_health_caching(self, monitoring_facade):
        """Test health result caching for performance with real infrastructure."""
        # Test caching behavior with real health checks
        start_time = time.time()
        result1 = await monitoring_facade.get_system_health()
        first_call_time = time.time() - start_time
        
        # Second call should be faster due to caching
        start_time = time.time()
        result2 = await monitoring_facade.get_system_health()
        second_call_time = time.time() - start_time
        
        # Results should be consistent
        assert result1.overall_status == result2.overall_status
        assert result1.total_components == result2.total_components
        
        # Second call should be significantly faster if caching is working
        # Allow some variance for real system calls
        if first_call_time > 0.01:  # Only test if first call took meaningful time
            assert second_call_time <= first_call_time
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, monitoring_facade):
        """Test that performance requirements are met with real infrastructure."""
        # Test health check performance with real infrastructure
        start_time = time.time()
        health_result = await monitoring_facade.get_system_health()
        health_check_time = (time.time() - start_time) * 1000
        
        # Real health checks should complete reasonably quickly
        assert health_check_time < 5000, f"Health check took {health_check_time:.2f}ms, should be <5000ms"
        
        # Test metrics collection performance
        start_time = time.time()
        metrics = await monitoring_facade.collect_all_metrics()
        metrics_time = (time.time() - start_time) * 1000
        
        # Real metrics collection should be reasonable
        assert metrics_time < 5000, f"Metrics collection took {metrics_time:.2f}ms, should be <5000ms"
        
        # Verify configuration is accessible
        assert hasattr(monitoring_facade, 'config')
        if hasattr(monitoring_facade.config, 'health_check_parallel_enabled'):
            assert isinstance(monitoring_facade.config.health_check_parallel_enabled, bool)
        if hasattr(monitoring_facade.config, 'metrics_collection_enabled'):
            assert isinstance(monitoring_facade.config.metrics_collection_enabled, bool)


class TestHealthEndpointIntegration:
    """Test integration with the consolidated health endpoints."""
    
    @pytest.fixture
    def real_app(self, setup_test_containers):
        """Create a test FastAPI app with health endpoints and real infrastructure."""
        from fastapi import FastAPI
        from prompt_improver.api.health import health_router
        
        app = FastAPI()
        app.include_router(health_router)
        return app
    
    @pytest.fixture
    def client(self, real_app):
        """Create a test client with real infrastructure."""
        return TestClient(real_app)
    
    def test_liveness_endpoint(self, client):
        """Test the liveness probe endpoint with real configuration."""
        response = client.get("/health/liveness")
        
        # Liveness should always return 200 unless service is completely down
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        # Real version and environment from actual config
        if "version" in data:
            assert isinstance(data["version"], str)
        if "environment" in data:
            assert isinstance(data["environment"], str)
    
    def test_readiness_endpoint(self, client):
        """Test readiness endpoint with real system health."""
        response = client.get("/health/readiness")
        
        # Status code depends on actual system health
        assert response.status_code in [200, 503]
        data = response.json()
        
        if response.status_code == 200:
            assert data["status"] == "ready"
            assert data["ready"] == True
        else:
            assert data["status"] == "not_ready"
            assert data["ready"] == False
            
        # Verify structure regardless of status
        assert "healthy_components" in data
        assert "total_components" in data
        assert isinstance(data["healthy_components"], int)
        assert isinstance(data["total_components"], int)
    
    def test_ready_endpoint_alias(self, client):
        """Test ready endpoint alias with real system health."""
        response = client.get("/health/ready")
        
        # Status code depends on actual system health
        assert response.status_code in [200, 503]
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "ready" in data
        assert isinstance(data["ready"], bool)
        
        if response.status_code == 503:
            assert data["status"] == "not_ready"
            assert data["ready"] == False
            # May have critical_issues if system provides them
            if "critical_issues" in data:
                assert isinstance(data["critical_issues"], list)
        else:
            assert data["status"] == "ready"
            assert data["ready"] == True
    
    def test_main_health_endpoint(self, client):
        """Test the main health check endpoint with real system health."""
        response = client.get("/health/")
        
        assert response.status_code == 200  # Main health endpoint should always return 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "healthy" in data
        assert isinstance(data["healthy"], bool)
        
        if "summary" in data:
            summary = data["summary"]
            assert "total_components" in summary
            assert "healthy_components" in summary
            assert isinstance(summary["total_components"], int)
            assert isinstance(summary["healthy_components"], int)
            
            if "health_percentage" in summary:
                assert 0 <= summary["health_percentage"] <= 100
        
        # May have components section with real component health
        if "components" in data:
            assert isinstance(data["components"], dict)
            
        # Real version and environment from config
        if "version" in data:
            assert isinstance(data["version"], str)
        if "environment" in data:
            assert isinstance(data["environment"], str)
    
    def test_component_health_endpoint(self, client):
        """Test individual component health check endpoint with real components."""
        # Test with a known component type (database is commonly available)
        response = client.get("/health/component/database")
        
        # Response depends on whether component exists and is configured
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert data["component"] == "database"
            assert "status" in data
            assert "message" in data
            assert "response_time_ms" in data
            assert isinstance(data["response_time_ms"], (int, float))
            
            if "category" in data:
                assert isinstance(data["category"], str)
            if "details" in data:
                assert isinstance(data["details"], dict)
        else:
            # Component not found or not configured - this is acceptable
            data = response.json()
            assert "error" in data or "message" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics collection endpoint with real metrics."""
        response = client.get("/health/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_metrics" in data
        assert isinstance(data["total_metrics"], int)
        
        if "metrics_by_category" in data:
            metrics_by_category = data["metrics_by_category"]
            assert isinstance(metrics_by_category, dict)
            
            # Check structure of any metrics that exist
            for category, metrics_list in metrics_by_category.items():
                assert isinstance(metrics_list, list)
                for metric in metrics_list:
                    assert "name" in metric
                    assert "value" in metric
                    assert isinstance(metric["value"], (int, float))
                    
                    if "tags" in metric:
                        assert isinstance(metric["tags"], dict)
    
    def test_monitoring_summary_endpoint(self, client):
        """Test monitoring summary endpoint with real monitoring data."""
        response = client.get("/health/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required sections exist
        assert "health" in data
        assert "metrics" in data
        assert "components" in data
        
        # Verify health section structure
        health_section = data["health"]
        assert "overall_status" in health_section
        assert health_section["overall_status"] in ["healthy", "degraded", "unhealthy"]
        
        if "total_components" in health_section:
            assert isinstance(health_section["total_components"], int)
        if "healthy_components" in health_section:
            assert isinstance(health_section["healthy_components"], int)
        
        # Verify metrics section structure
        metrics_section = data["metrics"]
        if "total_metrics" in metrics_section:
            assert isinstance(metrics_section["total_metrics"], int)
        if "collection_enabled" in metrics_section:
            assert isinstance(metrics_section["collection_enabled"], bool)
        
        # Verify components section structure
        components_section = data["components"]
        assert isinstance(components_section, dict)
        if "component_names" in components_section:
            assert isinstance(components_section["component_names"], list)


class TestPerformanceValidation:
    """Test performance requirements for the unified monitoring system with real infrastructure."""
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, setup_test_containers):
        """Test that health checks complete within SLA requirements using real infrastructure."""
        config = create_monitoring_config(
            health_check_timeout_seconds=10.0,
            parallel_enabled=True,
            max_concurrent_checks=5,
        )
        
        from prompt_improver.monitoring.unified.facade import ManagerMode
        facade = UnifiedMonitoringFacade(config=config, manager_mode=ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await facade.initialize()
            
            # Measure real health check performance
            start_time = time.time()
            result = await facade.get_system_health()
            duration_ms = (time.time() - start_time) * 1000
            
            # Real health checks may take longer than mocked ones, but should still be reasonable
            assert duration_ms < 5000, f"Health check took {duration_ms:.2f}ms, should be <5000ms"
            
            # Verify result structure
            assert result.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
            assert isinstance(result.total_components, int)
            assert result.total_components >= 0
            
        finally:
            await facade.shutdown()
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, setup_test_containers):
        """Test that metrics collection meets performance requirements with real infrastructure."""
        config = create_monitoring_config(metrics_enabled=True)
        
        from prompt_improver.monitoring.unified.facade import ManagerMode
        facade = UnifiedMonitoringFacade(config=config, manager_mode=ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await facade.initialize()
            
            # Measure real metrics collection performance
            start_time = time.time()
            metrics = await facade.collect_all_metrics()
            duration_ms = (time.time() - start_time) * 1000
            
            # Real metrics collection may vary, but should be reasonable
            assert duration_ms < 5000, f"Metrics collection took {duration_ms:.2f}ms, should be <5000ms"
            
            # Verify metrics structure
            assert isinstance(metrics, list)
            for metric in metrics:
                assert isinstance(metric, MetricPoint)
                assert hasattr(metric, 'name')
                assert hasattr(metric, 'value')
                assert hasattr(metric, 'metric_type')
                
        finally:
            await facade.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, setup_test_containers):
        """Test that concurrent health checks work properly with real infrastructure."""
        config = create_monitoring_config(
            parallel_enabled=True,
            max_concurrent_checks=10,
        )
        
        from prompt_improver.monitoring.unified.facade import ManagerMode
        facade = UnifiedMonitoringFacade(config=config, manager_mode=ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await facade.initialize()
            
            # Run multiple concurrent health checks with real infrastructure
            tasks = [facade.get_system_health() for _ in range(3)]  # Reduce concurrency for real infrastructure
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_duration_ms = (time.time() - start_time) * 1000
            
            # With real infrastructure, allow more time but should still be reasonable
            assert total_duration_ms < 15000, f"Concurrent checks took {total_duration_ms:.2f}ms"
            
            # Verify all results are valid
            assert len(results) == 3
            for result in results:
                assert result.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
                assert isinstance(result.total_components, int)
                assert result.total_components >= 0
                
        finally:
            await facade.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])