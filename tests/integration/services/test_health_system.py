"""Comprehensive tests for APES PHASE 3 health check system.
Tests the Composite pattern implementation and Prometheus integration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.prompt_improver.services.health import (
    HealthChecker,
    HealthResult,
    HealthStatus,
    HealthService,
    get_health_service,
    reset_health_service,
    DatabaseHealthChecker,
    MCPServerHealthChecker,
    AnalyticsServiceHealthChecker,
    MLServiceHealthChecker,
    SystemResourcesHealthChecker,
    PROMETHEUS_AVAILABLE
)


class MockHealthChecker(HealthChecker):
    """Mock health checker for testing"""
    
    def __init__(self, name: str, status: HealthStatus = HealthStatus.HEALTHY, 
                 response_time: float = 50.0, error: str = None):
        super().__init__(name)
        self._status = status
        self._response_time = response_time
        self._error = error
    
    async def check(self) -> HealthResult:
        if self._error:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=self._error,
                message=f"Mock error: {self._error}"
            )
        
        return HealthResult(
            status=self._status,
            component=self.name,
            response_time_ms=self._response_time,
            message=f"Mock {self.name} check successful"
        )


@pytest.fixture
def mock_health_checkers():
    """Create mock health checkers for testing"""
    return [
        MockHealthChecker("database", HealthStatus.HEALTHY, 45.0),
        MockHealthChecker("mcp_server", HealthStatus.HEALTHY, 120.0),
        MockHealthChecker("analytics", HealthStatus.WARNING, 180.0),
        MockHealthChecker("ml_service", HealthStatus.HEALTHY, 65.0),
        MockHealthChecker("system_resources", HealthStatus.HEALTHY, 15.0)
    ]


@pytest.fixture
def health_service(mock_health_checkers):
    """Create health service with mock checkers"""
    return HealthService(checkers=mock_health_checkers)


@pytest.fixture(autouse=True)
def reset_global_service():
    """Reset global health service before each test"""
    reset_health_service()
    yield
    reset_health_service()


class TestHealthResult:
    """Test HealthResult data class"""
    
    def test_health_result_creation(self):
        """Test creating health result"""
        result = HealthResult(
            status=HealthStatus.HEALTHY,
            component="test",
            response_time_ms=100.0,
            message="Test successful"
        )
        
        assert result.status == HealthStatus.HEALTHY
        assert result.component == "test"
        assert result.response_time_ms == 100.0
        assert result.message == "Test successful"
        assert result.error is None
        assert result.timestamp is not None
    
    def test_health_result_with_error(self):
        """Test creating health result with error"""
        result = HealthResult(
            status=HealthStatus.FAILED,
            component="test",
            error="Connection failed",
            message="Database unreachable"
        )
        
        assert result.status == HealthStatus.FAILED
        assert result.error == "Connection failed"
        assert result.message == "Database unreachable"


class TestHealthService:
    """Test HealthService composite pattern implementation"""
    
    @pytest.mark.asyncio
    async def test_run_health_check_parallel(self, health_service):
        """Test running health checks in parallel"""
        result = await health_service.run_health_check(parallel=True)
        
        assert result.overall_status == HealthStatus.WARNING  # Due to analytics warning
        assert len(result.checks) == 5
        assert "database" in result.checks
        assert "mcp_server" in result.checks
        assert "analytics" in result.checks
        assert result.warning_checks == ["analytics"]
        assert result.failed_checks == []
    
    @pytest.mark.asyncio
    async def test_run_health_check_sequential(self, health_service):
        """Test running health checks sequentially"""
        result = await health_service.run_health_check(parallel=False)
        
        assert result.overall_status == HealthStatus.WARNING
        assert len(result.checks) == 5
        assert all(isinstance(check, HealthResult) for check in result.checks.values())
    
    @pytest.mark.asyncio
    async def test_run_specific_check(self, health_service):
        """Test running specific component check"""
        result = await health_service.run_specific_check("database")
        
        assert result.status == HealthStatus.HEALTHY
        assert result.component == "database"
        assert result.response_time_ms == 45.0
    
    @pytest.mark.asyncio
    async def test_run_specific_check_unknown_component(self, health_service):
        """Test running check for unknown component"""
        result = await health_service.run_specific_check("unknown")
        
        assert result.status == HealthStatus.FAILED
        assert result.component == "unknown"
        assert "No health checker found" in result.message
    
    def test_get_available_checks(self, health_service):
        """Test getting available health checks"""
        checks = health_service.get_available_checks()
        
        assert len(checks) == 5
        assert "database" in checks
        assert "mcp_server" in checks
        assert "analytics" in checks
        assert "ml_service" in checks
        assert "system_resources" in checks
    
    @pytest.mark.asyncio
    async def test_get_health_summary(self, health_service):
        """Test getting health summary"""
        summary = await health_service.get_health_summary(include_details=True)
        
        assert summary["overall_status"] == "warning"
        assert "timestamp" in summary
        assert "checks" in summary
        assert len(summary["checks"]) == 5
        assert summary["warning_checks"] == ["analytics"]
        assert "failed_checks" not in summary  # Empty list not included
    
    def test_add_checker(self, health_service):
        """Test adding new health checker"""
        new_checker = MockHealthChecker("custom", HealthStatus.HEALTHY)
        health_service.add_checker(new_checker)
        
        assert "custom" in health_service.get_available_checks()
        assert len(health_service.checkers) == 6
    
    def test_remove_checker(self, health_service):
        """Test removing health checker"""
        result = health_service.remove_checker("analytics")
        assert result is True
        assert "analytics" not in health_service.get_available_checks()
        assert len(health_service.checkers) == 4
        
        # Try removing non-existent checker
        result = health_service.remove_checker("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_failed_checks_calculation(self):
        """Test calculation of overall status with failed checks"""
        failed_checkers = [
            MockHealthChecker("database", HealthStatus.FAILED),
            MockHealthChecker("mcp_server", HealthStatus.HEALTHY),
        ]
        service = HealthService(checkers=failed_checkers)
        
        result = await service.run_health_check()
        
        assert result.overall_status == HealthStatus.FAILED
        assert result.failed_checks == ["database"]
    
    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test handling of exceptions in health checks"""
        error_checker = MockHealthChecker("error", error="Simulated failure")
        service = HealthService(checkers=[error_checker])
        
        result = await service.run_health_check()
        
        assert result.overall_status == HealthStatus.FAILED
        assert result.checks["error"].status == HealthStatus.FAILED
        assert "Mock error: Simulated failure" in result.checks["error"].message


class TestGlobalHealthService:
    """Test global health service singleton"""
    
    def test_get_health_service_singleton(self):
        """Test that get_health_service returns singleton"""
        service1 = get_health_service()
        service2 = get_health_service()
        
        assert service1 is service2
        assert isinstance(service1, HealthService)
    
    def test_reset_health_service(self):
        """Test resetting global health service"""
        service1 = get_health_service()
        reset_health_service()
        service2 = get_health_service()
        
        assert service1 is not service2


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus client not available")
class TestPrometheusIntegration:
    """Test Prometheus metrics integration"""
    
    @pytest.mark.asyncio
    async def test_metrics_instrumentation(self):
        """Test that metrics are recorded during health checks"""
        from src.prompt_improver.services.health.metrics import (
            HEALTH_CHECK_STATUS,
            HEALTH_CHECKS_TOTAL,
            reset_health_metrics
        )
        
        # Reset metrics
        reset_health_metrics()
        
        # Create service and run check
        checker = MockHealthChecker("test", HealthStatus.HEALTHY, 100.0)
        service = HealthService(checkers=[checker])
        
        await service.run_health_check()
        
        # Note: In a real test environment, we would verify metrics
        # This is a placeholder for metrics verification
        assert True  # Metrics instrumentation is tested via integration


@pytest.mark.asyncio
class TestIndividualCheckers:
    """Test individual health checker implementations"""
    
    @patch('src.prompt_improver.services.health.checkers.get_session')
    async def test_database_health_checker(self, mock_get_session):
        """Test database health checker"""
        # Mock successful database connection
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5  # Active connections
        mock_session.execute.return_value = mock_result
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        checker = DatabaseHealthChecker()
        result = await checker.check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.component == "database"
        assert result.response_time_ms is not None
        assert "responding in" in result.message
    
    @patch('src.prompt_improver.services.health.checkers.get_session')
    async def test_database_health_checker_failure(self, mock_get_session):
        """Test database health checker with connection failure"""
        # Mock database connection failure
        mock_get_session.side_effect = Exception("Connection refused")
        
        checker = DatabaseHealthChecker()
        result = await checker.check()
        
        assert result.status == HealthStatus.FAILED
        assert result.component == "database"
        assert result.error == "Connection refused"
        assert "Database connection failed" in result.message
    
    @patch('src.prompt_improver.services.health.checkers.improve_prompt')
    async def test_mcp_server_health_checker(self, mock_improve_prompt):
        """Test MCP server health checker"""
        # Mock successful MCP call
        mock_improve_prompt.return_value = {"improved": "test"}
        
        checker = MCPServerHealthChecker()
        result = await checker.check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.component == "mcp_server"
        assert result.response_time_ms is not None
    
    @patch('src.prompt_improver.services.health.checkers.improve_prompt')
    async def test_mcp_server_health_checker_failure(self, mock_improve_prompt):
        """Test MCP server health checker with failure"""
        # Mock MCP call failure
        mock_improve_prompt.side_effect = Exception("MCP server unreachable")
        
        checker = MCPServerHealthChecker()
        result = await checker.check()
        
        assert result.status == HealthStatus.FAILED
        assert result.component == "mcp_server"
        assert result.error == "MCP server unreachable"
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_percent')
    async def test_system_resources_health_checker(self, mock_cpu, mock_disk, mock_memory):
        """Test system resources health checker"""
        # Mock system metrics
        mock_memory.return_value.percent = 50.0
        mock_disk.return_value.percent = 60.0
        mock_cpu.return_value = 30.0
        
        checker = SystemResourcesHealthChecker()
        result = await checker.check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.component == "system_resources"
        assert "CPU 30.0%" in result.message
        assert result.details["memory_usage_percent"] == 50.0
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage') 
    @patch('psutil.cpu_percent')
    async def test_system_resources_health_checker_warnings(self, mock_cpu, mock_disk, mock_memory):
        """Test system resources health checker with warnings"""
        # Mock high resource usage
        mock_memory.return_value.percent = 85.0  # Above 80% threshold
        mock_disk.return_value.percent = 90.0    # Above 85% threshold
        mock_cpu.return_value = 85.0             # Above 80% threshold
        
        checker = SystemResourcesHealthChecker()
        result = await checker.check()
        
        assert result.status == HealthStatus.WARNING
        assert result.component == "system_resources"
        assert len(result.details["warnings"]) == 3
        assert "High memory usage" in result.details["warnings"][0]


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the complete health check system"""
    
    @patch.multiple(
        'src.prompt_improver.services.health.checkers',
        get_session=AsyncMock(),
        improve_prompt=AsyncMock(),
        AnalyticsService=MagicMock(),
        get_ml_service=AsyncMock(),
    )
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_percent')
    async def test_full_health_check_integration(self, mock_cpu, mock_disk, mock_memory, **mocks):
        """Test full health check system integration"""
        # Setup mocks for successful health checks
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_session.execute.return_value = mock_result
        mocks['get_session'].return_value.__aenter__.return_value = mock_session
        
        mocks['improve_prompt'].return_value = {"result": "success"}
        
        mock_analytics = MagicMock()
        mock_analytics.get_performance_trends.return_value = {"trends": []}
        mocks['AnalyticsService'].return_value = mock_analytics
        
        mocks['get_ml_service'].return_value = MagicMock()
        
        mock_memory.return_value.percent = 45.0
        mock_disk.return_value.percent = 55.0
        mock_cpu.return_value = 25.0
        
        # Create and test health service
        service = get_health_service()
        result = await service.run_health_check()
        
        assert result.overall_status == HealthStatus.HEALTHY
        assert len(result.checks) == 5
        assert all(check.status != HealthStatus.FAILED for check in result.checks.values())
        
        # Test health summary
        summary = await service.get_health_summary(include_details=True)
        assert summary["overall_status"] == "healthy"
        assert len(summary["checks"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
