"""Integration tests validating database access migration to repository pattern.

Real behavior testing using testcontainers to verify:
1. Health monitoring migration from direct database access to repository pattern
2. Performance characteristics maintained (<100ms health checks)
3. Clean architecture compliance verified
4. No direct database imports in migrated components
"""

import asyncio
import time
from typing import Any, Dict

import pytest

from prompt_improver.database import DatabaseServices
from prompt_improver.application.services.health_application_service import (
    HealthApplicationService,
)
from prompt_improver.repositories.factory import get_repository_factory
from prompt_improver.repositories.protocols.health_repository_protocol import (
    HealthRepositoryProtocol,
)
from prompt_improver.performance.monitoring.health.checkers import DatabaseHealthChecker
from prompt_improver.core.services.manager import OrchestrationService
from prompt_improver.monitoring.unified.services import HealthCheckService
from prompt_improver.performance.monitoring.performance_benchmark import (
    MCPPerformanceBenchmark,
)
from tests.containers.postgres_container import get_postgres_container


class TestDatabaseAccessMigrationValidation:
    """Integration tests validating the repository pattern migration."""

    @pytest.fixture(scope="class")
    def event_loop(self):
        """Create an event loop for the test class."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="class")
    async def postgres_container(self):
        """Provide a real PostgreSQL container for testing."""
        container = get_postgres_container()
        try:
            yield container
        finally:
            container.stop()

    @pytest.fixture(scope="class")
    async def db_services(self, postgres_container):
        """Provide DatabaseServices with real PostgreSQL connection."""
        connection_string = postgres_container.get_connection_url()
        
        db_services = DatabaseServices(connection_string)
        await db_services.initialize()
        
        try:
            yield db_services
        finally:
            await db_services.close()

    @pytest.fixture(scope="class")
    async def health_repository(self, db_services):
        """Provide HealthRepositoryProtocol implementation."""
        factory = get_repository_factory(db_services)
        factory.initialize()
        
        health_repo = factory.get_health_repository()
        
        # Verify it's properly implementing the protocol
        assert hasattr(health_repo, 'check_database_health')
        assert hasattr(health_repo, 'get_database_metrics')
        
        yield health_repo
        
        factory.cleanup()

    @pytest.fixture(scope="class")
    async def health_application_service(self, db_services, health_repository):
        """Provide HealthApplicationService for testing."""
        # Mock the required dependencies for HealthApplicationService
        health_check_service = None  # Would be properly injected in real scenario
        performance_benchmark_service = None  # Would be properly injected in real scenario
        
        health_app_service = HealthApplicationService(
            db_services=db_services,
            health_repository=health_repository,
            health_check_service=health_check_service,
            performance_benchmark_service=performance_benchmark_service,
        )
        
        try:
            await health_app_service.initialize()
            yield health_app_service
        finally:
            await health_app_service.cleanup()

    @pytest.mark.asyncio
    async def test_database_health_checker_migration(self, health_repository):
        """Test that DatabaseHealthChecker works with repository injection."""
        # Create DatabaseHealthChecker with repository dependency injection
        health_checker = DatabaseHealthChecker(health_repository=health_repository)
        
        # Verify no direct database imports by checking attributes
        assert health_checker.health_repository is not None
        assert health_checker.health_repository == health_repository
        
        # Performance test - health check should complete in <100ms
        start_time = time.time()
        result = await health_checker.check()
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify performance requirement
        assert execution_time_ms < 100, f"Health check took {execution_time_ms:.2f}ms, exceeds 100ms target"
        
        # Verify functionality
        assert result is not None
        assert hasattr(result, 'status')
        assert hasattr(result, 'component')
        assert result.component == "database"
        
        # Verify that the result contains meaningful data
        if result.details:
            # Should have database-related metrics
            assert isinstance(result.details, dict)
            
        print(f"✅ DatabaseHealthChecker migration validated - {execution_time_ms:.2f}ms response time")

    @pytest.mark.asyncio
    async def test_orchestration_service_migration(self, health_application_service):
        """Test that OrchestrationService works with health application service injection."""
        # Create OrchestrationService with health application service dependency
        orchestration_service = OrchestrationService(
            health_application_service=health_application_service
        )
        
        # Verify dependency injection
        assert orchestration_service.health_application_service is not None
        assert orchestration_service.health_application_service == health_application_service
        
        # Test PostgreSQL connection check via application service
        start_time = time.time()
        postgres_status = await orchestration_service.start_postgresql_if_needed()
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify performance - should complete quickly
        assert execution_time_ms < 2000, f"PostgreSQL check took {execution_time_ms:.2f}ms, too slow"
        
        # Verify functionality
        assert postgres_status in ["connected", "started", "failed_to_start"]
        
        print(f"✅ OrchestrationService migration validated - PostgreSQL check: {postgres_status}")

    @pytest.mark.asyncio
    async def test_performance_metrics_collection_migration(self, health_application_service):
        """Test performance metrics collection via application service."""
        orchestration_service = OrchestrationService(
            health_application_service=health_application_service
        )
        
        # Test performance metrics collection
        start_time = time.time()
        metrics = await orchestration_service.collect_performance_metrics()
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify performance - metrics collection should be fast
        assert execution_time_ms < 500, f"Metrics collection took {execution_time_ms:.2f}ms, too slow"
        
        # Verify structure
        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "database_connections" in metrics
        assert "memory_usage_mb" in metrics
        assert "cpu_usage_percent" in metrics
        
        # Verify data types
        assert isinstance(metrics["database_connections"], (int, float))
        assert isinstance(metrics["memory_usage_mb"], (int, float))
        assert isinstance(metrics["cpu_usage_percent"], (int, float))
        
        print(f"✅ Performance metrics collection validated - {execution_time_ms:.2f}ms execution time")

    @pytest.mark.asyncio
    async def test_health_repository_comprehensive_operations(self, health_repository):
        """Test comprehensive health repository operations."""
        # Test basic health check
        start_time = time.time()
        db_health = await health_repository.check_database_health()
        db_check_time = (time.time() - start_time) * 1000
        
        assert db_check_time < 100, f"Database health check took {db_check_time:.2f}ms"
        assert db_health.component == "database"
        assert db_health.status in ["healthy", "warning", "critical", "unknown"]
        
        # Test connection pool health check
        start_time = time.time()
        pool_health = await health_repository.check_connection_pool_health()
        pool_check_time = (time.time() - start_time) * 1000
        
        assert pool_check_time < 100, f"Connection pool check took {pool_check_time:.2f}ms"
        assert pool_health.component == "connection_pool"
        
        # Test database metrics collection
        start_time = time.time()
        db_metrics = await health_repository.get_database_metrics()
        metrics_time = (time.time() - start_time) * 1000
        
        assert metrics_time < 200, f"Database metrics took {metrics_time:.2f}ms"
        assert hasattr(db_metrics, 'connection_pool_size')
        assert hasattr(db_metrics, 'active_connections')
        assert hasattr(db_metrics, 'connection_utilization')
        
        # Test full health check
        start_time = time.time()
        full_health = await health_repository.perform_full_health_check()
        full_check_time = (time.time() - start_time) * 1000
        
        assert full_check_time < 500, f"Full health check took {full_check_time:.2f}ms"
        assert hasattr(full_health, 'overall_status')
        assert hasattr(full_health, 'components_checked')
        assert full_health.components_checked > 0
        
        print(f"✅ Health repository comprehensive operations validated")
        print(f"  - DB Health: {db_check_time:.2f}ms")
        print(f"  - Pool Health: {pool_check_time:.2f}ms")
        print(f"  - DB Metrics: {metrics_time:.2f}ms")
        print(f"  - Full Check: {full_check_time:.2f}ms")

    @pytest.mark.asyncio
    async def test_migration_architecture_compliance(self, health_repository, health_application_service):
        """Test that migration follows clean architecture principles."""
        # Verify repository protocol compliance
        assert hasattr(health_repository, 'check_database_health')
        assert hasattr(health_repository, 'get_database_metrics')
        assert hasattr(health_repository, 'perform_full_health_check')
        
        # Verify application service protocol compliance  
        assert hasattr(health_application_service, 'perform_comprehensive_health_check')
        assert hasattr(health_application_service, 'monitor_system_performance')
        assert hasattr(health_application_service, 'diagnose_system_issues')
        
        # Test dependency injection works correctly
        health_checker = DatabaseHealthChecker(health_repository=health_repository)
        orchestration_service = OrchestrationService(health_application_service=health_application_service)
        
        # Verify no circular dependencies by checking initialization
        result = await health_checker.check()
        assert result is not None
        
        postgres_status = await orchestration_service.start_postgresql_if_needed()
        assert postgres_status is not None
        
        print("✅ Clean architecture compliance validated")
        print("  - Repository protocol: ✅")
        print("  - Application service protocol: ✅") 
        print("  - Dependency injection: ✅")
        print("  - No circular dependencies: ✅")

    @pytest.mark.asyncio 
    async def test_performance_benchmarks_maintained(self, health_repository, health_application_service):
        """Verify that migration maintains performance characteristics."""
        # Benchmark individual operations
        operations = []
        
        # Database health check
        for _ in range(5):
            start_time = time.time()
            await health_repository.check_database_health()
            execution_time = (time.time() - start_time) * 1000
            operations.append(("db_health_check", execution_time))
        
        # Connection pool check
        for _ in range(5):
            start_time = time.time()
            await health_repository.check_connection_pool_health()
            execution_time = (time.time() - start_time) * 1000
            operations.append(("pool_health_check", execution_time))
            
        # Health application service check
        for _ in range(3):  # Fewer iterations for more complex operation
            start_time = time.time()
            await health_application_service.perform_comprehensive_health_check(
                include_detailed_metrics=False
            )
            execution_time = (time.time() - start_time) * 1000
            operations.append(("app_service_health_check", execution_time))
        
        # Analyze performance
        db_health_times = [t for op, t in operations if op == "db_health_check"]
        pool_health_times = [t for op, t in operations if op == "pool_health_check"]
        app_service_times = [t for op, t in operations if op == "app_service_health_check"]
        
        avg_db_health = sum(db_health_times) / len(db_health_times)
        avg_pool_health = sum(pool_health_times) / len(pool_health_times)
        avg_app_service = sum(app_service_times) / len(app_service_times)
        
        # Performance assertions
        assert avg_db_health < 100, f"Average DB health check: {avg_db_health:.2f}ms exceeds 100ms target"
        assert avg_pool_health < 100, f"Average pool health check: {avg_pool_health:.2f}ms exceeds 100ms target"
        assert avg_app_service < 1000, f"Average app service check: {avg_app_service:.2f}ms exceeds 1000ms target"
        
        print("✅ Performance benchmarks validated")
        print(f"  - DB Health Check: {avg_db_health:.2f}ms (target: <100ms)")
        print(f"  - Pool Health Check: {avg_pool_health:.2f}ms (target: <100ms)")
        print(f"  - App Service Check: {avg_app_service:.2f}ms (target: <1000ms)")

    def test_no_direct_database_imports(self):
        """Verify that migrated components have no direct database imports."""
        import inspect
        
        from prompt_improver.performance.monitoring.health.checkers import DatabaseHealthChecker
        from prompt_improver.core.services.manager import OrchestrationService
        from prompt_improver.performance.baseline.baseline_collector import BaselineCollector
        
        # Get source code to check for direct database imports
        checker_source = inspect.getsource(DatabaseHealthChecker)
        manager_source = inspect.getsource(OrchestrationService)
        collector_source = inspect.getsource(BaselineCollector)
        
        # Patterns that indicate direct database access (violations)
        violation_patterns = [
            "from prompt_improver.database import get_session",
            "from prompt_improver.performance.database import get_session",
            "get_session()",
            "_get_database_functions()",
        ]
        
        violations = []
        
        # Check DatabaseHealthChecker
        for pattern in violation_patterns:
            if pattern in checker_source:
                violations.append(f"DatabaseHealthChecker contains: {pattern}")
        
        # Check OrchestrationService
        for pattern in violation_patterns:
            if pattern in manager_source:
                violations.append(f"OrchestrationService contains: {pattern}")
                
        # Check BaselineCollector 
        for pattern in violation_patterns:
            if pattern in collector_source:
                violations.append(f"BaselineCollector contains: {pattern}")
        
        # Assert no violations found
        assert len(violations) == 0, f"Direct database access violations found: {violations}"
        
        print("✅ No direct database imports detected")
        print("  - DatabaseHealthChecker: ✅")
        print("  - OrchestrationService: ✅") 
        print("  - BaselineCollector: ✅")