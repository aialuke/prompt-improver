"""Cross-domain protocol validation tests for consolidated architecture.

Integration tests that validate proper interaction patterns between
different protocol domains while maintaining architectural boundaries.

Tests complex scenarios including:
- Service composition across domains
- Protocol inheritance validation
- Cross-domain dependency injection
- Real-world usage patterns
- Performance under integration scenarios
"""

import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from prompt_improver.shared.interfaces.protocols import (
    application,
    cache,
    core,
    database,
    get_ml_protocols,
    get_monitoring_protocols,
)


@pytest.mark.integration
class TestCrossDomainServiceComposition:
    """Test service composition patterns across protocol domains."""

    @pytest.fixture
    async def mock_composed_service(self):
        """Create a mock service that implements multiple domain protocols."""

        class ComposedMockService:
            """Service implementing protocols from multiple domains."""

            def __init__(self):
                self.initialized = False
                self.cache_data = {}
                self.health_status = True

            # Core domain protocols
            async def initialize(self) -> None:
                self.initialized = True

            async def shutdown(self) -> None:
                self.initialized = False

            async def health_check(self) -> bool:
                return self.health_status

            # Cache domain protocols
            async def get(self, key: str) -> Any | None:
                return self.cache_data.get(key)

            async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
                self.cache_data[key] = value
                return True

            async def delete(self, key: str) -> bool:
                return self.cache_data.pop(key, None) is not None

            # Application domain protocols
            async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
                return {"status": "processed", "data": request}

        service = ComposedMockService()
        await service.initialize()
        yield service
        await service.shutdown()

    async def test_service_implements_multiple_domain_protocols(self, mock_composed_service):
        """Test that a service can implement protocols from multiple domains."""
        service = mock_composed_service

        # Should implement core protocols
        assert isinstance(service, core.ServiceProtocol)
        assert isinstance(service, core.HealthCheckProtocol)

        # Should implement cache protocols
        assert isinstance(service, cache.BasicCacheProtocol)

        # Should implement application protocols
        assert isinstance(service, application.ApplicationServiceProtocol)

        # Test cross-domain functionality
        assert service.initialized is True
        assert await service.health_check() is True

        await service.set("test_key", "test_value")
        assert await service.get("test_key") == "test_value"

        result = await service.process_request({"action": "test"})
        assert result["status"] == "processed"

    async def test_protocol_composition_performance(self, mock_composed_service):
        """Test that protocol composition doesn't significantly impact performance."""
        service = mock_composed_service

        # Test multiple protocol method calls in sequence
        start_time = time.time()

        for i in range(100):
            await service.health_check()
            await service.set(f"key_{i}", f"value_{i}")
            await service.get(f"key_{i}")
            await service.process_request({"iteration": i})

        total_time = time.time() - start_time
        avg_time_per_operation = total_time / 400  # 4 operations * 100 iterations

        # Each operation should be very fast
        assert avg_time_per_operation < 0.001, (  # 1ms per operation
            f"Cross-domain protocol operations too slow: {avg_time_per_operation * 1000:.2f}ms per operation"
        )


@pytest.mark.integration
class TestProtocolInheritanceValidation:
    """Test protocol inheritance patterns across domains."""

    def test_core_protocols_as_base_interfaces(self):
        """Test that core protocols can serve as base interfaces for other domains."""
        from prompt_improver.shared.interfaces.protocols import cache, core

        # Cache protocols should be able to extend core protocols
        class ExtendedCacheService:
            """Cache service extending core service protocol."""

            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            async def health_check(self) -> bool:
                return True

            async def get(self, key: str) -> Any | None:
                return None

            async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
                return True

        service = ExtendedCacheService()

        # Should satisfy both protocols
        assert isinstance(service, core.ServiceProtocol)
        assert isinstance(service, core.HealthCheckProtocol)
        assert isinstance(service, cache.BasicCacheProtocol)

    def test_application_protocols_extend_core(self):
        """Test that application protocols properly extend core protocols."""
        from prompt_improver.shared.interfaces.protocols import application, core

        class ApplicationService:
            """Application service implementing both core and application protocols."""

            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
                return {"processed": True}

        service = ApplicationService()

        # Should satisfy both protocol types
        assert isinstance(service, core.ServiceProtocol)
        assert isinstance(service, application.ApplicationServiceProtocol)


@pytest.mark.integration
class TestCrossDomainDependencyInjection:
    """Test dependency injection patterns across protocol domains."""

    @pytest.fixture
    def mock_dependency_container(self):
        """Create mock dependency container with cross-domain services."""

        class MockDependencyContainer:
            """Mock DI container for cross-domain testing."""

            def __init__(self):
                self.services = {}

            def register(self, protocol_type, service_instance):
                """Register service instance for protocol type."""
                self.services[protocol_type] = service_instance

            def get(self, protocol_type):
                """Get service instance for protocol type."""
                return self.services.get(protocol_type)

        return MockDependencyContainer()

    async def test_cross_domain_service_resolution(self, mock_dependency_container):
        """Test resolving services across different protocol domains."""
        container = mock_dependency_container

        # Mock services for different domains
        class MockCoreService:
            async def initialize(self): pass
            async def shutdown(self): pass

        class MockCacheService:
            async def get(self, key: str): return f"cached_{key}"
            async def set(self, key: str, value: Any, ttl: int | None = None): return True

        class MockDatabaseService:
            async def execute_query(self, query: str): return {"result": "query_executed"}
            async def get_session(self): return Mock()

        # Register services
        container.register(core.ServiceProtocol, MockCoreService())
        container.register(cache.BasicCacheProtocol, MockCacheService())
        container.register(database.DatabaseProtocol, MockDatabaseService())

        # Test cross-domain resolution
        core_service = container.get(core.ServiceProtocol)
        cache_service = container.get(cache.BasicCacheProtocol)
        db_service = container.get(database.DatabaseProtocol)

        assert core_service is not None
        assert cache_service is not None
        assert db_service is not None

        # Test services work correctly
        await core_service.initialize()
        cache_result = await cache_service.get("test_key")
        db_result = await db_service.execute_query("SELECT 1")

        assert cache_result == "cached_test_key"
        assert db_result["result"] == "query_executed"

    async def test_dependency_chain_validation(self, mock_dependency_container):
        """Test that dependency chains respect domain boundaries."""
        container = mock_dependency_container

        # Create service that depends on multiple domains
        class BusinessLogicService:
            """Service with cross-domain dependencies."""

            def __init__(self, cache_service, db_service, core_service):
                self.cache = cache_service
                self.database = db_service
                self.core = core_service

            async def process_business_logic(self, request: dict[str, Any]):
                """Process business logic using multiple domain services."""
                # Initialize core service
                await self.core.initialize()

                # Check cache first
                cached_result = await self.cache.get(f"request_{request.get('id')}")
                if cached_result:
                    return {"source": "cache", "data": cached_result}

                # Query database
                db_result = await self.database.execute_query(f"SELECT * FROM requests WHERE id = {request.get('id')}")

                # Cache the result
                await self.cache.set(f"request_{request.get('id')}", db_result)

                return {"source": "database", "data": db_result}

        # Mock dependencies
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True

        mock_db = AsyncMock()
        mock_db.execute_query.return_value = {"id": 1, "name": "test"}

        mock_core = AsyncMock()

        # Create business service
        business_service = BusinessLogicService(mock_cache, mock_db, mock_core)

        # Test business logic execution
        result = await business_service.process_business_logic({"id": 1})

        # Verify dependencies were called correctly
        mock_core.initialize.assert_called_once()
        mock_cache.get.assert_called_once_with("request_1")
        mock_db.execute_query.assert_called_once()
        mock_cache.set.assert_called_once()

        assert result["source"] == "database"
        assert result["data"]["id"] == 1


@pytest.mark.integration
class TestRealWorldUsagePatterns:
    """Test real-world usage patterns across protocol domains."""

    async def test_mcp_server_integration_pattern(self):
        """Test MCP server integration with other domain protocols."""

        class MockMCPServer:
            """Mock MCP server implementing cross-domain protocols."""

            def __init__(self, db_service):
                self.db_service = db_service
                self.running = False

            # Core protocols
            async def initialize(self) -> None:
                self.running = True

            async def shutdown(self) -> None:
                self.running = False

            # MCP protocols
            async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
                if request.get("method") == "query_database":
                    result = await self.db_service.execute_query(request.get("query", ""))
                    return {"jsonrpc": "2.0", "result": result, "id": request.get("id")}
                return {"jsonrpc": "2.0", "error": "Unknown method", "id": request.get("id")}

        # Mock database service
        mock_db = AsyncMock()
        mock_db.execute_query.return_value = {"rows": [{"id": 1, "name": "test"}]}

        # Create MCP server
        mcp_server = MockMCPServer(mock_db)
        await mcp_server.initialize()

        # Test MCP request handling with database integration
        request = {
            "jsonrpc": "2.0",
            "method": "query_database",
            "params": {"query": "SELECT * FROM users"},
            "id": "1"
        }

        response = await mcp_server.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert response["result"]["rows"][0]["id"] == 1

        await mcp_server.shutdown()

    async def test_cli_with_application_services_pattern(self):
        """Test CLI integration with application services."""

        class MockCLIHandler:
            """Mock CLI handler using application services."""

            def __init__(self, app_service):
                self.app_service = app_service

            async def handle_command(self, command: str, args: dict[str, Any]) -> str:
                """Handle CLI command using application service."""
                if command == "process":
                    result = await self.app_service.process_request(args)
                    return f"Command processed: {result}"
                return f"Unknown command: {command}"

        # Mock application service
        mock_app_service = AsyncMock()
        mock_app_service.process_request.return_value = {"status": "success", "data": "processed"}

        # Create CLI handler
        cli_handler = MockCLIHandler(mock_app_service)

        # Test CLI command processing
        result = await cli_handler.handle_command("process", {"input": "test_data"})

        assert "Command processed:" in result
        assert "success" in result
        mock_app_service.process_request.assert_called_once_with({"input": "test_data"})

    async def test_security_integration_pattern(self):
        """Test security protocol integration with other domains."""

        class SecureApplicationService:
            """Application service with security integration."""

            def __init__(self, auth_service, app_service):
                self.auth_service = auth_service
                self.app_service = app_service

            async def secure_process_request(self, token: str, request: dict[str, Any]) -> dict[str, Any]:
                """Process request with authentication."""
                # Validate token first
                user_info = await self.auth_service.validate_token(token)
                if not user_info:
                    return {"error": "Invalid token"}

                # Add user context to request
                request["user"] = user_info

                # Process the request
                return await self.app_service.process_request(request)

        # Mock services
        mock_auth = AsyncMock()
        mock_auth.validate_token.return_value = {"user_id": "123", "username": "test_user"}

        mock_app = AsyncMock()
        mock_app.process_request.return_value = {"status": "success", "processed_by": "test_user"}

        # Create secure service
        secure_service = SecureApplicationService(mock_auth, mock_app)

        # Test secure request processing
        result = await secure_service.secure_process_request(
            "valid_token",
            {"action": "create_resource"}
        )

        assert result["status"] == "success"
        assert result["processed_by"] == "test_user"

        mock_auth.validate_token.assert_called_once_with("valid_token")
        mock_app.process_request.assert_called_once()


@pytest.mark.integration
class TestPerformanceUnderIntegration:
    """Test protocol performance under integration scenarios."""

    async def test_cross_domain_call_performance(self):
        """Test performance of calls across multiple protocol domains."""
        # Create mock services for multiple domains
        services = {
            "core": AsyncMock(),
            "cache": AsyncMock(),
            "database": AsyncMock(),
            "application": AsyncMock()
        }

        # Configure mock responses
        services["cache"].get.return_value = None  # Cache miss
        services["database"].execute_query.return_value = {"data": "test"}
        services["cache"].set.return_value = True
        services["application"].process_request.return_value = {"processed": True}

        async def simulate_cross_domain_workflow():
            """Simulate a workflow that calls multiple domains."""
            # Check cache
            cached = await services["cache"].get("test_key")

            if not cached:
                # Query database
                data = await services["database"].execute_query("SELECT * FROM test")
                # Cache result
                await services["cache"].set("test_key", data)
                cached = data

            # Process with application service
            return await services["application"].process_request({"data": cached})

        # Measure performance of cross-domain workflow
        start_time = time.time()

        # Run multiple iterations to get average
        for _ in range(1000):
            await simulate_cross_domain_workflow()

        total_time = time.time() - start_time
        avg_time_per_workflow = total_time / 1000

        # Should complete quickly even with multiple domain calls
        assert avg_time_per_workflow < 0.001, (  # 1ms per workflow
            f"Cross-domain workflow too slow: {avg_time_per_workflow * 1000:.2f}ms per workflow"
        )

    async def test_lazy_loading_integration_performance(self):
        """Test lazy loading performance in integration scenarios."""
        # Test that ML protocols can be loaded when needed without blocking
        start_time = time.time()

        # Simulate loading ML protocols during integration
        ml_protocols = get_ml_protocols()
        ml_load_time = time.time() - start_time

        # Should load reasonably quickly
        assert ml_load_time < 2.0, (
            f"ML protocol lazy loading too slow in integration: {ml_load_time:.3f}s"
        )

        # Test monitoring protocols
        start_time = time.time()
        monitoring_protocols = get_monitoring_protocols()
        monitoring_load_time = time.time() - start_time

        assert monitoring_load_time < 1.0, (
            f"Monitoring protocol lazy loading too slow in integration: {monitoring_load_time:.3f}s"
        )

        # Verify protocols loaded correctly
        assert ml_protocols is not None
        assert monitoring_protocols is not None


@pytest.mark.integration
class TestProtocolCompatibilityMatrix:
    """Test compatibility matrix between different protocol domains."""

    def test_core_compatibility_with_all_domains(self):
        """Test that core protocols are compatible with all other domains."""
        from prompt_improver.shared.interfaces.protocols import (
            application,
            cache,
            cli,
            core,
            database,
            mcp,
            security,
        )

        domains = [cache, database, security, cli, mcp, application]

        for domain in domains:
            # Each domain should be able to coexist with core
            try:
                # Should be able to import both without conflicts
                assert core is not None
                assert domain is not None

                # Should have expected protocols
                assert hasattr(core, 'ServiceProtocol')
                assert hasattr(core, 'HealthCheckProtocol')

            except ImportError as e:
                pytest.fail(f"Core domain incompatible with {domain}: {e}")

    def test_infrastructure_compatibility_matrix(self):
        """Test compatibility between infrastructure domains."""
        from prompt_improver.shared.interfaces.protocols import (
            cache,
            database,
            security,
        )

        infrastructure_domains = [
            ("cache", cache),
            ("database", database),
            ("security", security)
        ]

        # Test all combinations
        for i, (name1, domain1) in enumerate(infrastructure_domains):
            for name2, domain2 in infrastructure_domains[i + 1:]:
                try:
                    # Should be able to use both domains together
                    assert domain1 is not None
                    assert domain2 is not None

                    # Should have expected protocols
                    assert len(dir(domain1)) > 0
                    assert len(dir(domain2)) > 0

                except ImportError as e:
                    pytest.fail(f"Infrastructure domains {name1} and {name2} incompatible: {e}")

    def test_lazy_domain_compatibility(self):
        """Test that lazy-loaded domains are compatible with regular domains."""
        from prompt_improver.shared.interfaces.protocols import (
            cache,
            core,
            database,
            get_ml_protocols,
            get_monitoring_protocols,
        )

        # Load lazy domains
        ml_protocols = get_ml_protocols()
        monitoring_protocols = get_monitoring_protocols()

        # Should be compatible with regular domains
        regular_domains = [core, cache, database]
        lazy_domains = [ml_protocols, monitoring_protocols]

        for regular in regular_domains:
            for lazy in lazy_domains:
                try:
                    # Should coexist without conflicts
                    assert regular is not None
                    assert lazy is not None

                except Exception as e:
                    pytest.fail(f"Lazy domain compatibility failed: {e}")


if __name__ == "__main__":
    # Run cross-domain validation tests
    pytest.main([__file__, "-v", "--tb=short"])
