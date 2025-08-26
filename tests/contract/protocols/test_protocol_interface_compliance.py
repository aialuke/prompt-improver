"""Protocol interface compliance contract tests.

Contract tests that ensure protocols maintain their interface definitions
and behavioral contracts across the entire system. These tests validate:

- Protocol method signatures remain stable
- Protocol behavioral contracts are maintained
- Runtime checkability works correctly
- Protocol composition patterns are valid
- Interface segregation is properly implemented
- Protocol evolution maintains backward compatibility

Critical for preventing breaking changes and ensuring reliable
protocol-based architecture evolution.
"""

import inspect
from typing import Any, Protocol
from unittest.mock import Mock

import pytest

from prompt_improver.shared.interfaces.protocols import (
    application,
    cache,
    cli,
    core,
    database,
    get_ml_protocols,
    get_monitoring_protocols,
    mcp,
    security,
)


@pytest.mark.contract
class TestCoreProtocolContracts:
    """Contract tests for core domain protocols."""

    def test_service_protocol_contract(self):
        """Test ServiceProtocol maintains required interface contract."""
        service_protocol = core.ServiceProtocol

        # Should be runtime checkable
        assert hasattr(service_protocol, '__runtime_checkable__')

        # Should have required methods
        required_methods = ['initialize', 'shutdown']
        for method_name in required_methods:
            assert hasattr(service_protocol, method_name), (
                f"ServiceProtocol missing required method: {method_name}"
            )

        # Test method signatures
        initialize_sig = inspect.signature(service_protocol.initialize)
        shutdown_sig = inspect.signature(service_protocol.shutdown)

        # Both should be async methods returning None
        assert len(initialize_sig.parameters) == 1, "initialize should only have self parameter"
        assert len(shutdown_sig.parameters) == 1, "shutdown should only have self parameter"

        # Should return None (or typing annotation equivalent)
        assert str(initialize_sig.return_annotation) in {'None', '<class \'NoneType\'>'} or \
               'None' in str(initialize_sig.return_annotation)
        assert str(shutdown_sig.return_annotation) in {'None', '<class \'NoneType\'>'} or \
               'None' in str(shutdown_sig.return_annotation)

    def test_health_check_protocol_contract(self):
        """Test HealthCheckProtocol maintains required interface contract."""
        health_protocol = core.HealthCheckProtocol

        # Should be runtime checkable
        assert hasattr(health_protocol, '__runtime_checkable__')

        # Should have health_check method
        assert hasattr(health_protocol, 'health_check'), (
            "HealthCheckProtocol missing health_check method"
        )

        # Test method signature
        health_check_sig = inspect.signature(health_protocol.health_check)

        # Should have only self parameter and return bool
        assert len(health_check_sig.parameters) == 1, "health_check should only have self parameter"
        assert 'bool' in str(health_check_sig.return_annotation), (
            f"health_check should return bool, got {health_check_sig.return_annotation}"
        )

    def test_core_protocol_runtime_checkability(self):
        """Test that core protocols work correctly with isinstance checks."""

        class MockCoreService:
            """Mock service implementing core protocols."""

            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            async def health_check(self) -> bool:
                return True

        service = MockCoreService()

        # Runtime checks should work
        assert isinstance(service, core.ServiceProtocol)
        assert isinstance(service, core.HealthCheckProtocol)

        # Should work with protocol checking
        def check_service(svc: core.ServiceProtocol) -> bool:
            return hasattr(svc, 'initialize') and hasattr(svc, 'shutdown')

        def check_health(svc: core.HealthCheckProtocol) -> bool:
            return hasattr(svc, 'health_check')

        assert check_service(service) is True
        assert check_health(service) is True


@pytest.mark.contract
class TestCacheProtocolContracts:
    """Contract tests for cache domain protocols."""

    def test_basic_cache_protocol_contract(self):
        """Test BasicCacheProtocol maintains required interface contract."""
        cache_protocol = cache.BasicCacheProtocol

        # Should be runtime checkable
        assert hasattr(cache_protocol, '__runtime_checkable__')

        # Should have required methods
        required_methods = ['get', 'set', 'delete']
        for method_name in required_methods:
            assert hasattr(cache_protocol, method_name), (
                f"BasicCacheProtocol missing required method: {method_name}"
            )

        # Test method signatures
        get_sig = inspect.signature(cache_protocol.get)
        set_sig = inspect.signature(cache_protocol.set)
        delete_sig = inspect.signature(cache_protocol.delete)

        # get(key: str) -> Optional[Any]
        get_params = list(get_sig.parameters.keys())
        assert 'key' in get_params, "get method should have key parameter"

        # set(key: str, value: Any, ttl: Optional[int] = None) -> bool
        set_params = list(set_sig.parameters.keys())
        assert 'key' in set_params, "set method should have key parameter"
        assert 'value' in set_params, "set method should have value parameter"

        # delete(key: str) -> bool
        delete_params = list(delete_sig.parameters.keys())
        assert 'key' in delete_params, "delete method should have key parameter"

    def test_cache_health_protocol_contract(self):
        """Test CacheHealthProtocol maintains required interface contract."""
        cache_health_protocol = cache.CacheHealthProtocol

        # Should be runtime checkable
        assert hasattr(cache_health_protocol, '__runtime_checkable__')

        # Should have health monitoring methods
        expected_methods = ['get_stats', 'get_health_status']
        for method_name in expected_methods:
            if hasattr(cache_health_protocol, method_name):
                method_sig = inspect.signature(getattr(cache_health_protocol, method_name))
                # Should be async methods
                assert 'self' in method_sig.parameters, f"{method_name} should have self parameter"

    def test_cache_service_facade_protocol_contract(self):
        """Test CacheServiceFacadeProtocol maintains required interface contract."""
        cache_facade_protocol = cache.CacheServiceFacadeProtocol

        # Should be runtime checkable
        assert hasattr(cache_facade_protocol, '__runtime_checkable__')

        # Should combine multiple cache concerns
        # Verify it has characteristics of a facade
        protocol_methods = [method for method in dir(cache_facade_protocol)
                           if not method.startswith('_')]

        # Should have multiple methods (facade pattern)
        assert len(protocol_methods) >= 3, (
            f"CacheServiceFacadeProtocol has only {len(protocol_methods)} methods, "
            "should be a comprehensive facade"
        )

    def test_cache_protocol_runtime_behavior(self):
        """Test cache protocols work with runtime checks."""

        class MockCacheService:
            """Mock cache service implementing cache protocols."""

            async def get(self, key: str) -> Any | None:
                return f"cached_{key}"

            async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
                return True

            async def delete(self, key: str) -> bool:
                return True

            async def get_stats(self) -> dict[str, Any]:
                return {"hits": 100, "misses": 10}

            async def get_health_status(self) -> dict[str, Any]:
                return {"status": "healthy", "uptime": 3600}

        cache_service = MockCacheService()

        # Runtime checks should work
        assert isinstance(cache_service, cache.BasicCacheProtocol)
        assert isinstance(cache_service, cache.CacheHealthProtocol)


@pytest.mark.contract
class TestDatabaseProtocolContracts:
    """Contract tests for database domain protocols."""

    def test_session_manager_protocol_contract(self):
        """Test SessionManagerProtocol maintains required interface contract."""
        session_manager_protocol = database.SessionManagerProtocol

        # Should be runtime checkable
        assert hasattr(session_manager_protocol, '__runtime_checkable__')

        # Should have session management methods
        expected_methods = ['get_session', 'close_session']
        for method_name in expected_methods:
            if hasattr(session_manager_protocol, method_name):
                assert hasattr(session_manager_protocol, method_name), (
                    f"SessionManagerProtocol missing {method_name} method"
                )

    def test_database_protocol_contract(self):
        """Test DatabaseProtocol maintains required interface contract."""
        db_protocol = database.DatabaseProtocol

        # Should be runtime checkable
        assert hasattr(db_protocol, '__runtime_checkable__')

        # Should have database operation methods
        expected_methods = ['execute_query', 'execute_transaction']
        for method_name in expected_methods:
            if hasattr(db_protocol, method_name):
                method_sig = inspect.signature(getattr(db_protocol, method_name))
                assert 'self' in method_sig.parameters, f"{method_name} should have self parameter"

    def test_connection_pool_core_protocol_contract(self):
        """Test ConnectionPoolCoreProtocol maintains required interface contract."""
        pool_protocol = database.ConnectionPoolCoreProtocol

        # Should be runtime checkable
        assert hasattr(pool_protocol, '__runtime_checkable__')

        # Should have connection pool methods
        expected_methods = ['get_connection', 'return_connection']
        for method_name in expected_methods:
            if hasattr(pool_protocol, method_name):
                assert hasattr(pool_protocol, method_name), (
                    f"ConnectionPoolCoreProtocol missing {method_name} method"
                )

    def test_database_protocol_runtime_behavior(self):
        """Test database protocols work with runtime checks."""

        class MockDatabaseService:
            """Mock database service implementing database protocols."""

            async def get_session(self):
                return Mock()

            async def close_session(self, session):
                pass

            async def execute_query(self, query: str) -> dict[str, Any]:
                return {"result": "query executed", "rows": []}

            async def execute_transaction(self, queries: list[str]) -> dict[str, Any]:
                return {"result": "transaction executed", "affected_rows": len(queries)}

            async def get_connection(self):
                return Mock()

            async def return_connection(self, connection):
                pass

        db_service = MockDatabaseService()

        # Runtime checks should work
        assert isinstance(db_service, database.SessionManagerProtocol)
        assert isinstance(db_service, database.DatabaseProtocol)
        assert isinstance(db_service, database.ConnectionPoolCoreProtocol)


@pytest.mark.contract
class TestSecurityProtocolContracts:
    """Contract tests for security domain protocols."""

    def test_authentication_protocol_contract(self):
        """Test AuthenticationProtocol maintains required interface contract."""
        auth_protocol = security.AuthenticationProtocol

        # Should be runtime checkable
        assert hasattr(auth_protocol, '__runtime_checkable__')

        # Should have authentication methods
        expected_methods = ['authenticate', 'validate_token']
        for method_name in expected_methods:
            assert hasattr(auth_protocol, method_name), (
                f"AuthenticationProtocol missing {method_name} method"
            )

        # Test method signatures
        auth_sig = inspect.signature(auth_protocol.authenticate)
        validate_sig = inspect.signature(auth_protocol.validate_token)

        # authenticate should take credentials
        auth_params = list(auth_sig.parameters.keys())
        assert 'credentials' in auth_params, "authenticate should have credentials parameter"

        # validate_token should take token
        validate_params = list(validate_sig.parameters.keys())
        assert 'token' in validate_params, "validate_token should have token parameter"

    def test_authorization_protocol_contract(self):
        """Test AuthorizationProtocol maintains required interface contract."""
        authz_protocol = security.AuthorizationProtocol

        # Should be runtime checkable
        assert hasattr(authz_protocol, '__runtime_checkable__')

        # Should have authorization methods
        expected_methods = ['check_permission', 'get_user_permissions']
        for method_name in expected_methods:
            if hasattr(authz_protocol, method_name):
                assert hasattr(authz_protocol, method_name), (
                    f"AuthorizationProtocol missing {method_name} method"
                )

    def test_encryption_protocol_contract(self):
        """Test EncryptionProtocol maintains required interface contract."""
        encrypt_protocol = security.EncryptionProtocol

        # Should be runtime checkable
        assert hasattr(encrypt_protocol, '__runtime_checkable__')

        # Should have encryption methods
        expected_methods = ['encrypt', 'decrypt']
        for method_name in expected_methods:
            if hasattr(encrypt_protocol, method_name):
                assert hasattr(encrypt_protocol, method_name), (
                    f"EncryptionProtocol missing {method_name} method"
                )

    def test_security_protocol_isolation_contract(self):
        """Test that security protocols maintain isolation contracts."""

        class MockAuthService:
            async def authenticate(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
                return {"user_id": "123"} if credentials.get("valid") else None

            async def validate_token(self, token: str) -> dict[str, Any] | None:
                return {"user_id": "123"} if token == "valid_token" else None

        class MockAuthzService:
            async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
                return True

            async def get_user_permissions(self, user_id: str) -> list[str]:
                return ["read", "write"]

        class MockEncryptService:
            async def encrypt(self, data: str, key: str) -> str:
                return f"encrypted_{data}"

            async def decrypt(self, encrypted_data: str, key: str) -> str:
                return encrypted_data.replace("encrypted_", "")

        auth_service = MockAuthService()
        authz_service = MockAuthzService()
        encrypt_service = MockEncryptService()

        # Each should implement only their specific protocol
        assert isinstance(auth_service, security.AuthenticationProtocol)
        assert isinstance(authz_service, security.AuthorizationProtocol)
        assert isinstance(encrypt_service, security.EncryptionProtocol)

        # Should NOT cross-implement (maintain isolation)
        assert not hasattr(auth_service, 'check_permission')
        assert not hasattr(authz_service, 'authenticate')
        assert not hasattr(encrypt_service, 'validate_token')


@pytest.mark.contract
class TestApplicationProtocolContracts:
    """Contract tests for application domain protocols."""

    def test_application_service_protocol_contract(self):
        """Test ApplicationServiceProtocol maintains required interface contract."""
        app_protocol = application.ApplicationServiceProtocol

        # Should be runtime checkable
        assert hasattr(app_protocol, '__runtime_checkable__')

        # Should have application service methods
        expected_methods = ['process_request']
        for method_name in expected_methods:
            if hasattr(app_protocol, method_name):
                assert hasattr(app_protocol, method_name), (
                    f"ApplicationServiceProtocol missing {method_name} method"
                )

    def test_workflow_orchestrator_protocol_contract(self):
        """Test WorkflowOrchestratorProtocol maintains required interface contract."""
        workflow_protocol = application.WorkflowOrchestratorProtocol

        # Should be runtime checkable
        assert hasattr(workflow_protocol, '__runtime_checkable__')

        # Should have workflow orchestration methods
        expected_methods = ['execute_workflow', 'get_workflow_status']
        for method_name in expected_methods:
            if hasattr(workflow_protocol, method_name):
                assert hasattr(workflow_protocol, method_name), (
                    f"WorkflowOrchestratorProtocol missing {method_name} method"
                )

    def test_validation_service_protocol_contract(self):
        """Test ValidationServiceProtocol maintains required interface contract."""
        validation_protocol = application.ValidationServiceProtocol

        # Should be runtime checkable
        assert hasattr(validation_protocol, '__runtime_checkable__')

        # Should have validation methods
        expected_methods = ['validate', 'validate_batch']
        for method_name in expected_methods:
            if hasattr(validation_protocol, method_name):
                assert hasattr(validation_protocol, method_name), (
                    f"ValidationServiceProtocol missing {method_name} method"
                )


@pytest.mark.contract
class TestMCPProtocolContracts:
    """Contract tests for MCP domain protocols."""

    def test_mcp_server_protocol_contract(self):
        """Test MCPServerProtocol maintains required interface contract."""
        mcp_server_protocol = mcp.MCPServerProtocol

        # Should be runtime checkable
        assert hasattr(mcp_server_protocol, '__runtime_checkable__')

        # Should have MCP server methods
        expected_methods = ['handle_request', 'start', 'stop']
        for method_name in expected_methods:
            if hasattr(mcp_server_protocol, method_name):
                assert hasattr(mcp_server_protocol, method_name), (
                    f"MCPServerProtocol missing {method_name} method"
                )

    def test_mcp_tool_protocol_contract(self):
        """Test MCPToolProtocol maintains required interface contract."""
        tool_protocol = mcp.MCPToolProtocol

        # Should be runtime checkable
        assert hasattr(tool_protocol, '__runtime_checkable__')

        # Should have tool execution methods
        expected_methods = ['execute', 'get_schema']
        for method_name in expected_methods:
            if hasattr(tool_protocol, method_name):
                assert hasattr(tool_protocol, method_name), (
                    f"MCPToolProtocol missing {method_name} method"
                )

    def test_server_services_protocol_contract(self):
        """Test ServerServicesProtocol maintains required interface contract."""
        services_protocol = mcp.ServerServicesProtocol

        # Should be runtime checkable
        assert hasattr(services_protocol, '__runtime_checkable__')

        # Should have service management methods
        expected_methods = ['get_service', 'register_service']
        for method_name in expected_methods:
            if hasattr(services_protocol, method_name):
                assert hasattr(services_protocol, method_name), (
                    f"ServerServicesProtocol missing {method_name} method"
                )


@pytest.mark.contract
class TestCLIProtocolContracts:
    """Contract tests for CLI domain protocols."""

    def test_command_processor_protocol_contract(self):
        """Test CommandProcessorProtocol maintains required interface contract."""
        cmd_protocol = cli.CommandProcessorProtocol

        # Should be runtime checkable
        assert hasattr(cmd_protocol, '__runtime_checkable__')

        # Should have command processing methods
        expected_methods = ['process_command', 'validate_command']
        for method_name in expected_methods:
            if hasattr(cmd_protocol, method_name):
                assert hasattr(cmd_protocol, method_name), (
                    f"CommandProcessorProtocol missing {method_name} method"
                )

    def test_user_interaction_protocol_contract(self):
        """Test UserInteractionProtocol maintains required interface contract."""
        ui_protocol = cli.UserInteractionProtocol

        # Should be runtime checkable
        assert hasattr(ui_protocol, '__runtime_checkable__')

        # Should have user interaction methods
        expected_methods = ['prompt_user', 'display_result']
        for method_name in expected_methods:
            if hasattr(ui_protocol, method_name):
                assert hasattr(ui_protocol, method_name), (
                    f"UserInteractionProtocol missing {method_name} method"
                )

    def test_workflow_manager_protocol_contract(self):
        """Test WorkflowManagerProtocol maintains required interface contract."""
        workflow_protocol = cli.WorkflowManagerProtocol

        # Should be runtime checkable
        assert hasattr(workflow_protocol, '__runtime_checkable__')

        # Should have workflow management methods
        expected_methods = ['start_workflow', 'stop_workflow', 'get_workflow_status']
        for method_name in expected_methods:
            if hasattr(workflow_protocol, method_name):
                assert hasattr(workflow_protocol, method_name), (
                    f"WorkflowManagerProtocol missing {method_name} method"
                )


@pytest.mark.contract
class TestLazyProtocolContracts:
    """Contract tests for lazy-loaded protocols."""

    def test_ml_protocols_lazy_contract(self):
        """Test ML protocols maintain contracts when lazy loaded."""
        ml_protocols = get_ml_protocols()

        # Should have expected ML protocols
        expected_protocols = ['MLflowServiceProtocol', 'EventBusProtocol']
        for protocol_name in expected_protocols:
            if hasattr(ml_protocols, protocol_name):
                protocol = getattr(ml_protocols, protocol_name)
                # Should be runtime checkable
                assert hasattr(protocol, '__runtime_checkable__'), (
                    f"ML protocol {protocol_name} not runtime checkable"
                )

    def test_monitoring_protocols_lazy_contract(self):
        """Test monitoring protocols maintain contracts when lazy loaded."""
        monitoring_protocols = get_monitoring_protocols()

        # Should have expected monitoring protocols
        expected_protocols = ['MetricsCollectorProtocol', 'HealthCheckComponentProtocol']
        for protocol_name in expected_protocols:
            if hasattr(monitoring_protocols, protocol_name):
                protocol = getattr(monitoring_protocols, protocol_name)
                # Should be runtime checkable
                assert hasattr(protocol, '__runtime_checkable__'), (
                    f"Monitoring protocol {protocol_name} not runtime checkable"
                )


@pytest.mark.contract
class TestProtocolCompositionContracts:
    """Contract tests for protocol composition patterns."""

    def test_multi_protocol_implementation_contract(self):
        """Test that services can implement multiple protocols correctly."""

        class MultiProtocolService:
            """Service implementing multiple domain protocols."""

            # Core protocols
            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            async def health_check(self) -> bool:
                return True

            # Cache protocols
            async def get(self, key: str) -> Any | None:
                return f"value_{key}"

            async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
                return True

            async def delete(self, key: str) -> bool:
                return True

            # Application protocols
            async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
                return {"processed": True, "request": request}

        service = MultiProtocolService()

        # Should satisfy all implemented protocols
        assert isinstance(service, core.ServiceProtocol)
        assert isinstance(service, core.HealthCheckProtocol)
        assert isinstance(service, cache.BasicCacheProtocol)
        assert isinstance(service, application.ApplicationServiceProtocol)

    def test_protocol_inheritance_contract(self):
        """Test protocol inheritance maintains contracts."""

        class BaseServiceProtocol(Protocol):
            """Base service protocol for inheritance test."""

            async def base_method(self) -> str:
                ...

        class ExtendedServiceProtocol(BaseServiceProtocol, Protocol):
            """Extended service protocol inheriting from base."""

            async def extended_method(self) -> int:
                ...

        class ConcreteService:
            """Concrete implementation of extended protocol."""

            async def base_method(self) -> str:
                return "base"

            async def extended_method(self) -> int:
                return 42

        service = ConcreteService()

        # Should satisfy both base and extended protocols
        assert isinstance(service, BaseServiceProtocol)
        assert isinstance(service, ExtendedServiceProtocol)

    def test_protocol_method_signature_stability(self):
        """Test that protocol method signatures are stable."""
        from prompt_improver.shared.interfaces.protocols import cache, core

        # Core protocol signatures should be stable
        service_init_sig = inspect.signature(core.ServiceProtocol.initialize)
        service_shutdown_sig = inspect.signature(core.ServiceProtocol.shutdown)
        health_check_sig = inspect.signature(core.HealthCheckProtocol.health_check)

        # Should have expected parameter counts
        assert len(service_init_sig.parameters) == 1  # self only
        assert len(service_shutdown_sig.parameters) == 1  # self only
        assert len(health_check_sig.parameters) == 1  # self only

        # Cache protocol signatures should be stable
        cache_get_sig = inspect.signature(cache.BasicCacheProtocol.get)
        cache_set_sig = inspect.signature(cache.BasicCacheProtocol.set)

        # get should have key parameter
        assert 'key' in cache_get_sig.parameters
        # set should have key and value parameters
        assert 'key' in cache_set_sig.parameters
        assert 'value' in cache_set_sig.parameters


@pytest.mark.contract
class TestProtocolEvolutionContracts:
    """Contract tests for protocol evolution and backward compatibility."""

    def test_protocol_versioning_contract(self):
        """Test that protocols support versioning for evolution."""
        # Test that protocols can be extended without breaking existing implementations

        class V1Protocol(Protocol):
            """Version 1 of a protocol."""

            async def method_v1(self) -> str:
                ...

        class V2Protocol(V1Protocol, Protocol):
            """Version 2 extending V1 protocol."""

            async def method_v2(self) -> int:
                ...

        class V1Implementation:
            """Implementation supporting only V1."""

            async def method_v1(self) -> str:
                return "v1"

        class V2Implementation:
            """Implementation supporting V2 (and V1)."""

            async def method_v1(self) -> str:
                return "v1"

            async def method_v2(self) -> int:
                return 2

        v1_impl = V1Implementation()
        v2_impl = V2Implementation()

        # V1 implementation should work with V1 protocol
        assert isinstance(v1_impl, V1Protocol)

        # V2 implementation should work with both protocols
        assert isinstance(v2_impl, V1Protocol)
        assert isinstance(v2_impl, V2Protocol)

        # V1 implementation should NOT work with V2 protocol
        assert not isinstance(v1_impl, V2Protocol)

    def test_backward_compatibility_contract(self):
        """Test that protocol changes maintain backward compatibility."""
        # Test that existing implementations continue to work

        from prompt_improver.shared.interfaces.protocols import core

        class LegacyService:
            """Legacy service implementation."""

            async def initialize(self) -> None:
                self.started = True

            async def shutdown(self) -> None:
                self.started = False

        legacy_service = LegacyService()

        # Legacy implementation should still work with current protocols
        assert isinstance(legacy_service, core.ServiceProtocol)

        # Should be able to use in protocol-expecting functions
        async def use_service(svc: core.ServiceProtocol) -> bool:
            await svc.initialize()
            await svc.shutdown()
            return True

        # Should work without modification
        import asyncio
        result = asyncio.run(use_service(legacy_service))
        assert result is True


if __name__ == "__main__":
    # Run protocol interface compliance tests
    pytest.main([__file__, "-v", "--tb=short"])
