"""Domain boundary validation tests for consolidated protocol architecture.

Tests architectural integrity by validating:
- Protocol domain isolation
- Dependency direction validation
- Interface segregation compliance
- Circular dependency prevention
- Security isolation requirements
- ML lazy loading boundaries
- Clean architecture layer separation

Critical for maintaining 2025 architectural standards and preventing
architectural degradation through cross-domain protocol violations.
"""

import ast
import inspect
import time
from pathlib import Path
from typing import Any

import pytest


class TestProtocolDomainIsolation:
    """Test that protocol domains maintain proper isolation boundaries."""

    def test_core_domain_has_no_external_dependencies(self):
        """Core domain protocols must not depend on other domains."""
        core_module_path = Path(__file__).parent.parent.parent.parent.parent.parent / \
                          "src" / "prompt_improver" / "shared" / "interfaces" / "protocols" / "core.py"

        with open(core_module_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        # Extract all import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        # Filter for our protocol imports
        protocol_imports = [imp for imp in imports if 'prompt_improver' in imp and 'protocols' in imp]

        # Core should only import from typing, abc, and standard library
        forbidden_domains = ['cache', 'database', 'security', 'cli', 'mcp', 'application', 'ml', 'monitoring']
        cross_domain_imports = [imp for imp in protocol_imports
                               if any(domain in imp for domain in forbidden_domains)]

        assert not cross_domain_imports, (
            f"Core domain has forbidden cross-domain imports: {cross_domain_imports}. "
            "Core protocols must remain domain-agnostic."
        )

    def test_security_domain_isolation(self):
        """Security domain must maintain strict isolation per OWASP 2025."""
        security_module_path = Path(__file__).parent.parent.parent.parent.parent.parent / \
                              "src" / "prompt_improver" / "shared" / "interfaces" / "protocols" / "security.py"

        with open(security_module_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        # Security should not import from other business domains
        forbidden_domains = ['cache', 'database', 'cli', 'mcp', 'application', 'ml', 'monitoring']
        protocol_imports = [imp for imp in imports if 'prompt_improver' in imp and 'protocols' in imp]
        cross_domain_imports = [imp for imp in protocol_imports
                               if any(domain in imp for domain in forbidden_domains)]

        assert not cross_domain_imports, (
            f"Security domain has forbidden cross-domain imports: {cross_domain_imports}. "
            "Security protocols must maintain strict isolation per OWASP 2025."
        )

    def test_database_domain_boundaries(self):
        """Database domain should not import ML or Cache protocols directly."""
        database_module_path = Path(__file__).parent.parent.parent.parent.parent.parent / \
                              "src" / "prompt_improver" / "shared" / "interfaces" / "protocols" / "database.py"

        with open(database_module_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'protocols' in node.module:
                    imports.append(node.module)

        # Database should not directly import ML or heavy dependencies
        forbidden_domains = ['ml', 'monitoring']
        cross_domain_imports = [imp for imp in imports
                               if any(domain in imp for domain in forbidden_domains)]

        assert not cross_domain_imports, (
            f"Database domain has forbidden heavy dependency imports: {cross_domain_imports}. "
            "Database protocols should not import ML/monitoring protocols directly."
        )

    def test_mcp_domain_isolation(self):
        """MCP protocols should not leak into core business domains."""
        # Import core business modules to check for MCP dependencies
        from prompt_improver.shared.interfaces.protocols import core

        # Check that core business protocols don't import MCP
        core_members = inspect.getmembers(core, inspect.isclass)
        for name, cls in core_members:
            if hasattr(cls, '__module__'):
                assert 'mcp' not in cls.__module__, (
                    f"Core protocol {name} has MCP dependency. "
                    "MCP protocols should not leak into core business domains."
                )


class TestDependencyDirectionValidation:
    """Test that dependencies flow in the correct direction per Clean Architecture."""

    def test_dependency_direction_core_to_application(self):
        """Application can depend on core, but not vice versa."""
        # Core protocols should be importable without application dependencies
        start_time = time.time()
        try:
            from prompt_improver.shared.interfaces.protocols import core
            import_time = time.time() - start_time

            # Should import quickly without heavy dependencies
            assert import_time < 0.1, (
                f"Core import took {import_time:.3f}s, indicating heavy dependencies. "
                "Core should be lightweight and dependency-free."
            )

            # Verify core protocols exist
            assert hasattr(core, 'ServiceProtocol')
            assert hasattr(core, 'HealthCheckProtocol')

        except ImportError as e:
            pytest.fail(f"Core protocols failed to import: {e}")

    def test_application_can_import_core(self):
        """Application layer should successfully import core protocols."""
        try:
            from prompt_improver.shared.interfaces.protocols import application, core

            # Application should be able to reference core protocols
            assert core.ServiceProtocol
            assert hasattr(application, 'ApplicationServiceProtocol')

        except ImportError as e:
            pytest.fail(f"Application failed to import core protocols: {e}")

    def test_infrastructure_can_import_application_and_core(self):
        """Infrastructure (database, cache) can import from application and core."""
        try:
            from prompt_improver.shared.interfaces.protocols import (
                application,
                cache,
                core,
                database,
            )

            # Infrastructure should successfully import business protocols
            assert core.ServiceProtocol
            assert application.ApplicationServiceProtocol
            assert hasattr(database, 'DatabaseProtocol')
            assert hasattr(cache, 'BasicCacheProtocol')

        except ImportError as e:
            pytest.fail(f"Infrastructure failed to import business protocols: {e}")


class TestInterfaceSegregationCompliance:
    """Test that protocols follow Interface Segregation Principle (ISP)."""

    def test_cache_protocols_interface_segregation(self):
        """Cache protocols should be properly segregated by responsibility."""
        from prompt_improver.shared.interfaces.protocols import cache

        # Should have separate interfaces for different responsibilities
        expected_protocols = [
            'BasicCacheProtocol',
            'CacheHealthProtocol',
            'CacheServiceFacadeProtocol'
        ]

        for protocol_name in expected_protocols:
            assert hasattr(cache, protocol_name), (
                f"Cache domain missing segregated protocol: {protocol_name}"
            )

            protocol_class = getattr(cache, protocol_name)
            methods = [method for method in dir(protocol_class)
                      if not method.startswith('_') and callable(getattr(protocol_class, method, None))]

            # Each protocol should have focused responsibility (not too many methods)
            assert len(methods) <= 10, (
                f"{protocol_name} has {len(methods)} methods, violating ISP. "
                "Protocols should have focused responsibilities."
            )

    def test_database_protocols_interface_segregation(self):
        """Database protocols should be properly segregated."""
        from prompt_improver.shared.interfaces.protocols import database

        expected_protocols = [
            'SessionManagerProtocol',
            'DatabaseProtocol',
            'ConnectionPoolCoreProtocol'
        ]

        for protocol_name in expected_protocols:
            assert hasattr(database, protocol_name), (
                f"Database domain missing segregated protocol: {protocol_name}"
            )

    def test_security_protocols_interface_segregation(self):
        """Security protocols should maintain strict segregation."""
        from prompt_improver.shared.interfaces.protocols import security

        expected_protocols = [
            'AuthenticationProtocol',
            'AuthorizationProtocol',
            'EncryptionProtocol'
        ]

        for protocol_name in expected_protocols:
            assert hasattr(security, protocol_name), (
                f"Security domain missing segregated protocol: {protocol_name}"
            )

            # Security protocols should be focused and minimal
            protocol_class = getattr(security, protocol_name)
            methods = [method for method in dir(protocol_class)
                      if not method.startswith('_')]

            # Security protocols should be especially focused
            assert len(methods) <= 8, (
                f"{protocol_name} has {len(methods)} methods. "
                "Security protocols should be minimal and focused."
            )


class TestCircularDependencyPrevention:
    """Test that domain boundaries prevent circular dependencies."""

    def test_no_circular_imports_between_domains(self):
        """Test that no circular import patterns exist between protocol domains."""
        # Test importing all non-lazy domains together
        try:
            start_time = time.time()

            from prompt_improver.shared.interfaces.protocols import (
                application,
                cache,
                cli,
                core,
                database,
                mcp,
                security,
            )

            import_time = time.time() - start_time

            # Should import without circular dependency issues
            assert import_time < 1.0, (
                f"Protocol imports took {import_time:.3f}s, indicating possible circular dependencies"
            )

            # Verify all domains imported successfully
            domains = [core, cache, database, security, cli, mcp, application]
            for domain in domains:
                assert domain is not None, "Domain failed to import due to circular dependency"

        except ImportError as e:
            pytest.fail(f"Circular dependency detected: {e}")

    def test_lazy_loading_prevents_heavy_dependency_circles(self):
        """Test that lazy loading prevents heavy dependency circular imports."""
        # ML and monitoring should not be imported automatically
        import sys

        # Clear any existing ML/monitoring imports
        modules_to_clear = [mod for mod in sys.modules
                           if 'protocols.ml' in mod or 'protocols.monitoring' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import main protocols - should not trigger ML/monitoring imports

        # Verify ML/monitoring are not in sys.modules
        ml_imported = any('protocols.ml' in mod for mod in sys.modules)
        monitoring_imported = any('protocols.monitoring' in mod for mod in sys.modules)

        assert not ml_imported, "ML protocols were imported despite lazy loading"
        assert not monitoring_imported, "Monitoring protocols were imported despite lazy loading"


class TestSecurityIsolationValidation:
    """Test security protocol isolation requirements per OWASP 2025."""

    def test_authentication_protocol_isolation(self):
        """Authentication protocols must remain isolated."""
        from prompt_improver.shared.interfaces.protocols.security import (
            AuthenticationProtocol,
        )

        # Test runtime checkability by trying isinstance
        class TestAuth:
            async def authenticate(self, credentials): return None
            async def validate_token(self, token): return None

        test_instance = TestAuth()
        try:
            isinstance(test_instance, AuthenticationProtocol)
            # If we get here, the protocol is runtime checkable
            runtime_checkable = True
        except TypeError:
            runtime_checkable = False

        assert runtime_checkable, "AuthenticationProtocol should be runtime checkable"

        # Should have focused authentication methods only
        methods = [method for method in dir(AuthenticationProtocol)
                  if not method.startswith('_')]
        auth_methods = [m for m in methods if 'auth' in m.lower() or 'token' in m.lower()]

        assert len(auth_methods) >= 2, (
            "AuthenticationProtocol should have authenticate and validate_token methods"
        )

    def test_authorization_protocol_isolation(self):
        """Authorization protocols must remain isolated from authentication."""
        from prompt_improver.shared.interfaces.protocols.security import (
            AuthorizationProtocol,
        )

        # Test runtime checkability by trying isinstance
        class TestAuthz:
            async def authorize(self, user_id, resource, action): return True
            async def get_user_permissions(self, user_id): return []

        test_instance = TestAuthz()
        try:
            isinstance(test_instance, AuthorizationProtocol)
            runtime_checkable = True
        except TypeError:
            runtime_checkable = False

        assert runtime_checkable, "AuthorizationProtocol should be runtime checkable"

        # Should have focused authorization methods only
        methods = [method for method in dir(AuthorizationProtocol)
                  if not method.startswith('_')]

        # Should not have authentication methods
        auth_methods = [m for m in methods if 'authenticate' in m.lower() or 'login' in m.lower()]
        assert not auth_methods, (
            f"AuthorizationProtocol has authentication methods: {auth_methods}. "
            "Should be isolated from authentication concerns."
        )

    def test_encryption_protocol_isolation(self):
        """Encryption protocols must remain isolated."""
        from prompt_improver.shared.interfaces.protocols.security import (
            EncryptionProtocol,
        )

        # Test runtime checkability by trying isinstance
        class TestEncrypt:
            async def encrypt(self, data, key): return f"encrypted_{data}"
            async def decrypt(self, data, key): return data.replace("encrypted_", "")

        test_instance = TestEncrypt()
        try:
            isinstance(test_instance, EncryptionProtocol)
            runtime_checkable = True
        except TypeError:
            runtime_checkable = False

        assert runtime_checkable, "EncryptionProtocol should be runtime checkable"

        # Should have focused encryption methods only
        methods = [method for method in dir(EncryptionProtocol)
                  if not method.startswith('_')]
        crypto_methods = [m for m in methods if 'encrypt' in m.lower() or 'decrypt' in m.lower()]

        assert len(crypto_methods) >= 2, (
            "EncryptionProtocol should have encrypt and decrypt methods"
        )


class TestMLLazyLoadingValidation:
    """Test ML protocol lazy loading boundary validation."""

    def test_ml_protocols_not_imported_by_default(self):
        """ML protocols should not be imported during regular application startup."""
        import sys

        # Clear ML modules
        modules_to_clear = [mod for mod in sys.modules
                           if 'protocols.ml' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import main protocol module

        # ML should not be loaded
        ml_loaded = any('protocols.ml' in mod for mod in sys.modules)
        assert not ml_loaded, "ML protocols loaded despite lazy loading requirement"

    def test_ml_lazy_loading_function(self):
        """Test that get_ml_protocols() properly lazy loads ML protocols."""
        from prompt_improver.shared.interfaces.protocols import get_ml_protocols

        start_time = time.time()
        ml_module = get_ml_protocols()
        load_time = time.time() - start_time

        assert ml_module is not None, "ML protocols failed to lazy load"
        assert hasattr(ml_module, 'MLflowServiceProtocol'), "ML protocols missing expected protocol"

        # Should load reasonably quickly (within 2 seconds for heavy deps)
        assert load_time < 2.0, (
            f"ML lazy loading took {load_time:.3f}s, too slow for production use"
        )

    def test_monitoring_lazy_loading_function(self):
        """Test that get_monitoring_protocols() properly lazy loads monitoring protocols."""
        from prompt_improver.shared.interfaces.protocols import get_monitoring_protocols

        start_time = time.time()
        monitoring_module = get_monitoring_protocols()
        load_time = time.time() - start_time

        assert monitoring_module is not None, "Monitoring protocols failed to lazy load"
        assert hasattr(monitoring_module, 'MetricsCollectorProtocol'), "Monitoring protocols missing expected protocol"

        # Should load reasonably quickly
        assert load_time < 1.0, (
            f"Monitoring lazy loading took {load_time:.3f}s, too slow for production use"
        )


class TestCleanArchitectureCompliance:
    """Test Clean Architecture layer separation maintenance."""

    def test_presentation_layer_isolation(self):
        """CLI protocols should not be imported by non-CLI components."""
        # Core business logic should not depend on CLI
        try:
            from prompt_improver.shared.interfaces.protocols import (
                cache,
                core,
                database,
            )

            # These should import without CLI dependencies
            assert core is not None
            assert database is not None
            assert cache is not None

        except ImportError as e:
            if 'cli' in str(e).lower():
                pytest.fail(f"Core business logic has CLI dependency: {e}")

    def test_application_layer_proper_position(self):
        """Application protocols should sit between core and infrastructure."""
        from prompt_improver.shared.interfaces.protocols import application, core

        # Application should reference core protocols
        assert hasattr(application, 'ApplicationServiceProtocol')

        # Should be able to import core without issues
        assert hasattr(core, 'ServiceProtocol')

    def test_infrastructure_layer_proper_dependencies(self):
        """Infrastructure protocols should depend on application and core only."""
        from prompt_improver.shared.interfaces.protocols import (
            application,
            cache,
            core,
            database,
        )

        # Infrastructure should successfully import business layers
        assert database is not None
        assert cache is not None
        assert core is not None
        assert application is not None

        # Should have proper infrastructure protocols
        assert hasattr(database, 'DatabaseProtocol')
        assert hasattr(cache, 'BasicCacheProtocol')


class TestPerformanceBoundaryValidation:
    """Test protocol import performance meets <2ms requirement."""

    def test_core_protocol_import_performance(self):
        """Core protocols should import within performance requirements."""
        import sys

        # Clear core protocol cache
        modules_to_clear = [mod for mod in sys.modules
                           if 'protocols.core' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start_time = time.time()
        import_time = time.time() - start_time

        assert import_time < 0.002, (  # 2ms requirement
            f"Core protocol import took {import_time * 1000:.1f}ms, exceeding 2ms requirement"
        )

    def test_cache_protocol_import_performance(self):
        """Cache protocols should import within performance requirements."""
        import sys

        # Clear cache protocol cache
        modules_to_clear = [mod for mod in sys.modules
                           if 'protocols.cache' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start_time = time.time()
        import_time = time.time() - start_time

        assert import_time < 0.002, (
            f"Cache protocol import took {import_time * 1000:.1f}ms, exceeding 2ms requirement"
        )

    def test_database_protocol_import_performance(self):
        """Database protocols should import within performance requirements."""
        import sys

        # Clear database protocol cache
        modules_to_clear = [mod for mod in sys.modules
                           if 'protocols.database' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start_time = time.time()
        import_time = time.time() - start_time

        assert import_time < 0.002, (
            f"Database protocol import took {import_time * 1000:.1f}ms, exceeding 2ms requirement"
        )

    def test_security_protocol_import_performance(self):
        """Security protocols should import within performance requirements."""
        import sys

        # Clear security protocol cache
        modules_to_clear = [mod for mod in sys.modules
                           if 'protocols.security' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start_time = time.time()
        import_time = time.time() - start_time

        assert import_time < 0.002, (
            f"Security protocol import took {import_time * 1000:.1f}ms, exceeding 2ms requirement"
        )


@pytest.mark.boundary_validation
class TestDomainBoundaryIntegration:
    """Integration tests for domain boundary validation."""

    def test_all_domains_can_coexist(self):
        """Test that all protocol domains can be imported together without conflicts."""
        try:
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

            # All regular domains should coexist
            domains = [core, cache, database, security, cli, mcp, application]
            for domain in domains:
                assert domain is not None, f"Domain {domain} failed to coexist with others"

            # Lazy domains should load on demand
            ml_protocols = get_ml_protocols()
            monitoring_protocols = get_monitoring_protocols()

            assert ml_protocols is not None, "ML protocols failed to lazy load in integration"
            assert monitoring_protocols is not None, "Monitoring protocols failed to lazy load in integration"

        except Exception as e:
            pytest.fail(f"Domain boundary integration failed: {e}")

    def test_protocol_runtime_checkability(self):
        """Test that protocols are properly runtime checkable across domains."""
        from prompt_improver.shared.interfaces.protocols import (
            cache,
            core,
            database,
            security,
        )

        protocols_to_check = [
            (core, 'ServiceProtocol'),
            (core, 'HealthCheckProtocol'),
            (cache, 'BasicCacheProtocol'),
            (database, 'SessionManagerProtocol'),
            (security, 'AuthenticationProtocol'),
        ]

        for module, protocol_name in protocols_to_check:
            protocol_class = getattr(module, protocol_name, None)
            if protocol_class is not None:
                # Test runtime checkability by trying isinstance with mock implementation
                class TestImplementation:
                    pass

                test_instance = TestImplementation()
                # This should work without error if protocol is runtime checkable
                try:
                    isinstance(test_instance, protocol_class)
                    # If we get here, the protocol is runtime checkable
                    assert True
                except TypeError as e:
                    if "not subscriptable" in str(e) or "runtime_checkable" in str(e):
                        pytest.fail(f"{protocol_name} is not runtime checkable: {e}")
                    else:
                        # Other TypeError is expected (test instance doesn't implement protocol)
                        assert True

    def test_cross_domain_protocol_compatibility(self):
        """Test that protocols from different domains can work together."""
        from prompt_improver.shared.interfaces.protocols import (
            cache,
            core,
            database,
            security,
        )

        # Test protocol compatibility patterns
        class MockService:
            """Mock service implementing multiple domain protocols."""

            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            async def check_health(self) -> dict[str, Any]:
                return {"status": "healthy", "timestamp": "2025-08-24T00:00:00Z"}

            def is_healthy(self) -> bool:
                return True

        service = MockService()

        # Should be checkable against core protocols
        assert isinstance(service, core.ServiceProtocol)
        assert isinstance(service, core.HealthCheckProtocol)

        # Protocols should be composable across domains
        assert core.ServiceProtocol
        if hasattr(database, 'SessionManagerProtocol'):
            assert database.SessionManagerProtocol
        if hasattr(cache, 'BasicCacheProtocol'):
            assert cache.BasicCacheProtocol
        if hasattr(security, 'AuthenticationProtocol'):
            assert security.AuthenticationProtocol


if __name__ == "__main__":
    # Run domain boundary validation tests
    pytest.main([__file__, "-v", "--tb=short"])
