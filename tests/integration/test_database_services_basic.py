"""
Basic integration tests for DatabaseServices architecture.
Simple tests without external containers.
"""

import pytest
import asyncio

from prompt_improver.database import (
    DatabaseServices,
    ManagerMode,
    get_database_services,
    create_security_context,
    SecurityTier,
)


class TestDatabaseServicesBasic:
    """Basic integration tests for DatabaseServices."""

    @pytest.mark.asyncio
    async def test_service_creation_all_modes(self):
        """Test DatabaseServices can be created in all modes."""
        modes = [
            ManagerMode.MCP_SERVER,
            ManagerMode.ML_TRAINING,
            ManagerMode.ADMIN,
            ManagerMode.ASYNC_MODERN,
            ManagerMode.HIGH_AVAILABILITY,
        ]
        
        for mode in modes:
            services = await get_database_services(mode)
            
            # Verify service creation
            assert services is not None
            assert isinstance(services, DatabaseServices)
            assert services.mode == mode
            
            # Verify composition structure
            assert hasattr(services, 'database')
            assert hasattr(services, 'cache') 
            assert hasattr(services, 'health_manager')
            assert hasattr(services, 'lock_manager')
            assert hasattr(services, 'pubsub')
            
            # Clean shutdown
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_security_context_creation(self):
        """Test security context creation with different tiers."""
        tiers = [
            SecurityTier.BASIC,
            SecurityTier.STANDARD,
            SecurityTier.PRIVILEGED,
            SecurityTier.ADMIN,
        ]
        
        for tier in tiers:
            context = create_security_context(f"user_{tier.value}", tier)
            
            # Verify context structure
            assert context.agent_id == f"user_{tier.value}"
            assert context.tier == tier
            assert context.created_at is not None
            assert context.expires_at is not None
            assert context.permissions is not None
            
            # Verify tier-specific permissions
            if tier == SecurityTier.BASIC:
                assert "read_only" in context.permissions
            elif tier == SecurityTier.ADMIN:
                assert "full_access" in context.permissions

    @pytest.mark.asyncio
    async def test_service_initialization_patterns(self):
        """Test different service initialization patterns."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        
        try:
            # Test initialization
            await services.initialize()
            
            # Verify initialization state
            # Note: without real DB, some services may not fully initialize
            # but they should not raise exceptions
            
            # Test health check structure
            health = await services.health_check()
            assert isinstance(health, dict)
            assert "status" in health
            assert "components" in health
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_service_creation(self):
        """Test concurrent service creation doesn't cause issues."""
        async def create_service(mode):
            services = await get_database_services(mode)
            await services.initialize()
            await asyncio.sleep(0.1)  # Simulate work
            await services.shutdown()
            return services.mode
        
        # Create multiple services concurrently
        modes = [ManagerMode.ASYNC_MODERN] * 5
        results = await asyncio.gather(
            *[create_service(mode) for mode in modes],
            return_exceptions=True
        )
        
        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Exceptions: {exceptions}"
        
        # Verify all completed
        assert len([r for r in results if r == ManagerMode.ASYNC_MODERN]) == 5

    def test_manager_mode_enum(self):
        """Test ManagerMode enum values."""
        expected_modes = {
            "MCP_SERVER",
            "ML_TRAINING", 
            "ADMIN",
            "ASYNC_MODERN",
            "HIGH_AVAILABILITY",
        }
        
        actual_modes = {mode.name for mode in ManagerMode}
        assert actual_modes == expected_modes

    def test_security_tier_enum(self):
        """Test SecurityTier enum values."""
        expected_tiers = {
            "BASIC",
            "STANDARD",
            "PRIVILEGED", 
            "ADMIN",
        }
        
        actual_tiers = {tier.name for tier in SecurityTier}
        assert actual_tiers == expected_tiers

    @pytest.mark.asyncio
    async def test_service_shutdown_cleanup(self):
        """Test proper cleanup during shutdown."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        await services.initialize()
        
        # Verify initialization
        assert not services.is_shutdown
        
        # Shutdown
        await services.shutdown()
        
        # Verify cleanup
        assert services.is_shutdown
        
        # Multiple shutdowns should be safe
        await services.shutdown()
        assert services.is_shutdown

    @pytest.mark.asyncio
    async def test_factory_functions(self):
        """Test factory functions work correctly."""
        from prompt_improver.database import (
            get_database_services,
            create_security_context,
            create_security_context_from_auth_result,
        )
        
        # Test basic factory
        services = await get_database_services(ManagerMode.ADMIN)
        assert services is not None
        await services.shutdown()
        
        # Test security context factory
        context = create_security_context("test", SecurityTier.STANDARD)
        assert context.agent_id == "test"
        
        # Test auth result factory (with minimal auth result)
        auth_result = {"user_id": "auth_test", "permissions": ["read"]}
        auth_context = create_security_context_from_auth_result(auth_result)
        assert auth_context.agent_id == "auth_test"

    @pytest.mark.asyncio
    async def test_composition_pattern(self):
        """Test the composition pattern is working correctly."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            # Verify composition structure - all services should be composed, not inherited
            assert hasattr(services, 'database')
            assert hasattr(services, 'cache')
            assert hasattr(services, 'health_manager')
            assert hasattr(services, 'lock_manager')
            assert hasattr(services, 'pubsub')
            
            # Verify services are separate objects (composition, not inheritance)
            assert services.database is not services.cache
            assert services.cache is not services.health_manager
            
            # Verify mode is accessible
            assert services.mode == ManagerMode.HIGH_AVAILABILITY
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling_patterns(self):
        """Test error handling in DatabaseServices."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        
        try:
            # Test initialization with potential errors
            await services.initialize()
            
            # Test health check with potential errors
            health = await services.health_check()
            
            # Health should return valid structure even with errors
            assert isinstance(health, dict)
            assert "status" in health
            
            # Status should be one of the expected values
            assert health["status"] in ["healthy", "degraded", "unhealthy"]
            
        finally:
            await services.shutdown()


class TestCompatibilityFunctions:
    """Test backwards compatibility functions."""

    @pytest.mark.asyncio
    async def test_compatibility_functions_exist(self):
        """Test that compatibility functions exist and work."""
        from prompt_improver.database import (
            get_unified_manager_async_modern,
            get_unified_manager_ml_training,
        )
        
        # Test async modern compatibility
        services1 = await get_unified_manager_async_modern()
        assert isinstance(services1, DatabaseServices)
        assert services1.mode == ManagerMode.ASYNC_MODERN
        await services1.shutdown()
        
        # Test ML training compatibility
        services2 = await get_unified_manager_ml_training()
        assert isinstance(services2, DatabaseServices)
        assert services2.mode == ManagerMode.ML_TRAINING
        await services2.shutdown()

    @pytest.mark.asyncio
    async def test_compatibility_function_equivalence(self):
        """Test compatibility functions return equivalent services."""
        # Direct call
        direct = await get_database_services(ManagerMode.ASYNC_MODERN)
        
        # Compatibility call
        from prompt_improver.database import get_unified_manager_async_modern
        compat = await get_unified_manager_async_modern()
        
        try:
            # Should be equivalent
            assert direct.mode == compat.mode
            assert type(direct) == type(compat)
            
        finally:
            await direct.shutdown()
            await compat.shutdown()


if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v"])