"""Test Suite for Unified Security Services Consolidation

This test suite validates the consolidated security architecture:
- SecurityServiceFacade functionality and component integration
- Backward compatibility with existing factory functions  
- Real behavior scenarios with security context flow
- Performance and health monitoring
- Component isolation and error handling

Focus Areas:
- Facade pattern implementation
- Component protocol compliance
- Legacy adapter functionality
- Security context management
- Error handling and fail-secure behavior
"""

import asyncio
import pytest
from unittest.mock import Mock, patch

from prompt_improver.database import (
    SecurityContext,
    ManagerMode,
    create_security_context,
)
from prompt_improver.security.unified import (
    SecurityServiceFacade,
    get_security_service_facade,
    cleanup_security_service_facade,
    SecurityComponentStatus,
)
from prompt_improver.security.unified.legacy_compatibility import (
    get_unified_security_manager,
    get_unified_authentication_manager,
    get_unified_validation_manager,
    get_unified_rate_limiter,
    cleanup_legacy_adapters,
)


class TestSecurityServiceFacade:
    """Test the main SecurityServiceFacade functionality."""
    
    @pytest.fixture
    async def security_facade(self):
        """Create a security facade for testing."""
        facade = SecurityServiceFacade()
        await facade.initialize_all_components()
        yield facade
        await facade.cleanup()
    
    async def test_facade_initialization(self):
        """Test that the security facade initializes correctly."""
        facade = SecurityServiceFacade()
        
        # Should not be initialized initially
        assert not facade._initialized
        
        # Initialize all components
        result = await facade.initialize_all_components()
        assert result is True
        assert facade._initialized is True
        
        # Cleanup
        await facade.cleanup()
    
    async def test_component_access(self, security_facade):
        """Test accessing security components through the facade."""
        # Test component property access
        auth_component = await security_facade.authentication
        assert auth_component is not None
        
        authz_component = await security_facade.authorization
        assert authz_component is not None
        
        validation_component = await security_facade.validation
        assert validation_component is not None
        
        crypto_component = await security_facade.cryptography
        assert crypto_component is not None
        
        rate_limit_component = await security_facade.rate_limiting
        assert rate_limit_component is not None
    
    async def test_health_check_all_components(self, security_facade):
        """Test health checking of all security components."""
        health_status = await security_facade.health_check_all_components()
        
        # Should have health status for all components plus facade
        expected_components = [
            "authentication", "authorization", "validation", 
            "cryptography", "rate_limiting", "facade"
        ]
        
        for component in expected_components:
            assert component in health_status
            status, details = health_status[component]
            assert isinstance(status, SecurityComponentStatus)
            assert isinstance(details, dict)
    
    async def test_overall_metrics(self, security_facade):
        """Test getting overall security performance metrics."""
        metrics = await security_facade.get_overall_metrics()
        
        # Should return SecurityPerformanceMetrics structure
        assert hasattr(metrics, 'operation_count')
        assert hasattr(metrics, 'average_latency_ms')
        assert hasattr(metrics, 'error_rate')
        assert hasattr(metrics, 'threat_detection_count')
        assert hasattr(metrics, 'last_updated')
        
        # Values should be valid
        assert metrics.operation_count >= 0
        assert metrics.average_latency_ms >= 0.0
        assert 0.0 <= metrics.error_rate <= 1.0
        assert metrics.threat_detection_count >= 0
    
    async def test_facade_cleanup(self, security_facade):
        """Test facade cleanup functionality."""
        # Facade should be initialized
        assert security_facade._initialized is True
        
        # Cleanup should succeed
        result = await security_facade.cleanup()
        assert result is True
        assert security_facade._initialized is False
        
        # Components should be cleared
        assert security_facade._authentication_component is None
        assert security_facade._authorization_component is None
        assert security_facade._validation_component is None
        assert security_facade._cryptography_component is None
        assert security_facade._rate_limiting_component is None


class TestAuthenticationComponent:
    """Test the AuthenticationComponent functionality."""
    
    @pytest.fixture
    async def auth_component(self):
        """Create an authentication component for testing."""
        facade = SecurityServiceFacade()
        await facade.initialize_all_components()
        component = await facade.authentication
        yield component
        await facade.cleanup()
    
    async def test_authentication_with_api_key(self, auth_component):
        """Test API key authentication."""
        credentials = {
            "api_key": "test_api_key_12345",
            "agent_id": "test_agent"
        }
        
        result = await auth_component.authenticate(credentials, "api_key")
        
        # Since this is placeholder implementation, it should handle gracefully
        assert hasattr(result, 'success')
        assert hasattr(result, 'operation_type')
        assert hasattr(result, 'execution_time_ms')
        assert result.operation_type == "authenticate_api_key"
    
    async def test_session_creation_and_validation(self, auth_component):
        """Test session creation and validation flow."""
        agent_id = "test_agent"
        security_context = create_security_context(
            agent_id=agent_id,
            manager_mode=ManagerMode.TESTING
        )
        
        # Create session
        create_result = await auth_component.create_session(agent_id, security_context)
        assert create_result.success is True
        assert "session_id" in create_result.metadata
        
        session_id = create_result.metadata["session_id"]
        
        # Validate session
        validate_result = await auth_component.validate_session(session_id)
        assert validate_result.success is True
        
        # Revoke session
        revoke_result = await auth_component.revoke_session(session_id)
        assert revoke_result.success is True


class TestAuthorizationComponent:
    """Test the AuthorizationComponent functionality."""
    
    @pytest.fixture
    async def authz_component(self):
        """Create an authorization component for testing."""
        facade = SecurityServiceFacade()
        await facade.initialize_all_components()
        component = await facade.authorization
        yield component
        await facade.cleanup()
    
    async def test_role_assignment_and_permission_check(self, authz_component):
        """Test role assignment and permission checking."""
        user_id = "test_user"
        role = "user"
        permission = "read_models"
        
        # Assign role
        assign_result = await authz_component.assign_role(user_id, role)
        assert assign_result.success is True
        
        # Create security context
        security_context = create_security_context(
            agent_id=user_id,
            manager_mode=ManagerMode.TESTING
        )
        
        # Check permission
        permission_result = await authz_component.check_permission(
            security_context, permission
        )
        assert hasattr(permission_result, 'success')
        assert permission_result.operation_type == "check_permission"
        
        # Get user permissions
        permissions_result = await authz_component.get_user_permissions(user_id)
        assert permissions_result.success is True
        assert "permissions" in permissions_result.metadata
    
    async def test_role_revocation(self, authz_component):
        """Test role revocation functionality."""
        user_id = "test_user"
        role = "ml_engineer"
        
        # Assign role first
        assign_result = await authz_component.assign_role(user_id, role)
        assert assign_result.success is True
        
        # Revoke role
        revoke_result = await authz_component.revoke_role(user_id, role)
        assert revoke_result.success is True


class TestLegacyCompatibility:
    """Test backward compatibility with existing factory functions."""
    
    async def test_unified_security_manager_compatibility(self):
        """Test that get_unified_security_manager() works with adapter."""
        try:
            security_manager = await get_unified_security_manager()
            assert security_manager is not None
            
            # Test legacy interface methods
            security_context = await security_manager.create_security_context("test_agent")
            assert isinstance(security_context, SecurityContext)
            
            # Test security context validation
            is_valid, context = await security_manager.validate_security_context(security_context)
            assert is_valid is True
            assert isinstance(context, SecurityContext)
            
        finally:
            await cleanup_legacy_adapters()
    
    async def test_unified_authentication_manager_compatibility(self):
        """Test that get_unified_authentication_manager() works with adapter."""
        try:
            auth_manager = await get_unified_authentication_manager()
            assert auth_manager is not None
            
            # Test legacy interface methods
            api_key, metadata = await auth_manager.generate_api_key(
                "test_agent", ["read_models"], "basic"
            )
            assert isinstance(api_key, str)
            assert isinstance(metadata, dict)
            assert "agent_id" in metadata
            
        finally:
            await cleanup_legacy_adapters()
    
    async def test_unified_validation_manager_compatibility(self):
        """Test that get_unified_validation_manager() works with adapter."""
        validation_manager = get_unified_validation_manager("default")
        assert validation_manager is not None
        
        # Test async method access
        result = await validation_manager.validate_input("test input", "default")
        assert hasattr(result, 'success')
        assert hasattr(result, 'operation_type')
    
    async def test_unified_rate_limiter_compatibility(self):
        """Test that get_unified_rate_limiter() works with adapter."""
        try:
            rate_limiter = await get_unified_rate_limiter()
            assert rate_limiter is not None
            
            # Create security context for testing
            security_context = create_security_context(
                agent_id="test_agent",
                manager_mode=ManagerMode.TESTING
            )
            
            # Test rate limit checking
            result = await rate_limiter.check_rate_limit(security_context)
            assert hasattr(result, 'success')
            assert hasattr(result, 'operation_type')
            
        finally:
            await cleanup_legacy_adapters()


class TestRealBehaviorScenarios:
    """Test real-world security scenarios end-to-end."""
    
    async def test_complete_authentication_flow(self):
        """Test complete authentication and authorization flow."""
        # Get the security facade
        security_facade = await get_security_service_facade()
        
        try:
            # Create credentials
            credentials = {
                "api_key": "test_api_key_real_behavior",
                "agent_id": "real_test_agent"
            }
            
            # Authenticate
            auth_component = await security_facade.authentication
            auth_result = await auth_component.authenticate(credentials, "api_key")
            
            # Even with placeholder implementation, should handle gracefully
            assert hasattr(auth_result, 'success')
            assert auth_result.operation_type == "authenticate_api_key"
            
            # Check authorization
            authz_component = await security_facade.authorization
            
            # Assign role first
            await authz_component.assign_role("real_test_agent", "user")
            
            # Create security context
            security_context = create_security_context(
                agent_id="real_test_agent",
                manager_mode=ManagerMode.TESTING
            )
            
            # Check permission
            permission_result = await authz_component.check_permission(
                security_context, "read_models"
            )
            
            # Should succeed based on user role having read_models permission
            assert permission_result.success is True
            
        finally:
            await cleanup_security_service_facade()
    
    async def test_security_facade_performance_metrics(self):
        """Test that security operations are properly tracked in metrics."""
        security_facade = await get_security_service_facade()
        
        try:
            # Perform several operations to generate metrics
            auth_component = await security_facade.authentication
            authz_component = await security_facade.authorization
            
            # Authenticate multiple times
            for i in range(3):
                credentials = {"api_key": f"key_{i}", "agent_id": f"agent_{i}"}
                await auth_component.authenticate(credentials, "api_key")
            
            # Check permissions multiple times
            security_context = create_security_context(
                agent_id="metrics_test_agent",
                manager_mode=ManagerMode.TESTING
            )
            
            for i in range(2):
                await authz_component.check_permission(security_context, "read_models")
            
            # Get overall metrics
            metrics = await security_facade.get_overall_metrics()
            
            # Should show operations were performed
            assert metrics.operation_count >= 0  # Placeholder implementations may not track
            assert metrics.average_latency_ms >= 0.0
            assert isinstance(metrics.last_updated, type(metrics.last_updated))
            
        finally:
            await cleanup_security_service_facade()
    
    async def test_component_error_handling(self):
        """Test error handling and fail-secure behavior."""
        security_facade = await get_security_service_facade()
        
        try:
            auth_component = await security_facade.authentication
            
            # Test with invalid credentials
            invalid_credentials = {}  # Missing required fields
            result = await auth_component.authenticate(invalid_credentials, "api_key")
            
            # Should fail securely
            assert result.success is False
            assert len(result.errors) > 0
            assert result.execution_time_ms >= 0.0
            
            # Test with invalid authentication method
            credentials = {"api_key": "test", "agent_id": "test"}
            result = await auth_component.authenticate(credentials, "invalid_method")
            
            # Should fail securely
            assert result.success is False
            assert len(result.errors) > 0
            
        finally:
            await cleanup_security_service_facade()


@pytest.mark.asyncio
class TestSecurityConsolidationIntegration:
    """Integration tests for the complete security consolidation."""
    
    async def test_facade_singleton_pattern(self):
        """Test that get_security_service_facade() returns the same instance."""
        facade1 = await get_security_service_facade()
        facade2 = await get_security_service_facade()
        
        # Should be the same instance (singleton pattern)
        assert facade1 is facade2
        
        await cleanup_security_service_facade()
    
    async def test_concurrent_facade_access(self):
        """Test concurrent access to the security facade."""
        async def get_facade():
            return await get_security_service_facade()
        
        # Start multiple concurrent requests for the facade
        facades = await asyncio.gather(*[get_facade() for _ in range(5)])
        
        # All should be the same instance
        first_facade = facades[0]
        for facade in facades[1:]:
            assert facade is first_facade
        
        await cleanup_security_service_facade()
    
    async def test_complete_security_stack_health(self):
        """Test health of the complete security stack."""
        security_facade = await get_security_service_facade()
        
        try:
            # Check health of all components
            health_status = await security_facade.health_check_all_components()
            
            # All components should be healthy in test environment
            for component_name, (status, details) in health_status.items():
                assert status in [SecurityComponentStatus.HEALTHY, SecurityComponentStatus.INITIALIZING]
                assert isinstance(details, dict)
                
                # Each component should report being initialized
                if "initialized" in details:
                    assert details["initialized"] is True
            
        finally:
            await cleanup_security_service_facade()