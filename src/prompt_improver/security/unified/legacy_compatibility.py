"""Legacy Compatibility Layer - Backward Compatibility Bridge

This module provides backward compatibility for existing security service
factory functions by delegating to the new SecurityServiceFacade components.
This ensures that existing code continues to work without modification while
benefiting from the consolidated security architecture.

Compatibility Functions:
- get_unified_security_manager() -> SecurityServiceFacade wrapper
- get_unified_authentication_manager() -> AuthenticationComponent wrapper
- get_unified_validation_manager() -> ValidationComponent wrapper
- get_unified_rate_limiter() -> RateLimitingComponent wrapper

Migration Strategy:
1. Existing factory functions delegate to facade components
2. Legacy interfaces are preserved through adapter classes
3. Gradual migration path allows incremental adoption
4. Full backward compatibility maintained during transition
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from prompt_improver.database import SecurityContext
from prompt_improver.security.unified.security_service_facade import get_security_service_facade

logger = logging.getLogger(__name__)


class UnifiedSecurityManagerAdapter:
    """Adapter that wraps SecurityServiceFacade to provide legacy interface."""
    
    def __init__(self, facade):
        self._facade = facade
        self._initialized = False
    
    async def initialize(self):
        """Initialize the adapter."""
        if not self._initialized:
            await self._facade.initialize_all_components()
            self._initialized = True
    
    async def create_security_context(
        self, 
        agent_id: str, 
        operation_type: str = "general"
    ) -> SecurityContext:
        """Create security context - delegates to database services."""
        from prompt_improver.database import create_security_context, ManagerMode
        return create_security_context(agent_id=agent_id, manager_mode=ManagerMode.PRODUCTION)
    
    async def validate_security_context(
        self,
        security_context: SecurityContext,
        operation_type: str = "general"
    ) -> tuple[bool, SecurityContext]:
        """Validate security context."""
        # Legacy interface - always return valid for now
        return True, security_context
    
    async def encrypt_data(
        self,
        security_context: SecurityContext,
        data: str,
        key_id: Optional[str] = None
    ) -> bytes:
        """Encrypt data using crypto component."""
        crypto = await self._facade.cryptography
        result = await crypto.encrypt_data(data, key_id, security_context)
        
        if result.success:
            return result.metadata.get("encrypted_data", b"")
        else:
            raise RuntimeError(f"Encryption failed: {result.errors}")
    
    async def decrypt_data(
        self,
        security_context: SecurityContext,
        encrypted_data: bytes,
        key_id: str
    ) -> str:
        """Decrypt data using crypto component."""
        crypto = await self._facade.cryptography
        result = await crypto.decrypt_data(encrypted_data, key_id, security_context)
        
        if result.success:
            return result.metadata.get("decrypted_data", "")
        else:
            raise RuntimeError(f"Decryption failed: {result.errors}")


class UnifiedAuthenticationManagerAdapter:
    """Adapter that wraps AuthenticationComponent to provide legacy interface."""
    
    def __init__(self, facade):
        self._facade = facade
        self._initialized = False
    
    async def initialize(self):
        """Initialize the adapter."""
        if not self._initialized:
            await self._facade.initialize_all_components()
            self._initialized = True
    
    async def authenticate_request(self, request_context: Dict[str, Any]):
        """Authenticate request using authentication component."""
        auth = await self._facade.authentication
        return await auth.authenticate(request_context)
    
    async def generate_api_key(
        self,
        agent_id: str,
        permissions: list[str],
        tier: str = "basic", 
        expires_hours: Optional[int] = None
    ) -> tuple[str, Dict[str, Any]]:
        """Generate API key - placeholder for now."""
        # This would need to be implemented with real API key generation
        import secrets
        api_key = secrets.token_urlsafe(32)
        metadata = {
            "agent_id": agent_id,
            "permissions": permissions,
            "tier": tier,
            "expires_hours": expires_hours
        }
        return api_key, metadata
    
    async def validate_api_key(self, api_key: str, agent_id: str) -> Dict[str, Any]:
        """Validate API key."""
        auth = await self._facade.authentication
        result = await auth.authenticate(
            {"api_key": api_key, "agent_id": agent_id},
            "api_key"
        )
        
        return {
            "valid": result.success,
            "agent_id": agent_id,
            "errors": result.errors if not result.success else []
        }


class UnifiedValidationManagerAdapter:
    """Adapter that wraps ValidationComponent to provide legacy interface."""
    
    def __init__(self, facade):
        self._facade = facade
        self._initialized = False
    
    async def initialize(self):
        """Initialize the adapter."""
        if not self._initialized:
            await self._facade.initialize_all_components()
            self._initialized = True
    
    async def validate_input(
        self,
        input_data: Any,
        validation_mode: str = "default",
        security_context: Optional[SecurityContext] = None
    ):
        """Validate input using validation component."""
        validation = await self._facade.validation
        return await validation.validate_input(input_data, validation_mode, security_context)
    
    async def sanitize_input(self, input_data: str, mode: str = "default"):
        """Sanitize input using validation component."""
        validation = await self._facade.validation
        result = await validation.sanitize_input(input_data, mode)
        
        if result.success:
            return result.metadata.get("sanitized_data", input_data)
        else:
            raise ValueError(f"Sanitization failed: {result.errors}")
    
    async def validate_output(self, output_data: str, security_context: SecurityContext):
        """Validate output using validation component."""
        validation = await self._facade.validation
        return await validation.validate_output(output_data, security_context)


class UnifiedRateLimiterAdapter:
    """Adapter that wraps RateLimitingComponent to provide legacy interface."""
    
    def __init__(self, facade):
        self._facade = facade
        self._initialized = False
    
    async def initialize(self):
        """Initialize the adapter."""
        if not self._initialized:
            await self._facade.initialize_all_components()
            self._initialized = True
    
    async def check_rate_limit(
        self,
        security_context: SecurityContext,
        operation_type: str = "default"
    ):
        """Check rate limit using rate limiting component."""
        rate_limiter = await self._facade.rate_limiting
        return await rate_limiter.check_rate_limit(security_context, operation_type)
    
    async def update_rate_limit(
        self,
        security_context: SecurityContext,
        operation_type: str = "default"
    ):
        """Update rate limit using rate limiting component."""
        rate_limiter = await self._facade.rate_limiting
        return await rate_limiter.update_rate_limit(security_context, operation_type)


# Global adapter instances  
_security_manager_adapter: Optional[UnifiedSecurityManagerAdapter] = None
_authentication_manager_adapter: Optional[UnifiedAuthenticationManagerAdapter] = None
_validation_manager_adapter: Optional[UnifiedValidationManagerAdapter] = None
_rate_limiter_adapter: Optional[UnifiedRateLimiterAdapter] = None
_adapter_lock = asyncio.Lock()


async def get_unified_security_manager():
    """Get unified security manager - delegates to SecurityServiceFacade."""
    global _security_manager_adapter
    
    if _security_manager_adapter is not None:
        return _security_manager_adapter
    
    async with _adapter_lock:
        if _security_manager_adapter is not None:
            return _security_manager_adapter
        
        try:
            facade = await get_security_service_facade()
            _security_manager_adapter = UnifiedSecurityManagerAdapter(facade)
            await _security_manager_adapter.initialize()
            
            logger.info("UnifiedSecurityManager compatibility adapter created")
            return _security_manager_adapter
            
        except Exception as e:
            logger.error(f"Failed to create UnifiedSecurityManager adapter: {e}")
            raise


async def get_unified_authentication_manager():
    """Get unified authentication manager - delegates to SecurityServiceFacade."""
    global _authentication_manager_adapter
    
    if _authentication_manager_adapter is not None:
        return _authentication_manager_adapter
    
    async with _adapter_lock:
        if _authentication_manager_adapter is not None:
            return _authentication_manager_adapter
        
        try:
            facade = await get_security_service_facade()
            _authentication_manager_adapter = UnifiedAuthenticationManagerAdapter(facade)
            await _authentication_manager_adapter.initialize()
            
            logger.info("UnifiedAuthenticationManager compatibility adapter created")
            return _authentication_manager_adapter
            
        except Exception as e:
            logger.error(f"Failed to create UnifiedAuthenticationManager adapter: {e}")
            raise


def get_unified_validation_manager(validation_mode: str = "default"):
    """Get unified validation manager - delegates to SecurityServiceFacade."""
    # Note: This function is synchronous in the original, but we need async facade
    # For now, we'll create a wrapper that handles the async initialization
    
    class ValidationManagerWrapper:
        def __init__(self, validation_mode: str):
            self.validation_mode = validation_mode
            self._adapter: Optional[UnifiedValidationManagerAdapter] = None
        
        async def _ensure_initialized(self):
            if self._adapter is None:
                facade = await get_security_service_facade()
                self._adapter = UnifiedValidationManagerAdapter(facade)
                await self._adapter.initialize()
        
        async def validate_input(self, *args, **kwargs):
            await self._ensure_initialized()
            return await self._adapter.validate_input(*args, **kwargs)
        
        async def sanitize_input(self, *args, **kwargs):
            await self._ensure_initialized()
            return await self._adapter.sanitize_input(*args, **kwargs)
        
        async def validate_output(self, *args, **kwargs):
            await self._ensure_initialized()
            return await self._adapter.validate_output(*args, **kwargs)
    
    return ValidationManagerWrapper(validation_mode)


async def get_unified_rate_limiter():
    """Get unified rate limiter - delegates to SecurityServiceFacade."""
    global _rate_limiter_adapter
    
    if _rate_limiter_adapter is not None:
        return _rate_limiter_adapter
    
    async with _adapter_lock:
        if _rate_limiter_adapter is not None:
            return _rate_limiter_adapter
        
        try:
            facade = await get_security_service_facade()
            _rate_limiter_adapter = UnifiedRateLimiterAdapter(facade)
            await _rate_limiter_adapter.initialize()
            
            logger.info("UnifiedRateLimiter compatibility adapter created")
            return _rate_limiter_adapter
            
        except Exception as e:
            logger.error(f"Failed to create UnifiedRateLimiter adapter: {e}")
            raise


# Legacy validation manager factory functions for backward compatibility
def get_mcp_validation_manager():
    """Get MCP validation manager."""
    return get_unified_validation_manager("mcp_server")


def get_api_validation_manager():
    """Get API validation manager."""
    return get_unified_validation_manager("api")


def get_internal_validation_manager():
    """Get internal validation manager."""
    return get_unified_validation_manager("internal")


def get_admin_validation_manager():
    """Get admin validation manager."""
    return get_unified_validation_manager("admin")


def get_high_security_validation_manager():
    """Get high security validation manager."""
    return get_unified_validation_manager("high_security")


def get_ml_validation_manager():
    """Get ML validation manager."""
    return get_unified_validation_manager("ml_processing")


def create_security_aware_validation_manager(security_context: SecurityContext):
    """Create security-aware validation manager."""
    return get_unified_validation_manager("security_aware")


def create_validation_test_adapter():
    """Create validation test adapter."""
    return get_unified_validation_manager("testing")


async def cleanup_legacy_adapters():
    """Cleanup all legacy adapters."""
    global _security_manager_adapter, _authentication_manager_adapter
    global _validation_manager_adapter, _rate_limiter_adapter
    
    async with _adapter_lock:
        _security_manager_adapter = None
        _authentication_manager_adapter = None
        _validation_manager_adapter = None
        _rate_limiter_adapter = None
        
        logger.info("Legacy compatibility adapters cleaned up")