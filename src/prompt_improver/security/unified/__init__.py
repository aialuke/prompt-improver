"""Unified Security Services - Consolidated Security Architecture

This package provides the consolidated security architecture that replaces
multiple overlapping security services with a unified facade pattern.

Key Components:
- SecurityServiceFacade: Main entry point for all security operations
- AuthenticationComponent: User authentication and session management
- AuthorizationComponent: Role-based access control and permissions
- ValidationComponent: Input/output validation and threat detection
- CryptographyComponent: Encryption, hashing, and key management
- RateLimitingComponent: Request rate limiting and throttling

Architecture Benefits:
- Single entry point eliminates duplicate functionality
- Clean component separation improves maintainability
- Protocol-based design enables easy testing and mocking
- Backward compatibility preserves existing integrations
- Performance optimization through unified caching and metrics

Usage:
    from prompt_improver.security.unified import get_security_service_facade
    
    security = await get_security_service_facade()
    auth_result = await security.authentication.authenticate(credentials)
    is_authorized = await security.authorization.check_permission(context, permission)
"""

from prompt_improver.security.unified.authentication_component import AuthenticationComponent
from prompt_improver.security.unified.authorization_component import AuthorizationComponent
from prompt_improver.security.unified.cryptography_component import CryptographyComponent
from prompt_improver.security.unified.protocols import (
    AuthenticationProtocol,
    AuthorizationProtocol,
    CryptographyProtocol,
    RateLimitingProtocol,
    SecurityComponent,
    SecurityComponentStatus,
    SecurityOperationResult,
    SecurityServiceFacadeProtocol,
    ValidationProtocol,
)
from prompt_improver.security.unified.rate_limiting_component import RateLimitingComponent
from prompt_improver.security.unified.security_service_facade import (
    SecurityServiceFacade,
    cleanup_security_service_facade,
    get_security_service_facade,
)
from prompt_improver.security.unified.validation_component import ValidationComponent

__all__ = [
    # Main facade
    "SecurityServiceFacade",
    "get_security_service_facade", 
    "cleanup_security_service_facade",
    
    # Components
    "AuthenticationComponent",
    "AuthorizationComponent", 
    "ValidationComponent",
    "CryptographyComponent",
    "RateLimitingComponent",
    
    # Protocols
    "SecurityServiceFacadeProtocol",
    "SecurityComponent",
    "AuthenticationProtocol",
    "AuthorizationProtocol",
    "ValidationProtocol", 
    "CryptographyProtocol",
    "RateLimitingProtocol",
    
    # Common types
    "SecurityComponentStatus",
    "SecurityOperationResult",
]