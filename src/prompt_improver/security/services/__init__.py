"""Decomposed Security Services - Clean Architecture Implementation

This module provides the decomposed security services that replace the unified
security manager god object. Each service has a single responsibility and
implements fail-secure design principles.

Services:
- AuthenticationService: User/system authentication with session management
- AuthorizationService: RBAC and permission-based access control  
- ValidationService: OWASP-compliant input/output validation
- CryptoService: NIST-compliant cryptographic operations
- SecurityMonitoringService: Real-time monitoring and threat detection
- SecurityServiceFacade: Unified entry point for all security operations

All services implement protocol-based interfaces for clean separation of concerns
and comprehensive security monitoring with fail-secure design.
"""

from .authentication_service import (
    AuthenticationService,
    create_authentication_service,
)
from .authorization_service import (
    AuthorizationService,
    Permission,
    Role,
    create_authorization_service,
)
from .crypto_service import (
    CryptoService,
    EncryptionResult,
    KeyDerivationConfig,
    create_crypto_service,
)
from .security_monitoring_service import (
    SecurityMonitoringService,
    SecurityEvent,
    SecurityIncident,
    ThreatSeverity,
    IncidentStatus,
    create_security_monitoring_service,
)
from .security_service_facade import (
    SecurityServiceFacade,
    get_security_service_facade,
    get_api_security_manager,
    get_mcp_security_manager,
    get_internal_security_manager,
    get_admin_security_manager,
    get_high_security_manager,
    cleanup_security_service_facade,
)
from .validation_service import (
    ValidationService,
    ValidationRule,
    ThreatPattern,
    create_validation_service,
)

__all__ = [
    # Core Services
    "AuthenticationService",
    "AuthorizationService", 
    "ValidationService",
    "CryptoService",
    "SecurityMonitoringService",
    
    # Facade and Factory Functions
    "SecurityServiceFacade",
    "get_security_service_facade",
    "get_api_security_manager",
    "get_mcp_security_manager", 
    "get_internal_security_manager",
    "get_admin_security_manager",
    "get_high_security_manager",
    "cleanup_security_service_facade",
    
    # Service Creation Functions
    "create_authentication_service",
    "create_authorization_service",
    "create_validation_service", 
    "create_crypto_service",
    "create_security_monitoring_service",
    
    # Data Classes and Types
    "Permission",
    "Role",
    "EncryptionResult",
    "KeyDerivationConfig",
    "SecurityEvent",
    "SecurityIncident", 
    "ValidationRule",
    "ThreatPattern",
    "ThreatSeverity",
    "IncidentStatus",
]