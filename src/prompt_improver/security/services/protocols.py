"""Security Service Protocols - Protocol-based Security Interface Contracts

Defines protocol interfaces for all security services to ensure clean separation
of concerns and fail-secure design patterns.

Each protocol enforces:
- Fail-secure by default (deny on error)
- Comprehensive audit logging
- Security incident escalation
- Performance monitoring integration
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple

from prompt_improver.database import SecurityContext, SecurityValidationResult


class SecurityStateManagerProtocol(Protocol):
    """Protocol for managing shared security state and metrics."""

    @abstractmethod
    async def record_security_operation(
        self,
        operation_type: str,
        success: bool,
        agent_id: str = "system",
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Record a security operation for audit and metrics."""
        ...

    @abstractmethod
    async def handle_security_incident(
        self,
        threat_level: str,
        operation_type: str,
        agent_id: str,
        details: Dict[str, Any],
    ) -> str:
        """Handle a security incident and return incident ID."""
        ...

    @abstractmethod
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics and statistics."""
        ...

    @abstractmethod
    async def is_agent_blocked(self, agent_id: str) -> bool:
        """Check if an agent is currently blocked."""
        ...

    @abstractmethod
    async def block_agent(self, agent_id: str, duration_hours: int = 1) -> None:
        """Block an agent for a specified duration."""
        ...

    @abstractmethod
    async def get_security_incidents(
        self, limit: int = 50, threat_level: str | None = None
    ) -> List[Dict[str, Any]]:
        """Get recent security incidents."""
        ...


class AuthenticationServiceProtocol(Protocol):
    """Protocol for authentication operations with fail-secure design."""

    @abstractmethod
    async def authenticate_agent(
        self,
        agent_id: str,
        credentials: Dict[str, Any],
        additional_context: Dict[str, Any] | None = None,
    ) -> Tuple[bool, SecurityContext]:
        """Authenticate an agent with comprehensive security checks.
        
        Returns:
            Tuple of (success, security_context)
            
        Fail-secure: Returns (False, minimal_context) on any error
        """
        ...

    @abstractmethod
    async def validate_credentials(
        self, agent_id: str, credentials: Dict[str, Any]
    ) -> bool:
        """Validate agent credentials against authentication store.
        
        Fail-secure: Returns False on any validation error
        """
        ...

    @abstractmethod
    async def check_authentication_rate_limit(self, agent_id: str) -> bool:
        """Check if agent is within authentication rate limits.
        
        Fail-secure: Returns False if rate limit exceeded or check fails
        """
        ...

    @abstractmethod
    async def record_authentication_attempt(
        self, agent_id: str, success: bool
    ) -> None:
        """Record an authentication attempt for rate limiting and audit."""
        ...

    @abstractmethod
    async def create_security_context(
        self,
        agent_id: str,
        authenticated: bool,
        additional_context: Dict[str, Any] | None = None,
    ) -> SecurityContext:
        """Create a security context with proper validation and metrics."""
        ...


class AuthorizationServiceProtocol(Protocol):
    """Protocol for authorization and access control operations."""

    @abstractmethod
    async def authorize_operation(
        self,
        security_context: SecurityContext,
        operation: str,
        resource: str,
        additional_checks: Dict[str, Any] | None = None,
    ) -> bool:
        """Authorize an operation with comprehensive security validation.
        
        Fail-secure: Returns False on any authorization error
        """
        ...

    @abstractmethod
    async def check_permissions(
        self,
        security_context: SecurityContext,
        required_permissions: List[str],
    ) -> bool:
        """Check if security context has required permissions.
        
        Fail-secure: Returns False if any permission missing or check fails
        """
        ...

    @abstractmethod
    async def validate_security_context(
        self, security_context: SecurityContext
    ) -> bool:
        """Validate security context integrity and expiration.
        
        Fail-secure: Returns False if context invalid or expired
        """
        ...

    @abstractmethod
    async def check_rate_limits(
        self, security_context: SecurityContext, operation: str
    ) -> bool:
        """Check operation-specific rate limits.
        
        Fail-secure: Returns False if rate limit exceeded or check fails
        """
        ...


class ValidationServiceProtocol(Protocol):
    """Protocol for input validation and sanitization operations."""

    @abstractmethod
    async def validate_input(
        self,
        security_context: SecurityContext,
        input_data: Any,
        validation_rules: Dict[str, Any] | None = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate input data with security checks.
        
        Returns:
            Tuple of (is_valid, validation_results)
            
        Fail-secure: Returns (False, error_details) on validation failure
        """
        ...

    @abstractmethod
    async def sanitize_input(
        self,
        input_data: Any,
        sanitization_rules: Dict[str, Any] | None = None,
    ) -> Any:
        """Sanitize input data to prevent injection attacks.
        
        Fail-secure: Returns safe default or raises exception on sanitization failure
        """
        ...

    @abstractmethod
    async def validate_output(
        self,
        security_context: SecurityContext,
        output_data: Any,
        validation_rules: Dict[str, Any] | None = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate output data before sending to prevent data leakage.
        
        Fail-secure: Returns (False, error_details) on validation failure
        """
        ...

    @abstractmethod
    async def check_content_security_policy(
        self, content: str, policy_rules: Dict[str, Any] | None = None
    ) -> bool:
        """Check content against security policies.
        
        Fail-secure: Returns False if content violates security policies
        """
        ...


class CryptoServiceProtocol(Protocol):
    """Protocol for cryptographic operations and key management."""

    @abstractmethod
    async def encrypt_data(
        self,
        security_context: SecurityContext,
        data: str | bytes,
        key_id: str | None = None,
    ) -> Tuple[bytes, str]:
        """Encrypt data using secure key management.
        
        Returns:
            Tuple of (encrypted_data, key_id_used)
            
        Fail-secure: Raises exception on encryption failure
        """
        ...

    @abstractmethod
    async def decrypt_data(
        self,
        security_context: SecurityContext,
        encrypted_data: bytes,
        key_id: str,
    ) -> bytes:
        """Decrypt data using secure key management.
        
        Fail-secure: Raises exception on decryption failure or invalid key
        """
        ...

    @abstractmethod
    async def hash_data(
        self,
        data: str | bytes,
        salt: str | None = None,
        algorithm: str = "sha256",
    ) -> str:
        """Create secure hash of data with optional salt.
        
        Fail-secure: Uses secure random salt if none provided
        """
        ...

    @abstractmethod
    async def verify_hash(
        self,
        data: str | bytes,
        hash_value: str,
        salt: str | None = None,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify data against hash value.
        
        Fail-secure: Returns False on verification failure or error
        """
        ...

    @abstractmethod
    async def generate_secure_token(
        self, length: int = 32, purpose: str = "general"
    ) -> str:
        """Generate cryptographically secure random token.
        
        Fail-secure: Uses system secure random generator
        """
        ...

    @abstractmethod
    async def rotate_keys(self, key_id: str | None = None) -> List[str]:
        """Rotate encryption keys for security.
        
        Returns:
            List of new key IDs
            
        Fail-secure: Maintains old keys until rotation confirmed
        """
        ...


class SecurityMonitoringServiceProtocol(Protocol):
    """Protocol for security monitoring and threat detection."""

    @abstractmethod
    async def detect_threat_patterns(
        self, security_context: SecurityContext, operation_data: Dict[str, Any]
    ) -> Tuple[bool, float, List[str]]:
        """Detect threat patterns in security operations.
        
        Returns:
            Tuple of (threat_detected, threat_score, threat_factors)
            
        Fail-secure: Returns (True, 1.0, ["detection_error"]) on detection failure
        """
        ...

    @abstractmethod
    async def analyze_security_behavior(
        self, agent_id: str, recent_operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze agent behavior for security anomalies.
        
        Fail-secure: Returns suspicious behavior indicators on analysis failure
        """
        ...

    @abstractmethod
    async def get_security_health_status(self) -> Dict[str, Any]:
        """Get overall security system health status."""
        ...

    @abstractmethod
    async def trigger_security_alert(
        self,
        alert_level: str,
        message: str,
        context: Dict[str, Any],
    ) -> None:
        """Trigger security alert for immediate attention."""
        ...