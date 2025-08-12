"""Security Component Protocols - Clean Interfaces for Consolidated Security Services

Defines the protocols and interfaces that security components must implement 
to work with the SecurityServiceFacade. These protocols ensure clean separation
of concerns while maintaining type safety and testability.

Protocol Hierarchy:
- SecurityComponent: Base protocol for all security components
- AuthenticationProtocol: User authentication and session management
- AuthorizationProtocol: Permission checking and access control  
- ValidationProtocol: Input/output validation and sanitization
- CryptographyProtocol: Encryption, hashing, and key management
- RateLimitingProtocol: Request rate limiting and throttling

Security Context Management:
All protocols use SecurityContext as the primary data structure for
security state management, ensuring consistency across components.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from pydantic import BaseModel, Field

from prompt_improver.database import (
    SecurityContext,
    SecurityPerformanceMetrics, 
    SecurityThreatScore,
    SecurityValidationResult
)


class SecurityComponentStatus(Enum):
    """Status of security components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    INITIALIZING = "initializing"
    MAINTENANCE = "maintenance"


class SecurityOperationResult(BaseModel):
    """Base result for all security operations."""
    success: bool = Field(description="Whether operation succeeded")
    operation_type: str = Field(description="Type of security operation")
    execution_time_ms: float = Field(ge=0, description="Operation execution time")
    security_context: SecurityContext | None = Field(default=None, description="Associated security context")
    errors: List[str] = Field(default_factory=list, description="List of error messages if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional operation metadata")


class SecurityComponent(Protocol):
    """Base protocol for all security components.
    
    Provides common interface for health checking, metrics collection,
    and lifecycle management that all security components must implement.
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the security component.
        
        Returns:
            True if initialization successful, False otherwise
        """
        ...
        
    @abstractmethod
    async def health_check(self) -> Tuple[SecurityComponentStatus, Dict[str, Any]]:
        """Check component health status.
        
        Returns:
            Tuple of (status, health_details)
        """
        ...
        
    @abstractmethod
    async def get_metrics(self) -> SecurityPerformanceMetrics:
        """Get security performance metrics for this component.
        
        Returns:
            Security performance metrics
        """
        ...
        
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup component resources.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        ...


class AuthenticationProtocol(SecurityComponent):
    """Protocol for authentication components.
    
    Handles user authentication, session management, and credential validation.
    All authentication operations must return consistent AuthenticationResult structures.
    """
    
    @abstractmethod
    async def authenticate(
        self, 
        credentials: Dict[str, Any],
        authentication_method: str = "api_key"
    ) -> SecurityOperationResult:
        """Authenticate user with provided credentials.
        
        Args:
            credentials: Authentication credentials (API key, session token, etc.)
            authentication_method: Method used for authentication
            
        Returns:
            SecurityOperationResult with authentication status
        """
        ...
        
    @abstractmethod
    async def create_session(
        self, 
        agent_id: str,
        security_context: SecurityContext
    ) -> SecurityOperationResult:
        """Create authenticated session for agent.
        
        Args:
            agent_id: Unique identifier for authenticated agent
            security_context: Security context for session
            
        Returns:
            SecurityOperationResult with session details
        """
        ...
        
    @abstractmethod
    async def validate_session(self, session_id: str) -> SecurityOperationResult:
        """Validate existing session.
        
        Args:
            session_id: Session identifier to validate
            
        Returns:
            SecurityOperationResult with session validation status
        """
        ...
        
    @abstractmethod
    async def revoke_session(self, session_id: str) -> SecurityOperationResult:
        """Revoke existing session.
        
        Args:
            session_id: Session identifier to revoke
            
        Returns:
            SecurityOperationResult with revocation status
        """
        ...


class AuthorizationProtocol(SecurityComponent):
    """Protocol for authorization components.
    
    Handles permission checking, role-based access control, and resource access validation.
    Implements RBAC (Role-Based Access Control) with fine-grained permissions.
    """
    
    @abstractmethod
    async def check_permission(
        self,
        security_context: SecurityContext,
        permission: str,
        resource: Optional[str] = None
    ) -> SecurityOperationResult:
        """Check if security context has required permission.
        
        Args:
            security_context: Security context to check
            permission: Permission identifier to check
            resource: Optional specific resource identifier
            
        Returns:
            SecurityOperationResult with authorization decision
        """
        ...
        
    @abstractmethod
    async def get_user_permissions(self, user_id: str) -> SecurityOperationResult:
        """Get all permissions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            SecurityOperationResult with user permissions
        """
        ...
        
    @abstractmethod
    async def assign_role(self, user_id: str, role: str) -> SecurityOperationResult:
        """Assign role to user.
        
        Args:
            user_id: User identifier
            role: Role identifier to assign
            
        Returns:
            SecurityOperationResult with assignment status
        """
        ...
        
    @abstractmethod
    async def revoke_role(self, user_id: str, role: str) -> SecurityOperationResult:
        """Revoke role from user.
        
        Args:
            user_id: User identifier  
            role: Role identifier to revoke
            
        Returns:
            SecurityOperationResult with revocation status
        """
        ...


class ValidationProtocol(SecurityComponent):
    """Protocol for validation components.
    
    Handles input/output validation, sanitization, and threat detection.
    Implements OWASP-compliant validation with comprehensive threat detection.
    """
    
    @abstractmethod
    async def validate_input(
        self,
        input_data: Any,
        validation_mode: str = "default",
        security_context: Optional[SecurityContext] = None
    ) -> SecurityOperationResult:
        """Validate input data for security threats.
        
        Args:
            input_data: Input data to validate
            validation_mode: Validation mode (strict, permissive, etc.)
            security_context: Optional security context for validation
            
        Returns:
            SecurityOperationResult with validation results and threat detection
        """
        ...
        
    @abstractmethod
    async def sanitize_input(
        self,
        input_data: str,
        sanitization_mode: str = "default"
    ) -> SecurityOperationResult:
        """Sanitize input data to prevent injection attacks.
        
        Args:
            input_data: Input string to sanitize
            sanitization_mode: Sanitization mode (html, sql, command, etc.)
            
        Returns:
            SecurityOperationResult with sanitized data
        """
        ...
        
    @abstractmethod
    async def validate_output(
        self,
        output_data: str,
        security_context: SecurityContext
    ) -> SecurityOperationResult:
        """Validate output data for security violations.
        
        Args:
            output_data: Output data to validate
            security_context: Security context for validation
            
        Returns:
            SecurityOperationResult with output validation results
        """
        ...
        
    @abstractmethod
    async def detect_threats(
        self,
        data: str,
        threat_types: Optional[List[str]] = None
    ) -> SecurityOperationResult:
        """Detect security threats in data.
        
        Args:
            data: Data to analyze for threats
            threat_types: Optional list of specific threat types to check
            
        Returns:
            SecurityOperationResult with threat detection results
        """
        ...


class CryptographyProtocol(SecurityComponent):
    """Protocol for cryptography components.
    
    Handles encryption, decryption, hashing, and key management operations.
    All cryptographic operations use NIST-approved algorithms and follow security best practices.
    """
    
    @abstractmethod
    async def hash_data(
        self,
        data: Union[str, bytes],
        algorithm: str = "sha256",
        salt: Optional[bytes] = None
    ) -> SecurityOperationResult:
        """Hash data using specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, sha512, etc.)
            salt: Optional salt for hashing
            
        Returns:
            SecurityOperationResult with hash value
        """
        ...
        
    @abstractmethod
    async def encrypt_data(
        self,
        data: Union[str, bytes],
        key_id: Optional[str] = None,
        security_context: Optional[SecurityContext] = None
    ) -> SecurityOperationResult:
        """Encrypt data using specified key.
        
        Args:
            data: Data to encrypt
            key_id: Optional specific key identifier
            security_context: Optional security context for key selection
            
        Returns:
            SecurityOperationResult with encrypted data
        """
        ...
        
    @abstractmethod
    async def decrypt_data(
        self,
        encrypted_data: bytes,
        key_id: str,
        security_context: SecurityContext
    ) -> SecurityOperationResult:
        """Decrypt data using specified key.
        
        Args:
            encrypted_data: Data to decrypt
            key_id: Key identifier for decryption
            security_context: Security context for authorization
            
        Returns:
            SecurityOperationResult with decrypted data
        """
        ...
        
    @abstractmethod
    async def generate_random(
        self,
        length: int,
        random_type: str = "bytes"
    ) -> SecurityOperationResult:
        """Generate cryptographically secure random data.
        
        Args:
            length: Length of random data to generate
            random_type: Type of random data (bytes, hex, urlsafe, token)
            
        Returns:
            SecurityOperationResult with random data
        """
        ...


class RateLimitingProtocol(SecurityComponent):
    """Protocol for rate limiting components.
    
    Handles request rate limiting, burst control, and traffic throttling.
    Implements sliding window rate limiting with Redis-based persistence.
    """
    
    @abstractmethod
    async def check_rate_limit(
        self,
        security_context: SecurityContext,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Check if request is within rate limits.
        
        Args:
            security_context: Security context containing agent info
            operation_type: Type of operation being rate limited
            
        Returns:
            SecurityOperationResult with rate limit status
        """
        ...
        
    @abstractmethod
    async def update_rate_limit(
        self,
        security_context: SecurityContext,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Update rate limit counters after successful operation.
        
        Args:
            security_context: Security context containing agent info  
            operation_type: Type of operation that was performed
            
        Returns:
            SecurityOperationResult with updated rate limit status
        """
        ...
        
    @abstractmethod
    async def get_rate_limit_status(
        self,
        agent_id: str,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Get current rate limit status for agent.
        
        Args:
            agent_id: Agent identifier
            operation_type: Type of operation to check
            
        Returns:
            SecurityOperationResult with rate limit status details
        """
        ...
        
    @abstractmethod
    async def reset_rate_limit(
        self,
        agent_id: str,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Reset rate limit counters for agent.
        
        Args:
            agent_id: Agent identifier
            operation_type: Type of operation to reset
            
        Returns:
            SecurityOperationResult with reset status
        """
        ...


class SecurityServiceFacadeProtocol(Protocol):
    """Protocol for the main SecurityServiceFacade.
    
    Defines the unified interface that consolidates all security operations
    through component delegation while maintaining clean separation of concerns.
    """
    
    @property
    @abstractmethod
    def authentication(self) -> AuthenticationProtocol:
        """Get authentication component."""
        ...
        
    @property  
    @abstractmethod
    def authorization(self) -> AuthorizationProtocol:
        """Get authorization component."""
        ...
        
    @property
    @abstractmethod
    def validation(self) -> ValidationProtocol:
        """Get validation component."""
        ...
        
    @property
    @abstractmethod
    def cryptography(self) -> CryptographyProtocol:
        """Get cryptography component."""
        ...
        
    @property
    @abstractmethod
    def rate_limiting(self) -> RateLimitingProtocol:
        """Get rate limiting component."""
        ...
        
    @abstractmethod
    async def initialize_all_components(self) -> bool:
        """Initialize all security components.
        
        Returns:
            True if all components initialized successfully
        """
        ...
        
    @abstractmethod
    async def health_check_all_components(self) -> Dict[str, Tuple[SecurityComponentStatus, Dict[str, Any]]]:
        """Check health of all security components.
        
        Returns:
            Dictionary mapping component names to their health status
        """
        ...
        
    @abstractmethod
    async def get_overall_metrics(self) -> SecurityPerformanceMetrics:
        """Get overall security performance metrics.
        
        Returns:
            Aggregated security performance metrics from all components
        """
        ...