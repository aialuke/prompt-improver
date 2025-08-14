"""Protocol interfaces for SecurityServiceFacade decomposition.

Comprehensive security infrastructure with fail-secure defaults.
Each service handles a specific security domain with focused responsibilities.
"""

from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable
from datetime import datetime
from enum import Enum
from uuid import UUID


class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ThreatLevel(Enum):
    """Threat level enumeration."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@runtime_checkable
class AuthenticationServiceProtocol(Protocol):
    """Protocol for authentication and session management."""
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        method: str = "standard"
    ) -> Dict[str, Any]:
        """Authenticate user with credentials.
        
        Args:
            credentials: Authentication credentials
            method: Authentication method
            
        Returns:
            Authentication result with tokens
        """
        ...
    
    async def validate_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Optional[Dict[str, Any]]:
        """Validate authentication token.
        
        Args:
            token: Token to validate
            token_type: Type of token
            
        Returns:
            Token claims if valid, None otherwise
        """
        ...
    
    async def refresh_token(
        self,
        refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        """Refresh authentication tokens.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New tokens if valid
        """
        ...
    
    async def create_session(
        self,
        user_id: UUID,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create authentication session.
        
        Args:
            user_id: User identifier
            metadata: Session metadata
            
        Returns:
            Session details
        """
        ...
    
    async def validate_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Validate active session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if valid
        """
        ...
    
    async def revoke_session(
        self,
        session_id: str
    ) -> bool:
        """Revoke authentication session.
        
        Args:
            session_id: Session to revoke
            
        Returns:
            Success status
        """
        ...
    
    async def get_active_sessions(
        self,
        user_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get active sessions for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active sessions
        """
        ...


@runtime_checkable
class AuthorizationServiceProtocol(Protocol):
    """Protocol for access control and permissions."""
    
    async def check_permission(
        self,
        user_id: UUID,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if user has permission.
        
        Args:
            user_id: User identifier
            resource: Resource identifier
            action: Action to perform
            context: Additional context
            
        Returns:
            Permission granted status
        """
        ...
    
    async def get_user_roles(
        self,
        user_id: UUID
    ) -> List[str]:
        """Get roles for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of role names
        """
        ...
    
    async def get_role_permissions(
        self,
        role: str
    ) -> List[Dict[str, Any]]:
        """Get permissions for role.
        
        Args:
            role: Role name
            
        Returns:
            List of permissions
        """
        ...
    
    async def grant_permission(
        self,
        user_id: UUID,
        resource: str,
        action: str,
        expiry: Optional[datetime] = None
    ) -> bool:
        """Grant permission to user.
        
        Args:
            user_id: User identifier
            resource: Resource identifier
            action: Action to grant
            expiry: Optional expiry time
            
        Returns:
            Success status
        """
        ...
    
    async def revoke_permission(
        self,
        user_id: UUID,
        resource: str,
        action: str
    ) -> bool:
        """Revoke permission from user.
        
        Args:
            user_id: User identifier
            resource: Resource identifier
            action: Action to revoke
            
        Returns:
            Success status
        """
        ...
    
    async def check_resource_access(
        self,
        user_id: UUID,
        resource_id: UUID,
        required_level: SecurityLevel
    ) -> bool:
        """Check resource access level.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            required_level: Required security level
            
        Returns:
            Access granted status
        """
        ...


@runtime_checkable
class ValidationServiceProtocol(Protocol):
    """Protocol for input validation and sanitization."""
    
    async def validate_input(
        self,
        data: Any,
        schema: Dict[str, Any],
        strict: bool = True
    ) -> Dict[str, Any]:
        """Validate input against schema.
        
        Args:
            data: Data to validate
            schema: Validation schema
            strict: Strict validation mode
            
        Returns:
            Validation results
        """
        ...
    
    async def sanitize_input(
        self,
        data: Any,
        sanitization_level: str = "standard"
    ) -> Any:
        """Sanitize input data.
        
        Args:
            data: Data to sanitize
            sanitization_level: Level of sanitization
            
        Returns:
            Sanitized data
        """
        ...
    
    async def check_sql_injection(
        self,
        query: str
    ) -> Dict[str, Any]:
        """Check for SQL injection attempts.
        
        Args:
            query: Query to check
            
        Returns:
            Detection results
        """
        ...
    
    async def check_xss(
        self,
        content: str
    ) -> Dict[str, Any]:
        """Check for XSS attempts.
        
        Args:
            content: Content to check
            
        Returns:
            Detection results
        """
        ...
    
    async def validate_file_upload(
        self,
        file_data: bytes,
        allowed_types: List[str],
        max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate file upload.
        
        Args:
            file_data: File data
            allowed_types: Allowed MIME types
            max_size: Maximum file size
            
        Returns:
            Validation results
        """
        ...
    
    async def validate_api_request(
        self,
        request_data: Dict[str, Any],
        endpoint: str
    ) -> Dict[str, Any]:
        """Validate API request.
        
        Args:
            request_data: Request data
            endpoint: API endpoint
            
        Returns:
            Validation results
        """
        ...


@runtime_checkable
class SecurityMonitoringServiceProtocol(Protocol):
    """Protocol for threat detection and audit logging."""
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[UUID],
        details: Dict[str, Any],
        threat_level: ThreatLevel = ThreatLevel.NONE
    ) -> bool:
        """Log security event.
        
        Args:
            event_type: Type of event
            user_id: Associated user
            details: Event details
            threat_level: Threat level
            
        Returns:
            Success status
        """
        ...
    
    async def detect_anomaly(
        self,
        activity: Dict[str, Any],
        user_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Detect anomalous activity.
        
        Args:
            activity: Activity data
            user_id: User identifier
            
        Returns:
            Anomaly details if detected
        """
        ...
    
    async def check_rate_limit(
        self,
        identifier: str,
        action: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Check rate limit.
        
        Args:
            identifier: Rate limit identifier
            action: Action being rate limited
            limit: Override default limit
            
        Returns:
            Rate limit status
        """
        ...
    
    async def get_threat_score(
        self,
        user_id: UUID,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get threat score for user.
        
        Args:
            user_id: User identifier
            time_window: Time window in minutes
            
        Returns:
            Threat score and details
        """
        ...
    
    async def get_audit_logs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit logs.
        
        Args:
            filters: Log filters
            limit: Result limit
            
        Returns:
            List of audit logs
        """
        ...
    
    async def trigger_security_alert(
        self,
        alert_type: str,
        severity: ThreatLevel,
        details: Dict[str, Any]
    ) -> bool:
        """Trigger security alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            details: Alert details
            
        Returns:
            Success status
        """
        ...
    
    async def get_security_metrics(
        self,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Get security metrics.
        
        Args:
            time_range: Time range for metrics
            
        Returns:
            Security metrics
        """
        ...


@runtime_checkable
class SecurityServiceFacadeProtocol(Protocol):
    """Protocol for unified SecurityServiceFacade."""
    
    @property
    def authentication(self) -> AuthenticationServiceProtocol:
        """Get authentication service."""
        ...
    
    @property
    def authorization(self) -> AuthorizationServiceProtocol:
        """Get authorization service."""
        ...
    
    @property
    def validation(self) -> ValidationServiceProtocol:
        """Get validation service."""
        ...
    
    @property
    def monitoring(self) -> SecurityMonitoringServiceProtocol:
        """Get security monitoring service."""
        ...
    
    async def secure_operation(
        self,
        operation: str,
        user_id: UUID,
        data: Dict[str, Any],
        required_level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> Dict[str, Any]:
        """Execute secure operation with full validation.
        
        Args:
            operation: Operation to execute
            user_id: User identifier
            data: Operation data
            required_level: Required security level
            
        Returns:
            Operation result
        """
        ...
    
    async def get_security_context(
        self,
        user_id: UUID
    ) -> Dict[str, Any]:
        """Get complete security context for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Security context
        """
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all security services.
        
        Returns:
            Health status of each service
        """
        ...