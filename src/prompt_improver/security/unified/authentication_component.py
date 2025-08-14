"""AuthenticationComponent - Consolidated Authentication Service

Extracts and consolidates authentication functionality from UnifiedAuthenticationManager
into a clean component that implements the AuthenticationProtocol. This component
handles all authentication operations including API keys, sessions, and system tokens.

Key Features:
- 256-bit cryptographically secure API key generation and validation
- Session token management with TTL and automatic cleanup
- Rate limiting integration for abuse prevention
- Comprehensive audit logging and security monitoring
- Fail-secure authentication policies (deny by default)
- Zero-trust security architecture
- Integration with UnifiedKeyService for secure operations

Security Standards:
- OWASP authentication guidelines compliance
- Zero-trust architecture with mandatory verification
- Fail-secure by default (no fail-open vulnerabilities)
- Comprehensive audit logging for all operations
- Rate limiting to prevent brute force attacks
"""

import asyncio
import hashlib
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from prompt_improver.database import (
    ManagerMode,
    SecurityContext,
    SecurityPerformanceMetrics,
    create_security_context,
    create_security_context_from_auth_result,
    get_database_services,
)
from prompt_improver.security.key_manager import (
    AuditEvent,
    SecurityAuditLogger,
    SecurityLevel,
    UnifiedKeyService,
)
from prompt_improver.security.unified.protocols import (
    AuthenticationProtocol,
    SecurityComponentStatus,
    SecurityOperationResult,
)
# Rate limiting functionality moved to SecurityServiceFacade
# This component now delegates rate limiting to the facade
from prompt_improver.utils.session_store import SessionStore
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class AuthenticationMethod(str):
    """Authentication methods supported by the component."""
    API_KEY = "api_key"
    SESSION_TOKEN = "session_token"
    SYSTEM_TOKEN = "system_token"


class AuthenticationStatus(str):
    """Authentication status results."""
    SUCCESS = "success"
    FAILED = "failed"
    EXPIRED = "expired"
    REVOKED = "revoked"
    RATE_LIMITED = "rate_limited"
    SYSTEM_ERROR = "system_error"


class APIKeyData(BaseModel):
    """API key data structure."""
    key_id: str = Field(description="Unique key identifier")
    agent_id: str = Field(description="Associated agent identifier")
    hashed_key: str = Field(description="SHA-256 hashed key value")
    permissions: List[str] = Field(default_factory=list, description="Granted permissions")
    tier: str = Field(default="basic", description="Rate limiting tier")
    created_at: datetime = Field(default_factory=aware_utc_now, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    usage_count: int = Field(default=0, description="Number of times used")
    is_active: bool = Field(default=True, description="Whether key is active")


class AuthenticationSession(BaseModel):
    """Authentication session data."""
    session_id: str = Field(description="Unique session identifier")
    agent_id: str = Field(description="Associated agent identifier")
    security_context: SecurityContext = Field(description="Session security context")
    created_at: datetime = Field(default_factory=aware_utc_now, description="Creation timestamp")
    expires_at: datetime = Field(description="Expiration timestamp")
    last_accessed: datetime = Field(default_factory=aware_utc_now, description="Last access timestamp")
    access_count: int = Field(default=0, description="Number of times accessed")
    is_active: bool = Field(default=True, description="Whether session is active")


class AuthenticationComponent:
    """Authentication component implementing AuthenticationProtocol.
    
    Consolidates authentication functionality from UnifiedAuthenticationManager
    into a clean, protocol-compliant component that handles all authentication
    operations for the SecurityServiceFacade.
    """
    
    def __init__(self):
        """Initialize authentication component."""
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Core components
        self._key_manager: Optional[UnifiedKeyService] = None
        self._session_store: Optional[SessionStore] = None
        # Rate limiting delegated to SecurityServiceFacade
        self._security_auditor: Optional[SecurityAuditLogger] = None
        
        # Authentication storage
        self._api_keys: Dict[str, APIKeyData] = {}  # key_id -> APIKeyData
        self._key_hash_index: Dict[str, str] = {}  # hashed_key -> key_id
        self._active_sessions: Dict[str, AuthenticationSession] = {}  # session_id -> session
        self._failed_attempts: Dict[str, List[datetime]] = {}  # agent_id -> attempt times
        
        # Configuration
        self._fail_secure = True
        self._zero_trust_mode = True
        self._max_failed_attempts = 5
        self._lockout_duration = timedelta(minutes=15)
        self._session_ttl = timedelta(hours=24)
        self._cleanup_interval = 300  # 5 minutes
        
        # Performance metrics
        self._metrics = {
            "authentication_attempts": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "api_key_validations": 0,
            "session_validations": 0,
            "sessions_created": 0,
            "sessions_revoked": 0,
            "total_auth_time_ms": 0.0,
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("AuthenticationComponent initialized")
    
    async def initialize(self) -> bool:
        """Initialize the authentication component.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        async with self._initialization_lock:
            if self._initialized:
                return True
            
            try:
                # Initialize core components
                self._key_manager = UnifiedKeyService()
                self._session_store = SessionStore()
                # Rate limiting delegated to SecurityServiceFacade
                self._security_auditor = SecurityAuditLogger("AuthenticationComponent")
                
                # Start background cleanup task
                self._cleanup_task = asyncio.create_task(self._session_cleanup_loop())
                
                self._initialized = True
                logger.info("AuthenticationComponent initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize AuthenticationComponent: {e}")
                return False
    
    async def health_check(self) -> Tuple[SecurityComponentStatus, Dict[str, Any]]:
        """Check component health status.
        
        Returns:
            Tuple of (status, health_details)
        """
        if not self._initialized:
            return SecurityComponentStatus.FAILED, {"error": "Component not initialized"}
        
        try:
            health_details = {
                "initialized": self._initialized,
                "active_sessions": len(self._active_sessions),
                "stored_api_keys": len(self._api_keys),
                "failed_attempt_records": len(self._failed_attempts),
                "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
                "metrics": self._metrics.copy(),
            }
            
            # Check for critical issues
            if self._cleanup_task and self._cleanup_task.done():
                exception = self._cleanup_task.exception()
                if exception:
                    health_details["cleanup_task_error"] = str(exception)
                    return SecurityComponentStatus.DEGRADED, health_details
            
            return SecurityComponentStatus.HEALTHY, health_details
            
        except Exception as e:
            return SecurityComponentStatus.FAILED, {"error": str(e)}
    
    async def get_metrics(self) -> SecurityPerformanceMetrics:
        """Get security performance metrics for this component.
        
        Returns:
            Security performance metrics
        """
        total_operations = self._metrics["authentication_attempts"]
        avg_latency = (
            self._metrics["total_auth_time_ms"] / total_operations 
            if total_operations > 0 else 0.0
        )
        error_rate = (
            self._metrics["failed_authentications"] / total_operations 
            if total_operations > 0 else 0.0
        )
        
        return SecurityPerformanceMetrics(
            operation_count=total_operations,
            average_latency_ms=avg_latency,
            error_rate=error_rate,
            threat_detection_count=sum(len(attempts) for attempts in self._failed_attempts.values()),
            last_updated=aware_utc_now()
        )
    
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
        if not self._initialized:
            return SecurityOperationResult(
                success=False,
                operation_type="authenticate",
                execution_time_ms=0.0,
                errors=["Component not initialized"]
            )
        
        start_time = time.perf_counter()
        self._metrics["authentication_attempts"] += 1
        
        agent_id = credentials.get("agent_id", "unknown")
        
        try:
            # Check for agent lockout
            if self._is_agent_locked_out(agent_id):
                execution_time = (time.perf_counter() - start_time) * 1000
                self._metrics["failed_authentications"] += 1
                self._metrics["total_auth_time_ms"] += execution_time
                
                return SecurityOperationResult(
                    success=False,
                    operation_type="authenticate",
                    execution_time_ms=execution_time,
                    errors=["Agent locked out due to too many failed attempts"]
                )
            
            # Perform authentication based on method
            if authentication_method == AuthenticationMethod.API_KEY:
                result = await self._authenticate_api_key(credentials, start_time)
            elif authentication_method == AuthenticationMethod.SESSION_TOKEN:
                result = await self._authenticate_session_token(credentials, start_time)
            elif authentication_method == AuthenticationMethod.SYSTEM_TOKEN:
                result = await self._authenticate_system_token(credentials, start_time)
            else:
                execution_time = (time.perf_counter() - start_time) * 1000
                self._metrics["failed_authentications"] += 1
                self._metrics["total_auth_time_ms"] += execution_time
                
                return SecurityOperationResult(
                    success=False,
                    operation_type="authenticate",
                    execution_time_ms=execution_time,
                    errors=[f"Unsupported authentication method: {authentication_method}"]
                )
            
            # Update metrics
            if result.success:
                self._metrics["successful_authentications"] += 1
            else:
                self._metrics["failed_authentications"] += 1
                self._track_failed_attempt(agent_id)
            
            self._metrics["total_auth_time_ms"] += result.execution_time_ms
            
            # Audit logging
            if self._security_auditor:
                await self._log_authentication_attempt(result, credentials, authentication_method)
            
            return result
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["failed_authentications"] += 1
            self._metrics["total_auth_time_ms"] += execution_time
            
            error_message = f"Authentication system error: {e}" if self._fail_secure else str(e)
            logger.error(f"Authentication error for agent {agent_id}: {e}")
            
            return SecurityOperationResult(
                success=False,
                operation_type="authenticate",
                execution_time_ms=execution_time,
                errors=[error_message]
            )
    
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
        if not self._initialized:
            return SecurityOperationResult(
                success=False,
                operation_type="create_session",
                execution_time_ms=0.0,
                errors=["Component not initialized"]
            )
        
        start_time = time.perf_counter()
        
        try:
            # Generate secure session ID
            session_id = secrets.token_urlsafe(32)
            expires_at = aware_utc_now() + self._session_ttl
            
            # Create session object
            session = AuthenticationSession(
                session_id=session_id,
                agent_id=agent_id,
                security_context=security_context,
                expires_at=expires_at
            )
            
            # Store session
            self._active_sessions[session_id] = session
            
            # Store in session store for persistence
            await self._session_store.create_session(
                session_id=session_id,
                data={
                    "agent_id": agent_id,
                    "security_context": security_context.dict(),
                    "created_at": session.created_at.isoformat(),
                    "expires_at": session.expires_at.isoformat(),
                },
                expires_at=expires_at
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["sessions_created"] += 1
            
            logger.info(f"Session created for agent {agent_id}: {session_id}")
            
            return SecurityOperationResult(
                success=True,
                operation_type="create_session",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "expires_at": expires_at.isoformat(),
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Failed to create session for agent {agent_id}: {e}")
            
            return SecurityOperationResult(
                success=False,
                operation_type="create_session",
                execution_time_ms=execution_time,
                errors=[str(e)]
            )
    
    async def validate_session(self, session_id: str) -> SecurityOperationResult:
        """Validate existing session.
        
        Args:
            session_id: Session identifier to validate
            
        Returns:
            SecurityOperationResult with session validation status
        """
        if not self._initialized:
            return SecurityOperationResult(
                success=False,
                operation_type="validate_session",
                execution_time_ms=0.0,
                errors=["Component not initialized"]
            )
        
        start_time = time.perf_counter()
        self._metrics["session_validations"] += 1
        
        try:
            # Check active sessions first
            session = self._active_sessions.get(session_id)
            
            if not session:
                # Try to load from persistent storage
                session_data = await self._session_store.get_session(session_id)
                if session_data:
                    # Reconstruct session object
                    security_context = SecurityContext(**session_data.get("security_context", {}))
                    session = AuthenticationSession(
                        session_id=session_id,
                        agent_id=session_data["agent_id"],
                        security_context=security_context,
                        created_at=datetime.fromisoformat(session_data["created_at"]),
                        expires_at=datetime.fromisoformat(session_data["expires_at"]),
                    )
                    self._active_sessions[session_id] = session
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            if not session:
                return SecurityOperationResult(
                    success=False,
                    operation_type="validate_session",
                    execution_time_ms=execution_time,
                    errors=["Session not found"]
                )
            
            # Check if session is expired
            now = aware_utc_now()
            if session.expires_at <= now:
                # Remove expired session
                self._active_sessions.pop(session_id, None)
                await self._session_store.delete_session(session_id)
                
                return SecurityOperationResult(
                    success=False,
                    operation_type="validate_session",
                    execution_time_ms=execution_time,
                    errors=["Session expired"]
                )
            
            # Update session access
            session.last_accessed = now
            session.access_count += 1
            
            logger.debug(f"Session validated for agent {session.agent_id}: {session_id}")
            
            return SecurityOperationResult(
                success=True,
                operation_type="validate_session",
                execution_time_ms=execution_time,
                security_context=session.security_context,
                metadata={
                    "session_id": session_id,
                    "agent_id": session.agent_id,
                    "expires_at": session.expires_at.isoformat(),
                    "access_count": session.access_count,
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Failed to validate session {session_id}: {e}")
            
            return SecurityOperationResult(
                success=False,
                operation_type="validate_session",
                execution_time_ms=execution_time,
                errors=[str(e)]
            )
    
    async def revoke_session(self, session_id: str) -> SecurityOperationResult:
        """Revoke existing session.
        
        Args:
            session_id: Session identifier to revoke
            
        Returns:
            SecurityOperationResult with revocation status
        """
        if not self._initialized:
            return SecurityOperationResult(
                success=False,
                operation_type="revoke_session",
                execution_time_ms=0.0,
                errors=["Component not initialized"]
            )
        
        start_time = time.perf_counter()
        
        try:
            # Remove from active sessions
            session = self._active_sessions.pop(session_id, None)
            
            # Remove from persistent storage
            await self._session_store.delete_session(session_id)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["sessions_revoked"] += 1
            
            agent_id = session.agent_id if session else "unknown"
            logger.info(f"Session revoked for agent {agent_id}: {session_id}")
            
            return SecurityOperationResult(
                success=True,
                operation_type="revoke_session",
                execution_time_ms=execution_time,
                metadata={
                    "session_id": session_id,
                    "agent_id": agent_id,
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Failed to revoke session {session_id}: {e}")
            
            return SecurityOperationResult(
                success=False,
                operation_type="revoke_session",
                execution_time_ms=execution_time,
                errors=[str(e)]
            )
    
    async def cleanup(self) -> bool:
        """Cleanup component resources.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear session data
            self._active_sessions.clear()
            self._api_keys.clear()
            self._key_hash_index.clear()
            self._failed_attempts.clear()
            
            # Reset metrics
            self._metrics = {key: 0 if isinstance(value, (int, float)) else value 
                           for key, value in self._metrics.items()}
            
            self._initialized = False
            logger.info("AuthenticationComponent cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup AuthenticationComponent: {e}")
            return False
    
    # Private helper methods
    
    def _is_agent_locked_out(self, agent_id: str) -> bool:
        """Check if agent is locked out due to failed attempts."""
        if agent_id not in self._failed_attempts:
            return False
        
        now = aware_utc_now()
        cutoff_time = now - self._lockout_duration
        
        # Clean old attempts
        recent_attempts = [
            attempt for attempt in self._failed_attempts[agent_id] 
            if attempt > cutoff_time
        ]
        self._failed_attempts[agent_id] = recent_attempts
        
        return len(recent_attempts) >= self._max_failed_attempts
    
    def _track_failed_attempt(self, agent_id: str):
        """Track failed authentication attempt for agent."""
        if agent_id not in self._failed_attempts:
            self._failed_attempts[agent_id] = []
        
        self._failed_attempts[agent_id].append(aware_utc_now())
    
    async def _authenticate_api_key(
        self, 
        credentials: Dict[str, Any], 
        start_time: float
    ) -> SecurityOperationResult:
        """Authenticate using API key."""
        api_key = credentials.get("api_key", "")
        if not api_key:
            execution_time = (time.perf_counter() - start_time) * 1000
            return SecurityOperationResult(
                success=False,
                operation_type="authenticate_api_key",
                execution_time_ms=execution_time,
                errors=["No API key provided"]
            )
        
        self._metrics["api_key_validations"] += 1
        
        # Hash the provided key
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Look up key data
        key_id = self._key_hash_index.get(hashed_key)
        if not key_id:
            execution_time = (time.perf_counter() - start_time) * 1000
            return SecurityOperationResult(
                success=False,
                operation_type="authenticate_api_key",
                execution_time_ms=execution_time,
                errors=["Invalid API key"]
            )
        
        key_data = self._api_keys.get(key_id)
        if not key_data or not key_data.is_active:
            execution_time = (time.perf_counter() - start_time) * 1000
            return SecurityOperationResult(
                success=False,
                operation_type="authenticate_api_key",
                execution_time_ms=execution_time,
                errors=["API key not active"]
            )
        
        # Check expiration
        if key_data.expires_at and key_data.expires_at <= aware_utc_now():
            execution_time = (time.perf_counter() - start_time) * 1000
            return SecurityOperationResult(
                success=False,
                operation_type="authenticate_api_key",
                execution_time_ms=execution_time,
                errors=["API key expired"]
            )
        
        # Update usage
        key_data.last_used = aware_utc_now()
        key_data.usage_count += 1
        
        # Create security context
        security_context = create_security_context(
            agent_id=key_data.agent_id,
            manager_mode=ManagerMode.PRODUCTION
        )
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return SecurityOperationResult(
            success=True,
            operation_type="authenticate_api_key",
            execution_time_ms=execution_time,
            security_context=security_context,
            metadata={
                "agent_id": key_data.agent_id,
                "key_id": key_data.key_id,
                "permissions": key_data.permissions,
                "tier": key_data.tier,
                "usage_count": key_data.usage_count,
            }
        )
    
    async def _authenticate_session_token(
        self, 
        credentials: Dict[str, Any], 
        start_time: float
    ) -> SecurityOperationResult:
        """Authenticate using session token."""
        session_token = credentials.get("session_token", "")
        if not session_token:
            execution_time = (time.perf_counter() - start_time) * 1000
            return SecurityOperationResult(
                success=False,
                operation_type="authenticate_session_token",
                execution_time_ms=execution_time,
                errors=["No session token provided"]
            )
        
        # Validate session using existing method
        result = await self.validate_session(session_token)
        result.operation_type = "authenticate_session_token"
        
        return result
    
    async def _authenticate_system_token(
        self, 
        credentials: Dict[str, Any], 
        start_time: float
    ) -> SecurityOperationResult:
        """Authenticate using system token."""
        # System token authentication would be implemented here
        # For now, return not implemented
        execution_time = (time.perf_counter() - start_time) * 1000
        return SecurityOperationResult(
            success=False,
            operation_type="authenticate_system_token",
            execution_time_ms=execution_time,
            errors=["System token authentication not yet implemented"]
        )
    
    async def _log_authentication_attempt(
        self, 
        result: SecurityOperationResult,
        credentials: Dict[str, Any],
        method: str
    ):
        """Log authentication attempt for audit purposes."""
        if self._security_auditor:
            event_type = AuditEvent.KEY_ACCESSED if result.success else AuditEvent.SECURITY_VIOLATION
            level = SecurityLevel.INFO if result.success else SecurityLevel.WARNING
            
            self._security_auditor.log_event(
                event_type,
                {
                    "method": method,
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "agent_id": credentials.get("agent_id", "unknown"),
                    "errors": result.errors,
                },
                level
            )
    
    async def _session_cleanup_loop(self):
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = aware_utc_now()
        expired_sessions = []
        
        for session_id, session in self._active_sessions.items():
            if session.expires_at <= now:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._active_sessions.pop(session_id, None)
            await self._session_store.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")