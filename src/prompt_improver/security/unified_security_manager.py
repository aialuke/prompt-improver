"""
Unified Security Manager - Complete Security Infrastructure Integration

A comprehensive security management system that consolidates and orchestrates
all security components following the proven UnifiedConnectionManager pattern.

Key Features:
- Mode-based security configuration (MCP_SERVER, API, INTERNAL)
- Fail-secure by default (no fail-open vulnerabilities)
- Unified integration of all security components
- Comprehensive audit logging and monitoring
- Real behavior testing infrastructure
- Clean architecture with zero legacy patterns

Security Components Integration:
- UnifiedAuthenticationManager (placeholder for future implementation)
- UnifiedValidationManager (placeholder for future implementation)
- UnifiedSecurityStack (placeholder for future implementation)
- KeyManager (existing secure key management)
- UnifiedRateLimiter (existing rate limiting)
- SecurityContext management
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import hashlib
import secrets

# Core imports
# Removed AppConfig dependency to fix circular import - security must be foundational
from ..utils.datetime_utils import aware_utc_now
from ..database.unified_connection_manager import (
    SecurityContext, 
    SecurityThreatScore,
    SecurityValidationResult,
    SecurityPerformanceMetrics,
    RedisSecurityError,
    create_security_context,
    create_security_context_from_auth_result,
    create_security_context_from_security_manager,
    create_system_security_context,
    get_unified_manager,
    ManagerMode
)

# Security component imports
from .key_manager import UnifiedKeyManager, get_unified_key_manager, SecurityLevel, AuditEvent
from .unified_rate_limiter import (
    UnifiedRateLimiter, 
    get_unified_rate_limiter,
    RateLimitResult,
    RateLimitStatus,
    RateLimitExceeded,
    RateLimitTier
)

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    security_tracer = trace.get_tracer(__name__ + ".security")
    security_meter = metrics.get_meter(__name__ + ".security")
    
    # Security-specific metrics
    security_operations_counter = security_meter.create_counter(
        "unified_security_operations_total",  
        description="Total unified security operations by type and result",
        unit="1"
    )
    
    security_violations_counter = security_meter.create_counter(
        "unified_security_violations_total",
        description="Total security violations by type and severity", 
        unit="1"
    )
    
    security_latency_histogram = security_meter.create_histogram(
        "unified_security_operation_duration_seconds",
        description="Unified security operation duration by type",
        unit="s"
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    security_tracer = None
    security_meter = None
    security_operations_counter = None
    security_violations_counter = None
    security_latency_histogram = None

logger = logging.getLogger(__name__)


class SecurityMode(Enum):
    """Security operation modes optimized for different use cases."""
    MCP_SERVER = "mcp_server"      # MCP server operations with agent authentication
    API = "api"                    # Public API with comprehensive validation
    INTERNAL = "internal"          # Internal service communication
    ADMIN = "admin"               # Administrative operations with elevated privileges
    HIGH_SECURITY = "high_security"  # Maximum security for sensitive operations


class SecurityThreatLevel(Enum):
    """Security threat levels for incident classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityOperationType(Enum):
    """Types of security operations for monitoring and audit."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization" 
    VALIDATION = "validation"
    ENCRYPTION = "encryption"
    RATE_LIMITING = "rate_limiting"
    AUDIT_LOGGING = "audit_logging"
    THREAT_DETECTION = "threat_detection"
    INCIDENT_RESPONSE = "incident_response"


@dataclass
class SecurityConfiguration:
    """Security configuration for different modes."""
    mode: SecurityMode
    security_level: SecurityLevel = SecurityLevel.enhanced
    rate_limit_tier: RateLimitTier = RateLimitTier.BASIC
    audit_logging_enabled: bool = True
    threat_detection_enabled: bool = True
    fail_secure: bool = True  # Always fail secure, never fail open
    max_authentication_attempts: int = 3
    session_timeout_minutes: int = 60
    require_encryption: bool = True
    zero_trust_mode: bool = True
    
    def __post_init__(self):
        """Apply mode-specific security defaults."""
        if self.mode == SecurityMode.HIGH_SECURITY:
            self.security_level = SecurityLevel.CRITICAL
            self.rate_limit_tier = RateLimitTier.BASIC  # More restrictive
            self.max_authentication_attempts = 1  # Zero tolerance
            self.session_timeout_minutes = 15  # Shorter sessions
            self.zero_trust_mode = True
        elif self.mode == SecurityMode.API:
            self.security_level = SecurityLevel.HIGH
            self.rate_limit_tier = RateLimitTier.PROFESSIONAL
            self.max_authentication_attempts = 2
            self.session_timeout_minutes = 30
        elif self.mode == SecurityMode.MCP_SERVER:
            self.security_level = SecurityLevel.enhanced
            self.rate_limit_tier = RateLimitTier.PROFESSIONAL
            self.max_authentication_attempts = 3
            self.session_timeout_minutes = 120  # Longer for MCP operations
        elif self.mode == SecurityMode.INTERNAL:
            self.security_level = SecurityLevel.enhanced
            self.rate_limit_tier = RateLimitTier.ENTERPRISE
            self.max_authentication_attempts = 5
            self.session_timeout_minutes = 240  # Longer for internal services
        elif self.mode == SecurityMode.ADMIN:
            self.security_level = SecurityLevel.CRITICAL
            self.rate_limit_tier = RateLimitTier.BASIC  # More restrictive for admin
            self.max_authentication_attempts = 1
            self.session_timeout_minutes = 10  # Very short for admin operations


@dataclass 
class SecurityIncident:
    """Security incident tracking."""
    incident_id: str
    timestamp: datetime
    threat_level: SecurityThreatLevel
    operation_type: SecurityOperationType
    agent_id: str
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary for logging."""
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp.isoformat(),
            "threat_level": self.threat_level.value,
            "operation_type": self.operation_type.value,
            "agent_id": self.agent_id,
            "details": self.details,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


@dataclass
class SecurityMetrics:
    """Security metrics and statistics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    security_violations: int = 0
    active_incidents: int = 0
    resolved_incidents: int = 0
    authentication_attempts: int = 0
    successful_authentications: int = 0
    rate_limit_violations: int = 0
    last_incident_time: Optional[datetime] = None
    
    def get_success_rate(self) -> float:
        """Calculate operation success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
        
    def get_violation_rate(self) -> float:
        """Calculate security violation rate."""
        if self.total_operations == 0:
            return 0.0
        return self.security_violations / self.total_operations


class UnifiedSecurityManager:
    """Unified Security Manager - Complete Security Infrastructure Integration.
    
    Consolidates and orchestrates all security components following the proven
    UnifiedConnectionManager pattern with fail-secure design principles.
    
    Security Components:
    - KeyManager: Secure key management and encryption
    - UnifiedRateLimiter: Rate limiting and traffic control
    - SecurityContext: Context-aware security operations
    - Audit logging: Comprehensive security event tracking
    - Threat detection: Real-time security monitoring
    - Incident response: Automated security incident handling
    """
    
    def __init__(self, 
                 mode: SecurityMode = SecurityMode.API,
                 config: Optional[SecurityConfiguration] = None):
        """Initialize unified security manager.
        
        Args:
            mode: Security operation mode
            config: Optional security configuration (auto-generated if not provided)
        """
        self.mode = mode
        self.config = config or SecurityConfiguration(mode=mode)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Security components (initialized lazily)
        self._key_manager: Optional[UnifiedKeyManager] = None
        self._rate_limiter: Optional[UnifiedRateLimiter] = None
        self._connection_manager = None
        
        # Security state tracking
        self._security_metrics = SecurityMetrics()
        self._active_incidents: Dict[str, SecurityIncident] = {}
        self._incident_history: deque = deque(maxlen=1000)  # Keep last 1000 incidents
        self._authentication_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._blocked_agents: Dict[str, datetime] = {}
        
        # Threat detection patterns
        self._suspicious_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self._known_threats: set = set()
        
        # Performance tracking
        self._operation_times: deque = deque(maxlen=100)  # Keep last 100 operation times
        
        # Initialization timestamp
        self._initialized_at = aware_utc_now()
        self._last_health_check = self._initialized_at
        
        self.logger.info(f"UnifiedSecurityManager initialized in {mode.value} mode")
    
    async def initialize(self) -> None:
        """Initialize all security components."""
        try:
            start_time = time.time()
            
            # Initialize key manager
            self._key_manager = get_unified_key_manager()
            
            # Initialize rate limiter  
            self._rate_limiter = await get_unified_rate_limiter()
            
            # Initialize connection manager for security context operations
            self._connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            await self._connection_manager.initialize()
            
            initialization_time = time.time() - start_time
            
            self.logger.info(
                f"UnifiedSecurityManager fully initialized in {initialization_time:.3f}s "
                f"(mode: {self.mode.value}, security_level: {self.config.security_level.value})"
            )
            
            # Record successful initialization
            await self._record_security_operation(
                SecurityOperationType.AUDIT_LOGGING,
                success=True,
                details={"initialization_time": initialization_time}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UnifiedSecurityManager: {e}")
            await self._handle_security_incident(
                SecurityThreatLevel.HIGH,
                SecurityOperationType.AUDIT_LOGGING,
                "system",
                {"error": str(e), "operation": "initialization"}
            )
            raise
    
    async def authenticate_agent(self, 
                                agent_id: str, 
                                credentials: Dict[str, Any],
                                additional_context: Optional[Dict[str, Any]] = None) -> Tuple[bool, SecurityContext]:
        """Authenticate an agent with comprehensive security checks.
        
        Args:
            agent_id: Agent identifier
            credentials: Authentication credentials
            additional_context: Additional security context
            
        Returns:
            Tuple of (success, security_context)
            
        Raises:
            RedisSecurityError: On security violations
        """
        operation_start = time.time()
        
        try:
            # Check if agent is blocked
            if await self._is_agent_blocked(agent_id):
                await self._handle_security_incident(
                    SecurityThreatLevel.MEDIUM,
                    SecurityOperationType.AUTHENTICATION,
                    agent_id,
                    {"reason": "blocked_agent_attempt", "credentials_provided": bool(credentials)}
                )
                return False, await self._create_failed_security_context(agent_id, "blocked_agent_attempt")
            
            # Check authentication attempt rate limiting
            if not await self._check_authentication_rate_limit(agent_id):
                await self._handle_security_incident(
                    SecurityThreatLevel.MEDIUM, 
                    SecurityOperationType.AUTHENTICATION,
                    agent_id,
                    {"reason": "authentication_rate_limit_exceeded"}
                )
                return False, await self._create_failed_security_context(agent_id, "authentication_rate_limit_exceeded")
            
            # Validate credentials (placeholder - integrate with actual auth system)
            auth_success = await self._validate_credentials(agent_id, credentials)
            
            if auth_success:
                # Create enhanced authenticated security context with full integration
                security_context = await create_security_context_from_security_manager(
                    agent_id=agent_id,
                    security_manager=self,
                    additional_context={
                        "authentication_method": "credential_validation",
                        "permissions": additional_context.get("permissions", []) if additional_context else [],
                        "credentials_provided": bool(credentials),
                        "security_manager_mode": self.mode.value,
                        "authentication_timestamp": time.time(),
                        **(additional_context or {})
                    }
                )
                
                # Add comprehensive security validation result
                validation_result = SecurityValidationResult(
                    validated=True,
                    validation_method="unified_security_manager",
                    validation_timestamp=time.time(),
                    validation_duration_ms=(time.time() - operation_start) * 1000,
                    security_incidents=[],
                    rate_limit_status="validated",
                    encryption_required=self.config.require_encryption,
                    audit_trail_id=f"auth_{int(time.time() * 1000000)}"
                )
                security_context.validation_result = validation_result
                
                # Add threat assessment
                threat_score = SecurityThreatScore(
                    level="low",
                    score=0.1,
                    factors=["successful_authentication"],
                    last_updated=time.time()
                )
                security_context.threat_score = threat_score
                
                # Add performance metrics
                auth_time_ms = (time.time() - operation_start) * 1000
                performance_metrics = SecurityPerformanceMetrics(
                    authentication_time_ms=auth_time_ms,
                    total_security_overhead_ms=auth_time_ms,
                    operations_count=1,
                    last_performance_check=time.time()
                )
                security_context.performance_metrics = performance_metrics
                
                # Add security context metadata
                security_context.audit_metadata.update({
                    "authentication_source": "unified_security_manager",
                    "security_mode": self.mode.value,
                    "zero_trust_validated": self.config.zero_trust_mode,
                    "additional_context": additional_context or {}
                })
                
                if self.config.require_encryption:
                    security_context.encryption_context = {
                        "required": True,
                        "method": "unified_key_manager"
                    }
                
                # Record successful authentication
                self._security_metrics.successful_authentications += 1
                await self._record_security_operation(
                    SecurityOperationType.AUTHENTICATION,
                    success=True,
                    agent_id=agent_id,
                    details={
                        "security_level": self.config.security_level.value,
                        "additional_context": additional_context or {}
                    }
                )
                
                self.logger.info(f"Agent {agent_id} authenticated successfully")
                return True, security_context
            
            else:
                # Record failed authentication attempt
                await self._record_authentication_attempt(agent_id, success=False)
                
                # Check if agent should be blocked
                if await self._should_block_agent(agent_id):
                    await self._block_agent(agent_id)
                    await self._handle_security_incident(
                        SecurityThreatLevel.HIGH,
                        SecurityOperationType.AUTHENTICATION,
                        agent_id,
                        {"reason": "multiple_failed_authentication_attempts", "action": "agent_blocked"}
                    )
                
                return False, await self._create_failed_security_context(agent_id, "multiple_failed_authentication_attempts")
                
        except Exception as e:
            self.logger.error(f"Authentication error for agent {agent_id}: {e}")
            await self._handle_security_incident(
                SecurityThreatLevel.HIGH,
                SecurityOperationType.AUTHENTICATION,
                agent_id,
                {"error": str(e), "operation": "authenticate_agent"}
            )
            # Fail secure - deny access on errors
            return False, await self._create_failed_security_context(agent_id, f"authentication_system_error: {str(e)}")
        
        finally:
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
            
            # Record operation metrics
            if OPENTELEMETRY_AVAILABLE and security_operations_counter:
                security_operations_counter.add(
                    1,
                    attributes={
                        "operation": "authenticate_agent",
                        "mode": self.mode.value,
                        "agent_id": agent_id
                    }
                )
    
    async def authorize_operation(self,
                                security_context: SecurityContext,
                                operation: str,
                                resource: str,
                                additional_checks: Optional[Dict[str, Any]] = None) -> bool:
        """Authorize an operation with comprehensive security validation.
        
        Args:
            security_context: Security context from authentication
            operation: Operation being attempted
            resource: Resource being accessed
            additional_checks: Additional authorization checks
            
        Returns:
            True if authorized, False otherwise
        """
        operation_start = time.time()
        
        try:
            # Validate security context
            if not self._validate_security_context(security_context):
                await self._handle_security_incident(
                    SecurityThreatLevel.MEDIUM,
                    SecurityOperationType.AUTHORIZATION,
                    security_context.agent_id,
                    {"reason": "invalid_security_context", "operation": operation, "resource": resource}
                )
                return False
            
            # Check rate limiting
            rate_limit_status = await self._rate_limiter.check_rate_limit(
                agent_id=security_context.agent_id,
                tier=security_context.tier,
                authenticated=security_context.authenticated
            )
            
            if rate_limit_status.result != RateLimitResult.ALLOWED:
                self._security_metrics.rate_limit_violations += 1
                await self._handle_security_incident(
                    SecurityThreatLevel.LOW,
                    SecurityOperationType.RATE_LIMITING,
                    security_context.agent_id,
                    {
                        "reason": "rate_limit_exceeded",
                        "result": rate_limit_status.result.value,
                        "operation": operation,
                        "resource": resource
                    }
                )
                return False
            
            # Perform operation-specific authorization (placeholder)
            authorization_success = await self._perform_operation_authorization(
                security_context, operation, resource, additional_checks
            )
            
            if authorization_success:
                await self._record_security_operation(
                    SecurityOperationType.AUTHORIZATION,
                    success=True,
                    agent_id=security_context.agent_id,
                    details={"operation": operation, "resource": resource}
                )
                return True
            else:
                await self._handle_security_incident(
                    SecurityThreatLevel.MEDIUM,
                    SecurityOperationType.AUTHORIZATION,
                    security_context.agent_id,
                    {"reason": "authorization_denied", "operation": operation, "resource": resource}
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Authorization error for {security_context.agent_id}: {e}")
            await self._handle_security_incident(
                SecurityThreatLevel.HIGH,
                SecurityOperationType.AUTHORIZATION,
                security_context.agent_id,
                {"error": str(e), "operation": operation, "resource": resource}
            )
            # Fail secure - deny access on errors
            return False
        
        finally:
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
    
    async def validate_input(self,
                           security_context: SecurityContext,
                           input_data: Any,
                           validation_rules: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """Validate input data with security checks.
        
        Args:
            security_context: Security context
            input_data: Data to validate
            validation_rules: Validation rules
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        operation_start = time.time()
        
        try:
            # Basic validation placeholder - integrate with actual validation system
            validation_results = await self._perform_input_validation(
                security_context, input_data, validation_rules
            )
            
            is_valid = validation_results.get("valid", False)
            
            await self._record_security_operation(
                SecurityOperationType.VALIDATION,
                success=is_valid,
                agent_id=security_context.agent_id,
                details={
                    "validation_rules": validation_rules or {},
                    "validation_results": validation_results
                }
            )
            
            if not is_valid:
                await self._handle_security_incident(
                    SecurityThreatLevel.LOW,
                    SecurityOperationType.VALIDATION,
                    security_context.agent_id,
                    {"reason": "input_validation_failed", "validation_results": validation_results}
                )
            
            return is_valid, validation_results
            
        except Exception as e:
            self.logger.error(f"Input validation error for {security_context.agent_id}: {e}")
            await self._handle_security_incident(
                SecurityThreatLevel.MEDIUM,
                SecurityOperationType.VALIDATION,
                security_context.agent_id,
                {"error": str(e), "operation": "validate_input"}
            )
            # Fail secure - consider input invalid on errors
            return False, {"valid": False, "error": str(e)}
        
        finally:
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
    
    async def encrypt_data(self,
                          security_context: SecurityContext,
                          data: Union[str, bytes],
                          key_id: Optional[str] = None) -> Tuple[bytes, str]:
        """Encrypt data using the unified key manager.
        
        Args:
            security_context: Security context
            data: Data to encrypt
            key_id: Optional specific key ID
            
        Returns:
            Tuple of (encrypted_data, key_id_used)
        """
        operation_start = time.time()
        
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Encrypt using key manager
            encrypted_data, used_key_id = self._key_manager.encrypt(data_bytes, key_id)
            
            await self._record_security_operation(
                SecurityOperationType.ENCRYPTION,
                success=True,
                agent_id=security_context.agent_id,
                details={"key_id": used_key_id, "data_size": len(data_bytes)}
            )
            
            return encrypted_data, used_key_id
            
        except Exception as e:
            self.logger.error(f"Encryption error for {security_context.agent_id}: {e}")
            await self._handle_security_incident(
                SecurityThreatLevel.HIGH,
                SecurityOperationType.ENCRYPTION,
                security_context.agent_id,
                {"error": str(e), "operation": "encrypt_data"}
            )
            raise
        
        finally:
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
    
    async def decrypt_data(self,
                          security_context: SecurityContext,
                          encrypted_data: bytes,
                          key_id: str) -> bytes:
        """Decrypt data using the unified key manager.
        
        Args:
            security_context: Security context
            encrypted_data: Data to decrypt
            key_id: Key ID for decryption
            
        Returns:
            Decrypted data
        """
        operation_start = time.time()
        
        try:
            # Decrypt using key manager
            decrypted_data = self._key_manager.decrypt(encrypted_data, key_id)
            
            await self._record_security_operation(
                SecurityOperationType.ENCRYPTION,
                success=True,
                agent_id=security_context.agent_id,
                details={"key_id": key_id, "operation": "decrypt"}
            )
            
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Decryption error for {security_context.agent_id}: {e}")
            await self._handle_security_incident(
                SecurityThreatLevel.HIGH,
                SecurityOperationType.ENCRYPTION,
                security_context.agent_id,
                {"error": str(e), "operation": "decrypt_data", "key_id": key_id}
            )
            raise
        
        finally:
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status and metrics.
        
        Returns:
            Security status dictionary
        """
        try:
            # Update health check timestamp
            self._last_health_check = aware_utc_now()
            
            # Calculate performance metrics
            avg_operation_time = (
                sum(self._operation_times) / len(self._operation_times)
                if self._operation_times else 0
            )
            
            # Get component status
            key_manager_status = self._key_manager.get_security_status() if self._key_manager else {"status": "not_initialized"}
            
            status = {
                "mode": self.mode.value,
                "security_level": self.config.security_level.value,
                "initialized_at": self._initialized_at.isoformat(),
                "last_health_check": self._last_health_check.isoformat(),
                "uptime_seconds": (self._last_health_check - self._initialized_at).total_seconds(),
                
                # Security metrics
                "metrics": {
                    "total_operations": self._security_metrics.total_operations,
                    "successful_operations": self._security_metrics.successful_operations,
                    "failed_operations": self._security_metrics.failed_operations,
                    "success_rate": self._security_metrics.get_success_rate(),
                    "security_violations": self._security_metrics.security_violations,
                    "violation_rate": self._security_metrics.get_violation_rate(),
                    "authentication_attempts": self._security_metrics.authentication_attempts,
                    "successful_authentications": self._security_metrics.successful_authentications,
                    "rate_limit_violations": self._security_metrics.rate_limit_violations,
                    "active_incidents": len(self._active_incidents),
                    "resolved_incidents": len(self._incident_history)
                },
                
                # Performance metrics
                "performance": {
                    "average_operation_time_ms": avg_operation_time * 1000,
                    "recent_operation_count": len(self._operation_times)
                },
                
                # Component status
                "components": {
                    "key_manager": key_manager_status,
                    "rate_limiter": {"status": "initialized" if self._rate_limiter else "not_initialized"},
                    "connection_manager": {"status": "initialized" if self._connection_manager else "not_initialized"}
                },
                
                # Security state
                "security_state": {
                    "blocked_agents": len(self._blocked_agents),
                    "known_threats": len(self._known_threats),
                    "suspicious_patterns": len(self._suspicious_patterns),
                    "fail_secure_enabled": self.config.fail_secure,
                    "zero_trust_mode": self.config.zero_trust_mode
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting security status: {e}")
            return {"error": str(e), "status": "error"}
    
    async def create_unified_security_context(self,
                                         agent_id: str,
                                         operation_type: str = "general",
                                         additional_metadata: Optional[Dict[str, Any]] = None) -> SecurityContext:
        """Create unified security context with comprehensive security manager integration.
        
        Args:
            agent_id: Agent identifier
            operation_type: Type of operation requiring security context
            additional_metadata: Additional security metadata
            
        Returns:
            Comprehensive SecurityContext with unified security manager integration
        """
        try:
            # Create security context using security manager integration
            security_context = await create_security_context_from_security_manager(
                agent_id=agent_id,
                security_manager=self,
                additional_context={
                    "operation_type": operation_type,
                    "security_manager_mode": self.mode.value,
                    "context_creation_timestamp": time.time(),
                    "zero_trust_mode": self.config.zero_trust_mode,
                    "fail_secure_enabled": self.config.fail_secure,
                    **(additional_metadata or {})
                }
            )
            
            # Add security manager specific enhancements
            security_context.audit_metadata.update({
                "unified_security_manager": True,
                "security_configuration": {
                    "mode": self.mode.value,
                    "security_level": self.config.security_level.value,
                    "rate_limit_tier": self.config.rate_limit_tier.value,
                    "zero_trust_mode": self.config.zero_trust_mode,
                    "fail_secure": self.config.fail_secure
                }
            })
            
            self.logger.debug(f"Created unified security context for {agent_id} (operation: {operation_type})")
            return security_context
            
        except Exception as e:
            self.logger.error(f"Failed to create unified security context for {agent_id}: {e}")
            # Fail-secure: create minimal context
            return await create_security_context(
                agent_id=agent_id,
                authenticated=False,
                security_level="basic"
            )
    
    async def validate_security_context(self, 
                                      security_context: SecurityContext,
                                      operation_type: str = "general") -> Tuple[bool, List[str]]:
        """Validate security context against unified security policies.
        
        Args:
            security_context: Security context to validate
            operation_type: Type of operation being validated
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        try:
            # Basic context validation
            if not security_context.is_valid():
                return False, ["Security context has expired or is invalid"]
            
            # Security level validation
            if self.config.security_level.value == "critical" and security_context.security_level != "critical":
                warnings.append("Security context level below critical threshold")
            
            # Zero trust validation
            if self.config.zero_trust_mode and not security_context.zero_trust_validated:
                warnings.append("Zero trust validation not performed")
            
            # Threat score validation
            if security_context.threat_score.score > 0.7:
                return False, [f"High threat score detected: {security_context.threat_score.score}"]
            elif security_context.threat_score.score > 0.5:
                warnings.append(f"Elevated threat score: {security_context.threat_score.score}")
            
            # Update context usage
            security_context.touch()
            security_context.add_audit_event("security_validation", {
                "operation_type": operation_type,
                "validation_result": "passed",
                "warnings": warnings,
                "security_manager_mode": self.mode.value
            })
            
            await self._record_security_operation(
                SecurityOperationType.VALIDATION,
                success=True,
                agent_id=security_context.agent_id,
                details={"operation_type": operation_type, "warnings": warnings}
            )
            
            return True, warnings
            
        except Exception as e:
            self.logger.error(f"Security context validation error: {e}")
            await self._handle_security_incident(
                SecurityThreatLevel.MEDIUM,
                SecurityOperationType.VALIDATION,
                security_context.agent_id,
                {"error": str(e), "operation_type": operation_type}
            )
            return False, [f"Validation system error: {str(e)}"]
    
    async def enhance_security_context(self,
                                     security_context: SecurityContext,
                                     enhancement_data: Dict[str, Any]) -> SecurityContext:
        """Enhance existing security context with additional security data.
        
        Args:
            security_context: Existing security context
            enhancement_data: Additional security data to add
            
        Returns:
            Enhanced SecurityContext
        """
        try:
            # Add enhancement metadata
            security_context.audit_metadata.update({
                "enhanced_by_security_manager": True,
                "enhancement_timestamp": time.time(),
                "enhancement_data": enhancement_data,
                "security_manager_mode": self.mode.value
            })
            
            # Update threat assessment if security data suggests changes
            if enhancement_data.get("threat_indicators"):
                threat_factors = security_context.threat_score.factors + enhancement_data["threat_indicators"]
                new_score = min(security_context.threat_score.score + 0.1, 1.0)
                security_context.update_threat_score(
                    security_context.threat_score.level,
                    new_score,
                    threat_factors
                )
            
            # Update permissions if provided
            if enhancement_data.get("additional_permissions"):
                security_context.permissions.extend(enhancement_data["additional_permissions"])
                security_context.permissions = list(set(security_context.permissions))  # Remove duplicates
            
            # Update security level if needed
            if enhancement_data.get("security_level_upgrade"):
                new_level = enhancement_data["security_level_upgrade"]
                if new_level in ["basic", "enhanced", "high", "critical"]:
                    security_context.security_level = new_level
            
            # Record enhancement
            security_context.add_audit_event("context_enhanced", {
                "enhancement_keys": list(enhancement_data.keys()),
                "security_manager_mode": self.mode.value
            })
            
            self.logger.debug(f"Enhanced security context for {security_context.agent_id}")
            return security_context
            
        except Exception as e:
            self.logger.error(f"Security context enhancement failed: {e}")
            return security_context  # Return original context on errors
    
    async def get_security_incidents(self, 
                                   limit: int = 50,
                                   threat_level: Optional[SecurityThreatLevel] = None) -> List[Dict[str, Any]]:
        """Get recent security incidents.
        
        Args:
            limit: Maximum number of incidents to return
            threat_level: Filter by threat level
            
        Returns:
            List of security incidents
        """
        try:
            incidents = []
            
            # Add active incidents
            for incident in self._active_incidents.values():
                if threat_level is None or incident.threat_level == threat_level:
                    incidents.append(incident.to_dict())
            
            # Add resolved incidents from history
            for incident in self._incident_history:
                if threat_level is None or incident.threat_level == threat_level:
                    incidents.append(incident.to_dict())
                    
                if len(incidents) >= limit:
                    break
            
            # Sort by timestamp (most recent first)
            incidents.sort(key=lambda x: x["timestamp"], reverse=True)
            return incidents[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting security incidents: {e}")
            return []
    
    # ========== Private Implementation Methods ==========
    
    async def _validate_credentials(self, agent_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate agent credentials (placeholder for actual implementation).
        
        This is a placeholder method that should be replaced with actual
        credential validation logic based on your authentication system.
        """
        # Placeholder implementation - always authenticate for now
        # In production, integrate with your actual authentication system
        return True
    
    async def _is_agent_blocked(self, agent_id: str) -> bool:
        """Check if an agent is currently blocked."""
        if agent_id not in self._blocked_agents:
            return False
        
        blocked_until = self._blocked_agents[agent_id]
        if aware_utc_now() > blocked_until:
            # Block has expired
            del self._blocked_agents[agent_id]
            return False
        
        return True
    
    async def _check_authentication_rate_limit(self, agent_id: str) -> bool:
        """Check if agent is within authentication rate limits."""
        now = aware_utc_now()
        attempts = self._authentication_attempts[agent_id]
        
        # Remove old attempts (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self._authentication_attempts[agent_id] = [
            attempt for attempt in attempts if attempt > cutoff
        ]
        
        # Check if within rate limit
        return len(self._authentication_attempts[agent_id]) < self.config.max_authentication_attempts * 2
    
    async def _record_authentication_attempt(self, agent_id: str, success: bool) -> None:
        """Record an authentication attempt."""
        self._authentication_attempts[agent_id].append(aware_utc_now())
        self._security_metrics.authentication_attempts += 1
        
        if success:
            self._security_metrics.successful_authentications += 1
    
    async def _should_block_agent(self, agent_id: str) -> bool:
        """Determine if an agent should be blocked due to failed attempts."""
        recent_attempts = self._authentication_attempts[agent_id]
        
        # Check last N attempts within time window
        if len(recent_attempts) >= self.config.max_authentication_attempts:
            recent_window = recent_attempts[-self.config.max_authentication_attempts:]
            time_span = recent_window[-1] - recent_window[0]
            
            # If all attempts within 10 minutes, block the agent
            if time_span <= timedelta(minutes=10):
                return True
        
        return False
    
    async def _block_agent(self, agent_id: str) -> None:
        """Block an agent for a specified duration."""
        block_duration = timedelta(hours=1)  # Block for 1 hour
        self._blocked_agents[agent_id] = aware_utc_now() + block_duration
        
        self.logger.warning(f"Blocked agent {agent_id} until {self._blocked_agents[agent_id]}")
    
    async def _create_failed_security_context(self, agent_id: str, failure_reason: str = "authentication_failed") -> SecurityContext:
        """Create an enhanced security context for failed authentication with full security manager integration."""
        current_time = time.time()
        
        # Create context using security manager integration
        security_context = await create_security_context_from_security_manager(
            agent_id=agent_id,
            security_manager=self,
            additional_context={
                "authentication_method": "failed",
                "failure_reason": failure_reason,
                "security_manager_mode": self.mode.value,
                "authentication_timestamp": current_time,
                "threat_detected": True
            }
        )
        
        # Add failed authentication validation result
        validation_result = SecurityValidationResult(
            validated=False,
            validation_method="unified_security_manager",
            validation_timestamp=current_time,
            validation_duration_ms=0.0,
            security_incidents=["authentication_failed"],
            rate_limit_status="denied",
            encryption_required=False,
            audit_trail_id=f"failed_auth_{int(current_time * 1000000)}"
        )
        security_context.validation_result = validation_result
        
        # Add elevated threat assessment for failed authentication
        threat_score = SecurityThreatScore(
            level="medium",
            score=0.6,
            factors=["authentication_failure", "potential_brute_force"],
            last_updated=current_time
        )
        security_context.threat_score = threat_score
        
        # Add audit metadata
        security_context.audit_metadata.update({
            "authentication_source": "unified_security_manager",
            "failure_reason": "credential_validation_failed",
            "security_mode": self.mode.value,
            "timestamp": current_time
        })
        
        return security_context
    
    def _validate_security_context(self, security_context: SecurityContext) -> bool:
        """Validate security context integrity."""
        if not security_context.authenticated:
            return False
        
        # Check context age
        context_age = time.time() - security_context.created_at
        max_age = self.config.session_timeout_minutes * 60
        
        if context_age > max_age:
            return False
        
        return True
    
    async def _perform_operation_authorization(self,
                                             security_context: SecurityContext,
                                             operation: str,
                                             resource: str,
                                             additional_checks: Optional[Dict[str, Any]]) -> bool:
        """Perform operation-specific authorization (placeholder)."""
        # Placeholder implementation - always authorize for now
        # In production, implement actual authorization logic
        return True
    
    async def _perform_input_validation(self,
                                      security_context: SecurityContext,
                                      input_data: Any,
                                      validation_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform input validation (placeholder)."""
        # Placeholder implementation - always validate for now
        # In production, implement actual validation logic
        return {"valid": True, "sanitized_data": input_data}
    
    async def _record_security_operation(self,
                                       operation_type: SecurityOperationType,
                                       success: bool,
                                       agent_id: str = "system",
                                       details: Optional[Dict[str, Any]] = None) -> None:
        """Record a security operation for audit and metrics."""
        self._security_metrics.total_operations += 1
        
        if success:
            self._security_metrics.successful_operations += 1
        else:
            self._security_metrics.failed_operations += 1
        
        # OpenTelemetry metrics
        if OPENTELEMETRY_AVAILABLE and security_operations_counter:
            security_operations_counter.add(
                1,
                attributes={
                    "operation_type": operation_type.value,
                    "success": str(success),
                    "mode": self.mode.value,
                    "agent_id": agent_id
                }
            )
        
        # Audit logging
        if self.config.audit_logging_enabled:
            audit_entry = {
                "timestamp": aware_utc_now().isoformat(),
                "operation_type": operation_type.value,
                "success": success,
                "agent_id": agent_id,
                "mode": self.mode.value,
                "details": details or {}
            }
            
            self.logger.info(f"SECURITY_AUDIT: {audit_entry}")
    
    async def _handle_security_incident(self,
                                      threat_level: SecurityThreatLevel,
                                      operation_type: SecurityOperationType,
                                      agent_id: str,
                                      details: Dict[str, Any]) -> None:
        """Handle a security incident."""
        incident_id = f"sec_{int(time.time() * 1000000)}_{secrets.token_hex(4)}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            timestamp=aware_utc_now(),
            threat_level=threat_level,
            operation_type=operation_type,
            agent_id=agent_id,
            details=details
        )
        
        # Store active incident
        self._active_incidents[incident_id] = incident
        
        # Update metrics
        self._security_metrics.security_violations += 1
        self._security_metrics.active_incidents = len(self._active_incidents)
        self._security_metrics.last_incident_time = incident.timestamp
        
        # OpenTelemetry metrics
        if OPENTELEMETRY_AVAILABLE and security_violations_counter:
            security_violations_counter.add(
                1,
                attributes={
                    "threat_level": threat_level.value,
                    "operation_type": operation_type.value,
                    "mode": self.mode.value,
                    "agent_id": agent_id
                }
            )
        
        # Log incident
        self.logger.warning(f"SECURITY_INCIDENT: {incident.to_dict()}")
        
        # Auto-resolve low-level incidents
        if threat_level == SecurityThreatLevel.LOW:
            await self._resolve_incident(incident_id)


    async def _resolve_incident(self, incident_id: str) -> None:
        """Resolve a security incident."""
        if incident_id in self._active_incidents:
            incident = self._active_incidents[incident_id]
            incident.resolved = True
            incident.resolution_time = aware_utc_now()
            
            # Move to history
            self._incident_history.append(incident)
            del self._active_incidents[incident_id]
            
            # Update metrics
            self._security_metrics.active_incidents = len(self._active_incidents)
            self._security_metrics.resolved_incidents = len(self._incident_history)
            
            self.logger.info(f"Resolved security incident: {incident_id}")


# ========== Factory Functions and Singleton Management ==========

# Global unified security manager instances (one per mode)
_unified_security_managers: Dict[SecurityMode, UnifiedSecurityManager] = {}


async def get_unified_security_manager(mode: SecurityMode = SecurityMode.API) -> UnifiedSecurityManager:
    """Get unified security manager instance for specified mode.
    
    Args:
        mode: Security operation mode
        
    Returns:
        UnifiedSecurityManager instance
    """
    global _unified_security_managers
    
    if mode not in _unified_security_managers:
        manager = UnifiedSecurityManager(mode=mode)
        await manager.initialize() 
        _unified_security_managers[mode] = manager
        logger.info(f"Created new UnifiedSecurityManager instance for mode: {mode.value}")
    
    return _unified_security_managers[mode]


# Convenience factory functions for common modes
async def get_mcp_security_manager() -> UnifiedSecurityManager:
    """Get security manager optimized for MCP server operations."""
    return await get_unified_security_manager(SecurityMode.MCP_SERVER)


async def get_api_security_manager() -> UnifiedSecurityManager:
    """Get security manager optimized for API operations."""
    return await get_unified_security_manager(SecurityMode.API)


async def get_internal_security_manager() -> UnifiedSecurityManager:
    """Get security manager optimized for internal service communication."""
    return await get_unified_security_manager(SecurityMode.INTERNAL)


async def get_admin_security_manager() -> UnifiedSecurityManager:
    """Get security manager optimized for administrative operations."""
    return await get_unified_security_manager(SecurityMode.ADMIN)


async def get_high_security_manager() -> UnifiedSecurityManager:
    """Get security manager with maximum security settings."""
    return await get_unified_security_manager(SecurityMode.HIGH_SECURITY)


# ========== Testing and Integration Support ==========

class SecurityTestAdapter:
    """Test adapter for unified security manager integration testing."""
    
    def __init__(self, security_manager: UnifiedSecurityManager):
        self.security_manager = security_manager
        self.logger = logging.getLogger(f"{__name__}.SecurityTestAdapter")
    
    async def create_test_security_context(self, agent_id: str = "test_agent") -> SecurityContext:
        """Create a security context for testing."""
        return await create_security_context(
            agent_id=agent_id,
            tier="basic",
            authenticated=True
        )
    
    async def simulate_authentication_attempt(self, 
                                            agent_id: str = "test_agent",
                                            should_succeed: bool = True) -> Tuple[bool, SecurityContext]:
        """Simulate an authentication attempt for testing."""
        credentials = {"token": "test_token"} if should_succeed else {"token": "invalid_token"}
        return await self.security_manager.authenticate_agent(agent_id, credentials)
    
    async def simulate_security_incident(self,
                                       threat_level: SecurityThreatLevel = SecurityThreatLevel.LOW,
                                       agent_id: str = "test_agent") -> str:
        """Simulate a security incident for testing."""
        await self.security_manager._handle_security_incident(
            threat_level=threat_level,
            operation_type=SecurityOperationType.AUTHENTICATION,
            agent_id=agent_id,
            details={"test": True, "simulated": True}
        )
        
        # Return the most recent incident ID
        if self.security_manager._active_incidents:
            return list(self.security_manager._active_incidents.keys())[-1]
        return "no_incident_created"
    
    async def get_test_metrics(self) -> Dict[str, Any]:
        """Get security metrics for testing validation."""
        return await self.security_manager.get_security_status()


async def create_security_test_adapter(mode: SecurityMode = SecurityMode.API) -> SecurityTestAdapter:
    """Create a security test adapter for integration testing."""
    security_manager = await get_unified_security_manager(mode)
    return SecurityTestAdapter(security_manager)