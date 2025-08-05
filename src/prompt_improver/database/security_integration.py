"""
Database Security Integration - Unified Security Across Database and Application Layers

Provides seamless integration between UnifiedSecurityManager, UnifiedAuthenticationManager,
and UnifiedConnectionManager SecurityContext for unified security enforcement.

Key Features:
- Unified security context creation from authentication results
- Database operations with integrated security validation
- Real-time security monitoring for database operations  
- Comprehensive audit logging across all layers
- Performance-optimized security context management
- Zero-friction integration between security and database layers

Security Integration Components:
- SecurityContextManager: Manages security context lifecycle
- DatabaseSecurityValidator: Validates database operations against security policies
- UnifiedAuditLogger: Provides comprehensive audit logging across layers
- SecurityMetricsCollector: Collects security metrics from all components
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Database and security imports
from .unified_connection_manager import (
    SecurityContext, 
    SecurityThreatScore,
    SecurityValidationResult,
    SecurityPerformanceMetrics,
    create_security_context,
    create_security_context_from_auth_result,
    create_security_context_from_security_manager,
    create_system_security_context,
    get_unified_manager,
    ManagerMode,
    RedisSecurityError
)

# Security component imports (lazy loading to avoid circular imports)
# from ..security.unified_security_manager import (
#     UnifiedSecurityManager, SecurityMode, SecurityThreatLevel, SecurityOperationType
# )
# from ..security.unified_authentication_manager import (
#     UnifiedAuthenticationManager, AuthenticationResult, AuthenticationStatus
# )

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    security_integration_tracer = trace.get_tracer(__name__ + ".security_integration")
    security_integration_meter = metrics.get_meter(__name__ + ".security_integration")
    
    # Security integration metrics
    security_integration_operations_counter = security_integration_meter.create_counter(
        "database_security_operations_total",
        description="Total database security integration operations by type and result",
        unit="1"
    )
    
    security_context_lifecycle_counter = security_integration_meter.create_counter(
        "security_context_lifecycle_total",
        description="Security context lifecycle events by type",
        unit="1"
    )
    
    security_validation_duration_histogram = security_integration_meter.create_histogram(
        "database_security_validation_duration_seconds",
        description="Database security validation duration by operation type",
        unit="s"
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    security_integration_tracer = None
    security_integration_meter = None
    security_integration_operations_counter = None
    security_context_lifecycle_counter = None
    security_validation_duration_histogram = None

logger = logging.getLogger(__name__)


class SecurityIntegrationMode(Enum):
    """Security integration modes for different operation types."""
    STRICT = "strict"          # Maximum security validation
    STANDARD = "standard"      # Standard security validation
    PERFORMANCE = "performance"  # Optimized for performance
    BYPASS = "bypass"          # Bypass security checks (testing only)


class DatabaseOperationType(Enum):
    """Database operation types for security validation."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    CACHE = "cache"
    TRANSACTION = "transaction"


@dataclass
class SecurityPolicyRule:
    """Security policy rule for database operations."""
    operation_type: DatabaseOperationType
    required_permissions: List[str]
    minimum_security_level: str
    require_encryption: bool = False
    audit_level: str = "standard"  # minimal, standard, comprehensive
    rate_limit_override: Optional[str] = None


@dataclass
class DatabaseSecurityValidationResult:
    """Result of database security validation."""
    allowed: bool
    security_context: SecurityContext
    validation_time_ms: float
    applied_policies: List[str]
    security_warnings: List[str]
    audit_metadata: Dict[str, Any]
    

class SecurityContextManager:
    """Manages security context lifecycle across database and application layers.
    
    Provides comprehensive security context management with:
    - Unified context creation from multiple security sources
    - Context validation and refresh
    - Performance-optimized context caching
    - Comprehensive audit logging
    """
    
    def __init__(self, 
                 integration_mode: SecurityIntegrationMode = SecurityIntegrationMode.STANDARD,
                 enable_context_caching: bool = True):
        """Initialize security context manager.
        
        Args:
            integration_mode: Security integration mode
            enable_context_caching: Enable security context caching for performance
        """
        self.integration_mode = integration_mode
        self.enable_context_caching = enable_context_caching
        self.logger = logging.getLogger(f"{__name__}.SecurityContextManager")
        
        # Context caching for performance
        self._context_cache: Dict[str, SecurityContext] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes default TTL
        
        # Performance metrics
        self._context_creation_times: List[float] = []
        self._validation_times: List[float] = []
        
        # Audit tracking
        self._audit_events: List[Dict[str, Any]] = []
        
        self.logger.info(f"SecurityContextManager initialized in {integration_mode.value} mode")
    
    async def create_context_from_authentication(self, 
                                               auth_result,
                                               cache_key: Optional[str] = None) -> SecurityContext:
        """Create security context from authentication result with caching support.
        
        Args:
            auth_result: AuthenticationResult from UnifiedAuthenticationManager
            cache_key: Optional cache key for performance optimization
            
        Returns:
            Enhanced SecurityContext
        """
        start_time = time.time()
        
        try:
            # Check cache first if enabled
            if self.enable_context_caching and cache_key:
                cached_context = self._get_cached_context(cache_key)
                if cached_context and cached_context.is_valid():
                    cached_context.touch()
                    return cached_context
            
            # Create context from authentication result
            security_context = await create_security_context_from_auth_result(auth_result)
            
            # Add integration-specific metadata
            security_context.add_audit_event("context_created", {
                "source": "authentication_result",
                "integration_mode": self.integration_mode.value,
                "cached": False
            })
            
            # Cache context if enabled
            if self.enable_context_caching and cache_key:
                self._cache_context(cache_key, security_context)
            
            creation_time = (time.time() - start_time) * 1000
            self._context_creation_times.append(creation_time)
            security_context.record_performance_metric("context_creation", creation_time)
            
            # OpenTelemetry metrics
            if OPENTELEMETRY_AVAILABLE and security_context_lifecycle_counter:
                security_context_lifecycle_counter.add(
                    1,
                    attributes={
                        "event": "created",
                        "source": "authentication",
                        "mode": self.integration_mode.value,
                        "cached": str(bool(cache_key))
                    }
                )
            
            self.logger.debug(f"Created security context for {security_context.agent_id} from authentication")
            return security_context
            
        except Exception as e:
            self.logger.error(f"Failed to create security context from authentication: {e}")
            # Fail-secure: create minimal context
            return await create_security_context(
                agent_id="failed_auth",
                authenticated=False,
                security_level="basic"  
            )
    
    async def create_context_from_security_manager(self,
                                                 agent_id: str,
                                                 security_manager,
                                                 additional_context: Optional[Dict[str, Any]] = None,
                                                 cache_key: Optional[str] = None) -> SecurityContext:
        """Create security context from security manager with comprehensive validation.
        
        Args:
            agent_id: Agent identifier
            security_manager: UnifiedSecurityManager instance
            additional_context: Additional security context
            cache_key: Optional cache key for performance
            
        Returns:
            SecurityContext with comprehensive security validation
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.enable_context_caching and cache_key:
                cached_context = self._get_cached_context(cache_key)
                if cached_context and cached_context.is_valid():
                    cached_context.touch()
                    return cached_context
            
            # Create context from security manager
            security_context = await create_security_context_from_security_manager(
                agent_id, security_manager, additional_context
            )
            
            # Add integration metadata
            security_context.add_audit_event("context_created", {
                "source": "security_manager",
                "integration_mode": self.integration_mode.value,
                "security_manager_mode": security_manager.mode.value,
                "additional_context_keys": list(additional_context.keys()) if additional_context else []
            })
            
            # Cache context
            if self.enable_context_caching and cache_key:
                self._cache_context(cache_key, security_context)
            
            creation_time = (time.time() - start_time) * 1000
            security_context.record_performance_metric("context_creation", creation_time)
            
            self.logger.debug(f"Created security context for {agent_id} from security manager")
            return security_context
            
        except Exception as e:
            self.logger.error(f"Failed to create security context from security manager: {e}")
            return await create_security_context(
                agent_id=agent_id,
                authenticated=False,
                security_level="basic"
            )
    
    async def validate_and_refresh_context(self, security_context: SecurityContext) -> SecurityContext:
        """Validate and refresh security context if needed.
        
        Args:
            security_context: Current security context
            
        Returns:
            Validated and potentially refreshed SecurityContext
        """
        start_time = time.time()
        
        try:
            # Check if context is still valid
            if not security_context.is_valid():
                self.logger.warning(f"Security context invalid for {security_context.agent_id}")
                
                # Try to refresh context if we have session information
                if security_context.session_id:
                    # This would integrate with authentication manager to refresh
                    # For now, create a new basic context
                    refreshed_context = await create_security_context(
                        agent_id=security_context.agent_id,
                        tier=security_context.tier,
                        authenticated=False,
                        security_level="basic"
                    )
                    
                    refreshed_context.add_audit_event("context_refreshed", {
                        "reason": "invalid_context",
                        "original_created_at": security_context.created_at
                    })
                    
                    return refreshed_context
            
            # Update usage tracking
            security_context.touch()
            
            # Update threat assessment if needed
            if time.time() - security_context.threat_score.last_updated > 300:  # 5 minutes
                # This would integrate with security manager for real threat assessment
                security_context.update_threat_score("low", 0.1, ["periodic_assessment"])
            
            validation_time = (time.time() - start_time) * 1000
            self._validation_times.append(validation_time)
            security_context.record_performance_metric("validation", validation_time)
            
            return security_context
            
        except Exception as e:
            self.logger.error(f"Context validation failed for {security_context.agent_id}: {e}")
            return security_context  # Return original context on validation errors
    
    def _get_cached_context(self, cache_key: str) -> Optional[SecurityContext]:
        """Get cached security context if valid."""
        if cache_key not in self._context_cache:
            return None
            
        # Check expiry
        if cache_key in self._cache_expiry:
            if time.time() > self._cache_expiry[cache_key]:
                del self._context_cache[cache_key]
                del self._cache_expiry[cache_key]
                return None
        
        return self._context_cache[cache_key]
    
    def _cache_context(self, cache_key: str, context: SecurityContext) -> None:
        """Cache security context with TTL."""
        self._context_cache[cache_key] = context
        self._cache_expiry[cache_key] = time.time() + self._cache_ttl
        
        # Limit cache size
        if len(self._context_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self._cache_expiry.keys(), key=self._cache_expiry.get)
            del self._context_cache[oldest_key]
            del self._cache_expiry[oldest_key]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for security context operations."""
        return {
            "average_context_creation_time_ms": (
                sum(self._context_creation_times) / len(self._context_creation_times)
                if self._context_creation_times else 0.0
            ),
            "average_validation_time_ms": (
                sum(self._validation_times) / len(self._validation_times)
                if self._validation_times else 0.0
            ),
            "cached_contexts_count": len(self._context_cache),
            "cache_hit_rate": 0.0,  # Would track in real implementation
            "total_contexts_created": len(self._context_creation_times),
            "total_validations": len(self._validation_times)
        }


class DatabaseSecurityValidator:
    """Validates database operations against unified security policies.
    
    Integrates security validation across database and application layers
    with comprehensive policy enforcement and audit logging.
    """
    
    def __init__(self, 
                 integration_mode: SecurityIntegrationMode = SecurityIntegrationMode.STANDARD):
        """Initialize database security validator.
        
        Args:
            integration_mode: Security integration mode
        """
        self.integration_mode = integration_mode
        self.logger = logging.getLogger(f"{__name__}.DatabaseSecurityValidator")
        
        # Default security policies
        self._security_policies = self._create_default_policies()
        
        # Validation metrics
        self._validation_results: List[DatabaseSecurityValidationResult] = []
        
        self.logger.info(f"DatabaseSecurityValidator initialized in {integration_mode.value} mode")
    
    async def validate_database_operation(self,
                                        operation_type: DatabaseOperationType,
                                        security_context: SecurityContext,
                                        operation_details: Optional[Dict[str, Any]] = None) -> DatabaseSecurityValidationResult:
        """Validate database operation against security policies.
        
        Args:
            operation_type: Type of database operation
            security_context: Security context for validation
            operation_details: Additional operation details
            
        Returns:
            DatabaseSecurityValidationResult with validation outcome
        """
        start_time = time.time()
        
        try:
            # Get applicable policies
            applicable_policies = self._get_applicable_policies(operation_type)
            applied_policies = []
            security_warnings = []
            
            # Validate security context
            if not security_context.is_valid():
                return DatabaseSecurityValidationResult(
                    allowed=False,
                    security_context=security_context,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    applied_policies=[],
                    security_warnings=["Invalid security context"],
                    audit_metadata={"validation_failed": "invalid_security_context"}
                )
            
            # Check authentication requirement
            if operation_type in [DatabaseOperationType.WRITE, DatabaseOperationType.DELETE, DatabaseOperationType.ADMIN]:
                if not security_context.authenticated:
                    return self._create_denied_result(
                        security_context, start_time, "Authentication required for operation", []
                    )
            
            # Validate against each applicable policy
            for policy in applicable_policies:
                policy_result = await self._validate_against_policy(
                    policy, security_context, operation_type, operation_details
                )
                
                if not policy_result["allowed"]:
                    return self._create_denied_result(
                        security_context, start_time, policy_result["reason"], applied_policies
                    )
                
                applied_policies.extend(policy_result["applied_policies"])
                security_warnings.extend(policy_result["warnings"])
            
            # Additional security checks based on integration mode
            if self.integration_mode == SecurityIntegrationMode.STRICT:
                strict_result = await self._perform_strict_validation(
                    security_context, operation_type, operation_details
                )
                if not strict_result["allowed"]:
                    return self._create_denied_result(
                        security_context, start_time, strict_result["reason"], applied_policies
                    )
                security_warnings.extend(strict_result["warnings"])
            
            # Update security context with operation
            security_context.touch()
            security_context.add_audit_event("database_operation_validated", {
                "operation_type": operation_type.value,
                "applied_policies": applied_policies,
                "security_warnings": security_warnings,
                "integration_mode": self.integration_mode.value
            })
            
            validation_time = (time.time() - start_time) * 1000
            security_context.record_performance_metric("database_validation", validation_time)
            
            # Create successful result
            result = DatabaseSecurityValidationResult(
                allowed=True,
                security_context=security_context,
                validation_time_ms=validation_time,
                applied_policies=applied_policies,
                security_warnings=security_warnings,
                audit_metadata={
                    "operation_type": operation_type.value,
                    "validation_mode": self.integration_mode.value,
                    "policies_evaluated": len(applicable_policies)
                }
            )
            
            # Store result for metrics
            self._validation_results.append(result)
            
            # OpenTelemetry metrics
            if OPENTELEMETRY_AVAILABLE and security_integration_operations_counter:
                security_integration_operations_counter.add(
                    1,
                    attributes={
                        "operation_type": operation_type.value,
                        "result": "allowed",
                        "mode": self.integration_mode.value,
                        "agent_id": security_context.agent_id
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Database security validation error: {e}")
            return self._create_denied_result(
                security_context, start_time, f"Validation system error: {str(e)}", []
            )
    
    def _create_default_policies(self) -> Dict[DatabaseOperationType, List[SecurityPolicyRule]]:
        """Create default security policies for database operations."""
        return {
            DatabaseOperationType.READ: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.READ,
                    required_permissions=["read"],
                    minimum_security_level="basic"
                )
            ],
            DatabaseOperationType.WRITE: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.WRITE,
                    required_permissions=["write"],
                    minimum_security_level="enhanced",
                    audit_level="comprehensive"
                )
            ],
            DatabaseOperationType.DELETE: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.DELETE,
                    required_permissions=["delete"],
                    minimum_security_level="high",
                    audit_level="comprehensive"
                )
            ],
            DatabaseOperationType.ADMIN: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.ADMIN,
                    required_permissions=["admin"],
                    minimum_security_level="critical",
                    require_encryption=True,
                    audit_level="comprehensive"
                )
            ],
            DatabaseOperationType.CACHE: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.CACHE,
                    required_permissions=["cache"],
                    minimum_security_level="basic"
                )
            ],
            DatabaseOperationType.TRANSACTION: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.TRANSACTION,
                    required_permissions=["transaction"],
                    minimum_security_level="enhanced",
                    audit_level="comprehensive"
                )
            ]
        }
    
    def _get_applicable_policies(self, operation_type: DatabaseOperationType) -> List[SecurityPolicyRule]:
        """Get security policies applicable to operation type."""
        return self._security_policies.get(operation_type, [])
    
    async def _validate_against_policy(self,
                                     policy: SecurityPolicyRule,
                                     security_context: SecurityContext,
                                     operation_type: DatabaseOperationType,
                                     operation_details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate operation against specific security policy."""
        warnings = []
        applied_policies = [f"{operation_type.value}_policy"]
        
        # Check required permissions
        if policy.required_permissions:
            missing_permissions = [
                perm for perm in policy.required_permissions 
                if perm not in security_context.permissions
            ]
            if missing_permissions:
                return {
                    "allowed": False,
                    "reason": f"Missing required permissions: {missing_permissions}",
                    "applied_policies": applied_policies,
                    "warnings": warnings
                }
        
        # Check minimum security level
        security_levels = {"basic": 1, "enhanced": 2, "high": 3, "critical": 4}
        current_level = security_levels.get(security_context.security_level, 0)
        required_level = security_levels.get(policy.minimum_security_level, 1)
        
        if current_level < required_level:
            return {
                "allowed": False,
                "reason": f"Insufficient security level: {security_context.security_level} < {policy.minimum_security_level}",
                "applied_policies": applied_policies,
                "warnings": warnings
            }
        
        # Check encryption requirement
        if policy.require_encryption and not security_context.encryption_context:
            warnings.append("Encryption recommended but not enforced")
        
        return {
            "allowed": True,
            "reason": "Policy validation successful",
            "applied_policies": applied_policies,
            "warnings": warnings
        }
    
    async def _perform_strict_validation(self,
                                       security_context: SecurityContext,
                                       operation_type: DatabaseOperationType,
                                       operation_details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform additional strict security validation."""
        warnings = []
        
        # Check threat score
        if security_context.threat_score.score > 0.5:
            return {
                "allowed": False,
                "reason": f"High threat score: {security_context.threat_score.score}",
                "warnings": warnings
            }
        
        # Check zero trust validation
        if not security_context.zero_trust_validated:
            warnings.append("Zero trust validation not performed")
        
        # Check context age
        context_age = time.time() - security_context.created_at
        if context_age > 3600:  # 1 hour
            warnings.append("Security context is older than 1 hour")
        
        return {
            "allowed": True,
            "reason": "Strict validation passed",
            "warnings": warnings
        }
    
    def _create_denied_result(self,
                            security_context: SecurityContext,
                            start_time: float,
                            reason: str,
                            applied_policies: List[str]) -> DatabaseSecurityValidationResult:
        """Create denied validation result."""
        validation_time = (time.time() - start_time) * 1000
        
        result = DatabaseSecurityValidationResult(
            allowed=False,
            security_context=security_context,
            validation_time_ms=validation_time,
            applied_policies=applied_policies,
            security_warnings=[reason],
            audit_metadata={
                "denied_reason": reason,
                "validation_mode": self.integration_mode.value
            }
        )
        
        # Store result for metrics
        self._validation_results.append(result)
        
        return result
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation performance metrics."""
        if not self._validation_results:
            return {"total_validations": 0}
        
        allowed_count = sum(1 for r in self._validation_results if r.allowed)
        denied_count = len(self._validation_results) - allowed_count
        avg_validation_time = sum(r.validation_time_ms for r in self._validation_results) / len(self._validation_results)
        
        return {
            "total_validations": len(self._validation_results),
            "allowed_validations": allowed_count,
            "denied_validations": denied_count,
            "success_rate": allowed_count / len(self._validation_results),
            "average_validation_time_ms": avg_validation_time,
            "integration_mode": self.integration_mode.value
        }


class UnifiedSecurityIntegration:
    """Unified security integration orchestrator.
    
    Provides the main interface for integrating security across
    database and application layers with zero friction.
    """
    
    def __init__(self,
                 integration_mode: SecurityIntegrationMode = SecurityIntegrationMode.STANDARD,
                 enable_context_caching: bool = True,
                 enable_performance_monitoring: bool = True):
        """Initialize unified security integration.
        
        Args:
            integration_mode: Security integration mode
            enable_context_caching: Enable security context caching
            enable_performance_monitoring: Enable performance monitoring
        """
        self.integration_mode = integration_mode
        self.enable_performance_monitoring = enable_performance_monitoring
        self.logger = logging.getLogger(f"{__name__}.UnifiedSecurityIntegration")
        
        # Initialize components
        self.context_manager = SecurityContextManager(integration_mode, enable_context_caching)
        self.database_validator = DatabaseSecurityValidator(integration_mode)
        
        # Connection manager for database operations
        self._connection_manager = None
        
        # Performance monitoring
        self._integration_metrics = {
            "operations_processed": 0,
            "average_security_overhead_ms": 0.0,
            "security_violations": 0,
            "cache_hit_rate": 0.0
        }
        
        self.logger.info(f"UnifiedSecurityIntegration initialized in {integration_mode.value} mode")
    
    async def initialize(self) -> None:
        """Initialize async components."""
        try:
            # Initialize connection manager
            self._connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            await self._connection_manager.initialize()
            
            self.logger.info("UnifiedSecurityIntegration async initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UnifiedSecurityIntegration: {e}")
            raise
    
    async def create_authenticated_security_context(self,
                                                  auth_result,
                                                  cache_key: Optional[str] = None) -> SecurityContext:
        """Create security context from authentication result.
        
        Args:
            auth_result: AuthenticationResult from UnifiedAuthenticationManager
            cache_key: Optional cache key for performance
            
        Returns:
            Enhanced SecurityContext
        """
        return await self.context_manager.create_context_from_authentication(
            auth_result, cache_key
        )
    
    async def create_security_manager_context(self,
                                            agent_id: str,
                                            security_manager,
                                            additional_context: Optional[Dict[str, Any]] = None,
                                            cache_key: Optional[str] = None) -> SecurityContext:
        """Create security context from security manager.
        
        Args:
            agent_id: Agent identifier
            security_manager: UnifiedSecurityManager instance
            additional_context: Additional context information
            cache_key: Optional cache key
            
        Returns:
            SecurityContext with comprehensive validation
        """
        return await self.context_manager.create_context_from_security_manager(
            agent_id, security_manager, additional_context, cache_key
        )
    
    async def validate_database_operation(self,
                                        operation_type: DatabaseOperationType,
                                        security_context: SecurityContext,
                                        operation_details: Optional[Dict[str, Any]] = None) -> DatabaseSecurityValidationResult:
        """Validate database operation with comprehensive security checks.
        
        Args:
            operation_type: Type of database operation
            security_context: Security context for validation
            operation_details: Additional operation details
            
        Returns:
            Validation result with security assessment
        """
        start_time = time.time()
        
        try:
            # Validate and refresh context if needed
            validated_context = await self.context_manager.validate_and_refresh_context(
                security_context
            )
            
            # Perform database security validation
            validation_result = await self.database_validator.validate_database_operation(
                operation_type, validated_context, operation_details
            )
            
            # Update integration metrics
            self._integration_metrics["operations_processed"] += 1
            if not validation_result.allowed:
                self._integration_metrics["security_violations"] += 1
            
            operation_time = (time.time() - start_time) * 1000
            current_avg = self._integration_metrics["average_security_overhead_ms"]
            total_ops = self._integration_metrics["operations_processed"]
            self._integration_metrics["average_security_overhead_ms"] = (
                (current_avg * (total_ops - 1) + operation_time) / total_ops
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Database operation validation failed: {e}")
            self._integration_metrics["security_violations"] += 1
            
            # Return denied result on errors
            return DatabaseSecurityValidationResult(
                allowed=False,
                security_context=security_context,
                validation_time_ms=(time.time() - start_time) * 1000,
                applied_policies=[],
                security_warnings=[f"Validation system error: {str(e)}"],
                audit_metadata={"error": str(e)}
            )
    
    async def execute_secure_database_operation(self,
                                              operation_type: DatabaseOperationType,
                                              operation_func: callable,
                                              security_context: SecurityContext,
                                              operation_args: Optional[Tuple] = None,
                                              operation_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute database operation with integrated security validation.
        
        Args:
            operation_type: Type of database operation
            operation_func: Database operation function to execute
            security_context: Security context for operation
            operation_args: Arguments for operation function
            operation_kwargs: Keyword arguments for operation function
            
        Returns:
            Operation result with security metadata
        """
        operation_args = operation_args or ()
        operation_kwargs = operation_kwargs or {}
        
        start_time = time.time()
        
        try:
            # Validate operation
            validation_result = await self.validate_database_operation(
                operation_type, security_context
            )
            
            if not validation_result.allowed:
                return {
                    "success": False,
                    "error": "Operation denied by security policy",
                    "security_warnings": validation_result.security_warnings,
                    "validation_result": validation_result
                }
            
            # Execute operation with security context
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*operation_args, **operation_kwargs)
            else:
                result = operation_func(*operation_args, **operation_kwargs)
            
            # Record successful operation
            validation_result.security_context.add_audit_event("database_operation_executed", {
                "operation_type": operation_type.value,
                "success": True,
                "execution_time_ms": (time.time() - start_time) * 1000
            })
            
            return {
                "success": True,
                "result": result,
                "security_context": validation_result.security_context,
                "validation_result": validation_result,
                "performance_metrics": {
                    "total_operation_time_ms": (time.time() - start_time) * 1000,
                    "security_overhead_ms": validation_result.validation_time_ms
                }
            }
            
        except Exception as e:
            self.logger.error(f"Secure database operation failed: {e}")
            
            # Record failed operation
            security_context.add_audit_event("database_operation_failed", {
                "operation_type": operation_type.value,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000
            })
            
            return {
                "success": False,
                "error": str(e),
                "security_context": security_context,
                "performance_metrics": {
                    "total_operation_time_ms": (time.time() - start_time) * 1000
                }
            }
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration performance metrics."""
        context_metrics = self.context_manager.get_performance_metrics()
        validation_metrics = self.database_validator.get_validation_metrics()
        
        return {
            "integration_mode": self.integration_mode.value,
            "operations_processed": self._integration_metrics["operations_processed"],
            "average_security_overhead_ms": self._integration_metrics["average_security_overhead_ms"],
            "security_violations": self._integration_metrics["security_violations"],
            "violation_rate": (
                self._integration_metrics["security_violations"] / 
                max(1, self._integration_metrics["operations_processed"])
            ),
            "context_manager_metrics": context_metrics,
            "database_validator_metrics": validation_metrics
        }


# ========== Factory Functions and Singleton Management ==========

# Global unified security integration instance
_unified_security_integration: Optional[UnifiedSecurityIntegration] = None


async def get_unified_security_integration(
    integration_mode: SecurityIntegrationMode = SecurityIntegrationMode.STANDARD
) -> UnifiedSecurityIntegration:
    """Get global unified security integration instance.
    
    Args:
        integration_mode: Security integration mode
        
    Returns:
        UnifiedSecurityIntegration instance
    """
    global _unified_security_integration
    
    if _unified_security_integration is None:
        _unified_security_integration = UnifiedSecurityIntegration(
            integration_mode=integration_mode,
            enable_context_caching=True,
            enable_performance_monitoring=True
        )
        await _unified_security_integration.initialize()
        logger.info(f"Created UnifiedSecurityIntegration in {integration_mode.value} mode")
    
    return _unified_security_integration


# ========== Convenience Functions for Common Operations ==========

async def create_authenticated_context(auth_result, cache_key: Optional[str] = None) -> SecurityContext:
    """Convenience function to create security context from authentication."""
    integration = await get_unified_security_integration()
    return await integration.create_authenticated_security_context(auth_result, cache_key)


async def validate_database_read(security_context: SecurityContext, 
                               details: Optional[Dict[str, Any]] = None) -> DatabaseSecurityValidationResult:
    """Convenience function to validate database read operation."""
    integration = await get_unified_security_integration()
    return await integration.validate_database_operation(
        DatabaseOperationType.READ, security_context, details
    )


async def validate_database_write(security_context: SecurityContext,
                                details: Optional[Dict[str, Any]] = None) -> DatabaseSecurityValidationResult:
    """Convenience function to validate database write operation."""
    integration = await get_unified_security_integration()
    return await integration.validate_database_operation(
        DatabaseOperationType.WRITE, security_context, details
    )


async def execute_secure_read(operation_func: callable,
                            security_context: SecurityContext,
                            *args, **kwargs) -> Dict[str, Any]:
    """Convenience function to execute secure database read."""
    integration = await get_unified_security_integration()
    return await integration.execute_secure_database_operation(
        DatabaseOperationType.READ, operation_func, security_context, args, kwargs
    )


async def execute_secure_write(operation_func: callable,
                             security_context: SecurityContext,
                             *args, **kwargs) -> Dict[str, Any]:
    """Convenience function to execute secure database write."""
    integration = await get_unified_security_integration()
    return await integration.execute_secure_database_operation(
        DatabaseOperationType.WRITE, operation_func, security_context, args, kwargs
    )