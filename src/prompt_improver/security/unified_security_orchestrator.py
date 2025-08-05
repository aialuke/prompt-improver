"""
Unified Security Orchestrator - Complete Integration Layer

Orchestrates seamless integration between UnifiedSecurityManager, 
UnifiedAuthenticationManager, and UnifiedConnectionManager SecurityContext
for unified security enforcement across all application layers.

Key Features:
- Single entry point for all security operations
- Unified security context management across authentication, authorization, database operations
- Comprehensive audit logging with single source of truth
- Unified security policy enforcement
- Performance-optimized with context caching and validation
- Zero-friction integration between all security components

Integration Components:
- UnifiedSecurityManager orchestration
- UnifiedAuthenticationManager integration
- SecurityContext lifecycle management
- Unified audit logging and monitoring
- Centralized security policy management
- Performance metrics and monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# Security component imports
from .unified_security_manager import (
    UnifiedSecurityManager, SecurityMode, SecurityThreatLevel, SecurityOperationType,
    get_unified_security_manager, SecurityConfiguration
)
from .unified_authentication_manager import (
    UnifiedAuthenticationManager, AuthenticationResult, AuthenticationStatus,
    get_unified_authentication_manager
)

# Database integration imports
from ..database.unified_connection_manager import (
    SecurityContext, SecurityThreatScore, SecurityValidationResult,
    SecurityPerformanceMetrics, create_security_context,
    create_security_context_from_auth_result,
    create_security_context_from_security_manager,
    get_unified_manager, ManagerMode
)

from ..database.security_integration import (
    UnifiedSecurityIntegration, SecurityIntegrationMode,
    DatabaseOperationType, get_unified_security_integration
)

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    orchestrator_tracer = trace.get_tracer(__name__ + ".security_orchestrator")
    orchestrator_meter = metrics.get_meter(__name__ + ".security_orchestrator")
    
    # Orchestrator-specific metrics
    security_orchestration_operations_counter = orchestrator_meter.create_counter(
        "unified_security_orchestration_operations_total",
        description="Total unified security orchestration operations by type and result",
        unit="1"
    )
    
    security_orchestration_latency_histogram = orchestrator_meter.create_histogram(
        "unified_security_orchestration_duration_seconds", 
        description="Unified security orchestration operation duration by type",
        unit="s"
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    orchestrator_tracer = None
    orchestrator_meter = None
    security_orchestration_operations_counter = None
    security_orchestration_latency_histogram = None

logger = logging.getLogger(__name__)


class SecurityOrchestrationMode(Enum):
    """Security orchestration modes for different deployment scenarios."""
    DEVELOPMENT = "development"      # Development environment with debugging
    PRODUCTION = "production"        # Production environment with full security
    HIGH_SECURITY = "high_security"  # Maximum security for sensitive environments
    TESTING = "testing"             # Testing environment with validation


@dataclass
class UnifiedSecurityPolicy:
    """Unified security policy for consistent enforcement across all layers."""
    policy_id: str
    name: str
    description: str
    required_security_level: str
    required_authentication: bool
    required_permissions: List[str]
    rate_limit_tier: str
    audit_level: str  # minimal, standard, comprehensive
    encryption_required: bool = False
    zero_trust_required: bool = False
    max_threat_score: float = 0.7
    context_ttl_minutes: int = 60
    
    def validates_operation(self, operation_type: str, security_context: SecurityContext) -> Tuple[bool, List[str]]:
        """Validate operation against this policy."""
        violations = []
        
        # Check authentication requirement
        if self.required_authentication and not security_context.authenticated:
            violations.append("Authentication required by policy")
        
        # Check security level
        security_levels = {"basic": 1, "enhanced": 2, "high": 3, "critical": 4}
        context_level = security_levels.get(security_context.security_level, 0)
        required_level = security_levels.get(self.required_security_level, 1)
        if context_level < required_level:
            violations.append(f"Insufficient security level: {security_context.security_level} < {self.required_security_level}")
        
        # Check permissions
        if self.required_permissions:
            missing_permissions = [
                perm for perm in self.required_permissions
                if perm not in security_context.permissions
            ]
            if missing_permissions:
                violations.append(f"Missing required permissions: {missing_permissions}")
        
        # Check threat score
        if security_context.threat_score.score > self.max_threat_score:
            violations.append(f"Threat score too high: {security_context.threat_score.score} > {self.max_threat_score}")
        
        # Check zero trust requirement
        if self.zero_trust_required and not security_context.zero_trust_validated:
            violations.append("Zero trust validation required by policy")
        
        return len(violations) == 0, violations


@dataclass
class SecurityOrchestrationResult:
    """Result of unified security orchestration operation."""
    success: bool
    operation_type: str
    security_context: SecurityContext
    authentication_result: Optional[AuthenticationResult]
    validation_result: Optional[SecurityValidationResult]
    applied_policies: List[str]
    security_warnings: List[str]
    performance_metrics: Dict[str, float]
    audit_metadata: Dict[str, Any]
    error_message: Optional[str] = None


class UnifiedSecurityOrchestrator:
    """Unified Security Orchestrator - Complete Integration Layer.
    
    Provides single entry point for all security operations with seamless
    integration between authentication, authorization, database operations,
    and security policy enforcement.
    
    Key Features:
    - Unified security context management across all application layers
    - Comprehensive audit logging with single source of truth
    - Performance-optimized with intelligent caching and validation
    - Centralized security policy management and enforcement
    - Zero-friction integration between all security components
    - Real-time security monitoring and threat assessment
    """
    
    def __init__(self,
                 orchestration_mode: SecurityOrchestrationMode = SecurityOrchestrationMode.PRODUCTION,
                 security_mode: SecurityMode = SecurityMode.API,
                 enable_comprehensive_logging: bool = True,
                 enable_performance_monitoring: bool = True):
        """Initialize unified security orchestrator.
        
        Args:
            orchestration_mode: Orchestration mode for deployment environment
            security_mode: Security mode for underlying security manager
            enable_comprehensive_logging: Enable comprehensive audit logging
            enable_performance_monitoring: Enable performance monitoring
        """
        self.orchestration_mode = orchestration_mode
        self.security_mode = security_mode
        self.enable_comprehensive_logging = enable_comprehensive_logging
        self.enable_performance_monitoring = enable_performance_monitoring
        self.logger = logging.getLogger(f"{__name__}.UnifiedSecurityOrchestrator")
        
        # Security component instances (initialized lazily)
        self._security_manager: Optional[UnifiedSecurityManager] = None
        self._authentication_manager: Optional[UnifiedAuthenticationManager] = None
        self._security_integration: Optional[UnifiedSecurityIntegration] = None
        self._connection_manager = None
        
        # Unified security policies
        self._security_policies: Dict[str, UnifiedSecurityPolicy] = {}
        self._default_policies_initialized = False
        
        # Performance and monitoring
        self._orchestration_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "authentication_operations": 0,
            "authorization_operations": 0,
            "database_operations": 0,
            "average_operation_time_ms": 0.0
        }
        
        self._operation_history: List[SecurityOrchestrationResult] = []
        self._audit_events: List[Dict[str, Any]] = []
        
        # Context caching for performance
        self._context_cache: Dict[str, SecurityContext] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes default
        
        self.logger.info(f"UnifiedSecurityOrchestrator initialized in {orchestration_mode.value} mode")
    
    async def initialize(self) -> None:
        """Initialize all security components and policies."""
        try:
            start_time = time.time()
            
            # Initialize security manager
            self._security_manager = await get_unified_security_manager(self.security_mode)
            
            # Initialize authentication manager
            self._authentication_manager = await get_unified_authentication_manager()
            
            # Initialize security integration
            integration_mode = SecurityIntegrationMode.STRICT if self.orchestration_mode == SecurityOrchestrationMode.HIGH_SECURITY else SecurityIntegrationMode.STANDARD
            self._security_integration = await get_unified_security_integration(integration_mode)
            
            # Initialize connection manager
            self._connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            await self._connection_manager.initialize()
            
            # Initialize default security policies
            await self._initialize_default_policies()
            
            initialization_time = time.time() - start_time
            self.logger.info(f"UnifiedSecurityOrchestrator fully initialized in {initialization_time:.3f}s")
            
            # Record successful initialization
            await self._record_audit_event("orchestrator_initialized", {
                "orchestration_mode": self.orchestration_mode.value,
                "security_mode": self.security_mode.value,
                "initialization_time_ms": initialization_time * 1000,
                "policies_loaded": len(self._security_policies)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UnifiedSecurityOrchestrator: {e}")
            raise
    
    async def authenticate_and_authorize(self,
                                       request_context: Dict[str, Any],
                                       operation_type: str = "general",
                                       required_permissions: Optional[List[str]] = None,
                                       additional_context: Optional[Dict[str, Any]] = None) -> SecurityOrchestrationResult:
        """Perform unified authentication and authorization with comprehensive security validation.
        
        Args:
            request_context: Request context containing headers and agent information
            operation_type: Type of operation being authenticated/authorized
            required_permissions: Additional permissions required for operation
            additional_context: Additional security context
            
        Returns:
            SecurityOrchestrationResult with comprehensive security information
        """
        start_time = time.time()
        self._orchestration_metrics["total_operations"] += 1
        self._orchestration_metrics["authentication_operations"] += 1
        
        try:
            # Step 1: Authenticate using authentication manager
            auth_result = await self._authentication_manager.authenticate_request(request_context)
            
            if not auth_result.success:
                return await self._create_failed_result(
                    "authentication_failed",
                    f"Authentication failed: {auth_result.error_message}",
                    start_time,
                    auth_result=auth_result
                )
            
            # Step 2: Create unified security context
            security_context = await self._security_integration.create_authenticated_security_context(
                auth_result,
                cache_key=f"auth_{auth_result.agent_id}_{operation_type}"
            )
            
            # Step 3: Enhance security context with additional metadata
            if required_permissions:
                enhancement_data = {"additional_permissions": required_permissions}
                security_context = await self._security_manager.enhance_security_context(
                    security_context, enhancement_data
                )
            
            # Step 4: Validate against unified security policies
            applicable_policies = await self._get_applicable_policies(operation_type, security_context)
            policy_validation_result, policy_warnings = await self._validate_against_policies(
                applicable_policies, operation_type, security_context
            )
            
            if not policy_validation_result:
                return await self._create_failed_result(
                    "policy_violation",
                    f"Security policy violations: {policy_warnings}",
                    start_time,
                    security_context=security_context,
                    auth_result=auth_result
                )
            
            # Step 5: Perform additional security manager validation
            context_validation_result, validation_warnings = await self._security_manager.validate_security_context(
                security_context, operation_type
            )
            
            if not context_validation_result:
                return await self._create_failed_result(
                    "security_validation_failed",
                    f"Security validation failed: {validation_warnings}",
                    start_time,
                    security_context=security_context,
                    auth_result=auth_result
                )
            
            # Step 6: Create successful result
            operation_time = (time.time() - start_time) * 1000
            all_warnings = policy_warnings + validation_warnings
            
            result = SecurityOrchestrationResult(
                success=True,
                operation_type=operation_type,
                security_context=security_context,
                authentication_result=auth_result,
                validation_result=None,  # Will be added by database operations
                applied_policies=[policy.policy_id for policy in applicable_policies],
                security_warnings=all_warnings,
                performance_metrics={
                    "total_operation_time_ms": operation_time,
                    "authentication_time_ms": auth_result.performance_metrics.get("total_auth_time_ms", 0),
                    "security_validation_time_ms": operation_time - auth_result.performance_metrics.get("total_auth_time_ms", 0)
                },
                audit_metadata={
                    "operation_type": operation_type,
                    "orchestration_mode": self.orchestration_mode.value,
                    "agent_id": auth_result.agent_id,
                    "authentication_method": auth_result.authentication_method.value,
                    "policies_applied": len(applicable_policies),
                    "warnings_count": len(all_warnings),
                    **(additional_context or {})
                }
            )
            
            # Update metrics and cache
            await self._update_success_metrics(result)
            await self._cache_security_context(security_context, operation_type)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Authentication and authorization error: {e}")
            return await self._create_failed_result(
                "system_error",
                f"Security system error: {str(e)}",
                start_time
            )
    
    async def execute_secure_operation(self,
                                     operation_func: Callable,
                                     security_context: SecurityContext,
                                     operation_type: str = "general",
                                     database_operation_type: Optional[DatabaseOperationType] = None,
                                     operation_args: Optional[Tuple] = None,
                                     operation_kwargs: Optional[Dict[str, Any]] = None) -> SecurityOrchestrationResult:
        """Execute operation with comprehensive security validation and audit logging.
        
        Args:
            operation_func: Function to execute securely
            security_context: Security context for operation
            operation_type: Type of operation being executed
            database_operation_type: Database operation type if applicable
            operation_args: Arguments for operation function
            operation_kwargs: Keyword arguments for operation function
            
        Returns:
            SecurityOrchestrationResult with operation result and security metadata
        """
        start_time = time.time()
        self._orchestration_metrics["total_operations"] += 1
        
        operation_args = operation_args or ()
        operation_kwargs = operation_kwargs or {}
        
        try:
            # Step 1: Validate and refresh security context
            context_validation_result, warnings = await self._security_manager.validate_security_context(
                security_context, operation_type
            )
            
            if not context_validation_result:
                return await self._create_failed_result(
                    "context_validation_failed",
                    f"Security context validation failed: {warnings}",
                    start_time,
                    security_context=security_context
                )
            
            # Step 2: Database security validation if applicable
            database_validation_result = None
            if database_operation_type:
                self._orchestration_metrics["database_operations"] += 1
                database_validation_result = await self._security_integration.validate_database_operation(
                    database_operation_type, security_context
                )
                
                if not database_validation_result.allowed:
                    return await self._create_failed_result(
                        "database_security_violation",
                        f"Database operation denied: {database_validation_result.security_warnings}",
                        start_time,
                        security_context=security_context,
                        validation_result=database_validation_result
                    )
            
            # Step 3: Execute operation with security context
            operation_start = time.time()
            
            if asyncio.iscoroutinefunction(operation_func):
                operation_result = await operation_func(*operation_args, **operation_kwargs)
            else:
                operation_result = operation_func(*operation_args, **operation_kwargs)
            
            operation_execution_time = (time.time() - operation_start) * 1000
            total_operation_time = (time.time() - start_time) * 1000
            
            # Step 4: Record successful operation
            security_context.add_audit_event("secure_operation_executed", {
                "operation_type": operation_type,
                "database_operation_type": database_operation_type.value if database_operation_type else None,
                "execution_time_ms": operation_execution_time,
                "total_time_ms": total_operation_time,
                "orchestrator_mode": self.orchestration_mode.value
            })
            
            # Step 5: Create successful result
            result = SecurityOrchestrationResult(
                success=True,
                operation_type=operation_type,
                security_context=security_context,
                authentication_result=None,
                validation_result=database_validation_result,
                applied_policies=[],  # Policies applied during initial auth/authz
                security_warnings=warnings,
                performance_metrics={
                    "total_operation_time_ms": total_operation_time,
                    "security_validation_time_ms": total_operation_time - operation_execution_time,
                    "operation_execution_time_ms": operation_execution_time,
                    "database_validation_time_ms": database_validation_result.validation_time_ms if database_validation_result else 0
                },
                audit_metadata={
                    "operation_type": operation_type,
                    "database_operation_type": database_operation_type.value if database_operation_type else None,
                    "operation_result_type": type(operation_result).__name__,
                    "orchestration_mode": self.orchestration_mode.value,
                    "agent_id": security_context.agent_id
                }
            )
            
            # Update metrics
            await self._update_success_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Secure operation execution error: {e}")
            
            # Record failed operation
            security_context.add_audit_event("secure_operation_failed", {
                "operation_type": operation_type,
                "error": str(e),
                "total_time_ms": (time.time() - start_time) * 1000
            })
            
            return await self._create_failed_result(
                "operation_execution_error",
                f"Operation execution failed: {str(e)}",
                start_time,
                security_context=security_context
            )
    
    async def get_comprehensive_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status across all integrated components."""
        try:
            # Get status from all components
            security_manager_status = await self._security_manager.get_security_status()
            auth_manager_metrics = await self._authentication_manager.get_performance_metrics()
            integration_metrics = self._security_integration.get_integration_metrics()
            
            return {
                "orchestrator": {
                    "mode": self.orchestration_mode.value,
                    "security_mode": self.security_mode.value,
                    "policies_loaded": len(self._security_policies),
                    "cached_contexts": len(self._context_cache),
                    "comprehensive_logging_enabled": self.enable_comprehensive_logging,
                    "performance_monitoring_enabled": self.enable_performance_monitoring
                },
                "metrics": self._orchestration_metrics,
                "security_manager": security_manager_status,
                "authentication_manager": auth_manager_metrics,
                "security_integration": integration_metrics,
                "performance": {
                    "recent_operations": len(self._operation_history),
                    "audit_events": len(self._audit_events),
                    "context_cache_utilization": f"{len(self._context_cache)}/{1000}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive security status: {e}")
            return {"error": str(e), "status": "error"}
    
    # ========== Private Implementation Methods ==========
    
    async def _initialize_default_policies(self) -> None:
        """Initialize default unified security policies."""
        default_policies = [
            UnifiedSecurityPolicy(
                policy_id="general_operations",
                name="General Operations Policy",
                description="Default policy for general application operations",
                required_security_level="basic",
                required_authentication=True,
                required_permissions=["read"],
                rate_limit_tier="basic",
                audit_level="standard"
            ),
            UnifiedSecurityPolicy(
                policy_id="data_modification",
                name="Data Modification Policy", 
                description="Policy for operations that modify data",
                required_security_level="enhanced",
                required_authentication=True,
                required_permissions=["write"],
                rate_limit_tier="professional",
                audit_level="comprehensive",
                max_threat_score=0.5
            ),
            UnifiedSecurityPolicy(
                policy_id="administrative_operations",
                name="Administrative Operations Policy",
                description="Policy for administrative and privileged operations",
                required_security_level="critical",
                required_authentication=True,
                required_permissions=["admin"],
                rate_limit_tier="basic",  # More restrictive
                audit_level="comprehensive",
                encryption_required=True,
                zero_trust_required=True,
                max_threat_score=0.3,
                context_ttl_minutes=15
            ),
            UnifiedSecurityPolicy(
                policy_id="high_security_operations",
                name="High Security Operations Policy",
                description="Policy for high-security sensitive operations",
                required_security_level="critical",
                required_authentication=True,
                required_permissions=["high_security"],
                rate_limit_tier="basic",
                audit_level="comprehensive",
                encryption_required=True,
                zero_trust_required=True,
                max_threat_score=0.2,
                context_ttl_minutes=10
            )
        ]
        
        for policy in default_policies:
            self._security_policies[policy.policy_id] = policy
        
        self._default_policies_initialized = True
        self.logger.info(f"Initialized {len(default_policies)} default security policies")
    
    async def _get_applicable_policies(self, operation_type: str, security_context: SecurityContext) -> List[UnifiedSecurityPolicy]:
        """Get security policies applicable to operation type and context."""
        applicable_policies = []
        
        # Policy selection logic based on operation type and context
        if operation_type in ["admin", "administrative", "system"]:
            if "administrative_operations" in self._security_policies:
                applicable_policies.append(self._security_policies["administrative_operations"])
        elif operation_type in ["write", "update", "delete", "modify"]:
            if "data_modification" in self._security_policies:
                applicable_policies.append(self._security_policies["data_modification"])
        elif self.orchestration_mode == SecurityOrchestrationMode.HIGH_SECURITY:
            if "high_security_operations" in self._security_policies:
                applicable_policies.append(self._security_policies["high_security_operations"])
        
        # Always apply general operations policy
        if "general_operations" in self._security_policies:
            applicable_policies.append(self._security_policies["general_operations"])
        
        return applicable_policies
    
    async def _validate_against_policies(self, policies: List[UnifiedSecurityPolicy],
                                       operation_type: str, security_context: SecurityContext) -> Tuple[bool, List[str]]:
        """Validate operation against applicable security policies."""
        all_warnings = []
        
        for policy in policies:
            is_valid, warnings = policy.validates_operation(operation_type, security_context)
            if not is_valid:
                return False, warnings
            all_warnings.extend(warnings)
        
        return True, all_warnings
    
    async def _cache_security_context(self, security_context: SecurityContext, operation_type: str) -> None:
        """Cache security context for performance optimization."""
        cache_key = f"{security_context.agent_id}_{operation_type}_{int(time.time() // 60)}"  # 1-minute buckets
        
        self._context_cache[cache_key] = security_context
        self._cache_expiry[cache_key] = time.time() + self._cache_ttl
        
        # Limit cache size
        if len(self._context_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self._cache_expiry.keys(), key=self._cache_expiry.get)
            del self._context_cache[oldest_key]
            del self._cache_expiry[oldest_key]
    
    async def _update_success_metrics(self, result: SecurityOrchestrationResult) -> None:
        """Update orchestration metrics with successful operation."""
        self._orchestration_metrics["successful_operations"] += 1
        
        # Update average operation time
        total_ops = self._orchestration_metrics["total_operations"]
        current_avg = self._orchestration_metrics["average_operation_time_ms"]
        new_time = result.performance_metrics["total_operation_time_ms"]
        
        self._orchestration_metrics["average_operation_time_ms"] = (
            (current_avg * (total_ops - 1) + new_time) / total_ops
        )
        
        # Store operation history (limited)
        self._operation_history.append(result)
        if len(self._operation_history) > 1000:
            self._operation_history.pop(0)
        
        # Record audit event
        await self._record_audit_event("operation_completed", result.audit_metadata)
    
    async def _create_failed_result(self, error_type: str, error_message: str, start_time: float,
                                  security_context: Optional[SecurityContext] = None,
                                  auth_result: Optional[AuthenticationResult] = None,
                                  validation_result: Optional[SecurityValidationResult] = None) -> SecurityOrchestrationResult:
        """Create standardized failed operation result."""
        self._orchestration_metrics["failed_operations"] += 1
        operation_time = (time.time() - start_time) * 1000
        
        result = SecurityOrchestrationResult(
            success=False,
            operation_type=error_type,
            security_context=security_context or await create_security_context("failed", authenticated=False),
            authentication_result=auth_result,
            validation_result=validation_result,
            applied_policies=[],
            security_warnings=[],
            performance_metrics={"total_operation_time_ms": operation_time},
            audit_metadata={
                "error_type": error_type,
                "orchestration_mode": self.orchestration_mode.value,
                "timestamp": time.time()
            },
            error_message=error_message
        )
        
        # Record audit event
        await self._record_audit_event("operation_failed", {
            "error_type": error_type,
            "error_message": error_message,
            "operation_time_ms": operation_time
        })
        
        return result
    
    async def _record_audit_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Record comprehensive audit event."""
        if not self.enable_comprehensive_logging:
            return
        
        audit_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "orchestration_mode": self.orchestration_mode.value,
            "security_mode": self.security_mode.value,
            **metadata
        }
        
        self._audit_events.append(audit_event)
        
        # Limit audit event storage
        if len(self._audit_events) > 10000:
            self._audit_events.pop(0)
        
        # Log to standard logging
        self.logger.info(f"SECURITY_AUDIT: {audit_event}")


# ========== Factory Functions and Singleton Management ==========

# Global unified security orchestrator instance
_unified_security_orchestrator: Optional[UnifiedSecurityOrchestrator] = None


async def get_unified_security_orchestrator(
    orchestration_mode: SecurityOrchestrationMode = SecurityOrchestrationMode.PRODUCTION,
    security_mode: SecurityMode = SecurityMode.API
) -> UnifiedSecurityOrchestrator:
    """Get global unified security orchestrator instance.
    
    Args:
        orchestration_mode: Orchestration mode for deployment environment
        security_mode: Security mode for underlying security manager
        
    Returns:
        UnifiedSecurityOrchestrator instance
    """
    global _unified_security_orchestrator
    
    if _unified_security_orchestrator is None:
        _unified_security_orchestrator = UnifiedSecurityOrchestrator(
            orchestration_mode=orchestration_mode,
            security_mode=security_mode,
            enable_comprehensive_logging=True,
            enable_performance_monitoring=True
        )
        await _unified_security_orchestrator.initialize()
        logger.info(f"Created UnifiedSecurityOrchestrator in {orchestration_mode.value} mode")
    
    return _unified_security_orchestrator


# ========== Convenience Functions for Common Security Operations ==========

async def authenticate_and_authorize_request(request_context: Dict[str, Any],
                                           operation_type: str = "general",
                                           required_permissions: Optional[List[str]] = None) -> SecurityOrchestrationResult:
    """Convenience function for request authentication and authorization."""
    orchestrator = await get_unified_security_orchestrator()
    return await orchestrator.authenticate_and_authorize(
        request_context, operation_type, required_permissions
    )


async def execute_with_security(operation_func: Callable,
                               security_context: SecurityContext,
                               operation_type: str = "general",
                               *args, **kwargs) -> SecurityOrchestrationResult:
    """Convenience function for executing operations with security validation."""
    orchestrator = await get_unified_security_orchestrator()
    return await orchestrator.execute_secure_operation(
        operation_func, security_context, operation_type, 
        operation_args=args, operation_kwargs=kwargs
    )


async def execute_secure_database_operation(operation_func: Callable,
                                          security_context: SecurityContext,
                                          database_operation_type: DatabaseOperationType,
                                          *args, **kwargs) -> SecurityOrchestrationResult:
    """Convenience function for secure database operations."""
    orchestrator = await get_unified_security_orchestrator()
    return await orchestrator.execute_secure_operation(
        operation_func, security_context, "database_operation",
        database_operation_type, args, kwargs
    )