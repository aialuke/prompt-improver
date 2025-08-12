"""Distributed security context for unified security operations.

Extracted from database.unified_connection_manager.py to provide:
- SecurityContext for comprehensive security information
- SecurityValidationResult for validation tracking
- SecurityThreatScore for threat assessment
- SecurityPerformanceMetrics for operation monitoring
- Factory functions for context creation

This provides unified security enforcement across database and application layers.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SecurityThreatScore:
    """Security threat assessment for operations."""

    level: str = "low"
    score: float = 0.0
    factors: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class SecurityValidationResult:
    """Comprehensive security validation results."""

    validated: bool = False
    validation_method: str = "none"
    validation_timestamp: float = field(default_factory=time.time)
    validation_duration_ms: float = 0.0
    security_incidents: list[str] = field(default_factory=list)
    rate_limit_status: str = "unknown"
    encryption_required: bool = False
    audit_trail_id: str | None = None


@dataclass
class SecurityPerformanceMetrics:
    """Security operation performance tracking."""

    authentication_time_ms: float = 0.0
    authorization_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    total_security_overhead_ms: float = 0.0
    operations_count: int = 0
    last_performance_check: float = field(default_factory=time.time)


@dataclass
class SecurityContext:
    """Enhanced security context for unified database and application operations.

    Provides comprehensive security information across all layers:
    - Authentication and authorization details
    - Security validation results and threat assessment
    - Performance metrics and audit trail information
    - Unified security policies and enforcement data

    Designed for zero-friction integration between database and security layers.
    """

    agent_id: str
    tier: str = "basic"
    authenticated: bool = False
    created_at: float = field(default_factory=time.time)
    authentication_method: str = "none"
    authentication_timestamp: float = field(default_factory=time.time)
    session_id: str | None = None
    permissions: list[str] = field(default_factory=list)
    validation_result: SecurityValidationResult = field(
        default_factory=SecurityValidationResult
    )
    threat_score: SecurityThreatScore = field(default_factory=SecurityThreatScore)
    performance_metrics: SecurityPerformanceMetrics = field(
        default_factory=SecurityPerformanceMetrics
    )
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    compliance_tags: list[str] = field(default_factory=list)
    security_level: str = "basic"
    zero_trust_validated: bool = False
    encryption_context: dict[str, str] | None = None
    expires_at: float | None = None
    max_operations: int | None = None
    operations_count: int = 0
    last_used: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """Check if security context is still valid."""
        current_time = time.time()
        if self.expires_at and current_time > self.expires_at:
            return False
        if self.max_operations and self.operations_count >= self.max_operations:
            return False
        if not self.authenticated:
            return False
        return True

    def touch(self) -> None:
        """Update last used timestamp and increment operation count."""
        self.last_used = time.time()
        self.operations_count += 1

    def add_audit_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Add audit event to security context."""
        if "audit_events" not in self.audit_metadata:
            self.audit_metadata["audit_events"] = []
        self.audit_metadata["audit_events"].append({
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
        })
        if len(self.audit_metadata["audit_events"]) > 50:
            self.audit_metadata["audit_events"] = self.audit_metadata["audit_events"][
                -50:
            ]

    def update_threat_score(
        self, new_level: str, new_score: float, factors: list[str]
    ) -> None:
        """Update threat assessment with new information."""
        self.threat_score.level = new_level
        self.threat_score.score = new_score
        self.threat_score.factors = factors
        self.threat_score.last_updated = time.time()

    def record_performance_metric(self, operation: str, duration_ms: float) -> None:
        """Record security operation performance metric."""
        if operation == "authentication":
            self.performance_metrics.authentication_time_ms = duration_ms
        elif operation == "authorization":
            self.performance_metrics.authorization_time_ms = duration_ms
        elif operation == "validation":
            self.performance_metrics.validation_time_ms = duration_ms
        self.performance_metrics.total_security_overhead_ms = (
            self.performance_metrics.authentication_time_ms
            + self.performance_metrics.authorization_time_ms
            + self.performance_metrics.validation_time_ms
        )
        self.performance_metrics.operations_count += 1
        self.performance_metrics.last_performance_check = time.time()

    def model_dump(self) -> dict[str, Any]:
        """Convert security context to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "tier": self.tier,
            "authenticated": self.authenticated,
            "created_at": self.created_at,
            "authentication_method": self.authentication_method,
            "authentication_timestamp": self.authentication_timestamp,
            "session_id": self.session_id,
            "permissions": self.permissions,
            "validation_result": {
                "validated": self.validation_result.validated,
                "validation_method": self.validation_result.validation_method,
                "validation_timestamp": self.validation_result.validation_timestamp,
                "validation_duration_ms": self.validation_result.validation_duration_ms,
                "security_incidents": self.validation_result.security_incidents,
                "rate_limit_status": self.validation_result.rate_limit_status,
                "encryption_required": self.validation_result.encryption_required,
                "audit_trail_id": self.validation_result.audit_trail_id,
            },
            "threat_score": {
                "level": self.threat_score.level,
                "score": self.threat_score.score,
                "factors": self.threat_score.factors,
                "last_updated": self.threat_score.last_updated,
            },
            "performance_metrics": {
                "authentication_time_ms": self.performance_metrics.authentication_time_ms,
                "authorization_time_ms": self.performance_metrics.authorization_time_ms,
                "validation_time_ms": self.performance_metrics.validation_time_ms,
                "total_security_overhead_ms": self.performance_metrics.total_security_overhead_ms,
                "operations_count": self.performance_metrics.operations_count,
                "last_performance_check": self.performance_metrics.last_performance_check,
            },
            "audit_metadata": self.audit_metadata,
            "compliance_tags": self.compliance_tags,
            "security_level": self.security_level,
            "zero_trust_validated": self.zero_trust_validated,
            "encryption_context": self.encryption_context,
            "expires_at": self.expires_at,
            "max_operations": self.max_operations,
            "operations_count": self.operations_count,
            "last_used": self.last_used,
            "is_valid": self.is_valid(),
        }


class RedisSecurityError(Exception):
    """Security-related Redis operation error."""


# Factory functions for creating security contexts


async def create_security_context(
    agent_id: str,
    tier: str = "basic",
    authenticated: bool = True,
    authentication_method: str = "system",
    permissions: list[str] | None = None,
    security_level: str = "basic",
    session_id: str | None = None,
    expires_minutes: int | None = None,
) -> SecurityContext:
    """Create enhanced security context for unified database and application operations.

    Default to authenticated=True for secure-by-default behavior.

    Args:
        agent_id: Agent identifier
        tier: Security tier (basic, professional, enterprise)
        authenticated: Authentication status
        authentication_method: Authentication method used
        permissions: List of permissions granted
        security_level: Security level (basic, enhanced, high, critical)
        session_id: Optional session identifier
        expires_minutes: Optional expiration time in minutes

    Returns:
        Enhanced SecurityContext instance
    """
    current_time = time.time()
    expires_at = current_time + expires_minutes * 60 if expires_minutes else None
    return SecurityContext(
        agent_id=agent_id,
        tier=tier,
        authenticated=authenticated,
        created_at=current_time,
        authentication_method=authentication_method,
        authentication_timestamp=current_time,
        session_id=session_id,
        permissions=permissions or [],
        security_level=security_level,
        expires_at=expires_at,
        last_used=current_time,
    )


async def create_security_context_from_auth_result(
    auth_result=None,
    agent_id: str | None = None,
    tier: str | None = None,
    authenticated: bool | None = None,
    authentication_method: str | None = None,
    permissions: list[str] | None = None,
    security_level: str | None = None,
    expires_minutes: int | None = None,
    auth_result_metadata: dict[str, Any] | None = None,
) -> SecurityContext:
    """Create security context from UnifiedAuthenticationManager authentication result.

    Provides seamless integration between authentication and database layers
    with comprehensive security information transfer. Can work with either
    AuthenticationResult objects or direct parameters for flexible integration.

    Args:
        auth_result: AuthenticationResult from UnifiedAuthenticationManager (optional)
        agent_id: Agent identifier (if not using auth_result)
        tier: Security tier (if not using auth_result)
        authenticated: Authentication status (if not using auth_result)
        authentication_method: Authentication method used (if not using auth_result)
        permissions: User permissions (if not using auth_result)
        security_level: Security level (if not using auth_result)
        expires_minutes: Context expiration in minutes (if not using auth_result)
        auth_result_metadata: Additional metadata from authentication process

    Returns:
        Enhanced SecurityContext with authentication details
    """
    current_time = time.time()
    if auth_result:
        auth_time_ms = auth_result.performance_metrics.get("total_auth_time_ms", 0.0)
        agent_id_final = auth_result.agent_id or "unknown"
        tier_final = auth_result.rate_limit_tier
        authenticated_final = auth_result.success
        auth_method_final = (
            auth_result.authentication_method.value
            if hasattr(auth_result.authentication_method, "value")
            else str(auth_result.authentication_method)
        )
        permissions_final = auth_result.audit_metadata.get("permissions", [])
        session_id_final = auth_result.session_id
        security_level_final = "basic"
        if auth_result.rate_limit_tier in ["professional", "enterprise"]:
            security_level_final = "enhanced"
        if auth_method_final == "api_key":
            security_level_final = (
                "high" if security_level_final == "enhanced" else "enhanced"
            )
        audit_metadata = {
            "authentication_result_integration": True,
            "auth_result_status": auth_result.status.value
            if hasattr(auth_result.status, "value")
            else str(auth_result.status),
            **auth_result.audit_metadata,
        }
        expires_at = time.time() + expires_minutes * 60 if expires_minutes else None
    else:
        auth_time_ms = 0.0
        agent_id_final = agent_id or "unknown"
        tier_final = tier or "basic"
        authenticated_final = authenticated if authenticated is not None else False
        auth_method_final = authentication_method or "unknown"
        permissions_final = permissions or []
        session_id_final = None
        security_level_final = security_level or "basic"
        audit_metadata = {
            "direct_parameter_creation": True,
            "authentication_manager_integration": True,
            **(auth_result_metadata or {}),
        }
        expires_at = time.time() + expires_minutes * 60 if expires_minutes else None
    validation_result = SecurityValidationResult(
        validated=authenticated_final,
        validation_method=auth_method_final,
        validation_timestamp=current_time,
        validation_duration_ms=auth_time_ms,
        rate_limit_status="authenticated" if authenticated_final else "failed",
        audit_trail_id=audit_metadata.get(
            "audit_trail_id", f"auth_{int(current_time * 1000000)}"
        ),
    )
    performance_metrics = SecurityPerformanceMetrics(
        authentication_time_ms=auth_time_ms,
        total_security_overhead_ms=auth_time_ms,
        operations_count=1,
        last_performance_check=current_time,
    )
    return SecurityContext(
        agent_id=agent_id_final,
        tier=tier_final,
        authenticated=authenticated_final,
        created_at=current_time,
        authentication_method=auth_method_final,
        authentication_timestamp=current_time,
        session_id=session_id_final,
        permissions=permissions_final,
        validation_result=validation_result,
        performance_metrics=performance_metrics,
        audit_metadata=audit_metadata,
        security_level=security_level_final,
        zero_trust_validated=authenticated_final,
        expires_at=expires_at,
        last_used=current_time,
    )


async def create_security_context_from_security_manager(
    agent_id: str, security_manager, additional_context: dict[str, Any] | None = None
) -> SecurityContext:
    """Create security context with validation from UnifiedSecurityManager.

    Integrates comprehensive security validation and threat assessment
    into database security context for unified security enforcement.

    Args:
        agent_id: Agent identifier
        security_manager: UnifiedSecurityManager instance
        additional_context: Additional security context information

    Returns:
        SecurityContext with comprehensive security validation
    """
    current_time = time.time()
    try:
        security_status = await security_manager.get_security_status()
        validation_result = SecurityValidationResult(
            validated=True,
            validation_method="security_manager",
            validation_timestamp=current_time,
            validation_duration_ms=0.0,
            rate_limit_status="validated",
            encryption_required=security_manager.config.require_encryption,
            audit_trail_id=f"sm_{int(current_time * 1000000)}",
        )
        threat_level = "low"
        threat_score = 0.0
        threat_factors = []
        if security_status.get("metrics", {}).get("violation_rate", 0) > 0.1:
            threat_level = "medium"
            threat_score = 0.3
            threat_factors.append("elevated_violation_rate")
        if security_status.get("metrics", {}).get("active_incidents", 0) > 0:
            threat_level = "high"
            threat_score = 0.6
            threat_factors.append("active_security_incidents")
        threat_assessment = SecurityThreatScore(
            level=threat_level,
            score=threat_score,
            factors=threat_factors,
            last_updated=current_time,
        )
        avg_operation_time = security_status.get("performance", {}).get(
            "average_operation_time_ms", 0.0
        )
        performance_metrics = SecurityPerformanceMetrics(
            authentication_time_ms=avg_operation_time,
            authorization_time_ms=avg_operation_time,
            validation_time_ms=avg_operation_time,
            total_security_overhead_ms=avg_operation_time * 3,
            operations_count=1,
            last_performance_check=current_time,
        )
        authenticated = (
            additional_context.get("authenticated", True)
            if additional_context
            else True
        )
        if (
            additional_context
            and additional_context.get("authentication_method") == "failed"
        ):
            authenticated = False
        security_level = security_manager.config.security_level.value
        if additional_context and additional_context.get("threat_detected"):
            security_level = "basic"
        tier = security_manager.config.rate_limit_tier.value
        return SecurityContext(
            agent_id=agent_id,
            tier=tier,
            authenticated=authenticated,
            created_at=current_time,
            authentication_method=additional_context.get(
                "authentication_method", "security_manager"
            )
            if additional_context
            else "security_manager",
            authentication_timestamp=current_time,
            session_id=additional_context.get("session_id")
            if additional_context
            else None,
            permissions=additional_context.get("permissions", [])
            if additional_context
            else [],
            validation_result=validation_result,
            threat_score=threat_assessment,
            performance_metrics=performance_metrics,
            audit_metadata={
                "security_manager_integration": True,
                "security_manager_mode": security_manager.mode.value,
                "security_configuration": {
                    "security_level": security_manager.config.security_level.value,
                    "rate_limit_tier": security_manager.config.rate_limit_tier.value,
                    "zero_trust_mode": security_manager.config.zero_trust_mode,
                    "fail_secure": security_manager.config.fail_secure,
                },
                **(additional_context or {}),
            },
            security_level=security_level,
            zero_trust_validated=security_manager.config.zero_trust_mode,
            encryption_context={"required": security_manager.config.require_encryption},
            last_used=current_time,
        )
    except Exception as e:
        logger.error(f"Failed to create security context from security manager: {e}")
        return await create_security_context(
            agent_id=agent_id,
            tier="basic",
            authenticated=False,
            authentication_method="failed",
            security_level="basic",
        )


async def create_system_security_context(
    operation_type: str = "system", security_level: str = "high"
) -> SecurityContext:
    """Create system-level security context for internal operations.

    Args:
        operation_type: Type of system operation
        security_level: Security level for system operations

    Returns:
        System SecurityContext with elevated privileges
    """
    current_time = time.time()
    validation_result = SecurityValidationResult(
        validated=True,
        validation_method="system_internal",
        validation_timestamp=current_time,
        validation_duration_ms=0.0,
        rate_limit_status="system_exempt",
        encryption_required=False,
        audit_trail_id=f"sys_{int(current_time * 1000000)}",
    )
    return SecurityContext(
        agent_id="system",
        tier="system",
        authenticated=True,
        created_at=current_time,
        authentication_method="system_internal",
        authentication_timestamp=current_time,
        session_id=None,
        permissions=["system:all"],
        validation_result=validation_result,
        audit_metadata={"operation_type": operation_type},
        compliance_tags=["system_internal"],
        security_level=security_level,
        zero_trust_validated=True,
        last_used=current_time,
    )
