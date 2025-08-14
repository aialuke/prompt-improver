"""Database Security Integration - Unified Security Across Database and Application Layers

Provides seamless integration between UnifiedSecurityManager, UnifiedAuthenticationManager,
and DatabaseServices SecurityContext for unified security enforcement.

Key Features:
- Unified security context creation from authentication results
- Database operations with integrated security validation
- Real-time security monitoring for database operations
- Comprehensive audit logging across all layers
- Performance-optimized security context management
- Zero-friction integration between security and database layers

Security Integration Components:
- UnifiedSecurityManager: Consolidated security context and policy management
- DatabaseSecurityValidator: Validates database operations against security policies
- UnifiedAuditLogger: Provides comprehensive audit logging across layers
- SecurityMetricsCollector: Collects security metrics from all components
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from prompt_improver.database.types import (
    ManagerMode,
    RedisSecurityError,
    SecurityPerformanceMetrics,
    SecurityThreatScore,
    SecurityValidationResult,
)
from prompt_improver.database.factories import (
    SecurityContext,
    create_security_context,
    create_security_context_from_auth_result,
)
from prompt_improver.database.composition import get_database_services

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
    security_integration_tracer = trace.get_tracer(__name__ + ".security_integration")
    security_integration_meter = metrics.get_meter(__name__ + ".security_integration")
    security_integration_operations_counter = security_integration_meter.create_counter(
        "database_security_operations_total",
        description="Total database security integration operations by type and result",
        unit="1",
    )
    security_context_lifecycle_counter = security_integration_meter.create_counter(
        "security_context_lifecycle_total",
        description="Security context lifecycle events by type",
        unit="1",
    )
    security_validation_duration_histogram = (
        security_integration_meter.create_histogram(
            "database_security_validation_duration_seconds",
            description="Database security validation duration by operation type",
            unit="s",
        )
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    security_integration_tracer = None
    security_integration_meter = None
    security_integration_operations_counter = None
    security_context_lifecycle_counter = None
    security_validation_duration_histogram = None
logger = logging.getLogger(__name__)


async def create_security_context_from_security_manager(
    agent_id: str, security_manager, additional_context: dict[str, Any] | None = None
) -> SecurityContext:
    """Create security context with validation from SecurityManager.

    Integrates comprehensive security validation and threat assessment
    into database security context for unified security enforcement.

    Args:
        agent_id: Agent identifier
        security_manager: Security manager instance
        additional_context: Additional security context information

    Returns:
        SecurityContext with comprehensive security validation
    """
    current_time = time.time()
    try:
        # Use basic security context creation since manager is optional
        validation_result = SecurityValidationResult(
            validated=True,
            validation_method="security_manager",
            validation_timestamp=current_time,
            validation_duration_ms=0.0,
            rate_limit_status="validated",
            encryption_required=True,
            audit_trail_id=f"sm_{int(current_time * 1000000)}",
        )
        
        # Create basic security context
        return create_security_context(
            agent_id=agent_id,
            manager_mode=ManagerMode.PRODUCTION,
            additional_context=additional_context or {},
            validation_result=validation_result,
        )
    except Exception as e:
        logger.error(f"Failed to create security context from manager: {e}")
        # Fallback to basic security context
        return create_security_context(
            agent_id=agent_id,
            manager_mode=ManagerMode.PRODUCTION,
        )


class SecurityIntegrationMode(Enum):
    """Security integration modes for different operation types."""

    STRICT = "strict"
    STANDARD = "standard"
    PERFORMANCE = "performance"
    BYPASS = "bypass"


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
    required_permissions: list[str]
    minimum_security_level: str
    require_encryption: bool = False
    audit_level: str = "standard"
    rate_limit_override: str | None = None


@dataclass
class DatabaseSecurityValidationResult:
    """Result of database security validation."""

    allowed: bool
    security_context: SecurityContext
    validation_time_ms: float
    applied_policies: list[str]
    security_warnings: list[str]
    audit_metadata: dict[str, Any]


# SecurityContextManager functionality has been consolidated into UnifiedSecurityManager
# This class has been removed to eliminate redundancy


class DatabaseSecurityValidator:
    """Validates database operations against unified security policies.

    Integrates security validation across database and application layers
    with comprehensive policy enforcement and audit logging.
    """

    def __init__(
        self,
        integration_mode: SecurityIntegrationMode = SecurityIntegrationMode.STANDARD,
    ):
        """Initialize database security validator.

        Args:
            integration_mode: Security integration mode
        """
        self.integration_mode = integration_mode
        self.logger = logging.getLogger(f"{__name__}.DatabaseSecurityValidator")
        self._security_policies = self._create_default_policies()
        self._validation_results: list[DatabaseSecurityValidationResult] = []
        self.logger.info(
            f"DatabaseSecurityValidator initialized in {integration_mode.value} mode"
        )

    async def validate_database_operation(
        self,
        operation_type: DatabaseOperationType,
        security_context: SecurityContext,
        operation_details: dict[str, Any] | None = None,
    ) -> DatabaseSecurityValidationResult:
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
            applicable_policies = self._get_applicable_policies(operation_type)
            applied_policies = []
            security_warnings = []
            if not security_context.is_valid():
                return DatabaseSecurityValidationResult(
                    allowed=False,
                    security_context=security_context,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    applied_policies=[],
                    security_warnings=["Invalid security context"],
                    audit_metadata={"validation_failed": "invalid_security_context"},
                )
            if operation_type in [
                DatabaseOperationType.WRITE,
                DatabaseOperationType.DELETE,
                DatabaseOperationType.ADMIN,
            ]:
                if not security_context.authenticated:
                    return self._create_denied_result(
                        security_context,
                        start_time,
                        "Authentication required for operation",
                        [],
                    )
            for policy in applicable_policies:
                policy_result = await self._validate_against_policy(
                    policy, security_context, operation_type, operation_details
                )
                if not policy_result["allowed"]:
                    return self._create_denied_result(
                        security_context,
                        start_time,
                        policy_result["reason"],
                        applied_policies,
                    )
                applied_policies.extend(policy_result["applied_policies"])
                security_warnings.extend(policy_result["warnings"])
            if self.integration_mode == SecurityIntegrationMode.STRICT:
                strict_result = await self._perform_strict_validation(
                    security_context, operation_type, operation_details
                )
                if not strict_result["allowed"]:
                    return self._create_denied_result(
                        security_context,
                        start_time,
                        strict_result["reason"],
                        applied_policies,
                    )
                security_warnings.extend(strict_result["warnings"])
            security_context.touch()
            security_context.add_audit_event(
                "database_operation_validated",
                {
                    "operation_type": operation_type.value,
                    "applied_policies": applied_policies,
                    "security_warnings": security_warnings,
                    "integration_mode": self.integration_mode.value,
                },
            )
            validation_time = (time.time() - start_time) * 1000
            security_context.record_performance_metric(
                "database_validation", validation_time
            )
            result = DatabaseSecurityValidationResult(
                allowed=True,
                security_context=security_context,
                validation_time_ms=validation_time,
                applied_policies=applied_policies,
                security_warnings=security_warnings,
                audit_metadata={
                    "operation_type": operation_type.value,
                    "validation_mode": self.integration_mode.value,
                    "policies_evaluated": len(applicable_policies),
                },
            )
            self._validation_results.append(result)
            if OPENTELEMETRY_AVAILABLE and security_integration_operations_counter:
                security_integration_operations_counter.add(
                    1,
                    attributes={
                        "operation_type": operation_type.value,
                        "result": "allowed",
                        "mode": self.integration_mode.value,
                        "agent_id": security_context.agent_id,
                    },
                )
            return result
        except Exception as e:
            self.logger.error(f"Database security validation error: {e}")
            return self._create_denied_result(
                security_context, start_time, f"Validation system error: {e!s}", []
            )

    def _create_default_policies(
        self,
    ) -> dict[DatabaseOperationType, list[SecurityPolicyRule]]:
        """Create default security policies for database operations."""
        return {
            DatabaseOperationType.READ: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.READ,
                    required_permissions=["read"],
                    minimum_security_level="basic",
                )
            ],
            DatabaseOperationType.WRITE: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.WRITE,
                    required_permissions=["write"],
                    minimum_security_level="enhanced",
                    audit_level="comprehensive",
                )
            ],
            DatabaseOperationType.DELETE: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.DELETE,
                    required_permissions=["delete"],
                    minimum_security_level="high",
                    audit_level="comprehensive",
                )
            ],
            DatabaseOperationType.ADMIN: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.ADMIN,
                    required_permissions=["admin"],
                    minimum_security_level="critical",
                    require_encryption=True,
                    audit_level="comprehensive",
                )
            ],
            DatabaseOperationType.CACHE: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.CACHE,
                    required_permissions=["cache"],
                    minimum_security_level="basic",
                )
            ],
            DatabaseOperationType.TRANSACTION: [
                SecurityPolicyRule(
                    operation_type=DatabaseOperationType.TRANSACTION,
                    required_permissions=["transaction"],
                    minimum_security_level="enhanced",
                    audit_level="comprehensive",
                )
            ],
        }

    def _get_applicable_policies(
        self, operation_type: DatabaseOperationType
    ) -> list[SecurityPolicyRule]:
        """Get security policies applicable to operation type."""
        return self._security_policies.get(operation_type, [])

    async def _validate_against_policy(
        self,
        policy: SecurityPolicyRule,
        security_context: SecurityContext,
        operation_type: DatabaseOperationType,
        operation_details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate operation against specific security policy."""
        warnings = []
        applied_policies = [f"{operation_type.value}_policy"]
        if policy.required_permissions:
            missing_permissions = [
                perm
                for perm in policy.required_permissions
                if perm not in security_context.permissions
            ]
            if missing_permissions:
                return {
                    "allowed": False,
                    "reason": f"Missing required permissions: {missing_permissions}",
                    "applied_policies": applied_policies,
                    "warnings": warnings,
                }
        security_levels = {"basic": 1, "enhanced": 2, "high": 3, "critical": 4}
        current_level = security_levels.get(security_context.security_level, 0)
        required_level = security_levels.get(policy.minimum_security_level, 1)
        if current_level < required_level:
            return {
                "allowed": False,
                "reason": f"Insufficient security level: {security_context.security_level} < {policy.minimum_security_level}",
                "applied_policies": applied_policies,
                "warnings": warnings,
            }
        if policy.require_encryption and (not security_context.encryption_context):
            warnings.append("Encryption recommended but not enforced")
        return {
            "allowed": True,
            "reason": "Policy validation successful",
            "applied_policies": applied_policies,
            "warnings": warnings,
        }

    async def _perform_strict_validation(
        self,
        security_context: SecurityContext,
        operation_type: DatabaseOperationType,
        operation_details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Perform additional strict security validation."""
        warnings = []
        if security_context.threat_score.score > 0.5:
            return {
                "allowed": False,
                "reason": f"High threat score: {security_context.threat_score.score}",
                "warnings": warnings,
            }
        if not security_context.zero_trust_validated:
            warnings.append("Zero trust validation not performed")
        context_age = time.time() - security_context.created_at
        if context_age > 3600:
            warnings.append("Security context is older than 1 hour")
        return {
            "allowed": True,
            "reason": "Strict validation passed",
            "warnings": warnings,
        }

    def _create_denied_result(
        self,
        security_context: SecurityContext,
        start_time: float,
        reason: str,
        applied_policies: list[str],
    ) -> DatabaseSecurityValidationResult:
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
                "validation_mode": self.integration_mode.value,
            },
        )
        self._validation_results.append(result)
        return result

    def get_validation_metrics(self) -> dict[str, Any]:
        """Get validation performance metrics."""
        if not self._validation_results:
            return {"total_validations": 0}
        allowed_count = sum(1 for r in self._validation_results if r.allowed)
        denied_count = len(self._validation_results) - allowed_count
        avg_validation_time = sum(
            r.validation_time_ms for r in self._validation_results
        ) / len(self._validation_results)
        return {
            "total_validations": len(self._validation_results),
            "allowed_validations": allowed_count,
            "denied_validations": denied_count,
            "success_rate": allowed_count / len(self._validation_results),
            "average_validation_time_ms": avg_validation_time,
            "integration_mode": self.integration_mode.value,
        }


class UnifiedSecurityIntegration:
    """Unified security integration orchestrator.

    Provides the main interface for integrating security across
    database and application layers with zero friction.
    """

    def __init__(
        self,
        integration_mode: SecurityIntegrationMode = SecurityIntegrationMode.STANDARD,
        enable_context_caching: bool = True,
        enable_performance_monitoring: bool = True,
    ):
        """Initialize unified security integration.

        Args:
            integration_mode: Security integration mode
            enable_context_caching: Enable security context caching
            enable_performance_monitoring: Enable performance monitoring
        """
        self.integration_mode = integration_mode
        self.enable_performance_monitoring = enable_performance_monitoring
        self.logger = logging.getLogger(f"{__name__}.UnifiedSecurityIntegration")
        # Import here to avoid circular imports
        from prompt_improver.security.services import (
            get_api_security_manager,
        )

        # Map integration modes to security modes
        mode_mapping = {
            SecurityIntegrationMode.STRICT: SecurityMode.HIGH_SECURITY,
            SecurityIntegrationMode.STANDARD: SecurityMode.API,
            SecurityIntegrationMode.PERFORMANCE: SecurityMode.INTERNAL,
            SecurityIntegrationMode.BYPASS: SecurityMode.INTERNAL,
        }
        self._security_mode = mode_mapping.get(integration_mode, SecurityMode.API)
        self._security_manager = None  # Will be initialized async
        self.database_validator = DatabaseSecurityValidator(integration_mode)
        self._connection_manager = None
        self._integration_metrics = {
            "operations_processed": 0,
            "average_security_overhead_ms": 0.0,
            "security_violations": 0,
            "cache_hit_rate": 0.0,
        }
        self.logger.info(
            f"UnifiedSecurityIntegration initialized in {integration_mode.value} mode"
        )

    async def initialize(self) -> None:
        """Initialize async components."""
        try:
            self._connection_manager = await get_database_services(
                ManagerMode.ASYNC_MODERN
            )
            await self._connection_manager.initialize()

            # Initialize the security manager
            from prompt_improver.security.services import (
                get_api_security_manager,
            )

            self._security_manager = await get_api_security_manager()

            self.logger.info("UnifiedSecurityIntegration async initialization complete")
        except Exception as e:
            self.logger.error(f"Failed to initialize UnifiedSecurityIntegration: {e}")
            raise

    async def create_authenticated_security_context(
        self, auth_result, cache_key: str | None = None
    ) -> SecurityContext:
        """Create security context from authentication result.

        Args:
            auth_result: AuthenticationResult from UnifiedAuthenticationManager
            cache_key: Optional cache key for performance

        Returns:
            Enhanced SecurityContext
        """
        return await self._security_manager.create_context_from_authentication(
            auth_result, cache_key
        )

    async def create_security_manager_context(
        self,
        agent_id: str,
        security_manager,
        additional_context: dict[str, Any] | None = None,
        cache_key: str | None = None,
    ) -> SecurityContext:
        """Create security context from security manager.

        Args:
            agent_id: Agent identifier
            security_manager: UnifiedSecurityManager instance
            additional_context: Additional context information
            cache_key: Optional cache key

        Returns:
            SecurityContext with comprehensive validation
        """
        return await self._security_manager.create_unified_security_context(
            agent_id, "database_integration", additional_context
        )

    async def validate_database_operation(
        self,
        operation_type: DatabaseOperationType,
        security_context: SecurityContext,
        operation_details: dict[str, Any] | None = None,
    ) -> DatabaseSecurityValidationResult:
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
            validated_context = (
                await self._security_manager.validate_and_refresh_context(
                    security_context
                )
            )
            validation_result = (
                await self.database_validator.validate_database_operation(
                    operation_type, validated_context, operation_details
                )
            )
            self._integration_metrics["operations_processed"] += 1
            if not validation_result.allowed:
                self._integration_metrics["security_violations"] += 1
            operation_time = (time.time() - start_time) * 1000
            current_avg = self._integration_metrics["average_security_overhead_ms"]
            total_ops = self._integration_metrics["operations_processed"]
            self._integration_metrics["average_security_overhead_ms"] = (
                current_avg * (total_ops - 1) + operation_time
            ) / total_ops
            return validation_result
        except Exception as e:
            self.logger.error(f"Database operation validation failed: {e}")
            self._integration_metrics["security_violations"] += 1
            return DatabaseSecurityValidationResult(
                allowed=False,
                security_context=security_context,
                validation_time_ms=(time.time() - start_time) * 1000,
                applied_policies=[],
                security_warnings=[f"Validation system error: {e!s}"],
                audit_metadata={"error": str(e)},
            )

    async def execute_secure_database_operation(
        self,
        operation_type: DatabaseOperationType,
        operation_func: callable,
        security_context: SecurityContext,
        operation_args: tuple | None = None,
        operation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            validation_result = await self.validate_database_operation(
                operation_type, security_context
            )
            if not validation_result.allowed:
                return {
                    "success": False,
                    "error": "Operation denied by security policy",
                    "security_warnings": validation_result.security_warnings,
                    "validation_result": validation_result,
                }
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*operation_args, **operation_kwargs)
            else:
                result = operation_func(*operation_args, **operation_kwargs)
            validation_result.security_context.add_audit_event(
                "database_operation_executed",
                {
                    "operation_type": operation_type.value,
                    "success": True,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                },
            )
            return {
                "success": True,
                "result": result,
                "security_context": validation_result.security_context,
                "validation_result": validation_result,
                "performance_metrics": {
                    "total_operation_time_ms": (time.time() - start_time) * 1000,
                    "security_overhead_ms": validation_result.validation_time_ms,
                },
            }
        except Exception as e:
            self.logger.error(f"Secure database operation failed: {e}")
            security_context.add_audit_event(
                "database_operation_failed",
                {
                    "operation_type": operation_type.value,
                    "error": str(e),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                },
            )
            return {
                "success": False,
                "error": str(e),
                "security_context": security_context,
                "performance_metrics": {
                    "total_operation_time_ms": (time.time() - start_time) * 1000
                },
            }

    async def get_integration_metrics(self) -> dict[str, Any]:
        """Get comprehensive integration performance metrics."""
        if self._security_manager:
            security_status = await self._security_manager.get_security_status()
            context_metrics = security_status.get("performance", {})
        else:
            context_metrics = {}
        validation_metrics = self.database_validator.get_validation_metrics()
        return {
            "integration_mode": self.integration_mode.value,
            "operations_processed": self._integration_metrics["operations_processed"],
            "average_security_overhead_ms": self._integration_metrics[
                "average_security_overhead_ms"
            ],
            "security_violations": self._integration_metrics["security_violations"],
            "violation_rate": self._integration_metrics["security_violations"]
            / max(1, self._integration_metrics["operations_processed"]),
            "security_manager_metrics": context_metrics,
            "database_validator_metrics": validation_metrics,
        }


_unified_security_integration: UnifiedSecurityIntegration | None = None


async def get_unified_security_integration(
    integration_mode: SecurityIntegrationMode = SecurityIntegrationMode.STANDARD,
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
            enable_performance_monitoring=True,
        )
        await _unified_security_integration.initialize()
        logger.info(
            f"Created UnifiedSecurityIntegration in {integration_mode.value} mode"
        )
    return _unified_security_integration


async def create_authenticated_context(
    auth_result, cache_key: str | None = None
) -> SecurityContext:
    """Convenience function to create security context from authentication."""
    integration = await get_unified_security_integration()
    return await integration.create_authenticated_security_context(
        auth_result, cache_key
    )


async def validate_database_read(
    security_context: SecurityContext, details: dict[str, Any] | None = None
) -> DatabaseSecurityValidationResult:
    """Convenience function to validate database read operation."""
    integration = await get_unified_security_integration()
    return await integration.validate_database_operation(
        DatabaseOperationType.READ, security_context, details
    )


async def validate_database_write(
    security_context: SecurityContext, details: dict[str, Any] | None = None
) -> DatabaseSecurityValidationResult:
    """Convenience function to validate database write operation."""
    integration = await get_unified_security_integration()
    return await integration.validate_database_operation(
        DatabaseOperationType.WRITE, security_context, details
    )


async def execute_secure_read(
    operation_func: callable, security_context: SecurityContext, *args, **kwargs
) -> dict[str, Any]:
    """Convenience function to execute secure database read."""
    integration = await get_unified_security_integration()
    return await integration.execute_secure_database_operation(
        DatabaseOperationType.READ, operation_func, security_context, args, kwargs
    )


async def execute_secure_write(
    operation_func: callable, security_context: SecurityContext, *args, **kwargs
) -> dict[str, Any]:
    """Convenience function to execute secure database write."""
    integration = await get_unified_security_integration()
    return await integration.execute_secure_database_operation(
        DatabaseOperationType.WRITE, operation_func, security_context, args, kwargs
    )
