"""Factory functions for database services and security contexts.

This module provides factory functions to create and manage database services
and security contexts following the clean architecture pattern established
by the service composition layer.

Key responsibilities:
- Security context creation and management
- Service factory functions for dependency injection
- Configuration-based service instantiation
- Thread-safe singleton management for shared resources
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import ManagerMode, PoolConfiguration, ServiceConfiguration

logger = logging.getLogger(__name__)


class SecurityTier(Enum):
    """Security tiers for cache operations and service access."""

    BASIC = "basic"
    STANDARD = "standard"
    PRIVILEGED = "privileged"
    ADMIN = "admin"


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
    tier: SecurityTier = SecurityTier.BASIC
    authenticated: bool = False
    created_at: float = field(default_factory=time.time)
    authentication_method: str = "none"
    authentication_timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    validation_result: Optional["SecurityValidationResult"] = None
    threat_score: Optional["SecurityThreatScore"] = None
    performance_metrics: Optional["SecurityPerformanceMetrics"] = None
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[str] = field(default_factory=list)
    security_level: str = "basic"
    zero_trust_validated: bool = False
    encryption_context: Optional[Dict[str, str]] = None
    expires_at: Optional[float] = None
    max_operations: Optional[int] = None
    operations_count: int = 0
    last_used: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize security context with default permissions."""
        if self.expires_at is None:
            self.expires_at = self.created_at + 3600  # 1 hour default

        # Set tier-based permissions if not provided
        if not self.permissions:
            self.permissions = self._get_tier_permissions()

        # Initialize validation result if not provided
        if self.validation_result is None:
            from .types import SecurityValidationResult

            self.validation_result = SecurityValidationResult()

        # Initialize threat score if not provided
        if self.threat_score is None:
            from .types import SecurityThreatScore

            self.threat_score = SecurityThreatScore()

        # Initialize performance metrics if not provided
        if self.performance_metrics is None:
            from .types import SecurityPerformanceMetrics

            self.performance_metrics = SecurityPerformanceMetrics()

    def _get_tier_permissions(self) -> List[str]:
        """Get permissions based on security tier."""
        base_permissions = ["cache_read"]

        if self.tier == SecurityTier.BASIC:
            base_permissions.extend(["cache_write"])
        elif self.tier == SecurityTier.STANDARD:
            base_permissions.extend(["cache_write", "database_read"])
        elif self.tier == SecurityTier.PRIVILEGED:
            base_permissions.extend(["cache_write", "database_read", "database_write"])
        elif self.tier == SecurityTier.ADMIN:
            base_permissions.extend([
                "cache_write",
                "database_read",
                "database_write",
                "admin_operations",
            ])

        return base_permissions

    def is_valid(self) -> bool:
        """Check if security context is still valid."""
        current_time = time.time()
        if self.expires_at and current_time > self.expires_at:
            return False
        if self.max_operations and self.operations_count >= self.max_operations:
            return False
        return True

    def is_expired(self) -> bool:
        """Check if security context has expired."""
        return not self.is_valid()

    def can_perform(self, operation: str) -> bool:
        """Check if context has permission for operation."""
        if not self.is_valid() or not self.authenticated:
            return False
        return operation in self.permissions

    def increment_operations(self) -> None:
        """Increment operations count and update last used timestamp."""
        self.operations_count += 1
        self.last_used = time.time()


# Global security context cache
_security_contexts: Dict[str, SecurityContext] = {}
_context_lock = asyncio.Lock()


async def create_security_context(
    agent_id: str,
    tier: str,
    authenticated: bool = True,
    session_id: Optional[str] = None,
    custom_permissions: Optional[Dict[str, bool]] = None,
) -> SecurityContext:
    """Create a new security context for service operations.

    Args:
        agent_id: Unique identifier for the agent/service
        tier: Security tier (basic, professional, enterprise, admin)
        authenticated: Whether the context is authenticated
        session_id: Optional session identifier
        custom_permissions: Optional custom permissions override

    Returns:
        SecurityContext instance for service authentication
    """
    async with _context_lock:
        try:
            security_tier = SecurityTier(tier.lower())
        except ValueError:
            logger.warning(f"Invalid security tier '{tier}', defaulting to BASIC")
            security_tier = SecurityTier.BASIC

        context = SecurityContext(
            agent_id=agent_id,
            tier=security_tier,
            authenticated=authenticated,
            permissions=custom_permissions or {},
            session_id=session_id,
        )

        # Cache context for reuse
        context_key = f"{agent_id}:{tier}:{session_id or 'default'}"
        _security_contexts[context_key] = context

        logger.debug(f"Created security context for {agent_id} with tier {tier}")
        return context


async def get_security_context(
    agent_id: str, tier: str, session_id: Optional[str] = None
) -> Optional[SecurityContext]:
    """Get existing security context if available and not expired.

    Args:
        agent_id: Agent identifier
        tier: Security tier
        session_id: Optional session identifier

    Returns:
        Existing SecurityContext or None if not found/expired
    """
    async with _context_lock:
        context_key = f"{agent_id}:{tier}:{session_id or 'default'}"
        context = _security_contexts.get(context_key)

        if context and not context.is_expired():
            return context
        elif context:
            # Remove expired context
            del _security_contexts[context_key]

        return None


async def create_service_configuration(
    mode: ManagerMode, overrides: Optional[Dict[str, Any]] = None
) -> ServiceConfiguration:
    """Create service configuration for a specific manager mode.

    Args:
        mode: Manager operation mode
        overrides: Optional configuration overrides

    Returns:
        ServiceConfiguration with mode-specific settings
    """
    base_config: ServiceConfiguration = {
        "mode": mode.value,
        "enable_caching": True,
        "enable_health_checks": True,
        "max_connections": 20,
        "timeout_seconds": 30,
        "retry_attempts": 3,
        "debug_logging": False,
    }

    # Mode-specific configurations
    mode_configs = {
        ManagerMode.MCP_SERVER: {
            "max_connections": 20,
            "timeout_seconds": 10,
            "debug_logging": False,
        },
        ManagerMode.ML_TRAINING: {
            "max_connections": 15,
            "timeout_seconds": 60,
            "retry_attempts": 5,
            "debug_logging": True,
        },
        ManagerMode.ADMIN: {
            "max_connections": 5,
            "timeout_seconds": 30,
            "enable_health_checks": True,
            "debug_logging": True,
        },
        ManagerMode.ASYNC_MODERN: {
            "max_connections": 12,
            "timeout_seconds": 20,
            "retry_attempts": 3,
        },
        ManagerMode.HIGH_AVAILABILITY: {
            "max_connections": 30,
            "timeout_seconds": 45,
            "retry_attempts": 5,
            "enable_health_checks": True,
        },
    }

    # Apply mode-specific config
    if mode in mode_configs:
        base_config.update(mode_configs[mode])

    # Apply user overrides
    if overrides:
        base_config.update(overrides)

    return base_config


async def cleanup_security_contexts():
    """Clean up expired security contexts."""
    async with _context_lock:
        if not _security_contexts:
            return

        expired_keys = []
        for key, context in _security_contexts.items():
            if context.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            del _security_contexts[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired security contexts")


# Periodic cleanup task management
_cleanup_task: Optional[asyncio.Task] = None


async def start_security_context_cleanup():
    """Start periodic cleanup of expired security contexts."""
    global _cleanup_task

    if _cleanup_task and not _cleanup_task.done():
        return

    async def cleanup_loop():
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await cleanup_security_contexts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in security context cleanup: {e}")

    _cleanup_task = asyncio.create_task(cleanup_loop())
    logger.info("Started security context cleanup task")


async def stop_security_context_cleanup():
    """Stop periodic cleanup of security contexts."""
    global _cleanup_task

    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None
        logger.info("Stopped security context cleanup task")


async def create_security_context_from_auth_result(
    agent_id: str,
    tier: str,
    authenticated: bool = True,
    auth_result: Optional[Dict[str, Any]] = None,
) -> SecurityContext:
    """Create security context from authentication result.

    This is a compatibility function for existing code that needs
    to create security contexts from authentication results.

    Args:
        agent_id: Agent identifier
        tier: Security tier
        authenticated: Authentication status
        auth_result: Authentication result data (optional)

    Returns:
        SecurityContext instance
    """
    # Extract session info from auth result if available
    session_id = None
    custom_permissions = None

    if auth_result:
        session_id = auth_result.get("session_id")
        custom_permissions = auth_result.get("permissions")

    return await create_security_context(
        agent_id=agent_id,
        tier=tier,
        authenticated=authenticated,
        session_id=session_id,
        custom_permissions=custom_permissions,
    )
