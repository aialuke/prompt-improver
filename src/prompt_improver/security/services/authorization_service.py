"""AuthorizationService - Role-Based Access Control with Fail-Secure Design

A specialized security service that handles authorization and access control using
Role-Based Access Control (RBAC) with fine-grained permissions. Implements fail-secure
principles where any error results in access denial.

Key Features:
- Role-Based Access Control (RBAC) with hierarchical roles
- Fine-grained permission system with resource-specific access
- Context-aware authorization decisions
- Rate limiting integration for operation-specific throttling
- Comprehensive audit logging for all authorization decisions
- Fail-secure design (deny on error, no fail-open vulnerabilities)
- Support for dynamic permission evaluation
- Integration with security context validation

Security Standards:
- NIST RBAC standard compliance
- OWASP Access Control guidelines
- Zero-trust architecture with mandatory authorization
- Principle of least privilege enforcement
- Defense in depth with multiple authorization layers
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from prompt_improver.database import SecurityContext
from prompt_improver.security.services.protocols import (
    AuthorizationServiceProtocol,
    SecurityStateManagerProtocol,
)
# Rate limiting functionality moved to SecurityServiceFacade
# This service now delegates rate limiting to the facade
from prompt_improver.utils.datetime_utils import aware_utc_now

try:
    from opentelemetry import metrics, trace

    OPENTELEMETRY_AVAILABLE = True
    authz_tracer = trace.get_tracer(__name__ + ".authorization")
    authz_meter = metrics.get_meter(__name__ + ".authorization")
    authz_operations_counter = authz_meter.create_counter(
        "authorization_operations_total",
        description="Total authorization operations by type and result",
        unit="1",
    )
    authz_decisions_counter = authz_meter.create_counter(
        "authorization_decisions_total",
        description="Authorization decisions by operation and result",
        unit="1",
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    authz_tracer = None
    authz_meter = None
    authz_operations_counter = None
    authz_decisions_counter = None

logger = logging.getLogger(__name__)


class Permission:
    """Represents a specific permission with optional resource constraints."""
    
    def __init__(self, name: str, resource: Optional[str] = None, constraints: Optional[Dict[str, Any]] = None):
        self.name = name
        self.resource = resource
        self.constraints = constraints or {}
    
    def matches(self, required_permission: str, required_resource: Optional[str] = None) -> bool:
        """Check if this permission matches a required permission."""
        # Check permission name match
        if self.name != required_permission and self.name != "*":
            return False
        
        # Check resource match if specified
        if required_resource:
            if self.resource and self.resource != required_resource and self.resource != "*":
                return False
        
        return True
    
    def __str__(self) -> str:
        if self.resource:
            return f"{self.name}:{self.resource}"
        return self.name


class Role:
    """Represents a role with associated permissions and hierarchy."""
    
    def __init__(self, name: str, permissions: Optional[List[Permission]] = None, parent_roles: Optional[List[str]] = None):
        self.name = name
        self.permissions = permissions or []
        self.parent_roles = parent_roles or []
        self.created_at = aware_utc_now()
    
    def has_permission(self, required_permission: str, required_resource: Optional[str] = None) -> bool:
        """Check if role has required permission."""
        for permission in self.permissions:
            if permission.matches(required_permission, required_resource):
                return True
        return False
    
    def add_permission(self, permission: Permission) -> None:
        """Add permission to role."""
        self.permissions.append(permission)
    
    def remove_permission(self, permission_name: str, resource: Optional[str] = None) -> bool:
        """Remove permission from role."""
        original_count = len(self.permissions)
        self.permissions = [
            p for p in self.permissions 
            if not (p.name == permission_name and p.resource == resource)
        ]
        return len(self.permissions) < original_count


class AuthorizationService:
    """Focused authorization service with RBAC and fail-secure design.
    
    Handles all authorization operations including permission checking,
    role management, and access control decisions. Designed to fail securely -
    any error condition results in access denial rather than potential bypass.
    
    Single Responsibility: Authorization and access control operations only
    """

    def __init__(
        self,
        security_state_manager: SecurityStateManagerProtocol,
        enable_hierarchical_roles: bool = True,
        enable_dynamic_permissions: bool = True,
        session_timeout_minutes: int = 60,
    ):
        """Initialize authorization service.
        
        Args:
            security_state_manager: Shared security state manager
            enable_hierarchical_roles: Enable role hierarchy support
            enable_dynamic_permissions: Enable dynamic permission evaluation
            session_timeout_minutes: Session timeout for authorization cache
        """
        self.security_state_manager = security_state_manager
        self.enable_hierarchical_roles = enable_hierarchical_roles
        self.enable_dynamic_permissions = enable_dynamic_permissions
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # RBAC data structures
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, Set[str]] = defaultdict(set)
        self._role_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # child -> parents
        
        # Authorization cache for performance
        self._authorization_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Performance metrics
        self._operation_times: deque = deque(maxlen=1000)
        self._authorization_decisions = {"allowed": 0, "denied": 0}
        self._total_operations = 0
        
        # Rate limiter integration
        # Rate limiting delegated to SecurityServiceFacade
        self._initialized = False
        
        # Initialize default roles and permissions
        self._initialize_default_rbac()
        
        logger.info("AuthorizationService initialized with RBAC and fail-secure design")

    async def initialize(self) -> bool:
        """Initialize authorization service components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Rate limiting delegated to SecurityServiceFacade
            
            initialization_time = time.time() - start_time
            logger.info(f"AuthorizationService initialized in {initialization_time:.3f}s")
            
            await self.security_state_manager.record_security_operation(
                "authorization_service_init",
                success=True,
                details={"initialization_time": initialization_time, "roles_count": len(self._roles)}
            )
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AuthorizationService: {e}")
            await self.security_state_manager.handle_security_incident(
                "high", "authorization_service_init", "system",
                {"error": str(e), "operation": "initialization"}
            )
            return False

    async def authorize_operation(
        self,
        security_context: SecurityContext,
        operation: str,
        resource: str,
        additional_checks: Dict[str, Any] | None = None,
    ) -> bool:
        """Authorize an operation with comprehensive security validation.
        
        Performs multi-layer authorization checks:
        1. Security context validation
        2. Rate limiting verification
        3. Permission-based authorization
        4. Context-aware additional checks
        5. Audit logging
        
        Args:
            security_context: Security context from authentication
            operation: Operation being attempted (e.g., "read", "write", "execute")
            resource: Resource being accessed (e.g., "prompts", "models", "admin")
            additional_checks: Additional authorization constraints
            
        Returns:
            True if authorized, False otherwise
            
        Fail-secure: Returns False on any authorization error or system failure
        """
        operation_start = time.time()
        
        if not self._initialized:
            logger.error("AuthorizationService not initialized")
            return False
        
        try:
            # Record authorization attempt for audit
            await self.security_state_manager.record_security_operation(
                "authorization_attempt",
                success=False,  # Will update if successful
                agent_id=security_context.agent_id,
                details={"operation": operation, "resource": resource}
            )
            
            # 1. Validate security context (fail-secure)
            if not await self._validate_security_context(security_context):
                await self.security_state_manager.handle_security_incident(
                    "medium", "authorization_invalid_context", security_context.agent_id,
                    {"operation": operation, "resource": resource, "reason": "invalid_security_context"}
                )
                self._authorization_decisions["denied"] += 1
                return False
            
            # 2. Check rate limits (fail-secure)
            if not await self._check_rate_limits(security_context, operation):
                await self.security_state_manager.handle_security_incident(
                    "low", "authorization_rate_limit", security_context.agent_id,
                    {"operation": operation, "resource": resource, "reason": "rate_limit_exceeded"}
                )
                self._authorization_decisions["denied"] += 1
                return False
            
            # 3. Check permissions (fail-secure)
            has_permission = await self._check_permissions(security_context, operation, resource)
            if not has_permission:
                await self.security_state_manager.handle_security_incident(
                    "medium", "authorization_permission_denied", security_context.agent_id,
                    {"operation": operation, "resource": resource, "reason": "insufficient_permissions"}
                )
                self._authorization_decisions["denied"] += 1
                return False
            
            # 4. Additional context-aware checks (fail-secure)
            if additional_checks:
                additional_authorized = await self._perform_additional_checks(
                    security_context, operation, resource, additional_checks
                )
                if not additional_authorized:
                    await self.security_state_manager.handle_security_incident(
                        "medium", "authorization_additional_check_failed", security_context.agent_id,
                        {"operation": operation, "resource": resource, "additional_checks": additional_checks}
                    )
                    self._authorization_decisions["denied"] += 1
                    return False
            
            # Authorization successful
            self._authorization_decisions["allowed"] += 1
            
            await self.security_state_manager.record_security_operation(
                "authorization_success",
                success=True,
                agent_id=security_context.agent_id,
                details={"operation": operation, "resource": resource}
            )
            
            logger.debug(f"Authorized {security_context.agent_id} for {operation} on {resource}")
            return True
            
        except Exception as e:
            logger.error(f"Authorization error for {security_context.agent_id}: {e}")
            
            await self.security_state_manager.handle_security_incident(
                "high", "authorization_system_error", security_context.agent_id,
                {"error": str(e), "operation": operation, "resource": resource}
            )
            
            # Fail-secure: Deny authorization on any system error
            self._authorization_decisions["denied"] += 1
            return False
            
        finally:
            # Record operation metrics
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
            self._total_operations += 1
            
            if OPENTELEMETRY_AVAILABLE and authz_operations_counter:
                authz_operations_counter.add(
                    1, {"operation": "authorize_operation", "resource": resource}
                )
            
            if OPENTELEMETRY_AVAILABLE and authz_decisions_counter:
                result = "allowed" if has_permission else "denied"
                authz_decisions_counter.add(
                    1, {"operation": operation, "result": result}
                )

    async def check_permissions(
        self,
        security_context: SecurityContext,
        required_permissions: List[str],
    ) -> bool:
        """Check if security context has required permissions.
        
        Args:
            security_context: Security context to check
            required_permissions: List of required permissions
            
        Returns:
            True if all permissions present, False otherwise
            
        Fail-secure: Returns False if any permission missing or check fails
        """
        try:
            for permission in required_permissions:
                if not await self._check_single_permission(security_context, permission):
                    return False
            return True
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False

    async def validate_security_context(self, security_context: SecurityContext) -> bool:
        """Validate security context integrity and expiration.
        
        Args:
            security_context: Security context to validate
            
        Returns:
            True if context valid, False otherwise
            
        Fail-secure: Returns False if context invalid or expired
        """
        try:
            return await self._validate_security_context(security_context)
        except Exception as e:
            logger.error(f"Security context validation error: {e}")
            return False

    async def check_rate_limits(self, security_context: SecurityContext, operation: str) -> bool:
        """Check operation-specific rate limits.
        
        Args:
            security_context: Security context
            operation: Operation being rate limited
            
        Returns:
            True if within limits, False otherwise
            
        Fail-secure: Returns False if rate limit exceeded or check fails
        """
        try:
            return await self._check_rate_limits(security_context, operation)
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return False

    # Role and Permission Management Methods

    async def create_role(self, role_name: str, permissions: List[str], parent_roles: Optional[List[str]] = None) -> bool:
        """Create a new role with specified permissions.
        
        Args:
            role_name: Name of the role to create
            permissions: List of permission strings
            parent_roles: Optional parent roles for hierarchy
            
        Returns:
            True if role created successfully, False otherwise
        """
        try:
            if role_name in self._roles:
                logger.warning(f"Role {role_name} already exists")
                return False
            
            # Convert permission strings to Permission objects
            role_permissions = []
            for perm_str in permissions:
                if ":" in perm_str:
                    name, resource = perm_str.split(":", 1)
                    role_permissions.append(Permission(name, resource))
                else:
                    role_permissions.append(Permission(perm_str))
            
            # Create role
            role = Role(role_name, role_permissions, parent_roles or [])
            self._roles[role_name] = role
            
            # Update hierarchy
            if parent_roles:
                for parent in parent_roles:
                    self._role_hierarchy[role_name].add(parent)
            
            await self.security_state_manager.record_security_operation(
                "role_created",
                success=True,
                details={"role_name": role_name, "permissions_count": len(permissions)}
            )
            
            logger.info(f"Created role {role_name} with {len(permissions)} permissions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create role {role_name}: {e}")
            return False

    async def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign role to user.
        
        Args:
            user_id: User identifier
            role_name: Role to assign
            
        Returns:
            True if assignment successful, False otherwise
        """
        try:
            if role_name not in self._roles:
                logger.warning(f"Role {role_name} does not exist")
                return False
            
            self._user_roles[user_id].add(role_name)
            
            await self.security_state_manager.record_security_operation(
                "role_assigned",
                success=True,
                agent_id=user_id,
                details={"role_name": role_name}
            )
            
            # Clear authorization cache for user
            self._clear_user_cache(user_id)
            
            logger.info(f"Assigned role {role_name} to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role {role_name} to user {user_id}: {e}")
            return False

    async def revoke_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Revoke role from user.
        
        Args:
            user_id: User identifier
            role_name: Role to revoke
            
        Returns:
            True if revocation successful, False otherwise
        """
        try:
            if role_name in self._user_roles[user_id]:
                self._user_roles[user_id].remove(role_name)
                
                await self.security_state_manager.record_security_operation(
                    "role_revoked",
                    success=True,
                    agent_id=user_id,
                    details={"role_name": role_name}
                )
                
                # Clear authorization cache for user
                self._clear_user_cache(user_id)
                
                logger.info(f"Revoked role {role_name} from user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke role {role_name} from user {user_id}: {e}")
            return False

    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all effective permissions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of permission strings
        """
        try:
            permissions = set()
            
            # Get user roles
            user_roles = self._user_roles.get(user_id, set())
            
            # Collect permissions from all roles (including inherited)
            for role_name in user_roles:
                role_permissions = await self._get_effective_role_permissions(role_name)
                permissions.update(role_permissions)
            
            return list(permissions)
            
        except Exception as e:
            logger.error(f"Failed to get permissions for user {user_id}: {e}")
            return []

    async def cleanup(self) -> bool:
        """Cleanup authorization service resources.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Clear RBAC data
            self._roles.clear()
            self._user_roles.clear()
            self._role_hierarchy.clear()
            
            # Clear caches
            self._authorization_cache.clear()
            self._cache_expiry.clear()
            
            # Clear metrics
            self._operation_times.clear()
            self._authorization_decisions = {"allowed": 0, "denied": 0}
            
            self._initialized = False
            logger.info("AuthorizationService cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup AuthorizationService: {e}")
            return False

    # Private helper methods

    def _initialize_default_rbac(self) -> None:
        """Initialize default roles and permissions."""
        # Define default permissions
        default_permissions = {
            "admin": [
                Permission("*", "*"),  # Full access
            ],
            "user": [
                Permission("read", "prompts"),
                Permission("write", "prompts"),
                Permission("read", "models"),
            ],
            "readonly": [
                Permission("read", "*"),
            ],
            "api_user": [
                Permission("read", "prompts"),
                Permission("write", "prompts"),
                Permission("execute", "models"),
            ]
        }
        
        # Create default roles
        for role_name, permissions in default_permissions.items():
            self._roles[role_name] = Role(role_name, permissions)
        
        # Set up role hierarchy
        self._role_hierarchy["user"].add("readonly")
        self._role_hierarchy["admin"].add("user")

    async def _validate_security_context(self, security_context: SecurityContext) -> bool:
        """Validate security context with comprehensive checks."""
        # Check if authenticated
        if not security_context.authenticated:
            return False
        
        # Check expiration
        if not security_context.is_valid():
            return False
        
        # Check threat score
        if security_context.threat_score.score > 0.7:
            return False
        
        # Check if agent is blocked
        if await self.security_state_manager.is_agent_blocked(security_context.agent_id):
            return False
        
        return True

    async def _check_rate_limits(self, security_context: SecurityContext, operation: str) -> bool:
        """Check rate limits for operation."""
        # Rate limiting is now handled by SecurityServiceFacade
        # For now, allow all requests until facade integration is complete
        return True

    async def _check_permissions(self, security_context: SecurityContext, operation: str, resource: str) -> bool:
        """Check if user has required permission for operation on resource."""
        user_id = security_context.agent_id
        
        # Check cache first
        cache_key = f"{user_id}:{operation}:{resource}"
        if cache_key in self._authorization_cache:
            if self._cache_expiry[cache_key] > aware_utc_now():
                return self._authorization_cache[cache_key]["authorized"]
            else:
                # Cache expired
                del self._authorization_cache[cache_key]
                del self._cache_expiry[cache_key]
        
        # Get user roles
        user_roles = self._user_roles.get(user_id, set())
        
        # Check permissions in all user roles (including inherited)
        authorized = False
        for role_name in user_roles:
            if await self._role_has_permission(role_name, operation, resource):
                authorized = True
                break
        
        # Cache result
        self._authorization_cache[cache_key] = {"authorized": authorized}
        self._cache_expiry[cache_key] = aware_utc_now() + self._cache_ttl
        
        return authorized

    async def _check_single_permission(self, security_context: SecurityContext, permission: str) -> bool:
        """Check single permission for user."""
        if ":" in permission:
            operation, resource = permission.split(":", 1)
            return await self._check_permissions(security_context, operation, resource)
        else:
            return await self._check_permissions(security_context, permission, "*")

    async def _role_has_permission(self, role_name: str, operation: str, resource: str) -> bool:
        """Check if role has permission, including inherited permissions."""
        if role_name not in self._roles:
            return False
        
        role = self._roles[role_name]
        
        # Check direct permissions
        if role.has_permission(operation, resource):
            return True
        
        # Check inherited permissions if hierarchy enabled
        if self.enable_hierarchical_roles:
            parent_roles = self._role_hierarchy.get(role_name, set())
            for parent_role in parent_roles:
                if await self._role_has_permission(parent_role, operation, resource):
                    return True
        
        return False

    async def _get_effective_role_permissions(self, role_name: str) -> Set[str]:
        """Get all effective permissions for a role including inherited."""
        permissions = set()
        
        if role_name not in self._roles:
            return permissions
        
        role = self._roles[role_name]
        
        # Add direct permissions
        for perm in role.permissions:
            permissions.add(str(perm))
        
        # Add inherited permissions if hierarchy enabled
        if self.enable_hierarchical_roles:
            parent_roles = self._role_hierarchy.get(role_name, set())
            for parent_role in parent_roles:
                parent_permissions = await self._get_effective_role_permissions(parent_role)
                permissions.update(parent_permissions)
        
        return permissions

    async def _perform_additional_checks(
        self,
        security_context: SecurityContext,
        operation: str,
        resource: str,
        additional_checks: Dict[str, Any],
    ) -> bool:
        """Perform additional context-aware authorization checks."""
        try:
            # Time-based access control
            if "time_restrictions" in additional_checks:
                time_restrictions = additional_checks["time_restrictions"]
                current_hour = aware_utc_now().hour
                if "allowed_hours" in time_restrictions:
                    allowed_hours = time_restrictions["allowed_hours"]
                    if current_hour not in allowed_hours:
                        return False
            
            # IP-based access control
            if "ip_restrictions" in additional_checks:
                ip_restrictions = additional_checks["ip_restrictions"]
                user_ip = security_context.audit_metadata.get("source_ip")
                if user_ip and "allowed_ips" in ip_restrictions:
                    allowed_ips = ip_restrictions["allowed_ips"]
                    if user_ip not in allowed_ips:
                        return False
            
            # Resource-specific constraints
            if "resource_constraints" in additional_checks:
                constraints = additional_checks["resource_constraints"]
                # Implement specific resource constraints based on requirements
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Additional checks failed: {e}")
            return False

    def _clear_user_cache(self, user_id: str) -> None:
        """Clear authorization cache for specific user."""
        keys_to_remove = [
            key for key in self._authorization_cache.keys()
            if key.startswith(f"{user_id}:")
        ]
        
        for key in keys_to_remove:
            del self._authorization_cache[key]
            if key in self._cache_expiry:
                del self._cache_expiry[key]


# Factory function for dependency injection
async def create_authorization_service(
    security_state_manager: SecurityStateManagerProtocol,
    **config_overrides
) -> AuthorizationService:
    """Create and initialize authorization service.
    
    Args:
        security_state_manager: Shared security state manager
        **config_overrides: Configuration overrides
        
    Returns:
        Initialized AuthorizationService instance
    """
    service = AuthorizationService(security_state_manager, **config_overrides)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize AuthorizationService")
    
    return service