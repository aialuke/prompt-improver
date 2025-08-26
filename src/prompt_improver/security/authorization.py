"""Real authorization service implementation for production use and integration testing."""

from enum import Enum


class Permission(Enum):
    """Permission enum for fine-grained access control."""

    READ_MODELS = "read_models"
    WRITE_MODELS = "write_models"
    DELETE_MODELS = "delete_models"
    MANAGE_USERS = "manage_users"
    VIEW_USERS = "view_users"
    ACCESS_DIFFERENTIAL_PRIVACY = "access_differential_privacy"
    CONFIGURE_PRIVACY_BUDGET = "configure_privacy_budget"
    RUN_ADVERSARIAL_TESTS = "run_adversarial_tests"
    ADMIN_SYSTEM = "admin_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    CONFIGURE_SECURITY = "configure_security"


class Role(Enum):
    """Role enum with predefined permission sets."""

    user = "user"
    ML_ENGINEER = "ml_engineer"
    PRIVACY_OFFICER = "privacy_officer"
    SECURITY_ADMIN = "security_admin"
    admin = "admin"


class AuthorizationService:
    """Real authorization service that implements RBAC (Role-Based Access Control)."""

    def __init__(self) -> None:
        self.role_permissions: dict[Role, set[Permission]] = {
            Role.user: {Permission.READ_MODELS, Permission.VIEW_USERS},
            Role.ML_ENGINEER: {
                Permission.READ_MODELS,
                Permission.WRITE_MODELS,
                Permission.RUN_ADVERSARIAL_TESTS,
                Permission.VIEW_USERS,
            },
            Role.PRIVACY_OFFICER: {
                Permission.READ_MODELS,
                Permission.ACCESS_DIFFERENTIAL_PRIVACY,
                Permission.CONFIGURE_PRIVACY_BUDGET,
                Permission.VIEW_AUDIT_LOGS,
                Permission.VIEW_USERS,
            },
            Role.SECURITY_ADMIN: {
                Permission.READ_MODELS,
                Permission.WRITE_MODELS,
                Permission.RUN_ADVERSARIAL_TESTS,
                Permission.CONFIGURE_SECURITY,
                Permission.VIEW_AUDIT_LOGS,
                Permission.VIEW_USERS,
                Permission.MANAGE_USERS,
            },
            Role.admin: set(Permission),
        }
        self.user_roles: dict[str, set[Role]] = {}
        self.user_permissions: dict[str, set[Permission]] = {}

    def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign a role to a user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
        return True

    def revoke_role(self, user_id: str, role: Role) -> bool:
        """Revoke a role from a user."""
        if user_id in self.user_roles and role in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role)
            return True
        return False

    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant a specific permission to a user (overrides role permissions)."""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()
        self.user_permissions[user_id].add(permission)
        return True

    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Revoke a specific permission from a user."""
        if (
            user_id in self.user_permissions
            and permission in self.user_permissions[user_id]
        ):
            self.user_permissions[user_id].remove(permission)
            return True
        return False

    def get_user_roles(self, user_id: str) -> set[Role]:
        """Get all roles assigned to a user."""
        return self.user_roles.get(user_id, set())

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        """Get all effective permissions for a user (from roles + custom permissions)."""
        all_permissions = set()
        user_roles = self.get_user_roles(user_id)
        for role in user_roles:
            all_permissions.update(self.role_permissions.get(role, set()))
        all_permissions.update(self.user_permissions.get(user_id, set()))
        return all_permissions

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions

    def has_role(self, user_id: str, role: Role) -> bool:
        """Check if a user has a specific role."""
        user_roles = self.get_user_roles(user_id)
        return role in user_roles

    def has_any_role(self, user_id: str, roles: list[Role]) -> bool:
        """Check if a user has any of the specified roles."""
        user_roles = self.get_user_roles(user_id)
        return any(role in user_roles for role in roles)

    def has_all_permissions(self, user_id: str, permissions: list[Permission]) -> bool:
        """Check if a user has all of the specified permissions."""
        user_permissions = self.get_user_permissions(user_id)
        return all(permission in user_permissions for permission in permissions)

    def can_access_resource(
        self, user_id: str, resource_type: str, action: str
    ) -> bool:
        """Check if user can perform action on resource type."""
        permission_map = {
            ("models", "read"): Permission.READ_MODELS,
            ("models", "write"): Permission.WRITE_MODELS,
            ("models", "delete"): Permission.DELETE_MODELS,
            ("users", "read"): Permission.VIEW_USERS,
            ("users", "write"): Permission.MANAGE_USERS,
            ("users", "delete"): Permission.MANAGE_USERS,
            ("privacy", "access"): Permission.ACCESS_DIFFERENTIAL_PRIVACY,
            ("privacy", "configure"): Permission.CONFIGURE_PRIVACY_BUDGET,
            ("security", "test"): Permission.RUN_ADVERSARIAL_TESTS,
            ("security", "configure"): Permission.CONFIGURE_SECURITY,
            ("audit", "view"): Permission.VIEW_AUDIT_LOGS,
            ("system", "admin"): Permission.ADMIN_SYSTEM,
        }
        required_permission = permission_map.get((resource_type, action))
        if required_permission is None:
            return False
        return self.has_permission(user_id, required_permission)

    def get_accessible_resources(self, user_id: str) -> dict[str, list[str]]:
        """Get all resources and actions accessible to a user."""
        permissions = self.get_user_permissions(user_id)
        accessible = {}
        permission_to_resource = {
            Permission.READ_MODELS: ("models", "read"),
            Permission.WRITE_MODELS: ("models", "write"),
            Permission.DELETE_MODELS: ("models", "delete"),
            Permission.VIEW_USERS: ("users", "read"),
            Permission.MANAGE_USERS: ("users", "write"),
            Permission.ACCESS_DIFFERENTIAL_PRIVACY: ("privacy", "access"),
            Permission.CONFIGURE_PRIVACY_BUDGET: ("privacy", "configure"),
            Permission.RUN_ADVERSARIAL_TESTS: ("security", "test"),
            Permission.CONFIGURE_SECURITY: ("security", "configure"),
            Permission.VIEW_AUDIT_LOGS: ("audit", "view"),
            Permission.ADMIN_SYSTEM: ("system", "admin"),
        }
        for permission in permissions:
            if permission in permission_to_resource:
                resource, action = permission_to_resource[permission]
                if resource not in accessible:
                    accessible[resource] = []
                accessible[resource].append(action)
        return accessible
