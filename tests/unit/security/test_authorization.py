"""
Authorization Security Tests

Tests authorization and access control mechanisms for ML components to ensure
proper role-based access control (RBAC) and permission validation across
Phase 3 privacy-preserving and security features.

Security Test Coverage:
- Role-based access control (RBAC)
- Permission validation for ML operations
- Resource-level access control
- Privacy-preserving ML access permissions
- Adversarial testing access control
- Administrative privilege separation
- Cross-component authorization
"""

import asyncio
import json
from enum import Enum

import pytest

from prompt_improver.core.config import AppConfig


class Permission(Enum):
    """ML system permissions"""

    READ_MODELS = "read_models"
    WRITE_MODELS = "write_models"
    DELETE_MODELS = "delete_models"
    DEPLOY_MODELS = "deploy_models"
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    READ_SENSITIVE_DATA = "read_sensitive_data"
    ACCESS_DIFFERENTIAL_PRIVACY = "access_differential_privacy"
    CONFIGURE_PRIVACY_BUDGET = "configure_privacy_budget"
    VIEW_PRIVACY_METRICS = "view_privacy_metrics"
    RUN_ADVERSARIAL_TESTS = "run_adversarial_tests"
    VIEW_SECURITY_LOGS = "view_security_logs"
    CONFIGURE_SECURITY = "configure_security"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    SYSTEM_ADMIN = "system_admin"


class Role(Enum):
    """System roles with associated permissions"""

    GUEST = "guest"
    USER = "user"
    ML_ANALYST = "ml_analyst"
    ML_ENGINEER = "ml_engineer"
    PRIVACY_OFFICER = "privacy_officer"
    SECURITY_ANALYST = "security_analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


ROLE_PERMISSIONS = {
    Role.GUEST: {Permission.READ_MODELS},
    Role.USER: {Permission.READ_MODELS, Permission.READ_DATA},
    Role.ML_ANALYST: {
        Permission.READ_MODELS,
        Permission.WRITE_MODELS,
        Permission.READ_DATA,
        Permission.VIEW_PRIVACY_METRICS,
    },
    Role.ML_ENGINEER: {
        Permission.READ_MODELS,
        Permission.WRITE_MODELS,
        Permission.DELETE_MODELS,
        Permission.DEPLOY_MODELS,
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.RUN_ADVERSARIAL_TESTS,
    },
    Role.PRIVACY_OFFICER: {
        Permission.READ_MODELS,
        Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        Permission.CONFIGURE_PRIVACY_BUDGET,
        Permission.VIEW_PRIVACY_METRICS,
        Permission.READ_SENSITIVE_DATA,
    },
    Role.SECURITY_ANALYST: {
        Permission.READ_MODELS,
        Permission.RUN_ADVERSARIAL_TESTS,
        Permission.VIEW_SECURITY_LOGS,
        Permission.CONFIGURE_SECURITY,
    },
    Role.ADMIN: {
        Permission.READ_MODELS,
        Permission.WRITE_MODELS,
        Permission.DELETE_MODELS,
        Permission.DEPLOY_MODELS,
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.MANAGE_USERS,
        Permission.MANAGE_ROLES,
        Permission.VIEW_SECURITY_LOGS,
    },
    Role.SUPER_ADMIN: set(Permission),
}


class PostgreSQLAuthorizationDB:
    """PostgreSQL database interface for authorization"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._ensure_tables()

    def _ensure_tables(self):
        """Create authorization tables if they don't exist"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "\n                    CREATE TABLE IF NOT EXISTS auth_users (\n                        user_id TEXT PRIMARY KEY,\n                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n                    )\n                "
                )
                cur.execute(
                    "\n                    CREATE TABLE IF NOT EXISTS auth_roles (\n                        role_name TEXT PRIMARY KEY,\n                        description TEXT\n                    )\n                "
                )
                cur.execute(
                    "\n                    CREATE TABLE IF NOT EXISTS auth_user_roles (\n                        user_id TEXT,\n                        role_name TEXT,\n                        assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                        PRIMARY KEY (user_id, role_name),\n                        FOREIGN KEY (user_id) REFERENCES auth_users(user_id),\n                        FOREIGN KEY (role_name) REFERENCES auth_roles(role_name)\n                    )\n                "
                )
                cur.execute(
                    "\n                    CREATE TABLE IF NOT EXISTS auth_resources (\n                        resource_id TEXT PRIMARY KEY,\n                        owner_id TEXT,\n                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                        FOREIGN KEY (owner_id) REFERENCES auth_users(user_id)\n                    )\n                "
                )
                cur.execute(
                    "\n                    CREATE TABLE IF NOT EXISTS auth_audit_log (\n                        id SERIAL PRIMARY KEY,\n                        user_id TEXT,\n                        action TEXT,\n                        resource_id TEXT,\n                        details JSONB,\n                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n                    )\n                "
                )
                for role in Role:
                    cur.execute(
                        "INSERT INTO auth_roles (role_name, description) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (role.value, f"Role: {role.value}"),
                    )
                conn.commit()

    def assign_role_to_user(self, user_id: str, role: Role):
        """Assign role to user in database"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO auth_users (user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (user_id,),
                )
                cur.execute(
                    "INSERT INTO auth_user_roles (user_id, role_name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (user_id, role.value),
                )
                cur.execute(
                    "INSERT INTO auth_audit_log (user_id, action, details) VALUES (%s, %s, %s)",
                    (user_id, "assign_role", json.dumps({"role": role.value})),
                )
                conn.commit()

    def remove_role_from_user(self, user_id: str, role: Role):
        """Remove role from user in database"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM auth_user_roles WHERE user_id = %s AND role_name = %s",
                    (user_id, role.value),
                )
                cur.execute(
                    "INSERT INTO auth_audit_log (user_id, action, details) VALUES (%s, %s, %s)",
                    (user_id, "remove_role", json.dumps({"role": role.value})),
                )
                conn.commit()

    def fetch_user_permissions(self, user_id: str) -> set[Permission]:
        """Fetch all permissions for a user based on their roles"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role_name FROM auth_user_roles WHERE user_id = %s",
                    (user_id,),
                )
                roles = [Role(row[0]) for row in cur.fetchall()]
                permissions = set()
                for role in roles:
                    permissions.update(ROLE_PERMISSIONS.get(role, set()))
                cur.execute(
                    "INSERT INTO auth_audit_log (user_id, action, details) VALUES (%s, %s, %s)",
                    (
                        user_id,
                        "permission_check",
                        json.dumps({"permissions_count": len(permissions)}),
                    ),
                )
                conn.commit()
                return permissions

    def check_resource_access(self, user_id: str, resource_id: str) -> bool:
        """Check if user can access resource"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT owner_id FROM auth_resources WHERE resource_id = %s",
                    (resource_id,),
                )
                result = cur.fetchone()
                if result:
                    owner_id = result[0]
                    if owner_id == user_id:
                        return True
                    user_permissions = self.fetch_user_permissions(user_id)
                    return Permission.SYSTEM_ADMIN in user_permissions
                return True

    def set_resource_owner(self, resource_id: str, owner_id: str):
        """Set resource owner in database"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO auth_users (user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (owner_id,),
                )
                cur.execute(
                    "INSERT INTO auth_resources (resource_id, owner_id) VALUES (%s, %s) ON CONFLICT (resource_id) DO UPDATE SET owner_id = %s",
                    (resource_id, owner_id, owner_id),
                )
                conn.commit()

    def get_audit_log(self) -> list[dict]:
        """Get audit log for testing purposes"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM auth_audit_log ORDER BY timestamp DESC LIMIT 100"
                )
                return cur.fetchall()

    def get_user_roles(self, user_id: str) -> set[Role]:
        """Get roles for a user (for testing)"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role_name FROM auth_user_roles WHERE user_id = %s",
                    (user_id,),
                )
                return {Role(row[0]) for row in cur.fetchall()}

    def cleanup(self):
        """Clean up test data"""
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM auth_audit_log")
                cur.execute("DELETE FROM auth_user_roles")
                cur.execute("DELETE FROM auth_resources")
                cur.execute("DELETE FROM auth_users")
                conn.commit()


class RealAuthorizationService:
    """Real authorization service using PostgreSQL database"""

    def __init__(self, db_interface: PostgreSQLAuthorizationDB):
        self.db = db_interface

    def assign_role(self, user_id: str, role: Role) -> None:
        """Assign role to user"""
        self.db.assign_role_to_user(user_id, role)

    def remove_role(self, user_id: str, role: Role) -> None:
        """Remove role from user"""
        self.db.remove_role_from_user(user_id, role)

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        """Get all permissions for user based on their roles"""
        return self.db.fetch_user_permissions(user_id)

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions

    def can_access_resource(
        self, user_id: str, resource_id: str, permission: Permission
    ) -> bool:
        """Check if user can access specific resource with permission"""
        if not self.has_permission(user_id, permission):
            return False
        return self.db.check_resource_access(user_id, resource_id)

    def set_resource_owner(self, resource_id: str, owner_id: str) -> None:
        """Set resource owner"""
        self.db.set_resource_owner(resource_id, owner_id)

    @property
    def user_roles(self) -> dict[str, set[Role]]:
        """Get user roles for testing compatibility"""
        all_roles = {}
        with psycopg.connect(self.db.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT user_id FROM auth_user_roles")
                user_ids = [row[0] for row in cur.fetchall()]
                for user_id in user_ids:
                    all_roles[user_id] = self.db.get_user_roles(user_id)
        return all_roles

    @property
    def audit_log(self) -> list[dict]:
        """Get audit log for testing compatibility"""
        return self.db.get_audit_log()


@pytest.fixture
async def auth_service():
    """Create authorization service with test data"""
    db_config = AppConfig().database
    db_interface = PostgreSQLAuthorizationDB(db_config.psycopg_connection_string)
    service = RealAuthorizationService(db_interface)
    service.assign_role("user_001", Role.USER)
    service.assign_role("analyst_001", Role.ML_ANALYST)
    service.assign_role("engineer_001", Role.ML_ENGINEER)
    service.assign_role("privacy_001", Role.PRIVACY_OFFICER)
    service.assign_role("security_001", Role.SECURITY_ANALYST)
    service.assign_role("admin_001", Role.ADMIN)
    service.assign_role("super_001", Role.SUPER_ADMIN)
    service.set_resource_owner("model_123", "analyst_001")
    service.set_resource_owner("dataset_456", "engineer_001")
    yield service
    db_interface.cleanup()


class TestRoleBasedAccessControl:
    """Test RBAC implementation"""

    def test_role_assignment_and_removal(self, auth_service):
        """Test role assignment and removal"""
        user_id = "test_user_rbac"
        assert auth_service.user_roles.get(user_id, set()) == set()
        auth_service.assign_role(user_id, Role.ML_ANALYST)
        assert Role.ML_ANALYST in auth_service.user_roles[user_id]
        auth_service.remove_role(user_id, Role.ML_ANALYST)
        assert Role.ML_ANALYST not in auth_service.user_roles.get(user_id, set())

    def test_multiple_roles_per_user(self, auth_service):
        """Test user can have multiple roles"""
        user_id = "multi_role_user"
        auth_service.assign_role(user_id, Role.ML_ANALYST)
        auth_service.assign_role(user_id, Role.SECURITY_ANALYST)
        user_roles = auth_service.user_roles[user_id]
        assert Role.ML_ANALYST in user_roles
        assert Role.SECURITY_ANALYST in user_roles
        assert len(user_roles) == 2

    def test_permission_inheritance_from_roles(self, auth_service):
        """Test permissions are correctly inherited from roles"""
        user_id = "inheritance_test"
        auth_service.assign_role(user_id, Role.ML_ENGINEER)
        permissions = auth_service.get_user_permissions(user_id)
        assert Permission.READ_MODELS in permissions
        assert Permission.WRITE_MODELS in permissions
        assert Permission.DELETE_MODELS in permissions
        assert Permission.DEPLOY_MODELS in permissions
        assert Permission.RUN_ADVERSARIAL_TESTS in permissions
        assert Permission.MANAGE_USERS not in permissions
        assert Permission.SYSTEM_ADMIN not in permissions


class TestPermissionValidation:
    """Test permission validation mechanisms"""

    def test_basic_permission_checking(self, auth_service):
        """Test basic permission checking"""
        assert (
            auth_service.has_permission("analyst_001", Permission.READ_MODELS) is True
        )
        assert (
            auth_service.has_permission("analyst_001", Permission.WRITE_MODELS) is True
        )
        assert (
            auth_service.has_permission("analyst_001", Permission.DELETE_MODELS)
            is False
        )
        assert auth_service.has_permission("user_001", Permission.READ_MODELS) is True
        assert auth_service.has_permission("user_001", Permission.WRITE_MODELS) is False

    def test_privacy_permission_validation(self, auth_service):
        """Test privacy-specific permission validation"""
        assert (
            auth_service.has_permission(
                "privacy_001", Permission.ACCESS_DIFFERENTIAL_PRIVACY
            )
            is True
        )
        assert (
            auth_service.has_permission(
                "privacy_001", Permission.CONFIGURE_PRIVACY_BUDGET
            )
            is True
        )
        assert (
            auth_service.has_permission("privacy_001", Permission.READ_SENSITIVE_DATA)
            is True
        )
        assert (
            auth_service.has_permission(
                "user_001", Permission.ACCESS_DIFFERENTIAL_PRIVACY
            )
            is False
        )
        assert (
            auth_service.has_permission(
                "analyst_001", Permission.CONFIGURE_PRIVACY_BUDGET
            )
            is False
        )

    def test_security_permission_validation(self, auth_service):
        """Test security-specific permission validation"""
        assert (
            auth_service.has_permission(
                "security_001", Permission.RUN_ADVERSARIAL_TESTS
            )
            is True
        )
        assert (
            auth_service.has_permission("security_001", Permission.VIEW_SECURITY_LOGS)
            is True
        )
        assert (
            auth_service.has_permission("security_001", Permission.CONFIGURE_SECURITY)
            is True
        )
        assert (
            auth_service.has_permission("user_001", Permission.VIEW_SECURITY_LOGS)
            is False
        )
        assert (
            auth_service.has_permission("analyst_001", Permission.CONFIGURE_SECURITY)
            is False
        )

    def test_admin_permission_validation(self, auth_service):
        """Test administrative permission validation"""
        assert auth_service.has_permission("admin_001", Permission.MANAGE_USERS) is True
        assert auth_service.has_permission("admin_001", Permission.MANAGE_ROLES) is True
        for permission in Permission:
            assert auth_service.has_permission("super_001", permission) is True
        assert auth_service.has_permission("user_001", Permission.MANAGE_USERS) is False
        assert (
            auth_service.has_permission("engineer_001", Permission.SYSTEM_ADMIN)
            is False
        )


class TestResourceLevelAccess:
    """Test resource-level access control"""

    def test_resource_ownership_access(self, auth_service):
        """Test resource ownership access control"""
        assert (
            auth_service.can_access_resource(
                "analyst_001", "model_123", Permission.READ_MODELS
            )
            is True
        )
        assert (
            auth_service.can_access_resource(
                "engineer_001", "dataset_456", Permission.READ_DATA
            )
            is True
        )
        assert (
            auth_service.can_access_resource(
                "user_001", "model_123", Permission.READ_MODELS
            )
            is False
        )

    def test_admin_override_resource_access(self, auth_service):
        """Test admin can override resource ownership"""
        assert (
            auth_service.can_access_resource(
                "super_001", "model_123", Permission.READ_MODELS
            )
            is True
        )
        assert (
            auth_service.can_access_resource(
                "super_001", "dataset_456", Permission.READ_DATA
            )
            is True
        )

    def test_insufficient_permission_for_resource(self, auth_service):
        """Test user without permission cannot access resource even if they own it"""
        auth_service.set_resource_owner("dataset_789", "analyst_001")
        assert (
            auth_service.can_access_resource(
                "analyst_001", "dataset_789", Permission.WRITE_DATA
            )
            is False
        )


class TestCrossComponentAuthorization:
    """Test authorization across different ML components"""

    def test_privacy_preserving_ml_authorization(self, auth_service):
        """Test authorization for privacy-preserving ML operations"""
        privacy_operations = [
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
            Permission.CONFIGURE_PRIVACY_BUDGET,
            Permission.VIEW_PRIVACY_METRICS,
        ]
        for operation in privacy_operations:
            assert auth_service.has_permission("privacy_001", operation) is True
            assert auth_service.has_permission("user_001", operation) is False
            assert auth_service.has_permission("super_001", operation) is True

    def test_adversarial_testing_authorization(self, auth_service):
        """Test authorization for adversarial testing operations"""
        assert (
            auth_service.has_permission(
                "engineer_001", Permission.RUN_ADVERSARIAL_TESTS
            )
            is True
        )
        assert (
            auth_service.has_permission(
                "security_001", Permission.RUN_ADVERSARIAL_TESTS
            )
            is True
        )
        assert (
            auth_service.has_permission("user_001", Permission.RUN_ADVERSARIAL_TESTS)
            is False
        )
        assert (
            auth_service.has_permission("analyst_001", Permission.RUN_ADVERSARIAL_TESTS)
            is False
        )

    def test_model_deployment_authorization(self, auth_service):
        """Test authorization for model deployment"""
        assert (
            auth_service.has_permission("engineer_001", Permission.DEPLOY_MODELS)
            is True
        )
        assert (
            auth_service.has_permission("admin_001", Permission.DEPLOY_MODELS) is True
        )
        assert (
            auth_service.has_permission("analyst_001", Permission.DEPLOY_MODELS)
            is False
        )
        assert (
            auth_service.has_permission("user_001", Permission.DEPLOY_MODELS) is False
        )


class TestPrivilegeEscalation:
    """Test privilege escalation prevention"""

    def test_prevent_unauthorized_role_assignment(self, auth_service):
        """Test preventing unauthorized role assignment"""
        can_manage_roles = auth_service.has_permission(
            "user_001", Permission.MANAGE_ROLES
        )
        assert can_manage_roles is False
        can_admin_manage = auth_service.has_permission(
            "admin_001", Permission.MANAGE_ROLES
        )
        assert can_admin_manage is True

    def test_prevent_permission_elevation(self, auth_service):
        """Test preventing permission elevation attacks"""
        user_id = "elevation_test"
        auth_service.assign_role(user_id, Role.USER)
        initial_permissions = auth_service.get_user_permissions(user_id)
        assert Permission.SYSTEM_ADMIN not in initial_permissions
        assert Permission.MANAGE_USERS not in initial_permissions
        assert auth_service.has_permission(user_id, Permission.SYSTEM_ADMIN) is False

    def test_role_separation_enforcement(self, auth_service):
        """Test role separation is enforced"""
        analyst_permissions = auth_service.get_user_permissions("analyst_001")
        assert Permission.READ_MODELS in analyst_permissions
        assert Permission.WRITE_MODELS in analyst_permissions
        assert Permission.MANAGE_USERS not in analyst_permissions
        assert Permission.SYSTEM_ADMIN not in analyst_permissions


class TestAuditLogging:
    """Test authorization audit logging"""

    def test_permission_checks_are_logged(self, auth_service):
        """Test permission checks are logged for audit"""
        initial_log_count = len(auth_service.audit_log)
        auth_service.has_permission("user_001", Permission.READ_MODELS)
        auth_service.has_permission("analyst_001", Permission.WRITE_MODELS)
        assert len(auth_service.audit_log) == initial_log_count + 2
        recent_logs = auth_service.audit_log[-2:]
        assert all(log["action"] == "permission_check" for log in recent_logs)
        assert recent_logs[0]["user_id"] == "user_001"
        assert recent_logs[1]["user_id"] == "analyst_001"

    def test_role_assignments_are_logged(self, auth_service):
        """Test role assignments are logged"""
        initial_log_count = len(auth_service.audit_log)
        auth_service.assign_role("test_user", Role.ML_ANALYST)
        auth_service.remove_role("test_user", Role.ML_ANALYST)
        assert len(auth_service.audit_log) == initial_log_count + 2
        recent_logs = auth_service.audit_log[-2:]
        assert recent_logs[0]["action"] == "assign_role"
        assert recent_logs[1]["action"] == "remove_role"

    def test_resource_access_is_logged(self, auth_service):
        """Test resource access attempts are logged"""
        initial_log_count = len(auth_service.audit_log)
        auth_service.can_access_resource(
            "analyst_001", "model_123", Permission.READ_MODELS
        )
        assert len(auth_service.audit_log) > initial_log_count
        resource_logs = [
            log for log in auth_service.audit_log if log["action"] == "resource_access"
        ]
        assert len(resource_logs) > 0
        latest_resource_log = resource_logs[-1]
        assert latest_resource_log["user_id"] == "analyst_001"
        assert latest_resource_log["details"]["resource_id"] == "model_123"


@pytest.mark.asyncio
class TestAsyncAuthorization:
    """Test asynchronous authorization patterns"""

    async def test_async_permission_checking(self, auth_service):
        """Test async permission checking patterns"""

        async def async_has_permission(user_id: str, permission: Permission) -> bool:
            await asyncio.sleep(0.001)
            return auth_service.has_permission(user_id, permission)

        result = await async_has_permission("user_001", Permission.READ_MODELS)
        assert isinstance(result, bool)
        tasks = [
            async_has_permission("user_001", Permission.READ_MODELS),
            async_has_permission("analyst_001", Permission.WRITE_MODELS),
            async_has_permission("admin_001", Permission.MANAGE_USERS),
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(isinstance(result, bool) for result in results)


@pytest.mark.performance
class TestAuthorizationPerformance:
    """Test authorization performance characteristics"""

    def test_permission_checking_performance(self, auth_service):
        """Test permission checking performance"""
        import time

        start_time = time.time()
        for _ in range(1000):
            auth_service.has_permission("analyst_001", Permission.READ_MODELS)
        elapsed_time = time.time() - start_time
        assert elapsed_time < 0.1
        avg_time_per_check = elapsed_time / 1000
        assert avg_time_per_check < 0.0001

    def test_role_permission_lookup_performance(self, auth_service):
        """Test role permission lookup performance"""
        import time

        user_id = "perf_test_user"
        for role in [Role.USER, Role.ML_ANALYST, Role.ML_ENGINEER]:
            auth_service.assign_role(user_id, role)
        start_time = time.time()
        for _ in range(100):
            auth_service.get_user_permissions(user_id)
        elapsed_time = time.time() - start_time
        assert elapsed_time < 0.05
        avg_time_per_lookup = elapsed_time / 100
        assert avg_time_per_lookup < 0.0005


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in authorization"""

    def test_nonexistent_user_permission_check(self, auth_service):
        """Test permission check for nonexistent user"""
        result = auth_service.has_permission("nonexistent_user", Permission.READ_MODELS)
        assert result is False

    def test_empty_roles_permission_check(self, auth_service):
        """Test permission check for user with no roles"""
        user_id = "no_roles_user"
        auth_service.user_roles[user_id] = set()
        result = auth_service.has_permission(user_id, Permission.READ_MODELS)
        assert result is False

    def test_resource_access_with_nonexistent_resource(self, auth_service):
        """Test resource access with nonexistent resource"""
        result = auth_service.can_access_resource(
            "analyst_001", "nonexistent_resource", Permission.READ_MODELS
        )
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
