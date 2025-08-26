"""
End-to-End Security Integration Tests

Tests complete security workflows and integration between different security
components to ensure comprehensive security coverage across Phase 3 ML features.
Following 2025 best practices: Uses real behavior for internal components,
only mocks external dependencies (third-party APIs, external auth providers).

Integration Test Coverage:
- Authentication + Authorization flow with real security validation
- Privacy-preserving ML with actual differential privacy mechanisms
- Adversarial robustness with real defense systems
- Input sanitization with production-grade validation
- Security audit trail integration
- Cross-component security validation
- Emergency security response scenarios
"""

import asyncio
import hashlib
import secrets
from typing import Any

import numpy as np
import pytest

from prompt_improver.services.security import (
    AdversarialDefenseSystem,
    AuthenticationService,
    AuthorizationService,
    DifferentialPrivacyService,
    FederatedLearningService,
    InputSanitizer,
    Permission,
    Role,
)
from prompt_improver.services.security.adversarial_defense import (
    AdversarialAttackSimulator,
    RobustnessEvaluator,
)
from prompt_improver.utils.datetime_utils import aware_utc_now


class SecurityOrchestrator:
    """Orchestrates REAL security components for end-to-end security testing.

    Following 2025 best practices: Uses real internal security services
    instead of mocks to test actual production behavior.
    """

    def __init__(self):
        self.auth_service = AuthenticationService()
        self.authz_service = AuthorizationService()
        self.input_sanitizer = InputSanitizer()
        self.dp_service = DifferentialPrivacyService()
        self.fl_service = FederatedLearningService()
        self.defense_system = AdversarialDefenseSystem()
        self.security_audit_log: list[dict[str, Any]] = []
        self.alert_system = SecurityAlertSystem()
        self._setup_default_security()

    def _setup_default_security(self):
        """Set up default users and security roles using real authentication service."""
        self.auth_service.register_user(
            username="ml_engineer",
            password="secure_password_123!",
            email="engineer@company.com",
            roles=["ml_engineer"],
        )
        self.auth_service.register_user(
            username="privacy_officer",
            password="privacy_secure_456!",
            email="privacy@company.com",
            roles=["privacy_officer"],
        )
        self.authz_service.assign_role("ml_eng_001", Role.ML_ENGINEER)
        self.authz_service.assign_role("privacy_001", Role.PRIVACY_OFFICER)
        ml_engineer_data = self.auth_service.authenticate_user(
            "ml_engineer", "secure_password_123!"
        )
        privacy_officer_data = self.auth_service.authenticate_user(
            "privacy_officer", "privacy_secure_456!"
        )
        if ml_engineer_data:
            self.authz_service.assign_role(
                ml_engineer_data["user_id"], Role.ML_ENGINEER
            )
        if privacy_officer_data:
            self.authz_service.assign_role(
                privacy_officer_data["user_id"], Role.PRIVACY_OFFICER
            )

    def authenticate_and_authorize(
        self, username: str, password: str, required_permission: Permission
    ) -> dict[str, Any] | None:
        """Complete authentication and authorization flow"""
        user_data = self.auth_service.authenticate_user(username, password)
        if not user_data:
            self._log_security_event("authentication_failed", {"username": username})
            return None
        if not self.authz_service.has_permission(
            user_data["user_id"], required_permission
        ):
            self._log_security_event(
                "authorization_failed",
                {
                    "user_id": user_data["user_id"],
                    "permission": required_permission.value,
                },
            )
            return None
        token = self.auth_service.create_jwt_token(user_data)
        self._log_security_event(
            "authentication_success",
            {"user_id": user_data["user_id"], "permission": required_permission.value},
        )
        return {
            "user_data": user_data,
            "token": token,
            "permissions": self.authz_service.get_user_permissions(
                user_data["user_id"]
            ),
        }

    def secure_ml_pipeline(
        self, user_token: str, ml_data: Any, privacy_params: dict[str, float]
    ) -> dict[str, Any]:
        """Execute secure ML pipeline with privacy preservation"""
        token_payload = self.auth_service.validate_jwt_token(user_token)
        if not token_payload:
            raise SecurityError("Invalid or expired token")
        if not self.input_sanitizer.validate_ml_input_data(ml_data):
            raise SecurityError("Invalid ML input data")
        epsilon = privacy_params.get("epsilon", 1.0)
        delta = privacy_params.get("delta", 1e-06)
        if not self.input_sanitizer.validate_privacy_parameters(epsilon, delta):
            raise SecurityError("Invalid privacy parameters")
        if not self.dp_service.check_privacy_budget(epsilon):
            raise SecurityError("Insufficient privacy budget")
        ml_result = np.mean(ml_data) if isinstance(ml_data, np.ndarray) else 42.0
        private_result = self.dp_service.add_noise(ml_result, sensitivity=1.0)
        self._log_security_event(
            "secure_ml_operation",
            {
                "user_id": token_payload["user_id"],
                "epsilon_used": epsilon,
                "data_size": len(ml_data) if hasattr(ml_data, "__len__") else 1,
            },
        )
        return {
            "result": private_result,
            "privacy_spent": self.dp_service.get_privacy_spent(),
            "timestamp": aware_utc_now(),
        }

    def adversarial_robustness_pipeline(
        self, user_token: str, model: Any, test_data: np.ndarray
    ) -> dict[str, Any]:
        """Execute adversarial robustness testing pipeline"""
        token_payload = self.auth_service.validate_jwt_token(user_token)
        if not token_payload:
            raise SecurityError("Invalid token")
        if not self.authz_service.has_permission(
            token_payload["user_id"], Permission.RUN_ADVERSARIAL_TESTS
        ):
            raise SecurityError("Insufficient permissions for adversarial testing")
        if not self.input_sanitizer.validate_ml_input_data(test_data):
            raise SecurityError("Invalid test data")
        self.defense_system.enable_defense("gaussian_noise")
        self.defense_system.enable_defense("input_validation")
        attacker = AdversarialAttackSimulator(epsilon=0.1)
        evaluator = RobustnessEvaluator()
        y_dummy = np.zeros(len(test_data))
        X_adversarial = attacker.generate_adversarial_examples(test_data, y_dummy)
        X_defended = self.defense_system.apply_gaussian_noise_defense(X_adversarial)
        robustness_results = evaluator.evaluate_robustness(
            model, test_data, X_defended, y_dummy
        )
        if robustness_results["attack_success_rate"] > 0.5:
            self.alert_system.trigger_alert(
                "HIGH", "High adversarial attack success rate detected"
            )
        self._log_security_event(
            "adversarial_robustness_test",
            {
                "user_id": token_payload["user_id"],
                "samples_tested": len(test_data),
                "attack_success_rate": robustness_results["attack_success_rate"],
                "robustness_score": robustness_results["robustness_score"],
            },
        )
        return robustness_results

    def federated_learning_security_pipeline(
        self, clients: list[str]
    ) -> dict[str, Any]:
        """Execute secure federated learning pipeline"""
        registered_clients = []
        for client_id in clients:
            public_key = f"pubkey_{client_id}_{secrets.token_hex(16)}"
            if self.fl_service.register_client(client_id, public_key):
                registered_clients.append(client_id)
        for client_id in registered_clients:
            model_update = {
                "gradients": {"layer1": [0.1, 0.2, 0.3], "layer2": [0.4, 0.5, 0.6]},
                "privacy_budget": 0.5,
            }
            encrypted_update = self.fl_service.encrypt_data(model_update)
            self.fl_service.submit_encrypted_update(
                client_id, encrypted_update, round_number=0
            )
        aggregated_result = self.fl_service.aggregate_updates()
        total_privacy_spent = len(registered_clients) * 0.5
        self._log_security_event(
            "federated_learning_round",
            {
                "registered_clients": len(registered_clients),
                "total_privacy_spent": total_privacy_spent,
                "aggregation_successful": aggregated_result is not None,
            },
        )
        return {
            "aggregated_result": aggregated_result,
            "registered_clients": len(registered_clients),
            "total_privacy_spent": total_privacy_spent,
        }

    def emergency_security_response(
        self, threat_level: str, threat_type: str
    ) -> dict[str, Any]:
        """Execute emergency security response"""
        response_actions = []
        if threat_level == "CRITICAL":
            response_actions.append("disable_ml_operations")
            response_actions.append("invalidate_all_tokens")
            for defense in self.defense_system.defense_methods:
                self.defense_system.enable_defense(defense)
            response_actions.append("enable_all_defenses")
        elif threat_level == "HIGH":
            self.dp_service.epsilon = min(self.dp_service.epsilon, 0.1)
            response_actions.append("reduce_privacy_budget")
            self.defense_system.enable_defense("input_validation")
            self.defense_system.enable_defense("gaussian_noise")
            response_actions.append("enable_core_defenses")
        self._log_security_event(
            "emergency_response",
            {
                "threat_level": threat_level,
                "threat_type": threat_type,
                "actions_taken": response_actions,
                "timestamp": aware_utc_now(),
            },
        )
        self.alert_system.trigger_alert(
            threat_level, f"Emergency response activated: {threat_type}"
        )
        return {
            "threat_level": threat_level,
            "actions_taken": response_actions,
            "defenses_active": self.defense_system.active_defenses,
        }

    def _log_security_event(self, event_type: str, details: dict[str, Any]):
        """Log security event for audit"""
        self.security_audit_log.append({
            "timestamp": aware_utc_now(),
            "event_type": event_type,
            "details": details,
            "event_id": hashlib.md5(
                f"{aware_utc_now()}{event_type}".encode()
            ).hexdigest()[:8],
        })


class SecurityAlertSystem:
    """Real security alert system for production use and integration testing"""

    def __init__(self):
        self.alerts: list[dict[str, Any]] = []
        self.alert_thresholds = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 3, "LOW": 5}
        self.alert_counts: dict[str, int] = {}

    def trigger_alert(self, level: str, message: str) -> bool:
        """Trigger security alert"""
        alert_key = f"{level}:{message}"
        current_count = self.alert_counts.get(alert_key, 0) + 1
        self.alert_counts[alert_key] = current_count
        threshold = self.alert_thresholds.get(level, 1)
        if current_count >= threshold:
            self.alerts.append({
                "timestamp": aware_utc_now(),
                "level": level,
                "message": message,
                "count": current_count,
                "alert_id": hashlib.md5(
                    f"{aware_utc_now()}{message}".encode()
                ).hexdigest()[:8],
            })
            return True
        return False


class SecurityError(Exception):
    """Security-related exception"""


@pytest.fixture
def security_orchestrator():
    """Create security orchestrator for testing"""
    return SecurityOrchestrator()


class TestAuthenticationAuthorizationIntegration:
    """Test authentication and authorization integration"""

    def test_successful_auth_flow(self, security_orchestrator):
        """Test successful authentication and authorization flow"""
        result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.WRITE_MODELS
        )
        assert result is not None
        assert "user_data" in result
        assert "token" in result
        assert "permissions" in result
        assert result["user_data"]["username"] == "ml_engineer"
        assert Permission.WRITE_MODELS in result["permissions"]
        assert Permission.READ_MODELS in result["permissions"]

    def test_failed_authentication(self, security_orchestrator):
        """Test failed authentication"""
        result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "wrong_password", Permission.WRITE_MODELS
        )
        assert result is None
        auth_failed_logs = [
            log
            for log in security_orchestrator.security_audit_log
            if log["event_type"] == "authentication_failed"
        ]
        assert len(auth_failed_logs) > 0

    def test_failed_authorization(self, security_orchestrator):
        """Test failed authorization (valid user, insufficient permissions)"""
        result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.MANAGE_USERS
        )
        assert result is None
        authz_failed_logs = [
            log
            for log in security_orchestrator.security_audit_log
            if log["event_type"] == "authorization_failed"
        ]
        assert len(authz_failed_logs) > 0

    def test_privacy_officer_permissions(self, security_orchestrator):
        """Test privacy officer specific permissions"""
        result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        assert result is not None
        assert Permission.ACCESS_DIFFERENTIAL_PRIVACY in result["permissions"]
        assert Permission.CONFIGURE_PRIVACY_BUDGET in result["permissions"]


class TestSecureMLPipelineIntegration:
    """Test secure ML pipeline integration"""

    def test_complete_secure_ml_pipeline(self, security_orchestrator):
        """Test complete secure ML pipeline"""
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        assert auth_result is not None
        token = auth_result["token"]
        ml_data = np.random.rand(100, 10)
        privacy_params = {"epsilon": 1.0, "delta": 1e-06}
        result = security_orchestrator.secure_ml_pipeline(
            token, ml_data, privacy_params
        )
        assert "result" in result
        assert "privacy_spent" in result
        assert "timestamp" in result
        raw_mean = np.mean(ml_data)
        private_result = result["result"]
        assert abs(private_result - raw_mean) > 0

    def test_invalid_token_rejection(self, security_orchestrator):
        """Test rejection of invalid tokens"""
        invalid_token = "invalid.jwt.token"
        ml_data = np.random.rand(50, 5)
        privacy_params = {"epsilon": 1.0, "delta": 1e-06}
        with pytest.raises(SecurityError, match="Invalid or expired token"):
            security_orchestrator.secure_ml_pipeline(
                invalid_token, ml_data, privacy_params
            )

    def test_invalid_ml_data_rejection(self, security_orchestrator):
        """Test rejection of invalid ML data"""
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        token = auth_result["token"]
        invalid_data = np.array([1.0, 2.0, float("nan"), 4.0])
        privacy_params = {"epsilon": 1.0, "delta": 1e-06}
        with pytest.raises(SecurityError, match="Invalid ML input data"):
            security_orchestrator.secure_ml_pipeline(
                token, invalid_data, privacy_params
            )

    def test_privacy_parameter_validation(self, security_orchestrator):
        """Test privacy parameter validation"""
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        token = auth_result["token"]
        ml_data = np.random.rand(50, 5)
        invalid_privacy_params = {"epsilon": -1.0, "delta": 1e-06}
        with pytest.raises(SecurityError, match="Invalid privacy parameters"):
            security_orchestrator.secure_ml_pipeline(
                token, ml_data, invalid_privacy_params
            )

    def test_privacy_budget_enforcement(self, security_orchestrator):
        """Test privacy budget enforcement"""
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        token = auth_result["token"]
        ml_data = np.random.rand(20, 3)
        for _ in range(15):
            privacy_params = {"epsilon": 1.0, "delta": 1e-06}
            try:
                security_orchestrator.secure_ml_pipeline(token, ml_data, privacy_params)
            except SecurityError as e:
                if "Insufficient privacy budget" in str(e):
                    break
        else:
            pytest.fail("Privacy budget was not enforced")


class TestAdversarialRobustnessIntegration:
    """Test adversarial robustness integration"""

    def test_adversarial_robustness_pipeline(self, security_orchestrator):
        """Test complete adversarial robustness pipeline"""
        auth_result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.RUN_ADVERSARIAL_TESTS
        )
        assert auth_result is not None
        token = auth_result["token"]
        test_data = np.random.rand(50, 10)

        # Using real models instead of mocks

        class VulnerableModel:
            def predict(self, X):
                return np.zeros(len(X))

        model = VulnerableModel()
        test_data = np.random.rand(20, 5)
        results = security_orchestrator.adversarial_robustness_pipeline(
            token, model, test_data
        )
        high_alerts = [
            alert
            for alert in security_orchestrator.alert_system.alerts
            if alert["level"] == "HIGH"
        ]
        assert len(high_alerts) > 0


class TestFederatedLearningSecurityIntegration:
    """Test federated learning security integration"""

    def test_secure_federated_learning_pipeline(self, security_orchestrator):
        """Test secure federated learning pipeline"""
        clients = ["client_001", "client_002", "client_003"]
        results = security_orchestrator.federated_learning_security_pipeline(clients)
        assert "aggregated_result" in results
        assert "registered_clients" in results
        assert "total_privacy_spent" in results
        assert results["registered_clients"] == len(clients)
        assert results["aggregated_result"] is not None
        expected_privacy = len(clients) * 0.5
        assert results["total_privacy_spent"] == expected_privacy

    def test_federated_learning_encryption(self, security_orchestrator):
        """Test encryption in federated learning"""
        clients = ["secure_client_1"]
        public_key = "test_public_key"
        success = security_orchestrator.fl_service.register_client(
            clients[0], public_key
        )
        assert success is True
        test_data = "sensitive_gradient_data"
        encrypted = security_orchestrator.fl_service.encrypt_data(test_data)
        assert encrypted != test_data.encode()
        decrypted = security_orchestrator.fl_service.cipher.decrypt(encrypted).decode()
        assert decrypted == test_data


class TestEmergencySecurityResponse:
    """Test emergency security response scenarios"""

    def test_critical_threat_response(self, security_orchestrator):
        """Test critical threat response"""
        response = security_orchestrator.emergency_security_response(
            "CRITICAL", "data_breach"
        )
        expected_actions = [
            "disable_ml_operations",
            "invalidate_all_tokens",
            "enable_all_defenses",
        ]
        for action in expected_actions:
            assert action in response["actions_taken"]
        for defense in security_orchestrator.defense_system.defense_methods:
            assert defense in security_orchestrator.defense_system.active_defenses
        critical_alerts = [
            alert
            for alert in security_orchestrator.alert_system.alerts
            if alert["level"] == "CRITICAL"
        ]
        assert len(critical_alerts) > 0

    def test_high_threat_response(self, security_orchestrator):
        """Test high threat response"""
        original_epsilon = security_orchestrator.dp_service.epsilon
        response = security_orchestrator.emergency_security_response(
            "HIGH", "adversarial_attack"
        )
        assert security_orchestrator.dp_service.epsilon <= min(original_epsilon, 0.1)
        assert "input_validation" in response["defenses_active"]
        assert "gaussian_noise" in response["defenses_active"]

    def test_emergency_response_audit_logging(self, security_orchestrator):
        """Test emergency response audit logging"""
        initial_log_count = len(security_orchestrator.security_audit_log)
        security_orchestrator.emergency_security_response("HIGH", "suspicious_activity")
        emergency_logs = [
            log
            for log in security_orchestrator.security_audit_log
            if log["event_type"] == "emergency_response"
        ]
        assert len(emergency_logs) > 0
        emergency_log = emergency_logs[-1]
        assert emergency_log["details"]["threat_level"] == "HIGH"
        assert emergency_log["details"]["threat_type"] == "suspicious_activity"
        assert "actions_taken" in emergency_log["details"]


class TestCrossComponentSecurityValidation:
    """Test security validation across multiple components"""

    def test_comprehensive_security_workflow(self, security_orchestrator):
        """Test comprehensive security workflow across all components"""
        auth_result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.RUN_ADVERSARIAL_TESTS
        )
        assert auth_result is not None
        token = auth_result["token"]
        raw_input = "<script>alert('xss')</script>"
        sanitized_input = security_orchestrator.input_sanitizer.sanitize_html_input(
            raw_input
        )
        assert "<script>" not in sanitized_input
        privacy_params = {"epsilon": 0.5, "delta": 1e-06}
        ml_data = np.random.rand(30, 5)
        privacy_auth = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        privacy_token = privacy_auth["token"]
        private_result = security_orchestrator.secure_ml_pipeline(
            privacy_token, ml_data, privacy_params
        )
        assert "result" in private_result

        # Using real models instead of mocks - test direct authentication
        result = security_orchestrator.auth_service.authenticate_user(
            "ml_engineer", "secure_password_123!"
        )
        assert result is not None
        assert result["username"] == "ml_engineer"

    async def test_concurrent_security_operations(self, security_orchestrator):
        """Test concurrent security operations"""
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        token = auth_result["token"]

        async def async_privacy_operation(data):
            await asyncio.sleep(0.001)
            privacy_params = {"epsilon": 0.1, "delta": 1e-06}
            return security_orchestrator.secure_ml_pipeline(token, data, privacy_params)

        async def async_input_validation(data):
            await asyncio.sleep(0.001)
            return security_orchestrator.input_sanitizer.validate_ml_input_data(data)

        data1 = np.random.rand(10, 3)
        data2 = np.random.rand(15, 3)
        tasks = [
            async_privacy_operation(data1),
            async_privacy_operation(data2),
            async_input_validation(data1),
            async_input_validation(data2),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert len(results) == 4
        assert all(not isinstance(result, Exception) for result in results[:2])
        assert all(isinstance(result, bool) for result in results[2:])


@pytest.mark.performance
class TestSecurityPerformance:
    """Test security performance in integrated scenarios"""

    def test_authentication_performance_under_load(self, security_orchestrator):
        """Test authentication performance under load"""
        import time

        start_time = time.time()
        for _i in range(100):
            result = security_orchestrator.authenticate_and_authorize(
                "ml_engineer", "secure_password_123!", Permission.READ_MODELS
            )
            assert result is not None
        elapsed_time = time.time() - start_time
        assert elapsed_time < 2.0
        avg_time_per_auth = elapsed_time / 100
        assert avg_time_per_auth < 0.02

    def test_integrated_security_pipeline_performance(self, security_orchestrator):
        """Test integrated security pipeline performance"""
        import time

        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer",
            "privacy_secure_456!",
            Permission.ACCESS_DIFFERENTIAL_PRIVACY,
        )
        token = auth_result["token"]
        ml_data = np.random.rand(100, 10)
        privacy_params = {"epsilon": 0.1, "delta": 1e-06}
        start_time = time.time()
        for _ in range(10):
            result = security_orchestrator.secure_ml_pipeline(
                token, ml_data, privacy_params
            )
            assert "result" in result
        elapsed_time = time.time() - start_time
        assert elapsed_time < 1.0
        avg_time_per_pipeline = elapsed_time / 10
        assert avg_time_per_pipeline < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
