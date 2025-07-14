"""
End-to-End Security Integration Tests

Tests complete security workflows and integration between different security
components to ensure comprehensive security coverage across Phase 3 ML features.

Integration Test Coverage:
- Authentication + Authorization flow
- Privacy-preserving ML with security validation
- Adversarial robustness in production pipeline
- Input sanitization throughout system
- Security audit trail integration
- Cross-component security validation
- Emergency security response scenarios
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import tempfile
import os
import json
import hashlib
import secrets

# Import security components for integration testing
from tests.unit.security.test_authentication import MockAuthenticationService
from tests.unit.security.test_authorization import MockAuthorizationService, Role, Permission
from tests.unit.security.test_input_sanitization import InputSanitizer
from tests.unit.security.test_privacy_preserving import MockDifferentialPrivacy, MockFederatedLearning
from tests.unit.security.test_adversarial_robustness import (
    MockAdversarialAttack, MockRobustnessEvaluator, MockDefenseSystem
)


class SecurityOrchestrator:
    """Orchestrates security components for end-to-end security"""
    
    def __init__(self):
        self.auth_service = MockAuthenticationService()
        self.authz_service = MockAuthorizationService()
        self.input_sanitizer = InputSanitizer()
        self.dp_service = MockDifferentialPrivacy()
        self.fl_service = MockFederatedLearning()
        self.defense_system = MockDefenseSystem()
        self.security_audit_log: List[Dict[str, Any]] = []
        self.alert_system = SecurityAlertSystem()
        
        # Set up default users and roles
        self._setup_default_security()
    
    def _setup_default_security(self):
        """Set up default users and security roles"""
        # Add test users
        self.auth_service.users["ml_engineer"] = {
            "user_id": "ml_eng_001",
            "email": "engineer@company.com",
            "password_hash": self.auth_service.hash_password("secure_password_123!"),
            "roles": ["ml_engineer"],
            "created_at": datetime.utcnow()
        }
        
        self.auth_service.users["privacy_officer"] = {
            "user_id": "privacy_001",
            "email": "privacy@company.com", 
            "password_hash": self.auth_service.hash_password("privacy_secure_456!"),
            "roles": ["privacy_officer"],
            "created_at": datetime.utcnow()
        }
        
        # Set up authorization roles
        self.authz_service.assign_role("ml_eng_001", Role.ML_ENGINEER)
        self.authz_service.assign_role("privacy_001", Role.PRIVACY_OFFICER)
    
    def authenticate_and_authorize(self, username: str, password: str, 
                                 required_permission: Permission) -> Optional[Dict[str, Any]]:
        """Complete authentication and authorization flow"""
        # Step 1: Authenticate user
        user_data = self.auth_service.authenticate_user(username, password)
        if not user_data:
            self._log_security_event("authentication_failed", {"username": username})
            return None
        
        # Step 2: Check authorization
        if not self.authz_service.has_permission(user_data["user_id"], required_permission):
            self._log_security_event("authorization_failed", {
                "user_id": user_data["user_id"],
                "permission": required_permission.value
            })
            return None
        
        # Step 3: Create session token
        token = self.auth_service.create_jwt_token(user_data)
        
        self._log_security_event("authentication_success", {
            "user_id": user_data["user_id"],
            "permission": required_permission.value
        })
        
        return {
            "user_data": user_data,
            "token": token,
            "permissions": self.authz_service.get_user_permissions(user_data["user_id"])
        }
    
    def secure_ml_pipeline(self, user_token: str, ml_data: Any, 
                          privacy_params: Dict[str, float]) -> Dict[str, Any]:
        """Execute secure ML pipeline with privacy preservation"""
        # Step 1: Validate token
        token_payload = self.auth_service.validate_jwt_token(user_token)
        if not token_payload:
            raise SecurityError("Invalid or expired token")
        
        # Step 2: Validate and sanitize ML input
        if not self.input_sanitizer.validate_ml_input_data(ml_data):
            raise SecurityError("Invalid ML input data")
        
        # Step 3: Validate privacy parameters
        epsilon = privacy_params.get("epsilon", 1.0)
        delta = privacy_params.get("delta", 1e-6)
        if not self.input_sanitizer.validate_privacy_parameters(epsilon, delta):
            raise SecurityError("Invalid privacy parameters")
        
        # Step 4: Check privacy budget
        if not self.dp_service.check_privacy_budget(epsilon):
            raise SecurityError("Insufficient privacy budget")
        
        # Step 5: Apply differential privacy
        # Simulate ML computation result
        ml_result = np.mean(ml_data) if isinstance(ml_data, np.ndarray) else 42.0
        private_result = self.dp_service.add_noise(ml_result, sensitivity=1.0)
        
        # Step 6: Log secure operation
        self._log_security_event("secure_ml_operation", {
            "user_id": token_payload["user_id"],
            "epsilon_used": epsilon,
            "data_size": len(ml_data) if hasattr(ml_data, '__len__') else 1
        })
        
        return {
            "result": private_result,
            "privacy_spent": self.dp_service.get_privacy_spent(),
            "timestamp": datetime.utcnow()
        }
    
    def adversarial_robustness_pipeline(self, user_token: str, model: Any, 
                                      test_data: np.ndarray) -> Dict[str, Any]:
        """Execute adversarial robustness testing pipeline"""
        # Step 1: Validate token and permissions
        token_payload = self.auth_service.validate_jwt_token(user_token)
        if not token_payload:
            raise SecurityError("Invalid token")
        
        if not self.authz_service.has_permission(token_payload["user_id"], 
                                               Permission.RUN_ADVERSARIAL_TESTS):
            raise SecurityError("Insufficient permissions for adversarial testing")
        
        # Step 2: Sanitize test data
        if not self.input_sanitizer.validate_ml_input_data(test_data):
            raise SecurityError("Invalid test data")
        
        # Step 3: Enable security defenses
        self.defense_system.enable_defense("gaussian_noise")
        self.defense_system.enable_defense("input_validation")
        
        # Step 4: Run adversarial attacks
        attacker = MockAdversarialAttack(epsilon=0.1)
        evaluator = MockRobustnessEvaluator()
        
        # Generate adversarial examples
        y_dummy = np.zeros(len(test_data))  # Dummy labels for testing
        X_adversarial = attacker.generate_adversarial_examples(test_data, y_dummy)
        
        # Apply defenses
        X_defended = self.defense_system.apply_gaussian_noise_defense(X_adversarial)
        
        # Evaluate robustness
        robustness_results = evaluator.evaluate_robustness(model, test_data, X_defended, y_dummy)
        
        # Step 5: Check for security-critical failures
        if robustness_results["attack_success_rate"] > 0.5:
            self.alert_system.trigger_alert("HIGH", "High adversarial attack success rate detected")
        
        # Step 6: Log robustness testing
        self._log_security_event("adversarial_robustness_test", {
            "user_id": token_payload["user_id"],
            "samples_tested": len(test_data),
            "attack_success_rate": robustness_results["attack_success_rate"],
            "robustness_score": robustness_results["robustness_score"]
        })
        
        return robustness_results
    
    def federated_learning_security_pipeline(self, clients: List[str]) -> Dict[str, Any]:
        """Execute secure federated learning pipeline"""
        # Step 1: Register clients with authentication
        registered_clients = []
        for client_id in clients:
            public_key = f"pubkey_{client_id}_{secrets.token_hex(16)}"
            if self.fl_service.register_client(client_id, public_key):
                registered_clients.append(client_id)
        
        # Step 2: Simulate encrypted model updates
        for client_id in registered_clients:
            # Simulate model update with privacy
            model_update = {
                "gradients": {
                    "layer1": [0.1, 0.2, 0.3],
                    "layer2": [0.4, 0.5, 0.6]
                },
                "privacy_budget": 0.5
            }
            
            # Encrypt and submit update
            encrypted_update = self.fl_service.encrypt_data(str(model_update))
            self.fl_service.submit_encrypted_update(client_id, encrypted_update)
        
        # Step 3: Secure aggregation
        aggregated_result = self.fl_service.aggregate_updates()
        
        # Step 4: Privacy accounting
        total_privacy_spent = len(registered_clients) * 0.5  # Sum of client privacy budgets
        
        # Step 5: Log federated learning operation
        self._log_security_event("federated_learning_round", {
            "registered_clients": len(registered_clients),
            "total_privacy_spent": total_privacy_spent,
            "aggregation_successful": aggregated_result is not None
        })
        
        return {
            "aggregated_result": aggregated_result,
            "registered_clients": len(registered_clients),
            "total_privacy_spent": total_privacy_spent
        }
    
    def emergency_security_response(self, threat_level: str, threat_type: str) -> Dict[str, Any]:
        """Execute emergency security response"""
        response_actions = []
        
        if threat_level == "CRITICAL":
            # Disable all ML operations
            response_actions.append("disable_ml_operations")
            
            # Invalidate all tokens
            response_actions.append("invalidate_all_tokens")
            
            # Enable maximum defenses
            for defense in self.defense_system.defense_methods:
                self.defense_system.enable_defense(defense)
            response_actions.append("enable_all_defenses")
            
        elif threat_level == "HIGH":
            # Reduce privacy budgets
            self.dp_service.epsilon = min(self.dp_service.epsilon, 0.1)
            response_actions.append("reduce_privacy_budget")
            
            # Enable core defenses
            self.defense_system.enable_defense("input_validation")
            self.defense_system.enable_defense("gaussian_noise")
            response_actions.append("enable_core_defenses")
        
        # Log emergency response
        self._log_security_event("emergency_response", {
            "threat_level": threat_level,
            "threat_type": threat_type,
            "actions_taken": response_actions,
            "timestamp": datetime.utcnow()
        })
        
        # Trigger alerts
        self.alert_system.trigger_alert(threat_level, f"Emergency response activated: {threat_type}")
        
        return {
            "threat_level": threat_level,
            "actions_taken": response_actions,
            "defenses_active": self.defense_system.active_defenses
        }
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for audit"""
        self.security_audit_log.append({
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "details": details,
            "event_id": hashlib.md5(f"{datetime.utcnow()}{event_type}".encode()).hexdigest()[:8]
        })


class SecurityAlertSystem:
    """Mock security alert system"""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "CRITICAL": 0,  # Always alert
            "HIGH": 1,      # Alert after 1 occurrence
            "MEDIUM": 3,    # Alert after 3 occurrences
            "LOW": 5        # Alert after 5 occurrences
        }
        self.alert_counts: Dict[str, int] = {}
    
    def trigger_alert(self, level: str, message: str) -> bool:
        """Trigger security alert"""
        alert_key = f"{level}:{message}"
        current_count = self.alert_counts.get(alert_key, 0) + 1
        self.alert_counts[alert_key] = current_count
        
        threshold = self.alert_thresholds.get(level, 1)
        
        if current_count >= threshold:
            self.alerts.append({
                "timestamp": datetime.utcnow(),
                "level": level,
                "message": message,
                "count": current_count,
                "alert_id": hashlib.md5(f"{datetime.utcnow()}{message}".encode()).hexdigest()[:8]
            })
            return True
        
        return False


class SecurityError(Exception):
    """Security-related exception"""
    pass


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
        
        # Check user data
        assert result["user_data"]["username"] == "ml_engineer"
        
        # Check permissions
        assert Permission.WRITE_MODELS in result["permissions"]
        assert Permission.READ_MODELS in result["permissions"]
    
    def test_failed_authentication(self, security_orchestrator):
        """Test failed authentication"""
        result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "wrong_password", Permission.WRITE_MODELS
        )
        
        assert result is None
        
        # Check audit log
        auth_failed_logs = [log for log in security_orchestrator.security_audit_log 
                           if log["event_type"] == "authentication_failed"]
        assert len(auth_failed_logs) > 0
    
    def test_failed_authorization(self, security_orchestrator):
        """Test failed authorization (valid user, insufficient permissions)"""
        result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.MANAGE_USERS  # Admin permission
        )
        
        assert result is None
        
        # Check audit log
        authz_failed_logs = [log for log in security_orchestrator.security_audit_log 
                            if log["event_type"] == "authorization_failed"]
        assert len(authz_failed_logs) > 0
    
    def test_privacy_officer_permissions(self, security_orchestrator):
        """Test privacy officer specific permissions"""
        result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        
        assert result is not None
        assert Permission.ACCESS_DIFFERENTIAL_PRIVACY in result["permissions"]
        assert Permission.CONFIGURE_PRIVACY_BUDGET in result["permissions"]


class TestSecureMLPipelineIntegration:
    """Test secure ML pipeline integration"""
    
    def test_complete_secure_ml_pipeline(self, security_orchestrator):
        """Test complete secure ML pipeline"""
        # Step 1: Authenticate and get token
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        assert auth_result is not None
        
        token = auth_result["token"]
        
        # Step 2: Prepare ML data
        ml_data = np.random.rand(100, 10)  # Valid ML data
        privacy_params = {"epsilon": 1.0, "delta": 1e-6}
        
        # Step 3: Execute secure ML pipeline
        result = security_orchestrator.secure_ml_pipeline(token, ml_data, privacy_params)
        
        assert "result" in result
        assert "privacy_spent" in result
        assert "timestamp" in result
        
        # Check that privacy was applied (result should be different from raw mean)
        raw_mean = np.mean(ml_data)
        private_result = result["result"]
        assert abs(private_result - raw_mean) > 0  # Noise should have been added
    
    def test_invalid_token_rejection(self, security_orchestrator):
        """Test rejection of invalid tokens"""
        invalid_token = "invalid.jwt.token"
        ml_data = np.random.rand(50, 5)
        privacy_params = {"epsilon": 1.0, "delta": 1e-6}
        
        with pytest.raises(SecurityError, match="Invalid or expired token"):
            security_orchestrator.secure_ml_pipeline(invalid_token, ml_data, privacy_params)
    
    def test_invalid_ml_data_rejection(self, security_orchestrator):
        """Test rejection of invalid ML data"""
        # Get valid token
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        token = auth_result["token"]
        
        # Invalid ML data (contains NaN)
        invalid_data = np.array([1.0, 2.0, float('nan'), 4.0])
        privacy_params = {"epsilon": 1.0, "delta": 1e-6}
        
        with pytest.raises(SecurityError, match="Invalid ML input data"):
            security_orchestrator.secure_ml_pipeline(token, invalid_data, privacy_params)
    
    def test_privacy_parameter_validation(self, security_orchestrator):
        """Test privacy parameter validation"""
        # Get valid token
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        token = auth_result["token"]
        
        ml_data = np.random.rand(50, 5)
        
        # Invalid privacy parameters
        invalid_privacy_params = {"epsilon": -1.0, "delta": 1e-6}  # Negative epsilon
        
        with pytest.raises(SecurityError, match="Invalid privacy parameters"):
            security_orchestrator.secure_ml_pipeline(token, ml_data, invalid_privacy_params)
    
    def test_privacy_budget_enforcement(self, security_orchestrator):
        """Test privacy budget enforcement"""
        # Get valid token
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        token = auth_result["token"]
        
        ml_data = np.random.rand(20, 3)
        
        # Exhaust privacy budget
        for _ in range(15):  # Should exceed budget
            privacy_params = {"epsilon": 1.0, "delta": 1e-6}
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
        # Step 1: Authenticate ML engineer
        auth_result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.RUN_ADVERSARIAL_TESTS
        )
        assert auth_result is not None
        
        token = auth_result["token"]
        
        # Step 2: Prepare test data and mock model
        test_data = np.random.rand(50, 10)
        
        # Mock model that just returns random predictions
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
        
        model = MockModel()
        
        # Step 3: Run adversarial robustness pipeline
        results = security_orchestrator.adversarial_robustness_pipeline(token, model, test_data)
        
        # Check results
        assert "clean_accuracy" in results
        assert "adversarial_accuracy" in results
        assert "robustness_score" in results
        assert "attack_success_rate" in results
        
        # Check that defenses were enabled
        assert "gaussian_noise" in security_orchestrator.defense_system.active_defenses
        assert "input_validation" in security_orchestrator.defense_system.active_defenses
    
    def test_insufficient_permissions_for_adversarial_testing(self, security_orchestrator):
        """Test insufficient permissions for adversarial testing"""
        # Step 1: Authenticate privacy officer (doesn't have adversarial testing permission)
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        token = auth_result["token"]
        
        # Step 2: Try to run adversarial testing (should fail)
        test_data = np.random.rand(30, 5)
        
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
        
        model = MockModel()
        
        with pytest.raises(SecurityError, match="Insufficient permissions for adversarial testing"):
            security_orchestrator.adversarial_robustness_pipeline(token, model, test_data)
    
    def test_security_alert_on_high_attack_success(self, security_orchestrator):
        """Test security alert on high attack success rate"""
        # Get valid token
        auth_result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.RUN_ADVERSARIAL_TESTS
        )
        token = auth_result["token"]
        
        # Mock model that's very vulnerable (always fails under attack)
        class VulnerableModel:
            def predict(self, X):
                # Return different predictions for clean vs adversarial (simulating vulnerability)
                return np.zeros(len(X))  # Always predict 0 for simplicity
        
        model = VulnerableModel()
        test_data = np.random.rand(20, 5)
        
        # This should trigger a security alert due to high vulnerability
        results = security_orchestrator.adversarial_robustness_pipeline(token, model, test_data)
        
        # Check that alert was triggered
        high_alerts = [alert for alert in security_orchestrator.alert_system.alerts 
                      if alert["level"] == "HIGH"]
        assert len(high_alerts) > 0


class TestFederatedLearningSecurityIntegration:
    """Test federated learning security integration"""
    
    def test_secure_federated_learning_pipeline(self, security_orchestrator):
        """Test secure federated learning pipeline"""
        clients = ["client_001", "client_002", "client_003"]
        
        # Run federated learning pipeline
        results = security_orchestrator.federated_learning_security_pipeline(clients)
        
        # Check results
        assert "aggregated_result" in results
        assert "registered_clients" in results
        assert "total_privacy_spent" in results
        
        # Check that clients were registered
        assert results["registered_clients"] == len(clients)
        
        # Check that aggregation was successful
        assert results["aggregated_result"] is not None
        
        # Check privacy accounting
        expected_privacy = len(clients) * 0.5  # Each client contributes 0.5 epsilon
        assert results["total_privacy_spent"] == expected_privacy
    
    def test_federated_learning_encryption(self, security_orchestrator):
        """Test encryption in federated learning"""
        clients = ["secure_client_1"]
        
        # Register client
        public_key = "test_public_key"
        success = security_orchestrator.fl_service.register_client(clients[0], public_key)
        assert success is True
        
        # Test encryption/decryption
        test_data = "sensitive_gradient_data"
        encrypted = security_orchestrator.fl_service.encrypt_data(test_data)
        
        # Encrypted data should be different from original
        assert encrypted != test_data.encode()
        
        # Should be able to decrypt
        decrypted = security_orchestrator.fl_service.cipher.decrypt(encrypted).decode()
        assert decrypted == test_data


class TestEmergencySecurityResponse:
    """Test emergency security response scenarios"""
    
    def test_critical_threat_response(self, security_orchestrator):
        """Test critical threat response"""
        response = security_orchestrator.emergency_security_response("CRITICAL", "data_breach")
        
        # Check response actions
        expected_actions = ["disable_ml_operations", "invalidate_all_tokens", "enable_all_defenses"]
        for action in expected_actions:
            assert action in response["actions_taken"]
        
        # Check that all defenses are enabled
        for defense in security_orchestrator.defense_system.defense_methods:
            assert defense in security_orchestrator.defense_system.active_defenses
        
        # Check alert was triggered
        critical_alerts = [alert for alert in security_orchestrator.alert_system.alerts 
                          if alert["level"] == "CRITICAL"]
        assert len(critical_alerts) > 0
    
    def test_high_threat_response(self, security_orchestrator):
        """Test high threat response"""
        original_epsilon = security_orchestrator.dp_service.epsilon
        
        response = security_orchestrator.emergency_security_response("HIGH", "adversarial_attack")
        
        # Check that privacy budget was reduced
        assert security_orchestrator.dp_service.epsilon <= min(original_epsilon, 0.1)
        
        # Check that core defenses were enabled
        assert "input_validation" in response["defenses_active"]
        assert "gaussian_noise" in response["defenses_active"]
    
    def test_emergency_response_audit_logging(self, security_orchestrator):
        """Test emergency response audit logging"""
        initial_log_count = len(security_orchestrator.security_audit_log)
        
        security_orchestrator.emergency_security_response("HIGH", "suspicious_activity")
        
        # Check that emergency response was logged
        emergency_logs = [log for log in security_orchestrator.security_audit_log 
                         if log["event_type"] == "emergency_response"]
        assert len(emergency_logs) > 0
        
        emergency_log = emergency_logs[-1]
        assert emergency_log["details"]["threat_level"] == "HIGH"
        assert emergency_log["details"]["threat_type"] == "suspicious_activity"
        assert "actions_taken" in emergency_log["details"]


class TestCrossComponentSecurityValidation:
    """Test security validation across multiple components"""
    
    def test_comprehensive_security_workflow(self, security_orchestrator):
        """Test comprehensive security workflow across all components"""
        # Step 1: Authentication and authorization
        auth_result = security_orchestrator.authenticate_and_authorize(
            "ml_engineer", "secure_password_123!", Permission.RUN_ADVERSARIAL_TESTS
        )
        assert auth_result is not None
        token = auth_result["token"]
        
        # Step 2: Input sanitization and validation
        raw_input = "<script>alert('xss')</script>"
        sanitized_input = security_orchestrator.input_sanitizer.sanitize_html_input(raw_input)
        assert "<script>" not in sanitized_input
        
        # Step 3: Privacy-preserving computation
        privacy_params = {"epsilon": 0.5, "delta": 1e-6}
        ml_data = np.random.rand(30, 5)
        
        # Switch to privacy officer for privacy operations
        privacy_auth = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        privacy_token = privacy_auth["token"]
        
        private_result = security_orchestrator.secure_ml_pipeline(
            privacy_token, ml_data, privacy_params
        )
        assert "result" in private_result
        
        # Step 4: Adversarial robustness testing
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
        
        model = MockModel()
        robustness_results = security_orchestrator.adversarial_robustness_pipeline(
            token, model, ml_data
        )
        assert "robustness_score" in robustness_results
        
        # Step 5: Verify comprehensive audit trail
        audit_events = [log["event_type"] for log in security_orchestrator.security_audit_log]
        expected_events = [
            "authentication_success",
            "secure_ml_operation", 
            "adversarial_robustness_test"
        ]
        
        for event in expected_events:
            assert event in audit_events
    
    def test_security_component_isolation(self, security_orchestrator):
        """Test that security component failures are isolated"""
        # Simulate component failure and ensure other components continue working
        
        # Step 1: Break authentication (simulate service failure)
        original_auth_method = security_orchestrator.auth_service.authenticate_user
        security_orchestrator.auth_service.authenticate_user = lambda u, p: None
        
        # Step 2: Verify other components still work independently
        # Input sanitization should still work
        test_input = "test<script>alert('xss')</script>input"
        sanitized = security_orchestrator.input_sanitizer.sanitize_html_input(test_input)
        assert "<script>" not in sanitized
        
        # Privacy components should still work
        dp_result = security_orchestrator.dp_service.add_noise(100.0)
        assert dp_result != 100.0  # Noise should be added
        
        # Defense system should still work
        test_data = np.random.rand(10, 3)
        defended_data = security_orchestrator.defense_system.apply_gaussian_noise_defense(test_data)
        assert not np.array_equal(defended_data, test_data)
        
        # Step 3: Restore authentication
        security_orchestrator.auth_service.authenticate_user = original_auth_method
    
    def test_security_configuration_consistency(self, security_orchestrator):
        """Test security configuration consistency across components"""
        # Test that security settings are consistently applied
        
        # Enable high security mode
        security_orchestrator.defense_system.enable_defense("gaussian_noise")
        security_orchestrator.defense_system.enable_defense("input_validation")
        
        # Reduce privacy budget for high security
        security_orchestrator.dp_service.epsilon = 0.1
        
        # Verify settings are applied consistently
        assert "gaussian_noise" in security_orchestrator.defense_system.active_defenses
        assert "input_validation" in security_orchestrator.defense_system.active_defenses
        assert security_orchestrator.dp_service.epsilon == 0.1
        
        # Test that operations respect high security settings
        ml_data = np.random.rand(20, 4)
        
        # Privacy operations should use reduced epsilon
        budget_available = security_orchestrator.dp_service.check_privacy_budget(0.05)
        assert budget_available is True
        
        budget_not_available = security_orchestrator.dp_service.check_privacy_budget(0.2)
        assert budget_not_available is False


@pytest.mark.asyncio
class TestAsyncSecurityIntegration:
    """Test asynchronous security integration"""
    
    async def test_async_authentication_flow(self, security_orchestrator):
        """Test asynchronous authentication flow"""
        async def async_authenticate(username: str, password: str):
            # Simulate async operation
            await asyncio.sleep(0.001)
            return security_orchestrator.auth_service.authenticate_user(username, password)
        
        # Test async authentication
        result = await async_authenticate("ml_engineer", "secure_password_123!")
        assert result is not None
        assert result["username"] == "ml_engineer"
    
    async def test_concurrent_security_operations(self, security_orchestrator):
        """Test concurrent security operations"""
        # Prepare authentication
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        token = auth_result["token"]
        
        # Define async operations
        async def async_privacy_operation(data):
            await asyncio.sleep(0.001)
            privacy_params = {"epsilon": 0.1, "delta": 1e-6}
            return security_orchestrator.secure_ml_pipeline(token, data, privacy_params)
        
        async def async_input_validation(data):
            await asyncio.sleep(0.001)
            return security_orchestrator.input_sanitizer.validate_ml_input_data(data)
        
        # Run operations concurrently
        data1 = np.random.rand(10, 3)
        data2 = np.random.rand(15, 3)
        
        tasks = [
            async_privacy_operation(data1),
            async_privacy_operation(data2),
            async_input_validation(data1),
            async_input_validation(data2)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all operations completed successfully
        assert len(results) == 4
        assert all(not isinstance(result, Exception) for result in results[:2])  # Privacy operations
        assert all(isinstance(result, bool) for result in results[2:])  # Validation operations


@pytest.mark.performance
class TestSecurityPerformance:
    """Test security performance in integrated scenarios"""
    
    def test_authentication_performance_under_load(self, security_orchestrator):
        """Test authentication performance under load"""
        import time
        
        start_time = time.time()
        
        # Perform many authentication attempts
        for i in range(100):
            result = security_orchestrator.authenticate_and_authorize(
                "ml_engineer", "secure_password_123!", Permission.READ_MODELS
            )
            assert result is not None
        
        elapsed_time = time.time() - start_time
        
        # Should handle 100 authentications quickly
        assert elapsed_time < 2.0
        
        avg_time_per_auth = elapsed_time / 100
        assert avg_time_per_auth < 0.02  # Less than 20ms per authentication
    
    def test_integrated_security_pipeline_performance(self, security_orchestrator):
        """Test integrated security pipeline performance"""
        import time
        
        # Get authentication token
        auth_result = security_orchestrator.authenticate_and_authorize(
            "privacy_officer", "privacy_secure_456!", Permission.ACCESS_DIFFERENTIAL_PRIVACY
        )
        token = auth_result["token"]
        
        # Test data
        ml_data = np.random.rand(100, 10)
        privacy_params = {"epsilon": 0.1, "delta": 1e-6}
        
        start_time = time.time()
        
        # Run complete security pipeline multiple times
        for _ in range(10):
            result = security_orchestrator.secure_ml_pipeline(token, ml_data, privacy_params)
            assert "result" in result
        
        elapsed_time = time.time() - start_time
        
        # Should complete 10 pipelines quickly
        assert elapsed_time < 1.0
        
        avg_time_per_pipeline = elapsed_time / 10
        assert avg_time_per_pipeline < 0.1  # Less than 100ms per pipeline


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])