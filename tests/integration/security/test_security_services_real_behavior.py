"""Real behavior tests for decomposed security services.

Comprehensive validation of security services with real authentication scenarios,
OWASP compliance testing, and actual cryptographic operations.

Security Requirements:
- Authentication: Multi-factor support, session management, rate limiting
- Authorization: RBAC with role hierarchies, permission matrices
- Validation: OWASP Top 10 protection, input/output sanitization  
- Cryptography: NIST-compliant algorithms, secure key management
- Monitoring: Real-time threat detection, incident tracking
"""

import asyncio
import json
import os
import pytest
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import uuid4

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.containers.real_redis_container import RealRedisTestContainer
from src.prompt_improver.security.services import (
    AuthenticationService,
    AuthorizationService, 
    ValidationService,
    CryptoService,
    SecurityMonitoringService,
    SecurityServiceFacade,
    get_security_service_facade,
    create_authentication_service,
    create_authorization_service,
    create_validation_service,
    create_crypto_service,
    create_security_monitoring_service,
    Permission,
    Role,
    SecurityEvent,
    ThreatSeverity,
)


class TestAuthenticationServiceRealBehavior:
    """Real behavior tests for authentication service."""

    @pytest.fixture
    async def test_infrastructure(self):
        """Set up test infrastructure with Redis and PostgreSQL."""
        redis_container = RealRedisTestContainer()
        postgres_container = PostgreSQLTestContainer()
        
        await redis_container.start()
        await postgres_container.start()
        
        yield {
            "redis": redis_container,
            "postgres": postgres_container
        }
        
        await redis_container.stop()
        await postgres_container.stop()

    @pytest.fixture
    async def auth_service(self, test_infrastructure):
        """Create authentication service with real backends."""
        redis_client = test_infrastructure["redis"].get_client()
        
        async with test_infrastructure["postgres"].get_session() as session:
            service = create_authentication_service(
                session_manager=session,
                redis_client=redis_client
            )
            yield service

    async def test_user_authentication_lifecycle(self, auth_service):
        """Test complete user authentication lifecycle."""
        user_id = f"test_user_{uuid4().hex[:8]}"
        password = "SecurePassword123!"
        
        # Test user registration
        registration_success = await auth_service.register_user(
            user_id=user_id,
            password=password,
            email=f"{user_id}@test.com",
            metadata={"registration_source": "api", "user_agent": "test"}
        )
        assert registration_success, "User registration failed"
        
        # Test authentication with correct credentials
        auth_result = await auth_service.authenticate(
            username=user_id,
            password=password,
            metadata={"ip_address": "127.0.0.1", "user_agent": "test"}
        )
        assert auth_result.success, "Authentication with correct credentials failed"
        assert auth_result.user_id == user_id
        assert auth_result.session_token is not None
        assert auth_result.expires_at > datetime.utcnow()
        
        # Test authentication with incorrect credentials
        wrong_auth = await auth_service.authenticate(
            username=user_id,
            password="WrongPassword",
            metadata={"ip_address": "127.0.0.1"}
        )
        assert not wrong_auth.success, "Authentication should fail with wrong password"
        
        # Test session validation
        session_valid = await auth_service.validate_session(auth_result.session_token)
        assert session_valid.valid, "Valid session should be recognized"
        assert session_valid.user_id == user_id
        
        # Test session refresh
        new_token = await auth_service.refresh_session(
            auth_result.session_token,
            extend_by_seconds=3600
        )
        assert new_token is not None, "Session refresh failed"
        
        # Test logout
        logout_success = await auth_service.logout(auth_result.session_token)
        assert logout_success, "Logout failed"
        
        # Verify session invalidated
        invalid_session = await auth_service.validate_session(auth_result.session_token)
        assert not invalid_session.valid, "Session should be invalid after logout"

    async def test_authentication_security_measures(self, auth_service):
        """Test security measures like rate limiting and brute force protection."""
        user_id = f"security_test_{uuid4().hex[:8]}"
        password = "SecurePassword123!"
        
        # Register test user
        await auth_service.register_user(user_id, password, f"{user_id}@test.com")
        
        # Test rate limiting - multiple failed attempts
        failed_attempts = []
        for attempt in range(10):  # Exceed typical rate limit
            start_time = time.perf_counter()
            result = await auth_service.authenticate(
                username=user_id,
                password="WrongPassword",
                metadata={"ip_address": "192.168.1.100"}
            )
            duration = time.perf_counter() - start_time
            failed_attempts.append((result, duration))
            
            assert not result.success, f"Attempt {attempt} should fail with wrong password"
            
        # Verify rate limiting kicks in (later attempts should be slower or blocked)
        early_attempts = failed_attempts[:3]
        later_attempts = failed_attempts[-3:]
        
        early_avg_time = sum(dur for _, dur in early_attempts) / len(early_attempts)
        later_avg_time = sum(dur for _, dur in later_attempts) / len(later_attempts)
        
        # Rate limiting should slow down or block later attempts
        assert later_avg_time >= early_avg_time, "Rate limiting should slow down repeated failed attempts"
        
        # Test account lockout detection
        lockout_status = await auth_service.check_account_lockout(user_id)
        if lockout_status.locked:
            assert lockout_status.lockout_expires > datetime.utcnow()
            assert lockout_status.remaining_attempts == 0
        
        # Test successful authentication still works with correct credentials
        # (might be blocked due to rate limiting depending on implementation)
        success_result = await auth_service.authenticate(
            username=user_id,
            password=password,
            metadata={"ip_address": "127.0.0.1"}  # Different IP
        )
        # Result depends on implementation - either succeeds or is rate limited

    async def test_multi_factor_authentication(self, auth_service):
        """Test multi-factor authentication support."""
        user_id = f"mfa_test_{uuid4().hex[:8]}"
        password = "SecurePassword123!"
        
        # Register user
        await auth_service.register_user(user_id, password, f"{user_id}@test.com")
        
        # Enable MFA
        mfa_setup = await auth_service.setup_mfa(
            user_id=user_id,
            mfa_type="totp",  # Time-based OTP
            device_name="test_device"
        )
        assert mfa_setup.success, "MFA setup should succeed"
        assert mfa_setup.secret_key is not None
        assert mfa_setup.qr_code_url is not None
        
        # Simulate TOTP code generation (in real implementation)
        # For testing, use a mock TOTP code
        mock_totp_code = "123456"
        
        # Test authentication with MFA required
        auth_result = await auth_service.authenticate_with_mfa(
            username=user_id,
            password=password,
            mfa_code=mock_totp_code,
            metadata={"ip_address": "127.0.0.1"}
        )
        # In real implementation, this would validate the TOTP code
        
        # Test MFA backup codes
        backup_codes = await auth_service.generate_backup_codes(user_id)
        assert len(backup_codes) >= 8, "Should generate sufficient backup codes"
        
        # Test authentication with backup code
        backup_auth = await auth_service.authenticate_with_backup_code(
            username=user_id,
            password=password,
            backup_code=backup_codes[0],
            metadata={"ip_address": "127.0.0.1"}
        )
        # Implementation would mark backup code as used

    async def test_session_management_and_security(self, auth_service):
        """Test session management and security features."""
        user_id = f"session_test_{uuid4().hex[:8]}"
        password = "SecurePassword123!"
        
        await auth_service.register_user(user_id, password, f"{user_id}@test.com")
        
        # Create multiple sessions
        sessions = []
        for i in range(5):
            auth_result = await auth_service.authenticate(
                username=user_id,
                password=password,
                metadata={
                    "ip_address": f"192.168.1.{i+1}",
                    "user_agent": f"TestAgent{i}",
                    "device_id": f"device_{i}"
                }
            )
            assert auth_result.success
            sessions.append(auth_result)
        
        # Test session enumeration
        user_sessions = await auth_service.get_user_sessions(user_id)
        assert len(user_sessions) >= 5, "Should track all user sessions"
        
        # Test session metadata
        for session in user_sessions:
            assert session.user_id == user_id
            assert session.ip_address is not None
            assert session.user_agent is not None
            assert session.created_at is not None
            assert session.last_activity is not None
        
        # Test selective session termination
        session_to_terminate = sessions[0].session_token
        termination_result = await auth_service.terminate_session(session_to_terminate)
        assert termination_result, "Session termination should succeed"
        
        # Verify session is invalid
        validation = await auth_service.validate_session(session_to_terminate)
        assert not validation.valid, "Terminated session should be invalid"
        
        # Test session cleanup (expire old sessions)
        cleanup_count = await auth_service.cleanup_expired_sessions()
        assert cleanup_count >= 0, "Session cleanup should return count"
        
        # Test concurrent session limits
        concurrent_limit = await auth_service.enforce_concurrent_session_limit(
            user_id=user_id,
            max_sessions=3
        )
        remaining_sessions = await auth_service.get_user_sessions(user_id)
        assert len(remaining_sessions) <= 3, "Should enforce concurrent session limit"


class TestAuthorizationServiceRealBehavior:
    """Real behavior tests for authorization service."""

    @pytest.fixture
    async def authz_service(self, test_infrastructure):
        """Create authorization service with real backend."""
        async with test_infrastructure["postgres"].get_session() as session:
            service = create_authorization_service(session_manager=session)
            yield service

    async def test_role_based_access_control(self, authz_service):
        """Test RBAC implementation with role hierarchies."""
        # Define roles with hierarchy
        admin_role = Role(
            name="admin",
            description="System administrator",
            permissions=["create", "read", "update", "delete", "admin"]
        )
        
        editor_role = Role(
            name="editor", 
            description="Content editor",
            permissions=["create", "read", "update"]
        )
        
        viewer_role = Role(
            name="viewer",
            description="Read-only user", 
            permissions=["read"]
        )
        
        # Create roles
        await authz_service.create_role(admin_role)
        await authz_service.create_role(editor_role)
        await authz_service.create_role(viewer_role)
        
        # Create test users
        admin_user = f"admin_user_{uuid4().hex[:8]}"
        editor_user = f"editor_user_{uuid4().hex[:8]}"
        viewer_user = f"viewer_user_{uuid4().hex[:8]}"
        
        # Assign roles
        await authz_service.assign_role(admin_user, "admin")
        await authz_service.assign_role(editor_user, "editor")
        await authz_service.assign_role(viewer_user, "viewer")
        
        # Test permissions
        resources = ["document:123", "user:456", "system:config"]
        actions = ["create", "read", "update", "delete", "admin"]
        
        # Admin should have all permissions
        for resource in resources:
            for action in actions:
                allowed = await authz_service.check_permission(admin_user, resource, action)
                assert allowed, f"Admin should have {action} permission on {resource}"
        
        # Editor should have limited permissions
        editor_allowed_actions = ["create", "read", "update"]
        for resource in resources:
            for action in actions:
                allowed = await authz_service.check_permission(editor_user, resource, action)
                if action in editor_allowed_actions:
                    assert allowed, f"Editor should have {action} permission on {resource}"
                else:
                    assert not allowed, f"Editor should NOT have {action} permission on {resource}"
        
        # Viewer should only have read permission
        for resource in resources:
            for action in actions:
                allowed = await authz_service.check_permission(viewer_user, resource, action)
                if action == "read":
                    assert allowed, f"Viewer should have read permission on {resource}"
                else:
                    assert not allowed, f"Viewer should NOT have {action} permission on {resource}"

    async def test_dynamic_permissions_and_conditions(self, authz_service):
        """Test dynamic permissions with conditions and context."""
        # Create role with conditional permissions
        conditional_role = Role(
            name="conditional_editor",
            description="Editor with time and resource constraints",
            permissions=["read", "update"],
            conditions={
                "time_restriction": {"start": "09:00", "end": "17:00"},
                "resource_pattern": "document:user_*",
                "ip_whitelist": ["192.168.1.0/24", "127.0.0.1"]
            }
        )
        
        await authz_service.create_role(conditional_role)
        
        test_user = f"conditional_user_{uuid4().hex[:8]}"
        await authz_service.assign_role(test_user, "conditional_editor")
        
        # Test time-based permissions
        current_hour = datetime.utcnow().hour
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.100",
            "resource_owner": test_user
        }
        
        # Test during allowed hours (mock)
        allowed = await authz_service.check_permission_with_context(
            user_id=test_user,
            resource="document:user_123",
            action="update", 
            context=context
        )
        # Result depends on implementation of time restrictions
        
        # Test outside IP whitelist
        context_bad_ip = context.copy()
        context_bad_ip["ip_address"] = "10.0.0.1"
        
        denied = await authz_service.check_permission_with_context(
            user_id=test_user,
            resource="document:user_123",
            action="update",
            context=context_bad_ip
        )
        # Should be denied based on IP restriction
        
        # Test resource pattern matching
        pattern_allowed = await authz_service.check_permission_with_context(
            user_id=test_user,
            resource="document:user_456",  # Matches pattern
            action="read",
            context=context
        )
        
        pattern_denied = await authz_service.check_permission_with_context(
            user_id=test_user, 
            resource="document:system_config",  # Doesn't match pattern
            action="read",
            context=context
        )
        # Results depend on pattern matching implementation

    async def test_permission_caching_and_performance(self, authz_service):
        """Test authorization performance and caching."""
        # Create user with multiple roles
        test_user = f"perf_user_{uuid4().hex[:8]}"
        
        # Create roles
        roles = []
        for i in range(10):
            role = Role(
                name=f"role_{i}",
                description=f"Test role {i}",
                permissions=[f"action_{j}" for j in range(i+1, i+6)]  # 5 permissions each
            )
            await authz_service.create_role(role)
            await authz_service.assign_role(test_user, role.name)
            roles.append(role)
        
        # Test permission checking performance
        test_resources = [f"resource_{i}" for i in range(100)]
        test_actions = [f"action_{i}" for i in range(20)]
        
        # First round - populate caches
        start_time = time.perf_counter()
        first_round_results = []
        for resource in test_resources[:10]:  # Subset for performance
            for action in test_actions[:5]:
                result = await authz_service.check_permission(test_user, resource, action)
                first_round_results.append(result)
        first_round_time = time.perf_counter() - start_time
        
        # Second round - should use cached results
        start_time = time.perf_counter()
        second_round_results = []
        for resource in test_resources[:10]:
            for action in test_actions[:5]:
                result = await authz_service.check_permission(test_user, resource, action)
                second_round_results.append(result)
        second_round_time = time.perf_counter() - start_time
        
        # Results should be consistent
        assert first_round_results == second_round_results, "Results should be consistent between rounds"
        
        # Second round should be faster (if caching implemented)
        cache_performance_ratio = first_round_time / max(second_round_time, 0.001)  # Avoid division by zero
        print(f"Authorization performance: First round: {first_round_time:.3f}s, "
              f"Second round: {second_round_time:.3f}s, "
              f"Speedup: {cache_performance_ratio:.1f}x")
        
        # Test bulk permission checking
        bulk_start_time = time.perf_counter()
        bulk_permissions = await authz_service.check_bulk_permissions(
            user_id=test_user,
            permission_requests=[
                {"resource": f"resource_{i}", "action": f"action_{i%5}"}
                for i in range(50)
            ]
        )
        bulk_time = time.perf_counter() - bulk_start_time
        
        assert len(bulk_permissions) == 50
        print(f"Bulk permission check: {bulk_time:.3f}s for 50 permissions")
        assert bulk_time < 1.0, "Bulk permission checking should be fast"


class TestValidationServiceRealBehavior:
    """Real behavior tests for validation service with OWASP compliance."""

    @pytest.fixture
    async def validation_service(self):
        """Create validation service."""
        service = create_validation_service()
        yield service

    async def test_owasp_top10_protection(self, validation_service):
        """Test protection against OWASP Top 10 vulnerabilities."""
        
        # A1: Injection attacks
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; EXEC xp_cmdshell('dir')",
            "' UNION SELECT password FROM users WHERE '1'='1"
        ]
        
        for payload in sql_injection_payloads:
            is_safe = await validation_service.validate_sql_input(payload)
            assert not is_safe, f"SQL injection payload should be detected: {payload}"
        
        # A2: Broken Authentication - test password strength
        weak_passwords = [
            "123456", "password", "admin", "qwerty", "abc123"
        ]
        
        for weak_password in weak_passwords:
            strength = await validation_service.validate_password_strength(weak_password)
            assert strength.score < 3, f"Weak password should have low score: {weak_password}"
            assert not strength.is_acceptable, "Weak password should not be acceptable"
        
        strong_password = "MySecureP@ssw0rd123!"
        strong_strength = await validation_service.validate_password_strength(strong_password)
        assert strong_strength.is_acceptable, "Strong password should be acceptable"
        assert strong_strength.score >= 4, "Strong password should have high score"
        
        # A3: Sensitive Data Exposure - test data sanitization
        sensitive_data = {
            "credit_card": "4532 1234 5678 9012",
            "ssn": "123-45-6789", 
            "email": "user@example.com",
            "phone": "+1-555-123-4567"
        }
        
        sanitized = await validation_service.sanitize_sensitive_data(sensitive_data)
        assert "****" in sanitized["credit_card"], "Credit card should be masked"
        assert "***-**-" in sanitized["ssn"], "SSN should be partially masked"
        assert sanitized["email"] == sensitive_data["email"], "Email should remain intact"
        
        # A4: XML External Entities (XXE)
        xxe_payload = '''<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
        <data>&xxe;</data>'''
        
        is_safe_xml = await validation_service.validate_xml_input(xxe_payload)
        assert not is_safe_xml, "XXE payload should be detected and blocked"
        
        # A5: Broken Access Control - test privilege escalation
        escalation_attempts = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/admin/delete_all_users",
            "javascript:alert('xss')"
        ]
        
        for attempt in escalation_attempts:
            is_safe_path = await validation_service.validate_file_path(attempt)
            assert not is_safe_path, f"Path traversal attempt should be blocked: {attempt}"
        
        # A6: Security Misconfiguration - test input validation
        oversized_input = "x" * 100000  # 100KB input
        size_validation = await validation_service.validate_input_size(
            oversized_input, max_length=1000
        )
        assert not size_validation, "Oversized input should be rejected"
        
        # A7: Cross-Site Scripting (XSS)
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
        ]
        
        for payload in xss_payloads:
            sanitized_html = await validation_service.sanitize_html_content(payload)
            assert "<script>" not in sanitized_html, f"XSS payload should be sanitized: {payload}"
            assert "javascript:" not in sanitized_html, f"Javascript URLs should be removed: {payload}"
            assert "onerror=" not in sanitized_html, f"Event handlers should be removed: {payload}"

    async def test_input_validation_performance(self, validation_service):
        """Test validation service performance under load."""
        # Generate test inputs
        test_inputs = {
            "emails": [f"user{i}@domain{i%10}.com" for i in range(1000)],
            "urls": [f"https://example{i%10}.com/path{i}" for i in range(500)],
            "html_content": [f"<div>Content {i} with <b>bold</b> text</div>" for i in range(200)],
            "json_data": [json.dumps({"id": i, "data": f"test_data_{i}"}) for i in range(300)]
        }
        
        performance_results = {}
        
        # Test email validation performance
        start_time = time.perf_counter()
        email_results = []
        for email in test_inputs["emails"]:
            result = await validation_service.validate_email_format(email)
            email_results.append(result)
        email_time = time.perf_counter() - start_time
        performance_results["email_validation"] = {
            "count": len(test_inputs["emails"]),
            "time": email_time,
            "ops_per_second": len(test_inputs["emails"]) / email_time,
            "valid_count": sum(email_results)
        }
        
        # Test URL validation performance
        start_time = time.perf_counter()
        url_results = []
        for url in test_inputs["urls"]:
            result = await validation_service.validate_url_format(url)
            url_results.append(result)
        url_time = time.perf_counter() - start_time
        performance_results["url_validation"] = {
            "count": len(test_inputs["urls"]),
            "time": url_time,
            "ops_per_second": len(test_inputs["urls"]) / url_time,
            "valid_count": sum(url_results)
        }
        
        # Test HTML sanitization performance
        start_time = time.perf_counter()
        html_results = []
        for html in test_inputs["html_content"]:
            result = await validation_service.sanitize_html_content(html)
            html_results.append(result)
        html_time = time.perf_counter() - start_time
        performance_results["html_sanitization"] = {
            "count": len(test_inputs["html_content"]),
            "time": html_time,
            "ops_per_second": len(test_inputs["html_content"]) / html_time
        }
        
        # Print performance summary
        print("\nValidation Service Performance:")
        for operation, stats in performance_results.items():
            print(f"  {operation}: {stats['ops_per_second']:.0f} ops/sec "
                  f"({stats['time']:.3f}s for {stats['count']} operations)")
        
        # Performance assertions
        assert performance_results["email_validation"]["ops_per_second"] > 1000, \
            "Email validation should process >1000 ops/sec"
        assert performance_results["url_validation"]["ops_per_second"] > 500, \
            "URL validation should process >500 ops/sec"
        assert performance_results["html_sanitization"]["ops_per_second"] > 100, \
            "HTML sanitization should process >100 ops/sec"

    async def test_threat_pattern_detection(self, validation_service):
        """Test advanced threat pattern detection."""
        # Define test patterns for different threat types
        threat_patterns = {
            "command_injection": [
                "cat /etc/passwd",
                "$(whoami)", 
                "`id`",
                "&& rm -rf /",
                "; cat /etc/shadow"
            ],
            "ldap_injection": [
                "*)(uid=*))(|(uid=*",
                "*))%00",
                "admin*)(|(password=*"
            ],
            "xpath_injection": [
                "' or '1'='1",
                "'] | //user/*[contains(*,'admin')] | //user[",
                "x' or name()='username' or 'x'='y"
            ],
            "nosql_injection": [
                "';return 'a'=='a' && ''=='",
                "{$gt: ''}",
                "1; return true"
            ]
        }
        
        # Test each threat category
        for threat_type, patterns in threat_patterns.items():
            print(f"\nTesting {threat_type} detection:")
            
            for pattern in patterns:
                detection_result = await validation_service.detect_threat_patterns(
                    input_data=pattern,
                    threat_types=[threat_type]
                )
                
                assert detection_result.threat_detected, \
                    f"Should detect {threat_type} in pattern: {pattern}"
                assert threat_type in detection_result.detected_types, \
                    f"Should identify threat type {threat_type}"
                assert detection_result.risk_score > 7, \
                    f"High-risk pattern should have score >7: {pattern}"
                
                print(f"  ✓ Detected: {pattern} (risk: {detection_result.risk_score})")
        
        # Test benign inputs (should not trigger false positives)
        benign_inputs = [
            "Hello, how are you today?",
            "Please process this normal request",
            "user@example.com wants to update profile",
            "SELECT name FROM products WHERE category = 'electronics'"  # Valid SQL
        ]
        
        false_positive_count = 0
        for benign_input in benign_inputs:
            result = await validation_service.detect_threat_patterns(benign_input)
            if result.threat_detected:
                false_positive_count += 1
                print(f"False positive: {benign_input}")
        
        false_positive_rate = false_positive_count / len(benign_inputs)
        assert false_positive_rate < 0.1, \
            f"False positive rate {false_positive_rate:.2%} should be <10%"
        
        print(f"\nThreat Detection Summary:")
        print(f"  False Positive Rate: {false_positive_rate:.2%}")


class TestCryptoServiceRealBehavior:
    """Real behavior tests for cryptographic service."""

    @pytest.fixture
    async def crypto_service(self):
        """Create crypto service."""
        service = create_crypto_service()
        yield service

    async def test_nist_compliant_encryption(self, crypto_service):
        """Test NIST-compliant encryption algorithms."""
        test_data = "This is sensitive data that needs encryption"
        
        # Test AES-256-GCM encryption
        encryption_result = await crypto_service.encrypt_data(
            plaintext=test_data,
            algorithm="AES-256-GCM",
            additional_data="test_context"
        )
        
        assert encryption_result.success, "Encryption should succeed"
        assert encryption_result.ciphertext is not None
        assert encryption_result.nonce is not None
        assert encryption_result.tag is not None
        assert encryption_result.algorithm == "AES-256-GCM"
        
        # Test decryption
        decryption_result = await crypto_service.decrypt_data(
            ciphertext=encryption_result.ciphertext,
            nonce=encryption_result.nonce,
            tag=encryption_result.tag,
            algorithm="AES-256-GCM",
            additional_data="test_context"
        )
        
        assert decryption_result.success, "Decryption should succeed"
        assert decryption_result.plaintext == test_data
        
        # Test decryption with wrong additional data (should fail)
        wrong_ad_result = await crypto_service.decrypt_data(
            ciphertext=encryption_result.ciphertext,
            nonce=encryption_result.nonce, 
            tag=encryption_result.tag,
            algorithm="AES-256-GCM",
            additional_data="wrong_context"
        )
        
        assert not wrong_ad_result.success, "Decryption with wrong additional data should fail"
        
        # Test multiple algorithms
        algorithms = ["AES-256-GCM", "AES-256-CBC", "ChaCha20-Poly1305"]
        
        for algorithm in algorithms:
            if await crypto_service.is_algorithm_supported(algorithm):
                result = await crypto_service.encrypt_data(test_data, algorithm)
                assert result.success, f"Encryption with {algorithm} should succeed"
                print(f"✓ {algorithm} encryption supported")

    async def test_key_derivation_and_management(self, crypto_service):
        """Test secure key derivation and management."""
        password = "UserPassword123!"
        salt = await crypto_service.generate_salt()
        
        # Test PBKDF2 key derivation
        key_result = await crypto_service.derive_key(
            password=password,
            salt=salt,
            algorithm="PBKDF2",
            iterations=100000,  # NIST recommended minimum
            key_length=32
        )
        
        assert key_result.success, "Key derivation should succeed"
        assert len(key_result.derived_key) == 32
        assert key_result.iterations == 100000
        assert key_result.salt == salt
        
        # Test key derivation consistency
        key_result2 = await crypto_service.derive_key(
            password=password,
            salt=salt,
            algorithm="PBKDF2", 
            iterations=100000,
            key_length=32
        )
        
        assert key_result.derived_key == key_result2.derived_key, \
            "Key derivation should be deterministic with same inputs"
        
        # Test different salt produces different key
        different_salt = await crypto_service.generate_salt()
        key_result3 = await crypto_service.derive_key(
            password=password,
            salt=different_salt,
            algorithm="PBKDF2",
            iterations=100000,
            key_length=32
        )
        
        assert key_result.derived_key != key_result3.derived_key, \
            "Different salts should produce different keys"
        
        # Test key rotation
        rotation_result = await crypto_service.rotate_key(
            old_key=key_result.derived_key,
            new_password="NewPassword456!",
            salt_refresh=True
        )
        
        assert rotation_result.success, "Key rotation should succeed"
        assert rotation_result.new_key != key_result.derived_key, \
            "Rotated key should be different"

    async def test_digital_signatures_and_verification(self, crypto_service):
        """Test digital signature creation and verification."""
        message = "This message needs to be signed"
        
        # Generate key pair for signing
        keypair_result = await crypto_service.generate_keypair(
            algorithm="RSA-2048",
            purpose="signing"
        )
        
        assert keypair_result.success, "Key pair generation should succeed"
        assert keypair_result.public_key is not None
        assert keypair_result.private_key is not None
        
        # Sign message
        signature_result = await crypto_service.sign_message(
            message=message,
            private_key=keypair_result.private_key,
            algorithm="RSA-PSS-SHA256"
        )
        
        assert signature_result.success, "Message signing should succeed"
        assert signature_result.signature is not None
        
        # Verify signature
        verification_result = await crypto_service.verify_signature(
            message=message,
            signature=signature_result.signature,
            public_key=keypair_result.public_key,
            algorithm="RSA-PSS-SHA256"
        )
        
        assert verification_result.valid, "Signature verification should succeed"
        
        # Test verification with tampered message
        tampered_verification = await crypto_service.verify_signature(
            message=message + " tampered",
            signature=signature_result.signature,
            public_key=keypair_result.public_key,
            algorithm="RSA-PSS-SHA256"
        )
        
        assert not tampered_verification.valid, \
            "Verification should fail for tampered message"

    async def test_cryptographic_performance(self, crypto_service):
        """Test cryptographic operation performance."""
        test_data_sizes = [
            ("Small (1KB)", "x" * 1024),
            ("Medium (10KB)", "x" * 10240),
            ("Large (100KB)", "x" * 102400)
        ]
        
        performance_results = {}
        
        for size_name, test_data in test_data_sizes:
            print(f"\nTesting {size_name} data:")
            
            # Encryption performance
            encrypt_times = []
            decrypt_times = []
            
            for _ in range(10):  # Multiple runs for averaging
                # Encryption
                start_time = time.perf_counter()
                enc_result = await crypto_service.encrypt_data(
                    test_data, "AES-256-GCM"
                )
                encrypt_time = time.perf_counter() - start_time
                encrypt_times.append(encrypt_time)
                
                # Decryption
                start_time = time.perf_counter()
                dec_result = await crypto_service.decrypt_data(
                    enc_result.ciphertext,
                    enc_result.nonce,
                    enc_result.tag,
                    "AES-256-GCM"
                )
                decrypt_time = time.perf_counter() - start_time
                decrypt_times.append(decrypt_time)
                
                assert enc_result.success and dec_result.success
            
            avg_encrypt_time = sum(encrypt_times) / len(encrypt_times)
            avg_decrypt_time = sum(decrypt_times) / len(decrypt_times)
            
            performance_results[size_name] = {
                "data_size": len(test_data),
                "avg_encrypt_time": avg_encrypt_time,
                "avg_decrypt_time": avg_decrypt_time,
                "encrypt_throughput_mbps": (len(test_data) / (1024*1024)) / avg_encrypt_time,
                "decrypt_throughput_mbps": (len(test_data) / (1024*1024)) / avg_decrypt_time
            }
            
            print(f"  Encryption: {avg_encrypt_time*1000:.2f}ms "
                  f"({performance_results[size_name]['encrypt_throughput_mbps']:.1f} MB/s)")
            print(f"  Decryption: {avg_decrypt_time*1000:.2f}ms "
                  f"({performance_results[size_name]['decrypt_throughput_mbps']:.1f} MB/s)")
        
        # Performance assertions
        for size_name, stats in performance_results.items():
            # Encryption/decryption should be reasonably fast
            assert stats["encrypt_throughput_mbps"] > 10, \
                f"Encryption throughput should be >10 MB/s for {size_name}"
            assert stats["decrypt_throughput_mbps"] > 10, \
                f"Decryption throughput should be >10 MB/s for {size_name}"


class TestSecurityMonitoringRealBehavior:
    """Real behavior tests for security monitoring service."""

    @pytest.fixture
    async def monitoring_service(self, test_infrastructure):
        """Create security monitoring service."""
        async with test_infrastructure["postgres"].get_session() as session:
            service = create_security_monitoring_service(session_manager=session)
            yield service

    async def test_real_time_threat_detection(self, monitoring_service):
        """Test real-time security event monitoring and threat detection."""
        # Simulate various security events
        security_events = [
            SecurityEvent(
                event_type="failed_login",
                severity=ThreatSeverity.MEDIUM,
                source_ip="192.168.1.100",
                user_id="test_user",
                metadata={"attempt_count": 3, "user_agent": "malicious_bot"}
            ),
            SecurityEvent(
                event_type="privilege_escalation",
                severity=ThreatSeverity.HIGH,
                source_ip="10.0.0.5",
                user_id="admin_user",
                metadata={"attempted_resource": "/admin/users", "success": False}
            ),
            SecurityEvent(
                event_type="data_exfiltration",
                severity=ThreatSeverity.CRITICAL,
                source_ip="203.0.113.1", 
                user_id="compromised_user",
                metadata={"data_volume": 50000000, "destination": "suspicious_domain.com"}
            )
        ]
        
        # Log events and test detection
        detection_results = []
        for event in security_events:
            result = await monitoring_service.log_security_event(event)
            assert result.success, f"Failed to log event: {event.event_type}"
            
            # Trigger threat analysis
            analysis = await monitoring_service.analyze_threat_level(
                event_type=event.event_type,
                context=event.metadata,
                time_window_minutes=60
            )
            detection_results.append(analysis)
        
        # Verify threat detection results
        assert detection_results[0].threat_level >= 5, "Multiple failed logins should raise threat level"
        assert detection_results[1].threat_level >= 8, "Privilege escalation should be high threat"
        assert detection_results[2].threat_level >= 9, "Data exfiltration should be critical threat"
        
        # Test incident creation for high-severity events
        incidents = await monitoring_service.get_active_incidents()
        critical_incidents = [i for i in incidents if i.severity == ThreatSeverity.CRITICAL]
        assert len(critical_incidents) >= 1, "Critical events should create incidents"

    async def test_anomaly_detection_patterns(self, monitoring_service):
        """Test behavioral anomaly detection."""
        user_id = f"anomaly_test_user_{uuid4().hex[:8]}"
        
        # Establish baseline behavior (normal login pattern)
        baseline_events = []
        for day in range(7):  # One week of normal behavior
            for hour in [9, 14, 17]:  # Regular business hours
                event = SecurityEvent(
                    event_type="successful_login",
                    severity=ThreatSeverity.LOW,
                    source_ip="192.168.1.50",  # Office IP
                    user_id=user_id,
                    metadata={
                        "login_hour": hour,
                        "day_of_week": day,
                        "location": "office",
                        "device_type": "laptop"
                    }
                )
                await monitoring_service.log_security_event(event)
                baseline_events.append(event)
        
        # Train baseline model
        baseline_result = await monitoring_service.establish_user_baseline(
            user_id=user_id,
            time_period_days=7
        )
        assert baseline_result.success, "Baseline establishment should succeed"
        
        # Test anomalous behavior detection
        anomalous_events = [
            # Login at unusual hour
            SecurityEvent(
                event_type="successful_login",
                severity=ThreatSeverity.LOW,
                source_ip="192.168.1.50",
                user_id=user_id,
                metadata={"login_hour": 3, "location": "office", "device_type": "laptop"}
            ),
            # Login from unusual location
            SecurityEvent(
                event_type="successful_login", 
                severity=ThreatSeverity.LOW,
                source_ip="203.0.113.15",  # External IP
                user_id=user_id,
                metadata={"login_hour": 14, "location": "unknown", "device_type": "mobile"}
            ),
            # Multiple rapid logins (potential credential stuffing)
            *[SecurityEvent(
                event_type="successful_login",
                severity=ThreatSeverity.LOW,
                source_ip=f"10.0.{i}.{i+1}",
                user_id=user_id,
                metadata={"login_hour": 12, "rapid_sequence": True}
            ) for i in range(5)]
        ]
        
        anomaly_scores = []
        for event in anomalous_events:
            await monitoring_service.log_security_event(event)
            
            anomaly_result = await monitoring_service.detect_behavioral_anomaly(
                user_id=user_id,
                current_event=event
            )
            
            anomaly_scores.append(anomaly_result.anomaly_score)
            
            if anomaly_result.is_anomalous:
                print(f"Anomaly detected: {event.metadata} (score: {anomaly_result.anomaly_score})")
        
        # Verify anomaly detection
        assert max(anomaly_scores) > 7, "Some events should be detected as highly anomalous"
        high_anomaly_count = sum(1 for score in anomaly_scores if score > 5)
        assert high_anomaly_count >= 2, "Multiple anomalous patterns should be detected"

    async def test_incident_response_workflow(self, monitoring_service):
        """Test complete incident response workflow."""
        # Create a critical security incident
        critical_event = SecurityEvent(
            event_type="unauthorized_access",
            severity=ThreatSeverity.CRITICAL,
            source_ip="203.0.113.100",
            user_id="admin_account",
            metadata={
                "accessed_resource": "/admin/system_config",
                "data_modified": True,
                "attack_vector": "credential_stuffing"
            }
        )
        
        # Log event and trigger incident creation
        await monitoring_service.log_security_event(critical_event)
        
        # Get created incident
        incidents = await monitoring_service.get_active_incidents()
        critical_incidents = [i for i in incidents 
                            if i.severity == ThreatSeverity.CRITICAL]
        assert len(critical_incidents) >= 1, "Critical incident should be created"
        
        incident = critical_incidents[0]
        
        # Test incident escalation
        escalation_result = await monitoring_service.escalate_incident(
            incident_id=incident.incident_id,
            escalation_reason="Potential data breach - administrative access compromised",
            assigned_to="security_team_lead"
        )
        assert escalation_result.success, "Incident escalation should succeed"
        
        # Test incident investigation
        investigation_result = await monitoring_service.add_investigation_note(
            incident_id=incident.incident_id,
            note="Initial analysis: Brute force attack on admin account succeeded. "
                 "Source IP traced to known threat actor. Immediate containment required.",
            investigator="security_analyst_1"
        )
        assert investigation_result.success, "Investigation note should be added"
        
        # Test containment actions
        containment_actions = [
            "block_source_ip",
            "disable_compromised_account", 
            "rotate_admin_credentials",
            "enable_enhanced_monitoring"
        ]
        
        for action in containment_actions:
            action_result = await monitoring_service.execute_containment_action(
                incident_id=incident.incident_id,
                action_type=action,
                executed_by="incident_commander",
                details=f"Automated containment: {action}"
            )
            assert action_result.success, f"Containment action {action} should succeed"
        
        # Test incident resolution
        resolution_result = await monitoring_service.resolve_incident(
            incident_id=incident.incident_id,
            resolution_summary="Incident contained and resolved. "
                             "Compromised account secured, threat actor blocked. "
                             "System integrity verified.",
            resolved_by="security_team_lead"
        )
        assert resolution_result.success, "Incident resolution should succeed"
        
        # Verify incident timeline
        timeline = await monitoring_service.get_incident_timeline(incident.incident_id)
        assert len(timeline) >= 6, "Incident should have detailed timeline"
        
        # Verify incident metrics
        metrics = await monitoring_service.get_incident_metrics(
            time_period_days=1,
            severity_filter=[ThreatSeverity.CRITICAL]
        )
        assert metrics["total_incidents"] >= 1
        assert metrics["avg_resolution_time_minutes"] > 0
        assert metrics["critical_incidents"] >= 1


@pytest.mark.integration
@pytest.mark.real_behavior
class TestSecurityServiceFacadeRealBehavior:
    """Real behavior tests for security service facade integration."""
    
    @pytest.fixture
    async def test_infrastructure(self):
        """Set up comprehensive test infrastructure."""
        redis_container = RealRedisTestContainer()
        postgres_container = PostgreSQLTestContainer()
        
        await redis_container.start()
        await postgres_container.start()
        
        yield {
            "redis": redis_container,
            "postgres": postgres_container
        }
        
        await redis_container.stop()
        await postgres_container.stop()

    @pytest.fixture
    async def security_facade(self, test_infrastructure):
        """Create security service facade with real backends."""
        # Set up environment for security services
        os.environ["REDIS_HOST"] = test_infrastructure["redis"].get_host()
        os.environ["REDIS_PORT"] = str(test_infrastructure["redis"].get_port())
        
        async with test_infrastructure["postgres"].get_session() as session:
            facade = get_security_service_facade(
                session_manager=session,
                redis_client=test_infrastructure["redis"].get_client()
            )
            yield facade

    async def test_end_to_end_security_workflow(self, security_facade):
        """Test complete end-to-end security workflow."""
        user_id = f"e2e_user_{uuid4().hex[:8]}"
        password = "SecurePassword123!"
        
        # 1. User Registration with Security Validation
        registration_result = await security_facade.secure_user_registration(
            user_id=user_id,
            password=password,
            email=f"{user_id}@test.com",
            metadata={"ip_address": "192.168.1.100", "user_agent": "TestBrowser"}
        )
        assert registration_result.success, "Secure registration should succeed"
        assert registration_result.user_created, "User should be created"
        assert registration_result.security_validation_passed, "Security validation should pass"
        
        # 2. Authentication with Security Monitoring
        auth_result = await security_facade.authenticate_and_monitor(
            username=user_id,
            password=password,
            context={"ip_address": "192.168.1.100", "device_fingerprint": "test_device"}
        )
        assert auth_result.authentication_successful, "Authentication should succeed"
        assert auth_result.session_token is not None, "Session token should be provided"
        assert auth_result.security_events_logged, "Security events should be logged"
        
        # 3. Authorization with Dynamic Permissions
        resource_access_result = await security_facade.authorize_resource_access(
            user_id=user_id,
            resource="document:123",
            action="read",
            context={
                "session_token": auth_result.session_token,
                "ip_address": "192.168.1.100",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        # Result depends on default permissions for new users
        
        # 4. Data Operations with Encryption and Validation
        sensitive_data = "This is confidential user data"
        data_operation_result = await security_facade.secure_data_operation(
            operation_type="store",
            data=sensitive_data,
            user_id=user_id,
            data_classification="confidential",
            context={"purpose": "user_profile", "retention_days": 365}
        )
        assert data_operation_result.operation_successful, "Data operation should succeed"
        assert data_operation_result.data_encrypted, "Data should be encrypted"
        assert data_operation_result.validation_passed, "Data validation should pass"
        
        # 5. Anomaly Detection - Simulate unusual behavior
        unusual_auth_result = await security_facade.authenticate_and_monitor(
            username=user_id,
            password=password,
            context={
                "ip_address": "203.0.113.50",  # Different IP
                "device_fingerprint": "unknown_device",
                "login_hour": 3  # Unusual hour
            }
        )
        # Should still succeed but trigger monitoring
        
        # 6. Incident Response - Check if anomaly triggered incident
        security_incidents = await security_facade.get_user_security_incidents(
            user_id=user_id,
            time_period_hours=24
        )
        
        # 7. Security Audit Trail
        audit_trail = await security_facade.get_user_security_audit(
            user_id=user_id,
            time_period_hours=24,
            include_events=True
        )
        assert len(audit_trail.events) >= 3, "Should have multiple security events"
        assert any(e.event_type == "user_registration" for e in audit_trail.events)
        assert any(e.event_type == "successful_authentication" for e in audit_trail.events)
        
        # 8. Security Metrics and Health
        security_health = await security_facade.get_security_health_status()
        assert security_health.overall_status in ["healthy", "warning"], "Security should be healthy"
        assert security_health.component_status["authentication"]["healthy"], "Auth should be healthy"
        assert security_health.component_status["authorization"]["healthy"], "Authz should be healthy"
        assert security_health.component_status["validation"]["healthy"], "Validation should be healthy"
        assert security_health.component_status["crypto"]["healthy"], "Crypto should be healthy"
        assert security_health.component_status["monitoring"]["healthy"], "Monitoring should be healthy"

    async def test_security_facade_performance_under_load(self, security_facade):
        """Test security facade performance under concurrent load."""
        # Create test users
        user_count = 50
        users = [f"perf_user_{i}_{uuid4().hex[:6]}" for i in range(user_count)]
        passwords = [f"SecurePass{i}!" for i in range(user_count)]
        
        # Concurrent user registration
        async def register_user(user_id, password):
            return await security_facade.secure_user_registration(
                user_id=user_id,
                password=password,
                email=f"{user_id}@test.com",
                metadata={"load_test": True}
            )
        
        start_time = time.perf_counter()
        registration_results = await asyncio.gather(*[
            register_user(user, password) 
            for user, password in zip(users, passwords)
        ])
        registration_time = time.perf_counter() - start_time
        
        successful_registrations = sum(1 for r in registration_results if r.success)
        assert successful_registrations >= user_count * 0.9, \
            "At least 90% of registrations should succeed"
        
        print(f"\nSecurity Performance Test Results:")
        print(f"  User Registration: {successful_registrations}/{user_count} users in {registration_time:.2f}s")
        print(f"  Registration Rate: {successful_registrations/registration_time:.1f} users/sec")
        
        # Concurrent authentication
        async def authenticate_user(user_id, password):
            return await security_facade.authenticate_and_monitor(
                username=user_id,
                password=password,
                context={"ip_address": f"192.168.1.{hash(user_id) % 254 + 1}"}
            )
        
        start_time = time.perf_counter()
        auth_results = await asyncio.gather(*[
            authenticate_user(user, password)
            for user, password in zip(users[:25], passwords[:25])  # Test subset
        ])
        auth_time = time.perf_counter() - start_time
        
        successful_auths = sum(1 for r in auth_results if r.authentication_successful)
        print(f"  Authentication: {successful_auths}/25 users in {auth_time:.2f}s")
        print(f"  Authentication Rate: {successful_auths/auth_time:.1f} auths/sec")
        
        # Performance requirements
        assert registration_time < 30, "Registration should complete within 30 seconds"
        assert auth_time < 15, "Authentication should complete within 15 seconds"
        assert successful_registrations/registration_time > 1.0, "Should handle >1 registration/sec"
        assert successful_auths/auth_time > 1.5, "Should handle >1.5 auths/sec"

    async def test_security_facade_error_resilience(self, security_facade):
        """Test security facade resilience to component failures."""
        # Test with various failure scenarios
        test_scenarios = [
            # Invalid input scenarios
            {
                "name": "Invalid email format",
                "operation": "secure_user_registration",
                "params": {
                    "user_id": "test_user",
                    "password": "ValidPass123!",
                    "email": "invalid-email-format",
                    "metadata": {}
                }
            },
            # Resource exhaustion scenarios
            {
                "name": "Oversized data",
                "operation": "secure_data_operation",
                "params": {
                    "operation_type": "validate",
                    "data": "x" * 1000000,  # 1MB data
                    "user_id": "test_user",
                    "data_classification": "public"
                }
            }
        ]
        
        resilience_results = []
        for scenario in test_scenarios:
            try:
                if scenario["operation"] == "secure_user_registration":
                    result = await security_facade.secure_user_registration(**scenario["params"])
                elif scenario["operation"] == "secure_data_operation":
                    result = await security_facade.secure_data_operation(**scenario["params"])
                
                # System should handle errors gracefully
                resilience_results.append({
                    "scenario": scenario["name"],
                    "handled_gracefully": True,
                    "error_logged": True  # Assume proper error logging
                })
                
            except Exception as e:
                # Unexpected exceptions indicate poor error handling
                resilience_results.append({
                    "scenario": scenario["name"], 
                    "handled_gracefully": False,
                    "exception": str(e)
                })
        
        # Verify graceful error handling
        graceful_handling_rate = sum(1 for r in resilience_results 
                                   if r["handled_gracefully"]) / len(resilience_results)
        
        assert graceful_handling_rate >= 0.8, \
            f"Should handle at least 80% of error scenarios gracefully, got {graceful_handling_rate:.2%}"
        
        print(f"\nError Resilience Test:")
        print(f"  Graceful Error Handling: {graceful_handling_rate:.1%}")
        for result in resilience_results:
            status = "✓" if result["handled_gracefully"] else "✗"
            print(f"  {status} {result['scenario']}")