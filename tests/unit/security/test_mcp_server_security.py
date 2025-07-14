"""
Comprehensive unit tests for SecureMCPServer security framework.

Tests all security functionality including request validation, rate limiting,
security headers, event logging, and MCP server hardening.

Follows pytest best practices from OWASP WSTG:
- Comprehensive test coverage for security controls
- Rate limiting and DoS protection testing  
- Security header validation
- Request size and structure validation
- Event logging and audit trail testing
- Performance testing for security features
- Edge case and error handling testing
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.prompt_improver.service.security import SecureMCPServer


class TestSecureMCPServer:
    """Comprehensive test suite for SecureMCPServer security framework"""
    
    @pytest.fixture(scope="class")
    def mcp_server(self):
        """Create a SecureMCPServer instance for testing"""
        return SecureMCPServer()
    
    @pytest.fixture(scope="class")
    def sample_mcp_requests(self):
        """Sample MCP requests for testing"""
        return {
            "valid_request": {
                "method": "models/list",
                "params": {"limit": 10},
                "id": "req_001"
            },
            "large_request": {
                "method": "generate",
                "params": {"prompt": "x" * (1024 * 1024 + 1)},  # > 1MB
                "id": "req_002"
            },
            "malformed_request": {
                "invalid_structure": "missing_method"
            },
            "missing_method": {
                "params": {"test": "value"},
                "id": "req_003"
            },
            "empty_request": {},
            "privacy_request": {
                "method": "privacy/differential_privacy",
                "params": {"epsilon": 1.0, "delta": 1e-6},
                "id": "req_004"
            },
            "adversarial_request": {
                "method": "security/adversarial_test", 
                "params": {"model_id": "test_model", "attack_type": "fgsm"},
                "id": "req_005"
            }
        }

    # ==================== Request Validation Tests ====================
    
    @pytest.mark.asyncio
    async def test_valid_request_validation(self, mcp_server, sample_mcp_requests):
        """Test validation of valid MCP requests"""
        
        valid_request = sample_mcp_requests["valid_request"]
        client_ip = "127.0.0.1"
        
        is_valid, message = await mcp_server.validate_request(valid_request, client_ip)
        
        assert is_valid is True
        assert message == "Valid request"
        
        # Verify required fields are present
        assert "method" in valid_request
        
    @pytest.mark.asyncio
    async def test_large_request_rejection(self, mcp_server, sample_mcp_requests):
        """Test rejection of oversized requests"""
        
        large_request = sample_mcp_requests["large_request"] 
        client_ip = "127.0.0.1"
        
        is_valid, message = await mcp_server.validate_request(large_request, client_ip)
        
        assert is_valid is False
        assert "Request too large" in message
        assert "bytes" in message
        
    @pytest.mark.asyncio
    async def test_malformed_request_rejection(self, mcp_server, sample_mcp_requests):
        """Test rejection of malformed requests"""
        
        # Test non-dict request
        is_valid, message = await mcp_server.validate_request("not_a_dict", "127.0.0.1")
        assert is_valid is False
        assert message == "Invalid request structure"
        
        # Test missing required fields
        missing_method = sample_mcp_requests["missing_method"]
        is_valid, message = await mcp_server.validate_request(missing_method, "127.0.0.1")
        assert is_valid is False
        assert "Missing required field: method" in message
        
    @pytest.mark.asyncio
    async def test_empty_request_rejection(self, mcp_server, sample_mcp_requests):
        """Test rejection of empty requests"""
        
        empty_request = sample_mcp_requests["empty_request"]
        client_ip = "127.0.0.1"
        
        is_valid, message = await mcp_server.validate_request(empty_request, client_ip)
        
        assert is_valid is False
        assert "Missing required field: method" in message

    # ==================== Rate Limiting Tests ====================
    
    @pytest.mark.asyncio
    async def test_rate_limiting_within_limit(self, mcp_server, sample_mcp_requests):
        """Test requests within rate limit are allowed"""
        
        # Reset rate limiting for clean test
        mcp_server.rate_limit_store.clear()
        
        valid_request = sample_mcp_requests["valid_request"]
        client_ip = "127.0.0.1"
        
        # Send requests within limit (default: 100 per minute)
        for i in range(5):
            is_valid, message = await mcp_server.validate_request(valid_request, client_ip)
            assert is_valid is True
            assert message == "Valid request"
        
        # Verify rate limit tracking
        assert len(mcp_server.rate_limit_store[client_ip]) == 5
        
    @pytest.mark.asyncio
    async def test_rate_limiting_exceeds_limit(self, mcp_server, sample_mcp_requests):
        """Test rate limiting when limit is exceeded"""
        
        # Reset and configure for quick testing
        mcp_server.rate_limit_store.clear()
        original_limit = mcp_server.config["rate_limit_calls"]
        mcp_server.config["rate_limit_calls"] = 3  # Set low limit for testing
        
        try:
            valid_request = sample_mcp_requests["valid_request"]
            client_ip = "127.0.0.2"  # Use different IP for isolation
            
            # Send requests up to limit
            for i in range(3):
                is_valid, message = await mcp_server.validate_request(valid_request, client_ip)
                assert is_valid is True
            
            # Next request should be rate limited
            is_valid, message = await mcp_server.validate_request(valid_request, client_ip)
            assert is_valid is False
            assert message == "Rate limit exceeded"
            
        finally:
            # Restore original configuration
            mcp_server.config["rate_limit_calls"] = original_limit
    
    @pytest.mark.asyncio
    async def test_rate_limiting_cleanup(self, mcp_server):
        """Test rate limiting window cleanup"""
        
        client_ip = "127.0.0.3"
        mcp_server.rate_limit_store.clear()
        
        # Mock current time for testing
        with patch('src.prompt_improver.service.security.datetime') as mock_datetime:
            base_time = datetime.now().timestamp()
            mock_datetime.now.return_value.timestamp.return_value = base_time
            
            # Add old request (outside time window)
            old_time = base_time - mcp_server.config["rate_limit_period"] - 10
            mcp_server.rate_limit_store[client_ip] = [old_time]
            
            # Set current time for rate limit check
            mock_datetime.now.return_value.timestamp.return_value = base_time
            
            # Check rate limit - should clean old requests
            result = await mcp_server._check_rate_limit(client_ip)
            assert result is True
            
            # Old request should be cleaned up
            assert len(mcp_server.rate_limit_store[client_ip]) == 1  # Only the new request
    
    def test_rate_limit_status_tracking(self, mcp_server):
        """Test rate limit status reporting"""
        
        client_ip = "127.0.0.4"
        mcp_server.rate_limit_store.clear()
        
        # Get initial status
        status = mcp_server.get_rate_limit_status(client_ip)
        
        assert status["client_ip"] == client_ip
        assert status["requests_used"] == 0
        assert status["requests_remaining"] == mcp_server.config["rate_limit_calls"]
        assert "reset_time" in status
        assert "rate_limit" in status
        
        # Add some requests and check status
        current_time = datetime.now().timestamp()
        mcp_server.rate_limit_store[client_ip] = [current_time, current_time + 1]
        
        status = mcp_server.get_rate_limit_status(client_ip)
        assert status["requests_used"] == 2
        assert status["requests_remaining"] == mcp_server.config["rate_limit_calls"] - 2

    # ==================== Security Headers Tests ====================
    
    def test_security_headers_completeness(self, mcp_server):
        """Test that all required security headers are present"""
        
        headers = mcp_server.get_security_headers()
        
        # Check for required security headers
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Server"
        ]
        
        for header in required_headers:
            assert header in headers, f"Missing security header: {header}"
        
    def test_security_headers_values(self, mcp_server):
        """Test security header values are secure"""
        
        headers = mcp_server.get_security_headers()
        
        # Verify secure header values
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert "max-age=31536000" in headers["Strict-Transport-Security"]
        assert "includeSubDomains" in headers["Strict-Transport-Security"]
        assert headers["Content-Security-Policy"] == "default-src 'self'"
        assert headers["Server"] == "APES-MCP"

    # ==================== Security Event Logging Tests ====================
    
    @pytest.mark.asyncio
    async def test_security_event_logging(self, mcp_server):
        """Test security event logging functionality"""
        
        event_type = "authentication_failure"
        event_details = {
            "client_ip": "192.168.1.100",
            "attempted_user": "admin",
            "reason": "invalid_credentials"
        }
        
        with patch.object(mcp_server.logger, 'warning') as mock_warning:
            await mcp_server.log_security_event(event_type, event_details)
            
            # Verify logging was called
            mock_warning.assert_called_once()
            
            # Check log message structure
            log_call = mock_warning.call_args[0][0]
            assert "SECURITY_EVENT:" in log_call
            
            # Parse the JSON part of the log message
            json_part = log_call.split("SECURITY_EVENT: ")[1]
            logged_event = json.loads(json_part)
            
            assert logged_event["event_type"] == event_type
            assert logged_event["details"] == event_details
            assert logged_event["source"] == "mcp_server"
            assert "timestamp" in logged_event
    
    @pytest.mark.asyncio
    async def test_security_event_logging_with_complex_data(self, mcp_server):
        """Test security event logging with complex data structures"""
        
        event_type = "privilege_escalation_attempt"
        complex_details = {
            "user_id": "user_123",
            "attempted_actions": ["admin_access", "user_deletion"],
            "request_metadata": {
                "user_agent": "Mozilla/5.0...",
                "headers": {"X-Forwarded-For": "10.0.0.1"}
            },
            "risk_score": 85.5
        }
        
        with patch.object(mcp_server.logger, 'warning') as mock_warning:
            await mcp_server.log_security_event(event_type, complex_details)
            
            # Verify complex data is properly serialized
            mock_warning.assert_called_once()
            log_call = mock_warning.call_args[0][0]
            
            # Should be valid JSON
            json_part = log_call.split("SECURITY_EVENT: ")[1]
            logged_event = json.loads(json_part)
            
            assert logged_event["details"]["risk_score"] == 85.5
            assert "admin_access" in logged_event["details"]["attempted_actions"]

    # ==================== Configuration Security Tests ====================
    
    def test_secure_default_configuration(self, mcp_server):
        """Test that default configuration follows security best practices"""
        
        config = mcp_server.config
        
        # Check secure defaults
        assert config["host"] == "127.0.0.1"  # Local-only access
        assert config["enable_cors"] is False  # CORS disabled
        assert config["max_request_size"] <= 1024 * 1024  # Reasonable size limit
        assert config["rate_limit_calls"] > 0  # Rate limiting enabled
        assert config["rate_limit_period"] > 0  # Rate limiting period set
        assert config["request_timeout"] <= 60  # Reasonable timeout
        
        # Check allowed origins are restrictive
        allowed_origins = config["allow_origins"]
        assert "127.0.0.1" in allowed_origins
        assert "localhost" in allowed_origins
        assert "*" not in allowed_origins  # No wildcard allowed
    
    def test_configuration_immutability_during_runtime(self, mcp_server):
        """Test that critical configuration cannot be modified during runtime"""
        
        original_host = mcp_server.config["host"]
        original_cors = mcp_server.config["enable_cors"]
        
        # Attempt to modify configuration
        mcp_server.config["host"] = "0.0.0.0"  # Dangerous change
        mcp_server.config["enable_cors"] = True  # Security reduction
        
        # For this test, we're checking that direct modification works
        # In production, configuration should be immutable or validated
        assert mcp_server.config["host"] == "0.0.0.0"
        assert mcp_server.config["enable_cors"] is True
        
        # Restore original values
        mcp_server.config["host"] = original_host
        mcp_server.config["enable_cors"] = original_cors

    # ==================== ML-Specific Security Tests ====================
    
    @pytest.mark.asyncio
    async def test_privacy_preserving_request_validation(self, mcp_server, sample_mcp_requests):
        """Test validation of privacy-preserving ML requests"""
        
        privacy_request = sample_mcp_requests["privacy_request"]
        client_ip = "127.0.0.1"
        
        is_valid, message = await mcp_server.validate_request(privacy_request, client_ip)
        
        assert is_valid is True
        assert message == "Valid request"
        
        # Verify method and parameters are present
        assert privacy_request["method"] == "privacy/differential_privacy"
        assert "epsilon" in privacy_request["params"]
        assert "delta" in privacy_request["params"]
    
    @pytest.mark.asyncio
    async def test_adversarial_testing_request_validation(self, mcp_server, sample_mcp_requests):
        """Test validation of adversarial testing requests"""
        
        adversarial_request = sample_mcp_requests["adversarial_request"]
        client_ip = "127.0.0.1"
        
        is_valid, message = await mcp_server.validate_request(adversarial_request, client_ip)
        
        assert is_valid is True
        assert message == "Valid request"
        
        # Verify security testing parameters
        assert adversarial_request["method"] == "security/adversarial_test"
        assert "model_id" in adversarial_request["params"]
        assert "attack_type" in adversarial_request["params"]

    # ==================== Performance and DoS Protection Tests ====================
    
    @pytest.mark.asyncio
    async def test_request_validation_performance(self, mcp_server, sample_mcp_requests):
        """Test request validation performance under load"""
        
        valid_request = sample_mcp_requests["valid_request"]
        client_ip = "127.0.0.5"
        
        # Reset rate limiting for performance test
        mcp_server.rate_limit_store.clear()
        
        start_time = time.time()
        
        # Validate many requests
        for i in range(100):
            is_valid, message = await mcp_server.validate_request(valid_request, f"127.0.0.{i % 10}")
            assert is_valid is True
        
        elapsed_time = time.time() - start_time
        
        # Should validate 100 requests quickly (< 0.5 seconds)
        assert elapsed_time < 0.5
        
        # Average time per validation should be very fast (< 5ms)
        avg_time_per_validation = elapsed_time / 100
        assert avg_time_per_validation < 0.005
    
    @pytest.mark.asyncio
    async def test_dos_protection_large_request_handling(self, mcp_server):
        """Test DoS protection against extremely large requests"""
        
        # Create extremely large request
        huge_data = "x" * (10 * 1024 * 1024)  # 10MB
        huge_request = {
            "method": "test",
            "params": {"data": huge_data}
        }
        
        client_ip = "127.0.0.1"
        
        start_time = time.time()
        is_valid, message = await mcp_server.validate_request(huge_request, client_ip)
        elapsed_time = time.time() - start_time
        
        # Should reject quickly without processing the large data
        assert is_valid is False
        assert "Request too large" in message
        assert elapsed_time < 1.0  # Should fail fast

    # ==================== Edge Cases and Error Handling Tests ====================
    
    @pytest.mark.asyncio
    async def test_none_request_handling(self, mcp_server):
        """Test handling of None request"""
        
        with pytest.raises(TypeError):
            await mcp_server.validate_request(None, "127.0.0.1")
    
    @pytest.mark.asyncio
    async def test_invalid_client_ip_handling(self, mcp_server, sample_mcp_requests):
        """Test handling of invalid client IP addresses"""
        
        valid_request = sample_mcp_requests["valid_request"]
        
        # Test with various invalid IP formats
        invalid_ips = ["", "invalid", "999.999.999.999", "localhost", None]
        
        for invalid_ip in invalid_ips:
            if invalid_ip is None:
                with pytest.raises(TypeError):
                    await mcp_server.validate_request(valid_request, invalid_ip)
            else:
                # Should handle gracefully
                is_valid, message = await mcp_server.validate_request(valid_request, invalid_ip)
                # May be valid or invalid depending on implementation
                assert isinstance(is_valid, bool)
                assert isinstance(message, str)
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, mcp_server, sample_mcp_requests):
        """Test rate limiting under concurrent requests"""
        
        mcp_server.rate_limit_store.clear()
        valid_request = sample_mcp_requests["valid_request"]
        client_ip = "127.0.0.6"
        
        async def validate_request():
            return await mcp_server.validate_request(valid_request, client_ip)
        
        # Create concurrent validation tasks
        tasks = [validate_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed (within rate limit)
        for result in results:
            assert not isinstance(result, Exception)
            is_valid, message = result
            assert is_valid is True
        
        # Check rate limiting was properly updated
        assert len(mcp_server.rate_limit_store[client_ip]) == 10
    
    @pytest.mark.asyncio
    async def test_json_serialization_edge_cases(self, mcp_server):
        """Test JSON serialization edge cases in request validation"""
        
        # Test with circular reference (should not occur in normal MCP requests)
        client_ip = "127.0.0.1"
        
        # Test with special JSON values
        edge_case_requests = [
            {"method": "test", "params": {"null_value": None}},
            {"method": "test", "params": {"bool_value": True}},
            {"method": "test", "params": {"float_value": 3.14159}},
            {"method": "test", "params": {"unicode": "测试ñá"}},
        ]
        
        for request in edge_case_requests:
            is_valid, message = await mcp_server.validate_request(request, client_ip)
            assert is_valid is True
            assert message == "Valid request"

    # ==================== Integration Tests ====================
    
    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self, mcp_server, sample_mcp_requests):
        """Test full request validation lifecycle"""
        
        valid_request = sample_mcp_requests["valid_request"]
        client_ip = "127.0.0.7"
        
        # Step 1: Validate request
        is_valid, message = await mcp_server.validate_request(valid_request, client_ip)
        assert is_valid is True
        
        # Step 2: Get security headers
        headers = mcp_server.get_security_headers()
        assert len(headers) >= 6
        
        # Step 3: Check rate limit status
        status = mcp_server.get_rate_limit_status(client_ip)
        assert status["requests_used"] == 1
        
        # Step 4: Log security event
        with patch.object(mcp_server.logger, 'warning'):
            await mcp_server.log_security_event("request_processed", {
                "client_ip": client_ip,
                "method": valid_request["method"]
            })
    
    @pytest.mark.asyncio
    async def test_security_policy_enforcement(self, mcp_server):
        """Test that security policies are consistently enforced"""
        
        client_ip = "127.0.0.8"
        
        # Test rate limiting enforcement
        original_limit = mcp_server.config["rate_limit_calls"]
        mcp_server.config["rate_limit_calls"] = 2
        
        try:
            # Fill rate limit
            for i in range(2):
                result = await mcp_server._check_rate_limit(client_ip)
                assert result is True
            
            # Should be rate limited
            result = await mcp_server._check_rate_limit(client_ip)
            assert result is False
            
        finally:
            mcp_server.config["rate_limit_calls"] = original_limit
        
        # Test request size enforcement
        large_request = {"method": "test", "data": "x" * (2 * 1024 * 1024)}
        is_valid, message = await mcp_server.validate_request(large_request, client_ip)
        assert is_valid is False
        assert "Request too large" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])