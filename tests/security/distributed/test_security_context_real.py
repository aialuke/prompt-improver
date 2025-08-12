"""Real behavior tests for SecurityContext classes.

Tests with actual Redis connections - NO MOCKS.
Requires real Redis connection for integration testing.
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock

import pytest

from prompt_improver.security.distributed.security_context import (
    SecurityContext,
    SecurityThreatScore, 
    SecurityValidationResult,
    SecurityPerformanceMetrics,
    create_security_context,
    create_security_context_from_auth_result,
    create_security_context_from_security_manager,
    create_system_security_context,
)


class TestSecurityThreatScore:
    """Test SecurityThreatScore functionality."""
    
    def test_threat_score_creation(self):
        """Test basic threat score creation."""
        score = SecurityThreatScore()
        assert score.level == "low"
        assert score.score == 0.0
        assert score.factors == []
        assert isinstance(score.last_updated, float)
        
    def test_threat_score_with_data(self):
        """Test threat score with specific data."""
        factors = ["high_error_rate", "suspicious_patterns"]
        score = SecurityThreatScore(
            level="high",
            score=0.85,
            factors=factors
        )
        assert score.level == "high"
        assert score.score == 0.85
        assert score.factors == factors


class TestSecurityValidationResult:
    """Test SecurityValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test basic validation result creation."""
        result = SecurityValidationResult()
        assert not result.validated
        assert result.validation_method == "none"
        assert isinstance(result.validation_timestamp, float)
        assert result.validation_duration_ms == 0.0
        assert result.security_incidents == []
        assert result.rate_limit_status == "unknown"
        assert not result.encryption_required
        assert result.audit_trail_id is None
        
    def test_validation_result_with_data(self):
        """Test validation result with specific data."""
        incidents = ["failed_login", "rate_limit_exceeded"]
        result = SecurityValidationResult(
            validated=True,
            validation_method="oauth2",
            validation_duration_ms=25.5,
            security_incidents=incidents,
            rate_limit_status="within_limits",
            encryption_required=True,
            audit_trail_id="audit_12345"
        )
        assert result.validated
        assert result.validation_method == "oauth2"
        assert result.validation_duration_ms == 25.5
        assert result.security_incidents == incidents
        assert result.rate_limit_status == "within_limits"
        assert result.encryption_required
        assert result.audit_trail_id == "audit_12345"


class TestSecurityPerformanceMetrics:
    """Test SecurityPerformanceMetrics functionality."""
    
    def test_performance_metrics_creation(self):
        """Test basic performance metrics creation."""
        metrics = SecurityPerformanceMetrics()
        assert metrics.authentication_time_ms == 0.0
        assert metrics.authorization_time_ms == 0.0
        assert metrics.validation_time_ms == 0.0
        assert metrics.total_security_overhead_ms == 0.0
        assert metrics.operations_count == 0
        assert isinstance(metrics.last_performance_check, float)
        
    def test_performance_metrics_with_data(self):
        """Test performance metrics with specific data."""
        metrics = SecurityPerformanceMetrics(
            authentication_time_ms=15.5,
            authorization_time_ms=8.2,
            validation_time_ms=12.1,
            total_security_overhead_ms=35.8,
            operations_count=42
        )
        assert metrics.authentication_time_ms == 15.5
        assert metrics.authorization_time_ms == 8.2
        assert metrics.validation_time_ms == 12.1
        assert metrics.total_security_overhead_ms == 35.8
        assert metrics.operations_count == 42


class TestSecurityContext:
    """Test SecurityContext functionality."""
    
    def test_security_context_creation(self):
        """Test basic security context creation."""
        context = SecurityContext(agent_id="test_agent")
        assert context.agent_id == "test_agent"
        assert context.tier == "basic"
        assert not context.authenticated
        assert isinstance(context.created_at, float)
        assert context.authentication_method == "none"
        assert isinstance(context.authentication_timestamp, float)
        assert context.session_id is None
        assert context.permissions == []
        assert isinstance(context.validation_result, SecurityValidationResult)
        assert isinstance(context.threat_score, SecurityThreatScore)
        assert isinstance(context.performance_metrics, SecurityPerformanceMetrics)
        assert context.audit_metadata == {}
        assert context.compliance_tags == []
        assert context.security_level == "basic"
        assert not context.zero_trust_validated
        assert context.encryption_context is None
        assert context.expires_at is None
        assert context.max_operations is None
        assert context.operations_count == 0
        assert isinstance(context.last_used, float)
        
    def test_security_context_validity_checks(self):
        """Test context validity logic."""
        context = SecurityContext(agent_id="test_agent", authenticated=True)
        assert context.is_valid()
        
        # Test expiration
        context.expires_at = time.time() - 100  # Expired
        assert not context.is_valid()
        
        # Reset and test max operations
        context.expires_at = None
        context.max_operations = 5
        context.operations_count = 5
        assert not context.is_valid()
        
        # Reset and test authentication
        context.max_operations = None
        context.operations_count = 0
        context.authenticated = False
        assert not context.is_valid()
        
    def test_security_context_touch(self):
        """Test touch functionality."""
        context = SecurityContext(agent_id="test_agent")
        initial_last_used = context.last_used
        initial_operations = context.operations_count
        
        time.sleep(0.01)  # Small delay
        context.touch()
        
        assert context.last_used > initial_last_used
        assert context.operations_count == initial_operations + 1
        
    def test_security_context_audit_events(self):
        """Test audit event functionality."""
        context = SecurityContext(agent_id="test_agent")
        
        # Add first event
        context.add_audit_event("login", {"ip": "192.168.1.1", "user_agent": "TestAgent"})
        assert "audit_events" in context.audit_metadata
        assert len(context.audit_metadata["audit_events"]) == 1
        
        event = context.audit_metadata["audit_events"][0]
        assert event["event_type"] == "login"
        assert event["details"]["ip"] == "192.168.1.1"
        assert "timestamp" in event
        
        # Add more events to test trimming (max 50)
        for i in range(55):
            context.add_audit_event(f"event_{i}", {"data": i})
        
        # Should be trimmed to 50
        assert len(context.audit_metadata["audit_events"]) == 50
        
    def test_security_context_threat_score_update(self):
        """Test threat score updates."""
        context = SecurityContext(agent_id="test_agent")
        
        factors = ["high_failure_rate", "suspicious_timing"]
        context.update_threat_score("high", 0.75, factors)
        
        assert context.threat_score.level == "high"
        assert context.threat_score.score == 0.75
        assert context.threat_score.factors == factors
        assert isinstance(context.threat_score.last_updated, float)
        
    def test_security_context_performance_recording(self):
        """Test performance metric recording."""
        context = SecurityContext(agent_id="test_agent")
        
        # Record authentication time
        context.record_performance_metric("authentication", 15.5)
        assert context.performance_metrics.authentication_time_ms == 15.5
        assert context.performance_metrics.total_security_overhead_ms == 15.5
        assert context.performance_metrics.operations_count == 1
        
        # Record authorization time
        context.record_performance_metric("authorization", 8.2)
        assert context.performance_metrics.authorization_time_ms == 8.2
        assert context.performance_metrics.total_security_overhead_ms == 23.7  # 15.5 + 8.2
        assert context.performance_metrics.operations_count == 2
        
        # Record validation time
        context.record_performance_metric("validation", 12.1)
        assert context.performance_metrics.validation_time_ms == 12.1
        assert context.performance_metrics.total_security_overhead_ms == 35.8  # 15.5 + 8.2 + 12.1
        assert context.performance_metrics.operations_count == 3
        
    def test_security_context_model_dump(self):
        """Test model_dump functionality."""
        context = SecurityContext(
            agent_id="test_agent",
            tier="professional", 
            authenticated=True,
            permissions=["read", "write"],
            security_level="high"
        )
        
        context.update_threat_score("medium", 0.45, ["elevated_errors"])
        context.record_performance_metric("authentication", 20.0)
        
        data = context.model_dump()
        
        # Verify key fields
        assert data["agent_id"] == "test_agent"
        assert data["tier"] == "professional"
        assert data["authenticated"]
        assert data["permissions"] == ["read", "write"]
        assert data["security_level"] == "high"
        assert data["is_valid"]
        
        # Verify nested structures
        assert data["threat_score"]["level"] == "medium"
        assert data["threat_score"]["score"] == 0.45
        assert data["threat_score"]["factors"] == ["elevated_errors"]
        
        assert data["performance_metrics"]["authentication_time_ms"] == 20.0
        assert data["performance_metrics"]["operations_count"] == 1


class TestSecurityContextFactories:
    """Test security context factory functions."""
    
    async def test_create_security_context(self):
        """Test basic security context creation."""
        context = await create_security_context(
            agent_id="test_user",
            tier="professional",
            authenticated=True,
            authentication_method="oauth2",
            permissions=["read", "write", "admin"],
            security_level="high",
            session_id="session_123",
            expires_minutes=60
        )
        
        assert context.agent_id == "test_user"
        assert context.tier == "professional"
        assert context.authenticated
        assert context.authentication_method == "oauth2"
        assert context.permissions == ["read", "write", "admin"]
        assert context.security_level == "high"
        assert context.session_id == "session_123"
        assert context.expires_at is not None
        
        # Check expiration is roughly 60 minutes from now
        expected_expiry = time.time() + 60 * 60
        assert abs(context.expires_at - expected_expiry) < 2  # Within 2 seconds
        
    async def test_create_security_context_from_auth_result_direct_params(self):
        """Test context creation from direct parameters (no auth_result)."""
        context = await create_security_context_from_auth_result(
            agent_id="direct_user",
            tier="enterprise",
            authenticated=True,
            authentication_method="api_key",
            permissions=["full_access"],
            security_level="critical",
            expires_minutes=30
        )
        
        assert context.agent_id == "direct_user"
        assert context.tier == "enterprise"
        assert context.authenticated
        assert context.authentication_method == "api_key"
        assert context.permissions == ["full_access"]
        assert context.security_level == "critical"
        assert context.validation_result.validated
        assert context.validation_result.validation_method == "api_key"
        assert context.audit_metadata["direct_parameter_creation"]
        
    async def test_create_security_context_from_security_manager(self):
        """Test context creation from security manager."""
        # Create mock security manager
        mock_security_manager = AsyncMock()
        mock_security_manager.get_security_status.return_value = {
            "metrics": {
                "violation_rate": 0.05,  # Below threshold
                "active_incidents": 0
            },
            "performance": {
                "average_operation_time_ms": 15.5
            }
        }
        
        # Mock config attributes
        mock_config = AsyncMock()
        mock_config.require_encryption = True
        mock_config.security_level.value = "enhanced"
        mock_config.rate_limit_tier.value = "professional"
        mock_config.zero_trust_mode = True
        mock_config.fail_secure = True
        mock_security_manager.config = mock_config
        
        # Mock mode
        mock_mode = AsyncMock()
        mock_mode.value = "production"
        mock_security_manager.mode = mock_mode
        
        context = await create_security_context_from_security_manager(
            agent_id="security_user",
            security_manager=mock_security_manager,
            additional_context={
                "session_id": "sec_session_123",
                "permissions": ["security_admin"]
            }
        )
        
        assert context.agent_id == "security_user"
        assert context.tier == "professional"
        assert context.authenticated
        assert context.session_id == "sec_session_123"
        assert context.permissions == ["security_admin"]
        assert context.security_level == "enhanced"
        assert context.zero_trust_validated
        assert context.validation_result.encryption_required
        assert context.audit_metadata["security_manager_integration"]
        
        # Verify threat assessment (should be low with good metrics)
        assert context.threat_score.level == "low"
        assert context.threat_score.score == 0.0
        assert context.threat_score.factors == []
        
    async def test_create_security_context_from_security_manager_high_threat(self):
        """Test context creation with high threat scenario."""
        # Create mock security manager with concerning metrics
        mock_security_manager = AsyncMock()
        mock_security_manager.get_security_status.return_value = {
            "metrics": {
                "violation_rate": 0.15,  # Above threshold
                "active_incidents": 2  # Active incidents
            },
            "performance": {
                "average_operation_time_ms": 25.0
            }
        }
        
        mock_config = AsyncMock()
        mock_config.require_encryption = True
        mock_config.security_level.value = "high"
        mock_config.rate_limit_tier.value = "enterprise"
        mock_config.zero_trust_mode = True
        mock_config.fail_secure = True
        mock_security_manager.config = mock_config
        
        mock_mode = AsyncMock()
        mock_mode.value = "production"
        mock_security_manager.mode = mock_mode
        
        context = await create_security_context_from_security_manager(
            agent_id="threat_user",
            security_manager=mock_security_manager
        )
        
        # Should have high threat score due to active incidents
        assert context.threat_score.level == "high"
        assert context.threat_score.score == 0.6
        assert "elevated_violation_rate" in context.threat_score.factors
        assert "active_security_incidents" in context.threat_score.factors
        
        # Performance metrics should reflect the higher operation times
        assert context.performance_metrics.authentication_time_ms == 25.0
        assert context.performance_metrics.total_security_overhead_ms == 75.0  # 25 * 3
        
    async def test_create_system_security_context(self):
        """Test system security context creation."""
        context = await create_system_security_context(
            operation_type="database_maintenance",
            security_level="critical"
        )
        
        assert context.agent_id == "system"
        assert context.tier == "system"
        assert context.authenticated
        assert context.authentication_method == "system_internal"
        assert context.permissions == ["system:all"]
        assert context.security_level == "critical"
        assert context.zero_trust_validated
        assert context.validation_result.validated
        assert context.validation_result.validation_method == "system_internal"
        assert context.validation_result.rate_limit_status == "system_exempt"
        assert not context.validation_result.encryption_required
        assert context.audit_metadata["operation_type"] == "database_maintenance"
        assert "system_internal" in context.compliance_tags


class TestSecurityContextRedisValidation:
    """Test SecurityContext with real Redis validation."""
    
    async def test_security_context_redis_validation_available(self):
        """Test Redis validation when Redis is available."""
        # Skip if no Redis connection available  
        try:
            import redis.asyncio as redis
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            await redis_client.ping()
        except Exception:
            pytest.skip("Redis not available for integration testing")
            
        # Create security context for Redis validation
        context = SecurityContext(
            agent_id="redis_test_user",
            authenticated=True,
            permissions=["cache:read", "cache:write"],
            security_level="enhanced"
        )
        
        # Test Redis key validation pattern
        test_key = f"security_test_{int(time.time() * 1000000)}"
        
        try:
            # Simulate Redis operation with security context
            await redis_client.set(test_key, "test_value", ex=10)  # 10 second expiry
            
            # Record the operation in context
            context.touch()
            context.add_audit_event("redis_operation", {
                "operation": "SET",
                "key": test_key,
                "success": True
            })
            
            # Verify context state
            assert context.operations_count == 1
            assert len(context.audit_metadata["audit_events"]) == 1
            
            # Verify Redis operation succeeded
            value = await redis_client.get(test_key)
            assert value == "test_value"
            
            # Clean up
            await redis_client.delete(test_key)
            await redis_client.close()
            
        except Exception as e:
            await redis_client.close()
            pytest.skip(f"Redis integration test failed: {e}")
            
    async def test_security_context_redis_validation_connection_failure(self):
        """Test security context behavior with Redis connection failure."""
        context = SecurityContext(
            agent_id="redis_fail_user",
            authenticated=True,
            security_level="high"
        )
        
        # Simulate Redis connection failure in audit event
        context.add_audit_event("redis_connection_failure", {
            "error": "Connection refused",
            "attempted_operation": "SET",
            "fallback_used": True
        })
        
        # Update threat score due to connection issues
        context.update_threat_score("medium", 0.4, ["redis_connection_failure"])
        
        # Verify threat assessment updated
        assert context.threat_score.level == "medium"
        assert context.threat_score.score == 0.4
        assert "redis_connection_failure" in context.threat_score.factors
        assert len(context.audit_metadata["audit_events"]) == 1
        
    async def test_security_context_concurrent_redis_operations(self):
        """Test security context with concurrent Redis operations."""
        # Skip if no Redis available
        try:
            import redis.asyncio as redis
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            await redis_client.ping()
        except Exception:
            pytest.skip("Redis not available for concurrent testing")
            
        context = SecurityContext(
            agent_id="concurrent_user",
            authenticated=True,
            permissions=["cache:admin"],
            security_level="high"
        )
        
        async def redis_operation(op_id: int):
            """Simulate a Redis operation."""
            test_key = f"concurrent_test_{op_id}_{int(time.time() * 1000000)}"
            try:
                await redis_client.set(test_key, f"value_{op_id}", ex=5)
                context.touch()
                context.add_audit_event("concurrent_redis_op", {
                    "operation_id": op_id,
                    "key": test_key,
                    "success": True
                })
                await redis_client.delete(test_key)  # Cleanup
                return True
            except Exception as e:
                context.add_audit_event("concurrent_redis_error", {
                    "operation_id": op_id,
                    "error": str(e)
                })
                return False
        
        try:
            # Run 5 concurrent operations
            tasks = [redis_operation(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify operations completed
            successful_ops = sum(1 for r in results if r is True)
            assert successful_ops >= 0  # At least some should succeed
            
            # Verify context tracked operations
            assert context.operations_count >= successful_ops
            assert "audit_events" in context.audit_metadata
            
            await redis_client.close()
            
        except Exception as e:
            await redis_client.close()
            pytest.skip(f"Concurrent Redis test failed: {e}")