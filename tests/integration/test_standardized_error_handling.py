"""Integration tests for standardized error handling implementation.

Tests the unified exception hierarchy, error propagation patterns,
and backwards compatibility across all application layers.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from prompt_improver.api.app import create_app
from prompt_improver.common.error_handling import (
    get_correlation_id,
    handle_repository_errors,
    handle_service_errors,
    set_correlation_id,
    wrap_external_exception,
)
from prompt_improver.common.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BusinessRuleViolationError,
    CacheError,
    ConfigurationError,
    ConnectionError,
    DatabaseError,
    DomainError,
    ExternalServiceError,
    InfrastructureError,
    InternalError,
    MLError,
    PromptImproverError,
    RateLimitError,
    ResourceExhaustedError,
    SecurityError,
    ServiceUnavailableError,
    SystemError,
    TimeoutError,
    ValidationError,
    create_error_response,
)
from prompt_improver.database.services.generation_service import GenerationDatabaseService


class TestUnifiedExceptionHierarchy:
    """Test the unified exception hierarchy structure and functionality."""
    
    def test_base_exception_structure(self):
        """Test PromptImproverError base functionality."""
        exc = PromptImproverError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"},
            context={"user_id": "123"}
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.details == {"key": "value"}
        assert exc.context == {"user_id": "123"}
        assert exc.severity == "ERROR"
        assert isinstance(exc.correlation_id, str)
        assert isinstance(exc.timestamp, datetime)
        
    def test_exception_hierarchy_inheritance(self):
        """Test that exception hierarchy follows the designed structure."""
        # Domain Errors
        assert issubclass(ValidationError, DomainError)
        assert issubclass(BusinessRuleViolationError, DomainError)
        assert issubclass(AuthorizationError, DomainError)
        
        # Infrastructure Errors  
        assert issubclass(DatabaseError, InfrastructureError)
        assert issubclass(CacheError, InfrastructureError)
        assert issubclass(ExternalServiceError, InfrastructureError)
        assert issubclass(ConnectionError, InfrastructureError)
        assert issubclass(MLError, InfrastructureError)
        assert issubclass(ServiceUnavailableError, InfrastructureError)
        
        # System Errors
        assert issubclass(ConfigurationError, SystemError)
        assert issubclass(ResourceExhaustedError, SystemError)
        assert issubclass(InternalError, SystemError)
        assert issubclass(TimeoutError, SystemError)
        assert issubclass(RateLimitError, SystemError)
        
        # All inherit from base
        assert issubclass(DomainError, PromptImproverError)
        assert issubclass(InfrastructureError, PromptImproverError)
        assert issubclass(SystemError, PromptImproverError)
        
    def test_exception_to_dict_conversion(self):
        """Test exception to dictionary conversion for logging and API responses."""
        exc = ValidationError(
            message="Invalid input",
            field="email",
            value="invalid-email",
            validation_rule="email_format",
            correlation_id="test-correlation-id"
        )
        
        result = exc.to_dict()
        
        assert result["error_code"] == "VALIDATION_ERROR"
        assert result["message"] == "Invalid input"
        assert result["correlation_id"] == "test-correlation-id"
        assert result["severity"] == "ERROR"
        assert "timestamp" in result
        assert result["details"]["field"] == "email"
        assert result["details"]["value"] == "invalid-email"
        
    def test_error_response_creation(self):
        """Test creating API error responses from exceptions."""
        exc = AuthenticationError(
            message="Invalid credentials",
            user_id="user123",
            auth_method="bearer_token"
        )
        
        # Test without debug info
        response = create_error_response(exc, include_debug=False)
        assert "error" in response
        assert response["error"]["code"] == "AUTH_ERROR"
        assert response["error"]["message"] == "Invalid credentials"
        assert "details" not in response["error"]
        assert "context" not in response["error"]
        
        # Test with debug info
        debug_response = create_error_response(exc, include_debug=True)
        assert "details" in debug_response["error"]
        assert debug_response["error"]["details"]["user_id"] == "user123"


class TestErrorWrapping:
    """Test external exception wrapping functionality."""
    
    def test_wrap_connection_error(self):
        """Test wrapping connection-related exceptions."""
        original_exc = OSError("Connection refused")
        wrapped = wrap_external_exception(
            original_exc,
            message="Database connection failed",
            correlation_id="test-id"
        )
        
        assert isinstance(wrapped, ConnectionError)
        assert wrapped.message == "Database connection failed"
        assert wrapped.correlation_id == "test-id"
        assert wrapped.cause == original_exc
        assert "caused_by" in wrapped.context
        
    def test_wrap_timeout_error(self):
        """Test wrapping timeout exceptions."""
        original_exc = asyncio.TimeoutError("Operation timed out")
        wrapped = wrap_external_exception(original_exc)
        
        assert isinstance(wrapped, TimeoutError)
        assert wrapped.cause == original_exc
        
    def test_wrap_value_error(self):
        """Test wrapping validation-related exceptions."""
        original_exc = ValueError("Invalid format")
        wrapped = wrap_external_exception(original_exc)
        
        assert isinstance(wrapped, ValidationError)
        assert wrapped.cause == original_exc
        
    def test_wrap_generic_exception(self):
        """Test wrapping unrecognized exceptions."""
        original_exc = RuntimeError("Something went wrong")
        wrapped = wrap_external_exception(original_exc)
        
        assert isinstance(wrapped, InternalError)
        assert wrapped.cause == original_exc


class TestRepositoryErrorHandling:
    """Test repository layer error handling patterns."""
    
    @pytest.mark.asyncio
    async def test_repository_decorator_validation_error(self):
        """Test repository decorator handling validation errors."""
        @handle_repository_errors()
        async def test_repo_method():
            raise ValueError("Invalid input")
        
        with pytest.raises(ValidationError) as exc_info:
            await test_repo_method()
        
        assert "Invalid input" in str(exc_info.value)
        assert exc_info.value.context["layer"] == "repository"
        
    @pytest.mark.asyncio  
    async def test_repository_decorator_database_error(self):
        """Test repository decorator handling database errors."""
        @handle_repository_errors()
        async def test_repo_method():
            raise SQLAlchemyError("Database connection failed")
        
        with pytest.raises(DatabaseError) as exc_info:
            await test_repo_method()
        
        assert "Database connection failed" in str(exc_info.value)
        assert exc_info.value.context["layer"] == "repository"
        
    @pytest.mark.asyncio
    async def test_repository_decorator_preserves_our_exceptions(self):
        """Test repository decorator preserves PromptImproverError exceptions."""
        @handle_repository_errors()
        async def test_repo_method():
            raise ValidationError("Already wrapped")
        
        with pytest.raises(ValidationError) as exc_info:
            await test_repo_method()
        
        assert exc_info.value.message == "Already wrapped"


class TestServiceErrorHandling:
    """Test service layer error handling patterns."""
    
    @pytest.mark.asyncio
    async def test_service_decorator_adds_context(self):
        """Test service decorator adds service context to existing errors."""
        @handle_service_errors()
        async def test_service_method():
            raise ValidationError("Service validation failed")
        
        with pytest.raises(ValidationError) as exc_info:
            await test_service_method()
        
        assert "service_layer" in exc_info.value.context
        assert exc_info.value.context["service_layer"]["function"] == "test_service_method"
        
    @pytest.mark.asyncio
    async def test_service_decorator_wraps_external_exceptions(self):
        """Test service decorator wraps external exceptions."""
        @handle_service_errors()
        async def test_service_method():
            raise RuntimeError("Unexpected error")
        
        with pytest.raises(InternalError) as exc_info:
            await test_service_method()
        
        assert exc_info.value.context["layer"] == "service"


class TestGenerationServiceErrorHandling:
    """Test updated GenerationDatabaseService error handling."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.rollback = AsyncMock()
        return session
        
    @pytest.mark.asyncio
    async def test_create_generation_session_validation_errors(self, mock_db_session):
        """Test validation errors in create_generation_session."""
        service = GenerationDatabaseService(mock_db_session)
        
        # Test negative target_samples
        with pytest.raises(ValidationError) as exc_info:
            await service.create_generation_session(
                generation_method="test",
                target_samples=-1
            )
        assert exc_info.value.field == "target_samples"
        
        # Test empty generation_method
        with pytest.raises(ValidationError) as exc_info:
            await service.create_generation_session(
                generation_method="",
                target_samples=100
            )
        assert exc_info.value.field == "generation_method"
        
        # Test invalid quality_threshold
        with pytest.raises(ValidationError) as exc_info:
            await service.create_generation_session(
                generation_method="test",
                target_samples=100,
                quality_threshold=1.5
            )
        assert exc_info.value.field == "quality_threshold"
        
    @pytest.mark.asyncio
    async def test_update_session_status_validation_errors(self, mock_db_session):
        """Test validation errors in update_session_status."""
        service = GenerationDatabaseService(mock_db_session)
        
        # Test empty session_id
        with pytest.raises(ValidationError) as exc_info:
            await service.update_session_status("", "completed")
        assert exc_info.value.field == "session_id"
        
        # Test invalid status
        with pytest.raises(ValidationError) as exc_info:
            await service.update_session_status("test-id", "invalid_status")
        assert exc_info.value.field == "status"
        
        # Test negative final_sample_count
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = MagicMock()
        with pytest.raises(ValidationError) as exc_info:
            await service.update_session_status("test-id", "completed", final_sample_count=-1)
        assert exc_info.value.field == "final_sample_count"


class TestFastAPIErrorHandling:
    """Test FastAPI error handling integration."""
    
    @pytest.fixture
    def client(self):
        """Create test FastAPI client."""
        app = create_app(testing=True)
        return TestClient(app)
        
    def test_correlation_id_in_response_headers(self, client):
        """Test that correlation ID is included in response headers."""
        # Make request without correlation ID
        response = client.get("/health/status")
        assert "X-Correlation-ID" in response.headers
        
        # Make request with custom correlation ID
        custom_id = str(uuid.uuid4())
        response = client.get(
            "/health/status",
            headers={"X-Correlation-ID": custom_id}
        )
        assert response.headers["X-Correlation-ID"] == custom_id
        
    def test_validation_error_response_format(self, client):
        """Test validation error response format."""
        # This would need a specific endpoint that triggers validation errors
        # For now, we'll test the format with a mock
        
        exc = ValidationError(
            message="Invalid input",
            field="email",
            value="invalid-email"
        )
        response_data = create_error_response(exc, include_debug=True)
        
        assert "error" in response_data
        assert response_data["error"]["code"] == "VALIDATION_ERROR"
        assert response_data["error"]["message"] == "Invalid input"
        assert "correlation_id" in response_data["error"]
        assert "timestamp" in response_data["error"]
        assert "details" in response_data["error"]


class TestCorrelationIdTracking:
    """Test correlation ID tracking across async boundaries."""
    
    @pytest.mark.asyncio
    async def test_correlation_context_setting(self):
        """Test correlation ID context setting and retrieval."""
        test_id = "test-correlation-id"
        set_correlation_id(test_id)
        
        assert get_correlation_id() == test_id
        
    @pytest.mark.asyncio
    async def test_correlation_context_in_exceptions(self):
        """Test correlation ID is preserved in exception context."""
        test_id = "test-correlation-id"
        set_correlation_id(test_id)
        
        @handle_repository_errors()
        async def test_method():
            raise ValueError("Test error")
        
        with pytest.raises(ValidationError) as exc_info:
            await test_method()
        
        # The decorator should use the context correlation ID
        assert exc_info.value.correlation_id == test_id


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing error handling."""
    
    def test_existing_exception_types_still_work(self):
        """Test that existing exception usage patterns still work."""
        # Test that we can still create exceptions the old way
        exc = DatabaseError("Database failed", query="SELECT * FROM test")
        
        assert exc.message == "Database failed"
        assert exc.query == "SELECT * FROM test"
        assert exc.error_code == "DATABASE_ERROR"
        
        # Test inheritance chain
        assert isinstance(exc, InfrastructureError)
        assert isinstance(exc, PromptImproverError)
        
    def test_exception_attributes_preserved(self):
        """Test that all original exception attributes are preserved."""
        # Test specific attributes are maintained
        cache_error = CacheError(
            message="Cache failed",
            cache_key="test:key",
            cache_type="redis",
            operation="get"
        )
        
        assert cache_error.cache_key == "test:key"
        assert cache_error.details["cache_type"] == "redis"
        assert cache_error.details["operation"] == "get"
        
    def test_error_logging_compatibility(self):
        """Test that error logging works with both old and new patterns."""
        with patch('prompt_improver.common.exceptions.logger') as mock_logger:
            exc = SecurityError(
                message="Security violation detected",
                security_check="input_validation",
                threat_level="HIGH"
            )
            
            exc.log_error()
            
            # Verify logging was called
            mock_logger.critical.assert_called_once()
            call_args = mock_logger.critical.call_args
            
            # Check log message format
            assert "SECURITY_ERROR" in call_args[0][0]
            assert "Security violation detected" in call_args[0][0]
            
            # Check structured logging data
            extra = call_args[1]["extra"]
            assert "correlation_id" in extra
            assert "error_details" in extra


class TestErrorRecoveryPatterns:
    """Test error recovery and retry mechanisms."""
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_after_failure(self):
        """Test retry mechanism succeeds after initial failures."""
        from prompt_improver.common.error_handling import create_retry_with_backoff
        
        attempt_count = 0
        
        @create_retry_with_backoff(max_retries=2, base_delay=0.01)
        async def failing_then_succeeding_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await failing_then_succeeding_operation()
        assert result == "success"
        assert attempt_count == 3
        
    @pytest.mark.asyncio
    async def test_retry_with_backoff_final_failure(self):
        """Test retry mechanism fails after max retries."""
        from prompt_improver.common.error_handling import create_retry_with_backoff
        
        @create_retry_with_backoff(max_retries=1, base_delay=0.01)
        async def always_failing_operation():
            raise ConnectionError("Permanent failure")
        
        with pytest.raises(ConnectionError) as exc_info:
            await always_failing_operation()
        
        assert "Permanent failure" in str(exc_info.value)


@pytest.mark.integration
class TestEndToEndErrorPropagation:
    """Test complete error propagation from repository to API response."""
    
    @pytest.mark.asyncio
    async def test_database_error_propagation_chain(self):
        """Test error propagation from database through service to API."""
        
        # This would require a more complex setup with actual service injection
        # For demonstration, we'll simulate the error propagation chain
        
        # 1. Repository raises database error
        @handle_repository_errors()
        async def mock_repository_method():
            raise SQLAlchemyError("Connection pool exhausted")
        
        # 2. Service catches and adds context
        @handle_service_errors()
        async def mock_service_method():
            return await mock_repository_method()
        
        # 3. Verify error propagation
        with pytest.raises(DatabaseError) as exc_info:
            await mock_service_method()
        
        error = exc_info.value
        
        # Verify error has both repository and service context
        assert error.context["layer"] == "repository"
        assert "service_layer" in error.context
        assert isinstance(error.cause, SQLAlchemyError)
        assert "Connection pool exhausted" in str(error.cause)
        
        # Verify error response format
        api_response = create_error_response(error, include_debug=True)
        assert api_response["error"]["code"] == "DATABASE_ERROR"
        assert "correlation_id" in api_response["error"]
        
    def test_error_severity_classification(self):
        """Test that errors are classified with appropriate severity levels."""
        # WARNING level errors
        auth_error = AuthenticationError("Invalid token")
        assert auth_error.severity == "WARNING"
        
        rate_limit_error = RateLimitError("Too many requests")  
        assert rate_limit_error.severity == "WARNING"
        
        # ERROR level errors
        validation_error = ValidationError("Invalid input")
        assert validation_error.severity == "ERROR"
        
        cache_error = CacheError("Redis unavailable")
        assert cache_error.severity == "ERROR"
        
        # CRITICAL level errors
        config_error = ConfigurationError("Missing required config")
        assert config_error.severity == "CRITICAL"
        
        internal_error = InternalError("Unexpected system failure")
        assert internal_error.severity == "CRITICAL"
        
        security_error = SecurityError("SQL injection attempt", threat_level="HIGH")
        assert security_error.severity == "CRITICAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])