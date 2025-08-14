"""Comprehensive Real Behavior Tests for Error Handling Services.

Tests the complete Error Handling Services decomposition with real behavior validation:
- ErrorHandlingFacade (unified error coordination)
- DatabaseErrorService (database error handling)
- NetworkErrorService (network error handling)  
- ValidationErrorService (validation error handling)

All tests use real errors and failure scenarios - no mocks.
Performance targets: <1ms error routing, <5ms end-to-end error processing.
"""

import asyncio
import logging
import pytest
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from prompt_improver.services.error_handling.facade import (
    ErrorHandlingFacade,
    ErrorHandlingFacadeProtocol,
    UnifiedErrorContext,
    ErrorServiceType,
    ErrorProcessingMode,
)
from prompt_improver.services.error_handling.database_error_service import (
    DatabaseErrorService,
    DatabaseErrorContext,
    DatabaseErrorCategory,
)
from prompt_improver.services.error_handling.network_error_service import (
    NetworkErrorService,
    NetworkErrorContext,
    NetworkErrorCategory,
)
from prompt_improver.services.error_handling.validation_error_service import (
    ValidationErrorService,
    ValidationErrorContext,
    ValidationErrorCategory,
)
from prompt_improver.core.services.security import SecurityLevel
from tests.integration.real_behavior_testing.containers.network_simulator import (
    NetworkSimulator,
    FailureType,
)

logger = logging.getLogger(__name__)


@pytest.fixture
async def database_error_service() -> DatabaseErrorService:
    """Create database error service with correlation tracking."""
    correlation_id = str(uuid.uuid4())[:8]
    service = DatabaseErrorService(correlation_id=correlation_id)
    yield service
    # Cleanup if needed


@pytest.fixture
async def network_error_service() -> NetworkErrorService:
    """Create network error service with correlation tracking."""
    correlation_id = str(uuid.uuid4())[:8]
    service = NetworkErrorService(correlation_id=correlation_id)
    yield service
    # Cleanup if needed


@pytest.fixture
async def validation_error_service() -> ValidationErrorService:
    """Create validation error service with correlation tracking."""
    correlation_id = str(uuid.uuid4())[:8]
    service = ValidationErrorService(correlation_id=correlation_id)
    yield service
    # Cleanup if needed


@pytest.fixture
async def error_handling_facade() -> ErrorHandlingFacade:
    """Create error handling facade with all services."""
    correlation_id = str(uuid.uuid4())[:8]
    facade = ErrorHandlingFacade(
        correlation_id=correlation_id,
        enable_caching=True,
        enable_batch_processing=True,
        default_security_level=SecurityLevel.internal
    )
    yield facade
    # Cleanup if needed


class TestDatabaseErrorService:
    """Test Database Error Service with real database error scenarios."""
    
    async def test_database_connection_error_handling(
        self,
        database_error_service: DatabaseErrorService,
        performance_tracker,
    ):
        """Test database connection error handling with real connection failures."""
        # Simulate real database connection errors
        connection_errors = [
            ConnectionError("Connection to database failed: Connection refused"),
            TimeoutError("Database connection timeout after 30 seconds"),
            RuntimeError("Database server not available"),
        ]
        
        for error in connection_errors:
            start_time = time.perf_counter()
            
            result = await database_error_service.handle_database_error(
                error=error,
                operation_name="database_connection",
                connection_details={"host": "localhost", "port": 5432, "database": "test_db"},
                query_info=None
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker("database_error_handling", duration, 5.0)  # <5ms target
            
            # Validate error handling result
            assert isinstance(result, DatabaseErrorContext)
            assert result.correlation_id == database_error_service.correlation_id
            assert result.operation_name == "database_connection"
            assert result.error_category in [category.value for category in DatabaseErrorCategory]
            assert result.processing_time_ms > 0
            
            # Connection errors should typically be retryable
            if isinstance(error, (ConnectionError, TimeoutError)):
                assert result.is_retryable
            
            logger.info(f"Database error handled: {error.__class__.__name__} -> {result.error_category}")

    async def test_database_query_error_analysis(
        self,
        database_error_service: DatabaseErrorService,
        performance_tracker,
    ):
        """Test database query error analysis and categorization."""
        query_errors = [
            RuntimeError("relation 'non_existent_table' does not exist"),
            ValueError("column 'invalid_column' does not exist"),
            RuntimeError("duplicate key value violates unique constraint"),
            TimeoutError("Query timeout: execution exceeded 30 seconds"),
            PermissionError("permission denied for table users"),
        ]
        
        query_info = {
            "sql": "SELECT * FROM users WHERE id = $1",
            "parameters": [123],
            "execution_plan": "Index Scan using users_pkey",
            "estimated_cost": 1.5,
        }
        
        results = []
        
        for error in query_errors:
            start_time = time.perf_counter()
            
            result = await database_error_service.handle_database_error(
                error=error,
                operation_name="query_execution",
                connection_details={"host": "localhost", "port": 5432},
                query_info=query_info
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker(f"query_error_{error.__class__.__name__}", duration, 10.0)
            
            results.append((error, result))
            
            # Validate query-specific error handling
            assert result.operation_name == "query_execution"
            assert result.query_info == query_info
            
            # Different error types should be categorized appropriately
            if "does not exist" in str(error):
                assert result.error_category in [DatabaseErrorCategory.SCHEMA_ERROR.value, DatabaseErrorCategory.QUERY_ERROR.value]
            elif "timeout" in str(error).lower():
                assert result.error_category == DatabaseErrorCategory.PERFORMANCE_ERROR.value
            elif "permission denied" in str(error):
                assert result.error_category == DatabaseErrorCategory.ACCESS_ERROR.value
        
        # Validate error analysis diversity
        categories = [result.error_category for _, result in results]
        unique_categories = set(categories)
        assert len(unique_categories) > 1, "Should categorize different error types differently"

    async def test_database_transaction_error_handling(
        self,
        database_error_service: DatabaseErrorService,
        performance_tracker,
    ):
        """Test database transaction error handling."""
        transaction_errors = [
            RuntimeError("current transaction is aborted, commands ignored until end of transaction block"),
            RuntimeError("deadlock detected"),
            RuntimeError("serialization failure, transaction rolled back"),
        ]
        
        transaction_context = {
            "transaction_id": "txn_123456",
            "isolation_level": "READ_COMMITTED",
            "operations_count": 5,
            "duration_ms": 250,
        }
        
        for error in transaction_errors:
            start_time = time.perf_counter()
            
            result = await database_error_service.handle_database_error(
                error=error,
                operation_name="transaction_commit",
                transaction_context=transaction_context,
                connection_details={"host": "localhost", "port": 5432}
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker("transaction_error_handling", duration, 8.0)
            
            # Transaction errors should include transaction context
            assert result.transaction_context == transaction_context
            assert result.error_category == DatabaseErrorCategory.TRANSACTION_ERROR.value
            
            # Most transaction errors should be retryable (with backoff)
            assert result.is_retryable

    async def test_database_error_statistics_tracking(
        self,
        database_error_service: DatabaseErrorService,
    ):
        """Test database error statistics tracking."""
        # Generate various errors to build statistics
        test_errors = [
            (ConnectionError("Connection failed"), "connection_test"),
            (TimeoutError("Query timeout"), "query_test"),
            (ValueError("Invalid parameter"), "validation_test"),
            (RuntimeError("Constraint violation"), "constraint_test"),
        ]
        
        for error, operation in test_errors:
            await database_error_service.handle_database_error(
                error=error,
                operation_name=operation,
                connection_details={"host": "test_host"}
            )
        
        # Get error statistics
        stats = database_error_service.get_error_statistics()
        
        # Validate statistics structure
        assert isinstance(stats, dict)
        assert "correlation_id" in stats
        assert "total_errors_handled" in stats
        assert "error_categories" in stats
        assert "operation_stats" in stats
        assert "performance_metrics" in stats
        
        # Should have processed multiple errors
        assert stats["total_errors_handled"] >= len(test_errors)
        
        # Should have multiple error categories
        error_categories = stats["error_categories"]
        assert len(error_categories) > 1
        
        # Performance metrics should be tracked
        performance_metrics = stats["performance_metrics"]
        assert "average_processing_time_ms" in performance_metrics
        assert performance_metrics["average_processing_time_ms"] > 0


class TestNetworkErrorService:
    """Test Network Error Service with real network error scenarios."""
    
    async def test_connection_error_handling(
        self,
        network_error_service: NetworkErrorService,
        network_simulator: NetworkSimulator,
        performance_tracker,
    ):
        """Test network connection error handling with real connection failures."""
        # Create network failure scenarios
        await network_simulator.start()
        await network_simulator.add_failure_scenario(
            "connection_test",
            FailureType.CONNECTION_REFUSED,
            probability=1.0  # Always fail for testing
        )
        
        try:
            # Simulate network operation that will fail
            await network_simulator.simulate_network_operation(
                operation_name="test_connection",
                target_host="unreachable.example.com",
                operation_type="http_request"
            )
        except Exception as network_error:
            start_time = time.perf_counter()
            
            result = await network_error_service.handle_network_error(
                error=network_error,
                operation_name="http_connection_test",
                request_details={
                    "url": "https://unreachable.example.com/api/test",
                    "method": "GET",
                    "timeout_ms": 5000,
                },
                connection_info={
                    "host": "unreachable.example.com",
                    "port": 443,
                    "protocol": "https",
                }
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker("network_connection_error", duration, 3.0)
            
            # Validate network error handling
            assert isinstance(result, NetworkErrorContext)
            assert result.correlation_id == network_error_service.correlation_id
            assert result.operation_name == "http_connection_test"
            assert result.error_category in [category.value for category in NetworkErrorCategory]
            assert result.processing_time_ms > 0
            
            # Connection errors should typically be retryable
            assert result.is_retryable
            
        finally:
            await network_simulator.stop()

    async def test_timeout_error_analysis(
        self,
        network_error_service: NetworkErrorService,
        performance_tracker,
    ):
        """Test network timeout error analysis."""
        timeout_errors = [
            asyncio.TimeoutError("Request timeout after 30 seconds"),
            TimeoutError("Connection timeout"),
            RuntimeError("Read timeout on socket"),
        ]
        
        request_details = {
            "url": "https://slow.example.com/api/slow-endpoint",
            "method": "POST",
            "timeout_ms": 30000,
            "retry_count": 0,
            "payload_size_bytes": 1024,
        }
        
        for error in timeout_errors:
            start_time = time.perf_counter()
            
            result = await network_error_service.handle_network_error(
                error=error,
                operation_name="slow_api_request",
                request_details=request_details,
                timing_info={
                    "dns_resolution_ms": 50,
                    "connection_establishment_ms": 100,
                    "request_sent_ms": 5,
                    "time_to_first_byte_ms": None,  # Timeout occurred
                }
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker(f"timeout_error_{error.__class__.__name__}", duration, 5.0)
            
            # Timeout errors should be categorized correctly
            assert result.error_category == NetworkErrorCategory.TIMEOUT_ERROR.value
            assert result.is_retryable  # Timeouts are generally retryable
            assert result.request_details == request_details

    async def test_http_status_error_handling(
        self,
        network_error_service: NetworkErrorService,
        performance_tracker,
    ):
        """Test HTTP status error handling."""
        http_errors = [
            RuntimeError("HTTP 404: Not Found"),
            RuntimeError("HTTP 500: Internal Server Error"),
            RuntimeError("HTTP 503: Service Unavailable"),
            RuntimeError("HTTP 429: Too Many Requests"),
            PermissionError("HTTP 403: Forbidden"),
            ValueError("HTTP 400: Bad Request"),
        ]
        
        for error in http_errors:
            start_time = time.perf_counter()
            
            # Extract status code from error message
            error_msg = str(error)
            status_code = None
            if "HTTP" in error_msg:
                try:
                    status_code = int(error_msg.split("HTTP ")[1].split(":")[0])
                except (IndexError, ValueError):
                    status_code = 500
            
            result = await network_error_service.handle_network_error(
                error=error,
                operation_name="api_request",
                request_details={
                    "url": "https://api.example.com/endpoint",
                    "method": "GET"
                },
                response_details={
                    "status_code": status_code,
                    "response_time_ms": 150,
                    "response_size_bytes": 256,
                }
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker(f"http_status_error_{status_code}", duration, 4.0)
            
            # HTTP errors should be categorized based on status code
            if status_code and 400 <= status_code < 500:
                assert result.error_category == NetworkErrorCategory.CLIENT_ERROR.value
                # Client errors generally not retryable (except 429)
                if status_code == 429:  # Rate limit
                    assert result.is_retryable
            elif status_code and 500 <= status_code < 600:
                assert result.error_category == NetworkErrorCategory.SERVER_ERROR.value
                assert result.is_retryable  # Server errors generally retryable

    async def test_network_error_retry_recommendations(
        self,
        network_error_service: NetworkErrorService,
    ):
        """Test network error retry recommendations."""
        retry_scenarios = [
            (asyncio.TimeoutError("Connection timeout"), True, "exponential_backoff"),
            (RuntimeError("HTTP 503: Service Unavailable"), True, "exponential_backoff"),
            (RuntimeError("HTTP 429: Too Many Requests"), True, "rate_limit_backoff"),
            (ValueError("HTTP 400: Bad Request"), False, None),
            (PermissionError("HTTP 403: Forbidden"), False, None),
        ]
        
        for error, should_retry, expected_strategy in retry_scenarios:
            result = await network_error_service.handle_network_error(
                error=error,
                operation_name="retry_analysis_test",
                request_details={"url": "https://example.com/api"}
            )
            
            assert result.is_retryable == should_retry
            
            if should_retry and expected_strategy:
                retry_recommendations = result.recovery_suggestions.get("retry_strategy")
                if retry_recommendations:
                    assert expected_strategy in str(retry_recommendations)


class TestValidationErrorService:
    """Test Validation Error Service with real validation scenarios."""
    
    async def test_input_validation_error_handling(
        self,
        validation_error_service: ValidationErrorService,
        performance_tracker,
    ):
        """Test input validation error handling."""
        validation_errors = [
            ValueError("Field 'email' is required but not provided"),
            ValueError("Invalid email format: 'not-an-email'"),
            ValueError("Password must be at least 8 characters long"),
            ValueError("Age must be between 1 and 150"),
        ]
        
        input_data = {
            "email": "not-an-email",
            "password": "123",
            "age": 200,
            "name": "",
        }
        
        validation_rules = {
            "email": {"required": True, "format": "email"},
            "password": {"required": True, "min_length": 8},
            "age": {"required": True, "min": 1, "max": 150},
            "name": {"required": True, "min_length": 1},
        }
        
        for error in validation_errors:
            start_time = time.perf_counter()
            
            result = await validation_error_service.handle_validation_error(
                error=error,
                operation_name="user_registration_validation",
                input_data=input_data,
                validation_rules=validation_rules,
                security_context={
                    "user_id": None,  # Anonymous registration
                    "ip_address": "192.168.1.1",
                    "user_agent": "Mozilla/5.0 (Test Browser)"
                }
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker("validation_error_handling", duration, 2.0)  # <2ms target
            
            # Validate validation error handling
            assert isinstance(result, ValidationErrorContext)
            assert result.correlation_id == validation_error_service.correlation_id
            assert result.operation_name == "user_registration_validation"
            assert result.error_category in [category.value for category in ValidationErrorCategory]
            assert result.input_data == input_data
            assert result.validation_rules == validation_rules
            
            # Validation errors are typically not retryable
            assert not result.is_retryable

    async def test_security_validation_error_detection(
        self,
        validation_error_service: ValidationErrorService,
        performance_tracker,
    ):
        """Test security-related validation error detection."""
        security_errors = [
            ValueError("Potential SQL injection detected in input"),
            ValueError("Cross-site scripting (XSS) attempt blocked"),
            ValueError("Suspicious file upload detected: executable content"),
            ValueError("Rate limit exceeded: too many requests"),
        ]
        
        suspicious_inputs = [
            {"query": "'; DROP TABLE users; --"},
            {"comment": "<script>alert('xss')</script>"},
            {"filename": "malware.exe.txt", "content": "MZ\x90\x00"},  # PE header
            {"api_key": "multiple_rapid_requests"},
        ]
        
        for error, suspicious_input in zip(security_errors, suspicious_inputs):
            start_time = time.perf_counter()
            
            result = await validation_error_service.handle_validation_error(
                error=error,
                operation_name="security_validation",
                input_data=suspicious_input,
                security_context={
                    "user_id": "anonymous",
                    "ip_address": "suspicious.ip.address",
                    "threat_level": "high",
                }
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker("security_validation_error", duration, 3.0)
            
            # Security validation errors should be detected
            if any(threat in str(error).lower() for threat in ["injection", "xss", "malware", "suspicious"]):
                assert result.threat_detected
                assert result.security_level == SecurityLevel.restricted
                assert result.error_category == ValidationErrorCategory.SECURITY_VIOLATION.value
            
            # Security violations should never be retryable
            assert not result.is_retryable

    async def test_business_rule_validation_errors(
        self,
        validation_error_service: ValidationErrorService,
        performance_tracker,
    ):
        """Test business rule validation errors."""
        business_rule_errors = [
            ValueError("Insufficient account balance for transaction"),
            ValueError("User not authorized to access this resource"),
            ValueError("Order quantity exceeds available inventory"),
            ValueError("Discount code expired or invalid"),
        ]
        
        business_contexts = [
            {"account_balance": 50.0, "transaction_amount": 100.0, "user_id": "user123"},
            {"user_role": "guest", "required_role": "admin", "resource": "/admin/users"},
            {"available_inventory": 5, "order_quantity": 10, "product_id": "prod456"},
            {"discount_code": "EXPIRED2023", "expiry_date": "2023-12-31", "current_date": "2024-01-15"},
        ]
        
        for error, business_context in zip(business_rule_errors, business_contexts):
            start_time = time.perf_counter()
            
            result = await validation_error_service.handle_validation_error(
                error=error,
                operation_name="business_rule_validation",
                input_data=business_context,
                business_context=business_context
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker("business_rule_validation", duration, 4.0)
            
            # Business rule validation should include business context
            assert result.business_context == business_context
            assert result.error_category == ValidationErrorCategory.BUSINESS_RULE_VIOLATION.value
            
            # Business rule violations typically not retryable
            assert not result.is_retryable

    async def test_validation_error_statistics(
        self,
        validation_error_service: ValidationErrorService,
    ):
        """Test validation error statistics tracking."""
        # Generate various validation errors
        test_scenarios = [
            (ValueError("Required field missing"), "field_validation"),
            (ValueError("Invalid format"), "format_validation"),
            (ValueError("Security threat detected"), "security_validation"),
            (ValueError("Business rule violation"), "business_validation"),
        ]
        
        for error, operation in test_scenarios:
            await validation_error_service.handle_validation_error(
                error=error,
                operation_name=operation,
                input_data={"test": "data"}
            )
        
        # Get validation error statistics
        stats = validation_error_service.get_error_statistics()
        
        # Validate statistics structure
        assert isinstance(stats, dict)
        assert "correlation_id" in stats
        assert "total_errors_handled" in stats
        assert "error_categories" in stats
        assert "threat_detection_stats" in stats
        assert "security_incidents" in stats
        
        # Should have processed multiple errors
        assert stats["total_errors_handled"] >= len(test_scenarios)
        
        # Should track threat detection
        threat_stats = stats["threat_detection_stats"]
        assert "total_threats_detected" in threat_stats
        assert "security_violations" in threat_stats


class TestErrorHandlingFacade:
    """Test Error Handling Facade integration and coordination."""
    
    async def test_unified_error_handling_with_classification(
        self,
        error_handling_facade: ErrorHandlingFacade,
        performance_tracker,
    ):
        """Test unified error handling with automatic error classification."""
        # Test errors that should be classified to different services
        test_errors = [
            (ConnectionError("Database connection failed"), "db_operation", ErrorServiceType.DATABASE),
            (asyncio.TimeoutError("HTTP request timeout"), "api_call", ErrorServiceType.NETWORK),
            (ValueError("Invalid input format"), "user_input_validation", ErrorServiceType.VALIDATION),
            (RuntimeError("System error occurred"), "system_operation", ErrorServiceType.SYSTEM),
        ]
        
        for error, operation_name, expected_service_type in test_errors:
            start_time = time.perf_counter()
            
            result = await error_handling_facade.handle_unified_error(
                error=error,
                operation_name=operation_name,
                processing_mode=ErrorProcessingMode.SYNCHRONOUS
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            performance_tracker("unified_error_handling", duration, 5.0)  # <5ms target
            
            # Validate unified error context
            assert isinstance(result, UnifiedErrorContext)
            assert result.correlation_id == error_handling_facade.correlation_id
            assert result.operation_name == operation_name
            assert result.service_type == expected_service_type
            assert result.processing_duration_ms > 0
            
            logger.info(f"Error classified: {error.__class__.__name__} -> {result.service_type.value}")

    async def test_facade_error_routing_performance(
        self,
        error_handling_facade: ErrorHandlingFacade,
        performance_tracker,
    ):
        """Test facade error routing performance (target: <1ms)."""
        # Test rapid error classification
        errors_to_classify = [
            ConnectionError("DB connection lost"),
            TimeoutError("Network timeout"),
            ValueError("Validation failed"),
            RuntimeError("System error"),
        ] * 10  # 40 total errors
        
        start_time = time.perf_counter()
        
        for i, error in enumerate(errors_to_classify):
            # Test just the classification (not full handling)
            service_type = error_handling_facade._classify_error_service_type(
                error, f"operation_{i}"
            )
            assert isinstance(service_type, ErrorServiceType)
        
        total_time = (time.perf_counter() - start_time) * 1000
        avg_time = total_time / len(errors_to_classify)
        
        performance_tracker("error_routing_performance", avg_time, 1.0)  # <1ms per routing
        
        logger.info(f"Error routing performance: {avg_time:.3f}ms average ({len(errors_to_classify)} errors)")

    async def test_batch_error_processing(
        self,
        error_handling_facade: ErrorHandlingFacade,
        performance_tracker,
    ):
        """Test batch error processing performance."""
        # Create batch of errors
        error_batch = [
            (ConnectionError(f"DB error {i}"), f"db_operation_{i}", {"context": f"batch_{i}"})
            for i in range(20)
        ] + [
            (TimeoutError(f"Network error {i}"), f"network_operation_{i}", {"context": f"batch_{i}"})
            for i in range(15)
        ] + [
            (ValueError(f"Validation error {i}"), f"validation_operation_{i}", {"context": f"batch_{i}"})
            for i in range(10)
        ]
        
        start_time = time.perf_counter()
        
        batch_results = await error_handling_facade.batch_handle_errors(
            errors=error_batch,
            processing_mode=ErrorProcessingMode.BATCH
        )
        
        duration = (time.perf_counter() - start_time) * 1000
        performance_tracker("batch_error_processing", duration, 500.0)  # Allow time for batch processing
        
        # Validate batch results
        assert isinstance(batch_results, list)
        assert len(batch_results) <= len(error_batch)  # May have some failures
        
        # Validate batch processing efficiency
        avg_time_per_error = duration / len(error_batch)
        logger.info(f"Batch processing: {len(error_batch)} errors in {duration:.1f}ms ({avg_time_per_error:.2f}ms per error)")
        
        # Batch processing should be more efficient than individual processing
        assert avg_time_per_error < 10.0, f"Batch processing too slow: {avg_time_per_error:.2f}ms per error"

    async def test_facade_caching_performance(
        self,
        error_handling_facade: ErrorHandlingFacade,
        performance_tracker,
    ):
        """Test facade error classification caching."""
        # Test repeated error classification (should benefit from caching)
        test_error = ConnectionError("Repeated database connection error")
        operation_name = "repeated_db_operation"
        
        # First classification (cache miss)
        start_time = time.perf_counter()
        first_service_type = error_handling_facade._classify_error_service_type(test_error, operation_name)
        first_duration = (time.perf_counter() - start_time) * 1000
        
        # Second classification (cache hit)
        start_time = time.perf_counter()
        second_service_type = error_handling_facade._classify_error_service_type(test_error, operation_name)
        second_duration = (time.perf_counter() - start_time) * 1000
        
        performance_tracker("error_classification_cache_miss", first_duration, 2.0)
        performance_tracker("error_classification_cache_hit", second_duration, 1.0)
        
        # Results should be identical
        assert first_service_type == second_service_type
        
        # Cache hit should be faster (or at least not slower)
        cache_speedup = first_duration > second_duration * 1.5  # 50% improvement threshold
        logger.info(f"Cache performance: first={first_duration:.3f}ms, second={second_duration:.3f}ms, speedup={cache_speedup}")

    async def test_facade_asynchronous_processing(
        self,
        error_handling_facade: ErrorHandlingFacade,
        performance_tracker,
    ):
        """Test facade asynchronous error processing."""
        # Test async processing mode
        async_errors = [
            RuntimeError(f"Async error {i}") for i in range(5)
        ]
        
        start_time = time.perf_counter()
        
        async_results = []
        for i, error in enumerate(async_errors):
            result = await error_handling_facade.handle_unified_error(
                error=error,
                operation_name=f"async_operation_{i}",
                processing_mode=ErrorProcessingMode.ASYNCHRONOUS
            )
            async_results.append(result)
        
        immediate_response_time = (time.perf_counter() - start_time) * 1000
        performance_tracker("async_error_processing_immediate", immediate_response_time, 50.0)
        
        # Async processing should return quickly (operations run in background)
        avg_immediate_time = immediate_response_time / len(async_errors)
        assert avg_immediate_time < 20.0, f"Async processing too slow: {avg_immediate_time:.1f}ms per error"
        
        # Wait briefly for background processing
        await asyncio.sleep(0.1)
        
        # All results should indicate async processing
        for result in async_results:
            assert result.processing_mode == ErrorProcessingMode.ASYNCHRONOUS
            assert "asynchronously" in result.recommended_action

    async def test_facade_comprehensive_statistics(
        self,
        error_handling_facade: ErrorHandlingFacade,
    ):
        """Test facade comprehensive error statistics."""
        # Generate various errors to build statistics
        test_errors = [
            (ConnectionError("DB error 1"), "db_op_1"),
            (ConnectionError("DB error 2"), "db_op_2"),
            (TimeoutError("Network error 1"), "net_op_1"),
            (ValueError("Validation error 1"), "val_op_1"),
            (RuntimeError("System error 1"), "sys_op_1"),
        ]
        
        for error, operation in test_errors:
            await error_handling_facade.handle_unified_error(
                error=error,
                operation_name=operation,
                processing_mode=ErrorProcessingMode.SYNCHRONOUS
            )
        
        # Get unified statistics
        stats = error_handling_facade.get_unified_error_statistics()
        
        # Validate comprehensive statistics
        assert isinstance(stats, dict)
        assert "correlation_id" in stats
        assert "facade_stats" in stats
        assert "service_statistics" in stats
        assert "service_health" in stats
        
        # Facade stats should show activity
        facade_stats = stats["facade_stats"]
        assert facade_stats["total_errors_processed"] >= len(test_errors)
        assert len(facade_stats["error_counts_by_service"]) > 1  # Multiple services used
        assert facade_stats["cache_enabled"]
        assert facade_stats["batch_processing_enabled"]
        
        # Service statistics should include all services
        service_stats = stats["service_statistics"]
        assert len(service_stats) >= 3  # Database, Network, Validation services


class TestIntegratedErrorHandlingWorkflow:
    """Test complete integrated error handling workflow scenarios."""
    
    async def test_end_to_end_error_handling_workflow(
        self,
        error_handling_facade: ErrorHandlingFacade,
        network_simulator: NetworkSimulator,
        performance_tracker,
    ):
        """Test end-to-end error handling workflow with realistic scenarios."""
        # Setup network simulator for realistic failures
        await network_simulator.start()
        await network_simulator.create_common_failure_scenarios()
        
        try:
            # Simulate complex application workflow with multiple error points
            workflow_errors = []
            
            # 1. Database connection error
            try:
                raise ConnectionError("Failed to connect to database: Connection refused (port 5432)")
            except Exception as e:
                workflow_errors.append(("database_initialization", e))
            
            # 2. Network API timeout
            try:
                await network_simulator.simulate_http_request(
                    url="https://api.external-service.com/data",
                    timeout_ms=1000
                )
            except Exception as e:
                workflow_errors.append(("external_api_call", e))
            
            # 3. Validation error
            try:
                raise ValueError("Invalid user input: email format is incorrect")
            except Exception as e:
                workflow_errors.append(("user_input_processing", e))
            
            # 4. System resource error
            try:
                raise RuntimeError("Insufficient system resources: disk space low")
            except Exception as e:
                workflow_errors.append(("resource_allocation", e))
            
            # Process all errors through facade
            start_time = time.perf_counter()
            
            error_results = []
            for operation_name, error in workflow_errors:
                result = await error_handling_facade.handle_unified_error(
                    error=error,
                    operation_name=operation_name,
                    processing_mode=ErrorProcessingMode.SYNCHRONOUS,
                    security_level=SecurityLevel.internal,
                    system_context={
                        "workflow_id": "test_workflow_001",
                        "step": operation_name,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                error_results.append((operation_name, result))
            
            total_time = (time.perf_counter() - start_time) * 1000
            performance_tracker("end_to_end_error_workflow", total_time, 100.0)  # <100ms for 4 errors
            
            # Validate comprehensive error handling
            assert len(error_results) == len(workflow_errors)
            
            # Each error should be handled appropriately
            for operation_name, result in error_results:
                assert result.operation_name == operation_name
                assert result.service_type in [ErrorServiceType.DATABASE, ErrorServiceType.NETWORK, 
                                               ErrorServiceType.VALIDATION, ErrorServiceType.SYSTEM]
                assert result.processing_duration_ms > 0
                
                logger.info(f"Workflow error handled: {operation_name} -> {result.service_type.value}")
            
            # Get final statistics
            final_stats = error_handling_facade.get_unified_error_statistics()
            logger.info(f"Workflow completed: {final_stats['facade_stats']['total_errors_processed']} total errors processed")
            
        finally:
            await network_simulator.stop()

    async def test_performance_targets_validation(
        self,
        error_handling_facade: ErrorHandlingFacade,
        performance_tracker,
    ):
        """Validate all Error Handling Services meet performance targets."""
        performance_results = {}
        
        # Test individual service performance
        test_scenarios = [
            ("database_service", ConnectionError("DB connection failed"), "db_operation"),
            ("network_service", TimeoutError("Network timeout"), "network_operation"),
            ("validation_service", ValueError("Invalid input"), "validation_operation"),
        ]
        
        for service_name, error, operation_name in test_scenarios:
            start_time = time.perf_counter()
            
            result = await error_handling_facade.handle_unified_error(
                error=error,
                operation_name=operation_name,
                processing_mode=ErrorProcessingMode.SYNCHRONOUS
            )
            
            duration = (time.perf_counter() - start_time) * 1000
            
            # Service-specific performance targets
            targets = {
                "database_service": 10.0,  # <10ms for database error handling
                "network_service": 8.0,    # <8ms for network error handling
                "validation_service": 5.0,  # <5ms for validation error handling
            }
            
            target = targets[service_name]
            performance_results[service_name] = {
                "duration_ms": duration,
                "target_ms": target,
                "met": duration < target
            }
            
            performance_tracker(f"{service_name}_performance", duration, target)
        
        # Test facade coordination performance (target: <1ms routing)
        routing_errors = [
            ConnectionError("Test DB error"),
            TimeoutError("Test network error"),
            ValueError("Test validation error"),
        ] * 5  # 15 total for averaging
        
        start_time = time.perf_counter()
        for i, error in enumerate(routing_errors):
            error_handling_facade._classify_error_service_type(error, f"test_op_{i}")
        
        total_routing_time = (time.perf_counter() - start_time) * 1000
        avg_routing_time = total_routing_time / len(routing_errors)
        
        performance_results["facade_routing"] = {
            "duration_ms": avg_routing_time,
            "target_ms": 1.0,
            "met": avg_routing_time < 1.0
        }
        performance_tracker("facade_routing_performance", avg_routing_time, 1.0)
        
        # Overall performance assessment
        total_targets_met = sum(1 for result in performance_results.values() if result["met"])
        total_targets = len(performance_results)
        performance_percentage = (total_targets_met / total_targets) * 100
        
        logger.info("Error Handling Services Performance Summary:")
        for service, result in performance_results.items():
            status = "✓" if result["met"] else "✗"
            logger.info(f"  {status} {service}: {result['duration_ms']:.2f}ms (target: {result['target_ms']}ms)")
        
        logger.info(f"Performance targets met: {total_targets_met}/{total_targets} ({performance_percentage:.1f}%)")
        
        # Should meet at least 80% of performance targets
        assert performance_percentage >= 80.0, f"Performance targets not met: {performance_percentage:.1f}%"

    async def test_error_handling_system_resilience(
        self,
        error_handling_facade: ErrorHandlingFacade,
        performance_tracker,
    ):
        """Test error handling system resilience under load."""
        # Generate high volume of errors to test system resilience
        error_volume = 100
        error_types = [
            lambda i: ConnectionError(f"High volume DB error {i}"),
            lambda i: TimeoutError(f"High volume network error {i}"),
            lambda i: ValueError(f"High volume validation error {i}"),
            lambda i: RuntimeError(f"High volume system error {i}"),
        ]
        
        start_time = time.perf_counter()
        
        # Process errors concurrently
        async def process_error(i):
            error_type = error_types[i % len(error_types)]
            error = error_type(i)
            return await error_handling_facade.handle_unified_error(
                error=error,
                operation_name=f"resilience_test_{i}",
                processing_mode=ErrorProcessingMode.SYNCHRONOUS
            )
        
        # Process errors in batches for manageable concurrency
        batch_size = 20
        results = []
        
        for batch_start in range(0, error_volume, batch_size):
            batch_end = min(batch_start + batch_size, error_volume)
            batch_tasks = [
                process_error(i) for i in range(batch_start, batch_end)
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        total_time = (time.perf_counter() - start_time) * 1000
        avg_time = total_time / error_volume
        
        performance_tracker("error_handling_resilience", avg_time, 20.0)  # <20ms average under load
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, UnifiedErrorContext)]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_results) / len(results) * 100
        
        logger.info(f"Resilience test results:")
        logger.info(f"  Total errors processed: {error_volume}")
        logger.info(f"  Successful processing: {len(successful_results)} ({success_rate:.1f}%)")
        logger.info(f"  Exceptions: {len(exceptions)}")
        logger.info(f"  Average processing time: {avg_time:.2f}ms")
        logger.info(f"  Total processing time: {total_time:.1f}ms")
        
        # System should maintain high success rate under load
        assert success_rate >= 95.0, f"Low success rate under load: {success_rate:.1f}%"
        
        # Get final health status
        health_status = error_handling_facade.get_unified_error_statistics()
        assert health_status["service_health"] == "operational", "System should remain operational under load"