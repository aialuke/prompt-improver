"""
Integration tests for enhanced psycopg error handling implementation.
Tests real-world scenarios and validates outputs to prevent false positives.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from src.prompt_improver.database.psycopg_client import TypeSafePsycopgClient
from src.prompt_improver.database.config import DatabaseConfig
from src.prompt_improver.database.error_handling import (
    RetryConfig, CircuitBreakerConfig, ErrorCategory, ErrorSeverity,
    DatabaseErrorClassifier, ErrorMetrics, CircuitBreakerState
)
from pydantic import BaseModel


class UserModel(BaseModel):
    """Test model for validation."""
    id: int
    name: str
    active: bool = True


class TestEnhancedErrorHandlingIntegration:
    """Integration tests for the complete error handling system."""
    
    @pytest.fixture
    def mock_connection_pool(self):
        """Create a mock connection pool for testing."""
        pool = Mock()
        
        # Create async context manager mocks
        connection_context = AsyncMock()
        connection = AsyncMock()
        connection_context.__aenter__ = AsyncMock(return_value=connection)
        connection_context.__aexit__ = AsyncMock(return_value=None)
        
        cursor_context = AsyncMock()
        cursor = AsyncMock()
        cursor_context.__aenter__ = AsyncMock(return_value=cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=None)
        
        # Setup mock chain
        pool.connection.return_value = connection_context
        connection.cursor.return_value = cursor_context
        
        return pool, connection, cursor
    
    @pytest.fixture
    def test_client_config(self):
        """Create test client configuration."""
        return {
            'config': DatabaseConfig(),
            'retry_config': RetryConfig(
                max_attempts=3,
                base_delay_ms=10,  # Fast for testing
                max_delay_ms=50,
                exponential_base=2.0,
                jitter=False
            ),
            'circuit_breaker_config': CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout_seconds=1,
                success_threshold=2,
                enabled=True
            ),
            'enable_error_metrics': True
        }
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retries(self, mock_connection_pool, test_client_config):
        """Test that successful operations don't trigger retries."""
        pool, connection, cursor = mock_connection_pool
        
        # Mock successful database response
        cursor.execute = AsyncMock()
        cursor.fetchall.return_value = [
            {'id': 1, 'name': 'test1', 'active': True},
            {'id': 2, 'name': 'test2', 'active': False}
        ]
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Execute successful operation
            start_time = time.perf_counter()
            result = await client.fetch_models(
                UserModel, 
                "SELECT id, name, active FROM test_table", 
                {"limit": 10}
            )
            duration = time.perf_counter() - start_time
            
            # Verify results
            assert len(result) == 2
            assert result[0].id == 1
            assert result[0].name == 'test1'
            assert result[1].active is False
            
            # Verify no retries occurred (should be fast)
            assert duration < 0.1
            
            # Verify execute was called exactly once
            cursor.execute.assert_called_once()
            
            # Verify metrics recorded success
            assert client.metrics.total_queries == 1
            assert len(client.metrics.slow_queries) == 0
            
            # Verify no errors in error metrics
            if client.error_metrics:
                summary = client.error_metrics.get_metrics_summary()
                assert summary['total_errors'] == 0
    
    @pytest.mark.asyncio
    async def test_transient_error_retry_success(self, mock_connection_pool, test_client_config):
        """Test that transient errors are retried and eventually succeed."""
        pool, connection, cursor = mock_connection_pool
        
        # Create a mock error with sqlstate for retryable connection failure
        class MockConnectionError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.sqlstate = "08006"  # Connection failure
        
        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 times
                raise MockConnectionError("Connection failed")
            # Succeed on 3rd attempt
            return None
        
        cursor.execute = mock_execute
        cursor.fetchall.return_value = [{'id': 1, 'name': 'success', 'active': True}]
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Execute operation that will fail and retry
            start_time = time.perf_counter()
            result = await client.fetch_models(
                UserModel, 
                "SELECT id, name, active FROM test_table"
            )
            duration = time.perf_counter() - start_time
            
            # Verify eventual success
            assert len(result) == 1
            assert result[0].name == 'success'
            
            # Verify retries occurred (should take longer due to delays)
            assert duration > 0.01  # At least some delay from retries
            
            # Verify execute was called 3 times (2 failures + 1 success)
            assert call_count == 3
            
            # Verify error metrics recorded the retries
            if client.error_metrics:
                summary = client.error_metrics.get_metrics_summary()
                assert summary['total_errors'] == 2  # 2 failed attempts
                assert ErrorCategory.CONNECTION.value in summary['error_counts_by_category']
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(self, mock_connection_pool, test_client_config):
        """Test that non-retryable errors fail immediately without retries."""
        pool, connection, cursor = mock_connection_pool
        
        # Create a mock syntax error (non-retryable)
        class MockSyntaxError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.sqlstate = "42601"  # Syntax error
        
        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise MockSyntaxError("Syntax error in SQL")
        
        cursor.execute = mock_execute
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Execute operation that will fail immediately
            start_time = time.perf_counter()
            with pytest.raises(MockSyntaxError):
                await client.fetch_models(
                    UserModel, 
                    "INVALID SQL SYNTAX"
                )
            duration = time.perf_counter() - start_time
            
            # Verify no retries occurred (should fail fast)
            assert duration < 0.01
            
            # Verify execute was called only once
            assert call_count == 1
            
            # Verify error was classified correctly
            if client.error_metrics:
                summary = client.error_metrics.get_metrics_summary()
                assert summary['total_errors'] == 1
                assert ErrorCategory.SYNTAX.value in summary['error_counts_by_category']
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, mock_connection_pool, test_client_config):
        """Test that circuit breaker opens after consecutive failures."""
        pool, connection, cursor = mock_connection_pool
        
        # Create a mock error that should open circuit breaker
        class MockDatabaseError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.sqlstate = "08006"  # Connection failure
        
        cursor.execute = AsyncMock(side_effect=MockDatabaseError("Database down"))
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Verify circuit breaker starts closed
            assert client.circuit_breaker.state == CircuitBreakerState.CLOSED
            
            # Execute operations to trigger circuit breaker
            failure_threshold = client.circuit_breaker.config.failure_threshold
            
            for i in range(failure_threshold):
                with pytest.raises(MockDatabaseError):
                    await client.fetch_models(UserModel, "SELECT * FROM test")
            
            # Verify circuit breaker opened
            assert client.circuit_breaker.state == CircuitBreakerState.OPEN
            assert client.circuit_breaker.failure_count >= failure_threshold
    
    @pytest.mark.asyncio
    async def test_health_check_comprehensive(self, mock_connection_pool, test_client_config):
        """Test comprehensive health check functionality."""
        pool, connection, cursor = mock_connection_pool
        
        # Mock successful health check responses
        cursor.execute = AsyncMock()
        cursor.fetchone.return_value = [1]  # For basic health check
        cursor.fetchall.return_value = [
            {
                'db_size': 1024000,
                'active_connections': 5,
                'idle_connections': 2,
                'blocked_queries': 0
            }
        ]
        
        # Mock pool stats
        pool_stats = Mock()
        pool_stats.pool_size = 10
        pool_stats.pool_available = 8
        pool_stats.pool_max = 20
        pool_stats.pool_min = 2
        pool_stats.requests_waiting = 0
        pool_stats.requests_errors = 0
        pool_stats.requests_num = 100
        pool_stats.usage_ms = 50
        pool_stats.connections_num = 10
        
        pool.get_stats.return_value = pool_stats
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Execute health check
            health_status = await client.health_check()
            
            # Verify health check structure
            assert 'overall_health' in health_status
            assert 'timestamp' in health_status
            assert 'checks' in health_status
            assert 'error_metrics' in health_status
            assert 'circuit_breaker_status' in health_status
            
            # Verify individual checks
            assert 'connection' in health_status['checks']
            assert 'pool' in health_status['checks']
            assert 'performance' in health_status['checks']
            assert 'server' in health_status['checks']
            
            # Verify connection check
            conn_check = health_status['checks']['connection']
            assert conn_check['status'] == 'HEALTHY'
            assert 'response_time_ms' in conn_check
            
            # Verify pool check
            pool_check = health_status['checks']['pool']
            assert pool_check['status'] == 'HEALTHY'
            assert pool_check['available_connections'] == 8
            assert pool_check['total_connections'] == 10
            
            # Verify overall health is determined correctly
            assert health_status['overall_health'] in ['HEALTHY', 'DEGRADED', 'UNHEALTHY']
    
    @pytest.mark.asyncio
    async def test_connection_test_with_retry(self, mock_connection_pool, test_client_config):
        """Test connection testing with retry functionality."""
        pool, connection, cursor = mock_connection_pool
        
        # Mock successful connection test
        cursor.execute = AsyncMock()
        cursor.fetchone.return_value = {'test': 1, 'timestamp': datetime.utcnow()}
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Execute connection test
            test_result = await client.test_connection_with_retry()
            
            # Verify successful result structure
            assert test_result['status'] == 'SUCCESS'
            assert 'result' in test_result
            assert 'response_time_ms' in test_result
            assert 'retry_count' in test_result
            assert test_result['retry_count'] == 0  # No retries for success
            
            # Verify timing information is reasonable
            assert isinstance(test_result['response_time_ms'], (int, float))
            assert test_result['response_time_ms'] >= 0
    
    @pytest.mark.asyncio
    async def test_error_metrics_accuracy(self, mock_connection_pool, test_client_config):
        """Test error metrics collection accuracy."""
        pool, connection, cursor = mock_connection_pool
        
        # Create different types of errors
        class MockConnectionError(Exception):
            def __init__(self): 
                super().__init__("Connection failed")
                self.sqlstate = "08006"
        
        class MockTimeoutError(Exception):
            def __init__(self): 
                super().__init__("Query timeout")
                self.sqlstate = "57014"
        
        class MockIntegrityError(Exception):
            def __init__(self): 
                super().__init__("Unique violation")
                self.sqlstate = "23505"
        
        errors = [MockConnectionError(), MockTimeoutError(), MockIntegrityError()]
        error_index = 0
        
        async def mock_execute(*args, **kwargs):
            nonlocal error_index
            if error_index < len(errors):
                error = errors[error_index]
                error_index += 1
                raise error
            return None
        
        cursor.execute = mock_execute
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Generate different types of errors
            for _ in range(3):
                try:
                    await client.fetch_models(UserModel, "SELECT * FROM test")
                except Exception:
                    pass  # Expected to fail
            
            # Verify error metrics accuracy
            if client.error_metrics:
                summary = client.error_metrics.get_metrics_summary()
                
                # Should have recorded 3 errors total
                assert summary['total_errors'] == 3
                
                # Should have different categories
                error_counts = summary['error_counts_by_category']
                assert ErrorCategory.CONNECTION.value in error_counts
                assert ErrorCategory.TIMEOUT.value in error_counts
                assert ErrorCategory.INTEGRITY.value in error_counts
                
                # Each category should have exactly 1 error
                assert error_counts[ErrorCategory.CONNECTION.value] == 1
                assert error_counts[ErrorCategory.TIMEOUT.value] == 1
                assert error_counts[ErrorCategory.INTEGRITY.value] == 1
                
                # Verify recent errors list
                assert len(summary['last_errors']) == 3
                
                # Verify error rate calculation
                assert summary['error_rate_per_minute'] > 0
    
    @pytest.mark.asyncio
    async def test_no_false_positive_retries(self, mock_connection_pool, test_client_config):
        """Test that successful operations don't trigger false positive retries."""
        pool, connection, cursor = mock_connection_pool
        
        # Mock consistently successful operations
        execute_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal execute_count
            execute_count += 1
            return None
        
        cursor.execute = mock_execute
        cursor.fetchall.return_value = [{'id': 1, 'name': 'test', 'active': True}]
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Execute multiple successful operations
            for i in range(10):
                result = await client.fetch_models(
                    UserModel, 
                    f"SELECT * FROM test WHERE id = {i}"
                )
                assert len(result) == 1
                assert result[0].name == 'test'
            
            # Verify execute was called exactly 10 times (no retries)
            assert execute_count == 10
            
            # Verify no errors in metrics
            if client.error_metrics:
                summary = client.error_metrics.get_metrics_summary()
                assert summary['total_errors'] == 0
            
            # Verify circuit breaker remains closed
            assert client.circuit_breaker.state == CircuitBreakerState.CLOSED
            assert client.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_accuracy(self, mock_connection_pool, test_client_config):
        """Test that performance metrics are accurately recorded."""
        pool, connection, cursor = mock_connection_pool
        
        # Mock operations with controlled timing
        async def mock_execute(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate 10ms query time
            return None
        
        cursor.execute = mock_execute
        cursor.fetchall.return_value = [{'id': 1, 'name': 'test', 'active': True}]
        
        # Mock pool stats for performance check
        pool_stats = Mock()
        pool_stats.pool_size = 10
        pool_stats.pool_available = 8
        pool_stats.pool_max = 20
        pool_stats.pool_min = 2
        pool_stats.requests_waiting = 0
        pool_stats.requests_errors = 0
        pool_stats.requests_num = 5
        pool_stats.usage_ms = 50
        pool_stats.connections_num = 10
        pool.get_stats.return_value = pool_stats
        
        with patch('src.prompt_improver.database.psycopg_client.AsyncConnectionPool', return_value=pool):
            client = TypeSafePsycopgClient(**test_client_config)
            client.pool = pool
            
            # Execute operations
            for _ in range(5):
                await client.fetch_models(UserModel, "SELECT * FROM test")
            
            # Get performance statistics
            perf_stats = await client.get_performance_stats()
            
            # Verify metrics accuracy
            assert perf_stats['total_queries'] == 5
            assert perf_stats['avg_query_time_ms'] >= 10  # Should be at least 10ms
            assert 'queries_under_50ms_percent' in perf_stats
            assert 'slow_query_count' in perf_stats
            assert 'pool_status' in perf_stats
            
            # Verify pool utilization calculation
            pool_status = perf_stats['pool_status']
            assert pool_status['pool_utilization'] == 20.0  # (10-8)/10 * 100


@pytest.mark.asyncio
async def test_error_classification_validation():
    """Test error classification with real-world scenarios."""
    
    # Test known SQLSTATE codes
    test_cases = [
        # (sqlstate, expected_category, expected_severity, expected_retryable)
        ("08006", ErrorCategory.CONNECTION, ErrorSeverity.HIGH, True),
        ("40P01", ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM, True),
        ("57014", ErrorCategory.TIMEOUT, ErrorSeverity.HIGH, True),
        ("23505", ErrorCategory.INTEGRITY, ErrorSeverity.MEDIUM, False),
        ("42601", ErrorCategory.SYNTAX, ErrorSeverity.LOW, False),
        ("53300", ErrorCategory.RESOURCE, ErrorSeverity.HIGH, True),
    ]
    
    for sqlstate, expected_category, expected_severity, expected_retryable in test_cases:
        # Create mock error with sqlstate
        class MockError(Exception):
            def __init__(self, sqlstate):
                super().__init__(f"Mock error with sqlstate {sqlstate}")
                self.sqlstate = sqlstate
        
        error = MockError(sqlstate)
        
        # Test classification
        category, severity = DatabaseErrorClassifier.classify_error(error)
        is_retryable = DatabaseErrorClassifier.is_retryable(error)
        
        # Verify results
        assert category == expected_category, f"SQLSTATE {sqlstate}: Expected {expected_category}, got {category}"
        assert severity == expected_severity, f"SQLSTATE {sqlstate}: Expected {expected_severity}, got {severity}"
        assert is_retryable == expected_retryable, f"SQLSTATE {sqlstate}: Expected retryable={expected_retryable}, got {is_retryable}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 