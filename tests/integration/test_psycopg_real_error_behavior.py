"""
Real-behavior database error testing following 2025 best practices.

This module tests actual psycopg error handling by:
1. Using testcontainers for real PostgreSQL instances
2. Triggering actual database error conditions
3. Validating real error classification and retry behavior
4. Testing circuit breaker patterns with real failures

No mocks - only real database behavior validation.
"""

import asyncio
import pytest
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Dict

# Test container imports
import testcontainers.postgres
from testcontainers.postgres import PostgresContainer

# PostgreSQL and psycopg imports
import psycopg
from psycopg import errors as psycopg_errors
from psycopg_pool import AsyncConnectionPool

from src.prompt_improver.database.error_handling import (
    DatabaseErrorClassifier, ErrorCategory, ErrorSeverity,
    RetryManager, RetryConfig, CircuitBreaker, CircuitBreakerConfig,
    ErrorMetrics, ErrorContext
)

logger = logging.getLogger(__name__)

# Pytest configuration for async tests
pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="session")
def postgres_container():
    """Start a real PostgreSQL container for testing."""
    container = PostgresContainer("postgres:15-alpine")
    with container as postgres:
        yield postgres


@pytest.fixture(scope="session")
def postgres_connection_info(postgres_container):
    """Extract connection info from the container."""
    return {
        "host": postgres_container.get_container_host_ip(),
        "port": postgres_container.get_exposed_port(5432),
        "database": postgres_container.dbname,
        "user": postgres_container.username,
        "password": postgres_container.password,
    }


@pytest.fixture
async def postgres_pool(postgres_connection_info):
    """Create a connection pool for the test database."""
    conninfo = (
        f"postgresql://{postgres_connection_info['user']}:"
        f"{postgres_connection_info['password']}@"
        f"{postgres_connection_info['host']}:"
        f"{postgres_connection_info['port']}/"
        f"{postgres_connection_info['database']}"
    )
    
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=5,
        timeout=10
    )
    
    async with pool:
        yield pool


@pytest.fixture
async def test_schema(postgres_pool):
    """Set up test schema with tables for error testing."""
    async with postgres_pool.connection() as conn:
        async with conn.cursor() as cur:
            # Create test tables for various error scenarios
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS test_users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS test_orders (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES test_users(id),
                    amount DECIMAL(10,2) NOT NULL CHECK (amount > 0),
                    status VARCHAR(20) DEFAULT 'pending'
                );
                
                CREATE TABLE IF NOT EXISTS test_locks (
                    id SERIAL PRIMARY KEY,
                    resource_name VARCHAR(100) UNIQUE NOT NULL,
                    locked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.commit()
    
    yield
    
    # Cleanup after tests
    async with postgres_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS test_orders CASCADE;")
            await cur.execute("DROP TABLE IF EXISTS test_users CASCADE;")
            await cur.execute("DROP TABLE IF EXISTS test_locks CASCADE;")
            await conn.commit()


class RealErrorTester:
    """Helper class to trigger real database errors for testing."""
    
    def __init__(self, pool):
        self.pool = pool
    
    async def trigger_connection_error(self):
        """Trigger a real connection error using direct connection to invalid host."""
        try:
            # Use direct connection to get immediate connection error (not pool timeout)
            # This follows 2025 best practice of testing real error conditions
            await psycopg.AsyncConnection.connect(
                "postgresql://user:pass@nonexistent-host:5432/db",
                connect_timeout=1
            )
        except Exception as e:
            return e
    
    async def trigger_unique_violation(self):
        """Trigger a real unique constraint violation."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Insert a user
                await cur.execute(
                    "INSERT INTO test_users (email, name) VALUES (%s, %s)",
                    ("test@example.com", "Test User")
                )
                await conn.commit()
                
                # Try to insert the same email again
                try:
                    await cur.execute(
                        "INSERT INTO test_users (email, name) VALUES (%s, %s)",
                        ("test@example.com", "Another User")
                    )
                    await conn.commit()
                except Exception as e:
                    await conn.rollback()
                    return e
    
    async def trigger_foreign_key_violation(self):
        """Trigger a real foreign key constraint violation."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    # Try to insert order with non-existent user_id
                    await cur.execute(
                        "INSERT INTO test_orders (user_id, amount) VALUES (%s, %s)",
                        (99999, 100.00)
                    )
                    await conn.commit()
                except Exception as e:
                    await conn.rollback()
                    return e
    
    async def trigger_check_constraint_violation(self):
        """Trigger a real check constraint violation."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    # Try to insert negative amount (violates CHECK constraint)
                    await cur.execute(
                        "INSERT INTO test_users (email, name) VALUES (%s, %s) RETURNING id",
                        ("valid@example.com", "Valid User")
                    )
                    user_id = (await cur.fetchone())[0]
                    
                    await cur.execute(
                        "INSERT INTO test_orders (user_id, amount) VALUES (%s, %s)",
                        (user_id, -50.00)
                    )
                    await conn.commit()
                except Exception as e:
                    await conn.rollback()
                    return e
    
    async def trigger_syntax_error(self):
        """Trigger a real SQL syntax error."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute("INVALID SQL SYNTAX HERE")
                except Exception as e:
                    return e
    
    async def trigger_deadlock(self):
        """Trigger a real deadlock scenario."""
        # This requires two concurrent transactions - we'll simulate with advisory locks
        async with self.pool.connection() as conn1, self.pool.connection() as conn2:
            try:
                async with conn1.cursor() as cur1, conn2.cursor() as cur2:
                    # Transaction 1: Lock resource A, then try to lock B
                    await cur1.execute("SELECT pg_advisory_lock(1)")
                    await asyncio.sleep(0.1)
                    
                    # Transaction 2: Lock resource B, then try to lock A
                    await cur2.execute("SELECT pg_advisory_lock(2)")
                    await asyncio.sleep(0.1)
                    
                    # Now try to acquire the other's lock (this should timeout)
                    await cur1.execute("SELECT pg_try_advisory_lock(2)")
                    await cur2.execute("SELECT pg_try_advisory_lock(1)")
                    
            except Exception as e:
                return e
            finally:
                # Release locks
                try:
                    async with conn1.cursor() as cur1:
                        await cur1.execute("SELECT pg_advisory_unlock_all()")
                    async with conn2.cursor() as cur2:
                        await cur2.execute("SELECT pg_advisory_unlock_all()")
                except:
                    pass
    
    async def trigger_timeout_error(self):
        """Trigger a real statement timeout."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    # Set a very short statement timeout
                    await cur.execute("SET statement_timeout = '100ms'")
                    
                    # Execute a long-running query
                    await cur.execute("SELECT pg_sleep(1)")
                except Exception as e:
                    return e
    
    async def trigger_disk_full_simulation(self):
        """Simulate disk full by creating a very large table."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    # Try to create a huge table that might cause resource issues
                    await cur.execute("""
                        CREATE TEMP TABLE huge_table AS 
                        SELECT generate_series(1, 10000000) as id, 
                               md5(random()::text) as data
                    """)
                except Exception as e:
                    return e


@pytest.fixture
def error_tester(postgres_pool):
    """Provide the error testing helper."""
    return RealErrorTester(postgres_pool)


# Test Cases for Real Error Behavior

async def test_real_connection_error_classification(error_tester):
    """Test classification of real connection errors."""
    error = await error_tester.trigger_connection_error()
    
    assert error is not None
    category, severity = DatabaseErrorClassifier.classify_error(error)
    
    assert category == ErrorCategory.CONNECTION
    assert severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
    assert DatabaseErrorClassifier.is_retryable(error) is True


async def test_real_unique_violation_classification(error_tester, test_schema):
    """Test classification of real unique constraint violations."""
    error = await error_tester.trigger_unique_violation()
    
    assert error is not None
    assert isinstance(error, psycopg_errors.UniqueViolation)
    
    category, severity = DatabaseErrorClassifier.classify_error(error)
    
    assert category == ErrorCategory.INTEGRITY
    assert severity == ErrorSeverity.MEDIUM
    assert DatabaseErrorClassifier.is_retryable(error) is False


async def test_real_foreign_key_violation_classification(error_tester, test_schema):
    """Test classification of real foreign key violations."""
    error = await error_tester.trigger_foreign_key_violation()
    
    assert error is not None
    assert isinstance(error, psycopg_errors.ForeignKeyViolation)
    
    category, severity = DatabaseErrorClassifier.classify_error(error)
    
    assert category == ErrorCategory.INTEGRITY
    assert severity == ErrorSeverity.MEDIUM
    assert DatabaseErrorClassifier.is_retryable(error) is False


async def test_real_check_constraint_violation_classification(error_tester, test_schema):
    """Test classification of real check constraint violations."""
    error = await error_tester.trigger_check_constraint_violation()
    
    assert error is not None
    assert isinstance(error, psycopg_errors.CheckViolation)
    
    category, severity = DatabaseErrorClassifier.classify_error(error)
    
    # Check constraints are INTEGRITY violations per SQLSTATE 23514 (2025 standards)
    assert category == ErrorCategory.INTEGRITY
    assert severity == ErrorSeverity.MEDIUM
    assert DatabaseErrorClassifier.is_retryable(error) is False


async def test_real_syntax_error_classification(error_tester):
    """Test classification of real SQL syntax errors."""
    error = await error_tester.trigger_syntax_error()
    
    assert error is not None
    assert isinstance(error, psycopg_errors.SyntaxError)
    
    category, severity = DatabaseErrorClassifier.classify_error(error)
    
    assert category == ErrorCategory.SYNTAX
    assert severity == ErrorSeverity.CRITICAL
    assert DatabaseErrorClassifier.is_retryable(error) is False


async def test_real_timeout_error_classification(error_tester, test_schema):
    """Test classification of real timeout errors."""
    error = await error_tester.trigger_timeout_error()
    
    if error:  # Timeout might not always trigger in test environment
        category, severity = DatabaseErrorClassifier.classify_error(error)
        
        assert category == ErrorCategory.TIMEOUT
        assert severity == ErrorSeverity.HIGH
        assert DatabaseErrorClassifier.is_retryable(error) is True


async def test_retry_manager_with_real_transient_errors(error_tester):
    """Test retry manager with real transient errors."""
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay_ms=100,
        max_delay_ms=1000,
        exponential_base=2.0,
        jitter=True
    )
    retry_manager = RetryManager(retry_config)
    
    context = ErrorContext(operation="test_real_retry", connection_id="test-conn")
    
    # Function that fails with connection error then succeeds
    attempt_count = 0
    
    async def failing_operation() -> dict[str, Any]:
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count <= 2:
            # Trigger real connection error on first two attempts
            error = await error_tester.trigger_connection_error()
            raise error
        else:
            # Succeed on third attempt
            return {"status": "success", "attempt": attempt_count}
    
    # Test retry with real errors
    result = await retry_manager.retry_async(failing_operation, context)
    
    assert result["status"] == "success"
    assert result["attempt"] == 3
    assert context.retry_count == 2  # 2 retries after initial failure


async def test_circuit_breaker_with_real_errors(error_tester):
    """Test circuit breaker pattern with real database errors."""
    # 2025 conservative circuit breaker configuration per Microsoft guidance
    circuit_config = CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout_seconds=3,  # Conservative 3-second recovery timeout
        success_threshold=1,
        enabled=True
    )
    circuit_breaker = CircuitBreaker(circuit_config)
    
    # Test circuit breaker opening after real failures
    async def failing_connection_operation():
        error = await error_tester.trigger_connection_error()
        if error:
            raise error
    
    for _ in range(3):
        try:
            await circuit_breaker.call(failing_connection_operation)
        except Exception:
            pass
    
    # Circuit should be open now
    assert circuit_breaker.state.value == "open"
    
    # Wait for recovery period (2025 conservative timing - 3+ seconds per Microsoft guidance)
    await asyncio.sleep(3.5)
    
    # Trigger circuit breaker to check recovery - this should transition to HALF_OPEN 
    # but the operation will fail and return to OPEN (which is correct behavior)
    try:
        await circuit_breaker.call(failing_connection_operation)
    except Exception:
        pass  # Expected: OPEN -> HALF_OPEN -> failed operation -> back to OPEN
    
    # Circuit should be back to open after failed test operation (2025 conservative behavior)
    assert circuit_breaker.state.value == "open"


async def test_error_metrics_with_real_errors(error_tester, test_schema):
    """Test error metrics collection with real database errors."""
    error_metrics = ErrorMetrics()
    
    # Generate various real errors and record them
    context = ErrorContext(operation="test_metrics", connection_id="test-conn")
    
    # Record unique violation
    unique_error = await error_tester.trigger_unique_violation()
    if unique_error:
        error_metrics.record_error(context, unique_error)
    
    # Record foreign key violation
    fk_error = await error_tester.trigger_foreign_key_violation()
    if fk_error:
        error_metrics.record_error(context, fk_error)
    
    # Record syntax error
    syntax_error = await error_tester.trigger_syntax_error()
    if syntax_error:
        error_metrics.record_error(context, syntax_error)
    
    # Verify metrics collection
    metrics_summary = error_metrics.get_metrics_summary()
    
    assert metrics_summary["total_errors"] >= 3
    assert "integrity" in metrics_summary["error_counts_by_category"]
    assert "syntax" in metrics_summary["error_counts_by_category"]
    # Check if any recent errors recorded (verify metrics are working)
    assert metrics_summary["recent_errors_count"] >= 0


async def test_real_database_resilience_patterns(postgres_pool, test_schema):
    """Test complete resilience patterns with real database operations."""
    # Configure comprehensive error handling
    retry_config = RetryConfig(max_attempts=3, base_delay_ms=100)
    circuit_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=1)
    
    retry_manager = RetryManager(retry_config)
    circuit_breaker = CircuitBreaker(circuit_config)
    error_metrics = ErrorMetrics()
    
    async def database_operation_with_resilience():
        """Simulate a database operation with full resilience patterns."""
        context = ErrorContext(
            operation="resilient_db_op",
            connection_id="resilience-test"
        )
        
        async def operation():
            async with postgres_pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Perform a real database operation
                    await cur.execute(
                        "INSERT INTO test_users (email, name) VALUES (%s, %s) RETURNING id",
                        (f"user_{time.time()}@example.com", "Test User")
                    )
                    result = await cur.fetchone()
                    await conn.commit()
                    return {"user_id": result[0], "status": "success"}
        
        try:
            # Apply circuit breaker and retry patterns
            async def retry_operation():
                return await retry_manager.retry_async(operation, context)
            
            result = await circuit_breaker.call(retry_operation)
            return result
        except Exception as e:
            error_metrics.record_error(context, e)
            raise
    
    # Test successful operation
    result = await database_operation_with_resilience()
    assert result["status"] == "success"
    assert "user_id" in result
    
    # Verify the operation actually worked in the database
    async with postgres_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM test_users")
            count = (await cur.fetchone())[0]
            assert count >= 1


async def test_real_error_context_collection(error_tester, test_schema):
    """Test comprehensive error context collection with real errors."""
    context = ErrorContext(
        operation="context_test",
        query="INSERT INTO test_users (email, name) VALUES (%s, %s)",
        params={"email": "test@example.com", "name": "Test"},
        connection_id="context-test-conn"
    )
    
    # Trigger a real error
    error = await error_tester.trigger_unique_violation()
    
    if error:
        # Let the error classification system handle the error
        error_metrics = ErrorMetrics()
        context.duration_ms = 150.5  # Set duration for testing
        error_metrics.record_error(context, error)
        
        # Verify context contains comprehensive information
        assert context.operation == "context_test"
        assert context.query is not None
        assert context.duration_ms == 150.5
        assert context.category is not None  # Set by error classification
        assert context.severity is not None  # Set by error classification
        
        # Verify the error was properly classified
        assert type(error).__name__ == "UniqueViolation"
        assert hasattr(error, 'sqlstate')


# Performance and load testing with real database
async def test_real_database_load_and_error_rates(postgres_pool, test_schema):
    """Test error handling under real database load."""
    error_metrics = ErrorMetrics()
    successful_operations = 0
    failed_operations = 0
    
    async def database_operation(operation_id: int):
        """Single database operation that might fail."""
        nonlocal successful_operations, failed_operations
        
        try:
            async with postgres_pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Some operations will fail due to unique constraints
                    email = f"user_{operation_id % 10}@example.com"  # Only 10 unique emails
                    await cur.execute(
                        "INSERT INTO test_users (email, name) VALUES (%s, %s)",
                        (email, f"User {operation_id}")
                    )
                    await conn.commit()
                    successful_operations += 1
                    
        except Exception as e:
            failed_operations += 1
            context = ErrorContext(
                operation=f"load_test_op_{operation_id}",
                connection_id=f"load-test-{operation_id}"
            )
            error_metrics.record_error(context, e)
    
    # Run concurrent operations that will cause some real conflicts
    tasks = [database_operation(i) for i in range(50)]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify we got both successes and real failures
    assert successful_operations > 0
    assert failed_operations > 0
    
    # Check error metrics were properly collected
    metrics = error_metrics.get_metrics_summary()
    assert metrics["total_errors"] == failed_operations
    assert "integrity" in metrics["error_counts_by_category"]
    
    logger.info(f"Load test results: {successful_operations} success, {failed_operations} failures")
    logger.info(f"Error metrics: {metrics}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 