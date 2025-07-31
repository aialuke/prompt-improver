# PostgreSQL Error Handling Best Practices 2025 (AsyncPG)

## Overview

This document outlines the comprehensive error handling implementation for PostgreSQL database operations using asyncpg, incorporating 2025 industry best practices for reliability, observability, and fault tolerance.

## Key Features

### 🔧 **Enhanced Error Classification**
- **SQLSTATE-based categorization**: Automatic error classification using PostgreSQL SQLSTATE codes
- **Severity assessment**: Errors are classified by severity (LOW, MEDIUM, HIGH, CRITICAL)
- **Smart fallback**: Handles both real asyncpg errors and test mocks gracefully

### 🔄 **Advanced Retry Mechanisms**
- **Exponential backoff**: Configurable delays with jitter to prevent thundering herd
- **Retryability detection**: Intelligent determination of which errors are worth retrying
- **Context preservation**: Full error context maintained across retry attempts

### ⚡ **Circuit Breaker Pattern**
- **Fault tolerance**: Prevents cascading failures in distributed systems
- **Automatic recovery**: Smart detection of service restoration
- **Configurable thresholds**: Customizable failure rates and timeouts

### 📊 **Comprehensive Observability**
- **Structured logging**: JSON-formatted logs with full context
- **Metrics integration**: OpenTelemetry metrics for monitoring
- **Error tracking**: Detailed error categorization and trending

## Error Classification System

### Error Categories

```python
class ErrorCategory(Enum):
    connection = "connection"      # Network/connection issues
    timeout = "timeout"           # Query timeouts
    transient = "transient"       # Temporary failures (deadlocks, etc.)
    constraint = "constraint"     # Data integrity violations
    syntax = "syntax"            # SQL syntax errors
    permission = "permission"     # Access control issues
    resource = "resource"        # System resource exhaustion
```

### AsyncPG Error Mappings

```python
ERROR_MAPPINGS = {
    # Connection errors
    asyncpg.ConnectionDoesNotExistError: (ErrorCategory.connection, ErrorSeverity.high),
    asyncpg.ConnectionFailureError: (ErrorCategory.connection, ErrorSeverity.high),
    asyncpg.InterfaceError: (ErrorCategory.connection, ErrorSeverity.high),

    # Timeout errors
    asyncio.CancelledError: (ErrorCategory.timeout, ErrorSeverity.medium),

    # Transient errors
    asyncpg.PostgresError: (ErrorCategory.transient, ErrorSeverity.medium),

    # Constraint violations
    asyncpg.IntegrityConstraintViolationError: (ErrorCategory.constraint, ErrorSeverity.low),
    asyncpg.UniqueViolationError: (ErrorCategory.constraint, ErrorSeverity.low),
    asyncpg.ForeignKeyViolationError: (ErrorCategory.constraint, ErrorSeverity.low),

    # Syntax errors
    asyncpg.SyntaxOrAccessError: (ErrorCategory.syntax, ErrorSeverity.high),
    asyncpg.UndefinedTableError: (ErrorCategory.syntax, ErrorSeverity.high),
    asyncpg.UndefinedColumnError: (ErrorCategory.syntax, ErrorSeverity.high),

    # Permission errors
    asyncpg.InsufficientPrivilegeError: (ErrorCategory.permission, ErrorSeverity.high),
}
```

## Usage Examples

### Basic Error Handling

```python
from prompt_improver.database import get_unified_manager, ManagerMode
from prompt_improver.database.error_handling import DatabaseErrorClassifier

async def safe_database_operation():
    """Example of safe database operation with error handling."""
    manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    
    try:
        async with manager.get_session() as session:
            result = await session.execute("SELECT * FROM rules WHERE active = true")
            return result.fetchall()
            
    except Exception as e:
        category, severity = DatabaseErrorClassifier.classify_error(e)
        
        if category == ErrorCategory.transient:
            # Retry transient errors
            logger.warning(f"Transient error detected: {e}")
            # Implement retry logic here
            
        elif category == ErrorCategory.connection:
            # Handle connection issues
            logger.error(f"Connection error: {e}")
            # Implement connection recovery
            
        else:
            # Log and re-raise other errors
            logger.error(f"Database error [{category.value}]: {e}")
            raise
```

### Integration with UnifiedConnectionManager

```python
from prompt_improver.database import get_unified_manager, ManagerMode

async def robust_query_execution():
    """Execute queries with built-in error handling and retry logic."""
    manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    
    # The UnifiedConnectionManager automatically handles:
    # - Connection pooling
    # - Retry logic for transient errors
    # - Circuit breaker patterns
    # - Health monitoring
    
    async with manager.get_session() as session:
        # This query benefits from all error handling features
        result = await session.execute(
            "SELECT id, content FROM rules WHERE category = :category",
            {"category": "clarity"}
        )
        return result.fetchall()
```

## Best Practices

### ✅ **Do:**
- Use the UnifiedConnectionManager for all database operations
- Classify errors appropriately for retry decisions
- Log errors with sufficient context for debugging
- Monitor error rates and patterns
- Test error scenarios with comprehensive test coverage

### ❌ **Don't:**
- Ignore transient errors that could be retried
- Retry non-retryable errors (syntax, permission)
- Log sensitive data in error messages
- Use bare except clauses without error classification
- Assume all PostgresError exceptions are the same

## Migration from psycopg

### Key Differences

1. **Exception Hierarchy**: AsyncPG has a different exception structure
2. **Connection Management**: AsyncPG uses connection pools differently
3. **Query Execution**: AsyncPG has different methods for query execution
4. **Type Handling**: AsyncPG handles PostgreSQL types differently

### Migration Checklist

- [ ] Replace psycopg imports with asyncpg
- [ ] Update error handling to use asyncpg exceptions
- [ ] Migrate connection strings from `postgresql+psycopg://` to `postgresql+asyncpg://`
- [ ] Update query execution patterns
- [ ] Test all error handling scenarios
- [ ] Update monitoring and logging

## Performance Considerations

### AsyncPG Advantages

- **Native async/await**: Built for Python's asyncio
- **Better performance**: Generally faster than psycopg for async operations
- **Type safety**: Better PostgreSQL type handling
- **Connection pooling**: Efficient built-in pooling

### Monitoring

```python
# Error metrics are automatically collected
from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry

metrics = get_metrics_registry()
# Metrics include:
# - database_errors_total (by category, severity)
# - database_retry_attempts_total
# - database_circuit_breaker_state
# - database_connection_pool_utilization
```

## Testing Error Scenarios

```python
import pytest
import asyncpg
from prompt_improver.database.error_handling import DatabaseErrorClassifier

@pytest.mark.asyncio
async def test_connection_error_handling():
    """Test handling of connection errors."""
    # Simulate connection failure
    with pytest.raises(asyncpg.ConnectionFailureError):
        await asyncpg.connect("postgresql://invalid:invalid@nonexistent:5432/test")

@pytest.mark.asyncio
async def test_constraint_violation_handling():
    """Test handling of constraint violations."""
    # Test unique constraint violation
    # Implementation depends on your test setup
    pass
```

## Conclusion

The asyncpg-based error handling system provides robust, observable, and maintainable database operations. By following these patterns and best practices, you can build resilient applications that gracefully handle database errors and provide excellent observability for production operations.

For more information, see:
- [AsyncPG Documentation](https://magicstack.github.io/asyncpg/)
- [UnifiedConnectionManager Documentation](../database/UNIFIED_CONNECTION_MANAGER.md)
- [OpenTelemetry Integration](../monitoring/OPENTELEMETRY_SETUP.md)
