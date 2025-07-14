# PostgreSQL Error Handling Best Practices 2025

## Overview

This document outlines the comprehensive error handling implementation for PostgreSQL database operations using psycopg3, incorporating 2025 industry best practices for reliability, observability, and fault tolerance.

## Key Features

### 🔧 **Enhanced Error Classification**
- **SQLSTATE-based categorization**: Automatic error classification using PostgreSQL SQLSTATE codes
- **Severity assessment**: Errors are classified by severity (LOW, MEDIUM, HIGH, CRITICAL)
- **Smart fallback**: Handles both real psycopg errors and test mocks gracefully

### 🔄 **Advanced Retry Mechanisms**
- **Exponential backoff**: Configurable delays with jitter to prevent thundering herd
- **Retryability detection**: Intelligent determination of which errors are worth retrying
- **Context preservation**: Full error context maintained across retry attempts

### ⚡ **Circuit Breaker Pattern**
- **Fault tolerance**: Prevents cascading failures in distributed systems
- **State management**: CLOSED → OPEN → HALF_OPEN state transitions
- **Recovery testing**: Automatic recovery detection and gradual restoration

### 📊 **Comprehensive Metrics**
- **Error tracking**: Detailed error counts by category and type
- **Performance monitoring**: Response times and retry counts
- **Health insights**: Circuit breaker status and connection health

## Implementation Architecture

### Error Categories

```python
class ErrorCategory(Enum):
    CONNECTION = "connection"      # Network/connection issues
    TIMEOUT = "timeout"           # Query or connection timeouts
    INTEGRITY = "integrity"       # Data integrity violations
    PERMISSION = "permission"     # Authorization failures
    SYNTAX = "syntax"            # SQL syntax errors
    DATA = "data"                # Data validation errors
    RESOURCE = "resource"        # Resource exhaustion
    TRANSIENT = "transient"      # Temporary/recoverable errors
    FATAL = "fatal"              # Unrecoverable errors
```

### Error Severity Levels

```python
class ErrorSeverity(Enum):
    LOW = "low"                  # Minor issues, log only
    MEDIUM = "medium"            # Noticeable but recoverable
    HIGH = "high"                # Significant issues requiring attention
    CRITICAL = "critical"        # Service-affecting, immediate action needed
```

## Configuration

### Retry Configuration

```python
retry_config = RetryConfig(
    max_attempts=3,              # Maximum retry attempts
    base_delay_ms=100,           # Initial delay between retries
    max_delay_ms=10000,          # Maximum delay cap
    exponential_base=2.0,        # Exponential backoff multiplier
    jitter=True                  # Add randomization to prevent thundering herd
)
```

### Circuit Breaker Configuration

```python
circuit_breaker_config = CircuitBreakerConfig(
    failure_threshold=5,         # Failures before opening circuit
    recovery_timeout_seconds=60, # Time before testing recovery
    success_threshold=3,         # Successes needed to close circuit
    enabled=True                 # Enable/disable circuit breaker
)
```

## Usage Examples

### Basic Client Setup

```python
from src.prompt_improver.database.psycopg_client import TypeSafePsycopgClient
from src.prompt_improver.database.error_handling import RetryConfig, CircuitBreakerConfig

# Create client with enhanced error handling
client = TypeSafePsycopgClient(
    config=database_config,
    retry_config=RetryConfig(max_attempts=5, base_delay_ms=200),
    circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
    enable_error_metrics=True
)
```

### Database Operations with Auto-Retry

```python
# Fetch models with automatic retry for transient errors
try:
    users = await client.fetch_models(User, "SELECT * FROM users WHERE active = %(active)s", {"active": True})
except psycopg.OperationalError as e:
    # Error was automatically retried based on classification
    logger.error(f"Database operation failed after retries: {e}")
```

### Health Monitoring

```python
# Comprehensive health check with error metrics
health_status = await client.health_check()
print(f"Database health: {health_status['overall_health']}")
print(f"Circuit breaker state: {health_status['circuit_breaker_status']['state']}")
print(f"Recent errors: {health_status['error_metrics']['total_errors']}")
```

### Connection Testing with Retry

```python
# Test connection with detailed error reporting
test_result = await client.test_connection_with_retry()
if test_result['status'] == 'SUCCESS':
    print(f"Connection OK (took {test_result['response_time_ms']}ms)")
else:
    print(f"Connection failed: {test_result['error']} ({test_result['error_category']})")
```

## Error Classification Details

### SQLSTATE Mapping

The system automatically maps PostgreSQL SQLSTATE codes to appropriate categories:

| SQLSTATE | Category | Severity | Retryable | Description |
|----------|----------|----------|-----------|-------------|
| 08006 | CONNECTION | HIGH | ✅ | Connection failure |
| 40P01 | TRANSIENT | MEDIUM | ✅ | Deadlock detected |
| 57014 | TIMEOUT | HIGH | ✅ | Query canceled |
| 23505 | INTEGRITY | MEDIUM | ❌ | Unique violation |
| 42601 | SYNTAX | LOW | ❌ | Syntax error |
| 53300 | RESOURCE | HIGH | ✅ | Too many connections |

### Retryability Logic

Errors are considered retryable based on:

1. **SQLSTATE classification**: Specific codes known to be transient
2. **Category-based rules**: CONNECTION, TIMEOUT, and TRANSIENT categories
3. **Message pattern matching**: For mock testing and edge cases
4. **Exception type checking**: psycopg-specific error types

## Monitoring and Observability

### Error Metrics

```python
# Get comprehensive error metrics
metrics = client.get_error_metrics_summary()
print(f"Total errors: {metrics['total_errors']}")
print(f"Error rate: {metrics['error_rate_per_minute']}")
print(f"Most common errors: {metrics['error_counts_by_category']}")
```

### Circuit Breaker Status

```python
# Monitor circuit breaker health
cb_status = client.get_circuit_breaker_status()
print(f"State: {cb_status['state']}")
print(f"Failure count: {cb_status['failure_count']}")
print(f"Success count: {cb_status['success_count']}")
```

## Testing Strategy

### Mock Error Testing

The system includes comprehensive test coverage with mock errors that simulate real psycopg behavior:

```python
# Example test setup
class MockPsycopgError(Exception):
    def __init__(self, message: str, sqlstate: str = None):
        super().__init__(message)
        self.sqlstate = sqlstate

# Test retryable connection error
conn_error = MockPsycopgError("Connection failed", sqlstate="08006")
assert DatabaseErrorClassifier.is_retryable(conn_error)
```

### Integration Testing

Run the comprehensive test suite:

```bash
python3 -m pytest tests/unit/database/test_enhanced_error_handling.py -v
```

## Performance Considerations

### Retry Impact

- **Exponential backoff**: Prevents overwhelming failed services
- **Jitter**: Distributes retry attempts to avoid synchronized thundering herd
- **Early termination**: Non-retryable errors fail fast without delays

### Circuit Breaker Benefits

- **Fast failure**: Open circuit prevents wasted resources on failed services
- **Gradual recovery**: Half-open state tests service recovery safely
- **Configurable thresholds**: Adapt to service-specific failure patterns

### Memory Management

- **Bounded error history**: Error metrics maintain fixed-size recent error lists
- **Efficient categorization**: Fast SQLSTATE lookup using sets
- **Context cleanup**: Error contexts are garbage collected after use

## Production Deployment

### Environment Configuration

```yaml
# config/database_config.yaml
retry_config:
  max_attempts: 5
  base_delay_ms: 200
  max_delay_ms: 30000
  exponential_base: 2.0
  jitter: true

circuit_breaker_config:
  failure_threshold: 10
  recovery_timeout_seconds: 120
  success_threshold: 5
  enabled: true
```

### Monitoring Setup

1. **Error rate alerts**: Monitor error categories and rates
2. **Circuit breaker alerts**: Alert when circuits open
3. **Connection health**: Track connection response times
4. **Retry patterns**: Monitor retry frequency and success rates

### Logging Configuration

```python
import logging
logging.getLogger('src.prompt_improver.database.error_handling').setLevel(logging.INFO)
```

## Migration Guide

### From Basic Error Handling

Replace simple try/catch blocks:

```python
# Before: Basic error handling
try:
    result = await cursor.execute(query)
except Exception as e:
    logger.error(f"Query failed: {e}")
    raise

# After: Enhanced error handling
# Error handling is automatic with TypeSafePsycopgClient
result = await client.fetch_models(Model, query, params)
```

### Updating Existing Code

1. Replace direct psycopg usage with `TypeSafePsycopgClient`
2. Configure retry and circuit breaker settings
3. Add health check endpoints
4. Update monitoring dashboards to include new metrics

## Best Practices Summary

✅ **Do:**
- Use the enhanced client for all database operations
- Configure retry settings based on your service requirements
- Monitor error metrics and circuit breaker status
- Test error scenarios with comprehensive test coverage
- Set appropriate logging levels for production

❌ **Don't:**
- Bypass the error handling by using raw psycopg connections
- Set overly aggressive retry configurations that could overwhelm services
- Ignore circuit breaker open states in your application logic
- Rely on error handling as a substitute for proper query optimization

## Support and Troubleshooting

### Common Issues

1. **High retry rates**: Check for systemic database issues
2. **Circuit breaker stuck open**: Verify underlying service health
3. **Memory usage from error tracking**: Adjust error history limits
4. **Test failures**: Ensure mock errors have proper sqlstate attributes

### Debug Information

Enable detailed logging to troubleshoot issues:

```python
logging.getLogger('src.prompt_improver.database').setLevel(logging.DEBUG)
```

For additional support, refer to the test suite examples and implementation details in:
- `src/prompt_improver/database/error_handling.py`
- `tests/unit/database/test_enhanced_error_handling.py` 