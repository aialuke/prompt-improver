# AdvancedPatternDiscovery Initialization Guide

## Overview

This guide documents the proper initialization of the `AdvancedPatternDiscovery` service and its integration with the `AprioriAnalyzer` component, following 2025 dependency injection best practices.

## Problem Fixed

**Original Issue**: `AprioriAnalyzer not initialized - database manager required advanced_pattern_discovery.py`

The issue occurred because:
1. `AdvancedPatternDiscovery` was being instantiated without a `DatabaseManager`
2. `AprioriAnalyzer` requires a synchronous database connection for association rule mining
3. The singleton pattern was causing initialization failures when `db_manager` was `None`

## Solution Implemented

### Lazy Initialization with Dependency Injection

We implemented a **lazy initialization pattern** with **thread-safe double-checked locking** that:

1. **Accepts DatabaseManager in constructor** - Follows dependency injection principles
2. **Lazy loads AprioriAnalyzer** - Only creates when actually needed
3. **Thread-safe initialization** - Uses `threading.Lock` for concurrent access
4. **Enhanced error handling** - Graceful degradation when dependencies unavailable
5. **Backward compatibility** - Existing code continues to work

### Key Technical Components

#### 1. Property-Based Lazy Loading
```python
@property
def apriori_analyzer(self):
    """Lazy initialization of AprioriAnalyzer with thread-safe double-checked locking."""
    if self._apriori_analyzer is None:
        with self._apriori_lock:
            if self._apriori_analyzer is None:
                if self._db_manager is not None:
                    self._apriori_analyzer = AprioriAnalyzer(db_manager=self._db_manager)
    return self._apriori_analyzer
```

#### 2. Enhanced Error Handling
```python
def _ensure_apriori_analyzer(self) -> bool:
    """Ensure AprioriAnalyzer is available with enhanced error handling."""
    if self.apriori_analyzer is None:
        logger.warning("AprioriAnalyzer not available - database manager required")
        return False
    return True
```

#### 3. Thread-Safe Database Manager Access
```python
@property
def db_manager(self):
    """Thread-safe access to database manager."""
    with self._db_lock:
        return self._db_manager
```

## Database Configuration

The solution uses **PostgreSQL** with proper driver configuration:

- **Synchronous connections**: `postgresql+psycopg://` (psycopg3)
- **Async connections**: `postgresql+asyncpg://` (asyncpg)

### Database URL Examples
```python
# Synchronous (for AprioriAnalyzer)
sync_url = "postgresql+psycopg://user:password@localhost:5432/database"

# Async (for general application use)
async_url = "postgresql+asyncpg://user:password@localhost:5432/database"
```

## Usage Examples

### ✅ Correct Initialization

```python
from prompt_improver.services.advanced_pattern_discovery import AdvancedPatternDiscovery
from prompt_improver.database.connection import DatabaseManager

# Create DatabaseManager first
db_manager = DatabaseManager("postgresql+psycopg://user:password@localhost:5432/db")

# Pass it to AdvancedPatternDiscovery
service = AdvancedPatternDiscovery(db_manager=db_manager)

# AprioriAnalyzer will be created lazily when needed
patterns = await service.discover_advanced_patterns(session, include_apriori=True)
```

### ✅ Handling Missing Database Manager

```python
# Service can handle None db_manager gracefully
service = AdvancedPatternDiscovery(db_manager=None)

# Methods will detect missing AprioriAnalyzer and handle appropriately
patterns = await service.discover_advanced_patterns(session, include_apriori=True)
# Will skip Apriori analysis and continue with other algorithms
```

### ❌ Incorrect Initialization (Fixed)

```python
# OLD: This would cause "AprioriAnalyzer not initialized" error
service = AdvancedPatternDiscovery()  # No db_manager parameter

# NEW: This works with lazy initialization
service = AdvancedPatternDiscovery(db_manager=None)  # Explicit None is OK
```

## Testing Best Practices

### Test Initialization
```python
import pytest
from prompt_improver.services.advanced_pattern_discovery import AdvancedPatternDiscovery
from prompt_improver.database.connection import DatabaseManager

@pytest.fixture
def discovery_service():
    """Create discovery service with proper database manager."""
    db_manager = DatabaseManager("postgresql+psycopg://user:password@localhost:5432/test_db")
    return AdvancedPatternDiscovery(db_manager=db_manager)

def test_service_initialization(discovery_service):
    """Test that service initializes properly."""
    assert discovery_service.db_manager is not None
    # AprioriAnalyzer should be lazily initialized
    assert discovery_service._apriori_analyzer is None  # Not yet created
```

### Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(db_url=st.text())
def test_graceful_database_handling(db_url):
    """Test that service handles various database configurations gracefully."""
    try:
        db_manager = DatabaseManager(db_url)
        service = AdvancedPatternDiscovery(db_manager=db_manager)
        # Should not crash during initialization
        assert service is not None
    except Exception:
        # Invalid URLs should fail gracefully
        pass
```

## Migration Guide

### For Existing Code

1. **Update instantiation calls**:
   ```python
   # OLD
   service = AdvancedPatternDiscovery()
   
   # NEW
   db_manager = DatabaseManager(database_url)
   service = AdvancedPatternDiscovery(db_manager=db_manager)
   ```

2. **Update test files**:
   ```python
   # OLD
   discovery_service = AdvancedPatternDiscovery()
   
   # NEW
   from prompt_improver.database.connection import DatabaseManager
   db_manager = DatabaseManager("postgresql+psycopg://user:password@localhost:5432/test_db")
   discovery_service = AdvancedPatternDiscovery(db_manager=db_manager)
   ```

3. **Update singleton usage**:
   ```python
   # OLD
   service = get_advanced_pattern_discovery()
   
   # NEW
   service = get_advanced_pattern_discovery(db_manager=db_manager)
   ```

## Performance Considerations

### Lazy Loading Benefits
- **Memory efficiency**: AprioriAnalyzer only created when needed
- **Faster startup**: No immediate database connection required
- **Resource optimization**: Unused components don't consume resources

### Thread Safety
- **Concurrent access**: Multiple threads can safely access the service
- **Double-checked locking**: Minimizes lock contention
- **Lock granularity**: Separate locks for different components

## Error Handling

The service provides multiple levels of error handling:

### 1. Initialization Level
```python
try:
    service = AdvancedPatternDiscovery(db_manager=db_manager)
except Exception as e:
    logger.error(f"Failed to initialize service: {e}")
    # Fallback to basic service without AprioriAnalyzer
    service = AdvancedPatternDiscovery(db_manager=None)
```

### 2. Runtime Level
```python
# Service methods check for component availability
if not service._ensure_apriori_analyzer():
    logger.warning("Skipping Apriori analysis - database manager required")
    # Continue with other algorithms
```

### 3. Database Level
```python
# DatabaseManager handles connection failures gracefully
try:
    with db_manager.get_connection() as conn:
        # Use connection
        pass
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    # Graceful degradation
```

## Monitoring and Logging

### Key Log Messages
- `"AprioriAnalyzer not available - database manager required"` - Expected when db_manager is None
- `"AprioriAnalyzer initialized successfully"` - Successful lazy initialization
- `"Database connection failed"` - Database connectivity issues

### Metrics to Monitor
- **Initialization time**: How long lazy loading takes
- **Memory usage**: AprioriAnalyzer memory footprint
- **Database connections**: Connection pool utilization
- **Error rates**: Frequency of initialization failures

## Future Enhancements

1. **Configuration-based initialization**: Load database settings from config files
2. **Health checks**: Periodic validation of database connectivity
3. **Retry mechanisms**: Automatic retry for transient database failures
4. **Performance monitoring**: Built-in metrics collection

## Related Documentation

- [`src/prompt_improver/services/advanced_pattern_discovery.py`](../src/prompt_improver/services/advanced_pattern_discovery.py) - Main implementation
- [`src/prompt_improver/database/connection.py`](../src/prompt_improver/database/connection.py) - Database connection management
- [`tests/integration/test_apriori_integration.py`](../tests/integration/test_apriori_integration.py) - Integration tests
- [`verify_fix.py`](../verify_fix.py) - Verification script