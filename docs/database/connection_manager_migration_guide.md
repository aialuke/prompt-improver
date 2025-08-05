# Connection Manager Migration Guide

## Overview
This guide provides step-by-step instructions for migrating from the legacy connection managers to the unified `UnifiedConnectionManagerV2` system.

## Migration Timeline
- **Phase 1**: Update imports and basic usage (Weeks 1-2)
- **Phase 2**: Migrate configuration and advanced features (Weeks 3-4)  
- **Phase 3**: Remove legacy code and optimize (Weeks 5-6)

## Before You Begin

### Prerequisites
- [ ] Python 3.8+ with async/await support
- [ ] SQLAlchemy 2.0+ installed
- [ ] asyncpg or psycopg3 for PostgreSQL
- [ ] pytest for testing migrations

### Backup Recommendations
```bash
# Backup current database configuration
cp config/database_config.yaml config/database_config.backup.yaml

# Backup existing connection code
git tag pre-connection-migration-$(date +%Y%m%d)
git push origin pre-connection-migration-$(date +%Y%m%d)
```

## Migration by Legacy Manager

### 1. DatabaseManager Migration

#### Before (Legacy Code)
```python
from prompt_improver.database import DatabaseManager
from prompt_improver.database.models import User

# Old synchronous usage
db_manager = DatabaseManager(config)
session = db_manager.get_session()

try:
    users = session.query(User).filter(User.active == True).all()
    # Process users...
finally:
    session.close()
```

#### After (Unified V2)
```python
from prompt_improver.database import UnifiedConnectionManagerV2
from prompt_improver.core.protocols import ConnectionMode
from prompt_improver.database.models import User

# New async usage
async def get_active_users():
    connection_manager = UnifiedConnectionManagerV2(config)
    
    async with connection_manager.get_connection(
        mode=ConnectionMode.READ_ONLY
    ) as conn:
        result = await conn.execute(
            select(User).where(User.active == True)
        )
        return result.scalars().all()
```

#### Migration Steps
1. **Update imports**:
   ```python
   # Replace this
   from prompt_improver.database import DatabaseManager
   
   # With this
   from prompt_improver.database import UnifiedConnectionManagerV2
   from prompt_improver.core.protocols import ConnectionMode
   ```

2. **Convert sync to async**:
   ```python
   # Old sync function
   def get_user_data(user_id: int):
       session = db_manager.get_session()
       # ... database operations
   
   # New async function
   async def get_user_data(user_id: int):
       async with connection_manager.get_connection() as conn:
           # ... database operations
   ```

3. **Update configuration**:
   ```yaml
   # old config
   database:
     host: localhost
     port: 5432
     sync_pool_size: 10
   
   # new unified config  
   database:
     host: localhost
     port: 5432
     pool_size: 20
     enable_ha: true
     enable_health_checks: true
   ```

### 2. HAConnectionManager Migration

#### Before (Legacy Code)
```python
from prompt_improver.database.ha_connection_manager import HAConnectionManager

ha_manager = HAConnectionManager(
    primary_host="db-primary",
    replica_hosts=["db-replica1", "db-replica2"],
    failover_timeout=5.0
)

# Manual failover handling
try:
    conn = await ha_manager.get_primary_connection()
    # ... operations
except ConnectionError:
    conn = await ha_manager.failover_to_replica()
```

#### After (Unified V2)
```python
from prompt_improver.database import UnifiedConnectionManagerV2
from prompt_improver.core.protocols import ConnectionMode

# HA features built-in
config = UnifiedDatabaseConfig(
    host="db-primary",
    replica_hosts=["db-replica1", "db-replica2"],
    enable_ha=True,
    failover_timeout=5.0
)

connection_manager = UnifiedConnectionManagerV2(config)

# Automatic failover handling
async with connection_manager.get_connection(
    mode=ConnectionMode.READ_WRITE
) as conn:
    # Automatic failover if primary fails
    # ... operations
```

#### Migration Steps
1. **Update configuration format**:
   ```python
   # Old HA config
   ha_config = {
       'primary_host': 'db-primary',
       'replica_hosts': ['db-replica1', 'db-replica2'],
       'failover_timeout': 5.0,
       'health_check_interval': 30
   }
   
   # New unified config
   config = UnifiedDatabaseConfig(
       host='db-primary',
       replica_hosts=['db-replica1', 'db-replica2'],
       enable_ha=True,
       failover_timeout=5.0,
       health_check_interval=30
   )
   ```

2. **Remove manual failover logic**:
   ```python
   # Remove manual failover handling - now automatic
   # Old code with manual failover can be simplified
   async with connection_manager.get_connection() as conn:
       # Built-in failover handling
       result = await conn.execute(query)
   ```

### 3. UnifiedConnectionManager Migration

#### Before (Legacy Code)
```python
from prompt_improver.database.unified_connection_manager import UnifiedConnectionManager

manager = UnifiedConnectionManager(config)

# Mode-based access
async with manager.get_mcp_connection() as conn:
    # MCP operations
    pass

async with manager.get_ml_connection() as conn:
    # ML operations  
    pass
```

#### After (Unified V2)
```python
from prompt_improver.database import UnifiedConnectionManagerV2
from prompt_improver.core.protocols import ConnectionMode

manager = UnifiedConnectionManagerV2(config)

# Simplified mode-based access
async with manager.get_connection(mode=ConnectionMode.READ_ONLY) as conn:
    # Read-only operations
    pass

async with manager.get_connection(mode=ConnectionMode.BATCH) as conn:
    # Batch operations
    pass
```

#### Migration Steps
1. **Update mode specifications**:
   ```python
   # Old modes
   get_mcp_connection() → get_connection(mode=ConnectionMode.READ_WRITE)
   get_ml_connection() → get_connection(mode=ConnectionMode.BATCH)
   get_readonly_connection() → get_connection(mode=ConnectionMode.READ_ONLY)
   get_admin_connection() → get_connection(mode=ConnectionMode.TRANSACTIONAL)
   ```

### 4. DatabaseSessionManager Migration

#### Before (Legacy Code)
```python
from prompt_improver.database.session_manager import DatabaseSessionManager

session_manager = DatabaseSessionManager(config)

async with session_manager.async_session() as session:
    result = await session.execute(select(User))
    users = result.scalars().all()
```

#### After (Unified V2)
```python
from prompt_improver.database import UnifiedConnectionManagerV2

connection_manager = UnifiedConnectionManagerV2(config)

async with connection_manager.get_connection() as conn:
    result = await conn.execute(select(User))
    users = result.scalars().all()
```

### 5. RegistryManager Migration

#### Before (Legacy Code)
```python
from prompt_improver.database.registry import RegistryManager

registry = RegistryManager()
registry.register_model(User)
registry.resolve_conflicts()
Base = registry.get_base()
```

#### After (Unified V2)
```python
# Registry management now built into connection manager
from prompt_improver.database import UnifiedConnectionManagerV2

connection_manager = UnifiedConnectionManagerV2(config)
# Registry management is automatic - no manual steps needed
```

## Configuration Migration

### Legacy Configuration Format
```yaml
# Multiple configuration sections
database:
  host: localhost
  port: 5432
  database: apes_db
  sync_pool_size: 10

ha_database:
  primary_host: localhost
  replica_hosts: []
  failover_timeout: 5.0

connection_pools:
  mcp_pool_size: 15
  ml_pool_size: 25
  admin_pool_size: 5

session_management:
  enable_async: true
  session_timeout: 3600
```

### Unified Configuration Format
```yaml
# Single unified configuration
database:
  # Basic connection
  host: localhost
  port: 5432
  database: apes_db
  username: ${DATABASE_USER}
  password: ${DATABASE_PASSWORD}
  
  # Pool configuration
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  
  # High availability
  replica_hosts: []
  failover_timeout: 5.0
  health_check_interval: 30
  
  # Performance tuning
  statement_cache_size: 1000
  prepared_statement_cache_size: 1000
  connection_timeout: 10.0
  
  # Feature flags
  enable_ha: true
  enable_metrics: true
  enable_health_checks: true
  strict_mode: false
```

### Configuration Migration Script
```python
# config_migration.py
import yaml
from pathlib import Path
from typing import Dict, Any

def migrate_database_config(old_config_path: Path, new_config_path: Path):
    """Migrate legacy database configuration to unified format"""
    
    with open(old_config_path) as f:
        old_config = yaml.safe_load(f)
    
    # Extract legacy settings
    db_config = old_config.get('database', {})
    ha_config = old_config.get('ha_database', {})
    pool_config = old_config.get('connection_pools', {})
    session_config = old_config.get('session_management', {})
    
    # Build unified configuration
    new_config = {
        'database': {
            # Basic connection (from database section)
            'host': db_config.get('host', 'localhost'),
            'port': db_config.get('port', 5432),
            'database': db_config.get('database', 'apes_db'),
            'username': db_config.get('username', '${DATABASE_USER}'),
            'password': db_config.get('password', '${DATABASE_PASSWORD}'),
            
            # Pool configuration (from connection_pools section)
            'pool_size': pool_config.get('mcp_pool_size', 20),
            'max_overflow': pool_config.get('max_overflow', 30),
            'pool_timeout': pool_config.get('pool_timeout', 30),
            'pool_recycle': pool_config.get('pool_recycle', 3600),
            
            # High availability (from ha_database section)
            'replica_hosts': ha_config.get('replica_hosts', []),
            'failover_timeout': ha_config.get('failover_timeout', 5.0),
            'health_check_interval': ha_config.get('health_check_interval', 30),
            
            # Performance tuning
            'statement_cache_size': 1000,
            'prepared_statement_cache_size': 1000,
            'connection_timeout': 10.0,
            
            # Feature flags
            'enable_ha': bool(ha_config.get('replica_hosts')),
            'enable_metrics': True,
            'enable_health_checks': True,
            'strict_mode': False
        }
    }
    
    # Write new configuration
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration migrated from {old_config_path} to {new_config_path}")

if __name__ == "__main__":
    migrate_database_config(
        Path("config/database_config.yaml"),
        Path("config/unified_database_config.yaml")
    )
```

## Testing Migration

### Unit Test Migration
```python
# Before (Legacy Tests)
import pytest
from prompt_improver.database import DatabaseManager

@pytest.fixture
def db_manager():
    config = TestDatabaseConfig()
    return DatabaseManager(config)

def test_user_creation(db_manager):
    session = db_manager.get_session()
    user = User(name="test", email="test@example.com")
    session.add(user)
    session.commit()
    
    assert user.id is not None
    session.close()

# After (Unified V2 Tests)
import pytest
from prompt_improver.database import UnifiedConnectionManagerV2
from prompt_improver.core.protocols import ConnectionMode

@pytest.fixture
async def connection_manager():
    config = UnifiedDatabaseConfig(
        host="localhost",
        database="test_db",
        enable_ha=False,  # Disable for testing
        enable_health_checks=False
    )
    return UnifiedConnectionManagerV2(config)

@pytest.mark.asyncio
async def test_user_creation(connection_manager):
    async with connection_manager.get_connection(
        mode=ConnectionMode.READ_WRITE
    ) as conn:
        user = User(name="test", email="test@example.com")
        conn.add(user)
        await conn.commit()
        
        assert user.id is not None
```

### Integration Test Updates
```python
# Integration test with health checks
@pytest.mark.asyncio
async def test_connection_health(connection_manager):
    """Test connection health monitoring"""
    health_result = await connection_manager.health_check()
    
    assert health_result['status'] == 'healthy'
    assert 'connection_pool' in health_result
    assert 'active_connections' in health_result

@pytest.mark.asyncio  
async def test_failover_behavior(connection_manager):
    """Test high availability failover"""
    # This test requires HA configuration
    config = UnifiedDatabaseConfig(
        host="unreachable-host",
        replica_hosts=["localhost"],
        enable_ha=True,
        failover_timeout=1.0
    )
    
    ha_manager = UnifiedConnectionManagerV2(config)
    
    # Should automatically failover to replica
    async with ha_manager.get_connection() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
```

## Performance Optimization

### Connection Pool Tuning
```python
# Optimize for your workload
config = UnifiedDatabaseConfig(
    # For high-throughput applications
    pool_size=50,
    max_overflow=100,
    pool_timeout=60,
    
    # For memory-constrained environments  
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,  # Recycle connections every 30min
    
    # For low-latency requirements
    connection_timeout=5.0,
    statement_cache_size=2000,
    prepared_statement_cache_size=2000
)
```

### Monitoring and Metrics
```python
# Enable built-in metrics collection
config = UnifiedDatabaseConfig(
    enable_metrics=True,
    enable_health_checks=True
)

connection_manager = UnifiedConnectionManagerV2(config)

# Access connection metrics
info = await connection_manager.get_connection_info()
print(f"Active connections: {info['active_connections']}")
print(f"Pool utilization: {info['pool_utilization']}%")
print(f"Average response time: {info['avg_response_time']}ms")
```

## Troubleshooting

### Common Migration Issues

#### Issue 1: Async/Sync Mixing
```python
# Problem: Mixing sync and async code
def sync_function():
    async with connection_manager.get_connection() as conn:  # ❌ Wrong
        return conn.execute(query)

# Solution: Make function async or use sync adapter
async def async_function():
    async with connection_manager.get_connection() as conn:  # ✅ Correct
        return await conn.execute(query)
```

#### Issue 2: Connection Mode Errors
```python
# Problem: Using wrong connection mode
async with connection_manager.get_connection(
    mode=ConnectionMode.READ_ONLY
) as conn:
    await conn.execute(insert_query)  # ❌ Write on read-only

# Solution: Use appropriate mode
async with connection_manager.get_connection(
    mode=ConnectionMode.READ_WRITE
) as conn:
    await conn.execute(insert_query)  # ✅ Correct
```

#### Issue 3: Configuration Errors
```python
# Problem: Missing required configuration
config = UnifiedDatabaseConfig(
    host="localhost"  # Missing database, username, password
)

# Solution: Complete configuration
config = UnifiedDatabaseConfig(
    host="localhost",
    database="apes_db",
    username=os.getenv("DATABASE_USER"),
    password=os.getenv("DATABASE_PASSWORD")
)
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger("prompt_improver.database").setLevel(logging.DEBUG)

# Enable connection debug mode
config = UnifiedDatabaseConfig(
    host="localhost",
    database="apes_db",
    # ... other config
    echo=True,  # Enable SQLAlchemy query logging
    debug_mode=True  # Enable connection manager debug mode
)
```

## Rollback Plan

### Preparation
```bash
# Tag current state before migration
git tag connection-migration-checkpoint-$(date +%Y%m%d)

# Keep legacy code temporarily
mkdir legacy_backup
cp -r src/prompt_improver/database/legacy/ legacy_backup/
```

### Rollback Steps
1. **Restore configuration**:
   ```bash
   cp config/database_config.backup.yaml config/database_config.yaml
   ```

2. **Restore imports**:
   ```python
   # Revert to legacy imports if needed
   from prompt_improver.database.legacy import DatabaseManager
   ```

3. **Restore legacy code**:
   ```bash
   git checkout connection-migration-checkpoint-$(date +%Y%m%d)
   ```

## Validation Checklist

### Pre-Migration
- [ ] All existing tests pass
- [ ] Performance benchmarks recorded
- [ ] Configuration backed up
- [ ] Legacy usage documented

### During Migration
- [ ] Import updates completed
- [ ] Configuration migrated and validated
- [ ] Tests updated and passing
- [ ] Performance comparable or improved

### Post-Migration  
- [ ] All functionality working as expected
- [ ] Performance meets or exceeds baseline
- [ ] Health checks operational
- [ ] Monitoring dashboards updated
- [ ] Legacy code removed
- [ ] Documentation updated

## Getting Help
- **Documentation**: [Connection Manager API Reference](../user/API_REFERENCE.md#connection-manager)
- **Examples**: [Connection Manager Examples](../../examples/connection_manager_examples.py)
- **Issues**: Create issues in the project repository with `migration` tag
- **Team Support**: Reach out to the database team for migration assistance

---

*Last Updated: 2025-01-28*  
*Migration Guide Version: 1.0*