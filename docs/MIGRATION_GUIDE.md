# Migration Guide: 2025 Architecture Modernization

This guide helps developers understand the major architectural changes made during the comprehensive cleanup and modernization effort, and how to work with the new patterns.

## Table of Contents

1. [Overview of Changes](#overview-of-changes)
2. [Service Architecture Migration](#service-architecture-migration)
3. [Database Services Migration](#database-services-migration)
4. [Security Services Migration](#security-services-migration)
5. [Repository Pattern Adoption](#repository-pattern-adoption)
6. [Dependency Injection Updates](#dependency-injection-updates)
7. [Testing Pattern Changes](#testing-pattern-changes)
8. [Common Migration Tasks](#common-migration-tasks)
9. [Troubleshooting](#troubleshooting)

## Overview of Changes

### Key Architectural Improvements

1. **Clean Architecture Implementation**: Strict separation of concerns across layers
2. **Service Consolidation**: Multiple managers consolidated into unified facades
3. **Repository Pattern**: Protocol-based data access abstraction
4. **Event-Driven ML**: Background ML processing with event bus communication
5. **Unified Security**: Consolidated security services with facade pattern
6. **Modern Async**: Async-first design with proper error handling

### Files Removed During Cleanup

- `src/prompt_improver/core/services/prompt_improvement_backup.py` - Replaced by clean service
- `examples/*_example.py` - Example files that are no longer relevant
- Legacy compatibility layers that are no longer needed
- Unused imports and dead code throughout the codebase

## Service Architecture Migration

### Before: Multiple Managers

```python
# Old pattern - multiple separate managers
from database.connection_manager import ConnectionManager
from database.query_manager import QueryManager
from security.auth_manager import AuthManager
from security.crypto_manager import CryptoManager
from cache.redis_manager import RedisManager

connection_manager = ConnectionManager()
query_manager = QueryManager(connection_manager)
auth_manager = AuthManager()
crypto_manager = CryptoManager()
redis_manager = RedisManager()
```

### After: Unified Service Composition

```python
# New pattern - unified service composition
from prompt_improver.database.services.composition import get_database_services, ManagerMode
from prompt_improver.security.unified.security_service_facade import get_security_service_facade

# Get composed services
database_services = await get_database_services(ManagerMode.PRODUCTION)
security_services = await get_security_service_facade()

# All operations through unified interfaces
async with database_services.get_session() as session:
    result = await session.execute(query)

auth_result = await security_services.authenticate(credentials)
```

## Database Services Migration

### Connection Management

**Before:**
```python
# Old direct database connections
import asyncpg
pool = await asyncpg.create_pool(DATABASE_URL)
connection = await pool.acquire()
```

**After:**
```python
# New database services composition
from prompt_improver.database.services.composition import get_database_services

services = await get_database_services(ManagerMode.PRODUCTION)
async with services.get_session() as session:
    # Session automatically managed
    result = await session.execute(query)
```

### Repository Pattern Usage

**Before:**
```python
# Old direct database queries
async def get_user(user_id: str):
    query = "SELECT * FROM users WHERE id = $1"
    result = await connection.fetchrow(query, user_id)
    return result
```

**After:**
```python
# New repository pattern
class UserRepository:
    def __init__(self, db_services: DatabaseServices):
        self.db = db_services
    
    async def get_user(self, user_id: str) -> User | None:
        async with self.db.get_session() as session:
            query = select(User).where(User.id == user_id)
            result = await session.execute(query)
            return result.scalar_one_or_none()
```

## Security Services Migration

### Unified Security Facade

**Before:**
```python
# Old multiple security managers
from security.unified_security_manager import get_unified_security_manager
from security.unified_authentication_manager import get_unified_authentication_manager
from security.unified_crypto_manager import get_unified_crypto_manager

security = await get_unified_security_manager()
auth = await get_unified_authentication_manager()
crypto = await get_unified_crypto_manager()
```

**After:**
```python
# New unified security facade
from prompt_improver.security.unified.security_service_facade import get_security_service_facade

security = await get_security_service_facade()

# All security operations through facade
auth = await security.authentication
crypto = await security.cryptography
validation = await security.validation
rate_limiting = await security.rate_limiting
```

### Legacy Compatibility

For existing code that uses legacy security interfaces, compatibility adapters are available:

```python
# Legacy compatibility (temporary)
from prompt_improver.security.unified.legacy_compatibility import (
    get_unified_security_manager,
    get_unified_authentication_manager
)

# These adapters delegate to the new facade
security_manager = await get_unified_security_manager()
auth_manager = await get_unified_authentication_manager()
```

## Repository Pattern Adoption

### Creating Repository Protocols

```python
from typing import Protocol

class IUserRepository(Protocol):
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        ...
    
    async def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        ...
    
    async def update_user(self, user_id: str, user_data: UserUpdate) -> User:
        """Update user data."""
        ...
```

### Implementing Repositories

```python
class UserRepository:
    def __init__(self, db_services: DatabaseServices):
        self.db = db_services
    
    async def create_user(self, user_data: UserCreate) -> User:
        async with self.db.get_session() as session:
            user = User(**user_data.dict())
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
```

### Using Repositories in Services

```python
class UserService:
    def __init__(self, user_repository: IUserRepository):
        self.user_repo = user_repository
    
    async def register_user(self, user_data: UserCreate) -> UserResponse:
        # Business logic here
        user = await self.user_repo.create_user(user_data)
        return UserResponse.from_user(user)
```

## Dependency Injection Updates

### Clean DI Container Usage

**Before:**
```python
# Old direct instantiation
from prompt_improver.core.di.container import get_container

container = await get_container()
service = await container.get(SomeService)
```

**After:**
```python
# New clean architecture container
from prompt_improver.core.di.clean_container import get_clean_container

container = await get_clean_container()
service = await container.get(ISomeService)  # Use protocol interface
```

### Service Registration

```python
# Register services with protocols
async def configure_services():
    container = await get_clean_container()
    
    # Register repository implementations
    db_services = await get_database_services()
    container.register(IUserRepository, UserRepository(db_services))
    
    # Register application services
    user_repo = await container.get(IUserRepository)
    container.register(IUserService, UserService(user_repo))
```

## Testing Pattern Changes

### Repository Testing

**Before:**
```python
# Old direct database testing
@pytest.fixture
async def database_connection():
    pool = await asyncpg.create_pool(TEST_DATABASE_URL)
    yield pool
    await pool.close()

async def test_user_creation(database_connection):
    # Direct database operations
    pass
```

**After:**
```python
# New repository-based testing
@pytest.fixture
async def user_repository():
    db_services = await get_database_services(ManagerMode.TESTING)
    return UserRepository(db_services)

@pytest.fixture
async def mock_user_repository():
    return MockUserRepository()

async def test_user_service_with_mock(mock_user_repository):
    """Unit test with mock repository."""
    service = UserService(mock_user_repository)
    result = await service.register_user(user_data)
    assert result is not None

async def test_user_service_integration(user_repository):
    """Integration test with real repository."""
    service = UserService(user_repository)
    result = await service.register_user(user_data)
    assert result is not None
```

### Service Testing

```python
class TestUserService:
    """Test user service at different levels."""
    
    async def test_unit_with_mock(self, mock_user_repository):
        """Unit test - fast, isolated."""
        service = UserService(mock_user_repository)
        # Test business logic
        pass
    
    async def test_integration_with_db(self, user_repository):
        """Integration test - real database."""
        service = UserService(user_repository)
        # Test with real data persistence
        pass
```

## Common Migration Tasks

### 1. Update Import Statements

**Before:**
```python
from database.connection_manager import ConnectionManager
from security.auth_manager import AuthManager
```

**After:**
```python
from prompt_improver.database.services.composition import get_database_services
from prompt_improver.security.unified.security_service_facade import get_security_service_facade
```

### 2. Replace Direct Database Calls

**Before:**
```python
async def get_data():
    query = "SELECT * FROM table"
    result = await connection.fetch(query)
    return result
```

**After:**
```python
async def get_data():
    async with self.db_services.get_session() as session:
        query = select(Model)
        result = await session.execute(query)
        return result.scalars().all()
```

### 3. Update Service Initialization

**Before:**
```python
class MyService:
    def __init__(self):
        self.db = ConnectionManager()
        self.auth = AuthManager()
```

**After:**
```python
class MyService:
    def __init__(
        self,
        user_repository: IUserRepository,
        security_service: SecurityServiceFacade
    ):
        self.user_repo = user_repository
        self.security = security_service
```

### 4. Update Configuration

**Before:**
```python
# Old configuration
DATABASE_URL = "postgresql://..."
REDIS_URL = "redis://..."
```

**After:**
```python
# New configuration with ManagerMode
DATABASE_URL = "postgresql://..."
REDIS_URL = "redis://..."
MANAGER_MODE = ManagerMode.PRODUCTION
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ImportError: cannot import name 'OldManager' from 'module'
```

**Solution:**
The old manager was removed. Use the new service composition:
```python
# Instead of importing old manager
from prompt_improver.database.services.composition import get_database_services
services = await get_database_services(ManagerMode.PRODUCTION)
```

#### 2. Service Not Registered

**Problem:**
```
ServiceNotRegisteredError: Service IUserRepository is not registered
```

**Solution:**
Register the service with the DI container:
```python
container = await get_clean_container()
container.register(IUserRepository, UserRepository(db_services))
```

#### 3. Async Context Issues

**Problem:**
```
RuntimeError: Session is not bound to a connection
```

**Solution:**
Use proper async context management:
```python
async with services.get_session() as session:
    # All database operations here
    result = await session.execute(query)
```

#### 4. Legacy Security Manager Issues

**Problem:**
```
AttributeError: 'UnifiedSecurityManagerAdapter' has no attribute 'new_method'
```

**Solution:**
Migrate to the new security facade:
```python
security = await get_security_service_facade()
auth = await security.authentication
result = await auth.authenticate(credentials)
```

### Performance Considerations

1. **Connection Pooling**: New service composition includes proper connection pooling
2. **Caching**: Unified cache coordinator provides multi-level caching
3. **Async Operations**: All I/O operations are properly async
4. **Resource Management**: Automatic resource cleanup with async context managers

### Testing Migration

1. **Update Test Fixtures**: Use new service composition in test fixtures
2. **Mock Repositories**: Create mock implementations of repository protocols
3. **Integration Tests**: Use real services for integration testing
4. **Performance Tests**: Validate that new architecture maintains performance

## Migration Checklist

- [ ] Update import statements to use new service composition
- [ ] Replace direct database connections with repository pattern
- [ ] Migrate security operations to unified facade
- [ ] Update dependency injection to use clean container
- [ ] Create repository protocols for data access
- [ ] Implement repositories with proper error handling
- [ ] Update tests to use new patterns
- [ ] Remove legacy imports and unused code
- [ ] Validate that all functionality still works
- [ ] Update documentation and comments

## Best Practices

### 1. Always Use Repository Protocols

```python
# Good - depends on abstraction
class UserService:
    def __init__(self, user_repo: IUserRepository):
        self.user_repo = user_repo

# Bad - depends on concrete implementation
class UserService:
    def __init__(self, db_connection: Connection):
        self.db = db_connection
```

### 2. Proper Error Handling

```python
async def get_user(self, user_id: str) -> User | None:
    try:
        async with self.db.get_session() as session:
            result = await session.execute(query)
            return result.scalar_one_or_none()
    except DatabaseError as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise ServiceError(f"User retrieval failed") from e
```

### 3. Use Type Hints

```python
from typing import Protocol, Optional

class IUserRepository(Protocol):
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        ...
```

### 4. Proper Async Context Management

```python
async def process_users():
    async with self.db_services.get_session() as session:
        # All database operations within this context
        users = await session.execute(select(User))
        for user in users.scalars():
            await self._process_user(user, session)
        # Automatic commit/rollback handling
```

## Getting Help

1. **Architecture Documentation**: See `/docs/ARCHITECTURE_PATTERNS.md`
2. **Code Examples**: Check `/examples/` directory for current patterns
3. **Test Examples**: Look at `/tests/integration/` for testing patterns
4. **Repository Examples**: See repository implementations in `/src/prompt_improver/repositories/`

This migration guide should help you understand and work with the new architectural patterns. The changes provide better testability, maintainability, and performance while following 2025 best practices for Python applications.