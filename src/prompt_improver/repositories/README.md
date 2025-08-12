# Repository Layer Architecture

This directory implements a comprehensive repository pattern that eliminates direct database access from the API layer, following clean architecture principles.

## Architecture Overview

```
API Layer → Service Layer → Repository Layer → Database Layer
```

### Key Components

#### 1. Repository Protocols (`protocols/`)
Abstract interfaces defining repository contracts:
- `BaseRepositoryProtocol` - Common CRUD operations and query patterns
- `AnalyticsRepositoryProtocol` - Session analysis and performance metrics
- `AprioriRepositoryProtocol` - Association rule mining and pattern discovery
- `MLRepositoryProtocol` - ML training and model management
- `RulesRepositoryProtocol` - Rule engine data access
- `UserFeedbackRepositoryProtocol` - User satisfaction and feedback
- `HealthRepositoryProtocol` - System health monitoring

#### 2. Repository Interfaces (`interfaces/`)
Concrete interface classes for dependency injection:
- Clean interface definitions combining protocols
- Type-safe contracts for service layer integration
- Protocol-based design following SOLID principles

#### 3. Base Repository (`base_repository.py`)
Foundation implementation providing:
- Generic CRUD operations with type safety
- Query builder with method chaining
- Transaction management with savepoint support
- Batch operations for performance optimization
- Health checking and diagnostics
- Integration with `DatabaseServices`

#### 4. Implementation Layer (`impl/`)
*Note: Concrete implementations will be created during API extraction phase*
- Domain-specific repository implementations
- Repository factory for dependency injection
- Query optimization and caching integration

## Benefits

### Clean Architecture
- **Separation of Concerns**: Clear boundaries between layers
- **Dependency Inversion**: API layer depends on abstractions, not implementations
- **Single Responsibility**: Each repository handles one domain
- **Interface Segregation**: Small, focused interfaces

### Database Access Improvements
- **Connection Pooling**: Leverages existing `UnifiedConnectionManager`
- **Query Optimization**: Built-in query building and optimization
- **Transaction Management**: Cross-repository transaction support
- **Batch Operations**: Efficient bulk database operations

### Developer Experience
- **Type Safety**: Generic types throughout with proper typing
- **Testability**: Easy repository mocking for unit tests
- **Consistency**: Standardized patterns across all domains
- **Maintainability**: Clear domain boundaries and responsibilities

### Performance & Scalability
- **Connection Reuse**: Efficient database connection management
- **Query Optimization**: Repository-level query optimization
- **Caching Integration**: Ready for cache layer integration
- **Horizontal Scaling**: Support for read/write separation

## Usage Example

```python
from prompt_improver.repositories import IRepositoryFactory, get_repository_factory

# Get repository factory (dependency injection)
factory = await get_repository_factory()

# Create domain repositories
analytics_repo = await factory.create_analytics_repository()
rules_repo = await factory.create_rules_repository()

# Use in service layer
async def get_session_analytics(start_date: datetime, end_date: datetime):
    return await analytics_repo.get_session_analytics(
        start_date=start_date,
        end_date=end_date
    )

async def get_top_rules():
    return await rules_repo.get_top_performing_rules(
        metric="improvement_score",
        limit=10
    )
```

## Integration with Existing Systems

### Database Layer Integration
- Uses existing `UnifiedConnectionManager` for connections
- Compatible with existing database models and schemas
- Leverages existing query optimization infrastructure
- Integrates with database health monitoring

### Service Layer Integration
- Clean interfaces for service layer consumption
- Repository factory provides dependency injection
- Transaction management across multiple repositories
- Error handling and logging integration

### Testing Integration
- Repository protocols enable easy mocking
- Base repository provides test utilities
- Health check endpoints for monitoring
- Integration test helpers for database operations

## Implementation Status

### ✅ Completed
- Repository protocols and interfaces
- Base repository implementation
- Query builder and transaction management
- Directory structure and documentation
- Integration architecture design

### 🚧 Pending (API Extraction Phase)
- Concrete repository implementations
- Repository factory implementation
- Service layer integration
- API endpoint migration to use repositories

## Database Compatibility

### Supported Operations
- **PostgreSQL**: Full support with JSONB operations
- **Connection Pooling**: AsyncPG with SQLAlchemy async
- **Transactions**: Full ACID compliance with savepoints
- **Query Optimization**: EXPLAIN plan integration
- **Health Monitoring**: Connection pool and performance metrics

### Migration Compatibility
- **Alembic Integration**: Compatible with existing migration system
- **Schema Evolution**: Repository layer adapts to schema changes
- **Backward Compatibility**: Maintains existing database contracts
- **Zero Downtime**: Repository swapping for live deployments

## Security Considerations

### Input Validation
- Type-safe query parameters
- SQL injection prevention through parameterized queries
- Input sanitization at repository boundary

### Access Control
- Repository-level access patterns
- Connection-level security through `UnifiedConnectionManager`
- Audit trail integration for data access

### Data Protection
- Automatic PII handling in user feedback repositories
- Encryption support for sensitive data fields
- Secure transaction handling

## Next Steps

1. **API Extraction Phase**: Create concrete repository implementations
2. **Service Layer Integration**: Update services to use repositories
3. **API Migration**: Remove direct database access from API endpoints
4. **Performance Tuning**: Optimize queries and add caching
5. **Monitoring Integration**: Add repository-level metrics and alerts

## File Structure

```
repositories/
├── __init__.py                 # Main module exports
├── README.md                  # This documentation
├── base_repository.py         # Base repository implementation
├── protocols/                 # Abstract interfaces
│   ├── __init__.py
│   ├── base_repository_protocol.py
│   ├── analytics_repository_protocol.py
│   ├── apriori_repository_protocol.py
│   ├── ml_repository_protocol.py
│   ├── rules_repository_protocol.py
│   ├── user_feedback_repository_protocol.py
│   └── health_repository_protocol.py
├── interfaces/                # Concrete interfaces for DI
│   ├── __init__.py
│   └── repository_interfaces.py
└── impl/                     # Implementation layer (pending)
    └── __init__.py
```

This repository architecture provides a solid foundation for eliminating the architecture violations identified in the legacy cleanup audit while maintaining high performance and developer productivity.