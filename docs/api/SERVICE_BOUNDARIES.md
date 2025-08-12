# API Documentation: Service Boundaries

This document describes the API boundaries and interfaces for the modernized APES system architecture.

## Table of Contents

1. [Service Architecture Overview](#service-architecture-overview)
2. [Database Services API](#database-services-api)
3. [Security Services API](#security-services-api)
4. [Repository Interfaces](#repository-interfaces)
5. [Application Services API](#application-services-api)
6. [ML Services API](#ml-services-api)
7. [Cache Services API](#cache-services-api)
8. [Error Handling](#error-handling)

## Service Architecture Overview

The system follows clean architecture with well-defined service boundaries:

```
External Interfaces (FastAPI, CLI)
    ↓
Application Services (Business Logic)
    ↓
Repository Protocols (Data Access Contracts)
    ↓
Infrastructure Services (Database, Cache, Security)
```

## Database Services API

### DatabaseServices Composition

**Module**: `prompt_improver.database.services.composition`

```python
from prompt_improver.database.services.composition import get_database_services, ManagerMode

# Get database services instance
services = await get_database_services(ManagerMode.PRODUCTION)
await services.initialize()
```

#### Manager Modes

```python
class ManagerMode(str, Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development" 
    TESTING = "testing"
    ASYNC_MODERN = "async_modern"
    HIGH_AVAILABILITY = "high_availability"
```

#### Core Methods

```python
class DatabaseServices:
    async def initialize(self) -> None:
        """Initialize all database services."""
        
    async def get_session(self) -> AsyncSession:
        """Get database session with automatic management."""
        
    async def health_check(self) -> HealthStatus:
        """Check database health status."""
        
    async def close(self) -> None:
        """Clean up database connections."""
```

### Session Management

```python
# Automatic session management
async with services.get_session() as session:
    query = select(Model).where(Model.id == id)
    result = await session.execute(query)
    # Automatic commit/rollback handling
```

### Health Monitoring

```python
health_status = await services.health_check()
print(f"Database health: {health_status.status}")
print(f"Connection count: {health_status.connection_count}")
```

## Security Services API

### SecurityServiceFacade

**Module**: `prompt_improver.security.unified.security_service_facade`

```python
from prompt_improver.security.unified.security_service_facade import get_security_service_facade

security = await get_security_service_facade()
```

#### Component Access

```python
# Authentication component
auth = await security.authentication
auth_result = await auth.authenticate(credentials, "password")

# Cryptography component
crypto = await security.cryptography
encrypted = await crypto.encrypt_data("sensitive data", key_id, security_context)

# Validation component
validation = await security.validation
is_valid = await validation.validate_input(user_input, "default", security_context)

# Rate limiting component
rate_limiter = await security.rate_limiting
allowed = await rate_limiter.check_rate_limit(security_context, "api_call")
```

### Security Context

```python
from prompt_improver.database import create_security_context, ManagerMode

security_context = create_security_context(
    agent_id="user123",
    manager_mode=ManagerMode.PRODUCTION
)
```

### Authentication Results

```python
@dataclass
class AuthResult:
    success: bool
    user_id: Optional[str]
    permissions: List[str]
    errors: List[str]
    metadata: Dict[str, Any]
```

## Repository Interfaces

### Base Repository Protocol

```python
from typing import Protocol, TypeVar, Generic

T = TypeVar('T')

class IRepository(Protocol, Generic[T]):
    async def create(self, entity: T) -> T:
        """Create new entity."""
        
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        
    async def update(self, id: str, entity: T) -> T:
        """Update entity."""
        
    async def delete(self, id: str) -> bool:
        """Delete entity."""
        
    async def list(self, filters: Optional[dict] = None) -> List[T]:
        """List entities with optional filters."""
```

### Prompt Repository Interface

```python
class IPromptRepository(Protocol):
    async def create_session(self, session: ImprovementSessionCreate) -> ImprovementSession:
        """Create improvement session."""
        
    async def get_session(self, session_id: str) -> Optional[ImprovementSession]:
        """Get session by ID."""
        
    async def get_sessions_by_user(self, user_id: str) -> List[ImprovementSession]:
        """Get all sessions for a user."""
        
    async def update_session(self, session_id: str, data: dict) -> ImprovementSession:
        """Update session data."""
```

### Rules Repository Interface

```python
class IRulesRepository(Protocol):
    async def get_rules(
        self, 
        filters: Optional[RuleFilter] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = False,
        limit: Optional[int] = None
    ) -> List[RuleMetadata]:
        """Get rules with filtering and sorting."""
        
    async def get_rule_by_id(self, rule_id: str) -> Optional[RuleMetadata]:
        """Get specific rule by ID."""
        
    async def create_rule_performance(self, performance: RulePerformanceCreate) -> RulePerformance:
        """Create rule performance record."""
        
    async def get_rule_performance(self, rule_id: str) -> List[RulePerformance]:
        """Get performance history for rule."""
```

### ML Repository Interface

```python
class IMLRepository(Protocol):
    async def store_training_data(self, data: TrainingData) -> bool:
        """Store ML training data."""
        
    async def get_training_data(
        self, 
        filters: Optional[dict] = None
    ) -> List[TrainingData]:
        """Retrieve training data."""
        
    async def cache_rule_intelligence(self, intelligence: List[RuleIntelligence]) -> bool:
        """Cache ML-generated rule intelligence."""
        
    async def get_rule_intelligence(self, rule_id: str) -> Optional[RuleIntelligence]:
        """Get cached intelligence for rule."""
```

## Application Services API

### PromptImprovementService

**Module**: `prompt_improver.core.services.prompt_improvement`

```python
class PromptImprovementService:
    async def improve_prompt(
        self,
        prompt: str,
        user_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        preferred_rules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Improve a prompt using intelligent rule selection."""
```

#### Response Format

```python
{
    "original_prompt": str,
    "improved_prompt": str,
    "applied_rules": List[Dict[str, Any]],
    "processing_time_ms": int,
    "session_id": str,
    "improvement_summary": Dict[str, Any],
    "confidence_score": float,
    "performance_data": List[Dict[str, Any]]
}
```

### RuleSelectionService

**Module**: `prompt_improver.core.services.rule_selection_service_clean`

```python
class CleanRuleSelectionService:
    async def get_optimal_rules(
        self,
        prompt_characteristics: Dict[str, Any],
        preferred_rules: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get optimal rules for prompt improvement."""
        
    async def get_active_rules(self) -> Dict[str, BasePromptRule]:
        """Get all active rules with caching."""
```

### AnalyticsService

**Module**: `prompt_improver.core.services.analytics_service`

```python
class AnalyticsService:
    async def get_performance_metrics(
        self,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        rule_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for rules."""
        
    async def calculate_improvement_trends(
        self,
        session_ids: List[str]
    ) -> Dict[str, Any]:
        """Calculate improvement trends over time."""
```

## ML Services API

### ML Event Bus

**Module**: `prompt_improver.core.events.ml_event_bus`

```python
from prompt_improver.core.events.ml_event_bus import get_ml_event_bus, MLEvent, MLEventType

event_bus = await get_ml_event_bus()

# Publish ML event
event = MLEvent(
    event_type=MLEventType.TRAINING_REQUEST,
    source="prompt_improvement_service",
    data={
        "operation": "optimize_rules",
        "features": feature_data,
        "effectiveness_scores": scores
    }
)

await event_bus.publish(event)
```

#### Event Types

```python
class MLEventType(str, Enum):
    TRAINING_REQUEST = "training_request"
    ANALYSIS_REQUEST = "analysis_request" 
    PERFORMANCE_METRICS_REQUEST = "performance_metrics_request"
    TRAINING_PROGRESS = "training_progress"
    SYSTEM_STATUS_REQUEST = "system_status_request"
    SHUTDOWN_REQUEST = "shutdown_request"
```

### Background Intelligence Processor

**Module**: `prompt_improver.ml.background.intelligence_processor`

```python
class IntelligenceProcessor:
    async def process_batch_intelligence(self) -> Dict[str, Any]:
        """Process batch ML intelligence operations."""
        
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
```

## Cache Services API

### Unified Cache Coordinator

**Module**: `prompt_improver.utils.unified_cache_coordinator`

```python
class UnifiedCacheCoordinator:
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 memory -> L2 Redis)."""
        
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache at all levels."""
        
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        
    async def clear_level(self, level: CacheLevel) -> bool:
        """Clear specific cache level."""
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
```

#### Cache Levels

```python
class CacheLevel(str, Enum):
    MEMORY = "memory"   # L1 cache
    REDIS = "redis"     # L2 cache
    ALL = "all"         # All levels
```

## Error Handling

### Service Exceptions

```python
class ServiceError(Exception):
    """Base service error."""
    pass

class DatabaseServiceError(ServiceError):
    """Database service specific errors."""
    pass

class SecurityServiceError(ServiceError):
    """Security service specific errors."""
    pass

class ValidationError(ServiceError):
    """Validation errors."""
    pass

class AuthenticationError(SecurityServiceError):
    """Authentication failures."""
    pass

class AuthorizationError(SecurityServiceError):
    """Authorization failures."""
    pass

class RateLimitError(SecurityServiceError):
    """Rate limiting errors."""
    pass
```

### Error Response Format

```python
{
    "error": {
        "type": str,      # Error type
        "message": str,   # Human readable message
        "code": str,      # Error code
        "details": dict,  # Additional error details
        "timestamp": str  # ISO timestamp
    },
    "request_id": str     # Request tracking ID
}
```

### Circuit Breaker Patterns

```python
class CircuitBreakerOpen(ServiceError):
    """Circuit breaker is open."""
    pass

class CircuitBreakerHalfOpen(ServiceError):
    """Circuit breaker is in half-open state."""
    pass
```

## Usage Examples

### Complete Service Setup

```python
async def setup_services():
    """Example service setup."""
    # Database services
    db_services = await get_database_services(ManagerMode.PRODUCTION)
    await db_services.initialize()
    
    # Security services
    security_services = await get_security_service_facade()
    await security_services.initialize_all_components()
    
    # Repository setup
    from prompt_improver.core.di.clean_container import get_clean_container
    container = await get_clean_container()
    
    # Register repositories
    container.register(IPromptRepository, PromptRepository(db_services))
    container.register(IRulesRepository, RulesRepository(db_services))
    
    # Application services
    prompt_repo = await container.get(IPromptRepository)
    rules_repo = await container.get(IRulesRepository)
    
    prompt_service = PromptImprovementService(
        prompt_repository=prompt_repo,
        rules_repository=rules_repo,
        user_feedback_repository=feedback_repo,
        ml_repository=ml_repo
    )
    
    return {
        "database": db_services,
        "security": security_services,
        "prompt_service": prompt_service,
        "container": container
    }
```

### API Request Flow

```python
async def handle_prompt_improvement_request(request_data: dict):
    """Example API request handling."""
    # 1. Security validation
    security_context = create_security_context(
        agent_id=request_data["user_id"],
        manager_mode=ManagerMode.PRODUCTION
    )
    
    security = await get_security_service_facade()
    validation = await security.validation
    
    # 2. Input validation
    validated_input = await validation.validate_input(
        request_data["prompt"],
        "api",
        security_context
    )
    
    # 3. Rate limiting check
    rate_limiter = await security.rate_limiting
    is_allowed = await rate_limiter.check_rate_limit(
        security_context,
        "prompt_improvement"
    )
    
    if not is_allowed:
        raise RateLimitError("Rate limit exceeded")
    
    # 4. Business logic
    services = await setup_services()
    result = await services["prompt_service"].improve_prompt(
        prompt=validated_input,
        user_context=request_data.get("context"),
        session_id=request_data.get("session_id")
    )
    
    # 5. Output validation
    validated_output = await validation.validate_output(
        result["improved_prompt"],
        security_context
    )
    
    return result
```

This API documentation provides clear interfaces and examples for working with the modernized service boundaries in the APES system.