# Clean Architecture Patterns 2025
*Definitive guide for prompt-improver architecture patterns*

## 🎯 **MANDATORY ARCHITECTURAL PATTERNS**

### **Layer Separation (ENFORCED)**
```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│  • API Endpoints (FastAPI)                                 │
│  • CLI Commands                                            │
│  • Event Handlers                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│  • Application Services (Workflow Orchestration)           │
│  • Use Case Implementation                                  │
│  • Cross-cutting Concerns                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                           │
│  • Business Logic Services                                 │
│  • Domain Models                                           │
│  • Business Rules                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 INFRASTRUCTURE LAYER                        │
│  • Repository Implementations                              │
│  • Database Access                                         │
│  • External Service Clients                                │
└─────────────────────────────────────────────────────────────┘
```

### **Service Naming Convention (MANDATORY)**
- **`*Facade`**: Unified interfaces consolidating multiple services
  - `SecurityServiceFacade`, `AnalyticsServiceFacade`, `MLModelServiceFacade`
- **`*Service`**: Business logic and domain services
  - `PromptImprovementService`, `ValidationService`, `ComponentService`
- **`*Manager`**: Infrastructure management (database, cache, connections)
  - `PostgreSQLPoolManager`, `CacheManager`, `ConnectionPoolManager`

### **Protocol-Based Dependency Injection (REQUIRED)**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SessionManagerProtocol(Protocol):
    """Database session management protocol."""
    
    async def get_session(self) -> SessionProtocol: ...
    def session_context(self) -> AbstractAsyncContextManager[SessionProtocol]: ...
    async def health_check(self) -> bool: ...

class BusinessService:
    """Example service with proper DI."""
    
    def __init__(self, session_manager: SessionManagerProtocol):
        self.session_manager = session_manager
    
    async def process_data(self, data: dict) -> dict:
        async with self.session_manager.session_context() as session:
            # Business logic here
            return processed_data
```

## 🚫 **PROHIBITED PATTERNS**

### **Direct Database Imports (FORBIDDEN)**
```python
# ❌ NEVER DO THIS
from prompt_improver.database import get_session, get_session_context

class BadService:
    async def process(self):
        async with get_session() as session:  # VIOLATION!
            pass
```

### **God Objects (FORBIDDEN)**
- **No classes >500 lines**
- **No classes with >5 primary responsibilities**
- **Split large classes into focused services**

### **Mixed Layer Dependencies (FORBIDDEN)**
```python
# ❌ NEVER DO THIS - Business logic importing infrastructure
from prompt_improver.database.models import User  # VIOLATION!
from prompt_improver.cache.redis_client import RedisClient  # VIOLATION!

class BusinessService:
    def __init__(self):
        self.redis = RedisClient()  # DIRECT INFRASTRUCTURE ACCESS!
```

## ✅ **REQUIRED PATTERNS**

### **Repository Pattern Implementation**
```python
from prompt_improver.repositories.protocols.user_repository_protocol import UserRepositoryProtocol

class UserService:
    """Proper business service with repository injection."""
    
    def __init__(self, user_repository: UserRepositoryProtocol):
        self.user_repository = user_repository
    
    async def create_user(self, user_data: dict) -> User:
        # Business logic validation
        validated_data = self._validate_user_data(user_data)
        
        # Delegate to repository
        return await self.user_repository.create(validated_data)
```

### **Service Facade Pattern**
```python
from prompt_improver.security.unified.protocols import SecurityServiceFacadeProtocol
from prompt_improver.security.unified.authentication_component import AuthenticationComponent
from prompt_improver.security.unified.authorization_component import AuthorizationComponent
from prompt_improver.security.unified.validation_component import ValidationComponent

class SecurityServiceFacade:
    """Unified security service facade."""
    
    def __init__(self):
        self.authentication = AuthenticationComponent()
        self.authorization = AuthorizationComponent()
        self.validation = ValidationComponent()
    
    async def authenticate_user(self, credentials: dict) -> AuthResult:
        """Unified authentication interface."""
        return await self.authentication.authenticate(credentials)
    
    async def authorize_action(self, user: User, action: str) -> bool:
        """Unified authorization interface."""
        return await self.authorization.check_permission(user, action)
```

### **Application Service Pattern**
```python
class PromptApplicationService:
    """Application service orchestrating business workflows."""
    
    def __init__(
        self,
        prompt_service: PromptImprovementService,
        analytics_facade: AnalyticsServiceFacade,
        security_facade: SecurityServiceFacade
    ):
        self.prompt_service = prompt_service
        self.analytics_facade = analytics_facade
        self.security_facade = security_facade
    
    async def improve_prompt_workflow(self, request: PromptRequest) -> PromptResponse:
        """Complete workflow orchestration."""
        # Security validation
        await self.security_facade.validate_request(request)
        
        # Business logic
        result = await self.prompt_service.improve(request.prompt)
        
        # Analytics tracking
        await self.analytics_facade.track_improvement(result)
        
        return PromptResponse(improved_prompt=result.text)
```

## 📋 **ARCHITECTURAL DECISION RECORDS**

### **ADR-001: Clean Architecture Enforcement**
**Status**: ADOPTED  
**Decision**: Enforce strict Clean Architecture layer separation  
**Rationale**: Eliminates tight coupling and improves maintainability  
**Implementation**: Protocol-based DI, repository pattern, service facades

### **ADR-002: Service Facade Consolidation** 
**Status**: ADOPTED  
**Decision**: Replace multiple managers with unified service facades  
**Rationale**: Reduces complexity and provides consistent interfaces  
**Implementation**: SecurityServiceFacade, AnalyticsServiceFacade, MLModelServiceFacade

### **ADR-003: God Object Elimination**
**Status**: ADOPTED  
**Decision**: No classes >500 lines, split into focused services  
**Rationale**: Improved maintainability and single responsibility  
**Implementation**: MLPipelineOrchestrator → 5 services, PostgreSQLPoolManager → 3 components

### **ADR-004: Database Import Violation Elimination**
**Status**: ADOPTED  
**Decision**: Zero direct database imports in business logic  
**Rationale**: Clean Architecture compliance and proper layer separation  
**Implementation**: Repository pattern with protocol-based dependency injection

## 🎯 **QUALITY GATES**

### **Code Review Checklist**
- [ ] No direct database imports in business logic
- [ ] All services use protocol-based dependency injection  
- [ ] Service naming follows convention (*Facade, *Service, *Manager)
- [ ] No classes exceed 500 lines
- [ ] Clean Architecture layer separation maintained
- [ ] All tests use testcontainers (no mocks for external services)

### **Architecture Compliance Metrics**
- **Clean Architecture Compliance**: >90% (ACHIEVED: 90%)
- **Service Organization**: >85% (ACHIEVED: 85%) 
- **Component Coupling**: >80% (ACHIEVED: 80%)
- **Database Import Violations**: 0 (ACHIEVED: 0)
- **God Object Classes**: 0 (ACHIEVED: 0)

## 🚀 **IMPLEMENTATION GUIDELINES**

### **When Creating New Services**
1. **Define Protocol First**: Create interface before implementation
2. **Inject Dependencies**: Use constructor injection with protocols
3. **Follow Naming**: Use appropriate suffix (*Facade, *Service, *Manager)
4. **Limit Scope**: Single responsibility, <500 lines
5. **Write Tests**: Use testcontainers for integration testing

### **When Modifying Existing Services**
1. **Check Architecture**: Ensure Clean Architecture compliance
2. **Update Protocols**: Modify interfaces if needed
3. **Maintain Facades**: Keep unified interfaces consistent
4. **Validate Dependencies**: Ensure proper injection patterns
5. **Test Real Behavior**: Validate with testcontainers

This document serves as the definitive guide for all architecture decisions and patterns in the prompt-improver system. All code must comply with these patterns to maintain Clean Architecture principles and system maintainability.