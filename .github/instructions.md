# GitHub Copilot Instructions for APES (Adaptive Prompt Enhancement System)

## Project Overview

APES is a high-performance, real-time prompt enhancement system built with **Clean Architecture principles** and **modern 2025 best practices**. The system transforms prompts in <200ms using ML optimization and rule-based transformations, achieving 96%+ performance improvements through comprehensive architectural refactoring.

### Core Value Proposition
- **Clean Architecture** - Strict layer separation with dependency inversion
- **High Performance** - 96.67% cache hit rates, <2ms response times  
- **Service Consolidation** - Unified facades replacing 70+ overlapping services
- **Real Behavior Testing** - Testcontainers over mocks for authentic validation
- **Protocol-Based DI** - Type-safe dependency injection without framework coupling

## 🔧 DEVELOPMENT WORKFLOW

### Critical Development Commands

**Development Server** (Hot reload with <50ms HMR):
```bash
./scripts/dev-server.sh  # Multi-service orchestration with performance monitoring
```

**Testing** (Real behavior with testcontainers):
```bash
# Core test workflows - use tasks over raw commands
task run:tests                    # VS Code task: Run Tests (preferred)
task run:tests-coverage          # VS Code task: Run Tests with Coverage
./scripts/run_tests.sh           # Full real behavior test suite
./scripts/run_integration_tests.py  # Integration tests only
```

**Database Operations**:
```bash
task db:start    # VS Code task: Database: Start (Docker Compose)
task db:stop     # VS Code task: Database: Stop  
task db:migrate  # VS Code task: Database: Migrate
```

**Code Quality**:
```bash
task format      # VS Code task: Format Code (Ruff)
task lint        # VS Code task: Lint Code (Ruff + fix)
task typecheck   # VS Code task: Type Check (Pyright)
```

**MCP Server**:
```bash
./start_mcp_server.sh           # Production MCP server startup
python -m prompt_improver.mcp_server.mcp_server  # Direct MCP server launch
```

### Before Writing Code

1. **Search Existing Solutions**:
   ```bash
   rg "exact_function_name|exact_class_name" --type py
   rg "validate.*email|email.*validation" --type py  
   ```

2. **Verify Architecture Compliance**:
   - Check for existing facades/services
   - Ensure protocol-based interfaces
   - Validate dependency injection patterns

3. **Plan Service Boundaries**:
   - Single responsibility per service
   - Clear protocol definitions
   - Proper facade orchestration

## 🏗️ MANDATORY ARCHITECTURAL PATTERNS

### 1. Clean Architecture Implementation (ENFORCED)

**REQUIRED**: Strict layered separation with dependency inversion

```
┌─────────────────────────────────────────────────────┐
│  Presentation Layer (API/CLI/MCP)                   │  
├─────────────────────────────────────────────────────┤
│  Application Services (Workflow Orchestration)     │
├─────────────────────────────────────────────────────┤
│  Domain Services (Business Logic)                  │
├─────────────────────────────────────────────────────┤
│  Repository Interfaces (Data Contracts)            │
├─────────────────────────────────────────────────────┤
│  Infrastructure (Database/Cache/Monitoring)        │
└─────────────────────────────────────────────────────┘
```

**MANDATORY**:
- Dependencies point inward only (dependency inversion principle)
- Business logic isolated from infrastructure concerns
- Repository pattern for all data access
- Application services for workflow orchestration

**PROHIBITED**:
- Direct database imports in business logic
- Infrastructure dependencies in domain layer
- Circular dependencies between layers

### 2. Service Facade Pattern (REQUIRED)

**MANDATORY**: Consolidate related services into unified facades with internal component specialization

```python
@runtime_checkable
class ServiceFacadeProtocol(Protocol):
    """Protocol defining facade interface"""
    async def primary_operation(self, data: Any) -> Dict[str, Any]: ...

class ServiceFacade(ServiceFacadeProtocol):
    """Facade implementation with internal components"""
    
    def __init__(
        self,
        component_a: ComponentAProtocol,
        component_b: ComponentBProtocol,
        component_c: ComponentCProtocol
    ):
        self.component_a = component_a
        self.component_b = component_b
        self.component_c = component_c
    
    async def primary_operation(self, data: Any) -> Dict[str, Any]:
        # Orchestrate internal components
        result_a = await self.component_a.process(data)
        result_b = await self.component_b.enhance(result_a)
        return await self.component_c.finalize(result_b)
```

**Service Naming Convention**:
- **\*Facade**: Unified interfaces consolidating multiple services
- **\*Service**: Business logic and domain operations  
- **\*Manager**: Infrastructure and resource management

### 3. Protocol-Based Dependency Injection (MANDATORY)

**REQUIRED**: All service dependencies via constructor using `typing.Protocol`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SessionManagerProtocol(Protocol):
    """Database session management protocol."""
    async def get_session(self) -> AsyncSession: ...
    async def close_session(self, session: AsyncSession) -> None: ...

class BusinessService:
    """Service using protocol-based DI"""
    
    def __init__(self, session_manager: SessionManagerProtocol):
        self.session_manager = session_manager  # Depends on abstraction
    
    async def process_data(self, data: Any) -> Dict[str, Any]:
        async with self.session_manager.get_session() as session:
            # Business logic here
            return result
```

**Requirements**:
- All protocols MUST be marked `@runtime_checkable`
- Constructor injection pattern for explicit dependencies
- Interface definition in domain layer, implementation in infrastructure
- Protocol-first development: define interface before implementation

### 4. Multi-Level Caching Architecture (MANDATORY)

**PERFORMANCE REQUIREMENT**: >80% cache hit rates (96.67% achieved)

```python
class CacheServiceFacade:
    def __init__(
        self,
        l1_cache: L1MemoryCacheService,     # ~0.001ms
        l2_cache: L2RedisCacheService,      # ~1-5ms  
        l3_cache: L3DatabaseCacheService    # ~10-50ms
    ):
        self.l1, self.l2, self.l3 = l1_cache, l2_cache, l3_cache
    
    async def get(self, key: str) -> Optional[Any]:
        # L1: Memory cache check first
        if value := await self.l1.get(key):
            return value
            
        # L2: Redis cache fallback  
        if value := await self.l2.get(key):
            await self.l1.set(key, value)  # Warm L1
            return value
            
        # L3: Database fallback
        if value := await self.l3.get(key):
            await self.l2.set(key, value)  # Warm L2
            await self.l1.set(key, value)  # Warm L1
            return value
        
        return None
```

**Performance Targets**:
- L1 Cache (Memory): <1ms operations
- L2 Cache (Redis): <10ms operations  
- L3 Cache (Database): <50ms operations
- Overall hit rate: >96% (achieved: 96.67%)

### 5. Real Behavior Testing (MANDATORY)

**NO MOCKS POLICY**: Use real services in sandboxed environments

```python
@pytest.fixture(scope="session")
async def postgres_container():
    """Start real PostgreSQL container for testing."""
    container = PostgresContainer("postgres:15-alpine")
    with container as postgres:
        yield postgres

@pytest.fixture
async def real_database_session(postgres_container):
    """Create real database session for testing."""
    engine = create_async_engine(postgres_container.get_connection_url())
    async with AsyncSession(engine) as session:
        yield session
```

**Testing Principles**:
- Real PostgreSQL/Redis containers via testcontainers
- Test actual error conditions, not simulated ones  
- Validate real constraints and schema violations
- Integration tests prioritized over unit tests (60-70% vs 20-30%)

## 🚫 PROHIBITED PATTERNS

### Anti-Patterns to Avoid

**FORBIDDEN**:
- Direct database imports in business logic (`from prompt_improver.database import get_session`)
- God objects >500 lines (Single Responsibility violation)
- Mock-based integration testing for database/cache operations
- Circular imports between architectural layers
- Hardcoded configuration values
- Backwards compatibility layers (clean break strategy)

### Import Restrictions

```python
# ❌ PROHIBITED: Direct infrastructure imports in business logic
from prompt_improver.database import get_session

# ✅ REQUIRED: Protocol-based dependency injection
def __init__(self, session_manager: SessionManagerProtocol):
    self.session_manager = session_manager
```

## 🎯 PERFORMANCE STANDARDS

### Response Time Requirements

```yaml
Performance Targets:
  P95: <100ms (all endpoints)
  Critical Paths: <2ms (achieved)
  Cache Operations:
    L1 Memory: <1ms (achieved: 0.001ms)
    L2 Redis: <10ms (achieved: 0.095ms)  
    L3 Database: <50ms (achieved: 25ms)

Memory Usage:
  Range: 10-1000MB
  Cache Efficiency: <1KB per entry (achieved: 358 bytes)

Cache Performance:
  Hit Rate: >80% required (achieved: 96.67%)
  Throughput: >500 req/s per service
```

### Achieved Performance Results

- **Analytics Service**: 114x throughput improvement
- **MCP Server**: 4.4x improvement (543μs → 123μs)
- **Prompt Processing**: 96.6% improvement
- **Overall Response Times**: 50-200ms → <2ms

## 🧪 TESTING STANDARDS (2025)

### Modern Test Pyramid

```
    /\      E2E Tests (5-10%)
   /  \     Critical user journeys
  /____\
 /      \   Integration Tests (60-70%)
/        \  Real behavior, contract testing
\________/
\        /  Unit Tests (20-30%)
 \______/   Business logic validation
```

### Real Behavior Testing Implementation

**DO**:
- ✅ Use real PostgreSQL containers for database testing
- ✅ Trigger actual error conditions instead of simulating them
- ✅ Test real constraints and schema validation
- ✅ Validate actual retry and circuit breaker behavior
- ✅ Use session-scoped containers to minimize startup overhead

**DON'T**:
- ❌ Mock database operations - use real databases
- ❌ Use SQLite as PostgreSQL substitute - use PostgreSQL containers
- ❌ Skip cleanup - ensure containers properly managed
- ❌ Test only happy paths - trigger real error scenarios

### Testcontainer Implementation

```python
@pytest.fixture(scope="session")
async def postgres_container():
    """Start real PostgreSQL container for testing."""
    container = PostgresContainer("postgres:15-alpine")
    with container as postgres:
        yield postgres

@pytest.fixture
async def real_database_session(postgres_container):
    """Create real database session for testing."""
    engine = create_async_engine(postgres_container.get_connection_url())
    async with AsyncSession(engine) as session:
        yield session
```

### Performance Testing Standards

**MANDATORY**: Include performance assertions in all service tests

```python
async def test_service_performance():
    import time
    start_time = time.perf_counter()
    result = await service.process_data(test_data)
    duration = time.perf_counter() - start_time
    
    # Assert SLA compliance
    assert duration < 0.100, f"Service took {duration:.3f}s, exceeds 100ms SLA"
    assert result is not None
    
    # Memory efficiency check
    import psutil
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    assert memory_usage < 50, f"Memory usage {memory_usage:.1f}MB exceeds 50MB limit"
```

## 🐍 MODERN PYTHON FEATURES (2025+)

### Contemporary Syntax and Patterns

**REQUIRED**: Use latest Python 3.12+ features for new code

```python
# ✅ Modern generic syntax (PEP 695) - 2025 Standard
class ServiceFacade[T]:
    def process(self, data: T) -> T: ...

class Repository[K, V]:
    def get(self, key: K) -> Optional[V]: ...
    def set(self, key: K, value: V) -> None: ...

# ✅ Modern type alias syntax
type ServiceConfig = Dict[str, Any]
type ProcessingResult = Dict[str, Union[str, int, bool]]
type CacheKey = str
type CacheValue = Any

# ✅ Pattern matching for error handling (Python 3.10+)
match result:
    case Success(data):
        return data
    case ValidationError(msg):
        raise ProcessingError(f"Validation failed: {msg}")
    case NetworkError(code, msg):
        raise ServiceUnavailableError(f"Network error {code}: {msg}")
    case _:
        raise UnknownError("Unexpected result type")

# ✅ Modern exception groups (Python 3.11+)
try:
    await process_multiple_items(items)
except* ValidationError as eg:
    for error in eg.exceptions:
        logger.error(f"Validation error: {error}")
except* NetworkError as eg:
    for error in eg.exceptions:
        logger.error(f"Network error: {error}")
```

### Advanced Type Annotations

```python
# ✅ Self type for builder patterns
from typing import Self

class ServiceBuilder:
    def with_cache(self, cache: CacheServiceFacade) -> Self:
        self.cache = cache
        return self
    
    def with_database(self, db: DatabaseManager) -> Self:
        self.database = db
        return self

# ✅ Literal types for strict validation
from typing import Literal

CacheLevel = Literal["L1", "L2", "L3"]
ProcessingMode = Literal["fast", "thorough", "experimental"]

def get_cache(level: CacheLevel) -> CacheServiceProtocol:
    match level:
        case "L1":
            return l1_memory_cache
        case "L2":
            return l2_redis_cache
        case "L3":
            return l3_database_cache
```

## 📁 PROJECT STRUCTURE

### Core Directories

```
src/prompt_improver/
├── core/                      # Domain layer (business logic)
│   ├── facades/              # Unified interfaces  
│   ├── protocols/            # Interface definitions
│   └── di/                   # Dependency injection containers
├── services/                 # Service facades
│   ├── prompt/facade.py      # PromptServiceFacade  
│   ├── analytics/facade.py   # AnalyticsServiceFacade
│   ├── security/facade.py    # SecurityServiceFacade
│   └── cache/facade.py       # CacheServiceFacade
├── application/              # Application services (workflow)
├── repositories/             # Data access layer
├── infrastructure/           # External concerns
├── mcp_server/              # MCP server implementation
│   ├── server.py            # Main server orchestrator
│   ├── tools.py             # Tool registration & handlers
│   ├── resources.py         # Resource registration & providers
│   ├── lifecycle.py         # Server lifecycle management
│   ├── security.py          # Security wiring & authentication
│   └── middleware.py        # Security middleware stack
└── database/                # Database migrations & schema

scripts/
├── dev-server.sh            # Development server with hot reload
├── run_tests.sh             # Real behavior test suite
├── run_integration_tests.py # Integration test runner
└── validate_*.py            # Architecture validation scripts

tests/
├── integration/             # Real behavior tests (60-70%)
│   ├── real_behavior_testing/
│   └── containers/          # Testcontainer implementations
├── unit/                   # Business logic tests (20-30%)
└── performance/            # Performance benchmarks
```

### Key Entry Points

- **MCP Server**: `src/prompt_improver/mcp_server/server.py` (APESMCPServer)
- **Dev Server**: `scripts/dev-server.sh` (hot reload with performance monitoring)
- **Main App**: `src/prompt_improver/main.py` (application entry point)
- **Service Facades**: `src/prompt_improver/services/*/facade.py`

### File Naming Conventions

- **Protocols**: `*_protocol.py` (interfaces)
- **Facades**: `facade.py` (unified interfaces)
- **Services**: `*_service.py` (business logic)
- **Managers**: `*_manager.py` (infrastructure)
- **Tests**: `test_*_real_behavior.py` (integration), `test_*.py` (unit)

## 🔧 CODING STANDARDS

### Type Hints and Documentation

```python
from typing import Protocol, Optional, Dict, Any, List
from typing import runtime_checkable

@runtime_checkable
class ServiceProtocol(Protocol):
    """Service protocol with comprehensive type hints."""
    
    async def process_data(
        self, 
        data: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process data with optional configuration.
        
        Args:
            data: Input data dictionary
            options: Optional processing configuration
            
        Returns:
            Processed data dictionary
            
        Raises:
            ValidationError: If data validation fails
            ProcessingError: If processing encounters errors
        """
        ...
```

**Requirements**:
- All functions MUST have type hints
- Use Google-style docstrings
- Protocol interfaces marked `@runtime_checkable`
- Comprehensive error handling with correlation tracking

### Error Handling Patterns

```python
from prompt_improver.core.exceptions import (
    ValidationError,
    ProcessingError,
    ServiceUnavailableError
)

class ServiceImplementation:
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Validation
            if not self._validate_input(data):
                raise ValidationError("Input validation failed", data=data)
            
            # Processing
            result = await self._process_internal(data)
            return result
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert to domain exception
            raise ProcessingError(f"Processing failed: {e}") from e
```

## 🎯 MCP SERVER SPECIFICS

### FastMCP Integration

The APES MCP Server implements 2025 FastMCP enhancements:

```python
# Middleware stack (auto-initialized)
server = APESMCPServer()
# Includes: Error Handling → Rate Limiting → Timing → Logging

# Progress-aware tools with Context support
async def improve_prompt(prompt: str, ctx: Context) -> str:
    await ctx.report_progress(0, "Starting validation")
    # ... processing with progress updates
    await ctx.report_progress(100, "Complete")
    return enhanced_prompt
```

### Resource Templates

```bash
# Wildcard resource access patterns
apes://sessions/{session_id}/history
apes://rules/{rule_category}/performance  
apes://metrics/{metric_type}
```

## 🎯 PROMPT IMPROVEMENT GUIDELINES

When suggesting code improvements:

1. **Always check existing facades first** - don't duplicate functionality
2. **Follow protocol-based patterns** - define interfaces before implementations  
3. **Implement real behavior tests** - use testcontainers, not mocks
4. **Validate performance requirements** - include timing assertions
5. **Maintain architectural boundaries** - respect clean architecture layers
6. **Use modern Python patterns** - f-strings, pathlib, async/await
7. **Document decisions** - comprehensive docstrings and type hints

## 📚 REFERENCE DOCUMENTATION

- **Architecture**: `docs/architecture/ARCHITECTURE_PATTERNS_2025_UPDATED.md`
- **Clean Architecture**: `docs/architecture/CLEAN_ARCHITECTURE_PATTERNS_2025.md`
- **Design Decisions**: `docs/DESIGN_DECISIONS_2025.md`
- **Testing Guide**: `docs/developer/REAL_BEHAVIOR_TESTING_2025.md`
- **Performance Standards**: `docs/cache/Cache_Performance_Monitoring_Guide.md`
- **Migration Guide**: `docs/MIGRATION_GUIDE.md`

---

**Remember**: This system prioritizes **performance**, **maintainability**, and **architectural integrity**. Every suggestion should align with these 2025+ best practices and the established patterns that have achieved 96%+ performance improvements.
