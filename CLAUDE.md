# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Principles

**Search existing solutions before creating new code:**
- Direct name search: `rg "exact_function_name|exact_class_name" --type py`
- Pattern search: `rg "validate.*email|email.*validation" --type py`
- Import search: `rg "from.*import.*{similar}|import.*{similar}" --type py`

**Apply Clean Architecture & SOLID principles (ENFORCED 2025):**
- **MANDATORY**: Use repository patterns with protocol-based DI (SessionManagerProtocol)
- **FORBIDDEN**: Direct database imports in business logic (`from prompt_improver.database import get_session`)
- **REQUIRED**: Service naming convention (*Facade, *Service, *Manager)
- **ENFORCED**: No classes >500 lines (god object elimination)
- **MANDATORY**: Real behavior testing with testcontainers (no mocks for external services)

**Code deletion verification:**
- `rg "ExactItemName" . --type py -n` (verify zero usage)
- `rg "from.*import.*ExactItemName|import.*ExactItemName" . --type py -n`

**Modernization patterns (2025):**
- Python: %formatting→f-strings, os.path→pathlib, dict.keys()→dict iteration
- Type hints: Any→specific types, missing annotations→explicit typing
- Async: callback patterns→async/await, threading→asyncio where appropriate
- Architecture: Direct database imports→repository protocols
- Services: Multiple managers→unified service facades

**Service Architecture Patterns (2025 COMPLETED TRANSFORMATION):**
- **Database**: Repository pattern with SessionManagerProtocol, ZERO direct imports (ACHIEVED)
- **Security**: SecurityServiceFacade consolidating authentication, authorization, validation, crypto
- **Analytics**: AnalyticsServiceFacade with 114x performance improvement and 96.67% cache hit rates
- **ML**: MLModelServiceFacade replacing 1,043-line god object with 5 focused services (COMPLETED)
- **Prompt Services**: PromptServiceFacade replacing 1,500+ line god object with 3 focused services (COMPLETED)
- **Monitoring**: UnifiedMonitoringFacade consolidating 8+ health checkers (<25ms operations)
- **Database Pool**: PostgreSQLPoolManager decomposed into 3 components (942 → 3×<400 lines)
- **Application Layer**: Application services for workflow orchestration (Clean Architecture)
- **Caching**: Multi-level strategy (L1 Memory, L2 Redis, L3 Database) achieving <2ms response times
- **Testing**: Real behavior testing with testcontainers (87.5% validation success achieved)

**REFERENCE**: See docs/architecture/ARCHITECTURE_PATTERNS_2025_UPDATED.md for complete patterns including god object decomposition

## God Object Decomposition (August 2025 COMPLETED)

**Clean Break Strategy - Zero Backwards Compatibility:**
- PromptServiceFacade: 1,500+ line god object → 3 focused services (PromptAnalysisService, RuleApplicationService, ValidationService)
- **ENFORCED**: No classes >500 lines (Single Responsibility Principle)
- **MANDATORY**: Facade pattern for unified interfaces with internal component specialization
- **REQUIRED**: Protocol-based dependency injection throughout decomposed services
- **VALIDATED**: Real behavior testing confirms architectural integrity

**Import Migration Pattern:**
```python
# OLD (REMOVED):
from prompt_improver.core.services.prompt_improvement import PromptImprovementService

# NEW (REQUIRED):
from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService
```

**Quality Gates Enforced:**
- Single responsibility maintained across all decomposed services
- Protocol-based interfaces for all service communication
- Zero circular imports through architectural cleanup
- Performance maintained (0.36ms facade coordination, 0.000413ms L1 cache)
- Complete test coverage with real behavior validation

## Evidence-Based Analysis

**When user requests:** verify, check, analyze, review, evaluate, assess, think, or "think hard"

**Required format:** [Finding]: [Evidence] 📍 Source: file:line

**Include:** Confidence level (HIGH/MEDIUM/LOW), scope examined ("X/Y files"), methodology used

**Validation levels:**
- Basic: Syntax, imports, type safety, functionality
- Integration: Database connections, external services, API contracts
- Performance: Response time 0.1-500ms, memory 10-1000MB, realistic test scenarios

**Always use parallel tool execution** - batch Read/Grep/Glob/Bash calls simultaneously for 3-10x performance improvement.

## Tool & Agent Usage

**Core Infrastructure Agents:**
- **database-specialist**: PostgreSQL/JSONB optimization, custom SQL migrations, query performance (APES schema)
- **ml-orchestrator**: ML pipelines (5 services), model registry (@champion/@production), feature engineering
- **performance-engineer**: Cross-cutting performance, SLO monitoring, cache optimization (96.67% hit rate)
- **security-architect**: Security policies, OWASP 2025, authentication/authorization design
- **infrastructure-specialist**: Docker/testcontainers, CI/CD, real behavior testing (no mocks)

**Specialized Domain Agents:**
- **data-pipeline-specialist**: ETL processes, analytics pipelines, data transformation
- **api-design-specialist**: FastAPI design, OpenAPI docs, REST architecture
- **monitoring-observability-specialist**: OpenTelemetry, distributed tracing, SLO monitoring
- **testing-strategy-specialist**: Real behavior testing, testcontainers, quality assurance
- **configuration-management-specialist**: Environment configs, settings management
- **documentation-specialist**: Technical docs, API documentation, ADRs

**Automatic Agent Selection Triggers:**
Database: query|schema|migration|PostgreSQL|SQL → database-specialist
ML/AI: model|training|feature|pipeline|ML → ml-orchestrator
Performance: slow|optimize|bottleneck|cache → performance-engineer
Security: auth|security|vulnerability|encrypt → security-architect
Infrastructure: docker|container|CI/CD|deploy → infrastructure-specialist
API: endpoint|FastAPI|OpenAPI|REST → api-design-specialist
Testing: test|testcontainer|quality → testing-strategy-specialist
Data: ETL|analytics|data processing → data-pipeline-specialist
Monitoring: observability|tracing|metrics → monitoring-observability-specialist
Config: config|environment|settings → configuration-management-specialist
Docs: documentation|ADR|technical writing → documentation-specialist

**Delegation Flow:**
performance-engineer → database-specialist (query optimization)
performance-engineer → ml-orchestrator (ML performance)
security-architect → infrastructure-specialist (security tool deployment)
ml-orchestrator → infrastructure-specialist (deployment infrastructure)
infrastructure-specialist → database-specialist (database infrastructure)

**APES Architecture Integration:**
- PostgreSQL 15+ with JSONB optimization (database-specialist)
- Custom SQL migrations in database/migrations/ (database-specialist)
- Testcontainers for real behavior testing (infrastructure-specialist + testing-strategy-specialist)
- Multi-level caching L1/L2/L3 achieving <2ms response (performance-engineer)
- Model registry aliases @champion/@production/@challenger (ml-orchestrator)

**External tool integration:**
- Search previous solutions via memory tools first
- Use Context7 for external dependencies and best practices
- Document decisions and patterns for future reference
- Fall back to manual analysis when MCP unavailable

**Parallel execution patterns:** Always batch independent operations - file reads, searches, validations. Default to parallel unless sequential dependencies exist.

**Comprehensive Documentation:** → docs/architecture/CLAUDE_CODE_AGENT_BEST_PRACTICES.md

## Extended Thinking & Context Management

**Use thinking blocks for complex reasoning (54% performance improvement):**
- After tool results: Reflect and determine next steps
- Before complex decisions: Multi-step reasoning and planning
- During problem decomposition: Breaking down complex tasks
- For iterative improvement: Analyzing and refining approaches

**Thinking triggers:** analyze, plan, evaluate, reason, decompose, optimize, strategize, debug

## Error Handling & Correction

**Priority hierarchy:** Security > Code Correctness > Type Safety > Code Quality > Test Passing

**Root cause analysis over quick fixes:**
- Investigate before applying fixes
- Document error type, location, specific cause
- Re-examine with different methodology when needed

**Development vs Production:**
- Temporary fixes acceptable in development with TODO documentation
- Production code requires proper fixes: import types correctly, fix type mismatches, maintain explicit typing
- Avoid masking issues: no # type: ignore, Any types, or deletion without understanding

**Always validate claims with concrete evidence before marking complete.**

## Architectural Compliance (2025 Standards)

**MANDATORY Patterns - Must Follow:**
- **Clean Architecture**: Strict layer separation (Presentation → Application → Domain → Repository → Infrastructure)
- **Repository Pattern**: All data access through protocol-based repository interfaces, zero database imports in business logic
- **Protocol-Based DI**: Use typing.Protocol for interfaces, constructor injection for dependencies
- **Service Facades**: Consolidate related functionality into unified facades with internal components
- **Application Services**: Business workflow orchestration between presentation and domain layers
- **Multi-Level Caching**: L1 (Memory) + L2 (Redis) + L3 (Database) for performance optimization
- **Structured Error Handling**: Exception hierarchy with correlation tracking, decorators for error propagation
- **Real Behavior Testing**: Integration tests use testcontainers, no mocks for external services

**PROHIBITED Patterns - Must Avoid:**
- **Direct Database Access**: From service or presentation layers (use repositories only)
- **God Objects**: Classes >500 lines (split into focused services)
- **Infrastructure in Core**: Database/cache imports in business logic
- **Service Proliferation**: Multiple overlapping services (consolidate into facades)
- **Hardcoded Values**: Configuration must be externalized with environment variables
- **Backwards Compatibility Layers**: Clean break strategy to eliminate technical debt
- **Mock Integration Tests**: Use real services via testcontainers for integration validation

**Performance Requirements:**
- **Response Times**: P95 <100ms for endpoints, <2ms achieved on critical paths
- **Cache Performance**: >80% hit rates (96.67% achieved)
- **Memory Usage**: 10-1000MB range maintained
- **Test Coverage**: 85%+ on service boundaries