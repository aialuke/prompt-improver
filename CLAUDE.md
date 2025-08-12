# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Principles

**Search existing solutions before creating new code:**
- Direct name search: `rg "exact_function_name|exact_class_name" --type py`
- Pattern search: `rg "validate.*email|email.*validation" --type py`
- Import search: `rg "from.*import.*{similar}|import.*{similar}" --type py`

**Apply Clean Architecture & SOLID principles:**
- Use repository patterns with protocol-based DI
- Extend existing code when ≤3 parameters, ≤50 lines, same domain
- Create new code when different domain or would break contracts  
- Refactor first when existing code unclear or violates SOLID principles

**Code deletion verification:**
- `rg "ExactItemName" . --type py -n` (verify zero usage)
- `rg "from.*import.*ExactItemName|import.*ExactItemName" . --type py -n`

**Modernization patterns (2025):**
- Python: %formatting→f-strings, os.path→pathlib, dict.keys()→dict iteration
- Type hints: Any→specific types, missing annotations→explicit typing
- Async: callback patterns→async/await, threading→asyncio where appropriate
- Architecture: Direct database imports→repository protocols
- Services: Multiple managers→unified service facades

**Service Architecture Patterns (2025 Refactoring):**
- **Database**: Use repository pattern with protocol-based interfaces, zero direct database imports in business logic
- **Security**: SecurityServiceFacade with component architecture (authentication, authorization, validation, crypto)
- **Analytics**: AnalyticsServiceFacade with 114x performance improvement and 96.67% cache hit rates
- **ML**: MLModelServiceFacade replacing 2,262-line god object with 6 focused services
- **Application Layer**: Use application services for workflow orchestration between presentation and domain
- **Caching**: Multi-level strategy (L1 Memory, L2 Redis, L3 Database) achieving <2ms response times
- **Error Handling**: Structured exception hierarchy with correlation tracking across all layers
- **Configuration**: Centralized with Pydantic validation and environment-specific profiles
- **Testing**: Real behavior testing with testcontainers, categorized boundaries (unit/integration/contract/e2e)

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

**Delegate to specialized agents for domain expertise:**
- **code-reviewer**: Code review, refactoring, technical debt analysis, quality assessment
- **system-reliability-engineer**: Monitoring, health checks, incident response, observability  
- **database-specialist**: Query optimization, schema design, migrations, connection pooling
- **security-architect**: Authentication, authorization, vulnerability assessment, security reviews
- **ml-orchestrator**: ML training issues, pipeline design, feature engineering, model optimization
- **performance-engineer**: Slow performance, profiling, load testing, bottleneck analysis
- **infrastructure-specialist**: Testcontainers, CI/CD, development environments, testing infrastructure

**Delegation triggers:** review, optimize, secure, monitor, test, deploy, analyze, refactor, troubleshoot

**External tool integration:**
- Search previous solutions via memory tools first
- Use Context7 for external dependencies and best practices
- Document decisions and patterns for future reference
- Fall back to manual analysis when MCP unavailable

**Parallel execution patterns:** Always batch independent operations - file reads, searches, validations. Default to parallel unless sequential dependencies exist.

## Extended Thinking & Context Management

**Use thinking blocks for complex reasoning (54% performance improvement):**
- After tool results: Reflect and determine next steps
- Before complex decisions: Multi-step reasoning and planning
- During problem decomposition: Breaking down complex tasks
- For iterative improvement: Analyzing and refining approaches

**Thinking triggers:** analyze, plan, evaluate, reason, decompose, optimize, strategize, debug

**Context optimization:**
- Use subagents to preserve main context during deep investigations
- Follow "Explore → Plan → Code → Commit" workflow for complex features
- Clean up temporary files/scripts at task completion
- Use /clear or /compact for context management when needed

**Iterative development:** Write tests first, use screenshots for UI/UX feedback, remember "Claude's outputs improve significantly with iteration"

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