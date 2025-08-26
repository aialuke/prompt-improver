# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Principles

**Pragmatic Problem Validation (FIRST STEP):**
- **Is this a real problem in production?** Theory loses to practice, every time
- **How many users actually encounter this?** Quantify impact before solving
- **Does complexity match severity?** Don't over-engineer edge cases
- **Can we measure this problem?** If not measurable, likely not real

**Search existing solutions before creating new code (AFTER validation):**
- Direct name search: `rg "exact_function_name|exact_class_name" --type py`
- Pattern search: `rg "validate.*email|email.*validation" --type py`
- Import search: `rg "from.*import.*{similar}|import.*{similar}" --type py`

**For Complex Analysis - Multi-Method Discovery:**
- **Filesystem**: `find . -name "*target*.py"` (comprehensive first)
- **Content**: `rg "pattern" --type py` (cross-validate findings)
- **Challenge assumptions** about file locations/patterns
- **Add 20% buffer** to scope estimates

**Apply Clean Architecture & SOLID principles (ENFORCED 2025):**
- **Philosophy**: Good programmers worry about data structures, not code. Proper data modeling eliminates complexity.
- **MANDATORY**: Use repository patterns with protocol-based DI (SessionManagerProtocol) - isolates data concerns
- **FORBIDDEN**: Direct database imports in business logic (`from prompt_improver.database import get_session`)
- **REQUIRED**: Service naming convention (*Facade, *Service, *Manager) - clear responsibility boundaries
- **ENFORCED**: No classes >500 lines (god object elimination) - maintain single responsibility
- **MANDATORY**: Real behavior testing with testcontainers (no mocks for external services) - practice over theory

**Code deletion verification:**
- `rg "ExactItemName" . --type py -n` (verify zero usage)
- `rg "from.*import.*ExactItemName|import.*ExactItemName" . --type py -n`

**Modernization patterns (2025):** Python: %formatting→f-strings, os.path→pathlib; Type hints: Any→specific types; Async: callbacks→async/await; Architecture: Direct database imports→repository protocols

**2025 Architecture Achievements:** Repository pattern with SessionManagerProtocol; Service facades replacing god objects; Multi-level caching <2ms response; Real behavior testing with testcontainers (87.5% success)


## Evidence-Based Analysis

**When user requests:** verify, check, analyze, review, evaluate, assess, think, or "think hard"

**PROBLEM VALIDATION (Do First):**
- **Reality Check**: Is this solving a real production problem or theoretical concern?
- **User Impact**: Quantify how many users/systems are affected
- **Complexity Assessment**: Does solution complexity match problem severity?
- **Measurability**: Can we measure the problem and its resolution?

**DISCOVERY REQUIREMENTS (Complex Tasks):**
- **Multi-Method Validation**: Use 2+ discovery methods (filesystem + content + dependency analysis)
- **Assumption Challenge**: Document and test initial scope assumptions
- **Scope Lock**: Complete discovery BEFORE implementation planning

**Format**: [Finding]: [Evidence] 📍 Source: file:line
**Include**: Confidence level, scope examined, methodology used
**Validation**: Basic (syntax, imports) → Integration (services, APIs) → Performance (response times, memory)

**CRITICAL**: Challenge initial estimates with systematic verification
**Batch Operations**: Use parallel tool execution for 3-10x performance improvement

## Tool & Agent Usage

**Core Infrastructure Agents:**
- **database-specialist**: PostgreSQL/JSONB optimization, query performance
- **ml-orchestrator**: ML pipelines, model registry (@champion/@production)
- **performance-engineer**: Cross-cutting performance, cache optimization (96.67% hit rate)
- **security-architect**: Security policies, OWASP 2025, auth/authorization design
- **infrastructure-specialist**: Docker/testcontainers, CI/CD, real behavior testing

**Specialized Domain Agents:**
- **data-pipeline-specialist**: ETL processes, analytics pipelines
- **api-design-specialist**: FastAPI design, OpenAPI docs
- **monitoring-observability-specialist**: OpenTelemetry, distributed tracing
- **testing-strategy-specialist**: Real behavior testing, quality assurance
- **configuration-management-specialist**: Environment configs, settings
- **documentation-specialist**: Technical docs, API documentation

**Automatic Agent Selection**: Use appropriate specialist for domain-specific tasks
**Delegation Flow**: performance-engineer → database-specialist (query optimization)
**Architecture Integration**: PostgreSQL 15+ JSONB, testcontainers, L1/L2/L3 caching <2ms, model registry @champion/@production

## Extended Thinking & Context Management

**Use thinking blocks for complex reasoning (54% performance improvement):**
- After tool results: Reflect and determine next steps
- Before complex decisions: Multi-step reasoning and planning
- During problem decomposition: Breaking down complex tasks

## Error Handling & Correction

**Priority hierarchy:** Security > Code Correctness > Simplicity > Type Safety > Code Quality > Test Passing

**Simplicity Check:**
- **Can this be done with fewer concepts?** Reduce by half, then half again
- **Are there special cases that could be eliminated?** Good taste means transforming edge cases into normal cases
- **Classic example**: Linked list deletion - from 10 lines with conditionals to 4 lines without

**Root cause analysis over quick fixes:**
- Investigate before applying fixes
- Document error type, location, specific cause
- Validate claims with concrete evidence before marking complete

## Architectural Compliance (2025 Standards)

**MANDATORY Patterns:**
- **Clean Architecture**: Strict layer separation (Presentation → Application → Domain → Repository → Infrastructure)
  - *Why*: Data flow clarity eliminates special cases and hidden dependencies
- **Repository Pattern**: All data access through protocol-based interfaces
  - *Why*: Focus on data structures and ownership, not implementation details
- **Protocol-Based DI**: Use typing.Protocol for interfaces
  - *Why*: Eliminates coupling, enables true dependency inversion
- **Service Facades**: Consolidate related functionality into unified facades
  - *Why*: Transform special cases into normal operations through proper abstraction
- **Multi-Level Caching**: L1 (Memory) + L2 (Redis) + L3 (Database) achieving <2ms response times
  - *Why*: Data structure optimization yields order-of-magnitude performance gains
- **Real Behavior Testing**: Integration tests use testcontainers, no mocks for external services
  - *Why*: Theory loses to practice - test real behavior, not assumptions

**PROHIBITED Patterns:**
- **Direct Database Access**: From service or presentation layers
- **God Objects**: Classes >500 lines (split into focused services)
- **Deep Nesting**: Functions with >3 levels of indentation (redesign the logic)
- **Special-Case Proliferation**: Multiple if/else for edge cases (redesign data structure instead)
- **Infrastructure in Core**: Database/cache imports in business logic
- **Mock Integration Tests**: Use real services via testcontainers
- **Unnecessary Abstraction**: Don't create patterns for problems that don't exist in production

**Performance Requirements:**
- **Response Times**: P95 <100ms for endpoints, <2ms on critical paths
- **Cache Performance**: >80% hit rates (96.67% achieved)
- **Test Coverage**: 85%+ on service boundaries

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
File Creation Context:
- **Refactoring**: Creating files to eliminate god objects is REQUIRED
- **Features**: New files need strong justification (prefer existing structure)
- **Documentation**: Only when explicitly requested by user
ALWAYS prefer editing existing files over creating new ones when adding functionality.