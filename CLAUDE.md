# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Principles

**Search existing solutions before creating new code:**
- Direct name search: `rg "exact_function_name|exact_class_name" --type py`
- Pattern search: `rg "validate.*email|email.*validation" --type py`
- Import search: `rg "from.*import.*{similar}|import.*{similar}" --type py`

**Apply YAGNI/KISS/DRY:**
- Extend existing code when ≤3 parameters, ≤50 lines, same domain
- Create new code when different domain or would break contracts  
- Refactor first when existing code unclear or violates SOLID principles

**Code deletion verification:**
- `rg "ExactItemName" . --type py -n` (verify zero usage)
- `rg "from.*import.*ExactItemName|import.*ExactItemName" . --type py -n`

**Modernization patterns:**
- Python: %formatting→f-strings, os.path→pathlib, dict.keys()→dict iteration
- Type hints: Any→specific types, missing annotations→explicit typing
- Async: callback patterns→async/await, threading→asyncio where appropriate

**Async background services:** Prefer EnhancedBackgroundTaskManager for persistent services; direct asyncio.create_task() acceptable for tests and short-lived operations.

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