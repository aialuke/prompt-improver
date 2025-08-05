# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow Rules

#### **RULE 0: MANDATORY FIRST RESPONSE PROTOCOL**
```
<first_response>
NON-NEGOTIABLE - Follow for EVERY user request:

1. <question_detection>
   Scan for ANY patterns:
   - Direct questions: "?", "Why did you", "How did you", "What made you"
   - Challenge patterns: "you didn't", "you ignored", "you missed"
   - Clarification demands: "explain why", "tell me why", "clarify your decision"
   - Instruction challenges: "I asked you to", "I told you to", "I explicitly said"
   - Confusion indicators: "you were supposed to", "the instruction was"
   
   If ANY detected → Answer the direct question FIRST with complete honesty before any other work
</question_detection>

2. <mandatory_output>Always state: "Checking CLAUDE.md rules for this task..."</mandatory_output>

3. <rule_listing>
   Always list which specific rules apply:
   - User preservation directives → "Applying: INSTRUCTION HIERARCHY PROTOCOL"
   - Creating/modifying code → "Applying: PRE-ACTION VERIFICATION + MINIMAL COMPLEXITY"
   - Background services/async tasks → "Applying: UNIFIED ASYNC INFRASTRUCTURE PROTOCOL + PRE-ACTION VERIFICATION"
   - Code deletion → "Applying: CRITICAL VERIFICATION RULE"
   - Analysis requests → "Applying: SYSTEMATIC ANALYSIS PROTOCOL"
   - External libraries → "Applying: MCP RESEARCH WORKFLOW"
   - Multi-step tasks → "Applying: SYSTEMATIC COMPLETION PROTOCOL"
   - Simple informational → "No specific rules apply to this informational request"
</rule_listing>

4. <compliance_check>
   Automated verification:
   - Protocol adherence verified
   - Rule trigger activated
   - Compliance status confirmed
</compliance_check>

5. ONLY THEN proceed with actual work
</first_response>
```

---

#### **RULE 1: SYSTEMATIC ANALYSIS PROTOCOL**
```
<analysis>
When user requests: verify, check, analyze, review, evaluate, assess, think, or "think hard"

Required workflow:
1. <evidence>Provide file:line citations for ALL claims</evidence>
2. <confidence>State HIGH/MEDIUM/LOW with specific justification</confidence>
3. <scope>Report "X/Y files examined" with methodology used</scope>
4. <completion>Continue until all areas show evidence OR systematic identification of what requires clarification</completion>

Format: [Finding]: [Evidence] ([Scope]) 📍 Source: file:line

**Parallel execution**: Always invoke relevant tools simultaneously for maximum efficiency.
</analysis>
```

---

#### **RULE 2: PRE-ACTION VERIFICATION & MINIMAL COMPLEXITY**
```
<verification>
Before ANY action, think hard about simplicity:

<complexity_assessment>
Apply core principles:
- YAGNI: Implement ONLY what is immediately necessary
- KISS: Choose the most straightforward approach available
- DRY: Use existing libraries rather than reimplementing functionality
</complexity_assessment>

<search_protocol>
MANDATORY 4-step search before creating new code:
1. Direct name search: rg "exact_function_name|exact_class_name" --type py
2. Pattern/keyword search: rg "validate.*email|email.*validation" --type py  
3. Import search: rg "from.*import.*{similar}|import.*{similar}" --type py
4. If unclear: Use comprehensive search tools for complete analysis
</search_protocol>

<decision_matrix>
EXTEND existing code when:
- ≤3 new parameters, ≤50 lines, maintains single responsibility
- Same domain/responsibility, preserves existing contracts
- Code is clear and follows SOLID principles

CREATE new code when:
- Different domain/responsibility, would break existing contracts
- Extension would violate single responsibility principle
- Existing code unclear or poorly structured

REFACTOR FIRST when:
- Existing code unclear, would violate SOLID principles
- Extension would create technical debt
- Current implementation not the simplest possible
</decision_matrix>

<deletion_verification>
For DELETING code - verify zero usage:
- rg "ExactItemName" . --type py -n
- rg "from.*import.*ExactItemName|import.*ExactItemName" . --type py -n
- rg "class.*ExactItemName|def.*ExactItemName" . --type py -n
- rg ": ExactItemName|ExactItemName\(" . --type py -n
Decision: Zero usage = safe removal
</deletion_verification>


Platform-agnostic: Use universal tools (rg, find, grep) for all environments
</verification>
```

---

### Rule 3: MCP RESEARCH WORKFLOW
*Streamlined MCP Implementation Tools with Web Research*

```
<research>
- Search previous solutions first via memory tools
- Use Context7 for external dependencies and best practices  
- Apply multi-step planning for complex tasks
- Invoke all relevant tools simultaneously (parallel execution)
- Document decisions and patterns for future reference
- Use thinking blocks for complex reasoning
- Fall back to manual analysis when MCP unavailable
</research>
```

---

#### **RULE 4: IMPLEMENTATION VALIDATION**
```
<validation>
Validation levels (apply in order):

<basic>
- Syntax correctness
- Import statements
- Type safety
- Basic functionality
</basic>

<integration>
- Database connections
- External services
- Cross-component compatibility
- API contract adherence
</integration>

<performance>
- Response time: 0.1-500ms expected
- Memory usage: 10-1000MB typical
- Validate metrics before claiming results
- Use realistic test scenarios
</performance>

Always validate claims with concrete evidence before marking complete.
</validation>
```

---

#### **RULE 5: ERROR HANDLING & CORRECTION**
```
<error_handling>
Error Priority Hierarchy:
1. Security & Safety - Never compromise
2. Code Correctness - Proper structure and logic  
3. Type Safety - Maintain proper type hints and validation
4. Code Quality - Readable, maintainable code
5. Test Passing - Only after above are satisfied

Immediate Response:
- Investigate root cause before applying fixes
- Document: error type, location, specific cause
- Avoid quick fixes that mask underlying issues

Correction Process:
- Re-examine with different methodology
- Update documentation and related claims
- Verify fix doesn't introduce new issues
- Test edge cases and error scenarios

❌ Never use quick fixes: Remove imports, add # type: ignore, use Any type, delete code
✅ Always use proper fixes: Import types correctly, fix type mismatches, maintain explicit typing
</error_handling>
```

---

#### **RULE 6: CONTINUOUS IMPROVEMENT**
```
<improvement>
Modernization patterns:
- Python: %formatting→f-strings, os.path→pathlib, dict.keys()→dict iteration
- Type hints: Any→specific types, missing annotations→explicit typing
- Async: callback patterns→async/await, threading→asyncio where appropriate
- Data structures: manual loops→comprehensions/built-in methods

Process:
1. Research latest practices via Context7
2. Recommend forward path, not backward compatibility
3. Provide clear migration strategies
4. Document rationale for modernization choices

Focus on: Performance, maintainability, readability, current best practices
</improvement>
```

---

#### **RULE 7: UNIFIED ASYNC INFRASTRUCTURE PROTOCOL**
```
<async_governance>
MANDATORY enforcement of unified async infrastructure patterns per ADR-007:

<adr_reference>
ARCHITECTURAL AUTHORITY: @docs/architecture/ADR-007-unified-async-infrastructure.md
STATUS: ACCEPTED - All async background services MUST comply
</adr_reference>

<background_service_detection>
Before creating ANY async background operation, analyze context:

DETECTION PATTERNS:
- File paths: /services/, /monitoring/, /background/, /lifecycle/, /orchestration/, /metrics/, /performance/, /health/
- Class patterns: *Manager, *Monitor, *Service, *Collector, *Worker, *Orchestrator, *Controller
- Method patterns: _loop, _monitor, _background, _periodic, _continuous, _worker, _daemon, _service
- Service patterns: while True loops, asyncio.sleep(), scheduled operations, service lifecycle methods

CONTEXT ANALYSIS:
- Persistent background operations → MANDATORY unified infrastructure
- Service monitoring/health checks → MANDATORY unified infrastructure  
- Data collection/aggregation → MANDATORY unified infrastructure
- Periodic/scheduled tasks → MANDATORY unified infrastructure
- Worker pools/processing queues → MANDATORY unified infrastructure
</background_service_detection>

<unified_infrastructure_mandate>
REQUIRED IMPLEMENTATION - EnhancedBackgroundTaskManager:

1. MANDATORY VERIFICATION:
   - rg "get_background_task_manager|EnhancedBackgroundTaskManager" . --type py
   - Confirm centralized task management is available and operational

2. REQUIRED PATTERN:
   ```python
   from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority
   
   task_manager = get_background_task_manager()
   await task_manager.submit_enhanced_task(
       task_id=f"service_name_{unique_identifier}",
       coroutine=background_function,
       priority=TaskPriority.HIGH,  # CRITICAL, HIGH, NORMAL, LOW, BACKGROUND
       tags={"service": "service_name", "type": "operation_type", "component": "component_name"}
   )
   ```

3. PROHIBITED PATTERNS:
   - Direct asyncio.create_task() for persistent background services
   - Custom async task management systems
   - Independent worker pools or schedulers
   - Ad-hoc background service implementations
</unified_infrastructure_mandate>

<legitimate_exceptions>
ALLOWED direct asyncio.create_task() usage:
- Test files (/tests/* directories) - parallel test execution
- Framework internals (EnhancedBackgroundTaskManager implementation)
- Request/response processing (short-lived operations)
- Coordinated parallel execution within functions (not persistent services)
- Temporary operations with clear lifecycle boundaries
</legitimate_exceptions>

<enforcement_protocol>
AUTOMATIC GOVERNANCE (enforced by pre-tool hooks):
1. Code analysis for asyncio.create_task() patterns
2. Context detection for background service indicators
3. Validation against legitimate exception patterns
4. Violation blocking with specific remediation guidance
5. ADR-007 compliance verification

VIOLATION RESPONSE:
- IMMEDIATE BLOCKING of non-compliant code generation
- SPECIFIC GUIDANCE with exact implementation requirements
- ADR-007 REFERENCE for architectural authority
- REMEDIATION STEPS with working code examples
</enforcement_protocol>
</async_governance>
```

---

### **RULE PRECEDENCE HIERARCHY**

When rules conflict, follow this order:
0. **Mandatory First Response Protocol** (governs all interactions)
1. **Error Correction** (safety first)
2. **Pre-Action Verification** (prevent issues)
3. **Unified Async Infrastructure Protocol** (architectural consistency)
4. **Systematic Analysis** (gather evidence)
5. **Implementation Validation** (ensure quality)
6. **MCP Research** (informed decisions)
7. **Continuous Improvement** (optimize over time)

---

### **AGENT DELEGATION**

**Delegate to specialized agents using the Task tool for domain expertise:**

- **code-reviewer**: Use PROACTIVELY for code review, refactoring, technical debt analysis, quality assessment
- **system-reliability-engineer**: Use PROACTIVELY for monitoring, health checks, incident response, observability setup
- **database-specialist**: Use PROACTIVELY for query optimization, schema design, migrations, connection pooling
- **security-architect**: Use PROACTIVELY for authentication, authorization, vulnerability assessment, security reviews
- **ml-orchestrator**: Use PROACTIVELY for ML training issues, pipeline design, feature engineering, model optimization
- **performance-engineer**: Use PROACTIVELY for slow performance, profiling, load testing, bottleneck analysis
- **infrastructure-specialist**: Use PROACTIVELY for testcontainers, CI/CD, development environments, testing infrastructure

**Delegation triggers:** review, optimize, secure, monitor, test, deploy, analyze, refactor, troubleshoot

---

### **PARALLEL PROCESSING**

**Maximize efficiency through concurrent execution:**

#### **Parallel Tool Calls**
- **Use when**: Multiple independent operations needed simultaneously
- **Pattern**: Invoke Read, Grep, Glob, Bash tools together in single response
- **Benefit**: 3-10x faster than sequential execution
- **Example**: `Read file1.py + Grep "pattern" + Bash "test command"` simultaneously

#### **Task Tool Parallelism** 
- **Use when**: Multiple complex tasks can run independently
- **Limit**: Up to 10 concurrent tasks automatically managed
- **Pattern**: Let Claude Code decide parallelism level for optimal performance
- **Benefit**: Concurrent agent execution and analysis

#### **Information Gathering**
- **Always parallel**: File reads, searches, validations, evidence collection
- **Pattern**: Batch all independent tool calls in single message
- **Rule**: Default to parallel unless one tool output feeds another

#### **Parallel Triggers**
**Use parallel execution for**: analyze + verify + search, read + grep + validate, test + profile + measure, scan + check + assess

**Avoid parallel for**: sequential dependencies, file writes, destructive operations

---

### **EXTENDED THINKING MODE**

**Use thinking blocks for complex reasoning and reflection:**

#### **When to Think**
- **After tool results**: Reflect on quality and determine optimal next steps
- **Before complex decisions**: Multi-step reasoning and planning
- **During problem decomposition**: Breaking down complex tasks
- **For iterative improvement**: Analyzing and refining approaches

#### **Think Tool Pattern**
- **Mid-response analysis**: Structured reasoning during complex task execution
- **Use for**: Tool output analysis, policy compliance, sequential high-stakes decisions
- **Benefit**: 54% performance improvement in complex domains
- **Pattern**: Pause → Analyze → Decide → Continue

#### **Thinking Triggers**
**Use extended thinking for**: analyze, plan, evaluate, reason, decompose, optimize, strategize, debug

---

### **CONTEXT MANAGEMENT**

**Optimize context usage and agent coordination:**

#### **Subagent Context Preservation**
- **Use subagents** to verify details and investigate complex problems
- **Benefit**: Preserves main context availability during deep investigations
- **Pattern**: Delegate specialized analysis to maintain primary workflow

#### **Iterative Development**
- **Explore → Plan → Code → Commit** workflow for complex features
- **Test-driven development**: Write tests first, then implement
- **Visual iteration**: Use screenshots for UI/UX feedback loops
- **Rule**: "Claude's outputs improve significantly with iteration"

#### **Tool Cleanup**
- **Always clean up**: Temporary files, scripts, helper files at task completion
- **Pattern**: Create → Use → Clean up in single workflow
- **Benefit**: Prevents context pollution and maintains clean workspace

---
