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

Parallel execution: Invoke all relevant tools simultaneously rather than sequentially for maximum efficiency.
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
1. Direct name search: rg "exact_function_name|exact_component_name" --type-add 'web:*.{ts,tsx,js,jsx}' -t web
2. Pattern/keyword search: rg "validate.*email|email.*validation" --type-add 'web:*.{ts,tsx,js,jsx}' -t web
3. Import/export search: rg "import.*{similar}|export.*{similar}" --type-add 'web:*.{ts,tsx,js,jsx}' -t web
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
- rg "ExactItemName" . --type ts --type tsx --type js --type jsx -n
- rg "import.*ExactItemName|export.*ExactItemName" . -n
- rg "<ExactItemName|ExactItemName>" . --type tsx --type jsx -n
- rg ": ExactItemName|extends ExactItemName" . -n
Decision: Zero usage = safe removal
</deletion_verification>

<error_hierarchy>
When fixing build/type errors, prioritize:
1. Security & Safety - Never compromise
2. Code Correctness - Proper structure and logic
3. Type Safety - Maintain proper TypeScript typing
4. Code Quality - Readable, maintainable code
5. Test Passing - Only after above are satisfied

❌ Never use quick fixes: Remove imports, add @ts-ignore, use any type, delete code
✅ Always use proper fixes: Import types correctly, fix type mismatches, maintain explicit typing
</error_hierarchy>

Platform-agnostic: Use universal tools (rg, find, grep) for all environments
</verification>
```

---

### Rule 3: MCP RESEARCH WORKFLOW
*Streamlined MCP Implementation Tools with Web Research*

```xml
<research>
  <memory>search previous solutions first</memory>
  <context7>mandatory for external dependencies and best practices</context7>
  <firecrawl>web research, documentation scraping, real-time information</firecrawl>
  <sequential>multi-step planning for complex tasks</sequential>
  <parallel>invoke all relevant tools simultaneously</parallel>
  <store>document decisions and patterns</store>
  <thinking>use thinking blocks for complex reasoning</thinking>
  <fallback>manual analysis when MCP unavailable</fallback>
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

#### **RULE 5: ERROR CORRECTION**
```
<error_handling>
When errors occur:

<immediate>
- Investigate root cause before applying fixes
- Document: error type, location, specific cause
- Avoid quick fixes that mask underlying issues
</immediate>

<correction>
- Re-examine with different methodology
- Update documentation and related claims
- Verify fix doesn't introduce new issues
- Test edge cases and error scenarios
</correction>

Error hierarchy: Security → Correctness → Type Safety → Code Quality → Test Passing
</error_handling>
```

---

#### **RULE 6: CONTINUOUS IMPROVEMENT**
```
<improvement>
Modernization patterns:
- JavaScript: var→const/let, .then()→async/await
- Python: %formatting→f-strings, os.path→pathlib
- TypeScript: any→specific types, manual loops→built-in methods

Process:
1. Research latest practices via Context7
2. Recommend forward path, not backward compatibility
3. Provide clear migration strategies
4. Document rationale for modernization choices

Focus on: Performance, maintainability, readability, current best practices
</improvement>
```

---

### **RULE PRECEDENCE HIERARCHY**

When rules conflict, follow this order:
0. **Mandatory First Response Protocol** (governs all interactions)
1. **Error Correction** (safety first)
2. **Pre-Action Verification** (prevent issues)
3. **Systematic Analysis** (gather evidence)
4. **Implementation Validation** (ensure quality)
5. **MCP Research** (informed decisions)
6. **Continuous Improvement** (optimize over time)

### **PARALLEL EXECUTION INSTRUCTION**

**Critical for Claude-4 Sonnet optimization:**
```
For maximum efficiency, whenever you perform multiple operations, invoke all relevant tools simultaneously rather than sequentially. This applies to:
- Reading multiple files
- Running multiple searches
- Gathering different types of evidence
- Performing independent validations

Default to parallel execution unless output of one tool is required for input of another.
```

### **EXTENDED THINKING MODE**

**For complex multi-step tasks:**
```
Use <thinking> tags for complex reasoning, planning, and analysis:
- After tool use for reflection
- Before making decisions
- When breaking down complex problems
- For step-by-step reasoning

This enables more deliberate and structured problem-solving.
```
---