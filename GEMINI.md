# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Project Overview
The Adaptive Prompt Enhancement System (APES) is an intelligent prompt optimization tool that automatically improves user prompts for AI agents through rule-based transformations enhanced by machine learning optimization. The system is currently undergoing a migration from a hybrid TypeScript/Python architecture to a pure Python implementation.

### Current State
- **Phase 1 & 2**: Foundation cleanup and Python scaffolding ✅ COMPLETED
- **Phase 3**: Core component implementation 🔄 IN PROGRESS  
- **Phase 4**: Testing & validation 🔄 IN PROGRESS
- **Phase 5**: Documentation update 📋 PENDING

## Development Commands

### Environment Setup
```bash
# Create and activate Python environment (uses uv if available, else standard venv)
./scripts/setup_development.sh

# Activate environment manually
source .venv/bin/activate
```

### Running the Application
```bash
# Start the FastAPI development server with auto-reload
./scripts/run_server.sh
# Server runs at http://127.0.0.1:8000
# API docs available at http://127.0.0.1:8000/docs
```

### Testing and Code Quality
```bash
# Run full test suite with linting
./scripts/run_tests.sh

# Run tests only
pytest tests/

# Run specific test file
pytest tests/rule_engine/test_clarity_rule.py

# Run linting only
ruff check src tests
ruff format --check src tests

# Format code
ruff format src tests
```

### ML Model Management
```bash
# View MLflow UI for experiment tracking
mlflow ui
# Access at http://127.0.0.1:5000

# Promote model to production (if needed)
python scripts/promote_model.py
```

## Architecture Overview

### Core Components

1. **Rule Engine** (`src/prompt_improver/rule_engine/`)
   - Base rule framework implementing check/apply pattern
   - Core rules based on Anthropic best practices
   - Technique rules for advanced prompt engineering
   - ML-discovered rules stored dynamically
   - Each rule inherits from `BasePromptRule` and implements:
     - `check()`: Determines if rule applies to prompt
     - `apply()`: Transforms the prompt
     - `to_llm_instruction()`: Generates LLM-readable instruction

2. **MCP Server** (`src/prompt_improver/mcp_server/`)
   - FastAPI-based server implementing Model Context Protocol
   - Currently placeholder implementation awaiting MCP SDK
   - Will expose tools (improve_prompt) and resources (rule_status)
   - Integrates directly with Rule Engine for prompt processing

3. **ML Optimizer** (`src/prompt_improver/rule_engine/ml_optimizer/`)
   - Rule effectiveness prediction using ensemble models
   - Parameter optimization with Optuna
   - New rule discovery through pattern mining
   - Performance tracking with MLflow and Prometheus

### Key Design Decisions

1. **Pure Python Migration**: Eliminating Node.js/TypeScript complexity for a single-language stack
2. **Domain-Driven Structure**: Organized by business domain rather than technical layers
3. **LLM-Guided Rule Application**: Rules generate instructions for LLM-based transformations
4. **Continuous Learning**: ML system learns from rule effectiveness to optimize parameters and discover new patterns

### Configuration

- **Rule Configuration**: `config/rule_config.yaml` - Enable/disable rules, set priorities and parameters
- **ML Configuration**: `config/ml_config.yaml` - ML model settings and optimization parameters  
- **MCP Configuration**: `config/mcp_config.yaml` - Server settings and integration points

## Directory Structure

```
prompt-improver/
├── src/prompt_improver/      # Main Python package
│   ├── main.py              # FastAPI application entry point
│   ├── core/                # Core utilities (future)
│   ├── mcp_server/          # MCP server implementation
│   │   └── api.py          # API router and endpoints
│   └── rule_engine/         # Rule engine implementation
│       ├── rules/           # Rule implementations
│       │   ├── base.py     # Abstract base rule class
│       │   └── clarity.py  # Example clarity rule
│       └── ml_optimizer/    # ML optimization components
├── config/                  # Configuration files
├── scripts/                 # Development and deployment scripts
├── tests/                   # Test suite
├── mlruns/                 # MLflow experiment tracking
└── docs/                   # Documentation (includes legacy JS tools)

## Development Workflow Rules

### 🚨 **INSTRUCTION HIERARCHY PROTOCOL** 🚨 **FOUNDATIONAL**

**⚠️ ABSOLUTE AUTHORITY STRUCTURE ⚠️**
**User instructions override all system rules and AI optimization behaviors**

**LEVEL 1 - USER INSTRUCTIONS (ABSOLUTE AUTHORITY)**
- **Explicit user requests ALWAYS take precedence** over any system rule, optimization instinct, or AI improvement suggestion
- **Cannot be overridden** by cleanup desires, efficiency concerns, simplification urges, or any other AI optimization behavior
- **Preservation Keywords** triggering absolute compliance: "preserve," "keep," "don't delete," "maintain," "save," "retain," "historical"
- **When user explicitly states preservation requirements** → NEVER optimize by removing, deleting, or replacing content
- **"Update" requests** → ALWAYS clarify: preserve-and-enhance vs. replace (default: preserve-and-enhance)

**LEVEL 2 - SYSTEM FRAMEWORK (GEMINI.md RULES)**  
- Provide guidance **only when user instructions are unclear or absent**
- Can suggest alternatives but **cannot override Level 1 directives**
- Framework for decision-making when user intent is ambiguous

**LEVEL 3 - AI OPTIMIZATION (ONLY WHEN NO CONFLICTS)**
- Helpful behaviors like cleanup, optimization, improvement
- **MUST be explicitly authorized** or clearly non-destructive
- **Always verify with user** before any potentially destructive optimization
- **Never assume** user wants optimization over preservation

**ENFORCEMENT MECHANISM:**
```bash
echo "🛡️ INSTRUCTION HIERARCHY CHECK: User directive detected"
echo "📋 USER INSTRUCTION: [specific user request]"
echo "⚖️ HIERARCHY LEVEL: [1-ABSOLUTE|2-FRAMEWORK|3-OPTIMIZATION]"
echo "✅ COMPLIANCE: Following Level 1 authority - no system overrides permitted"
```

---

### 🚨 **MANDATORY FIRST RESPONSE PROTOCOL** 🚨

**⚠️ NON-NEGOTIABLE ⚠️ - Follow for EVERY user request:**

1. **ENHANCED QUESTION DETECTION** - Scan for ANY of these patterns:
   **Direct Questions:** "?", "Why did you", "How did you", "What made you", "why", "how", "what"
   **Challenge Patterns:** "you didn't", "you ignored", "you missed", "you deleted", "you removed"
   **Clarification Demands:** "explain why", "tell me why", "clarify your decision"
   **Instruction Challenges:** "I asked you to", "I told you to", "I explicitly said"
   **Confusion Indicators:** "you were supposed to", "the instruction was", "I don't understand why"
   - **If ANY pattern detected** → Answer the direct question FIRST with complete honesty before any other work
   - **No rule consultation phase** - Direct accountability-focused response
   - **Include specific reasoning** for any decisions that were made

2. **MANDATORY OUTPUT**: "Checking GEMINI.md rules for this task..."

3. **MANDATORY RULE LISTING** - Always list which specific rules apply:
   **Auto-trigger detection**:
   - User preservation directives → "Applying: INSTRUCTION HIERARCHY PROTOCOL (Level 1 Authority)"
   - Documentation update requests → "Applying: PRE-ACTION VERIFICATION + DOCUMENTATION PRESERVATION"
   - Creating/modifying code → "Applying: MINIMAL COMPLEXITY PRINCIPLE"
   - Code deletion requested → "Applying: CRITICAL VERIFICATION RULE"  
   - External libraries mentioned → "Applying: MCP IMPLEMENTATION TOOLS"
   - Analysis requests ("verify", "check", "analyze") → "Applying: EVIDENCE ENUMERATION + CONFIDENCE ASSESSMENT"
   - Ambiguous requests → "Applying: PRE-ACTION VERIFICATION"
   - Multi-step tasks → "Applying: SYSTEMATIC COMPLETION PROTOCOL"
   - Error identification/correction → "Applying: ERROR CORRECTION PROTOCOL"
   - User challenges accuracy → "Applying: CLAIM RE-VERIFICATION PROTOCOL"
   - Simple informational → "No specific rules apply to this informational request"

4. **MANDATORY COMPLIANCE CHECK**:
   ```bash
   echo "🔍 AUTOMATED COMPLIANCE CHECK: Protocol adherence verified"
   echo "✅ RULE TRIGGER: [Specific rule] activated"
   echo "📊 COMPLIANCE STATUS: Following required protocol steps"
   ```

5. **ONLY THEN proceed with the actual work**

---

## Phase-Based Workflow: Context → Search → Implement → Validate

### **PRE-ACTION VERIFICATION PROTOCOL** 🛡️ CRITICAL

**BEFORE taking ANY action, verify understanding AND data sources:**

**PHASE 1: DIAGNOSTIC SOURCE VERIFICATION**
```bash
# MANDATORY: Verify available diagnostic tools first
echo "🔧 AVAILABLE TOOLS CHECK:"
echo "- IDE Diagnostics: [AVAILABLE/UNAVAILABLE] - mcp__ide__getDiagnostics"  
echo "- File System: [AVAILABLE/UNAVAILABLE] - Read/Grep/Glob tools"
echo "- External APIs: [AVAILABLE/UNAVAILABLE] - Context7/Web tools"

# Select most reliable data source
echo "📊 SELECTED DATA SOURCE: [IDE diagnostics|manual file analysis|external research] because [reliability justification]"
```

**PHASE 2: UNDERSTANDING VERIFICATION**
**Trigger verification when request contains:**
- Vague verbs: "fix", "improve", "optimize", "enhance", "update", "refactor"
- Undefined pronouns: "this", "that", "it" (without clear referent)
- Scope gaps: Missing what/where/which specifics
- Multi-step tasks (>2 distinct actions)
- **NEW**: Analysis requests without specified methodology
- **NEW**: Claims requiring code inspection without tool verification

**Required verification:**
```
Understanding Check:
- Actions required: [list each specific action]
- Undefined terms: [list ambiguous items]
- Missing context: [what needs clarification]
- Data source methodology: [tool/approach selected and why]
- Expected accuracy level: [high/medium/low based on available tools]
Decision: [PROCEED/CLARIFY] because [specific reason + data source confidence]
```

**Enhanced Decision criteria:**
- **PROCEED** if: All actions specific, no undefined terms, clear scope, reliable data source identified
- **CLARIFY** if: Vague actions, undefined terms, unclear scope, OR unreliable data source
- **REQUEST TOOLS** if: Analysis needed but diagnostic tools unavailable

---

**PHASE 2B: DOCUMENTATION PRESERVATION VERIFICATION** 📚 CRITICAL
**Trigger when user requests:** "update," "modify," "change" on documentation files containing historical data

**MANDATORY CLARIFICATION BEFORE ANY CHANGES:**
```
Documentation Update Clarification:
- Request type: [preserve-and-enhance|replace-sections|full-replacement]
- Preservation requirements: [maintain all historical data|keep specific sections|archive before replacing]
- Scope specification: [current status only|add new sections|modify existing sections|full rewrite]
- Historical value assessment: [reference material|implementation history|decision context|user explicitly values]
- User preservation signals: [previous "keep" requests|"preserve" keywords|historical importance mentioned]
Decision: [PRESERVE-AND-ENHANCE|REQUEST-SPECIFIC-AUTHORIZATION] because [preservation analysis + user directive compliance]
```

**DEFAULT BEHAVIOR HIERARCHY:**
1. **PRESERVE-AND-ENHANCE** (default) - Add new content while maintaining all existing content
2. **SECTION-SPECIFIC MODIFICATION** - Only with explicit user authorization of specific sections
3. **FULL REPLACEMENT** - Only with explicit user confirmation and backup offer

**IMPLEMENTATION PATTERN:**
- Use clear section markers: "## CURRENT STATUS" vs "## HISTORICAL REFERENCE"
- Add timestamps: "Updated [DATE]" for new information
- Preserve original sections with "HISTORICAL", "REFERENCE", or "ARCHIVE" markers
- **NEVER delete content** without explicit permission to remove specific sections
- When in doubt → Ask for clarification rather than assume destructive intent

---

### **MINIMAL COMPLEXITY PRINCIPLE** 🔍 CRITICAL

**Before creating ANY new code, think hard about simplicity:**

**Apply core principles:**
- **YAGNI**: Implement ONLY what is immediately necessary
- **KISS**: Choose the most straightforward approach available  
- **DRY**: Use existing libraries rather than reimplementing functionality

**MANDATORY SEARCH PROTOCOL:**
```bash
# 1. Direct name search
rg "exact_function_name|exact_component_name" --type-add 'web:*.{ts,tsx,js,jsx}' -t web

# 2. Pattern/keyword search
rg "validate.*email|email.*validation" --type-add 'web:*.{ts,tsx,js,jsx}' -t web

# 3. Import/export search
rg "import.*{similar}|export.*{similar}" --type-add 'web:*.{ts,tsx,js,jsx}' -t web

# 4. If unclear, use Task tool for comprehensive search
```

**Decision Matrix:**
- **EXTEND existing code** when: ≤3 new parameters, ≤50 lines, maintains single responsibility
- **CREATE new code** when: Different domain/responsibility, would break existing contracts
- **REFACTOR FIRST** when: Existing code unclear, would violate SOLID principles

**ERROR RESOLUTION HIERARCHY** (when fixing build/type errors):
1. **Security & Safety** - Never compromise
2. **Code Correctness** - Proper structure and logic
3. **Type Safety** - Maintain proper TypeScript typing
4. **Code Quality** - Readable, maintainable code
5. **Test Passing** - Only after above are satisfied

**❌ Never use quick fixes:** Remove imports, add @ts-ignore, use any type, delete code
**✅ Always use proper fixes:** Import types correctly, fix type mismatches, maintain explicit typing

---

### **CRITICAL VERIFICATION RULE** 🛑 CRITICAL

**⚠️ WARNING: CODE DELETION CAN CAUSE DATA LOSS ⚠️**

**MANDATORY verification before ANY code deletion:**

```bash
# Search for exact usage
rg "ExactItemName" . --type ts --type tsx --type js --type jsx -n

# Search for import/export usage
rg "import.*ExactItemName|export.*ExactItemName" . -n

# Search for React component usage (JSX)
rg "<ExactItemName|ExactItemName>" . --type tsx --type jsx -n

# Search for type annotations
rg ": ExactItemName|extends ExactItemName" . -n
```

**Decision Matrix:**
- ✅ **SAFE TO REMOVE**: Zero usage found in verification searches
- ❌ **DO NOT REMOVE**: Any usage found, even in same file
- ⚠️ **INVESTIGATE**: Usage only in comments or documentation

---

### **MCP IMPLEMENTATION TOOLS** 🔧 CRITICAL

**MANDATORY for external dependencies, complex implementations, AND code analysis:**

**Complete MCP Workflow:**

**0. IDE Diagnostic Integration (MANDATORY for code analysis)**
```bash
# ALWAYS check IDE diagnostics first for code analysis tasks
mcp__ide__getDiagnostics() # Get all diagnostics
mcp__ide__getDiagnostics("[specific_file_uri]") # Get file-specific diagnostics

# Use IDE diagnostics for:
# - Code quality analysis
# - Error detection
# - Unused variable identification  
# - Type checking issues
# - Syntax problems
```

**Tool Selection Decision Matrix:**
- **IDE Diagnostics FIRST** for: Code analysis, error detection, syntax validation
- **Manual File Analysis** when: IDE diagnostics unavailable or insufficient  
- **Context7 Research** for: External library/framework implementation
- **Sequential Thinking** for: Complex multi-step problem solving

**1. Context7 Research (MANDATORY for new implementations)**
```bash
# REQUIRED: Always research before writing code if not done in current planning
mcp__context7__resolve-library-id("[library/framework/tool-name]")
mcp__context7__get-library-docs("[resolved-id]" topic="[specific-implementation]")
```
**Required when:** Any new code, libraries, frameworks, or implementation patterns not researched in current session

**2. Memory Search (Always start here after Context7)**
```bash
mcp__memory__search_nodes("[implementation-type] patterns")
mcp__memory__search_nodes("[current-task-type] solutions")
```

**3. Additional Context7 Research (If needed for external dependencies)**
```bash
mcp__context7__resolve-library-id("[library-name]")
mcp__context7__get-library-docs("[resolved-id]", topic="[specific-topic]")
```

**4. Sequential Planning (For multi-step tasks)**
```bash
mcp__sequential-thinking__sequentialthinking("Plan [implementation] using [approach]")
```

**5. Memory Storage (Always finish with this)**
```bash
mcp__memory__create_entities({
  name: "[Decision/Pattern name]",
  entityType: "architecture|solution|pattern|decision",
  observations: ["Key learning", "Why this approach", "Trade-offs considered"]
})
```

**Enhanced Documentation Format:**
```
Tool Selection: [IDE diagnostics|manual analysis|context7] because [reliability/availability reason]
IDE Diagnostics: [X issues found] → [specific file:line references]
Context7 Research: [library/topic] → [3-5 key insights applied to implementation]
Memory Search: "[keywords]" - Found: [results or none]
Implementation: [brief description of approach using research]
Memory Storage: "[what-stored]" - [entity-type] - [key-observations]
```

**Fallback Strategy:**
```bash
# If IDE diagnostics fail:
echo "⚠️ IDE DIAGNOSTICS UNAVAILABLE - Using manual analysis"
echo "📋 FALLBACK: Manual file inspection + ripgrep searches"
echo "🔍 ACCURACY: Medium (manual) vs High (IDE diagnostics)"
```

---

### **EVIDENCE ENUMERATION REQUIREMENT** 📊 CRITICAL

**For analysis requests (verify, check, analyze, review):**

**MANDATORY VERIFICATION BEFORE CLAIMS:**
```bash
# 1. ALWAYS verify data sources first
echo "🔍 DATA SOURCE VERIFICATION: Using [diagnostic tool|manual analysis|file inspection]"

# 2. Execute verification commands
[specific verification commands based on analysis type]

# 3. Document methodology used
echo "📊 METHODOLOGY: [systematic search|IDE diagnostics|cross-reference verification]"
```

**Required format for ALL claims with MANDATORY CITATIONS:**
```
[Finding]: [Evidence] ([Scope]) 
📍 Source: <cite file="[file_path]" line="[line_number]">[specific_quote_or_finding]</cite>
```

**ENHANCED EXAMPLES:**
❌ "Services have duplications"
❌ "Found 3 duplicate functions" (insufficient evidence)
✅ "Found 3 duplicate validateEmail() functions: 
📍 Source: <cite file="src/utils/validation.ts" line="15">function validateEmail(email)</cite>
📍 Source: <cite file="src/auth/helper.ts" line="42">function validateEmail(input)</cite>  
📍 Source: <cite file="src/forms/validator.ts" line="28">const validateEmail = (addr)</cite>
(Scope: searched 15 TypeScript files via `find src -name "*.ts" | wc -l`)"

**MANDATORY SELF-VERIFICATION PROTOCOL:**
Before finalizing ANY analysis:
1. **Review each claim** - Can I cite specific evidence?
2. **Verify methodology** - Did I use the most reliable data source available?
3. **Check completeness** - Did I miss any counter-evidence?
4. **Document confidence** - What's my confidence level and why?

**If ANY claim lacks specific citable evidence → RETRACT and mark with [INSUFFICIENT_EVIDENCE]**

**Verification commands:**
```bash
# For finding examples with line numbers
rg "pattern" . --type ts -n | head -5

# For counting scope with methodology documentation
find src -name "*.ts" | wc -l
rg "pattern" . --type ts --files-with-matches | wc -l

# For IDE diagnostic verification (when available)
mcp__ide__getDiagnostics("[file_uri]")
```

---

### **SYSTEMATIC COMPLETION PROTOCOL** 📋

**For comprehensive analysis tasks:**

**Progress tracking format:**
```
Scope: [X areas defined] ([verification command used])
Progress: [Y/X areas analyzed] ([files per area])
Evidence: [Z findings documented] ([example locations])
Verification: [Implementation matches specification] ([specific checks performed])
Status: [CONTINUE/COMPLETE] because [objective reason + verification results]
```

**Enhanced Decision Criteria:**
- **CONTINUE**: Any area shows 0 files analyzed OR missing evidence OR verification gaps
- **COMPLETE**: All areas have evidence AND verification confirms specification compliance OR objective reason for exclusion

---

### **ERROR CORRECTION PROTOCOL** 🔄 CRITICAL

**MANDATORY when errors or inaccuracies are identified:**

**Phase 1: Error Acknowledgment**
```bash
echo "🚨 ERROR DETECTED: [specific error identified]"
echo "📍 LOCATION: [where error occurred - file, line, or analysis section]"
echo "🔍 ROOT CAUSE: [why error occurred - method, assumption, or data issue]"
```

**Phase 2: Systematic Re-verification**
```
MANDATORY RE-VERIFICATION CHECKLIST:
□ Re-examine original data sources with different method
□ Cross-reference against additional sources when available  
□ Verify methodology used was appropriate for the task
□ Check for confirmation bias in original analysis
□ Document what was missed or misinterpreted
□ Identify any related claims that may also be incorrect
```

**Phase 3: Correction Documentation**
```
ERROR CORRECTION REPORT:
- Original Claim: [exact original statement]
- Error Type: [methodology|data|interpretation|scope]
- Corrected Finding: [new accurate finding with evidence]
- Methodology Change: [what approach was used for correction]
- Related Impact: [other claims that needed review/correction]
- Prevention: [how to avoid this error type in future]
```

**Phase 4: Update Documentation**
```bash
# Update any documentation that contained the error
echo "📝 UPDATING: [specific files/reports updated]"
echo "✅ VERIFIED: All related claims checked and corrected"
echo "🔒 LOCKED: Final verification complete"
```

**Trigger Conditions:**
- User explicitly identifies an error ("you were wrong", "that's incorrect")
- Self-detection during verification process
- Contradiction found between sources
- Follow-up analysis reveals initial error

---

### **CONFIDENCE ASSESSMENT FRAMEWORK** 🎯 CRITICAL

**MANDATORY confidence level documentation for all analytical claims:**

**Confidence Level Definitions:**
```
🟢 HIGH CONFIDENCE (90-100%):
- IDE diagnostics used + manual verification
- Multiple reliable sources confirm finding
- Systematic methodology applied
- Evidence directly supports claim
- Format: "HIGH confidence: [finding] based on [specific evidence]"

🟡 MEDIUM CONFIDENCE (70-89%):
- Manual analysis with systematic approach
- Single reliable source or partial verification
- Some assumptions made but documented
- Evidence mostly supports claim
- Format: "MEDIUM confidence: [finding] based on [evidence + limitations]"

🔴 LOW CONFIDENCE (50-69%):
- Limited data sources available
- Methodology constraints acknowledged
- Significant assumptions made
- Evidence partially supports claim
- Format: "LOW confidence: [finding] - [specific limitations noted]"

⚫ INSUFFICIENT (0-49%):
- Inadequate evidence to make claim
- Unreliable or unclear data sources
- Must use: "[INSUFFICIENT_EVIDENCE] - cannot determine [finding]"
```

**Mandatory Confidence Documentation:**
```
CONFIDENCE ASSESSMENT:
- Evidence Quality: [high|medium|low] - [reason]
- Methodology Used: [systematic|limited|assumption-based]
- Data Source Reliability: [IDE|manual|external] - [reliability justification] 
- Verification Level: [cross-checked|single-source|unverified]
- Overall Confidence: [HIGH|MEDIUM|LOW|INSUFFICIENT] ([percentage])
- Limitations: [specific factors affecting confidence]
```

**Evidence Quality Requirements by Confidence Level:**
- **HIGH**: Direct citations with file:line references + systematic verification
- **MEDIUM**: Specific examples with file references + structured analysis  
- **LOW**: General findings with documented limitations + basic verification
- **INSUFFICIENT**: Must explicitly state inability to verify claim

**Uncertainty Acknowledgment Protocol:**
```bash
# When confidence is MEDIUM or below:
echo "⚠️ UNCERTAINTY ACKNOWLEDGED: [specific limitations]"
echo "🔍 ADDITIONAL VERIFICATION NEEDED: [what would increase confidence]"
echo "📊 CONFIDENCE LEVEL: [level] because [specific justification]"
```

---

### **CLAIM RE-VERIFICATION PROTOCOL** 🔍 CRITICAL

**MANDATORY when accuracy is questioned or systematic re-examination needed:**

**Trigger Conditions:**
- User challenges accuracy ("are you sure?", "verify this", "double-check")
- Contradictory evidence emerges
- High-stakes claims requiring validation
- Previous errors detected in same analysis

**Systematic Re-verification Process:**

**Phase 1: Original Claim Analysis**
```
ORIGINAL CLAIM REVIEW:
- Exact Claim: [copy original statement verbatim]
- Evidence Used: [list all evidence cited originally]
- Methodology: [describe approach used]
- Confidence Level: [what was claimed originally]
- Assumptions Made: [identify any assumptions]
```

**Phase 2: Independent Re-examination** 
```bash
# Use different methodology than original
echo "🔄 RE-VERIFICATION METHOD: [different approach from original]"

# Example approaches:
# - If used manual analysis → try IDE diagnostics
# - If used single file → expand to cross-file search  
# - If used grep → try AST-based analysis
# - If used assumptions → seek direct verification
```

**Phase 3: Cross-Reference Verification**
```bash
# Search for counter-evidence
rg "[contradiction_patterns]" . --type [relevant_types] -n

# Check multiple sources
echo "📚 SOURCES CHECKED:"
echo "- [source 1]: [finding]"
echo "- [source 2]: [finding]"  
echo "- [source 3]: [finding]"
```

**Phase 4: Verification Results**
```
RE-VERIFICATION RESULTS:
- Original Claim Status: [CONFIRMED|MODIFIED|CONTRADICTED|INSUFFICIENT_EVIDENCE]
- New Evidence Found: [list any new evidence]
- Contradictory Evidence: [list any conflicts]
- Methodology Comparison: [original vs re-verification approach]
- Final Determination: [corrected claim with confidence level]
- Documentation Updates: [what needs to be corrected]
```

**Multiple Source Verification Requirements:**
- **Critical Claims**: Minimum 2 independent verification methods
- **Code Analysis**: IDE diagnostics + manual inspection when available
- **External Dependencies**: Documentation + community sources + testing
- **Architecture Decisions**: Multiple examples + cross-reference patterns

---

## Unified Enforcement Framework

### **Risk Levels & Responses**

**🔴 CRITICAL RISK** (Data Safety, Security):
- MANDATORY enforcement - STOP if violated
- Zero tolerance for non-compliance

**🟡 HIGH RISK** (Workflow Integrity, Quality):
- REQUIRED enforcement - Flag warning if violated
- Extra verification steps applied

**🟢 MEDIUM RISK** (Best Practices, Optimization):
- RECOMMENDED enforcement - Note for improvement

### **Auto-Triggers**

**Context Detection:**
- File creation/modification → MINIMAL COMPLEXITY PRINCIPLE
- External dependencies → MCP IMPLEMENTATION TOOLS  
- Analysis language ("verify", "check") → EVIDENCE ENUMERATION + CONFIDENCE ASSESSMENT
- Deletion requests → CRITICAL VERIFICATION RULE
- Ambiguous requests → PRE-ACTION VERIFICATION
- Error identification → ERROR CORRECTION PROTOCOL
- Accuracy challenges → CLAIM RE-VERIFICATION PROTOCOL

### **Monthly Review Protocol**

**Key Metrics to Track:**
- Rule compliance rate per session
- Critical violations (should be zero)
- MCP tool usage for external dependencies
- Evidence provision for analysis claims
- Code reuse vs new creation ratio

**Simple Tracking Command:**
```bash
echo "📊 MONTHLY REVIEW: [Rule compliance %] - [Critical violations: X] - [Improvements needed]"
```

---

## Rule Integration Notes

**Workflow Triggers:**
- PRE-ACTION VERIFICATION → clarifies scope and verifies data sources before other rules apply
- MINIMAL COMPLEXITY → search existing before MCP IMPLEMENTATION  
- MCP IMPLEMENTATION → use memory/context7 for informed decisions + IDE diagnostics for code analysis
- EVIDENCE ENUMERATION → support analysis claims with citations and confidence assessment
- CONFIDENCE ASSESSMENT → document reliability of all analytical claims
- ERROR CORRECTION → systematic correction when errors identified
- CLAIM RE-VERIFICATION → independent verification when accuracy challenged
- CRITICAL VERIFICATION → prevent data loss through systematic verification

**Cross-Rule Dependencies:**
- Memory search informs complexity decisions
- Context7 research guides implementation approach
- Evidence requirements apply to all analytical claims with confidence levels
- Verification protocols apply to all deletion actions
- Error correction triggers re-verification protocols
- Confidence assessment guides evidence quality requirements
- IDE diagnostics enhance MCP tool selection and evidence quality

---

## Compliance Verification

**Before claiming task complete:**
□ Applied mandatory first response protocol
□ Followed phase-based workflow (Context → Search → Implement → Validate)  
□ Used appropriate verification for code changes
□ Applied MCP tools with IDE diagnostics for code analysis
□ Provided evidence for analytical claims with proper citations
□ Documented confidence levels for all analytical claims
□ Applied error correction protocol if any mistakes identified
□ Used re-verification protocol if accuracy was challenged
□ Stored institutional knowledge in memory
□ Maintained simplicity principles (YAGNI/KISS/DRY)
