# CLAUDE.md - Consolidated Version

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow Rules

### 🚨 **INSTRUCTION HIERARCHY PROTOCOL** 🚨 **FOUNDATIONAL**

**⚠️ ABSOLUTE AUTHORITY STRUCTURE ⚠️**
**User instructions override all system rules and AI optimization behaviors**

**LEVEL 1 - USER INSTRUCTIONS (ABSOLUTE AUTHORITY)**
- **Explicit user requests ALWAYS take precedence** over any system rule
- **Preservation Keywords** triggering absolute compliance: "preserve," "keep," "don't delete," "maintain," "save," "retain", "historical"
- **"Update" requests** → Default to preserve-and-enhance unless explicitly told to replace

**ENFORCEMENT:**
```bash
echo "🛡️ INSTRUCTION HIERARCHY: User directive detected - Level 1 authority engaged"
```

**DOCUMENTATION PRESERVATION** (Part of Level 1):
- When updating docs with historical data → ALWAYS preserve-and-enhance by default
- Use clear markers: "## CURRENT STATUS" vs "## HISTORICAL REFERENCE"
- NEVER delete without explicit permission for specific sections

---

### 🚨 **STREAMLINED FIRST RESPONSE PROTOCOL** 🚨

**For EVERY user request:**

1. **If question/challenge detected** ("?", "why", "you didn't", "explain") → Answer directly FIRST

2. **State applicable rules** (only if relevant):
   - Code creation/search → "Applying: MINIMAL COMPLEXITY PRINCIPLE"
   - Deletion request → "Applying: CRITICAL VERIFICATION RULE"  
   - Analysis/verification → "Applying: UNIFIED VERIFICATION PROTOCOL"
   - External dependencies → "Applying: MCP IMPLEMENTATION TOOLS"
   - Errors/accuracy challenges → "Applying: ERROR & RE-VERIFICATION PROTOCOL"

3. **Proceed with work**

---

## Streamlined Phase Workflow

### **PHASE 1: CONTEXT** 
**Quick verification before action:**
- Understand request fully (clarify if vague: "fix", "improve", "this")
- Select data source (IDE diagnostics > manual analysis > external)
- Check documentation preservation needs

**Gate: PROCEED if clear, CLARIFY if ambiguous**

### **PHASE 2: SEARCH**
**Find before building:**
- Use MCP tools for external dependencies  
- Search existing code (MINIMAL COMPLEXITY)
- Document findings with evidence

**Gate: PROCEED if research complete**

### **PHASE 3: IMPLEMENT**
**Build with validation:**
- Test-first approach
- Apply verification protocols
- Investigate errors immediately

**Gate: PROCEED if functional**

### **PHASE 4: VALIDATE**
**Confirm completion:**
- All tests passing realistically
- Documentation updated
- No TODOs remaining

**Gate: COMPLETE if validated**

---

### **MINIMAL COMPLEXITY PRINCIPLE** 🔍 CRITICAL

**Before creating ANY new code:**

**Core principles:** YAGNI, KISS, DRY

**Search Protocol:**
```bash
# 1. Direct search
rg "exact_function_name" --type-add 'web:*.{ts,tsx,js,jsx}' -t web

# 2. Pattern search  
rg "validate.*email|email.*validation" -t web

# 3. If unclear, use Task tool
```

**Decision Matrix:**
- EXTEND existing: ≤3 params, ≤50 lines, same responsibility
- CREATE new: Different domain, would break contracts
- REFACTOR first: Unclear existing code

**Never use quick fixes** (@ts-ignore, any, delete imports)

---

### **CRITICAL VERIFICATION RULE** 🛑 CRITICAL

**Before ANY deletion:**

```bash
# Verify zero usage
rg "ExactItemName" . --type ts --type tsx --type js --type jsx -n
rg "import.*ExactItemName|export.*ExactItemName" . -n
rg "<ExactItemName|ExactItemName>" . --type tsx --type jsx -n
```

**Decision:**
- ✅ SAFE: Zero usage found
- ❌ STOP: Any usage found
- ⚠️ INVESTIGATE: Only in comments

---

### **MCP IMPLEMENTATION TOOLS** 🔧 CRITICAL

**For external dependencies & complex implementations:**

**Workflow:**
1. **IDE Diagnostics** (if available): `mcp__ide__getDiagnostics()`
2. **Context7 Research**: `mcp__context7__resolve-library-id()` → `get-library-docs()`
3. **Memory Search**: `mcp__memory__search_nodes()`
4. **Sequential Planning**: `mcp__sequential-thinking__sequentialthinking()`
5. **Memory Storage**: `mcp__memory__create_entities()`

**Fallback if IDE unavailable:**
```bash
echo "⚠️ IDE DIAGNOSTICS UNAVAILABLE - Using manual analysis"
```

---

### **UNIFIED VERIFICATION PROTOCOL** 📊 CRITICAL

**Combines evidence, confidence, validation, and integration testing**

**1. Evidence Requirements** (for all claims):
```
[Finding]: [Evidence] (Scope: [coverage])
📍 Source: <cite file="[path]" line="[line]">[specific quote]</cite>
```

**2. Confidence Levels** (mandatory for analysis):
- 🟢 **HIGH** (90-100%): Multiple sources, systematic method
- 🟡 **MEDIUM** (70-89%): Single source, some assumptions  
- 🔴 **LOW** (50-69%): Limited data, documented constraints
- ⚫ **INSUFFICIENT** (<50%): Cannot verify claim

**3. Metrics Validation** (for performance claims):
```python
REALISTIC_RANGES = {
    "response_time_ms": {"min": 0.1, "max": 500},
    "memory_usage_mb": {"min": 10, "max": 1000},
    "database_connections": {"min": 1, "max": 50}
}
# Validate all metrics against realistic ranges
```

**4. Integration Testing** (for external services):
- Test connectivity and auth
- Verify realistic response times (1-100ms for DB)
- Check error handling
- Validate data format

---

### **ERROR & RE-VERIFICATION PROTOCOL** 🔄 CRITICAL

**Triggered by: errors, user challenges accuracy, contradictions found**

**Immediate Investigation** (for any error):
```bash
echo "🛑 STOP - Investigating: [error type]"
# 1. Examine full error
# 2. Verify file/import exists  
# 3. Test minimal reproduction
# 4. Check related functionality
```

**Systematic Re-verification** (when accuracy questioned):
1. Review original claim and evidence
2. Use DIFFERENT methodology for verification
3. Search for counter-evidence
4. Document findings:
   ```
   Original: [claim]
   Status: [CONFIRMED|MODIFIED|CONTRADICTED]
   New Evidence: [findings]
   Final: [corrected claim with confidence]
   ```

---

## Compliance Checklist

Before claiming complete:
□ User instructions followed (Level 1 authority)
□ Appropriate rules applied  
□ Evidence provided for claims
□ Confidence documented
□ Metrics validated as realistic
□ Errors investigated fully
□ Knowledge stored in memory

---

## Key Improvements in Consolidation:

1. **Reduced redundancy** - Phase gates no longer repeat rule content
2. **Unified verification** - All validation/evidence rules combined
3. **Streamlined error handling** - Error correction and re-verification merged
4. **Clearer hierarchy** - Rules organized by criticality and purpose
5. **Faster scanning** - ~50% reduction in size while preserving all functionality

The consolidated version maintains all critical safety checks while reducing cognitive load and context window usage.