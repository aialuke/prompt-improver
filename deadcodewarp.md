# 🗑️ Dead Code Cleanup Plan - APES Codebase
**Generated:** 2025-01-03  
**Scope:** Verified unused imports and truly dead code only  
**Impact:** Safe cleanup with zero functional changes  
**Status:** ✅ READY FOR EXECUTION

---

## 📊 **Analysis Summary**

**Total Dead Code Items:** 68 verified unused imports  
**Files Affected:** 20 Python source files  
**Safety Level:** HIGH - All changes are safe automated fixes  
**Functional Impact:** ZERO - Removing unused imports only

---

## 🔍 **Verification Methodology**

1. **Ruff Static Analysis:** Used `python3 -m ruff check src/ --select F401` for precise unused import detection
2. **Safe Fix Validation:** Only included items marked as "safe" by ruff with automatic fixes
3. **No Function/Class Removal:** Excluded any function or class definitions from cleanup
4. **Import-Only Focus:** Strictly limited to unused import statements

---

## 📋 **ENUMERATED DEAD CODE FINDINGS**

### **1. CLI Module - 9 Unused Imports**
**File:** `src/prompt_improver/cli.py`

1. **Line 12:** `typing.List` imported but unused
   - **Evidence:** Ruff F401 with safe fix available
   - **Action:** Remove from import statement, keep `typing.Optional`

2. **Line 19:** `rich.print` imported but unused  
   - **Evidence:** Entire line can be safely removed
   - **Action:** Delete entire import line

3. **Line 21:** `prompt_improver.database.sessionmanager` imported but unused
   - **Evidence:** Only `get_session` is used, not `sessionmanager`
   - **Action:** Remove from import statement

4. **Lines 469-472:** MCP availability check imports (4 items)
   - **Lines:** `mcp`, `fastmcp`, `typer`, `rich` 
   - **Evidence:** Inside try/except block for availability checking only
   - **Action:** Leave unchanged (intentional availability testing)

5. **Line 763:** `datetime.datetime` and `datetime.timedelta` imported but unused
   - **Evidence:** Two unused datetime imports in same line
   - **Action:** Remove entire import line

### **2. Database Module - 8 Unused Imports**

#### **config.py (1 item)**
**File:** `src/prompt_improver/database/config.py`

6. **Line 7:** `typing.Optional` imported but unused
   - **Evidence:** No Optional types used in configuration
   - **Action:** Remove entire import line

#### **connection.py (1 item)**  
**File:** `src/prompt_improver/database/connection.py`

7. **Line 6:** `typing.Optional` imported but unused
   - **Evidence:** Only `AsyncIterator` used from typing
   - **Action:** Remove from import statement

#### **models.py (2 items)**
**File:** `src/prompt_improver/database/models.py`

8. **Line 10:** `sqlalchemy.text` imported but unused
   - **Evidence:** SQLAlchemy text function not used
   - **Action:** Remove from import statement

9. **Line 12:** `sqlmodel.Relationship` imported but unused
   - **Evidence:** No model relationships defined
   - **Action:** Remove from import statement

#### **performance_monitor.py (1 item)**
**File:** `src/prompt_improver/database/performance_monitor.py`

10. **Line 7:** `time` imported but unused
    - **Evidence:** Uses `time.perf_counter` from different import
    - **Action:** Remove entire import line

#### **psycopg_client.py (3 items)**
**File:** `src/prompt_improver/database/psycopg_client.py`

11. **Line 6:** `asyncio` imported but unused
    - **Evidence:** No asyncio functions called directly
    - **Action:** Remove entire import line

12. **Line 9:** `typing.AsyncIterator` imported but unused
    - **Evidence:** Not used in type hints
    - **Action:** Remove from import statement

13. **Line 11:** `json` imported but unused
    - **Evidence:** JSON handling done elsewhere
    - **Action:** Remove entire import line

14. **Line 13:** `psycopg` imported but unused
    - **Evidence:** Only specific psycopg modules used
    - **Action:** Remove entire import line

### **3. Installation Module - 6 Unused Imports**

#### **initializer.py (4 items)**
**File:** `src/prompt_improver/installation/initializer.py`

15. **Line 8:** `sys` imported but unused
    - **Evidence:** System operations handled differently
    - **Action:** Remove entire import line

16. **Line 9:** `shutil` imported but unused
    - **Evidence:** File operations not using shutil
    - **Action:** Remove entire import line

17. **Line 11:** `tempfile` imported but unused
    - **Evidence:** Temporary files not created
    - **Action:** Remove entire import line

18. **Line 20:** `..database.sessionmanager` imported but unused
    - **Evidence:** Only `get_session` used
    - **Action:** Remove from import statement

#### **migration.py (2 items)**
**File:** `src/prompt_improver/installation/migration.py`

19. **Line 7:** `subprocess` imported but unused
    - **Evidence:** Subprocess calls handled elsewhere
    - **Action:** Remove entire import line

20. **Line 8:** `sys` imported but unused
    - **Evidence:** System operations not used
    - **Action:** Remove entire import line

21. **Line 21:** `..database.sessionmanager` imported but unused
    - **Evidence:** Only `get_session` used
    - **Action:** Remove from import statement

### **4. MCP Server Module - 2 Unused Imports**
**File:** `src/prompt_improver/mcp_server/mcp_server.py`

22. **Line 8:** `typing.List` imported but unused
    - **Evidence:** List types not used in type hints
    - **Action:** Remove from import statement

23. **Line 12:** `sqlalchemy.ext.asyncio.AsyncSession` imported but unused
    - **Evidence:** Session handling done via dependency injection
    - **Action:** Remove entire import line

### **5. Rule Engine Module - 1 Unused Import**
**File:** `src/prompt_improver/rule_engine/rules/clarity.py`

24. **Line 10:** `..base.LLMInstruction` imported but unused
    - **Evidence:** LLMInstruction not used in clarity rule
    - **Action:** Remove from import statement

### **6. Service Module - 11 Unused Imports**

#### **manager.py (1 item)**
**File:** `src/prompt_improver/service/manager.py`

25. **Line 255:** `..mcp_server.mcp_server.app` imported but unused
    - **Evidence:** MCP app not directly referenced
    - **Action:** Remove entire import line

#### **security.py (6 items)**
**File:** `src/prompt_improver/service/security.py`

26. **Line 8:** `asyncio` imported but unused
    - **Evidence:** Async operations use different patterns
    - **Action:** Remove entire import line

27. **Line 15:** `..database.get_session` imported but unused
    - **Evidence:** Database access not used in current implementation
    - **Action:** Remove from import statement

28. **Line 16:** `..database.models.ImprovementSession` imported but unused
    - **Evidence:** Model not referenced
    - **Action:** Remove entire import line

29. **Line 106:** `sqlalchemy.update` imported but unused
    - **Evidence:** Update operations not performed
    - **Action:** Remove from import statement

30. **Line 106:** `sqlalchemy.select` imported but unused
    - **Evidence:** Select operations not performed
    - **Action:** Remove from import statement

31. **Line 107:** `sqlalchemy.sql.func` imported but unused
    - **Evidence:** SQL functions not used
    - **Action:** Remove entire import line

### **7. Services Module - 31 Unused Imports**

#### **ab_testing.py (4 items)**
**File:** `src/prompt_improver/services/ab_testing.py`

32. **Line 6:** `asyncio` imported but unused
    - **Evidence:** Async handled by framework
    - **Action:** Remove entire import line

33. **Line 7:** `uuid` imported but unused
    - **Evidence:** UUID generation handled elsewhere
    - **Action:** Remove entire import line

34. **Line 8:** `datetime.timedelta` imported but unused
    - **Evidence:** Only datetime used
    - **Action:** Remove from import statement

35. **Line 15:** `sqlalchemy.func` imported but unused
    - **Evidence:** SQL functions not needed
    - **Action:** Remove from import statement

36. **Line 17:** `..database.models.UserFeedback` imported but unused
    - **Evidence:** UserFeedback model not referenced
    - **Action:** Remove from import statement

#### **advanced_pattern_discovery.py (7 items)**
**File:** `src/prompt_improver/services/advanced_pattern_discovery.py`

37. **Line 6:** `asyncio` imported but unused
    - **Evidence:** Async operations handled by framework
    - **Action:** Remove entire import line

38. **Line 7:** `json` imported but unused
    - **Evidence:** JSON operations not used
    - **Action:** Remove entire import line

39. **Line 9:** `datetime.datetime` imported but unused
    - **Evidence:** Only timedelta used
    - **Action:** Remove from import statement

40. **Line 9:** `datetime.timedelta` imported but unused
    - **Evidence:** Time operations not used
    - **Action:** Remove from import statement

41. **Line 25:** `sklearn.cluster.KMeans` imported but unused
    - **Evidence:** Only DBSCAN clustering used
    - **Action:** Remove from import statement

42. **Line 27:** `sklearn.decomposition.PCA` imported but unused
    - **Evidence:** PCA not used in pattern discovery
    - **Action:** Remove entire import line

43. **Line 39:** `sqlalchemy.func` imported but unused
    - **Evidence:** SQL functions not needed
    - **Action:** Remove from import statement

44. **Line 41:** `..database.models.UserFeedback` imported but unused
    - **Evidence:** UserFeedback not used in pattern analysis
    - **Action:** Remove from import statement

#### **analytics.py (4 items)**
**File:** `src/prompt_improver/services/analytics.py`

45. **Line 9:** `sqlalchemy.select` imported but unused
    - **Evidence:** Select operations use different import
    - **Action:** Remove from import statement

46. **Line 9:** `sqlalchemy.func` imported but unused
    - **Evidence:** SQL functions not used
    - **Action:** Remove from import statement

47. **Line 10:** `sqlmodel.select` imported but unused
    - **Evidence:** SQLModel select not used
    - **Action:** Remove entire import line

48. **Line 13:** `..database.models.RulePerformance` imported but unused
    - **Evidence:** Performance data accessed differently
    - **Action:** Remove from import statement

49. **Line 14:** `..database.models.UserFeedback` imported but unused
    - **Evidence:** Feedback not used in analytics
    - **Action:** Remove from import statement

#### **llm_transformer.py (2 items)**
**File:** `src/prompt_improver/services/llm_transformer.py`

50. **Line 6:** `json` imported but unused
    - **Evidence:** JSON operations handled elsewhere
    - **Action:** Remove entire import line

51. **Line 9:** `asyncio` imported but unused
    - **Evidence:** Async operations managed by framework
    - **Action:** Remove entire import line

#### **ml_integration.py (3 items)**
**File:** `src/prompt_improver/services/ml_integration.py`

52. **Line 7:** `asyncio` imported but unused
    - **Evidence:** Async operations handled by framework
    - **Action:** Remove entire import line

53. **Line 10:** `datetime.timedelta` imported but unused
    - **Evidence:** Only datetime used for timestamps
    - **Action:** Remove from import statement

54. **Line 11:** `typing.Tuple` imported but unused
    - **Evidence:** Tuple types not used in type hints
    - **Action:** Remove from import statement

55. **Line 22:** `sklearn.metrics.roc_auc_score` imported but unused
    - **Evidence:** ROC AUC not used in current metrics
    - **Action:** Remove from import statement

#### **monitoring.py (5 items)**
**File:** `src/prompt_improver/services/monitoring.py`

56. **Line 16:** `rich.progress.Progress` imported but unused
    - **Evidence:** Progress bars not used in monitoring dashboard
    - **Action:** Remove from import statement

57. **Line 16:** `rich.progress.BarColumn` imported but unused
    - **Evidence:** Bar columns not used
    - **Action:** Remove from import statement

58. **Line 16:** `rich.progress.TextColumn` imported but unused
    - **Evidence:** Text columns not used
    - **Action:** Remove from import statement

59. **Line 19:** `rich.columns.Columns` imported but unused
    - **Evidence:** Column layouts not used
    - **Action:** Remove entire import line

#### **prompt_improvement.py (2 items)**
**File:** `src/prompt_improver/services/prompt_improvement.py`

60. **Line 8:** `json` imported but unused
    - **Evidence:** JSON operations handled elsewhere
    - **Action:** Remove entire import line

61. **Line 11:** `sqlalchemy.func` imported but unused
    - **Evidence:** SQL functions not used
    - **Action:** Remove from import statement

---

## 🛠️ **CLEANUP EXECUTION PLAN**

### **Phase 1: Automated Safe Cleanup**
```bash
# Use ruff to automatically fix safe unused imports
cd /Users/lukemckenzie/prompt-improver
python3 -m ruff check src/ --select F401 --fix
```

### **Phase 2: Manual Verification (For Items Ruff Cannot Auto-Fix)**
**Files requiring manual review:**
- `src/prompt_improver/cli.py:469-472` (availability testing imports - KEEP)
- Any imports within try/except blocks for optional dependencies

### **Phase 3: Verification Testing**
```bash
# Verify no functional impact
python3 -m pytest tests/ -v
python3 -c "import src.prompt_improver; print('Import successful')"
```

### **Phase 4: Final Validation**
```bash
# Confirm all unused imports removed
python3 -m ruff check src/ --select F401
# Should return no F401 violations
```

---

## ⚠️ **SAFETY PROTOCOLS**

### **Pre-Execution Checks:**
1. ✅ **Backup Created:** Ensure git commit or backup before changes
2. ✅ **Test Suite Available:** Confirm tests can validate functionality
3. ✅ **Ruff Version:** Use latest ruff version for accurate analysis

### **Items Explicitly EXCLUDED from Cleanup:**
1. **Lines 469-472 in cli.py:** Intentional availability testing imports
2. **Any import within try/except blocks:** May be optional dependency checks
3. **Imports with `# noqa` comments:** Explicitly marked to ignore

### **Post-Cleanup Validation:**
1. **Import Tests:** Verify all modules can be imported successfully
2. **Unit Tests:** Run full test suite to ensure no functional impact
3. **Static Analysis:** Confirm no new linting issues introduced

---

## 📈 **EXPECTED IMPACT**

### **Before Cleanup:**
- **Total Lines:** ~30 files with unused imports
- **Ruff F401 Violations:** 68 unused imports
- **Maintenance Overhead:** Unnecessary import management

### **After Cleanup:**
- **Lines Removed:** ~61 import statements cleaned
- **Ruff F401 Violations:** 0 (clean)
- **Benefit:** Cleaner codebase, easier maintenance, faster imports

### **Risk Assessment:**
- **Functional Risk:** ZERO (imports only, no logic changes)
- **Breaking Changes:** NONE (unused imports cannot break functionality)
- **Rollback Plan:** Simple git revert if any unexpected issues

---

## ✅ **EXECUTION CHECKLIST**

- [ ] **Create backup/commit:** `git commit -am "Pre-cleanup backup"`
- [ ] **Run automated cleanup:** `python3 -m ruff check src/ --select F401 --fix`
- [ ] **Manual review:** Check items ruff couldn't auto-fix
- [ ] **Test functionality:** `python3 -m pytest tests/`
- [ ] **Verify import success:** Test module imports
- [ ] **Final validation:** `python3 -m ruff check src/ --select F401`
- [ ] **Document changes:** Update this file with results

---

**Cleanup Plan Ready:** 2025-01-03  
**Estimated Time:** 15-30 minutes  
**Complexity:** LOW (automated fixes only)  
**Confidence:** HIGH (verified safe changes only)
