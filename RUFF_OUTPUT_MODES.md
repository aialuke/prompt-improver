# Ruff Output Modes & Reporting Guide

## Overview

Yes! Ruff provides **comprehensive error reporting** by default. You have full control over whether to see issues, preview fixes, or auto-apply changes.

## 🔍 **Error Reporting Modes** (Default Behavior)

### **1. Basic Error Listing** (Default)
```bash
ruff check src/ --preview
```
**Output:** Detailed list of all errors with:
- ✅ File path and line numbers
- ✅ Error codes and descriptions  
- ✅ Helpful links to documentation
- ✅ Fix suggestions where available
- ✅ Count of total errors and fixable errors

**Example Output:**
```
src/prompt_improver/services/prompt_improvement.py:331:52: UP006 [*] Use `dict` instead of `Dict` for type annotation
    |
329 |         return max(0.0, min(1.0, structure_score))
330 |     
331 |     def _calculate_improvement_score(self, before: Dict[str, float], after: Dict[str, float]) -> float:
    |                                                    ^^^^ UP006
332 |         """Calculate improvement score based on metrics"""
    |
    = help: Replace with `dict`

Found 760 errors.
[*] 486 fixable with the `--fix` option (163 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

### **2. Statistics Summary**
```bash
ruff check src/ --preview --statistics
```
**Output:** Aggregated error counts by rule type:
```
174     W293    [*] blank-line-with-whitespace
 85     D415    [ ] missing-terminal-punctuation
 72     UP045   [*] non-pep604-annotation-optional
 70     UP006   [*] non-pep585-annotation
 63     COM812  [*] missing-trailing-comma
 57     D212    [*] multi-line-summary-first-line
 31     PLR6301 [ ] no-self-use
```

### **3. Show Changes Without Applying** (Preview Mode)
```bash
ruff check src/ --preview --diff
```
**Output:** Shows exactly what would be changed:
```diff
- from typing import Dict, List, Optional, Any
+ from typing import Any
```

### **4. JSON Output** (Programmatic Processing)
```bash
ruff check src/ --preview --output-format json
```
**Output:** Structured JSON for CI/CD integration:
```json
[
  {
    "code": "UP006",
    "filename": "/path/to/file.py",
    "location": {"column": 52, "row": 331},
    "message": "Use `dict` instead of `Dict` for type annotation",
    "url": "https://docs.astral.sh/ruff/rules/non-pep585-annotation"
  }
]
```

## 🔧 **Auto-Fix Modes** (Optional)

### **5. Safe Auto-Fix**
```bash
ruff check src/ --preview --fix
```
**Behavior:** 
- ✅ Automatically fixes safe issues (marked with `[*]`)
- ✅ Reports remaining unfixable issues
- ✅ Preserves original files as backups

### **6. Unsafe Auto-Fix** (Use with caution)
```bash
ruff check src/ --preview --fix --unsafe-fixes
```
**Behavior:**
- ⚠️ Applies potentially unsafe fixes
- ⚠️ May change code behavior
- ⚠️ Use only when you understand the changes

## 📊 **Specialized Output Formats**

### **7. GitHub Actions Format**
```bash
ruff check src/ --preview --output-format github
```
**Output:** GitHub-compatible annotations for CI/CD

### **8. GitLab CI Format**
```bash
ruff check src/ --preview --output-format gitlab
```
**Output:** GitLab CI-compatible format

### **9. Quiet Mode**
```bash
ruff check src/ --preview --quiet
```
**Output:** Minimal output, just essential errors

### **10. Verbose Mode**
```bash
ruff check src/ --preview --verbose
```
**Output:** Detailed debugging information

## 🎯 **Targeted Reporting**

### **11. Specific Rule Categories**
```bash
# Only show security issues
ruff check src/ --preview --select S

# Only show performance issues  
ruff check src/ --preview --select PERF

# Show multiple categories
ruff check src/ --preview --select E,W,F
```

### **12. File-Specific Checking**
```bash
# Check single file
ruff check src/prompt_improver/main.py --preview

# Check specific directory
ruff check src/services/ --preview

# Check with pattern
ruff check src/**/*.py --preview
```

### **13. Exit Code Behavior**
```bash
# Exit with code 0 even if errors found (CI-friendly)
ruff check src/ --preview --exit-zero

# Normal behavior (exit code 1 if errors found)
ruff check src/ --preview
```

## 🚀 **Practical Workflow Commands**

### **For Development** (See issues without changing code)
```bash
# Quick overview
ruff check src/ --preview --statistics

# Detailed errors for specific file
ruff check src/prompt_improver/services/prompt_improvement.py --preview

# See what would be fixed
ruff check src/ --preview --diff
```

### **For CI/CD** (Automated reporting)
```bash
# JSON output for processing
ruff check src/ --preview --output-format json > ruff-report.json

# GitHub Actions integration
ruff check src/ --preview --output-format github

# Fail build on errors
ruff check src/ --preview
```

### **For Gradual Adoption** (Incremental fixes)
```bash
# Fix only safe issues
ruff check src/ --preview --fix

# Fix specific rule category
ruff check src/ --preview --select W293 --fix

# Preview changes for specific rules
ruff check src/ --preview --select UP006,UP045 --diff
```

## 📈 **Dead Code Detection Output**

### **Unused Imports**
```bash
ruff check src/ --preview --select F401
```
**Output:**
```
src/file.py:1:1: F401 [*] `pandas` imported but unused
```

### **Unused Variables**
```bash
ruff check src/ --preview --select F841
```
**Output:**
```
src/file.py:10:5: F841 [*] Local variable `unused_var` is assigned to but never used
```

### **Unused Function Arguments**
```bash
ruff check src/ --preview --select ARG
```
**Output:**
```
src/file.py:15:20: ARG002 Unused method argument: `param`
```

## 🎛️ **Configuration-Based Reporting**

### **Respect Configuration**
```bash
# Uses settings from pyproject.toml
ruff check src/ --preview
```

### **Override Configuration**
```bash
# Ignore configuration, use specific rules
ruff check src/ --preview --select ALL --ignore D100,D103
```

## 💡 **Best Practices**

### **1. Development Workflow**
```bash
# Daily: Check for new issues
ruff check src/ --preview --statistics

# Before commit: See what would be fixed
ruff check src/ --preview --diff

# Clean commit: Fix safe issues
ruff check src/ --preview --fix
```

### **2. CI/CD Integration**
```bash
# Pre-commit hook
ruff check . --preview --output-format github

# Build pipeline
ruff check . --preview --output-format json --exit-zero
```

### **3. Large Codebases**
```bash
# Focus on critical issues first
ruff check src/ --preview --select E,W,F --statistics

# Gradual adoption by category
ruff check src/ --preview --select B,C4,SIM --fix
```

## 📋 **Summary**

**Default Behavior: ERROR REPORTING** ✅
- `ruff check` shows all issues without fixing
- Rich, detailed output with line numbers and explanations
- Clear indication of which errors can be auto-fixed

**Auto-fixing is OPTIONAL** ⚙️
- Must explicitly use `--fix` flag
- Safe fixes only unless `--unsafe-fixes` specified
- Always shows summary of changes made

**Multiple Output Formats** 📊
- Human-readable (default)
- JSON for automation
- GitHub/GitLab for CI/CD
- Statistics for overview

**Your Configuration** 🎯
With your new comprehensive configuration, you'll get:
- **760 total issues detected** (real example from your codebase)
- **486 auto-fixable issues** (formatting, imports, annotations)
- **274 manual review issues** (logic, complexity, documentation)
- **Detailed categorization** by rule type and severity

The configuration **prioritizes visibility** over silent fixes, giving you full control over your code quality improvements. 