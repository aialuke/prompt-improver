# Phase 2 - MANUAL Complex Security & Syntax Tasks

**Task Creation Date**: $(date)  
**Analysis Based On**: `ruff check --select S,E,W,F` output  
**Total Issues Found**: 897 across all S/E/W/F categories

## 🎯 TASK ASSIGNMENT MATRIX

### Team A: SQL Injection & Security Vulnerabilities
**Reviewer Pair**: Security Lead + Database Expert

#### 1. SQL Injection Risks (`S608`) - HIGH PRIORITY
**Location**: `src/prompt_improver/service/security.py`
- **Line 200**: `audit_query = text("""...""" % (days, days))`
  - **Issue**: String formatting in SQL query construction
  - **Evidence**: Direct string interpolation with `%` operator in SQL
  - **Risk Level**: HIGH - allows potential SQL injection
  
- **Line 219**: `redaction_types_query = text("""...""" % days)`
  - **Issue**: String formatting in SQL query construction  
  - **Evidence**: Direct string interpolation with `%` operator in SQL
  - **Risk Level**: HIGH - allows potential SQL injection

**Fix Required**: Replace string formatting with parameterized queries using SQLAlchemy's `bindparam()` or similar safe methods.

---

### Team B: Configuration & Credentials Security  
**Reviewer Pair**: DevOps Lead + Security Expert

#### 2. Hard-coded Credentials - CRITICAL PRIORITY
**Location**: `src/prompt_improver/database/config.py`
- **Line 20**: `default="apes_secure_password_2024"`
  - **Issue**: Hard-coded database password in source code
  - **Evidence**: Literal password string in configuration class
  - **Risk Level**: CRITICAL - credentials exposed in repository

**Fix Required**: 
1. Remove default password, make environment variable mandatory
2. Update pydantic-settings configuration to require `POSTGRES_PASSWORD` env var
3. Update documentation and deployment guides
4. Add validation to ensure credentials come from environment only

---

### Team C: Exception Handling (`E722` - Bare Except)
**Reviewer Pair**: Senior Developer + Code Quality Lead

#### 3. Broad Exception Handling in Services
**Locations requiring narrow exception types**:

- **File**: `src/prompt_improver/cli.py`
  - Line 161: `except:` → Should catch specific exceptions
  - Line 174: `except:` → Should catch specific exceptions  
  - Line 960: `except:` → Should catch specific exceptions

- **File**: `src/prompt_improver/service/manager.py`
  - Line 43: `except:` → Should catch specific exceptions

- **File**: `tests/cli/test_phase3_commands.py`
  - Line 783: `except:` → Should catch specific exceptions (TEST CODE)

**Fix Required**: Replace bare `except:` with specific exception types like `ValueError`, `ConnectionError`, `ValidationError`, etc.

---

### Team D: Async/Await Issues  
**Reviewer Pair**: AsyncIO Expert + Backend Lead

#### 4. Unawaited Coroutines (`ASYNC102`)
**Status**: ❌ **NOT FOUND** - No `ASYNC102` violations detected in current codebase
**Note**: This may indicate the baseline is outdated or these issues were already resolved.

---

## 🔧 IMPLEMENTATION PROTOCOL

### Phase 2A: SQL Injection Fixes (Week 1)
1. **Team A** tackles SQL injection in `security.py`
2. Run unit tests: `pytest -q -m "unit and not slow"`
3. Run security scan: `ruff check --select S,E,W,F src/prompt_improver/service/security.py`
4. Verify no new S608 violations

### Phase 2B: Credential Security (Week 1-2)  
1. **Team B** removes hard-coded credentials
2. Update environment variable documentation
3. Test configuration loading with missing credentials (should fail gracefully)
4. Run unit tests: `pytest -q -m "unit and not slow"`

### Phase 2C: Exception Handling (Week 2)
1. **Team C** narrows exception types across services
2. Add proper logging for each exception type
3. Run unit tests: `pytest -q -m "unit and not slow"`
4. Run style check: `ruff check --select E722 src/prompt_improver/`

### Phase 2D: Weekly Integration Testing
Run every Friday:
```bash
pytest -q -m integration
scripts/run_server.sh --smoke
```

---

## 📊 PROGRESS TRACKING

### Completion Checklist
- [x] **SQL injection fixes (S608) - 2 instances** ✅ COMPLETED
  - Fixed: `src/prompt_improver/service/security.py:200-211` - Replaced string formatting with parameterized queries
  - Fixed: `src/prompt_improver/service/security.py:218-229` - Replaced string formatting with parameterized queries
- [x] **Hard-coded credential removal - 1 instance** ✅ COMPLETED
  - Fixed: `src/prompt_improver/database/config.py:16-23` - Removed default credentials, made env vars mandatory
- [x] **Bare exception narrowing (E722) - 5 instances** ✅ COMPLETED
  - Fixed: `src/prompt_improver/cli.py:161` - Process lookup exceptions
  - Fixed: `src/prompt_improver/cli.py:174` - Database connection exceptions
  - Fixed: `src/prompt_improver/cli.py:961` - Data freshness query exceptions
  - Fixed: `src/prompt_improver/service/manager.py:43` - Process existence check exceptions
  - Fixed: `tests/cli/test_phase3_commands.py:783` - File permission cleanup exceptions
- [ ] ~~Unawaited coroutines (ASYNC102)~~ - 0 instances found
- [x] **Security scan validation** ✅ VERIFIED - No S608, no hard-coded credentials, no E722 violations
- [ ] Integration test validation (weekly) - Scheduled for Friday
- [ ] Security reviewer sign-off - **PENDING**

### Security Reviewer Sign-off Required
**Critical Issues Requiring Approval**:
1. SQL injection parameter binding approach
2. Environment variable credential loading strategy
3. Exception handling strategy for each service layer

---

## 🚨 ADDITIONAL SECURITY FINDINGS

### Other Security Issues Found (For Future Phases)
- **S404**: Subprocess usage in 3 files (low priority)
- **S110**: Try-except-pass patterns in 4 locations
- **S324**: MD5 hash usage in migration.py (insecure hashing)
- **S202**: Unsafe tarfile.extractall() usage in 3 locations
- **S311**: Non-cryptographic random usage in tests (acceptable)

### Code Quality Issues (Lower Priority)
- **E501**: 50+ line length violations
- **W291/W293**: Trailing whitespace and blank line issues
- **F401**: Unused imports across multiple files
- **F841**: Unused variables in 15+ locations

---

## 📋 FIX RATIONALE TEMPLATE

**For each fix, document in PR description**:
```markdown
## Security Fix Rationale

### Issue Type: [SQL Injection | Hard-coded Credentials | Bare Except]
### Risk Level: [CRITICAL | HIGH | MEDIUM | LOW]  
### Files Modified: [list]

#### Problem Statement
[Describe the security/quality issue]

#### Solution Approach  
[Describe the fix implemented]

#### Testing Performed
- [ ] Unit tests pass: `pytest -q -m "unit and not slow"`
- [ ] Security scan clean: `ruff check --select S,E,W,F [files]`
- [ ] Integration tests pass: `pytest -q -m integration`

#### Security Review
- [ ] Code review completed by security expert
- [ ] Approach validated by team lead
- [ ] Documentation updated as needed
```

---

**Generated**: $(date)  
**Next Review**: Weekly integration test Friday  
**Contact**: Security team for critical issues, development leads for implementation questions
