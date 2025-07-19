# Test Suite Remediation Plan

## Executive Summary
This plan addresses critical test suite issues preventing successful test execution. Based on research of 2025 best practices and analysis of 15+ distinct error patterns, we provide a prioritized, incremental approach to fix all issues.

## Priority 1: Critical Issues (Blocks All Tests)

### 1.1 Fix AutoML Import Error
**Issue**: `ModuleNotFoundError: No module named 'automl.test_automl_orchestrator'`
**Root Cause**: Empty `__init__.py` file in `tests/unit/automl/`
**Solution**:
```bash
# Create backup
cp tests/unit/automl/__init__.py tests/unit/automl/__init__.py.backup

# Fix the issue
echo "" > tests/unit/automl/__init__.py

# Alternative: If pytest discovery issues persist, try:
touch tests/unit/__init__.py
touch tests/unit/automl/__init__.py
```
**Rollback**: `cp tests/unit/automl/__init__.py.backup tests/unit/automl/__init__.py`

### 1.2 Configure Pytest Import Mode
**Issue**: Module resolution problems
**Solution**: Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = [
    "--import-mode=importlib",
]
pythonpath = ["src"]
```
**Best Practice**: As per pytest 2025 recommendations, use `importlib` mode for new projects

## Priority 2: Database Session Management

### 2.1 Fix SQLAlchemy Async Session Concurrency
**Issue**: "This session is provisioning a new connection; concurrent operations are not permitted"
**Root Cause**: Shared async sessions across tests
**Solution**:

Create `tests/fixtures/fixtures_db.py`:
```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="function")
async def test_db_session(test_db_engine):
    """Create isolated session per test"""
    async_session = sessionmaker(
        test_db_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()  # Ensure cleanup

# Configure pytest-asyncio
@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default policy but could switch to uvloop for performance"""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
```

Update `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

### 2.2 Fix Fixture Naming Mismatches
**Issue**: Tests expect `mock_db_session` but only `real_db_session` exists
**Solution**: 

Option A - Update test files (recommended):
```python
# In test files, replace:
# async def test_something(mock_db_session):
# With:
async def test_something(test_db_session):
```

Option B - Create alias fixture:
```python
# In conftest.py
@pytest.fixture
async def mock_db_session(test_db_session):
    """Alias for backward compatibility"""
    return test_db_session
```

## Priority 3: Environment and Dependencies

### 3.1 Handle Optional Dependencies
**Issue**: ML libraries not installed
**Solution**: Create test markers and skip conditions

`tests/conftest.py`:
```python
import pytest

# ML dependency checks
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import deap
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

# Create markers
pytest.mark.requires_sklearn = pytest.mark.skipif(
    not HAS_SKLEARN, reason="sklearn not installed"
)
pytest.mark.requires_deap = pytest.mark.skipif(
    not HAS_DEAP, reason="DEAP not installed"
)
```

Update test files:
```python
@pytest.mark.requires_sklearn
def test_ml_feature():
    from sklearn import ...
```

### 3.2 Infrastructure Requirements
**Issue**: Tests assume PostgreSQL and Docker are running
**Solution**: Add graceful checks

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def ensure_postgres():
    """Check PostgreSQL availability"""
    import psycopg
    try:
        conn = psycopg.connect("postgresql://localhost:5432/test")
        conn.close()
    except Exception:
        pytest.skip("PostgreSQL not available")

@pytest.fixture(scope="session") 
def ensure_docker():
    """Check Docker availability"""
    import subprocess
    try:
        subprocess.run(["docker", "ps"], check=True, capture_output=True)
    except Exception:
        pytest.skip("Docker not available")
```

## Priority 4: Test Organization Best Practices

### 4.1 Implement Test Structure
Following 2025 best practices, reorganize tests by pyramid level:

```bash
# Create directories if not exist
mkdir -p tests/unit tests/integration tests/e2e
mkdir -p tests/fixtures

# Move fixture files
mv tests/*fixtures*.py tests/fixtures/
```

### 4.2 Configure Test Discovery
Update `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    requires_db: Requires database
    requires_docker: Requires Docker
```

## Implementation Steps

### Phase 1: Emergency Fixes (Day 1)
1. **Hour 1**: Fix import error in automl test
2. **Hour 2**: Update pyproject.toml with pytest configurations
3. **Hour 3**: Fix database session fixtures
4. **Hour 4**: Run tests and verify basic functionality

### Phase 2: Systematic Fixes (Days 2-3)
1. Update all fixture references in test files
2. Add dependency checks and markers
3. Implement infrastructure checks
4. Test incrementally after each fix

### Phase 3: Best Practice Implementation (Days 4-5)
1. Reorganize test structure
2. Update CI/CD configurations
3. Document new test conventions
4. Train team on new practices

## Backup Strategy
Before each change:
```bash
# Create timestamped backup
tar -czf test_backup_$(date +%Y%m%d_%H%M%S).tar.gz tests/

# Git backup
git stash save "Pre-test-remediation backup"
```

## Validation Checklist
After each fix:
- [ ] Run `pytest --collect-only` to verify test discovery
- [ ] Run subset of tests: `pytest tests/unit -k "not ml" -v`
- [ ] Check for new errors: `pytest --tb=short -x`
- [ ] Verify no regression: `git diff tests/`

## Monitoring and Prevention
1. Add pre-commit hooks for test validation
2. Set up CI to run tests in isolated environments
3. Document dependency requirements clearly
4. Regular test suite health checks

## Success Metrics
- **Before**: 0/1033 tests collected due to import error
- **Target**: 1000+ tests collected, 90%+ passing
- **Measure**: Test execution time, coverage percentage

## Rollback Procedures
If issues arise:
1. Restore from backup: `tar -xzf test_backup_[timestamp].tar.gz`
2. Reset git: `git stash pop`
3. Document what failed for future attempts

## References
- [Pytest Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Pytest-asyncio Configuration](https://github.com/pytest-dev/pytest-asyncio)
- [SQLAlchemy Async Best Practices](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)