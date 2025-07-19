# Dependency Analysis Report

## Executive Summary

After comprehensive analysis of the project dependencies, I found:

1. **Mocking libraries are already consolidated** - Only `unittest.mock` is used throughout the codebase
2. **pytest-mock is unused** - Listed in requirements-test-real.txt but no actual usage found
3. **Many dependencies are actively used** - Core ML, database, and web framework dependencies are essential
4. **Some dependencies may be unused** - Several packages appear to have no direct imports

## Detailed Analysis

### Files Analyzed
- `requirements.lock` (264 dependencies)
- `requirements-dev.txt` (4 dependencies)  
- `requirements-test-real.txt` (29 dependencies)

### Mocking Libraries Status
✅ **ALREADY CONSOLIDATED**: All mocking uses `unittest.mock` (built-in)
❌ **UNUSED**: `pytest-mock` listed in requirements-test-real.txt but never imported

### Dependencies by Category

#### ✅ ACTIVELY USED (Keep)
**ML/Data Science:**
- mlflow, optuna (ML optimization)
- numpy, pandas, scipy (data processing)
- torch, transformers, sentence-transformers (ML models)
- scikit-learn, statsmodels (ML algorithms)

**Database:**
- redis, psycopg, asyncpg, sqlalchemy, alembic (database layers)

**Web Framework:**
- fastapi, starlette, uvicorn, pydantic (web API)

**CLI/UI:**
- typer, click, rich (command line interface)

**HTTP/Networking:**
- requests, httpx (HTTP clients)

**NLP:**
- nltk, textstat (natural language processing)

**Utilities:**
- joblib (parallel processing)
- prometheus_client (metrics)
- networkx (graph analysis)
- jsonschema (validation)
- mcp, mcp-context-sdk (MCP server)

#### ❓ POTENTIALLY UNUSED (Investigate)
**Azure/Cloud (likely unused):**
- adal, azure-* packages (25+ packages)
- boto3, botocore (AWS)
- google-* packages (5+ packages)

**Development Tools (possibly unused):**
- pip-audit, pipdeptree, pipreqs (dependency analysis tools)
- radon (code complexity)
- knack, mando, docopt, yarg (CLI frameworks not used)

**Legacy/Compatibility:**
- backports.*, six (Python 2/3 compatibility)
- contextlib2 (backport)

**Potentially Redundant:**
- CacheControl, cachetools (caching, but project uses Redis)
- durationpy, humanfriendly (time formatting)
- dynaconf (configuration management)

#### ⚠️ INDIRECT DEPENDENCIES (Keep - Required by others)
Many packages are likely indirect dependencies required by main packages:
- greenlet, grpcio (async/networking)
- packaging, setuptools, wheel (build tools)
- typing_extensions, importlib_metadata (Python compatibility)
- urllib3, certifi, charset-normalizer (HTTP/SSL)

## Recommendations

### Phase 1: Conservative Cleanup (LOW RISK)
1. **Remove pytest-mock** - Confirmed unused
2. **Remove obvious unused cloud packages** - Azure/AWS/Google packages if not used
3. **Remove deprecated development tools** - pip-audit, radon, etc. if not used in CI

### Phase 2: Careful Investigation (MEDIUM RISK)
1. **Audit legacy compatibility packages** - six, backports.*, contextlib2
2. **Review redundant caching libraries** - CacheControl, cachetools
3. **Check development CLI tools** - knack, mando, docopt, yarg

### Phase 3: Dependency Tree Analysis (HIGH RISK)
1. **Use pip-show/pipdeptree** to identify true indirect dependencies
2. **Remove only packages with zero reverse dependencies**
3. **Test extensively after each removal**

## Next Steps

1. Create backup of all requirement files
2. Start with Phase 1 (safe removals)
3. Run full test suite after each change
4. Use dependency analysis tools to verify removals
5. Document all changes made

## Files to Modify

- `requirements-test-real.txt` (remove pytest-mock)
- `requirements.lock` (remove confirmed unused packages)
- Update any imports if needed (none expected for unittest.mock)

## Safety Measures

- Backup all requirement files before changes
- Remove dependencies one at a time
- Run tests after each removal
- Use git to track changes for easy rollback
- Create verification script to detect unused dependencies