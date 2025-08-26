# P3.1 Syntax Error Remediation Roadmap

## ✅ Phase 3.1 COMPLETE - Inventory Analysis
**Status**: COMPLETE  
**Findings**: 48 syntax errors across 48 files identified and categorized  
**Root Cause**: Malformed ML lazy loading conversion patterns from Phase 2  

## 🎯 Phase 3.2: ML Lazy Loading Import Remediation (CRITICAL)

### Priority 1 Files - Core ML Algorithms (Est: 2 hours)
```bash
# High-impact ML algorithm files requiring immediate fix
src/prompt_improver/ml/analytics/session_comparison_analyzer.py:26
src/prompt_improver/ml/types.py:10  # Core type definitions
src/prompt_improver/ml/dimensionality/services/classical_reducer_service.py:19
src/prompt_improver/ml/dimensionality/services/neural_reducer_service.py:18
src/prompt_improver/ml/preprocessing/generators/statistical_generator.py:18
```

### Standard Remediation Pattern:
```python
# BEFORE (Broken):
from get_sklearn().cluster import KMeans
from get_numpy().typing import NDArray

# AFTER (Fixed):
from prompt_improver.core.utils.lazy_ml_loader import get_sklearn_cluster
from prompt_improver.core.utils.lazy_ml_loader import get_numpy

# Usage in code:
sklearn_cluster = get_sklearn_cluster()
KMeans = sklearn_cluster.KMeans

numpy = get_numpy() 
NDArray = numpy.typing.NDArray
```

## 🔧 Phase 3.3: Try-Except Block Reconstruction (HIGH)

### Priority 2 Files - Control Flow Restoration (Est: 1.5 hours)
```bash
# Files with interrupted try-except blocks
src/prompt_improver/analytics/unified/performance_analytics_component.py:681
src/prompt_improver/ml/analysis/linguistic_analyzer.py:489
src/prompt_improver/ml/health/ml_health_monitor.py:478
src/prompt_improver/ml/learning/algorithms/context_aware_weighter.py:22
```

### Standard Fix Pattern:
```python
# BEFORE (Broken):
try:
    processing_logic()
    import psutil
from prompt_improver.core.utils.lazy_ml_loader import get_numpy  # ← MISPLACED

# AFTER (Fixed):
from prompt_improver.core.utils.lazy_ml_loader import get_numpy
try:
    processing_logic()
    import psutil
except Exception as e:
    # Proper exception handling restored
```

## 🔄 Phase 3.4: Import Structure Consolidation (MEDIUM)

### Priority 3 Files - Import Block Cleanup (Est: 1 hour)
```bash
# Files with mixed import patterns
src/prompt_improver/analytics/unified/ab_testing_component.py:18
src/prompt_improver/analytics/unified/ml_analytics_component.py:95
src/prompt_improver/analytics/unified/session_analytics_component.py:104
```

### Standard Fix Pattern:
```python
# BEFORE (Broken):
from .protocols import (
from prompt_improver.core.utils.lazy_ml_loader import get_numpy  # ← MISPLACED
    ABTestingProtocol,
    ComponentHealth,
)

# AFTER (Fixed):
from .protocols import (
    ABTestingProtocol,
    ComponentHealth,
)
from prompt_improver.core.utils.lazy_ml_loader import get_numpy
```

## 🧹 Phase 3.5: Final Cleanup (LOW)

### Priority 4 Files - Structural Issues (Est: 30 minutes)
```bash
# Files with indentation/structural errors
src/prompt_improver/ml/evaluation/experiment_orchestrator.py:846
src/prompt_improver/performance/validation/memory_profiler.py:1004
src/prompt_improver/performance/validation/regression_detector.py:939
```

## 🧪 Validation Strategy

### Per-Phase Validation:
```bash
# After each phase, validate fixes
python3 -c "
import ast
from pathlib import Path
errors = []
for f in Path('src').rglob('*.py'):
    try:
        with open(f) as file:
            ast.parse(file.read())
    except SyntaxError as e:
        errors.append(f'{f}:{e.lineno}')
print(f'Remaining errors: {len(errors)}')
"
```

### ML-Specific Validation:
```bash
# Validate ML imports can be resolved
python3 -c "
from src.prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_sklearn_cluster
print('✅ Core ML lazy loaders functional')
try:
    np = get_numpy()
    sklearn = get_sklearn_cluster()
    print('✅ ML dependencies loadable')
except Exception as e:
    print(f'❌ ML dependency issue: {e}')
"
```

## 📊 Progress Tracking

### Phase 3.2 Success Criteria:
- [ ] All 23 ML lazy loading import errors resolved
- [ ] Core ML type definitions (types.py) functional
- [ ] No remaining `from get_*().module import` patterns
- [ ] ML algorithm imports validate successfully

### Phase 3.3 Success Criteria:
- [ ] All 12 try-except block errors resolved
- [ ] Control flow statements properly structured
- [ ] Exception handling restored in analytics components
- [ ] No remaining "expected except/finally" errors

### Phase 3.4 Success Criteria:  
- [ ] All mixed import pattern errors resolved
- [ ] Import blocks properly consolidated
- [ ] Analytics components initialize successfully
- [ ] Clean import structure across all modules

### Phase 3.5 Success Criteria:
- [ ] All 48 syntax errors resolved
- [ ] Full codebase AST compilation successful
- [ ] Performance validation scripts functional
- [ ] Ready for Phase 4: Functional validation

## ⚠️ Critical Dependencies

### Before Starting P3.2:
1. Verify `lazy_ml_loader` utilities exist and are functional
2. Confirm ML dependency installation status
3. Validate existing lazy loading patterns in working files

### Risk Mitigation:
1. **Backup Strategy**: Git commit before each phase
2. **Incremental Validation**: Test each file immediately after fix
3. **Rollback Plan**: Ability to revert to Phase 2 completion state
4. **Functional Testing**: Smoke test ML algorithms after import fixes

## 🚀 Implementation Timeline

**Total Estimated Time**: 5 hours  
**Phase 3.2**: 2 hours (ML imports - CRITICAL)  
**Phase 3.3**: 1.5 hours (Try-except blocks - HIGH)  
**Phase 3.4**: 1 hour (Import consolidation - MEDIUM)  
**Phase 3.5**: 0.5 hours (Final cleanup - LOW)  

**Next Immediate Action**: Begin P3.2 with `src/prompt_improver/ml/types.py` (core type definitions)

---
**P3.1 Status**: ✅ COMPLETE  
**Current Phase**: Ready for P3.2 execution  
**Critical Path**: ML lazy loading import fixes block all subsequent phases