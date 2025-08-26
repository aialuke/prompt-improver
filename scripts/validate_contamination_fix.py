#!/usr/bin/env python3
"""Validate NumPy contamination fix effectiveness.

This script measures startup performance improvements and memory reduction
after implementing lazy loading for ML dependencies.
"""

import subprocess
from pathlib import Path


def measure_import_time(import_statement: str, description: str) -> tuple[float, int]:
    """Measure import time and loaded ML modules."""
    script = f'''
import sys
import time

start = time.time()
{import_statement}
elapsed = (time.time() - start) * 1000

# Count ML modules
ml_modules = [name for name in sys.modules
              if any(lib in name.lower() for lib in ['numpy', 'scipy', 'sklearn', 'torch', 'beartype'])]

print(f"{{elapsed:.1f}},{{len(ml_modules)}}")
'''

    result = subprocess.run(['python3', '-c', script],
                          check=False, capture_output=True, text=True, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"Error measuring {description}: {result.stderr}")
        return 0.0, 0

    try:
        elapsed_ms, ml_count = result.stdout.strip().split(',')
        return float(elapsed_ms), int(ml_count)
    except:
        print(f"Failed to parse results for {description}")
        return 0.0, 0


def main():
    """Run contamination fix validation."""
    print("=" * 70)
    print("NUMPY CONTAMINATION FIX VALIDATION")
    print("=" * 70)

    results = {}

    # Add src to path for tests
    src_path = str(Path(__file__).parent.parent / "src")

    print("📊 Measuring import performance improvements...")
    print()

    # Test 1: Standalone lazy loader (our goal)
    elapsed, ml_count = measure_import_time(
        f"sys.path.append('{src_path}'); from standalone_lazy_ml_loader import get_numpy; np = get_numpy(); arr = np.array([1,2,3])",
        "Standalone lazy loader"
    )
    results['standalone'] = (elapsed, ml_count)
    print(f"✅ Standalone lazy loader:     {elapsed:6.1f}ms   ({ml_count:2d} ML modules)")

    # Test 2: Direct NumPy import (baseline comparison)
    elapsed, ml_count = measure_import_time(
        "import numpy as np; arr = np.array([1,2,3])",
        "Direct NumPy import"
    )
    results['direct_numpy'] = (elapsed, ml_count)
    print(f"📍 Direct NumPy import:        {elapsed:6.1f}ms   ({ml_count:2d} ML modules)")

    # Test 3: Package lazy loader (current state)
    elapsed, ml_count = measure_import_time(
        f"sys.path.append('{src_path}'); from prompt_improver.core.utils.lazy_ml_loader import get_numpy; np = get_numpy()",
        "Package lazy loader"
    )
    results['package_lazy'] = (elapsed, ml_count)
    print(f"⚠️  Package lazy loader:       {elapsed:6.1f}ms   ({ml_count:2d} ML modules)")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    standalone_time, standalone_ml = results['standalone']
    _direct_time, direct_ml = results['direct_numpy']
    package_time, package_ml = results['package_lazy']

    # Performance analysis
    print("🎯 PERFORMANCE IMPROVEMENT:")
    if package_time > 0:
        improvement = ((package_time - standalone_time) / package_time) * 100
        print(f"   Package import speed: {improvement:5.1f}% faster with standalone loader")

    print(f"   Startup overhead eliminated: {package_time - standalone_time:6.1f}ms")

    # Memory analysis
    print("\\n💾 MEMORY/MODULE REDUCTION:")
    print(f"   ML modules loaded (package): {package_ml}")
    print(f"   ML modules loaded (standalone): {standalone_ml}")
    print(f"   Module reduction: {package_ml - standalone_ml} fewer modules")

    # Contamination assessment
    print("\\n🚨 CONTAMINATION ASSESSMENT:")
    if standalone_ml <= direct_ml + 5:  # Allow small variance
        print("   ✅ Contamination eliminated - minimal ML modules loaded")
    else:
        print("   ⚠️  Contamination partially reduced but still present")

    if standalone_time < 100:
        print("   ✅ Startup performance acceptable (<100ms)")
    else:
        print("   ⚠️  Startup performance still needs improvement")

    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if package_time > standalone_time * 2:
        print("🔧 RECOMMENDED ACTIONS:")
        print("   1. Use standalone lazy loader in performance-critical paths")
        print("   2. Consider restructuring package imports to avoid deep chains")
        print("   3. Move coredis imports to lazy loading pattern")
        print("   4. Consider replacing coredis with redis-py (no beartype dependency)")
    else:
        print("✅ Package performance is acceptable")

    # Success criteria
    print()
    success_count = 0
    total_criteria = 3

    if standalone_time < 100:
        print("✅ Startup time <100ms: PASS")
        success_count += 1
    else:
        print("❌ Startup time <100ms: FAIL")

    if standalone_ml < package_ml:
        print("✅ Module reduction: PASS")
        success_count += 1
    else:
        print("❌ Module reduction: FAIL")

    if standalone_time < package_time * 0.5:
        print("✅ Performance improvement >50%: PASS")
        success_count += 1
    else:
        print("❌ Performance improvement >50%: FAIL")

    print()
    print(f"OVERALL: {success_count}/{total_criteria} success criteria met")

    if success_count == total_criteria:
        print("🎉 CONTAMINATION ELIMINATION: SUCCESS!")
    elif success_count >= 2:
        print("👍 CONTAMINATION ELIMINATION: PARTIAL SUCCESS")
    else:
        print("❌ CONTAMINATION ELIMINATION: NEEDS MORE WORK")


if __name__ == "__main__":
    main()
