#!/usr/bin/env python3
"""Diagnostic script to identify import contamination sources.

This script systematically tests different import paths to identify
where NumPy and other ML dependencies are being loaded during package import.
"""

import importlib.util
import sys
import time
from pathlib import Path


def time_import(description: str, import_func) -> float:
    """Time an import operation and print results."""
    print(f"Testing: {description}")
    start = time.time()
    try:
        result = import_func()
        elapsed = (time.time() - start) * 1000
        print(f"  ✓ Success: {elapsed:.1f}ms")
        return elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        print(f"  ✗ Failed: {elapsed:.1f}ms - {e}")
        return elapsed


def test_imports():
    """Test various import scenarios to identify contamination."""
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    print("=" * 60)
    print("IMPORT CONTAMINATION DIAGNOSIS")
    print("=" * 60)

    results = {}

    # Test 1: Direct file import (baseline)
    def direct_import():
        spec = importlib.util.spec_from_file_location(
            'lazy_ml_loader',
            str(src_path / 'prompt_improver/core/utils/lazy_ml_loader.py')
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    results['direct_file'] = time_import("Direct file import (baseline)", direct_import)

    # Test 2: Package import chain
    def package_import():
        from prompt_improver.core.utils import lazy_ml_loader
        return lazy_ml_loader

    results['package_import'] = time_import("Package import chain", package_import)

    # Test 3: Core module import
    def core_import():
        import prompt_improver.core
        return prompt_improver.core

    results['core_import'] = time_import("Core module import", core_import)

    # Test 4: Types import (converted file)
    def types_import():
        from prompt_improver.core import types
        return types

    results['types_import'] = time_import("Core types import (converted)", types_import)

    # Test 5: Database import
    def db_import():
        from prompt_improver import database
        return database

    results['db_import'] = time_import("Database module import", db_import)

    # Test 6: Services import
    def services_import():
        from prompt_improver import services
        return services

    results['services_import'] = time_import("Services module import", services_import)

    # Test 7: ML module import
    def ml_import():
        from prompt_improver import ml
        return ml

    results['ml_import'] = time_import("ML module import", ml_import)

    # Test 8: Performance module import
    def perf_import():
        from prompt_improver import performance
        return performance

    results['perf_import'] = time_import("Performance module import", perf_import)

    # Analysis
    print("\n" + "=" * 60)
    print("CONTAMINATION ANALYSIS")
    print("=" * 60)

    baseline = results.get('direct_file', 0)
    print(f"Baseline (direct file import): {baseline:.1f}ms")

    for test_name, duration in results.items():
        if test_name == 'direct_file':
            continue
        overhead = duration - baseline
        if overhead > 100:  # Significant overhead
            print(f"⚠️  {test_name}: +{overhead:.1f}ms overhead (CONTAMINATION DETECTED)")
        elif overhead > 10:
            print(f"⚠️  {test_name}: +{overhead:.1f}ms overhead (minor contamination)")
        else:
            print(f"✅ {test_name}: +{overhead:.1f}ms overhead (clean)")

    # Module loading check
    print("\n" + "=" * 60)
    print("LOADED MODULES ANALYSIS")
    print("=" * 60)

    ml_modules = [name for name in sys.modules if any(lib in name.lower() for lib in ['numpy', 'scipy', 'sklearn', 'torch', 'pandas'])]

    if ml_modules:
        print("🚨 ML modules loaded during import:")
        for mod in sorted(ml_modules)[:10]:  # Show first 10
            print(f"  - {mod}")
        if len(ml_modules) > 10:
            print(f"  ... and {len(ml_modules) - 10} more")
    else:
        print("✅ No ML modules detected in sys.modules")

    return results


if __name__ == "__main__":
    test_imports()
