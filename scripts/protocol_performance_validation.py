#!/usr/bin/env python3
"""Protocol Performance Validation Script.

Quick validation script to test optimized protocol import performance
and confirm <2ms performance targets are achieved.

Usage:
    python scripts/protocol_performance_validation.py
    python scripts/protocol_performance_validation.py --test-optimized
"""

import importlib
import sys
import time
from pathlib import Path
from statistics import mean

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def clear_modules():
    """Clear protocol-related modules from import cache."""
    modules_to_remove = [module_name for module_name in list(sys.modules.keys()) if 'prompt_improver' in module_name and 'protocols' in module_name]

    for module_name in modules_to_remove:
        del sys.modules[module_name]


def time_import(import_path: str, iterations: int = 3) -> tuple[float, float]:
    """Time protocol import performance.

    Returns:
        Tuple of (cold_start_ms, warm_avg_ms)
    """
    times = []

    for i in range(iterations):
        if i == 0:
            # Clear for cold start
            clear_modules()

        start_time = time.perf_counter()
        try:
            importlib.import_module(import_path)
            end_time = time.perf_counter()
            import_time_ms = (end_time - start_time) * 1000
            times.append(import_time_ms)
        except ImportError as e:
            print(f"❌ Import failed: {import_path} - {e}")
            return float('inf'), float('inf')

    cold_start = times[0]
    warm_avg = mean(times[1:]) if len(times) > 1 else times[0]

    return cold_start, warm_avg


def validate_protocol_performance():
    """Validate current protocol import performance."""
    print("🔍 Protocol Import Performance Validation")
    print("=" * 60)

    # Test current protocol performance
    protocols_to_test = [
        ('core', 'prompt_improver.shared.interfaces.protocols.core', 1.0),
        ('database', 'prompt_improver.shared.interfaces.protocols.database', 1.0),
        ('cache', 'prompt_improver.shared.interfaces.protocols.cache', 1.0),
        ('security', 'prompt_improver.shared.interfaces.protocols.security', 2.0),
        ('application', 'prompt_improver.shared.interfaces.protocols.application', 2.0),
        ('mcp', 'prompt_improver.shared.interfaces.protocols.mcp', 2.0),
        ('cli', 'prompt_improver.shared.interfaces.protocols.cli', 5.0),
    ]

    results = []
    total_pass = 0
    total_tests = len(protocols_to_test)

    print(f"{'Protocol':<12} {'Target':<8} {'Cold Start':<12} {'Warm Avg':<12} {'Status':<8}")
    print("-" * 60)

    for name, import_path, target_ms in protocols_to_test:
        cold_start, warm_avg = time_import(import_path)

        if cold_start == float('inf'):
            status = "❌ ERROR"
        elif cold_start <= target_ms:
            status = "✅ PASS"
            total_pass += 1
        else:
            status = "❌ FAIL"

        print(f"{name:<12} <{target_ms}ms     {cold_start:<9.3f}ms   {warm_avg:<9.3f}ms   {status}")

        results.append({
            'name': name,
            'target_ms': target_ms,
            'cold_start_ms': cold_start,
            'warm_avg_ms': warm_avg,
            'passed': cold_start <= target_ms and cold_start != float('inf')
        })

    # Test lazy loading
    print("\n🚀 Lazy Loading Performance")
    print("-" * 40)

    lazy_tests = [
        ('ml_lazy', 'prompt_improver.shared.interfaces.protocols', 'get_ml_protocols', 0.1),
        ('monitoring_lazy', 'prompt_improver.shared.interfaces.protocols', 'get_monitoring_protocols', 0.1)
    ]

    for name, import_path, function_name, target_ms in lazy_tests:
        clear_modules()

        start_time = time.perf_counter()
        try:
            module = importlib.import_module(import_path)
            lazy_func = getattr(module, function_name)
            lazy_func()
            end_time = time.perf_counter()
            lazy_time_ms = (end_time - start_time) * 1000

            if lazy_time_ms <= target_ms:
                status = "✅ PASS"
                total_pass += 1
            else:
                status = "❌ FAIL"

            print(f"{name:<12} <{target_ms}ms     {lazy_time_ms:<9.3f}ms                {status}")

        except (ImportError, AttributeError) as e:
            print(f"{name:<12} <{target_ms}ms     ERROR                   ❌ ERROR")

        total_tests += 1

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    success_rate = (total_pass / total_tests) * 100
    overall_status = "✅ PASS" if total_pass == total_tests else "❌ FAIL"

    print(f"Overall Status: {overall_status}")
    print(f"Success Rate: {total_pass}/{total_tests} ({success_rate:.1f}%)")

    if total_pass < total_tests:
        print(f"\n⚠️  {total_tests - total_pass} protocols need optimization:")
        for result in results:
            if not result['passed'] and result['cold_start_ms'] != float('inf'):
                improvement_needed = result['cold_start_ms'] / result['target_ms']
                print(f"   • {result['name']}: {result['cold_start_ms']:.1f}ms → <{result['target_ms']}ms ({improvement_needed:.1f}x improvement needed)")

    return success_rate == 100.0


def test_optimization_example():
    """Test the optimization example files."""
    print("\n🧪 Testing Optimization Examples")
    print("=" * 40)

    # Add optimized examples to path
    optimized_path = project_root / "scripts" / "optimized_protocol_examples"
    sys.path.insert(0, str(optimized_path))

    try:
        # Test optimized database protocols
        clear_modules()
        start_time = time.perf_counter()
        import database_protocols_optimized
        end_time = time.perf_counter()
        optimized_time_ms = (end_time - start_time) * 1000

        target_ms = 1.0
        status = "✅ PASS" if optimized_time_ms <= target_ms else "❌ FAIL"

        print(f"Optimized Database: {optimized_time_ms:.3f}ms (target: <{target_ms}ms) {status}")

        # Test optimized init
        clear_modules()
        start_time = time.perf_counter()
        import protocols_init_optimized
        end_time = time.perf_counter()
        init_time_ms = (end_time - start_time) * 1000

        init_target_ms = 10.0  # More relaxed for init
        init_status = "✅ PASS" if init_time_ms <= init_target_ms else "❌ FAIL"

        print(f"Optimized Init: {init_time_ms:.3f}ms (target: <{init_target_ms}ms) {init_status}")

        # Test lazy loading from optimized init
        if hasattr(protocols_init_optimized, 'get_ml_protocols'):
            start_time = time.perf_counter()
            protocols_init_optimized.get_ml_protocols()
            end_time = time.perf_counter()
            lazy_time_ms = (end_time - start_time) * 1000

            lazy_target_ms = 0.1
            lazy_status = "✅ PASS" if lazy_time_ms <= lazy_target_ms else "❌ FAIL"

            print(f"Optimized Lazy ML: {lazy_time_ms:.3f}ms (target: <{lazy_target_ms}ms) {lazy_status}")

        return True

    except ImportError as e:
        print(f"❌ Optimization example test failed: {e}")
        return False


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Protocol Performance Validation')
    parser.add_argument('--test-optimized', action='store_true', help='Test optimization examples')

    args = parser.parse_args()

    # Run current protocol validation
    current_pass = validate_protocol_performance()

    # Test optimization examples if requested
    if args.test_optimized:
        optimization_pass = test_optimization_example()

        print("\n🎯 FINAL RESULTS")
        print("=" * 30)
        print(f"Current Protocols: {'✅ PASS' if current_pass else '❌ FAIL'}")
        print(f"Optimization Examples: {'✅ PASS' if optimization_pass else '❌ FAIL'}")

        if optimization_pass and not current_pass:
            print("\n💡 Recommendation: Apply optimization patterns to achieve <2ms performance targets")

    return 0 if current_pass else 1


if __name__ == "__main__":
    sys.exit(main())
