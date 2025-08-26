#!/usr/bin/env python3
"""Analytics Import Performance Validation Script.

This script validates the >90% improvement target achieved through analytics component
lazy loading optimization by measuring import times and comparing against baseline measurements.

BASELINE MEASUREMENTS (from Task 1):
- SessionAnalyticsComponent: 897.19ms
- MLAnalyticsComponent: 436.47ms
- ABTestingComponent: 460.25ms
- PerformanceAnalyticsComponent: 444.26ms
- Total Combined Penalty: 2,238ms

TARGET VALIDATION: >90% improvement (2,238ms → <224ms)

Expected Results:
- Individual components: >90% import time reduction each
- Combined performance: 2,238ms → <224ms or better
- Real-world application startup improvement
- Memory efficiency gains
"""

import json
import statistics
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ImportMeasurement:
    """Individual import measurement result."""
    component_name: str
    import_statement: str
    import_time_ms: float
    success: bool
    error: str = ""
    measurement_method: str = "subprocess"


@dataclass
class LazyLoadingMeasurement:
    """Lazy loading behavior measurement."""
    component_name: str
    first_function_call_ms: float
    subsequent_calls_ms: list[float]
    cache_effective: bool
    numpy_loaded: bool
    scipy_loaded: bool


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    timestamp: str
    total_improvement_percent: float
    individual_improvements: dict[str, float]
    baseline_measurements: dict[str, float]
    optimized_measurements: dict[str, float]
    lazy_loading_validation: dict[str, Any]
    performance_targets_met: bool
    memory_analysis: dict[str, Any]
    recommendations: list[str]


class AnalyticsImportValidator:
    """Comprehensive analytics import performance validator."""

    def __init__(self) -> None:
        self.repo_root = Path(__file__).parent.parent
        self.src_path = self.repo_root / "src"

        # Baseline measurements from Task 1
        self.baseline_measurements = {
            "SessionAnalyticsComponent": 897.19,
            "MLAnalyticsComponent": 436.47,
            "ABTestingComponent": 460.25,
            "PerformanceAnalyticsComponent": 444.26,
        }

        self.total_baseline = sum(self.baseline_measurements.values())
        self.target_improvement = 0.90  # 90%
        self.target_total_time = self.total_baseline * (1 - self.target_improvement)

        # Components to test
        self.components = [
            {
                "name": "SessionAnalyticsComponent",
                "import_statement": "from prompt_improver.analytics.unified.session_analytics_component import SessionAnalyticsComponent"
            },
            {
                "name": "MLAnalyticsComponent",
                "import_statement": "from prompt_improver.analytics.unified.ml_analytics_component import MLAnalyticsComponent"
            },
            {
                "name": "ABTestingComponent",
                "import_statement": "from prompt_improver.analytics.unified.ab_testing_component import ABTestingComponent"
            },
            {
                "name": "PerformanceAnalyticsComponent",
                "import_statement": "from prompt_improver.analytics.unified.performance_analytics_component import PerformanceAnalyticsComponent"
            }
        ]

    def measure_import_time_subprocess(self, import_statement: str, component_name: str) -> ImportMeasurement:
        """Measure import time using subprocess for true first-time import."""
        try:
            # Create test script
            test_script = f'''
import sys
import time
sys.path.insert(0, r"{self.src_path}")

start_time = time.perf_counter()
{import_statement}
end_time = time.perf_counter()

print(f"IMPORT_TIME_MS:{{(end_time - start_time) * 1000:.6f}}")
'''

            # Execute in clean subprocess
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                check=False, capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Extract import time
                for line in result.stdout.split('\n'):
                    if line.startswith("IMPORT_TIME_MS:"):
                        import_time_ms = float(line.split(':')[1])
                        return ImportMeasurement(
                            component_name=component_name,
                            import_statement=import_statement,
                            import_time_ms=import_time_ms,
                            success=True,
                            measurement_method="subprocess"
                        )

                return ImportMeasurement(
                    component_name=component_name,
                    import_statement=import_statement,
                    import_time_ms=float('inf'),
                    success=False,
                    error="Could not extract import time from output",
                    measurement_method="subprocess"
                )
            return ImportMeasurement(
                component_name=component_name,
                import_statement=import_statement,
                import_time_ms=float('inf'),
                success=False,
                error=f"Subprocess failed: {result.stderr}",
                measurement_method="subprocess"
            )

        except Exception as e:
            return ImportMeasurement(
                component_name=component_name,
                import_statement=import_statement,
                import_time_ms=float('inf'),
                success=False,
                error=str(e),
                measurement_method="subprocess"
            )

    def measure_lazy_loading_behavior(self, component_name: str) -> LazyLoadingMeasurement:
        """Measure lazy loading behavior with function-level numpy/scipy imports."""
        try:
            # Create script that tests lazy loading
            test_script = f'''
import sys
import time
sys.path.insert(0, r"{self.src_path}")

# Import the component (should be fast)
{self.components[0]["import_statement"] if component_name == "SessionAnalyticsComponent" else ""}
{self.components[1]["import_statement"] if component_name == "MLAnalyticsComponent" else ""}
{self.components[2]["import_statement"] if component_name == "ABTestingComponent" else ""}
{self.components[3]["import_statement"] if component_name == "PerformanceAnalyticsComponent" else ""}

# Test lazy loading by calling _get_numpy() function
if "{component_name}" == "SessionAnalyticsComponent":
    from prompt_improver.analytics.unified.session_analytics_component import _get_numpy, _get_scipy_stats
elif "{component_name}" == "MLAnalyticsComponent":
    from prompt_improver.analytics.unified.ml_analytics_component import _get_numpy, _get_scipy_stats
elif "{component_name}" == "ABTestingComponent":
    from prompt_improver.analytics.unified.ab_testing_component import _get_numpy, _get_scipy_stats
elif "{component_name}" == "PerformanceAnalyticsComponent":
    from prompt_improver.analytics.unified.performance_analytics_component import _get_numpy, _get_scipy_stats

# First call to _get_numpy() (should trigger actual import)
start_time = time.perf_counter()
numpy = _get_numpy()
first_call_time = time.perf_counter() - start_time

# Subsequent calls (should be cached/fast)
subsequent_times = []
for i in range(5):
    start_time = time.perf_counter()
    numpy = _get_numpy()
    subsequent_times.append((time.perf_counter() - start_time) * 1000)

print(f"FIRST_CALL_MS:{{first_call_time * 1000:.6f}}")
print(f"SUBSEQUENT_CALLS_MS:{{','.join([str(t) for t in subsequent_times])}}")

# Check if numpy and scipy are loaded
import sys
print(f"NUMPY_LOADED:{{\\\"numpy\\\" in sys.modules}}")
print(f"SCIPY_LOADED:{{\\\"scipy\\\" in sys.modules}}")
'''

            result = subprocess.run(
                [sys.executable, "-c", test_script],
                check=False, capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                first_call_ms = 0.0
                subsequent_calls = []
                numpy_loaded = False
                scipy_loaded = False

                for line in result.stdout.split('\n'):
                    if line.startswith("FIRST_CALL_MS:"):
                        first_call_ms = float(line.split(':')[1])
                    elif line.startswith("SUBSEQUENT_CALLS_MS:"):
                        calls_str = line.split(':')[1]
                        subsequent_calls = [float(t) for t in calls_str.split(',') if t]
                    elif line.startswith("NUMPY_LOADED:"):
                        numpy_loaded = "True" in line
                    elif line.startswith("SCIPY_LOADED:"):
                        scipy_loaded = "True" in line

                cache_effective = len(subsequent_calls) > 0 and all(t < 1.0 for t in subsequent_calls)

                return LazyLoadingMeasurement(
                    component_name=component_name,
                    first_function_call_ms=first_call_ms,
                    subsequent_calls_ms=subsequent_calls,
                    cache_effective=cache_effective,
                    numpy_loaded=numpy_loaded,
                    scipy_loaded=scipy_loaded
                )
            return LazyLoadingMeasurement(
                component_name=component_name,
                first_function_call_ms=float('inf'),
                subsequent_calls_ms=[],
                cache_effective=False,
                numpy_loaded=False,
                scipy_loaded=False
            )

        except Exception as e:
            print(f"Error measuring lazy loading for {component_name}: {e}")
            return LazyLoadingMeasurement(
                component_name=component_name,
                first_function_call_ms=float('inf'),
                subsequent_calls_ms=[],
                cache_effective=False,
                numpy_loaded=False,
                scipy_loaded=False
            )

    def measure_memory_usage(self) -> dict[str, Any]:
        """Measure memory usage before and after imports."""
        try:
            # Memory before imports
            before_script = '''
import psutil
import os
process = psutil.Process(os.getpid())
print(f"MEMORY_BEFORE_MB:{process.memory_info().rss / 1024 / 1024:.2f}")
'''

            result_before = subprocess.run(
                [sys.executable, "-c", before_script],
                check=False, capture_output=True,
                text=True
            )

            # Memory after imports
            after_script = f'''
import psutil
import os
import sys
sys.path.insert(0, r"{self.src_path}")

# Import all analytics components
from prompt_improver.analytics.unified.session_analytics_component import SessionAnalyticsComponent
from prompt_improver.analytics.unified.ml_analytics_component import MLAnalyticsComponent
from prompt_improver.analytics.unified.ab_testing_component import ABTestingComponent
from prompt_improver.analytics.unified.performance_analytics_component import PerformanceAnalyticsComponent

process = psutil.Process(os.getpid())
print(f"MEMORY_AFTER_MB:{{process.memory_info().rss / 1024 / 1024:.2f}}")
'''

            result_after = subprocess.run(
                [sys.executable, "-c", after_script],
                check=False, capture_output=True,
                text=True
            )

            memory_before = 0.0
            memory_after = 0.0

            if result_before.returncode == 0:
                for line in result_before.stdout.split('\n'):
                    if line.startswith("MEMORY_BEFORE_MB:"):
                        memory_before = float(line.split(':')[1])

            if result_after.returncode == 0:
                for line in result_after.stdout.split('\n'):
                    if line.startswith("MEMORY_AFTER_MB:"):
                        memory_after = float(line.split(':')[1])

            return {
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_difference_mb": memory_after - memory_before,
                "measurement_success": result_before.returncode == 0 and result_after.returncode == 0
            }

        except Exception as e:
            return {
                "memory_before_mb": 0.0,
                "memory_after_mb": 0.0,
                "memory_difference_mb": 0.0,
                "measurement_success": False,
                "error": str(e)
            }

    def run_comprehensive_validation(self) -> ValidationReport:
        """Run comprehensive validation of analytics import optimization."""
        print("🚀 Analytics Import Performance Validation")
        print("=" * 60)
        print(f"Baseline total: {self.total_baseline:.2f}ms")
        print(f"Target improvement: {self.target_improvement:.1%}")
        print(f"Target total time: <{self.target_total_time:.2f}ms")
        print()

        # Phase 1: Measure optimized import times
        print("📊 Phase 1: Measuring Optimized Import Performance")
        print("-" * 50)

        optimized_measurements = {}
        import_results = []

        for component in self.components:
            print(f"Testing {component['name']}...")

            # Multiple measurements for statistical accuracy
            measurements = []
            for _i in range(5):
                result = self.measure_import_time_subprocess(
                    component['import_statement'],
                    component['name']
                )
                if result.success:
                    measurements.append(result.import_time_ms)
                import_results.append(result)

            if measurements:
                avg_time = statistics.mean(measurements)
                optimized_measurements[component['name']] = avg_time
                baseline = self.baseline_measurements[component['name']]
                improvement = ((baseline - avg_time) / baseline) * 100

                print(f"  ✓ {component['name']}: {avg_time:.2f}ms "
                      f"(baseline: {baseline:.2f}ms, improvement: {improvement:.1f}%)")
            else:
                optimized_measurements[component['name']] = float('inf')
                print(f"  ✗ {component['name']}: FAILED")

        # Phase 2: Lazy loading validation
        print("\n📋 Phase 2: Lazy Loading Behavior Validation")
        print("-" * 50)

        lazy_loading_results = {}
        for component in self.components:
            print(f"Testing lazy loading for {component['name']}...")

            result = self.measure_lazy_loading_behavior(component['name'])
            lazy_loading_results[component['name']] = asdict(result)

            if result.first_function_call_ms != float('inf'):
                print(f"  First numpy call: {result.first_function_call_ms:.3f}ms")
                if result.subsequent_calls_ms:
                    avg_subsequent = statistics.mean(result.subsequent_calls_ms)
                    print(f"  Subsequent calls: {avg_subsequent:.3f}ms (cached: {result.cache_effective})")
                print(f"  NumPy loaded: {result.numpy_loaded}, SciPy loaded: {result.scipy_loaded}")
            else:
                print("  ✗ Failed to measure lazy loading")

        # Phase 3: Memory analysis
        print("\n💾 Phase 3: Memory Usage Analysis")
        print("-" * 50)

        memory_analysis = self.measure_memory_usage()
        if memory_analysis["measurement_success"]:
            print(f"Memory before imports: {memory_analysis['memory_before_mb']:.2f}MB")
            print(f"Memory after imports: {memory_analysis['memory_after_mb']:.2f}MB")
            print(f"Memory increase: {memory_analysis['memory_difference_mb']:.2f}MB")
        else:
            print("✗ Memory measurement failed")

        # Phase 4: Calculate improvements and validate targets
        print("\n🎯 Phase 4: Performance Target Validation")
        print("-" * 50)

        individual_improvements = {}
        total_optimized = 0

        for component_name in self.baseline_measurements:
            baseline = self.baseline_measurements[component_name]
            optimized = optimized_measurements.get(component_name, float('inf'))

            if optimized != float('inf'):
                improvement_percent = ((baseline - optimized) / baseline) * 100
                individual_improvements[component_name] = improvement_percent
                total_optimized += optimized

                target_met = improvement_percent >= self.target_improvement * 100
                status = "✅" if target_met else "❌"
                print(f"{status} {component_name}: {improvement_percent:.1f}% improvement "
                      f"({baseline:.2f}ms → {optimized:.2f}ms)")
            else:
                individual_improvements[component_name] = 0.0
                total_optimized = float('inf')
                print(f"❌ {component_name}: MEASUREMENT FAILED")

        # Overall performance validation
        print("\n📈 Overall Performance Results:")
        print("-" * 50)

        if total_optimized != float('inf'):
            total_improvement_percent = ((self.total_baseline - total_optimized) / self.total_baseline) * 100
            targets_met = total_improvement_percent >= self.target_improvement * 100

            print(f"Combined baseline: {self.total_baseline:.2f}ms")
            print(f"Combined optimized: {total_optimized:.2f}ms")
            print(f"Total improvement: {total_improvement_percent:.1f}%")
            print(f"Target ({self.target_improvement:.1%}): {'✅ MET' if targets_met else '❌ NOT MET'}")
        else:
            total_improvement_percent = 0.0
            targets_met = False
            print("❌ Overall measurement failed")

        # Generate recommendations
        recommendations = self._generate_recommendations(
            individual_improvements,
            lazy_loading_results,
            memory_analysis
        )

        # Create validation report
        return ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_improvement_percent=total_improvement_percent,
            individual_improvements=individual_improvements,
            baseline_measurements=self.baseline_measurements,
            optimized_measurements=optimized_measurements,
            lazy_loading_validation=lazy_loading_results,
            performance_targets_met=targets_met,
            memory_analysis=memory_analysis,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        improvements: dict[str, float],
        lazy_loading_results: dict[str, Any],
        memory_analysis: dict[str, Any]
    ) -> list[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []

        # Import performance recommendations
        failed_components = [name for name, improvement in improvements.items() if improvement < 90.0]
        if failed_components:
            recommendations.append(
                f"Components not meeting 90% target: {', '.join(failed_components)} - "
                "review lazy loading implementation"
            )

        # Lazy loading recommendations
        ineffective_caching = [
            name for name, data in lazy_loading_results.items()
            if not data.get("cache_effective", False)
        ]
        if ineffective_caching:
            recommendations.append(
                f"Ineffective caching detected in: {', '.join(ineffective_caching)} - "
                "verify function-level import caching"
            )

        # Memory recommendations
        if memory_analysis.get("memory_difference_mb", 0) > 100:
            recommendations.append(
                "High memory usage detected - consider further optimization of large dependencies"
            )

        # Success recommendations
        if not failed_components and not ineffective_caching:
            recommendations.append(
                "Lazy loading optimization successfully implemented - "
                "maintain current architecture patterns"
            )
            recommendations.append(
                "Consider applying similar optimization to other heavy-dependency components"
            )

        return recommendations

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to file."""
        try:
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"\n📄 Detailed report saved to: {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")

    def print_final_summary(self, report: ValidationReport) -> None:
        """Print final validation summary."""
        print("\n" + "=" * 60)
        print("🏆 ANALYTICS OPTIMIZATION VALIDATION SUMMARY")
        print("=" * 60)

        if report.performance_targets_met:
            print("✅ SUCCESS: Analytics optimization achieved >90% improvement target")
            print(f"📊 Total improvement: {report.total_improvement_percent:.1f}%")
            print(f"⚡ Combined time: {sum(report.optimized_measurements.values()):.2f}ms "
                  f"(from {sum(report.baseline_measurements.values()):.2f}ms)")
        else:
            print("❌ PARTIAL SUCCESS: Some targets not met")
            print(f"📊 Total improvement: {report.total_improvement_percent:.1f}%")

        print("\n🔍 Individual Component Results:")
        for component, improvement in report.individual_improvements.items():
            status = "✅" if improvement >= 90.0 else "❌"
            print(f"  {status} {component}: {improvement:.1f}% improvement")

        print(f"\n💡 Key Recommendations ({len(report.recommendations)}):")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 60)


def main():
    """Main validation execution."""
    try:
        validator = AnalyticsImportValidator()

        # Run comprehensive validation
        report = validator.run_comprehensive_validation()

        # Save detailed report
        output_path = Path("analytics_import_performance_validation_report.json")
        validator.save_report(report, output_path)

        # Print final summary
        validator.print_final_summary(report)

        # Return appropriate exit code
        return 0 if report.performance_targets_met else 1

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Analytics validation failed: {e}")
        print(traceback.format_exc())
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
