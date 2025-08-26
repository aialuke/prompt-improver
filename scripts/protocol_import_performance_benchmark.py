#!/usr/bin/env python3
"""Protocol Import Performance Benchmark.

Comprehensive performance benchmarking for protocol import performance to ensure
compliance with <2ms requirement for critical system operations.

Performance Requirements:
- Core/Database/Cache: <1ms each (critical path)
- Security/Application/MCP: <2ms each (moderate priority)
- CLI: <5ms (lower priority, used in CLI contexts)
- ML/Monitoring (lazy): <0.1ms (must not load heavy dependencies)
- Total consolidated protocols: <10ms for all domains combined
- Memory overhead: <50MB additional memory usage

Usage:
    python scripts/protocol_import_performance_benchmark.py
    python scripts/protocol_import_performance_benchmark.py --verbose
    python scripts/protocol_import_performance_benchmark.py --concurrent-tests 10
"""

import gc
import importlib
import json
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import psutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class PerformanceCollector:
    """Collects detailed performance metrics during protocol imports."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = 0
        self.memory_samples = []

    def start_monitoring(self):
        """Start memory monitoring."""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.memory_samples = [self.initial_memory]

    def sample_memory(self):
        """Sample current memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_samples.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory

    def get_memory_stats(self) -> dict[str, float]:
        """Get memory usage statistics."""
        if not self.memory_samples:
            return {}

        return {
            "initial_mb": self.initial_memory,
            "peak_mb": self.peak_memory,
            "final_mb": self.memory_samples[-1],
            "delta_mb": self.memory_samples[-1] - self.initial_memory,
            "peak_delta_mb": self.peak_memory - self.initial_memory,
            "avg_mb": mean(self.memory_samples),
            "samples": len(self.memory_samples)
        }


class ProtocolImportBenchmark:
    """Main benchmark class for protocol import performance testing."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.results = {}
        self.performance_collector = PerformanceCollector()

        # Protocol domains to benchmark
        self.protocol_domains = {
            # Critical path protocols (<1ms requirement)
            'core': {
                'import_path': 'prompt_improver.shared.interfaces.protocols.core',
                'priority': 'critical',
                'target_ms': 1.0,
                'description': 'Core service and health protocols'
            },
            'database': {
                'import_path': 'prompt_improver.shared.interfaces.protocols.database',
                'priority': 'critical',
                'target_ms': 1.0,
                'description': 'Database session and connection protocols'
            },
            'cache': {
                'import_path': 'prompt_improver.shared.interfaces.protocols.cache',
                'priority': 'critical',
                'target_ms': 1.0,
                'description': 'Multi-level cache protocols'
            },

            # Moderate priority protocols (<2ms requirement)
            'security': {
                'import_path': 'prompt_improver.shared.interfaces.protocols.security',
                'priority': 'moderate',
                'target_ms': 2.0,
                'description': 'Authentication and authorization protocols'
            },
            'application': {
                'import_path': 'prompt_improver.shared.interfaces.protocols.application',
                'priority': 'moderate',
                'target_ms': 2.0,
                'description': 'Application service protocols'
            },
            'mcp': {
                'import_path': 'prompt_improver.shared.interfaces.protocols.mcp',
                'priority': 'moderate',
                'target_ms': 2.0,
                'description': 'MCP server and tool protocols'
            },

            # Lower priority protocols (<5ms requirement)
            'cli': {
                'import_path': 'prompt_improver.shared.interfaces.protocols.cli',
                'priority': 'low',
                'target_ms': 5.0,
                'description': 'CLI command and workflow protocols'
            },
        }

        # Lazy loading protocols (<0.1ms requirement)
        self.lazy_protocols = {
            'ml_lazy': {
                'import_path': 'prompt_improver.shared.interfaces.protocols',
                'lazy_function': 'get_ml_protocols',
                'target_ms': 0.1,
                'description': 'ML protocols with lazy loading'
            },
            'monitoring_lazy': {
                'import_path': 'prompt_improver.shared.interfaces.protocols',
                'lazy_function': 'get_monitoring_protocols',
                'target_ms': 0.1,
                'description': 'Monitoring protocols with lazy loading'
            }
        }

    @contextmanager
    def memory_monitoring(self):
        """Context manager for memory monitoring during benchmarks."""
        self.performance_collector.start_monitoring()
        try:
            yield self.performance_collector
        finally:
            self.performance_collector.sample_memory()

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}")

    def clear_import_cache(self):
        """Clear Python import cache to ensure cold starts."""
        # Remove from sys.modules
        modules_to_remove = [module_name for module_name in sys.modules if 'prompt_improver' in module_name]

        for module_name in modules_to_remove:
            del sys.modules[module_name]

        # Force garbage collection
        gc.collect()

        self.log(f"Cleared {len(modules_to_remove)} modules from import cache")

    def benchmark_single_import(self, import_path: str, iterations: int = 10) -> dict[str, Any]:
        """Benchmark a single protocol domain import."""
        times = []

        with self.memory_monitoring() as memory_monitor:
            for i in range(iterations):
                # Clear cache for cold start on first iteration
                if i == 0:
                    self.clear_import_cache()

                memory_monitor.sample_memory()

                # Measure import time
                start_time = time.perf_counter()
                try:
                    importlib.import_module(import_path)
                    end_time = time.perf_counter()
                    import_time_ms = (end_time - start_time) * 1000
                    times.append(import_time_ms)

                    self.log(f"  Iteration {i + 1}: {import_time_ms:.3f}ms")

                except ImportError as e:
                    self.log(f"  Import error: {e}")
                    times.append(float('inf'))

                memory_monitor.sample_memory()

                # Small delay to allow system stabilization
                time.sleep(0.001)

        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if not valid_times:
            return {
                'success': False,
                'error': 'All imports failed',
                'iterations': iterations
            }

        result = {
            'success': True,
            'times_ms': valid_times,
            'iterations': len(valid_times),
            'mean_ms': mean(valid_times),
            'median_ms': median(valid_times),
            'min_ms': min(valid_times),
            'max_ms': max(valid_times),
            'cold_start_ms': valid_times[0] if valid_times else None,
            'warm_avg_ms': mean(valid_times[1:]) if len(valid_times) > 1 else valid_times[0],
            'memory_stats': memory_monitor.get_memory_stats()
        }

        if len(valid_times) > 2:
            result['stdev_ms'] = stdev(valid_times)

        return result

    def benchmark_lazy_loading(self, import_path: str, lazy_function: str, iterations: int = 10) -> dict[str, Any]:
        """Benchmark lazy loading protocol imports."""
        times = []

        with self.memory_monitoring() as memory_monitor:
            for i in range(iterations):
                # Clear cache for cold start on first iteration
                if i == 0:
                    self.clear_import_cache()

                memory_monitor.sample_memory()

                # Measure lazy import time
                start_time = time.perf_counter()
                try:
                    module = importlib.import_module(import_path)
                    lazy_func = getattr(module, lazy_function)
                    lazy_func()  # This should be fast due to lazy loading
                    end_time = time.perf_counter()
                    import_time_ms = (end_time - start_time) * 1000
                    times.append(import_time_ms)

                    self.log(f"  Lazy {lazy_function} iteration {i + 1}: {import_time_ms:.3f}ms")

                except (ImportError, AttributeError) as e:
                    self.log(f"  Lazy import error: {e}")
                    times.append(float('inf'))

                memory_monitor.sample_memory()

                # Small delay to allow system stabilization
                time.sleep(0.001)

        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if not valid_times:
            return {
                'success': False,
                'error': 'All lazy imports failed',
                'iterations': iterations
            }

        result = {
            'success': True,
            'times_ms': valid_times,
            'iterations': len(valid_times),
            'mean_ms': mean(valid_times),
            'median_ms': median(valid_times),
            'min_ms': min(valid_times),
            'max_ms': max(valid_times),
            'cold_start_ms': valid_times[0] if valid_times else None,
            'warm_avg_ms': mean(valid_times[1:]) if len(valid_times) > 1 else valid_times[0],
            'memory_stats': memory_monitor.get_memory_stats()
        }

        if len(valid_times) > 2:
            result['stdev_ms'] = stdev(valid_times)

        return result

    def benchmark_concurrent_imports(self, concurrent_count: int = 5) -> dict[str, Any]:
        """Benchmark concurrent protocol imports to test performance under load."""
        self.log(f"Running concurrent import benchmark with {concurrent_count} threads")

        import_paths = [domain['import_path'] for domain in self.protocol_domains.values()]
        results = {}

        def worker_import(import_path: str, worker_id: int) -> tuple[str, float]:
            """Worker function for concurrent imports."""
            start_time = time.perf_counter()
            try:
                importlib.import_module(import_path)
                end_time = time.perf_counter()
                return import_path, (end_time - start_time) * 1000
            except ImportError:
                return import_path, float('inf')

        with self.memory_monitoring() as memory_monitor:
            # Clear cache for cold start
            self.clear_import_cache()

            # Run concurrent imports
            threads = []
            thread_results = {}

            for i in range(concurrent_count):
                import_path = import_paths[i % len(import_paths)]
                thread = threading.Thread(
                    target=lambda: thread_results.update([worker_import(import_path, i)]),
                    name=f"ImportWorker-{i}"
                )
                threads.append(thread)

            # Start all threads
            start_time = time.perf_counter()
            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            total_time = (time.perf_counter() - start_time) * 1000

        # Analyze results
        import_times = [t for t in thread_results.values() if t != float('inf')]
        failed_imports = len([t for t in thread_results.values() if t == float('inf')])

        return {
            'concurrent_threads': concurrent_count,
            'total_time_ms': total_time,
            'successful_imports': len(import_times),
            'failed_imports': failed_imports,
            'mean_import_time_ms': mean(import_times) if import_times else 0,
            'max_import_time_ms': max(import_times) if import_times else 0,
            'min_import_time_ms': min(import_times) if import_times else 0,
            'memory_stats': memory_monitor.get_memory_stats(),
            'results_by_path': thread_results
        }

    def benchmark_critical_path_sequence(self) -> dict[str, Any]:
        """Benchmark critical path protocols in sequence (typical API request)."""
        self.log("Running critical path sequence benchmark")

        critical_protocols = ['core', 'database', 'cache']

        with self.memory_monitoring() as memory_monitor:
            self.clear_import_cache()

            sequence_times = []
            total_start = time.perf_counter()

            for protocol_name in critical_protocols:
                import_path = self.protocol_domains[protocol_name]['import_path']

                start_time = time.perf_counter()
                try:
                    importlib.import_module(import_path)
                    end_time = time.perf_counter()
                    import_time = (end_time - start_time) * 1000
                    sequence_times.append((protocol_name, import_time))

                    self.log(f"  {protocol_name}: {import_time:.3f}ms")

                except ImportError as e:
                    self.log(f"  {protocol_name} failed: {e}")
                    sequence_times.append((protocol_name, float('inf')))

                memory_monitor.sample_memory()

            total_time = (time.perf_counter() - total_start) * 1000

        return {
            'total_sequence_time_ms': total_time,
            'individual_times': dict(sequence_times),
            'successful_imports': len([t for _, t in sequence_times if t != float('inf')]),
            'memory_stats': memory_monitor.get_memory_stats()
        }

    def run_comprehensive_benchmark(self, iterations: int = 10, concurrent_tests: int = 5) -> dict[str, Any]:
        """Run comprehensive protocol import performance benchmark."""
        print("🚀 Starting Protocol Import Performance Benchmark")
        print(f"📊 Configuration: {iterations} iterations, {concurrent_tests} concurrent threads")
        print("=" * 80)

        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'iterations': iterations,
                'concurrent_tests': concurrent_tests,
                'python_version': sys.version,
                'platform': sys.platform
            },
            'individual_domains': {},
            'lazy_loading': {},
            'concurrent_performance': {},
            'critical_path_sequence': {},
            'summary': {}
        }

        # 1. Individual domain benchmarks
        print("📦 Individual Protocol Domain Benchmarks")
        print("-" * 40)

        for domain_name, domain_config in self.protocol_domains.items():
            print(f"\n🔍 Benchmarking {domain_name} protocol ({domain_config['priority']} priority)")
            print(f"   Target: <{domain_config['target_ms']}ms")
            print(f"   Path: {domain_config['import_path']}")

            result = self.benchmark_single_import(domain_config['import_path'], iterations)
            result.update({
                'domain': domain_name,
                'priority': domain_config['priority'],
                'target_ms': domain_config['target_ms'],
                'description': domain_config['description']
            })

            benchmark_results['individual_domains'][domain_name] = result

            if result['success']:
                status = "✅ PASS" if result['mean_ms'] <= domain_config['target_ms'] else "❌ FAIL"
                print(f"   {status} - Mean: {result['mean_ms']:.3f}ms, Cold: {result['cold_start_ms']:.3f}ms")
                print(f"   Memory: +{result['memory_stats']['delta_mb']:.1f}MB")
            else:
                print(f"   ❌ FAIL - {result['error']}")

        # 2. Lazy loading benchmarks
        print("\n\n⚡ Lazy Loading Protocol Benchmarks")
        print("-" * 40)

        for lazy_name, lazy_config in self.lazy_protocols.items():
            print(f"\n🔍 Benchmarking {lazy_name}")
            print(f"   Target: <{lazy_config['target_ms']}ms")
            print(f"   Function: {lazy_config['lazy_function']}")

            result = self.benchmark_lazy_loading(
                lazy_config['import_path'],
                lazy_config['lazy_function'],
                iterations
            )
            result.update({
                'lazy_function': lazy_config['lazy_function'],
                'target_ms': lazy_config['target_ms'],
                'description': lazy_config['description']
            })

            benchmark_results['lazy_loading'][lazy_name] = result

            if result['success']:
                status = "✅ PASS" if result['mean_ms'] <= lazy_config['target_ms'] else "❌ FAIL"
                print(f"   {status} - Mean: {result['mean_ms']:.3f}ms")
                print(f"   Memory: +{result['memory_stats']['delta_mb']:.1f}MB")
            else:
                print(f"   ❌ FAIL - {result['error']}")

        # 3. Concurrent performance benchmark
        print("\n\n🔄 Concurrent Import Performance")
        print("-" * 40)

        concurrent_result = self.benchmark_concurrent_imports(concurrent_tests)
        benchmark_results['concurrent_performance'] = concurrent_result

        print(f"   Threads: {concurrent_result['concurrent_threads']}")
        print(f"   Total time: {concurrent_result['total_time_ms']:.3f}ms")
        print(f"   Successful: {concurrent_result['successful_imports']}")
        print(f"   Failed: {concurrent_result['failed_imports']}")
        print(f"   Mean per import: {concurrent_result['mean_import_time_ms']:.3f}ms")
        print(f"   Memory: +{concurrent_result['memory_stats']['delta_mb']:.1f}MB")

        # 4. Critical path sequence benchmark
        print("\n\n🎯 Critical Path Sequence Performance")
        print("-" * 40)

        critical_path_result = self.benchmark_critical_path_sequence()
        benchmark_results['critical_path_sequence'] = critical_path_result

        print(f"   Total sequence time: {critical_path_result['total_sequence_time_ms']:.3f}ms")
        print("   Target: <3ms (sum of critical protocol targets)")

        status = "✅ PASS" if critical_path_result['total_sequence_time_ms'] <= 3.0 else "❌ FAIL"
        print(f"   {status}")

        for protocol_name, import_time in critical_path_result['individual_times'].items():
            if import_time != float('inf'):
                print(f"     {protocol_name}: {import_time:.3f}ms")
            else:
                print(f"     {protocol_name}: FAILED")

        print(f"   Memory: +{critical_path_result['memory_stats']['delta_mb']:.1f}MB")

        # 5. Generate summary
        benchmark_results['summary'] = self.generate_performance_summary(benchmark_results)

        return benchmark_results

    def generate_performance_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive performance summary."""
        summary = {
            'overall_status': 'PASS',
            'critical_findings': [],
            'performance_metrics': {},
            'memory_impact': {},
            'recommendations': []
        }

        # Analyze individual domain performance
        domain_failures = []
        domain_performance = {}

        for domain_name, result in results['individual_domains'].items():
            if result['success']:
                domain_performance[domain_name] = {
                    'mean_ms': result['mean_ms'],
                    'target_ms': result['target_ms'],
                    'status': 'PASS' if result['mean_ms'] <= result['target_ms'] else 'FAIL'
                }

                if result['mean_ms'] > result['target_ms']:
                    domain_failures.append(f"{domain_name}: {result['mean_ms']:.3f}ms > {result['target_ms']}ms")
                    summary['overall_status'] = 'FAIL'
            else:
                domain_failures.append(f"{domain_name}: Import failed - {result['error']}")
                summary['overall_status'] = 'FAIL'

        # Analyze lazy loading performance
        lazy_failures = []
        for lazy_name, result in results['lazy_loading'].items():
            if result['success']:
                if result['mean_ms'] > result['target_ms']:
                    lazy_failures.append(f"{lazy_name}: {result['mean_ms']:.3f}ms > {result['target_ms']}ms")
                    summary['overall_status'] = 'FAIL'
            else:
                lazy_failures.append(f"{lazy_name}: Lazy loading failed - {result['error']}")
                summary['overall_status'] = 'FAIL'

        # Critical path analysis
        critical_path_time = results['critical_path_sequence']['total_sequence_time_ms']
        if critical_path_time > 3.0:
            summary['critical_findings'].append(f"Critical path sequence: {critical_path_time:.3f}ms > 3.0ms")
            summary['overall_status'] = 'FAIL'

        # Memory analysis
        total_memory_delta = 0
        for domain_result in results['individual_domains'].values():
            if 'memory_stats' in domain_result:
                total_memory_delta += domain_result['memory_stats'].get('delta_mb', 0)

        if total_memory_delta > 50:  # 50MB threshold
            summary['critical_findings'].append(f"Total memory impact: {total_memory_delta:.1f}MB > 50MB")

        # Compile findings
        summary['critical_findings'].extend(domain_failures)
        summary['critical_findings'].extend(lazy_failures)

        summary['performance_metrics'] = {
            'critical_path_time_ms': critical_path_time,
            'total_memory_delta_mb': total_memory_delta,
            'domain_performance': domain_performance,
            'concurrent_performance': {
                'mean_import_time_ms': results['concurrent_performance']['mean_import_time_ms'],
                'successful_imports': results['concurrent_performance']['successful_imports'],
                'failed_imports': results['concurrent_performance']['failed_imports']
            }
        }

        # Generate recommendations
        if domain_failures:
            summary['recommendations'].append("Optimize slow protocol domains with lazy loading or dependency reduction")
        if lazy_failures:
            summary['recommendations'].append("Investigate lazy loading implementation for ML/monitoring protocols")
        if critical_path_time > 3.0:
            summary['recommendations'].append("Critical path optimization needed for sub-3ms performance")
        if total_memory_delta > 50:
            summary['recommendations'].append("Consider protocol consolidation to reduce memory overhead")

        return summary

    def save_results(self, results: dict[str, Any], output_path: str | None = None):
        """Save benchmark results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"protocol_import_performance_benchmark_{timestamp}.json"

        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_path}")
        return output_path


def main():
    """Main benchmark execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Protocol Import Performance Benchmark')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--iterations', '-i', type=int, default=10, help='Number of iterations per test')
    parser.add_argument('--concurrent-tests', '-c', type=int, default=5, help='Number of concurrent import threads')
    parser.add_argument('--output', '-o', type=str, help='Output file path for results')

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = ProtocolImportBenchmark(verbose=args.verbose)

    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark(
            iterations=args.iterations,
            concurrent_tests=args.concurrent_tests
        )

        # Print final summary
        print("\n" + "=" * 80)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 80)

        summary = results['summary']
        status_icon = "✅" if summary['overall_status'] == 'PASS' else "❌"
        print(f"{status_icon} Overall Status: {summary['overall_status']}")

        if summary['critical_findings']:
            print(f"\n⚠️  Critical Findings ({len(summary['critical_findings'])}):")
            for finding in summary['critical_findings']:
                print(f"   • {finding}")

        print("\n📊 Performance Metrics:")
        metrics = summary['performance_metrics']
        print(f"   • Critical Path Time: {metrics['critical_path_time_ms']:.3f}ms (target: <3ms)")
        print(f"   • Total Memory Delta: {metrics['total_memory_delta_mb']:.1f}MB (target: <50MB)")
        print(f"   • Concurrent Success Rate: {metrics['concurrent_performance']['successful_imports']}/{metrics['concurrent_performance']['successful_imports'] + metrics['concurrent_performance']['failed_imports']}")

        if summary['recommendations']:
            print(f"\n💡 Recommendations ({len(summary['recommendations'])}):")
            for rec in summary['recommendations']:
                print(f"   • {rec}")

        # Save results
        output_path = benchmark.save_results(results, args.output)

        print("\n🎯 Benchmark completed successfully!")
        print(f"   Results: {output_path}")

        # Exit with appropriate code
        sys.exit(0 if summary['overall_status'] == 'PASS' else 1)

    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
