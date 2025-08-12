#!/usr/bin/env python3
"""Performance Baseline Measurement Script

This script runs comprehensive performance benchmarking to establish baselines
and validate that the system meets the performance targets outlined in
Validation_Consolidation.md.

Usage:
    python scripts/run_performance_baseline.py --operations 10000
    python scripts/run_performance_baseline.py --quick --ci-stage integration_tests
    python scripts/run_performance_baseline.py --memory-test --operations 100000
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from prompt_improver.performance.validation.ci_integration import (
    CIStage,
    run_performance_gate_cli,
)
from prompt_improver.performance.validation.comprehensive_benchmark import (
    run_validation_benchmark,
)
from prompt_improver.performance.validation.memory_profiler import (
    run_memory_leak_detection,
)
from prompt_improver.performance.validation.regression_detector import (
    get_regression_detector,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_comprehensive_baseline(
    operations: int = 10000,
    memory_operations: int = 100000,
    concurrent_sessions: int = 100,
    output_dir: str = "performance_results",
) -> None:
    """Run comprehensive performance baseline measurement."""
    logger.info("Starting comprehensive performance baseline measurement")
    logger.info(
        f"Operations: {operations:,}, Memory ops: {memory_operations:,}, Concurrent: {concurrent_sessions:,}"
    )

    # Create output directory
    results_dir = Path(output_dir)
    results_dir.mkdir(exist_ok=True)

    # 1. Run validation benchmarks
    logger.info("=" * 60)
    logger.info("Phase 1: Validation Performance Benchmarks")
    logger.info("=" * 60)

    benchmark_results = await run_validation_benchmark(
        operations=operations,
        memory_operations=memory_operations,
        concurrent_sessions=concurrent_sessions,
    )

    # Print immediate results
    print("\n" + "=" * 80)
    print("VALIDATION PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)

    for operation_name, result in benchmark_results.items():
        status = "✅ MEETS TARGET" if result.meets_target else "❌ BELOW TARGET"
        print(f"\n{operation_name.replace('_', ' ').title()}:")
        print(f"  Status: {status}")
        print(
            f"  Current: {result.current_performance_us:.2f}μs (target: {result.target_performance_us:.2f}μs)"
        )
        print(f"  P95: {result.p95_latency_us:.2f}μs")
        print(f"  P99: {result.p99_latency_us:.2f}μs")
        print(f"  Success Rate: {result.success_rate:.1%}")
        print(f"  Memory Usage: {result.memory_usage_kb:.1f}KB")
        print(f"  Improvement: {result.improvement_percent:+.1f}%")

        if result.regression_detected:
            print("  ⚠️  REGRESSION DETECTED")
        if result.memory_leak_detected:
            print("  🚨 MEMORY LEAK DETECTED")

    # 2. Run memory leak detection
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Memory Leak Detection")
    logger.info("=" * 60)

    memory_results = await run_memory_leak_detection(
        operations=memory_operations, output_dir=str(results_dir / "memory_analysis")
    )

    # Print memory results summary
    print("\n" + "=" * 80)
    print("MEMORY LEAK DETECTION RESULTS")
    print("=" * 80)

    if memory_results and "summary" in memory_results:
        summary = memory_results["summary"]
        print(f"Operations Tested: {summary.get('total_operations_tested', 0)}")
        print(f"Operations with Leaks: {summary.get('operations_with_leaks', 0)}")
        print(f"Critical Leaks: {summary.get('critical_leaks', 0)}")
        print(f"High Priority Leaks: {summary.get('high_leaks', 0)}")
        print(f"Medium Priority Leaks: {summary.get('medium_leaks', 0)}")
        print(f"Total Memory Growth: {summary.get('total_memory_growth_mb', 0):.1f}MB")
        print(f"Average GC Efficiency: {summary.get('gc_efficiency_average', 0):.1f}%")

        if summary.get("memory_leaks_detected", 0) > 0:
            print(
                "\n⚠️  Memory leaks detected - see detailed report for recommendations"
            )
        else:
            print("\n✅ No critical memory leaks detected")

    # 3. Generate summary report
    await generate_baseline_summary(benchmark_results, memory_results, results_dir)

    logger.info(
        f"\nComprehensive baseline measurement completed. Results saved to: {results_dir}"
    )


async def run_quick_check(operations: int = 1000) -> None:
    """Run quick performance check for development feedback."""
    logger.info("Running quick performance check...")

    benchmark_results = await run_validation_benchmark(
        operations=operations, memory_operations=operations * 2, concurrent_sessions=10
    )

    print("\n" + "=" * 60)
    print("QUICK PERFORMANCE CHECK RESULTS")
    print("=" * 60)

    all_targets_met = True
    for operation_name, result in benchmark_results.items():
        status_icon = "✅" if result.meets_target else "❌"
        print(
            f"{status_icon} {operation_name}: {result.current_performance_us:.1f}μs (target: {result.target_performance_us:.1f}μs)"
        )

        if not result.meets_target:
            all_targets_met = False

    if all_targets_met:
        print("\n🎉 All performance targets met!")
        return 0
    print("\n⚠️  Some performance targets not met - see detailed results above")
    return 1


async def run_ci_performance_gate(stage: str, quick: bool = False) -> int:
    """Run CI performance gate for a specific stage."""
    logger.info(f"Running CI performance gate for stage: {stage}")

    try:
        exit_code = await run_performance_gate_cli(
            stage=stage, quick=quick, config_dir=".performance"
        )
        return exit_code
    except Exception as e:
        logger.error(f"CI performance gate failed: {e}")
        return 2


async def generate_baseline_summary(
    benchmark_results: dict, memory_results: dict, output_dir: Path
) -> None:
    """Generate a comprehensive baseline summary report."""
    summary = {
        "timestamp": benchmark_results[
            list(benchmark_results.keys())[0]
        ].timestamp.isoformat(),
        "validation_benchmarks": {
            name: {
                "current_performance_us": result.current_performance_us,
                "target_performance_us": result.target_performance_us,
                "meets_target": result.meets_target,
                "improvement_factor": result.improvement_factor,
                "success_rate": result.success_rate,
                "memory_usage_kb": result.memory_usage_kb,
                "samples_count": result.samples_count,
            }
            for name, result in benchmark_results.items()
        },
        "memory_analysis": memory_results.get("summary", {}) if memory_results else {},
        "overall_assessment": {
            "total_operations_tested": len(benchmark_results),
            "operations_meeting_targets": sum(
                1 for r in benchmark_results.values() if r.meets_target
            ),
            "target_compliance_rate": sum(
                1 for r in benchmark_results.values() if r.meets_target
            )
            / len(benchmark_results),
            "memory_leaks_detected": memory_results.get("summary", {}).get(
                "memory_leaks_detected", 0
            )
            if memory_results
            else 0,
            "baseline_established": True,
        },
    }

    # Save JSON summary
    summary_file = output_dir / "performance_baseline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, indent=2, fp=f)

    # Generate human-readable report
    report_lines = [
        "=" * 80,
        "PERFORMANCE BASELINE SUMMARY REPORT",
        "=" * 80,
        f"Generated: {summary['timestamp']}",
        "",
        "VALIDATION PERFORMANCE TARGETS:",
        "-" * 40,
    ]

    # Performance targets from Validation_Consolidation.md
    targets = {
        "mcp_message_decode": {"current": 543.0, "target": 6.4, "improvement": "85x"},
        "config_instantiation": {"current": 54.3, "target": 8.4, "improvement": "6.5x"},
        "metrics_collection": {"current": 12.1, "target": 1.0, "improvement": "12x"},
    }

    for operation, target_info in targets.items():
        if operation in summary["validation_benchmarks"]:
            result = summary["validation_benchmarks"][operation]
            status = "✅ MET" if result["meets_target"] else "❌ NOT MET"
            report_lines.extend([
                f"  {operation.replace('_', ' ').title()}:",
                f"    Target: {target_info['target']}μs ({target_info['improvement']} improvement)",
                f"    Current: {result['current_performance_us']:.2f}μs",
                f"    Status: {status}",
                f"    Success Rate: {result['success_rate']:.1%}",
                "",
            ])

    # Overall assessment
    assessment = summary["overall_assessment"]
    compliance_rate = assessment["target_compliance_rate"] * 100

    report_lines.extend([
        "OVERALL ASSESSMENT:",
        "-" * 20,
        f"Operations Tested: {assessment['total_operations_tested']}",
        f"Targets Met: {assessment['operations_meeting_targets']}/{assessment['total_operations_tested']}",
        f"Compliance Rate: {compliance_rate:.1f}%",
        f"Memory Leaks: {assessment['memory_leaks_detected']}",
        "",
        "STATUS: "
        + (
            "🎉 BASELINE ESTABLISHED"
            if compliance_rate >= 80
            else "⚠️  PERFORMANCE OPTIMIZATION NEEDED"
        ),
        "",
        "NEXT STEPS:",
        "-" * 10,
    ])

    if compliance_rate >= 80:
        report_lines.extend([
            "• Baseline successfully established",
            "• Enable continuous performance monitoring",
            "• Set up CI/CD performance gates",
            "• Begin msgspec migration planning",
        ])
    else:
        report_lines.extend([
            "• Address performance targets not meeting requirements",
            "• Focus on operations with largest performance gaps",
            "• Review validation logic complexity",
            "• Consider immediate optimization work",
        ])

    report_lines.extend(["", "=" * 80])

    # Save text report
    report_file = output_dir / "performance_baseline_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nBaseline summary saved to: {summary_file}")
    print(f"Human-readable report saved to: {report_file}")


async def main():
    """Main entry point for performance baseline measurement."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Baseline Measurement")
    parser.add_argument(
        "--operations", type=int, default=10000, help="Operations per benchmark"
    )
    parser.add_argument(
        "--memory-operations",
        type=int,
        default=100000,
        help="Operations for memory testing",
    )
    parser.add_argument(
        "--concurrent-sessions",
        type=int,
        default=100,
        help="Concurrent sessions for stress testing",
    )
    parser.add_argument(
        "--output-dir", type=str, default="performance_results", help="Output directory"
    )
    parser.add_argument("--quick", action="store_true", help="Quick check mode")
    parser.add_argument(
        "--memory-test", action="store_true", help="Focus on memory leak detection"
    )
    parser.add_argument(
        "--ci-stage", choices=[s.value for s in CIStage], help="Run CI performance gate"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.ci_stage:
            # CI performance gate
            exit_code = await run_ci_performance_gate(args.ci_stage, args.quick)
            sys.exit(exit_code)

        elif args.quick:
            # Quick performance check
            exit_code = await run_quick_check(args.operations)
            sys.exit(exit_code)

        elif args.memory_test:
            # Focus on memory leak detection
            results = await run_memory_leak_detection(
                operations=args.memory_operations, output_dir=args.output_dir
            )

            if (
                results
                and results.get("summary", {}).get("memory_leaks_detected", 0) > 0
            ):
                print("⚠️  Memory leaks detected")
                sys.exit(1)
            else:
                print("✅ No memory leaks detected")
                sys.exit(0)

        else:
            # Full comprehensive baseline
            await run_comprehensive_baseline(
                operations=args.operations,
                memory_operations=args.memory_operations,
                concurrent_sessions=args.concurrent_sessions,
                output_dir=args.output_dir,
            )
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Baseline measurement interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Baseline measurement failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
