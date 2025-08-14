#!/usr/bin/env python3
"""Performance benchmark for string formatting modernization.

Compares performance of old vs new string formatting approaches.
"""

import statistics
import sys
import time
from datetime import UTC, datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.utils.datetime_utils import (
    format_compact_timestamp,
    format_date_only,
    format_display_date,
    format_log_timestamp,
)


def benchmark_datetime_formatting():
    """Benchmark datetime formatting performance."""
    iterations = 100000
    dt = datetime.now(UTC)

    print(f"Benchmarking datetime formatting ({iterations:,} iterations)...")

    # Old approach: strftime
    start = time.perf_counter()
    for _ in range(iterations):
        compact = dt.strftime("%Y%m%d_%H%M%S")
        display = dt.strftime("%Y-%m-%d %H:%M:%S")
        date_only = dt.strftime("%Y-%m-%d")
    old_time = time.perf_counter() - start

    # New approach: utility functions
    start = time.perf_counter()
    for _ in range(iterations):
        compact = format_compact_timestamp(dt)
        display = format_display_date(dt)
        date_only = format_date_only(dt)
    new_time = time.perf_counter() - start

    improvement = (old_time - new_time) / old_time * 100

    print(f"Old strftime approach: {old_time:.4f}s")
    print(f"New utility approach:  {new_time:.4f}s")
    print(f"Performance improvement: {improvement:+.1f}%")

    return improvement


def benchmark_string_formatting():
    """Benchmark various string formatting approaches."""
    iterations = 100000
    values = {"name": "test", "value": 42, "timestamp": "2025-01-01"}

    print(f"\nBenchmarking string formatting ({iterations:,} iterations)...")

    # Old % formatting
    start = time.perf_counter()
    for _ in range(iterations):
        result = "Processing %s with value %d at %s" % (
            values["name"],
            values["value"],
            values["timestamp"],
        )
    old_time = time.perf_counter() - start

    # New f-strings
    start = time.perf_counter()
    for _ in range(iterations):
        result = f"Processing {values['name']} with value {values['value']} at {values['timestamp']}"
    new_time = time.perf_counter() - start

    improvement = (old_time - new_time) / old_time * 100

    print(f"Old % formatting:    {old_time:.4f}s")
    print(f"New f-strings:       {new_time:.4f}s")
    print(f"Performance improvement: {improvement:+.1f}%")

    return improvement


def benchmark_complex_formatting():
    """Benchmark complex formatting scenarios."""
    iterations = 50000
    dt = datetime.now(UTC)

    print(f"\nBenchmarking complex formatting ({iterations:,} iterations)...")

    # Old approach with multiple strftime calls
    start = time.perf_counter()
    for _ in range(iterations):
        timestamp_id = f"task_{dt.strftime('%Y%m%d_%H%M%S')}_{42}"
        log_entry = (
            f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] Processing task {timestamp_id}"
        )
        filename = f"report_{dt.strftime('%Y-%m-%d')}.txt"
    old_time = time.perf_counter() - start

    # New approach with utilities
    start = time.perf_counter()
    for _ in range(iterations):
        timestamp_id = f"task_{format_compact_timestamp(dt)}_{42}"
        log_entry = f"[{format_display_date(dt)}] Processing task {timestamp_id}"
        filename = f"report_{format_date_only(dt)}.txt"
    new_time = time.perf_counter() - start

    improvement = (old_time - new_time) / old_time * 100

    print(f"Old complex formatting: {old_time:.4f}s")
    print(f"New utility formatting: {new_time:.4f}s")
    print(f"Performance improvement: {improvement:+.1f}%")

    return improvement


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmarks."""
    print("String Formatting Modernization Performance Benchmark")
    print("=" * 60)

    improvements = []

    # Datetime formatting
    improvements.append(benchmark_datetime_formatting())

    # String formatting
    improvements.append(benchmark_string_formatting())

    # Complex scenarios
    improvements.append(benchmark_complex_formatting())

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    avg_improvement = statistics.mean(improvements)

    print(f"Average performance improvement: {avg_improvement:+.1f}%")
    print("Individual improvements:")
    print(f"  - Datetime formatting: {improvements[0]:+.1f}%")
    print(f"  - String formatting:   {improvements[1]:+.1f}%")
    print(f"  - Complex scenarios:   {improvements[2]:+.1f}%")

    if avg_improvement >= 10:
        print(
            f"\n✅ Target achieved! Average improvement of {avg_improvement:.1f}% exceeds 10% goal"
        )
    elif avg_improvement >= 5:
        print(
            f"\n🟡 Good progress! Average improvement of {avg_improvement:.1f}% is approaching 10% goal"
        )
    else:
        print(
            f"\n🔴 More work needed. Average improvement of {avg_improvement:.1f}% is below target"
        )

    return avg_improvement


if __name__ == "__main__":
    run_comprehensive_benchmark()
