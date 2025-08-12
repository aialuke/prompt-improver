"""Performance Baseline Comparison Script
Compares two baseline reports and identifies performance changes.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def load_baseline(path: str) -> dict[str, Any]:
    """Load a baseline report from JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        sys.exit(1)


def compare_metric(
    baseline_value: float,
    current_value: float,
    metric_name: str,
    lower_is_better: bool = True,
) -> tuple[str, float, str]:
    """Compare two metric values and return status, change percentage, and interpretation."""
    if baseline_value == 0:
        return ("NEW", 0, f"{metric_name} is new")
    change_percent = (current_value - baseline_value) / baseline_value * 100
    if lower_is_better:
        if change_percent < -10:
            status = "IMPROVED"
            interpretation = f"{metric_name} improved significantly"
        elif change_percent < -5:
            status = "BETTER"
            interpretation = f"{metric_name} improved"
        elif change_percent > 20:
            status = "DEGRADED"
            interpretation = f"{metric_name} performance degraded significantly"
        elif change_percent > 10:
            status = "WORSE"
            interpretation = f"{metric_name} performance declined"
        else:
            status = "STABLE"
            interpretation = f"{metric_name} is stable"
    elif change_percent > 10:
        status = "IMPROVED"
        interpretation = f"{metric_name} improved significantly"
    elif change_percent > 5:
        status = "BETTER"
        interpretation = f"{metric_name} improved"
    elif change_percent < -20:
        status = "DEGRADED"
        interpretation = f"{metric_name} performance degraded significantly"
    elif change_percent < -10:
        status = "WORSE"
        interpretation = f"{metric_name} performance declined"
    else:
        status = "STABLE"
        interpretation = f"{metric_name} is stable"
    return (status, change_percent, interpretation)


def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    emojis = {
        "IMPROVED": "🚀",
        "BETTER": "✅",
        "STABLE": "➡️",
        "WORSE": "⚠️",
        "DEGRADED": "❌",
        "NEW": "🆕",
    }
    return emojis.get(status, "❓")


def compare_system_metrics(baseline: dict, current: dict) -> None:
    """Compare system-level metrics."""
    print("\n🖥️  SYSTEM METRICS COMPARISON")
    print("-" * 50)
    b_sys = baseline.get("system_metrics", {})
    c_sys = current.get("system_metrics", {})
    metrics = [
        ("cpu_percent", "CPU Usage", True),
        ("memory_percent", "Memory Usage", True),
        ("memory_used_gb", "Memory Used (GB)", True),
        ("disk_percent", "Disk Usage", True),
        ("process_count", "Process Count", True),
    ]
    for metric_key, metric_name, lower_is_better in metrics:
        if metric_key in b_sys and metric_key in c_sys:
            status, change, interpretation = compare_metric(
                b_sys[metric_key], c_sys[metric_key], metric_name, lower_is_better
            )
            emoji = get_status_emoji(status)
            print(
                f"{emoji} {metric_name:15} {b_sys[metric_key]:8.1f} → {c_sys[metric_key]:8.1f} ({change:+.1f}%)"
            )


def compare_ml_metrics(baseline: dict, current: dict) -> None:
    """Compare ML performance metrics."""
    print("\n🤖 ML PERFORMANCE COMPARISON")
    print("-" * 50)
    b_ml = baseline.get("ml_metrics", [])
    c_ml = current.get("ml_metrics", [])
    b_models = {ml["model_name"]: ml for ml in b_ml}
    c_models = {ml["model_name"]: ml for ml in c_ml}
    all_models = set(b_models.keys()) | set(c_models.keys())
    for model_name in sorted(all_models):
        print(f"\n   {model_name}:")
        if model_name in b_models and model_name in c_models:
            b_model = b_models[model_name]
            c_model = c_models[model_name]
            metrics = [
                ("initialization_time_ms", "Init Time", True),
                ("inference_time_ms", "Inference Time", True),
                ("memory_usage_mb", "Memory Usage", True),
                ("features_extracted", "Features", False),
            ]
            for metric_key, metric_name, lower_is_better in metrics:
                if metric_key in b_model and metric_key in c_model:
                    status, change, _ = compare_metric(
                        b_model[metric_key],
                        c_model[metric_key],
                        metric_name,
                        lower_is_better,
                    )
                    emoji = get_status_emoji(status)
                    print(
                        f"     {emoji} {metric_name:12} {b_model[metric_key]:8.2f} → {c_model[metric_key]:8.2f} ({change:+.1f}%)"
                    )
        elif model_name in c_models:
            print("     🆕 New model added")
        else:
            print("     ❌ Model removed")


def compare_cpu_benchmarks(baseline: dict, current: dict) -> None:
    """Compare CPU benchmark results."""
    print("\n⚡ CPU PERFORMANCE COMPARISON")
    print("-" * 50)
    b_cpu = baseline.get("cpu_benchmarks", {})
    c_cpu = current.get("cpu_benchmarks", {})
    benchmarks = [
        ("prime_calculation", "execution_time_ms", "Prime Calculation"),
        ("string_processing", "execution_time_ms", "String Processing"),
        ("json_processing", "execution_time_ms", "JSON Processing"),
    ]
    for bench_key, metric_key, bench_name in benchmarks:
        if bench_key in b_cpu and bench_key in c_cpu:
            b_val = b_cpu[bench_key].get(metric_key, 0)
            c_val = c_cpu[bench_key].get(metric_key, 0)
            if b_val > 0 and c_val > 0:
                status, change, _ = compare_metric(b_val, c_val, bench_name, True)
                emoji = get_status_emoji(status)
                print(
                    f"{emoji} {bench_name:18} {b_val:8.2f}ms → {c_val:8.2f}ms ({change:+.1f}%)"
                )


def compare_io_benchmarks(baseline: dict, current: dict) -> None:
    """Compare I/O benchmark results."""
    print("\n💾 I/O PERFORMANCE COMPARISON")
    print("-" * 50)
    b_io = baseline.get("io_benchmarks", {})
    c_io = current.get("io_benchmarks", {})
    io_metrics = [
        ("file_write", "write_speed_mb_per_sec", "Write Speed (MB/s)", False),
        ("file_read", "read_speed_mb_per_sec", "Read Speed (MB/s)", False),
        ("directory_operations", "execution_time_ms", "Directory Ops (ms)", True),
    ]
    for io_key, metric_key, metric_name, lower_is_better in io_metrics:
        if io_key in b_io and io_key in c_io:
            b_val = b_io[io_key].get(metric_key, 0)
            c_val = c_io[io_key].get(metric_key, 0)
            if b_val > 0 and c_val > 0:
                status, change, _ = compare_metric(
                    b_val, c_val, metric_name, lower_is_better
                )
                emoji = get_status_emoji(status)
                unit = "MB/s" if "speed" in metric_key else "ms"
                print(
                    f"{emoji} {metric_name:18} {b_val:8.2f}{unit} → {c_val:8.2f}{unit} ({change:+.1f}%)"
                )


def generate_summary(baseline: dict, current: dict) -> None:
    """Generate overall performance summary."""
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 50)
    status_counts = {"IMPROVED": 0, "BETTER": 0, "STABLE": 0, "WORSE": 0, "DEGRADED": 0}
    b_summary = baseline.get("summary", {})
    c_summary = current.get("summary", {})
    if (
        "cpu_utilization_percent" in b_summary
        and "cpu_utilization_percent" in c_summary
    ):
        cpu_status, cpu_change, _ = compare_metric(
            b_summary["cpu_utilization_percent"],
            c_summary["cpu_utilization_percent"],
            "CPU",
            True,
        )
        print(
            f"   Overall CPU Usage:     {get_status_emoji(cpu_status)} {cpu_change:+.1f}%"
        )
    if "total_memory_used_mb" in b_summary and "total_memory_used_mb" in c_summary:
        mem_status, mem_change, _ = compare_metric(
            b_summary["total_memory_used_mb"],
            c_summary["total_memory_used_mb"],
            "Memory",
            True,
        )
        print(
            f"   Overall Memory Usage:  {get_status_emoji(mem_status)} {mem_change:+.1f}%"
        )
    if (
        "benchmark_duration_seconds" in b_summary
        and "benchmark_duration_seconds" in c_summary
    ):
        duration_status, duration_change, _ = compare_metric(
            b_summary["benchmark_duration_seconds"],
            c_summary["benchmark_duration_seconds"],
            "Benchmark Duration",
            True,
        )
        print(
            f"   Benchmark Duration:    {get_status_emoji(duration_status)} {duration_change:+.1f}%"
        )


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare two performance baseline reports"
    )
    parser.add_argument("baseline", help="Path to baseline report JSON file")
    parser.add_argument("current", help="Path to current report JSON file")
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed comparison"
    )
    args = parser.parse_args()
    print("🔍 Loading performance baseline reports...")
    baseline = load_baseline(args.baseline)
    current = load_baseline(args.current)
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE BASELINE COMPARISON")
    print("=" * 60)
    baseline_time = baseline.get("timestamp", "unknown")
    current_time = current.get("timestamp", "unknown")
    print(f"Baseline:  {baseline_time}")
    print(f"Current:   {current_time}")
    compare_system_metrics(baseline, current)
    compare_ml_metrics(baseline, current)
    compare_cpu_benchmarks(baseline, current)
    compare_io_benchmarks(baseline, current)
    generate_summary(baseline, current)
    print("\n✅ Performance comparison completed!")
    print("\n📋 INTERPRETATION GUIDE:")
    print("   🚀 IMPROVED   - Significant improvement (>10%)")
    print("   ✅ BETTER     - Moderate improvement (5-10%)")
    print("   ➡️ STABLE     - No significant change (<5%)")
    print("   ⚠️ WORSE      - Moderate degradation (10-20%)")
    print("   ❌ DEGRADED   - Significant degradation (>20%)")
    print("   🆕 NEW        - New metric or component")


if __name__ == "__main__":
    main()
