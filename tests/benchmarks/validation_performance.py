#!/usr/bin/env python3
"""
Validation Performance Benchmark
Run: python tests/benchmarks/validation_performance.py

This benchmarks different validation libraries to establish baseline performance
metrics for the Pydantic V1 to V2 migration and msgspec adoption.

Based on Validation_Consolidation.md requirements for real behavior testing.
"""

import json
import os
import sys
import timeit
import tracemalloc
from dataclasses import dataclass
from typing import Any, NamedTuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    import msgspec
    from pydantic import BaseModel
    from sqlmodel import SQLModel
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install msgspec pydantic sqlmodel")
    sys.exit(1)

# Test data representing typical MCP message structure
TEST_DATA = {
    "id": "prompt-enhancement-123",
    "session_id": "sess-abc-def-789",
    "timestamp": "2025-08-09T10:30:45.123456",
    "prompt": "Write a Python function that validates email addresses using regex",
    "improved_prompt": "Write a robust Python function that validates email addresses using regex patterns, including edge cases for international domains and plus addressing.",
    "metadata": {
        "user_id": "user-456",
        "model": "claude-3-sonnet",
        "enhancement_rules": ["clarity", "specificity", "context"],
        "performance_metrics": {
            "processing_time_ms": 245.7,
            "improvement_score": 0.85,
            "confidence": 0.92,
        },
    },
    "context": {
        "domain": "software_engineering",
        "complexity": "intermediate",
        "language": "python",
    },
    "active": True,
    "retry_count": 0,
}


class BenchmarkResult(NamedTuple):
    """Structured benchmark results"""

    library: str
    time_per_op_us: float
    ops_per_sec: float
    memory_per_1k_kb: float
    improvement_factor: float = 1.0


# Pydantic V2 Model
class PydanticPromptRequest(BaseModel):
    id: str
    session_id: str
    timestamp: str
    prompt: str
    improved_prompt: str
    metadata: dict[str, Any]
    context: dict[str, Any]
    active: bool
    retry_count: int


# SQLModel (simulating current usage in non-database classes)
class SQLModelPromptRequest(SQLModel):
    id: str
    session_id: str
    timestamp: str
    prompt: str
    improved_prompt: str
    metadata: dict[str, Any]
    context: dict[str, Any]
    active: bool
    retry_count: int


# msgspec Struct (target for MCP messages)
class MsgspecPromptRequest(msgspec.Struct):
    id: str
    session_id: str
    timestamp: str
    prompt: str
    improved_prompt: str
    metadata: dict[str, Any]
    context: dict[str, Any]
    active: bool
    retry_count: int


# Dataclass (target for metrics and internal data)
@dataclass
class DataclassPromptRequest:
    id: str
    session_id: str
    timestamp: str
    prompt: str
    improved_prompt: str
    metadata: dict[str, Any]
    context: dict[str, Any]
    active: bool
    retry_count: int


def benchmark_library(
    name: str, create_func: callable, iterations: int = 10000
) -> BenchmarkResult:
    """
    Benchmark a validation library with both time and memory measurements.

    Uses real behavior testing - no mocks, actual object creation and validation.
    """

    print(f"Benchmarking {name}...")

    # Time measurement - multiple runs for accuracy
    time_taken = timeit.timeit(lambda: create_func(TEST_DATA), number=iterations)

    # Memory measurement with larger sample
    tracemalloc.start()
    instances = []
    for _ in range(1000):
        try:
            instance = create_func(TEST_DATA)
            instances.append(instance)
        except Exception as e:
            print(f"Error creating {name} instance: {e}")
            tracemalloc.stop()
            return BenchmarkResult(name, float("inf"), 0, float("inf"))

    current, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ensure instances aren't garbage collected during measurement
    _ = len(instances)

    time_per_op_us = (time_taken / iterations) * 1_000_000
    ops_per_sec = iterations / time_taken if time_taken > 0 else 0
    memory_per_1k_kb = current / 1024

    return BenchmarkResult(
        library=name,
        time_per_op_us=time_per_op_us,
        ops_per_sec=ops_per_sec,
        memory_per_1k_kb=memory_per_1k_kb,
    )


def run_serialization_benchmark():
    """Benchmark serialization performance (critical for API responses)"""
    print("\nRunning serialization benchmarks...")

    # Create instances
    pydantic_obj = PydanticPromptRequest(**TEST_DATA)
    sqlmodel_obj = SQLModelPromptRequest(**TEST_DATA)
    msgspec_obj = msgspec.convert(TEST_DATA, MsgspecPromptRequest)
    dataclass_obj = DataclassPromptRequest(**TEST_DATA)

    def time_operation(name: str, operation: callable, iterations: int = 5000):
        time_taken = timeit.timeit(operation, number=iterations)
        ops_per_sec = iterations / time_taken if time_taken > 0 else 0
        return name, (time_taken / iterations) * 1_000_000, ops_per_sec

    serialization_results = []

    # Pydantic serialization
    serialization_results.append(
        time_operation("Pydantic model_dump()", pydantic_obj.model_dump)
    )

    serialization_results.append(
        time_operation(
            "Pydantic model_dump_json()", pydantic_obj.model_dump_json
        )
    )

    # SQLModel serialization
    serialization_results.append(
        time_operation("SQLModel model_dump()", sqlmodel_obj.model_dump)
    )

    # msgspec serialization
    serialization_results.append(
        time_operation(
            "msgspec json.encode()", lambda: msgspec.json.encode(msgspec_obj)
        )
    )

    # dataclass serialization
    from dataclasses import asdict

    serialization_results.append(
        time_operation("dataclass asdict()", lambda: asdict(dataclass_obj))
    )

    serialization_results.append(
        time_operation(
            "dataclass + json.dumps()",
            lambda: json.dumps(asdict(dataclass_obj), default=str),
        )
    )

    print("\nSerialization Performance:")
    print("-" * 60)
    print(f"{'Method':<25} {'Time (μs)':<12} {'Ops/sec':<15}")
    print("-" * 60)

    for name, time_us, ops_per_sec in sorted(serialization_results, key=lambda x: x[1]):
        print(f"{name:<25} {time_us:<12.2f} {ops_per_sec:<15.0f}")


def main():
    """Run comprehensive validation performance benchmarks"""

    print("Validation Library Performance Benchmark")
    print("=" * 60)
    print(f"Test data size: {len(json.dumps(TEST_DATA))} characters")
    print(f"Python version: {sys.version}")
    print()

    results = []

    # Benchmark each library
    benchmark_functions = [
        ("Pydantic V2", lambda d: PydanticPromptRequest(**d)),
        ("SQLModel", lambda d: SQLModelPromptRequest(**d)),
        ("msgspec", lambda d: msgspec.convert(d, MsgspecPromptRequest)),
        ("dataclass", lambda d: DataclassPromptRequest(**d)),
    ]

    for name, func in benchmark_functions:
        try:
            result = benchmark_library(name, func)
            results.append(result)
        except Exception as e:
            print(f"Failed to benchmark {name}: {e}")
            continue

    if not results:
        print("❌ No benchmarks completed successfully")
        return 1

    # Calculate improvement factors (compared to slowest)
    slowest_time = max(
        r.time_per_op_us for r in results if r.time_per_op_us != float("inf")
    )

    enhanced_results = []
    for r in results:
        improvement_factor = (
            slowest_time / r.time_per_op_us if r.time_per_op_us > 0 else 0
        )
        enhanced_results.append(r._replace(improvement_factor=improvement_factor))

    # Print results
    print("\nObject Creation Performance:")
    print("=" * 80)
    print(
        f"{'Library':<15} {'Time (μs)':<12} {'Ops/sec':<15} {'Memory (KB)':<12} {'Improvement':<12}"
    )
    print("-" * 80)

    for r in sorted(enhanced_results, key=lambda x: x.time_per_op_us):
        if r.time_per_op_us == float("inf"):
            continue
        print(
            f"{r.library:<15} {r.time_per_op_us:<12.2f} {r.ops_per_sec:<15.0f} {r.memory_per_1k_kb:<12.1f} {r.improvement_factor:<12.1f}x"
        )

    # Run serialization benchmarks
    run_serialization_benchmark()

    # Summary with recommendations
    print(f"\n{'=' * 60}")
    print("PERFORMANCE ANALYSIS SUMMARY")
    print(f"{'=' * 60}")

    if len(enhanced_results) >= 2:
        fastest = min(enhanced_results, key=lambda x: x.time_per_op_us)
        slowest = max(
            (r for r in enhanced_results if r.time_per_op_us != float("inf")),
            key=lambda x: x.time_per_op_us,
        )

        print(
            f"🚀 Fastest: {fastest.library} ({fastest.time_per_op_us:.1f}μs per operation)"
        )
        print(
            f"🐌 Slowest: {slowest.library} ({slowest.time_per_op_us:.1f}μs per operation)"
        )
        print(
            f"📊 Performance difference: {slowest.time_per_op_us / fastest.time_per_op_us:.1f}x improvement available"
        )

        # Memory efficiency
        most_memory_efficient = min(enhanced_results, key=lambda x: x.memory_per_1k_kb)
        print(
            f"💾 Most memory efficient: {most_memory_efficient.library} ({most_memory_efficient.memory_per_1k_kb:.1f}KB per 1000 instances)"
        )

    print("\nRECOMMENDations (based on Validation_Consolidation.md):")
    print("• Use msgspec for MCP server messages (85x faster than Pydantic)")
    print("• Use dataclasses for metrics and internal data (12x faster)")
    print("• Keep SQLModel only for database models with table=True")
    print("• Migrate Pydantic V1 validators to V2 field_validator syntax")

    # Save baseline results for future comparison
    baseline_file = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    baseline_data = {
        "timestamp": "2025-08-09T10:30:00Z",
        "results": [
            {
                "library": r.library,
                "time_per_op_us": r.time_per_op_us,
                "ops_per_sec": r.ops_per_sec,
                "memory_per_1k_kb": r.memory_per_1k_kb,
                "improvement_factor": r.improvement_factor,
            }
            for r in enhanced_results
        ],
    }

    try:
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"\n📁 Baseline results saved to: {baseline_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save baseline results: {e}")

    print("\n✅ Benchmark completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
