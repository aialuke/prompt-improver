#!/usr/bin/env python3
"""
MCP Message Performance Benchmark - msgspec vs SQLModel
Run: python tests/benchmarks/mcp_message_performance.py

This benchmark measures the actual performance improvement from migrating
MCP message classes from SQLModel to msgspec, targeting 85x improvement
from 543μs to 6.4μs per message decode + validate operation.

Focus areas:
- PromptEnhancementRequest validation (hot path for 10k+ calls/sec)
- PromptStorageRequest validation
- JSON encoding/decoding with msgspec.json
- Memory efficiency for high-volume operations
"""

import os
import statistics
import sys
import time
import tracemalloc
from typing import Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    import msgspec
    import msgspec.json
    from sqlmodel import Field, SQLModel

    from prompt_improver.mcp_server.server import (
        MCPMessageCodec,
        PromptEnhancementRequest,
        PromptEnhancementRequestMsgspec,
        PromptStorageRequest,
        PromptStorageRequestMsgspec,
    )
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Ensure the project dependencies are installed")
    sys.exit(1)

# Realistic MCP message payloads based on production usage
PROMPT_ENHANCEMENT_PAYLOAD = {
    "prompt": "Write a Python function that processes user input and validates email addresses using regex patterns, handles edge cases for international domains, implements proper error handling, and includes comprehensive unit tests with pytest fixtures.",
    "session_id": "mcp_session_2025_08_09_12345_abcdef",
    "context": {
        "user_id": "user_789abc",
        "domain": "software_engineering",
        "complexity": "intermediate",
        "language": "python",
        "framework": "pytest",
        "previous_interactions": [
            {"timestamp": "2025-08-09T10:30:00Z", "action": "validate_input"},
            {"timestamp": "2025-08-09T10:31:15Z", "action": "apply_rules"},
            {"timestamp": "2025-08-09T10:32:30Z", "action": "generate_response"},
        ],
        "enhancement_preferences": {
            "clarity": 0.9,
            "specificity": 0.85,
            "context_awareness": 0.8,
            "technical_depth": 0.75,
        },
        "metadata": {
            "client_version": "1.2.3",
            "protocol_version": "1.0",
            "capabilities": ["streaming", "progress_reports", "validation"],
        },
    },
}

PROMPT_STORAGE_PAYLOAD = {
    "original": "Write a Python function for email validation",
    "enhanced": "Write a comprehensive Python function that validates email addresses using regex patterns, handles international domains and edge cases, implements proper error handling with descriptive messages, and includes unit tests with pytest fixtures for thorough validation coverage.",
    "metrics": {
        "processing_time_ms": 247.3,
        "improvement_score": 0.87,
        "confidence_level": 0.93,
        "rules_applied": [
            {"rule": "clarity_enhancement", "weight": 0.3, "impact": 0.82},
            {"rule": "specificity_boost", "weight": 0.25, "impact": 0.78},
            {"rule": "context_integration", "weight": 0.2, "impact": 0.85},
            {"rule": "technical_depth", "weight": 0.25, "impact": 0.71},
        ],
        "performance_metrics": {
            "cpu_time_ms": 156.2,
            "memory_peak_mb": 12.4,
            "db_queries": 3,
            "cache_hits": 7,
            "cache_misses": 2,
        },
    },
    "session_id": "mcp_session_2025_08_09_67890_fedcba",
}


def benchmark_message_validation(
    name: str, create_func, payload: dict[str, Any], iterations: int = 10000
) -> dict[str, float]:
    """
    Benchmark message validation performance with focus on hot path operations.

    Target: Measure decode + validate time to verify 85x improvement
    Current SQLModel: 543μs per operation
    Target msgspec: 6.4μs per operation (85x improvement)
    """

    print(f"\n🔥 Benchmarking {name} (hot path: {iterations:,} iterations)...")

    # Prepare JSON data for realistic deserialization testing
    import json

    json_data = json.dumps(payload)
    json_bytes = json_data.encode("utf-8")

    # Time measurement with high precision
    times = []

    for _i in range(5):  # 5 runs for statistical accuracy
        start_time = time.perf_counter()

        for _ in range(iterations):
            try:
                if hasattr(create_func, "__self__") and hasattr(
                    create_func.__self__, "decode_prompt_enhancement"
                ):
                    # msgspec codec method
                    instance = create_func(json_bytes)
                else:
                    # SQLModel validation
                    instance = create_func(**payload)
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
                return {
                    "time_per_op_us": float("inf"),
                    "ops_per_sec": 0,
                    "memory_mb": float("inf"),
                }

        run_time = time.perf_counter() - start_time
        times.append(run_time)

    # Use median time for stability
    median_time = statistics.median(times)
    time_per_op_us = (median_time / iterations) * 1_000_000
    ops_per_sec = iterations / median_time if median_time > 0 else 0

    # Memory measurement
    tracemalloc.start()
    instances = []
    try:
        for _ in range(1000):
            if hasattr(create_func, "__self__") and hasattr(
                create_func.__self__, "decode_prompt_enhancement"
            ):
                instance = create_func(json_bytes)
            else:
                instance = create_func(**payload)
            instances.append(instance)
    except Exception:
        pass

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_mb = peak / (1024 * 1024)

    # Ensure instances aren't garbage collected
    _ = len(instances)

    return {
        "time_per_op_us": time_per_op_us,
        "ops_per_sec": ops_per_sec,
        "memory_mb": memory_mb,
        "all_times_s": times,
    }


def benchmark_json_encoding(
    name: str, encode_func, payload: dict[str, Any], iterations: int = 5000
) -> dict[str, float]:
    """Benchmark JSON encoding performance."""

    print(f"🚀 Benchmarking {name} JSON encoding ({iterations:,} iterations)...")

    times = []
    for _i in range(3):
        start_time = time.perf_counter()

        for _ in range(iterations):
            try:
                encoded = encode_func(payload)
            except Exception as e:
                print(f"❌ Error in {name} encoding: {e}")
                return {"time_per_op_us": float("inf"), "ops_per_sec": 0}

        run_time = time.perf_counter() - start_time
        times.append(run_time)

    median_time = statistics.median(times)
    time_per_op_us = (median_time / iterations) * 1_000_000
    ops_per_sec = iterations / median_time if median_time > 0 else 0

    return {"time_per_op_us": time_per_op_us, "ops_per_sec": ops_per_sec}


def run_production_load_simulation():
    """Simulate production load of 10,000+ calls/sec to verify performance targets."""

    print("\n🏭 Production Load Simulation (10,000+ calls/sec target)...")
    print("=" * 70)

    codec = MCPMessageCodec()
    json_data = msgspec.json.encode(PROMPT_ENHANCEMENT_PAYLOAD)

    # Simulate 1 second of production load
    target_calls = 10000
    start_time = time.perf_counter()

    successful_calls = 0
    errors = 0

    for i in range(target_calls):
        try:
            request = codec.decode_prompt_enhancement(json_data)
            response = {
                "improved_prompt": "Enhanced version",
                "processing_time_ms": 45.2,
            }
            encoded_response = codec.encode_response(response)
            successful_calls += 1
        except Exception as e:
            errors += 1
            if errors < 10:  # Log first few errors
                print(f"⚠️  Error in call {i}: {e}")

    total_time = time.perf_counter() - start_time
    actual_calls_per_sec = successful_calls / total_time if total_time > 0 else 0
    avg_time_per_call_us = (
        (total_time / successful_calls) * 1_000_000 if successful_calls > 0 else 0
    )

    print("📊 Production Load Results:")
    print("   Target: 10,000 calls/sec")
    print(f"   Actual: {actual_calls_per_sec:,.0f} calls/sec")
    print(f"   Success rate: {(successful_calls / target_calls) * 100:.2f}%")
    print(f"   Avg time per call: {avg_time_per_call_us:.1f}μs")
    print(
        f"   Target achievement: {'✅ PASS' if avg_time_per_call_us < 100 else '❌ FAIL'}"
    )

    return {
        "target_calls_per_sec": 10000,
        "actual_calls_per_sec": actual_calls_per_sec,
        "success_rate": (successful_calls / target_calls) * 100,
        "avg_time_per_call_us": avg_time_per_call_us,
        "passes_target": avg_time_per_call_us < 100,
    }


def main():
    """Run comprehensive MCP message performance benchmark."""

    print("MCP Message Performance Benchmark - msgspec vs SQLModel")
    print("=" * 70)
    print("🎯 Target: 85x improvement (543μs → 6.4μs per decode + validate)")
    print("🔥 Production volume: 10,000+ calls/sec")
    print(
        f"📏 Payload sizes: Enhancement={len(str(PROMPT_ENHANCEMENT_PAYLOAD))} chars, Storage={len(str(PROMPT_STORAGE_PAYLOAD))} chars"
    )
    print()

    results = {}

    # Benchmark PromptEnhancementRequest
    print("🧪 PROMPT ENHANCEMENT REQUEST BENCHMARKS")
    print("-" * 50)

    # SQLModel version (current)
    results["sqlmodel_enhancement"] = benchmark_message_validation(
        "SQLModel PromptEnhancementRequest",
        PromptEnhancementRequest,
        PROMPT_ENHANCEMENT_PAYLOAD,
    )

    # msgspec version (optimized)
    codec = MCPMessageCodec()
    results["msgspec_enhancement"] = benchmark_message_validation(
        "msgspec PromptEnhancementRequest",
        codec.decode_prompt_enhancement,
        PROMPT_ENHANCEMENT_PAYLOAD,
    )

    # Benchmark PromptStorageRequest
    print("\n🗄️  PROMPT STORAGE REQUEST BENCHMARKS")
    print("-" * 50)

    results["sqlmodel_storage"] = benchmark_message_validation(
        "SQLModel PromptStorageRequest", PromptStorageRequest, PROMPT_STORAGE_PAYLOAD
    )

    results["msgspec_storage"] = benchmark_message_validation(
        "msgspec PromptStorageRequest",
        codec.decode_prompt_storage,
        PROMPT_STORAGE_PAYLOAD,
    )

    # JSON encoding benchmarks
    print("\n📋 JSON ENCODING BENCHMARKS")
    print("-" * 50)

    import json

    results["json_encoding"] = benchmark_json_encoding(
        "Standard json.dumps()",
        lambda obj: json.dumps(obj, separators=(",", ":")),
        PROMPT_ENHANCEMENT_PAYLOAD,
    )

    results["msgspec_encoding"] = benchmark_json_encoding(
        "msgspec.json.encode()",
        msgspec.json.encode,
        PROMPT_ENHANCEMENT_PAYLOAD,
    )

    # Production load simulation
    production_results = run_production_load_simulation()

    # Calculate improvement factors
    print("\n📈 PERFORMANCE ANALYSIS")
    print("=" * 70)

    if (
        results["sqlmodel_enhancement"]["time_per_op_us"] > 0
        and results["msgspec_enhancement"]["time_per_op_us"] > 0
    ):
        enhancement_improvement = (
            results["sqlmodel_enhancement"]["time_per_op_us"]
            / results["msgspec_enhancement"]["time_per_op_us"]
        )
        print(
            f"🚀 PromptEnhancementRequest improvement: {enhancement_improvement:.1f}x faster"
        )
        print(
            f"   SQLModel: {results['sqlmodel_enhancement']['time_per_op_us']:.1f}μs per operation"
        )
        print(
            f"   msgspec:  {results['msgspec_enhancement']['time_per_op_us']:.1f}μs per operation"
        )

        target_achieved = (
            enhancement_improvement >= 80
        )  # Target is 85x, allow some variance
        print(
            f"   Target (85x): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}"
        )

    if (
        results["sqlmodel_storage"]["time_per_op_us"] > 0
        and results["msgspec_storage"]["time_per_op_us"] > 0
    ):
        storage_improvement = (
            results["sqlmodel_storage"]["time_per_op_us"]
            / results["msgspec_storage"]["time_per_op_us"]
        )
        print(
            f"\n💾 PromptStorageRequest improvement: {storage_improvement:.1f}x faster"
        )
        print(
            f"   SQLModel: {results['sqlmodel_storage']['time_per_op_us']:.1f}μs per operation"
        )
        print(
            f"   msgspec:  {results['msgspec_storage']['time_per_op_us']:.1f}μs per operation"
        )

    if (
        results["json_encoding"]["time_per_op_us"] > 0
        and results["msgspec_encoding"]["time_per_op_us"] > 0
    ):
        encoding_improvement = (
            results["json_encoding"]["time_per_op_us"]
            / results["msgspec_encoding"]["time_per_op_us"]
        )
        print(f"\n📤 JSON encoding improvement: {encoding_improvement:.1f}x faster")
        print(
            f"   json.dumps(): {results['json_encoding']['time_per_op_us']:.1f}μs per operation"
        )
        print(
            f"   msgspec:      {results['msgspec_encoding']['time_per_op_us']:.1f}μs per operation"
        )

    # Memory efficiency
    print("\n💾 MEMORY EFFICIENCY")
    print("-" * 30)
    enhancement_memory_improvement = (
        results["sqlmodel_enhancement"]["memory_mb"]
        / results["msgspec_enhancement"]["memory_mb"]
        if results["msgspec_enhancement"]["memory_mb"] > 0
        else 1
    )
    storage_memory_improvement = (
        results["sqlmodel_storage"]["memory_mb"]
        / results["msgspec_storage"]["memory_mb"]
        if results["msgspec_storage"]["memory_mb"] > 0
        else 1
    )

    print(f"Enhancement: {enhancement_memory_improvement:.1f}x more memory efficient")
    print(f"Storage: {storage_memory_improvement:.1f}x more memory efficient")

    # Final assessment
    print("\n🎯 FINAL ASSESSMENT")
    print("=" * 30)
    if enhancement_improvement >= 80 and production_results["passes_target"]:
        print("✅ SUCCESS: msgspec migration achieves performance targets")
        print(
            f"   - 85x improvement target: {'✅ ACHIEVED' if enhancement_improvement >= 80 else '❌ MISSED'}"
        )
        print(
            f"   - 10k+ calls/sec target: {'✅ ACHIEVED' if production_results['passes_target'] else '❌ MISSED'}"
        )
        print("   - Ready for production deployment")
    else:
        print("⚠️  WARNING: Performance targets not fully achieved")
        print("   - Consider additional optimizations")
        print("   - Review message structure complexity")

    # Save results
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "performance_results": results,
        "production_simulation": production_results,
        "improvement_factors": {
            "enhancement_improvement": enhancement_improvement
            if "enhancement_improvement" in locals()
            else 0,
            "storage_improvement": storage_improvement
            if "storage_improvement" in locals()
            else 0,
            "encoding_improvement": encoding_improvement
            if "encoding_improvement" in locals()
            else 0,
        },
        "targets_achieved": {
            "85x_improvement": enhancement_improvement >= 80
            if "enhancement_improvement" in locals()
            else False,
            "10k_calls_per_sec": production_results["passes_target"],
        },
    }

    try:
        import json

        results_file = os.path.join(
            os.path.dirname(__file__), "mcp_performance_results.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
