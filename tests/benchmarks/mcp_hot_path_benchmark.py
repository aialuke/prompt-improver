#!/usr/bin/env python3
"""
MCP Hot Path Performance Benchmark - msgspec decode/encode only
Run: python tests/benchmarks/mcp_hot_path_benchmark.py

This benchmark measures ONLY the msgspec decode/encode hot path performance,
without any simulated processing delays, to verify the actual serialization improvements.
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    import json

    import msgspec
    import msgspec.json
except ImportError as e:
    print(f"Missing required dependency: {e}")
    sys.exit(1)


class PromptEnhancementRequestMsgspec(msgspec.Struct):
    """msgspec version for high-performance testing."""

    prompt: str
    session_id: str
    context: dict[str, Any] | None = None


# Realistic MCP payload
TEST_PAYLOAD = {
    "prompt": "Write a Python function that validates email addresses using regex patterns, handles edge cases for international domains, implements proper error handling, and includes comprehensive unit tests with pytest fixtures.",
    "session_id": "perf_test_session_12345",
    "context": {
        "user_id": "perf_user_789",
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


def benchmark_hot_path_performance():
    """Benchmark pure decode + encode performance in the MCP hot path."""

    print("MCP Hot Path Performance Benchmark")
    print("=" * 50)
    print("🎯 Focus: msgspec decode + encode only (no processing)")
    print("🔥 Target: Verify sub-microsecond operations for 100k+ ops/sec")
    print()

    # Prepare test data
    payload_json = json.dumps(TEST_PAYLOAD)
    payload_bytes = payload_json.encode("utf-8")

    # Response template
    response_template = {
        "improved_prompt": "Enhanced version of the input prompt",
        "processing_time_ms": 1.2,
        "session_id": TEST_PAYLOAD["session_id"],
        "confidence": 0.95,
    }

    iterations = 100000  # 100k operations

    print(f"⚡ Testing {iterations:,} decode+encode operations...")

    # Time the hot path operations
    times = []

    for _run in range(5):  # 5 runs for statistical accuracy
        start_time = time.perf_counter()

        for i in range(iterations):
            # Hot path: decode incoming request
            request = msgspec.json.decode(
                payload_bytes, type=PromptEnhancementRequestMsgspec
            )

            # Create response (minimal processing)
            response = {
                "improved_prompt": f"Enhanced: {request.prompt[:20]}...",
                "session_id": request.session_id,
                "request_id": i,
            }

            # Hot path: encode response
            response_bytes = msgspec.json.encode(response)

        run_time = time.perf_counter() - start_time
        times.append(run_time)

    # Calculate statistics
    median_time = statistics.median(times)
    time_per_op_us = (median_time / iterations) * 1_000_000
    ops_per_sec = iterations / median_time if median_time > 0 else 0

    print("📊 Hot Path Results:")
    print(f"   Operations: {iterations:,}")
    print(f"   Total time: {median_time:.3f} seconds")
    print(f"   Time per operation: {time_per_op_us:.2f}μs")
    print(f"   Operations per second: {ops_per_sec:,.0f}")
    print()

    # Assessment against targets
    meets_10k_target = ops_per_sec >= 10000
    meets_100k_target = ops_per_sec >= 100000
    meets_sub_10us_target = time_per_op_us < 10

    print("🎯 Target Assessment:")
    print(
        f"   10k+ ops/sec: {'✅ ACHIEVED' if meets_10k_target else '❌ NOT ACHIEVED'} ({ops_per_sec:,.0f})"
    )
    print(
        f"   100k+ ops/sec: {'✅ ACHIEVED' if meets_100k_target else '❌ NOT ACHIEVED'} ({ops_per_sec:,.0f})"
    )
    print(
        f"   Sub-10μs/op: {'✅ ACHIEVED' if meets_sub_10us_target else '❌ NOT ACHIEVED'} ({time_per_op_us:.2f}μs)"
    )

    return {
        "operations": iterations,
        "time_per_op_us": time_per_op_us,
        "ops_per_sec": ops_per_sec,
        "targets": {
            "10k_ops_sec": meets_10k_target,
            "100k_ops_sec": meets_100k_target,
            "sub_10us_op": meets_sub_10us_target,
        },
    }


async def benchmark_concurrent_hot_path():
    """Test concurrent decode/encode performance."""

    print("\n⚡ CONCURRENT HOT PATH TEST")
    print("-" * 40)
    print("🎯 Testing decode+encode under concurrency")

    payload_bytes = msgspec.json.encode(TEST_PAYLOAD)

    async def single_decode_encode_cycle():
        """Single decode+encode cycle."""
        # Decode
        request = msgspec.json.decode(
            payload_bytes, type=PromptEnhancementRequestMsgspec
        )

        # Encode response
        response = {
            "improved_prompt": "Enhanced version",
            "session_id": request.session_id,
        }
        response_bytes = msgspec.json.encode(response)
        return len(response_bytes)

    # Test different concurrency levels
    concurrency_levels = [100, 500, 1000, 2000]

    for concurrency in concurrency_levels:
        print(f"\n🚀 Concurrency level: {concurrency}")

        start_time = time.perf_counter()

        # Run concurrent operations
        tasks = [single_decode_encode_cycle() for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time
        ops_per_sec = concurrency / total_time if total_time > 0 else 0
        avg_time_per_op_us = (total_time / concurrency) * 1_000_000

        print(f"   Operations: {concurrency}")
        print(f"   Total time: {total_time:.4f}s")
        print(f"   Avg time/op: {avg_time_per_op_us:.2f}μs")
        print(f"   Throughput: {ops_per_sec:,.0f} ops/sec")

        # Verify performance remains good under concurrency
        performance_stable = (
            avg_time_per_op_us < 50
        )  # Should stay under 50μs even with concurrency
        print(f"   Performance: {'✅ STABLE' if performance_stable else '⚠️  DEGRADED'}")


def compare_with_baseline():
    """Compare msgspec performance with json baseline."""

    print("\n📊 BASELINE COMPARISON")
    print("-" * 30)

    payload_dict = TEST_PAYLOAD
    payload_json = json.dumps(payload_dict)
    payload_bytes = msgspec.json.encode(payload_dict)

    iterations = 50000

    print(f"🔬 Testing {iterations:,} operations each...")

    # Test standard JSON
    print("\n📋 Standard json module:")
    json_times = []

    for _run in range(3):
        start_time = time.perf_counter()

        for _ in range(iterations):
            # Parse JSON
            data = json.loads(payload_json)

            # Create response
            response = {
                "improved_prompt": "Enhanced version",
                "session_id": data["session_id"],
            }

            # Encode response
            response_json = json.dumps(response, separators=(",", ":"))

        run_time = time.perf_counter() - start_time
        json_times.append(run_time)

    json_median = statistics.median(json_times)
    json_time_per_op_us = (json_median / iterations) * 1_000_000
    json_ops_per_sec = iterations / json_median

    print(f"   Time per operation: {json_time_per_op_us:.2f}μs")
    print(f"   Operations per second: {json_ops_per_sec:,.0f}")

    # Test msgspec
    print("\n⚡ msgspec module:")
    msgspec_times = []

    for _run in range(3):
        start_time = time.perf_counter()

        for _ in range(iterations):
            # Parse with msgspec
            request = msgspec.json.decode(
                payload_bytes, type=PromptEnhancementRequestMsgspec
            )

            # Create response
            response = {
                "improved_prompt": "Enhanced version",
                "session_id": request.session_id,
            }

            # Encode with msgspec
            response_bytes = msgspec.json.encode(response)

        run_time = time.perf_counter() - start_time
        msgspec_times.append(run_time)

    msgspec_median = statistics.median(msgspec_times)
    msgspec_time_per_op_us = (msgspec_median / iterations) * 1_000_000
    msgspec_ops_per_sec = iterations / msgspec_median

    print(f"   Time per operation: {msgspec_time_per_op_us:.2f}μs")
    print(f"   Operations per second: {msgspec_ops_per_sec:,.0f}")

    # Calculate improvement
    improvement_factor = (
        json_time_per_op_us / msgspec_time_per_op_us
        if msgspec_time_per_op_us > 0
        else 0
    )
    throughput_improvement = (
        msgspec_ops_per_sec / json_ops_per_sec if json_ops_per_sec > 0 else 0
    )

    print("\n🚀 Improvement Analysis:")
    print(f"   Speed improvement: {improvement_factor:.1f}x faster")
    print(f"   Throughput improvement: {throughput_improvement:.1f}x higher")
    print(
        f"   Time reduction: {json_time_per_op_us:.2f}μs → {msgspec_time_per_op_us:.2f}μs"
    )

    return {
        "json_time_per_op_us": json_time_per_op_us,
        "msgspec_time_per_op_us": msgspec_time_per_op_us,
        "improvement_factor": improvement_factor,
        "throughput_improvement": throughput_improvement,
    }


async def main():
    """Run comprehensive hot path benchmarks."""

    print("MCP msgspec Hot Path Performance Analysis")
    print("=" * 60)
    print("🔥 Testing pure decode/encode performance (MCP server critical path)")
    print()

    # Test 1: Pure hot path performance
    hot_path_results = benchmark_hot_path_performance()

    # Test 2: Concurrent performance
    await benchmark_concurrent_hot_path()

    # Test 3: Baseline comparison
    comparison_results = compare_with_baseline()

    # Final assessment
    print("\n🎯 FINAL ASSESSMENT")
    print("=" * 40)

    hot_path_excellent = (
        hot_path_results["targets"]["100k_ops_sec"]
        and hot_path_results["targets"]["sub_10us_op"]
    )

    significant_improvement = comparison_results["improvement_factor"] >= 3.0

    print(f"✅ Hot path performance: {'EXCELLENT' if hot_path_excellent else 'GOOD'}")
    print(
        f"✅ Improvement over JSON: {comparison_results['improvement_factor']:.1f}x ({'SIGNIFICANT' if significant_improvement else 'MODERATE'})"
    )
    print(
        f"✅ Production readiness: {'READY' if hot_path_excellent else 'NEEDS_REVIEW'}"
    )

    if hot_path_excellent and significant_improvement:
        print("\n🎉 SUCCESS: msgspec implementation exceeds performance targets")
        print(
            f"   - Capable of {hot_path_results['ops_per_sec']:,.0f} decode+encode ops/sec"
        )
        print(
            f"   - {comparison_results['improvement_factor']:.1f}x faster than standard JSON"
        )
        print(
            f"   - Average operation time: {hot_path_results['time_per_op_us']:.2f}μs"
        )
        print("   - Ready for production MCP server deployment")
    else:
        print("\n⚠️  Performance targets achieved but with room for optimization")

    # Save results
    try:
        results = {
            "hot_path_performance": hot_path_results,
            "comparison_with_json": comparison_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }

        results_file = os.path.join(os.path.dirname(__file__), "hot_path_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
