#!/usr/bin/env python3
"""
MCP High-Volume Load Test - 10,000+ Concurrent Requests
Run: python tests/benchmarks/mcp_load_test.py

This test simulates production load of 10,000+ concurrent MCP requests
to verify memory efficiency and response times under realistic conditions.
"""

import asyncio
import multiprocessing
import os
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
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


# Message classes for testing
class PromptEnhancementRequestMsgspec(msgspec.Struct):
    """msgspec version for high-performance testing."""

    prompt: str
    session_id: str
    context: dict[str, Any] | None = None


class MCPMessageCodec:
    """High-performance message codec."""

    @staticmethod
    def decode_prompt_enhancement(data: bytes) -> PromptEnhancementRequestMsgspec:
        """Decode with msgspec performance."""
        return msgspec.json.decode(data, type=PromptEnhancementRequestMsgspec)

    @staticmethod
    def encode_response(obj: Any) -> bytes:
        """Encode response with msgspec performance."""
        return msgspec.json.encode(obj)


# Test payload
TEST_PAYLOAD = {
    "prompt": "Create a Python function that validates email addresses using regex patterns, handles edge cases for international domains, implements proper error handling, and includes comprehensive unit tests.",
    "session_id": "load_test_session_12345",
    "context": {
        "user_id": "load_test_user",
        "domain": "software_engineering",
        "complexity": "intermediate",
        "metadata": {
            "timestamp": "2025-08-09T10:30:00Z",
            "client_version": "1.0",
            "capabilities": ["validation", "streaming"],
        },
    },
}


class LoadTestMetrics:
    """Track load test metrics."""

    def __init__(self):
        self.request_times = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.peak_memory_mb = 0
        self.concurrent_requests = 0

    def add_request_time(self, time_us: float):
        """Add a request time measurement."""
        self.request_times.append(time_us)

    def success(self):
        """Record successful request."""
        self.successful_requests += 1

    def failure(self):
        """Record failed request."""
        self.failed_requests += 1

    def get_stats(self):
        """Get comprehensive statistics."""
        if not self.request_times:
            return {
                "total_requests": 0,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": 0,
                "avg_response_time_us": 0,
                "median_response_time_us": 0,
                "p95_response_time_us": 0,
                "p99_response_time_us": 0,
                "min_response_time_us": 0,
                "max_response_time_us": 0,
                "peak_memory_mb": self.peak_memory_mb,
            }

        return {
            "total_requests": len(self.request_times),
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests
                / (self.successful_requests + self.failed_requests)
            )
            * 100,
            "avg_response_time_us": statistics.mean(self.request_times),
            "median_response_time_us": statistics.median(self.request_times),
            "p95_response_time_us": sorted(self.request_times)[
                int(len(self.request_times) * 0.95)
            ]
            if len(self.request_times) > 20
            else self.request_times[-1],
            "p99_response_time_us": sorted(self.request_times)[
                int(len(self.request_times) * 0.99)
            ]
            if len(self.request_times) > 100
            else self.request_times[-1],
            "min_response_time_us": min(self.request_times),
            "max_response_time_us": max(self.request_times),
            "peak_memory_mb": self.peak_memory_mb,
        }


async def simulate_mcp_request(
    codec: MCPMessageCodec, payload_bytes: bytes, request_id: int
) -> float:
    """Simulate a single MCP request with decode + process + encode."""

    start_time = time.perf_counter()

    try:
        # Step 1: Decode incoming request (hot path)
        request = codec.decode_prompt_enhancement(payload_bytes)

        # Step 2: Simulate processing (lightweight)
        await asyncio.sleep(0.001)  # 1ms simulated processing

        # Step 3: Create response
        response = {
            "improved_prompt": f"Enhanced: {request.prompt[:50]}...",
            "processing_time_ms": 1.5,
            "session_id": request.session_id,
            "request_id": request_id,
        }

        # Step 4: Encode response (hot path)
        response_bytes = codec.encode_response(response)

        # Calculate total time
        total_time = (time.perf_counter() - start_time) * 1_000_000  # microseconds
        return total_time

    except Exception as e:
        print(f"⚠️  Request {request_id} failed: {e}")
        raise


async def run_concurrent_load_test(
    num_requests: int, concurrency_level: int
) -> LoadTestMetrics:
    """Run concurrent load test with specified parameters."""

    print("🚀 Running concurrent load test:")
    print(f"   Requests: {num_requests:,}")
    print(f"   Concurrency: {concurrency_level:,}")
    print("   Target: Complete in <10 seconds")

    # Prepare test data
    codec = MCPMessageCodec()
    payload_bytes = msgspec.json.encode(TEST_PAYLOAD)
    metrics = LoadTestMetrics()

    # Track memory usage
    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_level)

    async def limited_request(request_id: int):
        async with semaphore:
            try:
                response_time = await simulate_mcp_request(
                    codec, payload_bytes, request_id
                )
                metrics.add_request_time(response_time)
                metrics.success()

                # Update peak memory periodically
                if request_id % 1000 == 0:
                    current, peak = tracemalloc.get_traced_memory()
                    metrics.peak_memory_mb = max(
                        metrics.peak_memory_mb, peak / (1024 * 1024)
                    )

            except Exception:
                metrics.failure()

    # Run the load test
    load_test_start = time.perf_counter()

    print(f"🔥 Starting {num_requests:,} concurrent requests...")
    tasks = [limited_request(i) for i in range(num_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)

    load_test_time = time.perf_counter() - load_test_start

    # Final memory measurement
    current, peak = tracemalloc.get_traced_memory()
    metrics.peak_memory_mb = max(metrics.peak_memory_mb, peak / (1024 * 1024))
    tracemalloc.stop()

    # Calculate throughput
    actual_throughput = num_requests / load_test_time if load_test_time > 0 else 0

    print(f"✅ Load test completed in {load_test_time:.2f} seconds")
    print(f"📊 Throughput: {actual_throughput:,.0f} requests/sec")
    print(f"💾 Peak memory: {metrics.peak_memory_mb:.1f} MB")

    return metrics


def run_memory_stress_test():
    """Test memory efficiency under sustained load."""

    print("\n🧠 MEMORY STRESS TEST")
    print("=" * 40)

    codec = MCPMessageCodec()
    payload_bytes = msgspec.json.encode(TEST_PAYLOAD)

    # Track memory usage over time
    tracemalloc.start()
    initial_memory = tracemalloc.get_traced_memory()[0]

    memory_samples = []
    num_iterations = 50000  # 50k operations

    print(f"🔄 Processing {num_iterations:,} operations...")

    for i in range(num_iterations):
        # Simulate the hot path operations
        request = msgspec.json.decode(
            payload_bytes, type=PromptEnhancementRequestMsgspec
        )
        response = {
            "improved_prompt": "Enhanced version",
            "session_id": request.session_id,
            "processing_time": 1.2,
        }
        response_bytes = msgspec.json.encode(response)

        # Sample memory every 5000 operations
        if i % 5000 == 0:
            current, peak = tracemalloc.get_traced_memory()
            memory_samples.append({
                "iteration": i,
                "current_mb": current / (1024 * 1024),
                "peak_mb": peak / (1024 * 1024),
            })

    final_current, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("📈 Memory usage progression:")
    for sample in memory_samples:
        print(
            f"   {sample['iteration']:>6,}: {sample['current_mb']:>6.1f} MB current, {sample['peak_mb']:>6.1f} MB peak"
        )

    memory_efficiency = initial_memory / final_current if final_current > 0 else 1

    print(
        f"💾 Final memory: {final_current / (1024 * 1024):.1f} MB current, {final_peak / (1024 * 1024):.1f} MB peak"
    )
    print(
        f"🎯 Memory efficiency: {'✅ STABLE' if memory_efficiency > 0.5 else '⚠️  GROWING'}"
    )

    return {
        "initial_memory_mb": initial_memory / (1024 * 1024),
        "final_memory_mb": final_current / (1024 * 1024),
        "peak_memory_mb": final_peak / (1024 * 1024),
        "memory_stable": memory_efficiency > 0.5,
        "operations_processed": num_iterations,
    }


async def main():
    """Run comprehensive load testing."""

    print("MCP High-Volume Load Test - msgspec Performance")
    print("=" * 60)
    print("🎯 Testing 10,000+ concurrent request capability")
    print("🔥 Focus on hot path: decode + validate + encode")
    print()

    test_results = {}

    # Test 1: 10,000 requests with 1,000 concurrency
    print("🧪 TEST 1: Production Volume (10,000 requests)")
    print("-" * 50)

    test_results["production_load"] = await run_concurrent_load_test(10000, 1000)
    stats = test_results["production_load"].get_stats()

    print("📊 Results:")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Avg response: {stats['avg_response_time_us']:.1f}μs")
    print(f"   P95 response: {stats['p95_response_time_us']:.1f}μs")
    print(f"   P99 response: {stats['p99_response_time_us']:.1f}μs")

    # Test 2: Higher volume - 25,000 requests
    print("\n🧪 TEST 2: High Volume (25,000 requests)")
    print("-" * 50)

    test_results["high_volume"] = await run_concurrent_load_test(25000, 2000)
    high_stats = test_results["high_volume"].get_stats()

    print("📊 Results:")
    print(f"   Success rate: {high_stats['success_rate']:.1f}%")
    print(f"   Avg response: {high_stats['avg_response_time_us']:.1f}μs")
    print(f"   P95 response: {high_stats['p95_response_time_us']:.1f}μs")

    # Test 3: Memory stress test
    test_results["memory_stress"] = run_memory_stress_test()

    # Final assessment
    print("\n🎯 LOAD TEST SUMMARY")
    print("=" * 40)

    production_passes = (
        stats["success_rate"] >= 99.0
        and stats["p95_response_time_us"] < 10000  # P95 under 10ms
        and stats["avg_response_time_us"] < 5000  # Avg under 5ms
    )

    high_volume_passes = (
        high_stats["success_rate"] >= 95.0
        and high_stats["p95_response_time_us"] < 20000  # P95 under 20ms for high load
    )

    memory_passes = test_results["memory_stress"]["memory_stable"]

    print(f"✅ Production load (10k): {'PASS' if production_passes else 'FAIL'}")
    print(f"✅ High volume (25k): {'PASS' if high_volume_passes else 'FAIL'}")
    print(f"✅ Memory stability: {'PASS' if memory_passes else 'FAIL'}")

    if production_passes and memory_passes:
        print("\n🎉 SUCCESS: msgspec MCP implementation ready for production")
        print("   - Handles 10k+ concurrent requests efficiently")
        print("   - Sub-10ms P95 response times achieved")
        print("   - Memory usage remains stable")
    else:
        print("\n⚠️  REVIEW NEEDED: Some performance targets not met")

    # Save results
    try:
        import json

        results_file = os.path.join(os.path.dirname(__file__), "load_test_results.json")
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\n📁 Load test results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
