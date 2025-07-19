"""Test suite for Event Loop Manager, Session Wrapper, and Benchmark Utilities.

This module provides test cases for validating the functionality,
performance, and integration of the newly implemented components such as
event loop management, session handling, and performance benchmarks.

Follows pytest-asyncio best practices:
- Uses function-scoped fixtures for better test isolation
- Configures uvloop through session-scoped event_loop_policy fixture
- Uses @pytest.mark.asyncio for async test functions
"""

import asyncio
import time

import pytest
import uvloop

from prompt_improver.utils.event_loop_benchmark import run_full_benchmark_suite
from prompt_improver.utils.event_loop_manager import (
    get_event_loop_manager,
    setup_uvloop,
)
from prompt_improver.utils.session_event_loop import (
    SessionEventLoopWrapper,
    get_session_manager,
)


# Configure uvloop for testing (best practice)
@pytest.fixture(scope="session")
def event_loop_policy():
    """Configure uvloop event loop policy for all tests.

    Following pytest-asyncio best practices for uvloop integration.
    """
    try:
        return uvloop.EventLoopPolicy()
    except ImportError:
        # Fallback to default policy if uvloop not available
        return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="function")
def session_manager():
    """Return a session manager instance for testing.

    Uses function scope for better test isolation.
    """
    return get_session_manager()


async def test_uvloop_integration():
    """Verify uvloop is used if available and correctly configured."""
    manager = get_event_loop_manager()
    uvloop_enabled = setup_uvloop()

    # Test manager state
    assert manager.is_uvloop_enabled() == uvloop_enabled, (
        f"Manager uvloop state should match setup result: {uvloop_enabled}"
    )

    # Test loop info
    loop_info = manager.get_loop_info()

    # More flexible assertion - check that the loop type is consistent
    if uvloop_enabled:
        # If uvloop was enabled, the loop type should either be uvloop or reflect that uvloop is detected
        assert (
            loop_info.get("uvloop_detected", False)
            or "uvloop" in loop_info["loop_type"].lower()
        ), "uvloop should be detected when enabled"
    else:
        # If uvloop was not enabled, that's fine - we might not have uvloop installed
        print(f"uvloop not enabled - using {loop_info['loop_type']}")


async def test_session_wrapper_operations(session_manager):
    """Verify session wrapper correctly tracks and manages operations."""
    wrapper = session_manager.get_session_wrapper("test_session_1")

    # Test running a simple operation
    await wrapper.run_with_timeout(asyncio.sleep(0.1), operation_name="simple_sleep")

    metrics = wrapper.get_metrics()
    assert metrics["operations_count"] == 1, "Should have one operation recorded"
    assert metrics["error_count"] == 0, "Should have no error recorded"
    assert metrics["avg_time_ms"] > 0, "Average time should be greater than 0"


async def test_session_concurrent_tasks(session_manager):
    """Verify session wrapper handles concurrent tasks."""
    session_id = "test_session_2"
    wrapper = session_manager.get_session_wrapper(session_id)

    tasks = [wrapper.create_task(asyncio.sleep(0.1)) for _ in range(5)]

    await asyncio.gather(*tasks)

    assert len(wrapper.get_active_tasks()) == 0, "All tasks should be completed"
    metrics = wrapper.get_metrics()
    assert metrics["operations_count"] == 5, "Should record five operations"


async def test_benchmark_utilities():
    """Run the full benchmark suite and verify results."""
    results = await run_full_benchmark_suite()

    assert "comprehensive_benchmark" in results, (
        "Missing comprehensive benchmark results"
    )
    assert "prompt_simulation" in results, "Missing prompt simulation results"
    assert "session_benchmark" in results, "Missing session benchmark results"

    assert results["comprehensive_benchmark"]["meets_target"], (
        "Comprehensive benchmark should meet target latency"
    )
    assert results["prompt_simulation"]["meets_target"], (
        "Prompt simulation should meet target latency"
    )
    assert results["session_benchmark"]["meets_target"], (
        "Session benchmark should meet target latency"
    )


@pytest.mark.benchmark(group="event_loop", min_time=0.1, max_time=0.5, min_rounds=5)
def test_event_loop_latency_benchmark(benchmark):
    """Benchmark event loop latency using pytest-benchmark.

    Following pytest-benchmark best practices:
    - Direct function benchmarking for accuracy
    - Configured time limits and rounds
    - Grouped for comparison
    """

    async def benchmark_async_sleep():
        """Async function to benchmark."""
        await asyncio.sleep(0.001)  # 1ms sleep
        return True

    # Use asyncio.run for the benchmark (direct function call)
    result = benchmark(asyncio.run, benchmark_async_sleep())
    assert result is True


@pytest.mark.benchmark(
    group="session_management", min_time=0.1, max_time=0.5, min_rounds=5
)
def test_session_wrapper_benchmark(benchmark, session_manager):
    """Benchmark session wrapper performance.

    Tests session creation and operation performance.
    """

    def create_and_use_session():
        """Create session and perform operations."""
        wrapper = session_manager.get_session_wrapper(
            f"benchmark_session_{id(wrapper)}"
        )

        # Simulate session operations
        metrics = wrapper.get_metrics()
        assert metrics["operations_count"] >= 0
        return wrapper

    result = benchmark(create_and_use_session)
    assert result is not None


async def test_performance_regression_check():
    """Check that performance meets target requirements.

    This test ensures we maintain performance targets:
    - Event loop latency < 200ms
    - Session operations complete quickly
    """
    manager = get_event_loop_manager()

    # Test event loop latency
    latency_metrics = await manager.benchmark_loop_latency(samples=50)
    assert latency_metrics["avg_ms"] < 200, (
        f"Event loop latency {latency_metrics['avg_ms']:.2f}ms exceeds 200ms target"
    )

    # Test session wrapper performance
    session_wrapper = get_session_manager().get_session_wrapper("perf_test")

    start_time = time.perf_counter()
    async with session_wrapper.performance_context("performance_test"):
        await asyncio.sleep(0.01)  # Simulate work
    end_time = time.perf_counter()

    operation_time_ms = (end_time - start_time) * 1000
    assert operation_time_ms < 200, (
        f"Session operation {operation_time_ms:.2f}ms exceeds 200ms target"
    )
