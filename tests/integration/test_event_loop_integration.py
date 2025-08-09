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
from prompt_improver.utils.unified_loop_manager import get_unified_loop_manager

@pytest.fixture(scope='session')
def event_loop_policy():
    """Configure uvloop event loop policy for all tests.

    Following pytest-asyncio best practices for uvloop integration.
    """
    try:
        return uvloop.EventLoopPolicy()
    except ImportError:
        return asyncio.DefaultEventLoopPolicy()

@pytest.fixture(scope='function')
def unified_manager():
    """Return a unified loop manager instance for testing.

    Uses function scope for better test isolation.
    """
    return get_unified_loop_manager()

async def test_uvloop_integration():
    """Verify uvloop is used if available and correctly configured."""
    manager = get_unified_loop_manager()
    uvloop_enabled = manager.setup_uvloop()
    assert manager.is_uvloop_enabled() == uvloop_enabled, f'Manager uvloop state should match setup result: {uvloop_enabled}'
    loop_info = manager.get_loop_info()
    if uvloop_enabled:
        assert loop_info.get('uvloop_detected', False) or 'uvloop' in loop_info['loop_type'].lower(), 'uvloop should be detected when enabled'
    else:
        print(f"uvloop not enabled - using {loop_info['loop_type']}")

async def test_session_wrapper_operations(unified_manager):
    """Verify session wrapper correctly tracks and manages operations."""
    await unified_manager.run_with_session_tracking('test_session_1', asyncio.sleep(0.1), timeout=5.0)
    session_info = unified_manager.get_session_wrapper('test_session_1')
    metrics = session_info['metrics']
    assert metrics['operations_count'] == 1, 'Should have one operation recorded'
    assert metrics['error_count'] == 0, 'Should have no error recorded'
    assert metrics['avg_time_ms'] > 0, 'Average time should be greater than 0'

async def test_session_concurrent_tasks(unified_manager):
    """Verify session wrapper handles concurrent tasks."""
    session_id = 'test_session_2'
    tasks = [unified_manager.run_with_session_tracking(session_id, asyncio.sleep(0.1), timeout=5.0) for _ in range(5)]
    await asyncio.gather(*tasks)
    session_info = unified_manager.get_session_wrapper(session_id)
    metrics = session_info['metrics']
    assert metrics['operations_count'] == 5, 'Should record five operations'

async def test_benchmark_utilities():
    """Run the unified benchmark suite and verify results."""
    manager = get_unified_loop_manager()
    results = await manager.benchmark_unified_performance(latency_samples=50, throughput_tasks=100, session_count=3)
    assert 'loop_info' in results, 'Missing loop info'
    assert 'global_latency' in results, 'Missing global latency results'
    assert 'global_throughput' in results, 'Missing global throughput results'
    assert 'session_benchmarks' in results, 'Missing session benchmark results'
    assert results['global_latency']['avg_ms'] < 200, 'Global latency should meet target'
    assert results['global_throughput']['throughput_per_second'] > 10, 'Throughput should be reasonable'

@pytest.mark.benchmark(group='event_loop', min_time=0.1, max_time=0.5, min_rounds=5)
def test_event_loop_latency_benchmark(benchmark):
    """Benchmark event loop latency using pytest-benchmark.

    Following pytest-benchmark best practices:
    - Direct function benchmarking for accuracy
    - Configured time limits and rounds
    - Grouped for comparison
    """

    async def benchmark_async_sleep():
        """Async function to benchmark."""
        await asyncio.sleep(0.001)
        return True
    result = benchmark(asyncio.run, benchmark_async_sleep())
    assert result is True

@pytest.mark.benchmark(group='session_management', min_time=0.1, max_time=0.5, min_rounds=5)
def test_session_wrapper_benchmark(benchmark, unified_manager):
    """Benchmark session wrapper performance.

    Tests session creation and operation performance.
    """

    def create_and_use_session():
        """Create session and perform operations."""
        session_id = f'benchmark_session_{time.time()}'
        session_info = unified_manager.get_session_wrapper(session_id)
        metrics = session_info['metrics']
        assert metrics['operations_count'] >= 0
        return session_info
    result = benchmark(create_and_use_session)
    assert result is not None

async def test_performance_regression_check():
    """Check that performance meets target requirements.

    This test ensures we maintain performance targets:
    - Event loop latency < 200ms
    - Session operations complete quickly
    """
    manager = get_unified_loop_manager()
    latency_metrics = await manager.benchmark_loop_latency(samples=50)
    assert latency_metrics['avg_ms'] < 200, f"Event loop latency {latency_metrics['avg_ms']:.2f}ms exceeds 200ms target"
    session_id = 'perf_test'
    start_time = time.perf_counter()
    async with manager.session_context(session_id):
        await asyncio.sleep(0.01)
    end_time = time.perf_counter()
    operation_time_ms = (end_time - start_time) * 1000
    assert operation_time_ms < 200, f'Session operation {operation_time_ms:.2f}ms exceeds 200ms target'
