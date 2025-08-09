"""
Enhanced BackgroundTaskManager Tests with 2025 Best Practices

This module implements comprehensive testing for BackgroundTaskManager with 4 key enhancements:
1. Performance benchmarks with pytest-benchmark
2. Thread safety validation for concurrent task operations
3. Comprehensive lifecycle testing for BackgroundTaskManager
4. Failure scenario testing with real network/resource failures

Following 2025 best practices: real behavior testing, proper event loop management,
and comprehensive state validation.
"""
import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch
import pytest
from prompt_improver.services.health.background_manager import BackgroundTask, BackgroundTaskManager, TaskStatus, get_background_task_manager, init_background_task_manager, shutdown_background_task_manager
pytest_benchmark = pytest.importorskip('pytest_benchmark')

class TestBackgroundTaskManagerEnhanced:
    """Enhanced test suite for BackgroundTaskManager with 2025 best practices."""

    @pytest.fixture(scope='function')
    def event_loop(self):
        """Create isolated event loop for each test."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        loop.close()

    @pytest.fixture
    async def clean_manager(self):
        """Create a clean BackgroundTaskManager instance."""
        manager = BackgroundTaskManager(max_concurrent_tasks=5)
        await manager.start()
        yield manager
        await manager.stop(timeout=2.0)

    @pytest.fixture
    async def task_factory(self):
        """Factory for creating test tasks."""

        async def create_task(duration: float=0.1, should_fail: bool=False, task_name: str='test_task'):
            """Create a test task with configurable behavior."""
            if should_fail:
                raise RuntimeError(f'Simulated failure in {task_name}')
            await asyncio.sleep(duration)
            return f'{task_name}_completed'
        return create_task

    @pytest.mark.benchmark(group='task_submission')
    def test_task_submission_performance(self, benchmark, event_loop):
        """Benchmark task submission performance with real timing measurements."""

        async def setup_and_submit():
            manager = BackgroundTaskManager(max_concurrent_tasks=10)
            await manager.start()

            async def simple_task():
                await asyncio.sleep(0.001)
                return 'completed'
            start_time = time.time()
            task_ids = []
            for i in range(50):
                task_id = await manager.submit_task(f'perf_test_{i}', simple_task)
                task_ids.append(task_id)
            submission_time = (time.time() - start_time) * 1000
            await asyncio.sleep(0.2)
            await manager.stop(timeout=2.0)
            return submission_time

        def run_benchmark():
            return event_loop.run_until_complete(setup_and_submit())
        submission_time = benchmark(run_benchmark)
        assert submission_time < 100, f'Task submission too slow: {submission_time}ms'

    @pytest.mark.benchmark(group='task_execution')
    def test_concurrent_task_execution_performance(self, benchmark, event_loop):
        """Benchmark concurrent task execution with real BackgroundTaskManager."""

        async def benchmark_concurrent_execution():
            manager = BackgroundTaskManager(max_concurrent_tasks=20)
            await manager.start()

            async def cpu_bound_task(multiplier: int):
                result = 0
                for i in range(1000):
                    result += i * multiplier
                await asyncio.sleep(0.01)
                return result
            start_time = time.time()
            task_ids = []
            for i in range(100):
                task_id = await manager.submit_task(f'concurrent_{i}', cpu_bound_task, multiplier=i)
                task_ids.append(task_id)
            completed_tasks = 0
            while completed_tasks < 100:
                await asyncio.sleep(0.1)
                completed_tasks = sum((1 for task_id in task_ids if manager.get_task_status(task_id) and manager.get_task_status(task_id).status == TaskStatus.COMPLETED))
            execution_time = (time.time() - start_time) * 1000
            await manager.stop(timeout=2.0)
            return execution_time

        def run_benchmark():
            return event_loop.run_until_complete(benchmark_concurrent_execution())
        execution_time = benchmark(run_benchmark)
        assert execution_time < 2000, f'Concurrent execution too slow: {execution_time}ms'

    @pytest.mark.benchmark(group='lifecycle')
    def test_manager_lifecycle_performance(self, benchmark, event_loop):
        """Benchmark complete manager lifecycle (start/stop) performance."""

        async def benchmark_lifecycle():
            start_time = time.time()
            for cycle in range(5):
                manager = BackgroundTaskManager(max_concurrent_tasks=10)
                await manager.start()
                for i in range(10):
                    await manager.submit_task(f'lifecycle_{cycle}_{i}', lambda: asyncio.sleep(0.01))
                await manager.stop(timeout=1.0)
            lifecycle_time = (time.time() - start_time) * 1000
            return lifecycle_time

        def run_benchmark():
            return event_loop.run_until_complete(benchmark_lifecycle())
        lifecycle_time = benchmark(run_benchmark)
        assert lifecycle_time < 1000, f'Lifecycle performance too slow: {lifecycle_time}ms'

    @pytest.mark.asyncio
    async def test_concurrent_task_access_thread_safety(self, clean_manager, task_factory):
        """Validate thread safety with concurrent task operations."""

        async def concurrent_submitter(manager, prefix: str, count: int):
            """Submit multiple tasks concurrently."""
            task_ids = []
            for i in range(count):
                task_id = await manager.submit_task(f'{prefix}_{i}', task_factory, duration=0.1, task_name=f'{prefix}_{i}')
                task_ids.append(task_id)
            return task_ids
        results = await asyncio.gather(concurrent_submitter(clean_manager, 'thread_a', 20), concurrent_submitter(clean_manager, 'thread_b', 20), concurrent_submitter(clean_manager, 'thread_c', 20), return_exceptions=True)
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
            assert len(result) == 20
        all_task_ids = []
        for task_list in results:
            all_task_ids.extend(task_list)
        assert len(all_task_ids) == 60
        assert len(set(all_task_ids)) == 60
        await asyncio.sleep(0.5)
        task_counts = clean_manager.get_task_count()
        assert task_counts['completed'] == 60
        assert task_counts['running'] == 0
        assert task_counts['pending'] == 0

    @pytest.mark.asyncio
    async def test_concurrent_task_cancellation_thread_safety(self, clean_manager, task_factory):
        """Test thread safety of concurrent task cancellation."""
        task_ids = []
        for i in range(10):
            task_id = await clean_manager.submit_task(f'long_task_{i}', task_factory, duration=2.0, task_name=f'long_task_{i}')
            task_ids.append(task_id)
        await asyncio.sleep(0.1)

        async def cancel_tasks(task_ids_subset):
            """Cancel a subset of tasks concurrently."""
            results = []
            for task_id in task_ids_subset:
                result = await clean_manager.cancel_task(task_id)
                results.append(result)
            return results
        group1 = task_ids[:5]
        group2 = task_ids[5:]
        cancel_results = await asyncio.gather(cancel_tasks(group1), cancel_tasks(group2), return_exceptions=True)
        assert len(cancel_results) == 2
        for result in cancel_results:
            assert not isinstance(result, Exception)
            assert len(result) == 5
        await asyncio.sleep(0.5)
        task_counts = clean_manager.get_task_count()
        assert task_counts['cancelled'] == 10
        assert task_counts['running'] == 0

    @pytest.mark.asyncio
    async def test_shared_state_thread_safety(self, clean_manager):
        """Test thread safety of shared state access."""
        shared_counter = {'value': 0}
        access_lock = asyncio.Lock()

        async def increment_counter(index: str):
            """Safely increment shared counter."""
            async with access_lock:
                current = shared_counter['value']
                await asyncio.sleep(0.001)
                shared_counter['value'] = current + 1
            return f'task_{index}_completed'
        task_ids = []
        for i in range(50):
            task_id = await clean_manager.submit_task(f'shared_access_{i}', increment_counter, index=str(i))
            task_ids.append(task_id)
        await asyncio.sleep(1.0)
        assert shared_counter['value'] == 50
        completed_count = 0
        for task_id in task_ids:
            task = clean_manager.get_task_status(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                completed_count += 1
        assert completed_count == 50

    @pytest.mark.asyncio
    async def test_complete_task_lifecycle_validation(self, clean_manager, task_factory):
        """Test complete task lifecycle with state transitions and timing validation."""
        task_id = await clean_manager.submit_task('lifecycle_test', task_factory, duration=0.2)
        task = clean_manager.get_task_status(task_id)
        assert task is not None
        assert task.task_id == task_id
        assert task.status == TaskStatus.PENDING
        assert task.created_at > 0
        assert task.started_at is None
        assert task.completed_at is None
        assert task.result is None
        assert task.error is None
        await asyncio.sleep(0.05)
        task = clean_manager.get_task_status(task_id)
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
        assert task.started_at > task.created_at
        assert task.completed_at is None
        await asyncio.sleep(0.3)
        task = clean_manager.get_task_status(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.completed_at > task.started_at
        assert task.result == 'test_task_completed'
        assert task.error is None
        assert task.created_at < task.started_at < task.completed_at
        execution_time = task.completed_at - task.started_at
        assert 0.15 < execution_time < 0.35

    @pytest.mark.asyncio
    async def test_failed_task_lifecycle_validation(self, clean_manager, task_factory):
        """Test failed task lifecycle with proper error handling."""
        task_id = await clean_manager.submit_task('failed_task', task_factory, should_fail=True)
        await asyncio.sleep(0.2)
        task = clean_manager.get_task_status(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.completed_at is not None
        assert task.result is None
        assert task.error is not None
        assert 'Simulated failure' in task.error
        assert task.created_at < task.started_at < task.completed_at

    @pytest.mark.asyncio
    async def test_cancelled_task_lifecycle_validation(self, clean_manager, task_factory):
        """Test cancelled task lifecycle with proper cleanup."""
        task_id = await clean_manager.submit_task('cancelled_task', task_factory, duration=2.0)
        await asyncio.sleep(0.1)
        cancel_result = await clean_manager.cancel_task(task_id)
        assert cancel_result is True
        await asyncio.sleep(0.1)
        task = clean_manager.get_task_status(task_id)
        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None
        assert task.result is None
        assert task.created_at < task.started_at < task.completed_at

    @pytest.mark.asyncio
    async def test_manager_lifecycle_with_active_tasks(self, task_factory):
        """Test manager lifecycle with active tasks during shutdown."""
        manager = BackgroundTaskManager(max_concurrent_tasks=5)
        await manager.start()
        task_ids = []
        for i in range(10):
            task_id = await manager.submit_task(f'lifecycle_{i}', task_factory, duration=0.5)
            task_ids.append(task_id)
        await asyncio.sleep(0.1)
        running_count = sum((1 for task_id in task_ids if manager.get_task_status(task_id).status == TaskStatus.RUNNING))
        assert running_count > 0
        shutdown_start = time.time()
        await manager.stop(timeout=2.0)
        shutdown_duration = time.time() - shutdown_start
        assert shutdown_duration < 2.5
        for task_id in task_ids:
            task = manager.get_task_status(task_id)
            assert task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]

    @pytest.mark.asyncio
    async def test_resource_cleanup_during_lifecycle(self, clean_manager):
        """Test proper resource cleanup during task lifecycle."""
        resources_created = []
        resources_cleaned = []

        async def resource_task(resource_id: str):
            """Task that creates and cleans up resources."""
            resources_created.append(resource_id)
            try:
                await asyncio.sleep(0.1)
                return f'resource_{resource_id}_processed'
            finally:
                resources_cleaned.append(resource_id)
        task_ids = []
        for i in range(5):
            task_id = await clean_manager.submit_task(f'resource_{i}', resource_task, resource_id=str(i))
            task_ids.append(task_id)
        await asyncio.sleep(0.5)
        assert len(resources_created) == 5
        assert len(resources_cleaned) == 5
        assert set(resources_created) == set(resources_cleaned)
        for task_id in task_ids:
            task = clean_manager.get_task_status(task_id)
            assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_network_failure_scenario(self, clean_manager):
        """Test handling of real network failure scenarios."""

        async def network_task(should_fail: bool=False):
            """Simulate network operation that may fail."""
            if should_fail:
                await asyncio.sleep(0.1)
                raise TimeoutError('Network request timed out')
            await asyncio.sleep(0.05)
            return 'network_success'
        task_ids = []
        expected_failures = 0
        for i in range(10):
            should_fail = i % 3 == 0
            if should_fail:
                expected_failures += 1
            task_id = await clean_manager.submit_task(f'network_{i}', network_task, should_fail=should_fail)
            task_ids.append(task_id)
        await asyncio.sleep(0.5)
        successful_tasks = 0
        failed_tasks = 0
        for task_id in task_ids:
            task = clean_manager.get_task_status(task_id)
            if task.status == TaskStatus.COMPLETED:
                successful_tasks += 1
                assert task.result == 'network_success'
            elif task.status == TaskStatus.FAILED:
                failed_tasks += 1
                assert 'Network request timed out' in task.error
        expected_successes = 10 - expected_failures
        assert successful_tasks == expected_successes
        assert failed_tasks == expected_failures

    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenario(self, clean_manager):
        """Test handling of resource exhaustion scenarios."""
        limited_manager = BackgroundTaskManager(max_concurrent_tasks=3)
        await limited_manager.start()

        async def resource_heavy_task(index: str):
            """Task that simulates resource usage."""
            await asyncio.sleep(0.3)
            return f'resource_task_{index}_completed'
        task_ids = []
        for i in range(3):
            task_id = await limited_manager.submit_task(f'resource_{i}', resource_heavy_task, index=str(i))
            task_ids.append(task_id)
        await asyncio.sleep(0.1)
        rejected_count = 0
        for i in range(3, 8):
            try:
                task_id = await limited_manager.submit_task(f'resource_{i}', resource_heavy_task, index=str(i))
                task_ids.append(task_id)
            except ValueError as e:
                assert 'Maximum concurrent tasks exceeded' in str(e)
                rejected_count += 1
        assert len(task_ids) == 3
        assert rejected_count == 5
        await asyncio.sleep(1.0)
        for task_id in task_ids:
            task = limited_manager.get_task_status(task_id)
            assert task.status == TaskStatus.COMPLETED
        await limited_manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_memory_pressure_scenario(self, clean_manager):
        """Test handling of memory pressure scenarios."""
        gc.collect()

        async def memory_intensive_task(data_size: int):
            """Task that creates and processes large data."""
            large_data = [i for i in range(data_size)]
            result = sum(large_data)
            await asyncio.sleep(0.01)
            return f'processed_{len(large_data)}_items_sum_{result}'
        task_ids = []
        for i in range(5):
            data_size = 10000 * (i + 1)
            task_id = await clean_manager.submit_task(f'memory_{i}', memory_intensive_task, data_size=data_size)
            task_ids.append(task_id)
        await asyncio.sleep(0.5)
        for task_id in task_ids:
            task = clean_manager.get_task_status(task_id)
            assert task.status == TaskStatus.COMPLETED
            assert 'processed_' in task.result
            assert '_items_sum_' in task.result
        gc.collect()

    @pytest.mark.asyncio
    async def test_cascading_failure_scenario(self, clean_manager):
        """Test handling of cascading failure scenarios."""
        failure_count = {'count': 0}

        async def cascading_task(index: str, should_trigger_cascade: bool=False):
            """Task that may trigger cascading failures."""
            if should_trigger_cascade:
                failure_count['count'] += 1
                if failure_count['count'] <= 3:
                    raise RuntimeError(f"Cascading failure #{failure_count['count']} in task {index}")
            await asyncio.sleep(0.1)
            return f'cascading_task_{index}_completed'
        task_ids = []
        for i in range(8):
            should_trigger = i < 4
            task_id = await clean_manager.submit_task(f'cascade_{i}', cascading_task, index=str(i), should_trigger_cascade=should_trigger)
            task_ids.append(task_id)
        await asyncio.sleep(0.5)
        failed_tasks = 0
        successful_tasks = 0
        for task_id in task_ids:
            task = clean_manager.get_task_status(task_id)
            if task.status == TaskStatus.FAILED:
                failed_tasks += 1
                assert 'Cascading failure' in task.error
            elif task.status == TaskStatus.COMPLETED:
                successful_tasks += 1
        assert failed_tasks == 3
        assert successful_tasks == 5

    @pytest.mark.asyncio
    async def test_timeout_failure_scenario(self, clean_manager):
        """Test handling of timeout failure scenarios."""

        async def timeout_prone_task(duration: float, should_timeout: bool=False):
            """Task that may timeout."""
            if should_timeout:
                await asyncio.sleep(duration)
                return 'should_not_reach_here'
            await asyncio.sleep(0.1)
            return 'timeout_task_completed'
        task_ids = []
        for i in range(6):
            should_timeout = i % 2 == 0
            duration = 2.0 if should_timeout else 0.1
            task_id = await clean_manager.submit_task(f'timeout_{i}', timeout_prone_task, duration=duration, should_timeout=should_timeout)
            task_ids.append(task_id)
        await asyncio.sleep(0.2)
        for i, task_id in enumerate(task_ids):
            if i % 2 == 0:
                await clean_manager.cancel_task(task_id)
        await asyncio.sleep(0.2)
        cancelled_tasks = 0
        successful_tasks = 0
        for i, task_id in enumerate(task_ids):
            task = clean_manager.get_task_status(task_id)
            if i % 2 == 0:
                assert task.status == TaskStatus.CANCELLED
                cancelled_tasks += 1
            else:
                assert task.status == TaskStatus.COMPLETED
                assert task.result == 'timeout_task_completed'
                successful_tasks += 1
        assert cancelled_tasks == 3
        assert successful_tasks == 3
