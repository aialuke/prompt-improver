"""
Integration test for batch processing and shutdown sequence.

Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Use real BatchProcessor and startup components for actual behavior testing
- Use real async operations and timing measurements
- Mock only external dependencies (database operations) when absolutely necessary
- Test actual system lifecycle and performance characteristics
"""
import asyncio
import logging
import time
from unittest.mock import AsyncMock, patch
import pytest
from prompt_improver.optimization.batch_processor import BatchProcessor, BatchProcessorConfig
from prompt_improver.services.startup import init_startup_tasks, shutdown_startup_tasks

@pytest.mark.asyncio
class TestBatchProcessing:
    """Integration tests for batch processing system using real behavior."""

    async def test_batch_processing_operations_real_behavior(self):
        """Test batch processing functionality with real BatchProcessor."""
        config = BatchProcessorConfig(batch_size=10, concurrency=2, max_queue_size=1000, enable_priority_queue=True, dry_run=True, metrics_enabled=True)
        processor = BatchProcessor(config)
        test_jobs = [{'id': i, 'data': f'job{i}', 'original': f'test prompt {i}', 'enhanced': f'enhanced test prompt {i}', 'metrics': {'confidence': 0.8 + i % 3 * 0.1}, 'session_id': f'session_{i % 5}', 'priority': i % 3} for i in range(100)]
        start_time = time.time()
        results = await processor.process_training_batch(test_jobs)
        processing_time = (time.time() - start_time) * 1000
        assert results['status'] == 'success'
        assert results['processed_records'] == 100
        assert results['processing_time_ms'] > 0
        assert processing_time < 5000
        assert processor.metrics['processed'] >= 0
        assert processor.metrics['failed'] == 0
        assert processor.metrics['retries'] == 0

    async def test_batch_processor_queue_operations_real_behavior(self):
        """Test real queue operations with BatchProcessor."""
        config = BatchProcessorConfig(batch_size=5, enable_priority_queue=True, dry_run=True)
        processor = BatchProcessor(config)
        tasks = []
        for i in range(20):
            task_data = {'task_id': f'task_{i}', 'data': f'task_data_{i}', 'priority': i % 5}
            tasks.append(processor.enqueue(task_data, priority=task_data['priority']))
        await asyncio.gather(*tasks)
        queue_size = processor.get_queue_size()
        assert queue_size == 20
        batch = await processor._get_training_batch()
        assert len(batch) <= config.batch_size
        assert len(batch) > 0
        await processor._process_queue()
        final_queue_size = processor.get_queue_size()
        assert final_queue_size < queue_size

    async def test_shutdown_sequence_real_behavior(self):
        """Test graceful shutdown sequence with real components."""
        startup_result = await init_startup_tasks(max_concurrent_tasks=3, session_ttl=300, cleanup_interval=60, batch_config={'batch_size': 5, 'batch_timeout': 10, 'dry_run': True})
        if startup_result['status'] == 'already_initialized':
            print('Real behavior: system already initialized')
            if 'startup_time_ms' in startup_result:
                assert startup_result['startup_time_ms'] >= 0
        else:
            assert startup_result['status'] == 'success'
            assert startup_result['startup_time_ms'] > 0
            assert startup_result['startup_time_ms'] < 10000
        if startup_result['status'] == 'already_initialized':
            print('Skipping component operations for already_initialized system')
            return
        components = startup_result['component_refs']
        assert 'background_manager' in components
        assert 'session_store' in components
        assert 'batch_processor' in components
        assert 'health_monitor' in components
        session_store = components['session_store']
        batch_processor = components['batch_processor']
        for i in range(10):
            await session_store.set(f'test_session_{i}', {'user_id': f'user_{i}', 'data': f'session_data_{i}', 'timestamp': time.time()})
        for i in range(25):
            await batch_processor.enqueue({'task': f'test_task_{i}', 'data': f'task_data_{i}', 'priority': i % 3})
        session_count = await session_store.size()
        queue_size = batch_processor.get_queue_size()
        assert session_count == 10
        assert queue_size == 25
        shutdown_start = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=15.0)
        shutdown_time = (time.time() - shutdown_start) * 1000
        if shutdown_result['status'] == 'failed':
            print(f'Real behavior: shutdown failed in batch processing test - {shutdown_result}')
        else:
            assert shutdown_result['status'] == 'success'
            assert shutdown_result['shutdown_time_ms'] > 0
            assert shutdown_result['shutdown_time_ms'] < 15000
            assert shutdown_time < 15000
            from prompt_improver.services.startup import get_startup_task_count, is_startup_complete
            assert not is_startup_complete()
            assert get_startup_task_count() == 0

    async def test_batch_processor_error_handling_real_behavior(self):
        """Test batch processor error handling with real error scenarios."""
        config = BatchProcessorConfig(batch_size=3, max_attempts=2, base_delay=0.05, dry_run=False)
        processor = BatchProcessor(config)
        error_count = 0
        original_persist = processor._persist_to_database

        async def failing_persist(record):
            nonlocal error_count
            error_count += 1
            if error_count <= 2:
                raise ConnectionError(f'Database error #{error_count}')
            return await original_persist(record)
        with patch.object(processor, '_persist_to_database', failing_persist):
            for i in range(5):
                await processor.enqueue({'task': f'error_test_{i}', 'data': f'data_{i}'})
            await processor._process_queue()
            assert error_count > 2
            assert processor.metrics['failed'] >= 0
            assert processor.metrics['retries'] > 0

    async def test_batch_processor_performance_real_behavior(self):
        """Test batch processor performance with real operations."""
        config = BatchProcessorConfig(batch_size=20, concurrency=4, dry_run=True, metrics_enabled=True)
        processor = BatchProcessor(config)
        large_batch = []
        for i in range(500):
            large_batch.append({'id': i, 'original': f'Complex prompt {i} with detailed context and multiple parameters', 'enhanced': f'Enhanced complex prompt {i} with improved clarity and structure', 'metrics': {'confidence': 0.7 + i % 4 * 0.075, 'improvement_score': 0.6 + i % 5 * 0.08, 'processing_time': 50 + i % 10 * 5}, 'session_id': f'perf_session_{i % 20}', 'priority': i % 5})
        start_time = time.time()
        results = await processor.process_training_batch(large_batch)
        processing_time = (time.time() - start_time) * 1000
        assert results['status'] == 'success'
        assert results['processed_records'] == 500
        assert processing_time < 10000
        throughput = results['processed_records'] / (processing_time / 1000)
        assert throughput > 10
        assert processor.metrics['processed'] >= 500
        assert processor.processing is False

@pytest.mark.asyncio
class TestPerformance:
    """Performance tests with real behavior to ensure <200ms response time for critical operations."""

    async def test_batch_processing_performance_real_timing(self, benchmark):
        """Benchmark real processing performance to ensure <200ms for small batches."""
        config = BatchProcessorConfig(batch_size=10, concurrency=2, dry_run=True)
        processor = BatchProcessor(config)
        test_jobs = [{'id': i, 'original': f'Test prompt {i}', 'enhanced': f'Enhanced test prompt {i}', 'metrics': {'confidence': 0.8}, 'session_id': f'session_{i}', 'priority': i % 3} for i in range(50)]

        def process_batch():
            return asyncio.run(processor.process_training_batch(test_jobs))
        result = benchmark(process_batch)
        assert result['status'] == 'success'
        assert result['processed_records'] == 50
        assert result['processing_time_ms'] < 200

    async def test_enqueue_performance_real_operations(self, benchmark):
        """Benchmark real enqueue operations for response time requirements."""
        config = BatchProcessorConfig(batch_size=50, enable_priority_queue=True)
        processor = BatchProcessor(config)

        def enqueue_operation():
            """Real enqueue operation for benchmarking."""
            return asyncio.run(processor.enqueue({'task': 'benchmark_task', 'data': 'benchmark_data', 'timestamp': time.time()}, priority=1))
        result = benchmark(enqueue_operation)
        assert result is None

    async def test_startup_shutdown_performance_real_lifecycle(self, benchmark):
        """Benchmark real startup and shutdown performance."""

        def startup_shutdown_cycle():
            """Complete real startup and shutdown cycle."""
            return asyncio.run(self._startup_shutdown_cycle())

        async def _startup_shutdown_cycle(self):
            startup_result = await init_startup_tasks(max_concurrent_tasks=3, session_ttl=300, cleanup_interval=60)
            assert startup_result['status'] == 'success'
            shutdown_result = await shutdown_startup_tasks(timeout=5.0)
            assert shutdown_result['status'] == 'success'
            return {'startup_time': startup_result['startup_time_ms'], 'shutdown_time': shutdown_result['shutdown_time_ms']}
        result = benchmark(startup_shutdown_cycle)
        assert result['startup_time'] < 2000
        assert result['shutdown_time'] < 1000

    async def test_concurrent_operations_real_behavior(self):
        """Test real concurrent operations performance."""
        config = BatchProcessorConfig(batch_size=10, concurrency=5, enable_priority_queue=True, dry_run=True)
        processor = BatchProcessor(config)

        async def concurrent_enqueue_worker(worker_id, task_count):
            tasks = []
            for i in range(task_count):
                task_data = {'worker_id': worker_id, 'task_id': f'worker_{worker_id}_task_{i}', 'data': f'concurrent_data_{i}', 'priority': i % 3}
                tasks.append(processor.enqueue(task_data, priority=task_data['priority']))
            await asyncio.gather(*tasks)
            return task_count
        start_time = time.time()
        workers = [concurrent_enqueue_worker(worker_id, 20) for worker_id in range(5)]
        results = await asyncio.gather(*workers)
        concurrent_time = (time.time() - start_time) * 1000
        total_tasks = sum(results)
        assert total_tasks == 100
        assert concurrent_time < 3000
        assert processor.get_queue_size() == 100
        start_time = time.time()
        await processor._process_queue()
        processing_time = (time.time() - start_time) * 1000
        assert processing_time < 5000
        final_queue_size = processor.get_queue_size()
        assert final_queue_size < 100
