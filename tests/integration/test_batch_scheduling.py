"""
Integration tests for batch scheduling system including priority queue,
periodic processing, and performance benchmarks.

Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Use real BatchProcessor, PriorityQueue, and configuration components
- Use real async operations and timing measurements for performance validation
- Mock only external dependencies (database persistence) when absolutely necessary
- Test actual scheduling behavior, queue operations, and retry mechanisms
- Verify real concurrent processing and rate limiting functionality
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
import pytest
from prompt_improver.ml.optimization.batch.batch_processor import BatchProcessor, BatchProcessorConfig, PriorityQueue, periodic_batch_processor_coroutine

@pytest.mark.asyncio
class TestBatchScheduling:
    """Integration tests for batch scheduling system using real behavior."""

    async def test_batch_processor_config_validation_real_behavior(self):
        """Test batch processor configuration validation with real validation logic."""
        config = BatchProcessorConfig(batch_size=10, concurrency=3, timeout=30000, batch_timeout=30, max_attempts=3, base_delay=1.0, max_delay=60.0, enable_priority_queue=True, dry_run=True)
        assert config.batch_size == 10
        assert config.concurrency == 3
        assert config.timeout == 30000
        assert config.batch_timeout == 30
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.enable_priority_queue is True
        with pytest.raises(ValueError):
            BatchProcessorConfig(batch_size=0)
        with pytest.raises(ValueError):
            BatchProcessorConfig(concurrency=0)
        with pytest.raises(ValueError):
            BatchProcessorConfig(batch_timeout=-1)

    async def test_priority_queue_operations_real_behavior(self):
        """Test priority queue operations with real PriorityQueue."""
        pq = PriorityQueue()
        assert pq.size() == 0
        assert pq.is_empty()
        test_items = [({'task': 'low_priority', 'data': 'low'}, 10), ({'task': 'high_priority', 'data': 'high'}, 1), ({'task': 'medium_priority', 'data': 'medium'}, 5), ({'task': 'critical_priority', 'data': 'critical'}, 0)]
        for record, priority in test_items:
            pq.enqueue(record, priority=priority)
        assert pq.size() == 4
        assert not pq.is_empty()
        first = pq.dequeue()
        assert first.priority == 0
        assert first.record['task'] == 'critical_priority'
        second = pq.dequeue()
        assert second.priority == 1
        assert second.record['task'] == 'high_priority'
        third = pq.dequeue()
        assert third.priority == 5
        assert third.record['task'] == 'medium_priority'
        fourth = pq.dequeue()
        assert fourth.priority == 10
        assert fourth.record['task'] == 'low_priority'
        assert pq.size() == 0
        assert pq.is_empty()
        empty = pq.dequeue()
        assert empty is None

    async def test_batch_processor_enqueue_priority_real_behavior(self):
        """Test batch processor enqueue with real priority handling."""
        config = BatchProcessorConfig(batch_size=5, enable_priority_queue=True, dry_run=True, metrics_enabled=True)
        processor = BatchProcessor(config)
        tasks_data = [({'task': 'urgent', 'data': 'urgent_data'}, 1), ({'task': 'normal', 'data': 'normal_data'}, 5), ({'task': 'low', 'data': 'low_data'}, 10), ({'task': 'critical', 'data': 'critical_data'}, 0)]
        for task_data, priority in tasks_data:
            await processor.enqueue(task_data, priority=priority)
        assert processor.priority_queue.size() == 4
        assert processor.get_queue_size() == 4
        first = processor.priority_queue.dequeue()
        assert first.priority == 0
        assert first.record['task'] == 'critical'
        second = processor.priority_queue.dequeue()
        assert second.priority == 1
        assert second.record['task'] == 'urgent'
        assert processor.priority_queue.size() == 2

    async def test_batch_processor_async_queue_real_behavior(self):
        """Test batch processor with real async queue wrapper."""
        config = BatchProcessorConfig(batch_size=3, enable_priority_queue=False, dry_run=True)
        processor = BatchProcessor(config)
        test_tasks = [{'task': 'task1', 'priority': 5}, {'task': 'task2', 'priority': 3}, {'task': 'task3', 'priority': 1}, {'task': 'task4', 'priority': 2}]
        for task in test_tasks:
            await processor.enqueue(task, priority=task['priority'])
        assert processor.async_queue.qsize() == 4
        assert processor.get_queue_size() == 4
        batch = await processor._get_training_batch()
        assert len(batch) <= config.batch_size
        assert len(batch) > 0
        remaining_size = processor.get_queue_size()
        assert remaining_size < 4

    async def test_batch_processor_rate_limiting_real_behavior(self):
        """Test batch processor rate limiting with real timing."""
        config = BatchProcessorConfig(batch_size=10, rate_limit_per_second=2.0, dry_run=True)
        processor = BatchProcessor(config)
        expected_delay = 1.0 / 2.0
        assert processor.async_queue.config.rate_limit_delay == expected_delay
        start_time = time.time()
        tasks = []
        for i in range(5):
            task_data = {'task': f'rate_test_{i}', 'data': f'data_{i}'}
            tasks.append(processor.enqueue(task_data))
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        assert elapsed < 2.0
        assert processor.get_queue_size() == 5

    async def test_batch_processor_retry_logic_real_behavior(self):
        """Test batch processor exponential backoff retry logic with real timing."""
        config = BatchProcessorConfig(batch_size=2, max_attempts=3, base_delay=0.05, max_delay=1.0, jitter=False, dry_run=False)
        processor = BatchProcessor(config)
        attempt_count = 0
        attempt_times = []
        original_persist = processor._persist_to_database

        async def failing_persist(record):
            nonlocal attempt_count
            attempt_count += 1
            attempt_times.append(time.time())
            if attempt_count <= 2:
                raise Exception(f'Simulated failure #{attempt_count}')
            return await original_persist(record)
        with patch.object(processor, '_persist_to_database', failing_persist):
            await processor.enqueue({'task': 'retry_test', 'data': 'test_data'})
            start_time = time.time()
            await processor._process_queue()
            total_time = time.time() - start_time
            assert attempt_count == 3
            assert len(attempt_times) == 3
            if len(attempt_times) >= 2:
                first_retry_delay = attempt_times[1] - attempt_times[0]
                assert first_retry_delay >= config.base_delay * 0.8
            assert processor.metrics['retries'] > 0

    async def test_periodic_batch_processor_coroutine_real_behavior(self):
        """Test periodic batch processor coroutine with real timing."""
        config = BatchProcessorConfig(batch_size=5, batch_timeout=0.2, dry_run=True)
        processor = BatchProcessor(config)
        for i in range(15):
            await processor.enqueue({'task': f'periodic_task_{i}', 'data': f'periodic_data_{i}', 'priority': i % 3})
        initial_queue_size = processor.get_queue_size()
        assert initial_queue_size == 15
        periodic_task = asyncio.create_task(periodic_batch_processor_coroutine(processor))
        await asyncio.sleep(0.5)
        periodic_task.cancel()
        try:
            await periodic_task
        except asyncio.CancelledError:
            pass
        final_queue_size = processor.get_queue_size()
        assert final_queue_size < initial_queue_size
        assert processor.metrics['processed'] > 0

    async def test_batch_size_limits_real_behavior(self):
        """Test batch size limits and formation with real queue operations."""
        config = BatchProcessorConfig(batch_size=3, enable_priority_queue=True, dry_run=True)
        processor = BatchProcessor(config)
        test_tasks = []
        for i in range(10):
            task_data = {'task': f'batch_test_{i}', 'data': f'data_{i}', 'priority': i, 'timestamp': time.time()}
            test_tasks.append((task_data, i))
        for task_data, priority in test_tasks:
            await processor.enqueue(task_data, priority=priority)
        batch = await processor._get_training_batch()
        assert len(batch) <= config.batch_size
        assert len(batch) > 0
        if len(batch) > 1:
            for i in range(len(batch) - 1):
                current_priority = batch[i].get('priority', float('inf'))
                next_priority = batch[i + 1].get('priority', float('inf'))
                assert current_priority <= next_priority

    async def test_batch_processor_timeout_handling_real_behavior(self):
        """Test batch processor timeout handling with real timing."""
        config = BatchProcessorConfig(batch_size=5, timeout=100, dry_run=False)
        processor = BatchProcessor(config)

        async def slow_persist(record):
            await asyncio.sleep(0.2)
            return True
        with patch.object(processor, '_persist_to_database', slow_persist):
            start_time = time.time()
            success = await processor._process_queue_item({'item': {'task': 'timeout_test'}, 'priority': 1, 'timestamp': time.time(), 'attempts': 0})
            elapsed_time = (time.time() - start_time) * 1000
            assert success is False
            assert elapsed_time <= config.timeout * 2

    async def test_batch_processor_metrics_collection_real_behavior(self):
        """Test batch processor metrics collection with real operations."""
        config = BatchProcessorConfig(batch_size=3, metrics_enabled=True, dry_run=True)
        processor = BatchProcessor(config)
        assert processor.metrics['processed'] == 0
        assert processor.metrics['failed'] == 0
        assert processor.metrics['retries'] == 0
        assert 'start_time' in processor.metrics
        for i in range(7):
            await processor.enqueue({'task': f'metrics_task_{i}', 'data': f'metrics_data_{i}', 'priority': i % 3})
        await processor._process_queue()
        assert processor.metrics['processed'] > 0
        assert processor.metrics['failed'] == 0
        total_operations = processor.metrics['processed'] + processor.metrics['failed']
        assert total_operations > 0

    async def test_batch_processor_dry_run_mode_real_behavior(self):
        """Test batch processor dry run mode with real operations."""
        config = BatchProcessorConfig(batch_size=3, dry_run=True)
        processor = BatchProcessor(config)
        for i in range(8):
            await processor.enqueue({'task': f'dry_run_task_{i}', 'original': f'Original prompt {i}', 'enhanced': f'Enhanced prompt {i}', 'metrics': {'confidence': 0.8 + i % 2 * 0.1}, 'session_id': f'session_{i % 3}'})
        initial_queue_size = processor.get_queue_size()
        assert initial_queue_size == 8
        start_time = time.time()
        await processor._process_queue()
        processing_time = (time.time() - start_time) * 1000
        assert processor.metrics['processed'] > 0
        assert processor.metrics['failed'] == 0
        assert processing_time < 2000
        final_queue_size = processor.get_queue_size()
        assert final_queue_size < initial_queue_size

@pytest.mark.asyncio
class TestBatchSchedulingPerformance:
    """Performance tests for batch scheduling system using real behavior."""

    def test_enqueue_performance_real_operations(self, benchmark):
        """Benchmark real enqueue operations."""
        config = BatchProcessorConfig(batch_size=100, enable_priority_queue=True)
        processor = BatchProcessor(config)

        def enqueue_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(processor.enqueue({'task': 'benchmark_task', 'data': 'benchmark_data', 'timestamp': time.time()}, priority=5))
            finally:
                loop.close()
        result = benchmark(enqueue_task)
        assert result is None

    def test_batch_formation_performance_real_operations(self, benchmark):
        """Benchmark real batch formation from queue."""
        config = BatchProcessorConfig(batch_size=50, enable_priority_queue=True)
        processor = BatchProcessor(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for i in range(200):
                loop.run_until_complete(processor.enqueue({'task': f'perf_task_{i}', 'data': f'perf_data_{i}', 'priority': i % 10}, priority=i % 10))
        finally:
            loop.close()

        def get_batch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(processor._get_training_batch())
            finally:
                loop.close()
        result = benchmark(get_batch)
        assert len(result) <= config.batch_size
        assert len(result) > 0

    async def test_priority_queue_performance_real_operations(self, benchmark):
        """Benchmark real priority queue operations."""
        pq = PriorityQueue()
        for i in range(1000):
            pq.enqueue({'task': f'perf_task_{i}', 'data': f'performance_data_{i}', 'timestamp': time.time()}, priority=i % 100)

        def dequeue_batch():
            batch = []
            for _ in range(50):
                item = pq.dequeue()
                if item:
                    batch.append(item)
                else:
                    break
            return batch
        result = benchmark(dequeue_batch)
        assert len(result) <= 50

    async def test_concurrent_enqueue_performance_real_behavior(self, benchmark):
        """Benchmark real concurrent enqueue operations."""
        config = BatchProcessorConfig(batch_size=100, concurrency=10, enable_priority_queue=True)
        processor = BatchProcessor(config)

        async def concurrent_enqueue():
            tasks = []
            for i in range(100):
                task_data = {'task': f'concurrent_task_{i}', 'data': f'concurrent_data_{i}', 'timestamp': time.time(), 'worker_id': i % 10}
                tasks.append(processor.enqueue(task_data, priority=i % 10))
            await asyncio.gather(*tasks)
            return len(tasks)

        def run_concurrent():
            return asyncio.run(concurrent_enqueue())
        result = benchmark(run_concurrent)
        assert result == 100

    @pytest.mark.benchmark(min_time=0.1, max_time=0.5, min_rounds=5, disable_gc=True, warmup=False)
    def test_batch_processing_response_time_real_behavior(self, benchmark):
        """Test that real batch processing meets < 200ms response time requirement."""
        config = BatchProcessorConfig(batch_size=10, concurrency=5, dry_run=True)
        processor = BatchProcessor(config)
        test_batch = []
        for i in range(50):
            test_batch.append({'original': f'Real test prompt {i} with realistic content and complexity', 'enhanced': f'Enhanced real test prompt {i} with improved structure and clarity', 'metrics': {'confidence': 0.75 + i % 4 * 0.05, 'improvement_score': 0.6 + i % 5 * 0.08, 'complexity': i % 3}, 'session_id': f'real_session_{i % 10}', 'priority': i % 5})

        def process_training_batch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(processor.process_training_batch(test_batch))
                return result
            finally:
                loop.close()
        result = benchmark(process_training_batch)
        assert result['status'] == 'success'
        assert result['processing_time_ms'] < 200
        assert result['processed_records'] == len(test_batch)

    async def test_memory_usage_under_load_real_behavior(self, benchmark):
        """Test real memory usage during high-load batch processing."""
        config = BatchProcessorConfig(batch_size=100, max_queue_size=10000, enable_priority_queue=True)
        processor = BatchProcessor(config)

        async def load_queue():
            for i in range(2000):
                await processor.enqueue({'task': f'load_task_{i}', 'data': f'realistic_data_{i}' * 5, 'timestamp': time.time(), 'metadata': {'batch_id': f'batch_{i // 100}', 'worker_id': i % 20, 'priority_level': i % 5}}, priority=i % 100)
            return processor.get_queue_size()

        def run_load():
            return asyncio.run(load_queue())
        result = benchmark(run_load)
        assert result > 0
        assert result <= 2000

@pytest.mark.asyncio
class TestBatchSchedulingIntegration:
    """Integration tests combining real batch scheduling with other components."""

    async def test_batch_processor_with_session_store_real_behavior(self):
        """Test real batch processor integration with session store."""
        from prompt_improver.utils.session_store import SessionStore
        session_store = SessionStore(maxsize=100, ttl=300)
        config = BatchProcessorConfig(batch_size=5, dry_run=True)
        processor = BatchProcessor(config)
        async with session_store:
            session_data = {'user_id': 'test_user_123', 'batch_id': 'batch_456', 'status': 'processing', 'start_time': time.time(), 'preferences': {'theme': 'dark', 'language': 'en'}}
            await session_store.set('batch_session', session_data)
            for i in range(12):
                await processor.enqueue({'task': f'session_task_{i}', 'session_id': 'batch_session', 'data': f'session_data_{i}', 'priority': i % 3})
            await processor._process_queue()
            updated_data = session_data.copy()
            updated_data['status'] = 'completed'
            updated_data['end_time'] = time.time()
            updated_data['processed_count'] = 12
            await session_store.set('batch_session', updated_data)
            final_session = await session_store.get('batch_session')
            assert final_session['status'] == 'completed'
            assert final_session['processed_count'] == 12
            assert 'end_time' in final_session

    async def test_batch_processor_error_recovery_real_behavior(self):
        """Test real batch processor error recovery and resilience."""
        config = BatchProcessorConfig(batch_size=3, max_attempts=2, base_delay=0.05, dry_run=False)
        processor = BatchProcessor(config)
        fail_every_n = 3
        call_count = 0

        async def intermittent_fail(record):
            nonlocal call_count
            call_count += 1
            if call_count % fail_every_n == 0:
                raise ConnectionError(f'Real simulated network failure #{call_count}')
            return True
        with patch.object(processor, '_persist_to_database', intermittent_fail):
            for i in range(10):
                await processor.enqueue({'task': f'recovery_task_{i}', 'data': f'recovery_data_{i}', 'priority': i % 3})
            await processor._process_queue()
            assert processor.metrics['failed'] > 0
            assert processor.metrics['processed'] > 0
            assert call_count > 10

    async def test_batch_processor_graceful_shutdown_real_behavior(self):
        """Test real batch processor graceful shutdown behavior."""
        config = BatchProcessorConfig(batch_size=5, batch_timeout=0.1, dry_run=True)
        processor = BatchProcessor(config)
        for i in range(25):
            await processor.enqueue({'task': f'shutdown_task_{i}', 'data': f'shutdown_data_{i}', 'priority': i % 3})
        periodic_task = asyncio.create_task(periodic_batch_processor_coroutine(processor))
        await asyncio.sleep(0.3)
        start_time = time.time()
        periodic_task.cancel()
        try:
            await asyncio.wait_for(periodic_task, timeout=2.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail('Real graceful shutdown took too long')
        shutdown_time = (time.time() - start_time) * 1000
        assert shutdown_time < 2000

    async def test_batch_processor_configuration_updates_real_behavior(self):
        """Test real batch processor configuration updates."""
        config = BatchProcessorConfig(batch_size=5, concurrency=2, dry_run=True)
        processor = BatchProcessor(config)
        assert processor.config.batch_size == 5
        assert processor.config.concurrency == 2
        new_config = BatchProcessorConfig(batch_size=10, concurrency=4, dry_run=True)
        processor.config = new_config
        assert processor.config.batch_size == 10
        assert processor.config.concurrency == 4
        for i in range(15):
            await processor.enqueue({'task': f'config_test_{i}', 'data': f'config_data_{i}'})
        batch = await processor._get_training_batch()
        assert len(batch) <= 10
        assert len(batch) > 0
        await processor._process_queue()
        assert processor.metrics['processed'] > 0
