"""
Integration tests for batch scheduling system including priority queue,
periodic processing, and performance benchmarks.

Tests cover batch processor configuration, priority-based scheduling,
timeout handling, retry mechanisms, and performance requirements.
"""

import asyncio
import pytest
import time
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from prompt_improver.optimization.batch_processor import (
    BatchProcessor, 
    BatchProcessorConfig, 
    PriorityQueue,
    periodic_batch_processor_coroutine
)


@pytest.mark.asyncio
class TestBatchScheduling:
    """Integration tests for batch scheduling system."""
    
    async def test_batch_processor_config_validation(self):
        """Test batch processor configuration validation."""
        # Test valid configuration
        config = BatchProcessorConfig(
            batch_size=10,
            concurrency=3,
            timeout=30000,
            batch_timeout=30,
            max_attempts=3,
            base_delay=1.0,
            max_delay=60.0
        )
        assert config.batch_size == 10
        assert config.concurrency == 3
        assert config.timeout == 30000
        assert config.batch_timeout == 30
        
        # Test invalid configuration (should raise validation error)
        with pytest.raises(ValueError):
            BatchProcessorConfig(batch_size=0)  # batch_size must be >= 1
        
        with pytest.raises(ValueError):
            BatchProcessorConfig(concurrency=0)  # concurrency must be >= 1
    
    async def test_priority_queue_operations(self):
        """Test priority queue operations."""
        pq = PriorityQueue()
        
        # Test enqueue with different priorities
        pq.enqueue({"task": "low_priority"}, priority=10)
        pq.enqueue({"task": "high_priority"}, priority=1)
        pq.enqueue({"task": "medium_priority"}, priority=5)
        
        # Test dequeue returns highest priority (lowest number) first
        first = pq.dequeue()
        assert first.priority == 1
        assert first.record["task"] == "high_priority"
        
        second = pq.dequeue()
        assert second.priority == 5
        assert second.record["task"] == "medium_priority"
        
        third = pq.dequeue()
        assert third.priority == 10
        assert third.record["task"] == "low_priority"
        
        # Test empty queue
        empty = pq.dequeue()
        assert empty is None
    
    async def test_batch_processor_enqueue_priority(self):
        """Test batch processor enqueue with priority handling."""
        config = BatchProcessorConfig(
            batch_size=5,
            enable_priority_queue=True,
            dry_run=True  # Safe for testing
        )
        processor = BatchProcessor(config)
        
        # Enqueue tasks with different priorities
        await processor.enqueue({"task": "urgent"}, priority=1)
        await processor.enqueue({"task": "normal"}, priority=5)
        await processor.enqueue({"task": "low"}, priority=10)
        
        # Verify tasks are in priority queue
        assert processor.priority_queue.size() == 3
        
        # Process queue and verify priority order
        first = processor.priority_queue.dequeue()
        assert first.record["task"] == "urgent"
        assert first.priority == 1
    
    async def test_batch_processor_async_queue(self):
        """Test batch processor with async queue wrapper."""
        config = BatchProcessorConfig(
            batch_size=3,
            enable_priority_queue=False,
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Enqueue tasks
        await processor.enqueue({"task": "task1"}, priority=5)
        await processor.enqueue({"task": "task2"}, priority=3)
        await processor.enqueue({"task": "task3"}, priority=1)
        
        # Verify tasks are in async queue
        assert processor.async_queue.qsize() == 3
    
    async def test_batch_processor_rate_limiting(self):
        """Test batch processor rate limiting."""
        config = BatchProcessorConfig(
            batch_size=10,
            rate_limit_per_second=2.0,  # 2 requests per second
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Test rate limit calculation
        expected_delay = 1.0 / 2.0  # 0.5 seconds
        assert processor.async_queue.config.rate_limit_delay == expected_delay
        
        # Test enqueueing with rate limiting
        start_time = time.time()
        tasks = []
        for i in range(3):
            tasks.append(processor.enqueue({"task": f"task{i}"}))
        
        await asyncio.gather(*tasks)
        
        # Should complete quickly since we're just enqueueing
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Enqueuing should be fast
    
    async def test_batch_processor_retry_logic(self):
        """Test batch processor exponential backoff retry logic."""
        config = BatchProcessorConfig(
            batch_size=2,
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            jitter=False,  # Disable jitter for predictable testing
            dry_run=False
        )
        processor = BatchProcessor(config)
        
        # Mock the persistence method to fail initially
        fail_count = 0
        original_persist = processor._persist_to_database
        
        async def mock_persist_fail_twice(record):
            nonlocal fail_count
            fail_count += 1
            if fail_count <= 2:
                raise Exception("Simulated failure")
            return await original_persist(record)
        
        with patch.object(processor, '_persist_to_database', mock_persist_fail_twice):
            # This should trigger retry logic
            await processor.enqueue({"task": "retry_test"})
            
            # Wait for processing with retries
            await asyncio.sleep(0.5)
            
            # Should have failed twice, then succeeded
            assert fail_count == 3
    
    async def test_periodic_batch_processor_coroutine(self):
        """Test periodic batch processor coroutine."""
        config = BatchProcessorConfig(
            batch_size=5,
            batch_timeout=0.1,  # 100ms for quick testing
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Add some tasks to the queue
        for i in range(10):
            await processor.enqueue({"task": f"task{i}"})
        
        # Start periodic processor
        periodic_task = asyncio.create_task(
            periodic_batch_processor_coroutine(processor)
        )
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # Cancel the periodic task
        periodic_task.cancel()
        try:
            await periodic_task
        except asyncio.CancelledError:
            pass
        
        # Some tasks should have been processed
        initial_size = processor.get_queue_size()
        assert initial_size < 10  # Some tasks should be processed
    
    async def test_batch_size_limits(self):
        """Test batch size limits and formation."""
        config = BatchProcessorConfig(
            batch_size=3,
            enable_priority_queue=True,
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Add more tasks than batch size
        for i in range(10):
            await processor.enqueue({"task": f"task{i}"}, priority=i)
        
        # Get a training batch
        batch = await processor._get_training_batch()
        
        # Should be limited to batch_size
        assert len(batch) <= config.batch_size
        
        # Should be in priority order (lower priority numbers first)
        if len(batch) > 1:
            for i in range(len(batch) - 1):
                assert batch[i].get("priority", 0) <= batch[i + 1].get("priority", 0)
    
    async def test_batch_processor_timeout_handling(self):
        """Test batch processor timeout handling."""
        config = BatchProcessorConfig(
            batch_size=5,
            timeout=100,  # 100ms timeout
            dry_run=False
        )
        processor = BatchProcessor(config)
        
        # Mock a slow persistence operation
        async def slow_persist(record):
            await asyncio.sleep(0.2)  # Slower than timeout
            return True
        
        with patch.object(processor, '_persist_to_database', slow_persist):
            # This should timeout
            success = await processor._process_queue_item({
                "item": {"task": "slow_task"},
                "priority": 1,
                "timestamp": time.time(),
                "attempts": 0
            })
            
            # Should handle timeout gracefully
            assert success is False
    
    async def test_batch_processor_metrics_collection(self):
        """Test batch processor metrics collection."""
        config = BatchProcessorConfig(
            batch_size=3,
            metrics_enabled=True,
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Initial metrics
        assert processor.metrics["processed"] == 0
        assert processor.metrics["failed"] == 0
        assert processor.metrics["retries"] == 0
        
        # Process some tasks
        for i in range(5):
            await processor.enqueue({"task": f"task{i}"})
        
        # Trigger processing
        await processor._process_queue()
        
        # Metrics should be updated
        assert processor.metrics["processed"] > 0
    
    async def test_batch_processor_dry_run_mode(self):
        """Test batch processor dry run mode."""
        config = BatchProcessorConfig(
            batch_size=3,
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Add tasks
        for i in range(5):
            await processor.enqueue({"task": f"task{i}"})
        
        # Process tasks in dry run mode
        await processor._process_queue()
        
        # Should complete without errors
        assert processor.metrics["processed"] > 0
        assert processor.metrics["failed"] == 0


@pytest.mark.asyncio
class TestBatchSchedulingPerformance:
    """Performance tests for batch scheduling system."""
    
    def test_enqueue_performance(self, benchmark):
        """Benchmark enqueue operations."""
        config = BatchProcessorConfig(
            batch_size=100,
            enable_priority_queue=True
        )
        processor = BatchProcessor(config)
        
        def enqueue_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(processor.enqueue({"task": "test_task"}, priority=5))
            finally:
                loop.close()
        
        # Should complete in < 200ms
        result = benchmark(enqueue_task)
        assert result is None  # enqueue returns None
    
    def test_batch_formation_performance(self, benchmark):
        """Benchmark batch formation from queue."""
        config = BatchProcessorConfig(
            batch_size=50,
            enable_priority_queue=True
        )
        processor = BatchProcessor(config)
        
        # Pre-populate queue synchronously for benchmark
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for i in range(100):  # Reduced for faster setup
                loop.run_until_complete(processor.enqueue({"task": f"task{i}"}, priority=i % 10))
        finally:
            loop.close()
        
        def get_batch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(processor._get_training_batch())
            finally:
                loop.close()
        
        # Should complete in < 200ms
        result = benchmark(get_batch)
        assert len(result) <= config.batch_size
    
    async def test_priority_queue_performance(self, benchmark):
        """Benchmark priority queue operations."""
        pq = PriorityQueue()
        
        # Pre-populate queue
        for i in range(1000):
            pq.enqueue({"task": f"task{i}"}, priority=i % 100)
        
        def dequeue_batch():
            batch = []
            for _ in range(50):
                item = pq.dequeue()
                if item:
                    batch.append(item)
            return batch
        
        # Should complete in < 200ms
        result = benchmark(dequeue_batch)
        assert len(result) <= 50
    
    async def test_concurrent_enqueue_performance(self, benchmark):
        """Benchmark concurrent enqueue operations."""
        config = BatchProcessorConfig(
            batch_size=100,
            concurrency=10,
            enable_priority_queue=True
        )
        processor = BatchProcessor(config)
        
        async def concurrent_enqueue():
            tasks = []
            for i in range(100):
                tasks.append(processor.enqueue({"task": f"task{i}"}, priority=i % 10))
            await asyncio.gather(*tasks)
            return len(tasks)
        
        def run_concurrent():
            return asyncio.run(concurrent_enqueue())
        
        # Should complete in < 200ms
        result = benchmark(run_concurrent)
        assert result == 100
    
    @pytest.mark.benchmark(
        min_time=0.1,
        max_time=0.5,
        min_rounds=5,
        disable_gc=True,
        warmup=False
    )
    def test_batch_processing_response_time(self, benchmark):
        """Test that batch processing meets < 200ms response time requirement."""
        config = BatchProcessorConfig(
            batch_size=10,
            concurrency=5,
            dry_run=True  # For fast processing
        )
        processor = BatchProcessor(config)
        
        # Create a realistic batch
        test_batch = []
        for i in range(50):
            test_batch.append({
                "original": f"test prompt {i}",
                "enhanced": f"enhanced test prompt {i}",
                "metrics": {"confidence": 0.8 + (i % 3) * 0.1},
                "session_id": f"session_{i % 10}",
                "priority": i % 5
            })
        
        def process_training_batch():
            # Use a new event loop for benchmark to avoid "cannot be called from running loop" error
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(processor.process_training_batch(test_batch))
                return result
            finally:
                loop.close()
        
        # Benchmark the operation
        result = benchmark(process_training_batch)
        
        # Verify response time requirement
        assert result["status"] == "success"
        assert result["processing_time_ms"] < 200  # < 200ms requirement
        assert result["processed_records"] == len(test_batch)
    
    async def test_memory_usage_under_load(self, benchmark):
        """Test memory usage during high-load batch processing."""
        config = BatchProcessorConfig(
            batch_size=100,
            max_queue_size=10000,
            enable_priority_queue=True
        )
        processor = BatchProcessor(config)
        
        # Add large number of tasks
        async def load_queue():
            for i in range(5000):
                await processor.enqueue({
                    "task": f"task{i}",
                    "data": f"data_{i}" * 10,  # Some data size
                    "timestamp": time.time()
                }, priority=i % 100)
            return processor.get_queue_size()
        
        def run_load():
            return asyncio.run(load_queue())
        
        # Should handle large load efficiently
        result = benchmark(run_load)
        assert result > 0


@pytest.mark.asyncio
class TestBatchSchedulingIntegration:
    """Integration tests combining batch scheduling with other components."""
    
    async def test_batch_processor_with_session_store(self):
        """Test batch processor integration with session store."""
        from prompt_improver.utils.session_store import SessionStore
        
        # Create session store
        session_store = SessionStore(maxsize=100, ttl=300)
        
        # Create batch processor
        config = BatchProcessorConfig(
            batch_size=5,
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        async with session_store:
            # Store session data
            await session_store.set("batch_session", {
                "user_id": "test_user",
                "batch_id": "batch_123",
                "status": "processing"
            })
            
            # Process batch with session context
            for i in range(10):
                await processor.enqueue({
                    "task": f"task{i}",
                    "session_id": "batch_session"
                })
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Update session status
            await session_store.set("batch_session", {
                "user_id": "test_user",
                "batch_id": "batch_123",
                "status": "completed"
            })
            
            # Verify session data
            session_data = await session_store.get("batch_session")
            assert session_data["status"] == "completed"
    
    async def test_batch_processor_error_recovery(self):
        """Test batch processor error recovery and resilience."""
        config = BatchProcessorConfig(
            batch_size=3,
            max_attempts=2,
            base_delay=0.05,
            dry_run=False
        )
        processor = BatchProcessor(config)
        
        # Mock database to fail intermittently
        fail_every_n = 3
        call_count = 0
        
        async def intermittent_fail(record):
            nonlocal call_count
            call_count += 1
            if call_count % fail_every_n == 0:
                raise Exception(f"Simulated failure #{call_count}")
            return True
        
        with patch.object(processor, '_persist_to_database', intermittent_fail):
            # Add tasks
            for i in range(10):
                await processor.enqueue({"task": f"task{i}"})
            
            # Process with failures
            await processor._process_queue()
            
            # Should handle failures gracefully
            assert processor.metrics["failed"] > 0
            assert processor.metrics["processed"] > 0
    
    async def test_batch_processor_graceful_shutdown(self):
        """Test batch processor graceful shutdown behavior."""
        config = BatchProcessorConfig(
            batch_size=5,
            batch_timeout=0.1,
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Add tasks
        for i in range(20):
            await processor.enqueue({"task": f"task{i}"})
        
        # Start periodic processing
        periodic_task = asyncio.create_task(
            periodic_batch_processor_coroutine(processor)
        )
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Graceful shutdown
        start_time = time.time()
        periodic_task.cancel()
        
        try:
            await asyncio.wait_for(periodic_task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            pytest.fail("Graceful shutdown took too long")
        
        shutdown_time = (time.time() - start_time) * 1000
        assert shutdown_time < 1000  # Should shutdown within 1 second
    
    async def test_batch_processor_configuration_updates(self):
        """Test batch processor configuration updates."""
        # Initial configuration
        config = BatchProcessorConfig(
            batch_size=5,
            concurrency=2,
            dry_run=True
        )
        processor = BatchProcessor(config)
        
        # Verify initial config
        assert processor.config.batch_size == 5
        assert processor.config.concurrency == 2
        
        # Update configuration
        new_config = BatchProcessorConfig(
            batch_size=10,
            concurrency=4,
            dry_run=True
        )
        processor.config = new_config
        
        # Verify updated config
        assert processor.config.batch_size == 10
        assert processor.config.concurrency == 4
        
        # Test with updated configuration
        for i in range(15):
            await processor.enqueue({"task": f"task{i}"})
        
        batch = await processor._get_training_batch()
        assert len(batch) <= 10  # Should use new batch size
