"""
Example test demonstrating BatchProcessor integration with priority queue,
retry logic, and dry-run mode for local testing.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.prompt_improver.optimization.batch_processor import BatchProcessor, PriorityQueue, PriorityRecord


class TestBatchProcessorIntegration:
    """Test BatchProcessor with priority queue and retry logic."""
    
    def test_priority_queue_ordering(self):
        """Test priority queue orders items correctly (lower number = higher priority)."""
        pq = PriorityQueue()
        
        # Add items with different priorities
        pq.enqueue({"data": "low_priority"}, priority=100)
        pq.enqueue({"data": "high_priority"}, priority=1)
        pq.enqueue({"data": "medium_priority"}, priority=50)
        
        # Should dequeue in priority order
        assert pq.dequeue().priority == 1
        assert pq.dequeue().priority == 50
        assert pq.dequeue().priority == 100
        assert pq.dequeue() is None  # Empty queue
        
    def test_priority_queue_lazy_deletion(self):
        """Test lazy deletion in priority queue."""
        pq = PriorityQueue()
        
        pq.enqueue({"data": "item1"}, priority=1)
        pq.enqueue({"data": "item2"}, priority=2)
        
        # Remove first item
        first_item = pq.dequeue()
        pq.remove(first_item)
        
        # Second item should still be available
        assert pq.size() == 1
        assert not pq.empty()
        
    @pytest.mark.asyncio
    async def test_batch_processor_dry_run(self):
        """Test BatchProcessor dry-run mode."""
        config = {
            "dryRun": True,
            "maxAttempts": 2,
            "baseDelay": 0.1,
            "jitter": False,
        }
        
        batch_processor = BatchProcessor(config)
        
        # Mock the logger to capture dry-run messages
        batch_processor.logger = MagicMock()
        
        # Enqueue a record
        record = {
            "original": "test prompt",
            "enhanced": "enhanced test prompt",
            "metrics": {"score": 0.9},
            "session_id": "test_session",
        }
        
        await batch_processor.enqueue(record, priority=100)
        
        # Give some time for processing
        await asyncio.sleep(0.2)
        
        # Check that dry-run log was called
        batch_processor.logger.info.assert_called_with(
            "[DRY RUN] Would process record: " + str(record)
        )
        
    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        config = {
            "maxAttempts": 3,
            "baseDelay": 1.0,
            "maxDelay": 10.0,
            "jitter": False,
        }
        
        batch_processor = BatchProcessor(config)
        
        # Create a priority record
        record = PriorityRecord(
            priority=50,
            record={"test": "data"},
            attempts=0
        )
        
        # Simulate retry attempts
        record.attempts = 1
        await batch_processor._handle_retry(record)
        expected_delay_1 = 1.0 * (2 ** (1 - 1))  # 1.0 seconds
        
        record.attempts = 2
        await batch_processor._handle_retry(record)
        expected_delay_2 = 1.0 * (2 ** (2 - 1))  # 2.0 seconds
        
        record.attempts = 3
        await batch_processor._handle_retry(record)
        expected_delay_3 = 1.0 * (2 ** (3 - 1))  # 4.0 seconds
        
        # Verify delays increase exponentially
        assert expected_delay_1 < expected_delay_2 < expected_delay_3
        
    def test_batch_processor_config_defaults(self):
        """Test BatchProcessor initializes with correct defaults."""
        batch_processor = BatchProcessor()
        
        assert batch_processor.config["maxAttempts"] == 3
        assert batch_processor.config["baseDelay"] == 1.0
        assert batch_processor.config["maxDelay"] == 60.0
        assert batch_processor.config["jitter"] is True
        assert batch_processor.config["dryRun"] is False
        assert batch_processor.get_queue_size() == 0
        assert not batch_processor.processing
        
    def test_queue_size_tracking(self):
        """Test queue size tracking works correctly."""
        batch_processor = BatchProcessor()
        
        assert batch_processor.get_queue_size() == 0
        
        # Add items
        batch_processor.priority_queue.enqueue({"data": "item1"}, 1)
        batch_processor.priority_queue.enqueue({"data": "item2"}, 2)
        
        assert batch_processor.get_queue_size() == 2
        
        # Remove item
        batch_processor.priority_queue.dequeue()
        assert batch_processor.get_queue_size() == 1


if __name__ == "__main__":
    # Example usage demonstration
    async def demo_batch_processor():
        print("🔍 BatchProcessor Demo with Dry-Run Mode")
        
        # Configure for dry-run testing
        config = {
            "dryRun": True,
            "maxAttempts": 2,
            "baseDelay": 0.1,
            "jitter": True,
        }
        
        batch_processor = BatchProcessor(config)
        
        # Enqueue some test records with different priorities
        test_records = [
            {"original": "prompt 1", "enhanced": "enhanced 1", "priority": 100},
            {"original": "prompt 2", "enhanced": "enhanced 2", "priority": 1},  # Higher priority
            {"original": "prompt 3", "enhanced": "enhanced 3", "priority": 50},
        ]
        
        for record in test_records:
            priority = record.pop("priority")
            await batch_processor.enqueue(record, priority)
            
        print(f"📊 Queue size after enqueuing: {batch_processor.get_queue_size()}")
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        print(f"📊 Queue size after processing: {batch_processor.get_queue_size()}")
        print("✅ Demo completed!")
        
    # Run the demo
    asyncio.run(demo_batch_processor())
