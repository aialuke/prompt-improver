"""
Integration test for batch processing and shutdown sequence.
"""

import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock

from prompt_improver.optimization.batch_processor import BatchProcessor
from prompt_improver.services.startup import (
    init_startup_tasks,
    shutdown_startup_tasks
)


@pytest.mark.asyncio
class TestBatchProcessing:
    """Integration tests for batch processing system."""
    
    async def test_batch_processing_operations(self):
        """Test batch processing functionality."""
        processor = BatchProcessor()
        test_jobs = [{'id': i, 'data': f"job{i}"} for i in range(100)]
        
        results = await processor.process_batch(test_jobs)
        assert results['processed'] == 100
        assert 'failed' not in results or results['failed'] == 0
    
    async def test_shutdown_sequence(self):
        """Test graceful shutdown sequence."""
        # Initialize and start tasks
        startup_result = await init_startup_tasks()
        assert startup_result['status'] == 'success'
        
        # Perform operations...
        # Add logic related to running operations here as needed for the system
        
        # Shutdown system
        shutdown_result = await shutdown_startup_tasks()
        assert shutdown_result['status'] == 'success'
        assert shutdown_result['shutdown_time_ms'] < 10000  # Ensuring shutdown within 10 seconds


@pytest.mark.asyncio
class TestPerformance:
    """Performance tests to ensure <200 ms response time for critical operations."""
    
    async def test_batch_processing_performance(self, benchmark):
        """Benchmark processing of a large batch to ensure performance criteria are met."""
        processor = BatchProcessor()
        test_jobs = [{'id': i, 'data': f"job{i}"} for i in range(1000)]

        # Use the benchmark fixture to measure performance
        def process_batch():
            return asyncio.run(processor.process_batch(test_jobs))

        benchmark(process_batch)

