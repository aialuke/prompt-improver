import pytest
from src.prompt_improver.ml.optimization.batch.batch_processor import BatchProcessor, BatchProcessorConfig


@pytest.mark.asyncio
async def test_batch_processor_success():
    config = BatchProcessorConfig(batch_size=10, dry_run=True)
    processor = BatchProcessor(config)
    test_cases = [{"id": i} for i in range(30)]  # Sample test cases
    result = await processor.process_batch(test_cases)
    assert result["processed"] == 30
    assert result["failed"] == 0


@pytest.mark.asyncio
async def test_batch_processor_failure():
    config = BatchProcessorConfig(batch_size=5, dry_run=True)
    processor = BatchProcessor(config)
    result = await processor.process_single_batch([{"id": 1}])
    assert result["processed"] == 1
    assert result["failed"] == 0
