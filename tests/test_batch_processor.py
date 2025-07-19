import pytest
from src.prompt_improver.optimization.batch_processor import BatchProcessor


@pytest.mark.asyncio
async def test_batch_processor_success():
    processor = BatchProcessor()
    test_cases = [{"id": i} for i in range(30)]  # Sample test cases
    result = await processor.process_batch(test_cases)
    assert result["processed"] == 30
    assert result["failed"] == 0


@pytest.mark.asyncio
async def test_batch_processor_failure():
    processor = BatchProcessor()
    result = await processor.process_single_batch([{"id": 1}])
    assert result["processed"] == 1
    assert result["failed"] == 0
