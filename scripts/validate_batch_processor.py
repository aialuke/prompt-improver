#!/usr/bin/env python3
"""Quick validation script for batch processor enhancements."""

import asyncio
import json
import tempfile
import time
from pathlib import Path

# Test imports
try:
    from src.prompt_improver.ml.optimization.batch import (
        BatchProcessor,
        BatchProcessorConfig,
        StreamingBatchProcessor,
        StreamingBatchConfig,
        ChunkingStrategy
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def generate_test_data(num_records: int, file_path: str):
    """Generate test data."""
    with open(file_path, 'w') as f:
        for i in range(num_records):
            record = {
                "id": f"test_{i}",
                "data": f"test data {i}",
                "value": i * 0.1
            }
            f.write(json.dumps(record) + '\n')


def process_batch(items):
    """Simple processing function."""
    return [{"id": item.get("id"), "processed": True} for item in items]


async def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing Basic Functionality...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        
    try:
        # Generate test data
        num_records = 1000
        generate_test_data(num_records, test_file)
        print(f"  Generated {num_records} test records")
        
        # Test enhanced processor
        config = StreamingBatchConfig(
            chunk_size=100,
            worker_processes=2,
            chunking_strategy=ChunkingStrategy.FIXED_SIZE
        )
        
        async with StreamingBatchProcessor(config, process_batch) as processor:
            start_time = time.time()
            metrics = await processor.process_dataset(
                data_source=test_file,
                job_id="validation_test"
            )
            elapsed = time.time() - start_time
            
        print(f"  ✅ Processed {metrics.items_processed} items in {elapsed:.2f}s")
        print(f"  Throughput: {metrics.throughput_items_per_sec:.2f} items/sec")
        print(f"  Peak memory: {metrics.memory_peak_mb:.2f} MB")
        
        assert metrics.items_processed == num_records
        assert metrics.items_failed == 0
        
        print("✅ Basic functionality test passed!")
        
    finally:
        Path(test_file).unlink(missing_ok=True)


async def test_memory_efficiency():
    """Test memory efficiency."""
    print("\n🧪 Testing Memory Efficiency...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        
    try:
        # Generate larger dataset
        num_records = 10000
        generate_test_data(num_records, test_file)
        file_size_mb = Path(test_file).stat().st_size / (1024 * 1024)
        print(f"  Generated {num_records} records ({file_size_mb:.2f} MB)")
        
        # Test with memory constraints
        config = StreamingBatchConfig(
            chunk_size=500,
            max_chunk_memory_mb=10,
            worker_processes=2,
            memory_limit_mb=100,
            chunking_strategy=ChunkingStrategy.MEMORY_BASED,
            compression=True
        )
        
        async with StreamingBatchProcessor(config, process_batch) as processor:
            metrics = await processor.process_dataset(
                data_source=test_file,
                job_id="memory_test"
            )
            
        memory_efficiency = metrics.memory_peak_mb / file_size_mb
        print(f"  ✅ Peak memory: {metrics.memory_peak_mb:.2f} MB")
        print(f"  Memory efficiency: {memory_efficiency:.2f}x")
        print(f"  GC collections: {sum(metrics.gc_collections.values())}")
        
        assert metrics.memory_peak_mb < config.memory_limit_mb
        print("✅ Memory efficiency test passed!")
        
    finally:
        Path(test_file).unlink(missing_ok=True)


async def test_performance_improvement():
    """Test performance improvement vs baseline."""
    print("\n🧪 Testing Performance Improvement...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        
    try:
        # Generate dataset
        num_records = 5000
        generate_test_data(num_records, test_file)
        
        # Test baseline
        print("  Testing baseline processor...")
        baseline_config = BatchProcessorConfig(batch_size=100)
        baseline_processor = BatchProcessor(baseline_config)
        
        # Load all data for baseline
        records = []
        with open(test_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
                
        start_time = time.time()
        baseline_result = await baseline_processor.process_batch(records)
        baseline_time = time.time() - start_time
        baseline_throughput = baseline_result["processed"] / baseline_time
        
        print(f"  Baseline: {baseline_throughput:.2f} items/sec")
        
        # Test enhanced
        print("  Testing enhanced processor...")
        enhanced_config = StreamingBatchConfig(
            chunk_size=500,
            worker_processes=4,
            chunking_strategy=ChunkingStrategy.ADAPTIVE
        )
        
        async with StreamingBatchProcessor(enhanced_config, process_batch) as processor:
            metrics = await processor.process_dataset(
                data_source=test_file,
                job_id="performance_test"
            )
            
        print(f"  Enhanced: {metrics.throughput_items_per_sec:.2f} items/sec")
        
        improvement = metrics.throughput_items_per_sec / baseline_throughput
        print(f"  ✅ Performance improvement: {improvement:.2f}x")
        
        assert improvement > 1.5  # Should see at least 1.5x improvement
        print("✅ Performance improvement test passed!")
        
    finally:
        Path(test_file).unlink(missing_ok=True)


async def main():
    """Run all validation tests."""
    print("="*60)
    print("🔬 Batch Processor Enhancement Validation")
    print("="*60)
    
    try:
        await test_basic_functionality()
        await test_memory_efficiency()
        await test_performance_improvement()
        
        print("\n" + "="*60)
        print("✅ All validation tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())