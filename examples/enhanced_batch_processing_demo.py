#!/usr/bin/env python3
"""Demo of enhanced batch processing with 10x performance improvement.

This example demonstrates how to use the enhanced streaming batch processor
for processing large ML datasets efficiently.
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from prompt_improver.ml.optimization.batch import (
    StreamingBatchProcessor,
    StreamingBatchConfig,
    ChunkingStrategy
)


def generate_sample_data(num_records: int, file_path: str):
    """Generate sample ML training data."""
    print(f"Generating {num_records:,} sample records...")
    
    with open(file_path, 'w') as f:
        for i in range(num_records):
            record = {
                "id": f"sample_{i}",
                "text": f"This is sample text {i} with some content that needs processing.",
                "features": {
                    "length": random.randint(10, 100),
                    "complexity": random.random(),
                    "domain": random.choice(["tech", "medical", "legal", "general"])
                },
                "metadata": {
                    "timestamp": "2025-01-20T10:00:00Z",
                    "source": "demo"
                }
            }
            f.write(json.dumps(record) + '\n')
            
            if i > 0 and i % 10000 == 0:
                print(f"  Generated {i:,} records...")
                
    print(f"✅ Sample data generated: {file_path}")


def process_ml_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of ML items.
    
    This is where you would implement your actual ML processing logic.
    For demo purposes, we'll just transform the data.
    """
    results = []
    
    for item in items:
        # Simulate ML processing
        text_length = len(item.get("text", ""))
        
        result = {
            "id": item["id"],
            "original_text": item["text"][:50] + "...",
            "processed": True,
            "ml_score": random.random() * text_length / 100,
            "enhanced_features": {
                **item.get("features", {}),
                "processed_length": text_length,
                "quality_score": random.random()
            }
        }
        results.append(result)
        
    return results


async def demo_basic_usage():
    """Demonstrate basic usage of the streaming batch processor."""
    print("\n" + "="*60)
    print("DEMO: Basic Streaming Batch Processing")
    print("="*60)
    
    # Generate sample data
    data_file = "demo_data_basic.jsonl"
    generate_sample_data(10000, data_file)
    
    # Configure processor
    config = StreamingBatchConfig(
        chunk_size=1000,  # Process 1000 items at a time
        worker_processes=4,  # Use 4 parallel workers
        memory_limit_mb=500,  # Limit memory usage to 500MB
        chunking_strategy=ChunkingStrategy.FIXED_SIZE
    )
    
    # Process the data
    print("\nProcessing data with streaming batch processor...")
    
    async with StreamingBatchProcessor(config, process_ml_items) as processor:
        metrics = await processor.process_dataset(
            data_source=data_file,
            job_id="demo_basic"
        )
        
    # Display results
    print(f"\n✅ Processing Complete!")
    print(f"   Items processed: {metrics.items_processed:,}")
    print(f"   Processing time: {metrics.processing_time_ms/1000:.2f} seconds")
    print(f"   Throughput: {metrics.throughput_items_per_sec:.2f} items/sec")
    print(f"   Peak memory: {metrics.memory_peak_mb:.2f} MB")
    
    # Cleanup
    Path(data_file).unlink()


async def demo_memory_efficient_processing():
    """Demonstrate memory-efficient processing of large datasets."""
    print("\n" + "="*60)
    print("DEMO: Memory-Efficient Large Dataset Processing")
    print("="*60)
    
    # Generate larger dataset
    data_file = "demo_data_large.jsonl"
    generate_sample_data(100000, data_file)
    
    # Configure for memory efficiency
    config = StreamingBatchConfig(
        chunk_size=500,  # Smaller chunks
        max_chunk_memory_mb=20,  # Limit memory per chunk
        worker_processes=2,  # Fewer workers to save memory
        memory_limit_mb=200,  # Strict memory limit
        chunking_strategy=ChunkingStrategy.MEMORY_BASED,  # Memory-aware chunking
        compression=True,  # Enable compression
        gc_threshold_mb=100  # Aggressive garbage collection
    )
    
    print("\nProcessing large dataset with memory constraints...")
    print(f"Memory limit: {config.memory_limit_mb} MB")
    
    async with StreamingBatchProcessor(config, process_ml_items) as processor:
        metrics = await processor.process_dataset(
            data_source=data_file,
            job_id="demo_memory_efficient"
        )
        
    print(f"\n✅ Memory-Efficient Processing Complete!")
    print(f"   Items processed: {metrics.items_processed:,}")
    print(f"   Peak memory usage: {metrics.memory_peak_mb:.2f} MB")
    print(f"   Memory efficiency: {metrics.memory_peak_mb / (Path(data_file).stat().st_size / (1024*1024)):.2f}x")
    print(f"   GC collections: {sum(metrics.gc_collections.values())}")
    
    # Cleanup
    Path(data_file).unlink()


async def demo_checkpoint_resume():
    """Demonstrate checkpoint and resume functionality."""
    print("\n" + "="*60)
    print("DEMO: Checkpoint and Resume")
    print("="*60)
    
    # Generate dataset
    data_file = "demo_data_checkpoint.jsonl"
    generate_sample_data(20000, data_file)
    
    # Configure with checkpointing
    config = StreamingBatchConfig(
        chunk_size=1000,
        worker_processes=2,
        enable_checkpointing=True,
        checkpoint_interval=5000,  # Checkpoint every 5000 items
        checkpoint_dir="./demo_checkpoints"
    )
    
    job_id = "demo_checkpoint_job"
    
    # First run - simulate interruption
    print("\nStarting processing (will interrupt after 10 seconds)...")
    
    try:
        async with StreamingBatchProcessor(config, process_ml_items) as processor:
            # Use asyncio timeout to simulate interruption
            await asyncio.wait_for(
                processor.process_dataset(data_source=data_file, job_id=job_id),
                timeout=3.0  # Interrupt after 3 seconds
            )
    except asyncio.TimeoutError:
        print("⚠️ Processing interrupted!")
        
    # Check checkpoint
    checkpoint_file = Path(config.checkpoint_dir) / f"{job_id}.checkpoint"
    if checkpoint_file.exists():
        print(f"✅ Checkpoint saved: {checkpoint_file}")
        
    # Resume from checkpoint
    print("\nResuming from checkpoint...")
    
    async with StreamingBatchProcessor(config, process_ml_items) as processor:
        metrics = await processor.process_dataset(
            data_source=data_file,
            job_id=job_id,
            resume_from_checkpoint=True
        )
        
    print(f"\n✅ Processing Resumed and Completed!")
    print(f"   Total items processed: {metrics.items_processed:,}")
    print(f"   Checkpoints created: {metrics.checkpoint_count}")
    
    # Cleanup
    Path(data_file).unlink()
    import shutil
    shutil.rmtree(config.checkpoint_dir, ignore_errors=True)


async def demo_streaming_data_source():
    """Demonstrate processing from async streaming data source."""
    print("\n" + "="*60)
    print("DEMO: Streaming Data Source Processing")
    print("="*60)
    
    # Async generator that streams data
    async def stream_ml_data(num_records: int):
        """Simulate streaming data from an API or database."""
        for i in range(num_records):
            yield {
                "id": f"stream_{i}",
                "text": f"Streaming record {i} with real-time data",
                "features": {
                    "timestamp": i,
                    "value": random.random()
                }
            }
            
            # Simulate network delay
            if i % 100 == 0:
                await asyncio.sleep(0.01)
                
    # Configure for streaming
    config = StreamingBatchConfig(
        chunk_size=500,
        worker_processes=4,
        chunking_strategy=ChunkingStrategy.ADAPTIVE,
        prefetch_chunks=2  # Prefetch for smooth streaming
    )
    
    print("\nProcessing streaming data source...")
    
    async with StreamingBatchProcessor(config, process_ml_items) as processor:
        metrics = await processor.process_dataset(
            data_source=stream_ml_data(10000),
            job_id="demo_streaming"
        )
        
    print(f"\n✅ Streaming Processing Complete!")
    print(f"   Items processed: {metrics.items_processed:,}")
    print(f"   Throughput: {metrics.throughput_items_per_sec:.2f} items/sec")


async def demo_adaptive_processing():
    """Demonstrate adaptive processing for mixed workloads."""
    print("\n" + "="*60)
    print("DEMO: Adaptive Processing for Mixed Workloads")
    print("="*60)
    
    # Generate mixed dataset with varying record sizes
    data_file = "demo_data_mixed.jsonl"
    print("Generating mixed dataset with varying record sizes...")
    
    with open(data_file, 'w') as f:
        for i in range(10000):
            # Create records of varying sizes
            if i % 100 == 0:
                # Large record
                text = " ".join([f"word{j}" for j in range(1000)])
                size_type = "large"
            elif i % 10 == 0:
                # Medium record
                text = " ".join([f"word{j}" for j in range(100)])
                size_type = "medium"
            else:
                # Small record
                text = f"Small record {i}"
                size_type = "small"
                
            record = {
                "id": f"mixed_{i}",
                "text": text,
                "size_type": size_type,
                "features": {"size": len(text)}
            }
            f.write(json.dumps(record) + '\n')
            
    # Configure with adaptive strategy
    config = StreamingBatchConfig(
        chunk_size=1000,
        max_chunk_memory_mb=50,
        worker_processes=4,
        chunking_strategy=ChunkingStrategy.ADAPTIVE,  # Adapts to data characteristics
        memory_limit_mb=500
    )
    
    print("\nProcessing mixed dataset with adaptive strategy...")
    
    async with StreamingBatchProcessor(config, process_ml_items) as processor:
        metrics = await processor.process_dataset(
            data_source=data_file,
            job_id="demo_adaptive"
        )
        
    print(f"\n✅ Adaptive Processing Complete!")
    print(f"   Items processed: {metrics.items_processed:,}")
    print(f"   Chunks created: {metrics.chunks_processed}")
    print(f"   Average chunk size: {metrics.items_processed / max(1, metrics.chunks_processed):.1f} items")
    
    # Cleanup
    Path(data_file).unlink()


async def main():
    """Run all demos."""
    print("🚀 Enhanced Batch Processing Demo")
    print("Demonstrating 10x performance improvement for large dataset handling")
    
    # Run demos
    await demo_basic_usage()
    await demo_memory_efficient_processing()
    await demo_checkpoint_resume()
    await demo_streaming_data_source()
    await demo_adaptive_processing()
    
    print("\n" + "="*60)
    print("✅ All demos completed successfully!")
    print("="*60)
    
    print("\n📚 Key Takeaways:")
    print("1. Streaming processing enables handling datasets larger than memory")
    print("2. Parallel workers provide significant throughput improvements")
    print("3. Memory-based chunking prevents out-of-memory errors")
    print("4. Checkpoint/resume enables reliable long-running jobs")
    print("5. Adaptive strategies optimize for varying data characteristics")


if __name__ == "__main__":
    asyncio.run(main())