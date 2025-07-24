#!/usr/bin/env python3
"""Benchmark script for validating 10x batch processing performance improvement.

This script runs comprehensive benchmarks to validate the enhanced batch processor
achieves 10x improvement in large dataset handling.
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_improver.ml.optimization.batch.batch_processor import BatchProcessor, BatchProcessorConfig
from src.prompt_improver.ml.optimization.batch.enhanced_batch_processor import (
    StreamingBatchProcessor, StreamingBatchConfig, ChunkingStrategy
)
from tests.test_batch_processor_performance import DatasetGenerator, PerformanceBenchmark


def process_ml_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of ML items."""
    import numpy as np
    
    results = []
    for item in items:
        # Simulate ML processing
        result = {
            "id": item.get("id"),
            "processed": True,
            "score": np.random.random(),
            "features": np.random.randn(10).tolist()
        }
        results.append(result)
    return results


async def benchmark_baseline(dataset_path: str, num_records: int) -> Dict[str, Any]:
    """Benchmark the baseline batch processor."""
    print(f"\n📊 Benchmarking BASELINE processor with {num_records:,} records...")
    
    config = BatchProcessorConfig(
        batch_size=1000,
        concurrency=4,
        timeout=60000
    )
    
    processor = BatchProcessor(config)
    
    # Load dataset into memory (baseline approach)
    print("Loading dataset into memory...")
    start_load = time.time()
    records = []
    
    with open(dataset_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
            
    load_time = time.time() - start_load
    print(f"Dataset loaded in {load_time:.2f}s")
    
    # Process
    print("Processing batches...")
    start_time = time.time()
    
    results = await processor.process_batch(records)
    
    processing_time = time.time() - start_time
    
    metrics = {
        "processor": "baseline",
        "num_records": num_records,
        "load_time_sec": load_time,
        "processing_time_sec": processing_time,
        "total_time_sec": load_time + processing_time,
        "throughput_items_per_sec": results["processed"] / processing_time,
        "items_processed": results["processed"],
        "items_failed": results["failed"]
    }
    
    print(f"✅ Baseline completed: {results['processed']:,} items in {processing_time:.2f}s")
    print(f"   Throughput: {metrics['throughput_items_per_sec']:.2f} items/sec")
    
    return metrics


async def benchmark_enhanced(dataset_path: str, num_records: int, config_preset: str = "balanced") -> Dict[str, Any]:
    """Benchmark the enhanced streaming processor."""
    print(f"\n🚀 Benchmarking ENHANCED processor with {num_records:,} records (preset: {config_preset})...")
    
    # Configuration presets
    presets = {
        "memory_efficient": StreamingBatchConfig(
            chunk_size=1000,
            max_chunk_memory_mb=50,
            worker_processes=4,
            memory_limit_mb=500,
            chunking_strategy=ChunkingStrategy.MEMORY_BASED,
            compression=True,
            enable_checkpointing=True
        ),
        "high_throughput": StreamingBatchConfig(
            chunk_size=10000,
            max_chunk_memory_mb=200,
            worker_processes=8,
            memory_limit_mb=2000,
            chunking_strategy=ChunkingStrategy.FIXED_SIZE,
            compression=False,
            prefetch_chunks=4
        ),
        "balanced": StreamingBatchConfig(
            chunk_size=5000,
            max_chunk_memory_mb=100,
            worker_processes=os.cpu_count() or 4,
            memory_limit_mb=1000,
            chunking_strategy=ChunkingStrategy.ADAPTIVE,
            compression=True,
            enable_checkpointing=True,
            checkpoint_interval=50000
        )
    }
    
    config = presets.get(config_preset, presets["balanced"])
    
    print(f"Configuration: {config.worker_processes} workers, {config.chunk_size} chunk size, "
          f"{config.memory_limit_mb}MB memory limit")
    
    start_time = time.time()
    
    async with StreamingBatchProcessor(config, process_ml_batch) as processor:
        metrics = await processor.process_dataset(
            data_source=dataset_path,
            job_id=f"benchmark_{config_preset}_{int(time.time())}"
        )
        
    total_time = time.time() - start_time
    
    result = {
        "processor": f"enhanced_{config_preset}",
        "num_records": num_records,
        "processing_time_sec": metrics.processing_time_ms / 1000,
        "total_time_sec": total_time,
        "throughput_items_per_sec": metrics.throughput_items_per_sec,
        "items_processed": metrics.items_processed,
        "items_failed": metrics.items_failed,
        "memory_peak_mb": metrics.memory_peak_mb,
        "chunks_processed": metrics.chunks_processed,
        "gc_collections": sum(metrics.gc_collections.values()),
        "checkpoint_count": metrics.checkpoint_count
    }
    
    print(f"✅ Enhanced completed: {metrics.items_processed:,} items in {total_time:.2f}s")
    print(f"   Throughput: {metrics.throughput_items_per_sec:.2f} items/sec")
    print(f"   Peak Memory: {metrics.memory_peak_mb:.2f} MB")
    print(f"   Chunks: {metrics.chunks_processed}")
    
    return result


async def run_comprehensive_benchmark(output_dir: Path, dataset_sizes: List[int]):
    """Run comprehensive benchmarks across multiple dataset sizes."""
    print("=" * 80)
    print("🔬 BATCH PROCESSING PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Dataset sizes: {[f'{s:,}' for s in dataset_sizes]}")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    for num_records in dataset_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {num_records:,} records")
        print(f"{'='*60}")
        
        # Generate dataset
        dataset_path = output_dir / f"dataset_{num_records}.jsonl"
        
        if not dataset_path.exists():
            print(f"Generating dataset with {num_records:,} records...")
            dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_path))
            print(f"Dataset created: {dataset_info['file_size_mb']:.2f} MB in {dataset_info['generation_time_sec']:.2f}s")
        else:
            print(f"Using existing dataset: {dataset_path}")
            dataset_info = {
                "num_records": num_records,
                "file_size_mb": dataset_path.stat().st_size / (1024 * 1024)
            }
        
        results = {
            "dataset": dataset_info,
            "benchmarks": {}
        }
        
        # Run baseline (only for smaller datasets)
        if num_records <= 100000:
            try:
                baseline_result = await benchmark_baseline(str(dataset_path), num_records)
                results["benchmarks"]["baseline"] = baseline_result
            except Exception as e:
                print(f"⚠️ Baseline failed: {e}")
                results["benchmarks"]["baseline"] = {"error": str(e)}
        
        # Run enhanced with different presets
        for preset in ["memory_efficient", "balanced", "high_throughput"]:
            try:
                enhanced_result = await benchmark_enhanced(str(dataset_path), num_records, preset)
                results["benchmarks"][f"enhanced_{preset}"] = enhanced_result
            except Exception as e:
                print(f"⚠️ Enhanced {preset} failed: {e}")
                results["benchmarks"][f"enhanced_{preset}"] = {"error": str(e)}
        
        all_results.append(results)
        
        # Calculate improvements
        if "baseline" in results["benchmarks"] and "enhanced_balanced" in results["benchmarks"]:
            baseline = results["benchmarks"]["baseline"]
            enhanced = results["benchmarks"]["enhanced_balanced"]
            
            if "throughput_items_per_sec" in baseline and "throughput_items_per_sec" in enhanced:
                improvement = enhanced["throughput_items_per_sec"] / baseline["throughput_items_per_sec"]
                print(f"\n🎯 Performance Improvement: {improvement:.2f}x")
    
    # Generate report
    report = generate_benchmark_report(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON results
    results_file = output_dir / f"benchmark_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Markdown report
    report_file = output_dir / f"benchmark_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Results saved to:")
    print(f"   - {results_file}")
    print(f"   - {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(report)
    
    return all_results


def generate_benchmark_report(results: List[Dict[str, Any]]) -> str:
    """Generate a markdown report from benchmark results."""
    lines = []
    lines.append("# Batch Processing Performance Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Executive Summary")
    
    # Find max improvement
    max_improvement = 0
    for result in results:
        benchmarks = result["benchmarks"]
        if "baseline" in benchmarks and "enhanced_balanced" in benchmarks:
            baseline = benchmarks["baseline"]
            enhanced = benchmarks["enhanced_balanced"]
            if "throughput_items_per_sec" in baseline and "throughput_items_per_sec" in enhanced:
                improvement = enhanced["throughput_items_per_sec"] / baseline["throughput_items_per_sec"]
                max_improvement = max(max_improvement, improvement)
    
    if max_improvement > 0:
        lines.append(f"\n**Maximum Performance Improvement: {max_improvement:.2f}x** 🚀")
    
    # Results by dataset size
    lines.append("\n## Results by Dataset Size")
    
    for result in results:
        dataset = result["dataset"]
        benchmarks = result["benchmarks"]
        
        lines.append(f"\n### Dataset: {dataset['num_records']:,} records ({dataset['file_size_mb']:.2f} MB)")
        
        # Create comparison table
        lines.append("\n| Processor | Throughput (items/sec) | Processing Time (sec) | Peak Memory (MB) |")
        lines.append("|-----------|------------------------|----------------------|------------------|")
        
        for name, bench in benchmarks.items():
            if "error" not in bench:
                throughput = bench.get("throughput_items_per_sec", 0)
                proc_time = bench.get("processing_time_sec", 0)
                memory = bench.get("memory_peak_mb", "N/A")
                
                lines.append(f"| {name} | {throughput:,.2f} | {proc_time:.2f} | {memory} |")
        
        # Calculate improvements
        if "baseline" in benchmarks and "enhanced_balanced" in benchmarks:
            baseline = benchmarks["baseline"]
            enhanced = benchmarks["enhanced_balanced"]
            
            if "throughput_items_per_sec" in baseline and "throughput_items_per_sec" in enhanced:
                improvement = enhanced["throughput_items_per_sec"] / baseline["throughput_items_per_sec"]
                time_reduction = (baseline["processing_time_sec"] - enhanced["processing_time_sec"]) / baseline["processing_time_sec"] * 100
                
                lines.append(f"\n**Performance Improvement: {improvement:.2f}x** (🕐 {time_reduction:.1f}% time reduction)")
    
    # Performance characteristics
    lines.append("\n## Performance Characteristics")
    lines.append("\n### Enhanced Processor Benefits:")
    lines.append("- ✅ **Streaming Processing**: No need to load entire dataset into memory")
    lines.append("- ✅ **Parallel Processing**: Utilizes multiple CPU cores efficiently")
    lines.append("- ✅ **Memory Efficiency**: Constant memory usage regardless of dataset size")
    lines.append("- ✅ **Checkpoint/Resume**: Can recover from interruptions")
    lines.append("- ✅ **Adaptive Chunking**: Optimizes chunk size based on data characteristics")
    
    # Recommendations
    lines.append("\n## Recommendations")
    lines.append("\n### When to use Enhanced Processor:")
    lines.append("- Datasets larger than 100MB")
    lines.append("- Memory-constrained environments")
    lines.append("- Long-running batch jobs that need checkpoint/resume")
    lines.append("- When processing throughput is critical")
    
    lines.append("\n### Configuration Guidelines:")
    lines.append("- **Small datasets (<10MB)**: Use `memory_efficient` preset")
    lines.append("- **Medium datasets (10MB-1GB)**: Use `balanced` preset")
    lines.append("- **Large datasets (>1GB)**: Use `high_throughput` preset")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark batch processing performance improvements"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results (default: ./benchmark_results)"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 10000, 100000],
        help="Dataset sizes to test (default: 1000 10000 100000)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller datasets"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        dataset_sizes = [1000, 5000, 10000]
    else:
        dataset_sizes = args.sizes
    
    output_dir = Path(args.output_dir)
    
    # Run benchmarks
    asyncio.run(run_comprehensive_benchmark(output_dir, dataset_sizes))


if __name__ == "__main__":
    main()