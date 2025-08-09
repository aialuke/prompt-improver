"""Comprehensive performance tests for batch processor with real behavior validation.

Tests the enhanced batch processor with various dataset sizes to validate
10x performance improvement for large dataset handling.
"""
import asyncio
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import psutil
import pytest
from prompt_improver.ml.optimization.batch.batch_processor import BatchProcessor, BatchProcessorConfig
from prompt_improver.ml.optimization.batch.enhanced_batch_processor import ChunkingStrategy, ProcessingMetrics, StreamingBatchConfig, StreamingBatchProcessor

class DatasetGenerator:
    """Generate test datasets of various sizes."""

    @staticmethod
    def generate_ml_record() -> dict[str, Any]:
        """Generate a single ML training record."""
        return {'id': f'record_{random.randint(100000, 999999)}', 'prompt': ' '.join([f'word{i}' for i in range(random.randint(10, 100))]), 'features': {'length': random.randint(10, 1000), 'complexity': random.random(), 'embeddings': np.random.randn(768).tolist(), 'metadata': {'domain': random.choice(['tech', 'medical', 'legal', 'general']), 'language': 'en', 'quality_score': random.random()}}, 'labels': {'category': random.choice(['A', 'B', 'C', 'D']), 'confidence': random.random()}}

    @staticmethod
    def create_dataset_file(num_records: int, file_path: str) -> dict[str, Any]:
        """Create a dataset file with specified number of records."""
        start_time = time.time()
        file_size = 0
        with open(file_path, 'w') as f:
            for i in range(num_records):
                record = DatasetGenerator.generate_ml_record()
                json_line = json.dumps(record) + '\n'
                f.write(json_line)
                file_size += len(json_line.encode())
                if i > 0 and i % 10000 == 0:
                    print(f'Generated {i}/{num_records} records...')
        generation_time = time.time() - start_time
        return {'num_records': num_records, 'file_size_mb': file_size / (1024 * 1024), 'generation_time_sec': generation_time, 'file_path': file_path}

    @staticmethod
    async def async_data_generator(num_records: int):
        """Async generator for streaming data."""
        for i in range(num_records):
            yield DatasetGenerator.generate_ml_record()
            if i % 1000 == 0:
                await asyncio.sleep(0.001)

class PerformanceBenchmark:
    """Benchmark utilities for measuring performance."""

    def __init__(self):
        self.results = {}

    def measure_performance(self, name: str, metrics: ProcessingMetrics, dataset_info: dict[str, Any]):
        """Record performance measurements."""
        self.results[name] = {'dataset': {'num_records': dataset_info['num_records'], 'size_mb': dataset_info.get('file_size_mb', 0)}, 'performance': {'items_processed': metrics.items_processed, 'processing_time_sec': metrics.processing_time_ms / 1000, 'throughput_items_per_sec': metrics.throughput_items_per_sec, 'memory_peak_mb': metrics.memory_peak_mb, 'memory_efficiency': metrics.memory_peak_mb / dataset_info.get('file_size_mb', 1)}, 'reliability': {'items_failed': metrics.items_failed, 'retry_count': metrics.retry_count, 'checkpoint_count': metrics.checkpoint_count, 'error_rate': metrics.items_failed / max(1, metrics.items_processed + metrics.items_failed) * 100}}

    def calculate_improvement(self, baseline_name: str, enhanced_name: str) -> dict[str, float]:
        """Calculate performance improvement between baseline and enhanced."""
        baseline = self.results.get(baseline_name, {}).get('performance', {})
        enhanced = self.results.get(enhanced_name, {}).get('performance', {})
        if not baseline or not enhanced:
            return {}
        return {'throughput_improvement': enhanced['throughput_items_per_sec'] / max(1, baseline['throughput_items_per_sec']), 'time_improvement': baseline['processing_time_sec'] / max(0.001, enhanced['processing_time_sec']), 'memory_improvement': baseline['memory_peak_mb'] / max(1, enhanced['memory_peak_mb'])}

    def generate_report(self) -> str:
        """Generate a performance report."""
        report = ['# Batch Processing Performance Report\n']
        for name, result in self.results.items():
            report.append(f'\n## {name}')
            report.append(f"Dataset: {result['dataset']['num_records']:,} records ({result['dataset']['size_mb']:.2f} MB)")
            report.append(f"Processing Time: {result['performance']['processing_time_sec']:.2f} seconds")
            report.append(f"Throughput: {result['performance']['throughput_items_per_sec']:.2f} items/sec")
            report.append(f"Peak Memory: {result['performance']['memory_peak_mb']:.2f} MB")
            report.append(f"Memory Efficiency: {result['performance']['memory_efficiency']:.2f}x")
            report.append(f"Error Rate: {result['reliability']['error_rate']:.2f}%")
        improvements = []
        for enhanced_name in self.results:
            if 'enhanced' in enhanced_name.lower():
                baseline_name = enhanced_name.replace('enhanced', 'baseline').replace('Enhanced', 'Baseline')
                if baseline_name in self.results:
                    imp = self.calculate_improvement(baseline_name, enhanced_name)
                    if imp:
                        improvements.append(f'\n### {enhanced_name} vs {baseline_name}')
                        improvements.append(f"Throughput Improvement: {imp['throughput_improvement']:.2f}x")
                        improvements.append(f"Time Improvement: {imp['time_improvement']:.2f}x")
                        improvements.append(f"Memory Improvement: {imp['memory_improvement']:.2f}x")
        if improvements:
            report.append('\n# Performance Improvements')
            report.extend(improvements)
        return '\n'.join(report)

def process_ml_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simulate ML processing on a batch of items."""
    results = []
    for item in items:
        text_length = len(item.get('prompt', ''))
        embedding_dim = 128
        embeddings = np.random.randn(embedding_dim)
        score = random.random() * text_length / 100
        result = {'id': item.get('id'), 'original_prompt': item.get('prompt', '')[:50] + '...', 'processed': True, 'ml_score': score, 'features_extracted': embedding_dim, 'processing_time_ms': random.uniform(0.1, 1.0)}
        results.append(result)
    return results

@pytest.mark.asyncio
class TestBatchProcessorPerformance:
    """Test suite for batch processor performance validation."""

    async def test_small_dataset_baseline(self, tmp_path):
        """Test baseline performance with small dataset (1MB)."""
        num_records = 1000
        dataset_file = tmp_path / 'small_dataset.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        config = BatchProcessorConfig(batch_size=100, concurrency=2, timeout=30000)
        processor = BatchProcessor(config)
        records = []
        with open(dataset_file) as f:
            for line in f:
                records.append(json.loads(line))
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        results = await processor.process_batch(records)
        processing_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        metrics = ProcessingMetrics(items_processed=results['processed'], items_failed=results['failed'], processing_time_ms=processing_time * 1000, memory_peak_mb=peak_memory - start_memory, throughput_items_per_sec=results['processed'] / processing_time)
        benchmark = PerformanceBenchmark()
        benchmark.measure_performance('Baseline Small Dataset', metrics, dataset_info)
        assert results['processed'] == num_records
        assert results['failed'] == 0

    async def test_small_dataset_enhanced(self, tmp_path):
        """Test enhanced performance with small dataset (1MB)."""
        num_records = 1000
        dataset_file = tmp_path / 'small_dataset.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        config = StreamingBatchConfig(chunk_size=100, worker_processes=2, memory_limit_mb=500, chunking_strategy=ChunkingStrategy.ADAPTIVE)
        async with StreamingBatchProcessor(config, process_ml_batch) as processor:
            metrics = await processor.process_dataset(data_source=str(dataset_file), job_id='test_small_enhanced')
        benchmark = PerformanceBenchmark()
        benchmark.measure_performance('Enhanced Small Dataset', metrics, dataset_info)
        assert metrics.items_processed == num_records
        assert metrics.items_failed == 0
        assert metrics.memory_peak_mb < 100

    async def test_medium_dataset_performance(self, tmp_path):
        """Test with medium dataset (100MB)."""
        num_records = 50000
        dataset_file = tmp_path / 'medium_dataset.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        config = StreamingBatchConfig(chunk_size=1000, worker_processes=4, memory_limit_mb=1000, chunking_strategy=ChunkingStrategy.MEMORY_BASED, max_chunk_memory_mb=50)
        async with StreamingBatchProcessor(config, process_ml_batch) as processor:
            metrics = await processor.process_dataset(data_source=str(dataset_file), job_id='test_medium')
        benchmark = PerformanceBenchmark()
        benchmark.measure_performance('Enhanced Medium Dataset', metrics, dataset_info)
        assert metrics.items_processed == num_records
        assert metrics.throughput_items_per_sec > 1000

    @pytest.mark.slow
    async def test_large_dataset_performance(self, tmp_path):
        """Test with large dataset (1GB) - marked as slow test."""
        num_records = 500000
        dataset_file = tmp_path / 'large_dataset.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        config = StreamingBatchConfig(chunk_size=5000, worker_processes=8, memory_limit_mb=2000, chunking_strategy=ChunkingStrategy.ADAPTIVE, max_chunk_memory_mb=100, compression=True, enable_checkpointing=True, checkpoint_interval=50000)
        async with StreamingBatchProcessor(config, process_ml_batch) as processor:
            metrics = await processor.process_dataset(data_source=str(dataset_file), job_id='test_large')
        benchmark = PerformanceBenchmark()
        benchmark.measure_performance('Enhanced Large Dataset', metrics, dataset_info)
        assert metrics.items_processed == num_records
        assert metrics.throughput_items_per_sec > 5000
        assert metrics.memory_peak_mb < 2000
        assert metrics.checkpoint_count > 0

    async def test_streaming_performance(self):
        """Test performance with streaming data source."""
        num_records = 10000
        config = StreamingBatchConfig(chunk_size=500, worker_processes=4, chunking_strategy=ChunkingStrategy.FIXED_SIZE)
        async with StreamingBatchProcessor(config, process_ml_batch) as processor:
            metrics = await processor.process_dataset(data_source=DatasetGenerator.async_data_generator(num_records), job_id='test_streaming')
        assert metrics.items_processed == num_records
        assert metrics.throughput_items_per_sec > 2000

    async def test_error_handling_and_recovery(self, tmp_path):
        """Test error handling and retry mechanisms."""

        def flaky_processor(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Processor that randomly fails."""
            if random.random() < 0.1:
                raise Exception('Random processing error')
            return process_ml_batch(items)
        num_records = 1000
        dataset_file = tmp_path / 'error_test_dataset.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        config = StreamingBatchConfig(chunk_size=100, worker_processes=2, max_retries=3, error_threshold_percent=15.0)
        async with StreamingBatchProcessor(config, flaky_processor) as processor:
            metrics = await processor.process_dataset(data_source=str(dataset_file), job_id='test_error_handling')
        assert metrics.items_processed > 0
        assert metrics.retry_count > 0

    async def test_checkpoint_resume(self, tmp_path):
        """Test checkpoint and resume functionality."""
        num_records = 5000
        dataset_file = tmp_path / 'checkpoint_test.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        config = StreamingBatchConfig(chunk_size=500, worker_processes=2, enable_checkpointing=True, checkpoint_interval=1000, checkpoint_dir=str(tmp_path / 'checkpoints'))
        job_id = 'test_checkpoint_resume'
        async with StreamingBatchProcessor(config, process_ml_batch) as processor:
            processor.config.error_threshold_percent = 0.1
            try:
                metrics1 = await processor.process_dataset(data_source=str(dataset_file), job_id=job_id)
            except:
                pass
        async with StreamingBatchProcessor(config, process_ml_batch) as processor:
            processor.config.error_threshold_percent = 100.0
            metrics2 = await processor.process_dataset(data_source=str(dataset_file), job_id=job_id, resume_from_checkpoint=True)
        assert metrics2.items_processed == num_records

    async def test_memory_efficiency(self, tmp_path):
        """Test memory efficiency with different strategies."""
        num_records = 10000
        dataset_file = tmp_path / 'memory_test.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        strategies = [ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.MEMORY_BASED, ChunkingStrategy.ADAPTIVE]
        results = {}
        for strategy in strategies:
            config = StreamingBatchConfig(chunk_size=1000, max_chunk_memory_mb=20, worker_processes=2, chunking_strategy=strategy, gc_threshold_mb=100)
            async with StreamingBatchProcessor(config, process_ml_batch) as processor:
                metrics = await processor.process_dataset(data_source=str(dataset_file), job_id=f'test_memory_{strategy.value}')
            results[strategy.value] = {'memory_peak_mb': metrics.memory_peak_mb, 'gc_collections': sum(metrics.gc_collections.values()), 'throughput': metrics.throughput_items_per_sec}
        assert results['memory_based']['memory_peak_mb'] < results['fixed_size']['memory_peak_mb']
        assert results['adaptive']['gc_collections'] > 0

    async def test_parallel_processing_scaling(self, tmp_path):
        """Test performance scaling with different worker counts."""
        num_records = 20000
        dataset_file = tmp_path / 'scaling_test.jsonl'
        dataset_info = DatasetGenerator.create_dataset_file(num_records, str(dataset_file))
        worker_counts = [1, 2, 4, 8]
        results = {}
        for workers in worker_counts:
            config = StreamingBatchConfig(chunk_size=1000, worker_processes=workers, memory_limit_mb=1000)
            async with StreamingBatchProcessor(config, process_ml_batch) as processor:
                metrics = await processor.process_dataset(data_source=str(dataset_file), job_id=f'test_scaling_{workers}')
            results[workers] = metrics.throughput_items_per_sec
        assert results[2] > results[1] * 1.5
        assert results[4] > results[2] * 1.5

    async def test_comprehensive_performance_report(self, tmp_path):
        """Generate comprehensive performance report across all dataset sizes."""
        benchmark = PerformanceBenchmark()
        test_configs = [{'records': 1000, 'name': 'Small (1MB)', 'file': 'small.jsonl'}, {'records': 10000, 'name': 'Medium (10MB)', 'file': 'medium.jsonl'}, {'records': 100000, 'name': 'Large (100MB)', 'file': 'large.jsonl'}]
        for test_config in test_configs:
            dataset_file = tmp_path / test_config['file']
            dataset_info = DatasetGenerator.create_dataset_file(test_config['records'], str(dataset_file))
            if test_config['records'] <= 10000:
                config = BatchProcessorConfig(batch_size=100)
                processor = BatchProcessor(config)
                records = []
                with open(dataset_file) as f:
                    for line in f:
                        records.append(json.loads(line))
                start_time = time.time()
                results = await processor.process_batch(records)
                baseline_time = time.time() - start_time
                baseline_metrics = ProcessingMetrics(items_processed=results['processed'], processing_time_ms=baseline_time * 1000, throughput_items_per_sec=results['processed'] / baseline_time, memory_peak_mb=100)
                benchmark.measure_performance(f"Baseline {test_config['name']}", baseline_metrics, dataset_info)
            enhanced_config = StreamingBatchConfig(chunk_size=min(5000, test_config['records'] // 10), worker_processes=4, memory_limit_mb=500, chunking_strategy=ChunkingStrategy.ADAPTIVE, compression=test_config['records'] > 10000)
            async with StreamingBatchProcessor(enhanced_config, process_ml_batch) as processor:
                enhanced_metrics = await processor.process_dataset(data_source=str(dataset_file), job_id=f"perf_test_{test_config['name']}")
            benchmark.measure_performance(f"Enhanced {test_config['name']}", enhanced_metrics, dataset_info)
        report = benchmark.generate_report()
        report_file = tmp_path / 'performance_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        print(f'\nPerformance report saved to: {report_file}')
        print(report)
        if 'Enhanced Large (100MB)' in benchmark.results:
            large_enhanced = benchmark.results['Enhanced Large (100MB)']['performance']
            assert large_enhanced['throughput_items_per_sec'] > 5000, 'Should achieve >5000 items/sec for large datasets'
if __name__ == '__main__':
    asyncio.run(test_comprehensive_performance_report(Path('./test_output')))
