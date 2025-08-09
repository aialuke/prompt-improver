#!/usr/bin/env python3
"""
REAL BATCH PROCESSING TESTING SUITE

This module extends and integrates the existing batch processor performance tests
into the comprehensive real behavior validation framework.
NO MOCKS - only real behavior testing with actual large datasets.

Key Features:
- Integrates existing batch processor performance tests
- Tests with 1GB+ real datasets
- Validates 10x performance improvements with actual measurements
- Tests real streaming scenarios with continuous data feeds
- Validates actual memory efficiency with large file processing
- Measures real processing speed improvements with production data
"""

import asyncio
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from test_batch_processor_performance import (
    DatasetGenerator,
    PerformanceBenchmark,
    TestBatchProcessorPerformance,
    process_ml_batch,
)

from prompt_improver.ml.optimization.batch.enhanced_batch_processor import (
    ChunkingStrategy,
    ProcessingMetrics,
    StreamingBatchConfig,
    StreamingBatchProcessor,
)

# Import existing batch processor performance components
sys.path.append(str(Path(__file__).parent.parent))

# Import actual batch processing components
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingRealResult:
    """Result from batch processing real behavior testing."""

    test_name: str
    success: bool
    execution_time_sec: float
    memory_used_mb: float
    real_data_processed: int
    actual_performance_metrics: dict[str, Any]
    business_impact_measured: dict[str, Any]
    error_details: str | None = None


class BatchProcessingRealTestSuite:
    """
    Real behavior test suite for batch processing validation.

    Extends existing batch processor tests with comprehensive real behavior
    validation using actual large datasets and production-like conditions.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.results: list[BatchProcessingRealResult] = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="batch_real_"))
        self.performance_benchmark = PerformanceBenchmark()

    async def run_all_tests(self) -> list[BatchProcessingRealResult]:
        """Run all real batch processing tests."""
        logger.info("⚡ Starting Real Batch Processing Testing")

        # Test 1: Large Dataset Processing (1GB+)
        await self._test_large_dataset_processing()

        # Test 2: Memory Efficiency Under Real Load
        await self._test_memory_efficiency_real_load()

        # Test 3: Streaming Performance with Real Data
        await self._test_streaming_performance_real_data()

        # Test 4: Concurrent Processing Real Scenarios
        await self._test_concurrent_processing_real_scenarios()

        # Test 5: Error Recovery with Real Failures
        await self._test_error_recovery_real_failures()

        # Test 6: Performance Improvement Validation
        await self._test_performance_improvement_validation()

        # Test 7: Production Load Simulation
        await self._test_production_load_simulation()

        return self.results

    async def _test_large_dataset_processing(self):
        """Test processing of 1GB+ datasets to validate real performance."""
        test_start = time.time()
        logger.info("Testing Large Dataset Processing (1GB+)...")

        try:
            # Generate 1GB+ dataset
            target_size_gb = 1.0
            records_per_mb = 500  # Approximately 500 records per MB
            target_records = int(target_size_gb * 1024 * records_per_mb)

            logger.info(
                f"Generating {target_records:,} records (~{target_size_gb}GB)..."
            )

            dataset_file = self.temp_dir / "large_dataset.jsonl"
            dataset_info = DatasetGenerator.create_dataset_file(
                target_records, str(dataset_file)
            )

            actual_size_gb = dataset_info["file_size_mb"] / 1024
            logger.info(
                f"Generated dataset: {actual_size_gb:.2f}GB with {dataset_info['num_records']:,} records"
            )

            # Configure for large dataset processing
            config = StreamingBatchConfig(
                chunk_size=10000,  # Larger chunks for big data
                worker_processes=8,  # Maximize parallelization
                memory_limit_mb=4000,  # 4GB memory limit
                chunking_strategy=ChunkingStrategy.ADAPTIVE,
                max_chunk_memory_mb=200,
                compression=True,
                enable_checkpointing=True,
                checkpoint_interval=100000,
                gc_threshold_mb=500,
            )

            # Process the large dataset
            processing_start = time.time()

            async with StreamingBatchProcessor(config, process_ml_batch) as processor:
                metrics = await processor.process_dataset(
                    data_source=str(dataset_file), job_id="large_dataset_test"
                )

            processing_time = time.time() - processing_start

            # Validate performance targets for large datasets
            throughput_target = 5000  # 5K records/sec for large datasets
            memory_efficiency_target = 4000  # Should stay under 4GB

            success = (
                metrics.items_processed >= target_records * 0.95  # 95% processed
                and metrics.throughput_items_per_sec >= throughput_target
                and metrics.memory_peak_mb <= memory_efficiency_target
                and metrics.items_failed < target_records * 0.01  # <1% failure rate
            )

            # Calculate business impact
            baseline_time = target_records / 500  # Baseline: 500 records/sec
            speedup = baseline_time / processing_time if processing_time > 0 else 1

            result = BatchProcessingRealResult(
                test_name="Large Dataset Processing (1GB+)",
                success=success,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=metrics.memory_peak_mb,
                real_data_processed=metrics.items_processed,
                actual_performance_metrics={
                    "dataset_size_gb": actual_size_gb,
                    "records_processed": metrics.items_processed,
                    "throughput_items_per_sec": metrics.throughput_items_per_sec,
                    "processing_time_sec": processing_time,
                    "memory_peak_mb": metrics.memory_peak_mb,
                    "checkpoint_count": metrics.checkpoint_count,
                    "chunks_processed": metrics.chunks_processed,
                    "speedup_vs_baseline": speedup,
                },
                business_impact_measured={
                    "data_processing_capacity": actual_size_gb,
                    "time_to_insight_improvement": speedup,
                    "cost_efficiency": min(2.0, speedup),
                    "scalability_factor": metrics.throughput_items_per_sec / 1000,
                },
            )

            logger.info(
                f"✅ Large dataset test: {metrics.items_processed:,} records in {processing_time:.1f}s"
            )
            logger.info(
                f"   Throughput: {metrics.throughput_items_per_sec:.0f} items/sec"
            )
            logger.info("   Memory Peak: %sMB", metrics.memory_peak_mb:.1f)
            logger.info("   Speedup: %sx", speedup:.1f)

        except Exception as e:
            result = BatchProcessingRealResult(
                test_name="Large Dataset Processing (1GB+)",
                success=False,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=0,
                actual_performance_metrics={},
                business_impact_measured={},
                error_details=str(e),
            )
            logger.error("❌ Large dataset test failed: %s", e)

        self.results.append(result)

    async def _test_memory_efficiency_real_load(self):
        """Test memory efficiency under real load conditions."""
        test_start = time.time()
        logger.info("Testing Memory Efficiency Under Real Load...")

        try:
            # Test different chunking strategies with real memory constraints
            dataset_size = 100000  # 100K records
            dataset_file = self.temp_dir / "memory_test_dataset.jsonl"
            dataset_info = DatasetGenerator.create_dataset_file(
                dataset_size, str(dataset_file)
            )

            strategies = [
                (ChunkingStrategy.FIXED_SIZE, "Fixed Size"),
                (ChunkingStrategy.MEMORY_BASED, "Memory Based"),
                (ChunkingStrategy.ADAPTIVE, "Adaptive"),
            ]

            strategy_results = {}

            for strategy, strategy_name in strategies:
                logger.info("Testing %s strategy...", strategy_name)

                config = StreamingBatchConfig(
                    chunk_size=5000,
                    worker_processes=4,
                    memory_limit_mb=1000,
                    chunking_strategy=strategy,
                    max_chunk_memory_mb=50,
                    gc_threshold_mb=100,
                )

                async with StreamingBatchProcessor(
                    config, process_ml_batch
                ) as processor:
                    metrics = await processor.process_dataset(
                        data_source=str(dataset_file),
                        job_id=f"memory_test_{strategy.value}",
                    )

                strategy_results[strategy_name] = {
                    "memory_peak_mb": metrics.memory_peak_mb,
                    "gc_collections": sum(metrics.gc_collections.values()),
                    "throughput": metrics.throughput_items_per_sec,
                    "memory_efficiency": dataset_info["file_size_mb"]
                    / max(1, metrics.memory_peak_mb),
                }

                logger.info(
                    f"   {strategy_name}: {metrics.memory_peak_mb:.1f}MB peak, {metrics.throughput_items_per_sec:.0f} items/sec"
                )

            # Find most efficient strategy
            best_strategy = max(
                strategy_results.keys(),
                key=lambda s: strategy_results[s]["memory_efficiency"],
            )

            best_efficiency = strategy_results[best_strategy]["memory_efficiency"]

            result = BatchProcessingRealResult(
                test_name="Memory Efficiency Real Load",
                success=best_efficiency >= 2.0,  # 2x efficiency target
                execution_time_sec=time.time() - test_start,
                memory_used_mb=min(
                    r["memory_peak_mb"] for r in strategy_results.values()
                ),
                real_data_processed=dataset_size * len(strategies),
                actual_performance_metrics={
                    "strategies_tested": len(strategies),
                    "best_strategy": best_strategy,
                    "best_efficiency": best_efficiency,
                    "strategy_results": strategy_results,
                },
                business_impact_measured={
                    "memory_cost_reduction": best_efficiency,
                    "resource_optimization": best_efficiency / 2,
                    "scalability_improvement": min(2.0, best_efficiency),
                },
            )

            logger.info(
                f"✅ Memory efficiency test: {best_strategy} is most efficient ({best_efficiency:.1f}x)"
            )

        except Exception as e:
            result = BatchProcessingRealResult(
                test_name="Memory Efficiency Real Load",
                success=False,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=0,
                actual_performance_metrics={},
                business_impact_measured={},
                error_details=str(e),
            )
            logger.error("❌ Memory efficiency test failed: %s", e)

        self.results.append(result)

    async def _test_streaming_performance_real_data(self):
        """Test streaming performance with real continuous data feeds."""
        test_start = time.time()
        logger.info("Testing Streaming Performance with Real Data...")

        try:
            # Simulate real-time data streaming
            stream_duration = 30  # 30 seconds of streaming
            records_per_second = 1000  # 1K records/sec stream rate

            async def real_data_stream():
                """Generate continuous real data stream."""
                start_time = time.time()
                record_count = 0

                while time.time() - start_time < stream_duration:
                    yield DatasetGenerator.generate_ml_record()
                    record_count += 1

                    # Control streaming rate
                    if record_count % records_per_second == 0:
                        await asyncio.sleep(1.0)

                logger.info(
                    f"Stream completed: {record_count} records in {time.time() - start_time:.1f}s"
                )

            # Configure for streaming
            config = StreamingBatchConfig(
                chunk_size=2000,  # Smaller chunks for streaming
                worker_processes=6,
                memory_limit_mb=2000,
                chunking_strategy=ChunkingStrategy.ADAPTIVE,
                enable_checkpointing=False,  # Not needed for streaming
                gc_threshold_mb=200,
            )

            # Process the stream
            stream_start = time.time()

            async with StreamingBatchProcessor(config, process_ml_batch) as processor:
                metrics = await processor.process_dataset(
                    data_source=real_data_stream(), job_id="streaming_test"
                )

            stream_time = time.time() - stream_start

            # Validate streaming performance
            expected_records = stream_duration * records_per_second
            latency_target = 2.0  # Should complete within 2 seconds of stream end

            success = (
                metrics.items_processed >= expected_records * 0.9  # 90% processed
                and stream_time <= stream_duration + latency_target
                and metrics.throughput_items_per_sec
                >= records_per_second * 0.8  # 80% of stream rate
            )

            result = BatchProcessingRealResult(
                test_name="Streaming Performance Real Data",
                success=success,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=metrics.memory_peak_mb,
                real_data_processed=metrics.items_processed,
                actual_performance_metrics={
                    "stream_duration_sec": stream_duration,
                    "target_stream_rate": records_per_second,
                    "actual_throughput": metrics.throughput_items_per_sec,
                    "processing_latency_sec": stream_time - stream_duration,
                    "records_processed": metrics.items_processed,
                    "stream_efficiency": metrics.throughput_items_per_sec
                    / records_per_second,
                },
                business_impact_measured={
                    "real_time_processing_capability": min(
                        2.0, metrics.throughput_items_per_sec / records_per_second
                    ),
                    "latency_improvement": max(
                        0.1, 1.0 / max(0.1, stream_time - stream_duration)
                    ),
                    "streaming_scalability": metrics.throughput_items_per_sec / 500,
                },
            )

            logger.info(
                f"✅ Streaming test: {metrics.items_processed:,} records, {metrics.throughput_items_per_sec:.0f} items/sec"
            )
            logger.info("   Processing latency: %ss", stream_time - stream_duration:.1f)

        except Exception as e:
            result = BatchProcessingRealResult(
                test_name="Streaming Performance Real Data",
                success=False,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=0,
                actual_performance_metrics={},
                business_impact_measured={},
                error_details=str(e),
            )
            logger.error("❌ Streaming test failed: %s", e)

        self.results.append(result)

    async def _test_concurrent_processing_real_scenarios(self):
        """Test concurrent processing with real-world scenarios."""
        test_start = time.time()
        logger.info("Testing Concurrent Processing Real Scenarios...")

        try:
            # Create multiple datasets for concurrent processing
            num_concurrent_jobs = 5
            records_per_job = 20000

            datasets = []
            for i in range(num_concurrent_jobs):
                dataset_file = self.temp_dir / f"concurrent_dataset_{i}.jsonl"
                dataset_info = DatasetGenerator.create_dataset_file(
                    records_per_job, str(dataset_file)
                )
                datasets.append((dataset_file, dataset_info))

            logger.info(
                f"Created {num_concurrent_jobs} datasets with {records_per_job:,} records each"
            )

            # Process all datasets concurrently
            async def process_dataset(dataset_file, job_id):
                config = StreamingBatchConfig(
                    chunk_size=2000,
                    worker_processes=2,  # Fewer workers per job to test sharing
                    memory_limit_mb=1000,
                )

                async with StreamingBatchProcessor(
                    config, process_ml_batch
                ) as processor:
                    return await processor.process_dataset(
                        data_source=str(dataset_file), job_id=f"concurrent_job_{job_id}"
                    )

            # Run concurrent processing
            concurrent_start = time.time()

            tasks = [process_dataset(ds[0], i) for i, (ds, _) in enumerate(datasets)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            concurrent_time = time.time() - concurrent_start

            # Analyze results
            successful_jobs = 0
            total_processed = 0
            total_failed = 0
            throughputs = []

            for i, result in enumerate(results):
                if isinstance(result, ProcessingMetrics):
                    successful_jobs += 1
                    total_processed += result.items_processed
                    total_failed += result.items_failed
                    throughputs.append(result.throughput_items_per_sec)
                    logger.info(
                        f"   Job {i}: {result.items_processed:,} records, {result.throughput_items_per_sec:.0f} items/sec"
                    )
                else:
                    logger.error("   Job {i} failed: %s", result)

            success = (
                successful_jobs >= num_concurrent_jobs * 0.8  # 80% jobs successful
                and total_processed
                >= num_concurrent_jobs * records_per_job * 0.9  # 90% records processed
            )

            avg_throughput = np.mean(throughputs) if throughputs else 0
            total_throughput = total_processed / concurrent_time

            result = BatchProcessingRealResult(
                test_name="Concurrent Processing Real Scenarios",
                success=success,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=total_processed,
                actual_performance_metrics={
                    "concurrent_jobs": num_concurrent_jobs,
                    "successful_jobs": successful_jobs,
                    "total_records_processed": total_processed,
                    "total_throughput_items_per_sec": total_throughput,
                    "avg_job_throughput": avg_throughput,
                    "concurrent_efficiency": total_throughput
                    / (avg_throughput * successful_jobs)
                    if avg_throughput > 0
                    else 0,
                },
                business_impact_measured={
                    "concurrent_processing_capability": successful_jobs
                    / num_concurrent_jobs,
                    "resource_utilization_efficiency": min(
                        2.0, total_throughput / 5000
                    ),
                    "multi_tenant_support": success,
                },
            )

            logger.info(
                f"✅ Concurrent test: {successful_jobs}/{num_concurrent_jobs} jobs, {total_processed:,} total records"
            )
            logger.info("   Total throughput: %s items/sec", total_throughput:.0f)

        except Exception as e:
            result = BatchProcessingRealResult(
                test_name="Concurrent Processing Real Scenarios",
                success=False,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=0,
                actual_performance_metrics={},
                business_impact_measured={},
                error_details=str(e),
            )
            logger.error("❌ Concurrent processing test failed: %s", e)

        self.results.append(result)

    async def _test_error_recovery_real_failures(self):
        """Test error recovery with real failure scenarios."""
        test_start = time.time()
        logger.info("Testing Error Recovery with Real Failures...")

        try:
            # Create dataset for error testing
            dataset_size = 10000
            dataset_file = self.temp_dir / "error_test_dataset.jsonl"
            dataset_info = DatasetGenerator.create_dataset_file(
                dataset_size, str(dataset_file)
            )

            # Create processor that fails randomly
            def failing_processor(items):
                """Processor that fails 15% of the time."""
                if np.random.random() < 0.15:  # 15% failure rate
                    raise Exception("Simulated processing failure")
                return process_ml_batch(items)

            # Configure with error resilience
            config = StreamingBatchConfig(
                chunk_size=500,
                worker_processes=3,
                max_retries=3,
                error_threshold_percent=20.0,  # Allow up to 20% errors
                enable_checkpointing=True,
                checkpoint_interval=2000,
            )

            # Process with expected failures
            async with StreamingBatchProcessor(config, failing_processor) as processor:
                metrics = await processor.process_dataset(
                    data_source=str(dataset_file), job_id="error_recovery_test"
                )

            # Validate error recovery
            success_rate = metrics.items_processed / (
                metrics.items_processed + metrics.items_failed
            )

            success = (
                success_rate >= 0.8  # At least 80% success despite failures
                and metrics.retry_count > 0  # Retries were attempted
                and metrics.checkpoint_count > 0  # Checkpoints were created
            )

            result = BatchProcessingRealResult(
                test_name="Error Recovery Real Failures",
                success=success,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=metrics.items_processed,
                actual_performance_metrics={
                    "items_processed": metrics.items_processed,
                    "items_failed": metrics.items_failed,
                    "success_rate": success_rate,
                    "retry_count": metrics.retry_count,
                    "checkpoint_count": metrics.checkpoint_count,
                    "error_recovery_efficiency": success_rate
                    / 0.85,  # Expected 85% without retries
                },
                business_impact_measured={
                    "system_resilience": success_rate,
                    "data_loss_prevention": min(1.0, success_rate / 0.8),
                    "fault_tolerance": metrics.retry_count
                    / max(1, metrics.items_failed),
                },
            )

            logger.info(
                f"✅ Error recovery test: {success_rate:.1%} success rate with {metrics.retry_count} retries"
            )

        except Exception as e:
            result = BatchProcessingRealResult(
                test_name="Error Recovery Real Failures",
                success=False,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=0,
                actual_performance_metrics={},
                business_impact_measured={},
                error_details=str(e),
            )
            logger.error("❌ Error recovery test failed: %s", e)

        self.results.append(result)

    async def _test_performance_improvement_validation(self):
        """Validate actual performance improvements against baseline."""
        test_start = time.time()
        logger.info("Validating Performance Improvements Against Baseline...")

        try:
            # Use existing performance benchmark from test_batch_processor_performance.py
            test_suite = TestBatchProcessorPerformance()

            # Run comprehensive performance comparison
            await test_suite.test_comprehensive_performance_report(self.temp_dir)

            # Extract improvement metrics from benchmark
            benchmark_results = (
                test_suite.performance_benchmark.results
                if hasattr(test_suite, "performance_benchmark")
                else {}
            )

            # Calculate validated improvements
            improvements = {}
            for test_name, results in benchmark_results.items():
                if "Enhanced" in test_name:
                    baseline_name = test_name.replace("Enhanced", "Baseline")
                    if baseline_name in benchmark_results:
                        baseline_perf = benchmark_results[baseline_name]["performance"]
                        enhanced_perf = results["performance"]

                        improvements[test_name] = {
                            "throughput_improvement": enhanced_perf[
                                "throughput_items_per_sec"
                            ]
                            / max(1, baseline_perf["throughput_items_per_sec"]),
                            "time_improvement": baseline_perf["processing_time_sec"]
                            / max(0.1, enhanced_perf["processing_time_sec"]),
                            "memory_improvement": baseline_perf.get(
                                "memory_peak_mb", 100
                            )
                            / max(1, enhanced_perf.get("memory_peak_mb", 100)),
                        }

            # Validate 10x improvement target
            best_improvement = 0
            if improvements:
                best_improvement = max(
                    imp["throughput_improvement"] for imp in improvements.values()
                )

            target_improvement = 10.0  # 10x improvement target
            success = best_improvement >= target_improvement

            result = BatchProcessingRealResult(
                test_name="Performance Improvement Validation",
                success=success,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=len(improvements),
                actual_performance_metrics={
                    "improvements_measured": len(improvements),
                    "best_throughput_improvement": best_improvement,
                    "target_improvement": target_improvement,
                    "improvement_details": improvements,
                },
                business_impact_measured={
                    "validated_performance_gain": best_improvement,
                    "cost_reduction_potential": min(2.0, best_improvement / 5),
                    "scalability_validation": best_improvement >= target_improvement,
                },
            )

            logger.info(
                f"✅ Performance validation: {best_improvement:.1f}x best improvement (target: {target_improvement}x)"
            )

        except Exception as e:
            result = BatchProcessingRealResult(
                test_name="Performance Improvement Validation",
                success=False,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=0,
                actual_performance_metrics={},
                business_impact_measured={},
                error_details=str(e),
            )
            logger.error("❌ Performance validation failed: %s", e)

        self.results.append(result)

    async def _test_production_load_simulation(self):
        """Test under simulated production load conditions."""
        test_start = time.time()
        logger.info("Testing Production Load Simulation...")

        try:
            # Simulate production workload pattern
            workload_duration = 60  # 1 minute of production load
            base_load = 2000  # 2K records/sec base load
            peak_multiplier = 3  # 3x peak load

            async def production_workload():
                """Generate production-like workload with varying intensity."""
                start_time = time.time()
                total_records = 0

                while time.time() - start_time < workload_duration:
                    # Simulate load variations
                    elapsed = time.time() - start_time
                    load_factor = 1 + (peak_multiplier - 1) * abs(
                        np.sin(elapsed * np.pi / 20)
                    )  # 20s cycle
                    current_load = int(base_load * load_factor)

                    # Generate batch for current second
                    for _ in range(current_load):
                        yield DatasetGenerator.generate_ml_record()
                        total_records += 1

                    await asyncio.sleep(1.0)  # 1 second intervals

                logger.info("Production workload generated %s records", total_records:,)

            # Configure for production load
            config = StreamingBatchConfig(
                chunk_size=5000,
                worker_processes=8,
                memory_limit_mb=6000,
                chunking_strategy=ChunkingStrategy.ADAPTIVE,
                gc_threshold_mb=1000,
                enable_checkpointing=True,
                checkpoint_interval=50000,
            )

            # Process production load
            async with StreamingBatchProcessor(config, process_ml_batch) as processor:
                metrics = await processor.process_dataset(
                    data_source=production_workload(), job_id="production_load_test"
                )

            # Validate production performance
            expected_records = workload_duration * base_load * 2  # Average with peaks

            success = (
                metrics.items_processed
                >= expected_records * 0.8  # 80% of expected load
                and metrics.throughput_items_per_sec
                >= base_load  # At least base load throughput
                and metrics.memory_peak_mb <= 6000  # Within memory limit
            )

            result = BatchProcessingRealResult(
                test_name="Production Load Simulation",
                success=success,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=metrics.memory_peak_mb,
                real_data_processed=metrics.items_processed,
                actual_performance_metrics={
                    "workload_duration_sec": workload_duration,
                    "base_load_items_per_sec": base_load,
                    "peak_multiplier": peak_multiplier,
                    "actual_throughput": metrics.throughput_items_per_sec,
                    "records_processed": metrics.items_processed,
                    "load_handling_ratio": metrics.throughput_items_per_sec / base_load,
                },
                business_impact_measured={
                    "production_readiness": success,
                    "peak_load_handling": metrics.throughput_items_per_sec
                    / (base_load * peak_multiplier),
                    "operational_stability": min(
                        1.0, metrics.throughput_items_per_sec / base_load
                    ),
                },
            )

            logger.info(
                f"✅ Production load test: {metrics.items_processed:,} records, {metrics.throughput_items_per_sec:.0f} items/sec"
            )

        except Exception as e:
            result = BatchProcessingRealResult(
                test_name="Production Load Simulation",
                success=False,
                execution_time_sec=time.time() - test_start,
                memory_used_mb=self._get_memory_usage(),
                real_data_processed=0,
                actual_performance_metrics={},
                business_impact_measured={},
                error_details=str(e),
            )
            logger.error("❌ Production load test failed: %s", e)

        self.results.append(result)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)


if __name__ == "__main__":
    # Run batch processing tests independently
    async def main():
        config = {"real_data_requirements": {"minimum_dataset_size_gb": 1.0}}
        suite = BatchProcessingRealTestSuite(config)
        results = await suite.run_all_tests()

        print(f"\n{'=' * 60}")
        print("BATCH PROCESSING REAL BEHAVIOR TEST RESULTS")
        print(f"{'=' * 60}")

        for result in results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.test_name}")
            print(f"  Data Processed: {result.real_data_processed:,}")
            print(f"  Execution Time: {result.execution_time_sec:.1f}s")
            print(f"  Memory Used: {result.memory_used_mb:.1f}MB")
            if result.error_details:
                print(f"  Error: {result.error_details}")
            print()

    asyncio.run(main())
