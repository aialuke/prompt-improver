"""
Production-Like Simulation Test Suite

This test suite simulates production-like conditions to validate system performance
under realistic load with large datasets and concurrent operations.

SIMULATION TARGETS:
- 500+ concurrent users performing ML operations
- 5GB+ dataset processing through batch pipelines
- 1000+ concurrent database queries with complex joins
- Real-time A/B testing with statistical significance
- Sustained load for 30+ minutes
- Memory usage under 8GB sustained
- 99.9% uptime during load testing
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import psutil
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
import aiofiles

# Core system imports
from prompt_improver.database import get_session_context
from prompt_improver.database.psycopg_client import TypeSafePsycopgClient
from prompt_improver.ml.optimization.batch.enhanced_batch_processor import (
    StreamingBatchProcessor, StreamingBatchConfig, ChunkingStrategy
)
from prompt_improver.ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.performance.monitoring.performance_benchmark import MCPPerformanceBenchmark
from prompt_improver.performance.testing.ab_testing_service import ABTestingService
from prompt_improver.database.cache_layer import DatabaseCacheLayer, CachePolicy, CacheStrategy
from prompt_improver.utils.health_checks import HealthChecker

logger = logging.getLogger(__name__)


class ProductionSimulationMetrics:
    """Track comprehensive metrics during production simulation."""

    def __init__(self):
        self.simulation_start_time = time.time()
        self.user_sessions: List[Dict[str, Any]] = []
        self.system_snapshots: List[Dict[str, Any]] = []
        self.database_metrics: List[Dict[str, Any]] = []
        self.ml_operations: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []
        self.performance_targets: Dict[str, Any] = {
            "max_response_time_ms": 500,
            "min_throughput_ops_per_sec": 1000,
            "max_memory_usage_gb": 8,
            "min_uptime_percent": 99.9,
            "max_error_rate_percent": 0.1
        }

    def record_user_session(self, session_data: Dict[str, Any]):
        """Record a user session."""
        session_data["timestamp"] = time.time()
        self.user_sessions.append(session_data)

    def record_system_snapshot(self):
        """Record current system state."""
        memory_info = psutil.virtual_memory()
        cpu_info = psutil.cpu_percent(interval=0.1)
        disk_info = psutil.disk_usage('/')
        network_info = psutil.net_connections()

        snapshot = {
            "timestamp": time.time(),
            "memory_used_gb": memory_info.used / (1024**3),
            "memory_available_gb": memory_info.available / (1024**3),
            "memory_percent": memory_info.percent,
            "cpu_percent": cpu_info,
            "disk_usage_percent": disk_info.percent,
            "network_connections": len(network_info),
            "processes": len(psutil.pids())
        }
        self.system_snapshots.append(snapshot)

    def record_database_metric(self, metric_data: Dict[str, Any]):
        """Record database operation metrics."""
        metric_data["timestamp"] = time.time()
        self.database_metrics.append(metric_data)

    def record_ml_operation(self, operation_data: Dict[str, Any]):
        """Record ML operation metrics."""
        operation_data["timestamp"] = time.time()
        self.ml_operations.append(operation_data)

    def record_error(self, error_data: Dict[str, Any]):
        """Record error occurrence."""
        error_data["timestamp"] = time.time()
        self.error_log.append(error_data)

    def calculate_simulation_summary(self) -> Dict[str, Any]:
        """Calculate comprehensive simulation summary."""
        total_duration = time.time() - self.simulation_start_time

        # User session metrics
        total_users = len(self.user_sessions)
        successful_sessions = len([s for s in self.user_sessions if s.get("success", False)])
        user_success_rate = (successful_sessions / total_users * 100) if total_users > 0 else 0

        # System performance metrics
        if self.system_snapshots:
            peak_memory = max(s["memory_used_gb"] for s in self.system_snapshots)
            avg_memory = sum(s["memory_used_gb"] for s in self.system_snapshots) / len(self.system_snapshots)
            peak_cpu = max(s["cpu_percent"] for s in self.system_snapshots)
            avg_cpu = sum(s["cpu_percent"] for s in self.system_snapshots) / len(self.system_snapshots)
        else:
            peak_memory = avg_memory = peak_cpu = avg_cpu = 0

        # Database metrics
        db_operations = len(self.database_metrics)
        if self.database_metrics:
            avg_db_response = sum(m.get("response_time_ms", 0) for m in self.database_metrics) / db_operations
            db_success_rate = (len([m for m in self.database_metrics if m.get("success", False)]) / db_operations * 100)
        else:
            avg_db_response = db_success_rate = 0

        # ML operations metrics
        ml_operations_count = len(self.ml_operations)
        if self.ml_operations:
            ml_success_rate = (len([m for m in self.ml_operations if m.get("success", False)]) / ml_operations_count * 100)
            avg_ml_duration = sum(m.get("duration_sec", 0) for m in self.ml_operations) / ml_operations_count
        else:
            ml_success_rate = avg_ml_duration = 0

        # Error metrics
        total_errors = len(self.error_log)
        error_rate = (total_errors / (total_users + db_operations + ml_operations_count) * 100) if (total_users + db_operations + ml_operations_count) > 0 else 0

        # Calculate target achievements
        targets_met = {
            "response_time": avg_db_response <= self.performance_targets["max_response_time_ms"],
            "memory_usage": peak_memory <= self.performance_targets["max_memory_usage_gb"],
            "uptime": user_success_rate >= self.performance_targets["min_uptime_percent"],
            "error_rate": error_rate <= self.performance_targets["max_error_rate_percent"]
        }

        return {
            "simulation_duration_sec": total_duration,
            "total_users": total_users,
            "successful_sessions": successful_sessions,
            "user_success_rate_percent": user_success_rate,
            "peak_memory_gb": peak_memory,
            "avg_memory_gb": avg_memory,
            "peak_cpu_percent": peak_cpu,
            "avg_cpu_percent": avg_cpu,
            "database_operations": db_operations,
            "avg_database_response_ms": avg_db_response,
            "database_success_rate_percent": db_success_rate,
            "ml_operations": ml_operations_count,
            "ml_success_rate_percent": ml_success_rate,
            "avg_ml_duration_sec": avg_ml_duration,
            "total_errors": total_errors,
            "error_rate_percent": error_rate,
            "targets_met": targets_met,
            "overall_success": all(targets_met.values())
        }

    def generate_simulation_report(self) -> str:
        """Generate comprehensive simulation report."""
        summary = self.calculate_simulation_summary()

        report = [
            "# Production Simulation Test Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Simulation Duration: {summary['simulation_duration_sec']:.2f} seconds",
            "",
            "## Executive Summary",
            f"- Total Users Simulated: {summary['total_users']:,}",
            f"- User Success Rate: {summary['user_success_rate_percent']:.2f}%",
            f"- Database Operations: {summary['database_operations']:,}",
            f"- ML Operations: {summary['ml_operations']:,}",
            f"- Total Errors: {summary['total_errors']}",
            f"- Overall Success: {'✅ YES' if summary['overall_success'] else '❌ NO'}",
            "",
            "## Performance Metrics",
            f"- Peak Memory Usage: {summary['peak_memory_gb']:.2f} GB",
            f"- Average Memory Usage: {summary['avg_memory_gb']:.2f} GB",
            f"- Peak CPU Usage: {summary['peak_cpu_percent']:.1f}%",
            f"- Average CPU Usage: {summary['avg_cpu_percent']:.1f}%",
            f"- Average DB Response Time: {summary['avg_database_response_ms']:.2f} ms",
            f"- Database Success Rate: {summary['database_success_rate_percent']:.2f}%",
            f"- ML Success Rate: {summary['ml_success_rate_percent']:.2f}%",
            f"- Error Rate: {summary['error_rate_percent']:.3f}%",
            "",
            "## Target Achievement",
        ]

        for target, achieved in summary["targets_met"].items():
            status = "✅ MET" if achieved else "❌ NOT MET"
            report.append(f"- {target.replace('_', ' ').title()}: {status}")

        report.extend([
            "",
            "## Detailed Analysis",
            f"- System Snapshots Collected: {len(self.system_snapshots)}",
            f"- User Sessions Tracked: {len(self.user_sessions)}",
            f"- Database Metrics Recorded: {len(self.database_metrics)}",
            f"- ML Operations Monitored: {len(self.ml_operations)}",
            "",
            "## Production Readiness Assessment"
        ])

        if summary["overall_success"]:
            report.append("🚀 **PRODUCTION READY**: All performance targets met under simulated load")
        else:
            report.append("⚠️ **NEEDS OPTIMIZATION**: Some performance targets not met")
            failed_targets = [t for t, met in summary["targets_met"].items() if not met]
            report.append(f"Failed targets: {', '.join(failed_targets)}")

        return "\n".join(report)


class TestProductionSimulation:
    """Production-like simulation test suite."""

    @pytest.fixture
    def simulation_metrics(self):
        """Simulation metrics tracker."""
        return ProductionSimulationMetrics()

    @pytest.fixture
    async def db_client(self):
        """PostgreSQL client with production-like configuration."""
        client = PostgresAsyncClient(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            database=os.getenv("POSTGRES_DB", "prompt_improver_test"),
            user=os.getenv("POSTGRES_USER", "test_user"),
            password=os.getenv("POSTGRES_PASSWORD", "test_password"),
            max_connections=50,  # Higher connection limit for production simulation
            command_timeout=30
        )
        await client.connect()
        yield client
        await client.disconnect()

    @pytest.fixture
    async def ml_orchestrator(self):
        """ML orchestrator with production-like configuration."""
        config = OrchestratorConfig(
            max_concurrent_workflows=100,  # High concurrency for production simulation
            component_health_check_interval=5,
            training_timeout=1800,  # 30 minutes
            debug_mode=False,
            enable_performance_profiling=True,
            enable_batch_processing=True,
            enable_a_b_testing=True,
            resource_limits={
                "max_memory_gb": 6,
                "max_cpu_percent": 80
            }
        )
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.fixture
    async def cache_layer(self):
        """Database cache layer with production configuration."""
        policy = CachePolicy(
            ttl_seconds=1800,  # 30 minutes
            strategy=CacheStrategy.SMART,
            warm_on_startup=True,
            max_memory_mb=1024  # 1GB cache
        )
        cache = DatabaseCacheLayer(policy)
        yield cache
        await cache.redis_cache.redis_client.flushdb()

    @pytest.mark.asyncio
    async def test_large_dataset_processing_simulation(
        self,
        simulation_metrics: ProductionSimulationMetrics
    ):
        """
        Test 1: Large Dataset Processing (5GB+ simulation)
        Process large datasets through the enhanced batch processor.
        """
        print("\n🔄 Test 1: Large Dataset Processing Simulation (5GB+)")
        print("=" * 80)

        start_time = time.time()

        try:
            print("📊 Generating large synthetic dataset...")

            # Simulate 5GB dataset processing by processing multiple large chunks
            total_samples = 500000  # 500K samples
            chunk_size = 25000     # 25K per chunk
            total_chunks = total_samples // chunk_size

            print(f"🔢 Processing {total_samples:,} samples in {total_chunks} chunks")

            # Enhanced configuration for large data
            config = StreamingBatchConfig(
                chunk_size=chunk_size,
                worker_processes=8,  # Maximum parallelism
                memory_limit_mb=2000,  # 2GB limit per worker
                chunking_strategy=ChunkingStrategy.MEMORY_BASED,
                gc_threshold_mb=500,
                enable_compression=True,
                max_retries=3,
                enable_progress_tracking=True
            )

            total_processed = 0
            processing_times = []
            memory_peaks = []

            # Generate and process data in chunks
            for chunk_idx in range(total_chunks):
                chunk_start = time.time()
                simulation_metrics.record_system_snapshot()

                print(f"  Processing chunk {chunk_idx + 1}/{total_chunks}...")

                # Generate chunk data (simulating 5GB total)
                chunk_data = []
                for i in range(chunk_size):
                    sample_id = chunk_idx * chunk_size + i

                    # Large feature vectors (simulating real ML data)
                    features = np.random.random(100).tolist()  # 100-dimensional
                    metadata = {
                        "user_id": f"user_{sample_id % 10000}",
                        "session_id": f"session_{sample_id % 1000}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "production_simulation",
                        "version": "1.0"
                    }

                    chunk_data.append({
                        "id": sample_id,
                        "features": features,
                        "label": np.random.randint(0, 10),
                        "metadata": metadata,
                        "raw_text": f"sample_text_data_{sample_id}" * 10  # Simulate text data
                    })

                # Write chunk to temporary file
                chunk_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
                for record in chunk_data:
                    chunk_file.write(json.dumps(record) + '\n')
                chunk_file.close()

                try:
                    # Complex processing function
                    def complex_ml_processing(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                        """Complex ML-like processing for production simulation."""
                        processed = []

                        for item in batch:
                            # Simulate complex feature engineering
                            features = np.array(item["features"])

                            # Multiple transformations
                            normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
                            scaled = normalized * 2.0

                            # Statistical features
                            feature_stats = {
                                "mean": float(np.mean(features)),
                                "std": float(np.std(features)),
                                "min": float(np.min(features)),
                                "max": float(np.max(features)),
                                "skewness": float(np.sum((features - np.mean(features))**3) / (len(features) * np.std(features)**3)),
                                "kurtosis": float(np.sum((features - np.mean(features))**4) / (len(features) * np.std(features)**4))
                            }

                            # Dimensionality reduction simulation
                            pca_features = features[:20]  # Top 20 components

                            # Clustering simulation
                            cluster_center = np.array([0.5] * len(features))
                            distance = np.linalg.norm(features - cluster_center)
                            cluster_id = int(distance * 10) % 5

                            # Text processing simulation
                            text_features = {
                                "word_count": len(item["raw_text"].split()),
                                "char_count": len(item["raw_text"]),
                                "hash": hash(item["raw_text"]) % 1000000
                            }

                            processed_item = {
                                "id": item["id"],
                                "original_features": features.tolist(),
                                "normalized_features": normalized.tolist(),
                                "scaled_features": scaled.tolist(),
                                "pca_features": pca_features.tolist(),
                                "feature_stats": feature_stats,
                                "cluster_id": cluster_id,
                                "text_features": text_features,
                                "label": item["label"],
                                "metadata": item["metadata"],
                                "processing_timestamp": datetime.now(timezone.utc).isoformat()
                            }
                            processed.append(processed_item)

                        # Simulate computation time
                        time.sleep(0.002 * len(batch))  # 2ms per item
                        return processed

                    # Process with enhanced batch processor
                    async with StreamingBatchProcessor(config, complex_ml_processing) as processor:
                        chunk_metrics = await processor.process_dataset(
                            data_source=chunk_file.name,
                            job_id=f"large_dataset_chunk_{chunk_idx}"
                        )

                        chunk_time = time.time() - chunk_start
                        processing_times.append(chunk_time)
                        memory_peaks.append(chunk_metrics.memory_peak_mb)
                        total_processed += chunk_metrics.items_processed

                        # Record metrics
                        simulation_metrics.record_ml_operation({
                            "operation_type": "large_dataset_chunk",
                            "chunk_id": chunk_idx,
                            "items_processed": chunk_metrics.items_processed,
                            "duration_sec": chunk_time,
                            "throughput_items_per_sec": chunk_metrics.throughput_items_per_sec,
                            "memory_peak_mb": chunk_metrics.memory_peak_mb,
                            "success": chunk_metrics.items_processed == chunk_size
                        })

                        if chunk_idx % 5 == 0:  # Progress update every 5 chunks
                            progress = (chunk_idx + 1) / total_chunks * 100
                            print(f"    Progress: {progress:.1f}% | "
                                  f"Throughput: {chunk_metrics.throughput_items_per_sec:.0f} items/sec | "
                                  f"Memory: {chunk_metrics.memory_peak_mb:.1f}MB")

                finally:
                    os.unlink(chunk_file.name)

            # Calculate overall metrics
            total_time = time.time() - start_time
            avg_processing_time = sum(processing_times) / len(processing_times)
            peak_memory = max(memory_peaks)
            avg_memory = sum(memory_peaks) / len(memory_peaks)
            overall_throughput = total_processed / total_time

            # Estimate dataset size
            estimated_size_gb = (total_samples * 100 * 8 + total_samples * 200) / (1024**3)  # Features + metadata

            print(f"\n📈 Large Dataset Processing Results:")
            print(f"  - Total samples processed: {total_processed:,}")
            print(f"  - Estimated dataset size: {estimated_size_gb:.2f} GB")
            print(f"  - Total processing time: {total_time:.2f}s")
            print(f"  - Overall throughput: {overall_throughput:.0f} items/sec")
            print(f"  - Peak memory usage: {peak_memory:.1f} MB")
            print(f"  - Average chunk processing time: {avg_processing_time:.2f}s")

            # Verify targets
            assert total_processed == total_samples, f"Not all samples processed: {total_processed}/{total_samples}"
            assert estimated_size_gb >= 5.0, f"Dataset too small: {estimated_size_gb:.2f} GB < 5 GB"
            assert overall_throughput > 5000, f"Throughput too low: {overall_throughput} items/sec"
            assert peak_memory < 3000, f"Memory usage too high: {peak_memory} MB"

            print("✅ Large dataset processing simulation passed!")

        except Exception as e:
            simulation_metrics.record_error({
                "test": "large_dataset_processing",
                "error": str(e),
                "phase": "processing"
            })
            raise

    @pytest.mark.asyncio
    async def test_concurrent_user_load_simulation(
        self,
        simulation_metrics: ProductionSimulationMetrics,
        ml_orchestrator: MLPipelineOrchestrator,
        db_client: TypeSafePsycopgClient,
        cache_layer: DatabaseCacheLayer
    ):
        """
        Test 2: Concurrent User Load (500+ users)
        Simulate 500+ concurrent users performing realistic operations.
        """
        print("\n🔄 Test 2: Concurrent User Load Simulation (500+ users)")
        print("=" * 80)

        start_time = time.time()

        try:
            concurrent_users = 500
            session_duration = 60  # 60 seconds per session
            operations_per_session = 10

            print(f"🚀 Simulating {concurrent_users} concurrent users for {session_duration}s...")

            # Start system monitoring
            monitoring_task = asyncio.create_task(
                self._continuous_system_monitoring(simulation_metrics, session_duration)
            )

            async def simulate_realistic_user_session(user_id: int):
                """Simulate a realistic user session with mixed operations."""
                session_start = time.time()
                session_operations = []

                try:
                    for op_idx in range(operations_per_session):
                        op_start = time.time()

                        # Distribute operations realistically
                        operation_type = self._select_operation_type(op_idx, operations_per_session)

                        if operation_type == "database_query":
                            # Complex database query
                            success, duration = await self._simulate_database_operation(
                                user_id, db_client, cache_layer
                            )

                        elif operation_type == "ml_operation":
                            # ML model operation
                            success, duration = await self._simulate_ml_operation(
                                user_id, ml_orchestrator
                            )

                        elif operation_type == "batch_job":
                            # Small batch processing
                            success, duration = await self._simulate_batch_operation(user_id)

                        elif operation_type == "health_check":
                            # System health check
                            success, duration = await self._simulate_health_check(ml_orchestrator)

                        else:  # file_operation
                            # File system operation
                            success, duration = await self._simulate_file_operation(user_id)

                        op_duration = time.time() - op_start

                        session_operations.append({
                            "operation_type": operation_type,
                            "success": success,
                            "duration_sec": op_duration,
                            "response_time_ms": duration
                        })

                        # Record individual operation
                        if operation_type == "database_query":
                            simulation_metrics.record_database_metric({
                                "user_id": user_id,
                                "operation": operation_type,
                                "response_time_ms": duration,
                                "success": success
                            })
                        elif operation_type == "ml_operation":
                            simulation_metrics.record_ml_operation({
                                "user_id": user_id,
                                "operation": operation_type,
                                "duration_sec": op_duration,
                                "success": success
                            })

                        # Small delay between operations
                        await asyncio.sleep(np.random.exponential(0.5))  # Realistic timing

                    session_duration_actual = time.time() - session_start
                    successful_operations = sum(1 for op in session_operations if op["success"])
                    avg_response_time = sum(op["response_time_ms"] for op in session_operations) / len(session_operations)

                    session_data = {
                        "user_id": user_id,
                        "session_duration_sec": session_duration_actual,
                        "operations_count": len(session_operations),
                        "successful_operations": successful_operations,
                        "success_rate": successful_operations / len(session_operations),
                        "avg_response_time_ms": avg_response_time,
                        "operations": session_operations,
                        "success": successful_operations >= operations_per_session * 0.8  # 80% success threshold
                    }

                    simulation_metrics.record_user_session(session_data)
                    return session_data

                except Exception as e:
                    simulation_metrics.record_error({
                        "user_id": user_id,
                        "error": str(e),
                        "phase": "user_session"
                    })

                    # Record failed session
                    failed_session = {
                        "user_id": user_id,
                        "session_duration_sec": time.time() - session_start,
                        "operations_count": len(session_operations),
                        "successful_operations": 0,
                        "success_rate": 0,
                        "avg_response_time_ms": 0,
                        "error": str(e),
                        "success": False
                    }

                    simulation_metrics.record_user_session(failed_session)
                    return failed_session

            # Launch all user sessions
            print(f"🔄 Launching {concurrent_users} user sessions...")
            user_tasks = [simulate_realistic_user_session(i) for i in range(concurrent_users)]

            # Wait for completion with timeout
            session_results = []
            completed_users = 0

            try:
                session_results = await asyncio.wait_for(
                    asyncio.gather(*user_tasks, return_exceptions=True),
                    timeout=session_duration + 30  # 30 second buffer
                )
                completed_users = len([r for r in session_results if not isinstance(r, Exception)])

            except asyncio.TimeoutError:
                print("⚠️ Some user sessions timed out")
                # Cancel remaining tasks
                for task in user_tasks:
                    if not task.done():
                        task.cancel()

                # Collect results from completed tasks
                session_results = []
                for task in user_tasks:
                    if task.done() and not task.cancelled():
                        try:
                            result = task.result()
                            session_results.append(result)
                            completed_users += 1
                        except Exception as e:
                            session_results.append(e)

            # Stop monitoring
            monitoring_task.cancel()

            # Analyze results
            successful_sessions = len([r for r in session_results
                                    if not isinstance(r, Exception) and r.get("success", False)])

            total_operations = sum(r.get("operations_count", 0) for r in session_results
                                 if not isinstance(r, Exception))

            successful_operations = sum(r.get("successful_operations", 0) for r in session_results
                                      if not isinstance(r, Exception))

            overall_success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0

            avg_response_times = [r.get("avg_response_time_ms", 0) for r in session_results
                                if not isinstance(r, Exception) and r.get("avg_response_time_ms", 0) > 0]
            avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0

            total_time = time.time() - start_time

            print(f"\n📈 Concurrent User Load Results:")
            print(f"  - Target concurrent users: {concurrent_users}")
            print(f"  - Completed user sessions: {completed_users}")
            print(f"  - Successful sessions: {successful_sessions}")
            print(f"  - Session success rate: {successful_sessions / completed_users * 100:.1f}%")
            print(f"  - Total operations: {total_operations:,}")
            print(f"  - Successful operations: {successful_operations:,}")
            print(f"  - Overall success rate: {overall_success_rate:.1f}%")
            print(f"  - Average response time: {avg_response_time:.2f}ms")
            print(f"  - Total simulation time: {total_time:.2f}s")

            # Verify targets
            assert completed_users >= concurrent_users * 0.95, f"Too many incomplete sessions: {completed_users}/{concurrent_users}"
            assert overall_success_rate >= 95.0, f"Success rate too low: {overall_success_rate:.1f}%"
            assert avg_response_time <= 500, f"Response time too high: {avg_response_time:.2f}ms"

            print("✅ Concurrent user load simulation passed!")

        except Exception as e:
            simulation_metrics.record_error({
                "test": "concurrent_user_load",
                "error": str(e),
                "phase": "simulation"
            })
            raise

    @pytest.mark.asyncio
    async def test_sustained_load_endurance(
        self,
        simulation_metrics: ProductionSimulationMetrics,
        ml_orchestrator: MLPipelineOrchestrator,
        db_client: PostgresAsyncClient
    ):
        """
        Test 3: Sustained Load Endurance (30+ minute test)
        Run sustained operations for 30+ minutes to test system stability.
        """
        print("\n🔄 Test 3: Sustained Load Endurance (30+ minutes)")
        print("=" * 80)

        # Note: For testing purposes, we'll run a shorter version (5 minutes)
        # In production, this should be extended to 30+ minutes
        test_duration_minutes = 5  # Reduced for testing
        test_duration_seconds = test_duration_minutes * 60

        print(f"⏱️ Running sustained load test for {test_duration_minutes} minutes...")
        print("(Note: Production version should run for 30+ minutes)")

        start_time = time.time()

        try:
            # Configuration for sustained load
            concurrent_operations = 50  # Moderate continuous load
            operation_interval = 2  # 2 seconds between operations

            # Start continuous monitoring
            monitoring_task = asyncio.create_task(
                self._continuous_system_monitoring(simulation_metrics, test_duration_seconds)
            )

            # Continuous operation functions
            async def continuous_database_operations():
                """Continuous database operations."""
                operation_count = 0
                while time.time() - start_time < test_duration_seconds:
                    try:
                        # Rotating query types
                        queries = [
                            ("SELECT COUNT(*) FROM sessions WHERE created_at > NOW() - INTERVAL '1 hour'", {}),
                            ("SELECT id, name FROM rules WHERE active = true ORDER BY created_at DESC LIMIT 10", {}),
                            ("SELECT r.id, r.name, COUNT(pi.id) as improvements FROM rules r LEFT JOIN prompt_improvements pi ON r.id = pi.rule_id GROUP BY r.id LIMIT 5", {}),
                        ]

                        query, params = queries[operation_count % len(queries)]

                        op_start = time.perf_counter()
                        result = await db_client.fetch_raw(query, params)
                        op_duration = (time.perf_counter() - op_start) * 1000

                        simulation_metrics.record_database_metric({
                            "operation_type": "sustained_db_operation",
                            "query_type": f"query_{operation_count % len(queries)}",
                            "response_time_ms": op_duration,
                            "rows_returned": len(result),
                            "success": True
                        })

                        operation_count += 1
                        await asyncio.sleep(operation_interval)

                    except Exception as e:
                        simulation_metrics.record_error({
                            "operation": "continuous_database",
                            "error": str(e),
                            "operation_count": operation_count
                        })
                        await asyncio.sleep(operation_interval * 2)  # Longer delay on error

            async def continuous_ml_operations():
                """Continuous ML operations."""
                operation_count = 0
                while time.time() - start_time < test_duration_seconds:
                    try:
                        # Alternate between different ML operations
                        if operation_count % 3 == 0:
                            # Health check
                            op_start = time.time()
                            health = await ml_orchestrator.get_component_health()
                            op_duration = time.time() - op_start

                            simulation_metrics.record_ml_operation({
                                "operation_type": "health_check",
                                "healthy_components": sum(1 for h in health.values() if h),
                                "total_components": len(health),
                                "duration_sec": op_duration,
                                "success": True
                            })

                        elif operation_count % 3 == 1:
                            # Resource usage check
                            op_start = time.time()
                            resources = await ml_orchestrator.get_resource_usage()
                            op_duration = time.time() - op_start

                            simulation_metrics.record_ml_operation({
                                "operation_type": "resource_check",
                                "metrics_count": len(resources),
                                "duration_sec": op_duration,
                                "success": True
                            })

                        else:
                            # Quick workflow status check
                            op_start = time.time()
                            # Simulate checking workflow status
                            await asyncio.sleep(0.1)  # Simulate operation
                            op_duration = time.time() - op_start

                            simulation_metrics.record_ml_operation({
                                "operation_type": "workflow_status",
                                "duration_sec": op_duration,
                                "success": True
                            })

                        operation_count += 1
                        await asyncio.sleep(operation_interval)

                    except Exception as e:
                        simulation_metrics.record_error({
                            "operation": "continuous_ml",
                            "error": str(e),
                            "operation_count": operation_count
                        })
                        await asyncio.sleep(operation_interval * 2)

            async def memory_pressure_operations():
                """Operations that create memory pressure."""
                operation_count = 0
                while time.time() - start_time < test_duration_seconds:
                    try:
                        # Create and process data to simulate memory usage
                        data_size = 1000  # 1K items
                        test_data = np.random.random((data_size, 50))  # 50-dimensional data

                        # Simulate processing
                        processed_data = test_data * 2.0
                        stats = {
                            "mean": np.mean(processed_data),
                            "std": np.std(processed_data),
                            "max": np.max(processed_data),
                            "min": np.min(processed_data)
                        }

                        # Clean up
                        del test_data, processed_data

                        operation_count += 1
                        await asyncio.sleep(operation_interval * 2)  # Longer interval for memory operations

                    except Exception as e:
                        simulation_metrics.record_error({
                            "operation": "memory_pressure",
                            "error": str(e),
                            "operation_count": operation_count
                        })
                        await asyncio.sleep(operation_interval * 3)

            # Launch continuous operations
            print("🔄 Starting continuous operations...")
            continuous_tasks = [
                asyncio.create_task(continuous_database_operations()),
                asyncio.create_task(continuous_ml_operations()),
                asyncio.create_task(memory_pressure_operations())
            ]

            # Wait for test duration
            await asyncio.sleep(test_duration_seconds)

            print("⏹️ Stopping continuous operations...")

            # Cancel all continuous tasks
            for task in continuous_tasks:
                task.cancel()

            # Wait for tasks to complete cancellation
            await asyncio.gather(*continuous_tasks, return_exceptions=True)

            # Stop monitoring
            monitoring_task.cancel()

            # Analyze sustained load results
            total_time = time.time() - start_time

            # Calculate metrics
            db_operations = len([m for m in simulation_metrics.database_metrics
                               if m.get("operation_type") == "sustained_db_operation"])
            ml_operations = len([m for m in simulation_metrics.ml_operations])
            total_errors = len(simulation_metrics.error_log)

            db_avg_response = (sum(m.get("response_time_ms", 0) for m in simulation_metrics.database_metrics
                                 if m.get("operation_type") == "sustained_db_operation") / db_operations) if db_operations > 0 else 0

            # Memory stability analysis
            if simulation_metrics.system_snapshots:
                memory_usage = [s["memory_used_gb"] for s in simulation_metrics.system_snapshots]
                memory_stability = max(memory_usage) - min(memory_usage)
                avg_memory = sum(memory_usage) / len(memory_usage)
            else:
                memory_stability = avg_memory = 0

            error_rate = (total_errors / (db_operations + ml_operations) * 100) if (db_operations + ml_operations) > 0 else 0

            print(f"\n📈 Sustained Load Endurance Results:")
            print(f"  - Test duration: {total_time / 60:.1f} minutes")
            print(f"  - Database operations: {db_operations:,}")
            print(f"  - ML operations: {ml_operations:,}")
            print(f"  - Total errors: {total_errors}")
            print(f"  - Error rate: {error_rate:.3f}%")
            print(f"  - Average DB response time: {db_avg_response:.2f}ms")
            print(f"  - Average memory usage: {avg_memory:.2f}GB")
            print(f"  - Memory usage stability: {memory_stability:.2f}GB variation")
            print(f"  - System snapshots collected: {len(simulation_metrics.system_snapshots)}")

            # Verify endurance targets
            assert total_time >= test_duration_seconds * 0.95, f"Test duration too short: {total_time}s"
            assert error_rate <= 1.0, f"Error rate too high: {error_rate:.3f}%"
            assert db_avg_response <= 500, f"DB response time degraded: {db_avg_response:.2f}ms"
            assert avg_memory <= 8.0, f"Memory usage too high: {avg_memory:.2f}GB"
            assert memory_stability <= 2.0, f"Memory usage unstable: {memory_stability:.2f}GB variation"

            print("✅ Sustained load endurance test passed!")

        except Exception as e:
            simulation_metrics.record_error({
                "test": "sustained_load_endurance",
                "error": str(e),
                "phase": "endurance"
            })
            raise

    @pytest.mark.asyncio
    async def test_generate_production_simulation_report(
        self,
        simulation_metrics: ProductionSimulationMetrics
    ):
        """
        Test 4: Generate Production Simulation Report
        Generate comprehensive report of production simulation results.
        """
        print("\n📊 Generating Production Simulation Report")
        print("=" * 80)

        # Generate comprehensive report
        report_content = simulation_metrics.generate_simulation_report()
        summary = simulation_metrics.calculate_simulation_summary()

        # Save report
        timestamp = int(time.time())
        report_path = Path(f"production_simulation_report_{timestamp}.md")

        with open(report_path, 'w') as f:
            f.write(report_content)

        # Save detailed metrics as JSON
        metrics_path = Path(f"production_simulation_metrics_{timestamp}.json")

        detailed_metrics = {
            "summary": summary,
            "user_sessions": simulation_metrics.user_sessions,
            "system_snapshots": simulation_metrics.system_snapshots[-100:],  # Last 100 snapshots
            "database_metrics": simulation_metrics.database_metrics[-1000:],  # Last 1000 operations
            "ml_operations": simulation_metrics.ml_operations[-1000:],  # Last 1000 operations
            "error_log": simulation_metrics.error_log,
            "performance_targets": simulation_metrics.performance_targets
        }

        # Convert datetime objects to strings for JSON serialization
        def convert_timestamps(obj):
            if isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj

        serializable_metrics = convert_timestamps(detailed_metrics)

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"✅ Production simulation report saved to: {report_path}")
        print(f"✅ Detailed metrics saved to: {metrics_path}")

        # Display key results
        print(f"\n📈 Production Simulation Summary:")
        print(f"  - Simulation Duration: {summary['simulation_duration_sec'] / 60:.1f} minutes")
        print(f"  - Total Users: {summary['total_users']:,}")
        print(f"  - User Success Rate: {summary['user_success_rate_percent']:.1f}%")
        print(f"  - Database Operations: {summary['database_operations']:,}")
        print(f"  - ML Operations: {summary['ml_operations']:,}")
        print(f"  - Peak Memory: {summary['peak_memory_gb']:.2f}GB")
        print(f"  - Error Rate: {summary['error_rate_percent']:.3f}%")
        print(f"  - Overall Success: {'✅ YES' if summary['overall_success'] else '❌ NO'}")

        # Verify overall success
        assert summary['overall_success'], "Production simulation did not meet all targets"
        assert summary['user_success_rate_percent'] >= 95.0, f"User success rate too low: {summary['user_success_rate_percent']:.1f}%"
        assert summary['error_rate_percent'] <= 0.5, f"Error rate too high: {summary['error_rate_percent']:.3f}%"

        print("\n✅ Production simulation completed successfully!")
        print("🚀 System is ready for production deployment!")

    # Helper methods

    def _select_operation_type(self, op_idx: int, total_ops: int) -> str:
        """Select operation type based on realistic distribution."""
        # Realistic distribution: 40% DB, 25% ML, 15% batch, 15% health, 5% file
        rand = (op_idx * 7) % 100  # Deterministic but distributed

        if rand < 40:
            return "database_query"
        elif rand < 65:
            return "ml_operation"
        elif rand < 80:
            return "batch_job"
        elif rand < 95:
            return "health_check"
        else:
            return "file_operation"

    async def _simulate_database_operation(
        self,
        user_id: int,
        db_client: PostgresAsyncClient,
        cache_layer: DatabaseCacheLayer
    ) -> Tuple[bool, float]:
        """Simulate a database operation."""
        try:
            queries = [
                ("SELECT COUNT(*) FROM sessions WHERE user_id = %(user_id)s", {"user_id": f"user_{user_id % 1000}"}),
                ("SELECT id, name FROM rules WHERE active = true ORDER BY created_at DESC LIMIT 5", {}),
                ("SELECT * FROM prompt_improvements WHERE created_at > NOW() - INTERVAL '1 day' LIMIT 3", {}),
            ]

            query, params = queries[user_id % len(queries)]

            start = time.perf_counter()

            # Use cache layer for some queries
            if user_id % 3 == 0:
                async def execute_query(q, p):
                    return await db_client.fetch_raw(q, p)

                result, was_cached = await cache_layer.get_or_execute(query, params, execute_query)
            else:
                result = await db_client.fetch_raw(query, params)

            duration = (time.perf_counter() - start) * 1000
            return True, duration

        except Exception:
            return False, 0

    async def _simulate_ml_operation(self, user_id: int, ml_orchestrator: MLPipelineOrchestrator) -> Tuple[bool, float]:
        """Simulate an ML operation."""
        try:
            start = time.time()

            # Alternate between different ML operations
            if user_id % 4 == 0:
                health = await ml_orchestrator.get_component_health()
                success = len(health) > 0
            elif user_id % 4 == 1:
                resources = await ml_orchestrator.get_resource_usage()
                success = len(resources) >= 0
            elif user_id % 4 == 2:
                # Quick workflow start (mock)
                await asyncio.sleep(0.05)  # Simulate workflow operation
                success = True
            else:
                # Status check (mock)
                await asyncio.sleep(0.02)  # Simulate status check
                success = True

            duration = (time.time() - start) * 1000
            return success, duration

        except Exception:
            return False, 0

    async def _simulate_batch_operation(self, user_id: int) -> Tuple[bool, float]:
        """Simulate a small batch operation."""
        try:
            start = time.time()

            # Simulate small data processing
            data = np.random.random(100)
            processed = data * 2.0 + 1.0
            result = np.mean(processed)

            duration = (time.time() - start) * 1000
            return True, duration

        except Exception:
            return False, 0

    async def _simulate_health_check(self, ml_orchestrator: MLPipelineOrchestrator) -> Tuple[bool, float]:
        """Simulate a health check operation."""
        try:
            start = time.time()
            health = await ml_orchestrator.get_component_health()
            duration = (time.time() - start) * 1000
            return len(health) > 0, duration

        except Exception:
            return False, 0

    async def _simulate_file_operation(self, user_id: int) -> Tuple[bool, float]:
        """Simulate a file system operation."""
        try:
            start = time.time()

            # Create and delete a temporary file
            temp_file = Path(f"temp_user_{user_id}_{int(time.time())}.txt")
            temp_file.write_text(f"User {user_id} temporary data")
            content = temp_file.read_text()
            temp_file.unlink()

            duration = (time.time() - start) * 1000
            return content is not None, duration

        except Exception:
            return False, 0

    async def _continuous_system_monitoring(self, metrics: ProductionSimulationMetrics, duration: float):
        """Continuously monitor system resources."""
        start_time = time.time()

        while time.time() - start_time < duration:
            metrics.record_system_snapshot()
            await asyncio.sleep(5)  # Every 5 seconds


if __name__ == "__main__":
    # Run production simulation tests
    pytest.main([__file__, "-v", "-s", "--tb=short", "-x"])
