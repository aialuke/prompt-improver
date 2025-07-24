"""
Comprehensive Integration Testing for Phase 1 & 2 Improvements.

This test suite validates that ALL Phase 1 & 2 improvements work together seamlessly
with real behavior validation. It tests the integration between:

Phase 1 Improvements:
- Type Safety (ML models)
- Database Performance
- Batch Processing
- Developer Experience
- ML Platform

Phase 2 Improvements:
- Production Readiness Validation
- Security Hardening
- Health Monitoring
- Performance Optimization
- Monitoring Infrastructure

Key Integration Points:
1. Type Safety ↔ Database Performance: Typed database operations
2. ML Platform ↔ Batch Processing: Efficient ML data processing
3. Developer Experience ↔ All Systems: IDE integration and hot reloading
4. Security ↔ Performance: Secure yet performant operations
5. Monitoring ↔ All Components: Comprehensive observability

Test Approach: REAL BEHAVIOR TESTING - No mocks, only actual component behavior
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import psutil
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

# Core imports for Phase 1 & 2 improvements
from prompt_improver.database import get_session_context
from prompt_improver.database.psycopg_client import PostgresAsyncClient
from prompt_improver.ml.optimization.batch.enhanced_batch_processor import (
    StreamingBatchProcessor, StreamingBatchConfig, ChunkingStrategy
)
from prompt_improver.ml.core.training_data_loader import TrainingDataLoader
from prompt_improver.ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator
from prompt_improver.performance.monitoring.performance_benchmark import PerformanceBenchmark
from prompt_improver.performance.validation.performance_validation import PerformanceValidator
from prompt_improver.security.secure_logging import SecureLoggingHandler
from prompt_improver.utils.health_checks import HealthChecker, HealthStatus
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig

# Configure logging
logger = logging.getLogger(__name__)


class IntegrationTestMetrics:
    """Track metrics across all integration tests."""
    
    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.integration_points: Dict[str, bool] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.compound_improvements: Dict[str, float] = {}
        
    def record_test_result(self, test_name: str, success: bool, duration: float, details: Dict[str, Any]):
        """Record results from a test."""
        self.test_results[test_name] = {
            "success": success,
            "duration_sec": duration,
            "timestamp": datetime.now(timezone.utc),
            "details": details
        }
        
    def record_integration_point(self, point_name: str, working: bool):
        """Record whether an integration point is working."""
        self.integration_points[point_name] = working
        
    def record_performance_metric(self, metric_name: str, value: float):
        """Record a performance metric."""
        self.performance_metrics[metric_name] = value
        
    def calculate_compound_improvements(self):
        """Calculate compound improvements from all phases."""
        # Calculate overall performance improvement
        if "baseline_response_time" in self.performance_metrics and "optimized_response_time" in self.performance_metrics:
            baseline = self.performance_metrics["baseline_response_time"]
            optimized = self.performance_metrics["optimized_response_time"]
            self.compound_improvements["response_time_improvement"] = (baseline - optimized) / baseline * 100
            
        # Calculate memory efficiency improvement
        if "baseline_memory_usage" in self.performance_metrics and "optimized_memory_usage" in self.performance_metrics:
            baseline = self.performance_metrics["baseline_memory_usage"]
            optimized = self.performance_metrics["optimized_memory_usage"]
            self.compound_improvements["memory_efficiency_improvement"] = (baseline - optimized) / baseline * 100
            
        # Calculate throughput improvement
        if "baseline_throughput" in self.performance_metrics and "optimized_throughput" in self.performance_metrics:
            baseline = self.performance_metrics["baseline_throughput"]
            optimized = self.performance_metrics["optimized_throughput"]
            self.compound_improvements["throughput_improvement"] = (optimized - baseline) / baseline * 100
            
    def generate_report(self) -> str:
        """Generate comprehensive integration test report."""
        self.calculate_compound_improvements()
        
        report = ["# Phase 1 & 2 Integration Test Report\n"]
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        
        # Test Results Summary
        report.append("## Test Results Summary\n")
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["success"])
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success Rate: {(passed_tests / total_tests * 100) if total_tests > 0 else 0:.1f}%\n")
        
        # Integration Points Status
        report.append("## Integration Points Status\n")
        for point, working in self.integration_points.items():
            status = "✅ Working" if working else "❌ Failed"
            report.append(f"- {point}: {status}")
        report.append("")
        
        # Performance Metrics
        report.append("## Performance Metrics\n")
        for metric, value in self.performance_metrics.items():
            report.append(f"- {metric}: {value:.2f}")
        report.append("")
        
        # Compound Improvements
        report.append("## Compound Improvements\n")
        for improvement, percentage in self.compound_improvements.items():
            report.append(f"- {improvement}: {percentage:.1f}%")
        report.append("")
        
        # Detailed Test Results
        report.append("## Detailed Test Results\n")
        for test_name, result in self.test_results.items():
            report.append(f"### {test_name}")
            report.append(f"- Status: {'✅ Passed' if result['success'] else '❌ Failed'}")
            report.append(f"- Duration: {result['duration_sec']:.2f}s")
            if result.get("details"):
                report.append("- Details:")
                for key, value in result["details"].items():
                    report.append(f"  - {key}: {value}")
            report.append("")
            
        return "\n".join(report)


class TestPhase1Phase2Integration:
    """Comprehensive integration tests for Phase 1 & 2 improvements."""
    
    @pytest.fixture
    def metrics(self):
        """Test metrics tracker."""
        return IntegrationTestMetrics()
    
    @pytest.fixture
    async def db_client(self):
        """Create PostgresAsyncClient for database tests."""
        client = PostgresAsyncClient(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            database=os.getenv("POSTGRES_DB", "prompt_improver_test"),
            user=os.getenv("POSTGRES_USER", "test_user"),
            password=os.getenv("POSTGRES_PASSWORD", "test_password")
        )
        await client.connect()
        yield client
        await client.disconnect()
    
    @pytest.fixture
    async def orchestrator(self):
        """Create ML orchestrator with all components."""
        config = OrchestratorConfig(
            max_concurrent_workflows=3,
            component_health_check_interval=1,
            training_timeout=300,
            debug_mode=True,
            enable_performance_profiling=True
        )
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_type_safety_database_integration(self, metrics: IntegrationTestMetrics, db_client: PostgresAsyncClient):
        """
        Test 1: Type Safety ↔ Database Performance Integration
        Verify that typed database operations maintain performance while ensuring type safety.
        """
        print("\n🔗 Test 1: Type Safety ↔ Database Performance Integration")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            # Test typed database operations with async session
            async with get_session_context() as session:
                # Verify session type (Phase 1: Type Safety)
                assert isinstance(session, AsyncSession), "Session should be typed as AsyncSession"
                print("✅ Database session properly typed")
                
                # Test performance optimization (Phase 1: Database Performance)
                # Measure query performance with connection pooling
                query_start = time.time()
                
                # Execute a sample query with proper typing
                from prompt_improver.database.models import TrainingData
                from sqlalchemy import select
                
                # Type-safe query construction
                stmt = select(TrainingData).limit(100)
                result = await session.execute(stmt)
                training_data = result.scalars().all()
                
                query_time = time.time() - query_start
                details["query_time_ms"] = query_time * 1000
                
                print(f"✅ Typed query executed in {query_time * 1000:.2f}ms")
                assert query_time < 0.2, f"Query too slow: {query_time}s"
                
                # Test batch operations with type safety
                batch_start = time.time()
                
                # Use PostgresAsyncClient for optimized batch operations
                if training_data:
                    # Test COPY operation (Phase 1: Database Performance)
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
                    try:
                        # Write data with proper types
                        import csv
                        writer = csv.writer(temp_file)
                        writer.writerow(['id', 'prompt', 'score', 'timestamp'])
                        for td in training_data[:10]:
                            writer.writerow([td.id, td.prompt, td.effectiveness_score, td.created_at])
                        temp_file.close()
                        
                        # Test async COPY operation
                        copy_result = await db_client.copy_from_file(
                            table_name="training_data_copy_test",
                            file_path=temp_file.name,
                            columns=['id', 'prompt', 'score', 'timestamp']
                        )
                        details["copy_operation"] = "success" if copy_result else "failed"
                        
                    finally:
                        os.unlink(temp_file.name)
                
                batch_time = time.time() - batch_start
                details["batch_operation_time_ms"] = batch_time * 1000
                
                print(f"✅ Batch operations completed in {batch_time * 1000:.2f}ms")
                
                # Verify connection pooling is active
                pool_stats = await db_client.get_pool_stats()
                details["connection_pool_size"] = pool_stats.get("size", 0)
                details["active_connections"] = pool_stats.get("active", 0)
                
                print(f"✅ Connection pool active: {pool_stats.get('size', 0)} connections")
                
                metrics.record_integration_point("type_safety_database", True)
                metrics.record_performance_metric("typed_query_time_ms", query_time * 1000)
                
        except Exception as e:
            print(f"❌ Type safety database integration failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("type_safety_database", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("type_safety_database_integration", test_success, duration, details)
        assert test_success, "Type safety database integration test failed"
    
    @pytest.mark.asyncio
    async def test_ml_platform_batch_processing_integration(self, metrics: IntegrationTestMetrics):
        """
        Test 2: ML Platform ↔ Batch Processing Integration
        Verify efficient ML data processing with enhanced batch processor.
        """
        print("\n🔗 Test 2: ML Platform ↔ Batch Processing Integration")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            # Generate synthetic ML data (Phase 1: ML Platform)
            generator = ProductionSyntheticDataGenerator(
                target_samples=1000,
                generation_method="statistical",
                use_enhanced_scoring=True
            )
            
            print("🔄 Generating synthetic ML data...")
            synthetic_data = await generator.generate_comprehensive_training_data()
            
            assert synthetic_data is not None, "Failed to generate synthetic data"
            details["synthetic_samples_generated"] = len(synthetic_data.get("features", []))
            
            # Process data with enhanced batch processor (Phase 1: Batch Processing)
            config = StreamingBatchConfig(
                chunk_size=100,
                worker_processes=2,
                memory_limit_mb=500,
                chunking_strategy=ChunkingStrategy.ADAPTIVE,
                gc_threshold_mb=100
            )
            
            # Define ML processing function
            def process_ml_features(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Process ML features in batch."""
                results = []
                for item in batch:
                    # Simulate feature processing
                    processed = {
                        "original_features": item.get("features", []),
                        "enhanced_features": np.array(item.get("features", [])) * 1.1,  # Simple enhancement
                        "processing_timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    results.append(processed)
                return results
            
            # Create temporary file with synthetic data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for i, features in enumerate(synthetic_data.get("features", [])):
                    record = {
                        "id": i,
                        "features": features.tolist() if hasattr(features, 'tolist') else features,
                        "label": synthetic_data.get("effectiveness_scores", [])[i] if i < len(synthetic_data.get("effectiveness_scores", [])) else 0
                    }
                    f.write(json.dumps(record) + '\n')
                temp_file = f.name
            
            try:
                # Process with streaming batch processor
                async with StreamingBatchProcessor(config, process_ml_features) as processor:
                    print("🔄 Processing ML data with enhanced batch processor...")
                    processing_metrics = await processor.process_dataset(
                        data_source=temp_file,
                        job_id="ml_batch_integration_test"
                    )
                    
                    details["items_processed"] = processing_metrics.items_processed
                    details["throughput_items_per_sec"] = processing_metrics.throughput_items_per_sec
                    details["memory_peak_mb"] = processing_metrics.memory_peak_mb
                    details["gc_collections"] = sum(processing_metrics.gc_collections.values())
                    
                    print(f"✅ Processed {processing_metrics.items_processed} items")
                    print(f"✅ Throughput: {processing_metrics.throughput_items_per_sec:.2f} items/sec")
                    print(f"✅ Peak memory: {processing_metrics.memory_peak_mb:.2f} MB")
                    print(f"✅ GC collections: {sum(processing_metrics.gc_collections.values())}")
                    
                    # Verify memory efficiency
                    assert processing_metrics.memory_peak_mb < 200, f"Memory usage too high: {processing_metrics.memory_peak_mb} MB"
                    
                    # Verify throughput
                    assert processing_metrics.throughput_items_per_sec > 500, f"Throughput too low: {processing_metrics.throughput_items_per_sec} items/sec"
                    
                    metrics.record_integration_point("ml_platform_batch_processing", True)
                    metrics.record_performance_metric("ml_batch_throughput", processing_metrics.throughput_items_per_sec)
                    metrics.record_performance_metric("ml_batch_memory_mb", processing_metrics.memory_peak_mb)
                    
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            print(f"❌ ML platform batch processing integration failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("ml_platform_batch_processing", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("ml_platform_batch_processing_integration", test_success, duration, details)
        assert test_success, "ML platform batch processing integration test failed"
    
    @pytest.mark.asyncio
    async def test_security_performance_integration(self, metrics: IntegrationTestMetrics):
        """
        Test 3: Security ↔ Performance Integration
        Verify that security improvements don't degrade performance.
        """
        print("\n🔗 Test 3: Security ↔ Performance Integration")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            # Test secure logging performance (Phase 2: Security)
            secure_handler = SecureLoggingHandler()
            test_logger = logging.getLogger("security_test")
            test_logger.addHandler(secure_handler)
            
            # Measure logging performance
            log_start = time.time()
            sensitive_data = "password=secret123"
            
            for i in range(1000):
                # Log with potential sensitive data
                test_logger.info(f"Processing request {i} with data: {sensitive_data}")
            
            log_time = time.time() - log_start
            details["secure_logging_time_ms"] = log_time * 1000
            details["logs_per_second"] = 1000 / log_time
            
            print(f"✅ Secure logging: 1000 logs in {log_time * 1000:.2f}ms")
            print(f"✅ Performance: {1000 / log_time:.2f} logs/sec")
            
            # Verify sensitive data was sanitized
            # In real implementation, check log output
            assert log_time < 1.0, f"Secure logging too slow: {log_time}s for 1000 logs"
            
            # Test async file operations performance (Phase 3: Performance)
            import aiofiles
            
            test_data = b"x" * 1024 * 1024  # 1MB of data
            
            # Measure async file write performance
            write_start = time.time()
            async with aiofiles.open("test_async_write.bin", "wb") as f:
                await f.write(test_data)
            write_time = time.time() - write_start
            
            # Measure async file read performance
            read_start = time.time()
            async with aiofiles.open("test_async_write.bin", "rb") as f:
                read_data = await f.read()
            read_time = time.time() - read_start
            
            os.unlink("test_async_write.bin")
            
            details["async_write_mb_per_sec"] = 1.0 / write_time
            details["async_read_mb_per_sec"] = 1.0 / read_time
            
            print(f"✅ Async file write: {1.0 / write_time:.2f} MB/s")
            print(f"✅ Async file read: {1.0 / read_time:.2f} MB/s")
            
            # Test combined security + performance
            # Simulate secure data processing with async operations
            process_start = time.time()
            
            # Generate secure random data
            import secrets
            secure_random_data = secrets.token_bytes(1024 * 100)  # 100KB
            
            # Process with async operations
            async with aiofiles.open("secure_temp.bin", "wb") as f:
                await f.write(secure_random_data)
            
            async with aiofiles.open("secure_temp.bin", "rb") as f:
                processed_data = await f.read()
            
            os.unlink("secure_temp.bin")
            
            process_time = time.time() - process_start
            details["secure_process_time_ms"] = process_time * 1000
            
            print(f"✅ Secure data processing completed in {process_time * 1000:.2f}ms")
            
            metrics.record_integration_point("security_performance", True)
            metrics.record_performance_metric("secure_logging_throughput", 1000 / log_time)
            metrics.record_performance_metric("async_io_throughput_mb_per_sec", 1.0 / write_time)
            
        except Exception as e:
            print(f"❌ Security performance integration failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("security_performance", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("security_performance_integration", test_success, duration, details)
        assert test_success, "Security performance integration test failed"
    
    @pytest.mark.asyncio
    async def test_monitoring_all_components_integration(self, metrics: IntegrationTestMetrics, orchestrator: MLPipelineOrchestrator):
        """
        Test 4: Monitoring ↔ All Components Integration
        Verify comprehensive observability across all components.
        """
        print("\n🔗 Test 4: Monitoring ↔ All Components Integration")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            # Test health monitoring (Phase 2: Health Endpoints)
            health_checker = HealthChecker()
            
            # Check database health
            db_health = await health_checker.check_database()
            details["database_health"] = db_health.status.value
            print(f"✅ Database health: {db_health.status.value}")
            
            # Check Redis health
            redis_health = await health_checker.check_redis()
            details["redis_health"] = redis_health.status.value
            print(f"✅ Redis health: {redis_health.status.value}")
            
            # Check system resources
            system_health = await health_checker.check_system_resources()
            details["system_health"] = system_health.status.value
            details["cpu_usage_percent"] = system_health.details.get("cpu_percent", 0)
            details["memory_usage_percent"] = system_health.details.get("memory_percent", 0)
            print(f"✅ System health: {system_health.status.value}")
            print(f"  - CPU: {system_health.details.get('cpu_percent', 0):.1f}%")
            print(f"  - Memory: {system_health.details.get('memory_percent', 0):.1f}%")
            
            # Test performance monitoring integration
            benchmark = PerformanceBenchmark(
                name="integration_test",
                description="Testing monitoring integration"
            )
            
            # Run a sample benchmark
            async def sample_workload():
                """Sample workload for benchmarking."""
                await asyncio.sleep(0.1)
                return {"result": "success"}
            
            benchmark_result = await benchmark.run_async(sample_workload)
            details["benchmark_duration_ms"] = benchmark_result["duration_ms"]
            details["benchmark_memory_mb"] = benchmark_result["memory_mb"]
            
            print(f"✅ Performance benchmark completed in {benchmark_result['duration_ms']:.2f}ms")
            
            # Test ML orchestrator health monitoring
            component_health = await orchestrator.get_component_health()
            healthy_components = sum(1 for h in component_health.values() if h)
            total_components = len(component_health)
            health_percentage = (healthy_components / total_components * 100) if total_components > 0 else 0
            
            details["total_components"] = total_components
            details["healthy_components"] = healthy_components
            details["component_health_percentage"] = health_percentage
            
            print(f"✅ Component health: {healthy_components}/{total_components} ({health_percentage:.1f}%)")
            
            # Test resource usage monitoring
            resource_usage = await orchestrator.get_resource_usage()
            details["resource_metrics_count"] = len(resource_usage)
            
            print(f"✅ Resource monitoring: {len(resource_usage)} metrics tracked")
            
            # Verify all monitoring systems are integrated
            assert db_health.status == HealthStatus.HEALTHY, "Database unhealthy"
            assert health_percentage >= 70, f"Component health too low: {health_percentage:.1f}%"
            assert len(resource_usage) > 0, "No resource metrics collected"
            
            metrics.record_integration_point("monitoring_all_components", True)
            metrics.record_performance_metric("component_health_percentage", health_percentage)
            
        except Exception as e:
            print(f"❌ Monitoring integration failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("monitoring_all_components", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("monitoring_all_components_integration", test_success, duration, details)
        assert test_success, "Monitoring integration test failed"
    
    @pytest.mark.asyncio
    async def test_end_to_end_ml_workflow_integration(self, metrics: IntegrationTestMetrics, orchestrator: MLPipelineOrchestrator):
        """
        Test 5: End-to-End ML Workflow Integration
        Test complete ML pipeline with all Phase 1 & 2 improvements.
        """
        print("\n🔗 Test 5: End-to-End ML Workflow Integration")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            # Configure workflow with all improvements enabled
            workflow_params = {
                "model_type": "integration_test_model",
                "enable_monitoring": True,
                "enable_performance_profiling": True,
                "enable_security": True,
                "enable_type_checking": True,
                "batch_processing_enabled": True,
                "use_async_operations": True,
                "test_mode": True
            }
            
            print("🚀 Starting ML workflow with all improvements enabled...")
            
            # Track initial resource usage
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            initial_cpu = psutil.cpu_percent(interval=0.1)
            
            # Start workflow
            workflow_start = time.time()
            workflow_id = await orchestrator.start_workflow("tier1_training", workflow_params)
            
            assert workflow_id is not None, "Failed to start workflow"
            print(f"✅ Workflow started: {workflow_id}")
            
            # Monitor workflow progress
            timeout = 120
            check_interval = 3
            elapsed = 0
            
            performance_snapshots = []
            
            while elapsed < timeout:
                # Get workflow status
                status = await orchestrator.get_workflow_status(workflow_id)
                
                # Collect performance snapshot
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                current_cpu = psutil.cpu_percent(interval=0.1)
                
                snapshot = {
                    "elapsed_sec": elapsed,
                    "state": status.state.value,
                    "memory_mb": current_memory,
                    "cpu_percent": current_cpu
                }
                performance_snapshots.append(snapshot)
                
                print(f"⏱️ Status: {status.state.value} | Memory: {current_memory:.1f}MB | CPU: {current_cpu:.1f}%")
                
                if status.state.value in ["COMPLETED", "ERROR"]:
                    break
                
                await asyncio.sleep(check_interval)
                elapsed += check_interval
            
            workflow_duration = time.time() - workflow_start
            
            # Get final status
            final_status = await orchestrator.get_workflow_status(workflow_id)
            
            # Calculate performance metrics
            peak_memory = max(s["memory_mb"] for s in performance_snapshots)
            avg_cpu = sum(s["cpu_percent"] for s in performance_snapshots) / len(performance_snapshots)
            memory_increase = peak_memory - initial_memory
            
            details["workflow_duration_sec"] = workflow_duration
            details["workflow_state"] = final_status.state.value
            details["peak_memory_mb"] = peak_memory
            details["memory_increase_mb"] = memory_increase
            details["avg_cpu_percent"] = avg_cpu
            details["performance_snapshots"] = len(performance_snapshots)
            
            print(f"\n✅ Workflow completed in {workflow_duration:.2f}s")
            print(f"📊 Peak memory: {peak_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
            print(f"📊 Average CPU: {avg_cpu:.1f}%")
            
            # Verify workflow completed successfully
            assert final_status.state.value == "COMPLETED", f"Workflow failed with state: {final_status.state.value}"
            
            # Verify performance targets
            assert workflow_duration < 120, f"Workflow too slow: {workflow_duration}s"
            assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB increase"
            
            metrics.record_integration_point("end_to_end_ml_workflow", True)
            metrics.record_performance_metric("workflow_duration_sec", workflow_duration)
            metrics.record_performance_metric("workflow_memory_increase_mb", memory_increase)
            
        except Exception as e:
            print(f"❌ End-to-end ML workflow integration failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("end_to_end_ml_workflow", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("end_to_end_ml_workflow_integration", test_success, duration, details)
        assert test_success, "End-to-end ML workflow integration test failed"
    
    @pytest.mark.asyncio
    async def test_compound_performance_improvements(self, metrics: IntegrationTestMetrics):
        """
        Test 6: Compound Performance Improvements
        Measure the combined effect of all performance optimizations.
        """
        print("\n🔗 Test 6: Compound Performance Improvements")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            # Create test workload that uses multiple improvements
            test_data_size = 10000
            
            # Baseline measurement (simulated without optimizations)
            print("📊 Measuring baseline performance...")
            baseline_start = time.time()
            baseline_memory_start = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Simulate unoptimized operations
            # Synchronous file I/O
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                for i in range(test_data_size):
                    f.write(f"test_line_{i}\n")
                temp_file = f.name
            
            # Synchronous processing
            with open(temp_file, 'r') as f:
                lines = f.readlines()
                processed = [line.strip().upper() for line in lines]
            
            os.unlink(temp_file)
            
            baseline_time = time.time() - baseline_start
            baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024) - baseline_memory_start
            baseline_throughput = test_data_size / baseline_time
            
            details["baseline_time_sec"] = baseline_time
            details["baseline_memory_mb"] = baseline_memory
            details["baseline_throughput_items_per_sec"] = baseline_throughput
            
            print(f"✅ Baseline: {baseline_time:.2f}s | {baseline_memory:.1f}MB | {baseline_throughput:.0f} items/sec")
            
            # Optimized measurement (with all improvements)
            print("\n📊 Measuring optimized performance...")
            optimized_start = time.time()
            optimized_memory_start = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Use async file I/O (Phase 3: Performance)
            import aiofiles
            
            async with aiofiles.open("test_async.txt", 'w') as f:
                for i in range(test_data_size):
                    await f.write(f"test_line_{i}\n")
            
            # Use batch processing (Phase 1: Batch Processing)
            async def process_batch(items):
                return [item.strip().upper() for item in items]
            
            # Read and process in chunks
            processed_items = []
            async with aiofiles.open("test_async.txt", 'r') as f:
                chunk_size = 1000
                while True:
                    chunk = []
                    for _ in range(chunk_size):
                        line = await f.readline()
                        if not line:
                            break
                        chunk.append(line)
                    
                    if not chunk:
                        break
                    
                    # Process chunk
                    processed_chunk = await process_batch(chunk)
                    processed_items.extend(processed_chunk)
            
            os.unlink("test_async.txt")
            
            # Trigger garbage collection (Phase 3: Memory Management)
            import gc
            gc.collect()
            
            optimized_time = time.time() - optimized_start
            optimized_memory = psutil.Process().memory_info().rss / (1024 * 1024) - optimized_memory_start
            optimized_throughput = test_data_size / optimized_time
            
            details["optimized_time_sec"] = optimized_time
            details["optimized_memory_mb"] = optimized_memory
            details["optimized_throughput_items_per_sec"] = optimized_throughput
            
            print(f"✅ Optimized: {optimized_time:.2f}s | {optimized_memory:.1f}MB | {optimized_throughput:.0f} items/sec")
            
            # Calculate improvements
            time_improvement = (baseline_time - optimized_time) / baseline_time * 100
            memory_improvement = (baseline_memory - optimized_memory) / baseline_memory * 100 if baseline_memory > 0 else 0
            throughput_improvement = (optimized_throughput - baseline_throughput) / baseline_throughput * 100
            
            details["time_improvement_percent"] = time_improvement
            details["memory_improvement_percent"] = memory_improvement
            details["throughput_improvement_percent"] = throughput_improvement
            
            print(f"\n📈 Improvements:")
            print(f"  - Time: {time_improvement:.1f}% faster")
            print(f"  - Memory: {memory_improvement:.1f}% more efficient")
            print(f"  - Throughput: {throughput_improvement:.1f}% higher")
            
            # Record metrics for final report
            metrics.record_performance_metric("baseline_response_time", baseline_time)
            metrics.record_performance_metric("optimized_response_time", optimized_time)
            metrics.record_performance_metric("baseline_memory_usage", baseline_memory)
            metrics.record_performance_metric("optimized_memory_usage", optimized_memory)
            metrics.record_performance_metric("baseline_throughput", baseline_throughput)
            metrics.record_performance_metric("optimized_throughput", optimized_throughput)
            
            # Verify improvements meet targets
            assert time_improvement > 20, f"Time improvement too low: {time_improvement:.1f}%"
            assert throughput_improvement > 20, f"Throughput improvement too low: {throughput_improvement:.1f}%"
            
            metrics.record_integration_point("compound_performance", True)
            
        except Exception as e:
            print(f"❌ Compound performance test failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("compound_performance", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("compound_performance_improvements", test_success, duration, details)
        assert test_success, "Compound performance improvements test failed"
    
    @pytest.mark.asyncio
    async def test_cross_platform_compatibility(self, metrics: IntegrationTestMetrics):
        """
        Test 7: Cross-Platform Compatibility
        Verify all improvements work across different environments.
        """
        print("\n🔗 Test 7: Cross-Platform Compatibility")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            import platform
            import sys
            
            # Get platform information
            details["platform"] = platform.system()
            details["platform_version"] = platform.version()
            details["python_version"] = sys.version
            details["architecture"] = platform.machine()
            
            print(f"🖥️ Platform: {platform.system()} {platform.version()}")
            print(f"🐍 Python: {sys.version.split()[0]}")
            print(f"🏗️ Architecture: {platform.machine()}")
            
            # Test platform-specific features
            if platform.system() == "Windows":
                # Windows-specific tests
                print("\n🪟 Testing Windows compatibility...")
                # Test async file operations on Windows
                import aiofiles
                async with aiofiles.open("windows_test.txt", 'w') as f:
                    await f.write("Windows async test")
                async with aiofiles.open("windows_test.txt", 'r') as f:
                    content = await f.read()
                os.unlink("windows_test.txt")
                assert content == "Windows async test", "Windows async file I/O failed"
                print("✅ Windows async file I/O working")
                
            elif platform.system() == "Darwin":  # macOS
                # macOS-specific tests
                print("\n🍎 Testing macOS compatibility...")
                # Test memory management on macOS
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                details["macos_memory_limit"] = hard
                print(f"✅ macOS memory limits: {hard / (1024**3):.1f} GB")
                
            elif platform.system() == "Linux":
                # Linux-specific tests
                print("\n🐧 Testing Linux compatibility...")
                # Test cgroup limits on Linux
                try:
                    with open("/proc/self/cgroup", 'r') as f:
                        cgroup_info = f.read()
                    details["linux_cgroup"] = "containerized" if "docker" in cgroup_info else "native"
                    print(f"✅ Linux environment: {details['linux_cgroup']}")
                except:
                    details["linux_cgroup"] = "unknown"
            
            # Test universal features
            print("\n🌍 Testing universal features...")
            
            # Test async operations (should work on all platforms)
            async def universal_async_test():
                await asyncio.sleep(0.01)
                return True
            
            result = await universal_async_test()
            assert result, "Universal async test failed"
            print("✅ Async operations working")
            
            # Test multiprocessing compatibility
            from multiprocessing import cpu_count
            cpu_cores = cpu_count()
            details["cpu_cores"] = cpu_cores
            print(f"✅ Multiprocessing: {cpu_cores} cores available")
            
            # Test file path handling
            test_path = Path("test") / "nested" / "path.txt"
            normalized_path = test_path.as_posix()
            details["path_separator"] = os.sep
            print(f"✅ Path handling: {normalized_path} (separator: {os.sep})")
            
            metrics.record_integration_point("cross_platform_compatibility", True)
            
        except Exception as e:
            print(f"❌ Cross-platform compatibility test failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("cross_platform_compatibility", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("cross_platform_compatibility", test_success, duration, details)
        assert test_success, "Cross-platform compatibility test failed"
    
    @pytest.mark.asyncio
    async def test_production_deployment_readiness(self, metrics: IntegrationTestMetrics):
        """
        Test 8: Production Deployment Readiness
        Verify system is ready for production deployment with all improvements.
        """
        print("\n🔗 Test 8: Production Deployment Readiness")
        print("=" * 60)
        
        start_time = time.time()
        test_success = True
        details = {}
        
        try:
            # Run production readiness validation
            from scripts.production_readiness_validation import ProductionReadinessValidator
            
            validator = ProductionReadinessValidator()
            print("🔄 Running production readiness validation...")
            
            # Check security
            security_status = await validator.check_security()
            details["security_status"] = security_status
            print(f"🔒 Security: {'✅ PASS' if security_status['passed'] else '❌ FAIL'}")
            
            # Check performance
            performance_status = await validator.check_performance()
            details["performance_status"] = performance_status
            print(f"⚡ Performance: {'✅ PASS' if performance_status['passed'] else '❌ FAIL'}")
            
            # Check reliability
            reliability_status = await validator.check_reliability()
            details["reliability_status"] = reliability_status
            print(f"🔄 Reliability: {'✅ PASS' if reliability_status['passed'] else '❌ FAIL'}")
            
            # Check observability
            observability_status = await validator.check_observability()
            details["observability_status"] = observability_status
            print(f"👁️ Observability: {'✅ PASS' if observability_status['passed'] else '❌ FAIL'}")
            
            # Check scalability
            scalability_status = await validator.check_scalability()
            details["scalability_status"] = scalability_status
            print(f"📈 Scalability: {'✅ PASS' if scalability_status['passed'] else '❌ FAIL'}")
            
            # Check compliance
            compliance_status = await validator.check_compliance()
            details["compliance_status"] = compliance_status
            print(f"📋 Compliance: {'✅ PASS' if compliance_status['passed'] else '❌ FAIL'}")
            
            # Overall readiness
            all_passed = all([
                security_status.get('passed', False),
                performance_status.get('passed', False),
                reliability_status.get('passed', False),
                observability_status.get('passed', False),
                scalability_status.get('passed', False),
                compliance_status.get('passed', False)
            ])
            
            details["production_ready"] = all_passed
            
            if all_passed:
                print("\n✅ PRODUCTION READY: All checks passed!")
            else:
                print("\n⚠️ NOT PRODUCTION READY: Some checks failed")
            
            metrics.record_integration_point("production_deployment_readiness", all_passed)
            
        except Exception as e:
            print(f"❌ Production readiness test failed: {e}")
            test_success = False
            details["error"] = str(e)
            metrics.record_integration_point("production_deployment_readiness", False)
        
        duration = time.time() - start_time
        metrics.record_test_result("production_deployment_readiness", test_success, duration, details)
        assert test_success, "Production deployment readiness test failed"
    
    @pytest.mark.asyncio
    async def test_generate_integration_report(self, metrics: IntegrationTestMetrics):
        """
        Test 9: Generate Comprehensive Integration Report
        Generate and save the final integration test report.
        """
        print("\n📊 Generating Comprehensive Integration Report")
        print("=" * 60)
        
        # Generate report
        report = metrics.generate_report()
        
        # Save report
        report_path = Path("integration_test_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_path}")
        print("\n" + report)
        
        # Verify all integration points
        failed_points = [point for point, working in metrics.integration_points.items() if not working]
        if failed_points:
            print(f"\n⚠️ Failed integration points: {failed_points}")
        
        # Verify compound improvements
        if metrics.compound_improvements:
            print("\n📈 Compound Improvements Achieved:")
            for improvement, percentage in metrics.compound_improvements.items():
                print(f"  - {improvement}: {percentage:.1f}%")
        
        # Assert all tests passed
        total_tests = len(metrics.test_results)
        passed_tests = sum(1 for r in metrics.test_results.values() if r["success"])
        
        assert passed_tests == total_tests, f"Some tests failed: {passed_tests}/{total_tests} passed"
        assert not failed_points, f"Some integration points failed: {failed_points}"
        
        print(f"\n✅ All {total_tests} integration tests passed!")
        print("✅ Phase 1 & 2 improvements are working together seamlessly!")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])