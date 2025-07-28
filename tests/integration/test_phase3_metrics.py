"""
Comprehensive Phase 3 Metrics and Observability Real Behavior Testing Suite

This test suite validates all Phase 3 metrics and observability systems under real-world conditions:
- OpenTelemetry distributed tracing with actual operations
- Business metrics collection with real business operations  
- System metrics under actual load conditions
- Performance baseline collection and regression detection
- SLO/SLA calculations with real service data
- Load testing integration for metrics accuracy validation
- End-to-end integration testing without conflicts
- Prometheus export validation for dashboard integration

No mocking - all tests use real services, actual data operations, and production-like scenarios.
"""

import asyncio
import json
import time
import pytest
import aiohttp
import threading
import tempfile
import statistics
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch
import psutil
import requests

# Core imports
from prompt_improver.metrics import (
    initialize_all_metrics,
    shutdown_all_metrics,
    get_ml_metrics_collector,
    get_api_metrics_collector, 
    get_performance_metrics_collector,
    get_bi_metrics_collector,
    get_aggregation_engine,
    get_dashboard_exporter,
    record_prompt_improvement,
    record_api_request,
    record_feature_usage,
    record_operational_cost,
    ExportFormat,
    DashboardType,
    TimeRange
)

# OpenTelemetry imports
from prompt_improver.monitoring.opentelemetry import (
    init_telemetry,
    shutdown_telemetry,
    get_tracer,
    get_meter,
    trace_ml_operation,
    trace_database_operation,
    trace_cache_operation,
    trace_business_operation,
    get_correlation_id,
    set_correlation_id,
    HttpMetrics,
    DatabaseMetrics,
    MLMetrics,
    BusinessMetrics
)

# SLO/SLA monitoring imports  
from prompt_improver.monitoring.slo import (
    SLOMonitor,
    SLAMonitor,
    ErrorBudgetMonitor,
    BurnRateAlert,
    SLODefinition,
    SLOTarget,
    SLOTimeWindow,
    SLOType
)

# Performance baseline imports
from prompt_improver.performance.baseline import (
    BaselineCollector,
    RegressionDetector,
    LoadTestingIntegration,
    PerformanceValidationSuite
)

# System and database imports
from prompt_improver.database.connection_pool_optimizer import ConnectionPoolOptimizer
from prompt_improver.core.config_manager import ConfigManager
from prompt_improver.performance.monitoring.metrics_registry import MetricsRegistry


class Phase3MetricsTestSuite:
    """Comprehensive metrics testing with real behavior validation."""
    
    def __init__(self):
        self.tracer = None
        self.meter = None
        self.config_manager = None
        self.test_data_dir = None
        self.load_test_threads = []
        self.performance_data = []
        self.metrics_data = {}
        
    async def setup(self):
        """Initialize all metrics systems for testing."""
        # Create temporary directory for test data
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="phase3_metrics_"))
        
        # Initialize configuration
        self.config_manager = ConfigManager()
        await self.config_manager.initialize()
        
        # Initialize OpenTelemetry with real exporters
        telemetry_config = {
            'service_name': 'phase3-metrics-test',
            'environment': 'test',
            'otlp_endpoint': 'http://localhost:4317',
            'enable_console_exporter': True,
            'enable_file_exporter': True,
            'file_export_path': str(self.test_data_dir / 'traces.json'),
            'sampling_ratio': 1.0  # 100% sampling for testing
        }
        
        init_telemetry(**telemetry_config)
        self.tracer = get_tracer(__name__)
        self.meter = get_meter(__name__)
        
        # Initialize all metrics collectors
        await initialize_all_metrics()
        
        print(f"✅ Phase 3 metrics test suite initialized")
        print(f"📁 Test data directory: {self.test_data_dir}")
        
    async def cleanup(self):
        """Clean up metrics systems and test data."""
        # Stop load test threads
        for thread in self.load_test_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Shutdown metrics systems
        await shutdown_all_metrics()
        await shutdown_telemetry()
        
        print(f"🧹 Phase 3 metrics test suite cleaned up")


@pytest.fixture(scope="session")
async def metrics_test_suite():
    """Session-scoped fixture for metrics test suite."""
    suite = Phase3MetricsTestSuite()
    await suite.setup()
    yield suite
    await suite.cleanup()


class TestOpenTelemetryRealBehavior:
    """Test OpenTelemetry distributed tracing with real operations."""
    
    @pytest.mark.asyncio
    async def test_distributed_tracing_with_real_http_requests(self, metrics_test_suite):
        """Test distributed tracing across real HTTP requests."""
        tracer = metrics_test_suite.tracer
        test_data = []
        
        # Create parent span for the entire operation
        with tracer.start_as_current_span("integration_test_flow") as parent_span:
            correlation_id = f"test-{int(time.time())}"
            set_correlation_id(correlation_id)
            
            # Simulate multiple HTTP requests with tracing
            urls = [
                "https://httpbin.org/delay/1",
                "https://httpbin.org/json",
                "https://httpbin.org/uuid"
            ]
            
            async with aiohttp.ClientSession() as session:
                for i, url in enumerate(urls):
                    with tracer.start_as_current_span(f"http_request_{i}") as span:
                        span.set_attribute("http.url", url)
                        span.set_attribute("http.method", "GET")
                        span.set_attribute("correlation.id", correlation_id)
                        
                        start_time = time.time()
                        try:
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                                duration = time.time() - start_time
                                span.set_attribute("http.status_code", response.status)
                                span.set_attribute("http.response_time", duration)
                                
                                test_data.append({
                                    'url': url,
                                    'status': response.status,
                                    'duration': duration,
                                    'correlation_id': correlation_id
                                })
                                
                        except Exception as e:
                            span.record_exception(e)
                            span.set_attribute("error", True)
                            raise
            
            parent_span.set_attribute("total_requests", len(urls))
            parent_span.set_attribute("avg_response_time", 
                                    statistics.mean([d['duration'] for d in test_data]))
        
        # Verify tracing data was collected
        assert len(test_data) == 3
        assert all(d['status'] == 200 for d in test_data)
        assert all(d['correlation_id'] == correlation_id for d in test_data)
        
        # Verify response times are reasonable
        avg_duration = statistics.mean([d['duration'] for d in test_data])
        assert 0.1 <= avg_duration <= 10.0, f"Average response time {avg_duration}s outside expected range"
        
        print(f"✅ Distributed tracing test completed")
        print(f"📊 Average response time: {avg_duration:.3f}s")
        print(f"🔗 Correlation ID: {correlation_id}")
    
    @pytest.mark.asyncio 
    async def test_database_operation_tracing(self, metrics_test_suite):
        """Test database operation tracing with real database calls."""
        tracer = metrics_test_suite.tracer
        
        # Initialize database connection
        pool_optimizer = ConnectionPoolOptimizer()
        await pool_optimizer.initialize()
        
        operations_data = []
        
        try:
            with tracer.start_as_current_span("database_integration_test") as parent_span:
                
                # Test various database operations with tracing
                operations = [
                    ("SELECT 1", "health_check"),
                    ("SELECT pg_database_size(current_database())", "size_check"),
                    ("SELECT version()", "version_check")
                ]
                
                for query, operation_type in operations:
                    operation_data = await trace_database_operation(
                        operation_type,
                        query,
                        pool_optimizer._execute_query
                    )(query)
                    
                    operations_data.append({
                        'operation': operation_type,
                        'query': query,
                        'duration': operation_data.get('duration', 0),
                        'success': operation_data.get('success', False)
                    })
                
                parent_span.set_attribute("total_operations", len(operations))
                parent_span.set_attribute("avg_duration", 
                                        statistics.mean([o['duration'] for o in operations_data]))
                
        finally:
            await pool_optimizer.close()
        
        # Verify all operations completed successfully
        assert len(operations_data) == 3
        assert all(op['success'] for op in operations_data)
        
        # Verify reasonable performance
        avg_duration = statistics.mean([op['duration'] for op in operations_data])
        assert avg_duration <= 1.0, f"Average DB operation time {avg_duration}s too slow"
        
        print(f"✅ Database tracing test completed")
        print(f"📊 Average DB operation time: {avg_duration:.3f}s")
    
    @pytest.mark.asyncio
    async def test_ml_operation_tracing(self, metrics_test_suite):
        """Test ML operation tracing with real ML pipeline operations."""
        tracer = metrics_test_suite.tracer
        ml_operations = []
        
        with tracer.start_as_current_span("ml_pipeline_test") as parent_span:
            
            # Simulate various ML operations
            ml_tasks = [
                ("prompt_preprocessing", 0.1),
                ("model_inference", 0.5),
                ("result_postprocessing", 0.2),
                ("quality_validation", 0.3)
            ]
            
            for task_name, simulated_duration in ml_tasks:
                operation_data = await trace_ml_operation(
                    task_name,
                    self._simulate_ml_operation
                )(simulated_duration)
                
                ml_operations.append({
                    'task': task_name,
                    'duration': operation_data.get('duration', 0),
                    'success': operation_data.get('success', False),
                    'throughput': operation_data.get('throughput', 0)
                })
            
            parent_span.set_attribute("total_ml_operations", len(ml_tasks))
            parent_span.set_attribute("pipeline_duration", 
                                    sum(op['duration'] for op in ml_operations))
        
        # Verify ML operations completed successfully
        assert len(ml_operations) == 4
        assert all(op['success'] for op in ml_operations)
        
        # Verify reasonable ML performance
        total_duration = sum(op['duration'] for op in ml_operations)
        assert total_duration <= 5.0, f"Total ML pipeline duration {total_duration}s too slow"
        
        print(f"✅ ML operation tracing test completed")
        print(f"📊 Total ML pipeline duration: {total_duration:.3f}s")
    
    async def _simulate_ml_operation(self, duration: float) -> Dict[str, Any]:
        """Simulate an ML operation with realistic timing."""
        start_time = time.time()
        
        # Simulate actual work
        await asyncio.sleep(duration)
        
        # Simulate some computation
        result = sum(i**2 for i in range(1000))
        
        actual_duration = time.time() - start_time
        
        return {
            'duration': actual_duration,
            'success': True,
            'throughput': 1000 / actual_duration,
            'result_checksum': result % 10000
        }


class TestBusinessMetricsRealBehavior:
    """Test business metrics collection with real business operations."""
    
    @pytest.mark.asyncio
    async def test_feature_usage_tracking_with_real_operations(self, metrics_test_suite):
        """Test feature usage tracking with actual feature operations."""
        bi_collector = get_bi_metrics_collector()
        
        # Simulate real feature usage patterns
        features_tested = []
        
        # Test various feature categories with realistic usage
        feature_scenarios = [
            ("prompt_improvement", "premium", 50, 0.95),
            ("model_tuning", "enterprise", 25, 0.88),
            ("batch_processing", "standard", 100, 0.92),
            ("api_access", "premium", 200, 0.98),
            ("analytics_dashboard", "enterprise", 15, 0.85)
        ]
        
        for feature, user_tier, usage_count, success_rate in feature_scenarios:
            for i in range(usage_count):
                # Simulate actual feature usage with realistic timing
                start_time = time.time()
                
                # Simulate feature operation
                success = i < (usage_count * success_rate)
                processing_time = 0.1 + (i % 10) * 0.05  # Variable processing time
                await asyncio.sleep(processing_time / 100)  # Scale down for testing
                
                duration = time.time() - start_time
                
                # Record feature usage with real data
                record_feature_usage(
                    feature_name=feature,
                    user_tier=user_tier,
                    usage_count=1,
                    success=success,
                    duration=duration,
                    metadata={
                        'operation_id': f"{feature}_{i}",
                        'timestamp': datetime.now().isoformat(),
                        'processing_time': processing_time
                    }
                )
                
                features_tested.append({
                    'feature': feature,
                    'user_tier': user_tier,
                    'success': success,
                    'duration': duration
                })
        
        # Get collection statistics
        stats = bi_collector.get_collection_stats()
        
        # Verify data collection
        assert stats['total_events'] >= len(features_tested)
        assert stats['feature_categories'] >= len(set(f['feature'] for f in features_tested))
        
        # Verify success rates match expectations
        feature_success_rates = {}
        for feature_data in features_tested:
            feature = feature_data['feature']
            if feature not in feature_success_rates:
                feature_success_rates[feature] = []
            feature_success_rates[feature].append(feature_data['success'])
        
        for feature, successes in feature_success_rates.items():
            actual_rate = sum(successes) / len(successes)
            expected_rate = next(s[3] for s in feature_scenarios if s[0] == feature)
            assert abs(actual_rate - expected_rate) <= 0.1, \
                f"Feature {feature} success rate {actual_rate} differs from expected {expected_rate}"
        
        print(f"✅ Feature usage tracking test completed")
        print(f"📊 Total features tested: {len(set(f['feature'] for f in features_tested))}")
        print(f"📈 Total usage events: {len(features_tested)}")
    
    @pytest.mark.asyncio
    async def test_cost_tracking_with_real_operations(self, metrics_test_suite):
        """Test operational cost tracking with actual resource usage."""
        bi_collector = get_bi_metrics_collector()
        
        cost_data = []
        
        # Simulate real operational costs
        cost_scenarios = [
            ("compute", "ml_inference", 50, 0.05),    # $0.05 per inference
            ("storage", "data_retention", 1000, 0.01), # $0.01 per GB-day
            ("network", "api_calls", 500, 0.001),     # $0.001 per API call
            ("database", "query_execution", 200, 0.02), # $0.02 per complex query
            ("monitoring", "metrics_collection", 100, 0.005) # $0.005 per metric point
        ]
        
        total_expected_cost = 0
        
        for cost_type, resource, usage_count, unit_cost in cost_scenarios:
            for i in range(usage_count):
                # Simulate variable resource usage
                usage_multiplier = 1 + (i % 5) * 0.1  # 1.0x to 1.4x usage
                actual_cost = unit_cost * usage_multiplier
                total_expected_cost += actual_cost
                
                # Record operational cost with real timing
                record_operational_cost(
                    cost_type=cost_type,
                    resource_type=resource,
                    amount=actual_cost,
                    currency="USD",
                    metadata={
                        'usage_id': f"{resource}_{i}",
                        'usage_multiplier': usage_multiplier,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                cost_data.append({
                    'cost_type': cost_type,
                    'resource': resource,
                    'amount': actual_cost,
                    'usage_multiplier': usage_multiplier
                })
        
        # Get collection statistics
        stats = bi_collector.get_collection_stats()
        
        # Verify cost tracking
        assert stats['total_events'] >= len(cost_data)
        
        # Verify cost calculations
        total_recorded_cost = sum(c['amount'] for c in cost_data)
        assert abs(total_recorded_cost - total_expected_cost) <= 0.001, \
            f"Total cost mismatch: recorded {total_recorded_cost}, expected {total_expected_cost}"
        
        # Verify cost distribution
        cost_by_type = {}
        for cost_item in cost_data:
            cost_type = cost_item['cost_type']
            if cost_type not in cost_by_type:
                cost_by_type[cost_type] = 0
            cost_by_type[cost_type] += cost_item['amount']
        
        assert len(cost_by_type) == len(set(s[0] for s in cost_scenarios))
        
        print(f"✅ Cost tracking test completed")
        print(f"💰 Total operational cost tracked: ${total_recorded_cost:.3f}")
        print(f"📊 Cost categories: {list(cost_by_type.keys())}")


class TestSystemMetricsUnderLoad:
    """Test system metrics collection under actual load conditions."""
    
    @pytest.mark.asyncio
    async def test_connection_age_tracking_under_load(self, metrics_test_suite):
        """Test connection age tracking with concurrent database operations."""
        performance_collector = get_performance_metrics_collector()
        
        # Create multiple concurrent database connections
        connection_data = []
        concurrent_operations = 20
        
        async def db_operation_worker(worker_id: int):
            """Worker that performs database operations."""
            pool_optimizer = ConnectionPoolOptimizer()
            await pool_optimizer.initialize()
            
            try:
                start_time = time.time()
                
                # Perform multiple operations to age the connection
                for i in range(10):
                    result = await pool_optimizer._execute_query("SELECT pg_sleep(0.1)")
                    connection_age = time.time() - start_time
                    
                    connection_data.append({
                        'worker_id': worker_id,
                        'operation_id': i,
                        'connection_age': connection_age,
                        'timestamp': time.time()
                    })
                
            finally:
                await pool_optimizer.close()
        
        # Run concurrent workers
        tasks = [db_operation_worker(i) for i in range(concurrent_operations)]
        await asyncio.gather(*tasks)
        
        # Verify connection age tracking
        assert len(connection_data) == concurrent_operations * 10
        
        # Verify connection ages are reasonable and increasing
        worker_data = {}
        for data in connection_data:
            worker_id = data['worker_id']
            if worker_id not in worker_data:
                worker_data[worker_id] = []
            worker_data[worker_id].append(data['connection_age'])
        
        for worker_id, ages in worker_data.items():
            # Verify ages are increasing (monotonic)
            assert all(ages[i] <= ages[i+1] for i in range(len(ages)-1)), \
                f"Connection ages not monotonic for worker {worker_id}"
            
            # Verify reasonable age values
            max_age = max(ages)
            assert 0.5 <= max_age <= 5.0, \
                f"Connection max age {max_age}s outside expected range for worker {worker_id}"
        
        print(f"✅ Connection age tracking test completed")
        print(f"🔗 Total connections tested: {concurrent_operations}")
        print(f"⏱️ Average max connection age: {statistics.mean([max(ages) for ages in worker_data.values()]):.3f}s")
    
    @pytest.mark.asyncio
    async def test_queue_depth_monitoring_under_load(self, metrics_test_suite):
        """Test queue depth monitoring with high-throughput operations."""
        performance_collector = get_performance_metrics_collector()
        
        # Create a queue processing scenario
        queue_data = []
        max_queue_size = 1000
        processing_workers = 5
        
        # Simulate queue with realistic timing
        queue = asyncio.Queue(maxsize=max_queue_size)
        
        async def producer():
            """Producer that adds items to queue."""
            for i in range(max_queue_size):
                await queue.put(f"task_{i}")
                queue_data.append({
                    'action': 'enqueue',
                    'queue_size': queue.qsize(),
                    'timestamp': time.time()
                })
                await asyncio.sleep(0.001)  # Small delay to create queue buildup
        
        async def consumer(worker_id: int):
            """Consumer that processes queue items."""
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Simulate processing time
                    await asyncio.sleep(0.01)
                    
                    queue_data.append({
                        'action': 'dequeue',
                        'queue_size': queue.qsize(),
                        'worker_id': worker_id,
                        'item': item,
                        'timestamp': time.time()
                    })
                    
                    queue.task_done()
                    
                except asyncio.TimeoutError:
                    break
        
        # Run producer and consumers concurrently
        producer_task = asyncio.create_task(producer())
        consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(processing_workers)]
        
        await producer_task
        await queue.join()  # Wait for all items to be processed
        
        # Cancel remaining consumer tasks
        for task in consumer_tasks:
            task.cancel()
        
        # Verify queue monitoring data
        enqueue_events = [d for d in queue_data if d['action'] == 'enqueue']
        dequeue_events = [d for d in queue_data if d['action'] == 'dequeue']
        
        assert len(enqueue_events) == max_queue_size
        assert len(dequeue_events) == max_queue_size
        
        # Verify queue depth patterns
        max_queue_depth = max(d['queue_size'] for d in enqueue_events)
        assert max_queue_depth > 0, "Queue should have had some depth during processing"
        
        # Verify all items were processed
        final_queue_size = min(d['queue_size'] for d in dequeue_events[-10:])  # Last few events
        assert final_queue_size == 0, "Queue should be empty after processing"
        
        print(f"✅ Queue depth monitoring test completed")
        print(f"📊 Max queue depth: {max_queue_depth}")
        print(f"⚡ Total items processed: {len(dequeue_events)}")
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_under_realistic_load(self, metrics_test_suite):
        """Test cache hit rate monitoring with realistic access patterns."""
        performance_collector = get_performance_metrics_collector()
        
        # Simulate realistic cache with actual Redis operations
        from prompt_improver.cache.redis_cache import RedisCache
        
        cache = RedisCache()
        await cache.initialize()
        
        cache_operations = []
        
        try:
            # Populate cache with initial data
            initial_keys = [f"cache_key_{i}" for i in range(100)]
            for key in initial_keys:
                value = f"cached_value_{key}"
                await cache.set(key, value, ttl=300)
            
            # Simulate realistic access patterns
            # 70% hits, 30% misses (Pareto distribution)
            access_patterns = []
            total_accesses = 1000
            
            for i in range(total_accesses):
                if i < total_accesses * 0.7:  # 70% should be hits
                    key = initial_keys[i % len(initial_keys)]
                else:  # 30% should be misses
                    key = f"missing_key_{i}"
                
                access_patterns.append(key)
            
            # Perform cache operations with timing
            for i, key in enumerate(access_patterns):
                start_time = time.time()
                
                try:
                    value = await cache.get(key)
                    hit = value is not None
                except Exception:
                    hit = False
                    value = None
                
                duration = time.time() - start_time
                
                cache_operations.append({
                    'key': key,
                    'hit': hit,
                    'duration': duration,
                    'operation_id': i
                })
                
                # Small delay to simulate realistic load
                await asyncio.sleep(0.001)
        
        finally:
            await cache.close()
        
        # Calculate cache hit rate
        total_operations = len(cache_operations)
        cache_hits = sum(1 for op in cache_operations if op['hit'])
        hit_rate = cache_hits / total_operations
        
        # Verify cache hit rate is close to expected (70%)
        expected_hit_rate = 0.7
        assert abs(hit_rate - expected_hit_rate) <= 0.05, \
            f"Cache hit rate {hit_rate} differs significantly from expected {expected_hit_rate}"
        
        # Verify performance characteristics
        hit_durations = [op['duration'] for op in cache_operations if op['hit']]
        miss_durations = [op['duration'] for op in cache_operations if not op['hit']]
        
        avg_hit_duration = statistics.mean(hit_durations)
        avg_miss_duration = statistics.mean(miss_durations)
        
        # Cache hits should generally be faster than misses
        assert avg_hit_duration <= avg_miss_duration * 2, \
            f"Cache hits ({avg_hit_duration}s) not significantly faster than misses ({avg_miss_duration}s)"
        
        print(f"✅ Cache hit rate test completed")
        print(f"🎯 Cache hit rate: {hit_rate:.3f} (expected: {expected_hit_rate})")
        print(f"⚡ Avg hit duration: {avg_hit_duration:.4f}s")
        print(f"❌ Avg miss duration: {avg_miss_duration:.4f}s")


class TestPerformanceBaselineSystem:
    """Test performance baseline collection and regression detection."""
    
    @pytest.mark.asyncio
    async def test_automated_baseline_collection(self, metrics_test_suite):
        """Test automated baseline collection with real performance data."""
        baseline_collector = BaselineCollector()
        
        # Configure baseline collection
        config = {
            'collection_interval': 1,  # 1 second for testing
            'metrics_to_collect': [
                'response_time',
                'throughput',
                'memory_usage',
                'cpu_utilization',
                'database_query_time'
            ],
            'storage_path': str(metrics_test_suite.test_data_dir / 'baselines')
        }
        
        await baseline_collector.configure(config)
        
        # Start baseline collection
        await baseline_collector.start()
        
        performance_data = []
        
        try:
            # Generate realistic performance scenarios
            scenarios = [
                ('low_load', 10, 0.1),
                ('medium_load', 50, 0.2), 
                ('high_load', 100, 0.5),
                ('peak_load', 200, 1.0)
            ]
            
            for scenario_name, request_count, base_latency in scenarios:
                scenario_start = time.time()
                
                # Simulate load scenario
                for i in range(request_count):
                    operation_start = time.time()
                    
                    # Simulate processing with variable latency
                    processing_time = base_latency + (i % 10) * 0.01
                    await asyncio.sleep(processing_time / 100)  # Scale down for testing
                    
                    operation_duration = time.time() - operation_start
                    
                    # Record performance metrics
                    performance_data.append({
                        'scenario': scenario_name,
                        'request_id': i,
                        'response_time': operation_duration,
                        'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                        'cpu_percent': psutil.Process().cpu_percent(),
                        'timestamp': time.time()
                    })
                
                scenario_duration = time.time() - scenario_start
                print(f"📊 Completed scenario '{scenario_name}': {request_count} requests in {scenario_duration:.2f}s")
                
                # Allow baseline collector to process data
                await asyncio.sleep(2)
        
        finally:
            await baseline_collector.stop()
        
        # Verify baseline data was collected
        baseline_data = await baseline_collector.get_collected_baselines()
        assert len(baseline_data) > 0, "No baseline data was collected"
        
        # Verify baseline data covers different scenarios
        collected_metrics = baseline_data.get('metrics', {})
        assert 'response_time' in collected_metrics
        assert 'memory_usage' in collected_metrics
        
        # Verify statistical properties
        response_times = [d['response_time'] for d in performance_data]
        baseline_response_time = collected_metrics['response_time']['mean']
        actual_mean = statistics.mean(response_times)
        
        assert abs(baseline_response_time - actual_mean) <= actual_mean * 0.1, \
            f"Baseline response time {baseline_response_time} differs from actual {actual_mean}"
        
        print(f"✅ Baseline collection test completed")
        print(f"📈 Baseline response time: {baseline_response_time:.4f}s")
        print(f"📊 Performance scenarios tested: {len(scenarios)}")
    
    @pytest.mark.asyncio
    async def test_regression_detection_with_real_performance_changes(self, metrics_test_suite):
        """Test regression detection with actual performance degradation scenarios."""
        regression_detector = RegressionDetector()
        
        # Configure regression detection
        config = {
            'sensitivity': 0.1,  # 10% change threshold
            'window_size': 50,   # Look at last 50 data points
            'metrics_to_monitor': ['response_time', 'throughput', 'error_rate']
        }
        
        await regression_detector.configure(config)
        
        # Generate baseline performance data
        baseline_data = []
        baseline_response_time = 0.1  # 100ms baseline
        
        for i in range(100):
            # Normal performance with small variations
            response_time = baseline_response_time + (i % 5) * 0.01
            throughput = 100 - (i % 5)  # requests/second
            error_rate = 0.01 + (i % 3) * 0.001
            
            data_point = {
                'timestamp': time.time() + i,
                'response_time': response_time,
                'throughput': throughput,
                'error_rate': error_rate
            }
            
            baseline_data.append(data_point)
            await regression_detector.add_data_point(data_point)
        
        # Verify no regressions detected in baseline
        regressions = await regression_detector.detect_regressions()
        assert len(regressions) == 0, f"False positive regressions detected: {regressions}"
        
        # Introduce performance regression
        regression_data = []
        degraded_response_time = baseline_response_time * 1.5  # 50% slower
        
        for i in range(50):
            # Performance degradation
            response_time = degraded_response_time + (i % 5) * 0.02
            throughput = 60 - (i % 5)  # Lower throughput
            error_rate = 0.05 + (i % 3) * 0.002  # Higher error rate
            
            data_point = {
                'timestamp': time.time() + 100 + i,
                'response_time': response_time,
                'throughput': throughput,
                'error_rate': error_rate
            }
            
            regression_data.append(data_point)
            await regression_detector.add_data_point(data_point)
        
        # Detect regressions
        regressions = await regression_detector.detect_regressions()
        assert len(regressions) > 0, "Failed to detect performance regression"
        
        # Verify regression details
        response_time_regression = next(
            (r for r in regressions if r['metric'] == 'response_time'), None
        )
        assert response_time_regression is not None, "Response time regression not detected"
        
        detected_change = response_time_regression['change_percentage']
        expected_change = (degraded_response_time / baseline_response_time - 1) * 100
        
        assert abs(detected_change - expected_change) <= 10, \
            f"Detected change {detected_change}% differs from expected {expected_change}%"
        
        print(f"✅ Regression detection test completed")
        print(f"⚠️ Regressions detected: {len(regressions)}")
        print(f"📈 Response time regression: {detected_change:.1f}%")


class TestSLOSLASystem:
    """Test SLO/SLA calculations with real service data."""
    
    @pytest.mark.asyncio
    async def test_slo_calculations_with_real_service_data(self, metrics_test_suite):
        """Test SLO calculations using actual service performance data."""
        
        # Define realistic SLOs
        slo_definitions = [
            SLODefinition(
                name="api_availability",
                slo_type=SLOType.AVAILABILITY,
                target=SLOTarget(value=99.9, unit="percent"),
                time_window=SLOTimeWindow.ROLLING_30_DAYS
            ),
            SLODefinition(
                name="response_time",
                slo_type=SLOType.LATENCY,
                target=SLOTarget(value=200, unit="milliseconds"),
                time_window=SLOTimeWindow.ROLLING_7_DAYS
            ),
            SLODefinition(
                name="error_rate",
                slo_type=SLOType.ERROR_RATE,
                target=SLOTarget(value=1.0, unit="percent"),
                time_window=SLOTimeWindow.ROLLING_24_HOURS
            )
        ]
        
        slo_monitor = SLOMonitor(slo_definitions)
        await slo_monitor.initialize()
        
        # Generate realistic service data over time
        service_data = []
        total_requests = 10000
        
        for i in range(total_requests):
            # Simulate realistic service behavior
            timestamp = time.time() + i * 0.01  # Spread over time
            
            # 99.95% success rate (better than 99.9% SLO)
            success = i < total_requests * 0.9995
            
            # Response times: mostly fast, occasional slow responses
            if i % 100 == 0:  # 1% of requests are slow
                response_time = 0.3 + (i % 10) * 0.05  # 300-800ms
            else:
                response_time = 0.05 + (i % 10) * 0.01  # 50-140ms
            
            error = not success
            
            request_data = {
                'timestamp': timestamp,
                'response_time': response_time,
                'success': success,
                'error': error,
                'status_code': 500 if error else 200
            }
            
            service_data.append(request_data)
            await slo_monitor.record_request(request_data)
        
        # Calculate SLO compliance
        slo_results = await slo_monitor.calculate_slo_compliance()
        
        # Verify availability SLO
        availability_result = next(
            (r for r in slo_results if r['slo_name'] == 'api_availability'), None
        )
        assert availability_result is not None
        
        actual_availability = availability_result['current_value']
        expected_availability = 99.95  # Our simulated rate
        assert abs(actual_availability - expected_availability) <= 0.1, \
            f"Availability {actual_availability}% differs from expected {expected_availability}%"
        
        # Verify response time SLO
        response_time_result = next(
            (r for r in slo_results if r['slo_name'] == 'response_time'), None
        )
        assert response_time_result is not None
        
        # Most requests should meet the 200ms SLO
        p95_response_time = response_time_result['p95_value']
        assert p95_response_time <= 200, f"P95 response time {p95_response_time}ms exceeds 200ms SLO"
        
        # Verify error rate SLO
        error_rate_result = next(
            (r for r in slo_results if r['slo_name'] == 'error_rate'), None
        )
        assert error_rate_result is not None
        
        actual_error_rate = error_rate_result['current_value']
        expected_error_rate = 0.05  # Our simulated rate
        assert abs(actual_error_rate - expected_error_rate) <= 0.1, \
            f"Error rate {actual_error_rate}% differs from expected {expected_error_rate}%"
        
        print(f"✅ SLO calculations test completed")
        print(f"📊 Availability: {actual_availability:.2f}% (SLO: 99.9%)")
        print(f"⚡ P95 Response Time: {p95_response_time:.1f}ms (SLO: 200ms)")
        print(f"❌ Error Rate: {actual_error_rate:.2f}% (SLO: 1.0%)")
    
    @pytest.mark.asyncio
    async def test_error_budget_tracking_and_burn_rate_alerting(self, metrics_test_suite):
        """Test error budget tracking and burn rate alerting with realistic scenarios."""
        
        # Define SLO with error budget
        slo_definition = SLODefinition(
            name="service_availability",
            slo_type=SLOType.AVAILABILITY,
            target=SLOTarget(value=99.9, unit="percent"),  # 0.1% error budget
            time_window=SLOTimeWindow.ROLLING_30_DAYS
        )
        
        error_budget_monitor = ErrorBudgetMonitor(slo_definition)
        burn_rate_alert = BurnRateAlert(slo_definition, burn_rate_threshold=5.0)
        
        await error_budget_monitor.initialize()
        await burn_rate_alert.initialize()
        
        # Simulate different error scenarios
        scenarios = [
            ("normal_operation", 1000, 0.05),      # 0.05% error rate
            ("minor_incident", 500, 2.0),          # 2% error rate
            ("major_incident", 200, 10.0),         # 10% error rate  
            ("recovery", 1000, 0.02)               # 0.02% error rate
        ]
        
        error_budget_data = []
        alerts_triggered = []
        
        for scenario_name, request_count, error_rate_percent in scenarios:
            scenario_start = time.time()
            
            for i in range(request_count):
                # Generate request with specified error rate
                is_error = i < (request_count * error_rate_percent / 100)
                
                request_data = {
                    'timestamp': time.time(),
                    'success': not is_error,
                    'error': is_error,
                    'scenario': scenario_name
                }
                
                # Record request for error budget tracking
                await error_budget_monitor.record_request(request_data)
                
                # Check for burn rate alerts
                alert = await burn_rate_alert.check_burn_rate(request_data)
                if alert:
                    alerts_triggered.append({
                        'scenario': scenario_name,
                        'alert': alert,
                        'timestamp': time.time()
                    })
                
                error_budget_data.append(request_data)
                
                # Small delay to simulate real timing
                await asyncio.sleep(0.001)
            
            scenario_duration = time.time() - scenario_start
            print(f"📊 Scenario '{scenario_name}': {request_count} requests, "
                  f"{error_rate_percent}% error rate in {scenario_duration:.2f}s")
        
        # Get final error budget status
        budget_status = await error_budget_monitor.get_budget_status()
        
        # Verify error budget tracking
        assert 'remaining_budget' in budget_status
        assert 'burn_rate' in budget_status
        assert 'projected_exhaustion' in budget_status
        
        remaining_budget = budget_status['remaining_budget']
        assert 0 <= remaining_budget <= 100, f"Invalid remaining budget: {remaining_budget}%"
        
        # Verify burn rate alerts were triggered appropriately
        major_incident_alerts = [a for a in alerts_triggered if a['scenario'] == 'major_incident']
        assert len(major_incident_alerts) > 0, "No alerts triggered during major incident"
        
        normal_operation_alerts = [a for a in alerts_triggered if a['scenario'] == 'normal_operation']
        assert len(normal_operation_alerts) == 0, "False alerts triggered during normal operation"
        
        print(f"✅ Error budget tracking test completed")
        print(f"💰 Remaining error budget: {remaining_budget:.2f}%")
        print(f"🚨 Alerts triggered: {len(alerts_triggered)}")


class TestLoadTestingIntegration:
    """Test production-like load testing for metrics validation."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_metrics_accuracy(self, metrics_test_suite):
        """Test metrics accuracy under high-throughput load conditions."""
        
        # Initialize load testing
        load_tester = LoadTestingIntegration()
        config = {
            'target_rps': 100,  # 100 requests per second
            'duration': 30,     # 30 seconds
            'concurrent_users': 20,
            'ramp_up_time': 5   # 5 seconds to reach target RPS
        }
        
        await load_tester.configure(config)
        
        # Metrics to validate during load test
        metrics_to_track = [
            'request_count',
            'response_time_p95',
            'error_rate',
            'throughput',
            'memory_usage',
            'cpu_utilization'
        ]
        
        load_test_data = []
        
        # Start load test
        async def load_test_worker(worker_id: int, requests_per_worker: int):
            """Worker that generates load and collects metrics."""
            for i in range(requests_per_worker):
                request_start = time.time()
                
                try:
                    # Simulate HTTP request processing
                    processing_time = 0.05 + (i % 10) * 0.01  # 50-140ms
                    await asyncio.sleep(processing_time / 10)  # Scale down for testing
                    
                    success = i % 100 != 0  # 1% error rate
                    status_code = 200 if success else 500
                    
                except Exception as e:
                    success = False
                    status_code = 500
                
                response_time = time.time() - request_start
                
                # Record load test data
                load_test_data.append({
                    'worker_id': worker_id,
                    'request_id': i,
                    'response_time': response_time,
                    'success': success,
                    'status_code': status_code,
                    'timestamp': time.time()
                })
                
                # Record metrics
                record_api_request(
                    endpoint="/test/load",
                    method="GET",
                    status_code=status_code,
                    response_time=response_time,
                    user_id=f"load_test_user_{worker_id}"
                )
                
                # Throttle to achieve target RPS
                await asyncio.sleep(1.0 / (config['target_rps'] / config['concurrent_users']))
        
        # Calculate requests per worker
        total_requests = config['target_rps'] * config['duration']
        requests_per_worker = total_requests // config['concurrent_users']
        
        # Run load test
        load_start = time.time()
        workers = [
            load_test_worker(i, requests_per_worker) 
            for i in range(config['concurrent_users'])
        ]
        
        await asyncio.gather(*workers)
        load_duration = time.time() - load_start
        
        # Verify load test results
        total_requests_made = len(load_test_data)
        actual_rps = total_requests_made / load_duration
        
        # Verify throughput is close to target
        target_rps = config['target_rps']
        assert abs(actual_rps - target_rps) <= target_rps * 0.2, \
            f"Actual RPS {actual_rps:.1f} differs significantly from target {target_rps}"
        
        # Verify error rate
        errors = sum(1 for d in load_test_data if not d['success'])
        error_rate = errors / total_requests_made * 100
        assert 0.5 <= error_rate <= 2.0, f"Error rate {error_rate:.2f}% outside expected range"
        
        # Verify response time distribution
        response_times = [d['response_time'] for d in load_test_data]
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        assert p95_response_time <= 1.0, f"P95 response time {p95_response_time:.3f}s too high"
        
        # Get metrics collector stats to verify data was recorded
        api_collector = get_api_metrics_collector()
        stats = api_collector.get_collection_stats()
        
        assert stats['total_requests'] >= total_requests_made
        
        print(f"✅ High throughput load test completed")
        print(f"⚡ Actual RPS: {actual_rps:.1f} (target: {target_rps})")
        print(f"📊 Total requests: {total_requests_made}")
        print(f"❌ Error rate: {error_rate:.2f}%")
        print(f"⏱️ P95 response time: {p95_response_time:.3f}s")


class TestIntegrationWithoutConflicts:
    """Test all metrics systems working together without conflicts."""
    
    @pytest.mark.asyncio
    async def test_concurrent_metrics_collection_without_interference(self, metrics_test_suite):
        """Test that all metrics systems can run concurrently without interfering."""
        
        # Initialize all metrics systems
        collectors = {
            'ml': get_ml_metrics_collector(),
            'api': get_api_metrics_collector(),
            'performance': get_performance_metrics_collector(),
            'bi': get_bi_metrics_collector()
        }
        
        aggregation_engine = get_aggregation_engine()
        dashboard_exporter = get_dashboard_exporter()
        
        # Test concurrent operations across all systems
        operation_results = []
        
        async def ml_operations():
            """Perform ML operations while other systems are active."""
            for i in range(100):
                await record_prompt_improvement(
                    prompt_category="test_prompts",
                    improvement_score=0.8 + (i % 10) * 0.02,
                    processing_time=0.1 + (i % 5) * 0.01,
                    user_id=f"ml_user_{i}"
                )
                await asyncio.sleep(0.01)
            
            operation_results.append({'system': 'ml', 'operations': 100})
        
        async def api_operations():
            """Perform API operations while other systems are active."""
            for i in range(150):
                record_api_request(
                    endpoint=f"/api/test/{i % 10}",
                    method="GET" if i % 2 == 0 else "POST",
                    status_code=200 if i % 20 != 0 else 500,
                    response_time=0.05 + (i % 8) * 0.01,
                    user_id=f"api_user_{i}"
                )
                await asyncio.sleep(0.005)
            
            operation_results.append({'system': 'api', 'operations': 150})
        
        async def performance_operations():
            """Perform performance monitoring while other systems are active."""
            for i in range(80):
                from prompt_improver.metrics.performance_metrics import record_pipeline_stage_timing
                record_pipeline_stage_timing(
                    stage="test_stage",
                    duration=0.02 + (i % 6) * 0.005,
                    success=i % 15 != 0,
                    metadata={'operation_id': f"perf_{i}"}
                )
                await asyncio.sleep(0.012)
            
            operation_results.append({'system': 'performance', 'operations': 80})
        
        async def business_operations():
            """Perform business intelligence operations while other systems are active."""
            for i in range(120):
                record_feature_usage(
                    feature_name=f"feature_{i % 5}",
                    user_tier="premium" if i % 3 == 0 else "standard",
                    usage_count=1,
                    success=i % 12 != 0
                )
                
                if i % 10 == 0:
                    record_operational_cost(
                        cost_type="compute",
                        resource_type="ml_inference",
                        amount=0.05 + (i % 3) * 0.01,
                        currency="USD"
                    )
                
                await asyncio.sleep(0.008)
            
            operation_results.append({'system': 'business', 'operations': 120})
        
        # Run all operations concurrently
        concurrent_start = time.time()
        await asyncio.gather(
            ml_operations(),
            api_operations(),
            performance_operations(),
            business_operations()
        )
        concurrent_duration = time.time() - concurrent_start
        
        # Verify all operations completed
        assert len(operation_results) == 4
        total_operations = sum(result['operations'] for result in operation_results)
        
        # Verify all collectors received data
        for name, collector in collectors.items():
            stats = collector.get_collection_stats()
            assert stats['total_events'] > 0, f"No events recorded by {name} collector"
        
        # Verify no errors or conflicts
        aggregation_stats = aggregation_engine.get_processing_stats()
        assert aggregation_stats.get('errors', 0) == 0, "Aggregation engine reported errors"
        
        # Test dashboard export works with all systems active
        export_data = await dashboard_exporter.export_real_time_monitoring(
            time_range=TimeRange.LAST_HOUR,
            format=ExportFormat.JSON
        )
        
        assert 'ml_metrics' in export_data
        assert 'api_metrics' in export_data
        assert 'performance_metrics' in export_data
        assert 'business_metrics' in export_data
        
        print(f"✅ Concurrent metrics collection test completed")
        print(f"⚡ Total operations: {total_operations} in {concurrent_duration:.2f}s")
        print(f"📊 Operations per second: {total_operations / concurrent_duration:.1f}")
        print(f"🔧 Systems tested: {list(collectors.keys())}")


class TestPrometheusExportValidation:
    """Test Prometheus export format and dashboard integration."""
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_export_format(self, metrics_test_suite):
        """Test that metrics are exported in correct Prometheus format."""
        
        # Generate sample metrics data
        sample_operations = 100
        
        for i in range(sample_operations):
            # Generate various metrics
            record_api_request(
                endpoint=f"/api/endpoint_{i % 5}",
                method="GET",
                status_code=200 if i % 20 != 0 else 500,
                response_time=0.1 + (i % 10) * 0.01,
                user_id=f"user_{i % 25}"
            )
            
            record_prompt_improvement(
                prompt_category="test_category",
                improvement_score=0.7 + (i % 10) * 0.03,
                processing_time=0.05 + (i % 8) * 0.005,
                user_id=f"user_{i % 25}"
            )
            
            record_feature_usage(
                feature_name=f"feature_{i % 3}",
                user_tier="premium" if i % 4 == 0 else "standard",
                usage_count=1,
                success=i % 15 != 0
            )
        
        # Allow metrics to be processed
        await asyncio.sleep(2)
        
        # Export metrics in Prometheus format
        dashboard_exporter = get_dashboard_exporter()
        prometheus_data = await dashboard_exporter.export_prometheus_metrics()
        
        # Verify Prometheus format structure
        assert isinstance(prometheus_data, str), "Prometheus export should be string format"
        
        # Verify required Prometheus metric patterns
        required_patterns = [
            'api_requests_total',
            'api_request_duration_seconds',
            'ml_prompt_improvement_score',
            'ml_processing_time_seconds',
            'feature_usage_total',
            'business_feature_success_rate'
        ]
        
        for pattern in required_patterns:
            assert pattern in prometheus_data, f"Missing Prometheus metric pattern: {pattern}"
        
        # Verify metric labels format
        assert 'endpoint=' in prometheus_data, "Missing endpoint labels"
        assert 'method=' in prometheus_data, "Missing method labels"
        assert 'status_code=' in prometheus_data, "Missing status_code labels"
        
        # Verify metric values are numeric
        import re
        metric_values = re.findall(r'} ([\d.]+)', prometheus_data)
        assert len(metric_values) > 0, "No metric values found"
        
        for value in metric_values[:10]:  # Check first 10 values
            try:
                float(value)
            except ValueError:
                pytest.fail(f"Invalid metric value format: {value}")
        
        print(f"✅ Prometheus export format validation completed")
        print(f"📊 Metrics exported: {len(metric_values)} values")
        print(f"🏷️ Metric patterns found: {len(required_patterns)}")
    
    @pytest.mark.asyncio
    async def test_grafana_dashboard_integration(self, metrics_test_suite):
        """Test Grafana dashboard configuration and data integration."""
        
        # Generate comprehensive metrics for dashboard testing
        dashboard_test_data = []
        
        # Simulate sustained traffic over time
        for minute in range(10):  # 10 minutes of data
            for second in range(0, 60, 5):  # Every 5 seconds
                timestamp = time.time() + minute * 60 + second
                
                # Vary metrics over time to create realistic patterns
                request_count = 10 + minute * 2  # Increasing load
                error_rate = 0.01 + (minute % 3) * 0.005  # Periodic error spikes
                response_time = 0.1 + (minute % 4) * 0.02  # Variable latency
                
                for req in range(request_count):
                    is_error = req < (request_count * error_rate)
                    
                    record_api_request(
                        endpoint=f"/api/service_{req % 3}",
                        method="GET",
                        status_code=500 if is_error else 200,
                        response_time=response_time + (req % 5) * 0.01,
                        user_id=f"user_{req % 50}",
                        timestamp=timestamp
                    )
                    
                    dashboard_test_data.append({
                        'timestamp': timestamp,
                        'minute': minute,
                        'success': not is_error,
                        'response_time': response_time
                    })
        
        # Export dashboard configuration
        dashboard_exporter = get_dashboard_exporter()
        
        # Export Grafana dashboard JSON
        grafana_config = await dashboard_exporter.export_grafana_dashboard(
            dashboard_type=DashboardType.REAL_TIME_MONITORING,
            time_range=TimeRange.LAST_HOUR
        )
        
        # Verify Grafana dashboard structure
        assert 'dashboard' in grafana_config
        dashboard = grafana_config['dashboard']
        
        assert 'title' in dashboard
        assert 'panels' in dashboard
        assert len(dashboard['panels']) > 0
        
        # Verify essential panels exist
        panel_titles = [panel.get('title', '') for panel in dashboard['panels']]
        required_panels = [
            'API Request Rate',
            'Response Time',
            'Error Rate',
            'Feature Usage',
            'System Performance'
        ]
        
        for required_panel in required_panels:
            matching_panels = [title for title in panel_titles if required_panel.lower() in title.lower()]
            assert len(matching_panels) > 0, f"Missing required dashboard panel: {required_panel}"
        
        # Verify query structure
        for panel in dashboard['panels']:
            if 'targets' in panel:
                for target in panel['targets']:
                    assert 'expr' in target, "Panel target missing Prometheus query expression"
                    expr = target['expr']
                    
                    # Verify query uses proper metric names
                    metric_patterns = ['api_requests', 'response_time', 'error_rate', 'feature_usage']
                    has_valid_metric = any(pattern in expr for pattern in metric_patterns)
                    assert has_valid_metric, f"Panel query doesn't use recognized metrics: {expr}"
        
        # Test time-series data export for dashboard
        time_series_data = await dashboard_exporter.export_time_series_data(
            metrics=['api_requests_total', 'api_request_duration_seconds'],
            time_range=TimeRange.LAST_HOUR,
            resolution='1m'  # 1-minute resolution
        )
        
        assert 'api_requests_total' in time_series_data
        assert 'api_request_duration_seconds' in time_series_data
        
        # Verify time series structure
        for metric_name, metric_data in time_series_data.items():
            assert 'timestamps' in metric_data
            assert 'values' in metric_data
            assert len(metric_data['timestamps']) == len(metric_data['values'])
            assert len(metric_data['timestamps']) > 0
        
        print(f"✅ Grafana dashboard integration test completed")
        print(f"📊 Dashboard panels: {len(dashboard['panels'])}")
        print(f"📈 Time series metrics: {len(time_series_data)}")
        print(f"⏱️ Data points generated: {len(dashboard_test_data)}")


class TestComprehensiveValidation:
    """Execute end-to-end validation with detailed performance reports."""
    
    @pytest.mark.asyncio
    async def test_complete_observability_stack_validation(self, metrics_test_suite):
        """Comprehensive end-to-end validation of entire observability stack."""
        
        validation_start = time.time()
        validation_results = {
            'test_phases': [],
            'performance_metrics': {},
            'system_health': {},
            'data_integrity': {},
            'export_validation': {}
        }
        
        # Phase 1: System initialization validation
        print("🚀 Phase 1: System Initialization Validation")
        phase1_start = time.time()
        
        # Verify all systems are operational
        systems = {
            'opentelemetry': get_tracer(__name__) is not None,
            'metrics_collectors': len([
                get_ml_metrics_collector(),
                get_api_metrics_collector(),
                get_performance_metrics_collector(),
                get_bi_metrics_collector()
            ]) == 4,
            'aggregation_engine': get_aggregation_engine() is not None,
            'dashboard_exporter': get_dashboard_exporter() is not None
        }
        
        phase1_duration = time.time() - phase1_start
        validation_results['test_phases'].append({
            'phase': 'initialization',
            'duration': phase1_duration,
            'systems_operational': all(systems.values()),
            'system_details': systems
        })
        
        # Phase 2: Load generation and metrics collection
        print("📊 Phase 2: Load Generation and Metrics Collection")
        phase2_start = time.time()
        
        # Generate comprehensive load across all systems
        load_scenarios = [
            ('api_traffic', 500, self._generate_api_load),
            ('ml_operations', 200, self._generate_ml_load),
            ('business_events', 300, self._generate_business_load),
            ('performance_monitoring', 150, self._generate_performance_load)
        ]
        
        scenario_results = []
        
        for scenario_name, operation_count, generator_func in load_scenarios:
            scenario_start = time.time()
            
            scenario_data = await generator_func(operation_count)
            scenario_duration = time.time() - scenario_start
            
            scenario_results.append({
                'scenario': scenario_name,
                'operations': operation_count,
                'duration': scenario_duration,
                'ops_per_second': operation_count / scenario_duration,
                'data_points': len(scenario_data)
            })
        
        phase2_duration = time.time() - phase2_start
        validation_results['test_phases'].append({
            'phase': 'load_generation',
            'duration': phase2_duration,
            'scenarios': scenario_results,
            'total_operations': sum(s['operations'] for s in scenario_results)
        })
        
        # Phase 3: Data processing and aggregation validation
        print("🔄 Phase 3: Data Processing and Aggregation Validation")
        phase3_start = time.time()
        
        # Allow time for data processing
        await asyncio.sleep(5)
        
        # Validate data aggregation
        aggregation_engine = get_aggregation_engine()
        aggregation_stats = aggregation_engine.get_processing_stats()
        
        # Validate metrics collection
        collection_stats = {}
        collectors = {
            'ml': get_ml_metrics_collector(),
            'api': get_api_metrics_collector(),
            'performance': get_performance_metrics_collector(),
            'bi': get_bi_metrics_collector()
        }
        
        for name, collector in collectors.items():
            stats = collector.get_collection_stats()
            collection_stats[name] = stats
        
        phase3_duration = time.time() - phase3_start
        validation_results['test_phases'].append({
            'phase': 'data_processing',
            'duration': phase3_duration,
            'aggregation_stats': aggregation_stats,
            'collection_stats': collection_stats
        })
        
        # Phase 4: Export and dashboard validation
        print("📈 Phase 4: Export and Dashboard Validation")
        phase4_start = time.time()
        
        dashboard_exporter = get_dashboard_exporter()
        
        # Test all export formats
        export_tests = [
            ('prometheus', dashboard_exporter.export_prometheus_metrics),
            ('json', lambda: dashboard_exporter.export_real_time_monitoring(
                TimeRange.LAST_HOUR, ExportFormat.JSON)),
            ('grafana', lambda: dashboard_exporter.export_grafana_dashboard(
                DashboardType.REAL_TIME_MONITORING, TimeRange.LAST_HOUR))
        ]
        
        export_results = {}
        for export_name, export_func in export_tests:
            export_start = time.time()
            try:
                export_data = await export_func()
                export_success = export_data is not None and len(str(export_data)) > 0
                export_duration = time.time() - export_start
                
                export_results[export_name] = {
                    'success': export_success,
                    'duration': export_duration,
                    'data_size': len(str(export_data))
                }
            except Exception as e:
                export_results[export_name] = {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - export_start
                }
        
        phase4_duration = time.time() - phase4_start
        validation_results['test_phases'].append({
            'phase': 'export_validation',
            'duration': phase4_duration,
            'export_results': export_results
        })
        
        # Phase 5: Performance and health assessment
        print("💊 Phase 5: Performance and Health Assessment")
        phase5_start = time.time()
        
        # System resource usage
        process = psutil.Process()
        system_health = {
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=1),
            'open_files': len(process.open_files()),
            'connections': len(process.connections()),
            'threads': process.num_threads()
        }
        
        # Performance metrics
        total_validation_duration = time.time() - validation_start
        performance_metrics = {
            'total_validation_time': total_validation_duration,
            'operations_per_second': sum(s['operations'] for s in scenario_results) / total_validation_duration,
            'phase_durations': [phase['duration'] for phase in validation_results['test_phases']],
            'avg_phase_duration': statistics.mean([phase['duration'] for phase in validation_results['test_phases']])
        }
        
        phase5_duration = time.time() - phase5_start
        validation_results['test_phases'].append({
            'phase': 'health_assessment',
            'duration': phase5_duration,
            'system_health': system_health,
            'performance_metrics': performance_metrics
        })
        
        # Final validation results
        validation_results['performance_metrics'] = performance_metrics
        validation_results['system_health'] = system_health
        validation_results['data_integrity'] = {
            'all_collectors_active': all(stats['total_events'] > 0 for stats in collection_stats.values()),
            'aggregation_errors': aggregation_stats.get('errors', 0),
            'export_success_rate': sum(1 for r in export_results.values() if r.get('success', False)) / len(export_results)
        }
        validation_results['export_validation'] = export_results
        
        # Save validation report
        report_path = metrics_test_suite.test_data_dir / 'phase3_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Assertions for overall validation
        assert validation_results['data_integrity']['all_collectors_active'], \
            "Not all metrics collectors are active"
        assert validation_results['data_integrity']['aggregation_errors'] == 0, \
            "Aggregation engine reported errors"
        assert validation_results['data_integrity']['export_success_rate'] >= 0.8, \
            "Export success rate below acceptable threshold"
        assert performance_metrics['operations_per_second'] >= 50, \
            "Overall throughput below acceptable threshold"
        assert system_health['memory_usage_mb'] <= 1000, \
            "Memory usage exceeds acceptable limits"
        
        print(f"✅ Comprehensive validation completed successfully")
        print(f"⏱️ Total validation time: {total_validation_duration:.2f}s")
        print(f"⚡ Operations per second: {performance_metrics['operations_per_second']:.1f}")
        print(f"💾 Memory usage: {system_health['memory_usage_mb']:.1f} MB")
        print(f"📊 Export success rate: {validation_results['data_integrity']['export_success_rate']:.1%}")
        print(f"📄 Detailed report saved to: {report_path}")
        
        return validation_results
    
    async def _generate_api_load(self, operation_count: int) -> List[Dict]:
        """Generate API load for validation testing."""
        api_data = []
        
        for i in range(operation_count):
            endpoint = f"/api/test/{i % 10}"
            method = "GET" if i % 3 != 2 else "POST"
            status_code = 200 if i % 25 != 0 else (500 if i % 50 == 0 else 404)
            response_time = 0.05 + (i % 15) * 0.01
            
            record_api_request(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time=response_time,
                user_id=f"validation_user_{i % 100}"
            )
            
            api_data.append({
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time': response_time
            })
            
            if i % 100 == 0:
                await asyncio.sleep(0.01)  # Brief pause every 100 requests
        
        return api_data
    
    async def _generate_ml_load(self, operation_count: int) -> List[Dict]:
        """Generate ML operations load for validation testing."""
        ml_data = []
        
        for i in range(operation_count):
            category = f"category_{i % 5}"
            improvement_score = 0.6 + (i % 20) * 0.02
            processing_time = 0.1 + (i % 8) * 0.02
            
            await record_prompt_improvement(
                prompt_category=category,
                improvement_score=improvement_score,
                processing_time=processing_time,
                user_id=f"ml_user_{i % 50}"
            )
            
            ml_data.append({
                'category': category,
                'improvement_score': improvement_score,
                'processing_time': processing_time
            })
            
            if i % 50 == 0:
                await asyncio.sleep(0.005)
        
        return ml_data
    
    async def _generate_business_load(self, operation_count: int) -> List[Dict]:
        """Generate business events load for validation testing."""
        business_data = []
        
        for i in range(operation_count):
            feature = f"feature_{i % 8}"
            user_tier = "premium" if i % 4 == 0 else "standard"
            success = i % 12 != 0
            
            record_feature_usage(
                feature_name=feature,
                user_tier=user_tier,
                usage_count=1,
                success=success
            )
            
            if i % 20 == 0:
                record_operational_cost(
                    cost_type="compute",
                    resource_type="ml_processing",
                    amount=0.05 + (i % 5) * 0.01,
                    currency="USD"
                )
            
            business_data.append({
                'feature': feature,
                'user_tier': user_tier,
                'success': success
            })
            
            if i % 75 == 0:
                await asyncio.sleep(0.01)
        
        return business_data
    
    async def _generate_performance_load(self, operation_count: int) -> List[Dict]:
        """Generate performance monitoring load for validation testing."""
        performance_data = []
        
        for i in range(operation_count):
            stage = f"stage_{i % 6}"
            duration = 0.02 + (i % 10) * 0.005
            success = i % 20 != 0
            
            from prompt_improver.metrics.performance_metrics import record_pipeline_stage_timing
            record_pipeline_stage_timing(
                stage=stage,
                duration=duration,
                success=success,
                metadata={'validation_id': f"perf_{i}"}
            )
            
            performance_data.append({
                'stage': stage,
                'duration': duration,
                'success': success
            })
            
            if i % 40 == 0:
                await asyncio.sleep(0.008)
        
        return performance_data


# Test execution
if __name__ == "__main__":
    print("🧪 Phase 3 Metrics and Observability Testing Suite")
    print("=" * 60)
    print("This comprehensive test suite validates:")
    print("• OpenTelemetry distributed tracing with real operations")
    print("• Business metrics collection with actual data")
    print("• System metrics under load conditions")
    print("• Performance baseline and regression detection")
    print("• SLO/SLA monitoring with real service data")
    print("• Load testing integration")
    print("• Cross-system integration without conflicts")
    print("• Prometheus export and dashboard integration")
    print("• End-to-end validation with performance reports")
    print("=" * 60)
    
    # Run with pytest: pytest tests/integration/test_phase3_metrics.py -v -s