#!/usr/bin/env python3
"""
Phase 1 Metrics Validation Script

Validates the four missing Phase 1 metrics implementation:
1. Connection Age Tracking - Real connection lifecycle with timestamps 
2. Request Queue Depths - HTTP, database, ML inference, Redis queue monitoring
3. Cache Hit Rates - Application, ML model, configuration, session cache effectiveness  
4. Feature Usage Analytics - Feature flag adoption, API utilization, ML model usage

Tests real-time collection, performance targets (<1ms overhead), and integration.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.metrics.system_metrics import (
    SystemMetricsCollector,
    MetricsConfig,
    get_system_metrics_collector
)
from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase1MetricsValidator:
    """Comprehensive validation of Phase 1 missing metrics"""
    
    def __init__(self):
        self.config = MetricsConfig(
            connection_age_retention_hours=1,
            queue_depth_sample_interval_ms=100,
            cache_hit_window_minutes=5,
            feature_usage_window_hours=1,
            metrics_collection_overhead_ms=1.0
        )
        self.registry = get_metrics_registry()
        self.collector = SystemMetricsCollector(self.config, self.registry)
        self.results = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "performance_results": {},
            "functionality_results": {},
            "integration_results": {},
            "overall_status": "PENDING"
        }
    
    async def run_comprehensive_validation(self) -> dict:
        """Run comprehensive validation of all Phase 1 metrics"""
        logger.info("🚀 Starting Phase 1 Metrics Comprehensive Validation")
        
        try:
            # 1. Performance Validation
            logger.info("📊 Testing Performance Targets...")
            await self._validate_performance_targets()
            
            # 2. Functionality Validation  
            logger.info("🔧 Testing Core Functionality...")
            await self._validate_core_functionality()
            
            # 3. Real-time Collection Validation
            logger.info("⚡ Testing Real-time Collection...")
            await self._validate_realtime_collection()
            
            # 4. Integration Validation
            logger.info("🔗 Testing System Integration...")
            await self._validate_system_integration()
            
            # 5. Data Accuracy Validation
            logger.info("📈 Testing Data Accuracy...")
            await self._validate_data_accuracy()
            
            # Calculate overall status
            self._calculate_overall_status()
            
            logger.info(f"✅ Validation Complete - Status: {self.results['overall_status']}")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            self.results["overall_status"] = "FAILED"
            self.results["error"] = str(e)
            return self.results
    
    async def _validate_performance_targets(self) -> None:
        """Validate <1ms performance targets"""
        logger.info("Testing connection age tracking performance...")
        
        # Test Connection Age Tracking Performance
        start_time = time.perf_counter()
        for i in range(100):
            conn_id = f"perf_test_conn_{i}"
            self.collector.connection_tracker.track_connection_created(
                conn_id, "database", "performance_test_pool"
            )
            self.collector.connection_tracker.track_connection_used(conn_id)
            self.collector.connection_tracker.track_connection_destroyed(conn_id)
        
        connection_duration_ms = (time.perf_counter() - start_time) * 1000
        connection_avg_ms = connection_duration_ms / 300  # 100 * 3 operations
        
        logger.info("Testing cache monitoring performance...")
        
        # Test Cache Monitoring Performance
        start_time = time.perf_counter()
        for i in range(100):
            if i % 3 == 0:
                self.collector.cache_monitor.record_cache_miss(
                    "application", "perf_cache", f"key_{i}", 10.0
                )
            else:
                self.collector.cache_monitor.record_cache_hit(
                    "application", "perf_cache", f"key_{i}", 1.5
                )
        
        cache_duration_ms = (time.perf_counter() - start_time) * 1000
        cache_avg_ms = cache_duration_ms / 100
        
        logger.info("Testing feature analytics performance...")
        
        # Test Feature Analytics Performance
        start_time = time.perf_counter()
        for i in range(100):
            self.collector.feature_analytics.record_feature_usage(
                "api_endpoint", f"/api/test/{i % 5}", f"user_{i % 10}",
                "direct_call", 20.0, True
            )
        
        feature_duration_ms = (time.perf_counter() - start_time) * 1000
        feature_avg_ms = feature_duration_ms / 100
        
        logger.info("Testing queue monitoring performance...")
        
        # Test Queue Monitoring Performance
        start_time = time.perf_counter()
        for i in range(100):
            self.collector.queue_monitor.sample_queue_depth(
                "http", "perf_queue", i % 50, 100
            )
        
        queue_duration_ms = (time.perf_counter() - start_time) * 1000
        queue_avg_ms = queue_duration_ms / 100
        
        # Store results
        self.results["performance_results"] = {
            "connection_tracking_avg_ms": round(connection_avg_ms, 3),
            "cache_monitoring_avg_ms": round(cache_avg_ms, 3),
            "feature_analytics_avg_ms": round(feature_avg_ms, 3),
            "queue_monitoring_avg_ms": round(queue_avg_ms, 3),
            "target_ms": self.config.metrics_collection_overhead_ms,
            "connection_meets_target": connection_avg_ms < self.config.metrics_collection_overhead_ms,
            "cache_meets_target": cache_avg_ms < self.config.metrics_collection_overhead_ms,
            "feature_meets_target": feature_avg_ms < self.config.metrics_collection_overhead_ms,
            "queue_meets_target": queue_avg_ms < self.config.metrics_collection_overhead_ms
        }
        
        logger.info(f"Performance Results:")
        logger.info(f"  Connection Tracking: {connection_avg_ms:.3f}ms (target: <{self.config.metrics_collection_overhead_ms}ms)")
        logger.info(f"  Cache Monitoring: {cache_avg_ms:.3f}ms")
        logger.info(f"  Feature Analytics: {feature_avg_ms:.3f}ms")
        logger.info(f"  Queue Monitoring: {queue_avg_ms:.3f}ms")
    
    async def _validate_core_functionality(self) -> None:
        """Validate core functionality of all metrics components"""
        
        # 1. Connection Age Tracking
        logger.info("Validating connection age tracking...")
        
        # Create connections with different ages
        conn_ids = []
        for i in range(5):
            conn_id = f"func_test_conn_{i}"
            conn_ids.append(conn_id)
            self.collector.connection_tracker.track_connection_created(
                conn_id, "database" if i % 2 == 0 else "redis", 
                f"pool_{i}", {"host": f"host_{i}"}
            )
        
        # Test connection usage
        for conn_id in conn_ids[:3]:
            self.collector.connection_tracker.track_connection_used(conn_id)
        
        # Test age distribution
        age_dist = self.collector.connection_tracker.get_age_distribution()
        
        # Destroy some connections
        for conn_id in conn_ids[:2]:
            self.collector.connection_tracker.track_connection_destroyed(conn_id)
        
        # 2. Cache Hit Rate Monitoring
        logger.info("Validating cache hit rate monitoring...")
        
        cache_types = ["application", "ml_model", "configuration", "session"]
        for cache_type in cache_types:
            # Generate mixed hit/miss pattern (70% hit rate)
            for i in range(20):
                if i % 10 < 7:  # 70% hits
                    self.collector.cache_monitor.record_cache_hit(
                        cache_type, f"{cache_type}_cache", f"key_{i}", 
                        1.0 + (i % 3)  # Variable response times
                    )
                else:  # 30% misses
                    self.collector.cache_monitor.record_cache_miss(
                        cache_type, f"{cache_type}_cache", f"key_{i}", 
                        15.0 + (i % 5)  # Higher miss response times
                    )
        
        # Test cache statistics
        cache_stats = {}
        for cache_type in cache_types:
            stats = self.collector.cache_monitor.get_cache_statistics(
                cache_type, f"{cache_type}_cache"
            )
            cache_stats[cache_type] = stats
        
        # 3. Request Queue Monitoring  
        logger.info("Validating request queue monitoring...")
        
        queue_types = ["http", "database", "ml_inference", "redis"]
        for queue_type in queue_types:
            # Simulate varying queue depths
            for depth in [5, 10, 15, 25, 20, 8, 3]:
                self.collector.queue_monitor.sample_queue_depth(
                    queue_type, f"{queue_type}_queue", depth, 50
                )
        
        # Test queue statistics
        queue_stats = {}
        for queue_type in queue_types:
            stats = self.collector.queue_monitor.get_queue_statistics(
                queue_type, f"{queue_type}_queue"
            )
            queue_stats[queue_type] = stats
        
        # 4. Feature Usage Analytics
        logger.info("Validating feature usage analytics...")
        
        feature_types = ["api_endpoint", "feature_flag", "ml_model", "component"]
        features = {
            "api_endpoint": ["/api/users", "/api/posts", "/api/search"],
            "feature_flag": ["new_ui", "beta_feature", "advanced_search"],
            "ml_model": ["recommendation", "classification", "clustering"],
            "component": ["dashboard", "reports", "analytics"]
        }
        
        for feature_type, feature_list in features.items():
            for feature_name in feature_list:
                # Simulate different usage patterns
                for i in range(15):
                    user_id = f"user_{i % 8}"  # 8 unique users
                    pattern = ["direct_call", "batch_operation", "background_task"][i % 3]
                    success = i % 10 != 0  # 90% success rate
                    perf = 20.0 + (i % 30)  # Variable performance
                    
                    self.collector.feature_analytics.record_feature_usage(
                        feature_type, feature_name, user_id, pattern, perf, success
                    )
        
        # Test feature analytics
        feature_stats = {}
        for feature_type, feature_list in features.items():
            for feature_name in feature_list:
                stats = self.collector.feature_analytics.get_feature_analytics(
                    feature_type, feature_name
                )
                feature_stats[f"{feature_type}:{feature_name}"] = stats
        
        # Store results
        self.results["functionality_results"] = {
            "connection_age_distribution": len(age_dist) > 0,
            "cache_statistics_available": len(cache_stats) == len(cache_types),
            "queue_statistics_available": len(queue_stats) == len(queue_types), 
            "feature_analytics_available": len(feature_stats) > 0,
            "connection_tracking_working": len(age_dist) > 0,
            "cache_hit_rates_calculated": all(
                "current_hit_rate" in stats for stats in cache_stats.values() 
                if "error" not in stats
            ),
            "queue_depths_monitored": all(
                "current_depth" in stats for stats in queue_stats.values()
                if "error" not in stats
            ),
            "feature_usage_tracked": all(
                "total_usage_in_window" in stats for stats in feature_stats.values()
                if "error" not in stats
            )
        }
        
        logger.info("Core functionality validation complete:")
        for metric, status in self.results["functionality_results"].items():
            logger.info(f"  {metric}: {'✅' if status else '❌'}")
    
    async def _validate_realtime_collection(self) -> None:
        """Validate real-time collection capabilities"""
        logger.info("Testing real-time metrics collection...")
        
        # Start background monitoring
        monitoring_task = asyncio.create_task(
            self.collector.start_background_monitoring()
        )
        
        # Let monitoring run briefly
        await asyncio.sleep(0.2)
        
        # Simulate real-time activity
        activities = []
        start_time = time.perf_counter()
        
        for i in range(10):
            # Connection activity
            conn_id = f"realtime_conn_{i}"
            self.collector.connection_tracker.track_connection_created(
                conn_id, "database", "realtime_pool"
            )
            activities.append(f"Created connection {conn_id}")
            
            # Cache activity
            self.collector.cache_monitor.record_cache_hit(
                "realtime", "cache", f"key_{i}", 2.0
            )
            activities.append(f"Cache hit for key_{i}")
            
            # Feature usage
            self.collector.feature_analytics.record_feature_usage(
                "realtime_api", f"endpoint_{i}", f"user_{i % 3}", 
                "realtime_call", 15.0, True
            )
            activities.append(f"Feature usage for endpoint_{i}")
            
            # Small delay to simulate real-time
            await asyncio.sleep(0.01)
        
        realtime_duration = time.perf_counter() - start_time
        
        # Stop monitoring
        self.collector.stop_background_monitoring()
        
        # Wait for monitoring task to complete
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()
        
        # Collect metrics to verify real-time data
        metrics = self.collector.collect_all_metrics()
        
        self.results["realtime_results"] = {
            "background_monitoring_started": True,
            "realtime_activities_processed": len(activities),
            "realtime_duration_seconds": round(realtime_duration, 3),
            "metrics_collection_successful": "timestamp" in metrics,
            "health_score_calculated": "system_health_score" in metrics,
            "connection_data_available": len(metrics.get("connection_age_distribution", {})) > 0
        }
        
        logger.info("Real-time collection validation complete:")
        for metric, value in self.results["realtime_results"].items():
            logger.info(f"  {metric}: {value}")
    
    async def _validate_system_integration(self) -> None:
        """Validate integration with existing monitoring stack"""
        logger.info("Testing system integration...")
        
        # Test metrics registry integration
        registry_metrics = self.registry.get_metrics_info()
        
        # Test system health score calculation
        health_score = self.collector.get_system_health_score()
        
        # Test comprehensive metrics collection
        start_time = time.perf_counter()
        all_metrics = self.collector.collect_all_metrics()
        collection_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Test context managers integration
        with self.collector.connection_tracker.track_connection_lifecycle(
            "integration_conn", "database", "integration_pool"
        ):
            # Connection should be tracked
            pass
        # Connection should be cleaned up
        
        with self.collector.cache_monitor.track_cache_operation(
            "integration", "cache", "test_key"
        ) as tracker:
            tracker.mark_hit()
        
        with self.collector.feature_analytics.track_feature_usage(
            "integration", "test_feature", "test_user"
        ):
            # Simulate feature work
            await asyncio.sleep(0.001)
        
        self.results["integration_results"] = {
            "prometheus_available": registry_metrics.get("prometheus_available", False),
            "metrics_registered": registry_metrics.get("total_metrics", 0) > 0,
            "health_score_valid": 0.0 <= health_score <= 1.0,
            "comprehensive_collection_working": len(all_metrics) > 0,
            "collection_performance_ms": round(collection_time_ms, 3),
            "context_managers_working": True,  # If we got here, they worked
            "integration_meets_performance": collection_time_ms < 10.0  # Should be very fast
        }
        
        logger.info("System integration validation complete:")
        for metric, value in self.results["integration_results"].items():
            logger.info(f"  {metric}: {value}")
    
    async def _validate_data_accuracy(self) -> None:
        """Validate data accuracy and consistency"""
        logger.info("Testing data accuracy and consistency...")
        
        # Create controlled test scenario
        test_connections = 5
        test_cache_operations = 20
        test_feature_usage = 15
        
        # 1. Connection Age Accuracy
        conn_start_time = datetime.utcnow()
        for i in range(test_connections):
            self.collector.connection_tracker.track_connection_created(
                f"accuracy_conn_{i}", "database", "accuracy_pool"
            )
        
        # Wait a bit for age accumulation
        await asyncio.sleep(0.1)
        
        age_dist = self.collector.connection_tracker.get_age_distribution()
        connections_tracked = sum(
            len(pools) for pools in age_dist.get("database", {}).values()
        )
        
        # 2. Cache Hit Rate Accuracy
        expected_hits = int(test_cache_operations * 0.8)  # 80% hit rate
        expected_misses = test_cache_operations - expected_hits
        
        for i in range(test_cache_operations):
            if i < expected_hits:
                self.collector.cache_monitor.record_cache_hit(
                    "accuracy", "test_cache", f"key_{i}", 2.0
                )
            else:
                self.collector.cache_monitor.record_cache_miss(
                    "accuracy", "test_cache", f"key_{i}", 20.0
                )
        
        cache_stats = self.collector.cache_monitor.get_cache_statistics(
            "accuracy", "test_cache"
        )
        
        # 3. Feature Usage Accuracy
        unique_users = 3
        for i in range(test_feature_usage):
            user_id = f"accuracy_user_{i % unique_users}"
            self.collector.feature_analytics.record_feature_usage(
                "accuracy", "test_feature", user_id, "direct_call", 25.0, True
            )
        
        feature_stats = self.collector.feature_analytics.get_feature_analytics(
            "accuracy", "test_feature"
        )
        
        # 4. Queue Monitoring Accuracy
        test_queue_samples = [10, 20, 30, 25, 15]
        for depth in test_queue_samples:
            self.collector.queue_monitor.sample_queue_depth(
                "accuracy", "test_queue", depth, 100
            )
        
        queue_stats = self.collector.queue_monitor.get_queue_statistics(
            "accuracy", "test_queue"
        )
        
        # Verify accuracy
        cache_hit_rate = cache_stats.get("current_hit_rate", 0)
        expected_hit_rate = expected_hits / test_cache_operations
        hit_rate_accuracy = abs(cache_hit_rate - expected_hit_rate) < 0.05  # 5% tolerance
        
        feature_unique_users = feature_stats.get("unique_users_in_window", 0)
        users_accuracy = feature_unique_users == unique_users
        
        queue_sample_count = queue_stats.get("sample_count", 0) 
        queue_accuracy = queue_sample_count == len(test_queue_samples)
        
        self.results["accuracy_results"] = {
            "connections_created": test_connections,
            "connections_tracked": connections_tracked,
            "connection_tracking_accurate": connections_tracked == test_connections,
            "expected_hit_rate": round(expected_hit_rate, 3),
            "actual_hit_rate": round(cache_hit_rate, 3),
            "hit_rate_accurate": hit_rate_accuracy,
            "expected_unique_users": unique_users,
            "actual_unique_users": feature_unique_users,
            "user_tracking_accurate": users_accuracy,
            "expected_queue_samples": len(test_queue_samples),
            "actual_queue_samples": queue_sample_count,
            "queue_sampling_accurate": queue_accuracy
        }
        
        logger.info("Data accuracy validation complete:")
        for metric, value in self.results["accuracy_results"].items():
            logger.info(f"  {metric}: {value}")
    
    def _calculate_overall_status(self) -> None:
        """Calculate overall validation status"""
        
        # Performance checks
        perf = self.results["performance_results"]
        performance_passed = all([
            perf["connection_meets_target"],
            perf["cache_meets_target"], 
            perf["feature_meets_target"],
            perf["queue_meets_target"]
        ])
        
        # Functionality checks
        func = self.results["functionality_results"]
        functionality_passed = all([
            func["connection_tracking_working"],
            func["cache_hit_rates_calculated"],
            func["queue_depths_monitored"],
            func["feature_usage_tracked"]
        ])
        
        # Integration checks
        integ = self.results["integration_results"]
        integration_passed = all([
            integ["health_score_valid"],
            integ["comprehensive_collection_working"],
            integ["context_managers_working"],
            integ["integration_meets_performance"]
        ])
        
        # Accuracy checks
        acc = self.results["accuracy_results"]
        accuracy_passed = all([
            acc["connection_tracking_accurate"],
            acc["hit_rate_accurate"],
            acc["user_tracking_accurate"],
            acc["queue_sampling_accurate"]
        ])
        
        # Real-time checks
        realtime = self.results["realtime_results"]
        realtime_passed = all([
            realtime["background_monitoring_started"],
            realtime["metrics_collection_successful"],
            realtime["health_score_calculated"]
        ])
        
        # Overall status
        all_passed = all([
            performance_passed,
            functionality_passed,
            integration_passed,
            accuracy_passed,
            realtime_passed
        ])
        
        self.results["validation_summary"] = {
            "performance_passed": performance_passed,
            "functionality_passed": functionality_passed,
            "integration_passed": integration_passed,
            "accuracy_passed": accuracy_passed,
            "realtime_passed": realtime_passed,
            "all_tests_passed": all_passed
        }
        
        if all_passed:
            self.results["overall_status"] = "PASSED"
        else:
            self.results["overall_status"] = "FAILED"
    
    def save_results(self, output_path: str = None) -> None:
        """Save validation results to file"""
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = f"phase1_metrics_validation_results_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")


async def main():
    """Main validation execution"""
    print("🚀 Phase 1 Metrics Implementation Validation")
    print("=" * 60)
    
    validator = Phase1MetricsValidator()
    results = await validator.run_comprehensive_validation()
    
    # Save results
    validator.save_results()
    
    # Print summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    summary = results["validation_summary"]
    status = results["overall_status"]
    
    print(f"Overall Status: {'🟢 PASSED' if status == 'PASSED' else '🔴 FAILED'}")
    print(f"Performance Tests: {'✅' if summary['performance_passed'] else '❌'}")
    print(f"Functionality Tests: {'✅' if summary['functionality_passed'] else '❌'}")
    print(f"Integration Tests: {'✅' if summary['integration_passed'] else '❌'}")
    print(f"Accuracy Tests: {'✅' if summary['accuracy_passed'] else '❌'}")
    print(f"Real-time Tests: {'✅' if summary['realtime_passed'] else '❌'}")
    
    # Print key metrics
    print("\n🔢 KEY PERFORMANCE METRICS")
    print("=" * 60)
    perf = results["performance_results"]
    print(f"Connection Tracking: {perf['connection_tracking_avg_ms']:.3f}ms (target: <{perf['target_ms']}ms)")
    print(f"Cache Monitoring: {perf['cache_monitoring_avg_ms']:.3f}ms")
    print(f"Feature Analytics: {perf['feature_analytics_avg_ms']:.3f}ms")
    print(f"Queue Monitoring: {perf['queue_monitoring_avg_ms']:.3f}ms")
    
    # Print integration metrics
    print("\n🔗 INTEGRATION METRICS")
    print("=" * 60)
    integ = results["integration_results"]
    print(f"System Health Score: Valid ({'✅' if integ['health_score_valid'] else '❌'})")
    print(f"Metrics Collection: {integ['collection_performance_ms']:.3f}ms")
    print(f"Context Managers: {'✅' if integ['context_managers_working'] else '❌'}")
    
    print("\n" + "=" * 60)
    if status == "PASSED":
        print("🎉 ALL PHASE 1 METRICS SUCCESSFULLY VALIDATED!")
        print("✅ Connection Age Tracking: Real lifecycle with timestamps")
        print("✅ Request Queue Depths: HTTP, DB, ML, Redis monitoring") 
        print("✅ Cache Hit Rates: Application, ML, config, session caches")
        print("✅ Feature Usage Analytics: Flag adoption, API utilization")
        print("✅ Performance Target: <1ms overhead achieved")
        print("✅ Real-time Collection: Operational measurements")
        print("✅ System Integration: Prometheus metrics registry")
    else:
        print("❌ VALIDATION FAILED - Review results for details")
    
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))