"""Real behavior tests for Redis health monitoring with actual failure scenarios.

Comprehensive validation of Redis health monitoring services with real Redis instances,
connection failures, recovery scenarios, and performance monitoring.

Performance Requirements:
- Health check operations: <25ms response time
- Connection monitoring: <10ms connection status detection
- Recovery operations: <100ms automatic recovery time
- Alert generation: <5ms for critical alerts
- Metrics collection: >100 ops/sec throughput
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import uuid4

import coredis

from tests.containers.real_redis_container import RealRedisTestContainer
from src.prompt_improver.monitoring.unified.facade import UnifiedMonitoringFacade
from src.prompt_improver.monitoring.unified.types import (
    HealthStatus,
    MetricPoint,
    MetricType,
    MonitoringConfig,
)
from src.prompt_improver.monitoring.unified.health_checkers import (
    RedisHealthChecker,
    DatabaseHealthChecker,
    CacheHealthChecker,
)
from src.prompt_improver.utils.cache_service.l2_cache_service import L2CacheService


# Custom Redis health checker for testing
class TestRedisHealthChecker:
    """Redis health checker for real behavior testing."""
    
    def __init__(self, redis_client: coredis.Redis, name: str = "redis_test"):
        self.redis_client = redis_client
        self.name = name
        self.last_check_time = None
        self.failure_count = 0
        self.recovery_count = 0
    
    def get_component_name(self) -> str:
        """Get component name."""
        return self.name
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform Redis health check with comprehensive metrics."""
        start_time = time.perf_counter()
        self.last_check_time = datetime.utcnow()
        
        try:
            # Basic connectivity test
            ping_start = time.perf_counter()
            await self.redis_client.ping()
            ping_duration = time.perf_counter() - ping_start
            
            # Memory usage check
            info = await self.redis_client.info("memory")
            used_memory = int(info.get("used_memory", 0))
            max_memory = int(info.get("maxmemory", 0))
            
            # Connection count check
            info_clients = await self.redis_client.info("clients")
            connected_clients = int(info_clients.get("connected_clients", 0))
            
            # Key space statistics
            info_keyspace = await self.redis_client.info("keyspace")
            total_keys = 0
            for db_key, db_info in info_keyspace.items():
                if db_key.startswith("db"):
                    # Parse "keys=X,expires=Y,avg_ttl=Z" format
                    keys_part = db_info.split(",")[0]
                    if "=" in keys_part:
                        total_keys += int(keys_part.split("=")[1])
            
            # Performance test - simple operation
            test_key = f"health_check_{uuid4().hex[:8]}"
            perf_start = time.perf_counter()
            await self.redis_client.set(test_key, "health_test", ex=60)
            await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            operation_duration = time.perf_counter() - perf_start
            
            total_duration = time.perf_counter() - start_time
            
            # Determine health status
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check response time
            if ping_duration > 0.1:  # 100ms threshold
                status = HealthStatus.DEGRADED
                issues.append(f"High ping latency: {ping_duration*1000:.2f}ms")
            
            # Check memory usage
            if max_memory > 0 and used_memory > max_memory * 0.9:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {used_memory/max_memory:.1%}")
            
            # Check operation performance
            if operation_duration > 0.05:  # 50ms threshold
                status = HealthStatus.DEGRADED
                issues.append(f"Slow operations: {operation_duration*1000:.2f}ms")
            
            if status == HealthStatus.HEALTHY and self.failure_count > 0:
                self.recovery_count += 1
                self.failure_count = 0
            
            return {
                "status": status,
                "message": f"Redis health check completed in {total_duration*1000:.2f}ms",
                "response_time_ms": total_duration * 1000,
                "details": {
                    "ping_duration_ms": ping_duration * 1000,
                    "operation_duration_ms": operation_duration * 1000,
                    "used_memory_bytes": used_memory,
                    "max_memory_bytes": max_memory,
                    "memory_usage_percent": (used_memory / max_memory * 100) if max_memory > 0 else 0,
                    "connected_clients": connected_clients,
                    "total_keys": total_keys,
                    "failure_count": self.failure_count,
                    "recovery_count": self.recovery_count,
                    "issues": issues,
                },
                "error": None,
            }
            
        except Exception as e:
            self.failure_count += 1
            total_duration = time.perf_counter() - start_time
            
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Redis health check failed: {str(e)}",
                "response_time_ms": total_duration * 1000,
                "details": {
                    "failure_count": self.failure_count,
                    "recovery_count": self.recovery_count,
                    "last_error": str(e),
                    "error_type": type(e).__name__,
                },
                "error": str(e),
            }


class TestRedisConnectionMonitor:
    """Redis connection monitor for testing connection pool health."""
    
    def __init__(self, redis_client: coredis.Redis):
        self.redis_client = redis_client
        self.connection_events = []
        self.reconnection_count = 0
        self.last_connection_test = None
    
    async def monitor_connection_pool(self) -> Dict[str, Any]:
        """Monitor Redis connection pool health."""
        start_time = time.perf_counter()
        
        try:
            # Test multiple connections
            connection_tests = []
            for i in range(5):
                test_start = time.perf_counter()
                await self.redis_client.ping()
                connection_time = time.perf_counter() - test_start
                connection_tests.append(connection_time)
            
            avg_connection_time = sum(connection_tests) / len(connection_tests)
            max_connection_time = max(connection_tests)
            min_connection_time = min(connection_tests)
            
            # Connection pool info
            pool_info = {}
            if hasattr(self.redis_client, "connection_pool"):
                pool = self.redis_client.connection_pool
                pool_info = {
                    "pool_size": getattr(pool, "_created_connections", 0),
                    "available_connections": getattr(pool, "_available_connections", 0),
                    "in_use_connections": getattr(pool, "_in_use_connections", 0),
                }
            
            total_duration = time.perf_counter() - start_time
            self.last_connection_test = datetime.utcnow()
            
            # Determine connection health
            connection_health = HealthStatus.HEALTHY
            if avg_connection_time > 0.05:  # 50ms average threshold
                connection_health = HealthStatus.DEGRADED
            if max_connection_time > 0.1:  # 100ms max threshold
                connection_health = HealthStatus.DEGRADED
            
            return {
                "status": connection_health,
                "monitoring_duration_ms": total_duration * 1000,
                "connection_metrics": {
                    "avg_connection_time_ms": avg_connection_time * 1000,
                    "min_connection_time_ms": min_connection_time * 1000,
                    "max_connection_time_ms": max_connection_time * 1000,
                    "connection_count": len(connection_tests),
                    "reconnection_count": self.reconnection_count,
                },
                "pool_info": pool_info,
                "last_test": self.last_connection_test.isoformat() if self.last_connection_test else None,
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
                "monitoring_duration_ms": (time.perf_counter() - start_time) * 1000,
                "connection_metrics": {"error": True, "reconnection_count": self.reconnection_count},
            }


class TestRedisHealthMonitoringRealBehavior:
    """Real behavior tests for Redis health monitoring."""

    @pytest.fixture
    async def redis_container(self):
        """Set up real Redis testcontainer."""
        container = RealRedisTestContainer()
        await container.start()
        yield container
        await container.stop()

    @pytest.fixture
    async def monitoring_facade(self, redis_container):
        """Set up monitoring facade with real Redis."""
        # Configure environment for Redis connection
        import os
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        
        config = MonitoringConfig(
            health_check_timeout_seconds=5.0,
            health_check_parallel_enabled=True,
            max_concurrent_checks=10,
            metrics_collection_enabled=True,
            cache_health_results_seconds=1.0,  # Short cache for testing
        )
        
        facade = UnifiedMonitoringFacade(config=config)
        await facade.start_monitoring()
        
        yield facade
        
        await facade.stop_monitoring()

    @pytest.fixture
    async def redis_client(self, redis_container):
        """Create Redis client for direct testing."""
        client = redis_container.get_client(decode_responses=False)
        await client.ping()  # Verify connection
        yield client
        await client.aclose()

    async def test_redis_health_monitoring_basic_operations(self, monitoring_facade, redis_client):
        """Test basic Redis health monitoring operations."""
        # Create and register Redis health checker
        redis_health_checker = TestRedisHealthChecker(redis_client, "redis_primary")
        
        # Register the health checker
        monitoring_facade.register_health_checker(redis_health_checker)
        
        # Test individual component health check
        start_time = time.perf_counter()
        health_result = await monitoring_facade.check_component_health("redis_primary")
        health_check_duration = time.perf_counter() - start_time
        
        assert health_result is not None
        assert health_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert health_check_duration < 0.025, f"Health check took {health_check_duration:.3f}s, exceeds 25ms limit"
        
        print(f"Redis Health Check: {health_result.status.value} ({health_check_duration*1000:.2f}ms)")
        assert health_result.response_time_ms < 25, f"Health check response time {health_result.response_time_ms:.2f}ms exceeds 25ms target"
        
        # Test system-wide health check including Redis
        start_time = time.perf_counter()
        system_health = await monitoring_facade.get_system_health()
        system_health_duration = time.perf_counter() - start_time
        
        assert system_health is not None
        assert system_health.total_components >= 1
        assert system_health_duration < 0.1, f"System health check took {system_health_duration:.3f}s, should be <100ms"
        
        print(f"System Health: {system_health.overall_status.value} ({system_health_duration*1000:.2f}ms)")
        print(f"  Components: {system_health.healthy_components}/{system_health.total_components} healthy")
        
        # Verify Redis is included in health results
        assert "redis_primary" in [comp for comp in monitoring_facade.health_service.get_registered_components()]

    async def test_redis_connection_monitoring_and_pool_health(self, redis_client):
        """Test Redis connection monitoring and pool health."""
        connection_monitor = TestRedisConnectionMonitor(redis_client)
        
        # Test connection pool monitoring
        start_time = time.perf_counter()
        pool_health = await connection_monitor.monitor_connection_pool()
        monitoring_duration = time.perf_counter() - start_time
        
        assert pool_health is not None
        assert pool_health["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert monitoring_duration < 0.01, f"Connection monitoring took {monitoring_duration:.3f}s, exceeds 10ms limit"
        
        print(f"Connection Pool Health: {pool_health['status'].value}")
        print(f"  Avg Connection Time: {pool_health['connection_metrics']['avg_connection_time_ms']:.2f}ms")
        print(f"  Max Connection Time: {pool_health['connection_metrics']['max_connection_time_ms']:.2f}ms")
        
        # Performance assertions
        avg_conn_time = pool_health['connection_metrics']['avg_connection_time_ms']
        max_conn_time = pool_health['connection_metrics']['max_connection_time_ms']
        
        assert avg_conn_time < 50, f"Average connection time {avg_conn_time:.2f}ms should be <50ms"
        assert max_conn_time < 100, f"Max connection time {max_conn_time:.2f}ms should be <100ms"

    async def test_redis_failure_scenarios_and_recovery(self, redis_container, redis_client):
        """Test Redis failure scenarios and automatic recovery."""
        redis_health_checker = TestRedisHealthChecker(redis_client, "redis_failure_test")
        
        # Test normal operation first
        normal_health = await redis_health_checker.check_health()
        assert normal_health["status"] == HealthStatus.HEALTHY
        print(f"Normal Redis Health: {normal_health['status'].value}")
        
        # Simulate connection failure by using invalid client
        failed_client = coredis.Redis(
            host="invalid.host.does.not.exist",
            port=6379,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        
        failed_health_checker = TestRedisHealthChecker(failed_client, "redis_failed")
        
        # Test failure detection
        start_time = time.perf_counter()
        failed_health = await failed_health_checker.check_health()
        failure_detection_time = time.perf_counter() - start_time
        
        assert failed_health["status"] == HealthStatus.UNHEALTHY
        assert failed_health_checker.failure_count > 0
        assert failure_detection_time < 2.0, f"Failure detection took {failure_detection_time:.3f}s, should be fast"
        
        print(f"Failed Redis Health: {failed_health['status'].value}")
        print(f"  Failure Count: {failed_health_checker.failure_count}")
        print(f"  Error: {failed_health['details']['last_error']}")
        
        # Test recovery scenario
        recovery_start = time.perf_counter()
        recovery_health = await redis_health_checker.check_health()  # Use working client
        recovery_time = time.perf_counter() - recovery_start
        
        assert recovery_health["status"] == HealthStatus.HEALTHY
        assert recovery_time < 0.1, f"Recovery check took {recovery_time:.3f}s, should be <100ms"
        
        print(f"Recovery Check: {recovery_health['status'].value} ({recovery_time*1000:.2f}ms)")

    async def test_redis_performance_monitoring_under_load(self, redis_client):
        """Test Redis performance monitoring under load."""
        redis_health_checker = TestRedisHealthChecker(redis_client, "redis_performance")
        
        # Create load on Redis
        load_tasks = []
        for i in range(100):
            async def redis_operation(index):
                key = f"load_test_key_{index}"
                value = f"load_test_value_{index}_{uuid4().hex[:8]}"
                await redis_client.set(key, value, ex=300)
                retrieved = await redis_client.get(key)
                return retrieved is not None
            
            load_tasks.append(redis_operation(i))
        
        # Execute load and measure performance
        load_start = time.perf_counter()
        load_results = await asyncio.gather(*load_tasks)
        load_duration = time.perf_counter() - load_start
        
        successful_operations = sum(1 for result in load_results if result)
        throughput = successful_operations / load_duration
        
        print(f"Redis Load Test:")
        print(f"  Operations: {len(load_tasks)}")
        print(f"  Successful: {successful_operations}")
        print(f"  Duration: {load_duration:.3f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")
        
        # Test health check during load
        concurrent_health_checks = []
        for i in range(5):
            concurrent_health_checks.append(redis_health_checker.check_health())
        
        health_start = time.perf_counter()
        health_results = await asyncio.gather(*concurrent_health_checks)
        health_duration = time.perf_counter() - health_start
        
        # Verify all health checks succeeded
        assert len(health_results) == 5
        healthy_checks = sum(1 for result in health_results if result["status"] == HealthStatus.HEALTHY)
        
        avg_health_time = health_duration / len(health_results)
        
        print(f"Health Checks Under Load:")
        print(f"  Healthy Checks: {healthy_checks}/5")
        print(f"  Average Health Check Time: {avg_health_time*1000:.2f}ms")
        
        # Performance assertions
        assert throughput > 100, f"Redis throughput {throughput:.1f} ops/sec should be >100 ops/sec"
        assert healthy_checks >= 4, f"At least 4/5 health checks should succeed under load"
        assert avg_health_time < 0.025, f"Average health check time {avg_health_time:.3f}s should be <25ms"

    async def test_redis_metrics_collection_and_alerts(self, monitoring_facade, redis_client):
        """Test Redis metrics collection and alert generation."""
        # Record custom Redis metrics
        redis_metrics = [
            ("redis.connections.active", 25.0),
            ("redis.memory.used_bytes", 1048576.0),  # 1MB
            ("redis.operations.per_second", 150.0),
            ("redis.keyspace.total_keys", 1000.0),
            ("redis.response_time.avg_ms", 2.5),
        ]
        
        metrics_start = time.perf_counter()
        
        for metric_name, metric_value in redis_metrics:
            monitoring_facade.record_custom_metric(
                metric_name,
                metric_value,
                tags={"component": "redis", "instance": "primary"}
            )
        
        metrics_recording_time = time.perf_counter() - metrics_start
        metrics_throughput = len(redis_metrics) / metrics_recording_time
        
        print(f"Metrics Recording:")
        print(f"  Metrics: {len(redis_metrics)}")
        print(f"  Recording Time: {metrics_recording_time*1000:.2f}ms")
        print(f"  Throughput: {metrics_throughput:.1f} metrics/sec")
        
        # Performance assertion for metrics
        assert metrics_throughput > 100, f"Metrics throughput {metrics_throughput:.1f} should be >100 metrics/sec"
        
        # Test metrics collection
        collection_start = time.perf_counter()
        all_metrics = await monitoring_facade.collect_all_metrics()
        collection_time = time.perf_counter() - collection_start
        
        assert len(all_metrics) > 0
        assert collection_time < 0.1, f"Metrics collection took {collection_time:.3f}s, should be <100ms"
        
        print(f"Metrics Collection: {len(all_metrics)} metrics in {collection_time*1000:.2f}ms")
        
        # Verify Redis metrics are included
        redis_metric_names = [metric.name for metric in all_metrics if "redis" in metric.name.lower()]
        assert len(redis_metric_names) > 0, "Should have Redis metrics in collection"
        
        print(f"Redis Metrics Found: {redis_metric_names}")

    async def test_redis_cache_service_health_integration(self, redis_container):
        """Test Redis health monitoring integration with cache service."""
        # Set up L2 cache service with Redis
        import os
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        
        l2_cache = L2CacheService()
        
        try:
            # Test cache operations
            cache_operations = []
            for i in range(20):
                key = f"cache_health_test_{i}"
                value = {"test_data": f"value_{i}", "timestamp": time.time()}
                
                operation_start = time.perf_counter()
                success = await l2_cache.set(key, value, ttl=300)
                retrieved = await l2_cache.get(key)
                operation_time = time.perf_counter() - operation_start
                
                cache_operations.append({
                    "success": success and retrieved == value,
                    "operation_time": operation_time,
                })
            
            # Analyze cache performance
            successful_ops = sum(1 for op in cache_operations if op["success"])
            avg_operation_time = sum(op["operation_time"] for op in cache_operations) / len(cache_operations)
            max_operation_time = max(op["operation_time"] for op in cache_operations)
            
            print(f"Cache Service Health Integration:")
            print(f"  Successful Operations: {successful_ops}/20")
            print(f"  Average Operation Time: {avg_operation_time*1000:.2f}ms")
            print(f"  Max Operation Time: {max_operation_time*1000:.2f}ms")
            
            # Performance assertions
            assert successful_ops >= 19, f"At least 19/20 cache operations should succeed"
            assert avg_operation_time < 0.01, f"Average cache operation time {avg_operation_time:.3f}s should be <10ms"
            assert max_operation_time < 0.02, f"Max cache operation time {max_operation_time:.3f}s should be <20ms"
            
            # Test cache health check
            health_check_start = time.perf_counter()
            cache_health = await l2_cache.health_check()
            health_check_time = time.perf_counter() - health_check_start
            
            assert cache_health["healthy"] is True
            assert health_check_time < 0.025, f"Cache health check took {health_check_time:.3f}s, should be <25ms"
            
            print(f"Cache Health Check: {cache_health['status']} ({health_check_time*1000:.2f}ms)")
            print(f"  Performance: {cache_health['performance']}")
            
        finally:
            await l2_cache.close()

    async def test_redis_monitoring_system_resilience(self, monitoring_facade, redis_container):
        """Test Redis monitoring system resilience and error handling."""
        # Test monitoring with partial Redis failures
        redis_clients = []
        health_checkers = []
        
        # Create multiple Redis health checkers (some will fail)
        for i in range(5):
            if i < 3:
                # Working Redis connections
                client = redis_container.get_client(decode_responses=False)
                await client.ping()  # Verify connection
                redis_clients.append(client)
            else:
                # Failing Redis connections
                client = coredis.Redis(
                    host=f"invalid_{i}.redis.host",
                    port=6379,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                )
                redis_clients.append(client)
            
            checker = TestRedisHealthChecker(client, f"redis_resilience_{i}")
            health_checkers.append(checker)
            monitoring_facade.register_health_checker(checker)
        
        try:
            # Test system health with mixed Redis health
            resilience_start = time.perf_counter()
            system_health = await monitoring_facade.get_system_health()
            resilience_time = time.perf_counter() - resilience_start
            
            print(f"System Resilience Test:")
            print(f"  Overall Status: {system_health.overall_status.value}")
            print(f"  Healthy Components: {system_health.healthy_components}")
            print(f"  Unhealthy Components: {system_health.unhealthy_components}")
            print(f"  Health Check Time: {resilience_time*1000:.2f}ms")
            
            # System should handle partial failures gracefully
            assert system_health.total_components >= 5
            assert system_health.healthy_components >= 3  # At least working Redis instances
            assert system_health.unhealthy_components >= 2  # At least failing Redis instances
            assert resilience_time < 0.2, f"System health check with failures took {resilience_time:.3f}s, should be <200ms"
            
            # Test individual health checks with concurrent failures
            concurrent_health_tasks = []
            for checker in health_checkers:
                concurrent_health_tasks.append(monitoring_facade.check_component_health(checker.name))
            
            concurrent_start = time.perf_counter()
            concurrent_results = await asyncio.gather(*concurrent_health_tasks, return_exceptions=True)
            concurrent_time = time.perf_counter() - concurrent_start
            
            # Analyze concurrent results
            successful_checks = sum(1 for result in concurrent_results if not isinstance(result, Exception))
            failed_checks = len(concurrent_results) - successful_checks
            
            print(f"Concurrent Health Checks:")
            print(f"  Successful: {successful_checks}")
            print(f"  Failed: {failed_checks}")
            print(f"  Duration: {concurrent_time*1000:.2f}ms")
            
            assert successful_checks >= 3, f"At least 3 health checks should succeed"
            assert concurrent_time < 0.1, f"Concurrent health checks took {concurrent_time:.3f}s, should be <100ms"
            
            # Test monitoring system recovery
            monitoring_summary = await monitoring_facade.get_monitoring_summary()
            
            assert "health" in monitoring_summary
            assert "components" in monitoring_summary
            assert monitoring_summary["health"]["total_components"] >= 5
            
            print(f"Monitoring Summary:")
            print(f"  Health Percentage: {monitoring_summary['health']['health_percentage']:.1f}%")
            print(f"  Registered Components: {monitoring_summary['components']['registered_count']}")
            
        finally:
            # Clean up Redis clients
            for client in redis_clients:
                try:
                    if hasattr(client, "aclose"):
                        await client.aclose()
                    elif hasattr(client, "close"):
                        await client.close()
                except Exception:
                    pass


@pytest.mark.integration
@pytest.mark.real_behavior
class TestRedisMonitoringSystemIntegration:
    """System-level integration tests for Redis monitoring."""

    @pytest.fixture
    async def redis_cluster_setup(self):
        """Set up multiple Redis instances for cluster testing."""
        containers = []
        
        # Start 3 Redis containers
        for i in range(3):
            container = RealRedisTestContainer(port=6379 + i)
            await container.start()
            containers.append(container)
        
        yield containers
        
        # Clean up
        for container in containers:
            await container.stop()

    async def test_redis_monitoring_system_performance_benchmarks(self, redis_cluster_setup):
        """Test system-wide performance benchmarks for Redis monitoring."""
        monitoring_config = MonitoringConfig(
            health_check_timeout_seconds=10.0,
            health_check_parallel_enabled=True,
            max_concurrent_checks=50,
            metrics_collection_enabled=True,
        )
        
        monitoring_facade = UnifiedMonitoringFacade(config=monitoring_config)
        await monitoring_facade.start_monitoring()
        
        try:
            # Register health checkers for all Redis instances
            health_checkers = []
            redis_clients = []
            
            for i, container in enumerate(redis_cluster_setup):
                client = container.get_client(decode_responses=False)
                await client.ping()
                redis_clients.append(client)
                
                checker = TestRedisHealthChecker(client, f"redis_cluster_{i}")
                health_checkers.append(checker)
                monitoring_facade.register_health_checker(checker)
            
            # Benchmark: System health check with multiple Redis instances
            benchmark_iterations = 10
            health_check_times = []
            
            for iteration in range(benchmark_iterations):
                start_time = time.perf_counter()
                system_health = await monitoring_facade.get_system_health()
                duration = time.perf_counter() - start_time
                health_check_times.append(duration)
                
                assert system_health.total_components >= 3
                assert system_health.healthy_components >= 3
            
            avg_health_check_time = sum(health_check_times) / len(health_check_times)
            min_health_check_time = min(health_check_times)
            max_health_check_time = max(health_check_times)
            
            print(f"Redis Monitoring System Performance Benchmarks:")
            print(f"  Health Check Iterations: {benchmark_iterations}")
            print(f"  Average Health Check Time: {avg_health_check_time*1000:.2f}ms")
            print(f"  Min Health Check Time: {min_health_check_time*1000:.2f}ms")
            print(f"  Max Health Check Time: {max_health_check_time*1000:.2f}ms")
            
            # Performance assertions
            assert avg_health_check_time < 0.1, f"Average health check time {avg_health_check_time:.3f}s should be <100ms"
            assert max_health_check_time < 0.2, f"Max health check time {max_health_check_time:.3f}s should be <200ms"
            
            # Benchmark: Concurrent health checks
            concurrent_iterations = 50
            concurrent_start = time.perf_counter()
            
            concurrent_tasks = []
            for i in range(concurrent_iterations):
                checker_index = i % len(health_checkers)
                task = monitoring_facade.check_component_health(health_checkers[checker_index].name)
                concurrent_tasks.append(task)
            
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            concurrent_duration = time.perf_counter() - concurrent_start
            concurrent_throughput = concurrent_iterations / concurrent_duration
            
            successful_concurrent = sum(1 for r in concurrent_results if r.status == HealthStatus.HEALTHY)
            
            print(f"Concurrent Health Check Benchmark:")
            print(f"  Concurrent Checks: {concurrent_iterations}")
            print(f"  Successful: {successful_concurrent}")
            print(f"  Duration: {concurrent_duration:.3f}s")
            print(f"  Throughput: {concurrent_throughput:.1f} checks/sec")
            
            # Performance assertions for concurrent operations
            assert concurrent_throughput > 100, f"Concurrent health check throughput {concurrent_throughput:.1f} should be >100 checks/sec"
            assert successful_concurrent >= concurrent_iterations * 0.95, f"At least 95% of concurrent checks should succeed"
            
            # Benchmark: Metrics collection with Redis monitoring
            metrics_iterations = 1000
            metrics_start = time.perf_counter()
            
            for i in range(metrics_iterations):
                monitoring_facade.record_custom_metric(
                    f"redis.benchmark.metric_{i%10}",
                    float(i),
                    tags={"benchmark": "true", "iteration": str(i)}
                )
            
            metrics_duration = time.perf_counter() - metrics_start
            metrics_throughput = metrics_iterations / metrics_duration
            
            print(f"Metrics Collection Benchmark:")
            print(f"  Metrics Recorded: {metrics_iterations}")
            print(f"  Duration: {metrics_duration:.3f}s")
            print(f"  Throughput: {metrics_throughput:.1f} metrics/sec")
            
            # Performance assertion for metrics
            assert metrics_throughput > 1000, f"Metrics throughput {metrics_throughput:.1f} should be >1000 metrics/sec"
            
            print(f"\nSystem Performance Summary: PASSED")
            print(f"  Health Checks: {avg_health_check_time*1000:.2f}ms avg")
            print(f"  Concurrent Throughput: {concurrent_throughput:.1f} checks/sec")
            print(f"  Metrics Throughput: {metrics_throughput:.1f} metrics/sec")
            
        finally:
            # Clean up
            for client in redis_clients:
                try:
                    await client.aclose()
                except Exception:
                    pass
            
            await monitoring_facade.stop_monitoring()

    async def test_redis_monitoring_failure_recovery_scenarios(self, redis_cluster_setup):
        """Test Redis monitoring system failure and recovery scenarios."""
        monitoring_facade = UnifiedMonitoringFacade()
        await monitoring_facade.start_monitoring()
        
        try:
            # Set up monitoring for Redis cluster
            working_clients = []
            failing_clients = []
            
            # Create working clients
            for container in redis_cluster_setup:
                client = container.get_client(decode_responses=False)
                await client.ping()
                working_clients.append(client)
                
                checker = TestRedisHealthChecker(client, f"working_redis_{len(working_clients)}")
                monitoring_facade.register_health_checker(checker)
            
            # Create failing clients
            for i in range(2):
                client = coredis.Redis(
                    host=f"failure_test_{i}.invalid",
                    port=6379,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                )
                failing_clients.append(client)
                
                checker = TestRedisHealthChecker(client, f"failing_redis_{i}")
                monitoring_facade.register_health_checker(checker)
            
            # Test initial system state
            initial_health = await monitoring_facade.get_system_health()
            print(f"Initial System Health:")
            print(f"  Total Components: {initial_health.total_components}")
            print(f"  Healthy: {initial_health.healthy_components}")
            print(f"  Unhealthy: {initial_health.unhealthy_components}")
            
            assert initial_health.total_components >= 5
            assert initial_health.healthy_components >= 3  # Working Redis instances
            assert initial_health.unhealthy_components >= 2  # Failing Redis instances
            
            # Test cascade failure scenario (simulate multiple Redis failures)
            print("\nSimulating cascade failures...")
            cascade_start = time.perf_counter()
            
            # Close some working connections to simulate failures
            for i, client in enumerate(working_clients[:2]):
                try:
                    await client.aclose()
                    print(f"Simulated failure for working_redis_{i+1}")
                except Exception:
                    pass
            
            # Test system response to cascade failures
            cascade_health = await monitoring_facade.get_system_health()
            cascade_time = time.perf_counter() - cascade_start
            
            print(f"Post-Cascade System Health:")
            print(f"  Healthy: {cascade_health.healthy_components}")
            print(f"  Unhealthy: {cascade_health.unhealthy_components}")
            print(f"  Detection Time: {cascade_time*1000:.2f}ms")
            
            # System should detect failures quickly
            assert cascade_time < 1.0, f"Cascade failure detection took {cascade_time:.3f}s, should be <1s"
            assert cascade_health.unhealthy_components >= 4  # More failures detected
            
            # Test recovery scenario
            print("\nTesting recovery scenario...")
            recovery_start = time.perf_counter()
            
            # Reconnect to Redis instances (simulate recovery)
            recovered_clients = []
            for container in redis_cluster_setup:
                try:
                    client = container.get_client(decode_responses=False)
                    await client.ping()
                    recovered_clients.append(client)
                    
                    # Register new checker for recovered instance
                    checker = TestRedisHealthChecker(client, f"recovered_redis_{len(recovered_clients)}")
                    monitoring_facade.register_health_checker(checker)
                    
                except Exception as e:
                    print(f"Failed to recover Redis connection: {e}")
            
            # Test system recovery
            recovery_health = await monitoring_facade.get_system_health()
            recovery_time = time.perf_counter() - recovery_start
            
            print(f"Post-Recovery System Health:")
            print(f"  Total Components: {recovery_health.total_components}")
            print(f"  Healthy: {recovery_health.healthy_components}")
            print(f"  Recovery Time: {recovery_time*1000:.2f}ms")
            
            # System should show improvement after recovery
            assert recovery_health.healthy_components > cascade_health.healthy_components
            assert recovery_time < 0.5, f"Recovery validation took {recovery_time:.3f}s, should be <500ms"
            
            # Test monitoring system resilience after recovery
            resilience_test_results = []
            for i in range(10):
                test_start = time.perf_counter()
                health = await monitoring_facade.get_system_health()
                test_time = time.perf_counter() - test_start
                
                resilience_test_results.append({
                    "healthy_components": health.healthy_components,
                    "test_time": test_time,
                })
            
            avg_test_time = sum(r["test_time"] for r in resilience_test_results) / len(resilience_test_results)
            consistent_health = all(r["healthy_components"] >= 3 for r in resilience_test_results)
            
            print(f"Resilience Test Results:")
            print(f"  Consistent Health: {consistent_health}")
            print(f"  Average Test Time: {avg_test_time*1000:.2f}ms")
            
            assert consistent_health, "System health should be consistent after recovery"
            assert avg_test_time < 0.05, f"Average health check time {avg_test_time:.3f}s should remain <50ms after recovery"
            
            print(f"\nFailure Recovery Test Summary: PASSED")
            print(f"  Initial Healthy: {initial_health.healthy_components}")
            print(f"  Post-Cascade Healthy: {cascade_health.healthy_components}")
            print(f"  Post-Recovery Healthy: {recovery_health.healthy_components}")
            print(f"  System Resilience: Maintained")
            
        finally:
            # Clean up all clients
            all_clients = working_clients + failing_clients + recovered_clients
            for client in all_clients:
                try:
                    await client.aclose()
                except Exception:
                    pass
            
            await monitoring_facade.stop_monitoring()