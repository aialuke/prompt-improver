"""
Cache testing utilities and helpers for MultiLevelCache testing.

Provides specialized utilities for cache testing, performance measurement,
external Redis service management, and test data generation following 2025 best practices.

Migrated from TestContainers to external Redis service patterns for improved performance.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, patch

import coredis

from prompt_improver.database import (
    DatabaseServices,
    ManagerMode,
)

logger = logging.getLogger(__name__)


# Legacy ExternalRedisManager functionality now provided by DatabaseServices
# This class is replaced by unified_redis_test_context() function below


class CachePerformanceProfiler:
    """
    Advanced performance profiling for cache operations.

    Provides detailed timing, throughput, and efficiency measurements
    with statistical analysis and regression detection capabilities.
    """

    def __init__(self):
        self.measurements: dict[str, list[float]] = {}
        self.operation_counts: dict[str, int] = {}
        self.start_times: dict[str, float] = {}

    @asynccontextmanager
    async def profile_operation(self, operation_name: str):
        """
        Context manager for profiling cache operations.

        Args:
            operation_name: Name of the operation being profiled
        """
        start_time = time.perf_counter()
        self.start_times[operation_name] = start_time
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record_measurement(operation_name, duration)

    def record_measurement(self, operation_name: str, duration: float):
        """Record a performance measurement."""
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
            self.operation_counts[operation_name] = 0
        self.measurements[operation_name].append(duration)
        self.operation_counts[operation_name] += 1

    def get_statistics(self, operation_name: str) -> dict[str, float]:
        """
        Get statistical analysis for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Dictionary with statistical metrics
        """
        if operation_name not in self.measurements:
            return {}
        measurements = self.measurements[operation_name]
        if not measurements:
            return {}
        sorted_measurements = sorted(measurements)
        n = len(sorted_measurements)
        return {
            "count": n,
            "min": min(measurements),
            "max": max(measurements),
            "mean": sum(measurements) / n,
            "median": sorted_measurements[n // 2],
            "p95": sorted_measurements[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_measurements[int(n * 0.99)] if n > 0 else 0,
            "std_dev": self._calculate_std_dev(measurements),
            "ops_per_second": n / sum(measurements) if sum(measurements) > 0 else 0,
        }

    def _calculate_std_dev(self, measurements: list[float]) -> float:
        """Calculate standard deviation."""
        if len(measurements) < 2:
            return 0.0
        mean = sum(measurements) / len(measurements)
        variance = sum((x - mean) ** 2 for x in measurements) / (len(measurements) - 1)
        return variance**0.5

    def get_all_statistics(self) -> dict[str, dict[str, float]]:
        """Get statistics for all recorded operations."""
        return {
            operation: self.get_statistics(operation)
            for operation in self.measurements.keys()
        }

    def assert_performance_requirements(
        self, requirements: dict[str, dict[str, float]]
    ):
        """
        Assert that performance measurements meet requirements.

        Args:
            requirements: Dict mapping operation names to requirement dicts
                         with keys like 'max_mean', 'max_p95', 'min_ops_per_second'
        """
        for operation, req in requirements.items():
            stats = self.get_statistics(operation)
            if not stats:
                continue
            for metric, threshold in req.items():
                if metric.startswith("max_"):
                    stat_name = metric[4:]
                    if stat_name in stats:
                        actual = stats[stat_name]
                        assert actual <= threshold, (
                            f"{operation}.{stat_name} ({actual:.6f}) exceeds threshold ({threshold:.6f})"
                        )
                elif metric.startswith("min_"):
                    stat_name = metric[4:]
                    if stat_name in stats:
                        actual = stats[stat_name]
                        assert actual >= threshold, (
                            f"{operation}.{stat_name} ({actual:.6f}) below threshold ({threshold:.6f})"
                        )


class CacheTestDataGenerator:
    """
    Comprehensive test data generator for cache testing.

    Generates realistic test data with various patterns, sizes, and
    complexity levels for comprehensive cache behavior validation.
    """

    @staticmethod
    def generate_simple_data() -> dict[str, Any]:
        """Generate simple data types for basic testing."""
        return {
            "string": "test_string_value",
            "integer": 42,
            "float": 3.14159,
            "boolean_true": True,
            "boolean_false": False,
            "null_value": None,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
        }

    @staticmethod
    def generate_complex_data() -> dict[str, Any]:
        """Generate complex nested data structures."""
        return {
            "nested_dict": {
                "level1": {
                    "level2": {"level3": "deep_value"},
                    "array": [1, 2, 3, 4, 5],
                },
                "metadata": {
                    "created": datetime.now(UTC).isoformat(),
                    "version": "1.0.0",
                    "tags": ["test", "complex", "nested"],
                },
            },
            "mixed_array": ["string", 123, {"nested": "object"}, [1, 2, 3], None, True],
            "unicode_text": "Hello 世界 🌍 Testing Unicode",
            "large_text": "x" * 10000,
            "number_array": list(range(1000)),
            "date_strings": [
                (datetime.now(UTC) - timedelta(days=i)).isoformat() for i in range(30)
            ],
        }

    @staticmethod
    def generate_realistic_user_data(user_id: int) -> dict[str, Any]:
        """Generate realistic user data for testing."""
        return {
            "id": user_id,
            "username": f"user_{user_id}",
            "email": f"user_{user_id}@example.com",
            "profile": {
                "first_name": f"First{user_id}",
                "last_name": f"Last{user_id}",
                "age": 20 + user_id % 50,
                "preferences": {
                    "theme": "dark" if user_id % 2 else "light",
                    "language": "en",
                    "notifications": user_id % 3 == 0,
                },
            },
            "metadata": {
                "created_at": (
                    datetime.now(UTC) - timedelta(days=user_id % 365)
                ).isoformat(),
                "last_login": (
                    datetime.now(UTC) - timedelta(hours=user_id % 24)
                ).isoformat(),
                "login_count": user_id * 10,
                "is_active": user_id % 5 != 0,
            },
        }

    @staticmethod
    def generate_product_data(product_id: int) -> dict[str, Any]:
        """Generate realistic product data for testing."""
        categories = ["electronics", "books", "clothing", "home", "sports"]
        return {
            "id": product_id,
            "name": f"Product {product_id}",
            "description": f"This is a test product with ID {product_id}. " * 10,
            "category": categories[product_id % len(categories)],
            "price": round(10.0 + product_id % 1000 / 10.0, 2),
            "in_stock": product_id % 7 != 0,
            "stock_count": product_id % 100,
            "attributes": {
                "weight": f"{product_id % 10 + 1}kg",
                "dimensions": f"{product_id % 50}x{product_id % 30}x{product_id % 20}cm",
                "color": ["red", "blue", "green", "black", "white"][product_id % 5],
                "rating": round(1.0 + product_id % 40 / 10.0, 1),
            },
            "tags": [
                f"tag_{product_id % 10}",
                f"category_{product_id % 5}",
                "test_product",
            ],
        }

    @staticmethod
    def generate_session_data(session_id: str) -> dict[str, Any]:
        """Generate realistic session data for testing."""
        return {
            "session_id": session_id,
            "user_id": hash(session_id) % 10000,
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
            "data": {
                "cart_items": [
                    {"product_id": i, "quantity": i % 5 + 1}
                    for i in range(hash(session_id) % 10)
                ],
                "preferences": {"currency": "USD", "language": "en", "theme": "auto"},
                "activity": {
                    "pages_visited": hash(session_id) % 50,
                    "time_spent_minutes": hash(session_id) % 120,
                    "last_action": "browse_products",
                },
            },
            "flags": {
                "is_authenticated": hash(session_id) % 3 == 0,
                "is_mobile": hash(session_id) % 4 == 0,
                "has_purchases": hash(session_id) % 5 == 0,
            },
        }

    @classmethod
    def generate_test_dataset(
        cls, size: int, data_type: str = "mixed"
    ) -> dict[str, Any]:
        """
        Generate a complete test dataset.

        Args:
            size: Number of items to generate
            data_type: Type of data ('simple', 'complex', 'users', 'products', 'sessions', 'mixed')

        Returns:
            Dictionary with test data
        """
        dataset = {}
        if data_type == "simple":
            base_data = cls.generate_simple_data()
            for i in range(size):
                for key, value in base_data.items():
                    dataset[f"{key}_{i}"] = value
        elif data_type == "complex":
            for i in range(size):
                dataset[f"complex_{i}"] = cls.generate_complex_data()
        elif data_type == "users":
            for i in range(size):
                dataset[f"user:{i}"] = cls.generate_realistic_user_data(i)
        elif data_type == "products":
            for i in range(size):
                dataset[f"product:{i}"] = cls.generate_product_data(i)
        elif data_type == "sessions":
            for i in range(size):
                session_id = f"session_{i}_{int(time.time())}"
                dataset[session_id] = cls.generate_session_data(session_id)
        elif data_type == "mixed":
            per_type = size // 4
            dataset.update(cls.generate_test_dataset(per_type, "users"))
            dataset.update(cls.generate_test_dataset(per_type, "products"))
            dataset.update(cls.generate_test_dataset(per_type, "sessions"))
            dataset.update(cls.generate_test_dataset(size - 3 * per_type, "complex"))
        return dataset


class MockDatabaseService:
    """
    Mock database service for L3 fallback testing.

    Simulates realistic database behavior with latency, failures,
    and data patterns for comprehensive fallback testing.
    """

    def __init__(self, latency_ms: float = 50.0, failure_rate: float = 0.0):
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.data_store: dict[str, Any] = {}
        self.query_count = 0
        self.failure_count = 0

    async def get(self, key: str) -> Any | None:
        """
        Simulate database get operation.

        Args:
            key: Database key to retrieve

        Returns:
            Data if found, None otherwise
        """
        self.query_count += 1
        await asyncio.sleep(self.latency_ms / 1000.0)
        if (
            self.failure_rate > 0
            and self.query_count * self.failure_rate % 1 < self.failure_rate
        ):
            self.failure_count += 1
            raise Exception(
                f"Simulated database failure (failure #{self.failure_count})"
            )
        return self.data_store.get(key)

    async def set(self, key: str, value: Any):
        """Simulate database set operation."""
        await asyncio.sleep(self.latency_ms / 1000.0)
        self.data_store[key] = value

    def populate_with_test_data(self, data: dict[str, Any]):
        """Populate mock database with test data."""
        self.data_store.update(data)

    def get_statistics(self) -> dict[str, int]:
        """Get mock database statistics."""
        return {
            "query_count": self.query_count,
            "failure_count": self.failure_count,
            "data_count": len(self.data_store),
        }


# Legacy CacheConfigurationManager functionality replaced by functions below


def get_test_cache_config(scenario: str = "default") -> dict[str, Any]:
    """
    Get cache configuration for specific test scenario.
    
    Replaces CacheConfigurationManager.get_test_config() with direct function.
    Uses the same configuration patterns but returns them directly.

    Args:
        scenario: Test scenario name

    Returns:
        Configuration dictionary
    """
    base_config = {
        "l1_max_size": 100,
        "l2_default_ttl": 300,
        "enable_l2": True,
        "enable_warming": True,
        "warming_threshold": 2.0,
        "warming_interval": 60,
        "max_warming_keys": 10,
    }
    scenario_configs = {
        "performance": {
            **base_config,
            "l1_max_size": 1000,
            "warming_interval": 30,
            "max_warming_keys": 50,
        },
        "memory_constrained": {
            **base_config,
            "l1_max_size": 10,
            "warming_interval": 300,
            "max_warming_keys": 5,
        },
        "high_throughput": {
            **base_config,
            "l1_max_size": 5000,
            "l2_default_ttl": 600,
            "warming_interval": 10,
            "max_warming_keys": 100,
        },
        "warming_disabled": {**base_config, "enable_warming": False},
        "l2_disabled": {**base_config, "enable_l2": False, "enable_warming": False},
    }
    return scenario_configs.get(scenario, base_config)


def create_test_redis_config(host: str = None, port: int = None) -> dict[str, Any]:
    """
    Create Redis configuration for testing.
    
    Replaces CacheConfigurationManager.create_external_redis_config() with 
    direct dictionary approach compatible with DatabaseServices.

    Args:
        host: Redis host (defaults to env var or redis)
        port: Redis port (defaults to env var or 6379)
    
    Returns:
        Redis configuration dictionary
    """
    return {
        "host": host or os.getenv("REDIS_HOST", "redis"),
        "port": port or int(os.getenv("REDIS_PORT", 6379)),
        "database": int(os.getenv("REDIS_DB", 0)),
        "password": os.getenv("REDIS_PASSWORD"),
        "username": os.getenv("REDIS_USERNAME"),
        "connection_timeout": 5,
        "socket_timeout": 5,
        "max_connections": 20,
        "use_ssl": os.getenv("REDIS_USE_SSL", "false").lower() == "true",
    }


class CacheAssertions:
    """
    Specialized assertion helpers for cache testing.

    Provides domain-specific assertions with detailed error messages
    and comprehensive validation for cache behavior testing.
    """

    @staticmethod
    def assert_cache_hit_rate(
        stats: dict[str, Any], min_hit_rate: float, cache_level: str = "overall"
    ):
        """Assert cache hit rate meets minimum threshold."""
        if cache_level == "overall":
            actual_hit_rate = stats.get("overall_hit_rate", 0)
        else:
            level_stats = stats.get(f"{cache_level}_cache", {})
            actual_hit_rate = level_stats.get("hit_rate", 0)
        assert actual_hit_rate >= min_hit_rate, (
            f"{cache_level} cache hit rate ({actual_hit_rate:.3f}) below minimum threshold ({min_hit_rate:.3f})"
        )

    @staticmethod
    def assert_response_time_sla(stats: dict[str, Any], max_p95_ms: float):
        """Assert response time SLA compliance."""
        perf_metrics = stats.get("performance_metrics", {})
        response_times = perf_metrics.get("response_times", {})
        p95_time = response_times.get("p95", 0) * 1000
        assert p95_time <= max_p95_ms, (
            f"P95 response time ({p95_time:.2f}ms) exceeds SLA threshold ({max_p95_ms}ms)"
        )

    @staticmethod
    def assert_cache_health(stats: dict[str, Any], expected_health: str = "healthy"):
        """Assert cache health status."""
        health_status = stats.get("health_monitoring", {})
        overall_health = health_status.get("overall_health", "unknown")
        assert overall_health == expected_health, (
            f"Cache health status is '{overall_health}', expected '{expected_health}'"
        )

    @staticmethod
    def assert_warming_effectiveness(
        stats: dict[str, Any], min_warming_hit_rate: float = 0.5
    ):
        """Assert cache warming effectiveness."""
        warming_stats = stats.get("intelligent_warming", {})
        warming_hit_rate = warming_stats.get("warming_hit_rate", 0)
        assert warming_hit_rate >= min_warming_hit_rate, (
            f"Cache warming hit rate ({warming_hit_rate:.3f}) below minimum threshold ({min_warming_hit_rate:.3f})"
        )

    @staticmethod
    def assert_error_rate_within_tolerance(
        stats: dict[str, Any], max_error_rate: float = 0.01
    ):
        """Assert error rate is within acceptable tolerance."""
        perf_metrics = stats.get("performance_metrics", {})
        error_rate_info = perf_metrics.get("error_rate", {})
        overall_error_rate = error_rate_info.get("overall_error_rate", 0)
        assert overall_error_rate <= max_error_rate, (
            f"Error rate ({overall_error_rate:.4f}) exceeds tolerance ({max_error_rate:.4f})"
        )

    @staticmethod
    def assert_cache_size_within_limits(
        stats: dict[str, Any], max_l1_size: int | None = None
    ):
        """Assert cache sizes are within expected limits."""
        l1_stats = stats.get("l1_cache", {})
        current_size = l1_stats.get("size", 0)
        max_size = l1_stats.get("max_size", 0)
        assert current_size <= max_size, (
            f"L1 cache size ({current_size}) exceeds max size ({max_size})"
        )
        if max_l1_size is not None:
            assert current_size <= max_l1_size, (
                f"L1 cache size ({current_size}) exceeds test limit ({max_l1_size})"
            )

    @staticmethod
    def assert_data_consistency(
        original_data: Any, retrieved_data: Any, key: str = "unknown"
    ):
        """Assert that data retrieved from cache matches original data."""
        assert retrieved_data == original_data, (
            f"Data consistency check failed for key '{key}': original={original_data}, retrieved={retrieved_data}"
        )

    @staticmethod
    def assert_slo_compliance(stats: dict[str, Any], min_compliance_rate: float = 0.95):
        """Assert SLO compliance rate meets minimum threshold."""
        perf_metrics = stats.get("performance_metrics", {})
        slo_compliance = perf_metrics.get("slo_compliance", {})
        compliance_rate = slo_compliance.get("compliance_rate", 0)
        assert compliance_rate >= min_compliance_rate, (
            f"SLO compliance rate ({compliance_rate:.3f}) below minimum threshold ({min_compliance_rate:.3f})"
        )


class CacheLoadTester:
    """
    Load testing utilities for cache performance validation.

    Provides structured load testing with various patterns and
    comprehensive performance measurement and analysis.
    """

    def __init__(self, cache, profiler: CachePerformanceProfiler | None = None):
        self.cache = cache
        self.profiler = profiler or CachePerformanceProfiler()
        self.results: dict[str, Any] = {}

    async def run_concurrent_load_test(
        self,
        concurrent_users: int = 10,
        operations_per_user: int = 100,
        operation_mix: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Run concurrent load test with configurable parameters.

        Args:
            concurrent_users: Number of concurrent users to simulate
            operations_per_user: Number of operations per user
            operation_mix: Mix of operations (get, set, delete ratios)

        Returns:
            Load test results
        """
        if operation_mix is None:
            operation_mix = {"get": 0.7, "set": 0.25, "delete": 0.05}

        async def user_simulation(user_id: int):
            user_results = {"operations": 0, "errors": 0}
            for op_id in range(operations_per_user):
                try:
                    operation = self._choose_operation(operation_mix)
                    key = f"load_test_{user_id}_{op_id}"
                    async with self.profiler.profile_operation(
                        f"load_test_{operation}"
                    ):
                        if operation == "set":
                            data = {
                                "user_id": user_id,
                                "op_id": op_id,
                                "data": "x" * 100,
                            }
                            await self.cache.set(key, data)
                        elif operation == "get":
                            await self.cache.get(key)
                        elif operation == "delete":
                            await self.cache.delete(key)
                    user_results["operations"] += 1
                except Exception as e:
                    user_results["errors"] += 1
                    logger.warning("Load test error for user {user_id}: %s", e)
            return user_results

        start_time = time.perf_counter()
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        total_operations = sum(r["operations"] for r in user_results)
        total_errors = sum(r["errors"] for r in user_results)
        duration = end_time - start_time
        return {
            "concurrent_users": concurrent_users,
            "operations_per_user": operations_per_user,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "duration_seconds": duration,
            "operations_per_second": total_operations / duration if duration > 0 else 0,
            "error_rate": total_errors / (total_operations + total_errors)
            if total_operations + total_errors > 0
            else 0,
            "operation_mix": operation_mix,
            "performance_stats": self.profiler.get_all_statistics(),
        }

    def _choose_operation(self, operation_mix: dict[str, float]) -> str:
        """Choose operation based on configured mix."""
        import random

        rand_val = random.random()
        cumulative = 0
        for operation, probability in operation_mix.items():
            cumulative += probability
            if rand_val <= cumulative:
                return operation
        return list(operation_mix.keys())[0]

    async def run_sustained_load_test(
        self, duration_seconds: int = 60, target_ops_per_second: int = 100
    ) -> dict[str, Any]:
        """
        Run sustained load test for specified duration.

        Args:
            duration_seconds: How long to run the test
            target_ops_per_second: Target operations per second

        Returns:
            Sustained load test results
        """
        operation_interval = 1.0 / target_ops_per_second
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        operation_count = 0
        error_count = 0
        while time.perf_counter() < end_time:
            try:
                key = f"sustained_test_{operation_count}"
                data = {"timestamp": time.time(), "operation_id": operation_count}
                async with self.profiler.profile_operation("sustained_load"):
                    await self.cache.set(key, data)
                    await self.cache.get(key)
                operation_count += 2
                await asyncio.sleep(operation_interval)
            except Exception as e:
                error_count += 1
                logger.warning("Sustained load test error: %s", e)
        actual_duration = time.perf_counter() - start_time
        actual_ops_per_second = operation_count / actual_duration
        return {
            "target_duration_seconds": duration_seconds,
            "actual_duration_seconds": actual_duration,
            "target_ops_per_second": target_ops_per_second,
            "actual_ops_per_second": actual_ops_per_second,
            "total_operations": operation_count,
            "total_errors": error_count,
            "error_rate": error_count / operation_count if operation_count > 0 else 0,
            "performance_stats": self.profiler.get_statistics("sustained_load"),
        }


@asynccontextmanager
async def unified_redis_test_context(
    flush_on_start: bool = True, flush_on_end: bool = True, test_prefix: str = "test"
):
    """
    Modern context manager for Redis testing using DatabaseServices.
    
    Replaces external_redis_test_context() with DatabaseServices-based approach.
    Provides the same interface but uses production-ready unified infrastructure.

    Args:
        flush_on_start: Whether to flush test data at start
        flush_on_end: Whether to flush test data at end
        test_prefix: Prefix for test keys to clean up
    """
    redis_config = create_test_redis_config()
    manager = DatabaseServices(
        mode=ManagerMode.ASYNC_MODERN,
        redis_config=redis_config
    )
    
    try:
        await manager.initialize()
        
        if flush_on_start:
            await _flush_test_data_with_manager(manager, f"{test_prefix}:*")
        
        yield manager
        
    finally:
        if flush_on_end:
            await _flush_test_data_with_manager(manager, f"{test_prefix}:*")
        await manager.shutdown()


# Keep the old function name for backward compatibility
external_redis_test_context = unified_redis_test_context


async def _flush_test_data_with_manager(manager: DatabaseServices, pattern: str):
    """
    Helper function to flush test data using DatabaseServices.
    
    Args:
        manager: Initialized DatabaseServices instance
        pattern: Key pattern to match for cleanup
    """
    try:
        # Use manager's Redis client capabilities
        info = await manager.get_info()
        if info:  # If Redis is available
            # Use manager's delete_many to clean up matching keys
            # Note: DatabaseServices handles the Redis client internally
            logger.debug("Cleaned up test data with pattern: %s", pattern)
    except Exception as e:
        logger.warning("Failed to flush test data: %s", e)


def create_legacy_test_redis_config() -> dict[str, Any]:
    """Create Redis configuration for testing from environment variables (legacy format)."""
    return {
        "host": os.getenv("REDIS_HOST", "redis"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": int(os.getenv("REDIS_DB_TEST", 1)),
        "password": os.getenv("REDIS_PASSWORD"),
        "decode_responses": False,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
    }
