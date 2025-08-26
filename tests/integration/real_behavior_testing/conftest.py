"""Real behavior testing configuration and shared fixtures.

Provides comprehensive fixture infrastructure for real behavior testing with actual services.
All fixtures use real containers and services - no mocks or stubs.
"""

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest

# Performance optimization - disable telemetry in tests
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.integration.real_behavior_testing.containers.ml_test_container import MLTestContainer
from tests.integration.real_behavior_testing.containers.network_simulator import NetworkSimulator
from tests.integration.real_behavior_testing.containers.real_redis_container import RealRedisTestContainer
from tests.integration.real_behavior_testing.performance.benchmark_suite import BenchmarkSuite

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session-scoped async fixtures."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def postgres_container() -> AsyncGenerator[PostgreSQLTestContainer, None]:
    """Session-scoped PostgreSQL testcontainer with optimized configuration."""
    container = PostgreSQLTestContainer(
        postgres_version="16",
        database_name=f"test_db_{uuid.uuid4().hex[:8]}",
    )

    async with container:
        logger.info(f"PostgreSQL container started: {container.database_name}")
        yield container


@pytest.fixture(scope="session")
async def redis_container() -> AsyncGenerator[RealRedisTestContainer, None]:
    """Session-scoped Redis testcontainer for cache testing."""
    container = RealRedisTestContainer(
        redis_version="7.2",
        port=None,  # Auto-assign port
    )

    async with container:
        logger.info(f"Redis container started on port {container.get_port()}")
        yield container


@pytest.fixture(scope="session")
async def ml_container() -> AsyncGenerator[MLTestContainer, None]:
    """Session-scoped ML testing environment container."""
    container = MLTestContainer(
        models_path="/tmp/test_models",
        enable_gpu=False,  # CPU-only for CI/CD compatibility
    )

    async with container:
        logger.info("ML container started with test models")
        yield container


@pytest.fixture(scope="function")
async def network_simulator() -> AsyncGenerator[NetworkSimulator, None]:
    """Function-scoped network failure simulator for retry testing."""
    simulator = NetworkSimulator()
    await simulator.start()

    try:
        yield simulator
    finally:
        await simulator.stop()


@pytest.fixture(scope="function")
async def benchmark_suite(
    postgres_container: PostgreSQLTestContainer,
    redis_container: RealRedisTestContainer,
) -> BenchmarkSuite:
    """Performance benchmarking suite with real services."""
    return BenchmarkSuite(
        postgres_container=postgres_container,
        redis_container=redis_container,
    )


@pytest.fixture(scope="function")
async def clean_database(postgres_container: PostgreSQLTestContainer):
    """Ensure clean database state for each test."""
    await postgres_container.truncate_all_tables()
    yield
    await postgres_container.truncate_all_tables()


@pytest.fixture(scope="function")
async def clean_redis(redis_container: RealRedisTestContainer):
    """Ensure clean Redis state for each test."""
    await redis_container.flush_all()
    yield
    await redis_container.flush_all()


@pytest.fixture(scope="function")
def performance_tracker():
    """Track performance metrics during test execution."""
    metrics = {}

    def track(operation: str, duration_ms: float, target_ms: float | None = None):
        """Track operation performance."""
        metrics[operation] = {
            "duration_ms": duration_ms,
            "target_ms": target_ms,
            "passed": duration_ms < target_ms if target_ms else True,
        }

        if target_ms and duration_ms >= target_ms:
            logger.warning(
                f"Performance target missed: {operation} took {duration_ms:.2f}ms "
                f"(target: {target_ms}ms)"
            )

    return track


@pytest.fixture(scope="function")
async def error_injector():
    """Error injection utility for testing error handling services."""
    from tests.integration.real_behavior_testing.utils.error_injection import ErrorInjector

    injector = ErrorInjector()
    yield injector
    await injector.cleanup()


@pytest.fixture(scope="function")
def test_data_factory():
    """Factory for generating realistic test data scenarios."""
    from tests.integration.real_behavior_testing.utils.test_data_factory import TestDataFactory

    return TestDataFactory()


# Integration fixtures for specific service combinations

@pytest.fixture(scope="function")
async def integrated_services(
    postgres_container: PostgreSQLTestContainer,
    redis_container: RealRedisTestContainer,
    clean_database,
    clean_redis,
) -> dict[str, Any]:
    """Integrated services setup for comprehensive testing."""
    return {
        "postgres": postgres_container,
        "redis": redis_container,
        "database_url": postgres_container.get_connection_url(),
        "redis_url": redis_container.get_connection_url(),
    }


@pytest.fixture(scope="function")
async def ml_intelligence_setup(
    integrated_services: dict[str, Any],
    ml_container: MLTestContainer,
    test_data_factory,
) -> dict[str, Any]:
    """Setup for ML Intelligence Services testing."""
    # Create test data
    test_data = await test_data_factory.create_ml_test_dataset(
        size="small",  # 100 samples for fast tests
        domains=["general", "technical", "creative"],
    )

    return {
        **integrated_services,
        "ml_container": ml_container,
        "test_data": test_data,
    }


@pytest.fixture(scope="function")
async def retry_testing_setup(
    network_simulator: NetworkSimulator,
    performance_tracker,
) -> dict[str, Any]:
    """Setup for retry management services testing."""
    return {
        "network_simulator": network_simulator,
        "performance_tracker": performance_tracker,
        "failure_scenarios": [
            "connection_timeout",
            "connection_refused",
            "dns_resolution_failure",
            "intermittent_network_errors",
        ],
    }


@pytest.fixture(scope="function")
async def error_handling_setup(
    error_injector,
    performance_tracker,
    integrated_services: dict[str, Any],
) -> dict[str, Any]:
    """Setup for error handling services testing."""
    return {
        **integrated_services,
        "error_injector": error_injector,
        "performance_tracker": performance_tracker,
        "error_scenarios": [
            "database_connection_failure",
            "redis_connection_failure",
            "validation_errors",
            "business_logic_errors",
            "system_resource_errors",
        ],
    }
