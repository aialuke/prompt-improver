"""
Unit test configuration and fixtures.

Provides completely isolated unit testing environment with comprehensive mocking.
Tests in this category should run in <100ms and have no external dependencies.
"""

import asyncio
import logging
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Disable all external services for unit tests
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["REDIS_URL"] = "redis://mock-redis:6379"
os.environ["DATABASE_URL"] = "postgresql://mock:mock@mock:5432/mock"


# Mock external dependencies at module level
@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Automatically mock all external dependencies for unit tests."""
    with patch('asyncpg.connect', new_callable=AsyncMock) as mock_db, \
         patch('coredis.Redis', new_callable=MagicMock) as mock_redis, \
         patch('opentelemetry.trace.get_tracer', return_value=MagicMock()) as mock_tracer, \
         patch('opentelemetry.metrics.get_meter', return_value=MagicMock()) as mock_meter:
        yield {
            'db': mock_db,
            'redis': mock_redis,
            'tracer': mock_tracer,
            'meter': mock_meter
        }


@pytest.fixture
def mock_database_session():
    """Mock database session for unit tests."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.query = MagicMock()
    return session


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for unit tests."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=False)
    client.expire = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_datetime_service():
    """Mock datetime service for consistent testing."""
    service = MagicMock()
    fixed_time = datetime(2025, 1, 15, 12, 0, 0)
    service.now.return_value = fixed_time
    service.utc_now.return_value = fixed_time
    return service


@pytest.fixture
def sample_prompt():
    """Sample prompt data for testing."""
    return "Fix this issue with the code"


@pytest.fixture
def sample_improvement_result():
    """Sample improvement result for testing."""
    return {
        "improved_prompt": "Please fix this specific issue with the code by analyzing the error and implementing a solution",
        "confidence": 0.85,
        "rules_applied": ["clarity", "specificity"],
        "processing_time_ms": 45,
        "metadata": {
            "session_id": str(uuid4()),
            "timestamp": datetime.now().isoformat()
        }
    }


@pytest.fixture
def mock_ml_service():
    """Mock ML service for unit tests."""
    service = AsyncMock()
    service.predict = AsyncMock(return_value=0.85)
    service.train = AsyncMock(return_value={"status": "success"})
    service.evaluate = AsyncMock(return_value={"accuracy": 0.92})
    return service


@pytest.fixture
def mock_rule_engine():
    """Mock rule engine for unit tests."""
    engine = MagicMock()
    engine.apply_rules = MagicMock(return_value={
        "improved_prompt": "Improved prompt text",
        "confidence": 0.85,
        "rules_applied": ["clarity", "specificity"]
    })
    return engine


# Performance timing for unit tests
@pytest.fixture(autouse=True)
def unit_test_performance_check(request):
    """Ensure unit tests run under 100ms."""
    start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop() else 0
    yield
    if asyncio.get_event_loop():
        end_time = asyncio.get_event_loop().time()
        duration = (end_time - start_time) * 1000
        if duration > 100:
            pytest.fail(f"Unit test {request.node.name} took {duration:.2f}ms (should be <100ms)")


# Logging configuration for unit tests
logging.getLogger().setLevel(logging.CRITICAL)  # Suppress all logging in unit tests
