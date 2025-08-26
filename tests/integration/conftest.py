"""
Integration test configuration and fixtures.

Provides test container infrastructure for service boundary testing.
Tests use real databases and services but mock external APIs.
"""

import logging
import os
import time
from uuid import uuid4

import coredis
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer


# Test container management
@pytest.fixture(scope="session", autouse=True)
def setup_test_containers():
    """Set up test containers for integration tests."""
    # Start PostgreSQL container
    postgres = PostgresContainer("postgres:15-alpine")
    postgres.start()

    # Start Redis container
    redis = RedisContainer("redis:7-alpine")
    redis.start()

    # Set environment variables
    os.environ["TEST_DATABASE_URL"] = postgres.get_connection_url().replace("psycopg2", "asyncpg")
    os.environ["TEST_REDIS_URL"] = f"redis://localhost:{redis.get_exposed_port(6379)}"

    yield {
        "postgres": postgres,
        "redis": redis
    }

    # Cleanup
    postgres.stop()
    redis.stop()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    database_url = os.environ["TEST_DATABASE_URL"]
    engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=5,
        max_overflow=0,
        pool_pre_ping=True
    )

    yield engine

    await engine.dispose()


@pytest.fixture
async def test_db_session(test_engine):
    """Create test database session with automatic rollback."""
    async_session = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session, session.begin():
        yield session
            # Automatic rollback happens when context exits


@pytest.fixture
async def test_redis_client():
    """Create test Redis client."""
    redis_url = os.environ["TEST_REDIS_URL"]
    client = coredis.from_url(redis_url)

    yield client

    # Cleanup test keys
    await client.flushdb()
    await client.connection_pool.disconnect()


@pytest.fixture
def integration_test_data():
    """Generate test data for integration tests."""
    return {
        "session_id": str(uuid4()),
        "user_id": "test_user",
        "prompts": [
            "Fix this bug",
            "Make this code better",
            "Explain how this works"
        ],
        "context": {
            "domain": "software_development",
            "language": "python",
            "complexity": "medium"
        }
    }


@pytest.fixture
async def setup_test_database_schema(test_db_session):
    """Set up test database schema."""
    # Create necessary tables for testing
    schema_sql = """
    CREATE TABLE IF NOT EXISTS prompt_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id VARCHAR(255) UNIQUE NOT NULL,
        user_id VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        context JSONB
    );

    CREATE TABLE IF NOT EXISTS prompt_improvements (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id VARCHAR(255) NOT NULL,
        original_prompt TEXT NOT NULL,
        improved_prompt TEXT NOT NULL,
        confidence FLOAT,
        rules_applied JSONB,
        processing_time_ms INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS ml_experiments (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        experiment_name VARCHAR(255) NOT NULL,
        model_type VARCHAR(100),
        parameters JSONB,
        metrics JSONB,
        status VARCHAR(50) DEFAULT 'running',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    from sqlalchemy import text
    await test_db_session.execute(text(schema_sql))
    await test_db_session.commit()


# Performance monitoring for integration tests
@pytest.fixture(autouse=True)
def integration_test_performance_check(request):
    """Ensure integration tests run under 1 second."""
    start_time = time.time()
    yield
    duration = (time.time() - start_time) * 1000
    if duration > 1000:
        logging.warning(f"Integration test {request.node.name} took {duration:.2f}ms (should be <1000ms)")


# Service mocking for external dependencies
@pytest.fixture
def mock_external_services():
    """Mock external services for integration tests."""
    from unittest.mock import MagicMock, patch

    with patch('requests.post') as mock_http_post, \
         patch('requests.get') as mock_http_get, \
         patch('openai.OpenAI') as mock_openai:

        # Mock HTTP responses
        mock_http_post.return_value.status_code = 200
        mock_http_post.return_value.json.return_value = {"status": "success"}
        mock_http_get.return_value.status_code = 200
        mock_http_get.return_value.json.return_value = {"data": "test"}

        # Mock OpenAI API
        mock_openai.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Improved prompt"))]
        )

        yield {
            "http_post": mock_http_post,
            "http_get": mock_http_get,
            "openai": mock_openai
        }


# Real service factories for integration testing
@pytest.fixture
async def prompt_improvement_service(test_db_session, test_redis_client):
    """Create real prompt improvement service for integration testing."""
    from src.prompt_improver.services.prompt.facade import (
        PromptServiceFacade as PromptImprovementService,
    )

    return PromptImprovementService(
        db_session=test_db_session,
        redis_client=test_redis_client
    )


@pytest.fixture
async def ml_service(test_db_session):
    """Create real ML service for integration testing."""
    from src.prompt_improver.core.services.ml_service import MLService

    return MLService(db_session=test_db_session)


# Logging configuration for integration tests
logging.basicConfig(level=logging.INFO)
logging.getLogger("testcontainers").setLevel(logging.WARNING)
