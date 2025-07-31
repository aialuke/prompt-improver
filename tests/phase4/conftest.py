"""
Phase 4 Testing Configuration - 2025 Best Practices

Implements testcontainers for database testing following 2025 industry standards:
- Real PostgreSQL containers for integration testing
- Async SQLAlchemy 2.0 session management
- Proper test isolation and cleanup
- Performance optimized test database setup

2025 Solution: Uses pytest.ini configuration with asyncio_mode=auto
and asyncio_default_fixture_loop_scope=session to handle scope properly.
"""

import asyncio
import json
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from testcontainers.postgres import PostgresContainer
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

from prompt_improver.database.models import SQLModel, RuleIntelligenceCache


@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL container for testing session.

    2025 Best Practice: Use testcontainers for real database testing
    instead of mocking for better integration test coverage.
    """
    with PostgresContainer(
        image="postgres:16-alpine",  # Latest stable PostgreSQL
        username="test_user",
        password="test_password",
        dbname="test_db",
        port=5432
    ) as postgres:
        yield postgres


@pytest.fixture(scope="session")
def database_url(postgres_container):
    """Get async database URL from container."""
    # Get connection details from container
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    username = postgres_container.username
    password = postgres_container.password
    database = postgres_container.dbname

    # Create async URL for SQLAlchemy 2.0 with psycopg3 driver (2025 best practice)
    async_url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    return async_url


@pytest_asyncio.fixture(scope="session")
async def async_engine(database_url):
    """Create async SQLAlchemy engine for testing.

    2025 Best Practice: With pytest.ini configuration, async fixtures
    work properly at session scope without scope mismatch errors.
    """
    engine = create_async_engine(
        database_url,
        echo=False,  # Set to True for SQL debugging
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def async_session_factory(async_engine):
    """Create async session factory for testing."""
    return async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


@pytest_asyncio.fixture
async def db_session(async_session_factory) -> AsyncGenerator[AsyncSession, None]:
    """Provide clean database session for each test.

    2025 Best Practice: Each test gets a fresh transaction that's
    rolled back after the test for perfect isolation.
    """
    async with async_session_factory() as session:
        # Start a transaction
        transaction = await session.begin()

        try:
            yield session
        finally:
            # Always rollback to ensure test isolation
            await transaction.rollback()
            await session.close()


@pytest_asyncio.fixture
async def sample_rule_performance_data(db_session: AsyncSession):
    """Create sample rule performance data for testing."""
    # Insert sample data for testing
    sample_data = [
        {
            "rule_id": "test_rule_1",
            "rule_name": "Test Rule 1",
            "improvement_score": 0.8,
            "effectiveness_rating": 4,
            "prompt_characteristics": {
                "prompt_type": "instructional",
                "complexity_level": 0.7,
                "domain": "technical",
                "length_category": "medium",
                "reasoning_required": True,
                "specificity_level": 0.8,
                "context_richness": 0.6,
                "task_type": "problem_solving",
                "language_style": "formal",
                "custom_attributes": {}
            },
            "prompt_type": "instructional"
        },
        {
            "rule_id": "test_rule_2",
            "rule_name": "Test Rule 2",
            "improvement_score": 0.75,
            "effectiveness_rating": 4,
            "prompt_characteristics": {
                "prompt_type": "analytical",
                "complexity_level": 0.6,
                "domain": "business",
                "length_category": "short",
                "reasoning_required": False,
                "specificity_level": 0.7,
                "context_richness": 0.5,
                "task_type": "analysis",
                "language_style": "professional",
                "custom_attributes": {}
            },
            "prompt_type": "analytical"
        }
    ]

    # Insert sample rule performance data
    for data in sample_data:
        insert_query = text("""
            INSERT INTO rule_performance (
                rule_id, rule_name, improvement_score, effectiveness_rating,
                prompt_characteristics, prompt_type, created_at
            ) VALUES (
                :rule_id, :rule_name, :improvement_score, :effectiveness_rating,
                :prompt_characteristics, :prompt_type, NOW()
            )
        """)

        await db_session.execute(insert_query, {
            "rule_id": data["rule_id"],
            "rule_name": data["rule_name"],
            "improvement_score": data["improvement_score"],
            "effectiveness_rating": data["effectiveness_rating"],
            "prompt_characteristics": json.dumps(data["prompt_characteristics"]),
            "prompt_type": data["prompt_type"]
        })

    await db_session.commit()
    return sample_data


@pytest_asyncio.fixture
async def sample_rule_intelligence_cache(db_session: AsyncSession):
    """Create sample rule intelligence cache data for testing."""
    # Use SQLModel ORM to ensure all defaults are properly handled
    cache_entry = RuleIntelligenceCache(
        cache_key="test_cache_key_1",
        rule_id="test_rule_1",
        rule_name="Test Rule 1",
        prompt_characteristics_hash="hash_1",
        effectiveness_score=0.8,
        characteristic_match_score=0.75,
        historical_performance_score=0.85,
        ml_prediction_score=0.7,
        recency_score=0.9,
        total_score=0.8,
        confidence_level=0.85,
        sample_size=25,
        pattern_insights={"key": "value"},
        optimization_recommendations=["rec1", "rec2"],
        performance_trend="improving"
    )

    db_session.add(cache_entry)
    await db_session.commit()
    await db_session.refresh(cache_entry)

    return [cache_entry]


# 2025 Performance Testing Configuration
@pytest.fixture(scope="session")
def performance_test_config():
    """Configuration for performance testing following 2025 standards."""
    return {
        "max_response_time_ms": 200,  # <200ms SLA requirement
        "batch_size_limits": {
            "small": 50,
            "medium": 100,
            "large": 500
        },
        "parallel_worker_limits": {
            "min": 2,
            "max": 8,
            "default": 4
        },
        "confidence_thresholds": {
            "minimum": 0.6,
            "good": 0.8,
            "excellent": 0.9
        }
    }
