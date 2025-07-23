"""
Centralized pytest configuration and shared fixtures for APES system testing.
Provides comprehensive fixture infrastructure following pytest-asyncio best practices.

Features:
- Deterministic RNG seeding for reproducible test results
- Optimized fixture data generation for faster test execution
- Comprehensive database session management
- Performance monitoring and profiling capabilities
"""

import asyncio
import os
import random
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from typer.testing import CliRunner
from testcontainers.redis import RedisContainer
import redis.asyncio as aioredis

from prompt_improver.database.models import (
    ABExperiment,
    ImprovementSession,
    RuleMetadata,
    RulePerformance,
    UserFeedback,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

# ML dependency checks for graceful test skipping
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import deap
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

try:
    import pymc
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

try:
    import umap
    HAS_UMAP = True  
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# Create skip markers for optional dependencies
requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
requires_deap = pytest.mark.skipif(not HAS_DEAP, reason="DEAP not installed") 
requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
requires_umap = pytest.mark.skipif(not HAS_UMAP, reason="UMAP not installed")
requires_hdbscan = pytest.mark.skipif(not HAS_HDBSCAN, reason="HDBSCAN not installed")


# ===============================
# DETERMINISTIC TEST SETUP
# ===============================

# Set deterministic seed for reproducible test results
TEST_RANDOM_SEED = 42

@pytest.fixture(scope="session", autouse=True)
def seed_random_generators():
    """Seed all random number generators for reproducible test results.
    
    This fixture runs automatically for every test session to ensure
    deterministic behavior across all tests.
    """
    # Seed Python's random module
    random.seed(TEST_RANDOM_SEED)
    
    # Seed NumPy's random generator
    np.random.seed(TEST_RANDOM_SEED)
    
    # Set environment variable for subprocess determinism
    os.environ["PYTHONHASHSEED"] = str(TEST_RANDOM_SEED)
    
    # Additional seeding for ML libraries if available
    try:
        import sklearn
        # sklearn uses numpy's random state, so np.random.seed() is sufficient
        pass
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(TEST_RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TEST_RANDOM_SEED)
    except ImportError:
        pass


@pytest.fixture(scope="function")
def deterministic_rng():
    """Provide a deterministic random number generator for individual tests.
    
    This fixture ensures that each test gets a fresh, seeded RNG state
    while maintaining reproducibility.
    """
    # Create a new random state for this test
    rng = np.random.RandomState(TEST_RANDOM_SEED)
    return rng


# ===============================
# CLI TESTING INFRASTRUCTURE
# ===============================
@pytest.fixture(scope="session")
def cli_runner():
    """Session-scoped CLI runner for testing commands.

    Session scope prevents recreation overhead while maintaining isolation
    through CliRunner's built-in isolation mechanisms.
    """
    return CliRunner()


@pytest.fixture(scope="function")
def isolated_cli_runner():
    """Function-scoped CLI runner for tests requiring complete isolation."""
    return CliRunner()


# Database Testing Infrastructure
@pytest.fixture(scope="function")
async def real_db_session(test_db_engine):
    """Create a real database session for testing."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy.orm import sessionmaker
    from prompt_improver.database.models import SQLModel

    async_session = async_sessionmaker(
        bind=test_db_engine, expire_on_commit=False, class_=AsyncSession
    )

    # Tables are already created by test_db_engine fixture
    async with async_session() as session:
        yield session
        await session.rollback()  # Ensure rollback after each test to maintain isolation


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create test database engine using existing PostgreSQL configuration with retry logic."""
    from prompt_improver.database.config import DatabaseConfig
    from tests.database_helpers import (
        create_test_engine_with_retry,
        cleanup_test_database,
        wait_for_postgres_async,
    )
    import uuid

    config = DatabaseConfig()
    postgres_ready = await wait_for_postgres_async(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        database=config.postgres_database,  # Use main database (apes_production) instead of 'postgres'
        max_retries=30,
        retry_delay=1.0,
    )

    if not postgres_ready:
        pytest.skip("PostgreSQL server not available after 30 attempts")

    # Create a unique test database for each test to prevent conflicts
    test_db_name = f"apes_test_{uuid.uuid4().hex[:8]}"
    
    # Clean up test database for completely fresh state
    db_cleaned = await cleanup_test_database(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        test_db_name=test_db_name,
    )

    if not db_cleaned:
        pytest.skip("Could not clean up test database")

    test_db_url = f"postgresql+psycopg://{config.postgres_username}:{config.postgres_password}@{config.postgres_host}:{config.postgres_port}/{test_db_name}"

    engine = await create_test_engine_with_retry(
        test_db_url,
        max_retries=3,
        connect_args={
            "application_name": "apes_test_suite",
            "connect_timeout": 10,
        },
        pool_timeout=10,
    )

    if engine is None:
        pytest.skip("Could not create database engine")

    try:
        yield engine
    finally:
        try:
            await engine.dispose()
        except Exception:
            pass
        
        # Clean up the unique test database
        try:
            import asyncpg
            
            conn = await asyncpg.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_username,
                password=config.postgres_password,
                database=config.postgres_database,  # Use main database (apes_production) instead of 'postgres'
                timeout=5.0,
            )
            
            # Terminate connections to test database
            await conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
                test_db_name
            )
            
            # Drop the unique test database
            await conn.execute(f'DROP DATABASE IF EXISTS "{test_db_name}"')
            await conn.close()
            
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
async def test_db_session(test_db_engine):
    """Create test database session with transaction rollback for isolation."""
    async_session = async_sessionmaker(
        bind=test_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        # Start a transaction
        trans = await session.begin()
        try:
            yield session
        finally:
            # Always rollback to ensure test isolation
            try:
                if trans.is_active:
                    await trans.rollback()
            except Exception:
                # Transaction may already be closed, ignore
                pass


@pytest.fixture
async def mock_db_session(test_db_session):
    """Alias for backward compatibility with tests expecting mock_db_session."""
    return test_db_session


@pytest.fixture
async def populate_ab_experiment(real_db_session):
    """
    Populate database with ABExperiment and related RulePerformance records for testing.
    Creates comprehensive experiment configurations with sufficient data for statistical validation.
    """
    import uuid
    import numpy as np
    from datetime import datetime, timedelta
    from prompt_improver.database.models import PromptSession

    # Create rule metadata records first to satisfy foreign key constraints
    rule_metadata_records = [
        RuleMetadata(
            rule_id="clarity_rule",
            rule_name="Clarity Enhancement Rule",
            category="core",
            description="Improves prompt clarity",
            enabled=True,
            priority=5,
            default_parameters={"weight": 1.0, "threshold": 0.7},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
        ),
        RuleMetadata(
            rule_id="specificity_rule",
            rule_name="Specificity Enhancement Rule",
            category="core",
            description="Improves prompt specificity",
            enabled=True,
            priority=4,
            default_parameters={"weight": 0.8, "threshold": 0.6},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
        ),
        RuleMetadata(
            rule_id="chain_of_thought_rule",
            rule_name="Chain of Thought Rule",
            category="advanced",
            description="Adds chain of thought reasoning",
            enabled=True,
            priority=3,
            default_parameters={"weight": 0.9, "threshold": 0.5},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
        ),
        RuleMetadata(
            rule_id="example_rule",
            rule_name="Example Enhancement Rule",
            category="enhancement",
            description="Adds examples to prompts",
            enabled=True,
            priority=2,
            default_parameters={"weight": 0.7, "threshold": 0.6},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
        ),
    ]

    # Create multiple experiments with different configurations
    experiments = [
        ABExperiment(
            experiment_id=str(uuid.uuid4()),
            experiment_name="Clarity Enhancement A/B Test",
            description="Testing impact of clarity rule improvements",
            control_rules={"rule_ids": ["clarity_rule"]},
            treatment_rules={"rule_ids": ["clarity_rule", "specificity_rule"]},
            target_metric="improvement_score",
            sample_size_per_group=150,
            current_sample_size=300,
            significance_threshold=0.05,
            status="running",
            started_at=aware_utc_now() - timedelta(days=5),
            experiment_metadata={
                "control_description": "Basic clarity rule only",
                "treatment_description": "Clarity + specificity rules",
                "expected_effect_size": 0.1
            }
        ),
        ABExperiment(
            experiment_id=str(uuid.uuid4()),
            experiment_name="Multi-Rule Performance Test",
            description="Testing comprehensive rule set performance",
            control_rules={"rule_ids": ["clarity_rule", "specificity_rule"]},
            treatment_rules={"rule_ids": ["clarity_rule", "specificity_rule", "chain_of_thought_rule"]},
            target_metric="improvement_score",
            sample_size_per_group=200,
            current_sample_size=400,
            significance_threshold=0.01,
            status="running",
            started_at=aware_utc_now() - timedelta(days=10),
            experiment_metadata={
                "control_description": "Standard two-rule set",
                "treatment_description": "Enhanced three-rule set",
                "expected_effect_size": 0.15
            }
        ),
        ABExperiment(
            experiment_id=str(uuid.uuid4()),
            experiment_name="Completed Experiment",
            description="Already completed experiment for historical analysis",
            control_rules={"rule_ids": ["clarity_rule"]},
            treatment_rules={"rule_ids": ["clarity_rule", "example_rule"]},
            target_metric="improvement_score",
            sample_size_per_group=100,
            current_sample_size=200,
            significance_threshold=0.05,
            status="completed",
            started_at=aware_utc_now() - timedelta(days=30),
            completed_at=aware_utc_now() - timedelta(days=15),
            results={
                "control_mean": 0.72,
                "treatment_mean": 0.78,
                "p_value": 0.023,
                "effect_size": 0.06,
                "statistical_significance": True
            }
        )
    ]

    # Calculate total sessions needed for all experiments
    total_sessions_needed = sum(exp.current_sample_size for exp in experiments)
    
    # Create corresponding PromptSession records for RulePerformance foreign keys
    prompt_sessions = []
    for i in range(total_sessions_needed):  # Sufficient sessions for all experiments
        session = PromptSession(
            session_id=f"exp_session_{i}",
            original_prompt=f"Test prompt {i}",
            improved_prompt=f"Enhanced test prompt {i} with better clarity and specificity",
            user_context={"experiment_type": "ab_test", "session_number": i},
            quality_score=np.random.uniform(0.6, 0.9),
            improvement_score=np.random.uniform(0.5, 0.9),
            confidence_level=np.random.uniform(0.7, 0.95),
            created_at=aware_utc_now() - timedelta(hours=i // 10),
            updated_at=aware_utc_now() - timedelta(hours=i // 10)
        )
        prompt_sessions.append(session)

    # Create RulePerformance records for statistical validation
    rule_performance_records = []
    session_idx = 0
    
    # For each experiment, create realistic performance data
    for exp_idx, experiment in enumerate(experiments):
        experiment_sessions = experiment.current_sample_size
        half_sessions = experiment_sessions // 2
        
        # Control group performance (based on control_rules)
        control_base_score = 0.70 + (exp_idx * 0.02)  # Vary by experiment
        for i in range(half_sessions):
            score = np.random.normal(control_base_score, 0.08)
            score = max(0.0, min(1.0, score))  # Clamp to valid range
            
            record = RulePerformance(
                session_id=f"exp_session_{session_idx}",
                rule_id="clarity_rule",  # Primary rule in control
                improvement_score=score,
                execution_time_ms=np.random.normal(120, 20),
                confidence_level=np.random.uniform(0.8, 0.95),
                parameters_used={"weight": 1.0, "threshold": 0.7, "experiment_arm": "control"},
                created_at=aware_utc_now() - timedelta(hours=session_idx // 10)
            )
            rule_performance_records.append(record)
            session_idx += 1
        
        # Treatment group performance (based on treatment_rules)
        treatment_base_score = control_base_score + 0.05 + (exp_idx * 0.01)  # Slight improvement
        for i in range(half_sessions):
            score = np.random.normal(treatment_base_score, 0.08)
            score = max(0.0, min(1.0, score))  # Clamp to valid range
            
            # Alternate between treatment rules for variety
            rule_id = "specificity_rule" if i % 2 == 0 else "chain_of_thought_rule"
            if exp_idx == 2:  # Completed experiment uses different treatment
                rule_id = "example_rule"
            
            record = RulePerformance(
                session_id=f"exp_session_{session_idx}",
                rule_id=rule_id,
                improvement_score=score,
                execution_time_ms=np.random.normal(135, 25),  # Slightly higher for treatment
                confidence_level=np.random.uniform(0.75, 0.92),
                parameters_used={"weight": 0.8, "threshold": 0.6, "experiment_arm": "treatment"},
                created_at=aware_utc_now() - timedelta(hours=session_idx // 10)
            )
            rule_performance_records.append(record)
            session_idx += 1

    # Add all records to database
    real_db_session.add_all(rule_metadata_records + experiments + prompt_sessions + rule_performance_records)
    await real_db_session.commit()

    # Refresh to get database-generated values
    for experiment in experiments:
        await real_db_session.refresh(experiment)
    
    for session in prompt_sessions:
        await real_db_session.refresh(session)

    for record in rule_performance_records:
        await real_db_session.refresh(record)

    return experiments, rule_performance_records

# ===============================
# REDIS CONTAINER SETUP
# ===============================

@pytest.fixture(scope="session")
def redis_container():
    """Starts a Redis container for the duration of the testing session.
    
    Uses Testcontainers to spin up a disposable Redis instance that is
    automatically cleaned up after the test session completes.
    """
    with RedisContainer() as container:
        yield container

@pytest.fixture(scope="function")
async def redis_client(redis_container):
    """Provides a fresh Redis client for each test and flushes the database.
    
    This fixture ensures clean Redis state between tests by:
    1. Creating a new Redis client connection
    2. Flushing the database before each test
    3. Properly cleaning up the connection after each test
    
    Args:
        redis_container: Session-scoped Redis container fixture
        
    Yields:
        redis.asyncio.Redis: Async Redis client instance
    """
    # Get Redis client from container
    sync_client = redis_container.get_client()
    
    # Create async client using connection parameters
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    
    client = aioredis.Redis(
        host=host,
        port=port,
        decode_responses=False
    )
    
    # Ensure clean state for each test
    await client.flushdb()
    
    yield client
    
    # Cleanup connection
    await client.close()

# Temporary File Infrastructure
@pytest.fixture(scope="function")
def test_data_dir(tmp_path):
    """Function-scoped temporary directory for test data.

    Uses pytest's tmp_path for automatic cleanup and proper isolation.
    """
    data_dir = tmp_path / "test_apes_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    (data_dir / "data").mkdir()
    (data_dir / "config").mkdir()
    (data_dir / "logs").mkdir()
    (data_dir / "temp").mkdir()

    return data_dir


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Async Event Loop Management
@pytest.fixture(scope="function")
def event_loop():
    """Provide a fresh event loop for each test function.

    Function scope ensures complete isolation between async tests.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Sample Data Fixtures
@pytest.fixture(scope="session")
def sample_training_data():
    """Session-scoped sample data for ML testing.

    Expensive to generate, safe to reuse across tests.
    """
    return {
        "features": [
            [0.8, 150, 1.0, 5, 0.7, 1.0],  # High effectiveness
            [0.6, 200, 0.8, 4, 0.6, 1.0],  # Medium effectiveness
            [0.4, 300, 0.6, 3, 0.5, 0.0],  # Low effectiveness
            [0.9, 100, 1.0, 5, 0.8, 1.0],  # Best performance
            [0.3, 400, 0.4, 2, 0.4, 0.0],  # Poor performance
        ]
        * 5,  # 25 samples total for reliable ML testing
        "effectiveness_scores": [0.8, 0.6, 0.4, 0.9, 0.3] * 5,
    }


@pytest.fixture(scope="session")
def sample_ml_training_data():
    """Sample ML training data for testing."""
    return {
        "features": [
            [0.8, 150, 1.0, 5, 0.9, 6, 0.7, 1.0, 0.1, 0.5],  # High performance
            [0.7, 200, 0.8, 4, 0.8, 5, 0.6, 1.0, 0.2, 0.4],  # Medium performance
            [0.6, 250, 0.6, 3, 0.7, 4, 0.5, 0.0, 0.3, 0.3],  # Lower performance
            [0.9, 100, 1.0, 5, 0.95, 7, 0.8, 1.0, 0.05, 0.6],  # Best performance
            [0.5, 300, 0.4, 2, 0.6, 3, 0.4, 0.0, 0.4, 0.2],  # Poor performance
        ]
        * 10,  # 50 samples total
        "effectiveness_scores": [0.8, 0.7, 0.6, 0.9, 0.5] * 10,
    }


# Configuration Override Fixture
@pytest.fixture(scope="function")
def test_config():
    """Function-scoped test configuration override."""
    return {
        "database": {"host": "localhost", "database": "apes_test", "user": "test_user"},
        "performance": {"target_response_time_ms": 200, "timeout_seconds": 5},
        "ml": {"min_training_samples": 10, "optimization_timeout": 60},
    }


# Real Database Fixtures - Replace mocks with actual database records
@pytest.fixture
async def real_rule_metadata(real_db_session):
    """Create real rule metadata in the test database."""
    import uuid
    from sqlalchemy import select
    
    test_suffix = str(uuid.uuid4())[:8]
    
    metadata_records = [
        RuleMetadata(
            rule_id=f"clarity_rule_{test_suffix}",
            rule_name="Clarity Enhancement Rule",
            category="core",
            description="Improves prompt clarity",
            enabled=True,
            priority=5,
            default_parameters={"weight": 1.0, "threshold": 0.7},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
        ),
        RuleMetadata(
            rule_id=f"specificity_rule_{test_suffix}",
            rule_name="Specificity Enhancement Rule",
            category="core",
            description="Improves prompt specificity",
            enabled=True,
            priority=4,
            default_parameters={"weight": 0.8, "threshold": 0.6},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
        ),
    ]
    
    # Add records to database
    for record in metadata_records:
        real_db_session.add(record)
    await real_db_session.commit()
    
    # Refresh to get database-generated values
    for record in metadata_records:
        await real_db_session.refresh(record)
    
    return metadata_records


@pytest.fixture
async def real_prompt_sessions(real_db_session):
    """Create real prompt sessions in the test database."""
    from datetime import datetime, timedelta
    from prompt_improver.database.models import PromptSession
    
    session_records = [
        PromptSession(
            id=1,
            session_id="test_session_1",
            original_prompt="Make this better",
            improved_prompt="Please improve the clarity and specificity of this document",
            user_context={"context": "document_improvement"},
            quality_score=0.8,
            improvement_score=0.75,
            confidence_level=0.9,
            created_at=aware_utc_now(),
            updated_at=aware_utc_now(),
        ),
        PromptSession(
            id=2,
            session_id="test_session_2",
            original_prompt="Help me with this task", 
            improved_prompt="Please provide step-by-step guidance for completing this specific task",
            user_context={"context": "task_guidance"},
            quality_score=0.85,
            improvement_score=0.8,
            confidence_level=0.88,
            created_at=aware_utc_now() - timedelta(hours=1),
            updated_at=aware_utc_now() - timedelta(hours=1),
        ),
    ]
    
    # Add records to database
    for record in session_records:
        real_db_session.add(record)
    await real_db_session.commit()
    
    # Refresh to get database-generated values
    for record in session_records:
        await real_db_session.refresh(record)
    
    return session_records


# PromptSession fixtures for referential integrity
@pytest.fixture
def sample_prompt_sessions():
    """Sample prompt sessions for testing with proper database relationships."""
    from datetime import datetime, timedelta
    from prompt_improver.database.models import PromptSession
    
    return [
        PromptSession(
            id=1,
            session_id="test_session_1",
            original_prompt="Make this better",
            improved_prompt="Please improve the clarity and specificity of this document",
            user_context={"context": "document_improvement"},
            quality_score=0.8,
            improvement_score=0.75,
            confidence_level=0.9,
            created_at=aware_utc_now(),
            updated_at=aware_utc_now(),
        ),
        PromptSession(
            id=2,
            session_id="test_session_2",
            original_prompt="Help me with this task", 
            improved_prompt="Please provide step-by-step guidance for completing this specific task",
            user_context={"context": "task_guidance"},
            quality_score=0.85,
            improvement_score=0.8,
            confidence_level=0.88,
            created_at=aware_utc_now() - timedelta(hours=1),
            updated_at=aware_utc_now() - timedelta(hours=1),
        ),
    ]


# Legacy fixture for backward compatibility
@pytest.fixture
def sample_rule_metadata():
    """Sample rule metadata for testing with unique IDs per test."""
    import uuid

    # Generate unique suffix for this test run
    test_suffix = str(uuid.uuid4())[:8]

    return [
        RuleMetadata(
            rule_id=f"clarity_rule_{test_suffix}",
            rule_name="Clarity Enhancement Rule",
            category="core",
            description="Improves prompt clarity",
            enabled=True,
            priority=5,
            default_parameters={"weight": 1.0, "threshold": 0.7},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
            created_at=aware_utc_now(),
        ),
        RuleMetadata(
            rule_id=f"specificity_rule_{test_suffix}",
            rule_name="Specificity Enhancement Rule",
            category="core",
            description="Improves prompt specificity",
            enabled=True,
            priority=4,
            default_parameters={"weight": 0.8, "threshold": 0.6},
            parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
            created_at=aware_utc_now(),
        ),
    ]


@pytest.fixture
def sample_rule_performance(sample_rule_metadata, sample_prompt_sessions):
    """Sample rule performance data for testing with unique rule IDs and proper database relationships."""
    import uuid
    from datetime import datetime, timedelta

    # Use the same unique rule IDs from sample_rule_metadata and session_ids from sample_prompt_sessions
    base_data = [
        RulePerformance(
            id=1,
            session_id=sample_prompt_sessions[0].session_id,  # "test_session_1"
            rule_id=sample_rule_metadata[0].rule_id,  # clarity_rule with unique suffix
            improvement_score=0.8,
            confidence_level=0.9,
            execution_time_ms=150,
            parameters_used={"weight": 1.0, "threshold": 0.7},
            created_at=aware_utc_now(),
        ),
        RulePerformance(
            id=2,
            session_id=sample_prompt_sessions[1].session_id,  # "test_session_2"
            rule_id=sample_rule_metadata[
                1
            ].rule_id,  # specificity_rule with unique suffix
            improvement_score=0.7,
            confidence_level=0.8,
            execution_time_ms=200,
            parameters_used={"weight": 0.8, "threshold": 0.6},
            created_at=aware_utc_now() - timedelta(minutes=30),
        ),
    ]

    # Create multiple records with unique IDs and ensure no constraint violations
    result = []
    for i in range(15):  # 15 records each = 30 total
        for j, base in enumerate(base_data):
            # Create new instance with unique ID and constrained scores
            improvement_score = max(0.0, min(1.0, base.improvement_score + (i * 0.01)))
            confidence_level = max(0.0, min(1.0, base.confidence_level + (i * 0.005)))

            new_record = RulePerformance(
                id=i * len(base_data) + j + 1,  # Unique ID for each record
                session_id=base.session_id,  # Use session_id from base data (referencing existing sessions)
                rule_id=base.rule_id,
                improvement_score=improvement_score,  # Keep within 0-1 range
                confidence_level=confidence_level,  # Keep within 0-1 range
                execution_time_ms=base.execution_time_ms + i,
                parameters_used=base.parameters_used,
                created_at=aware_utc_now() - timedelta(minutes=i*5),  # Staggered creation times
            )
            result.append(new_record)

    return result


@pytest.fixture
def sample_user_feedback():
    """Sample user feedback for testing."""
    return [
        UserFeedback(
            id=1,
            session_id="test_session_1",
            rating=4,
            feedback_text="Good improvement",
            improvement_areas=["clarity", "specificity"],
            is_processed=False,
            ml_optimized=False,
            model_id=None,
            created_at=aware_utc_now(),
        ),
        UserFeedback(
            id=2,
            session_id="test_session_2",
            rating=5,
            feedback_text="Excellent improvement",
            improvement_areas=["clarity"],
            is_processed=True,
            ml_optimized=True,
            model_id="model_123",
            created_at=aware_utc_now() - timedelta(hours=2),
        ),
    ]


@pytest.fixture
def sample_improvement_sessions():
    """Sample improvement sessions for testing."""
    return [
        ImprovementSession(
            id=1,
            session_id="test_session_1",
            original_prompt="Make this better",
            final_prompt="Please improve the clarity and specificity of this document",
            rules_applied=["clarity_rule"],
            user_context={"context": "document_improvement"},
            improvement_metrics={"clarity": 0.8, "specificity": 0.7},
            created_at=aware_utc_now(),
        ),
        ImprovementSession(
            id=2,
            session_id="test_session_2",
            original_prompt="Help me with this task",
            final_prompt="Please provide step-by-step guidance for completing this specific task",
            rules_applied=["clarity_rule", "specificity_rule"],
            user_context={"context": "task_guidance"},
            improvement_metrics={"clarity": 0.9, "specificity": 0.8},
            created_at=aware_utc_now() - timedelta(hours=1),
        ),
    ]


# Service Instance Fixtures
@pytest.fixture
def ml_service():
    """Create ML service instance for testing."""
    with patch("prompt_improver.ml.core.ml_integration.mlflow"):
        from prompt_improver.ml.core.ml_integration import MLModelService

        return MLModelService()


@pytest.fixture
def prompt_service():
    """Create PromptImprovementService instance."""
    from prompt_improver.core.services.prompt_improvement import PromptImprovementService

    return PromptImprovementService()


# LLM Transformer Service Fixtures for Unit Testing
@pytest.fixture
def mock_llm_transformer():
    """Mock LLMTransformerService for unit testing rule logic.

    Provides realistic transformation responses without external dependencies.
    Function-scoped to ensure test isolation.
    """
    from unittest.mock import AsyncMock, MagicMock

    service = MagicMock()

    # Mock enhance_clarity method with realistic responses
    async def mock_enhance_clarity(prompt, vague_words, context=None):
        enhanced_prompt = prompt
        transformations = []

        # Simulate realistic clarity improvements
        for word in vague_words:
            if word.lower() == "thing":
                enhanced_prompt = enhanced_prompt.replace(word, "specific item")
                transformations.append({
                    "type": "clarity_enhancement",
                    "original_word": word,
                    "replacement": "specific item",
                    "reason": "Improved specificity",
                })
            elif word.lower() == "stuff":
                enhanced_prompt = enhanced_prompt.replace(word, "relevant details")
                transformations.append({
                    "type": "clarity_enhancement",
                    "original_word": word,
                    "replacement": "relevant details",
                    "reason": "Improved specificity",
                })

        return {
            "enhanced_prompt": enhanced_prompt,
            "transformations": transformations,
            "confidence": 0.8,
            "improvement_type": "clarity",
        }

    # Mock enhance_specificity method with realistic responses
    async def mock_enhance_specificity(prompt, context=None):
        enhanced_prompt = prompt
        transformations = []

        # Simulate specificity improvements based on prompt length and content
        if len(prompt.split()) < 5:  # Short prompts get more enhancement
            enhanced_prompt += (
                "\n\nFormat: Please provide specific details and examples."
            )
            transformations.append({
                "type": "format_specification",
                "addition": "Format: Please provide specific details and examples.",
                "reason": "Added output format requirements",
            })

        return {
            "enhanced_prompt": enhanced_prompt,
            "transformations": transformations,
            "confidence": 0.75,
            "improvement_type": "specificity",
        }

    service.enhance_clarity = AsyncMock(side_effect=mock_enhance_clarity)
    service.enhance_specificity = AsyncMock(side_effect=mock_enhance_specificity)

    return service


@pytest.fixture
def sample_test_prompts():
    """Sample prompts for testing rule behavior.

    Provides variety of prompt types for comprehensive rule testing.
    """
    return {
        "vague_prompts": [
            "fix this thing",
            "make this stuff better",
            "help me with this",
            "analyze this data",
        ],
        "clear_prompts": [
            "Please rewrite the following paragraph to be suitable for a fifth-grade reading level.",
            "Write a Python function named 'calculate_fibonacci' that takes an integer n as input.",
            "Create a detailed project timeline for implementing user authentication.",
        ],
        "short_prompts": ["help", "summarize", "explain", "analyze"],
        "specific_prompts": [
            "Write a Python function that takes a list of integers and returns the second largest value.",
            "Create a SQL query to find all users who registered in the last 30 days.",
            "Design a RESTful API endpoint for updating user profile information.",
        ],
    }


@pytest.fixture
def mock_rule_metadata_corrected():
    """Mock rule metadata with correct field names matching database schema.

    Uses 'default_parameters' instead of 'parameters' to match RuleMetadata model.
    """
    from datetime import datetime

    from prompt_improver.database.models import RuleMetadata

    return [
        RuleMetadata(
            rule_id="clarity_rule",
            rule_name="Clarity Enhancement Rule",
            category="core",
            description="Improves prompt clarity by replacing vague terms",
            enabled=True,
            priority=5,
            default_parameters={"vague_threshold": 0.7, "confidence_weight": 1.0},
            parameter_constraints={"vague_threshold": {"min": 0.0, "max": 1.0}},
        ),
        RuleMetadata(
            rule_id="specificity_rule",
            rule_name="Specificity Enhancement Rule",
            category="core",
            description="Improves prompt specificity by adding constraints and examples",
            enabled=True,
            priority=4,
            default_parameters={"min_length": 10, "add_format": True},
            parameter_constraints={"min_length": {"min": 1, "max": 100}},
        ),
    ]


# Mock Service Fixtures
@pytest.fixture
def mock_ml_service():
    """Mock ML service for testing."""
    service = MagicMock()
    service.optimize_rules = AsyncMock(
        return_value={
            "status": "success",
            "model_id": "test_model_123",
            "best_score": 0.85,
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
            "processing_time_ms": 1500,
        }
    )
    service.predict_rule_effectiveness = AsyncMock(
        return_value={
            "status": "success",
            "prediction": 0.8,
            "confidence": 0.9,
            "probabilities": [0.1, 0.9],
            "processing_time_ms": 2,
        }
    )
    service.optimize_ensemble_rules = AsyncMock(
        return_value={
            "status": "success",
            "ensemble_score": 0.88,
            "ensemble_std": 0.05,
            "processing_time_ms": 3000,
        }
    )
    service.discover_patterns = AsyncMock(
        return_value={
            "status": "success",
            "patterns_discovered": 3,
            "patterns": [
                {
                    "parameters": {"weight": 1.0},
                    "avg_effectiveness": 0.85,
                    "support_count": 10,
                },
                {
                    "parameters": {"weight": 0.9},
                    "avg_effectiveness": 0.82,
                    "support_count": 8,
                },
                {
                    "parameters": {"weight": 0.8},
                    "avg_effectiveness": 0.79,
                    "support_count": 7,
                },
            ],
            "total_analyzed": 100,
            "processing_time_ms": 1200,
        }
    )
    return service


@pytest.fixture
def mock_prompt_service():
    """Mock prompt improvement service for testing."""
    service = MagicMock()
    service.improve_prompt = AsyncMock(
        return_value={
            "original_prompt": "Test prompt",
            "improved_prompt": "Enhanced test prompt with better clarity and specificity",
            "applied_rules": [{"rule_id": "clarity_rule", "confidence": 0.9}],
            "processing_time_ms": 100,
            "session_id": "test_session_123",
        }
    )
    service.trigger_optimization = AsyncMock(
        return_value={
            "status": "success",
            "performance_score": 0.85,
            "training_samples": 25,
        }
    )
    service.run_ml_optimization = AsyncMock(
        return_value={
            "status": "success",
            "best_score": 0.88,
            "model_id": "optimized_model_456",
        }
    )
    service.discover_patterns = AsyncMock(
        return_value={"status": "success", "patterns_discovered": 2}
    )
    return service


# mock_analytics_service fixture removed - using real RealTimeAnalyticsService instead
# This supports the transition from mocked to real services in orchestrator tests


# MLflow and Optuna Mock Fixtures
@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing ML operations."""
    with (
        patch("mlflow.start_run") as mock_start,
        patch("mlflow.log_params") as mock_log_params,
        patch("mlflow.log_metrics") as mock_log_metrics,
        patch("mlflow.sklearn.log_model") as mock_log_model,
        patch("mlflow.active_run") as mock_active_run,
    ):
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        yield {
            "start_run": mock_start,
            "log_params": mock_log_params,
            "log_metrics": mock_log_metrics,
            "log_model": mock_log_model,
            "active_run": mock_active_run,
        }


@pytest.fixture
def mock_optuna():
    """Mock Optuna for testing hyperparameter optimization."""
    with patch("optuna.create_study") as mock_create_study:
        mock_study = MagicMock()
        mock_study.best_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
        }
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study

        yield {"create_study": mock_create_study, "study": mock_study}


# Performance Testing Utilities
@pytest.fixture
def performance_threshold():
    """Performance thresholds for Phase 3 requirements."""
    return {
        "prediction_latency_ms": 5,  # <5ms for predictions
        "optimization_timeout_s": 300,  # 5 minute timeout for optimization
        "cache_hit_ratio": 0.9,  # >90% cache hit ratio target
        "database_query_ms": 50,  # <50ms for database queries
    }


# Async Context Manager Helper
class AsyncContextManager:
    """Helper class for testing async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Factory for creating async context managers."""
    return AsyncContextManager


# Test Data Generation Utilities
def generate_test_features(n_samples: int = 25, n_features: int = 10):
    """Generate test feature data for ML testing."""
    import numpy as np

    return np.random.rand(n_samples, n_features).tolist()


def generate_test_effectiveness_scores(n_samples: int = 25):
    """Generate test effectiveness scores for ML testing."""
    import numpy as np

    return np.random.uniform(0.3, 0.9, n_samples).tolist()


@pytest.fixture
def test_data_generator():
    """Factory for generating test data."""
    return {
        "features": generate_test_features,
        "effectiveness_scores": generate_test_effectiveness_scores,
    }


# Database Population Utilities
async def populate_test_database(
    session: AsyncSession,
    rule_metadata_list=None,
    rule_performance_list=None,
    user_feedback_list=None,
):
    """Populate test database with sample data."""
    if rule_metadata_list:
        for rule in rule_metadata_list:
            session.add(rule)

    if rule_performance_list:
        for perf in rule_performance_list:
            session.add(perf)

    if user_feedback_list:
        for feedback in user_feedback_list:
            session.add(feedback)

    await session.commit()


@pytest.fixture
def populate_db():
    """Database population utility."""
    return populate_test_database


# Async Testing Utilities
def async_test(f):
    """Decorator for async test functions."""
    import functools

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@pytest.fixture
def async_test_decorator():
    """Async test decorator utility."""
    return async_test
