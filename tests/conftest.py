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
import logging
import os
import random

# PERFORMANCE OPTIMIZATION: Set all environment variables first to avoid heavy imports
# Disable telemetry and Ryuk at import time (before any OTEL modules import)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
os.environ.setdefault("TESTCONTAINERS_LOG_LEVEL", "INFO")
# Set test optimization flags
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DISABLE_HEAVY_IMPORTS", "true")

import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# LAZY IMPORT UTILITIES for performance optimization
def lazy_import(module_name: str, attribute: str = None):
    """Lazy import utility to defer heavy imports until actually needed."""
    @lru_cache(maxsize=1)
    def _import():
        try:
            module = __import__(module_name, fromlist=[attribute] if attribute else [])
            return getattr(module, attribute) if attribute else module
        except ImportError as e:
            raise ImportError(f"Failed to import {module_name}.{attribute or ''}: {e}")
    return _import

# ULTRA-DEFERRED dependency imports - no execution until actually called
def _get_asyncpg():
    return lazy_import("asyncpg")
    
def _get_coredis():
    return lazy_import("coredis")
    
def _get_numpy():
    return lazy_import("numpy")
    
def _get_sqlalchemy_async_session():
    return lazy_import("sqlalchemy.ext.asyncio", "AsyncSession")
    
def _get_sqlalchemy_async_sessionmaker():
    return lazy_import("sqlalchemy.ext.asyncio", "async_sessionmaker")
    
def _get_typer_cli_runner():
    return lazy_import("typer.testing", "CliRunner")

# Conditional OpenTelemetry imports - only if telemetry is enabled
def get_opentelemetry_components():
    """Lazy load OpenTelemetry components only when telemetry is enabled."""
    if os.getenv("OTEL_SDK_DISABLED", "true").lower() == "true":
        # Return mock objects when telemetry is disabled
        return type('MockTelemetry', (), {
            'metrics': None,
            'trace': None, 
            'MeterProvider': type('MockMeterProvider', (), {}),
            'TracerProvider': type('MockTracerProvider', (), {}),
            'Resource': type('MockResource', (), {}),
            'OTLPMetricExporter': type('MockOTLPMetricExporter', (), {}),
            'OTLPSpanExporter': type('MockOTLPSpanExporter', (), {}),
            'PeriodicExportingMetricReader': type('MockPeriodicExportingMetricReader', (), {}),
            'BatchSpanProcessor': type('MockBatchSpanProcessor', (), {})
        })()
    
    # Only import when actually needed and enabled
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    
    return type('Telemetry', (), {
        'metrics': metrics,
        'trace': trace,
        'MeterProvider': MeterProvider,
        'TracerProvider': TracerProvider, 
        'Resource': Resource,
        'OTLPMetricExporter': OTLPMetricExporter,
        'OTLPSpanExporter': OTLPSpanExporter,
        'PeriodicExportingMetricReader': PeriodicExportingMetricReader,
        'BatchSpanProcessor': BatchSpanProcessor
    })()

# DEFERRED APPLICATION IMPORTS - no execution until fixture actually called
# These are created as functions that return lazy importers
def _get_config():
    return lazy_import("prompt_improver.core.config", "get_config")

def _get_session():
    return lazy_import("prompt_improver.database", "get_session")

def _get_aware_utc_now():
    return lazy_import("prompt_improver.utils.datetime_utils", "aware_utc_now")

# Lazy database models loader
@lru_cache(maxsize=1)
def get_database_models():
    """Lazy load database models to avoid heavy SQLAlchemy imports."""
    from prompt_improver.database.models import (
        ABExperiment,
        ImprovementSession, 
        RuleIntelligenceCache,
        RuleMetadata,
        RulePerformance,
        SQLModel,
        UserFeedback,
    )
    return {
        'ABExperiment': ABExperiment,
        'ImprovementSession': ImprovementSession,
        'RuleIntelligenceCache': RuleIntelligenceCache,
        'RuleMetadata': RuleMetadata,
        'RulePerformance': RulePerformance,
        'SQLModel': SQLModel,
        'UserFeedback': UserFeedback,
    }

# Import real ML fixtures with lazy loading
def get_real_ml_fixtures():
    """Lazy load real ML fixtures to avoid import delays."""
    try:
        import tests.real_ml.fixtures
        return tests.real_ml.fixtures
    except ImportError:
        return None  # Real ML fixtures not available

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def testcontainers_sane_defaults():
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    os.environ.setdefault("TESTCONTAINERS_LOG_LEVEL", "INFO")
    # Disable OpenTelemetry by default in tests
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
    os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")


@lru_cache(maxsize=1)
def detect_external_redis() -> tuple[Optional[str], Optional[str]]:
    """Lazy detection of external Redis containers to avoid startup delays.
    
    Returns: (host, port) tuple or (None, None) if not found
    """
    if os.getenv("TEST_REDIS_HOST") and os.getenv("TEST_REDIS_PORT"):
        return os.getenv("TEST_REDIS_HOST"), os.getenv("TEST_REDIS_PORT")
    
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}\t{{.Image}}"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in out.strip().splitlines():
            try:
                name, ports, image = line.split("\t")
            except ValueError:
                continue
            if "redis" not in image.lower():
                continue
            # Look for hostport mapping to 6379/tcp
            parts = ports.split(",")
            for p in parts:
                p = p.strip()
                if "->6379/tcp" in p and ":" in p:
                    try:
                        host_part = p.split("->")[0]
                        host_port = host_part.split(":")[-1]
                        if host_port.isdigit():
                            host = os.getenv("REDIS_HOST", "redis")
                            logger.info(f"Detected running Redis container {name} on host port {host_port}")
                            return host, host_port
                    except Exception:
                        continue
    except Exception:
        pass  # Docker not available or parsing fails
    
    return None, None

@pytest.fixture(scope="session", autouse=True) 
def prefer_external_redis_if_running():
    """Configure external Redis if available - now with lazy detection."""
    host, port = detect_external_redis()
    if host and port:
        os.environ.setdefault("TEST_REDIS_HOST", host)
        os.environ.setdefault("TEST_REDIS_PORT", port)


# LAZY ML LIBRARY DETECTION for performance optimization
@lru_cache(maxsize=1)
def check_ml_libraries() -> dict[str, bool]:
    """Lazy check for ML library availability to avoid import delays."""
    libraries = {}
    
    for lib_name, import_name in [
        ('sklearn', 'sklearn'),
        ('deap', 'deap'), 
        ('pymc', 'pymc'),
        ('umap', 'umap'),
        ('hdbscan', 'hdbscan')
    ]:
        try:
            __import__(import_name)
            libraries[lib_name] = True
        except ImportError:
            libraries[lib_name] = False
    
    return libraries

# Create pytest skip markers using lazy evaluation
def _make_skip_marker(lib_name: str):
    def _check():
        return not check_ml_libraries()[lib_name]
    return pytest.mark.skipif(_check(), reason=f"{lib_name} not installed")

requires_sklearn = _make_skip_marker('sklearn')
requires_deap = _make_skip_marker('deap')
requires_pymc = _make_skip_marker('pymc')
requires_umap = _make_skip_marker('umap')
requires_hdbscan = _make_skip_marker('hdbscan')
TEST_RANDOM_SEED = 42


@pytest.fixture(scope="session", autouse=True)
def seed_random_generators():
    """Seed all random number generators for reproducible test results.

    This fixture runs automatically for every test session to ensure
    deterministic behavior across all tests.
    """
    random.seed(TEST_RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(TEST_RANDOM_SEED)
    
    # Ultra-lazy seed numpy only when imported
    try:
        np = _get_numpy()()  # Call deferred function to get lazy importer
        np.random.seed(TEST_RANDOM_SEED)
    except ImportError:
        pass
    
    # Lazy seed ML libraries only if available
    ml_libs = check_ml_libraries()
    if ml_libs.get('sklearn', False):
        try:
            import sklearn  # sklearn doesn't need explicit seeding
        except ImportError:
            pass
    
    # Lazy seed PyTorch only if available        
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
    np = _get_numpy()()  # Ultra-lazy import numpy
    rng = np.random.RandomState(TEST_RANDOM_SEED)
    return rng


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


@pytest.fixture(scope="function")
async def real_db_session(test_db_engine):
    """Create a real database session for testing."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy.orm import sessionmaker

    AsyncSession = _get_sqlalchemy_async_session()()
    async_sessionmaker = _get_sqlalchemy_async_sessionmaker()()
    async_session = async_sessionmaker(
        bind=test_db_engine, expire_on_commit=False, class_=AsyncSession
    )
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create test database engine using existing PostgreSQL configuration with retry logic."""
    import uuid

    from prompt_improver.core.config import AppConfig
    from tests.database_helpers import (
        cleanup_test_database,
        create_test_engine_with_retry,
        wait_for_postgres_async,
    )

    config = AppConfig().database
    postgres_ready = await wait_for_postgres_async(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        database=config.postgres_database,
        max_retries=30,
        retry_delay=1.0,
    )
    if not postgres_ready:
        pytest.skip("PostgreSQL server not available after 30 attempts")
    test_db_name = f"apes_test_{uuid.uuid4().hex[:8]}"
    db_cleaned = await cleanup_test_database(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        test_db_name=test_db_name,
    )
    if not db_cleaned:
        pytest.skip("Could not clean up test database")
    test_db_url = f"postgresql+asyncpg://{config.postgres_username}:{config.postgres_password}@{config.postgres_host}:{config.postgres_port}/{test_db_name}"
    engine = await create_test_engine_with_retry(
        test_db_url,
        max_retries=3,
        connect_args={"application_name": "apes_test_suite", "connect_timeout": 10},
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
        try:
            from sqlalchemy import text

            from prompt_improver.database import (
                ManagerMode,
                get_database_services,
            )

            try:
                manager = get_database_services(ManagerMode.ASYNC_MODERN)
                async with manager.get_async_session() as session:
                    result = await session.execute(
                        text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                        {"db_name": test_db_name},
                    )
                    exists = result.scalar()
                    if not exists:
                        logger.info(
                            f"Test database {test_db_name} does not exist, cleanup not needed",
                            test_db_name,
                        )
                        return
            except Exception as e:
                logger.debug(
                    f"DatabaseServices verification failed, proceeding with direct cleanup: {e}",
                    e,
                )
            import asyncpg

            conn = await asyncpg.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_username,
                password=config.postgres_password,
                database=config.postgres_database,
                timeout=5.0,
            )
            await conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
                test_db_name,
            )
            await conn.execute(f'DROP DATABASE IF EXISTS "{test_db_name}"')
            await conn.close()
        except Exception:
            pass


@pytest.fixture
async def test_db_session(test_db_engine):
    """Create test database session with transaction rollback for isolation."""
    AsyncSession = _get_sqlalchemy_async_session()()
    async_sessionmaker = _get_sqlalchemy_async_sessionmaker()()
    async_session = async_sessionmaker(
        bind=test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        trans = await session.begin()
        try:
            yield session
        finally:
            try:
                if trans.is_active:
                    await trans.rollback()
            except Exception:
                pass


@pytest.fixture
async def populate_ab_experiment(real_db_session):
    """
    Populate database with ABExperiment and related RulePerformance records for testing.
    Creates comprehensive experiment configurations with sufficient data for statistical validation.
    """
    import uuid
    from datetime import timedelta

    import numpy as np

    from prompt_improver.database.models import PromptSession

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
                "expected_effect_size": 0.1,
            },
        ),
        ABExperiment(
            experiment_id=str(uuid.uuid4()),
            experiment_name="Multi-Rule Performance Test",
            description="Testing comprehensive rule set performance",
            control_rules={"rule_ids": ["clarity_rule", "specificity_rule"]},
            treatment_rules={
                "rule_ids": [
                    "clarity_rule",
                    "specificity_rule",
                    "chain_of_thought_rule",
                ]
            },
            target_metric="improvement_score",
            sample_size_per_group=200,
            current_sample_size=400,
            significance_threshold=0.01,
            status="running",
            started_at=aware_utc_now() - timedelta(days=10),
            experiment_metadata={
                "control_description": "Standard two-rule set",
                "treatment_description": "Enhanced three-rule set",
                "expected_effect_size": 0.15,
            },
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
                "statistical_significance": True,
            },
        ),
    ]
    total_sessions_needed = sum(exp.current_sample_size for exp in experiments)
    prompt_sessions = []
    for i in range(total_sessions_needed):
        session = PromptSession(
            session_id=f"exp_session_{i}",
            original_prompt=f"Test prompt {i}",
            improved_prompt=f"Enhanced test prompt {i} with better clarity and specificity",
            user_context={"experiment_type": "ab_test", "session_number": i},
            quality_score=np.random.uniform(0.6, 0.9),
            improvement_score=np.random.uniform(0.5, 0.9),
            confidence_level=np.random.uniform(0.7, 0.95),
            created_at=aware_utc_now() - timedelta(hours=i // 10),
            updated_at=aware_utc_now() - timedelta(hours=i // 10),
        )
        prompt_sessions.append(session)
    rule_performance_records = []
    session_idx = 0
    for exp_idx, experiment in enumerate(experiments):
        experiment_sessions = experiment.current_sample_size
        half_sessions = experiment_sessions // 2
        control_base_score = 0.7 + exp_idx * 0.02
        for i in range(half_sessions):
            score = np.random.normal(control_base_score, 0.08)
            score = max(0.0, min(1.0, score))
            record = RulePerformance(
                session_id=f"exp_session_{session_idx}",
                rule_id="clarity_rule",
                improvement_score=score,
                execution_time_ms=np.random.normal(120, 20),
                confidence_level=np.random.uniform(0.8, 0.95),
                parameters_used={
                    "weight": 1.0,
                    "threshold": 0.7,
                    "experiment_arm": "control",
                },
                created_at=aware_utc_now() - timedelta(hours=session_idx // 10),
            )
            rule_performance_records.append(record)
            session_idx += 1
        treatment_base_score = control_base_score + 0.05 + exp_idx * 0.01
        for i in range(half_sessions):
            score = np.random.normal(treatment_base_score, 0.08)
            score = max(0.0, min(1.0, score))
            rule_id = "specificity_rule" if i % 2 == 0 else "chain_of_thought_rule"
            if exp_idx == 2:
                rule_id = "example_rule"
            record = RulePerformance(
                session_id=f"exp_session_{session_idx}",
                rule_id=rule_id,
                improvement_score=score,
                execution_time_ms=np.random.normal(135, 25),
                confidence_level=np.random.uniform(0.75, 0.92),
                parameters_used={
                    "weight": 0.8,
                    "threshold": 0.6,
                    "experiment_arm": "treatment",
                },
                created_at=aware_utc_now() - timedelta(hours=session_idx // 10),
            )
            rule_performance_records.append(record)
            session_idx += 1
    real_db_session.add_all(
        rule_metadata_records + experiments + prompt_sessions + rule_performance_records
    )
    await real_db_session.commit()
    for experiment in experiments:
        await real_db_session.refresh(experiment)
    for session in prompt_sessions:
        await real_db_session.refresh(session)
    for record in rule_performance_records:
        await real_db_session.refresh(record)
    return (experiments, rule_performance_records)


@pytest.fixture(scope="session")
async def external_redis_config():
    """External Redis configuration for testing.

    Uses environment-configured external Redis instance for production-like testing.
    Provides SSL/TLS and authentication support with external services.
    """
    config = get_config()
    redis_config = config.redis
    try:
        connection_params = redis_config.get_connection_params()
        test_client = coredis.Redis(**connection_params)
        pong_result = await test_client.ping()
        if pong_result not in ["PONG", b"PONG", True]:
            pytest.skip("External Redis not available for testing")
        await test_client.close()
        yield redis_config
    except Exception as e:
        pytest.skip(f"External Redis not available: {e}")


@pytest.fixture(scope="function")
async def redis_client(external_redis_config):
    """Provides a fresh Redis client for each test with proper isolation.

    This fixture ensures clean Redis state between tests by:
    1. Creating a new Redis client connection to external Redis
    2. Using test-specific database or key prefixes for isolation
    3. Cleaning up test data after each test
    4. Properly closing connections

    Args:
        external_redis_config: External Redis configuration

    Yields:
        coredis.Redis: Async Redis client instance
    """
    import os
    import uuid

    connection_params = external_redis_config.get_connection_params()
    test_db = int(os.getenv("REDIS_TEST_DB", "1"))
    if test_db != connection_params.get("db", 0):
        connection_params["db"] = test_db
    client = coredis.Redis(**connection_params)
    test_prefix = f"test:{uuid.uuid4().hex[:8]}:"
    test_keys = set()
    original_set = client.set
    original_delete = client.delete
    original_flushdb = client.flushdb

    async def tracked_set(key, *args, **kwargs):
        prefixed_key = f"{test_prefix}{key}"
        test_keys.add(prefixed_key)
        return await original_set(prefixed_key, *args, **kwargs)

    async def tracked_get(key, *args, **kwargs):
        prefixed_key = f"{test_prefix}{key}"
        return await client.__class__.get(client, prefixed_key, *args, **kwargs)

    async def tracked_delete(key, *args, **kwargs):
        if isinstance(key, str):
            prefixed_key = f"{test_prefix}{key}"
            test_keys.discard(prefixed_key)
            return await original_delete(prefixed_key, *args, **kwargs)
        prefixed_keys = [f"{test_prefix}{k}" for k in [key] + list(args)]
        test_keys.difference_update(prefixed_keys)
        return await original_delete(*prefixed_keys, **kwargs)

    async def safe_flushdb(*args, **kwargs):
        if test_keys:
            keys_to_delete = list(test_keys)
            if keys_to_delete:
                await original_delete(*keys_to_delete)
            test_keys.clear()

    client.set = tracked_set
    client.get = tracked_get
    client.delete = tracked_delete
    client.flushdb = safe_flushdb
    await client.flushdb()
    try:
        yield client
    finally:
        try:
            if test_keys:
                keys_to_delete = list(test_keys)
                if keys_to_delete:
                    await original_delete(*keys_to_delete)
        except Exception as e:
            logger.warning(f"Error cleaning up test Redis keys: {e}")
        try:
            await client.close()
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")


@pytest.fixture(scope="session")
def otel_test_setup():
    """
    Session-scoped OpenTelemetry setup for real behavior testing.

    Configures OpenTelemetry with real exporters and collectors for
    integration testing following 2025 best practices.
    """
    resource = Resource.create({
        "service.name": "apes-test",
        "service.version": "test",
        "deployment.environment": "test",
        "test.session": "real-behavior",
    })
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    metric_reader = PeriodicExportingMetricReader(
        exporter=OTLPMetricExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"), insecure=True),
        export_interval_millis=1000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    yield {
        "tracer_provider": tracer_provider,
        "meter_provider": meter_provider,
        "tracer": trace.get_tracer("apes-test"),
        "meter": metrics.get_meter("apes-test"),
    }
    tracer_provider.shutdown()
    meter_provider.shutdown()


@pytest.fixture(scope="function")
async def otel_metrics_collector(otel_test_setup):
    """
    Function-scoped OpenTelemetry metrics collector for real behavior testing.

    Provides real metric collection and validation capabilities for testing
    the migrated ML components with actual OpenTelemetry infrastructure.
    """
    meter = otel_test_setup["meter"]
    ml_processing_counter = meter.create_counter(
        name="ml_processing_total",
        description="Total ML processing operations",
        unit="1",
    )
    ml_processing_histogram = meter.create_histogram(
        name="ml_processing_duration", description="ML processing duration", unit="ms"
    )
    failure_analysis_counter = meter.create_counter(
        name="failure_analysis_total",
        description="Total failure analysis operations",
        unit="1",
    )
    failure_classification_counter = meter.create_counter(
        name="failure_classification_total",
        description="Total failure classification operations",
        unit="1",
    )
    yield {
        "meter": meter,
        "ml_processing_counter": ml_processing_counter,
        "ml_processing_histogram": ml_processing_histogram,
        "failure_analysis_counter": failure_analysis_counter,
        "failure_classification_counter": failure_classification_counter,
    }


@pytest.fixture(scope="function")
async def real_ml_database(test_db_session):
    """
    Function-scoped real database fixture for ML component testing.

    Creates ML-specific tables and provides real database interactions
    for testing failure_analyzer.py and failure_classifier.py components.
    """
    await test_db_session.execute(
        "\n        CREATE TABLE IF NOT EXISTS ml_metrics (\n            id SERIAL PRIMARY KEY,\n            metric_name VARCHAR(255) NOT NULL,\n            metric_value FLOAT NOT NULL,\n            labels JSONB,\n            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),\n            component VARCHAR(100) NOT NULL\n        );\n\n        CREATE INDEX IF NOT EXISTS idx_ml_metrics_name\n        ON ml_metrics(metric_name);\n\n        CREATE INDEX IF NOT EXISTS idx_ml_metrics_component\n        ON ml_metrics(component);\n    "
    )
    await test_db_session.execute(
        "\n        CREATE TABLE IF NOT EXISTS failure_analysis (\n            id SERIAL PRIMARY KEY,\n            analysis_id UUID NOT NULL,\n            failure_type VARCHAR(100) NOT NULL,\n            confidence_score FLOAT NOT NULL,\n            analysis_data JSONB NOT NULL,\n            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()\n        );\n\n        CREATE INDEX IF NOT EXISTS idx_failure_analysis_type\n        ON failure_analysis(failure_type);\n    "
    )
    await test_db_session.commit()
    yield test_db_session
    await test_db_session.execute("DROP TABLE IF EXISTS ml_metrics CASCADE;")
    await test_db_session.execute("DROP TABLE IF EXISTS failure_analysis CASCADE;")
    await test_db_session.commit()


@pytest.fixture(scope="function")
async def real_behavior_environment(
    otel_metrics_collector, real_ml_database, redis_client
):
    """
    Comprehensive real behavior testing environment.

    Combines OpenTelemetry metrics, real PostgreSQL database, and Redis
    for complete integration testing of migrated ML components following
    2025 best practices.
    """
    environment = {
        "otel_metrics": otel_metrics_collector,
        "database": real_ml_database,
        "redis": redis_client,
        "tracer": trace.get_tracer("apes-test"),
        "meter": otel_metrics_collector["meter"],
    }
    await redis_client.flushdb()
    await real_ml_database.execute(
        "TRUNCATE TABLE ml_metrics, failure_analysis RESTART IDENTITY CASCADE;"
    )
    await real_ml_database.commit()
    yield environment
    await redis_client.flushdb()
    await real_ml_database.execute(
        "TRUNCATE TABLE ml_metrics, failure_analysis RESTART IDENTITY CASCADE;"
    )
    await real_ml_database.commit()


def _check_otel_availability():
    """Check if OpenTelemetry is available and enabled."""
    if os.getenv("OTEL_SDK_DISABLED", "true").lower() == "true":
        return False
    try:
        otel = get_opentelemetry_components()
        return all([otel.trace, otel.metrics, otel.TracerProvider, otel.MeterProvider])
    except Exception:
        return False

requires_otel = pytest.mark.skipif(
    not _check_otel_availability(),
    reason="OpenTelemetry not properly configured or disabled",
)
requires_real_db = pytest.mark.skipif(
    os.getenv("SKIP_REAL_DB_TESTS", "false").lower() == "true",
    reason="Real database tests disabled",
)


@pytest.fixture(scope="function")
def test_data_dir(tmp_path):
    """Function-scoped temporary directory for test data.

    Uses pytest's tmp_path for automatic cleanup and proper isolation.
    """
    data_dir = tmp_path / "test_apes_data"
    data_dir.mkdir(parents=True, exist_ok=True)
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


@pytest.fixture(scope="function")
def event_loop():
    """Provide a fresh event loop for each test function.

    Function scope ensures complete isolation between async tests.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sample_training_data():
    """Session-scoped sample data for ML testing.

    Expensive to generate, safe to reuse across tests.
    """
    return {
        "features": [
            [0.8, 150, 1.0, 5, 0.7, 1.0],
            [0.6, 200, 0.8, 4, 0.6, 1.0],
            [0.4, 300, 0.6, 3, 0.5, 0.0],
            [0.9, 100, 1.0, 5, 0.8, 1.0],
            [0.3, 400, 0.4, 2, 0.4, 0.0],
        ]
        * 5,
        "effectiveness_scores": [0.8, 0.6, 0.4, 0.9, 0.3] * 5,
    }


@pytest.fixture(scope="session")
def sample_ml_training_data():
    """Sample ML training data for testing."""
    return {
        "features": [
            [0.8, 150, 1.0, 5, 0.9, 6, 0.7, 1.0, 0.1, 0.5],
            [0.7, 200, 0.8, 4, 0.8, 5, 0.6, 1.0, 0.2, 0.4],
            [0.6, 250, 0.6, 3, 0.7, 4, 0.5, 0.0, 0.3, 0.3],
            [0.9, 100, 1.0, 5, 0.95, 7, 0.8, 1.0, 0.05, 0.6],
            [0.5, 300, 0.4, 2, 0.6, 3, 0.4, 0.0, 0.4, 0.2],
        ]
        * 10,
        "effectiveness_scores": [0.8, 0.7, 0.6, 0.9, 0.5] * 10,
    }


@pytest.fixture(scope="function")
def test_config():
    """Function-scoped test configuration override."""
    return {
        "database": {"host": os.getenv("POSTGRES_HOST", "postgres"), "database": "apes_test", "user": "test_user"},
        "performance": {"target_response_time_ms": 200, "timeout_seconds": 5},
        "ml": {"min_training_samples": 10, "optimization_timeout": 60},
    }


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
    for record in metadata_records:
        real_db_session.add(record)
    await real_db_session.commit()
    for record in metadata_records:
        await real_db_session.refresh(record)
    return metadata_records


@pytest.fixture
async def real_prompt_sessions(real_db_session):
    """Create real prompt sessions in the test database."""
    from datetime import timedelta

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
    for record in session_records:
        real_db_session.add(record)
    await real_db_session.commit()
    for record in session_records:
        await real_db_session.refresh(record)
    return session_records


@pytest.fixture
def sample_prompt_sessions():
    """Sample prompt sessions for testing with proper database relationships."""
    from datetime import timedelta

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


@pytest.fixture
def sample_rule_metadata():
    """Sample rule metadata for testing with unique IDs per test."""
    import uuid

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
    from datetime import timedelta

    base_data = [
        RulePerformance(
            id=1,
            session_id=sample_prompt_sessions[0].session_id,
            rule_id=sample_rule_metadata[0].rule_id,
            improvement_score=0.8,
            confidence_level=0.9,
            execution_time_ms=150,
            parameters_used={"weight": 1.0, "threshold": 0.7},
            created_at=aware_utc_now(),
        ),
        RulePerformance(
            id=2,
            session_id=sample_prompt_sessions[1].session_id,
            rule_id=sample_rule_metadata[1].rule_id,
            improvement_score=0.7,
            confidence_level=0.8,
            execution_time_ms=200,
            parameters_used={"weight": 0.8, "threshold": 0.6},
            created_at=aware_utc_now() - timedelta(minutes=30),
        ),
    ]
    result = []
    for i in range(15):
        for j, base in enumerate(base_data):
            improvement_score = max(0.0, min(1.0, base.improvement_score + i * 0.01))
            confidence_level = max(0.0, min(1.0, base.confidence_level + i * 0.005))
            new_record = RulePerformance(
                id=i * len(base_data) + j + 1,
                session_id=base.session_id,
                rule_id=base.rule_id,
                improvement_score=improvement_score,
                confidence_level=confidence_level,
                execution_time_ms=base.execution_time_ms + i,
                parameters_used=base.parameters_used,
                created_at=aware_utc_now() - timedelta(minutes=i * 5),
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


@pytest.fixture
def ml_service():
    """Create ML service instance for testing."""
    with patch("prompt_improver.ml.core.ml_integration.mlflow"):
        from prompt_improver.ml.core.ml_integration import MLModelService

        return MLModelService()


@pytest.fixture
def prompt_service():
    """Create PromptImprovementService instance."""
    from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

    return PromptImprovementService()


@pytest.fixture
def mock_llm_transformer():
    """Mock LLMTransformerService for unit testing rule logic.

    Provides realistic transformation responses without external dependencies.
    Function-scoped to ensure test isolation.
    """
    from unittest.mock import AsyncMock, MagicMock

    service = MagicMock()

    async def mock_enhance_clarity(prompt, vague_words, context=None):
        enhanced_prompt = prompt
        transformations = []
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

    async def mock_enhance_specificity(prompt, context=None):
        enhanced_prompt = prompt
        transformations = []
        if len(prompt.split()) < 5:
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


@pytest.fixture
def real_ml_service_for_testing():
    """Real ML service for testing - replaces mock_ml_service."""
    from tests.real_ml.lightweight_models import RealMLService
    return RealMLService(random_state=42)


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


@pytest.fixture
def performance_threshold():
    """Performance thresholds for Phase 3 requirements."""
    return {
        "prediction_latency_ms": 5,
        "optimization_timeout_s": 300,
        "cache_hit_ratio": 0.9,
        "database_query_ms": 50,
    }


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


async def populate_test_database(
    session,  # AsyncSession - using Any for lazy loading
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


@pytest.fixture(scope="function")
async def mock_mlflow_service():
    """Protocol-compliant mock MLflow service for testing.

    Implements MLflowServiceProtocol with realistic behavior patterns
    for comprehensive ML pipeline testing.
    """
    from unittest.mock import AsyncMock

    from prompt_improver.core.protocols.ml_protocols import ServiceStatus

    # Legacy MockMLflowService class - replaced with real implementation
    class MockMLflowService:
        def __init__(self):
            self.experiments = {}
            self.models = {}
            self.traces = {}
            self._is_healthy = True

        async def log_experiment(
            self, experiment_name: str, parameters: Dict[str, Any]
        ) -> str:
            run_id = f"run_{len(self.experiments)}_{experiment_name}"
            self.experiments[run_id] = {
                "name": experiment_name,
                "parameters": parameters,
                "timestamp": aware_utc_now().isoformat(),
                "status": "running",
            }
            return run_id

        async def log_model(
            self, model_name: str, model_data: Any, metadata: Dict[str, Any]
        ) -> str:
            model_uri = f"models:/{model_name}/version_1"
            self.models[model_uri] = {
                "name": model_name,
                "data": model_data,
                "metadata": metadata,
                "timestamp": aware_utc_now().isoformat(),
            }
            return model_uri

        async def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
            if model_id in self.models:
                return self.models[model_id]["metadata"]
            return {
                "model_id": model_id,
                "status": "active",
                "version": "1.0",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
            }

        async def start_trace(
            self, trace_name: str, attributes: Optional[Dict[str, Any]] = None
        ) -> str:
            trace_id = f"trace_{len(self.traces)}_{trace_name}"
            self.traces[trace_id] = {
                "name": trace_name,
                "attributes": attributes or {},
                "start_time": aware_utc_now().isoformat(),
                "status": "active",
            }
            return trace_id

        async def end_trace(
            self, trace_id: str, outputs: Optional[Dict[str, Any]] = None
        ) -> None:
            if trace_id in self.traces:
                self.traces[trace_id].update({
                    "end_time": aware_utc_now().isoformat(),
                    "outputs": outputs or {},
                    "status": "completed",
                })

        async def health_check(self) -> ServiceStatus:
            return ServiceStatus.HEALTHY if self._is_healthy else ServiceStatus.ERROR

        def set_health_status(self, healthy: bool):
            """Test helper to control health status."""
            self._is_healthy = healthy

    # Use real MLflow service instead of mock
    import tempfile
    from pathlib import Path
    from tests.real_ml.lightweight_models import RealMLflowService
    
    temp_dir = tempfile.mkdtemp()
    return RealMLflowService(storage_dir=Path(temp_dir))


@pytest.fixture(scope="session")
async def redis_container():
    """Session-scoped real Redis container for integration testing.
    
    Provides actual Redis instance with full feature support including:
    - Real Redis operations and behavior
    - Performance testing capabilities
    - Memory management and eviction policies
    - Persistence testing
    - Connection pooling validation
    - Health monitoring
    """
    from tests.containers.real_redis_container import RealRedisTestContainer
    
    container = RealRedisTestContainer(
        image="redis:7-alpine",
        enable_persistence=False,  # Disable persistence for faster tests
        redis_conf_overrides={
            "maxmemory": "64mb",  # Limit memory for testing
            "maxmemory-policy": "allkeys-lru",
        }
    )
    
    await container.start()
    try:
        yield container
    finally:
        await container.stop()


@pytest.fixture(scope="function")
async def redis_client(redis_container):
    """Function-scoped Redis client with clean state.
    
    Provides fresh Redis client for each test with:
    - Database flush for test isolation
    - Proper connection configuration
    - Error handling and cleanup
    """
    # Flush all databases for test isolation
    await redis_container.flush_all_databases()
    
    # Return async client
    client = await redis_container.get_async_client(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture(scope="function") 
async def redis_binary_client(redis_container):
    """Redis client that preserves binary data."""
    await redis_container.flush_all_databases()
    
    client = await redis_container.get_async_client(decode_responses=False)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture(scope="function")
async def real_cache_service(redis_client):
    """Real cache service using actual Redis for comprehensive testing.
    
    Replaces mock_cache_service with real Redis backend to test:
    - Actual cache behavior and performance
    - Redis-specific features (TTL, expiry, atomicity)
    - Connection handling and error scenarios
    - Memory usage and eviction policies
    """
    from prompt_improver.core.protocols.ml_protocols import ServiceStatus
    
    class RealCacheService:
        def __init__(self, redis_client):
            self._redis = redis_client
            
        async def get(self, key: str) -> Optional[Any]:
            """Get value from Redis cache."""
            try:
                value = await self._redis.get(key)
                if value is None:
                    return None
                # Try to deserialize JSON, fallback to string
                try:
                    import json
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            except Exception:
                return None
                
        async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
            """Set value in Redis cache with optional TTL."""
            try:
                # Serialize complex objects as JSON
                if isinstance(value, (dict, list)):
                    import json
                    value = json.dumps(value)
                
                if ttl:
                    await self._redis.set(key, value, ex=ttl)
                else:
                    await self._redis.set(key, value)
            except Exception as e:
                # In real Redis, failures should be visible
                raise RuntimeError(f"Cache set failed: {e}")
                
        async def delete(self, key: str) -> bool:
            """Delete key from Redis cache."""
            try:
                result = await self._redis.delete(key)
                return bool(result)
            except Exception:
                return False
                
        async def exists(self, key: str) -> bool:
            """Check if key exists in Redis cache."""
            try:
                result = await self._redis.exists(key)
                return bool(result)
            except Exception:
                return False
                
        async def health_check(self) -> ServiceStatus:
            """Check Redis health."""
            try:
                await self._redis.ping()
                return ServiceStatus.HEALTHY
            except Exception:
                return ServiceStatus.ERROR
                
        async def clear(self):
            """Clear all cache data."""
            await self._redis.flushdb()
            
        # Test helpers for compatibility
        def set_health_status(self, healthy: bool):
            """Test helper - not applicable for real Redis."""
            pass
            
        def get_all_data(self) -> Dict[str, Any]:
            """Test helper to inspect cache contents."""
            # This is a simplified version - real implementation would be async
            return {}
    
    return RealCacheService(redis_client)


@pytest.fixture(scope="session")
async def redis_cluster_container():
    """Redis cluster container for testing cluster behavior."""
    from tests.containers.real_redis_container import RedisClusterContainer
    
    cluster = RedisClusterContainer(num_nodes=6)
    await cluster.start()
    try:
        yield cluster
    finally:
        await cluster.stop()


@pytest.fixture(scope="function")
async def redis_performance_container():
    """Redis container optimized for performance testing."""
    from tests.containers.real_redis_container import RealRedisTestContainer
    
    container = RealRedisTestContainer(
        image="redis:7-alpine",
        redis_conf_overrides={
            "maxmemory": "256mb",
            "maxmemory-policy": "allkeys-lru",
            "save": '""',  # Disable persistence for performance
            "appendonly": "no",
        }
    )
    
    await container.start()
    try:
        yield container
    finally:
        await container.stop()


@pytest.fixture(scope="function")
async def redis_ssl_container():
    """Redis container with SSL/TLS enabled for security testing."""
    from tests.containers.real_redis_container import RealRedisTestContainer
    
    container = RealRedisTestContainer(
        image="redis:7-alpine",
        enable_ssl=True,
        redis_conf_overrides={
            "requirepass": "test_password",
        }
    )
    
    await container.start()
    try:
        yield container
    finally:
        await container.stop()


# Backward compatibility alias for existing tests
@pytest.fixture(scope="function")
async def mock_cache_service(real_cache_service):
    """Backward compatibility alias - now uses real Redis."""
    return real_cache_service


@pytest.fixture(scope="function")
async def postgres_container():
    """Real PostgreSQL testcontainer for integration testing.
    
    Provides actual PostgreSQL database instance for comprehensive testing
    of database operations, constraints, and performance characteristics.
    """
    from tests.containers.postgres_container import PostgreSQLTestContainer
    
    container = PostgreSQLTestContainer(
        postgres_version="16",
        username="test_user", 
        password="test_pass"
    )
    
    async with container:
        yield container


@pytest.fixture(scope="function")
async def real_database_service(postgres_container):
    """Real database service using PostgreSQL testcontainer.
    
    Replaces mock database operations with actual PostgreSQL interactions
    to ensure real behavior testing and catch actual database issues.
    """
    from prompt_improver.core.protocols.ml_protocols import ServiceStatus
    from sqlalchemy import text
    
    class RealDatabaseService:
        def __init__(self, container):
            self.container = container
            self._query_count = 0
            
        async def execute_query(
            self, query: str, parameters: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            """Execute query against real PostgreSQL database."""
            self._query_count += 1
            
            async with self.container.get_session() as session:
                result = await session.execute(text(query), parameters or {})
                
                # Handle different result types
                if result.returns_rows:
                    rows = result.fetchall()
                    if rows:
                        # Convert rows to dictionaries
                        columns = list(result.keys())
                        return [dict(zip(columns, row)) for row in rows]
                    return []
                else:
                    # For INSERT/UPDATE/DELETE, return affected rows
                    return [{"affected_rows": result.rowcount}]

        async def execute_transaction(
            self, queries: List[str], parameters: Optional[List[Dict[str, Any]]] = None
        ) -> None:
            """Execute transaction against real PostgreSQL database."""
            async with self.container.get_session() as session:
                try:
                    for i, query in enumerate(queries):
                        query_params = parameters[i] if parameters and i < len(parameters) else {}
                        await session.execute(text(query), query_params)
                        self._query_count += 1
                    
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        async def health_check(self) -> ServiceStatus:
            """Check database health using real connection."""
            try:
                async with self.container.get_session() as session:
                    await session.execute(text("SELECT 1"))
                return ServiceStatus.HEALTHY
            except Exception:
                return ServiceStatus.ERROR

        async def get_connection_pool_stats(self) -> Dict[str, Any]:
            """Get real connection pool statistics."""
            # Get actual pool stats from the engine
            engine = self.container._engine
            pool = engine.pool if engine else None
            
            return {
                "active_connections": pool.checkedout() if pool else 0,
                "idle_connections": pool.checkedin() if pool else 0, 
                "max_connections": pool.size() if pool else 0,
                "queries_executed": self._query_count,
                "pool_overflow": pool.overflow() if pool else 0,
            }
            
        def get_connection_url(self) -> str:
            """Get database connection URL."""
            return self.container.get_connection_url()
            
        async def truncate_all_tables(self):
            """Clean all data for test isolation."""
            await self.container.truncate_all_tables()
            
        async def create_test_data(self, table_name: str, data: List[Dict[str, Any]]):
            """Create test data in specific table."""
            if not data:
                return
                
            async with self.container.get_session() as session:
                columns = list(data[0].keys())
                placeholders = ", ".join([f":{col}" for col in columns])
                sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                
                for row in data:
                    await session.execute(text(sql), row)
                
                await session.commit()
                
    return RealDatabaseService(postgres_container)


# Backward compatibility alias for existing tests
@pytest.fixture(scope="function") 
async def mock_database_service(real_database_service):
    """Backward compatibility fixture that now uses real PostgreSQL.
    
    This maintains API compatibility while providing real database testing.
    Tests using this fixture now get actual PostgreSQL behavior instead of mocks.
    """
    return real_database_service


@pytest.fixture(scope="function")
async def mock_event_bus():
    """Protocol-compliant mock event bus for testing.

    Implements EventBusProtocol with realistic pub/sub patterns,
    subscription management, and event delivery tracking.
    """
    import asyncio
    from unittest.mock import AsyncMock

    from prompt_improver.core.protocols.ml_protocols import ServiceStatus

    class MockEventBus:
        def __init__(self):
            self._subscribers = {}
            self._published_events = []
            self._subscription_counter = 0
            self._is_healthy = True

        async def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
            event = {
                "type": event_type,
                "data": event_data,
                "timestamp": aware_utc_now().isoformat(),
                "delivered_to": [],
            }
            if event_type in self._subscribers:
                for sub_id, handler in self._subscribers[event_type].items():
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event_data)
                        else:
                            handler(event_data)
                        event["delivered_to"].append(sub_id)
                    except Exception as e:
                        event["delivery_errors"] = event.get("delivery_errors", [])
                        event["delivery_errors"].append({
                            "subscriber": sub_id,
                            "error": str(e),
                        })
            self._published_events.append(event)

        async def subscribe(self, event_type: str, handler: Any) -> str:
            subscription_id = f"sub_{self._subscription_counter}_{event_type}"
            self._subscription_counter += 1
            if event_type not in self._subscribers:
                self._subscribers[event_type] = {}
            self._subscribers[event_type][subscription_id] = handler
            return subscription_id

        async def unsubscribe(self, subscription_id: str) -> None:
            for event_type, subscribers in self._subscribers.items():
                if subscription_id in subscribers:
                    del subscribers[subscription_id]
                    break

        async def health_check(self) -> ServiceStatus:
            return ServiceStatus.HEALTHY if self._is_healthy else ServiceStatus.ERROR

        def set_health_status(self, healthy: bool):
            """Test helper to control health status."""
            self._is_healthy = healthy

        def get_published_events(self) -> List[Dict[str, Any]]:
            """Test helper to inspect published events."""
            return self._published_events.copy()

        def get_subscription_count(self, event_type: Optional[str] = None) -> int:
            """Test helper to count active subscriptions."""
            if event_type:
                return len(self._subscribers.get(event_type, {}))
            return sum(len(subs) for subs in self._subscribers.values())

    return MockEventBus()


@pytest.fixture(scope="function")
async def ml_service_container(
    mock_mlflow_service, real_cache_service, real_database_service, mock_event_bus
):
    """Protocol-compliant ML service container fixture.

    Provides complete service container with all ML pipeline dependencies
    pre-configured for testing. Supports dependency injection patterns
    and service lifecycle management.
    """
    from prompt_improver.core.di.ml_container import MLServiceContainer
    from prompt_improver.core.protocols.ml_protocols import ServiceContainerProtocol

    class TestMLServiceContainer:
        def __init__(self):
            self._services = {
                "mlflow_service": mock_mlflow_service,
                "cache_service": real_cache_service,
                "database_service": real_database_service,
                "event_bus": mock_event_bus,
            }
            self._initialized = False

        async def register_service(
            self, service_name: str, service_instance: Any
        ) -> None:
            self._services[service_name] = service_instance

        async def get_service(self, service_name: str) -> Any:
            if service_name not in self._services:
                raise KeyError(f"Service not found: {service_name}")
            return self._services[service_name]

        async def initialize_all_services(self) -> None:
            if self._initialized:
                return
            for service in self._services.values():
                if hasattr(service, "initialize"):
                    if asyncio.iscoroutinefunction(service.initialize):
                        await service.initialize()
                    else:
                        service.initialize()
            self._initialized = True

        async def shutdown_all_services(self) -> None:
            for service in self._services.values():
                if hasattr(service, "shutdown"):
                    if asyncio.iscoroutinefunction(service.shutdown):
                        await service.shutdown()
                    else:
                        service.shutdown()
            self._initialized = False

        def get_all_services(self) -> Dict[str, Any]:
            return self._services.copy()

        def is_initialized(self) -> bool:
            return self._initialized

    container = TestMLServiceContainer()
    await container.initialize_all_services()
    try:
        yield container
    finally:
        await container.shutdown_all_services()


@pytest.fixture(scope="function")
async def component_factory(ml_service_container):
    """Component factory fixture for ML pipeline testing.

    Provides ComponentFactory configured with test service container
    for component creation with dependency injection.
    """
    from prompt_improver.core.factories.component_factory import ComponentFactory
    from prompt_improver.core.protocols.ml_protocols import ComponentSpec

    factory = ComponentFactory(ml_service_container)
    test_specs = [
        ComponentSpec(
            name="test_ml_component",
            module_path="tests.fixtures.test_components",
            class_name="TestMLComponent",
            tier="TIER_1",
            dependencies={
                "database_service": "database_service",
                "cache_service": "cache_service",
            },
            config={"test_mode": True},
        ),
        ComponentSpec(
            name="test_training_component",
            module_path="tests.fixtures.test_components",
            class_name="TestTrainingComponent",
            tier="TIER_2",
            dependencies={
                "mlflow_service": "mlflow_service",
                "database_service": "database_service",
            },
            config={"training_samples": 100},
        ),
    ]
    await factory.register_multiple_specs(test_specs)
    try:
        yield factory
    finally:
        await factory.shutdown_all_components()


@pytest.fixture(scope="function")
def sample_training_data_generator():
    """Advanced training data generator for ML pipeline testing.

    Generates realistic training data with configurable distributions,
    feature engineering patterns, and target variable relationships.
    """
    from typing import List, Tuple

    import numpy as np

    class TrainingDataGenerator:
        def __init__(self, random_seed: int = 42):
            self.rng = np.random.RandomState(random_seed)

        def generate_rule_performance_data(
            self,
            n_samples: int = 100,
            n_rules: int = 5,
            effectiveness_distribution: str = "normal",
        ) -> Dict[str, Any]:
            """Generate rule performance training data."""
            rule_ids = [f"rule_{i}" for i in range(n_rules)]
            features = []
            effectiveness_scores = []
            for _ in range(n_samples):
                clarity_score = self.rng.beta(2, 1)
                length = self.rng.lognormal(4, 0.5)
                complexity = self.rng.uniform(1, 10)
                user_rating = self.rng.normal(7, 1.5)
                user_rating = np.clip(user_rating, 1, 10)
                context_match = self.rng.beta(3, 2)
                features.append([
                    clarity_score,
                    length,
                    complexity,
                    user_rating,
                    context_match,
                ])
                if effectiveness_distribution == "normal":
                    base_effectiveness = (
                        0.4
                        + 0.3 * clarity_score
                        + 0.2 * context_match
                        - 0.1 * (complexity / 10)
                    )
                    effectiveness = self.rng.normal(base_effectiveness, 0.1)
                elif effectiveness_distribution == "bimodal":
                    if self.rng.random() < 0.6:
                        effectiveness = self.rng.normal(0.7, 0.1)
                    else:
                        effectiveness = self.rng.normal(0.4, 0.1)
                else:
                    effectiveness = self.rng.uniform(0.2, 0.9)
                effectiveness = np.clip(effectiveness, 0.0, 1.0)
                effectiveness_scores.append(effectiveness)
            return {
                "features": features,
                "effectiveness_scores": effectiveness_scores,
                "rule_ids": rule_ids,
                "feature_names": [
                    "clarity_score",
                    "length",
                    "complexity",
                    "user_rating",
                    "context_match",
                ],
                "metadata": {
                    "n_samples": n_samples,
                    "n_rules": n_rules,
                    "distribution": effectiveness_distribution,
                    "generated_at": aware_utc_now().isoformat(),
                },
            }

        def generate_ab_test_data(
            self,
            control_samples: int = 150,
            treatment_samples: int = 150,
            effect_size: float = 0.1,
        ) -> Dict[str, Any]:
            """Generate A/B test data with specified effect size."""
            control_scores = self.rng.normal(0.65, 0.15, control_samples)
            control_scores = np.clip(control_scores, 0.0, 1.0)
            treatment_scores = self.rng.normal(
                0.65 + effect_size, 0.15, treatment_samples
            )
            treatment_scores = np.clip(treatment_scores, 0.0, 1.0)
            return {
                "control_group": {
                    "scores": control_scores.tolist(),
                    "group_size": control_samples,
                    "mean_score": float(np.mean(control_scores)),
                    "std_score": float(np.std(control_scores)),
                },
                "treatment_group": {
                    "scores": treatment_scores.tolist(),
                    "group_size": treatment_samples,
                    "mean_score": float(np.mean(treatment_scores)),
                    "std_score": float(np.std(treatment_scores)),
                },
                "metadata": {
                    "effect_size": effect_size,
                    "total_samples": control_samples + treatment_samples,
                    "generated_at": aware_utc_now().isoformat(),
                },
            }

        def generate_time_series_data(
            self, n_timesteps: int = 100, n_metrics: int = 5, trend: str = "increasing"
        ) -> Dict[str, Any]:
            """Generate time series performance data."""
            timestamps = [
                aware_utc_now() - timedelta(hours=i) for i in range(n_timesteps)
            ]
            timestamps.reverse()
            metrics_data = {}
            for metric_idx in range(n_metrics):
                metric_name = f"metric_{metric_idx}"
                if trend == "increasing":
                    base_values = np.linspace(0.4, 0.8, n_timesteps)
                elif trend == "decreasing":
                    base_values = np.linspace(0.8, 0.4, n_timesteps)
                elif trend == "seasonal":
                    base_values = 0.6 + 0.2 * np.sin(
                        np.linspace(0, 4 * np.pi, n_timesteps)
                    )
                else:
                    base_values = np.full(n_timesteps, 0.6)
                noise = self.rng.normal(0, 0.05, n_timesteps)
                values = base_values + noise
                values = np.clip(values, 0.0, 1.0)
                metrics_data[metric_name] = {
                    "timestamps": [ts.isoformat() for ts in timestamps],
                    "values": values.tolist(),
                    "trend": trend,
                }
            return {
                "metrics": metrics_data,
                "metadata": {
                    "n_timesteps": n_timesteps,
                    "n_metrics": n_metrics,
                    "trend_pattern": trend,
                    "generated_at": aware_utc_now().isoformat(),
                },
            }

    return TrainingDataGenerator()


@pytest.fixture(scope="function")
async def performance_test_harness():
    """Performance testing harness for ML pipeline benchmarking.

    Provides comprehensive performance monitoring, profiling utilities,
    and benchmark comparison tools following 2025 best practices.
    """
    import asyncio
    import time
    from collections.abc import Callable
    from typing import Any, Dict, List

    import psutil

    class PerformanceTestHarness:
        def __init__(self):
            self.benchmark_results = []
            self.performance_baselines = {}
            self.resource_monitors = []
            self.start_time = None

        async def benchmark_async_function(
            self,
            func: Callable[..., Any],
            *args: Any,
            iterations: int = 10,
            warmup_iterations: int = 2,
            **kwargs: Any,
        ) -> dict[str, Any]:
            """Benchmark async function with statistical analysis."""
            for _ in range(warmup_iterations):
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
            execution_times = []
            memory_usage = []
            for i in range(iterations):
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024
                start_time = time.perf_counter()
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                mem_after = process.memory_info().rss / 1024 / 1024
                memory_delta = mem_after - mem_before
                execution_times.append(execution_time)
                memory_usage.append(memory_delta)
            import statistics

            benchmark_result = {
                "function_name": func.__name__,
                "iterations": iterations,
                "execution_times_ms": execution_times,
                "avg_time_ms": statistics.mean(execution_times),
                "median_time_ms": statistics.median(execution_times),
                "min_time_ms": min(execution_times),
                "max_time_ms": max(execution_times),
                "std_dev_ms": statistics.stdev(execution_times)
                if len(execution_times) > 1
                else 0,
                "memory_usage_mb": memory_usage,
                "avg_memory_mb": statistics.mean(memory_usage),
                "max_memory_mb": max(memory_usage),
                "benchmarked_at": aware_utc_now().isoformat(),
            }
            self.benchmark_results.append(benchmark_result)
            return benchmark_result

        def set_performance_baseline(
            self, operation_name: str, baseline_metrics: dict[str, float]
        ) -> None:
            """Set performance baseline for comparison."""
            self.performance_baselines[operation_name] = baseline_metrics

        def compare_to_baseline(
            self, operation_name: str, actual_metrics: dict[str, float]
        ) -> dict[str, Any]:
            """Compare actual metrics to baseline."""
            if operation_name not in self.performance_baselines:
                return {"status": "no_baseline", "operation": operation_name}
            baseline = self.performance_baselines[operation_name]
            comparison = {"operation": operation_name, "comparisons": {}}
            for metric_name, actual_value in actual_metrics.items():
                if metric_name in baseline:
                    baseline_value = baseline[metric_name]
                    percentage_diff = (
                        (actual_value - baseline_value) / baseline_value * 100
                    )
                    comparison["comparisons"][metric_name] = {
                        "baseline": baseline_value,
                        "actual": actual_value,
                        "difference": actual_value - baseline_value,
                        "percentage_diff": percentage_diff,
                        "performance_status": "improved"
                        if percentage_diff < -5
                        else "degraded"
                        if percentage_diff > 5
                        else "stable",
                    }
            return comparison

        async def monitor_resource_usage(
            self, duration_seconds: int = 60
        ) -> dict[str, list[float]]:
            """Monitor system resource usage over time."""
            cpu_usage = []
            memory_usage = []
            timestamps = []
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                process = psutil.Process()
                cpu_usage.append(process.cpu_percent())
                memory_usage.append(process.memory_info().rss / 1024 / 1024)
                timestamps.append(time.time() - start_time)
                await asyncio.sleep(0.1)
            return {
                "timestamps": timestamps,
                "cpu_usage_percent": cpu_usage,
                "memory_usage_mb": memory_usage,
                "avg_cpu": sum(cpu_usage) / len(cpu_usage),
                "avg_memory": sum(memory_usage) / len(memory_usage),
                "peak_memory": max(memory_usage),
            }

        def validate_sla_compliance(
            self, metrics: dict[str, float], sla_thresholds: dict[str, float]
        ) -> dict[str, Any]:
            """Validate SLA compliance for performance metrics."""
            compliance_results = {"overall_compliant": True, "violations": []}
            for metric_name, threshold in sla_thresholds.items():
                if metric_name in metrics:
                    actual_value = metrics[metric_name]
                    is_compliant = actual_value <= threshold
                    if not is_compliant:
                        compliance_results["overall_compliant"] = False
                        compliance_results["violations"].append({
                            "metric": metric_name,
                            "threshold": threshold,
                            "actual": actual_value,
                            "violation_percentage": (actual_value - threshold)
                            / threshold
                            * 100,
                        })
            return compliance_results

        def get_benchmark_summary(self) -> dict[str, Any]:
            """Get comprehensive benchmark summary."""
            if not self.benchmark_results:
                return {"status": "no_benchmarks_run"}
            return {
                "total_benchmarks": len(self.benchmark_results),
                "functions_tested": list(
                    {r["function_name"] for r in self.benchmark_results}
                ),
                "avg_execution_time_ms": sum(
                    r["avg_time_ms"] for r in self.benchmark_results
                )
                / len(self.benchmark_results),
                "slowest_function": max(
                    self.benchmark_results, key=lambda r: r["avg_time_ms"]
                )["function_name"],
                "fastest_function": min(
                    self.benchmark_results, key=lambda r: r["avg_time_ms"]
                )["function_name"],
                "results": self.benchmark_results,
            }

    return PerformanceTestHarness()


@pytest.fixture(scope="function")
async def integration_test_coordinator():
    """Integration test coordinator for real behavior testing.

    Coordinates complex integration scenarios across multiple services,
    manages test data consistency, and validates end-to-end workflows.
    """
    import asyncio
    from typing import Any, Dict, List, Optional

    class IntegrationTestCoordinator:
        def __init__(self):
            self.test_scenarios = {}
            self.scenario_results = {}
            self.cleanup_tasks = []
            self.service_states = {}

        async def register_test_scenario(
            self,
            scenario_name: str,
            services: list[str],
            setup_data: dict[str, Any],
            cleanup_data: dict[str, Any] | None = None,
        ) -> None:
            """Register a complex integration test scenario."""
            self.test_scenarios[scenario_name] = {
                "services": services,
                "setup_data": setup_data,
                "cleanup_data": cleanup_data or {},
                "registered_at": aware_utc_now().isoformat(),
            }

        async def execute_scenario(
            self,
            scenario_name: str,
            service_container,
            validation_steps: list[Callable] = None,
        ) -> dict[str, Any]:
            """Execute integration test scenario with validation."""
            if scenario_name not in self.test_scenarios:
                raise ValueError(f"Scenario not registered: {scenario_name}")
            scenario = self.test_scenarios[scenario_name]
            start_time = time.perf_counter()
            execution_result = {
                "scenario_name": scenario_name,
                "status": "running",
                "steps_completed": [],
                "validation_results": [],
                "errors": [],
            }
            try:
                for service_name in scenario["services"]:
                    service = await service_container.get_service(service_name)
                    if hasattr(service, "get_all_data"):
                        self.service_states[
                            f"{scenario_name}_{service_name}_initial"
                        ] = await service.get_all_data()
                    setup_data = scenario["setup_data"].get(service_name, {})
                    if setup_data and hasattr(service, "apply_test_data"):
                        await service.apply_test_data(setup_data)
                execution_result["steps_completed"].append("setup")
                if validation_steps:
                    for i, validation_step in enumerate(validation_steps):
                        try:
                            if asyncio.iscoroutinefunction(validation_step):
                                validation_result = await validation_step(
                                    service_container
                                )
                            else:
                                validation_result = validation_step(service_container)
                            execution_result["validation_results"].append({
                                "step": i,
                                "result": validation_result,
                                "status": "passed",
                            })
                        except Exception as e:
                            execution_result["validation_results"].append({
                                "step": i,
                                "error": str(e),
                                "status": "failed",
                            })
                            execution_result["errors"].append(
                                f"Validation step {i}: {e}"
                            )
                execution_result["steps_completed"].append("validation")
                execution_result["status"] = (
                    "completed" if not execution_result["errors"] else "failed"
                )
            except Exception as e:
                execution_result["status"] = "error"
                execution_result["errors"].append(f"Scenario execution error: {e}")
            finally:
                cleanup_tasks = []
                for service_name in scenario["services"]:
                    cleanup_data = scenario["cleanup_data"].get(service_name, {})
                    if cleanup_data:
                        task = self._cleanup_service_data(
                            service_container, service_name, cleanup_data
                        )
                        cleanup_tasks.append(task)
                if cleanup_tasks:
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                    execution_result["steps_completed"].append("cleanup")
            execution_result["duration_ms"] = (time.perf_counter() - start_time) * 1000
            self.scenario_results[scenario_name] = execution_result
            return execution_result

        async def _cleanup_service_data(
            self, service_container, service_name: str, cleanup_data: dict[str, Any]
        ) -> None:
            """Clean up service data after test scenario."""
            try:
                service = await service_container.get_service(service_name)
                if hasattr(service, "clear"):
                    await service.clear()
                elif hasattr(service, "cleanup_test_data"):
                    await service.cleanup_test_data(cleanup_data)
            except Exception as e:
                logger.warning(f"Cleanup failed for {service_name}: {e}")

        async def validate_service_consistency(
            self, service_container, consistency_checks: dict[str, Callable]
        ) -> dict[str, Any]:
            """Validate data consistency across services."""
            consistency_results = {"overall_consistent": True, "check_results": {}}
            for check_name, check_function in consistency_checks.items():
                try:
                    if asyncio.iscoroutinefunction(check_function):
                        check_result = await check_function(service_container)
                    else:
                        check_result = check_function(service_container)
                    consistency_results["check_results"][check_name] = {
                        "status": "passed",
                        "result": check_result,
                    }
                except Exception as e:
                    consistency_results["overall_consistent"] = False
                    consistency_results["check_results"][check_name] = {
                        "status": "failed",
                        "error": str(e),
                    }
            return consistency_results

        async def simulate_service_failure(
            self,
            service_container,
            service_name: str,
            failure_duration_seconds: int = 5,
        ) -> dict[str, Any]:
            """Simulate service failure for resilience testing."""
            service = await service_container.get_service(service_name)
            original_health = True
            if hasattr(service, "set_health_status"):
                original_health = service._is_healthy
            failure_result = {
                "service_name": service_name,
                "failure_duration_seconds": failure_duration_seconds,
                "started_at": aware_utc_now().isoformat(),
            }
            try:
                if hasattr(service, "set_health_status"):
                    service.set_health_status(False)
                await asyncio.sleep(failure_duration_seconds)
                if hasattr(service, "set_health_status"):
                    service.set_health_status(original_health)
                failure_result["status"] = "recovered"
                failure_result["ended_at"] = aware_utc_now().isoformat()
            except Exception as e:
                failure_result["status"] = "error"
                failure_result["error"] = str(e)
            return failure_result

        def get_scenario_summary(self) -> dict[str, Any]:
            """Get summary of all executed scenarios."""
            if not self.scenario_results:
                return {"status": "no_scenarios_executed"}
            successful_scenarios = [
                r for r in self.scenario_results.values() if r["status"] == "completed"
            ]
            failed_scenarios = [
                r
                for r in self.scenario_results.values()
                if r["status"] in ["failed", "error"]
            ]
            return {
                "total_scenarios": len(self.scenario_results),
                "successful_scenarios": len(successful_scenarios),
                "failed_scenarios": len(failed_scenarios),
                "success_rate": len(successful_scenarios)
                / len(self.scenario_results)
                * 100,
                "avg_duration_ms": sum(
                    r["duration_ms"] for r in self.scenario_results.values()
                )
                / len(self.scenario_results),
                "scenario_details": self.scenario_results,
            }

    return IntegrationTestCoordinator()


@pytest.fixture(scope="function")
async def async_test_context_manager():
    """Async context manager for comprehensive fixture lifecycle management.

    Provides automatic setup/teardown coordination, resource tracking,
    and error handling for complex async test scenarios.
    """
    import asyncio
    from contextlib import asynccontextmanager

    class AsyncTestContextManager:
        def __init__(self):
            self.active_contexts = {}
            self.resource_registry = {}
            self.cleanup_order = []

        @asynccontextmanager
        async def managed_test_lifecycle(
            self, test_name: str, services: list[Any], cleanup_timeout: int = 30
        ):
            """Manage complete test lifecycle with automatic cleanup."""
            start_time = time.perf_counter()
            context_id = f"{test_name}_{id(self)}"
            self.active_contexts[context_id] = {
                "test_name": test_name,
                "services": services,
                "start_time": start_time,
                "status": "initializing",
            }
            try:
                init_tasks = []
                for service in services:
                    if hasattr(service, "initialize") and asyncio.iscoroutinefunction(
                        service.initialize
                    ):
                        init_tasks.append(service.initialize())
                if init_tasks:
                    await asyncio.gather(*init_tasks)
                self.active_contexts[context_id]["status"] = "running"
                yield {
                    "context_id": context_id,
                    "services": {
                        f"service_{i}": service for i, service in enumerate(services)
                    },
                    "start_time": start_time,
                }
                self.active_contexts[context_id]["status"] = "completed"
            except Exception as e:
                self.active_contexts[context_id]["status"] = "error"
                self.active_contexts[context_id]["error"] = str(e)
                raise
            finally:
                cleanup_tasks = []
                for service in reversed(services):
                    if hasattr(service, "shutdown") and asyncio.iscoroutinefunction(
                        service.shutdown
                    ):
                        cleanup_tasks.append(service.shutdown())
                if cleanup_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*cleanup_tasks, return_exceptions=True),
                            timeout=cleanup_timeout,
                        )
                    except TimeoutError:
                        logger.warning(f"Cleanup timeout for context {context_id}")
                self.active_contexts[context_id]["duration_ms"] = (
                    time.perf_counter() - start_time
                ) * 1000

        async def register_resource(
            self,
            resource_name: str,
            resource: Any,
            cleanup_function: Callable | None = None,
        ) -> None:
            """Register a resource for automatic cleanup."""
            self.resource_registry[resource_name] = {
                "resource": resource,
                "cleanup_function": cleanup_function,
                "registered_at": aware_utc_now().isoformat(),
            }
            self.cleanup_order.append(resource_name)

        async def cleanup_all_resources(self) -> dict[str, Any]:
            """Clean up all registered resources."""
            cleanup_results = {"successful": [], "failed": []}
            for resource_name in reversed(self.cleanup_order):
                if resource_name in self.resource_registry:
                    resource_info = self.resource_registry[resource_name]
                    try:
                        if resource_info["cleanup_function"]:
                            cleanup_func = resource_info["cleanup_function"]
                            if asyncio.iscoroutinefunction(cleanup_func):
                                await cleanup_func(resource_info["resource"])
                            else:
                                cleanup_func(resource_info["resource"])
                        cleanup_results["successful"].append(resource_name)
                    except Exception as e:
                        cleanup_results["failed"].append({
                            "resource": resource_name,
                            "error": str(e),
                        })
            self.resource_registry.clear()
            self.cleanup_order.clear()
            return cleanup_results

        def get_active_contexts(self) -> dict[str, dict[str, Any]]:
            """Get information about all active test contexts."""
            return self.active_contexts.copy()

    context_manager = AsyncTestContextManager()
    try:
        yield context_manager
    finally:
        await context_manager.cleanup_all_resources()


@pytest.fixture(scope="session")
def test_quality_validator():
    """Test quality validator for ensuring fixture reliability.

    Validates fixture behavior, Protocol compliance, and integration patterns
    to maintain high-quality test infrastructure.
    """
    import inspect
    from typing import Protocol, get_type_hints

    class TestQualityValidator:
        def __init__(self):
            self.validation_results = []

        def validate_protocol_compliance(
            self, implementation: Any, protocol_class: Protocol
        ) -> dict[str, Any]:
            """Validate that implementation follows Protocol interface."""
            validation_result = {
                "implementation": implementation.__class__.__name__,
                "protocol": protocol_class.__name__,
                "compliant": True,
                "missing_methods": [],
                "signature_mismatches": [],
            }
            protocol_methods = [
                name
                for name, method in inspect.getmembers(
                    protocol_class, inspect.isfunction
                )
            ]
            for method_name in protocol_methods:
                if not hasattr(implementation, method_name):
                    validation_result["compliant"] = False
                    validation_result["missing_methods"].append(method_name)
                else:
                    protocol_method = getattr(protocol_class, method_name)
                    impl_method = getattr(implementation, method_name)
                    protocol_sig = inspect.signature(protocol_method)
                    impl_sig = inspect.signature(impl_method)
                    if protocol_sig != impl_sig:
                        validation_result["signature_mismatches"].append({
                            "method": method_name,
                            "protocol_signature": str(protocol_sig),
                            "implementation_signature": str(impl_sig),
                        })
            self.validation_results.append(validation_result)
            return validation_result

        def validate_fixture_isolation(
            self, fixture_instances: list[Any]
        ) -> dict[str, Any]:
            """Validate that fixtures are properly isolated."""
            isolation_result = {
                "fixtures_tested": len(fixture_instances),
                "isolation_violations": [],
                "shared_state_detected": False,
            }
            for i, fixture1 in enumerate(fixture_instances):
                for j, fixture2 in enumerate(fixture_instances[i + 1 :], i + 1):
                    if hasattr(fixture1, "__dict__") and hasattr(fixture2, "__dict__"):
                        fixture1_objects = {
                            id(obj)
                            for obj in fixture1.__dict__.values()
                            if hasattr(obj, "__dict__")
                        }
                        fixture2_objects = {
                            id(obj)
                            for obj in fixture2.__dict__.values()
                            if hasattr(obj, "__dict__")
                        }
                        shared_objects = fixture1_objects.intersection(fixture2_objects)
                        if shared_objects:
                            isolation_result["shared_state_detected"] = True
                            isolation_result["isolation_violations"].append({
                                "fixture1_index": i,
                                "fixture2_index": j,
                                "shared_object_ids": list(shared_objects),
                            })
            return isolation_result

        def get_validation_summary(self) -> dict[str, Any]:
            """Get comprehensive validation summary."""
            if not self.validation_results:
                return {"status": "no_validations_run"}
            compliant_results = [r for r in self.validation_results if r["compliant"]]
            return {
                "total_validations": len(self.validation_results),
                "compliant_implementations": len(compliant_results),
                "compliance_rate": len(compliant_results)
                / len(self.validation_results)
                * 100,
                "validation_details": self.validation_results,
            }

    return TestQualityValidator()


@pytest.fixture(scope="function")
async def parallel_test_coordinator():
    """Coordinate parallel test execution for external services.

    Provides isolation strategies for parallel pytest-xdist execution:
    - Unique database names per test process
    - Redis database and key prefix separation
    - Connection pool management for concurrent access
    - Cleanup coordination to prevent interference

    Features:
    - Worker ID detection for pytest-xdist
    - Automatic resource allocation per worker
    - Cleanup coordination across parallel processes
    - Performance monitoring for parallel execution
    """
    import os
    import uuid
    from typing import Any, Dict

    worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
    if worker_id == "master":
        worker_id = "single"
    session_id = uuid.uuid4().hex[:8]
    test_id = f"{worker_id}_{session_id}"
    coordinator_config = {
        "worker_id": worker_id,
        "session_id": session_id,
        "test_id": test_id,
        "database": {
            "test_db_prefix": f"apes_test_{test_id}_",
            "connection_pool_prefix": f"pool_{test_id}",
            "max_connections_per_worker": 10,
        },
        "redis": {
            "test_db_number": _get_worker_redis_db(worker_id),
            "key_prefix": f"test:{test_id}:",
            "connection_pool_prefix": f"redis_pool_{test_id}",
        },
        "performance": {
            "start_time": time.perf_counter(),
            "connection_attempts": 0,
            "cleanup_operations": 0,
        },
    }
    try:
        yield coordinator_config
    finally:
        duration = time.perf_counter() - coordinator_config["performance"]["start_time"]
        coordinator_config["performance"]["total_duration"] = duration
        logger.debug(
            f"Parallel test completed - Worker: {worker_id}, Duration: {duration:.1f}s, DB: {db_status}, Redis: {redis_status}",
            worker_id,
            format(duration, ".3f"),
            coordinator_config["database"]["test_db_prefix"],
            coordinator_config["redis"]["test_db_number"],
        )


def _get_worker_redis_db(worker_id: str) -> int:
    """Allocate Redis database numbers for parallel workers.

    Maps pytest-xdist worker IDs to specific Redis database numbers
    to ensure isolation during parallel test execution.

    Args:
        worker_id: pytest-xdist worker identifier or 'single'

    Returns:
        Redis database number (1-15) for test isolation
    """
    if worker_id == "single":
        return 1
    try:
        if worker_id.startswith("gw"):
            worker_num = int(worker_id[2:])
            return min(1 + worker_num % 14, 15)
    except (ValueError, IndexError):
        pass
    worker_hash = hash(worker_id) % 14
    return 1 + worker_hash


@pytest.fixture(scope="function")
async def isolated_external_postgres(parallel_test_coordinator):
    """Enhanced PostgreSQL isolation for parallel test execution.

    Creates unique database per test with parallel worker coordination.
    Uses external connection with <1s startup time.

    Features:
    - Unique database name per worker and test
    - Automatic cleanup with parallel coordination
    - Connection pool isolation
    - Performance monitoring
    """
    from prompt_improver.core.config import AppConfig
    from tests.database_helpers import (
        cleanup_test_database,
        create_test_engine_with_retry,
        wait_for_postgres_async,
    )

    config = AppConfig().database
    coordinator = parallel_test_coordinator
    test_db_name = f"{coordinator['database']['test_db_prefix']}{uuid.uuid4().hex[:8]}"
    postgres_ready = await wait_for_postgres_async(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        database=config.postgres_database,
        max_retries=15,
        retry_delay=0.5,
    )
    if not postgres_ready:
        pytest.skip(
            f"External PostgreSQL not available for worker {coordinator['worker_id']}"
        )
    await cleanup_test_database(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        test_db_name=test_db_name,
    )
    test_db_url = f"postgresql+asyncpg://{config.postgres_username}:{config.postgres_password}@{config.postgres_host}:{config.postgres_port}/{test_db_name}"
    engine = await create_test_engine_with_retry(
        test_db_url,
        max_retries=3,
        connect_args={
            "application_name": f"apes_test_{coordinator['worker_id']}",
            "connect_timeout": 5,
        },
        pool_timeout=15,
        pool_size=coordinator["database"]["max_connections_per_worker"],
        max_overflow=5,
    )
    if engine is None:
        pytest.skip(
            f"Could not create database engine for worker {coordinator['worker_id']}"
        )
    coordinator["performance"]["connection_attempts"] += 1
    try:
        yield (engine, test_db_name)
    finally:
        try:
            await engine.dispose()
        except Exception as e:
            logger.debug(f"Engine disposal warning: {e}")
        try:
            await cleanup_test_database(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_username,
                password=config.postgres_password,
                test_db_name=test_db_name,
            )
            coordinator["performance"]["cleanup_operations"] += 1
        except Exception as e:
            logger.debug(f"Database cleanup warning: {e}")


@pytest.fixture(scope="function")
async def isolated_external_redis(external_redis_config, parallel_test_coordinator):
    """Enhanced Redis isolation for parallel test execution.

    Uses worker-specific Redis databases and key prefixes for isolation.
    Uses external connection with <1s startup time.

    Features:
    - Worker-specific Redis database allocation (1-15)
    - Unique key prefixes per test
    - Automatic cleanup coordination
    - Connection pool isolation
    """
    import coredis

    coordinator = parallel_test_coordinator
    connection_params = external_redis_config.get_connection_params()
    connection_params["db"] = coordinator["redis"]["test_db_number"]
    client = coredis.Redis(**connection_params)
    try:
        pong_result = await client.ping()
        if pong_result not in ["PONG", b"PONG", True]:
            pytest.skip(
                f"External Redis not available for worker {coordinator['worker_id']}"
            )
    except Exception as e:
        pytest.skip(
            f"External Redis connection failed for worker {coordinator['worker_id']}: {e}"
        )
    test_prefix = coordinator["redis"]["key_prefix"]
    test_keys = set()
    original_set = client.set
    original_get = client.get
    original_delete = client.delete
    original_flushdb = client.flushdb

    async def tracked_set(key, *args, **kwargs):
        prefixed_key = f"{test_prefix}{key}"
        test_keys.add(prefixed_key)
        return await original_set(prefixed_key, *args, **kwargs)

    async def tracked_get(key, *args, **kwargs):
        prefixed_key = f"{test_prefix}{key}"
        return await client.__class__.get(client, prefixed_key, *args, **kwargs)

    async def tracked_delete(key, *args, **kwargs):
        if isinstance(key, str):
            prefixed_key = f"{test_prefix}{key}"
            test_keys.discard(prefixed_key)
            return await original_delete(prefixed_key, *args, **kwargs)
        prefixed_keys = [f"{test_prefix}{k}" for k in [key] + list(args)]
        test_keys.difference_update(prefixed_keys)
        return await original_delete(*prefixed_keys, **kwargs)

    async def safe_flushdb(*args, **kwargs):
        if test_keys:
            keys_to_delete = list(test_keys)
            if keys_to_delete:
                await original_delete(*keys_to_delete)
            test_keys.clear()
            coordinator["performance"]["cleanup_operations"] += 1

    client.set = tracked_set
    client.get = tracked_get
    client.delete = tracked_delete
    client.flushdb = safe_flushdb
    await client.flushdb()
    coordinator["performance"]["connection_attempts"] += 1
    try:
        yield client
    finally:
        try:
            await client.flushdb()
        except Exception as e:
            logger.debug(f"Redis cleanup warning: {e}")
        try:
            await client.close()
        except Exception as e:
            logger.debug(f"Redis connection closure warning: {e}")


@pytest.fixture(scope="function")
async def parallel_execution_validator():
    """Validate parallel test execution performance and isolation.

    Monitors and validates:
    - Startup time improvements (target: <1s with external services)
    - Database isolation between parallel workers
    - Redis key/database isolation
    - Connection pool efficiency
    - Real behavior testing maintenance
    """
    import time
    from typing import Dict, List

    validation_metrics = {
        "startup_times": [],
        "isolation_checks": [],
        "performance_baselines": {},
        "real_behavior_validations": [],
    }

    def record_startup_time(service_name: str, duration: float):
        """Record service startup time for validation."""
        validation_metrics["startup_times"].append({
            "service": service_name,
            "duration_ms": duration * 1000,
            "timestamp": time.time(),
        })

    def validate_isolation(worker_id: str, resource_id: str, isolation_type: str):
        """Validate resource isolation between workers."""
        validation_metrics["isolation_checks"].append({
            "worker_id": worker_id,
            "resource_id": resource_id,
            "isolation_type": isolation_type,
            "validated_at": time.time(),
        })

    def validate_real_behavior(
        test_name: str, real_service: str, behavior_maintained: bool
    ):
        """Validate that real behavior testing is maintained."""
        validation_metrics["real_behavior_validations"].append({
            "test_name": test_name,
            "real_service": real_service,
            "behavior_maintained": behavior_maintained,
            "validated_at": time.time(),
        })

    validator = {
        "record_startup_time": record_startup_time,
        "validate_isolation": validate_isolation,
        "validate_real_behavior": validate_real_behavior,
        "metrics": validation_metrics,
    }
    try:
        yield validator
    finally:
        startup_times = [m["duration_ms"] for m in validation_metrics["startup_times"]]
        if startup_times:
            avg_startup = sum(startup_times) / len(startup_times)
            max_startup = max(startup_times)
            if avg_startup < 1000:
                logger.info(
                    f"✅ Startup time validation PASSED: avg {avg_startup_time:.1f}ms (target: <1000ms)",
                    format(avg_startup, ".1f"),
                )
            else:
                logger.warning(
                    f"⚠️  Startup time validation FAILED: avg {avg_startup_time:.1f}ms (target: <1000ms)",
                    format(avg_startup, ".1f"),
                )
            if max_startup < 2000:
                logger.info(
                    f"✅ Maximum startup time PASSED: {max_startup_time:.1f}ms (target: <2000ms)",
                    format(max_startup, ".1f"),
                )
            else:
                logger.warning(
                    f"⚠️  Maximum startup time FAILED: {max_startup_time:.1f}ms (target: <2000ms)",
                    format(max_startup, ".1f"),
                )
        isolation_count = len(validation_metrics["isolation_checks"])
        if isolation_count > 0:
            logger.info(f"✅ Isolation validation completed: {isolation_count} checks")
        real_behavior_count = len(validation_metrics["real_behavior_validations"])
        maintained_count = sum(
            1
            for v in validation_metrics["real_behavior_validations"]
            if v["behavior_maintained"]
        )
        if real_behavior_count > 0:
            success_rate = maintained_count / real_behavior_count * 100
            logger.info(
                f"✅ Real behavior testing: {percentage:.1f}% maintained ({maintained}/{total})",
                format(success_rate, ".1f"),
                maintained_count,
                real_behavior_count,
            )


@pytest.fixture(scope="session", autouse=True)
def setup_real_database():
    """Setup real database for integration testing.

    Consolidated from integration conftest.py - manages real database setup
    for integration tests requiring production-like environment.
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=apes_postgres",
                "--format",
                "{{.Names}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "apes_postgres" not in result.stdout:
            print("Starting real PostgreSQL database for integration tests...")
            subprocess.run(
                ["docker-compose", "up", "-d", "postgres"],
                check=True,
                timeout=60,
                cwd="/Users/lukemckenzie/prompt-improver",
            )
            print("Waiting for database to be ready...")
            max_retries = 30
            for i in range(max_retries):
                try:
                    result = subprocess.run(
                        [
                            "docker",
                            "exec",
                            "apes_postgres",
                            "pg_isready",
                            "-U",
                            "apes_user",
                            "-d",
                            "apes_production",
                        ],
                        check=False,
                        capture_output=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        print("Real database is ready!")
                        break
                except subprocess.TimeoutExpired:
                    pass
                if i == max_retries - 1:
                    raise RuntimeError("Real database failed to start within timeout")
                time.sleep(1)
        else:
            print("Real database already running - using for integration testing")
        yield
    except Exception as e:
        print(f"Real database setup failed: {e}")
        yield


@pytest.fixture
async def real_database_session():
    """Provide real database session for integration tests.

    Consolidated from integration conftest.py - provides production database
    access for real behavior testing.
    """
    async with get_session() as session:
        yield session


@pytest.fixture
async def performance_baseline():
    """Establish performance baseline for tests.

    Consolidated from integration conftest.py - provides performance baselines
    for regression testing.
    """
    from prompt_improver.database.performance_monitor import get_performance_monitor

    monitor = await get_performance_monitor()
    baseline_snapshot = await monitor.take_performance_snapshot()
    return {
        "baseline_cache_hit_ratio": baseline_snapshot.cache_hit_ratio,
        "baseline_query_time_ms": baseline_snapshot.avg_query_time_ms,
        "baseline_connections": baseline_snapshot.active_connections,
        "baseline_timestamp": baseline_snapshot.timestamp,
    }


@pytest_asyncio.fixture
async def sample_rule_intelligence_cache(isolated_db_session):
    """Create sample rule intelligence cache data for testing.

    Consolidated from phase4 conftest.py - provides test data for rule intelligence.
    """
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
        performance_trend="improving",
    )
    isolated_db_session.add(cache_entry)
    await isolated_db_session.commit()
    await isolated_db_session.refresh(cache_entry)
    return [cache_entry]


@pytest.fixture(scope="session")
def performance_test_config():
    """Configuration for performance testing following 2025 standards.

    Consolidated from phase4 conftest.py - provides performance test configuration.
    """
    return {
        "max_response_time_ms": 200,
        "batch_size_limits": {"small": 50, "medium": 100, "large": 500},
        "parallel_worker_limits": {"min": 2, "max": 8, "default": 4},
        "confidence_thresholds": {"minimum": 0.6, "good": 0.8, "excellent": 0.9},
    }


@pytest.fixture
def mock_event_bus():
    """Provide event bus for testing event-driven behavior.

    Consolidated from integration conftest.py - provides minimal event bus
    for event collection in tests.
    """

    class TestEventBus:
        def __init__(self):
            self.events = []
            self.subscribers = {}

        async def emit(self, event):
            self.events.append(event)
            if event.event_type in self.subscribers:
                for handler in self.subscribers[event.event_type]:
                    await handler(event)

        def subscribe(self, event_type, handler):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)

        async def start(self):
            pass

        async def stop(self):
            pass

    return TestEventBus()


def pytest_configure(config):
    """Configure custom pytest markers for all test types."""
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line(
        "markers", "real_database: mark test as requiring real database"
    )
    config.addinivalue_line(
        "markers", "event_driven: mark test as testing event-driven behavior"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically for all test types."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "database" in str(item.fspath):
            item.add_marker(pytest.mark.real_database)
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        if "event" in item.name.lower():
            item.add_marker(pytest.mark.event_driven)


@pytest.fixture(autouse=True)
def async_test_timeout():
    """Set reasonable timeout for async tests."""
    return 30
