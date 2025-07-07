"""
Centralized pytest configuration and shared fixtures for APES system testing.
Provides comprehensive fixture infrastructure following pytest-asyncio best practices.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from prompt_improver.database.models import (
    ImprovementSession,
    RuleMetadata,
    RulePerformance,
)


# CLI Testing Infrastructure
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
def mock_db_session():
    """Mock database session with proper async patterns.

    Function-scoped to ensure test isolation and prevent state leakage.
    """
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = AsyncMock()

    # Configure common query patterns
    session.scalar_one_or_none = AsyncMock()
    session.fetchall = AsyncMock()
    session.refresh = AsyncMock()

    return session


@pytest.fixture
async def test_db_engine():
    """Create test database engine using existing PostgreSQL configuration with retry logic."""
    from prompt_improver.database.config import DatabaseConfig
    from tests.database_helpers import (
        wait_for_postgres_async,
        ensure_test_database_exists,
        create_test_engine_with_retry
    )

    # Use existing database config but with test database name
    config = DatabaseConfig()
    
    # First, wait for PostgreSQL to be ready
    postgres_ready = await wait_for_postgres_async(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        database="postgres",  # Connect to default database first
        max_retries=30,
        retry_delay=1.0
    )
    
    if not postgres_ready:
        pytest.skip("PostgreSQL server not available after 30 attempts")
    
    # Ensure test database exists
    db_exists = await ensure_test_database_exists(
        host=config.postgres_host,
        port=config.postgres_port,
        user=config.postgres_username,
        password=config.postgres_password,
        test_db_name="apes_test"
    )
    
    if not db_exists:
        pytest.skip("Could not create test database")
    
    # Now create the engine with retry logic
    test_db_url = f"postgresql+asyncpg://{config.postgres_username}:{config.postgres_password}@{config.postgres_host}:{config.postgres_port}/apes_test"
    
    engine = await create_test_engine_with_retry(
        test_db_url,
        max_retries=3,
        connect_args={
            "server_settings": {
                "application_name": "apes_test_suite",
            },
            "command_timeout": 10,  # Increased timeout
        },
        pool_timeout=10,  # Increased timeout
    )
    
    if engine is None:
        pytest.skip("Could not create database engine")

    try:
        yield engine
    finally:
        # Cleanup
        try:
            await engine.dispose()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
async def test_db_session(test_db_engine):
    """Create test database session with transaction rollback for isolation."""
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
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


# Sample Model Data Fixtures
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
            rule_category="core",
            rule_description="Improves prompt clarity",
            enabled=True,
            priority=5,
            rule_version="1.0",
            default_parameters={"weight": 1.0, "threshold": 0.7},
        ),
        RuleMetadata(
            rule_id=f"specificity_rule_{test_suffix}",
            rule_name="Specificity Enhancement Rule",
            rule_category="core",
            rule_description="Improves prompt specificity",
            enabled=True,
            priority=4,
            rule_version="1.0",
            default_parameters={"weight": 0.8, "threshold": 0.6},
        ),
    ]


@pytest.fixture
def sample_rule_performance(sample_rule_metadata):
    """Sample rule performance data for testing with unique rule IDs."""
    import uuid

    # Use the same unique rule IDs from sample_rule_metadata
    base_data = [
        RulePerformance(
            rule_id=sample_rule_metadata[0].rule_id,  # clarity_rule with unique suffix
            rule_name="Clarity Enhancement Rule",
            improvement_score=0.8,
            confidence_level=0.9,
            execution_time_ms=150,
            prompt_characteristics={"length": 50, "complexity": 0.6},
            before_metrics={"clarity": 0.5, "specificity": 0.6},
            after_metrics={"clarity": 0.8, "specificity": 0.6},
        ),
        RulePerformance(
            rule_id=sample_rule_metadata[1].rule_id,  # specificity_rule with unique suffix
            rule_name="Specificity Enhancement Rule",
            improvement_score=0.7,
            confidence_level=0.8,
            execution_time_ms=200,
            prompt_characteristics={"length": 45, "complexity": 0.5},
            before_metrics={"clarity": 0.7, "specificity": 0.4},
            after_metrics={"clarity": 0.7, "specificity": 0.7},
        ),
    ]

    # Create multiple records with unique prompt_ids
    result = []
    for i in range(15):  # 15 records each = 30 total
        for base in base_data:
            # Create new instance with unique prompt_id
            new_record = RulePerformance(
                rule_id=base.rule_id,
                rule_name=base.rule_name,
                prompt_id=uuid.uuid4(),  # Unique prompt_id for each record
                improvement_score=base.improvement_score + (i * 0.01),  # Slight variation
                confidence_level=base.confidence_level,
                execution_time_ms=base.execution_time_ms + i,
                prompt_characteristics=base.prompt_characteristics,
                before_metrics=base.before_metrics,
                after_metrics=base.after_metrics,
            )
            result.append(new_record)

    return result


@pytest.fixture
def sample_user_feedback():
    """Sample user feedback for testing."""
    return [
        UserFeedback(
            original_prompt="Make this better",
            improved_prompt="Please improve the clarity and specificity of this document",
            user_rating=4,
            applied_rules={"rules": ["clarity_rule"]},
            improvement_areas={"areas": ["clarity", "specificity"]},
            user_notes="Good improvement",
            session_id="test_session_1",
            ml_optimized=False,
            model_id=None,
            created_at=datetime.utcnow(),
        ),
        UserFeedback(
            original_prompt="Help me with this task",
            improved_prompt="Please provide step-by-step guidance for completing this specific task",
            user_rating=5,
            applied_rules={"rules": ["clarity_rule", "specificity_rule"]},
            improvement_areas={"areas": ["clarity"]},
            user_notes="Excellent improvement",
            session_id="test_session_2",
            ml_optimized=True,
            model_id="model_123",
            created_at=datetime.utcnow() - timedelta(hours=2),
        ),
    ]


@pytest.fixture
def sample_improvement_sessions():
    """Sample improvement sessions for testing."""
    return [
        ImprovementSession(
            session_id="test_session_1",
            original_prompt="Make this better",
            final_prompt="Please improve the clarity and specificity of this document",
            rules_applied=[{"rule_id": "clarity_rule", "confidence": 0.9}],
            iteration_count=1,
            session_metadata={"user_context": "document_improvement"},
            status="completed",
            created_at=datetime.utcnow(),
        ),
        ImprovementSession(
            session_id="test_session_2",
            original_prompt="Help me with this task",
            final_prompt="Please provide step-by-step guidance for completing this specific task",
            rules_applied=[
                {"rule_id": "clarity_rule", "confidence": 0.8},
                {"rule_id": "specificity_rule", "confidence": 0.9},
            ],
            iteration_count=2,
            session_metadata={"user_context": "task_guidance"},
            status="completed",
            created_at=datetime.utcnow() - timedelta(hours=1),
        ),
    ]


# Service Instance Fixtures
@pytest.fixture
def ml_service():
    """Create ML service instance for testing."""
    with patch("prompt_improver.services.ml_integration.mlflow"):
        from prompt_improver.services.ml_integration import MLModelService

        return MLModelService()


@pytest.fixture
def prompt_service():
    """Create PromptImprovementService instance."""
    from prompt_improver.services.prompt_improvement import PromptImprovementService

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
            rule_category="core",
            rule_description="Improves prompt clarity by replacing vague terms",
            enabled=True,
            priority=5,
            rule_version="1.0",
            default_parameters={"vague_threshold": 0.7, "confidence_weight": 1.0},
        ),
        RuleMetadata(
            rule_id="specificity_rule",
            rule_name="Specificity Enhancement Rule",
            rule_category="core",
            rule_description="Improves prompt specificity by adding constraints and examples",
            enabled=True,
            priority=4,
            rule_version="1.0",
            default_parameters={"min_length": 10, "add_format": True},
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


@pytest.fixture
def mock_analytics_service():
    """Mock analytics service for testing."""
    service = MagicMock()
    service.get_performance_summary = AsyncMock(
        return_value={
            "total_sessions": 100,
            "avg_improvement": 0.75,
            "success_rate": 0.95,
        }
    )
    service.get_rule_effectiveness = AsyncMock(
        return_value={"clarity_rule": 0.85, "specificity_rule": 0.78}
    )
    service.get_ml_performance_summary = AsyncMock(
        return_value={
            "total_models": 5,
            "avg_performance": 0.82,
            "last_training": datetime.utcnow() - timedelta(hours=6),
        }
    )
    return service


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
