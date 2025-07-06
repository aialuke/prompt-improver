"""
Pytest configuration and shared fixtures for Phase 3 testing.
Provides database fixtures, mock services, and testing utilities.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
import tempfile
import os
from datetime import datetime, timedelta

from prompt_improver.database.models import (
    RuleMetadata, RulePerformance, UserFeedback, ImprovementSession,
    MLModelPerformance, ABExperiment, DiscoveredPattern
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db_engine():
    """Create test database engine with in-memory SQLite."""
    # Use in-memory SQLite for fast testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_db_session(test_db_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def mock_db_session():
    """Mock database session for unit tests."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def sample_rule_metadata():
    """Sample rule metadata for testing."""
    return [
        RuleMetadata(
            rule_id="clarity_rule",
            rule_name="Clarity Enhancement Rule",
            rule_category="core",
            rule_description="Improves prompt clarity",
            enabled=True,
            priority=5,
            rule_version="1.0",
            parameters={"weight": 1.0, "threshold": 0.7},
            effectiveness_score=0.85,
            weight=1.0,
            active=True,
            updated_by="system",
            updated_at=datetime.utcnow()
        ),
        RuleMetadata(
            rule_id="specificity_rule",
            rule_name="Specificity Enhancement Rule", 
            rule_category="core",
            rule_description="Improves prompt specificity",
            enabled=True,
            priority=4,
            rule_version="1.0",
            parameters={"weight": 0.8, "threshold": 0.6},
            effectiveness_score=0.78,
            weight=0.8,
            active=True,
            updated_by="system",
            updated_at=datetime.utcnow()
        )
    ]


@pytest.fixture
def sample_rule_performance():
    """Sample rule performance data for testing."""
    return [
        RulePerformance(
            rule_id="clarity_rule",
            rule_name="Clarity Enhancement Rule",
            improvement_score=0.8,
            confidence_level=0.9,
            execution_time_ms=150,
            prompt_characteristics={"length": 50, "complexity": 0.6},
            before_metrics={"clarity": 0.5, "specificity": 0.6},
            after_metrics={"clarity": 0.8, "specificity": 0.6},
            user_satisfaction_score=0.9,
            session_id="test_session_1",
            created_at=datetime.utcnow()
        ),
        RulePerformance(
            rule_id="specificity_rule",
            rule_name="Specificity Enhancement Rule",
            improvement_score=0.7,
            confidence_level=0.8,
            execution_time_ms=200,
            prompt_characteristics={"length": 45, "complexity": 0.5},
            before_metrics={"clarity": 0.7, "specificity": 0.4},
            after_metrics={"clarity": 0.7, "specificity": 0.7},
            user_satisfaction_score=0.8,
            session_id="test_session_2",
            created_at=datetime.utcnow() - timedelta(hours=1)
        )
    ] * 15  # Create 30 total records for sufficient test data


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
            created_at=datetime.utcnow()
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
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
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
            created_at=datetime.utcnow()
        ),
        ImprovementSession(
            session_id="test_session_2", 
            original_prompt="Help me with this task",
            final_prompt="Please provide step-by-step guidance for completing this specific task",
            rules_applied=[
                {"rule_id": "clarity_rule", "confidence": 0.8},
                {"rule_id": "specificity_rule", "confidence": 0.9}
            ],
            iteration_count=2,
            session_metadata={"user_context": "task_guidance"},
            status="completed",
            created_at=datetime.utcnow() - timedelta(hours=1)
        )
    ]


@pytest.fixture
def sample_ml_training_data():
    """Sample ML training data for testing."""
    return {
        "features": [
            [0.8, 150, 1.0, 5, 0.9, 6, 0.7, 1.0, 0.1, 0.5],  # High performance
            [0.7, 200, 0.8, 4, 0.8, 5, 0.6, 1.0, 0.2, 0.4],  # Medium performance
            [0.6, 250, 0.6, 3, 0.7, 4, 0.5, 0.0, 0.3, 0.3],  # Lower performance
            [0.9, 100, 1.0, 5, 0.95, 7, 0.8, 1.0, 0.05, 0.6], # Best performance
            [0.5, 300, 0.4, 2, 0.6, 3, 0.4, 0.0, 0.4, 0.2],  # Poor performance
        ] * 10,  # 50 samples total
        "effectiveness_scores": [0.8, 0.7, 0.6, 0.9, 0.5] * 10
    }


@pytest.fixture
def mock_ml_service():
    """Mock ML service for testing."""
    service = MagicMock()
    service.optimize_rules = AsyncMock(return_value={
        "status": "success",
        "model_id": "test_model_123",
        "best_score": 0.85,
        "accuracy": 0.90,
        "precision": 0.88,
        "recall": 0.92,
        "processing_time_ms": 1500
    })
    service.predict_rule_effectiveness = AsyncMock(return_value={
        "status": "success",
        "prediction": 0.8,
        "confidence": 0.9,
        "probabilities": [0.1, 0.9],
        "processing_time_ms": 2
    })
    service.optimize_ensemble_rules = AsyncMock(return_value={
        "status": "success",
        "ensemble_score": 0.88,
        "ensemble_std": 0.05,
        "processing_time_ms": 3000
    })
    service.discover_patterns = AsyncMock(return_value={
        "status": "success",
        "patterns_discovered": 3,
        "patterns": [
            {"parameters": {"weight": 1.0}, "avg_effectiveness": 0.85, "support_count": 10},
            {"parameters": {"weight": 0.9}, "avg_effectiveness": 0.82, "support_count": 8},
            {"parameters": {"weight": 0.8}, "avg_effectiveness": 0.79, "support_count": 7}
        ],
        "total_analyzed": 100,
        "processing_time_ms": 1200
    })
    return service


@pytest.fixture
def mock_prompt_service():
    """Mock prompt improvement service for testing."""
    service = MagicMock()
    service.improve_prompt = AsyncMock(return_value={
        "original_prompt": "Test prompt",
        "improved_prompt": "Enhanced test prompt with better clarity and specificity",
        "applied_rules": [{"rule_id": "clarity_rule", "confidence": 0.9}],
        "processing_time_ms": 100,
        "session_id": "test_session_123"
    })
    service.trigger_optimization = AsyncMock(return_value={
        "status": "success",
        "performance_score": 0.85,
        "training_samples": 25
    })
    service.run_ml_optimization = AsyncMock(return_value={
        "status": "success",
        "best_score": 0.88,
        "model_id": "optimized_model_456"
    })
    service.discover_patterns = AsyncMock(return_value={
        "status": "success",
        "patterns_discovered": 2
    })
    return service


@pytest.fixture
def mock_analytics_service():
    """Mock analytics service for testing."""
    service = MagicMock()
    service.get_performance_summary = AsyncMock(return_value={
        "total_sessions": 100,
        "avg_improvement": 0.75,
        "success_rate": 0.95
    })
    service.get_rule_effectiveness = AsyncMock(return_value={
        "clarity_rule": 0.85,
        "specificity_rule": 0.78
    })
    service.get_ml_performance_summary = AsyncMock(return_value={
        "total_models": 5,
        "avg_performance": 0.82,
        "last_training": datetime.utcnow() - timedelta(hours=6)
    })
    return service


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing ML operations."""
    with patch('mlflow.start_run') as mock_start, \
         patch('mlflow.log_params') as mock_log_params, \
         patch('mlflow.log_metrics') as mock_log_metrics, \
         patch('mlflow.sklearn.log_model') as mock_log_model, \
         patch('mlflow.active_run') as mock_active_run:
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run
        
        yield {
            'start_run': mock_start,
            'log_params': mock_log_params,
            'log_metrics': mock_log_metrics,
            'log_model': mock_log_model,
            'active_run': mock_active_run
        }


@pytest.fixture
def mock_optuna():
    """Mock Optuna for testing hyperparameter optimization."""
    with patch('optuna.create_study') as mock_create_study:
        mock_study = MagicMock()
        mock_study.best_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5
        }
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study
        
        yield {
            'create_study': mock_create_study,
            'study': mock_study
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


# Performance testing utilities
@pytest.fixture
def performance_threshold():
    """Performance thresholds for Phase 3 requirements."""
    return {
        "prediction_latency_ms": 5,  # <5ms for predictions
        "optimization_timeout_s": 300,  # 5 minute timeout for optimization
        "cache_hit_ratio": 0.9,  # >90% cache hit ratio target
        "database_query_ms": 50  # <50ms for database queries
    }


# Test data generation utilities
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
        'features': generate_test_features,
        'effectiveness_scores': generate_test_effectiveness_scores
    }


# Database population utilities
async def populate_test_database(session: AsyncSession, 
                                rule_metadata_list=None,
                                rule_performance_list=None,
                                user_feedback_list=None):
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


# Async testing utilities
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