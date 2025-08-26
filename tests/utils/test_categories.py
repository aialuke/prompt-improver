"""
Testing category utilities and helpers.

Provides utilities for test categorization, performance validation, and test execution management.
"""

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestPerformanceValidator:
    """Validates test performance requirements by category."""

    PERFORMANCE_LIMITS = {
        "unit": 100,        # Unit tests: <100ms
        "integration": 1000,  # Integration tests: <1s
        "contract": 5000,    # Contract tests: <5s
        "e2e": 10000        # E2E tests: <10s
    }

    @classmethod
    def validate_performance(cls, test_name: str, category: str, duration_ms: float):
        """Validate test performance meets category requirements."""
        limit = cls.PERFORMANCE_LIMITS.get(category, 1000)
        if duration_ms > limit:
            pytest.fail(
                f"{category.title()} test '{test_name}' took {duration_ms:.2f}ms "
                f"(should be <{limit}ms)"
            )

    @classmethod
    def performance_fixture(cls, category: str):
        """Create performance validation fixture for test category."""
        @pytest.fixture(autouse=True)
        def _performance_validator(request):
            start_time = time.time()
            yield
            duration = (time.time() - start_time) * 1000
            cls.validate_performance(request.node.name, category, duration)

        return _performance_validator


def unit_test_performance(func):
    """Decorator to enforce unit test performance requirements."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000
        TestPerformanceValidator.validate_performance(func.__name__, "unit", duration)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000
        TestPerformanceValidator.validate_performance(func.__name__, "unit", duration)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class MockFactory:
    """Factory for creating standardized mocks by test category."""

    @staticmethod
    def create_database_mock() -> AsyncMock:
        """Create standardized database mock for unit tests."""
        mock_db = AsyncMock()
        mock_db.execute.return_value = MagicMock()
        mock_db.commit.return_value = None
        mock_db.rollback.return_value = None
        mock_db.close.return_value = None
        return mock_db

    @staticmethod
    def create_redis_mock() -> AsyncMock:
        """Create standardized Redis mock for unit tests."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.expire.return_value = True
        return mock_redis

    @staticmethod
    def create_service_mock(service_name: str) -> MagicMock:
        """Create standardized service mock."""
        mock_service = MagicMock()
        mock_service.name = service_name
        mock_service.is_healthy = MagicMock(return_value=True)
        return mock_service


class TestDataFactory:
    """Factory for creating test data by category."""

    @staticmethod
    def create_prompt_data(complexity: str = "medium") -> dict[str, Any]:
        """Create standardized prompt test data."""
        prompts_by_complexity = {
            "simple": "Fix bug",
            "medium": "Fix the authentication bug in the login system",
            "complex": "Debug and fix the intermittent authentication timeout issue in the distributed login system that occurs under high load conditions"
        }

        return {
            "original_prompt": prompts_by_complexity[complexity],
            "expected_length_increase": {"simple": 2, "medium": 1.5, "complex": 1.2}[complexity],
            "expected_confidence_range": {"simple": (0.7, 0.9), "medium": (0.75, 0.95), "complex": (0.8, 0.98)}[complexity]
        }

    @staticmethod
    def create_session_data() -> dict[str, Any]:
        """Create standardized session test data."""
        from uuid import uuid4
        return {
            "session_id": str(uuid4()),
            "user_id": "test_user",
            "context": {
                "domain": "software_development",
                "language": "python",
                "complexity": "medium"
            }
        }

    @staticmethod
    def create_improvement_result(prompt: str, confidence: float = 0.85) -> dict[str, Any]:
        """Create standardized improvement result."""
        return {
            "improved_prompt": f"Please {prompt.lower()} by analyzing the specific requirements and implementing a comprehensive solution with proper error handling",
            "confidence": confidence,
            "rules_applied": ["clarity", "specificity", "structure"],
            "processing_time_ms": 45,
            "metadata": {
                "model_version": "test_v1",
                "rule_engine_version": "test_v1"
            }
        }


def pytest_collection_modifyitems(config, items):
    """Automatically categorize tests based on location and add appropriate markers."""
    for item in items:
        # Get test path relative to tests directory
        test_path = str(item.fspath.relative_to(item.config.rootpath))

        # Categorize by directory structure
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/contract/" in test_path:
            item.add_marker(pytest.mark.contract)
            if "/rest/" in test_path:
                item.add_marker(pytest.mark.api)
            elif "/mcp/" in test_path:
                item.add_marker(pytest.mark.mcp)
        elif "/e2e/" in test_path:
            item.add_marker(pytest.mark.e2e)
            if "/workflows/" in test_path:
                item.add_marker(pytest.mark.workflow)
            elif "/scenarios/" in test_path:
                item.add_marker(pytest.mark.scenario)

        # Add performance marker for performance tests
        if "performance" in item.name.lower() or "load" in item.name.lower():
            item.add_marker(pytest.mark.performance)

        # Add slow marker for tests that might be slow
        if any(marker in test_path for marker in ["/e2e/", "load_test", "stress_test"]):
            item.add_marker(pytest.mark.slow)


class TestCategoryValidator:
    """Validates test categorization and boundaries."""

    @staticmethod
    def validate_unit_test_isolation(test_func: Callable):
        """Validate unit test has no external dependencies."""
        # This would be extended with actual dependency checking
        # For now, it's a placeholder for validation logic

    @staticmethod
    def validate_integration_test_boundaries(test_func: Callable):
        """Validate integration test uses appropriate boundaries."""

    @staticmethod
    def validate_contract_test_compliance(test_func: Callable):
        """Validate contract test follows protocol specifications."""


def skip_if_no_containers():
    """Skip test if test containers are not available."""
    try:
        import testcontainers
        return pytest.mark.skipif(False, reason="Test containers available")
    except ImportError:
        return pytest.mark.skipif(True, reason="Test containers not available")


def skip_if_no_docker():
    """Skip test if Docker is not available."""
    import subprocess
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return pytest.mark.skipif(False, reason="Docker available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return pytest.mark.skipif(True, reason="Docker not available")


# Test execution utilities
class TestExecutionManager:
    """Manages test execution strategies by category."""

    @staticmethod
    def should_run_category(category: str, config) -> bool:
        """Determine if test category should run based on configuration."""
        # This would integrate with CI/CD configuration
        return True

    @staticmethod
    def get_parallel_workers(category: str) -> int | None:
        """Get recommended parallel workers for test category."""
        workers = {
            "unit": 8,        # Unit tests can run highly parallel
            "integration": 4,  # Integration tests need moderate parallelism
            "contract": 2,    # Contract tests need careful coordination
            "e2e": 1         # E2E tests should run sequentially
        }
        return workers.get(category)


# Export test category markers
UNIT_MARKER = pytest.mark.unit
INTEGRATION_MARKER = pytest.mark.integration
CONTRACT_MARKER = pytest.mark.contract
E2E_MARKER = pytest.mark.e2e
PERFORMANCE_MARKER = pytest.mark.performance
