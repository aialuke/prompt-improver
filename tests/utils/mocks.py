"""Mock utilities for testing.

Provides mock objects and utilities for test scenarios with consistent
interfaces matching production protocols.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock


class MockRedisClient:
    """Mock Redis client for testing scenarios."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._ttl_data: dict[str, int] = {}
        self.is_connected = True
        self.call_count = 0

    async def get(self, key: str) -> Any:
        """Mock Redis GET operation."""
        self.call_count += 1
        return self._data.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        """Mock Redis SET operation with optional expiration."""
        self.call_count += 1
        self._data[key] = value
        if ex:
            self._ttl_data[key] = ex
        return True

    async def delete(self, *keys: str) -> int:
        """Mock Redis DELETE operation."""
        self.call_count += 1
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                if key in self._ttl_data:
                    del self._ttl_data[key]
                deleted += 1
        return deleted

    async def exists(self, *keys: str) -> int:
        """Mock Redis EXISTS operation."""
        self.call_count += 1
        return sum(1 for key in keys if key in self._data)

    async def ping(self) -> bytes:
        """Mock Redis PING operation."""
        self.call_count += 1
        if not self.is_connected:
            raise ConnectionError("Mock Redis connection failed")
        return b"PONG"

    async def flushall(self) -> bool:
        """Mock Redis FLUSHALL operation."""
        self.call_count += 1
        self._data.clear()
        self._ttl_data.clear()
        return True

    async def info(self, section: str | None = None) -> dict[str, Any]:
        """Mock Redis INFO operation."""
        self.call_count += 1
        return {
            "redis_version": "7.0.0",
            "used_memory": 1024000,
            "connected_clients": 5,
            "total_commands_processed": self.call_count,
        }

    def disconnect(self) -> None:
        """Mock Redis disconnect."""
        self.is_connected = False

    def reset_mock(self) -> None:
        """Reset mock state for testing."""
        self._data.clear()
        self._ttl_data.clear()
        self.call_count = 0
        self.is_connected = True


class MockDatabaseSession:
    """Mock database session for testing."""

    def __init__(self):
        self.data: list[dict[str, Any]] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    async def execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Mock database execute operation."""
        # Simple mock that returns predefined data
        result = MagicMock()
        result.fetchall.return_value = self.data
        result.rowcount = len(self.data)
        return result

    async def commit(self) -> None:
        """Mock database commit."""
        self.committed = True

    async def rollback(self) -> None:
        """Mock database rollback."""
        self.rolled_back = True

    async def close(self) -> None:
        """Mock database close."""
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
        await self.close()


class MockCacheService:
    """Mock cache service for testing."""

    def __init__(self):
        self.redis = MockRedisClient()
        self.hit_count = 0
        self.miss_count = 0

    async def get(self, key: str) -> Any:
        """Mock cache get operation."""
        value = await self.redis.get(key)
        if value is not None:
            self.hit_count += 1
        else:
            self.miss_count += 1
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Mock cache set operation."""
        await self.redis.set(key, value, ex=ttl)

    async def delete(self, key: str) -> bool:
        """Mock cache delete operation."""
        count = await self.redis.delete(key)
        return count > 0

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class MockMLModel:
    """Mock ML model for testing."""

    def __init__(self, model_id: str = "test_model"):
        self.model_id = model_id
        self.prediction_count = 0
        self.is_trained = True

    async def predict(self, inputs: Any) -> Any:
        """Mock model prediction."""
        self.prediction_count += 1
        # Return simple mock prediction
        return {"prediction": 0.85, "confidence": 0.92}

    async def predict_batch(self, inputs: list[Any]) -> list[Any]:
        """Mock batch prediction."""
        self.prediction_count += len(inputs)
        return [await self.predict(inp) for inp in inputs]

    def get_model_info(self) -> dict[str, Any]:
        """Mock model info."""
        return {
            "model_id": self.model_id,
            "version": "1.0.0",
            "trained": self.is_trained,
            "predictions_made": self.prediction_count,
        }


class MockHealthChecker:
    """Mock health checker for testing."""

    def __init__(self, component_name: str = "test_component"):
        self.component_name = component_name
        self.is_healthy = True
        self.check_count = 0

    async def check_health(self) -> dict[str, Any]:
        """Mock health check."""
        self.check_count += 1
        return {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "component_name": self.component_name,
            "response_time_ms": 15.0,
            "checks_performed": self.check_count,
            "details": {"mock": True}
        }

    def get_component_name(self) -> str:
        """Get component name."""
        return self.component_name

    def get_timeout_seconds(self) -> float:
        """Get timeout for health check."""
        return 5.0


# Utility functions for creating mocks
def create_mock_database_services():
    """Create mock database services for testing."""
    mock = AsyncMock()
    mock.get_session.return_value.__aenter__.return_value = MockDatabaseSession()
    return mock


def create_mock_cache_services():
    """Create mock cache services for testing."""
    return MockCacheService()


def create_mock_health_services():
    """Create mock health services for testing."""
    mock = AsyncMock()
    mock.check_component_health.return_value = {
        "status": "healthy",
        "response_time_ms": 12.5,
        "details": {"mock": True}
    }
    return mock
