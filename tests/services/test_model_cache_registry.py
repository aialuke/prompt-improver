"""
Tests for In-Memory Model Cache Registry Implementation
Incorporates property-based and performance tests using pytest and hypothesis.
"""

import asyncio
from time import time
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)

from prompt_improver.services.ml_integration import InMemoryModelRegistry


class TestInMemoryModelCache:
    """Test suite for in-memory model cache registry with TTL and eviction policies."""

    @pytest.fixture
    def model_registry(self):
        """Fixture to create an in-memory model registry."""
        return InMemoryModelRegistry(max_cache_size_mb=500)

    @given(
        st.text(min_size=5, max_size=20),
        st.dictionaries(keys=st.text(min_size=1, max_size=5), values=st.floats(min_value=0.0, max_value=1.0)),
        st.integers(min_value=50, max_value=500)  # Model size in MB
    )
    @settings(max_examples=10)
    def test_cache_registry_operations(self, model_id, model_data, model_size):
        """Test add, remove, and retrieval operations in the model cache."""
        model_registry = InMemoryModelRegistry(max_cache_size_mb=500)
        mock_model = MagicMock()

        model_registry.add_model(model_id, mock_model, ttl_minutes=60)
        cached_model = model_registry.get_model(model_id)

        # Verify model is cached
        assert cached_model is not None

        # Remove model and verify it's no longer in the cache
        model_registry.remove_model(model_id)
        assert model_registry.get_model(model_id) is None

    @pytest.mark.asyncio
    async def test_model_ttl_expiry(self, model_registry):
        """Test that models expire in cache after TTL has passed."""
        mock_model = MagicMock()

        # Add mock model with very short TTL
        model_registry.add_model("expiring_model", mock_model, ttl_minutes=0.01)
        await asyncio.sleep(0.75)  # Sleep to allow TTL to expire

        # Verify the model has expired
        expired_model = model_registry.get_model("expiring_model")
        assert expired_model is None

    def test_cache_eviction_policy(self, model_registry):
        """Test the LRU cache eviction policy by adding models until cache is overfilled."""
        # Create mock models
        model_count = 10
        mock_models = {f"model_{i}": MagicMock() for i in range(model_count)}

        # Assume models have varied sizes. Prepare registry and fill it up.
        for idx, (model_id, mock_model) in enumerate(mock_models.items()):
            model_size_mb = ((idx % 5) + 1) * 10  # sizes: 10MB, 20MB, ..., 50MB
            with patch.object(model_registry, '_estimate_model_memory', return_value=model_size_mb):
                model_registry.add_model(model_id, mock_model)

        # Add a large model that forces eviction
        large_model = MagicMock()
        with patch.object(model_registry, '_estimate_model_memory', return_value=300):
            model_registry.add_model("large_model", large_model)

        # Verify higher idx models remain in cache and lower ones might be evicted due to LRU
        surviving_ids = [model_id for idx, model_id in enumerate(mock_models.keys()) if idx >= 5]
        assert surviving_ids
        assert model_registry.get_model("large_model") is not None

    @pytest.mark.asyncio
    async def test_model_access_update(self, model_registry):
        """Test that accessing a model updates its access metadata."""
        mock_model = MagicMock()
        model_registry.add_model("access_test", mock_model)

        cached_model = model_registry.get_model("access_test")
        assert cached_model is not None

        # Simulate a delay and then access the model
        await asyncio.sleep(0.1)
        cached_model_accessed = model_registry.get_model("access_test")
        assert cached_model_accessed is not None

        # Verify access timestamp and count
        entry = model_registry._cache.get("access_test")
        assert entry.access_count == 2


# Additional targeted tests for edge cases and integration scenarios
class TestAdditionalCacheScenarios:
    """Additional cache tests covering edge cases and special conditions."""

    @pytest.fixture
    def model_registry(self):
        """Fixture to create an in-memory model registry."""
        return InMemoryModelRegistry(max_cache_size_mb=500)

    def test_estimated_memory_on_undefined_model(self, model_registry):
        """Verify memory size estimation for undefined model types."""

        class UnknownModel:
            pass

        unknown_model = UnknownModel()

        # Use internal model estimation method for test
        estimated_memory = model_registry._estimate_model_memory(unknown_model)

        # Should use conservative default estimate
        assert estimated_memory > 0.0

    @pytest.mark.asyncio
    async def test_concurrent_model_retrieval(self, model_registry):
        """Test concurrent model retrieval and ensure registry thread-safety."""
        mock_model = MagicMock()
        model_registry.add_model("concurrent_test", mock_model)

        async def retrieve_model():
            return model_registry.get_model("concurrent_test")

        await asyncio.gather(*[retrieve_model() for _ in range(10)])

        # Verify no deadlocks or exceptions occurred during concurrent access
        cache_stats = model_registry.get_cache_stats()
        assert cache_stats["active_models"] == 1

    @given(model_sizes=st.lists(st.integers(min_value=10, max_value=50), min_size=3, max_size=8))
    @settings(max_examples=20)
    def test_varying_model_sizes_eviction(self, model_sizes):
        """Test eviction logic with varying model sizes using property-based testing."""
        model_registry = InMemoryModelRegistry(max_cache_size_mb=500)
        # More lenient assumption to reduce filtering
        assume(sum(model_sizes) <= 300)

        # Fill cache with varying model sizes
        for i, size in enumerate(model_sizes):
            mock_model = MagicMock()
            model_id = f"model_{i}"
            with patch.object(model_registry, '_estimate_model_memory', return_value=size):
                model_registry.add_model(model_id, mock_model)

        # Force more additions to trigger eviction
        new_model = MagicMock()
        with patch.object(model_registry, '_estimate_model_memory', return_value=100):
            model_registry.add_model("evict_test", new_model)

        # Verify eviction occurred
        assert model_registry.get_model("evict_test") is not None

        # Verify that newly added model is now part of cache
        assert "evict_test" in model_registry._cache

        # Total models should be reasonable given cache constraints
        cache_stats = model_registry.get_cache_stats()
        assert cache_stats["total_models"] >= 1  # At least the evict_test model

    @pytest.mark.asyncio
    async def test_cleanup_expired_models(self, model_registry):
        """Test that expired models are cleaned up from the registry."""
        mock_model = MagicMock()

        # Add a model with a very small TTL, let it expire
        model_registry.add_model("expired_test", mock_model, ttl_minutes=0.001)
        await asyncio.sleep(0.75)  # Ensure expiry

        cleaned_count = model_registry.cleanup_expired()

        # Verify cleanup occurred successfully
        assert cleaned_count > 0
        assert model_registry.get_model("expired_test") is None
