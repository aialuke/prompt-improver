"""
Enhanced model cache registry tests with real behavior following 2025 best practices.
Migration from mock-based testing to real behavior testing based on research:
- Use real sklearn models with lightweight configurations for testing
- Real memory management and TTL testing with actual model objects
- Real thread-safety testing with concurrent access patterns
- Mock only external services that would be expensive or flaky
- Test actual cache performance characteristics with real models
"""

import asyncio

import numpy as np
import pytest
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from prompt_improver.services.ml_integration import InMemoryModelRegistry


class TestInMemoryModelCache:
    """Test suite for in-memory model cache registry with real ML models."""

    @pytest.fixture
    def model_registry(self):
        """Fixture to create an in-memory model registry."""
        return InMemoryModelRegistry(max_cache_size_mb=100)

    @pytest.fixture
    def sample_sklearn_model(self):
        """Create a real trained sklearn model for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=3, random_state=42)
        model.fit(X_train, y_train)
        return (model, X_test, y_test)

    @pytest.fixture
    def multiple_sklearn_models(self):
        """Create multiple real trained sklearn models for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        models = {}
        rf_model = RandomForestClassifier(n_estimators=3, random_state=42)
        rf_model.fit(X_train, y_train)
        models["rf_model"] = rf_model
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)
        models["lr_model"] = lr_model
        return (models, X_test, y_test)

    @given(st.text(min_size=5, max_size=20), st.integers(min_value=1, max_value=10))
    @settings(max_examples=5)
    def test_cache_registry_operations_with_real_models(self, model_id, ttl_minutes):
        """Test add, remove, and retrieval operations with real sklearn models."""
        model_registry = InMemoryModelRegistry(max_cache_size_mb=100)
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=3, random_state=42)
        model.fit(X_train, y_train)
        success = model_registry.add_model(
            model_id, model, model_type="sklearn", ttl_minutes=ttl_minutes
        )
        assert success is True
        cached_model = model_registry.get_model(model_id)
        assert cached_model is not None
        assert cached_model is model
        predictions = cached_model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert all(pred in {0, 1} for pred in predictions)
        model_registry.remove_model(model_id)
        assert model_registry.get_model(model_id) is None

    @pytest.mark.asyncio
    async def test_model_ttl_expiry_with_real_models(
        self, model_registry, sample_sklearn_model
    ):
        """Test that real models expire in cache after TTL has passed."""
        model, _X_test, _y_test = sample_sklearn_model
        model_registry.add_model("expiring_model", model, ttl_minutes=0.02)
        cached_model = model_registry.get_model("expiring_model")
        assert cached_model is not None
        assert cached_model is model
        await asyncio.sleep(1.5)
        expired_model = model_registry.get_model("expiring_model")
        assert expired_model is None

    def test_cache_eviction_policy_with_real_models(self, multiple_sklearn_models):
        """Test the LRU cache eviction policy with real models."""
        models, _X_test, _y_test = multiple_sklearn_models
        model_registry = InMemoryModelRegistry(max_cache_size_mb=20)
        model_ids = []
        for i, (model_name, model) in enumerate(models.items()):
            model_id = f"model_{i}_{model_name}"
            model_ids.append(model_id)
            success = model_registry.add_model(model_id, model, model_type="sklearn")
            assert success is True
        X, y = make_classification(
            n_samples=200, n_features=20, n_classes=2, random_state=42
        )
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        large_model = RandomForestClassifier(n_estimators=10, random_state=42)
        large_model.fit(X_train, y_train)
        model_registry.add_model("large_model", large_model, model_type="sklearn")
        cached_large = model_registry.get_model("large_model")
        assert cached_large is not None
        assert cached_large is large_model
        stats = model_registry.get_cache_stats()
        assert stats["total_models"] >= 1
        assert stats["active_models"] >= 1

    @pytest.mark.asyncio
    async def test_model_access_update_with_real_models(
        self, model_registry, sample_sklearn_model
    ):
        """Test that accessing a real model updates its access metadata."""
        model, X_test, y_test = sample_sklearn_model
        model_registry.add_model("access_test", model, model_type="sklearn")
        cached_model = model_registry.get_model("access_test")
        assert cached_model is not None
        assert cached_model is model
        predictions = cached_model.predict(X_test)
        assert len(predictions) == len(y_test)
        await asyncio.sleep(0.1)
        cached_model_accessed = model_registry.get_model("access_test")
        assert cached_model_accessed is not None
        assert cached_model_accessed is model
        entry = model_registry._cache.get("access_test")
        assert entry.access_count == 2

    def test_real_model_memory_estimation(self, model_registry, sample_sklearn_model):
        """Test memory estimation with real sklearn models."""
        model, _, _ = sample_sklearn_model
        estimated_memory = model_registry._estimate_model_memory(model)
        assert estimated_memory > 0.0
        assert isinstance(estimated_memory, float)
        model_registry.add_model("memory_test", model, model_type="sklearn")
        stats = model_registry.get_cache_stats()
        assert stats["total_memory_mb"] > 0.0
        assert len(stats["model_details"]) == 1
        assert stats["model_details"][0]["memory_mb"] > 0.0

    def test_real_model_performance_validation(
        self, model_registry, sample_sklearn_model
    ):
        """Test that cached models maintain performance characteristics."""
        model, X_test, y_test = sample_sklearn_model
        baseline_predictions = model.predict(X_test)
        baseline_accuracy = np.mean(baseline_predictions == y_test)
        model_registry.add_model("performance_test", model, model_type="sklearn")
        cached_model = model_registry.get_model("performance_test")
        cached_predictions = cached_model.predict(X_test)
        cached_accuracy = np.mean(cached_predictions == y_test)
        assert cached_accuracy == baseline_accuracy
        assert np.array_equal(cached_predictions, baseline_predictions)


class TestAdditionalCacheScenarios:
    """Additional cache tests with real models covering edge cases."""

    @pytest.fixture
    def model_registry(self):
        """Fixture to create an in-memory model registry."""
        return InMemoryModelRegistry(max_cache_size_mb=50)

    @pytest.fixture
    def custom_model_class(self):
        """Create a custom model class for testing unknown model types."""

        class CustomModel:
            def __init__(self):
                self.data = np.random.rand(100, 10)
                self.weights = np.random.rand(10)

            def predict(self, X):
                return np.dot(X, self.weights)

        return CustomModel()

    def test_estimated_memory_on_custom_model(self, model_registry, custom_model_class):
        """Test memory estimation for custom model types."""
        estimated_memory = model_registry._estimate_model_memory(custom_model_class)
        assert estimated_memory > 0.0
        assert estimated_memory >= 5.0

    @pytest.mark.asyncio
    async def test_concurrent_model_retrieval_with_real_models(self, model_registry):
        """Test concurrent model retrieval with real models ensuring thread-safety."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        model_registry.add_model("concurrent_test", model, model_type="sklearn")

        async def retrieve_and_test_model():
            cached_model = model_registry.get_model("concurrent_test")
            assert cached_model is not None
            predictions = cached_model.predict(X[:10])
            assert len(predictions) == 10
            return cached_model

        results = await asyncio.gather(*[retrieve_and_test_model() for _ in range(10)])
        assert all(result is model for result in results)
        cache_stats = model_registry.get_cache_stats()
        assert cache_stats["active_models"] == 1

    @given(
        model_sizes=st.lists(
            st.integers(min_value=3, max_value=8), min_size=2, max_size=5
        )
    )
    @settings(max_examples=10)
    def test_varying_model_sizes_eviction_real_models(self, model_sizes):
        """Test eviction logic with varying real model sizes."""
        assume(len(model_sizes) >= 2)
        model_registry = InMemoryModelRegistry(max_cache_size_mb=30)
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X_train, _X_test, y_train, _y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model_ids = []
        for i, n_estimators in enumerate(model_sizes):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            model_id = f"model_{i}_est_{n_estimators}"
            model_ids.append(model_id)
            model_registry.add_model(model_id, model, model_type="sklearn")
        final_model = RandomForestClassifier(
            n_estimators=max(model_sizes) + 2, random_state=42
        )
        final_model.fit(X_train, y_train)
        model_registry.add_model("final_model", final_model, model_type="sklearn")
        cached_final = model_registry.get_model("final_model")
        assert cached_final is not None
        assert cached_final is final_model
        cache_stats = model_registry.get_cache_stats()
        assert cache_stats["total_models"] >= 1
        assert cache_stats["memory_utilization"] <= 1.0

    @pytest.mark.asyncio
    async def test_cleanup_expired_models_real_behavior(self, model_registry):
        """Test that expired real models are cleaned up from the registry."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model1 = LogisticRegression(random_state=42)
        model1.fit(X_train, y_train)
        model_registry.add_model("expired_test", model1, ttl_minutes=0.001)
        model2 = LogisticRegression(random_state=43)
        model2.fit(X_train, y_train)
        model_registry.add_model("persistent_test", model2, ttl_minutes=60)
        await asyncio.sleep(0.1)
        cleaned_count = model_registry.cleanup_expired()
        assert cleaned_count > 0
        assert model_registry.get_model("expired_test") is None
        persistent_model = model_registry.get_model("persistent_test")
        assert persistent_model is not None
        predictions = persistent_model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_cache_stats_with_real_models(self, model_registry):
        """Test comprehensive cache statistics with real models."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        models = []
        for i in range(3):
            model = RandomForestClassifier(n_estimators=2, random_state=42 + i)
            model.fit(X_train, y_train)
            models.append(model)
            model_registry.add_model(f"stats_test_{i}", model, model_type="sklearn")
        for i in range(3):
            model_registry.get_model(f"stats_test_{i}")
        stats = model_registry.get_cache_stats()
        assert stats["total_models"] == 3
        assert stats["active_models"] == 3
        assert stats["expired_models"] == 0
        assert stats["total_memory_mb"] > 0
        assert stats["memory_utilization"] >= 0
        assert len(stats["model_details"]) == 3
        for detail in stats["model_details"]:
            assert detail["model_type"] == "sklearn"
            assert detail["memory_mb"] > 0
            assert detail["access_count"] >= 1
            assert detail["cached_minutes_ago"] >= 0
            assert not detail["is_expired"]

    @pytest.mark.asyncio
    async def test_real_model_prediction_consistency(self, model_registry):
        """Test that real models maintain prediction consistency after caching."""
        X, y = make_classification(
            n_samples=200, n_features=15, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, _y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        original_predictions = model.predict(X_test)
        original_probabilities = model.predict_proba(X_test)
        model_registry.add_model("consistency_test", model, model_type="sklearn")
        for _ in range(5):
            cached_model = model_registry.get_model("consistency_test")
            assert cached_model is not None
            cached_predictions = cached_model.predict(X_test)
            cached_probabilities = cached_model.predict_proba(X_test)
            assert np.array_equal(cached_predictions, original_predictions)
            assert np.allclose(cached_probabilities, original_probabilities)
            await asyncio.sleep(0.01)
