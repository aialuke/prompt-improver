"""Integration tests for cache invalidation flow using pattern.invalidate events.
Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Real Redis container via testcontainers for pub/sub and cache operations
- Real database connections for database-dependent tests
- Mock only external dependencies (MLflow)
- Test actual cache invalidation behavior with real Redis operations
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_improver.database.models import RuleMetadata
from prompt_improver.services.ml_integration import get_ml_service
from prompt_improver.utils.datetime_utils import aware_utc_now
from prompt_improver.utils.redis_cache import (
    CacheSubscriber,
    start_cache_subscriber,
    stop_cache_subscriber,
)


@pytest.fixture
async def ml_service():
    """Get ML service instance for testing."""
    return await get_ml_service()


@pytest.fixture
async def cache_subscriber(redis_client):
    """Create cache subscriber for testing with real Redis client."""
    # Configure cache module to use real Redis client
    import prompt_improver.utils.redis_cache as cache_module
    original_client = getattr(cache_module, 'redis_client', None)
    cache_module.redis_client = redis_client
    
    try:
        subscriber = CacheSubscriber()
        yield subscriber
        await subscriber.stop()
    finally:
        if original_client is not None:
            cache_module.redis_client = original_client


@pytest.fixture
async def real_rule_metadata(real_db_session):
    """Create real rule metadata for testing cache invalidation."""
    import uuid

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
            created_at=aware_utc_now(),
            updated_at=aware_utc_now(),
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
            updated_at=aware_utc_now(),
        ),
    ]
    for record in metadata_records:
        real_db_session.add(record)
    await real_db_session.commit()
    for record in metadata_records:
        await real_db_session.refresh(record)
    return metadata_records


@pytest.mark.asyncio
class TestCacheInvalidationFlow:
    """Test cache invalidation flow from ML training to cache invalidation."""

    async def test_rule_parameter_update_emits_invalidation_event(
        self, ml_service, real_db_session, redis_client, real_rule_metadata
    ):
        """Test that rule parameter updates emit pattern.invalidate events using real Redis."""
        rule_ids = [rule.rule_id for rule in real_rule_metadata]
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("pattern.invalidate")
        try:
            with patch(
                "prompt_improver.services.ml_integration.redis_client", redis_client
            ):
                optimized_params = {"n_estimators": 100, "max_depth": 10}
                await ml_service._update_rule_parameters(
                    real_db_session, rule_ids, optimized_params, 0.85, "test_model_123"
                )
                await asyncio.sleep(0.1)
                message = await pubsub.get_message(timeout=1.0)
                if message and message["type"] == "subscribe":
                    message = await pubsub.get_message(timeout=1.0)
                assert message is not None
                assert message["type"] == "message"
                assert message["channel"] == b"pattern.invalidate"
                event_data = json.loads(message["data"])
                assert event_data["type"] == "rule_parameters_updated"
                assert set(event_data["rule_ids"]) == set(rule_ids)
                assert event_data["effectiveness_score"] == 0.85
                assert event_data["model_id"] == "test_model_123"
                assert "apes:pattern:" in event_data["cache_prefixes"]
                assert "rule:" in event_data["cache_prefixes"]
                assert "timestamp" in event_data
        finally:
            await pubsub.close()

    async def test_model_training_completion_emits_invalidation_event(
        self, ml_service, real_db_session, redis_client
    ):
        """Test that model training completion emits pattern.invalidate events using real Redis."""
        training_data = {
            "features": [[0.5, 100, 0.8, 0.3, 0.7]] * 25,
            "effectiveness_scores": [
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
        }
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("pattern.invalidate")
        try:
            with (
                patch(
                    "prompt_improver.services.ml_integration.redis_client", redis_client
                ),
                patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
            ):
                mock_mlflow.active_run.return_value = None
                mock_mlflow.start_run.return_value.__enter__.return_value = None
                mock_mlflow.start_run.return_value.__exit__.return_value = None
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.active_run.return_value = mock_run
                result = await ml_service.optimize_rules(training_data, real_db_session)
                assert result["status"] == "success"
                assert "model_id" in result
                await asyncio.sleep(0.1)
                messages = []
                while True:
                    try:
                        message = await pubsub.get_message(timeout=0.5)
                        if message:
                            messages.append(message)
                        else:
                            break
                    except TimeoutError:
                        break
                cache_invalidation_message = None
                for message in messages:
                    if (
                        message["type"] == "message"
                        and message["channel"] == b"pattern.invalidate"
                    ):
                        try:
                            event_data = json.loads(message["data"])
                            if event_data["type"] == "model_training_completed":
                                cache_invalidation_message = message
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
                assert cache_invalidation_message is not None
                event_data = json.loads(cache_invalidation_message["data"])
                assert event_data["type"] == "model_training_completed"
                assert event_data["model_id"] == result["model_id"]
                assert event_data["model_type"] == "RandomForestClassifier"
                assert "apes:pattern:" in event_data["cache_prefixes"]
                assert "rule:" in event_data["cache_prefixes"]
                assert "ml:model:" in event_data["cache_prefixes"]
                assert "timestamp" in event_data
        finally:
            await pubsub.close()

    async def test_cache_subscriber_handles_invalidation_events(
        self, cache_subscriber, redis_client
    ):
        """Test that cache subscriber properly handles pattern.invalidate events using real Redis."""
        test_keys = [
            "apes:pattern:test_key_1",
            "apes:pattern:test_key_2",
            "rule:test_key_1",
            "rule:test_key_2",
        ]
        for key in test_keys:
            await redis_client.set(key, "test_value")
        for key in test_keys:
            assert await redis_client.exists(key)
        try:
            await cache_subscriber.start()
            await asyncio.sleep(0.1)
            test_event = {
                "type": "rule_parameters_updated",
                "rule_ids": ["test_rule_1", "test_rule_2"],
                "cache_prefixes": ["apes:pattern:", "rule:"],
            }
            await redis_client.publish("pattern.invalidate", json.dumps(test_event))
            await asyncio.sleep(0.2)
            for key in test_keys:
                assert not await redis_client.exists(key)
        finally:
            await cache_subscriber.stop()
            for key in test_keys:
                await redis_client.delete(key)

    async def test_cache_subscriber_startup_and_shutdown(self, redis_client):
        """Test cache subscriber startup and shutdown functionality using real Redis."""
        # Configure cache module to use real Redis
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = redis_client
        
        try:
            await start_cache_subscriber()
            await asyncio.sleep(0.1)
            await stop_cache_subscriber()
            await asyncio.sleep(0.1)
        finally:
            if original_client is not None:
                cache_module.redis_client = original_client

    async def test_cache_subscriber_error_handling(self, redis_client):
        """Test cache subscriber error handling using real Redis with simulated error conditions."""
        # Configure cache module to use real Redis
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = redis_client
        
        try:
            subscriber = CacheSubscriber()
            try:
                await subscriber.start()
                await asyncio.sleep(0.1)
                original_get_message = subscriber.pubsub.get_message
                with patch.object(
                    subscriber.pubsub,
                    "get_message",
                    side_effect=Exception("Simulated Redis error"),
                ):
                    await asyncio.sleep(0.1)
                assert subscriber.pubsub is not None
                assert subscriber.is_running
            finally:
                await subscriber.stop()

    async def test_cache_invalidation_by_prefix(self, redis_client):
        """Test cache invalidation by prefix functionality using real Redis."""
        # Configure cache module to use real Redis
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = redis_client
        
        try:
            subscriber = CacheSubscriber()
            test_keys = [
                "apes:pattern:key1",
                "apes:pattern:key2",
                "apes:pattern:key3",
                "other:key1",
            ]
            for key in test_keys:
                await redis_client.set(key, "test_value")
            for key in test_keys:
                assert await redis_client.exists(key)
            deleted_count = await subscriber._invalidate_by_prefix("apes:pattern:")
            assert not await redis_client.exists("apes:pattern:key1")
            assert not await redis_client.exists("apes:pattern:key2")
            assert not await redis_client.exists("apes:pattern:key3")
            assert await redis_client.exists("other:key1")
            assert deleted_count == 3
            await redis_client.delete("other:key1")

    async def test_ensemble_model_training_emits_invalidation_event(
        self, ml_service, real_db_session, redis_client
    ):
        """Test that ensemble model training emits pattern.invalidate events using real Redis."""
        training_data = {
            "features": [[0.5, 100, 0.8, 0.3, 0.7]] * 25,
            "effectiveness_scores": [
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
        }
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("pattern.invalidate")
        try:
            with (
                patch(
                    "prompt_improver.services.ml_integration.redis_client", redis_client
                ),
                patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow,
            ):
                mock_mlflow.active_run.return_value = None
                mock_mlflow.start_run.return_value.__enter__.return_value = None
                mock_mlflow.start_run.return_value.__exit__.return_value = None
                mock_run = MagicMock()
                mock_run.info.run_id = "ensemble_run_123"
                mock_mlflow.active_run.return_value = mock_run
                result = await ml_service.optimize_ensemble_rules(
                    training_data, real_db_session
                )
                assert result["status"] == "success"
                assert "model_id" in result
                await asyncio.sleep(0.1)
                messages = []
                while True:
                    try:
                        message = await pubsub.get_message(timeout=0.5)
                        if message:
                            messages.append(message)
                        else:
                            break
                    except TimeoutError:
                        break
                cache_invalidation_message = None
                for message in messages:
                    if (
                        message["type"] == "message"
                        and message["channel"] == b"pattern.invalidate"
                    ):
                        try:
                            event_data = json.loads(message["data"])
                            if event_data["type"] == "model_training_completed":
                                cache_invalidation_message = message
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
                assert cache_invalidation_message is not None
                event_data = json.loads(cache_invalidation_message["data"])
                assert event_data["type"] == "model_training_completed"
                assert event_data["model_id"] == result["model_id"]
                assert event_data["model_type"] == "StackingClassifier"
                assert "apes:pattern:" in event_data["cache_prefixes"]
                assert "rule:" in event_data["cache_prefixes"]
                assert "ml:model:" in event_data["cache_prefixes"]
                assert "timestamp" in event_data
        finally:
            await pubsub.close()

    async def test_cache_invalidation_maintains_data_freshness(self, redis_client):
        """Test that cache invalidation maintains data freshness using real Redis."""
        # Configure cache module to use real Redis
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = redis_client
        
        try:
            subscriber = CacheSubscriber()
            cached_keys = [
                "apes:pattern:rule_123",
                "apes:pattern:model_456",
                "rule:clarity_rule",
                "rule:specificity_rule",
            ]
            for key in cached_keys:
                await redis_client.set(key, "cached_value")
            for key in cached_keys:
                assert await redis_client.exists(key)
            event_data = {
                "type": "rule_parameters_updated",
                "rule_ids": ["clarity_rule", "specificity_rule"],
                "cache_prefixes": ["apes:pattern:", "rule:"],
            }
            await subscriber._handle_invalidate_event({
                "type": "message",
                "data": json.dumps(event_data),
            })
            for key in cached_keys:
                assert not await redis_client.exists(key)
            pattern_keys = []
            async for key in redis_client.scan_iter(match="apes:pattern:*"):
                pattern_keys.append(key)
            rule_keys = []
            async for key in redis_client.scan_iter(match="rule:*"):
                rule_keys.append(key)
            assert len(pattern_keys) == 0
            assert len(rule_keys) == 0
