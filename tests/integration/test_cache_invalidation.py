"""Integration tests for cache invalidation flow using pattern.invalidate events.
Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Real Redis container via testcontainers for pub/sub and cache operations
- Real database connections for database-dependent tests
- Mock only external dependencies (MLflow)
- Test actual cache invalidation behavior with real Redis operations
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prompt_improver.services.ml_integration import get_ml_service
from prompt_improver.utils.redis_cache import CacheSubscriber, start_cache_subscriber, stop_cache_subscriber
from prompt_improver.database.models import RuleMetadata
from prompt_improver.utils.datetime_utils import aware_utc_now


@pytest.fixture
async def ml_service():
    """Get ML service instance for testing."""
    return await get_ml_service()


@pytest.fixture
async def cache_subscriber(redis_client):
    """Create cache subscriber for testing with real Redis client."""
    # Patch the redis_client in the module to use our test Redis instance
    with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
        subscriber = CacheSubscriber()
        yield subscriber
        # Ensure proper cleanup
        await subscriber.stop()


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
    
    # Add records to database
    for record in metadata_records:
        real_db_session.add(record)
    await real_db_session.commit()
    
    # Refresh to get database-generated values
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
        # Extract rule IDs from real metadata
        rule_ids = [rule.rule_id for rule in real_rule_metadata]
        
        # Create a pub/sub subscriber to capture published events
        pubsub = redis_client.pubsub()
        await pubsub.subscribe('pattern.invalidate')
        
        try:
            # Patch the redis_client in ml_integration to use our test Redis instance
            with patch('prompt_improver.services.ml_integration.redis_client', redis_client):
                # Test rule parameter update
                optimized_params = {"n_estimators": 100, "max_depth": 10}
                await ml_service._update_rule_parameters(
                    real_db_session, 
                    rule_ids, 
                    optimized_params, 
                    0.85, 
                    "test_model_123"
                )
                
                # Wait for message to be published and received
                await asyncio.sleep(0.1)
                
                # Get the published message
                message = await pubsub.get_message(timeout=1.0)
                # Skip the subscription confirmation message
                if message and message['type'] == 'subscribe':
                    message = await pubsub.get_message(timeout=1.0)
                
                # Verify cache invalidation event was published
                assert message is not None
                assert message['type'] == 'message'
                assert message['channel'] == b'pattern.invalidate'
                
                # Verify the event details
                event_data = json.loads(message['data'])
                assert event_data['type'] == 'rule_parameters_updated'
                assert set(event_data['rule_ids']) == set(rule_ids)
                assert event_data['effectiveness_score'] == 0.85
                assert event_data['model_id'] == 'test_model_123'
                assert 'apes:pattern:' in event_data['cache_prefixes']
                assert 'rule:' in event_data['cache_prefixes']
                assert 'timestamp' in event_data
        finally:
            await pubsub.close()

    async def test_model_training_completion_emits_invalidation_event(
        self, ml_service, real_db_session, redis_client
    ):
        """Test that model training completion emits pattern.invalidate events using real Redis."""
        # Setup training data
        training_data = {
            "features": [[0.5, 100, 0.8, 0.3, 0.7]] * 25,
            "effectiveness_scores": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6, 
                                   0.7, 0.8, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                   0.3, 0.4, 0.5, 0.6, 0.7]
        }
        
        # Create a pub/sub subscriber to capture published events
        pubsub = redis_client.pubsub()
        await pubsub.subscribe('pattern.invalidate')
        
        try:
            # Patch both redis_client and MLflow
            with patch('prompt_improver.services.ml_integration.redis_client', redis_client), \
                 patch('prompt_improver.services.ml_integration.mlflow') as mock_mlflow:
                
                # Mock MLflow (external dependency)
                mock_mlflow.active_run.return_value = None
                mock_mlflow.start_run.return_value.__enter__.return_value = None
                mock_mlflow.start_run.return_value.__exit__.return_value = None
                
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.active_run.return_value = mock_run
                
                result = await ml_service.optimize_rules(training_data, real_db_session)
                
                # Verify successful training
                assert result["status"] == "success"
                assert "model_id" in result
                
                # Wait for message to be published and received
                await asyncio.sleep(0.1)
                
                # Get messages and find cache invalidation event
                messages = []
                while True:
                    try:
                        message = await pubsub.get_message(timeout=0.5)
                        if message:
                            messages.append(message)
                        else:
                            break
                    except asyncio.TimeoutError:
                        break
                
                # Find the cache invalidation message
                cache_invalidation_message = None
                for message in messages:
                    if (message['type'] == 'message' and 
                        message['channel'] == b'pattern.invalidate'):
                        try:
                            event_data = json.loads(message['data'])
                            if event_data['type'] == 'model_training_completed':
                                cache_invalidation_message = message
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                assert cache_invalidation_message is not None
                
                # Verify the event details
                event_data = json.loads(cache_invalidation_message['data'])
                assert event_data['type'] == 'model_training_completed'
                assert event_data['model_id'] == result['model_id']
                assert event_data['model_type'] == 'RandomForestClassifier'
                assert 'apes:pattern:' in event_data['cache_prefixes']
                assert 'rule:' in event_data['cache_prefixes']
                assert 'ml:model:' in event_data['cache_prefixes']
                assert 'timestamp' in event_data
        finally:
            await pubsub.close()

    async def test_cache_subscriber_handles_invalidation_events(
        self, cache_subscriber, redis_client
    ):
        """Test that cache subscriber properly handles pattern.invalidate events using real Redis."""
        # Setup real cache keys in Redis
        test_keys = [
            'apes:pattern:test_key_1',
            'apes:pattern:test_key_2', 
            'rule:test_key_1',
            'rule:test_key_2'
        ]
        
        # Add some test keys to Redis
        for key in test_keys:
            await redis_client.set(key, "test_value")
        
        # Verify keys exist before invalidation
        for key in test_keys:
            assert await redis_client.exists(key)
        
        try:
            # Start subscriber
            await cache_subscriber.start()
            
            # Give time for subscription to be established
            await asyncio.sleep(0.1)
            
            # Publish test invalidation event
            test_event = {
                'type': 'rule_parameters_updated',
                'rule_ids': ['test_rule_1', 'test_rule_2'],
                'cache_prefixes': ['apes:pattern:', 'rule:']
            }
            
            # Publish the event to trigger cache invalidation
            await redis_client.publish('pattern.invalidate', json.dumps(test_event))
            
            # Give time for event processing
            await asyncio.sleep(0.2)
            
            # Verify cache keys were deleted
            for key in test_keys:
                assert not await redis_client.exists(key)
                
        finally:
            # Stop subscriber
            await cache_subscriber.stop()
            
            # Cleanup any remaining keys
            for key in test_keys:
                await redis_client.delete(key)

    async def test_cache_subscriber_startup_and_shutdown(
        self, redis_client
    ):
        """Test cache subscriber startup and shutdown functionality using real Redis."""
        # Patch the redis_client for global subscriber functions
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            # Test startup
            await start_cache_subscriber()
            
            # Give time for subscription to be established
            await asyncio.sleep(0.1)
            
            # Verify subscriber is active by checking Redis connections
            # This is a simple test - in practice the subscriber would be listening
            
            # Test shutdown
            await stop_cache_subscriber()
            
            # Give time for shutdown to complete
            await asyncio.sleep(0.1)
            
            # Subscriber should be stopped (no direct assertion available,
            # but the function should complete without errors)

    async def test_cache_subscriber_error_handling(
        self, redis_client
    ):
        """Test cache subscriber error handling using real Redis with simulated error conditions."""
        # Create subscriber with real Redis
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            subscriber = CacheSubscriber()
            
            try:
                # Start subscriber
                await subscriber.start()
                
                # Give time for subscription to be established
                await asyncio.sleep(0.1)
                
                # Simulate error condition by disconnecting Redis temporarily
                # This tests real error handling in the subscriber
                original_get_message = subscriber.pubsub.get_message
                
                # Mock get_message to raise an error, then restore
                with patch.object(subscriber.pubsub, 'get_message', side_effect=Exception("Simulated Redis error")):
                    # Give time for error to occur and be handled
                    await asyncio.sleep(0.1)
                
                # Verify subscriber is still running after error
                assert subscriber.pubsub is not None
                assert subscriber.is_running
                
            finally:
                # Stop subscriber
                await subscriber.stop()

    async def test_cache_invalidation_by_prefix(
        self, redis_client
    ):
        """Test cache invalidation by prefix functionality using real Redis."""
        # Create subscriber with real Redis
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            subscriber = CacheSubscriber()
            
            # Setup real test keys in Redis
            test_keys = [
                'apes:pattern:key1',
                'apes:pattern:key2',
                'apes:pattern:key3',
                'other:key1'  # This should not be deleted
            ]
            
            # Add keys to Redis
            for key in test_keys:
                await redis_client.set(key, "test_value")
            
            # Verify all keys exist
            for key in test_keys:
                assert await redis_client.exists(key)
            
            # Test prefix invalidation
            deleted_count = await subscriber._invalidate_by_prefix('apes:pattern:')
            
            # Verify correct keys were deleted
            assert not await redis_client.exists('apes:pattern:key1')
            assert not await redis_client.exists('apes:pattern:key2')
            assert not await redis_client.exists('apes:pattern:key3')
            assert await redis_client.exists('other:key1')  # Should not be deleted
            
            # Verify correct count returned
            assert deleted_count == 3
            
            # Cleanup
            await redis_client.delete('other:key1')

    async def test_ensemble_model_training_emits_invalidation_event(
        self, ml_service, real_db_session, redis_client
    ):
        """Test that ensemble model training emits pattern.invalidate events using real Redis."""
        # Setup training data for ensemble
        training_data = {
            "features": [[0.5, 100, 0.8, 0.3, 0.7]] * 25,
            "effectiveness_scores": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6, 
                                   0.7, 0.8, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                   0.3, 0.4, 0.5, 0.6, 0.7]
        }
        
        # Create a pub/sub subscriber to capture published events
        pubsub = redis_client.pubsub()
        await pubsub.subscribe('pattern.invalidate')
        
        try:
            # Patch both redis_client and MLflow
            with patch('prompt_improver.services.ml_integration.redis_client', redis_client), \
                 patch('prompt_improver.services.ml_integration.mlflow') as mock_mlflow:
                
                # Mock MLflow (external dependency)
                mock_mlflow.active_run.return_value = None
                mock_mlflow.start_run.return_value.__enter__.return_value = None
                mock_mlflow.start_run.return_value.__exit__.return_value = None
                
                mock_run = MagicMock()
                mock_run.info.run_id = "ensemble_run_123"
                mock_mlflow.active_run.return_value = mock_run
                
                result = await ml_service.optimize_ensemble_rules(training_data, real_db_session)
                
                # Verify successful training
                assert result["status"] == "success"
                assert "model_id" in result
                
                # Wait for message to be published and received
                await asyncio.sleep(0.1)
                
                # Get messages and find cache invalidation event
                messages = []
                while True:
                    try:
                        message = await pubsub.get_message(timeout=0.5)
                        if message:
                            messages.append(message)
                        else:
                            break
                    except asyncio.TimeoutError:
                        break
                
                # Find the cache invalidation message
                cache_invalidation_message = None
                for message in messages:
                    if (message['type'] == 'message' and 
                        message['channel'] == b'pattern.invalidate'):
                        try:
                            event_data = json.loads(message['data'])
                            if event_data['type'] == 'model_training_completed':
                                cache_invalidation_message = message
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                assert cache_invalidation_message is not None
                
                # Verify the event details
                event_data = json.loads(cache_invalidation_message['data'])
                assert event_data['type'] == 'model_training_completed'
                assert event_data['model_id'] == result['model_id']
                assert event_data['model_type'] == 'StackingClassifier'
                assert 'apes:pattern:' in event_data['cache_prefixes']
                assert 'rule:' in event_data['cache_prefixes']
                assert 'ml:model:' in event_data['cache_prefixes']
                assert 'timestamp' in event_data
        finally:
            await pubsub.close()

    async def test_cache_invalidation_maintains_data_freshness(
        self, redis_client
    ):
        """Test that cache invalidation maintains data freshness using real Redis."""
        # Create subscriber with real Redis
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            subscriber = CacheSubscriber()
            
            # Setup real cached keys in Redis
            cached_keys = [
                'apes:pattern:rule_123',
                'apes:pattern:model_456',
                'rule:clarity_rule',
                'rule:specificity_rule'
            ]
            
            # Add keys to Redis
            for key in cached_keys:
                await redis_client.set(key, "cached_value")
            
            # Verify all keys exist before invalidation
            for key in cached_keys:
                assert await redis_client.exists(key)
            
            # Simulate invalidation event
            event_data = {
                'type': 'rule_parameters_updated',
                'rule_ids': ['clarity_rule', 'specificity_rule'],
                'cache_prefixes': ['apes:pattern:', 'rule:']
            }
            
            # Test event handling
            await subscriber._handle_invalidate_event({
                'type': 'message',
                'data': json.dumps(event_data)
            })
            
            # Verify all relevant cache keys were invalidated
            for key in cached_keys:
                assert not await redis_client.exists(key)
            
            # Verify data freshness is maintained by checking cache is empty
            # This ensures stale data won't be served
            pattern_keys = []
            async for key in redis_client.scan_iter(match='apes:pattern:*'):
                pattern_keys.append(key)
            
            rule_keys = []
            async for key in redis_client.scan_iter(match='rule:*'):
                rule_keys.append(key)
            
            assert len(pattern_keys) == 0
            assert len(rule_keys) == 0
