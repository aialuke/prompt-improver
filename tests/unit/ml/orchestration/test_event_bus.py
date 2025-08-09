"""
Tests for Event Bus system.
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
import pytest
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_bus import EventBus, EventSubscription
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent

class TestEventBus:
    """Test suite for Event Bus."""

    @pytest.fixture
    async def event_bus(self):
        """Create event bus instance for testing."""
        config = OrchestratorConfig(event_bus_buffer_size=100, event_handler_timeout=5)
        bus = EventBus(config)
        await bus.initialize()
        yield bus
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_bus_initialization(self, event_bus):
        """Test event bus initialization."""
        assert event_bus._is_initialized is True
        assert event_bus.subscribers == {}
        assert len(event_bus.event_history) == 0

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self, event_bus):
        """Test event subscription and emission."""
        events_received = []

        def event_handler(event: MLEvent):
            events_received.append(event)
        subscription = await event_bus.subscribe(EventType.TRAINING_STARTED, event_handler)
        assert subscription is not None
        assert subscription.subscription_id is not None
        assert subscription.event_type == EventType.TRAINING_STARTED
        event = MLEvent(event_type=EventType.TRAINING_STARTED, source='test_component', data={'workflow_id': 'test-123', 'model': 'test_model'})
        await event_bus.emit(event)
        await asyncio.sleep(0.1)
        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.TRAINING_STARTED
        assert events_received[0].data['workflow_id'] == 'test-123'

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers for the same event type."""
        events_received_1 = []
        events_received_2 = []

        def handler_1(event: MLEvent):
            events_received_1.append(event)

        def handler_2(event: MLEvent):
            events_received_2.append(event)
        await event_bus.subscribe(EventType.WORKFLOW_STARTED, handler_1)
        await event_bus.subscribe(EventType.WORKFLOW_STARTED, handler_2)
        event = MLEvent(event_type=EventType.WORKFLOW_STARTED, source='test', data={'workflow_id': 'multi-test'})
        await event_bus.emit(event)
        await asyncio.sleep(0.1)
        assert len(events_received_1) == 1
        assert len(events_received_2) == 1
        assert events_received_1[0].data['workflow_id'] == 'multi-test'
        assert events_received_2[0].data['workflow_id'] == 'multi-test'

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test event unsubscription."""
        events_received = []

        def event_handler(event: MLEvent):
            events_received.append(event)
        subscription = await event_bus.subscribe(EventType.TRAINING_COMPLETED, event_handler)
        event1 = MLEvent(event_type=EventType.TRAINING_COMPLETED, source='test', data={'message': 'first'})
        await event_bus.emit(event1)
        await asyncio.sleep(0.1)
        assert len(events_received) == 1
        success = await event_bus.unsubscribe(subscription.subscription_id)
        assert success is True
        event2 = MLEvent(event_type=EventType.TRAINING_COMPLETED, source='test', data={'message': 'second'})
        await event_bus.emit(event2)
        await asyncio.sleep(0.1)
        assert len(events_received) == 1
        assert events_received[0].data['message'] == 'first'

    @pytest.mark.asyncio
    async def test_async_event_handler(self, event_bus):
        """Test async event handlers."""
        events_received = []

        async def async_event_handler(event: MLEvent):
            await asyncio.sleep(0.01)
            events_received.append(event)
        await event_bus.subscribe(EventType.EVALUATION_STARTED, async_event_handler)
        event = MLEvent(event_type=EventType.EVALUATION_STARTED, source='test', data={'async_test': True})
        await event_bus.emit(event)
        await asyncio.sleep(0.1)
        assert len(events_received) == 1
        assert events_received[0].data['async_test'] is True

    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history tracking."""
        for i in range(5):
            event = MLEvent(event_type=EventType.WORKFLOW_STARTED, source='test', data={'sequence': i})
            await event_bus.emit(event)
        await asyncio.sleep(0.1)
        history = await event_bus.get_event_history()
        assert len(history) == 5
        for i, event in enumerate(history):
            assert event.data['sequence'] == i

    @pytest.mark.asyncio
    async def test_event_filtering(self, event_bus):
        """Test event filtering by type."""
        training_event = MLEvent(EventType.TRAINING_STARTED, 'test', {'type': 'training'})
        evaluation_event = MLEvent(EventType.EVALUATION_STARTED, 'test', {'type': 'evaluation'})
        deployment_event = MLEvent(EventType.DEPLOYMENT_STARTED, 'test', {'type': 'deployment'})
        await event_bus.emit(training_event)
        await event_bus.emit(evaluation_event)
        await event_bus.emit(deployment_event)
        await asyncio.sleep(0.1)
        training_history = await event_bus.get_event_history(event_type=EventType.TRAINING_STARTED)
        assert len(training_history) == 1
        assert training_history[0].data['type'] == 'training'
        evaluation_history = await event_bus.get_event_history(event_type=EventType.EVALUATION_STARTED)
        assert len(evaluation_history) == 1
        assert evaluation_history[0].data['type'] == 'evaluation'

    @pytest.mark.asyncio
    async def test_handler_error_handling(self, event_bus):
        """Test error handling in event handlers."""
        events_received = []

        def failing_handler(event: MLEvent):
            raise ValueError('Handler error')

        def working_handler(event: MLEvent):
            events_received.append(event)
        await event_bus.subscribe(EventType.WORKFLOW_FAILED, failing_handler)
        await event_bus.subscribe(EventType.WORKFLOW_FAILED, working_handler)
        event = MLEvent(event_type=EventType.WORKFLOW_FAILED, source='test', data={'error_test': True})
        await event_bus.emit(event)
        await asyncio.sleep(0.1)
        assert len(events_received) == 1
        assert events_received[0].data['error_test'] is True

    @pytest.mark.asyncio
    async def test_event_statistics(self, event_bus):
        """Test event statistics collection."""
        events_to_emit = [MLEvent(EventType.TRAINING_STARTED, 'test', {}), MLEvent(EventType.TRAINING_COMPLETED, 'test', {}), MLEvent(EventType.EVALUATION_STARTED, 'test', {}), MLEvent(EventType.TRAINING_STARTED, 'test', {})]
        for event in events_to_emit:
            await event_bus.emit(event)
        await asyncio.sleep(0.1)
        stats = await event_bus.get_statistics()
        assert stats['total_events_emitted'] == 4
        assert stats['total_subscribers'] >= 0
        assert 'events_by_type' in stats
        assert stats['events_by_type'][EventType.TRAINING_STARTED.value] == 2
        assert stats['events_by_type'][EventType.TRAINING_COMPLETED.value] == 1
        assert stats['events_by_type'][EventType.EVALUATION_STARTED.value] == 1

    @pytest.mark.asyncio
    async def test_bulk_event_emission(self, event_bus):
        """Test bulk event emission performance."""
        events_received = []

        def bulk_handler(event: MLEvent):
            events_received.append(event)
        await event_bus.subscribe(EventType.TRAINING_PROGRESS, bulk_handler)
        events_to_emit = []
        for i in range(50):
            event = MLEvent(event_type=EventType.TRAINING_PROGRESS, source='bulk_test', data={'progress': i / 50.0})
            events_to_emit.append(event_bus.emit(event))
        await asyncio.gather(*events_to_emit)
        await asyncio.sleep(0.2)
        assert len(events_received) == 50
        progress_values = [event.data['progress'] for event in events_received]
        expected_values = [i / 50.0 for i in range(50)]
        progress_values.sort()
        expected_values.sort()
        assert progress_values == expected_values

class TestEventSubscription:
    """Test suite for EventSubscription."""

    def test_subscription_creation(self):
        """Test event subscription creation."""
        subscription = EventSubscription(subscription_id='sub-123', event_type=EventType.TRAINING_STARTED, handler=lambda x: None, subscriber_info='test_subscriber')
        assert subscription.subscription_id == 'sub-123'
        assert subscription.event_type == EventType.TRAINING_STARTED
        assert subscription.subscriber_info == 'test_subscriber'
        assert subscription.created_at is not None

    def test_subscription_serialization(self):
        """Test subscription to/from dict conversion."""
        subscription = EventSubscription(subscription_id='sub-456', event_type=EventType.EVALUATION_COMPLETED, handler=lambda x: None, subscriber_info='test_subscriber')
        subscription_dict = subscription.to_dict()
        assert subscription_dict['subscription_id'] == 'sub-456'
        assert subscription_dict['event_type'] == EventType.EVALUATION_COMPLETED.value
        assert subscription_dict['subscriber_info'] == 'test_subscriber'
        assert 'created_at' in subscription_dict

class TestMLEvent:
    """Test suite for MLEvent."""

    def test_event_creation_with_correlation(self):
        """Test event creation with correlation ID."""
        correlation_id = 'correlation-123'
        event = MLEvent(event_type=EventType.DEPLOYMENT_COMPLETED, source='deployment_service', data={'deployment_id': 'deploy-456'}, correlation_id=correlation_id)
        assert event.event_type == EventType.DEPLOYMENT_COMPLETED
        assert event.source == 'deployment_service'
        assert event.data['deployment_id'] == 'deploy-456'
        assert event.correlation_id == correlation_id
        assert event.timestamp is not None
        assert event.event_id is not None

    def test_event_metadata(self):
        """Test event metadata properties."""
        event = MLEvent(event_type=EventType.RESOURCE_ALLOCATED, source='resource_manager', data={'resource_type': 'CPU', 'amount': 2.0})
        assert event.timestamp <= datetime.now(timezone.utc)
        assert len(event.event_id) > 0
        assert event.event_id.startswith('evt_')

    def test_event_with_tags(self):
        """Test event creation with tags."""
        event = MLEvent(event_type=EventType.TRAINING_FAILED, source='training_coordinator', data={'error': 'Out of memory'}, tags=['error', 'memory', 'training'])
        assert event.tags == ['error', 'memory', 'training']
        event_dict = event.to_dict()
        assert event_dict['tags'] == ['error', 'memory', 'training']
if __name__ == '__main__':

    async def smoke_test():
        """Basic smoke test for event bus."""
        print('Running Event Bus smoke test...')
        config = OrchestratorConfig()
        bus = EventBus(config)
        try:
            await bus.initialize()
            print('✓ Event bus initialized successfully')
            events_received = []

            def test_handler(event):
                events_received.append(event)
            subscription = await bus.subscribe(EventType.TRAINING_STARTED, test_handler)
            print(f'✓ Subscribed to events: {subscription.subscription_id}')
            event = MLEvent(event_type=EventType.TRAINING_STARTED, source='smoke_test', data={'test': True})
            await bus.emit(event)
            await asyncio.sleep(0.1)
            print(f'✓ Event emitted and received: {len(events_received)} events')
            stats = await bus.get_statistics()
            print(f"✓ Event statistics: {stats['total_events_emitted']} total events")
            success = await bus.unsubscribe(subscription.subscription_id)
            print(f'✓ Unsubscribed successfully: {success}')
            print('✓ All basic tests passed!')
        except Exception as e:
            print(f'✗ Test failed: {e}')
            raise
        finally:
            await bus.shutdown()
            print('✓ Event bus shut down gracefully')
    asyncio.run(smoke_test())
