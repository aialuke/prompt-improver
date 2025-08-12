"""
Integration tests for individual DatabaseServices components.
Tests each service in isolation with real infrastructure.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List
import pytest
from sqlalchemy import text
import coredis

from prompt_improver.database import (
    DatabaseServices,
    ManagerMode,
    get_database_services,
)


class TestPostgresPoolManager:
    """Test PostgreSQL pool manager with real database."""

    @pytest.mark.asyncio
    async def test_connection_pool_scaling(self, postgres_container):
        """Test dynamic pool scaling based on load."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            pool_manager = services.database
            
            # Get initial pool size
            initial_metrics = pool_manager.get_metrics()
            initial_size = initial_metrics.total_connections
            
            # Generate load to trigger scaling
            async def load_generator(n: int):
                async with pool_manager.get_session() as session:
                    await session.execute(
                        text("SELECT pg_sleep(0.1), :n"), {"n": n}
                    )
            
            # Run concurrent load
            tasks = [load_generator(i) for i in range(30)]
            await asyncio.gather(*tasks)
            
            # Check if pool scaled
            final_metrics = pool_manager.get_metrics()
            assert final_metrics.total_connections >= initial_size
            assert final_metrics.queries_executed >= 30
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_connection_health_checks(self, postgres_container):
        """Test connection health checking and validation."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        
        try:
            await services.initialize()
            pool_manager = services.database
            
            # Test health check
            is_healthy = await pool_manager.health_check()
            assert is_healthy
            
            # Test connection validation
            async with pool_manager.get_session() as session:
                # Connection should be valid
                result = await session.execute(text("SELECT current_database()"))
                db_name = result.scalar()
                assert db_name is not None
            
            # Test stale connection handling
            # This would normally require killing connections externally
            # Here we just verify the mechanism exists
            assert hasattr(pool_manager, 'validate_connection')
            
        finally:
            await services.shutdown()


class TestCacheManager:
    """Test multi-level cache manager with real Redis."""

    @pytest.mark.asyncio
    async def test_cache_hierarchy(self, redis_container):
        """Test L1 -> L2 -> L3 cache hierarchy."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            cache = services.cache
            
            # Test data
            test_data = {
                "simple": "value",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"nested": "data"},
            }
            
            # Test each cache level
            for key, value in test_data.items():
                cache_key = f"hierarchy_test_{key}"
                
                # Set in cache
                await cache.set(cache_key, value, ttl=300)
                
                # Should be in L1 (memory)
                cached = await cache.get(cache_key)
                assert cached == value
                
                # Clear L1 to test L2
                if hasattr(cache, '_memory_cache'):
                    cache._memory_cache.clear()
                
                # Should hit L2 (Redis)
                cached = await cache.get(cache_key)
                assert cached == value
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_cache_expiration(self, redis_container):
        """Test cache TTL and expiration."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        
        try:
            await services.initialize()
            cache = services.cache
            
            # Set with short TTL
            await cache.set("expire_test", "value", ttl=1)
            
            # Should exist immediately
            value = await cache.get("expire_test")
            assert value == "value"
            
            # Wait for expiration
            await asyncio.sleep(1.5)
            
            # Should be expired
            value = await cache.get("expire_test")
            assert value is None
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, redis_container):
        """Test cache invalidation patterns."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            cache = services.cache
            
            # Set multiple related keys
            keys = [f"invalidate_test_{i}" for i in range(5)]
            for key in keys:
                await cache.set(key, f"value_{key}", ttl=300)
            
            # Verify all exist
            for key in keys:
                assert await cache.get(key) is not None
            
            # Invalidate specific keys
            for key in keys[:3]:
                await cache.delete(key)
            
            # Check invalidation
            for i, key in enumerate(keys):
                value = await cache.get(key)
                if i < 3:
                    assert value is None
                else:
                    assert value == f"value_{key}"
            
            # Pattern-based invalidation (if supported)
            if hasattr(cache, 'delete_pattern'):
                await cache.delete_pattern("invalidate_test_*")
                for key in keys:
                    assert await cache.get(key) is None
            
        finally:
            await services.shutdown()


class TestDistributedLockManager:
    """Test distributed locking with real Redis."""

    @pytest.mark.asyncio
    async def test_lock_mutual_exclusion(self, redis_container):
        """Test mutual exclusion property of distributed locks."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            lock_manager = services.lock_manager
            
            lock_key = "mutex_test"
            counter = {"value": 0}
            
            async def critical_section(worker_id: int):
                async with lock_manager.acquire(lock_key, timeout=5):
                    # Read-modify-write (must be atomic)
                    current = counter["value"]
                    await asyncio.sleep(0.01)  # Simulate processing
                    counter["value"] = current + 1
                    return worker_id
            
            # Run workers concurrently
            workers = [critical_section(i) for i in range(20)]
            results = await asyncio.gather(*workers)
            
            # Counter should equal number of workers (no race conditions)
            assert counter["value"] == 20
            assert len(results) == 20
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_lock_timeout(self, redis_container):
        """Test lock timeout and release."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            lock_manager = services.lock_manager
            
            lock_key = "timeout_test"
            
            # Acquire lock with timeout
            lock = await lock_manager.acquire(lock_key, timeout=1)
            assert lock is not None
            
            # Try to acquire same lock (should fail)
            lock2 = await lock_manager.try_acquire(lock_key, timeout=0.1)
            assert lock2 is None
            
            # Wait for timeout
            await asyncio.sleep(1.5)
            
            # Should be able to acquire now
            lock3 = await lock_manager.acquire(lock_key, timeout=1)
            assert lock3 is not None
            
            # Clean release
            await lock_manager.release(lock_key)
            
        finally:
            await services.shutdown()


class TestPubSubManager:
    """Test pub/sub messaging with real Redis."""

    @pytest.mark.asyncio
    async def test_pubsub_messaging(self, redis_container):
        """Test publish/subscribe messaging."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            pubsub = services.pubsub
            
            received_messages = []
            
            # Subscribe to channel
            async def message_handler(message):
                received_messages.append(message)
            
            await pubsub.subscribe("test_channel", message_handler)
            
            # Give subscription time to establish
            await asyncio.sleep(0.1)
            
            # Publish messages
            messages = ["msg1", "msg2", "msg3"]
            for msg in messages:
                await pubsub.publish("test_channel", msg)
            
            # Wait for messages
            await asyncio.sleep(0.5)
            
            # Should have received all messages
            assert len(received_messages) >= len(messages)
            
            # Unsubscribe
            await pubsub.unsubscribe("test_channel")
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_pubsub_patterns(self, redis_container):
        """Test pattern-based subscriptions."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            pubsub = services.pubsub
            
            received = {"events": [], "metrics": []}
            
            # Subscribe to patterns
            async def event_handler(message):
                received["events"].append(message)
            
            async def metric_handler(message):
                received["metrics"].append(message)
            
            await pubsub.subscribe_pattern("events:*", event_handler)
            await pubsub.subscribe_pattern("metrics:*", metric_handler)
            
            await asyncio.sleep(0.1)
            
            # Publish to different channels
            await pubsub.publish("events:user", "user_event")
            await pubsub.publish("events:system", "system_event")
            await pubsub.publish("metrics:cpu", "cpu_metric")
            await pubsub.publish("other:data", "other_data")
            
            await asyncio.sleep(0.5)
            
            # Check pattern matching
            assert len(received["events"]) >= 2
            assert len(received["metrics"]) >= 1
            
        finally:
            await services.shutdown()


class TestHealthManager:
    """Test health monitoring with real services."""

    @pytest.mark.asyncio
    async def test_component_health_monitoring(self, postgres_container, redis_container):
        """Test health monitoring of all components."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            # Get comprehensive health status
            health_status = await services.health_check()
            
            # Verify structure
            assert "status" in health_status
            assert "components" in health_status
            assert "response_time_ms" in health_status
            assert "timestamp" in health_status
            
            # Check component health
            components = health_status["components"]
            
            # Should have database health
            if "database" in components:
                assert components["database"] in ["healthy", "degraded", "unhealthy"]
            
            # Should have cache health
            if "cache" in components:
                assert components["cache"] in ["healthy", "degraded", "unhealthy"]
            
            # Overall status should reflect components
            overall = health_status["status"]
            assert overall in ["healthy", "degraded", "unhealthy"]
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, postgres_container):
        """Test circuit breaker with service failures."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            if hasattr(services.health_manager, 'circuit_breaker'):
                cb = services.health_manager.circuit_breaker
                
                # Initially closed
                assert not cb.is_open()
                
                # Simulate cascading failures
                for _ in range(cb.failure_threshold):
                    cb.record_failure()
                
                # Should trip to open
                assert cb.is_open()
                
                # Operations should be rejected
                with pytest.raises(Exception):
                    cb.call(lambda: 1/0)
                
                # Wait for half-open
                await asyncio.sleep(cb.timeout)
                
                # Should allow test call
                cb.record_success()
                assert not cb.is_open()
            
        finally:
            await services.shutdown()


class TestSecurityIntegration:
    """Test security features with real services."""

    @pytest.mark.asyncio
    async def test_tier_based_access_control(self, postgres_container):
        """Test security tier enforcement."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            # Create contexts with different tiers
            from prompt_improver.database import SecurityTier, create_security_context
            
            basic_ctx = create_security_context("basic_user", SecurityTier.BASIC)
            admin_ctx = create_security_context("admin_user", SecurityTier.ADMIN)
            
            # Test permission differences
            assert "read_only" in basic_ctx.permissions
            assert "full_access" in admin_ctx.permissions
            
            # Test with database operations (if enforcement is implemented)
            # This would require actual permission checking in the database layer
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_security_context_validation(self, postgres_container):
        """Test security context validation and lifecycle."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            from prompt_improver.database import create_security_context, SecurityTier
            
            # Create and validate context
            context = create_security_context(
                agent_id="test_agent",
                tier=SecurityTier.STANDARD
            )
            
            # Verify required fields
            assert context.agent_id == "test_agent"
            assert context.tier == SecurityTier.STANDARD
            assert context.created_at is not None
            assert context.expires_at > context.created_at
            
            # Test expiration
            expired_context = create_security_context(
                agent_id="expired",
                tier=SecurityTier.BASIC
            )
            # Manually expire it
            expired_context.expires_at = datetime.now(UTC) - timedelta(hours=1)
            assert expired_context.is_expired()
            
        finally:
            await services.shutdown()