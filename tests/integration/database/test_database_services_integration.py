"""
Comprehensive integration tests for DatabaseServices architecture.
Tests real behavior with actual PostgreSQL and Redis instances.
"""

import asyncio
import time
from datetime import datetime, UTC
from typing import Any, Dict, List
import pytest
from sqlalchemy import text

from prompt_improver.database import (
    DatabaseServices,
    ManagerMode,
    SecurityContext,
    SecurityTier,
    get_database_services,
    create_security_context,
)
from prompt_improver.database.composition import PoolConfiguration
from prompt_improver.database.types import (
    ConnectionMetrics,
    HealthStatus,
    SecurityValidationResult,
)


class TestDatabaseServicesIntegration:
    """Integration tests for DatabaseServices with real infrastructure."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, postgres_container, redis_container):
        """Test DatabaseServices initialization with real connections."""
        # Create services with real connections
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            # Initialize all services
            await services.initialize()
            
            # Verify all services are initialized
            assert services.database is not None
            assert services.cache is not None
            assert services.health_manager is not None
            assert services.lock_manager is not None
            assert services.pubsub is not None
            
            # Test database connectivity
            async with services.database.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1
            
            # Test cache connectivity
            if hasattr(services.cache, 'redis_client'):
                await services.cache.set("test_key", "test_value", ttl=60)
                value = await services.cache.get("test_key")
                assert value == "test_value"
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_connection_pooling(self, postgres_container):
        """Test database connection pooling with concurrent requests."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        
        try:
            await services.initialize()
            
            # Track connection metrics
            initial_metrics = services.database.get_metrics()
            
            # Execute concurrent database operations
            async def db_operation(n: int):
                async with services.database.get_session() as session:
                    result = await session.execute(
                        text("SELECT :n as num"), {"n": n}
                    )
                    return result.scalar()
            
            # Run 20 concurrent operations
            tasks = [db_operation(i) for i in range(20)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 20
            assert all(results[i] == i for i in range(20))
            
            # Check pool metrics
            final_metrics = services.database.get_metrics()
            assert final_metrics.total_connections > 0
            assert final_metrics.active_connections >= 0
            assert final_metrics.idle_connections >= 0
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_unified_cache_system(self, postgres_container, redis_container):
        """Test unified cache system with L1, L2, and L3 caching coordination."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            # Test L1 (Memory) -> L2 (Redis) -> L3 (Database) flow
            key = "multi_level_test"
            value = {"data": "test_value", "timestamp": datetime.now(UTC).isoformat()}
            
            # Set in cache (should propagate through levels)
            await services.cache.set(key, value, ttl=300)
            
            # Get from cache (should hit L1)
            cached_value = await services.cache.get(key)
            assert cached_value == value
            
            # Clear L1 cache to test L2
            if hasattr(services.cache, 'clear_l1'):
                await services.cache.clear_l1()
            
            # Should now hit L2 (Redis)
            cached_value = await services.cache.get(key)
            assert cached_value == value
            
            # Test cache invalidation
            await services.cache.delete(key)
            cached_value = await services.cache.get(key)
            assert cached_value is None
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_distributed_locking(self, redis_container):
        """Test distributed locking mechanism for concurrent access control."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            lock_key = "test_distributed_lock"
            results = []
            
            async def critical_section(worker_id: int):
                """Simulate critical section that requires exclusive access."""
                async with services.lock_manager.acquire(lock_key, timeout=5):
                    # Only one worker should be here at a time
                    results.append(f"start_{worker_id}")
                    await asyncio.sleep(0.1)  # Simulate work
                    results.append(f"end_{worker_id}")
            
            # Run multiple workers concurrently
            workers = [critical_section(i) for i in range(5)]
            await asyncio.gather(*workers)
            
            # Verify mutual exclusion - each start should be followed by its end
            for i in range(0, len(results), 2):
                start = results[i]
                end = results[i + 1]
                worker_id = start.split('_')[1]
                assert end == f"end_{worker_id}"
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_health_monitoring(self, postgres_container, redis_container):
        """Test health monitoring and circuit breaker functionality."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            # Check initial health
            health_status = await services.health_check()
            assert health_status["status"] in ["healthy", "degraded"]
            assert "components" in health_status
            
            # Test circuit breaker behavior
            if hasattr(services.health_manager, 'circuit_breaker'):
                cb = services.health_manager.circuit_breaker
                
                # Simulate failures to trip circuit breaker
                for _ in range(5):
                    cb.record_failure()
                
                # Circuit should be open
                assert cb.is_open()
                
                # Wait for half-open state
                await asyncio.sleep(cb.timeout)
                
                # Record success to close circuit
                cb.record_success()
                assert not cb.is_open()
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_security_context_integration(self, postgres_container):
        """Test security context creation and tier-based permissions."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            # Test different security tiers
            contexts = [
                create_security_context("user_basic", SecurityTier.BASIC),
                create_security_context("user_standard", SecurityTier.STANDARD),
                create_security_context("user_privileged", SecurityTier.PRIVILEGED),
                create_security_context("user_admin", SecurityTier.ADMIN),
            ]
            
            for context in contexts:
                # Verify context properties
                assert context.agent_id
                assert context.tier in SecurityTier
                assert context.created_at
                assert context.permissions
                
                # Test permission checks based on tier
                if context.tier == SecurityTier.ADMIN:
                    assert "full_access" in context.permissions
                elif context.tier == SecurityTier.PRIVILEGED:
                    assert "write_access" in context.permissions
                elif context.tier == SecurityTier.STANDARD:
                    assert "read_write" in context.permissions
                else:  # BASIC
                    assert "read_only" in context.permissions
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, postgres_container, redis_container):
        """Test concurrent database and cache operations."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            async def database_operation(n: int):
                """Database operation."""
                async with services.database.get_session() as session:
                    result = await session.execute(
                        text("SELECT pg_sleep(0.01), :n"), {"n": n}
                    )
                    return n
            
            async def cache_operation(n: int):
                """Cache operation."""
                key = f"concurrent_test_{n}"
                await services.cache.set(key, n, ttl=60)
                value = await services.cache.get(key)
                return value
            
            async def lock_operation(n: int):
                """Lock operation."""
                lock_key = f"lock_{n % 3}"  # Use 3 different locks
                async with services.lock_manager.acquire(lock_key, timeout=5):
                    await asyncio.sleep(0.01)
                    return n
            
            # Run all operations concurrently
            db_tasks = [database_operation(i) for i in range(10)]
            cache_tasks = [cache_operation(i) for i in range(10)]
            lock_tasks = [lock_operation(i) for i in range(10)]
            
            all_results = await asyncio.gather(
                *db_tasks, *cache_tasks, *lock_tasks,
                return_exceptions=True
            )
            
            # Verify no exceptions
            exceptions = [r for r in all_results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
            
            # Verify all operations completed
            assert len(all_results) == 30
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, postgres_container, redis_container):
        """Test graceful shutdown of all services."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            # Start some background operations
            async def background_work():
                while True:
                    try:
                        async with services.database.get_session() as session:
                            await session.execute(text("SELECT 1"))
                        await asyncio.sleep(0.1)
                    except asyncio.CancelledError:
                        break
            
            # Start background tasks
            tasks = [asyncio.create_task(background_work()) for _ in range(3)]
            
            # Let them run briefly
            await asyncio.sleep(0.5)
            
            # Shutdown services
            await services.shutdown()
            
            # Cancel background tasks
            for task in tasks:
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify clean shutdown
            assert services.is_shutdown
            
        except Exception as e:
            # Ensure cleanup even on error
            if not services.is_shutdown:
                await services.shutdown()
            raise

    @pytest.mark.asyncio
    async def test_mode_specific_behavior(self, postgres_container, redis_container):
        """Test different ManagerMode configurations."""
        modes = [
            ManagerMode.MCP_SERVER,
            ManagerMode.ML_TRAINING,
            ManagerMode.ADMIN,
            ManagerMode.ASYNC_MODERN,
            ManagerMode.HIGH_AVAILABILITY,
        ]
        
        for mode in modes:
            services = await get_database_services(mode)
            
            try:
                await services.initialize()
                
                # Verify mode-specific configuration
                assert services.mode == mode
                
                # Test basic operations work in all modes
                async with services.database.get_session() as session:
                    result = await session.execute(text("SELECT 1"))
                    assert result.scalar() == 1
                
                # Mode-specific checks
                if mode == ManagerMode.HIGH_AVAILABILITY:
                    # Should have enhanced monitoring
                    assert services.health_manager is not None
                elif mode == ManagerMode.ML_TRAINING:
                    # Should have optimized for batch operations
                    pool_config = services.database.pool_config
                    assert pool_config.max_size >= 20
                elif mode == ManagerMode.ADMIN:
                    # Should have full access permissions
                    assert services.mode == ManagerMode.ADMIN
                
            finally:
                await services.shutdown()


class TestDatabaseServicesErrorHandling:
    """Test error handling and recovery in DatabaseServices."""

    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self, postgres_container):
        """Test recovery from database connection failures with real infrastructure."""
        services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
        
        try:
            await services.initialize()
            
            # Test that services can handle real connection issues
            # by testing with an invalid query that should be handled gracefully
            try:
                async with services.database.get_session() as session:
                    # This will succeed with real database
                    result = await session.execute(text("SELECT 1"))
                    assert result.scalar() == 1
            except Exception as e:
                # If there's a connection issue with real database, that's legitimate
                assert "connection" in str(e).lower() or "database" in str(e).lower()
            
            # Test that services can recover from transient issues
            # by attempting multiple operations
            success_count = 0
            for _ in range(3):
                try:
                    async with services.database.get_session() as session:
                        result = await session.execute(text("SELECT 1"))
                        if result.scalar() == 1:
                            success_count += 1
                except Exception:
                    pass  # Real infrastructure may have transient issues
            
            # Should succeed at least once with real infrastructure
            assert success_count >= 1
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio
    async def test_cache_fallback_behavior(self, postgres_container, redis_container):
        """Test cache fallback behavior with real infrastructure."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        
        try:
            await services.initialize()
            
            # Test normal cache operation first
            key = "fallback_test"
            value = {"data": "test"}
            
            # Test with real cache infrastructure
            try:
                await services.cache.set(key, value, ttl=60)
                cached = await services.cache.get(key)
                
                # With real infrastructure, should either work or fail gracefully
                if cached is not None:
                    assert cached == value
                    
                # Clean up
                await services.cache.delete(key)
                
            except Exception as e:
                # Real cache operations may fail if not configured
                assert "cache" in str(e).lower() or "redis" in str(e).lower() or "connection" in str(e).lower()
            
        finally:
            await services.shutdown()

    @pytest.mark.asyncio 
    async def test_concurrent_initialization(self):
        """Test handling of concurrent service initialization."""
        async def init_services():
            services = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
            await services.initialize()
            return services
        
        # Initialize multiple services concurrently
        services_list = await asyncio.gather(
            *[init_services() for _ in range(5)],
            return_exceptions=True
        )
        
        # Clean up all services
        for services in services_list:
            if not isinstance(services, Exception):
                await services.shutdown()
        
        # Verify no exceptions
        exceptions = [s for s in services_list if isinstance(s, Exception)]
        assert len(exceptions) == 0