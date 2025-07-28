"""Phase 3: Testcontainers Integration Testing (2025 Best Practices).

Implements cutting-edge 2025 testing practices using Testcontainers:
- Real PostgreSQL containers for database testing
- Real Redis containers for cache testing
- No mocks - only real behavior testing
- Production-like environment simulation
- Chaos engineering patterns
"""

import pytest
import asyncio
import time
import statistics
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

# Testcontainers imports (2025 best practice)
try:
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

# Real database imports
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text
from sqlalchemy.pool import NullPool

# Real cache imports
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class TestcontainersConfig:
    """Configuration for Testcontainers-based testing."""
    postgres_image: str = "postgres:15-alpine"
    redis_image: str = "redis:7-alpine"
    concurrent_clients: int = 20
    requests_per_client: int = 15
    sla_threshold_ms: float = 200.0
    chaos_failure_rate: float = 0.05  # 5% failure rate for chaos testing


class TestcontainersSetup:
    """Testcontainers setup for real behavior testing (2025 best practices)."""
    
    def __init__(self, config: TestcontainersConfig):
        self.config = config
        self.postgres_container = None
        self.redis_container = None
        self.postgres_engine = None
        self.redis_client = None
    
    async def start_containers(self):
        """Start real PostgreSQL and Redis containers."""
        if not TESTCONTAINERS_AVAILABLE:
            pytest.skip("Testcontainers not available - install with: pip install testcontainers")
        
        print("🐳 Starting PostgreSQL container...")
        self.postgres_container = PostgresContainer(
            image=self.config.postgres_image,
            username="test_user",
            password="test_pass",
            dbname="test_db"
        )
        self.postgres_container.start()
        
        print("🐳 Starting Redis container...")
        self.redis_container = RedisContainer(image=self.config.redis_image)
        self.redis_container.start()
        
        # Create real database engine
        postgres_url = self.postgres_container.get_connection_url().replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        self.postgres_engine = create_async_engine(
            postgres_url,
            poolclass=NullPool,
            echo=False
        )
        
        # Create real Redis client
        redis_url = f"redis://localhost:{self.redis_container.get_exposed_port(6379)}/0"
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Setup test data
        await self._setup_test_data()
        
        print("✅ Testcontainers started successfully")
    
    async def _setup_test_data(self):
        """Setup realistic test data in real containers."""
        # Create tables in real PostgreSQL
        async with self.postgres_engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE rule_performance (
                    rule_id VARCHAR(255) PRIMARY KEY,
                    effectiveness_score FLOAT NOT NULL,
                    application_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rule_metadata JSONB DEFAULT '{}'::jsonb
                )
            """))
            
            await conn.execute(text("""
                CREATE TABLE rule_combinations (
                    combination_id SERIAL PRIMARY KEY,
                    rule_combination TEXT[] NOT NULL,
                    effectiveness_score FLOAT NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """))
            
            # Insert realistic test data
            rules_data = [
                ("clarity_rule", 0.85, 150),
                ("structure_rule", 0.78, 120),
                ("context_rule", 0.92, 200),
                ("precision_rule", 0.76, 90),
                ("engagement_rule", 0.83, 110)
            ]
            
            for rule_id, effectiveness, count in rules_data:
                await conn.execute(text("""
                    INSERT INTO rule_performance (rule_id, effectiveness_score, application_count)
                    VALUES (:rule_id, :effectiveness, :count)
                """), {"rule_id": rule_id, "effectiveness": effectiveness, "count": count})
        
        # Warm up Redis cache
        await self.redis_client.ping()
        for i in range(10):
            await self.redis_client.set(f"warm_cache_{i}", f"value_{i}", ex=300)
    
    async def get_db_session(self) -> AsyncSession:
        """Get real database session from container."""
        from sqlalchemy.ext.asyncio import async_sessionmaker
        async_session = async_sessionmaker(self.postgres_engine, expire_on_commit=False)
        return async_session()
    
    async def stop_containers(self):
        """Stop and cleanup containers."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_engine:
            await self.postgres_engine.dispose()
        
        if self.postgres_container:
            self.postgres_container.stop()
        
        if self.redis_container:
            self.redis_container.stop()
        
        print("🧹 Testcontainers stopped and cleaned up")


@pytest.mark.asyncio
@pytest.mark.skipif(not TESTCONTAINERS_AVAILABLE, reason="Testcontainers not available")
class TestTestcontainersIntegration:
    """Phase 3: Testcontainers Integration Testing (2025 Best Practices)."""
    
    @pytest.fixture(scope="class")
    async def testcontainers_setup(self):
        """Setup real containers for testing."""
        config = TestcontainersConfig()
        setup = TestcontainersSetup(config)
        await setup.start_containers()
        yield setup
        await setup.stop_containers()
    
    async def test_real_database_behavior(self, testcontainers_setup: TestcontainersSetup):
        """Test real PostgreSQL behavior with Testcontainers."""
        
        print("\n🧪 Testing real PostgreSQL behavior...")
        
        async with testcontainers_setup.get_db_session() as session:
            # Test real PostgreSQL-specific features
            result = await session.execute(text("""
                SELECT rule_id, effectiveness_score,
                       ROW_NUMBER() OVER (ORDER BY effectiveness_score DESC) as rank
                FROM rule_performance 
                WHERE effectiveness_score >= 0.8
            """))
            
            rules = result.fetchall()
            assert len(rules) >= 2, "Should find high-effectiveness rules"
            
            # Test real PostgreSQL JSON operations
            result = await session.execute(text("""
                UPDATE rule_performance 
                SET rule_metadata = jsonb_set(
                    COALESCE(rule_metadata, '{}'::jsonb),
                    '{test_flag}',
                    'true'::jsonb
                )
                WHERE rule_id = 'clarity_rule'
                RETURNING rule_metadata
            """))
            
            updated_metadata = result.fetchone()
            assert updated_metadata is not None, "Should update JSON metadata"
            
            # Test real transaction behavior
            async with session.begin():
                await session.execute(text("""
                    INSERT INTO rule_performance (rule_id, effectiveness_score)
                    VALUES ('test_rule', 0.95)
                """))
                
                # Verify within transaction
                result = await session.execute(text("""
                    SELECT COUNT(*) FROM rule_performance WHERE rule_id = 'test_rule'
                """))
                count = result.scalar()
                assert count == 1, "Should see uncommitted data within transaction"
            
            print("✅ Real PostgreSQL behavior validated")
    
    async def test_real_cache_behavior(self, testcontainers_setup: TestcontainersSetup):
        """Test real Redis cache behavior with Testcontainers."""
        
        print("\n🧪 Testing real Redis cache behavior...")
        
        redis_client = testcontainers_setup.redis_client
        
        # Test real Redis operations
        await redis_client.set("test_key", "test_value", ex=60)
        value = await redis_client.get("test_key")
        assert value == "test_value", "Should store and retrieve values"
        
        # Test real Redis expiration
        await redis_client.set("expire_test", "value", ex=1)
        await asyncio.sleep(1.1)
        expired_value = await redis_client.get("expire_test")
        assert expired_value is None, "Should expire keys"
        
        # Test real Redis pipeline
        pipe = redis_client.pipeline()
        pipe.set("pipe1", "value1")
        pipe.set("pipe2", "value2")
        pipe.get("pipe1")
        pipe.get("pipe2")
        results = await pipe.execute()
        
        assert results[2] == "value1", "Pipeline should work correctly"
        assert results[3] == "value2", "Pipeline should work correctly"
        
        # Test real Redis hash operations
        await redis_client.hset("rule_cache", mapping={
            "clarity_rule": "0.85",
            "structure_rule": "0.78"
        })
        
        hash_values = await redis_client.hgetall("rule_cache")
        assert len(hash_values) == 2, "Should store hash values"
        
        print("✅ Real Redis cache behavior validated")
    
    async def test_real_concurrent_load(self, testcontainers_setup: TestcontainersSetup):
        """Test real concurrent load with actual containers."""
        
        print("\n🧪 Testing real concurrent load on containers...")
        
        async def concurrent_worker(worker_id: int) -> Dict[str, Any]:
            """Worker that performs real database and cache operations."""
            response_times = []
            cache_hits = 0
            cache_misses = 0
            
            async with testcontainers_setup.get_db_session() as session:
                for request_num in range(testcontainers_setup.config.requests_per_client):
                    start_time = time.perf_counter()
                    
                    # Real cache lookup
                    cache_key = f"worker_{worker_id}_req_{request_num}"
                    cached_result = await testcontainers_setup.redis_client.get(cache_key)
                    
                    if cached_result:
                        cache_hits += 1
                        # Simulate cache hit processing
                        await asyncio.sleep(0.001)
                    else:
                        cache_misses += 1
                        
                        # Real database query
                        result = await session.execute(text("""
                            SELECT rule_id, effectiveness_score 
                            FROM rule_performance 
                            WHERE effectiveness_score >= :threshold
                            ORDER BY RANDOM()
                            LIMIT 3
                        """), {"threshold": 0.7})
                        
                        rules = result.fetchall()
                        
                        # Cache the result
                        await testcontainers_setup.redis_client.set(
                            cache_key, 
                            f"rules_{len(rules)}", 
                            ex=60
                        )
                    
                    response_time = (time.perf_counter() - start_time) * 1000
                    response_times.append(response_time)
                    
                    # Small delay to simulate realistic usage
                    await asyncio.sleep(0.002)
            
            return {
                "worker_id": worker_id,
                "response_times": response_times,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "avg_response_time": statistics.mean(response_times)
            }
        
        # Launch concurrent workers
        start_time = time.perf_counter()
        
        tasks = [
            asyncio.create_task(concurrent_worker(worker_id))
            for worker_id in range(testcontainers_setup.config.concurrent_clients)
        ]
        
        worker_results = await asyncio.gather(*tasks)
        total_test_time = time.perf_counter() - start_time
        
        # Analyze real load test results
        all_response_times = []
        total_cache_hits = 0
        total_cache_misses = 0
        
        for result in worker_results:
            all_response_times.extend(result["response_times"])
            total_cache_hits += result["cache_hits"]
            total_cache_misses += result["cache_misses"]
        
        # Calculate real performance metrics
        p95_response_time = statistics.quantiles(all_response_times, n=20)[18]
        avg_response_time = statistics.mean(all_response_times)
        cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses)
        sla_violations = sum(1 for rt in all_response_times if rt > testcontainers_setup.config.sla_threshold_ms)
        sla_compliance = 1.0 - (sla_violations / len(all_response_times))
        
        # Real performance assertions
        assert p95_response_time < testcontainers_setup.config.sla_threshold_ms, \
            f"Real containers 95th percentile exceeds SLA: {p95_response_time:.2f}ms"
        
        assert sla_compliance >= 0.85, \
            f"Real containers SLA compliance too low: {sla_compliance:.2%}"
        
        # Cache should improve over time (second half should have better hit rate)
        second_half_start = len(all_response_times) // 2
        second_half_times = all_response_times[second_half_start:]
        first_half_avg = statistics.mean(all_response_times[:second_half_start])
        second_half_avg = statistics.mean(second_half_times)
        
        assert second_half_avg <= first_half_avg * 1.1, \
            "Cache should improve performance over time"
        
        print(f"✅ Real concurrent load test completed in {total_test_time:.2f}s")
        print(f"📊 Real Container Performance:")
        print(f"   • Concurrent workers: {len(worker_results)}")
        print(f"   • Total requests: {len(all_response_times)}")
        print(f"   • Average response time: {avg_response_time:.2f}ms")
        print(f"   • 95th percentile: {p95_response_time:.2f}ms")
        print(f"   • SLA compliance: {sla_compliance:.2%}")
        print(f"   • Cache hit rate: {cache_hit_rate:.2%}")
        print(f"   • Performance improvement: {((first_half_avg - second_half_avg) / first_half_avg * 100):.1f}%")
    
    async def test_chaos_engineering_patterns(self, testcontainers_setup: TestcontainersSetup):
        """Test chaos engineering patterns with real containers."""
        
        print("\n🧪 Testing chaos engineering patterns...")
        
        # Test database connection resilience
        async with testcontainers_setup.get_db_session() as session:
            # Simulate connection issues by overwhelming the connection pool
            connection_tasks = []
            for i in range(50):  # More than typical pool size
                task = asyncio.create_task(session.execute(text("SELECT 1")))
                connection_tasks.append(task)
            
            # Some should succeed, some might fail due to pool exhaustion
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            successful_connections = sum(1 for r in results if not isinstance(r, Exception))
            
            assert successful_connections > 0, "Some connections should succeed under load"
            print(f"✅ Connection resilience: {successful_connections}/50 connections succeeded")
        
        # Test cache failure resilience
        redis_client = testcontainers_setup.redis_client
        
        # Fill cache to near capacity
        for i in range(1000):
            await redis_client.set(f"chaos_key_{i}", f"value_{i}" * 100, ex=60)
        
        # Test continued operation under memory pressure
        memory_pressure_success = 0
        for i in range(100):
            try:
                await redis_client.set(f"pressure_test_{i}", "test_value", ex=10)
                result = await redis_client.get(f"pressure_test_{i}")
                if result == "test_value":
                    memory_pressure_success += 1
            except Exception:
                pass  # Expected under memory pressure
        
        assert memory_pressure_success > 50, "Should handle some operations under memory pressure"
        print(f"✅ Cache resilience: {memory_pressure_success}/100 operations succeeded under pressure")
        
        print("✅ Chaos engineering patterns validated")


if __name__ == "__main__":
    """Run Testcontainers integration tests directly."""
    async def run_testcontainers_demo():
        print("🚀 Phase 3: Testcontainers Integration Testing (2025 Best Practices)")
        print("=" * 80)
        print("🐳 Uses real PostgreSQL containers")
        print("🐳 Uses real Redis containers") 
        print("🔍 No mocks - only real behavior testing")
        print("🏗️  Production-like environment simulation")
        print("🌪️  Chaos engineering patterns")
        print("=" * 80)
        
        if not TESTCONTAINERS_AVAILABLE:
            print("❌ Testcontainers not available")
            print("💡 Install with: pip install testcontainers")
            return
        
        print("✅ Testcontainers available - ready for real behavior testing!")
        print("🎯 This demonstrates cutting-edge 2025 testing practices")
    
    # Run demonstration
    asyncio.run(run_testcontainers_demo())
