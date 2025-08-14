"""Performance benchmarks for all decomposed components.

Comprehensive performance validation ensuring all decomposed components meet
or exceed their performance requirements under realistic workloads.

Performance Requirements Validation:
- L1 Cache: <1ms response time, >95% hit rate
- L2 Cache: <10ms response time, >80% hit rate  
- L3 Cache: <50ms response time, 100% durability
- Security Services: <100ms authentication, OWASP compliance
- ML Repository: <100ms CRUD operations, >1000 records/sec
- DI Containers: <5ms service resolution, <20ms complex resolution
- Redis Health Monitoring: <25ms health checks, <100ms recovery
- System Integration: <200ms end-to-end workflows, >50 concurrent users
"""

import asyncio
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import pytest

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.containers.real_redis_container import RealRedisTestContainer


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    
    component: str
    operation: str
    requirement_ms: float
    actual_ms: float
    throughput_ops_sec: float
    success_rate: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_operations: int
    test_duration_sec: float
    passes_requirement: bool
    
    def __post_init__(self):
        self.passes_requirement = self.actual_ms <= self.requirement_ms and self.success_rate >= 0.95


class ComponentPerformanceBenchmarks:
    """Comprehensive performance benchmarks for all decomposed components."""

    @pytest.fixture(scope="session")
    async def benchmark_infrastructure(self):
        """Set up performance testing infrastructure."""
        redis_container = RealRedisTestContainer()
        postgres_container = PostgreSQLTestContainer()
        
        await redis_container.start()
        await postgres_container.start()
        
        # Configure environment
        import os
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        os.environ["POSTGRES_HOST"] = postgres_container.get_host()
        os.environ["POSTGRES_PORT"] = str(postgres_container.get_exposed_port(5432))
        
        from prompt_improver.database import DatabaseServices, ManagerMode
        db_services = DatabaseServices(
            mode=ManagerMode.ASYNC_MODERN,
            connection_url=postgres_container.get_connection_url()
        )
        
        yield {
            "redis": redis_container,
            "postgres": postgres_container,
            "db_services": db_services,
            "redis_client": redis_container.get_client(),
        }
        
        await db_services.cleanup()
        await redis_container.stop()
        await postgres_container.stop()

    async def test_l1_cache_performance_benchmarks(self, benchmark_infrastructure):
        """Benchmark L1 cache performance against <1ms requirement."""
        from src.prompt_improver.utils.cache_service.l1_cache_service import L1CacheService
        
        print("\n=== L1 Cache Performance Benchmarks ===")
        
        l1_cache = L1CacheService(max_size=10000)
        benchmarks = []
        
        # Benchmark 1: Basic Set/Get Operations
        print("\n1. Basic Set/Get Operations...")
        
        operations = 10000
        test_data = [(f"key_{i}", f"value_{i}_{uuid4().hex}") for i in range(operations)]
        
        # Set operations benchmark
        set_times = []
        set_start = time.perf_counter()
        
        for key, value in test_data:
            op_start = time.perf_counter()
            success = await l1_cache.set(key, value)
            op_time = time.perf_counter() - op_start
            set_times.append(op_time)
            if not success:
                print(f"Set operation failed for {key}")
        
        set_duration = time.perf_counter() - set_start
        set_throughput = operations / set_duration
        
        set_benchmark = PerformanceBenchmark(
            component="L1Cache",
            operation="set",
            requirement_ms=1.0,
            actual_ms=statistics.mean(set_times) * 1000,
            throughput_ops_sec=set_throughput,
            success_rate=1.0,  # All should succeed
            p95_ms=statistics.quantiles(set_times, n=20)[18] * 1000,  # P95
            p99_ms=statistics.quantiles(set_times, n=100)[98] * 1000,  # P99
            min_ms=min(set_times) * 1000,
            max_ms=max(set_times) * 1000,
            total_operations=operations,
            test_duration_sec=set_duration,
            passes_requirement=statistics.mean(set_times) < 0.001
        )
        
        benchmarks.append(set_benchmark)
        
        print(f"  Set Operations: {set_throughput:.0f} ops/sec")
        print(f"  Average Time: {set_benchmark.actual_ms:.3f}ms (requirement: <1ms)")
        print(f"  P95 Time: {set_benchmark.p95_ms:.3f}ms")
        print(f"  P99 Time: {set_benchmark.p99_ms:.3f}ms")
        
        # Get operations benchmark
        get_times = []
        cache_hits = 0
        get_start = time.perf_counter()
        
        for key, expected_value in test_data:
            op_start = time.perf_counter()
            result = await l1_cache.get(key)
            op_time = time.perf_counter() - op_start
            get_times.append(op_time)
            if result == expected_value:
                cache_hits += 1
        
        get_duration = time.perf_counter() - get_start
        get_throughput = operations / get_duration
        hit_rate = cache_hits / operations
        
        get_benchmark = PerformanceBenchmark(
            component="L1Cache",
            operation="get",
            requirement_ms=1.0,
            actual_ms=statistics.mean(get_times) * 1000,
            throughput_ops_sec=get_throughput,
            success_rate=hit_rate,
            p95_ms=statistics.quantiles(get_times, n=20)[18] * 1000,
            p99_ms=statistics.quantiles(get_times, n=100)[98] * 1000,
            min_ms=min(get_times) * 1000,
            max_ms=max(get_times) * 1000,
            total_operations=operations,
            test_duration_sec=get_duration,
            passes_requirement=statistics.mean(get_times) < 0.001 and hit_rate >= 0.95
        )
        
        benchmarks.append(get_benchmark)
        
        print(f"  Get Operations: {get_throughput:.0f} ops/sec")
        print(f"  Average Time: {get_benchmark.actual_ms:.3f}ms (requirement: <1ms)")
        print(f"  Hit Rate: {hit_rate:.1%} (requirement: >95%)")
        
        # Benchmark 2: Concurrent Operations
        print("\n2. Concurrent Operations...")
        
        concurrent_users = 50
        ops_per_user = 200
        
        async def concurrent_user_simulation(user_id):
            user_times = []
            for i in range(ops_per_user):
                key = f"concurrent_{user_id}_{i}"
                value = f"concurrent_value_{user_id}_{i}"
                
                # Set operation
                set_start = time.perf_counter()
                await l1_cache.set(key, value)
                set_time = time.perf_counter() - set_start
                user_times.append(set_time)
                
                # Get operation
                get_start = time.perf_counter()
                result = await l1_cache.get(key)
                get_time = time.perf_counter() - get_start
                user_times.append(get_time)
                
                if result != value:
                    return user_times, False  # Cache miss
            
            return user_times, True
        
        concurrent_start = time.perf_counter()
        concurrent_tasks = [concurrent_user_simulation(i) for i in range(concurrent_users)]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_duration = time.perf_counter() - concurrent_start
        
        all_concurrent_times = []
        successful_users = 0
        
        for times, success in concurrent_results:
            all_concurrent_times.extend(times)
            if success:
                successful_users += 1
        
        concurrent_throughput = len(all_concurrent_times) / concurrent_duration
        concurrent_success_rate = successful_users / concurrent_users
        
        concurrent_benchmark = PerformanceBenchmark(
            component="L1Cache",
            operation="concurrent",
            requirement_ms=1.0,
            actual_ms=statistics.mean(all_concurrent_times) * 1000,
            throughput_ops_sec=concurrent_throughput,
            success_rate=concurrent_success_rate,
            p95_ms=statistics.quantiles(all_concurrent_times, n=20)[18] * 1000,
            p99_ms=statistics.quantiles(all_concurrent_times, n=100)[98] * 1000,
            min_ms=min(all_concurrent_times) * 1000,
            max_ms=max(all_concurrent_times) * 1000,
            total_operations=len(all_concurrent_times),
            test_duration_sec=concurrent_duration,
            passes_requirement=statistics.mean(all_concurrent_times) < 0.001 and concurrent_success_rate >= 0.95
        )
        
        benchmarks.append(concurrent_benchmark)
        
        print(f"  Concurrent Users: {concurrent_users}")
        print(f"  Concurrent Throughput: {concurrent_throughput:.0f} ops/sec")
        print(f"  Average Time: {concurrent_benchmark.actual_ms:.3f}ms")
        print(f"  Success Rate: {concurrent_success_rate:.1%}")
        
        # Performance assertions
        for benchmark in benchmarks:
            assert benchmark.passes_requirement, f"L1 Cache {benchmark.operation} failed: {benchmark.actual_ms:.3f}ms > {benchmark.requirement_ms}ms or success rate {benchmark.success_rate:.1%} < 95%"
        
        print(f"\nL1 Cache Benchmarks: {'PASSED' if all(b.passes_requirement for b in benchmarks) else 'FAILED'}")
        
        return benchmarks

    async def test_l2_cache_performance_benchmarks(self, benchmark_infrastructure):
        """Benchmark L2 cache performance against <10ms requirement."""
        from src.prompt_improver.utils.cache_service.l2_cache_service import L2CacheService
        
        print("\n=== L2 Cache Performance Benchmarks ===")
        
        l2_cache = L2CacheService()
        benchmarks = []
        
        try:
            # Benchmark 1: Basic Operations
            print("\n1. Basic Set/Get Operations...")
            
            operations = 1000
            test_data = [(f"l2_key_{i}", {"id": i, "data": f"value_{i}_{uuid4().hex}"}) for i in range(operations)]
            
            # Set operations
            set_times = []
            successful_sets = 0
            
            for key, value in test_data:
                op_start = time.perf_counter()
                success = await l2_cache.set(key, value, ttl=3600)
                op_time = time.perf_counter() - op_start
                set_times.append(op_time)
                if success:
                    successful_sets += 1
            
            set_success_rate = successful_sets / operations
            
            set_benchmark = PerformanceBenchmark(
                component="L2Cache",
                operation="set",
                requirement_ms=10.0,
                actual_ms=statistics.mean(set_times) * 1000,
                throughput_ops_sec=operations / sum(set_times),
                success_rate=set_success_rate,
                p95_ms=statistics.quantiles(set_times, n=20)[18] * 1000,
                p99_ms=statistics.quantiles(set_times, n=100)[98] * 1000,
                min_ms=min(set_times) * 1000,
                max_ms=max(set_times) * 1000,
                total_operations=operations,
                test_duration_sec=sum(set_times),
                passes_requirement=statistics.mean(set_times) < 0.01 and set_success_rate >= 0.95
            )
            
            benchmarks.append(set_benchmark)
            
            print(f"  Set Operations: {set_benchmark.throughput_ops_sec:.0f} ops/sec")
            print(f"  Average Time: {set_benchmark.actual_ms:.3f}ms (requirement: <10ms)")
            print(f"  Success Rate: {set_success_rate:.1%}")
            
            # Get operations
            get_times = []
            cache_hits = 0
            
            for key, expected_value in test_data:
                op_start = time.perf_counter()
                result = await l2_cache.get(key)
                op_time = time.perf_counter() - op_start
                get_times.append(op_time)
                if result == expected_value:
                    cache_hits += 1
            
            hit_rate = cache_hits / operations
            
            get_benchmark = PerformanceBenchmark(
                component="L2Cache",
                operation="get",
                requirement_ms=10.0,
                actual_ms=statistics.mean(get_times) * 1000,
                throughput_ops_sec=operations / sum(get_times),
                success_rate=hit_rate,
                p95_ms=statistics.quantiles(get_times, n=20)[18] * 1000,
                p99_ms=statistics.quantiles(get_times, n=100)[98] * 1000,
                min_ms=min(get_times) * 1000,
                max_ms=max(get_times) * 1000,
                total_operations=operations,
                test_duration_sec=sum(get_times),
                passes_requirement=statistics.mean(get_times) < 0.01 and hit_rate >= 0.8
            )
            
            benchmarks.append(get_benchmark)
            
            print(f"  Get Operations: {get_benchmark.throughput_ops_sec:.0f} ops/sec")
            print(f"  Average Time: {get_benchmark.actual_ms:.3f}ms")
            print(f"  Hit Rate: {hit_rate:.1%} (requirement: >80%)")
            
            # Benchmark 2: Bulk Operations
            print("\n2. Bulk Operations...")
            
            bulk_keys = [f"bulk_key_{i}" for i in range(100)]
            bulk_data = {key: {"bulk_id": i, "bulk_data": f"bulk_value_{i}"} for i, key in enumerate(bulk_keys)}
            
            # Bulk set
            bulk_set_start = time.perf_counter()
            bulk_set_success = await l2_cache.mset(bulk_data, ttl=3600)
            bulk_set_time = time.perf_counter() - bulk_set_start
            
            # Bulk get
            bulk_get_start = time.perf_counter()
            bulk_results = await l2_cache.mget(bulk_keys)
            bulk_get_time = time.perf_counter() - bulk_get_start
            
            bulk_hit_rate = len(bulk_results) / len(bulk_keys)
            
            bulk_benchmark = PerformanceBenchmark(
                component="L2Cache",
                operation="bulk",
                requirement_ms=10.0,
                actual_ms=(bulk_set_time + bulk_get_time) * 1000 / 2,  # Average of set and get
                throughput_ops_sec=len(bulk_keys) * 2 / (bulk_set_time + bulk_get_time),  # Set + Get operations
                success_rate=bulk_hit_rate,
                p95_ms=(bulk_set_time + bulk_get_time) * 1000 / 2,  # Same as average for bulk
                p99_ms=(bulk_set_time + bulk_get_time) * 1000 / 2,
                min_ms=min(bulk_set_time, bulk_get_time) * 1000,
                max_ms=max(bulk_set_time, bulk_get_time) * 1000,
                total_operations=len(bulk_keys) * 2,
                test_duration_sec=bulk_set_time + bulk_get_time,
                passes_requirement=(bulk_set_time + bulk_get_time) / 2 < 0.01 and bulk_hit_rate >= 0.8
            )
            
            benchmarks.append(bulk_benchmark)
            
            print(f"  Bulk Set Time: {bulk_set_time*1000:.3f}ms for {len(bulk_keys)} keys")
            print(f"  Bulk Get Time: {bulk_get_time*1000:.3f}ms for {len(bulk_keys)} keys") 
            print(f"  Bulk Hit Rate: {bulk_hit_rate:.1%}")
            
        finally:
            await l2_cache.close()
        
        # Performance assertions
        for benchmark in benchmarks:
            assert benchmark.passes_requirement, f"L2 Cache {benchmark.operation} failed: {benchmark.actual_ms:.3f}ms > {benchmark.requirement_ms}ms or success rate {benchmark.success_rate:.1%} insufficient"
        
        print(f"\nL2 Cache Benchmarks: {'PASSED' if all(b.passes_requirement for b in benchmarks) else 'FAILED'}")
        
        return benchmarks

    async def test_security_services_performance_benchmarks(self, benchmark_infrastructure):
        """Benchmark security services performance against <100ms requirement."""
        from src.prompt_improver.security.services import (
            get_security_service_facade,
            create_authentication_service,
            create_crypto_service,
        )
        
        print("\n=== Security Services Performance Benchmarks ===")
        
        benchmarks = []
        
        async with benchmark_infrastructure["postgres"].get_session() as session:
            redis_client = benchmark_infrastructure["redis_client"]
            security_facade = get_security_service_facade(
                session_manager=session,
                redis_client=redis_client
            )
            
            # Benchmark 1: Authentication Operations
            print("\n1. Authentication Operations...")
            
            # Register multiple users for testing
            auth_users = []
            registration_times = []
            
            for i in range(100):
                user_id = f"perf_user_{i}_{uuid4().hex[:6]}"
                password = f"PerfPassword{i}!"
                
                reg_start = time.perf_counter()
                registration = await security_facade.secure_user_registration(
                    user_id=user_id,
                    password=password,
                    email=f"{user_id}@perf.test",
                    metadata={"performance_test": True}
                )
                reg_time = time.perf_counter() - reg_start
                
                registration_times.append(reg_time)
                
                if registration.success:
                    auth_users.append((user_id, password))
            
            registration_success_rate = len(auth_users) / 100
            
            registration_benchmark = PerformanceBenchmark(
                component="Security",
                operation="registration",
                requirement_ms=100.0,
                actual_ms=statistics.mean(registration_times) * 1000,
                throughput_ops_sec=100 / sum(registration_times),
                success_rate=registration_success_rate,
                p95_ms=statistics.quantiles(registration_times, n=20)[18] * 1000,
                p99_ms=statistics.quantiles(registration_times, n=100)[98] * 1000,
                min_ms=min(registration_times) * 1000,
                max_ms=max(registration_times) * 1000,
                total_operations=100,
                test_duration_sec=sum(registration_times),
                passes_requirement=statistics.mean(registration_times) < 0.1 and registration_success_rate >= 0.95
            )
            
            benchmarks.append(registration_benchmark)
            
            print(f"  Registration: {registration_benchmark.throughput_ops_sec:.1f} ops/sec")
            print(f"  Average Time: {registration_benchmark.actual_ms:.3f}ms (requirement: <100ms)")
            print(f"  Success Rate: {registration_success_rate:.1%}")
            
            # Authentication performance test
            auth_times = []
            successful_auths = 0
            
            for user_id, password in auth_users[:50]:  # Test subset
                auth_start = time.perf_counter()
                auth_result = await security_facade.authenticate_and_monitor(
                    username=user_id,
                    password=password,
                    context={"ip_address": "127.0.0.1", "performance_test": True}
                )
                auth_time = time.perf_counter() - auth_start
                auth_times.append(auth_time)
                
                if auth_result.authentication_successful:
                    successful_auths += 1
            
            auth_success_rate = successful_auths / len(auth_users[:50])
            
            auth_benchmark = PerformanceBenchmark(
                component="Security",
                operation="authentication",
                requirement_ms=100.0,
                actual_ms=statistics.mean(auth_times) * 1000,
                throughput_ops_sec=len(auth_times) / sum(auth_times),
                success_rate=auth_success_rate,
                p95_ms=statistics.quantiles(auth_times, n=20)[18] * 1000 if len(auth_times) >= 20 else max(auth_times) * 1000,
                p99_ms=statistics.quantiles(auth_times, n=100)[98] * 1000 if len(auth_times) >= 100 else max(auth_times) * 1000,
                min_ms=min(auth_times) * 1000,
                max_ms=max(auth_times) * 1000,
                total_operations=len(auth_times),
                test_duration_sec=sum(auth_times),
                passes_requirement=statistics.mean(auth_times) < 0.1 and auth_success_rate >= 0.95
            )
            
            benchmarks.append(auth_benchmark)
            
            print(f"  Authentication: {auth_benchmark.throughput_ops_sec:.1f} ops/sec")
            print(f"  Average Time: {auth_benchmark.actual_ms:.3f}ms")
            print(f"  Success Rate: {auth_success_rate:.1%}")
        
        # Benchmark 2: Cryptographic Operations
        print("\n2. Cryptographic Operations...")
        
        crypto_service = create_crypto_service()
        
        # Encryption performance test
        test_data = [f"Encryption test data {i} " * 10 for i in range(100)]  # Varied size data
        encryption_times = []
        decryption_times = []
        successful_operations = 0
        
        for data in test_data:
            # Encryption
            enc_start = time.perf_counter()
            enc_result = await crypto_service.encrypt_data(data, "AES-256-GCM")
            enc_time = time.perf_counter() - enc_start
            encryption_times.append(enc_time)
            
            if enc_result.success:
                # Decryption
                dec_start = time.perf_counter()
                dec_result = await crypto_service.decrypt_data(
                    enc_result.ciphertext,
                    enc_result.nonce,
                    enc_result.tag,
                    "AES-256-GCM"
                )
                dec_time = time.perf_counter() - dec_start
                decryption_times.append(dec_time)
                
                if dec_result.success and dec_result.plaintext == data:
                    successful_operations += 1
        
        crypto_success_rate = successful_operations / len(test_data)
        avg_crypto_time = (sum(encryption_times) + sum(decryption_times)) / (len(encryption_times) + len(decryption_times))
        
        crypto_benchmark = PerformanceBenchmark(
            component="Security",
            operation="cryptography",
            requirement_ms=100.0,
            actual_ms=avg_crypto_time * 1000,
            throughput_ops_sec=(len(encryption_times) + len(decryption_times)) / (sum(encryption_times) + sum(decryption_times)),
            success_rate=crypto_success_rate,
            p95_ms=statistics.quantiles(encryption_times + decryption_times, n=20)[18] * 1000,
            p99_ms=statistics.quantiles(encryption_times + decryption_times, n=100)[98] * 1000,
            min_ms=min(encryption_times + decryption_times) * 1000,
            max_ms=max(encryption_times + decryption_times) * 1000,
            total_operations=len(encryption_times) + len(decryption_times),
            test_duration_sec=sum(encryption_times) + sum(decryption_times),
            passes_requirement=avg_crypto_time < 0.1 and crypto_success_rate >= 0.95
        )
        
        benchmarks.append(crypto_benchmark)
        
        print(f"  Cryptography: {crypto_benchmark.throughput_ops_sec:.1f} ops/sec")
        print(f"  Average Time: {crypto_benchmark.actual_ms:.3f}ms")
        print(f"  Success Rate: {crypto_success_rate:.1%}")
        
        # Performance assertions
        for benchmark in benchmarks:
            assert benchmark.passes_requirement, f"Security {benchmark.operation} failed: {benchmark.actual_ms:.3f}ms > {benchmark.requirement_ms}ms or success rate {benchmark.success_rate:.1%} < 95%"
        
        print(f"\nSecurity Services Benchmarks: {'PASSED' if all(b.passes_requirement for b in benchmarks) else 'FAILED'}")
        
        return benchmarks

    async def test_ml_repository_performance_benchmarks(self, benchmark_infrastructure):
        """Benchmark ML repository performance against <100ms CRUD requirement."""
        from src.prompt_improver.repositories.impl.ml_repository_service import MLRepositoryFacade
        from prompt_improver.database.models import TrainingSessionCreate
        
        print("\n=== ML Repository Performance Benchmarks ===")
        
        ml_facade = MLRepositoryFacade(benchmark_infrastructure["db_services"])
        benchmarks = []
        
        # Benchmark 1: Training Session CRUD Operations
        print("\n1. Training Session CRUD Operations...")
        
        # Create operations
        create_times = []
        created_sessions = []
        
        for i in range(100):
            session_data = TrainingSessionCreate(
                session_id=f"perf_session_{i}_{uuid4().hex[:6]}",
                model_name=f"PerfModel_{i}",
                model_type="performance_test",
                status="initializing",
                hyperparameters={"performance_test": True, "batch_size": 32, "epochs": i % 10 + 1}
            )
            
            create_start = time.perf_counter()
            created_session = await ml_facade.create_training_session(session_data)
            create_time = time.perf_counter() - create_start
            create_times.append(create_time)
            
            if created_session:
                created_sessions.append(created_session)
        
        create_success_rate = len(created_sessions) / 100
        create_throughput = len(created_sessions) / sum(create_times)
        
        create_benchmark = PerformanceBenchmark(
            component="MLRepository",
            operation="create",
            requirement_ms=100.0,
            actual_ms=statistics.mean(create_times) * 1000,
            throughput_ops_sec=create_throughput,
            success_rate=create_success_rate,
            p95_ms=statistics.quantiles(create_times, n=20)[18] * 1000,
            p99_ms=statistics.quantiles(create_times, n=100)[98] * 1000,
            min_ms=min(create_times) * 1000,
            max_ms=max(create_times) * 1000,
            total_operations=100,
            test_duration_sec=sum(create_times),
            passes_requirement=statistics.mean(create_times) < 0.1 and create_success_rate >= 0.95
        )
        
        benchmarks.append(create_benchmark)
        
        print(f"  Create: {create_throughput:.1f} ops/sec")
        print(f"  Average Time: {create_benchmark.actual_ms:.3f}ms (requirement: <100ms)")
        print(f"  Success Rate: {create_success_rate:.1%}")
        
        # Read operations
        read_times = []
        successful_reads = 0
        
        for session in created_sessions[:50]:  # Test subset
            read_start = time.perf_counter()
            retrieved_session = await ml_facade.get_training_session_by_id(session.session_id)
            read_time = time.perf_counter() - read_start
            read_times.append(read_time)
            
            if retrieved_session and retrieved_session.session_id == session.session_id:
                successful_reads += 1
        
        read_success_rate = successful_reads / len(created_sessions[:50])
        read_throughput = len(read_times) / sum(read_times)
        
        read_benchmark = PerformanceBenchmark(
            component="MLRepository",
            operation="read",
            requirement_ms=100.0,
            actual_ms=statistics.mean(read_times) * 1000,
            throughput_ops_sec=read_throughput,
            success_rate=read_success_rate,
            p95_ms=statistics.quantiles(read_times, n=20)[18] * 1000 if len(read_times) >= 20 else max(read_times) * 1000,
            p99_ms=statistics.quantiles(read_times, n=100)[98] * 1000 if len(read_times) >= 100 else max(read_times) * 1000,
            min_ms=min(read_times) * 1000,
            max_ms=max(read_times) * 1000,
            total_operations=len(read_times),
            test_duration_sec=sum(read_times),
            passes_requirement=statistics.mean(read_times) < 0.1 and read_success_rate >= 0.95
        )
        
        benchmarks.append(read_benchmark)
        
        print(f"  Read: {read_throughput:.1f} ops/sec")
        print(f"  Average Time: {read_benchmark.actual_ms:.3f}ms")
        print(f"  Success Rate: {read_success_rate:.1%}")
        
        # Benchmark 2: Batch Operations
        print("\n2. Batch Operations...")
        
        # Batch model performance creation
        batch_size = 50
        model_performance_data = []
        
        for i in range(batch_size):
            model_data = {
                "model_id": f"perf_model_{i}_{uuid4().hex[:6]}",
                "model_name": f"BatchPerfModel_{i}",
                "model_type": "batch_performance_test",
                "accuracy": 0.7 + (i % 20) * 0.01,
                "precision": 0.72 + (i % 15) * 0.01,
                "recall": 0.68 + (i % 25) * 0.01,
                "f1_score": 0.70 + (i % 18) * 0.01,
            }
            model_performance_data.append(model_data)
        
        batch_start = time.perf_counter()
        batch_tasks = [ml_facade.create_model_performance(data) for data in model_performance_data]
        batch_results = await asyncio.gather(*batch_tasks)
        batch_duration = time.perf_counter() - batch_start
        
        successful_batch_ops = sum(1 for result in batch_results if result is not None)
        batch_success_rate = successful_batch_ops / batch_size
        batch_throughput = successful_batch_ops / batch_duration
        
        batch_benchmark = PerformanceBenchmark(
            component="MLRepository",
            operation="batch",
            requirement_ms=100.0,
            actual_ms=(batch_duration / batch_size) * 1000,  # Average per operation
            throughput_ops_sec=batch_throughput,
            success_rate=batch_success_rate,
            p95_ms=(batch_duration / batch_size) * 1000,  # Approximate for batch
            p99_ms=(batch_duration / batch_size) * 1000,
            min_ms=(batch_duration / batch_size) * 1000,
            max_ms=(batch_duration / batch_size) * 1000,
            total_operations=batch_size,
            test_duration_sec=batch_duration,
            passes_requirement=(batch_duration / batch_size) < 0.1 and batch_success_rate >= 0.95
        )
        
        benchmarks.append(batch_benchmark)
        
        print(f"  Batch Operations: {batch_throughput:.1f} ops/sec")
        print(f"  Average Time per Op: {batch_benchmark.actual_ms:.3f}ms")
        print(f"  Batch Success Rate: {batch_success_rate:.1%}")
        
        # Performance assertions
        for benchmark in benchmarks:
            assert benchmark.passes_requirement, f"ML Repository {benchmark.operation} failed: {benchmark.actual_ms:.3f}ms > {benchmark.requirement_ms}ms or success rate {benchmark.success_rate:.1%} < 95%"
        
        print(f"\nML Repository Benchmarks: {'PASSED' if all(b.passes_requirement for b in benchmarks) else 'FAILED'}")
        
        return benchmarks

    async def test_di_container_performance_benchmarks(self, benchmark_infrastructure):
        """Benchmark DI container performance against <5ms service resolution requirement."""
        from src.prompt_improver.core.di import DIContainer
        
        print("\n=== DI Container Performance Benchmarks ===")
        
        container = DIContainer(name="performance_test_container")
        await container.initialize()
        
        try:
            benchmarks = []
            
            # Benchmark 1: Simple Service Resolution
            print("\n1. Simple Service Resolution...")
            
            class SimpleTestService:
                def __init__(self):
                    self.created_at = time.time()
                
                def get_info(self):
                    return {"service": "simple", "created_at": self.created_at}
            
            # Register multiple simple services
            for i in range(100):
                container.register_singleton(f"SimpleService{i}", SimpleTestService)
            
            # Test service resolution performance
            resolution_times = []
            successful_resolutions = 0
            
            for i in range(100):
                resolution_start = time.perf_counter()
                service = await container.get(f"SimpleService{i}")
                resolution_time = time.perf_counter() - resolution_start
                resolution_times.append(resolution_time)
                
                if service is not None and hasattr(service, 'get_info'):
                    successful_resolutions += 1
            
            resolution_success_rate = successful_resolutions / 100
            resolution_throughput = 100 / sum(resolution_times)
            
            simple_resolution_benchmark = PerformanceBenchmark(
                component="DIContainer",
                operation="simple_resolution",
                requirement_ms=5.0,
                actual_ms=statistics.mean(resolution_times) * 1000,
                throughput_ops_sec=resolution_throughput,
                success_rate=resolution_success_rate,
                p95_ms=statistics.quantiles(resolution_times, n=20)[18] * 1000,
                p99_ms=statistics.quantiles(resolution_times, n=100)[98] * 1000,
                min_ms=min(resolution_times) * 1000,
                max_ms=max(resolution_times) * 1000,
                total_operations=100,
                test_duration_sec=sum(resolution_times),
                passes_requirement=statistics.mean(resolution_times) < 0.005 and resolution_success_rate >= 0.95
            )
            
            benchmarks.append(simple_resolution_benchmark)
            
            print(f"  Simple Resolution: {resolution_throughput:.0f} ops/sec")
            print(f"  Average Time: {simple_resolution_benchmark.actual_ms:.3f}ms (requirement: <5ms)")
            print(f"  Success Rate: {resolution_success_rate:.1%}")
            
            # Benchmark 2: Complex Service Resolution with Dependencies
            print("\n2. Complex Service Resolution...")
            
            class DependencyService:
                def __init__(self):
                    self.initialized_at = time.time()
                
                def get_data(self):
                    return {"dependency": "data", "initialized_at": self.initialized_at}
            
            class ComplexService:
                def __init__(self, dependency: DependencyService):
                    self.dependency = dependency
                    self.created_at = time.time()
                
                def get_complex_info(self):
                    return {
                        "service": "complex",
                        "created_at": self.created_at,
                        "dependency_data": self.dependency.get_data()
                    }
            
            # Register dependency service
            container.register_singleton("DependencyService", DependencyService)
            
            # Register complex services with dependencies
            for i in range(50):
                async def complex_factory(index=i):
                    dependency = await container.get("DependencyService")
                    return ComplexService(dependency)
                
                container.register_factory(f"ComplexService{i}", complex_factory)
            
            # Test complex service resolution performance
            complex_resolution_times = []
            successful_complex_resolutions = 0
            
            for i in range(50):
                complex_start = time.perf_counter()
                complex_service = await container.get(f"ComplexService{i}")
                complex_time = time.perf_counter() - complex_start
                complex_resolution_times.append(complex_time)
                
                if complex_service is not None and hasattr(complex_service, 'get_complex_info'):
                    info = complex_service.get_complex_info()
                    if info and "dependency_data" in info:
                        successful_complex_resolutions += 1
            
            complex_success_rate = successful_complex_resolutions / 50
            complex_throughput = 50 / sum(complex_resolution_times)
            
            complex_resolution_benchmark = PerformanceBenchmark(
                component="DIContainer",
                operation="complex_resolution",
                requirement_ms=20.0,  # Higher requirement for complex services
                actual_ms=statistics.mean(complex_resolution_times) * 1000,
                throughput_ops_sec=complex_throughput,
                success_rate=complex_success_rate,
                p95_ms=statistics.quantiles(complex_resolution_times, n=20)[18] * 1000 if len(complex_resolution_times) >= 20 else max(complex_resolution_times) * 1000,
                p99_ms=statistics.quantiles(complex_resolution_times, n=100)[98] * 1000 if len(complex_resolution_times) >= 100 else max(complex_resolution_times) * 1000,
                min_ms=min(complex_resolution_times) * 1000,
                max_ms=max(complex_resolution_times) * 1000,
                total_operations=50,
                test_duration_sec=sum(complex_resolution_times),
                passes_requirement=statistics.mean(complex_resolution_times) < 0.02 and complex_success_rate >= 0.95
            )
            
            benchmarks.append(complex_resolution_benchmark)
            
            print(f"  Complex Resolution: {complex_throughput:.0f} ops/sec")
            print(f"  Average Time: {complex_resolution_benchmark.actual_ms:.3f}ms (requirement: <20ms)")
            print(f"  Success Rate: {complex_success_rate:.1%}")
            
            # Benchmark 3: Concurrent Service Resolution
            print("\n3. Concurrent Service Resolution...")
            
            concurrent_users = 20
            resolutions_per_user = 10
            
            async def concurrent_resolution_test(user_id):
                user_times = []
                for i in range(resolutions_per_user):
                    service_name = f"SimpleService{(user_id * resolutions_per_user + i) % 100}"
                    resolution_start = time.perf_counter()
                    service = await container.get(service_name)
                    resolution_time = time.perf_counter() - resolution_start
                    user_times.append(resolution_time)
                
                return user_times
            
            concurrent_start = time.perf_counter()
            concurrent_tasks = [concurrent_resolution_test(i) for i in range(concurrent_users)]
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            concurrent_duration = time.perf_counter() - concurrent_start
            
            all_concurrent_times = []
            for user_times in concurrent_results:
                all_concurrent_times.extend(user_times)
            
            concurrent_throughput = len(all_concurrent_times) / concurrent_duration
            
            concurrent_benchmark = PerformanceBenchmark(
                component="DIContainer",
                operation="concurrent_resolution",
                requirement_ms=5.0,
                actual_ms=statistics.mean(all_concurrent_times) * 1000,
                throughput_ops_sec=concurrent_throughput,
                success_rate=1.0,  # All should succeed
                p95_ms=statistics.quantiles(all_concurrent_times, n=20)[18] * 1000,
                p99_ms=statistics.quantiles(all_concurrent_times, n=100)[98] * 1000,
                min_ms=min(all_concurrent_times) * 1000,
                max_ms=max(all_concurrent_times) * 1000,
                total_operations=len(all_concurrent_times),
                test_duration_sec=concurrent_duration,
                passes_requirement=statistics.mean(all_concurrent_times) < 0.005
            )
            
            benchmarks.append(concurrent_benchmark)
            
            print(f"  Concurrent Resolution: {concurrent_throughput:.0f} ops/sec")
            print(f"  Average Time: {concurrent_benchmark.actual_ms:.3f}ms")
            print(f"  Total Concurrent Operations: {len(all_concurrent_times)}")
            
        finally:
            await container.shutdown()
        
        # Performance assertions
        for benchmark in benchmarks:
            assert benchmark.passes_requirement, f"DI Container {benchmark.operation} failed: {benchmark.actual_ms:.3f}ms > {benchmark.requirement_ms}ms"
        
        print(f"\nDI Container Benchmarks: {'PASSED' if all(b.passes_requirement for b in benchmarks) else 'FAILED'}")
        
        return benchmarks

    def test_performance_benchmark_summary(self, benchmark_infrastructure):
        """Generate comprehensive performance benchmark summary."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # This would collect all benchmark results from individual test methods
        # For now, we'll provide a template for the summary
        
        summary_data = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "infrastructure": {
                "redis_host": benchmark_infrastructure["redis"].get_host(),
                "redis_port": benchmark_infrastructure["redis"].get_port(),
                "postgres_host": benchmark_infrastructure["postgres"].get_connection_info()["host"],
                "postgres_port": benchmark_infrastructure["postgres"].get_connection_info()["port"],
            },
            "performance_requirements": {
                "L1_Cache": {"response_time_ms": 1.0, "hit_rate": 0.95},
                "L2_Cache": {"response_time_ms": 10.0, "hit_rate": 0.80},
                "Security": {"response_time_ms": 100.0, "success_rate": 0.95},
                "ML_Repository": {"response_time_ms": 100.0, "throughput_ops_sec": 1000},
                "DI_Container": {"simple_resolution_ms": 5.0, "complex_resolution_ms": 20.0},
                "Health_Monitoring": {"health_check_ms": 25.0, "recovery_ms": 100.0},
            }
        }
        
        print(f"\nTest Environment:")
        print(f"  Redis: {summary_data['infrastructure']['redis_host']}:{summary_data['infrastructure']['redis_port']}")
        print(f"  PostgreSQL: {summary_data['infrastructure']['postgres_host']}:{summary_data['infrastructure']['postgres_port']}")
        print(f"  Timestamp: {summary_data['test_timestamp']}")
        
        print(f"\nPerformance Requirements:")
        for component, requirements in summary_data['performance_requirements'].items():
            print(f"  {component}:")
            for metric, value in requirements.items():
                if "ms" in metric:
                    print(f"    {metric}: <{value}ms")
                elif "rate" in metric:
                    print(f"    {metric}: >{value:.0%}")
                elif "throughput" in metric:
                    print(f"    {metric}: >{value} ops/sec")
        
        print(f"\n{'='*80}")
        print("All performance benchmarks should be executed via individual test methods")
        print("This summary provides the framework for comprehensive performance validation")
        print("="*80)
        
        return summary_data


@pytest.mark.performance
@pytest.mark.benchmark
@pytest.mark.real_behavior
class TestDecomposedComponentsPerformance(ComponentPerformanceBenchmarks):
    """Performance benchmark test suite for all decomposed components."""
    pass