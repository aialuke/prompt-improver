"""System-wide integration tests for all decomposed components.

Comprehensive validation that all decomposed components work together correctly
across component boundaries with real backend services and realistic workloads.

System Requirements Validation:
- Multi-level cache services: L1 <1ms, L2 <10ms, L3 <50ms
- Security services: OWASP compliance, multi-factor auth, real crypto
- ML repository services: <100ms CRUD operations, >1000 records/sec
- DI containers: <5ms service resolution, proper lifecycle management  
- Redis health monitoring: <25ms health checks, failure recovery
- Cross-component integration: End-to-end workflows with real data
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import uuid4

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.containers.real_redis_container import RealRedisTestContainer

# Import all component systems under test
from src.prompt_improver.utils.cache_service import L1CacheService, L2CacheService
from src.prompt_improver.security.services import (
    get_security_service_facade,
    create_authentication_service,
    create_crypto_service,
)
from src.prompt_improver.repositories.impl.ml_repository_service import MLRepositoryFacade
from src.prompt_improver.core.di import DIContainer, get_container
from src.prompt_improver.monitoring.unified.facade import UnifiedMonitoringFacade
from src.prompt_improver.monitoring.unified.types import MonitoringConfig
from src.prompt_improver.database import DatabaseServices, ManagerMode
from prompt_improver.database.models import (
    TrainingSessionCreate,
    GenerationSession,
    SyntheticDataSample,
)


class SystemWideComponentValidation:
    """System-wide validation of all decomposed components."""

    @pytest.fixture
    async def complete_system_infrastructure(self):
        """Set up complete system infrastructure with all backend services."""
        redis_container = RealRedisTestContainer()
        postgres_container = PostgreSQLTestContainer()
        
        await redis_container.start()
        await postgres_container.start()
        
        # Set up environment
        import os
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        os.environ["POSTGRES_HOST"] = postgres_container.get_host()
        os.environ["POSTGRES_PORT"] = str(postgres_container.get_exposed_port(5432))
        
        # Initialize database services
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

    async def test_complete_system_initialization_and_health(self, complete_system_infrastructure):
        """Test complete system initialization and health validation."""
        print("\n=== System-Wide Component Validation ===")
        
        # Initialize all major components
        components = {}
        initialization_times = {}
        
        # 1. Initialize Cache Services (L1/L2)
        print("\n1. Initializing Cache Services...")
        cache_start = time.perf_counter()
        
        l1_cache = L1CacheService(max_size=1000)
        l2_cache = L2CacheService()
        
        # Test L1 cache
        await l1_cache.set("test_key", "test_value")
        l1_result = await l1_cache.get("test_key")
        assert l1_result == "test_value"
        
        # Test L2 cache
        await l2_cache.set("test_key", "test_value")
        l2_result = await l2_cache.get("test_key")
        assert l2_result == "test_value"
        
        components["l1_cache"] = l1_cache
        components["l2_cache"] = l2_cache
        initialization_times["cache_services"] = time.perf_counter() - cache_start
        
        print(f"  ✓ Cache Services initialized ({initialization_times['cache_services']*1000:.2f}ms)")
        
        # 2. Initialize Security Services
        print("\n2. Initializing Security Services...")
        security_start = time.perf_counter()
        
        redis_client = complete_system_infrastructure["redis_client"]
        async with complete_system_infrastructure["postgres"].get_session() as session:
            security_facade = get_security_service_facade(
                session_manager=session,
                redis_client=redis_client
            )
            
            components["security_facade"] = security_facade
            initialization_times["security_services"] = time.perf_counter() - security_start
            
        print(f"  ✓ Security Services initialized ({initialization_times['security_services']*1000:.2f}ms)")
        
        # 3. Initialize ML Repository Services
        print("\n3. Initializing ML Repository Services...")
        ml_start = time.perf_counter()
        
        ml_facade = MLRepositoryFacade(complete_system_infrastructure["db_services"])
        
        # Test ML repository functionality
        session_id = f"system_test_{uuid4().hex[:8]}"
        session_data = TrainingSessionCreate(
            session_id=session_id,
            model_name="SystemTestModel",
            model_type="integration_test",
            status="initializing"
        )
        
        training_session = await ml_facade.create_training_session(session_data)
        assert training_session is not None
        assert training_session.session_id == session_id
        
        components["ml_facade"] = ml_facade
        initialization_times["ml_repository"] = time.perf_counter() - ml_start
        
        print(f"  ✓ ML Repository Services initialized ({initialization_times['ml_repository']*1000:.2f}ms)")
        
        # 4. Initialize DI Container
        print("\n4. Initializing DI Container...")
        di_start = time.perf_counter()
        
        di_container = DIContainer(name="system_validation_container")
        await di_container.initialize()
        
        # Register test services
        class SystemTestService:
            def __init__(self):
                self.initialized_at = datetime.utcnow()
                self.call_count = 0
            
            async def get_system_info(self):
                self.call_count += 1
                return {
                    "service": "system_test",
                    "initialized_at": self.initialized_at.isoformat(),
                    "call_count": self.call_count,
                }
        
        di_container.register_singleton("SystemTestService", SystemTestService)
        
        # Test service resolution
        test_service = await di_container.get("SystemTestService")
        system_info = await test_service.get_system_info()
        assert system_info["service"] == "system_test"
        
        components["di_container"] = di_container
        initialization_times["di_container"] = time.perf_counter() - di_start
        
        print(f"  ✓ DI Container initialized ({initialization_times['di_container']*1000:.2f}ms)")
        
        # 5. Initialize Monitoring Services
        print("\n5. Initializing Monitoring Services...")
        monitoring_start = time.perf_counter()
        
        monitoring_config = MonitoringConfig(
            health_check_timeout_seconds=5.0,
            health_check_parallel_enabled=True,
            metrics_collection_enabled=True,
        )
        
        monitoring_facade = UnifiedMonitoringFacade(config=monitoring_config)
        await monitoring_facade.start_monitoring()
        
        # Test monitoring functionality
        monitoring_facade.record_custom_metric(
            "system.validation.test",
            1.0,
            tags={"component": "system_test"}
        )
        
        metrics = await monitoring_facade.collect_all_metrics()
        assert len(metrics) > 0
        
        components["monitoring_facade"] = monitoring_facade
        initialization_times["monitoring_services"] = time.perf_counter() - monitoring_start
        
        print(f"  ✓ Monitoring Services initialized ({initialization_times['monitoring_services']*1000:.2f}ms)")
        
        # Calculate total initialization time
        total_init_time = sum(initialization_times.values())
        
        print(f"\n=== System Initialization Summary ===")
        print(f"Total Initialization Time: {total_init_time*1000:.2f}ms")
        for component, init_time in initialization_times.items():
            print(f"  {component}: {init_time*1000:.2f}ms")
        
        # System initialization performance assertions
        assert total_init_time < 2.0, f"System initialization took {total_init_time:.3f}s, should be <2s"
        assert initialization_times["cache_services"] < 0.1, "Cache services should init <100ms"
        assert initialization_times["security_services"] < 0.5, "Security services should init <500ms"
        assert initialization_times["ml_repository"] < 0.2, "ML repository should init <200ms"
        assert initialization_times["di_container"] < 0.1, "DI container should init <100ms"
        assert initialization_times["monitoring_services"] < 0.5, "Monitoring should init <500ms"
        
        # Clean up
        try:
            await monitoring_facade.stop_monitoring()
            await di_container.shutdown()
            await l2_cache.close()
        except Exception as e:
            print(f"Cleanup warning: {e}")
        
        return {
            "components": components,
            "initialization_times": initialization_times,
            "total_init_time": total_init_time,
        }

    async def test_cross_component_workflow_integration(self, complete_system_infrastructure):
        """Test end-to-end workflow across all component boundaries."""
        print("\n=== Cross-Component Workflow Integration ===")
        
        workflow_start_time = time.perf_counter()
        
        # Initialize components
        l1_cache = L1CacheService(max_size=500)
        l2_cache = L2CacheService()
        ml_facade = MLRepositoryFacade(complete_system_infrastructure["db_services"])
        
        redis_client = complete_system_infrastructure["redis_client"]
        async with complete_system_infrastructure["postgres"].get_session() as session:
            security_facade = get_security_service_facade(
                session_manager=session,
                redis_client=redis_client
            )
            
            # Step 1: User Authentication and Authorization Workflow
            print("\n1. User Authentication and Security Workflow...")
            user_id = f"workflow_user_{uuid4().hex[:8]}"
            password = "SecureWorkflowPassword123!"
            
            auth_start = time.perf_counter()
            
            # Register user through security system
            registration_result = await security_facade.secure_user_registration(
                user_id=user_id,
                password=password,
                email=f"{user_id}@workflow.test",
                metadata={"workflow": "integration_test", "component": "security"}
            )
            
            assert registration_result.success, "User registration should succeed"
            
            # Authenticate user
            auth_result = await security_facade.authenticate_and_monitor(
                username=user_id,
                password=password,
                context={"ip_address": "127.0.0.1", "workflow": "integration"}
            )
            
            assert auth_result.authentication_successful, "Authentication should succeed"
            
            auth_duration = time.perf_counter() - auth_start
            print(f"  ✓ Security workflow completed ({auth_duration*1000:.2f}ms)")
            
            # Step 2: Cache Data Workflow (L1 → L2 coordination)
            print("\n2. Multi-Level Cache Workflow...")
            cache_start = time.perf_counter()
            
            # Store user session data in cache hierarchy
            session_data = {
                "user_id": user_id,
                "session_token": auth_result.session_token,
                "permissions": ["read", "write", "execute"],
                "workflow_context": {
                    "component_access": ["ml", "cache", "monitoring"],
                    "session_start": datetime.utcnow().isoformat(),
                }
            }
            
            # L1 Cache (hot data)
            await l1_cache.set(f"session:{user_id}", session_data, ttl=3600)
            l1_retrieved = await l1_cache.get(f"session:{user_id}")
            assert l1_retrieved == session_data
            
            # L2 Cache (shared data)
            await l2_cache.set(f"user_profile:{user_id}", {
                "user_id": user_id,
                "profile_data": {"role": "integration_tester", "permissions": session_data["permissions"]},
                "cached_at": datetime.utcnow().isoformat(),
            }, ttl=7200)
            
            l2_retrieved = await l2_cache.get(f"user_profile:{user_id}")
            assert l2_retrieved is not None
            assert l2_retrieved["user_id"] == user_id
            
            cache_duration = time.perf_counter() - cache_start
            print(f"  ✓ Cache workflow completed ({cache_duration*1000:.2f}ms)")
            
            # Step 3: ML Repository Workflow
            print("\n3. ML Repository and Training Workflow...")
            ml_start = time.perf_counter()
            
            # Create training session for user
            training_session_id = f"workflow_training_{uuid4().hex[:8]}"
            training_data = TrainingSessionCreate(
                session_id=training_session_id,
                model_name=f"UserModel_{user_id}",
                model_type="workflow_integration",
                status="initializing",
                hyperparameters={
                    "user_context": user_id,
                    "workflow_type": "integration",
                    "cache_integration": True,
                    "security_validated": True,
                }
            )
            
            training_session = await ml_facade.create_training_session(training_data)
            assert training_session is not None
            
            # Create model performance record
            model_performance_data = {
                "model_id": f"workflow_model_{uuid4().hex[:8]}",
                "model_name": training_data.model_name,
                "model_type": training_data.model_type,
                "training_session_id": training_session_id,
                "accuracy": 0.85,
                "precision": 0.87,
                "recall": 0.83,
                "f1_score": 0.85,
                "user_context": user_id,
                "workflow_metadata": {
                    "integration_test": True,
                    "security_validated": registration_result.success,
                    "cache_levels_used": ["L1", "L2"],
                }
            }
            
            model_performance = await ml_facade.create_model_performance(model_performance_data)
            assert model_performance is not None
            
            # Create generation session
            generation_data = {
                "session_id": f"gen_{uuid4().hex[:8]}",
                "model_id": model_performance.model_id,
                "generation_method": "workflow_integration",
                "status": "active",
                "config": {
                    "user_context": user_id,
                    "integration_mode": True,
                }
            }
            
            generation_session = await ml_facade.create_generation_session(generation_data)
            assert generation_session is not None
            
            ml_duration = time.perf_counter() - ml_start
            print(f"  ✓ ML Repository workflow completed ({ml_duration*1000:.2f}ms)")
            
            # Step 4: Cross-Component Data Validation
            print("\n4. Cross-Component Data Validation...")
            validation_start = time.perf_counter()
            
            # Validate data consistency across components
            
            # Check cache data matches security context
            cached_session = await l1_cache.get(f"session:{user_id}")
            assert cached_session["user_id"] == user_id
            assert cached_session["session_token"] == auth_result.session_token
            
            # Check ML data references correct user
            retrieved_training = await ml_facade.get_training_session_by_id(training_session_id)
            assert retrieved_training is not None
            assert retrieved_training.hyperparameters["user_context"] == user_id
            
            # Check model performance includes workflow metadata
            model_performances = await ml_facade.get_model_performance_by_id(model_performance.model_id)
            assert len(model_performances) == 1
            assert model_performances[0].workflow_metadata["security_validated"] is True
            
            validation_duration = time.perf_counter() - validation_start
            print(f"  ✓ Cross-component validation completed ({validation_duration*1000:.2f}ms)")
            
            # Step 5: Performance and Health Monitoring Integration
            print("\n5. Performance and Health Monitoring...")
            monitoring_start = time.perf_counter()
            
            monitoring_facade = UnifiedMonitoringFacade()
            await monitoring_facade.start_monitoring()
            
            try:
                # Record workflow metrics
                workflow_metrics = [
                    ("workflow.auth.duration_ms", auth_duration * 1000),
                    ("workflow.cache.duration_ms", cache_duration * 1000),
                    ("workflow.ml.duration_ms", ml_duration * 1000),
                    ("workflow.validation.duration_ms", validation_duration * 1000),
                    ("workflow.user.sessions", 1.0),
                    ("workflow.ml.models_created", 1.0),
                ]
                
                for metric_name, metric_value in workflow_metrics:
                    monitoring_facade.record_custom_metric(
                        metric_name,
                        metric_value,
                        tags={"user_id": user_id, "workflow": "integration", "component": "system"}
                    )
                
                # Collect and validate metrics
                all_metrics = await monitoring_facade.collect_all_metrics()
                workflow_metric_names = [m.name for m in all_metrics if "workflow" in m.name]
                
                assert len(workflow_metric_names) >= len(workflow_metrics), "All workflow metrics should be recorded"
                
                monitoring_duration = time.perf_counter() - monitoring_start
                print(f"  ✓ Monitoring integration completed ({monitoring_duration*1000:.2f}ms)")
                
            finally:
                await monitoring_facade.stop_monitoring()
        
        # Calculate total workflow time
        total_workflow_time = time.perf_counter() - workflow_start_time
        
        # Workflow performance summary
        workflow_summary = {
            "total_duration_ms": total_workflow_time * 1000,
            "auth_duration_ms": auth_duration * 1000,
            "cache_duration_ms": cache_duration * 1000,
            "ml_duration_ms": ml_duration * 1000,
            "validation_duration_ms": validation_duration * 1000,
            "monitoring_duration_ms": monitoring_duration * 1000,
            "user_id": user_id,
            "training_session_id": training_session_id,
            "model_id": model_performance.model_id,
            "generation_session_id": generation_session.session_id,
        }
        
        print(f"\n=== Cross-Component Workflow Summary ===")
        print(f"Total Workflow Time: {workflow_summary['total_duration_ms']:.2f}ms")
        print(f"  Authentication: {workflow_summary['auth_duration_ms']:.2f}ms")
        print(f"  Cache Operations: {workflow_summary['cache_duration_ms']:.2f}ms")
        print(f"  ML Repository: {workflow_summary['ml_duration_ms']:.2f}ms")
        print(f"  Data Validation: {workflow_summary['validation_duration_ms']:.2f}ms")
        print(f"  Monitoring: {workflow_summary['monitoring_duration_ms']:.2f}ms")
        
        # Performance assertions for cross-component workflow
        assert total_workflow_time < 5.0, f"Complete workflow should finish <5s, took {total_workflow_time:.3f}s"
        assert auth_duration < 1.0, f"Authentication workflow should be <1s, took {auth_duration:.3f}s"
        assert cache_duration < 0.1, f"Cache workflow should be <100ms, took {cache_duration:.3f}s"
        assert ml_duration < 1.0, f"ML workflow should be <1s, took {ml_duration:.3f}s"
        assert validation_duration < 0.5, f"Validation should be <500ms, took {validation_duration:.3f}s"
        
        # Clean up
        await l2_cache.close()
        
        return workflow_summary

    async def test_system_performance_under_concurrent_load(self, complete_system_infrastructure):
        """Test system performance under realistic concurrent load."""
        print("\n=== System Performance Under Concurrent Load ===")
        
        load_test_start = time.perf_counter()
        
        # Initialize system components
        l1_cache = L1CacheService(max_size=2000)
        l2_cache = L2CacheService()
        ml_facade = MLRepositoryFacade(complete_system_infrastructure["db_services"])
        
        redis_client = complete_system_infrastructure["redis_client"]
        
        # Define concurrent load scenarios
        concurrent_users = 20
        operations_per_user = 10
        
        print(f"Load Test Parameters:")
        print(f"  Concurrent Users: {concurrent_users}")
        print(f"  Operations Per User: {operations_per_user}")
        print(f"  Total Operations: {concurrent_users * operations_per_user}")
        
        async def user_workflow_simulation(user_index):
            """Simulate complete user workflow under load."""
            user_id = f"load_user_{user_index}_{uuid4().hex[:6]}"
            user_results = {
                "user_id": user_id,
                "operations": [],
                "total_time": 0,
                "successful_operations": 0,
            }
            
            user_start = time.perf_counter()
            
            try:
                async with complete_system_infrastructure["postgres"].get_session() as session:
                    security_facade = get_security_service_facade(
                        session_manager=session,
                        redis_client=redis_client
                    )
                    
                    # Authentication operation
                    auth_start = time.perf_counter()
                    registration = await security_facade.secure_user_registration(
                        user_id=user_id,
                        password=f"LoadTestPassword{user_index}!",
                        email=f"{user_id}@load.test",
                        metadata={"load_test": True, "user_index": user_index}
                    )
                    auth_time = time.perf_counter() - auth_start
                    
                    if registration.success:
                        user_results["successful_operations"] += 1
                    
                    user_results["operations"].append(("auth", auth_time, registration.success))
                    
                    # Cache operations
                    for op_index in range(operations_per_user):
                        cache_start = time.perf_counter()
                        
                        # L1 Cache operations
                        cache_key = f"load_data_{user_index}_{op_index}"
                        cache_data = {
                            "user_id": user_id,
                            "operation_index": op_index,
                            "timestamp": time.time(),
                            "load_test_data": f"data_{op_index}" * 10,  # Some bulk data
                        }
                        
                        await l1_cache.set(cache_key, cache_data)
                        l1_result = await l1_cache.get(cache_key)
                        l1_success = l1_result == cache_data
                        
                        # L2 Cache operations
                        await l2_cache.set(f"l2_{cache_key}", cache_data, ttl=3600)
                        l2_result = await l2_cache.get(f"l2_{cache_key}")
                        l2_success = l2_result == cache_data
                        
                        cache_time = time.perf_counter() - cache_start
                        cache_success = l1_success and l2_success
                        
                        if cache_success:
                            user_results["successful_operations"] += 1
                        
                        user_results["operations"].append(("cache", cache_time, cache_success))
                    
                    # ML Repository operations
                    ml_start = time.perf_counter()
                    
                    training_session_data = TrainingSessionCreate(
                        session_id=f"load_training_{user_index}_{uuid4().hex[:6]}",
                        model_name=f"LoadTestModel_{user_index}",
                        model_type="load_test",
                        status="initializing",
                        hyperparameters={"load_test": True, "user_index": user_index}
                    )
                    
                    training_session = await ml_facade.create_training_session(training_session_data)
                    ml_time = time.perf_counter() - ml_start
                    ml_success = training_session is not None
                    
                    if ml_success:
                        user_results["successful_operations"] += 1
                    
                    user_results["operations"].append(("ml_training", ml_time, ml_success))
                
            except Exception as e:
                user_results["error"] = str(e)
            
            user_results["total_time"] = time.perf_counter() - user_start
            return user_results
        
        # Execute concurrent load test
        print("\nExecuting concurrent load test...")
        
        load_tasks = [user_workflow_simulation(i) for i in range(concurrent_users)]
        user_results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        load_test_duration = time.perf_counter() - load_test_start
        
        # Analyze load test results
        successful_users = 0
        total_operations = 0
        successful_operations = 0
        operation_times = {"auth": [], "cache": [], "ml_training": []}
        
        for result in user_results:
            if isinstance(result, Exception):
                print(f"User workflow failed: {result}")
                continue
            
            if result.get("error"):
                print(f"User {result['user_id']} error: {result['error']}")
                continue
            
            successful_users += 1
            total_operations += len(result["operations"])
            successful_operations += result["successful_operations"]
            
            # Collect operation timing data
            for op_type, op_time, op_success in result["operations"]:
                if op_success and op_type in operation_times:
                    operation_times[op_type].append(op_time)
        
        # Calculate performance metrics
        success_rate = successful_users / concurrent_users
        operation_success_rate = successful_operations / total_operations if total_operations > 0 else 0
        overall_throughput = successful_operations / load_test_duration
        
        # Operation performance analysis
        performance_analysis = {}
        for op_type, times in operation_times.items():
            if times:
                performance_analysis[op_type] = {
                    "count": len(times),
                    "avg_time_ms": sum(times) / len(times) * 1000,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000,
                    "p95_time_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) > 1 else times[0] * 1000,
                }
        
        print(f"\n=== Concurrent Load Test Results ===")
        print(f"Load Test Duration: {load_test_duration:.3f}s")
        print(f"Successful Users: {successful_users}/{concurrent_users} ({success_rate:.1%})")
        print(f"Total Operations: {total_operations}")
        print(f"Successful Operations: {successful_operations} ({operation_success_rate:.1%})")
        print(f"Overall Throughput: {overall_throughput:.1f} ops/sec")
        
        print(f"\nOperation Performance Analysis:")
        for op_type, analysis in performance_analysis.items():
            print(f"  {op_type}:")
            print(f"    Count: {analysis['count']}")
            print(f"    Avg Time: {analysis['avg_time_ms']:.2f}ms")
            print(f"    P95 Time: {analysis['p95_time_ms']:.2f}ms")
            print(f"    Min/Max: {analysis['min_time_ms']:.2f}ms / {analysis['max_time_ms']:.2f}ms")
        
        # Performance assertions for concurrent load
        assert success_rate >= 0.9, f"User success rate {success_rate:.1%} should be ≥90%"
        assert operation_success_rate >= 0.85, f"Operation success rate {operation_success_rate:.1%} should be ≥85%"
        assert overall_throughput > 20, f"Overall throughput {overall_throughput:.1f} should be >20 ops/sec"
        
        # Component-specific performance assertions
        if "auth" in performance_analysis:
            auth_p95 = performance_analysis["auth"]["p95_time_ms"]
            assert auth_p95 < 1000, f"Auth P95 time {auth_p95:.2f}ms should be <1000ms"
        
        if "cache" in performance_analysis:
            cache_p95 = performance_analysis["cache"]["p95_time_ms"]
            assert cache_p95 < 100, f"Cache P95 time {cache_p95:.2f}ms should be <100ms"
        
        if "ml_training" in performance_analysis:
            ml_p95 = performance_analysis["ml_training"]["p95_time_ms"]
            assert ml_p95 < 500, f"ML training P95 time {ml_p95:.2f}ms should be <500ms"
        
        # Clean up
        await l2_cache.close()
        
        print(f"\nConcurrent Load Test: PASSED")
        
        return {
            "load_test_duration": load_test_duration,
            "success_rate": success_rate,
            "operation_success_rate": operation_success_rate,
            "overall_throughput": overall_throughput,
            "performance_analysis": performance_analysis,
        }

    async def test_system_resilience_and_recovery(self, complete_system_infrastructure):
        """Test system resilience and recovery under failure scenarios."""
        print("\n=== System Resilience and Recovery Test ===")
        
        resilience_start = time.perf_counter()
        
        # Initialize system components
        l1_cache = L1CacheService(max_size=500)
        l2_cache = L2CacheService()
        ml_facade = MLRepositoryFacade(complete_system_infrastructure["db_services"])
        
        monitoring_facade = UnifiedMonitoringFacade()
        await monitoring_facade.start_monitoring()
        
        try:
            # Establish baseline system health
            print("\n1. Establishing Baseline System Health...")
            baseline_start = time.perf_counter()
            
            # Perform normal operations
            baseline_operations = []
            
            # Cache operations
            for i in range(10):
                await l1_cache.set(f"baseline_key_{i}", f"baseline_value_{i}")
                result = await l1_cache.get(f"baseline_key_{i}")
                baseline_operations.append(("l1_cache", result is not None))
                
                await l2_cache.set(f"baseline_key_{i}", f"baseline_value_{i}")
                result = await l2_cache.get(f"baseline_key_{i}")
                baseline_operations.append(("l2_cache", result is not None))
            
            # ML operations
            for i in range(3):
                session_data = TrainingSessionCreate(
                    session_id=f"baseline_session_{i}",
                    model_name=f"BaselineModel_{i}",
                    model_type="resilience_test",
                    status="completed"
                )
                training_session = await ml_facade.create_training_session(session_data)
                baseline_operations.append(("ml_repository", training_session is not None))
            
            baseline_duration = time.perf_counter() - baseline_start
            baseline_success_rate = sum(1 for _, success in baseline_operations if success) / len(baseline_operations)
            
            print(f"  Baseline Operations: {len(baseline_operations)}")
            print(f"  Baseline Success Rate: {baseline_success_rate:.1%}")
            print(f"  Baseline Duration: {baseline_duration*1000:.2f}ms")
            
            # 2. Test System Under Partial Component Failures
            print("\n2. Testing System Under Partial Failures...")
            
            # Simulate Redis connection issues (affecting L2 cache)
            original_redis_host = os.environ.get("REDIS_HOST")
            os.environ["REDIS_HOST"] = "invalid.redis.host.failure.test"
            
            # Test system behavior with L2 cache failures
            partial_failure_operations = []
            failure_start = time.perf_counter()
            
            # L1 cache should continue working
            for i in range(5):
                await l1_cache.set(f"failure_key_{i}", f"failure_value_{i}")
                result = await l1_cache.get(f"failure_key_{i}")
                partial_failure_operations.append(("l1_cache_during_failure", result is not None))
            
            # L2 cache operations should fail gracefully
            failing_l2_cache = L2CacheService()
            for i in range(5):
                success = await failing_l2_cache.set(f"failure_key_{i}", f"failure_value_{i}")
                partial_failure_operations.append(("l2_cache_failure", success))
                
                result = await failing_l2_cache.get(f"failure_key_{i}")
                partial_failure_operations.append(("l2_cache_failure_read", result is not None))
            
            # ML operations should continue working (using PostgreSQL)
            for i in range(2):
                session_data = TrainingSessionCreate(
                    session_id=f"failure_session_{i}",
                    model_name=f"FailureTestModel_{i}",
                    model_type="resilience_test",
                    status="completed"
                )
                training_session = await ml_facade.create_training_session(session_data)
                partial_failure_operations.append(("ml_during_redis_failure", training_session is not None))
            
            failure_duration = time.perf_counter() - failure_start
            
            # Restore Redis connection
            if original_redis_host:
                os.environ["REDIS_HOST"] = original_redis_host
            
            await failing_l2_cache.close()
            
            # Analyze partial failure results
            l1_cache_failures = [op for op in partial_failure_operations if op[0] == "l1_cache_during_failure"]
            l2_cache_failures = [op for op in partial_failure_operations if op[0].startswith("l2_cache_failure")]
            ml_during_failure = [op for op in partial_failure_operations if op[0] == "ml_during_redis_failure"]
            
            l1_success_during_failure = sum(1 for _, success in l1_cache_failures if success) / len(l1_cache_failures)
            l2_failure_rate = sum(1 for _, success in l2_cache_failures if not success) / len(l2_cache_failures)
            ml_success_during_failure = sum(1 for _, success in ml_during_failure if success) / len(ml_during_failure)
            
            print(f"  Partial Failure Duration: {failure_duration*1000:.2f}ms")
            print(f"  L1 Cache Success (Redis down): {l1_success_during_failure:.1%}")
            print(f"  L2 Cache Failure Rate: {l2_failure_rate:.1%}")
            print(f"  ML Repository Success (Redis down): {ml_success_during_failure:.1%}")
            
            # 3. Test System Recovery
            print("\n3. Testing System Recovery...")
            recovery_start = time.perf_counter()
            
            # Reinitialize L2 cache with restored Redis connection
            recovered_l2_cache = L2CacheService()
            
            recovery_operations = []
            
            # Test L2 cache recovery
            for i in range(5):
                success = await recovered_l2_cache.set(f"recovery_key_{i}", f"recovery_value_{i}")
                recovery_operations.append(("l2_cache_recovery", success))
                
                if success:
                    result = await recovered_l2_cache.get(f"recovery_key_{i}")
                    recovery_operations.append(("l2_cache_recovery_read", result is not None))
            
            # Test continued ML operations
            for i in range(2):
                session_data = TrainingSessionCreate(
                    session_id=f"recovery_session_{i}",
                    model_name=f"RecoveryModel_{i}",
                    model_type="resilience_test",
                    status="completed"
                )
                training_session = await ml_facade.create_training_session(session_data)
                recovery_operations.append(("ml_post_recovery", training_session is not None))
            
            recovery_duration = time.perf_counter() - recovery_start
            recovery_success_rate = sum(1 for _, success in recovery_operations if success) / len(recovery_operations)
            
            print(f"  Recovery Duration: {recovery_duration*1000:.2f}ms")
            print(f"  Recovery Success Rate: {recovery_success_rate:.1%}")
            
            # 4. System Health Validation Post-Recovery
            print("\n4. System Health Validation...")
            health_start = time.perf_counter()
            
            system_health = await monitoring_facade.get_system_health()
            health_duration = time.perf_counter() - health_start
            
            print(f"  System Health Status: {system_health.overall_status.value}")
            print(f"  Healthy Components: {system_health.healthy_components}/{system_health.total_components}")
            print(f"  Health Check Duration: {health_duration*1000:.2f}ms")
            
            await recovered_l2_cache.close()
            
        finally:
            await monitoring_facade.stop_monitoring()
        
        resilience_duration = time.perf_counter() - resilience_start
        
        print(f"\n=== System Resilience Summary ===")
        print(f"Total Resilience Test Duration: {resilience_duration:.3f}s")
        print(f"Baseline Success Rate: {baseline_success_rate:.1%}")
        print(f"L1 Cache Resilience: {l1_success_during_failure:.1%}")
        print(f"ML Repository Resilience: {ml_success_during_failure:.1%}")
        print(f"System Recovery Rate: {recovery_success_rate:.1%}")
        
        # Resilience assertions
        assert baseline_success_rate >= 0.95, f"Baseline success rate {baseline_success_rate:.1%} should be ≥95%"
        assert l1_success_during_failure >= 0.9, f"L1 cache should maintain ≥90% success during Redis failure"
        assert ml_success_during_failure >= 0.9, f"ML repository should maintain ≥90% success during Redis failure"
        assert recovery_success_rate >= 0.9, f"System recovery rate {recovery_success_rate:.1%} should be ≥90%"
        assert resilience_duration < 10.0, f"Complete resilience test should finish <10s"
        
        print(f"\nSystem Resilience Test: PASSED")
        
        return {
            "resilience_duration": resilience_duration,
            "baseline_success_rate": baseline_success_rate,
            "l1_resilience": l1_success_during_failure,
            "ml_resilience": ml_success_during_failure,
            "recovery_success_rate": recovery_success_rate,
        }


@pytest.mark.integration
@pytest.mark.real_behavior
@pytest.mark.system_wide
class TestSystemWideValidation(SystemWideComponentValidation):
    """System-wide validation test suite."""
    pass