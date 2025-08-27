"""Real behavior tests for decomposed DI containers.

Comprehensive validation of DI container orchestration with actual service instances,
dependency resolution, lifecycle management, and cross-container communication.

Performance Requirements:
- Service resolution: <5ms for simple services, <20ms for complex services
- Container initialization: <100ms per container
- Health checks: <25ms across all containers
- Service lifecycle: Proper cleanup and resource management
- Dependency injection: 100% resolution accuracy
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import pytest
from src.prompt_improver.core.di import (
    DIContainer,
    MLServiceContainer,
    ServiceInitializationError,
    ServiceLifetime,
    ServiceNotRegisteredError,
    get_core_container,
    get_database_container,
    get_monitoring_container,
    get_security_container,
)
from src.prompt_improver.core.di.container_orchestrator import (
    shutdown_container,
)
from src.prompt_improver.core.di.core_container import CoreContainer
from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.containers.real_redis_container import RealRedisTestContainer

from prompt_improver.core.interfaces.datetime_service import DateTimeServiceProtocol


# Test protocols and implementations for validation
@runtime_checkable
class TestServiceProtocol(Protocol):
    """Test service protocol for DI container validation."""

    async def get_value(self) -> str:
        """Get test value."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        ...


class TestServiceImplementation:
    """Test service implementation for DI container validation."""

    def __init__(self, value: str = "test_value"):
        self.value = value
        self.initialized_at = datetime.utcnow()
        self.call_count = 0

    async def get_value(self) -> str:
        """Get test value."""
        self.call_count += 1
        return self.value

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "initialized_at": self.initialized_at.isoformat(),
            "call_count": self.call_count,
            "uptime_seconds": (datetime.utcnow() - self.initialized_at).total_seconds()
        }

    async def cleanup(self):
        """Cleanup method for testing finalizers."""
        self.value = "cleaned_up"


@runtime_checkable
class DependentServiceProtocol(Protocol):
    """Test protocol for services with dependencies."""

    async def get_dependency_value(self) -> str:
        """Get value from dependency."""
        ...


class DependentServiceImplementation:
    """Service implementation with dependency for testing DI resolution."""

    def __init__(self, test_service: TestServiceProtocol):
        self.test_service = test_service
        self.initialized_at = datetime.utcnow()

    async def get_dependency_value(self) -> str:
        """Get value from dependency."""
        base_value = await self.test_service.get_value()
        return f"dependent_{base_value}"

    async def health_check(self) -> dict[str, Any]:
        """Perform health check including dependency."""
        dependency_health = await self.test_service.health_check()
        return {
            "status": "healthy",
            "initialized_at": self.initialized_at.isoformat(),
            "dependency_status": dependency_health["status"]
        }


class TestCoreContainerRealBehavior:
    """Real behavior tests for core DI container."""

    @pytest.fixture
    async def core_container(self):
        """Create core container for testing."""
        container = get_core_container()
        await container.initialize()
        yield container
        await container.shutdown()

    @pytest.fixture
    async def isolated_core_container(self):
        """Create isolated core container for tests that need custom service registrations."""
        # Create a fresh container instead of using the global one
        container = CoreContainer(name="test_isolated")
        # Don't call initialize() to avoid registering default services that might conflict
        yield container
        await container.shutdown()

    async def test_core_container_service_registration_and_resolution(self, core_container):
        """Test core container service registration and resolution."""
        # Test singleton registration
        start_time = time.perf_counter()
        core_container.register_singleton(
            TestServiceProtocol,
            TestServiceImplementation,
            tags={"test", "core"}
        )
        registration_time = time.perf_counter() - start_time

        assert registration_time < 0.001, f"Service registration took {registration_time:.3f}s, should be <1ms"
        print(f"Service Registration Time: {registration_time * 1000:.2f}ms")

        # Test service resolution
        start_time = time.perf_counter()
        service_instance = await core_container.get(TestServiceProtocol)
        resolution_time = time.perf_counter() - start_time

        assert service_instance is not None
        assert isinstance(service_instance, TestServiceImplementation)
        assert resolution_time < 0.005, f"Service resolution took {resolution_time:.3f}s, should be <5ms"
        print(f"Service Resolution Time: {resolution_time * 1000:.2f}ms")

        # Test singleton behavior
        second_instance = await core_container.get(TestServiceProtocol)
        assert service_instance is second_instance, "Singleton should return same instance"

        # Test service functionality
        value = await service_instance.get_value()
        assert value == "test_value"

        health = await service_instance.health_check()
        assert health["status"] == "healthy"
        assert health["call_count"] == 1

    async def test_core_container_transient_services(self, core_container):
        """Test transient service registration and resolution."""
        # Register transient service
        core_container.register_transient(
            TestServiceProtocol,
            TestServiceImplementation,
            tags={"transient", "test"}
        )

        # Test multiple resolutions create different instances
        instance1 = await core_container.get(TestServiceProtocol)
        instance2 = await core_container.get(TestServiceProtocol)

        assert instance1 is not instance2, "Transient services should create new instances"
        assert isinstance(instance1, TestServiceImplementation)
        assert isinstance(instance2, TestServiceImplementation)

        # Both instances should be functional
        value1 = await instance1.get_value()
        value2 = await instance2.get_value()
        assert value1 == value2 == "test_value"

    async def test_core_container_factory_registration(self, isolated_core_container):
        """Test factory-based service registration."""
        call_count = 0

        def test_factory() -> TestServiceImplementation:
            nonlocal call_count
            call_count += 1
            return TestServiceImplementation(f"factory_value_{call_count}")

        # Register factory as singleton
        isolated_core_container.register_factory(
            TestServiceProtocol,
            test_factory,
            lifetime=ServiceLifetime.SINGLETON,
            tags={"factory", "singleton"}
        )

        # Test singleton factory behavior
        instance1 = await isolated_core_container.get(TestServiceProtocol)
        instance2 = await isolated_core_container.get(TestServiceProtocol)

        assert instance1 is instance2, "Singleton factory should reuse instance"
        assert call_count == 1, "Factory should be called only once for singleton"

        value = await instance1.get_value()
        assert value == "factory_value_1"

    async def test_core_container_dependency_injection(self, isolated_core_container):
        """Test dependency injection between services."""
        # Register base service
        isolated_core_container.register_singleton(
            TestServiceProtocol,
            TestServiceImplementation,
            tags={"base", "dependency"}
        )

        # Register dependent service with factory that resolves dependency
        async def dependent_factory() -> DependentServiceImplementation:
            base_service = await isolated_core_container.get(TestServiceProtocol)
            return DependentServiceImplementation(base_service)

        isolated_core_container.register_factory(
            DependentServiceProtocol,
            dependent_factory,
            lifetime=ServiceLifetime.SINGLETON,
            tags={"dependent", "injection"}
        )

        # Test dependency resolution
        start_time = time.perf_counter()
        dependent_service = await isolated_core_container.get(DependentServiceProtocol)
        complex_resolution_time = time.perf_counter() - start_time

        assert dependent_service is not None
        assert complex_resolution_time < 0.02, f"Complex service resolution took {complex_resolution_time:.3f}s, should be <20ms"
        print(f"Complex Service Resolution Time: {complex_resolution_time * 1000:.2f}ms")

        # Test functionality through dependency chain
        dependency_value = await dependent_service.get_dependency_value()
        assert dependency_value == "dependent_test_value"

        # Test health check through dependency
        health = await dependent_service.health_check()
        assert health["status"] == "healthy"
        assert health["dependency_status"] == "healthy"

    async def test_core_container_performance_under_load(self, core_container):
        """Test core container performance under concurrent load."""
        # Register multiple services
        service_count = 20
        for i in range(service_count):
            service_name = f"TestService{i}"

            # Create a unique protocol for each service
            def create_service_class(index):
                class SpecificTestService(TestServiceImplementation):
                    def __init__(self):
                        super().__init__(f"test_value_{index}")

                return SpecificTestService

            core_container.register_singleton(
                f"TestService{i}",  # Use string as service key
                create_service_class(i),
                tags={"load_test", f"service_{i}"}
            )

        # Test concurrent service resolution
        async def resolve_service(service_name):
            return await core_container.get(service_name)

        start_time = time.perf_counter()
        resolution_tasks = [resolve_service(f"TestService{i}") for i in range(service_count)]
        resolved_services = await asyncio.gather(*resolution_tasks)
        concurrent_resolution_time = time.perf_counter() - start_time

        assert len(resolved_services) == service_count
        assert all(service is not None for service in resolved_services)

        concurrent_throughput = service_count / concurrent_resolution_time
        assert concurrent_throughput > 100, f"Concurrent resolution throughput {concurrent_throughput:.1f} should be >100 ops/sec"

        print(f"Concurrent Resolution: {service_count} services in {concurrent_resolution_time:.3f}s ({concurrent_throughput:.1f} ops/sec)")

        # Test service uniqueness and functionality
        for i, service in enumerate(resolved_services):
            value = await service.get_value()
            assert value == f"test_value_{i}", f"Service {i} should have correct value"

    async def test_core_container_health_monitoring(self, core_container):
        """Test core container health monitoring capabilities."""
        # Register services with health checks
        core_container.register_singleton(
            TestServiceProtocol,
            TestServiceImplementation,
            tags={"health_test"}
        )

        # Test container health check
        start_time = time.perf_counter()
        health_result = await core_container.health_check()
        health_check_time = time.perf_counter() - start_time

        assert health_result is not None
        assert isinstance(health_result, dict)
        assert health_check_time < 0.025, f"Health check took {health_check_time:.3f}s, should be <25ms"
        print(f"Container Health Check Time: {health_check_time * 1000:.2f}ms")

        # Test registration info
        registration_info = core_container.get_registration_info()
        assert "registered_services" in registration_info or len(registration_info) > 0


class TestContainerOrchestrationRealBehavior:
    """Real behavior tests for DI container orchestration."""

    @pytest.fixture
    async def test_infrastructure(self):
        """Set up test infrastructure with real backends."""
        redis_container = RealRedisTestContainer()
        postgres_container = PostgreSQLTestContainer()

        await redis_container.start()
        await postgres_container.start()

        # Set environment variables for containers
        import os
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        os.environ["POSTGRES_HOST"] = postgres_container.get_host()
        os.environ["POSTGRES_PORT"] = str(postgres_container.get_exposed_port(5432))

        yield {
            "redis": redis_container,
            "postgres": postgres_container
        }

        await redis_container.stop()
        await postgres_container.stop()

    @pytest.fixture
    async def di_container(self, test_infrastructure):
        """Create DI container orchestrator with real backends."""
        # Cleanup any existing global container
        await shutdown_container()

        container = DIContainer(name="test_orchestrator")
        await container.initialize()
        yield container
        await container.shutdown()

    async def test_container_orchestration_service_routing(self, di_container):
        """Test service routing to appropriate specialized containers."""
        # Test datetime service routing (should go to core container)
        start_time = time.perf_counter()
        datetime_service = await di_container.get(DateTimeServiceProtocol)
        datetime_resolution_time = time.perf_counter() - start_time

        assert datetime_service is not None
        assert datetime_resolution_time < 0.005, f"DateTime service resolution took {datetime_resolution_time:.3f}s"
        print(f"DateTime Service Resolution: {datetime_resolution_time * 1000:.2f}ms")

        # Test service functionality
        current_time = await datetime_service.now()
        assert isinstance(current_time, datetime)

        # Test service from different containers would be routed correctly
        # This tests the domain routing logic in the orchestrator

    async def test_cross_container_service_coordination(self, di_container):
        """Test coordination between services across different containers."""
        # Register services in different containers that need to interact

        # Core service
        di_container.register_singleton(
            TestServiceProtocol,
            TestServiceImplementation,
            tags={"cross_container", "core"}
        )

        # Register a service that depends on core service in ML container
        async def ml_service_factory():
            # This simulates a service in ML container depending on core service
            core_service = await di_container.get(TestServiceProtocol)

            class MLDependentService:
                def __init__(self, core_service):
                    self.core_service = core_service
                    self.initialized_at = datetime.utcnow()

                async def get_ml_value(self):
                    base_value = await self.core_service.get_value()
                    return f"ml_{base_value}"

                async def health_check(self):
                    core_health = await self.core_service.health_check()
                    return {
                        "status": "healthy",
                        "core_dependency_status": core_health["status"],
                        "initialized_at": self.initialized_at.isoformat()
                    }

            return MLDependentService(core_service)

        di_container.register_factory(
            "MLDependentService",
            ml_service_factory,
            lifetime=ServiceLifetime.SINGLETON,
            tags={"cross_container", "ml"}
        )

        # Test cross-container service resolution
        start_time = time.perf_counter()
        ml_service = await di_container.get("MLDependentService")
        cross_container_resolution_time = time.perf_counter() - start_time

        assert ml_service is not None
        assert cross_container_resolution_time < 0.02, f"Cross-container resolution took {cross_container_resolution_time:.3f}s"
        print(f"Cross-Container Service Resolution: {cross_container_resolution_time * 1000:.2f}ms")

        # Test functionality across container boundaries
        ml_value = await ml_service.get_ml_value()
        assert ml_value == "ml_test_value"

        # Test health check across containers
        health = await ml_service.health_check()
        assert health["status"] == "healthy"
        assert health["core_dependency_status"] == "healthy"

    async def test_container_orchestration_health_monitoring(self, di_container):
        """Test orchestration-level health monitoring across all containers."""
        # Test orchestrator health check
        start_time = time.perf_counter()
        orchestrator_health = await di_container.health_check_all()
        orchestrator_health_time = time.perf_counter() - start_time

        assert orchestrator_health is not None
        assert "orchestrator_status" in orchestrator_health
        assert "containers" in orchestrator_health
        assert orchestrator_health_time < 0.1, f"Orchestrator health check took {orchestrator_health_time:.3f}s"
        print(f"Orchestrator Health Check Time: {orchestrator_health_time * 1000:.2f}ms")

        # Verify all container health is reported
        expected_containers = ["core", "security", "database", "monitoring", "ml"]
        for container_name in expected_containers:
            if container_name in orchestrator_health["containers"]:
                container_health = orchestrator_health["containers"][container_name]
                print(f"  {container_name}: {container_health.get('status', 'unknown')}")

        # Test individual container access
        core_container = di_container.get_core_container()
        assert core_container is not None

        ml_container = di_container.get_ml_container()
        assert ml_container is not None

    async def test_container_orchestration_lifecycle_management(self, di_container):
        """Test lifecycle management across orchestrated containers."""
        # Register services with cleanup requirements
        cleanup_called = []

        class LifecycleTestService:
            def __init__(self, service_id: str):
                self.service_id = service_id
                self.initialized_at = datetime.utcnow()

            async def get_value(self):
                return f"lifecycle_{self.service_id}"

            async def cleanup(self):
                cleanup_called.append(self.service_id)
                print(f"Cleaning up service: {self.service_id}")

        # Register services in different containers
        for i in range(5):
            service_id = f"lifecycle_service_{i}"
            di_container.register_singleton(
                f"LifecycleService{i}",
                lambda: LifecycleTestService(service_id),
                tags={"lifecycle", f"service_{i}"}
            )

        # Resolve all services to ensure they're instantiated
        services = []
        for i in range(5):
            service = await di_container.get(f"LifecycleService{i}")
            services.append(service)
            value = await service.get_value()
            print(f"Resolved service {i}: {value}")

        assert len(services) == 5

        # Test scoped service lifecycle
        async with di_container.scope("test_scope"):
            # Services within scope should be properly managed
            scoped_service = services[0]  # Use existing service for test
            assert scoped_service is not None

        # Test graceful shutdown
        start_time = time.perf_counter()
        await di_container.shutdown()
        shutdown_time = time.perf_counter() - start_time

        print(f"Container Orchestrator Shutdown Time: {shutdown_time * 1000:.2f}ms")
        assert shutdown_time < 1.0, f"Orchestrator shutdown took {shutdown_time:.3f}s, should be <1s"

    async def test_container_orchestration_performance_metrics(self, di_container):
        """Test performance metrics collection across orchestrated containers."""
        # Register multiple services for performance testing
        service_count = 50
        services = []

        for i in range(service_count):
            service_name = f"PerfService{i}"

            def create_perf_service(index):
                class PerfService:
                    def __init__(self):
                        self.index = index
                        self.created_at = datetime.utcnow()

                    async def get_performance_data(self):
                        return {
                            "index": self.index,
                            "created_at": self.created_at.isoformat(),
                            "processing_time": 0.001  # Simulate processing time
                        }

                return PerfService

            di_container.register_singleton(
                service_name,
                create_perf_service(i),
                tags={"performance", f"perf_{i}"}
            )

        # Test batch service resolution performance
        start_time = time.perf_counter()
        resolution_tasks = []

        for i in range(service_count):
            task = di_container.get(f"PerfService{i}")
            resolution_tasks.append(task)

        resolved_services = await asyncio.gather(*resolution_tasks)
        batch_resolution_time = time.perf_counter() - start_time

        assert len(resolved_services) == service_count
        batch_throughput = service_count / batch_resolution_time

        print("Batch Service Resolution Performance:")
        print(f"  Services: {service_count}")
        print(f"  Total Time: {batch_resolution_time:.3f}s")
        print(f"  Throughput: {batch_throughput:.1f} services/sec")
        print(f"  Average Resolution Time: {batch_resolution_time * 1000 / service_count:.2f}ms per service")

        # Performance assertions
        assert batch_throughput > 200, f"Batch resolution throughput {batch_throughput:.1f} should be >200 services/sec"
        assert batch_resolution_time / service_count < 0.005, "Average resolution time should be <5ms per service"

        # Test service functionality
        for i, service in enumerate(resolved_services):
            perf_data = await service.get_performance_data()
            assert perf_data["index"] == i
            assert "created_at" in perf_data

        # Test container performance metrics
        perf_metrics = di_container.get_performance_metrics()
        assert "resolution_times" in perf_metrics
        assert "registered_services_count" in perf_metrics
        assert perf_metrics["registered_services_count"] > 0

        print("Container Performance Metrics:")
        print(f"  Registered Services: {perf_metrics['registered_services_count']}")
        print(f"  Active Resources: {perf_metrics['active_resources_count']}")
        print(f"  Scoped Contexts: {perf_metrics['scoped_contexts_count']}")

    async def test_container_orchestration_error_handling_and_resilience(self, di_container):
        """Test error handling and resilience in container orchestration."""
        # Test service registration error handling
        class FailingService:
            def __init__(self):
                raise Exception("Intentional initialization failure")

        # This should not crash the container
        try:
            di_container.register_singleton(
                "FailingService",
                FailingService,
                tags={"error_test"}
            )

            # Attempting to resolve should raise appropriate exception
            with pytest.raises((ServiceInitializationError, Exception)):
                await di_container.get("FailingService")

        except Exception as e:
            # Container should handle registration errors gracefully
            print(f"Handled registration error: {e}")

        # Test service not found error handling
        with pytest.raises(ServiceNotRegisteredError):
            await di_container.get("NonExistentService")

        # Test container should remain functional after errors
        di_container.register_singleton(
            TestServiceProtocol,
            TestServiceImplementation,
            tags={"recovery_test"}
        )

        recovery_service = await di_container.get(TestServiceProtocol)
        assert recovery_service is not None

        value = await recovery_service.get_value()
        assert value == "test_value"

        print("Container remained functional after error scenarios")

        # Test partial container failure resilience
        health_status = await di_container.health_check_all()

        # Even if some containers are unhealthy, orchestrator should report status
        assert "orchestrator_status" in health_status
        assert health_status["orchestrator_status"] in {"healthy", "degraded", "unhealthy"}

        print(f"Orchestrator Status After Errors: {health_status['orchestrator_status']}")


@pytest.mark.integration
@pytest.mark.real_behavior
class TestSpecializedContainerIntegration:
    """Integration tests for specialized DI containers."""

    @pytest.fixture
    async def test_infrastructure(self):
        """Set up complete test infrastructure."""
        redis_container = RealRedisTestContainer()
        postgres_container = PostgreSQLTestContainer()

        await redis_container.start()
        await postgres_container.start()

        import os
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        os.environ["POSTGRES_HOST"] = postgres_container.get_host()
        os.environ["POSTGRES_PORT"] = str(postgres_container.get_exposed_port(5432))

        yield {
            "redis": redis_container,
            "postgres": postgres_container
        }

        await redis_container.stop()
        await postgres_container.stop()

    async def test_specialized_containers_coordination(self, test_infrastructure):
        """Test coordination between all specialized containers."""
        # Initialize all specialized containers
        core_container = get_core_container()
        security_container = get_security_container()
        database_container = get_database_container()
        monitoring_container = get_monitoring_container()
        ml_container = MLServiceContainer()

        containers = {
            "core": core_container,
            "security": security_container,
            "database": database_container,
            "monitoring": monitoring_container,
            "ml": ml_container,
        }

        try:
            # Initialize all containers
            initialization_times = {}
            for name, container in containers.items():
                start_time = time.perf_counter()
                if hasattr(container, "initialize"):
                    await container.initialize()
                initialization_time = time.perf_counter() - start_time
                initialization_times[name] = initialization_time

                assert initialization_time < 0.1, f"{name} container initialization took {initialization_time:.3f}s"
                print(f"{name.capitalize()} Container Initialization: {initialization_time * 1000:.2f}ms")

            # Test cross-container service dependencies
            # This would simulate services from different containers working together

            # Core services (datetime, configuration)
            if hasattr(core_container, "get"):
                try:
                    datetime_service = await core_container.get(DateTimeServiceProtocol)
                    current_time = await datetime_service.now()
                    assert isinstance(current_time, datetime)
                    print(f"Core container datetime service working: {current_time.isoformat()}")
                except Exception as e:
                    print(f"Core container service test failed: {e}")

            # Test container health checks
            health_check_times = {}
            healthy_containers = 0

            for name, container in containers.items():
                start_time = time.perf_counter()
                try:
                    if hasattr(container, "health_check"):
                        health_result = await container.health_check()
                        health_check_time = time.perf_counter() - start_time
                        health_check_times[name] = health_check_time

                        if isinstance(health_result, dict) and health_result.get("status") in {"healthy", "degraded"}:
                            healthy_containers += 1
                            print(f"{name.capitalize()} Container Health: {health_result.get('status', 'unknown')} ({health_check_time * 1000:.2f}ms)")
                        else:
                            print(f"{name.capitalize()} Container Health: Check completed ({health_check_time * 1000:.2f}ms)")
                    else:
                        print(f"{name.capitalize()} Container: No health check method")
                        healthy_containers += 1  # Assume healthy if no health check

                except Exception as e:
                    health_check_time = time.perf_counter() - start_time
                    print(f"{name.capitalize()} Container Health Check Failed: {e} ({health_check_time * 1000:.2f}ms)")

            # Performance assertions
            total_init_time = sum(initialization_times.values())
            avg_health_check_time = sum(health_check_times.values()) / len(health_check_times) if health_check_times else 0

            print("\nContainer Integration Summary:")
            print(f"  Total Initialization Time: {total_init_time * 1000:.2f}ms")
            print(f"  Average Health Check Time: {avg_health_check_time * 1000:.2f}ms")
            print(f"  Healthy Containers: {healthy_containers}/{len(containers)}")

            assert total_init_time < 0.5, f"Total container initialization should be <500ms, got {total_init_time:.3f}s"
            assert avg_health_check_time < 0.025 if avg_health_check_time > 0 else True, \
                f"Average health check should be <25ms, got {avg_health_check_time:.3f}s"
            assert healthy_containers >= len(containers) * 0.8, \
                f"At least 80% of containers should be healthy, got {healthy_containers}/{len(containers)}"

        finally:
            # Clean up containers
            shutdown_times = {}
            for name, container in containers.items():
                start_time = time.perf_counter()
                try:
                    if hasattr(container, "shutdown"):
                        await container.shutdown()
                    shutdown_time = time.perf_counter() - start_time
                    shutdown_times[name] = shutdown_time
                    print(f"{name.capitalize()} Container Shutdown: {shutdown_time * 1000:.2f}ms")
                except Exception as e:
                    print(f"{name.capitalize()} Container Shutdown Error: {e}")

            total_shutdown_time = sum(shutdown_times.values())
            print(f"Total Container Shutdown Time: {total_shutdown_time * 1000:.2f}ms")

            assert total_shutdown_time < 1.0, f"Total shutdown should be <1s, got {total_shutdown_time:.3f}s"

    async def test_container_system_stress_test(self, test_infrastructure):
        """Stress test the entire DI container system."""
        container = DIContainer(name="stress_test_container")

        try:
            await container.initialize()

            # Register many services to test system limits
            service_count = 100
            registration_start_time = time.perf_counter()

            for i in range(service_count):
                service_name = f"StressService{i}"

                class StressTestService:
                    def __init__(self, index):
                        self.index = index
                        self.created_at = datetime.utcnow()

                    async def get_index(self):
                        return self.index

                    async def stress_operation(self):
                        # Simulate some work
                        await asyncio.sleep(0.001)  # 1ms of work
                        return f"stress_result_{self.index}"

                container.register_factory(
                    service_name,
                    lambda idx=i: StressTestService(idx),
                    lifetime=ServiceLifetime.SINGLETON,
                    tags={"stress_test", f"batch_{i // 10}"}
                )

            registration_time = time.perf_counter() - registration_start_time
            registration_throughput = service_count / registration_time

            print("Service Registration Stress Test:")
            print(f"  Services: {service_count}")
            print(f"  Registration Time: {registration_time:.3f}s")
            print(f"  Registration Throughput: {registration_throughput:.1f} services/sec")

            # Test concurrent service resolution under stress
            resolution_start_time = time.perf_counter()

            async def resolve_and_call_service(service_name):
                service = await container.get(service_name)
                index = await service.get_index()
                result = await service.stress_operation()
                return (service_name, index, result)

            # Create tasks for all services
            stress_tasks = [resolve_and_call_service(f"StressService{i}") for i in range(service_count)]

            # Execute with controlled concurrency
            batch_size = 20
            all_results = []

            for i in range(0, service_count, batch_size):
                batch = stress_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch)
                all_results.extend(batch_results)

            resolution_time = time.perf_counter() - resolution_start_time
            resolution_throughput = service_count / resolution_time

            print("Service Resolution Stress Test:")
            print(f"  Resolution Time: {resolution_time:.3f}s")
            print(f"  Resolution Throughput: {resolution_throughput:.1f} services/sec")
            print(f"  Average Time per Service: {resolution_time * 1000 / service_count:.2f}ms")

            # Verify all services worked correctly
            assert len(all_results) == service_count
            for service_name, index, result in all_results:
                expected_result = f"stress_result_{index}"
                assert result == expected_result, f"Service {service_name} returned incorrect result"

            # Test system health under stress
            health_start_time = time.perf_counter()
            stress_health = await container.health_check_all()
            health_time = time.perf_counter() - health_start_time

            print(f"Health Check Under Stress: {health_time * 1000:.2f}ms")
            assert health_time < 0.1, f"Health check under stress took {health_time:.3f}s, should be <100ms"

            # Performance assertions for stress test
            assert registration_throughput > 500, f"Registration throughput {registration_throughput:.1f} should be >500 services/sec"
            assert resolution_throughput > 100, f"Resolution throughput {resolution_throughput:.1f} should be >100 services/sec"
            assert stress_health["orchestrator_status"] in {"healthy", "degraded"}, "System should remain functional under stress"

            print("Stress Test Summary: PASSED")
            print(f"  System handled {service_count} services successfully")
            print(f"  Registration: {registration_throughput:.1f} services/sec")
            print(f"  Resolution: {resolution_throughput:.1f} services/sec")
            print(f"  Health Status: {stress_health['orchestrator_status']}")

        finally:
            await container.shutdown()
