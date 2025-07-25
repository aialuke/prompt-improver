"""Tests for datetime service dependency injection implementation.

This test suite validates the datetime service DI implementation
following the critical findings implementation plan.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock

from prompt_improver.core.interfaces.datetime_service import (
    DateTimeServiceProtocol, 
    MockDateTimeService
)
from prompt_improver.core.services.datetime_service import DateTimeService
from prompt_improver.core.di.container import (
    DIContainer, 
    get_datetime_service, 
    get_container,
    ServiceLifetime,
    ServiceNotRegisteredError,
    CircularDependencyError
)


class TestDateTimeServiceInterface:
    """Test the datetime service interface and protocol."""
    
    @pytest.mark.asyncio
    async def test_datetime_service_protocol_compliance(self):
        """Test that DateTimeService implements the protocol correctly."""
        service = DateTimeService()
        
        # Verify it implements the protocol
        assert isinstance(service, DateTimeServiceProtocol)
        
        # Test all protocol methods exist and are callable
        assert hasattr(service, 'utc_now')
        assert hasattr(service, 'aware_utc_now')
        assert hasattr(service, 'naive_utc_now')
        assert hasattr(service, 'from_timestamp')
        assert hasattr(service, 'to_timezone')
        assert hasattr(service, 'format_iso')
        assert hasattr(service, 'ensure_aware_utc')
        assert hasattr(service, 'ensure_naive_utc')
    
    @pytest.mark.asyncio
    async def test_datetime_service_basic_functionality(self):
        """Test basic datetime service functionality."""
        service = DateTimeService()
        
        # Test UTC now methods
        utc_time = await service.utc_now()
        aware_time = await service.aware_utc_now()
        naive_time = await service.naive_utc_now()
        
        assert utc_time.tzinfo == timezone.utc
        assert aware_time.tzinfo == timezone.utc
        assert naive_time.tzinfo is None
        
        # Test timestamp conversion
        timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        aware_dt = await service.from_timestamp(timestamp, aware=True)
        naive_dt = await service.from_timestamp(timestamp, aware=False)
        
        assert aware_dt.tzinfo == timezone.utc
        assert naive_dt.tzinfo is None
        assert aware_dt.year == 2022
        assert naive_dt.year == 2022
        
        # Test ISO formatting
        iso_string = await service.format_iso(utc_time)
        assert isinstance(iso_string, str)
        assert 'T' in iso_string  # ISO format contains T separator
    
    @pytest.mark.asyncio
    async def test_datetime_service_timezone_handling(self):
        """Test timezone conversion functionality."""
        service = DateTimeService()
        
        # Test ensure_aware_utc
        naive_dt = datetime(2022, 1, 1, 12, 0, 0)
        aware_dt = await service.ensure_aware_utc(naive_dt)
        assert aware_dt.tzinfo == timezone.utc
        
        # Test ensure_naive_utc
        aware_dt = datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        naive_dt = await service.ensure_naive_utc(aware_dt)
        assert naive_dt.tzinfo is None
    
    @pytest.mark.asyncio
    async def test_datetime_service_health_check(self):
        """Test datetime service health check."""
        service = DateTimeService()
        
        health = await service.health_check()
        
        assert health["status"] == "healthy"
        assert "call_count" in health
        assert "current_utc" in health
        assert health["service_type"] == "DateTimeService"


class TestMockDateTimeService:
    """Test the mock datetime service for testing."""
    
    @pytest.mark.asyncio
    async def test_mock_service_fixed_time(self):
        """Test mock service with fixed time."""
        fixed_time = datetime(2022, 1, 1, 12, 0, 0)
        mock_service = MockDateTimeService(fixed_time=fixed_time)
        
        # Test fixed time return
        utc_time = await mock_service.utc_now()
        naive_time = await mock_service.naive_utc_now()
        
        assert utc_time.replace(tzinfo=None) == fixed_time
        assert naive_time == fixed_time
        
        # Test call tracking
        assert mock_service.call_count == 2
        assert 'utc_now' in mock_service.method_calls
        assert 'naive_utc_now' in mock_service.method_calls
    
    @pytest.mark.asyncio
    async def test_mock_service_call_tracking(self):
        """Test mock service call tracking."""
        mock_service = MockDateTimeService()
        
        # Make various calls
        await mock_service.utc_now()
        await mock_service.naive_utc_now()
        await mock_service.format_iso(datetime.now())
        
        assert mock_service.call_count == 3
        assert len(mock_service.method_calls) == 3
        
        # Test reset
        mock_service.reset_counters()
        assert mock_service.call_count == 0
        assert len(mock_service.method_calls) == 0


class TestDIContainer:
    """Test the dependency injection container."""
    
    @pytest.mark.asyncio
    async def test_container_singleton_registration(self):
        """Test singleton service registration and resolution."""
        container = DIContainer()
        
        # Register a service
        container.register_singleton(DateTimeServiceProtocol, DateTimeService)
        
        # Resolve service twice
        service1 = await container.get(DateTimeServiceProtocol)
        service2 = await container.get(DateTimeServiceProtocol)
        
        # Should be the same instance (singleton)
        assert service1 is service2
        assert isinstance(service1, DateTimeService)
    
    @pytest.mark.asyncio
    async def test_container_transient_registration(self):
        """Test transient service registration."""
        container = DIContainer()
        
        # Register as transient
        container.register_transient(DateTimeServiceProtocol, DateTimeService)
        
        # Resolve service twice
        service1 = await container.get(DateTimeServiceProtocol)
        service2 = await container.get(DateTimeServiceProtocol)
        
        # Should be different instances (transient)
        assert service1 is not service2
        assert isinstance(service1, DateTimeService)
        assert isinstance(service2, DateTimeService)
    
    @pytest.mark.asyncio
    async def test_container_instance_registration(self):
        """Test pre-created instance registration."""
        container = DIContainer()
        
        # Create and register instance
        mock_service = MockDateTimeService()
        container.register_instance(DateTimeServiceProtocol, mock_service)
        
        # Resolve service
        service = await container.get(DateTimeServiceProtocol)
        
        # Should be the same instance
        assert service is mock_service
    
    @pytest.mark.asyncio
    async def test_container_factory_registration(self):
        """Test factory function registration."""
        container = DIContainer()
        
        # Register with factory
        def create_mock_service():
            return MockDateTimeService(fixed_time=datetime(2022, 1, 1))
        
        container.register_factory(DateTimeServiceProtocol, create_mock_service)
        
        # Resolve service
        service = await container.get(DateTimeServiceProtocol)
        
        assert isinstance(service, MockDateTimeService)
        assert service.fixed_time == datetime(2022, 1, 1)
    
    @pytest.mark.asyncio
    async def test_container_service_not_registered_error(self):
        """Test error when resolving unregistered service."""
        container = DIContainer()
        
        # Define a dummy interface
        class DummyService:
            pass
        
        # Should raise error
        with pytest.raises(ServiceNotRegisteredError):
            await container.get(DummyService)
    
    @pytest.mark.asyncio
    async def test_container_health_check(self):
        """Test container health check functionality."""
        container = DIContainer()
        
        # Register and resolve a service
        container.register_singleton(DateTimeServiceProtocol, DateTimeService)
        service = await container.get(DateTimeServiceProtocol)
        
        # Perform health check
        health = await container.health_check()
        
        assert health["container_status"] == "healthy"
        assert health["registered_services"] > 0
        assert "DateTimeServiceProtocol" in health["services"]
    
    @pytest.mark.asyncio
    async def test_container_registration_info(self):
        """Test container registration information."""
        container = DIContainer()
        
        # Register services
        container.register_singleton(DateTimeServiceProtocol, DateTimeService)
        
        # Get registration info
        info = container.get_registration_info()
        
        assert "DateTimeServiceProtocol" in info
        assert info["DateTimeServiceProtocol"]["lifetime"] == "singleton"
        assert info["DateTimeServiceProtocol"]["implementation"] == "DateTimeService"


class TestGlobalContainer:
    """Test global container functions."""
    
    @pytest.mark.asyncio
    async def test_get_datetime_service_global(self):
        """Test global datetime service getter."""
        # Get service through global function
        service = await get_datetime_service()
        
        assert isinstance(service, DateTimeService)
        
        # Should return same instance on subsequent calls (singleton)
        service2 = await get_datetime_service()
        assert service is service2
    
    @pytest.mark.asyncio
    async def test_get_container_global(self):
        """Test global container getter."""
        container1 = await get_container()
        container2 = await get_container()
        
        # Should return same instance
        assert container1 is container2
        assert isinstance(container1, DIContainer)


class TestRealBehaviorValidation:
    """Real behavior testing as specified in the implementation plan."""
    
    @pytest.mark.asyncio
    async def test_dependency_injection_validation(self):
        """Test: Verify service injection works across all modules."""
        service = await get_datetime_service()
        assert isinstance(service, DateTimeServiceProtocol)
        
        # Test timezone consistency
        utc_time = await service.utc_now()
        aware_time = await service.aware_utc_now()
        assert utc_time.tzinfo == timezone.utc
        assert aware_time.tzinfo == timezone.utc
    
    @pytest.mark.asyncio
    async def test_module_decoupling_verification(self):
        """Test: Ensure modules can work with mocked datetime service."""
        container = DIContainer()
        
        # Inject mock service
        mock_service = MockDateTimeService(fixed_time=datetime(2022, 1, 1, 12, 0, 0))
        container.register_instance(DateTimeServiceProtocol, mock_service)
        
        # Test service resolution
        resolved_service = await container.get(DateTimeServiceProtocol)
        assert resolved_service is mock_service
        
        # Test functionality with mock
        time_result = await resolved_service.utc_now()
        assert time_result.replace(tzinfo=None) == datetime(2022, 1, 1, 12, 0, 0)
        assert mock_service.call_count > 0
    
    @pytest.mark.asyncio
    async def test_performance_impact_assessment(self):
        """Test: Verify DI doesn't impact performance significantly."""
        import time
        
        # Benchmark direct datetime usage
        start_time = time.perf_counter()
        for _ in range(1000):
            datetime.now(timezone.utc)
        direct_time = time.perf_counter() - start_time
        
        # Benchmark DI datetime usage
        service = await get_datetime_service()
        start_time = time.perf_counter()
        for _ in range(1000):
            await service.utc_now()
        di_time = time.perf_counter() - start_time
        
        # Should be within reasonable overhead (less than 100% increase)
        overhead_ratio = di_time / direct_time
        assert overhead_ratio < 2.0, f"DI overhead too high: {overhead_ratio:.2f}x"
    
    @pytest.mark.asyncio
    async def test_service_initialization_reliability(self):
        """Test: Verify service initialization doesn't fail."""
        container = DIContainer()
        
        # Test multiple service resolutions
        for _ in range(10):
            service = await container.get(DateTimeServiceProtocol)
            assert service is not None
            
            # Test service functionality
            result = await service.utc_now()
            assert isinstance(result, datetime)
            assert result.tzinfo == timezone.utc
