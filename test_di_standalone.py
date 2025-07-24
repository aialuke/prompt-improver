#!/usr/bin/env python3
"""Standalone test for datetime service dependency injection.

This test validates the DI implementation without importing
the full prompt_improver module to avoid circular dependencies.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for direct imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Direct imports to avoid circular dependencies
from prompt_improver.core.interfaces.datetime_service import (
    DateTimeServiceProtocol, 
    MockDateTimeService
)
from prompt_improver.core.services.datetime_service import DateTimeService
from prompt_improver.core.di.container import DIContainer


async def test_datetime_service_basic():
    """Test basic datetime service functionality."""
    print("Testing DateTimeService basic functionality...")
    
    service = DateTimeService()
    
    # Test UTC now methods
    utc_time = await service.utc_now()
    aware_time = await service.aware_utc_now()
    naive_time = await service.naive_utc_now()
    
    assert utc_time.tzinfo == timezone.utc
    assert aware_time.tzinfo == timezone.utc
    assert naive_time.tzinfo is None
    
    print(f"✓ UTC time: {utc_time}")
    print(f"✓ Aware time: {aware_time}")
    print(f"✓ Naive time: {naive_time}")
    
    # Test timestamp conversion
    timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
    aware_dt = await service.from_timestamp(timestamp, aware=True)
    naive_dt = await service.from_timestamp(timestamp, aware=False)
    
    assert aware_dt.tzinfo == timezone.utc
    assert naive_dt.tzinfo is None
    assert aware_dt.year == 2022
    assert naive_dt.year == 2022
    
    print(f"✓ Timestamp conversion: {aware_dt}")
    
    # Test ISO formatting
    iso_string = await service.format_iso(utc_time)
    assert isinstance(iso_string, str)
    assert 'T' in iso_string
    
    print(f"✓ ISO format: {iso_string}")
    
    # Test health check
    health = await service.health_check()
    assert health["status"] == "healthy"
    
    print(f"✓ Health check: {health['status']}")
    print("DateTimeService basic tests passed!\n")


async def test_mock_service():
    """Test mock datetime service."""
    print("Testing MockDateTimeService...")
    
    fixed_time = datetime(2022, 1, 1, 12, 0, 0)
    mock_service = MockDateTimeService(fixed_time=fixed_time)
    
    # Test fixed time return
    utc_time = await mock_service.utc_now()
    naive_time = await mock_service.naive_utc_now()
    
    assert utc_time.replace(tzinfo=None) == fixed_time
    assert naive_time == fixed_time
    
    print(f"✓ Fixed time UTC: {utc_time}")
    print(f"✓ Fixed time naive: {naive_time}")
    
    # Test call tracking
    assert mock_service.call_count == 2
    assert 'utc_now' in mock_service.method_calls
    assert 'naive_utc_now' in mock_service.method_calls
    
    print(f"✓ Call count: {mock_service.call_count}")
    print(f"✓ Method calls: {mock_service.method_calls}")
    
    # Test reset
    mock_service.reset_counters()
    assert mock_service.call_count == 0
    assert len(mock_service.method_calls) == 0
    
    print("✓ Counter reset successful")
    print("MockDateTimeService tests passed!\n")


async def test_di_container():
    """Test dependency injection container."""
    print("Testing DIContainer...")
    
    container = DIContainer()
    
    # Test singleton registration and resolution
    container.register_singleton(DateTimeServiceProtocol, DateTimeService)
    
    service1 = await container.get(DateTimeServiceProtocol)
    service2 = await container.get(DateTimeServiceProtocol)
    
    # Should be the same instance (singleton)
    assert service1 is service2
    assert isinstance(service1, DateTimeService)
    
    print("✓ Singleton registration and resolution")
    
    # Test transient registration
    container.register_transient(DateTimeServiceProtocol, DateTimeService)
    
    service3 = await container.get(DateTimeServiceProtocol)
    service4 = await container.get(DateTimeServiceProtocol)
    
    # Should be different instances (transient)
    assert service3 is not service4
    assert isinstance(service3, DateTimeService)
    assert isinstance(service4, DateTimeService)
    
    print("✓ Transient registration and resolution")
    
    # Test instance registration
    mock_service = MockDateTimeService()
    container.register_instance(DateTimeServiceProtocol, mock_service)
    
    service5 = await container.get(DateTimeServiceProtocol)
    assert service5 is mock_service
    
    print("✓ Instance registration")
    
    # Test factory registration
    def create_mock_service():
        return MockDateTimeService(fixed_time=datetime(2022, 1, 1))
    
    container.register_factory(DateTimeServiceProtocol, create_mock_service)
    
    service6 = await container.get(DateTimeServiceProtocol)
    assert isinstance(service6, MockDateTimeService)
    assert service6.fixed_time == datetime(2022, 1, 1)
    
    print("✓ Factory registration")
    
    # Test health check
    health = await container.health_check()
    assert health["container_status"] in ["healthy", "degraded"]
    assert health["registered_services"] > 0
    
    print(f"✓ Container health: {health['container_status']}")
    
    # Test registration info
    info = container.get_registration_info()
    assert "DateTimeServiceProtocol" in info
    
    print(f"✓ Registration info: {len(info)} services registered")
    print("DIContainer tests passed!\n")


async def test_performance_impact():
    """Test performance impact of DI vs direct usage."""
    print("Testing performance impact...")
    
    import time
    
    # Benchmark direct datetime usage
    start_time = time.perf_counter()
    for _ in range(1000):
        datetime.now(timezone.utc)
    direct_time = time.perf_counter() - start_time
    
    # Benchmark DI datetime usage
    container = DIContainer()
    service = await container.get(DateTimeServiceProtocol)
    
    start_time = time.perf_counter()
    for _ in range(1000):
        await service.utc_now()
    di_time = time.perf_counter() - start_time
    
    overhead_ratio = di_time / direct_time
    
    print(f"✓ Direct time: {direct_time:.4f}s")
    print(f"✓ DI time: {di_time:.4f}s")
    print(f"✓ Overhead ratio: {overhead_ratio:.2f}x")
    
    # Should be within reasonable overhead
    assert overhead_ratio < 5.0, f"DI overhead too high: {overhead_ratio:.2f}x"
    
    print("Performance impact test passed!\n")


async def test_real_behavior_scenarios():
    """Test real behavior scenarios from implementation plan."""
    print("Testing real behavior scenarios...")
    
    # Scenario 1: Dependency injection validation
    container = DIContainer()
    service = await container.get(DateTimeServiceProtocol)
    assert isinstance(service, DateTimeServiceProtocol)
    
    utc_time = await service.utc_now()
    aware_time = await service.aware_utc_now()
    assert utc_time.tzinfo == timezone.utc
    assert aware_time.tzinfo == timezone.utc
    
    print("✓ Dependency injection validation")
    
    # Scenario 2: Module decoupling verification
    mock_service = MockDateTimeService(fixed_time=datetime(2022, 1, 1, 12, 0, 0))
    container.register_instance(DateTimeServiceProtocol, mock_service)
    
    resolved_service = await container.get(DateTimeServiceProtocol)
    assert resolved_service is mock_service
    
    time_result = await resolved_service.utc_now()
    assert time_result.replace(tzinfo=None) == datetime(2022, 1, 1, 12, 0, 0)
    assert mock_service.call_count > 0
    
    print("✓ Module decoupling verification")
    
    # Scenario 3: Service initialization reliability
    for _ in range(10):
        new_container = DIContainer()
        service = await new_container.get(DateTimeServiceProtocol)
        assert service is not None
        
        result = await service.utc_now()
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc
    
    print("✓ Service initialization reliability")
    print("Real behavior scenarios passed!\n")


async def main():
    """Run all tests."""
    print("=== DateTime Service DI Implementation Tests ===\n")
    
    try:
        await test_datetime_service_basic()
        await test_mock_service()
        await test_di_container()
        await test_performance_impact()
        await test_real_behavior_scenarios()
        
        print("🎉 All tests passed successfully!")
        print("\n=== Implementation Status ===")
        print("✅ DateTimeServiceProtocol interface defined")
        print("✅ DateTimeService implementation complete")
        print("✅ MockDateTimeService for testing ready")
        print("✅ DIContainer with full functionality")
        print("✅ Performance impact within acceptable limits")
        print("✅ Real behavior validation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
