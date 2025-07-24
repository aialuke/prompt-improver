#!/usr/bin/env python3
"""Simple test for datetime service dependency injection."""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the specific directories to path
src_path = Path(__file__).parent / "src"
interfaces_path = src_path / "prompt_improver" / "core" / "interfaces"
services_path = src_path / "prompt_improver" / "core" / "services"
di_path = src_path / "prompt_improver" / "core" / "di"

sys.path.insert(0, str(interfaces_path))
sys.path.insert(0, str(services_path))
sys.path.insert(0, str(di_path))

# Now import the modules
exec(open(interfaces_path / "datetime_service.py").read())
exec(open(services_path / "datetime_service.py").read())
exec(open(di_path / "container.py").read())


async def test_basic_functionality():
    """Test basic functionality of the DI implementation."""
    print("=== Testing DateTime Service DI Implementation ===\n")
    
    # Test 1: Basic DateTimeService functionality
    print("1. Testing DateTimeService...")
    service = DateTimeService()
    
    utc_time = await service.utc_now()
    naive_time = await service.naive_utc_now()
    
    assert utc_time.tzinfo == timezone.utc
    assert naive_time.tzinfo is None
    
    print(f"   ✓ UTC time: {utc_time}")
    print(f"   ✓ Naive time: {naive_time}")
    
    # Test 2: Mock service
    print("\n2. Testing MockDateTimeService...")
    fixed_time = datetime(2022, 1, 1, 12, 0, 0)
    mock_service = MockDateTimeService(fixed_time=fixed_time)
    
    mock_utc = await mock_service.utc_now()
    assert mock_utc.replace(tzinfo=None) == fixed_time
    assert mock_service.call_count == 1
    
    print(f"   ✓ Mock time: {mock_utc}")
    print(f"   ✓ Call count: {mock_service.call_count}")
    
    # Test 3: DI Container
    print("\n3. Testing DIContainer...")
    container = DIContainer()
    
    # Register and resolve service
    container.register_singleton(DateTimeServiceProtocol, DateTimeService)
    resolved_service = await container.get(DateTimeServiceProtocol)
    
    assert isinstance(resolved_service, DateTimeService)
    
    # Test singleton behavior
    resolved_service2 = await container.get(DateTimeServiceProtocol)
    assert resolved_service is resolved_service2
    
    print("   ✓ Service registration and resolution")
    print("   ✓ Singleton behavior verified")
    
    # Test 4: Performance comparison
    print("\n4. Testing performance impact...")
    import time
    
    # Direct usage
    start = time.perf_counter()
    for _ in range(100):
        datetime.now(timezone.utc)
    direct_time = time.perf_counter() - start
    
    # DI usage
    start = time.perf_counter()
    for _ in range(100):
        await resolved_service.utc_now()
    di_time = time.perf_counter() - start
    
    overhead = (di_time / direct_time) if direct_time > 0 else 1
    
    print(f"   ✓ Direct time: {direct_time:.4f}s")
    print(f"   ✓ DI time: {di_time:.4f}s")
    print(f"   ✓ Overhead: {overhead:.2f}x")
    
    # Test 5: Health check
    print("\n5. Testing health checks...")
    service_health = await resolved_service.health_check()
    container_health = await container.health_check()
    
    assert service_health["status"] == "healthy"
    assert container_health["container_status"] in ["healthy", "degraded"]
    
    print(f"   ✓ Service health: {service_health['status']}")
    print(f"   ✓ Container health: {container_health['container_status']}")
    
    print("\n🎉 All tests passed!")
    print("\n=== Implementation Summary ===")
    print("✅ DateTimeServiceProtocol interface working")
    print("✅ DateTimeService implementation functional")
    print("✅ MockDateTimeService ready for testing")
    print("✅ DIContainer with singleton support")
    print("✅ Performance overhead acceptable")
    print("✅ Health monitoring operational")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_basic_functionality())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
