"""
Simple test script to verify the DI container functionality.

This script tests the container.py implementation directly without importing
the full APES system to avoid dependency issues.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
logging.basicConfig(level=logging.INFO)


async def test_container_import():
    """Test that we can import the container without errors."""
    print("🧪 Testing container import...")
    try:
        from prompt_improver.core.di.container import DIContainer, ServiceLifetime

        print("✅ Container import successful!")
        return (DIContainer, ServiceLifetime)
    except Exception as e:
        print(f"❌ Container import failed: {e}")
        import traceback

        traceback.print_exc()
        return (None, None)


async def test_basic_container_functionality(DIContainer, ServiceLifetime):
    """Test basic container functionality."""
    print("🧪 Testing basic container functionality...")
    if not DIContainer:
        print("❌ Skipping test - container not available")
        return None
    try:
        container = DIContainer(name="test_container")

        class TestService:
            def __init__(self):
                self.value = "test_value"

        container.register_singleton(TestService, TestService)
        service = await container.get(TestService)
        assert service is not None
        assert service.value == "test_value"
        service2 = await container.get(TestService)
        assert service is service2
        print("✅ Basic container functionality works!")
        return True
    except Exception as e:
        print(f"❌ Basic container test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_factory_registration(DIContainer, ServiceLifetime):
    """Test factory registration functionality."""
    print("🧪 Testing factory registration...")
    if not DIContainer:
        print("❌ Skipping test - container not available")
        return None
    try:
        container = DIContainer(name="factory_test_container")

        class FactoryService:
            def __init__(self, config_value):
                self.config_value = config_value

        def create_factory_service():
            return FactoryService("factory_created")

        container.register_factory(
            FactoryService, create_factory_service, ServiceLifetime.SINGLETON
        )
        service = await container.get(FactoryService)
        assert service is not None
        assert service.config_value == "factory_created"
        print("✅ Factory registration works!")
        return True
    except Exception as e:
        print(f"❌ Factory registration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_container_health_check(DIContainer, ServiceLifetime):
    """Test container health check functionality."""
    print("🧪 Testing container health check...")
    if not DIContainer:
        print("❌ Skipping test - container not available")
        return None
    try:
        container = DIContainer(name="health_test_container")

        class HealthTestService:
            def __init__(self):
                self.status = "healthy"

        container.register_singleton(HealthTestService, HealthTestService)
        await container.get(HealthTestService)
        health_result = await container.health_check()
        assert "container_status" in health_result
        assert "registered_services" in health_result
        assert "services" in health_result
        print(
            f"✅ Container health check works! Status: {health_result['container_status']}"
        )
        print(f"   Registered services: {health_result['registered_services']}")
        return True
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_container_cleanup(DIContainer, ServiceLifetime):
    """Test container cleanup functionality."""
    print("🧪 Testing container cleanup...")
    if not DIContainer:
        print("❌ Skipping test - container not available")
        return None
    try:
        container = DIContainer(name="cleanup_test_container")

        class CleanupTestService:
            def __init__(self):
                self.cleaned_up = False

            def cleanup(self):
                self.cleaned_up = True

        container.register_singleton(CleanupTestService, CleanupTestService)
        service = await container.get(CleanupTestService)
        assert service is not None
        await container.cleanup()
        print("✅ Container cleanup completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Cleanup test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Simple DI Container Tests\n")
    DIContainer, ServiceLifetime = await test_container_import()
    print()
    if not DIContainer:
        print("❌ Cannot proceed with tests - container import failed")
        sys.exit(1)
    tests = [
        test_basic_container_functionality,
        test_factory_registration,
        test_container_health_check,
        test_container_cleanup,
    ]
    passed = 0
    total = len(tests)
    for test in tests:
        try:
            result = await test(DIContainer, ServiceLifetime)
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            print()
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests completed successfully!")
        print("✅ The DI container is working correctly with real behavior testing.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
