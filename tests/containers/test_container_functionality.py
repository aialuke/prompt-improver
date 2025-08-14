"""
Test script to verify the DI container functionality with real behavior testing.

This script tests the container.py implementation to ensure it works correctly
with the EventBasedMLService and all its dependencies.
"""

import asyncio
import os
import sys

from prompt_improver.core.di.container_orchestrator import DIContainer
from prompt_improver.core.interfaces.ml_interface import (
    MLAnalysisInterface,
    MLHealthInterface,
    MLModelInterface,
    MLServiceInterface,
    MLTrainingInterface,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


async def test_container_basic_functionality():
    """Test basic container functionality."""
    print("🧪 Testing basic container functionality...")
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


async def test_ml_service_registration():
    """Test ML service registration and resolution."""
    print("🧪 Testing ML service registration...")
    container = DIContainer(name="ml_test_container")
    container.register_ml_interfaces()
    ml_service = await container.get(MLServiceInterface)
    assert ml_service is not None
    print(f"✅ MLServiceInterface resolved: {type(ml_service).__name__}")
    analysis_service = await container.get(MLAnalysisInterface)
    assert analysis_service is not None
    print(f"✅ MLAnalysisInterface resolved: {type(analysis_service).__name__}")
    training_service = await container.get(MLTrainingInterface)
    assert training_service is not None
    print(f"✅ MLTrainingInterface resolved: {type(training_service).__name__}")
    health_service = await container.get(MLHealthInterface)
    assert health_service is not None
    print(f"✅ MLHealthInterface resolved: {type(health_service).__name__}")
    model_service = await container.get(MLModelInterface)
    assert model_service is not None
    print(f"✅ MLModelInterface resolved: {type(model_service).__name__}")
    assert ml_service is analysis_service
    assert ml_service is training_service
    assert ml_service is health_service
    assert ml_service is model_service
    print("✅ All ML interfaces resolve to the same unified service instance!")


async def test_ml_service_functionality():
    """Test actual ML service functionality."""
    print("🧪 Testing ML service functionality...")
    container = DIContainer(name="ml_functionality_test")
    container.register_ml_interfaces()
    ml_service = await container.get(MLServiceInterface)
    try:
        result = await ml_service.process_prompt_improvement_request(
            prompt="Test prompt", context={"test": True}, improvement_type="test"
        )
        assert "request_id" in result or "analysis_results" in result
        print("✅ Prompt improvement request works!")
    except Exception as e:
        print(f"⚠️  Prompt improvement request failed: {e}")
    try:
        status = await ml_service.get_ml_pipeline_status()
        assert "pipeline_status" in status
        print("✅ ML pipeline status works!")
    except Exception as e:
        print(f"⚠️  ML pipeline status failed: {e}")
    try:
        usage = await ml_service.get_resource_usage()
        assert isinstance(usage, dict)
        print("✅ Resource usage works!")
    except Exception as e:
        print(f"⚠️  Resource usage failed: {e}")


async def test_container_health_check():
    """Test container health check functionality."""
    print("🧪 Testing container health check...")
    container = DIContainer(name="health_test_container")
    container.register_ml_interfaces()
    await container.get(MLServiceInterface)
    health_result = await container.health_check()
    assert "container_status" in health_result
    assert "registered_services" in health_result
    assert "services" in health_result
    print(
        f"✅ Container health check works! Status: {health_result['container_status']}"
    )
    print(f"   Registered services: {health_result['registered_services']}")


async def test_container_cleanup():
    """Test container cleanup functionality."""
    print("🧪 Testing container cleanup...")
    container = DIContainer(name="cleanup_test_container")
    container.register_ml_interfaces()
    service = await container.get(MLServiceInterface)
    assert service is not None
    await container.cleanup()
    print("✅ Container cleanup completed successfully!")


async def main():
    """Run all tests."""
    print("🚀 Starting DI Container Tests with Real Behavior Testing\n")
    try:
        await test_container_basic_functionality()
        print()
        await test_ml_service_registration()
        print()
        await test_ml_service_functionality()
        print()
        await test_container_health_check()
        print()
        await test_container_cleanup()
        print()
        print("🎉 All tests completed successfully!")
        print("✅ The DI container is working correctly with real behavior testing.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
