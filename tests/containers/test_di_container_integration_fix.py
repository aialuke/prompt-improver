"""
DI Container Integration Fix Validation
======================================

Tests the fix for the RealTimeAnalyticsService dependency injection issue
following 2025 best practices for handling deprecated services.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_di_container_integration_fix():
    """
    Comprehensive test for DI container integration fix

    Tests:
    1. Health monitor factory registration
    2. Deprecated RealTimeAnalyticsService handling
    3. Modern analytics factory pattern
    4. AutoML callback integration
    5. Error handling for missing services
    """
    print("🧪 DI Container Integration Fix Validation")
    print("=" * 50)
    try:
        from prompt_improver.core.di.container_orchestrator import get_container

        container = await get_container()
        print("✅ DI container created successfully")
        print("\n1️⃣ Testing Health Monitor Factory Registration...")
        container.register_health_monitor_factory()
        from prompt_improver.performance.monitoring.health.unified_health_system import (
            HealthService,
        )

        health_service = await container.get(HealthService)
        assert health_service is not None, "Health service should not be None"
        print(f"✅ HealthService resolved: {type(health_service).__name__}")
        print("\n2️⃣ Testing Modern Service Registration...")
        info = container.get_registration_info()
        modern_services = [
            name
            for name, details in info.items()
            if "deprecated" not in str(details).lower()
        ]
        assert len(modern_services) > 0, "Should have modern services registered"
        print(f"✅ Modern services registered: {len(modern_services)} services")
        print("\n3️⃣ Testing Modern Analytics Factory Pattern...")
        from prompt_improver.core.services.analytics_factory import get_analytics_router

        analytics_router = get_analytics_router()
        assert analytics_router is not None, "Analytics router should not be None"
        print(f"✅ Modern analytics router: {type(analytics_router).__name__}")
        print("\n4️⃣ Testing AutoML Callback Integration...")
        from prompt_improver.ml.automl.callbacks import (
            RealTimeAnalyticsCallback,
            create_standard_callbacks,
        )

        callback = RealTimeAnalyticsCallback(analytics_router)
        assert callback is not None, "Callback should not be None"
        print(f"✅ RealTimeAnalyticsCallback created: {type(callback).__name__}")

        class MockOrchestrator:
            def __init__(self):
                self.experiment_orchestrator = None
                self.model_manager = None

        mock_orchestrator = MockOrchestrator()
        callbacks = create_standard_callbacks(mock_orchestrator)
        assert len(callbacks) >= 1, "Should create at least one callback"
        print(f"✅ Standard callbacks created: {len(callbacks)} callbacks")
        analytics_callbacks = [
            cb for cb in callbacks if isinstance(cb, RealTimeAnalyticsCallback)
        ]
        print(
            f"✅ Analytics callbacks using modern pattern: {len(analytics_callbacks)}"
        )
        print("\n5️⃣ Testing Error Handling...")
        from prompt_improver.core.di.container_orchestrator import ServiceNotRegisteredError

        NonExistentService = type("NonExistentService", (), {})
        try:
            await container.get(NonExistentService)
            assert False, "Should have raised ServiceNotRegisteredError"
        except ServiceNotRegisteredError as e:
            print(f"✅ Proper error handling: {type(e).__name__}")
        print("\n6️⃣ Testing Clean Architecture...")
        info = container.get_registration_info()
        deprecated_count = len([
            name
            for name, details in info.items()
            if "deprecated" in str(details).lower() or "legacy" in str(details).lower()
        ])
        assert deprecated_count == 0, (
            f"Found {deprecated_count} deprecated services - should be 0"
        )
        print("✅ Clean architecture: No deprecated services registered")
        print("\n🎉 All DI Container Integration Tests Passed!")
        print("\n📋 Summary:")
        print("  ✅ Health monitor factory registration working")
        print("  ✅ Modern service registration only - no deprecated services")
        print("  ✅ Modern analytics factory pattern adopted")
        print("  ✅ AutoML callbacks using modern services")
        print("  ✅ Proper error handling for missing services")
        print("  ✅ Clean architecture with no backward compatibility shims")
        print("\n🏆 DI Container Integration Fix: SUCCESS")
        print("🔥 Following 2025 dependency injection best practices!")
        return True
    except Exception as e:
        print(f"\n❌ DI Container Integration Test Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the integration fix validation"""
    success = await test_di_container_integration_fix()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
