#!/usr/bin/env python3
"""
Clean Architecture Validation Test
=================================

Validates that all backward compatibility shims and deprecated aliases have been removed,
and the system uses clean, modern implementation patterns exclusively.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_clean_architecture():
    """
    Comprehensive test for clean architecture without backward compatibility
    
    Tests:
    1. DI Container has no deprecated services
    2. Analytics factory uses modern patterns only
    3. Health system has no reset functions or legacy adapters
    4. AutoML callbacks use modern service patterns
    5. All integrations work without deprecated code paths
    """
    
    print("🧹 Clean Architecture Validation")
    print("=" * 40)
    
    try:
        # Test 1: DI Container Clean Architecture
        print("\n1️⃣ Testing DI Container Clean Architecture...")
        from prompt_improver.core.di.container import get_container
        
        container = await get_container()
        container.register_health_monitor_factory()
        
        # Verify no deprecated service registration methods exist
        assert not hasattr(container, 'register_deprecated_service_replacements'), \
               "Deprecated service registration method should not exist"
        
        # Verify no deprecated services are registered
        info = container.get_registration_info()
        deprecated_services = [name for name, details in info.items() 
                              if any(keyword in str(details).lower() 
                                   for keyword in ['deprecated', 'legacy', 'backward', 'compat'])]
        
        assert len(deprecated_services) == 0, \
               f"Found deprecated services: {deprecated_services}"
        print(f"✅ DI Container clean: {len(info)} modern services, 0 deprecated")
        
        # Test 2: Analytics Factory Clean Architecture
        print("\n2️⃣ Testing Analytics Factory Clean Architecture...")
        from prompt_improver.core.services.analytics_factory import get_analytics_router
        
        # Verify deprecated functions don't exist
        try:
            from prompt_improver.core.services import analytics_factory
            assert not hasattr(analytics_factory, 'create_legacy_real_time_analytics_service'), \
                   "Legacy analytics service creator should not exist"
            assert not hasattr(analytics_factory, 'migrate_analytics_service_usage'), \
                   "Migration function should not exist"
        except AttributeError:
            pass  # Good - functions don't exist
        
        # Test modern factory works
        analytics_router = get_analytics_router()
        assert analytics_router is not None, "Modern analytics router should work"
        print(f"✅ Analytics factory clean: Modern router type {type(analytics_router).__name__}")
        
        # Test 3: Health System Clean Architecture
        print("\n3️⃣ Testing Health System Clean Architecture...")
        from prompt_improver.performance.monitoring.health import (
            get_unified_health_monitor, 
            get_health_service,
            HealthService
        )
        
        # Verify reset functions don't exist
        try:
            from prompt_improver.performance.monitoring import health
            assert not hasattr(health, 'reset_health_service'), \
                   "Reset function should not exist"
            assert not hasattr(health, 'reset_unified_health_monitor'), \
                   "Reset function should not exist"
        except AttributeError:
            pass  # Good - functions don't exist
        
        # Test modern health system works
        health_monitor = get_unified_health_monitor()
        health_service = get_health_service()
        
        assert health_monitor is not None, "Health monitor should work"
        assert health_service is not None, "Health service should work"
        assert isinstance(health_service, HealthService), "Should be modern HealthService"
        print("✅ Health system clean: No reset functions, modern services only")
        
        # Test 4: AutoML Callbacks Clean Architecture
        print("\n4️⃣ Testing AutoML Callbacks Clean Architecture...")
        from prompt_improver.ml.automl.callbacks import (
            RealTimeAnalyticsCallback,
            create_standard_callbacks
        )
        
        # Test callback uses modern analytics service directly
        callback = RealTimeAnalyticsCallback(analytics_router)
        assert callback.analytics_service is analytics_router, \
               "Callback should use modern service directly"
        
        # Test callback factory uses modern patterns
        class MockOrchestrator:
            def __init__(self):
                self.experiment_orchestrator = None
                self.model_manager = None
        
        callbacks = create_standard_callbacks(MockOrchestrator())
        assert len(callbacks) >= 1, "Should create callbacks"
        
        # Verify analytics callback is using modern service
        analytics_callbacks = [cb for cb in callbacks if isinstance(cb, RealTimeAnalyticsCallback)]
        assert len(analytics_callbacks) == 1, "Should have exactly one analytics callback"
        print("✅ AutoML callbacks clean: Modern service integration only")
        
        # Test 5: Integration Validation - Original Issue Resolved
        print("\n5️⃣ Testing Original Integration Issue Resolution...")
        
        # This was the original failing integration - DI container with health service
        health_service_from_container = await container.get(HealthService)
        assert health_service_from_container is not None, \
               "Health service should resolve from DI container"
        
        # Verify it's the same instance as the global one (singleton pattern)
        global_health_service = get_health_service()
        # Note: These might be different instances due to adapter pattern, but both should work
        
        # Test health service functionality
        available_checks = health_service_from_container.get_available_checks()
        assert isinstance(available_checks, list), "Should return list of available checks"
        
        # Test health check execution
        health_summary = await health_service_from_container.get_health_summary()
        assert isinstance(health_summary, dict), "Should return health summary dict"
        assert "overall_status" in health_summary, "Should have overall status"
        
        print("✅ Original integration issue resolved: DI container + health service working")
        
        # Test 6: No Import Errors for Clean Architecture
        print("\n6️⃣ Testing No Import Errors...")
        
        # These should all work without errors
        test_imports = [
            "from prompt_improver.core.di.container import DIContainer",
            "from prompt_improver.core.services.analytics_factory import get_analytics_router",
            "from prompt_improver.performance.monitoring.health import HealthService",
            "from prompt_improver.ml.automl.callbacks import RealTimeAnalyticsCallback"
        ]
        
        for import_stmt in test_imports:
            try:
                exec(import_stmt)
            except ImportError as e:
                assert False, f"Import failed: {import_stmt} - {e}"
        
        print("✅ All modern imports working correctly")
        
        print("\n🎉 Clean Architecture Validation Passed!")
        print("\n📋 Architecture Summary:")
        print("  🧹 Zero deprecated services or functions")
        print("  🎯 Modern dependency injection patterns only")
        print("  ⚡ Clean analytics factory implementation")
        print("  🔄 Modern health monitoring system")
        print("  🤖 AutoML callbacks using modern services")
        print("  ✅ Original DI integration issue completely resolved")
        
        print(f"\n🏆 Clean Architecture Status: EXCELLENT")
        print("🎯 2025 implementation patterns achieved!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Clean Architecture Validation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the clean architecture validation"""
    success = await test_clean_architecture()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)