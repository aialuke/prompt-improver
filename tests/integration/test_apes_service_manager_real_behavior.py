#!/usr/bin/env python3
"""
Real behavior integration test for APESServiceManager with ML Pipeline Orchestrator.
Tests actual functionality without mocks to ensure successful integration.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APESServiceManagerRealBehaviorTest:
    """Test APESServiceManager real behavior integration."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  Details: {details}")
        self.test_results[test_name] = {"passed": passed, "details": details}
        
    async def test_orchestrator_service_lifecycle(self):
        """Test complete service lifecycle through orchestrator."""
        try:
            from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
            
            # Initialize orchestrator
            orchestrator = MLPipelineOrchestrator()
            await orchestrator.initialize()
            
            self.log_test("Orchestrator initializes with APESServiceManager", True)
            
            # Verify APESServiceManager is loaded and initialized
            loaded_components = orchestrator.component_loader.loaded_components
            self.log_test("APESServiceManager loaded in orchestrator", 
                         "apes_service_manager" in loaded_components)
            
            if "apes_service_manager" in loaded_components:
                component = loaded_components["apes_service_manager"]
                self.log_test("APESServiceManager is initialized", component.is_initialized)
                self.log_test("APESServiceManager has instance", component.instance is not None)
                
                if component.instance:
                    # Test real behavior - service status
                    status = await orchestrator.invoke_component("apes_service_manager", "get_service_status")
                    self.log_test("Service status returns valid data", isinstance(status, dict))
                    self.log_test("Status includes orchestrator integration", 
                                 status.get("orchestrator_integration", False))
                    self.log_test("Event bus connection detected in status", 
                                 status.get("event_bus_connected", False))
                    
                    # Test event emission capability
                    instance = component.instance
                    if hasattr(instance, '_emit_service_event'):
                        try:
                            await instance._emit_service_event("service.test", {"test": True})
                            self.log_test("Service can emit events to orchestrator", True)
                        except Exception as e:
                            self.log_test("Service can emit events to orchestrator", False, str(e))
                    
            await orchestrator.shutdown()
            
        except Exception as e:
            self.log_test("Orchestrator initializes with APESServiceManager", False, str(e))
            
    async def test_service_manager_event_integration(self):
        """Test service manager event integration with real event bus."""
        try:
            from prompt_improver.ml.orchestration.events.event_bus import EventBus
            from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
            from prompt_improver.ml.orchestration.events.event_types import EventType
            from prompt_improver.core.services.manager import APESServiceManager
            from rich.console import Console
            
            # Create real event bus
            config = OrchestratorConfig()
            event_bus = EventBus(config)
            await event_bus.initialize()
            
            # Create service manager with event bus
            console = Console()
            service_manager = APESServiceManager(console, event_bus)
            
            self.log_test("Service manager creates with real event bus", True)
            
            # Test real event emission and reception
            events_received = []
            
            def event_handler(event):
                events_received.append(event)
                logger.info(f"Received event: {event.event_type.value} from {event.source}")
                
            # Subscribe to component events
            subscription_id = event_bus.subscribe(EventType.COMPONENT_STARTED, event_handler)
            
            # Give event bus time to start processing
            await asyncio.sleep(0.1)
            
            # Emit real service event
            await service_manager._emit_service_event("service.started", {
                "status": "running",
                "test_mode": True,
                "timestamp": "2025-01-23T00:00:00Z"
            })
            
            # Wait for event processing
            await asyncio.sleep(0.2)
            
            self.log_test("Real events are emitted and received", len(events_received) > 0)
            
            if events_received:
                event = events_received[0]
                self.log_test("Event has correct source", event.source == "apes_service_manager")
                self.log_test("Event has correct type", event.event_type == EventType.COMPONENT_STARTED)
                self.log_test("Event has data payload", "test_mode" in event.data)
                
            await event_bus.shutdown()
            
        except Exception as e:
            self.log_test("Service manager creates with real event bus", False, str(e))
            
    async def test_service_status_real_behavior(self):
        """Test service status with real behavior patterns."""
        try:
            from prompt_improver.core.services.manager import APESServiceManager
            from rich.console import Console
            
            # Create service manager
            console = Console()
            service_manager = APESServiceManager(console)
            
            # Test real status behavior
            status = service_manager.get_service_status()
            
            self.log_test("Service status returns dict", isinstance(status, dict))
            
            # Verify enhanced status fields
            required_fields = [
                "running", "pid", "started_at", "uptime_seconds", "memory_usage_mb",
                "service_status", "is_initialized", "event_bus_connected", "orchestrator_integration"
            ]
            
            missing_fields = [field for field in required_fields if field not in status]
            self.log_test("Status includes all enhanced fields", len(missing_fields) == 0, 
                         f"Missing: {missing_fields}" if missing_fields else "")
            
            # Test orchestrator integration flag
            self.log_test("Orchestrator integration flag is True", 
                         status.get("orchestrator_integration") is True)
            
            # Test service status tracking
            self.log_test("Service status is tracked", 
                         status.get("service_status") in ["unknown", "stopped", "starting", "running", "failed"])
            
        except Exception as e:
            self.log_test("Service status returns dict", False, str(e))
            
    async def test_component_discovery_integration(self):
        """Test that APESServiceManager is properly discovered by orchestrator."""
        try:
            from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
            from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
            
            loader = DirectComponentLoader()
            
            # Test component is in the loader configuration
            tier4_components = loader.component_paths.get(ComponentTier.TIER_4_PERFORMANCE, {})
            self.log_test("APESServiceManager in Tier 4 components", 
                         "apes_service_manager" in tier4_components)
            
            # Test component can be loaded
            component = await loader.load_component("apes_service_manager", ComponentTier.TIER_4_PERFORMANCE)
            self.log_test("Component loads successfully", component is not None)
            
            if component:
                self.log_test("Component has correct name", component.name == "apes_service_manager")
                self.log_test("Component has correct module path", 
                             "prompt_improver.core.services.manager" in component.module_path)
                
                # Test component can be initialized
                success = await loader.initialize_component("apes_service_manager")
                self.log_test("Component initializes through loader", success)
                
                if success:
                    instance = component.instance
                    self.log_test("Component instance is created", instance is not None)
                    
                    # Test instance has required methods
                    required_methods = ["start_service", "stop_service", "get_service_status", "shutdown_service"]
                    missing_methods = [method for method in required_methods if not hasattr(instance, method)]
                    self.log_test("Instance has all required methods", len(missing_methods) == 0,
                                 f"Missing: {missing_methods}" if missing_methods else "")
                    
        except Exception as e:
            self.log_test("APESServiceManager in Tier 4 components", False, str(e))
            
    async def run_all_tests(self):
        """Run all real behavior tests."""
        logger.info("🚀 Starting APESServiceManager Real Behavior Integration Tests")
        logger.info("=" * 70)
        
        await self.test_component_discovery_integration()
        await self.test_service_status_real_behavior()
        await self.test_service_manager_event_integration()
        await self.test_orchestrator_service_lifecycle()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 Real Behavior Test Results Summary")
        logger.info("=" * 70)
        
        passed_count = sum(1 for result in self.test_results.values() if result["passed"])
        total_count = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅" if result["passed"] else "❌"
            logger.info(f"{status} {test_name}")
            
        logger.info(f"\n📈 Overall: {passed_count}/{total_count} tests passed")
        
        if passed_count == total_count:
            logger.info("🎉 All real behavior tests passed! APESServiceManager integration is fully functional.")
            return True
        else:
            logger.info("⚠️  Some tests failed. Real behavior integration needs attention.")
            return False


async def main():
    """Main test execution."""
    test_runner = APESServiceManagerRealBehaviorTest()
    success = await test_runner.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
