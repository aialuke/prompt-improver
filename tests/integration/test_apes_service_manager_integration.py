#!/usr/bin/env python3
"""
Test APESServiceManager integration with ML Pipeline Orchestrator.
Verifies 2025 best practices for service management integration.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APESServiceManagerIntegrationTest:
    """Test APESServiceManager integration with orchestrator."""
    
    def __init__(self):
        self.test_results = {}
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  Details: {details}")
        self.test_results[test_name] = {"passed": passed, "details": details}
        
    async def test_component_loading(self):
        """Test APESServiceManager loads through orchestrator."""
        try:
            from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
            from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
            
            loader = DirectComponentLoader()
            
            # Test component loading
            loaded_component = await loader.load_component("apes_service_manager", ComponentTier.TIER_4_PERFORMANCE)
            self.log_test("Component loads through orchestrator", loaded_component is not None)
            
            if loaded_component:
                # Verify component class
                from prompt_improver.core.services.manager import APESServiceManager
                self.log_test("Correct component class loaded", loaded_component.component_class == APESServiceManager)
                
                # Test component initialization
                success = await loader.initialize_component("apes_service_manager")
                self.log_test("Component initializes successfully", success)
                
                if success:
                    instance = loaded_component.instance
                    self.log_test("Component instance created", instance is not None)
                    self.log_test("Instance is APESServiceManager", isinstance(instance, APESServiceManager))
                    
                    # Test service manager methods
                    self.log_test("Has start_service method", hasattr(instance, 'start_service'))
                    self.log_test("Has stop_service method", hasattr(instance, 'stop_service'))
                    self.log_test("Has get_service_status method", hasattr(instance, 'get_service_status'))
                    
                    # Test service status (should work without actual service running)
                    try:
                        status = instance.get_service_status()
                        self.log_test("Service status check works", isinstance(status, dict))
                    except Exception as e:
                        self.log_test("Service status check works", False, str(e))
                        
        except Exception as e:
            self.log_test("Component loads through orchestrator", False, str(e))
            
    async def test_orchestrator_integration(self):
        """Test APESServiceManager integration with full orchestrator."""
        try:
            from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
            
            # Initialize orchestrator
            orchestrator = MLPipelineOrchestrator()
            await orchestrator.initialize()
            
            self.log_test("Orchestrator initializes", True)
            
            # Test component invocation through orchestrator
            try:
                # This should work if APESServiceManager is properly integrated
                result = await orchestrator.invoke_component("apes_service_manager", "get_service_status")
                self.log_test("Component invocation through orchestrator", isinstance(result, dict))
            except Exception as e:
                self.log_test("Component invocation through orchestrator", False, str(e))
                
        except Exception as e:
            self.log_test("Orchestrator initializes", False, str(e))
            
    async def test_service_lifecycle_integration(self):
        """Test service lifecycle integration patterns."""
        try:
            from prompt_improver.core.services.manager import APESServiceManager
            from rich.console import Console
            
            # Create service manager
            console = Console()
            service_manager = APESServiceManager(console)
            
            # Test initialization
            self.log_test("Service manager creates successfully", True)
            
            # Test configuration
            self.log_test("Has data directory", hasattr(service_manager, 'data_dir'))
            self.log_test("Has PID file path", hasattr(service_manager, 'pid_file'))
            self.log_test("Has logging setup", hasattr(service_manager, 'logger'))
            
            # Test async methods exist
            self.log_test("Has async start_service", hasattr(service_manager, 'start_service'))
            self.log_test("Has async shutdown_service", hasattr(service_manager, 'shutdown_service'))
            
            # Test event integration capability
            self.log_test("Has shutdown event", hasattr(service_manager, 'shutdown_event'))
            
        except Exception as e:
            self.log_test("Service manager creates successfully", False, str(e))
            
    async def test_event_bus_integration(self):
        """Test event bus integration patterns for service management."""
        try:
            from prompt_improver.ml.orchestration.events.event_bus import EventBus
            from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
            from prompt_improver.ml.orchestration.events.event_types import MLEvent, EventType
            
            # Create event bus
            config = OrchestratorConfig()
            event_bus = EventBus(config)
            await event_bus.initialize()
            
            self.log_test("Event bus initializes", True)
            
            # Test service manager could integrate with events
            events_received = []
            
            def service_event_handler(event: MLEvent):
                events_received.append(event)
                
            # Subscribe to service events
            subscription_id = event_bus.subscribe(EventType.COMPONENT_STARTED, service_event_handler)
            self.log_test("Event subscription works", subscription_id is not None)
            
            # Emit a test service event
            test_event = MLEvent(
                event_type=EventType.COMPONENT_STARTED,
                source="apes_service_manager",
                data={"service": "test", "status": "started"}
            )
            
            await event_bus.emit(test_event)
            
            # Give event time to process
            await asyncio.sleep(0.1)
            
            self.log_test("Service events can be emitted", len(events_received) > 0)
            
            await event_bus.shutdown()

        except Exception as e:
            self.log_test("Event bus initializes", False, str(e))

    async def test_enhanced_service_manager_integration(self):
        """Test enhanced service manager with event bus integration."""
        try:
            from prompt_improver.ml.orchestration.events.event_bus import EventBus
            from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
            from prompt_improver.core.services.manager import APESServiceManager
            from rich.console import Console

            # Create event bus
            config = OrchestratorConfig()
            event_bus = EventBus(config)
            await event_bus.initialize()

            # Create service manager with event bus
            console = Console()
            service_manager = APESServiceManager(console, event_bus)

            self.log_test("Service manager with event bus creates", True)

            # Test enhanced status
            status = service_manager.get_service_status()
            self.log_test("Enhanced status includes orchestrator fields",
                         "orchestrator_integration" in status and status["orchestrator_integration"])
            self.log_test("Event bus connection detected", status.get("event_bus_connected", False))

            # Test event emission capability
            events_received = []

            def service_event_handler(event):
                events_received.append(event)

            from prompt_improver.ml.orchestration.events.event_types import EventType
            event_bus.subscribe(EventType.COMPONENT_STARTED, service_event_handler)

            # Test event emission
            await service_manager._emit_service_event("service.started", {"test": True})

            # Give event time to process
            await asyncio.sleep(0.1)

            self.log_test("Service manager can emit events", len(events_received) > 0)

            await event_bus.shutdown()

        except Exception as e:
            self.log_test("Service manager with event bus creates", False, str(e))
            
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("🚀 Starting APESServiceManager Integration Tests")
        logger.info("=" * 60)
        
        await self.test_component_loading()
        await self.test_orchestrator_integration()
        await self.test_service_lifecycle_integration()
        await self.test_event_bus_integration()
        await self.test_enhanced_service_manager_integration()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 Test Results Summary")
        logger.info("=" * 60)
        
        passed_count = sum(1 for result in self.test_results.values() if result["passed"])
        total_count = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅" if result["passed"] else "❌"
            logger.info(f"{status} {test_name}")
            
        logger.info(f"\n📈 Overall: {passed_count}/{total_count} tests passed")
        
        if passed_count == total_count:
            logger.info("🎉 All tests passed! APESServiceManager integration is working correctly.")
            return True
        else:
            logger.info("⚠️  Some tests failed. Integration needs attention.")
            return False


async def main():
    """Main test execution."""
    test_runner = APESServiceManagerIntegrationTest()
    success = await test_runner.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
