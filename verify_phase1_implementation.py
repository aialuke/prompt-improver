#!/usr/bin/env python3
"""
Quick verification script for Phase 1 ML Pipeline Orchestration implementation.
Runs essential checks to confirm the implementation is working correctly.
"""

import asyncio
import sys
import os
import time

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def verify_phase1():
    """Run essential verification checks for Phase 1."""
    print("🔍 Phase 1 ML Pipeline Orchestration - Quick Verification")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 6
    
    # Check 1: Event System
    try:
        from prompt_improver.ml.orchestration.events.event_bus import EventBus
        from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        event_bus = EventBus(OrchestratorConfig())
        await event_bus.initialize()
        
        # Test event emission
        test_event = MLEvent(
            event_type=EventType.TRAINING_STARTED,
            source="verification",
            data={"test": True}
        )
        await event_bus.emit(test_event)
        await event_bus.shutdown()
        
        print("✅ Event System: Working")
        checks_passed += 1
        
    except Exception as e:
        print(f"❌ Event System: Failed - {e}")
    
    # Check 2: Configuration System
    try:
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        from prompt_improver.ml.orchestration.config.component_definitions import ComponentDefinitions
        
        config = OrchestratorConfig()
        errors = config.validate()
        assert len(errors) == 0
        
        comp_defs = ComponentDefinitions()
        components = comp_defs.get_all_component_definitions()
        assert len(components) >= 19
        
        print(f"✅ Configuration System: Working ({len(components)} components)")
        checks_passed += 1
        
    except Exception as e:
        print(f"❌ Configuration System: Failed - {e}")
    
    # Check 3: Training Coordinator
    try:
        from prompt_improver.ml.orchestration.coordinators.training_workflow_coordinator import (
            TrainingWorkflowCoordinator, TrainingWorkflowConfig
        )
        
        coordinator = TrainingWorkflowCoordinator(TrainingWorkflowConfig())
        active = await coordinator.list_active_workflows()
        assert isinstance(active, list)
        
        print("✅ Training Coordinator: Working")
        checks_passed += 1
        
    except Exception as e:
        print(f"❌ Training Coordinator: Failed - {e}")
    
    # Check 4: Component Connectors
    try:
        from prompt_improver.ml.orchestration.connectors.tier1_connectors import (
            TrainingDataLoaderConnector, Tier1ConnectorFactory
        )
        
        connector = TrainingDataLoaderConnector()
        capabilities = connector.get_capabilities()
        assert len(capabilities) > 0
        
        available = Tier1ConnectorFactory.list_available_components()
        assert len(available) >= 11
        
        print(f"✅ Component Connectors: Working ({len(available)} available)")
        checks_passed += 1
        
    except Exception as e:
        print(f"❌ Component Connectors: Failed - {e}")
    
    # Check 5: Component Registry
    try:
        from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry(OrchestratorConfig())
        await registry.initialize()
        
        components = await registry.list_components()
        assert len(components) >= 19
        
        health = await registry.get_health_summary()
        assert health["total_components"] >= 19
        
        await registry.shutdown()
        
        print(f"✅ Component Registry: Working ({health['total_components']} components)")
        checks_passed += 1
        
    except Exception as e:
        print(f"❌ Component Registry: Failed - {e}")
    
    # Check 6: End-to-End Workflow
    try:
        event_bus = EventBus(OrchestratorConfig())
        await event_bus.initialize()
        
        coordinator = TrainingWorkflowCoordinator(TrainingWorkflowConfig(), event_bus)
        
        start_time = time.time()
        await coordinator.start_training_workflow("verification_workflow", {"test": True})
        execution_time = time.time() - start_time
        
        status = await coordinator.get_workflow_status("verification_workflow")
        assert status["status"] == "completed"
        assert len(status["steps_completed"]) == 3
        
        await event_bus.shutdown()
        
        print(f"✅ End-to-End Workflow: Working ({execution_time:.3f}s)")
        checks_passed += 1
        
    except Exception as e:
        print(f"❌ End-to-End Workflow: Failed - {e}")
    
    # Summary
    print("-" * 60)
    success_rate = (checks_passed / total_checks) * 100
    
    if checks_passed == total_checks:
        print(f"🎉 SUCCESS: All {checks_passed}/{total_checks} verification checks passed ({success_rate:.1f}%)")
        print("Phase 1 implementation is working correctly!")
        return True
    else:
        print(f"⚠️  PARTIAL: {checks_passed}/{total_checks} verification checks passed ({success_rate:.1f}%)")
        print("Some components may need attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_phase1())
    sys.exit(0 if success else 1)