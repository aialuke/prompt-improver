#!/usr/bin/env python3
"""
Comprehensive test for Phase 2 ML Pipeline Orchestrator implementation.

Tests all core components:
- MLPipelineOrchestrator
- WorkflowExecutionEngine with real component calls
- ResourceManager with actual allocation algorithms
- ComponentRegistry with component discovery
- All workflow coordinators (Training, Evaluation, Deployment)
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.coordinators.training_workflow_coordinator import TrainingWorkflowCoordinator, TrainingWorkflowConfig
from prompt_improver.ml.orchestration.coordinators.evaluation_pipeline_manager import EvaluationPipelineManager, EvaluationConfig
from prompt_improver.ml.orchestration.coordinators.deployment_controller import DeploymentController, DeploymentConfig
from prompt_improver.ml.orchestration.core.resource_manager import ResourceManager, ResourceType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase2TestRunner:
    """Comprehensive test runner for Phase 2 implementation."""
    
    def __init__(self):
        """Initialize test runner."""
        self.config = OrchestratorConfig()
        self.orchestrator = None
        self.test_results = {}
        
    async def run_all_tests(self) -> None:
        """Run all Phase 2 tests."""
        logger.info("🚀 Starting Phase 2 ML Pipeline Orchestrator Tests")
        
        tests = [
            ("test_orchestrator_initialization", self.test_orchestrator_initialization),
            ("test_component_registry", self.test_component_registry),
            ("test_resource_manager", self.test_resource_manager),
            ("test_workflow_execution_engine", self.test_workflow_execution_engine),
            ("test_training_coordinator", self.test_training_coordinator),
            ("test_evaluation_coordinator", self.test_evaluation_coordinator),
            ("test_deployment_controller", self.test_deployment_controller),
            ("test_end_to_end_workflow", self.test_end_to_end_workflow),
            ("test_resource_allocation", self.test_resource_allocation),
            ("test_component_health_monitoring", self.test_component_health_monitoring)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                logger.info(f"▶️  Running {test_name}")
                await test_func()
                self.test_results[test_name] = "PASSED"
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            except Exception as e:
                self.test_results[test_name] = f"FAILED: {e}"
                logger.error(f"❌ {test_name} FAILED: {e}")
                failed += 1
        
        # Print summary
        logger.info(f"\n📊 Test Summary:")
        logger.info(f"   ✅ Passed: {passed}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   📈 Success Rate: {passed / (passed + failed) * 100:.1f}%")
        
        if failed == 0:
            logger.info("🎉 All Phase 2 tests passed! Implementation is ready.")
        else:
            logger.error("⚠️  Some tests failed. Review implementation.")
    
    async def test_orchestrator_initialization(self) -> None:
        """Test MLPipelineOrchestrator initialization."""
        self.orchestrator = MLPipelineOrchestrator(self.config)
        
        # Test initialization
        await self.orchestrator.initialize()
        assert self.orchestrator._is_initialized, "Orchestrator should be initialized"
        
        # Test component health
        health = await self.orchestrator.get_component_health()
        assert isinstance(health, dict), "Health should be a dictionary"
        
        # Test resource usage
        usage = await self.orchestrator.get_resource_usage()
        assert isinstance(usage, dict), "Resource usage should be a dictionary"
        
        logger.info("   - Orchestrator initialized successfully")
        logger.info(f"   - Component health: {len(health)} components")
        logger.info(f"   - Resource usage tracking: {len(usage)} resource types")
    
    async def test_component_registry(self) -> None:
        """Test ComponentRegistry functionality."""
        registry = self.orchestrator.component_registry
        
        # Test component listing
        components = await registry.list_components()
        assert len(components) > 0, "Should have registered components"
        
        # Test component discovery
        discovered = await registry.discover_components()
        assert isinstance(discovered, list), "Discovery should return a list"
        
        # Test health summary
        health_summary = await registry.get_health_summary()
        assert "total_components" in health_summary, "Health summary should have total count"
        assert health_summary["total_components"] > 0, "Should have components registered"
        
        logger.info(f"   - Registry has {len(components)} components")
        logger.info(f"   - Health summary: {health_summary['overall_health_percentage']:.1f}% healthy")
        
        # Test component capability search
        training_components = await registry.get_components_by_capability("data_loading")
        logger.info(f"   - Found {len(training_components)} components with data_loading capability")
    
    async def test_resource_manager(self) -> None:
        """Test ResourceManager allocation algorithms."""
        resource_manager = self.orchestrator.resource_manager
        
        # Test resource allocation
        allocation_id = await resource_manager.allocate_resource(
            ResourceType.MEMORY, 
            1024 * 1024 * 1024,  # 1GB
            "test_component"
        )
        assert allocation_id is not None, "Should return allocation ID"
        
        # Test usage stats
        stats = await resource_manager.get_usage_stats()
        assert "memory" in stats, "Should have memory usage stats"
        assert stats["memory"]["currently_allocated"] > 0, "Should show allocated memory"
        
        # Test resource release
        released = await resource_manager.release_resource(allocation_id)
        assert released, "Should successfully release resource"
        
        logger.info(f"   - Successfully allocated and released memory resource")
        logger.info(f"   - Resource stats: {len(stats)} types tracked")
    
    async def test_workflow_execution_engine(self) -> None:
        """Test WorkflowExecutionEngine with real component calls."""
        engine = self.orchestrator.workflow_engine
        
        # Test workflow definitions
        definitions = await engine.list_workflow_definitions()
        assert len(definitions) > 0, "Should have workflow definitions"
        
        # Test starting a workflow
        workflow_id = "test_workflow_" + str(asyncio.get_event_loop().time())
        await engine.start_workflow(workflow_id, "training", {"batch_size": 500})
        
        # Wait for workflow to complete
        await asyncio.sleep(0.5)
        
        # Test workflow status
        status = await engine.get_workflow_status(workflow_id)
        assert status["workflow_id"] == workflow_id, "Should return correct workflow ID"
        
        logger.info(f"   - Executed workflow {workflow_id}")
        logger.info(f"   - Workflow status: {status['is_running']}")
        logger.info(f"   - Available definitions: {len(definitions)}")
    
    async def test_training_coordinator(self) -> None:
        """Test TrainingWorkflowCoordinator."""
        config = TrainingWorkflowConfig()
        coordinator = TrainingWorkflowCoordinator(
            config, 
            self.orchestrator.event_bus,
            self.orchestrator.resource_manager
        )
        
        # Test training workflow
        workflow_id = "training_test_" + str(asyncio.get_event_loop().time())
        await coordinator.start_training_workflow(workflow_id, {"batch_size": 1000})
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Test status
        status = await coordinator.get_workflow_status(workflow_id)
        assert status["status"] == "completed", "Training workflow should complete"
        assert len(status["steps_completed"]) == 3, "Should complete all 3 steps"
        
        logger.info(f"   - Training workflow completed successfully")
        logger.info(f"   - Steps completed: {status['steps_completed']}")
    
    async def test_evaluation_coordinator(self) -> None:
        """Test EvaluationPipelineManager."""
        config = EvaluationConfig()
        manager = EvaluationPipelineManager(
            config, 
            self.orchestrator.event_bus,
            self.orchestrator.component_registry
        )
        
        await manager.initialize()
        
        # Test evaluation workflow
        workflow_id = "evaluation_test_" + str(asyncio.get_event_loop().time())
        await manager.start_evaluation_workflow(workflow_id, {"data_size": 5000})
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Test status
        status = await manager.get_evaluation_status(workflow_id)
        assert status["status"] == "completed", "Evaluation workflow should complete"
        assert "evaluation_results" in status, "Should have evaluation results"
        
        logger.info(f"   - Evaluation workflow completed successfully")
        logger.info(f"   - Results: {status['evaluation_results'].get('aggregated_results', {}).get('overall_score', 'N/A')}")
    
    async def test_deployment_controller(self) -> None:
        """Test DeploymentController."""
        config = DeploymentConfig()
        controller = DeploymentController(
            config,
            self.orchestrator.event_bus,
            self.orchestrator.resource_manager
        )
        
        # Test deployment
        deployment_id = "deployment_test_" + str(asyncio.get_event_loop().time())
        deployment_params = {
            "model_version": "v1.2.3",
            "model_artifact_path": "/models/test_model.pkl",
            "strategy": "blue_green"
        }
        
        await controller.start_deployment(deployment_id, deployment_params)
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Test status
        status = await controller.get_deployment_status(deployment_id)
        assert status["status"] == "completed", "Deployment should complete"
        assert len(status["health_checks"]) > 0, "Should have health checks"
        
        logger.info(f"   - Deployment completed successfully")
        logger.info(f"   - Strategy: {status['strategy'].value}")
        logger.info(f"   - Health checks: {len(status['health_checks'])}")
    
    async def test_end_to_end_workflow(self) -> None:
        """Test complete end-to-end workflow through orchestrator."""
        # Start a complete ML workflow
        workflow_id = await self.orchestrator.start_workflow(
            "training", 
            {
                "batch_size": 1000,
                "model_type": "transformer",
                "max_epochs": 10
            }
        )
        
        # Wait for workflow to process
        await asyncio.sleep(1.0)
        
        # Check workflow status
        workflow_status = await self.orchestrator.get_workflow_status(workflow_id)
        assert workflow_status.workflow_id == workflow_id, "Should track workflow correctly"
        
        # List all workflows
        workflows = await self.orchestrator.list_workflows()
        assert len(workflows) > 0, "Should have active workflows"
        
        logger.info(f"   - End-to-end workflow {workflow_id} executed")
        logger.info(f"   - Workflow state: {workflow_status.state.value}")
        logger.info(f"   - Total active workflows: {len(workflows)}")
    
    async def test_resource_allocation(self) -> None:
        """Test resource allocation across multiple components."""
        resource_manager = self.orchestrator.resource_manager
        
        # Allocate resources for multiple components
        allocations = []
        for i in range(3):
            allocation_id = await resource_manager.allocate_resource(
                ResourceType.CPU,
                1.0,  # 1 CPU core
                f"component_{i}"
            )
            allocations.append(allocation_id)
        
        # Check usage
        usage = await resource_manager.get_usage_stats()
        cpu_usage = usage.get("cpu", {})
        assert cpu_usage.get("currently_allocated", 0) >= 3.0, "Should allocate 3 CPU cores"
        
        # Release all allocations
        for allocation_id in allocations:
            await resource_manager.release_resource(allocation_id)
        
        logger.info(f"   - Successfully allocated and released {len(allocations)} resources")
        logger.info(f"   - Peak CPU usage: {cpu_usage.get('usage_percentage', 0):.1f}%")
    
    async def test_component_health_monitoring(self) -> None:
        """Test component health monitoring."""
        registry = self.orchestrator.component_registry
        
        # Get all components
        components = await registry.list_components()
        
        # Check health of each component
        healthy_count = 0
        for component in components:
            status = await registry.check_component_health(component.name)
            if status.value in ["healthy", "starting"]:
                healthy_count += 1
        
        # Get health summary
        health_summary = await registry.get_health_summary()
        
        assert healthy_count > 0, "Should have some healthy components"
        assert health_summary["total_components"] == len(components), "Summary should match component count"
        
        logger.info(f"   - Health monitoring: {healthy_count}/{len(components)} components healthy")
        logger.info(f"   - Overall health: {health_summary['overall_health_percentage']:.1f}%")
    
    async def cleanup(self) -> None:
        """Clean up after tests."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
            logger.info("   - Orchestrator shutdown complete")


async def main():
    """Main test execution."""
    runner = Phase2TestRunner()
    
    try:
        await runner.run_all_tests()
    finally:
        await runner.cleanup()
    
    # Check if all tests passed
    failed_tests = [name for name, result in runner.test_results.items() if not result.startswith("PASSED")]
    
    if failed_tests:
        logger.error(f"❌ Tests failed: {failed_tests}")
        sys.exit(1)
    else:
        logger.info("🎉 All Phase 2 tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())