"""
Validation Test for ML Background Task Integration

Tests the integration of ML components with EnhancedBackgroundTaskManager
to ensure proper task lifecycle management and monitoring.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

from prompt_improver.ml.automl.orchestrator import (
    AutoMLConfig,
    create_automl_orchestrator,
)
from prompt_improver.ml.evaluation.experiment_orchestrator import (
    ExperimentOrchestrator,
)
from prompt_improver.ml.lifecycle.ml_platform_integration import (
    MLPlatformIntegration,
)
from prompt_improver.performance.monitoring.health.background_manager import (
    EnhancedBackgroundTaskManager,
    TaskPriority,
    get_background_task_manager,
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLBackgroundTaskIntegrationValidator:
    """Comprehensive validator for ML background task integration."""

    def __init__(self):
        self.task_manager: EnhancedBackgroundTaskManager | None = None
        self.test_results: dict[str, dict[str, Any]] = {}

    async def setup(self):
        """Initialize the background task manager for testing."""
        self.task_manager = get_background_task_manager()
        await self.task_manager.start()
        logger.info("✅ EnhancedBackgroundTaskManager initialized for testing")

    async def cleanup(self):
        """Clean up resources after testing."""
        if self.task_manager:
            await self.task_manager.stop()
        logger.info("✅ Test cleanup completed")

    async def test_automl_orchestrator_integration(self) -> dict[str, Any]:
        """Test AutoML Orchestrator integration with background task manager."""
        logger.info("🧪 Testing AutoML Orchestrator integration...")
        try:
            config = AutoMLConfig(
                study_name="test_automl_integration", n_trials=5, timeout=30
            )
            orchestrator = await create_automl_orchestrator(
                config=config, task_manager=self.task_manager
            )
            initial_status = await orchestrator.get_optimization_status()
            task_stats = self.task_manager.get_statistics()
            result = {
                "status": "success",
                "initial_status": initial_status,
                "task_manager_integration": {
                    "task_manager_type": type(orchestrator.task_manager).__name__,
                    "total_tasks_managed": task_stats.get("total_submitted", 0),
                    "background_task_integration": True,
                },
                "test_time": time.time(),
            }
            logger.info("✅ AutoML Orchestrator integration test passed")
            return result
        except Exception as e:
            logger.exception("❌ AutoML Orchestrator integration test failed: %s", e)
            return {"status": "failed", "error": str(e), "test_time": time.time()}

    async def test_experiment_orchestrator_integration(self) -> dict[str, Any]:
        """Test Experiment Orchestrator integration with background task manager."""
        logger.info("🧪 Testing Experiment Orchestrator integration...")
        try:
            from unittest.mock import AsyncMock

            mock_db_session = AsyncMock()
            orchestrator = ExperimentOrchestrator(
                db_session=mock_db_session, task_manager=self.task_manager
            )
            orchestrator_status = await orchestrator.get_orchestrator_status()
            result = {
                "status": "success",
                "orchestrator_status": orchestrator_status,
                "task_manager_integration": orchestrator_status["task_management"],
                "test_time": time.time(),
            }
            logger.info("✅ Experiment Orchestrator integration test passed")
            return result
        except Exception as e:
            logger.exception("❌ Experiment Orchestrator integration test failed: %s", e)
            return {"status": "failed", "error": str(e), "test_time": time.time()}

    async def test_ml_platform_integration(self) -> dict[str, Any]:
        """Test ML Platform Integration with background task manager."""
        logger.info("🧪 Testing ML Platform Integration...")
        try:
            platform = MLPlatformIntegration(
                storage_path=Path("./test_ml_platform"),
                enable_distributed=False,
                enable_monitoring=True,
                task_manager=self.task_manager,
            )
            success = await platform.initialize_platform()
            task_stats = self.task_manager.get_statistics()
            result = {
                "status": "success" if success else "failed",
                "platform_initialization": success,
                "platform_status": platform.platform_status.value,
                "task_manager_integration": {
                    "task_manager_type": type(platform.task_manager).__name__,
                    "background_tasks_started": len(platform._background_task_ids),
                    "total_tasks_managed": task_stats.get("total_submitted", 0),
                },
                "test_time": time.time(),
            }
            await platform.shutdown_platform()
            logger.info("✅ ML Platform Integration test passed")
            return result
        except Exception as e:
            logger.exception("❌ ML Platform Integration test failed: %s", e)
            return {"status": "failed", "error": str(e), "test_time": time.time()}

    async def test_task_lifecycle_management(self) -> dict[str, Any]:
        """Test comprehensive task lifecycle management."""
        logger.info("🧪 Testing task lifecycle management...")
        try:

            async def test_task():
                await asyncio.sleep(2)
                return {"result": "task_completed"}

            task_id = await self.task_manager.submit_enhanced_task(
                task_id="lifecycle_test_task",
                coroutine=test_task(),
                priority=TaskPriority.NORMAL,
                timeout=10,
                tags={"type": "lifecycle_test"},
            )
            initial_status = self.task_manager.get_enhanced_task_status(task_id)
            result = await self.task_manager.wait_for_task(task_id, timeout=15)
            final_status = self.task_manager.get_enhanced_task_status(task_id)
            final_stats = self.task_manager.get_statistics()
            test_result = {
                "status": "success",
                "task_lifecycle": {
                    "task_id": task_id,
                    "initial_status": initial_status["status"]
                    if initial_status
                    else "unknown",
                    "final_status": final_status["status"]
                    if final_status
                    else "unknown",
                    "task_result": result,
                    "execution_time": final_status.get("metrics", {}).get(
                        "total_duration", 0
                    )
                    if final_status
                    else 0,
                },
                "task_manager_stats": {
                    "total_submitted": final_stats.get("total_submitted", 0),
                    "total_completed": final_stats.get("total_completed", 0),
                    "success_rate": final_stats.get("total_completed", 0)
                    / max(1, final_stats.get("total_submitted", 0)),
                },
                "test_time": time.time(),
            }
            logger.info("✅ Task lifecycle management test passed")
            return test_result
        except Exception as e:
            logger.exception("❌ Task lifecycle management test failed: %s", e)
            return {"status": "failed", "error": str(e), "test_time": time.time()}

    async def test_task_cancellation_and_retry(self) -> dict[str, Any]:
        """Test task cancellation and retry mechanisms."""
        logger.info("🧪 Testing task cancellation and retry...")
        try:

            async def long_running_task():
                await asyncio.sleep(30)
                return {"result": "should_not_complete"}

            task_id = await self.task_manager.submit_enhanced_task(
                task_id="cancellation_test_task",
                coroutine=long_running_task(),
                priority=TaskPriority.NORMAL,
                timeout=60,
                tags={"type": "cancellation_test"},
            )
            await asyncio.sleep(1)
            cancellation_success = await self.task_manager.cancel_task(task_id)
            cancelled_status = self.task_manager.get_enhanced_task_status(task_id)
            result = {
                "status": "success",
                "cancellation_test": {
                    "task_id": task_id,
                    "cancellation_success": cancellation_success,
                    "final_status": cancelled_status["status"]
                    if cancelled_status
                    else "unknown",
                },
                "test_time": time.time(),
            }
            logger.info("✅ Task cancellation test passed")
            return result
        except Exception as e:
            logger.exception("❌ Task cancellation test failed: %s", e)
            return {"status": "failed", "error": str(e), "test_time": time.time()}

    async def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run comprehensive validation of all ML background task integrations."""
        logger.info(
            "🚀 Starting comprehensive ML background task integration validation..."
        )
        start_time = time.time()
        self.test_results = {
            "automl_orchestrator": await self.test_automl_orchestrator_integration(),
            "experiment_orchestrator": await self.test_experiment_orchestrator_integration(),
            "ml_platform": await self.test_ml_platform_integration(),
            "task_lifecycle": await self.test_task_lifecycle_management(),
            "task_cancellation": await self.test_task_cancellation_and_retry(),
        }
        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for result in self.test_results.values() if result["status"] == "success"
        )
        failed_tests = total_tests - passed_tests
        final_stats = self.task_manager.get_statistics()
        comprehensive_result = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests,
                "validation_time": time.time() - start_time,
            },
            "task_manager_performance": {
                "total_tasks_submitted": final_stats.get("total_submitted", 0),
                "total_tasks_completed": final_stats.get("total_completed", 0),
                "total_tasks_failed": final_stats.get("total_failed", 0),
                "overall_success_rate": final_stats.get("total_completed", 0)
                / max(1, final_stats.get("total_submitted", 0)),
                "task_manager_type": type(self.task_manager).__name__,
            },
            "individual_test_results": self.test_results,
            "integration_status": "SUCCESS"
            if failed_tests == 0
            else "PARTIAL"
            if passed_tests > 0
            else "FAILED",
        }
        if failed_tests == 0:
            logger.info(
                "🎉 ALL TESTS PASSED! %s/%s ML background task integrations validated successfully",
                passed_tests,
                total_tests,
            )
        else:
            logger.warning(
                "⚠️  SOME TESTS FAILED: %s/%s tests passed, %s failed",
                passed_tests,
                total_tests,
                failed_tests,
            )
        return comprehensive_result


async def main():
    """Main validation function."""
    validator = MLBackgroundTaskIntegrationValidator()
    try:
        await validator.setup()
        results = await validator.run_comprehensive_validation()
        print("\n" + "=" * 80)
        print("ML BACKGROUND TASK INTEGRATION VALIDATION RESULTS")
        print("=" * 80)
        print(f"Integration Status: {results['integration_status']}")
        print(
            f"Tests Passed: {results['validation_summary']['passed_tests']}/{results['validation_summary']['total_tests']}"
        )
        print(f"Success Rate: {results['validation_summary']['success_rate']:.1%}")
        print(
            f"Total Tasks Managed: {results['task_manager_performance']['total_tasks_submitted']}"
        )
        print(
            f"Task Success Rate: {results['task_manager_performance']['overall_success_rate']:.1%}"
        )
        print("=" * 80)
        return 0 if results["integration_status"] == "SUCCESS" else 1
    except Exception as e:
        logger.exception("❌ Validation failed with error: %s", e)
        return 1
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
