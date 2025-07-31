#!/usr/bin/env python3
"""
Checkpoint 2A: Simplified ML Pipeline Orchestration Validation
=============================================================

Focused validation of Week 3 ML infrastructure consolidation with core functionality tests.
Tests the integrated functionality of consolidated components without complex dependencies.

Success Criteria:
- AdaptiveEventBus operational and processing events
- EnhancedBackgroundTaskManager handling ML tasks  
- UnifiedConnectionManager database operations working
- Integration between components functioning
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Core consolidated components
from src.prompt_improver.ml.orchestration.events.adaptive_event_bus import (
    AdaptiveEventBus
)
from src.prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent
from src.prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from src.prompt_improver.database.unified_connection_manager import (
    UnifiedConnectionManager, ManagerMode
)
from src.prompt_improver.performance.monitoring.health.background_manager import (
    get_background_task_manager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedCheckpoint2AValidator:
    """Simplified ML infrastructure consolidation validator."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    async def run_validation(self) -> Dict[str, Any]:
        """Execute simplified validation tests."""
        logger.info("🚀 Starting Simplified Checkpoint 2A Validation")
        
        results = {
            "validation_name": "Checkpoint_2A_Simplified",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "tests": []
        }
        
        try:
            # Test 1: AdaptiveEventBus Basic Functionality
            event_bus_result = await self.test_event_bus_basic()
            results["tests"].append(event_bus_result)
            
            # Test 2: Background Task Manager Basic Functionality
            task_manager_result = await self.test_background_tasks_basic()
            results["tests"].append(task_manager_result)
            
            # Test 3: Database Manager Basic Functionality
            database_result = await self.test_database_basic()
            results["tests"].append(database_result)
            
            # Test 4: Basic Integration Test
            integration_result = await self.test_basic_integration()
            results["tests"].append(integration_result)
            
            # Overall assessment
            successful_tests = sum(1 for test in results["tests"] if test["success"])
            total_tests = len(results["tests"])
            
            results["summary"] = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "overall_success": successful_tests >= 3,  # At least 3/4 tests must pass
                "execution_time_seconds": time.time() - self.start_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results["error"] = str(e)
            results["success"] = False
            return results
    
    async def test_event_bus_basic(self) -> Dict[str, Any]:
        """Test basic AdaptiveEventBus functionality."""
        logger.info("🔄 Testing AdaptiveEventBus Basic Functionality")
        test_start = time.time()
        
        try:
            # Create event bus with minimal configuration
            config = OrchestratorConfig(
                max_concurrent_workflows=2,
                event_bus_buffer_size=1000
            )
            
            event_bus = AdaptiveEventBus(config=config)
            
            # Test initialization
            await event_bus.start()
            
            # Test event emission and processing
            events_processed = []
            
            async def test_handler(event: MLEvent):
                events_processed.append(event.data.get("test_id"))
            
            # Subscribe to test events
            subscription_id = event_bus.subscribe(
                EventType.TRAINING_STARTED,
                test_handler,
                priority=5
            )
            
            # Emit test events
            for i in range(10):
                event = MLEvent(
                    event_type=EventType.TRAINING_STARTED,
                    source="checkpoint_2a_test",
                    data={"test_id": f"test_{i}"}
                )
                await event_bus.emit(event)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Get metrics
            metrics = event_bus.get_performance_metrics()
            
            # Cleanup
            event_bus.unsubscribe(subscription_id)
            await event_bus.stop()
            
            success = len(events_processed) >= 8  # At least 80% processed
            
            return {
                "test_name": "AdaptiveEventBus_Basic",
                "success": success,
                "execution_time": time.time() - test_start,
                "events_emitted": 10,
                "events_processed": len(events_processed),
                "processing_rate": len(events_processed) / 10,
                "metrics": {
                    "events_processed": metrics.get("events_processed", 0),
                    "worker_count": metrics.get("worker_count", 0),
                    "state": metrics.get("state", "unknown")
                }
            }
            
        except Exception as e:
            return {
                "test_name": "AdaptiveEventBus_Basic",
                "success": False,
                "execution_time": time.time() - test_start,
                "error": str(e)
            }
    
    async def test_background_tasks_basic(self) -> Dict[str, Any]:
        """Test basic background task manager functionality."""
        logger.info("⚙️ Testing Background Task Manager Basic Functionality")
        test_start = time.time()
        
        try:
            task_manager = get_background_task_manager()
            
            # Test task submission and execution
            async def test_task(task_id: str):
                await asyncio.sleep(0.1)  # Simulate work
                return {"task_id": task_id, "result": "completed"}
            
            # Submit multiple tasks
            task_ids = []
            for i in range(5):
                task_id = f"test_task_{i}"
                submitted_id = await task_manager.submit_enhanced_task(
                    task_id=task_id,
                    coroutine=test_task(task_id),
                    priority="MEDIUM",
                    timeout=10
                )
                task_ids.append(submitted_id)
            
            # Wait for completion and collect results
            completed_tasks = []
            for task_id in task_ids:
                try:
                    result = await task_manager.get_task_result(task_id, timeout=15)
                    if result:
                        completed_tasks.append(result)
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task_id} timed out")
            
            # Get status
            status = await task_manager.get_comprehensive_status()
            
            success = len(completed_tasks) >= 4  # At least 80% completed
            
            return {
                "test_name": "BackgroundTaskManager_Basic",
                "success": success,
                "execution_time": time.time() - test_start,
                "tasks_submitted": len(task_ids),
                "tasks_completed": len(completed_tasks),
                "completion_rate": len(completed_tasks) / len(task_ids) if task_ids else 0,
                "task_manager_status": {
                    "active_tasks": status.get("active_tasks", 0),
                    "total_tasks": status.get("metrics", {}).get("tasks_completed", 0)
                }
            }
            
        except Exception as e:
            return {
                "test_name": "BackgroundTaskManager_Basic",
                "success": False,
                "execution_time": time.time() - test_start,
                "error": str(e)
            }
    
    async def test_database_basic(self) -> Dict[str, Any]:
        """Test basic database manager functionality."""
        logger.info("💾 Testing Database Manager Basic Functionality")
        test_start = time.time()
        
        try:
            # Initialize database manager for ML training mode
            db_manager = UnifiedConnectionManager(mode=ManagerMode.ML_TRAINING)
            await db_manager.initialize()
            
            # Test basic database operations
            query_results = []
            
            for i in range(5):
                try:
                    async with db_manager.get_async_session() as session:
                        from sqlalchemy import text
                        result = await session.execute(text("SELECT 1 as test_value"))
                        value = result.scalar()
                        query_results.append(value)
                except Exception as e:
                    logger.warning(f"Query {i} failed: {e}")
            
            # Test health check
            health_check = await db_manager.health_check()
            
            # Get performance metrics
            metrics = await db_manager.get_performance_metrics()
            
            # Cleanup
            await db_manager.close()
            
            success = (
                len(query_results) >= 4 and  # At least 80% queries successful
                health_check.get("status") == "healthy"
            )
            
            return {
                "test_name": "DatabaseManager_Basic",
                "success": success,
                "execution_time": time.time() - test_start,
                "queries_attempted": 5,
                "queries_successful": len(query_results),
                "success_rate": len(query_results) / 5,
                "health_status": health_check.get("status", "unknown"),
                "metrics": {
                    "avg_response_time_ms": metrics.get("avg_response_time_ms", 0),
                    "pool_utilization": metrics.get("pool_utilization", 0),
                    "total_connections": metrics.get("total_connections", 0)
                }
            }
            
        except Exception as e:
            return {
                "test_name": "DatabaseManager_Basic",
                "success": False,
                "execution_time": time.time() - test_start,
                "error": str(e)
            }
    
    async def test_basic_integration(self) -> Dict[str, Any]:
        """Test basic integration between components."""
        logger.info("🔗 Testing Basic Component Integration")
        test_start = time.time()
        
        try:
            # Initialize components
            config = OrchestratorConfig(max_concurrent_workflows=1)
            event_bus = AdaptiveEventBus(config=config)
            task_manager = get_background_task_manager()
            db_manager = UnifiedConnectionManager(mode=ManagerMode.ML_TRAINING)
            
            await event_bus.start()
            await db_manager.initialize()
            
            # Integration test: Task that uses database and emits events
            integration_steps = []
            
            async def integration_task():
                try:
                    # Step 1: Database operation
                    async with db_manager.get_async_session() as session:
                        from sqlalchemy import text
                        await session.execute(text("SELECT 1"))
                    integration_steps.append("database_query")
                    
                    # Step 2: Event emission
                    event = MLEvent(
                        event_type=EventType.TRAINING_PROGRESS,
                        source="integration_test",
                        data={"step": "integration_complete"}
                    )
                    await event_bus.emit(event)
                    integration_steps.append("event_emitted")
                    
                    return {"status": "completed", "steps": len(integration_steps)}
                    
                except Exception as e:
                    integration_steps.append(f"error: {str(e)}")
                    return {"status": "failed", "error": str(e)}
            
            # Execute integration task
            task_id = await task_manager.submit_enhanced_task(
                task_id="integration_test",
                coroutine=integration_task(),
                priority="HIGH",
                timeout=10
            )
            
            result = await task_manager.get_task_result(task_id, timeout=15)
            
            # Cleanup
            await event_bus.stop()
            await db_manager.close()
            
            success = (
                result is not None and
                result.get("status") == "completed" and
                len(integration_steps) >= 2
            )
            
            return {
                "test_name": "BasicIntegration",
                "success": success,
                "execution_time": time.time() - test_start,
                "integration_steps": integration_steps,
                "steps_completed": len(integration_steps),
                "task_result": result
            }
            
        except Exception as e:
            return {
                "test_name": "BasicIntegration",
                "success": False,
                "execution_time": time.time() - test_start,
                "error": str(e)
            }

async def main():
    """Execute simplified Checkpoint 2A validation."""
    validator = SimplifiedCheckpoint2AValidator()
    
    try:
        results = await validator.run_validation()
        
        # Save results
        results_file = Path("checkpoint_2a_simplified_results.json")
        with results_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print results
        print("\n" + "="*70)
        print("🎯 CHECKPOINT 2A: SIMPLIFIED ML ORCHESTRATION VALIDATION")
        print("="*70)
        
        summary = results.get("summary", {})
        if summary.get("overall_success", False):
            print("✅ VALIDATION PASSED - Core consolidated components functional")
        else:
            print("❌ VALIDATION FAILED - Issues found in core components")
        
        print(f"\n📊 Test Results:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Successful: {summary.get('successful_tests', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"  Execution Time: {summary.get('execution_time_seconds', 0):.2f}s")
        
        print(f"\n📋 Individual Test Results:")
        for test in results.get("tests", []):
            status = "✅" if test["success"] else "❌"
            print(f"  {status} {test['test_name']}: {test.get('execution_time', 0):.2f}s")
            if not test["success"] and "error" in test:
                print(f"      Error: {test['error']}")
        
        print(f"\n📁 Results saved to: {results_file.absolute()}")
        print("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Validation execution failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    asyncio.run(main())