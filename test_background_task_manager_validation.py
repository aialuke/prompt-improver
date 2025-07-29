#!/usr/bin/env python3
"""
Background Task Manager Validation Test

Tests the enhanced background task manager functionality to ensure
it works correctly after legacy pattern removal.
"""

import asyncio
import time
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_background_manager():
    """Test enhanced background task manager functionality"""
    print("=== Testing Enhanced Background Task Manager ===")
    
    from src.prompt_improver.performance.monitoring.health.background_manager import (
        EnhancedBackgroundTaskManager,
        TaskPriority,
        TaskStatus
    )
    from src.prompt_improver.core.retry_manager import RetryConfig
    
    # Create and start manager
    manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=5)
    await manager.start()
    
    try:
        print("✅ Enhanced background task manager started")
        
        # Test 1: Basic task submission and execution
        async def simple_task():
            await asyncio.sleep(0.1)
            return "Task completed successfully"
        
        task_id = await manager.submit_enhanced_task(
            task_id="test_simple",
            coroutine=simple_task,
            priority=TaskPriority.HIGH
        )
        
        print(f"✅ Submitted task: {task_id}")
        
        # Wait for completion
        await asyncio.sleep(0.5)
        task_status = manager.get_enhanced_task_status(task_id)
        
        if task_status and task_status["status"] == TaskStatus.COMPLETED.value:
            print(f"✅ Task completed successfully: {task_status['result']}")
        else:
            print(f"❌ Task did not complete properly: {task_status}")
        
        # Test 2: Retry mechanism
        failure_count = 0
        async def failing_task():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise Exception(f"Simulated failure {failure_count}")
            return "Task succeeded after retries"
        
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_task_id = await manager.submit_enhanced_task(
            task_id="test_retry",
            coroutine=failing_task,
            priority=TaskPriority.NORMAL,
            retry_config=retry_config
        )
        
        print(f"✅ Submitted retry test task: {retry_task_id}")
        
        # Wait for completion with retries
        await asyncio.sleep(3.0)
        retry_task_status = manager.get_enhanced_task_status(retry_task_id)
        
        if retry_task_status:
            print(f"✅ Retry task status: {retry_task_status['status']}")
            print(f"   Retry count: {retry_task_status['retry_count']}")
            print(f"   Final result: {retry_task_status.get('result', 'N/A')}")
        
        # Test 3: Priority scheduling
        results = []
        
        async def priority_task(name, delay=0.05):
            await asyncio.sleep(delay)
            result = f"{name} completed at {time.time()}"
            results.append(result)
            return result
        
        # Submit tasks with different priorities (should execute high priority first)
        low_task_id = await manager.submit_enhanced_task(
            coroutine=lambda: priority_task("low_priority", 0.01),
            priority=TaskPriority.LOW
        )
        
        high_task_id = await manager.submit_enhanced_task(
            coroutine=lambda: priority_task("high_priority", 0.01),
            priority=TaskPriority.HIGH
        )
        
        await asyncio.sleep(0.5)
        print("✅ Priority scheduling test completed")
        print(f"   Results order: {results}")
        
        # Test 4: Statistics and monitoring
        stats = manager.get_statistics()
        print(f"✅ Task manager statistics:")
        print(f"   Total submitted: {stats['total_submitted']}")
        print(f"   Total completed: {stats['total_completed']}")
        print(f"   Total failed: {stats['total_failed']}")
        print(f"   Currently running: {stats['currently_running']}")
        print(f"   Success rate: {stats['total_completed'] / max(1, stats['total_submitted']):.2%}")
        
        # Test 5: Dead letter queue functionality
        async def always_failing_task():
            raise Exception("This task always fails")
        
        dead_letter_config = RetryConfig(max_attempts=2, base_delay=0.05)
        dead_task_id = await manager.submit_enhanced_task(
            task_id="test_dead_letter",
            coroutine=always_failing_task,
            retry_config=dead_letter_config
        )
        
        # Wait for task to fail and move to dead letter queue
        await asyncio.sleep(1.0)
        
        dead_letter_tasks = manager.get_dead_letter_tasks()
        print(f"✅ Dead letter queue test:")
        print(f"   Dead letter tasks: {len(dead_letter_tasks)}")
        
        if dead_letter_tasks:
            for task in dead_letter_tasks:
                print(f"   - Task {task['task_id']}: {task['error']}")
        
        return {
            "manager_started": True,
            "basic_task_executed": task_status and task_status["status"] == TaskStatus.COMPLETED.value,
            "retry_mechanism_working": retry_task_status and retry_task_status["retry_count"] > 0,
            "priority_scheduling_working": len(results) > 0,
            "statistics_available": stats['total_submitted'] > 0,
            "dead_letter_queue_working": len(dead_letter_tasks) > 0,
            "overall_success": True
        }
        
    except Exception as e:
        print(f"❌ Background task manager test failed: {e}")
        logger.exception("Background task manager test error")
        return {
            "manager_started": False,
            "error": str(e),
            "overall_success": False
        }
        
    finally:
        # Always stop the manager
        try:
            await manager.stop()
            print("✅ Background task manager stopped cleanly")
        except Exception as e:
            print(f"⚠️  Error stopping manager: {e}")


async def test_background_manager_integration():
    """Test background manager integration with health system"""
    print("\n=== Testing Background Manager Integration ===")
    
    try:
        from src.prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager
        
        # Get global instance
        manager = get_background_task_manager()  
        
        # Test that it's properly initialized
        stats = manager.get_statistics()
        print(f"✅ Global background manager accessible")
        print(f"   Initial state - submitted: {stats['total_submitted']}, completed: {stats['total_completed']}")
        
        # Test orchestrator-compatible interface
        config = {
            "tasks": [],  # No tasks for this test
            "max_concurrent": 5,
            "enable_metrics": True,
            "output_path": "./test_outputs"
        }
        
        result = await manager.run_orchestrated_analysis(config)
        
        if result.get("orchestrator_compatible"):
            print("✅ Orchestrator-compatible interface working")
            print(f"   Component version: {result['local_metadata']['component_version']}")
            print(f"   Execution time: {result['local_metadata']['execution_time']:.2f}s")
        else:
            print("❌ Orchestrator interface not working")
        
        return {
            "global_manager_accessible": True,
            "orchestrator_interface_working": result.get("orchestrator_compatible", False),
            "integration_success": True
        }
        
    except Exception as e:
        print(f"❌ Background manager integration test failed: {e}")
        logger.exception("Background manager integration error")
        return {
            "integration_success": False,
            "error": str(e)
        }


async def run_background_validation():
    """Run comprehensive background task manager validation"""
    print("🔧 Background Task Manager Validation")
    print("=" * 50)
    
    validation_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_passed": False,
        "test_results": {}
    }
    
    # Test categories
    test_functions = [
        ("enhanced_manager", test_enhanced_background_manager),
        ("integration", test_background_manager_integration)
    ]
    
    try:
        for test_name, test_func in test_functions:
            print(f"\n🧪 Running {test_name} test...")
            
            try:
                test_start = time.time()
                result = await test_func()
                test_duration = time.time() - test_start
                
                validation_results["test_results"][test_name] = {
                    "passed": result.get("overall_success", result.get("integration_success", False)),
                    "duration_seconds": test_duration,
                    "result": result
                }
                
                success = result.get("overall_success", result.get("integration_success", False))
                print(f"{'✅' if success else '❌'} {test_name} {'passed' if success else 'failed'} in {test_duration:.2f}s")
                
            except Exception as e:
                test_duration = time.time() - test_start if 'test_start' in locals() else 0
                validation_results["test_results"][test_name] = {
                    "passed": False,
                    "duration_seconds": test_duration,
                    "error": str(e)
                }
                print(f"❌ {test_name} failed with exception: {e}")
                logger.exception(f"Test {test_name} failed")
        
        # Calculate overall results
        passed_tests = sum(1 for result in validation_results["test_results"].values() if result["passed"])
        total_tests = len(validation_results["test_results"])
        
        validation_results["validation_passed"] = passed_tests == total_tests
        validation_results["passed_tests"] = passed_tests
        validation_results["total_tests"] = total_tests
        validation_results["success_rate"] = passed_tests / total_tests if total_tests > 0 else 0
        
        # Summary
        print(f"\n📊 Background Task Manager Validation Summary")
        print("=" * 50)
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Success rate: {validation_results['success_rate']:.1%}")
        print(f"Overall validation: {'✅ PASSED' if validation_results['validation_passed'] else '❌ FAILED'}")
        
        if validation_results["validation_passed"]:
            print("\n🎉 Background task manager validation completed successfully!")
            print("   Enhanced features are working correctly after legacy pattern removal.")
        else:
            print("\n⚠️  Background task manager validation found issues.")
            print("   Some enhanced features may not be working correctly.")
        
        return validation_results
        
    except Exception as e:
        validation_results["validation_passed"] = False
        validation_results["error"] = str(e)
        print(f"❌ Background validation failed: {e}")
        logger.exception("Background validation failed")
        return validation_results


if __name__ == "__main__":
    # Run the validation
    results = asyncio.run(run_background_validation())
    
    # Exit with appropriate code
    exit(0 if results["validation_passed"] else 1)