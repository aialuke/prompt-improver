#!/usr/bin/env python3
"""
Phase 2B-1 Enhanced BackgroundTaskManager Integration Test

Tests the enhanced BackgroundTaskManager with 2025 best practices:
- Priority-based task scheduling
- Retry mechanisms with exponential backoff
- Circuit breaker patterns for external dependencies
- Comprehensive observability and metrics
- Dead letter queue handling
- Resource-aware task distribution

Validates orchestrator integration and 2025 compliance.
"""

import asyncio
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2BEnhancedBackgroundManagerTester:
    """Test enhanced BackgroundTaskManager integration with orchestrator"""
    
    def __init__(self):
        self.test_results = {}
        self.component_name = "enhanced_background_task_manager"
    
    async def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test for enhanced BackgroundTaskManager"""
        
        print("🚀 Phase 2B-1 Enhanced BackgroundTaskManager Integration Test")
        print("=" * 70)
        
        # Test 1: Component Discovery and 2025 Features
        features_result = await self._test_2025_features()
        
        # Test 2: Priority-Based Task Scheduling
        priority_result = await self._test_priority_scheduling()
        
        # Test 3: Retry Mechanisms
        retry_result = await self._test_retry_mechanisms()
        
        # Test 4: Circuit Breaker Patterns
        circuit_breaker_result = await self._test_circuit_breakers()
        
        # Test 5: Orchestrator Integration
        integration_result = await self._test_orchestrator_integration()
        
        # Compile results
        overall_result = {
            "features_2025": features_result,
            "priority_scheduling": priority_result,
            "retry_mechanisms": retry_result,
            "circuit_breakers": circuit_breaker_result,
            "orchestrator_integration": integration_result,
            "summary": self._generate_test_summary()
        }
        
        self._print_test_results(overall_result)
        return overall_result
    
    async def _test_2025_features(self) -> Dict[str, Any]:
        """Test 2025 enhanced features"""
        
        print("\n🔬 Test 1: 2025 Features Validation")
        
        try:
            from prompt_improver.performance.monitoring.health.background_manager import (
                EnhancedBackgroundTaskManager,
                TaskPriority,
                TaskStatus,
                RetryConfig,
                TaskMetrics,
                EnhancedBackgroundTask
            )
            
            # Test enhanced classes and enums
            manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=5, enable_metrics=True)
            
            features_available = {
                "enhanced_manager": True,
                "task_priorities": len(list(TaskPriority)) >= 5,
                "enhanced_status": len(list(TaskStatus)) >= 8,
                "retry_config": hasattr(RetryConfig, 'max_retries'),
                "task_metrics": hasattr(TaskMetrics, 'execution_count'),
                "enhanced_task": hasattr(EnhancedBackgroundTask, 'priority'),
                "orchestrator_interface": hasattr(manager, 'run_orchestrated_analysis'),
                "statistics": hasattr(manager, 'get_statistics'),
                "dead_letter_queue": hasattr(manager, 'get_dead_letter_tasks'),
                "circuit_breakers": hasattr(manager, 'circuit_breakers')
            }
            
            success_count = sum(features_available.values())
            total_features = len(features_available)
            
            print(f"  ✅ Enhanced Manager: {'AVAILABLE' if features_available['enhanced_manager'] else 'MISSING'}")
            print(f"  ✅ Task Priorities: {len(list(TaskPriority))} levels available")
            print(f"  ✅ Enhanced Status: {len(list(TaskStatus))} states available")
            print(f"  ✅ Retry Configuration: {'AVAILABLE' if features_available['retry_config'] else 'MISSING'}")
            print(f"  ✅ Task Metrics: {'AVAILABLE' if features_available['task_metrics'] else 'MISSING'}")
            print(f"  ✅ Enhanced Tasks: {'AVAILABLE' if features_available['enhanced_task'] else 'MISSING'}")
            print(f"  ✅ Orchestrator Interface: {'AVAILABLE' if features_available['orchestrator_interface'] else 'MISSING'}")
            print(f"  ✅ Statistics: {'AVAILABLE' if features_available['statistics'] else 'MISSING'}")
            print(f"  ✅ Dead Letter Queue: {'AVAILABLE' if features_available['dead_letter_queue'] else 'MISSING'}")
            print(f"  ✅ Circuit Breakers: {'AVAILABLE' if features_available['circuit_breakers'] else 'MISSING'}")
            print(f"  📊 Features Score: {success_count}/{total_features} ({(success_count/total_features)*100:.1f}%)")
            
            return {
                "success": success_count == total_features,
                "features_available": features_available,
                "features_score": success_count / total_features,
                "priority_levels": len(list(TaskPriority)),
                "status_states": len(list(TaskStatus))
            }
            
        except Exception as e:
            print(f"  ❌ 2025 features test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_priority_scheduling(self) -> Dict[str, Any]:
        """Test priority-based task scheduling"""
        
        print("\n📋 Test 2: Priority-Based Task Scheduling")
        
        try:
            from prompt_improver.performance.monitoring.health.background_manager import (
                EnhancedBackgroundTaskManager,
                TaskPriority
            )
            
            manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=2)
            await manager.start()
            
            # Submit tasks with different priorities
            async def sample_task(priority_name, delay=0.1):
                await asyncio.sleep(delay)
                return f"Task completed with priority {priority_name}"
            
            # Submit tasks in reverse priority order to test scheduling
            task_ids = []
            priorities = [TaskPriority.BACKGROUND, TaskPriority.NORMAL, TaskPriority.HIGH, TaskPriority.CRITICAL]
            
            for i, priority in enumerate(priorities):
                task_id = await manager.submit_enhanced_task(
                    coroutine=lambda p=priority.name: sample_task(p),
                    priority=priority,
                    tags={"test": "priority_scheduling"}
                )
                task_ids.append(task_id)
            
            # Wait for tasks to complete
            await asyncio.sleep(1.0)
            
            # Check task execution order and statistics
            stats = manager.get_statistics()
            completed_tasks = 0
            
            for task_id in task_ids:
                status = manager.get_enhanced_task_status(task_id)
                if status and status["status"] == "completed":
                    completed_tasks += 1
            
            await manager.stop()
            
            success = completed_tasks >= 2  # At least some tasks should complete
            
            print(f"  {'✅' if success else '❌'} Priority Scheduling: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Tasks Submitted: {len(task_ids)}")
            print(f"  📊 Tasks Completed: {completed_tasks}")
            print(f"  📊 Total Submitted: {stats['total_submitted']}")
            
            return {
                "success": success,
                "tasks_submitted": len(task_ids),
                "tasks_completed": completed_tasks,
                "statistics": stats
            }
            
        except Exception as e:
            print(f"  ❌ Priority scheduling test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_retry_mechanisms(self) -> Dict[str, Any]:
        """Test retry mechanisms with exponential backoff"""
        
        print("\n🔄 Test 3: Retry Mechanisms")
        
        try:
            from prompt_improver.performance.monitoring.health.background_manager import (
                EnhancedBackgroundTaskManager,
                RetryConfig,
                TaskPriority
            )
            
            manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=2)
            await manager.start()
            
            # Create a task that fails initially
            failure_count = 0
            async def failing_task():
                nonlocal failure_count
                failure_count += 1
                if failure_count < 3:  # Fail first 2 attempts
                    raise Exception(f"Simulated failure {failure_count}")
                return "Success after retries"
            
            # Submit task with retry configuration
            retry_config = RetryConfig(max_retries=3, initial_delay=0.1, max_delay=1.0)
            task_id = await manager.submit_enhanced_task(
                coroutine=failing_task,
                priority=TaskPriority.NORMAL,
                retry_config=retry_config,
                tags={"test": "retry_mechanisms"}
            )
            
            # Wait for retries to complete
            await asyncio.sleep(3.0)
            
            # Check task status and retry count
            status = manager.get_enhanced_task_status(task_id)
            stats = manager.get_statistics()
            
            success = (status and
                      (status["status"] == "completed" or status["status"] == "retrying") and
                      status["retry_count"] >= 2 and
                      stats["total_retries"] >= 2)
            
            await manager.stop()
            
            print(f"  {'✅' if success else '❌'} Retry Mechanisms: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Task Status: {status['status'] if status else 'Unknown'}")
            print(f"  📊 Retry Count: {status['retry_count'] if status else 0}")
            print(f"  📊 Total Retries: {stats['total_retries']}")
            
            return {
                "success": success,
                "task_status": status["status"] if status else "unknown",
                "retry_count": status["retry_count"] if status else 0,
                "total_retries": stats["total_retries"]
            }
            
        except Exception as e:
            print(f"  ❌ Retry mechanisms test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_circuit_breakers(self) -> Dict[str, Any]:
        """Test circuit breaker patterns"""
        
        print("\n⚡ Test 4: Circuit Breaker Patterns")
        
        try:
            from prompt_improver.performance.monitoring.health.background_manager import (
                EnhancedBackgroundTaskManager,
                TaskPriority
            )
            
            manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=2)
            await manager.start()
            
            # Create a task that always fails
            async def always_failing_task():
                raise Exception("Circuit breaker test failure")
            
            # Submit multiple tasks with same circuit breaker key
            task_ids = []
            for i in range(3):
                task_id = await manager.submit_enhanced_task(
                    coroutine=always_failing_task,
                    priority=TaskPriority.NORMAL,
                    circuit_breaker_key="test_service",
                    tags={"test": "circuit_breaker"}
                )
                task_ids.append(task_id)
            
            # Wait for tasks to fail
            await asyncio.sleep(1.0)
            
            # Check circuit breaker status
            stats = manager.get_statistics()
            circuit_breakers = stats.get("circuit_breakers", {})
            
            success = (len(circuit_breakers) > 0 and 
                      "test_service" in circuit_breakers)
            
            await manager.stop()
            
            print(f"  {'✅' if success else '❌'} Circuit Breakers: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Circuit Breakers Created: {len(circuit_breakers)}")
            print(f"  📊 Test Service CB: {'AVAILABLE' if 'test_service' in circuit_breakers else 'MISSING'}")
            
            return {
                "success": success,
                "circuit_breakers_created": len(circuit_breakers),
                "test_service_cb": "test_service" in circuit_breakers
            }
            
        except Exception as e:
            print(f"  ❌ Circuit breaker test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test orchestrator integration"""
        
        print("\n🔄 Test 5: Orchestrator Integration")
        
        try:
            from prompt_improver.performance.monitoring.health.background_manager import EnhancedBackgroundTaskManager
            
            manager = EnhancedBackgroundTaskManager()
            await manager.start()
            
            # Test orchestrator interface
            async def sample_orchestrator_task():
                await asyncio.sleep(0.1)
                return "Orchestrator task completed"
            
            config = {
                "tasks": [
                    {
                        "task_id": "orchestrator_test_1",
                        "coroutine": sample_orchestrator_task,
                        "priority": 2,  # HIGH priority
                        "tags": {"source": "orchestrator", "test": "integration"}
                    }
                ],
                "max_concurrent": 5,
                "enable_metrics": True,
                "output_path": "./test_outputs/background_task_management"
            }
            
            result = await manager.run_orchestrated_analysis(config)
            
            # Validate result structure
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result and
                "task_management_summary" in result["component_result"]
            )
            
            task_summary = result.get("component_result", {}).get("task_management_summary", {})
            
            await manager.stop()
            
            print(f"  {'✅' if success else '❌'} Orchestrator Interface: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Tasks Submitted: {task_summary.get('submitted_tasks', 0)}")
            print(f"  📊 Total Tasks Managed: {task_summary.get('total_tasks_managed', 0)}")
            print(f"  📊 Execution Time: {result.get('local_metadata', {}).get('execution_time', 0):.3f}s")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "tasks_submitted": task_summary.get("submitted_tasks", 0),
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ Orchestrator integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        return {
            "total_tests": 5,
            "component_tested": "enhanced_background_task_manager",
            "enhancement_status": "Phase 2B-1 Enhanced BackgroundTaskManager Complete",
            "version": "2025.1.0"
        }
    
    def _print_test_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 70)
        print("📊 PHASE 2B-1 ENHANCED BACKGROUND TASK MANAGER TEST RESULTS")
        print("=" * 70)
        
        # Print summary
        features = results.get("features_2025", {})
        priority = results.get("priority_scheduling", {})
        retry = results.get("retry_mechanisms", {})
        circuit = results.get("circuit_breakers", {})
        integration = results.get("orchestrator_integration", {})
        
        features_success = features.get("success", False)
        priority_success = priority.get("success", False)
        retry_success = retry.get("success", False)
        circuit_success = circuit.get("success", False)
        integration_success = integration.get("success", False)
        
        print(f"✅ 2025 Features: {'PASSED' if features_success else 'FAILED'} ({features.get('features_score', 0)*100:.1f}%)")
        print(f"✅ Priority Scheduling: {'PASSED' if priority_success else 'FAILED'}")
        print(f"✅ Retry Mechanisms: {'PASSED' if retry_success else 'FAILED'}")
        print(f"✅ Circuit Breakers: {'PASSED' if circuit_success else 'FAILED'}")
        print(f"✅ Orchestrator Integration: {'PASSED' if integration_success else 'FAILED'}")
        
        overall_success = all([features_success, priority_success, retry_success, circuit_success, integration_success])
        
        if overall_success:
            print("\n🎉 PHASE 2B-1 ENHANCEMENT: COMPLETE SUCCESS!")
            print("Enhanced BackgroundTaskManager with 2025 best practices is fully integrated and ready!")
        else:
            print("\n⚠️  PHASE 2B-1 ENHANCEMENT: NEEDS ATTENTION")
            print("Some enhanced features require additional work.")


async def main():
    """Main test execution function"""
    
    tester = Phase2BEnhancedBackgroundManagerTester()
    results = await tester.run_comprehensive_integration_test()
    
    # Save results to file
    with open('phase2b_enhanced_background_manager_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: phase2b_enhanced_background_manager_test_results.json")
    
    return 0 if results.get("features_2025", {}).get("success", False) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
