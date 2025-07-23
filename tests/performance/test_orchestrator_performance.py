"""
Performance tests for ML Pipeline Orchestrator.
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator, PipelineState
)
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig


class TestOrchestratorPerformance:
    """Performance tests for ML Pipeline Orchestrator."""
    
    @pytest.fixture
    async def high_performance_orchestrator(self):
        """Create orchestrator optimized for performance testing."""
        config = OrchestratorConfig(
            max_concurrent_workflows=20,
            component_health_check_interval=5,
            pipeline_status_update_interval=2,
            event_bus_buffer_size=1000,
            event_handler_timeout=10
        )
        
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        
        yield orchestrator
        
        await orchestrator.shutdown()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_workflow_startup_performance(self, high_performance_orchestrator):
        """Test workflow startup performance."""
        orchestrator = high_performance_orchestrator
        
        # Test single workflow startup time
        start_time = time.perf_counter()
        workflow_id = await orchestrator.start_workflow("tier1_training", {"performance_test": True})
        end_time = time.perf_counter()
        
        startup_time = end_time - start_time
        
        # Workflow should start within 100ms
        assert startup_time < 0.1, f"Workflow startup took {startup_time:.3f}s, expected < 0.1s"
        
        # Verify workflow started successfully
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.RUNNING
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_workflow_startup_performance(self, high_performance_orchestrator):
        """Test concurrent workflow startup performance."""
        orchestrator = high_performance_orchestrator
        
        # Test concurrent startup of multiple workflows
        num_workflows = 10
        
        start_time = time.perf_counter()
        
        # Start workflows concurrently
        tasks = []
        for i in range(num_workflows):
            task = orchestrator.start_workflow("tier1_training", {"workflow_index": i})
            tasks.append(task)
        
        workflow_ids = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_startup_time = end_time - start_time
        
        # All workflows should start within 2 seconds
        assert total_startup_time < 2.0, f"Concurrent startup took {total_startup_time:.3f}s, expected < 2.0s"
        
        # Verify all workflows started
        assert len(workflow_ids) == num_workflows
        assert len(set(workflow_ids)) == num_workflows  # All unique
        
        # Check startup time per workflow
        avg_startup_time = total_startup_time / num_workflows
        assert avg_startup_time < 0.2, f"Average startup time {avg_startup_time:.3f}s, expected < 0.2s"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_workflow_status_query_performance(self, high_performance_orchestrator):
        """Test workflow status query performance."""
        orchestrator = high_performance_orchestrator
        
        # Start some workflows
        workflow_ids = []
        for i in range(5):
            workflow_id = await orchestrator.start_workflow("tier1_training", {"index": i})
            workflow_ids.append(workflow_id)
        
        # Test single status query performance
        query_times = []
        for _ in range(100):  # 100 status queries
            start_time = time.perf_counter()
            await orchestrator.get_workflow_status(workflow_ids[0])
            end_time = time.perf_counter()
            query_times.append(end_time - start_time)
        
        # Calculate performance statistics
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)
        p95_query_time = statistics.quantiles(query_times, n=20)[18]  # 95th percentile
        
        # Performance assertions
        assert avg_query_time < 0.001, f"Average query time {avg_query_time:.6f}s, expected < 1ms"
        assert max_query_time < 0.01, f"Max query time {max_query_time:.6f}s, expected < 10ms"
        assert p95_query_time < 0.005, f"95th percentile {p95_query_time:.6f}s, expected < 5ms"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_allocation_performance(self, high_performance_orchestrator):
        """Test resource allocation performance."""
        orchestrator = high_performance_orchestrator
        
        # Test resource allocation times
        allocation_times = []
        
        for i in range(50):  # 50 resource allocations
            workflow_id = f"perf-test-{i}"
            
            start_time = time.perf_counter()
            
            # Start workflow (triggers resource allocation)
            await orchestrator.start_workflow("tier1_training", {
                "workflow_id": workflow_id,
                "resource_requirements": {
                    "cpu": 2.0,
                    "memory": 4096,
                    "gpu": 1
                }
            })
            
            end_time = time.perf_counter()
            allocation_times.append(end_time - start_time)
        
        # Calculate performance statistics
        avg_allocation_time = statistics.mean(allocation_times)
        max_allocation_time = max(allocation_times)
        
        # Resource allocation should be fast
        assert avg_allocation_time < 0.05, f"Average allocation time {avg_allocation_time:.3f}s, expected < 50ms"
        assert max_allocation_time < 0.2, f"Max allocation time {max_allocation_time:.3f}s, expected < 200ms"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_event_emission_performance(self, high_performance_orchestrator):
        """Test event emission performance."""
        orchestrator = high_performance_orchestrator
        
        events_received = []
        
        def fast_event_handler(event):
            events_received.append(event)
        
        # Subscribe to events
        await orchestrator.event_bus.subscribe("WORKFLOW_STARTED", fast_event_handler)
        
        # Test event emission performance
        num_workflows = 100
        
        start_time = time.perf_counter()
        
        # Start workflows rapidly (triggers events)
        for i in range(num_workflows):
            await orchestrator.start_workflow("tier1_training", {"event_test": i})
        
        # Wait for all events to be processed
        await asyncio.sleep(1)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 5.0, f"Event emission took {total_time:.3f}s for {num_workflows} workflows"
        
        # Verify events were processed
        await asyncio.sleep(2)  # Additional time for event processing
        assert len(events_received) >= num_workflows * 0.8  # At least 80% of events should be received
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_component_health_check_performance(self, high_performance_orchestrator):
        """Test component health check performance."""
        orchestrator = high_performance_orchestrator
        
        # Test component health check times
        health_check_times = []
        
        for _ in range(20):  # 20 health checks
            start_time = time.perf_counter()
            health = await orchestrator.get_component_health()
            end_time = time.perf_counter()
            
            health_check_times.append(end_time - start_time)
            
            # Verify health data was returned
            assert isinstance(health, dict)
        
        # Calculate performance statistics
        avg_health_check_time = statistics.mean(health_check_times)
        max_health_check_time = max(health_check_times)
        
        # Health checks should be fast
        assert avg_health_check_time < 0.1, f"Average health check time {avg_health_check_time:.3f}s, expected < 100ms"
        assert max_health_check_time < 0.5, f"Max health check time {max_health_check_time:.3f}s, expected < 500ms"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_workflow_list_performance(self, high_performance_orchestrator):
        """Test workflow listing performance with many workflows."""
        orchestrator = high_performance_orchestrator
        
        # Start many workflows
        num_workflows = 50
        workflow_ids = []
        
        for i in range(num_workflows):
            workflow_id = await orchestrator.start_workflow("tier1_training", {"list_test": i})
            workflow_ids.append(workflow_id)
        
        # Test workflow listing performance
        list_times = []
        
        for _ in range(10):  # 10 list operations
            start_time = time.perf_counter()
            workflows = await orchestrator.list_workflows()
            end_time = time.perf_counter()
            
            list_times.append(end_time - start_time)
            
            # Verify list contains workflows
            assert len(workflows) >= num_workflows
        
        # Calculate performance statistics
        avg_list_time = statistics.mean(list_times)
        max_list_time = max(list_times)
        
        # Listing should be fast even with many workflows
        assert avg_list_time < 0.02, f"Average list time {avg_list_time:.3f}s, expected < 20ms"
        assert max_list_time < 0.1, f"Max list time {max_list_time:.3f}s, expected < 100ms"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, high_performance_orchestrator):
        """Test memory usage under high load."""
        orchestrator = high_performance_orchestrator
        
        # Get initial resource usage
        initial_usage = await orchestrator.get_resource_usage()
        initial_memory = initial_usage.get("total_memory_allocated", 0)
        
        # Create sustained load
        num_workflows = 100
        batch_size = 10
        
        for batch in range(0, num_workflows, batch_size):
            # Start batch of workflows
            batch_workflows = []
            for i in range(batch, min(batch + batch_size, num_workflows)):
                workflow_id = await orchestrator.start_workflow("tier1_training", {
                    "memory_test": i,
                    "simulate_memory_usage": True
                })
                batch_workflows.append(workflow_id)
            
            # Let some workflows complete
            await asyncio.sleep(0.5)
        
        # Check memory usage after load
        peak_usage = await orchestrator.get_resource_usage()
        peak_memory = peak_usage.get("total_memory_allocated", 0)
        
        # Memory usage should be reasonable
        memory_increase = peak_memory - initial_memory
        assert memory_increase < 10000, f"Memory increase {memory_increase}MB too high, expected < 10GB"
        
        # Wait for workflows to complete and memory to be released
        await asyncio.sleep(5)
        
        final_usage = await orchestrator.get_resource_usage()
        final_memory = final_usage.get("total_memory_allocated", 0)
        
        # Memory should be mostly released
        memory_remaining = final_memory - initial_memory
        assert memory_remaining < 1000, f"Memory not released properly: {memory_remaining}MB remaining"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_frequency_operations(self, high_performance_orchestrator):
        """Test performance under high-frequency operations."""
        orchestrator = high_performance_orchestrator
        
        # Test rapid-fire operations
        operations_per_second = []
        test_duration = 10  # 10 seconds
        
        start_time = time.perf_counter()
        operation_count = 0
        
        while time.perf_counter() - start_time < test_duration:
            # Mix of different operations
            if operation_count % 4 == 0:
                # Start workflow
                await orchestrator.start_workflow("tier1_training", {"hf_test": operation_count})
            elif operation_count % 4 == 1:
                # Check component health
                await orchestrator.get_component_health()
            elif operation_count % 4 == 2:
                # Get resource usage
                await orchestrator.get_resource_usage()
            else:
                # List workflows
                await orchestrator.list_workflows()
            
            operation_count += 1
            
            # Record operations per second every second
            elapsed = time.perf_counter() - start_time
            if elapsed >= len(operations_per_second) + 1:
                ops_per_sec = operation_count / elapsed
                operations_per_second.append(ops_per_sec)
        
        # Calculate final performance
        total_elapsed = time.perf_counter() - start_time
        final_ops_per_sec = operation_count / total_elapsed
        
        # Should handle at least 50 operations per second
        assert final_ops_per_sec >= 50, f"Achieved {final_ops_per_sec:.1f} ops/sec, expected >= 50"
        
        # Performance should be consistent (no degradation over time)
        if len(operations_per_second) >= 5:
            early_performance = statistics.mean(operations_per_second[:3])
            late_performance = statistics.mean(operations_per_second[-3:])
            performance_ratio = late_performance / early_performance
            
            assert performance_ratio >= 0.8, f"Performance degraded by {(1-performance_ratio)*100:.1f}%"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_orchestrator_scalability(self, high_performance_orchestrator):
        """Test orchestrator scalability with increasing load."""
        orchestrator = high_performance_orchestrator
        
        # Test scalability with increasing numbers of workflows
        load_levels = [10, 25, 50, 100]
        performance_results = []
        
        for load_level in load_levels:
            # Measure performance at this load level
            start_time = time.perf_counter()
            
            # Start workflows concurrently
            tasks = []
            for i in range(load_level):
                task = orchestrator.start_workflow("tier1_training", {
                    "scalability_test": True,
                    "load_level": load_level,
                    "workflow_index": i
                })
                tasks.append(task)
            
            workflow_ids = await asyncio.gather(*tasks)
            
            # Measure time to start all workflows
            startup_time = time.perf_counter() - start_time
            
            # Measure time for status queries
            status_start = time.perf_counter()
            for workflow_id in workflow_ids[:10]:  # Sample 10 workflows
                await orchestrator.get_workflow_status(workflow_id)
            status_time = time.perf_counter() - status_start
            
            performance_results.append({
                "load_level": load_level,
                "startup_time": startup_time,
                "startup_time_per_workflow": startup_time / load_level,
                "status_query_time": status_time / 10
            })
            
            # Clean up some workflows to avoid resource exhaustion
            await asyncio.sleep(1)
        
        # Analyze scalability
        for i, result in enumerate(performance_results):
            load = result["load_level"]
            startup_per_workflow = result["startup_time_per_workflow"]
            status_time = result["status_query_time"]
            
            # Performance should not degrade significantly with load
            assert startup_per_workflow < 0.1, f"At load {load}, startup time per workflow {startup_per_workflow:.3f}s > 100ms"
            assert status_time < 0.01, f"At load {load}, status query time {status_time:.3f}s > 10ms"
        
        # Overall scalability check
        first_result = performance_results[0]
        last_result = performance_results[-1]
        
        scalability_ratio = last_result["startup_time_per_workflow"] / first_result["startup_time_per_workflow"]
        
        # Performance degradation should be minimal (less than 3x slower at 10x load)
        assert scalability_ratio < 3.0, f"Performance degraded by {scalability_ratio:.1f}x with increased load"


class TestOrchestratorStressTests:
    """Stress tests for ML Pipeline Orchestrator."""
    
    @pytest.fixture
    async def stress_test_orchestrator(self):
        """Create orchestrator for stress testing."""
        config = OrchestratorConfig(
            max_concurrent_workflows=100,
            component_health_check_interval=10,
            event_bus_buffer_size=5000
        )
        
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        
        yield orchestrator
        
        await orchestrator.shutdown()
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_orchestrator_endurance(self, stress_test_orchestrator):
        """Test orchestrator endurance under continuous load."""
        orchestrator = stress_test_orchestrator
        
        test_duration = 30  # 30 seconds of continuous operation
        workflow_interval = 0.1  # Start workflow every 100ms
        
        start_time = time.time()
        workflows_started = 0
        errors_encountered = 0
        
        while time.time() - start_time < test_duration:
            try:
                # Start workflow
                workflow_id = await orchestrator.start_workflow("tier1_training", {
                    "endurance_test": True,
                    "start_time": time.time()
                })
                workflows_started += 1
                
                # Occasional status checks
                if workflows_started % 10 == 0:
                    await orchestrator.get_component_health()
                
                await asyncio.sleep(workflow_interval)
                
            except Exception as e:
                errors_encountered += 1
                if errors_encountered > workflows_started * 0.1:  # More than 10% error rate
                    pytest.fail(f"Too many errors during endurance test: {errors_encountered}/{workflows_started}")
        
        # Verify endurance test results
        assert workflows_started >= test_duration / workflow_interval * 0.8  # At least 80% of expected workflows
        error_rate = errors_encountered / workflows_started if workflows_started > 0 else 1
        assert error_rate < 0.05, f"Error rate {error_rate:.1%} too high, expected < 5%"
        
        # Verify orchestrator is still responsive
        final_health = await orchestrator.get_component_health()
        assert isinstance(final_health, dict)
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, stress_test_orchestrator):
        """Test handling of resource exhaustion scenarios."""
        orchestrator = stress_test_orchestrator
        
        # Try to exhaust resources by requesting very high resource requirements
        high_resource_workflows = []
        resource_allocation_failures = 0
        
        for i in range(20):  # Try to start 20 very resource-intensive workflows
            try:
                workflow_id = await orchestrator.start_workflow("tier1_training", {
                    "resource_exhaustion_test": True,
                    "resource_requirements": {
                        "cpu": 32.0,  # Very high CPU requirement
                        "memory": 65536,  # 64GB memory
                        "gpu": 8  # 8 GPUs
                    }
                })
                high_resource_workflows.append(workflow_id)
                
            except Exception as e:
                resource_allocation_failures += 1
        
        # Some failures are expected when resources are exhausted
        # But orchestrator should continue functioning
        
        # Try to start a normal workflow
        normal_workflow_id = await orchestrator.start_workflow("tier1_training", {
            "normal_workflow": True,
            "resource_requirements": {
                "cpu": 1.0,
                "memory": 512
            }
        })
        
        # Normal workflow should succeed
        assert normal_workflow_id is not None
        
        # Orchestrator should still be responsive
        status = await orchestrator.get_workflow_status(normal_workflow_id)
        assert status.state == PipelineState.RUNNING
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_access_stress(self, stress_test_orchestrator):
        """Test orchestrator under high concurrent access."""
        orchestrator = stress_test_orchestrator
        
        # Define concurrent operations
        async def concurrent_operations(operation_id):
            operations_completed = 0
            errors = 0
            
            for i in range(50):  # 50 operations per concurrent task
                try:
                    if i % 5 == 0:
                        # Start workflow
                        await orchestrator.start_workflow("tier1_training", {
                            "concurrent_test": True,
                            "operation_id": operation_id,
                            "iteration": i
                        })
                    elif i % 5 == 1:
                        # Get component health
                        await orchestrator.get_component_health()
                    elif i % 5 == 2:
                        # Get resource usage
                        await orchestrator.get_resource_usage()
                    elif i % 5 == 3:
                        # List workflows
                        await orchestrator.list_workflows()
                    else:
                        # Get system status
                        # Just do a quick operation
                        await asyncio.sleep(0.001)
                    
                    operations_completed += 1
                    
                except Exception as e:
                    errors += 1
            
            return operations_completed, errors
        
        # Run concurrent operations
        num_concurrent_tasks = 20
        
        start_time = time.perf_counter()
        
        tasks = [concurrent_operations(i) for i in range(num_concurrent_tasks)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        total_operations = sum(completed for completed, _ in results)
        total_errors = sum(errors for _, errors in results)
        
        operations_per_second = total_operations / total_time
        error_rate = total_errors / total_operations if total_operations > 0 else 1
        
        # Performance assertions
        assert operations_per_second >= 100, f"Achieved {operations_per_second:.1f} ops/sec under concurrent access"
        assert error_rate < 0.1, f"Error rate {error_rate:.1%} too high under concurrent access"
        
        # Verify orchestrator is still functional
        final_status = await orchestrator.get_component_health()
        assert isinstance(final_status, dict)


if __name__ == "__main__":
    # Run basic performance test
    async def performance_test():
        """Basic performance test for orchestrator."""
        print("Running Orchestrator Performance Test...")
        
        # Create high-performance orchestrator
        config = OrchestratorConfig(max_concurrent_workflows=10)
        orchestrator = MLPipelineOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            print("✓ Orchestrator initialized")
            
            # Test workflow startup performance
            start_time = time.perf_counter()
            workflow_id = await orchestrator.start_workflow("tier1_training", {"perf_test": True})
            end_time = time.perf_counter()
            
            startup_time = end_time - start_time
            print(f"✓ Workflow startup time: {startup_time:.3f}s")
            
            # Test status query performance
            query_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                await orchestrator.get_workflow_status(workflow_id)
                end_time = time.perf_counter()
                query_times.append(end_time - start_time)
            
            avg_query_time = statistics.mean(query_times)
            print(f"✓ Average status query time: {avg_query_time:.6f}s")
            
            # Test concurrent operations
            start_time = time.perf_counter()
            
            tasks = []
            for i in range(5):
                task = orchestrator.start_workflow("tier1_training", {"concurrent": i})
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            end_time = time.perf_counter()
            
            concurrent_time = end_time - start_time
            print(f"✓ Concurrent startup time (5 workflows): {concurrent_time:.3f}s")
            
            # Performance validation
            assert startup_time < 0.1, f"Startup too slow: {startup_time:.3f}s"
            assert avg_query_time < 0.001, f"Query too slow: {avg_query_time:.6f}s"
            assert concurrent_time < 1.0, f"Concurrent startup too slow: {concurrent_time:.3f}s"
            
            print("✓ All performance tests passed!")
            
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
            raise
        finally:
            await orchestrator.shutdown()
            print("✓ Orchestrator shut down")
    
    # Run the performance test
    asyncio.run(performance_test())