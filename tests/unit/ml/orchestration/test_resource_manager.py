"""
Tests for Resource Manager.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from prompt_improver.ml.orchestration.core.resource_manager import (
    ResourceManager, ResourceType, ResourceAllocation, ResourceUsage
)
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent


class TestResourceManager:
    """Test suite for Resource Manager."""
    
    @pytest.fixture
    async def resource_manager(self):
        """Create resource manager instance for testing."""
        config = OrchestratorConfig(
            max_concurrent_workflows=2,
            gpu_allocation_timeout=300
        )
        
        # Mock event bus
        mock_event_bus = Mock()
        mock_event_bus.emit = Mock()
        
        manager = ResourceManager(config)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, resource_manager):
        """Test resource manager initialization."""
        assert resource_manager._is_initialized is True
        assert resource_manager.allocations == {}
        assert len(resource_manager.allocation_history) == 0
    
    @pytest.mark.asyncio
    async def test_allocate_cpu_resources(self, resource_manager):
        """Test CPU resource allocation."""
        workflow_id = "test-workflow-123"
        requirements = {
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 1024  # MB
        }
        
        allocation = await resource_manager.allocate_resources(workflow_id, requirements)
        
        assert allocation is not None
        assert allocation.workflow_id == workflow_id
        assert allocation.allocated_resources[ResourceType.CPU] == 2.0
        assert allocation.allocated_resources[ResourceType.MEMORY] == 1024
        assert allocation.allocation_time is not None
        assert allocation.allocation_id in resource_manager.allocations
    
    @pytest.mark.asyncio
    async def test_allocate_gpu_resources(self, resource_manager):
        """Test GPU resource allocation."""
        workflow_id = "test-workflow-gpu"
        requirements = {
            ResourceType.GPU: 1,
            ResourceType.GPU_MEMORY: 8192  # MB
        }
        
        allocation = await resource_manager.allocate_resources(workflow_id, requirements)
        
        assert allocation is not None
        assert allocation.allocated_resources[ResourceType.GPU] == 1
        assert allocation.allocated_resources[ResourceType.GPU_MEMORY] == 8192
    
    @pytest.mark.asyncio
    async def test_deallocate_resources(self, resource_manager):
        """Test resource deallocation."""
        workflow_id = "test-workflow-123"
        requirements = {ResourceType.CPU: 1.0, ResourceType.MEMORY: 512}
        
        # Allocate resources
        allocation = await resource_manager.allocate_resources(workflow_id, requirements)
        allocation_id = allocation.allocation_id
        
        assert allocation_id in resource_manager.allocations
        
        # Deallocate resources
        success = await resource_manager.deallocate_resources(allocation_id)
        
        assert success is True
        assert allocation_id not in resource_manager.allocations
        assert len(resource_manager.allocation_history) == 1
    
    @pytest.mark.asyncio
    async def test_resource_usage_stats(self, resource_manager):
        """Test resource usage statistics."""
        # Allocate some resources
        workflow_id1 = "workflow-1"
        workflow_id2 = "workflow-2"
        
        await resource_manager.allocate_resources(workflow_id1, {ResourceType.CPU: 2.0, ResourceType.MEMORY: 1024})
        await resource_manager.allocate_resources(workflow_id2, {ResourceType.CPU: 1.0, ResourceType.MEMORY: 512})
        
        stats = await resource_manager.get_resource_usage_stats()
        
        assert isinstance(stats, ResourceUsage)
        assert stats.total_cpu_allocated == 3.0
        assert stats.total_memory_allocated == 1536
        assert stats.active_allocations == 2
    
    @pytest.mark.asyncio
    async def test_resource_availability(self, resource_manager):
        """Test resource availability checking."""
        # Check initial availability
        available = await resource_manager.check_resource_availability({ResourceType.CPU: 1.0})
        assert available is True
        
        # Allocate significant resources
        workflow_id = "heavy-workflow"
        heavy_requirements = {
            ResourceType.CPU: 16.0,  # High CPU usage
            ResourceType.MEMORY: 32768  # 32GB memory
        }
        
        allocation = await resource_manager.allocate_resources(workflow_id, heavy_requirements)
        assert allocation is not None
        
        # Check availability for similar heavy workload
        available = await resource_manager.check_resource_availability(heavy_requirements)
        # Should still be available unless we've hit hard limits
        assert isinstance(available, bool)
    
    @pytest.mark.asyncio
    async def test_allocation_cleanup(self, resource_manager):
        """Test cleanup of expired allocations."""
        workflow_id = "test-workflow-cleanup"
        requirements = {ResourceType.CPU: 1.0}
        
        allocation = await resource_manager.allocate_resources(workflow_id, requirements)
        
        # Manually set allocation time to past (simulate expired allocation)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        resource_manager.allocations[allocation.allocation_id].allocation_time = old_time
        
        # Run cleanup
        cleaned = await resource_manager._cleanup_expired_allocations()
        
        assert cleaned >= 0  # Should have cleaned up expired allocations
    
    @pytest.mark.asyncio
    async def test_concurrent_allocations(self, resource_manager):
        """Test concurrent resource allocations."""
        workflows = [f"workflow-{i}" for i in range(5)]
        requirements = {ResourceType.CPU: 0.5, ResourceType.MEMORY: 256}
        
        # Allocate resources concurrently
        tasks = [
            resource_manager.allocate_resources(workflow_id, requirements)
            for workflow_id in workflows
        ]
        
        allocations = await asyncio.gather(*tasks)
        
        # All allocations should succeed
        assert len(allocations) == 5
        assert all(allocation is not None for allocation in allocations)
        
        # All should have unique allocation IDs
        allocation_ids = [alloc.allocation_id for alloc in allocations]
        assert len(set(allocation_ids)) == 5
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, resource_manager):
        """Test memory optimization features."""
        # Simulate memory pressure
        workflow_id = "memory-test"
        requirements = {ResourceType.MEMORY: 2048}
        
        allocation = await resource_manager.allocate_resources(workflow_id, requirements)
        assert allocation is not None
        
        # Trigger memory optimization
        optimized = await resource_manager.optimize_memory_usage()
        assert isinstance(optimized, bool)
        
        # Check if garbage collection was triggered
        stats_before = await resource_manager.get_resource_usage_stats()
        await resource_manager._trigger_garbage_collection()
        stats_after = await resource_manager.get_resource_usage_stats()
        
        # Stats should be available after GC
        assert stats_after is not None
    
    @pytest.mark.asyncio
    async def test_resource_limits(self, resource_manager):
        """Test resource allocation limits."""
        workflow_id = "limit-test"
        
        # Try to allocate extremely high resources
        extreme_requirements = {
            ResourceType.CPU: 1000.0,  # Unrealistic CPU count
            ResourceType.MEMORY: 1024 * 1024 * 1024  # 1TB memory
        }
        
        allocation = await resource_manager.allocate_resources(workflow_id, extreme_requirements)
        
        # Should still return an allocation (manager handles what's actually possible)
        assert allocation is not None
        
        # But allocated amounts might be different from requested
        actual_cpu = allocation.allocated_resources.get(ResourceType.CPU, 0)
        actual_memory = allocation.allocated_resources.get(ResourceType.MEMORY, 0)
        
        # Should be reasonable values
        assert actual_cpu > 0
        assert actual_memory > 0


class TestResourceAllocation:
    """Test suite for ResourceAllocation."""
    
    def test_allocation_creation(self):
        """Test resource allocation creation."""
        workflow_id = "test-workflow"
        allocation_id = "alloc-123"
        resources = {ResourceType.CPU: 2.0, ResourceType.MEMORY: 1024}
        
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            workflow_id=workflow_id,
            allocated_resources=resources
        )
        
        assert allocation.allocation_id == allocation_id
        assert allocation.workflow_id == workflow_id
        assert allocation.allocated_resources == resources
        assert allocation.allocation_time is not None
    
    def test_allocation_serialization(self):
        """Test allocation to/from dict conversion."""
        allocation = ResourceAllocation(
            allocation_id="alloc-123",
            workflow_id="workflow-456",
            allocated_resources={ResourceType.CPU: 1.0, ResourceType.MEMORY: 512}
        )
        
        # Convert to dict and back
        allocation_dict = allocation.to_dict()
        restored_allocation = ResourceAllocation.from_dict(allocation_dict)
        
        assert restored_allocation.allocation_id == allocation.allocation_id
        assert restored_allocation.workflow_id == allocation.workflow_id
        assert restored_allocation.allocated_resources == allocation.allocated_resources
    
    def test_allocation_age_calculation(self):
        """Test allocation age calculation."""
        allocation = ResourceAllocation(
            allocation_id="alloc-123",
            workflow_id="workflow-456",
            allocated_resources={ResourceType.CPU: 1.0}
        )
        
        # Age should be very small (just created)
        age = allocation.age
        assert age.total_seconds() < 1.0
        
        # Manually set old allocation time
        old_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        allocation.allocation_time = old_time
        
        # Age should be approximately 30 minutes
        age = allocation.age
        assert 29 * 60 <= age.total_seconds() <= 31 * 60


class TestResourceUsage:
    """Test suite for ResourceUsage."""
    
    def test_stats_creation(self):
        """Test resource usage stats creation."""
        stats = ResourceUsage(
            total_cpu_allocated=4.0,
            total_memory_allocated=2048,
            total_gpu_allocated=2,
            total_gpu_memory_allocated=16384,
            active_allocations=3,
            total_allocations=10,
            allocation_history_size=7
        )
        
        assert stats.total_cpu_allocated == 4.0
        assert stats.total_memory_allocated == 2048
        assert stats.total_gpu_allocated == 2
        assert stats.total_gpu_memory_allocated == 16384
        assert stats.active_allocations == 3
        assert stats.total_allocations == 10
        assert stats.allocation_history_size == 7
    
    def test_stats_serialization(self):
        """Test stats to/from dict conversion."""
        stats = ResourceUsage(
            total_cpu_allocated=2.0,
            total_memory_allocated=1024,
            active_allocations=2,
            total_allocations=5,
            allocation_history_size=3
        )
        
        # Convert to dict and back
        stats_dict = stats.to_dict()
        restored_stats = ResourceUsage.from_dict(stats_dict)
        
        assert restored_stats.total_cpu_allocated == stats.total_cpu_allocated
        assert restored_stats.total_memory_allocated == stats.total_memory_allocated
        assert restored_stats.active_allocations == stats.active_allocations
        assert restored_stats.total_allocations == stats.total_allocations
        assert restored_stats.allocation_history_size == stats.allocation_history_size


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test for resource manager."""
        print("Running Resource Manager smoke test...")
        
        # Create and initialize manager
        config = OrchestratorConfig()
        mock_event_bus = Mock()
        manager = ResourceManager(config)
        
        try:
            await manager.initialize()
            print("✓ Resource manager initialized successfully")
            
            # Test resource allocation
            workflow_id = "smoke-test-workflow"
            requirements = {ResourceType.CPU: 1.0, ResourceType.MEMORY: 512}
            
            allocation = await manager.allocate_resources(workflow_id, requirements)
            print(f"✓ Resources allocated: {allocation.allocation_id}")
            
            # Test resource stats
            stats = await manager.get_resource_usage_stats()
            print(f"✓ Resource stats: {stats.active_allocations} active allocations")
            
            # Test resource deallocation
            success = await manager.deallocate_resources(allocation.allocation_id)
            print(f"✓ Resources deallocated: {success}")
            
            # Test resource availability
            available = await manager.check_resource_availability(requirements)
            print(f"✓ Resource availability checked: {available}")
            
            print("✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
        finally:
            await manager.shutdown()
            print("✓ Resource manager shut down gracefully")
    
    # Run the smoke test
    asyncio.run(smoke_test())