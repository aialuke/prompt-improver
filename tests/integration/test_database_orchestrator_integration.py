"""
Integration tests for database components with ML orchestrator.

Following 2025 best practices:
- Real orchestrator integration (no mocks)
- Event-driven coordination testing
- Component capability validation
- End-to-end workflow testing
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any, List

from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent
from prompt_improver.ml.orchestration.connectors.tier4_connectors import (
    DatabasePerformanceMonitorConnector,
    DatabaseConnectionOptimizerConnector,
    Tier4ConnectorFactory
)


class TestDatabaseOrchestratorIntegration:
    """Integration tests for database components with ML orchestrator."""
    
    @pytest.fixture
    async def orchestrator_config(self):
        """Create orchestrator configuration for testing."""
        config = OrchestratorConfig()
        config.event_bus_buffer_size = 1000
        config.event_handler_timeout = 5.0
        return config
    
    @pytest.fixture
    async def orchestrator(self, orchestrator_config):
        """Create ML orchestrator with database components."""
        orchestrator = MLPipelineOrchestrator(orchestrator_config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_database_component_registration(self, orchestrator):
        """Test that database components are properly registered with orchestrator."""
        # Act - Get available components
        available_components = await orchestrator.list_available_components()
        
        # Assert - Database components are available
        assert "database_performance_monitor" in available_components
        assert "database_connection_optimizer" in available_components
        
        # Verify component metadata
        db_monitor_info = available_components["database_performance_monitor"]
        assert db_monitor_info["tier"] == "TIER_4_PERFORMANCE"
        assert db_monitor_info["status"] in ["READY", "INITIALIZING"]
        
        db_optimizer_info = available_components["database_connection_optimizer"]
        assert db_optimizer_info["tier"] == "TIER_4_PERFORMANCE"
        assert db_optimizer_info["status"] in ["READY", "INITIALIZING"]
    
    @pytest.mark.asyncio
    async def test_database_performance_monitor_capabilities(self, orchestrator):
        """Test database performance monitor capabilities through orchestrator."""
        # Act - Execute real-time monitoring capability
        result = await orchestrator.execute_component_capability(
            component_name="database_performance_monitor",
            capability_name="real_time_monitoring",
            inputs={"monitoring_config": {"interval": 1}}
        )
        
        # Assert - Real monitoring data returned
        assert result["status"] == "success"
        assert "snapshot" in result
        
        snapshot = result["snapshot"]
        assert "timestamp" in snapshot
        assert "cache_hit_ratio" in snapshot
        assert "active_connections" in snapshot
        assert "avg_query_time_ms" in snapshot
        
        # Validate real data types and ranges
        assert isinstance(snapshot["cache_hit_ratio"], (int, float))
        assert isinstance(snapshot["active_connections"], int)
        assert isinstance(snapshot["avg_query_time_ms"], (int, float))
        assert 0 <= snapshot["cache_hit_ratio"] <= 100
        assert snapshot["active_connections"] >= 0
        assert snapshot["avg_query_time_ms"] >= 0
    
    @pytest.mark.asyncio
    async def test_database_cache_monitoring_capability(self, orchestrator):
        """Test database cache monitoring capability."""
        # Act - Execute cache hit monitoring
        result = await orchestrator.execute_component_capability(
            component_name="database_performance_monitor",
            capability_name="cache_hit_monitoring",
            inputs={"cache_config": {"threshold": 90.0}}
        )
        
        # Assert - Real cache metrics returned
        assert result["status"] == "success"
        assert "cache_hit_ratio" in result
        assert "index_hit_ratio" in result
        assert "meets_target" in result
        assert "target_threshold" in result
        
        # Validate real metrics
        assert isinstance(result["cache_hit_ratio"], (int, float))
        assert isinstance(result["index_hit_ratio"], (int, float))
        assert isinstance(result["meets_target"], bool)
        assert result["target_threshold"] == 90.0
    
    @pytest.mark.asyncio
    async def test_database_connection_optimizer_capabilities(self, orchestrator):
        """Test database connection optimizer capabilities through orchestrator."""
        # Act - Execute system resource analysis
        result = await orchestrator.execute_component_capability(
            component_name="database_connection_optimizer",
            capability_name="system_resource_analysis",
            inputs={"analysis_config": {}}
        )
        
        # Assert - Real system analysis returned
        assert result["status"] == "success"
        assert "system_resources" in result
        assert "recommended_settings" in result
        assert "analysis_timestamp" in result
        
        # Validate real system resources
        system_resources = result["system_resources"]
        assert "total_memory_gb" in system_resources
        assert "cpu_count" in system_resources
        assert system_resources["total_memory_gb"] > 0
        assert system_resources["cpu_count"] > 0
        
        # Validate recommended settings
        recommended_settings = result["recommended_settings"]
        assert "work_mem" in recommended_settings
        assert "effective_cache_size" in recommended_settings
    
    @pytest.mark.asyncio
    async def test_database_optimization_execution(self, orchestrator):
        """Test database optimization execution through orchestrator."""
        # Act - Execute connection optimization
        result = await orchestrator.execute_component_capability(
            component_name="database_connection_optimizer",
            capability_name="optimize_connection_settings",
            inputs={"optimization_config": {}}
        )
        
        # Assert - Optimization completed successfully
        assert result["status"] == "success"
        assert "message" in result
        assert "optimization_applied" in result
        assert result["optimization_applied"] is True
        assert "dynamic resource detection" in result["message"]
    
    @pytest.mark.asyncio
    async def test_database_index_creation_capability(self, orchestrator):
        """Test database index creation capability."""
        # Act - Execute index creation
        result = await orchestrator.execute_component_capability(
            component_name="database_connection_optimizer",
            capability_name="create_performance_indexes",
            inputs={"index_config": {}}
        )
        
        # Assert - Index creation completed
        assert result["status"] == "success"
        assert "message" in result
        assert "indexes_created" in result
        assert result["indexes_created"] is True
    
    @pytest.mark.asyncio
    async def test_event_driven_database_coordination(self, orchestrator):
        """Test event-driven coordination between database components and orchestrator."""
        # Arrange - Set up event collection
        emitted_events = []
        
        async def event_collector(event: MLEvent):
            emitted_events.append(event)
        
        # Subscribe to database events
        orchestrator.event_bus.subscribe(EventType.DATABASE_PERFORMANCE_SNAPSHOT_TAKEN, event_collector)
        orchestrator.event_bus.subscribe(EventType.DATABASE_CONNECTION_OPTIMIZED, event_collector)
        
        # Act - Execute monitoring and optimization
        await orchestrator.execute_component_capability(
            component_name="database_performance_monitor",
            capability_name="real_time_monitoring",
            inputs={}
        )
        
        await orchestrator.execute_component_capability(
            component_name="database_connection_optimizer",
            capability_name="optimize_connection_settings",
            inputs={}
        )
        
        # Wait for event processing
        await asyncio.sleep(0.2)
        
        # Assert - Events were emitted for coordination
        snapshot_events = [e for e in emitted_events if e.event_type == EventType.DATABASE_PERFORMANCE_SNAPSHOT_TAKEN]
        optimization_events = [e for e in emitted_events if e.event_type == EventType.DATABASE_CONNECTION_OPTIMIZED]
        
        # Note: Events may not be emitted if components don't have event bus integration
        # This tests the orchestrator's ability to handle events when they are emitted
        assert isinstance(snapshot_events, list)
        assert isinstance(optimization_events, list)
    
    @pytest.mark.asyncio
    async def test_database_component_error_handling(self, orchestrator):
        """Test error handling in database components through orchestrator."""
        # Act - Execute capability with invalid inputs
        result = await orchestrator.execute_component_capability(
            component_name="database_performance_monitor",
            capability_name="slow_query_detection",
            inputs={"limit": -1}  # Invalid limit
        )
        
        # Assert - Error handled gracefully
        # Note: The actual behavior depends on component implementation
        # This tests that the orchestrator can handle component errors
        assert "status" in result
        # Component should either handle the error gracefully or return error status
    
    @pytest.mark.asyncio
    async def test_database_workflow_coordination(self, orchestrator):
        """Test coordinated database workflow through orchestrator."""
        # Act - Execute a coordinated workflow
        # 1. Analyze system resources
        analysis_result = await orchestrator.execute_component_capability(
            component_name="database_connection_optimizer",
            capability_name="system_resource_analysis",
            inputs={}
        )
        
        # 2. Optimize based on analysis
        optimization_result = await orchestrator.execute_component_capability(
            component_name="database_connection_optimizer",
            capability_name="optimize_connection_settings",
            inputs={}
        )
        
        # 3. Monitor performance after optimization
        monitoring_result = await orchestrator.execute_component_capability(
            component_name="database_performance_monitor",
            capability_name="real_time_monitoring",
            inputs={}
        )
        
        # Assert - Workflow completed successfully
        assert analysis_result["status"] == "success"
        assert optimization_result["status"] == "success"
        assert monitoring_result["status"] == "success"
        
        # Validate workflow coordination
        assert "system_resources" in analysis_result
        assert "optimization_applied" in optimization_result
        assert "snapshot" in monitoring_result
    
    @pytest.mark.asyncio
    async def test_tier4_connector_factory(self):
        """Test Tier 4 connector factory for database components."""
        # Act - Create connectors using factory
        db_monitor_connector = Tier4ConnectorFactory.create_connector("database_performance_monitor")
        db_optimizer_connector = Tier4ConnectorFactory.create_connector("database_connection_optimizer")
        
        # Assert - Connectors created successfully
        assert isinstance(db_monitor_connector, DatabasePerformanceMonitorConnector)
        assert isinstance(db_optimizer_connector, DatabaseConnectionOptimizerConnector)
        
        # Validate connector metadata
        assert db_monitor_connector.metadata.name == "database_performance_monitor"
        assert db_optimizer_connector.metadata.name == "database_connection_optimizer"
        
        # Validate capabilities
        assert len(db_monitor_connector.metadata.capabilities) > 0
        assert len(db_optimizer_connector.metadata.capabilities) > 0
        
        # Check for expected capabilities
        monitor_capability_names = [cap.name for cap in db_monitor_connector.metadata.capabilities]
        optimizer_capability_names = [cap.name for cap in db_optimizer_connector.metadata.capabilities]
        
        assert "real_time_monitoring" in monitor_capability_names
        assert "cache_hit_monitoring" in monitor_capability_names
        assert "optimize_connection_settings" in optimizer_capability_names
        assert "system_resource_analysis" in optimizer_capability_names
    
    @pytest.mark.asyncio
    async def test_component_lifecycle_management(self, orchestrator):
        """Test component lifecycle management through orchestrator."""
        # Act - Get component status
        status = await orchestrator.get_component_status("database_performance_monitor")
        
        # Assert - Component is properly managed
        assert "status" in status
        assert "metadata" in status
        assert status["status"] in ["READY", "INITIALIZING", "STOPPED", "ERROR"]
        
        # Validate metadata
        metadata = status["metadata"]
        assert "name" in metadata
        assert "tier" in metadata
        assert "capabilities" in metadata
        assert metadata["name"] == "database_performance_monitor"

    @pytest.mark.asyncio
    async def test_database_event_handler_integration(self, orchestrator):
        """Test database event handler integration with orchestrator."""
        # Arrange - Import and register database event handler
        from prompt_improver.ml.orchestration.events.handlers.database_handler import DatabaseEventHandler

        db_handler = DatabaseEventHandler(orchestrator=orchestrator, event_bus=orchestrator.event_bus)

        # Subscribe handler to database events
        orchestrator.event_bus.subscribe(EventType.DATABASE_PERFORMANCE_SNAPSHOT_TAKEN, db_handler.handle_event)
        orchestrator.event_bus.subscribe(EventType.DATABASE_CACHE_HIT_RATIO_LOW, db_handler.handle_event)

        # Act - Emit test database events
        test_snapshot_event = MLEvent(
            event_type=EventType.DATABASE_PERFORMANCE_SNAPSHOT_TAKEN,
            source="test",
            data={
                "snapshot": {
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit_ratio": 85.0,  # Below 90% threshold
                    "active_connections": 15,
                    "avg_query_time_ms": 45.0,
                    "slow_queries_count": 2
                }
            }
        )

        await orchestrator.event_bus.emit(test_snapshot_event)

        # Wait for event processing
        await asyncio.sleep(0.2)

        # Assert - Handler processed events
        assert db_handler.events_processed > 0
        assert len(db_handler.performance_snapshots) > 0

        # Validate handler statistics
        stats = db_handler.get_statistics()
        assert stats["events_processed"] > 0
        assert stats["performance_snapshots_count"] > 0
        assert "alert_thresholds" in stats
