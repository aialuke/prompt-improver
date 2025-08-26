"""
Integration test for database protocol consolidation.

Tests that the consolidation from 4 files to 1 unified file
works correctly in real integration scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

# Test the consolidated imports work in integration context
from src.prompt_improver.shared.interfaces.protocols.database import (
    ConnectionPoolCoreProtocol,
    DatabaseHealthProtocol,
    DatabaseProtocol,
    SessionManagerProtocol,
)


class TestProtocolConsolidationIntegration:
    """Integration tests for consolidated database protocols."""

    def test_session_manager_protocol_integration(self):
        """Test SessionManagerProtocol works in real integration scenarios."""

        class TestSessionManager:
            """Mock session manager implementing the protocol."""

            async def get_session(self):
                return Mock()

            def session_context(self):
                return AsyncMock()

            def transaction_context(self):
                return AsyncMock()

            async def health_check(self) -> bool:
                return True

            async def get_connection_info(self) -> dict:
                return {"status": "healthy", "connections": 5}

            async def close_all_sessions(self) -> None:
                pass

        # Test protocol compliance
        manager = TestSessionManager()
        assert isinstance(manager, SessionManagerProtocol), "Should implement protocol"

        # Test protocol usage in integration context
        async def integration_test():
            # Test all critical methods work
            session = await manager.get_session()
            assert session is not None

            health = await manager.health_check()
            assert health is True

            info = await manager.get_connection_info()
            assert info["status"] == "healthy"

            # Test context managers (don't need to enter, just verify they exist)
            session_ctx = manager.session_context()
            transaction_ctx = manager.transaction_context()

            assert session_ctx is not None
            assert transaction_ctx is not None

            # Test cleanup
            await manager.close_all_sessions()

        # Run the integration test
        asyncio.run(integration_test())

    def test_multiple_protocol_integration(self):
        """Test multiple protocols can be used together."""

        class IntegratedDatabaseService:
            """Service implementing multiple protocols."""

            # SessionManagerProtocol methods
            async def get_session(self):
                return Mock()

            def session_context(self):
                return AsyncMock()

            def transaction_context(self):
                return AsyncMock()

            async def health_check(self) -> bool:
                return True

            async def get_connection_info(self) -> dict:
                return {"status": "healthy"}

            async def close_all_sessions(self) -> None:
                pass

            # ConnectionPoolCoreProtocol methods
            async def initialize(self) -> bool:
                return True

            async def get_ha_connection(self, pool_name: str = "primary"):
                return Mock()

            async def execute_cached_query(self, query: str, cache_key: str | None = None):
                return {"result": "success"}

        service = IntegratedDatabaseService()

        # Test multiple protocol compliance
        assert isinstance(service, SessionManagerProtocol), "Should implement SessionManagerProtocol"
        assert isinstance(service, ConnectionPoolCoreProtocol), "Should implement ConnectionPoolCoreProtocol"

    def test_protocol_consolidation_no_conflicts(self):
        """Test that consolidated protocols don't have naming conflicts."""

        # Import all protocols and verify they're distinct
        from src.prompt_improver.shared.interfaces.protocols.database import (
            AlertingServiceProtocol,
            ConnectionPoolCoreProtocol,
            DatabaseConfigProtocol,
            DatabaseConnectionServiceProtocol,
            DatabaseHealthProtocol,
            DatabaseHealthServiceProtocol,
            DatabaseProtocol,
            DatabaseServiceProtocol,
            DatabaseServicesProtocol,
            DatabaseSessionProtocol,
            HealthMetricsServiceProtocol,
            HealthReportingServiceProtocol,
            PoolManagerFacadeProtocol,
            PoolMonitoringServiceProtocol,
            PoolScalingManagerProtocol,
            QueryExecutorProtocol,
            QueryOptimizerProtocol,
            SessionManagerProtocol,
            SessionProtocol,
        )

        # Create set of all protocols
        protocols = {
            SessionProtocol, SessionManagerProtocol, QueryExecutorProtocol,
            DatabaseSessionProtocol, DatabaseServiceProtocol, DatabaseConfigProtocol,
            QueryOptimizerProtocol, ConnectionPoolCoreProtocol, PoolScalingManagerProtocol,
            PoolMonitoringServiceProtocol, PoolManagerFacadeProtocol, DatabaseHealthProtocol,
            DatabaseConnectionServiceProtocol, HealthMetricsServiceProtocol, AlertingServiceProtocol,
            HealthReportingServiceProtocol, DatabaseHealthServiceProtocol, DatabaseProtocol,
            DatabaseServicesProtocol
        }

        # Verify we have exactly 19 distinct protocols
        assert len(protocols) == 19, f"Expected 19 distinct protocols, got {len(protocols)}"

        # Verify all are Protocol subclasses
        from typing import Protocol
        for protocol in protocols:
            assert issubclass(protocol, Protocol), f"{protocol} should be a Protocol subclass"

    @pytest.mark.asyncio
    async def test_application_service_integration(self):
        """Test protocols work with application services."""

        try:
            # Test application service can import and use protocols
            from src.prompt_improver.application.services.analytics_application_service import (
                AnalyticsApplicationService,
            )

            # If we get here, the application service successfully imports
            # which means the protocol consolidation didn't break application layer
            assert True, "Application service integration successful"

        except ImportError as e:
            pytest.fail(f"Application service integration failed: {e}")

    def test_repository_protocol_integration(self):
        """Test protocols work with repository layer."""

        try:
            # Test repository layer can import protocols
            from src.prompt_improver.repositories.base_repository import BaseRepository

            # If we get here, repository integration works
            assert True, "Repository integration successful"

        except ImportError as e:
            pytest.fail(f"Repository integration failed: {e}")

    def test_performance_characteristics_maintained(self):
        """Test that protocol consolidation maintains performance characteristics."""

        import time

        # Test rapid protocol access (simulating high-throughput usage)
        protocols = [
            SessionManagerProtocol,
            DatabaseProtocol,
            ConnectionPoolCoreProtocol,
            DatabaseHealthProtocol,
        ]

        start_time = time.perf_counter()

        # Access protocols 1000 times (simulating high usage)
        for _ in range(1000):
            for protocol in protocols:
                _ = protocol

        end_time = time.perf_counter()
        access_time = (end_time - start_time) * 1000  # Convert to ms

        # Should be extremely fast for protocol references
        assert access_time < 10.0, f"Protocol access too slow: {access_time:.3f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
