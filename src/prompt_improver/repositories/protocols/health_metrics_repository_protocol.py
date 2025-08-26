"""Health Metrics Repository Protocol.

Defines the interface for health monitoring and database performance metrics
collection following clean architecture principles and repository pattern.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HealthMetricsRepositoryProtocol(Protocol):
    """Protocol for health metrics and database monitoring operations."""

    async def check_database_connectivity(self) -> dict[str, Any]:
        """Check basic database connectivity with response time measurement.

        Returns:
            Dict containing connectivity status, response_time_ms, and error info
        """
        ...

    async def get_active_connections(self) -> int:
        """Get count of active database connections.

        Returns:
            Number of active connections
        """
        ...

    async def get_long_running_queries(self, threshold_seconds: int = 30) -> list[dict[str, Any]]:
        """Get information about long-running queries.

        Args:
            threshold_seconds: Query duration threshold in seconds

        Returns:
            List of long-running query information
        """
        ...

    async def get_query_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive query performance statistics.

        Returns:
            Dict containing query performance metrics
        """
        ...

    async def get_connection_pool_status(self) -> dict[str, Any]:
        """Get connection pool health and utilization metrics.

        Returns:
            Dict containing pool status and metrics
        """
        ...

    async def execute_health_query(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a health monitoring query safely.

        Args:
            query: SQL query string
            parameters: Query parameters

        Returns:
            Query result
        """
        ...

    async def measure_query_latency(self, test_query: str = "SELECT 1") -> float:
        """Measure database query latency.

        Args:
            test_query: Query to use for latency measurement

        Returns:
            Latency in milliseconds
        """
        ...

    async def get_database_size_metrics(self) -> dict[str, Any]:
        """Get database size and storage metrics.

        Returns:
            Dict containing database size information
        """
        ...

    async def get_table_statistics(self) -> list[dict[str, Any]]:
        """Get table-level statistics for health monitoring.

        Returns:
            List of table statistics
        """
        ...

    async def get_index_health_metrics(self) -> list[dict[str, Any]]:
        """Get index health and usage statistics.

        Returns:
            List of index health metrics
        """
        ...
