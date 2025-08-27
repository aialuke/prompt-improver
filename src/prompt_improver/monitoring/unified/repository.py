"""Monitoring repository implementation.

Provides repository pattern implementation for monitoring data persistence,
following clean architecture principles with database services integration.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from prompt_improver.database.types import ManagerMode
from prompt_improver.monitoring.unified.types import HealthCheckResult, MetricPoint

if TYPE_CHECKING:
    from prompt_improver.shared.types import SecurityContext

logger = logging.getLogger(__name__)


class MonitoringRepository:
    """Repository implementation for monitoring data persistence."""

    def __init__(self, manager_mode: ManagerMode = ManagerMode.HIGH_AVAILABILITY) -> None:
        self.manager_mode = manager_mode
        self._security_context: SecurityContext | None = None
        self._database_services = None

    async def _ensure_initialized(self) -> None:
        """Ensure repository is initialized with database services."""
        if self._database_services is None:
            # Use delayed imports to avoid circular dependencies
            from prompt_improver.database.composition import get_database_services
            self._database_services = await get_database_services(self.manager_mode)

        if self._security_context is None:
            # Use delayed imports to avoid circular dependencies
            from prompt_improver.database.security_integration import (
                create_security_context,
            )
            self._security_context = await create_security_context(
                agent_id="monitoring_repository",
                tier="basic"
            )

    async def store_health_result(self, result: HealthCheckResult) -> None:
        """Store health check result."""
        try:
            await self._ensure_initialized()

            # Store in cache for quick retrieval
            cache_key = f"health_check:{result.component_name}:latest"
            health_data = {
                "status": result.status.value,
                "component_name": result.component_name,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
                "timestamp": result.timestamp.isoformat(),
                "category": result.category.value,
                "details": result.details,
                "error": result.error,
            }

            await self._database_services.set_cached(
                cache_key,
                health_data,
                ttl_seconds=300,  # 5 minutes cache
                security_context=self._security_context,
            )

            # Also store historical data with timestamp
            history_key = f"health_check:{result.component_name}:history:{int(result.timestamp.timestamp())}"
            await self._database_services.set_cached(
                history_key,
                health_data,
                ttl_seconds=86400,  # 24 hours retention
                security_context=self._security_context,
            )

        except Exception as e:
            logger.exception(f"Failed to store health result for {result.component_name}: {e}")
            raise

    async def store_metrics(self, metrics: list[MetricPoint]) -> None:
        """Store multiple metric points."""
        if not metrics:
            return

        try:
            await self._ensure_initialized()

            for metric in metrics:
                # Store latest metric value
                cache_key = f"metric:{metric.name}:latest"
                metric_data = {
                    "name": metric.name,
                    "value": metric.value,
                    "metric_type": metric.metric_type.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "tags": metric.tags,
                    "unit": metric.unit,
                    "description": metric.description,
                }

                await self._database_services.set_cached(
                    cache_key,
                    metric_data,
                    ttl_seconds=300,  # 5 minutes cache
                    security_context=self._security_context,
                )

                # Store historical data
                history_key = f"metric:{metric.name}:history:{int(metric.timestamp.timestamp())}"
                await self._database_services.set_cached(
                    history_key,
                    metric_data,
                    ttl_seconds=86400,  # 24 hours retention
                    security_context=self._security_context,
                )

        except Exception as e:
            logger.exception(f"Failed to store metrics: {e}")
            raise

    async def get_health_history(
        self,
        component_name: str,
        hours_back: int = 24
    ) -> list[HealthCheckResult]:
        """Get health check history for component."""
        results = []

        try:
            await self._ensure_initialized()

            # Get current time and calculate time range
            now = datetime.now(UTC)
            start_time = now - timedelta(hours=hours_back)

            # For demonstration, get latest result
            # In production, this would query historical data more efficiently
            cache_key = f"health_check:{component_name}:latest"
            cached_data = await self._database_services.get_cached(
                cache_key, self._security_context
            )

            if cached_data:
                results.append(self._dict_to_health_result(cached_data))

        except Exception as e:
            logger.exception(f"Failed to get health history for {component_name}: {e}")

        return results

    async def get_metrics_history(
        self,
        metric_name: str,
        hours_back: int = 24,
        tags: dict[str, str] | None = None
    ) -> list[MetricPoint]:
        """Get metrics history."""
        metrics = []

        try:
            await self._ensure_initialized()

            # For demonstration, get latest metric
            # In production, this would query historical data more efficiently
            cache_key = f"metric:{metric_name}:latest"
            cached_data = await self._database_services.get_cached(
                cache_key, self._security_context
            )

            if cached_data:
                metric = self._dict_to_metric_point(cached_data)

                # Filter by tags if provided
                if tags is None or self._tags_match(metric.tags, tags):
                    metrics.append(metric)

        except Exception as e:
            logger.exception(f"Failed to get metrics history for {metric_name}: {e}")

        return metrics

    async def cleanup_old_data(self, retention_hours: int) -> int:
        """Clean up old monitoring data."""
        cleaned_count = 0

        try:
            await self._ensure_initialized()

            # Calculate cutoff timestamp
            cutoff_time = datetime.now(UTC) - timedelta(hours=retention_hours)
            cutoff_timestamp = int(cutoff_time.timestamp())

            # This is a simplified cleanup - in production you'd want to
            # implement more sophisticated cleanup logic
            logger.info(f"Cleanup would remove data older than {cutoff_time}")

        except Exception as e:
            logger.exception(f"Failed to cleanup old monitoring data: {e}")

        return cleaned_count

    def _dict_to_health_result(self, data: dict[str, Any]) -> HealthCheckResult:
        """Convert dictionary to HealthCheckResult."""
        from prompt_improver.monitoring.unified.types import (
            ComponentCategory,
            HealthStatus,
        )

        status = HealthStatus(data.get("status", "unknown"))
        category = ComponentCategory(data.get("category", "custom"))
        timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now(UTC).isoformat()))

        return HealthCheckResult(
            status=status,
            component_name=data.get("component_name", ""),
            message=data.get("message", ""),
            details=data.get("details", {}),
            response_time_ms=data.get("response_time_ms", 0.0),
            timestamp=timestamp,
            error=data.get("error"),
            category=category,
        )

    def _dict_to_metric_point(self, data: dict[str, Any]) -> MetricPoint:
        """Convert dictionary to MetricPoint."""
        from prompt_improver.monitoring.unified.types import MetricType

        metric_type = MetricType(data.get("metric_type", "gauge"))
        timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now(UTC).isoformat()))

        return MetricPoint(
            name=data.get("name", ""),
            value=data.get("value", 0.0),
            metric_type=metric_type,
            tags=data.get("tags", {}),
            timestamp=timestamp,
            unit=data.get("unit", ""),
            description=data.get("description", ""),
        )

    def _tags_match(self, metric_tags: dict[str, str], filter_tags: dict[str, str]) -> bool:
        """Check if metric tags match filter criteria."""
        return all(metric_tags.get(key) == value for key, value in filter_tags.items())
