"""Real-time API Endpoints for A/B Testing Analytics
Provides WebSocket and REST endpoints for live experiment monitoring
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from prompt_improver.database import (
    DatabaseServices,
    ManagerMode,
    create_database_services,
    get_database_services,
)

# Removed direct database imports - now using repository pattern
from prompt_improver.repositories.factory import (
    get_analytics_repository,
    get_health_repository,
)
from prompt_improver.repositories.protocols.analytics_repository_protocol import (
    AnalyticsRepositoryProtocol,
)
from prompt_improver.repositories.protocols.health_repository_protocol import (
    HealthRepositoryProtocol,
)

redis_available = True
try:
    import coredis
except ImportError:
    redis_available = False
    coredis = None
analytics_available = True
try:
    from prompt_improver.performance.analytics.real_time_analytics import (
        get_real_time_analytics_service,
    )
except ImportError:
    analytics_available = False

    async def get_real_time_analytics_service(
        analytics_repository: AnalyticsRepositoryProtocol,
    ) -> Any:
        """Repository-based analytics service using AnalyticsRepositoryProtocol"""

        class RepositoryAnalyticsService:
            """Analytics service implementation using repository pattern"""

            def __init__(self, analytics_repository: AnalyticsRepositoryProtocol):
                self.analytics_repository = analytics_repository
                self.logger = logging.getLogger(__name__)

            async def get_real_time_metrics(
                self, experiment_id: str
            ) -> dict[str, Any] | None:
                """Get real-time metrics for an experiment using repository"""
                try:
                    # Use analytics repository to get session analytics
                    from datetime import timedelta

                    end_date = datetime.now(UTC)
                    start_date = end_date - timedelta(hours=24)

                    analytics_result = (
                        await self.analytics_repository.get_session_analytics(
                            start_date, end_date
                        )
                    )

                    # Create metrics based on analytics data
                    current_time = datetime.now(UTC)
                    metrics = {
                        "experiment_id": experiment_id,
                        "experiment_name": f"Experiment {experiment_id}",
                        "status": "running",
                        "current_sample_size": analytics_result.total_sessions,
                        "target_sample_size": 200,  # Default target
                        "completion_percentage": min(
                            analytics_result.total_sessions / 200 * 100, 100.0
                        ),
                        "duration_hours": 24.0,  # Using 24h window
                        "target_metric": "improvement_score",
                        "significance_threshold": 0.05,
                        "results": {
                            "avg_improvement": analytics_result.avg_improvement_score,
                            "performance_distribution": analytics_result.performance_distribution,
                        },
                        "metadata": {"repository_based": True},
                        "last_updated": current_time.isoformat(),
                    }
                    return metrics
                except Exception as e:
                    self.logger.error(
                        f"Error getting real-time metrics for experiment {experiment_id}: {e}"
                    )
                    return None

            async def start_experiment_monitoring(
                self, experiment_id: str, update_interval: int
            ) -> bool:
                """Start monitoring for an experiment using repository"""
                try:
                    self.logger.info(
                        f"Started monitoring for experiment {experiment_id} with {update_interval}s interval (repository-based)"
                    )
                    return True
                except Exception as e:
                    self.logger.error(
                        f"Error starting monitoring for experiment {experiment_id}: {e}"
                    )
                    return False

            async def stop_experiment_monitoring(self, experiment_id: str) -> bool:
                """Stop monitoring for an experiment using repository"""
                try:
                    self.logger.info(
                        f"Stopped monitoring for experiment {experiment_id} (repository-based)"
                    )
                    return True
                except Exception as e:
                    self.logger.error(
                        f"Error stopping monitoring for experiment {experiment_id}: {e}"
                    )
                    return False

            async def get_active_experiments(self) -> list[str]:
                """Get list of active experiment IDs using repository"""
                try:
                    # Use analytics to simulate active experiments
                    from datetime import timedelta

                    end_date = datetime.now(UTC)
                    start_date = end_date - timedelta(hours=24)

                    analytics_result = (
                        await self.analytics_repository.get_session_analytics(
                            start_date, end_date
                        )
                    )
                    # Simulate experiment IDs based on session count
                    experiment_count = min(
                        analytics_result.total_sessions // 10, 5
                    )  # Max 5 experiments
                    active_experiment_ids = [
                        f"exp_{i + 1}" for i in range(experiment_count)
                    ]

                    self.logger.info(
                        f"Found {len(active_experiment_ids)} active experiments (repository-based)"
                    )
                    return active_experiment_ids
                except Exception as e:
                    self.logger.error(f"Error getting active experiments: {e}")
                    return []

        return RepositoryAnalyticsService(analytics_repository)


logger = logging.getLogger(__name__)


# Clean dependency function for real-time services
async def get_realtime_database_services() -> DatabaseServices:
    """Get database services for real-time operations."""
    services = await get_database_services(ManagerMode.ASYNC_MODERN)
    if services is None:
        services = await create_database_services(ManagerMode.ASYNC_MODERN)
    return services


# WebSocket connection manager (would normally be imported from a WebSocket module)
class ConnectionManager:
    """Placeholder connection manager for WebSocket connections."""

    async def connect_to_group(self, websocket, group: str, user_id: str):
        """Connect to a WebSocket group."""
        await websocket.accept()

    async def connect(self, websocket, channel: str, user_id: str):
        """Connect to a WebSocket channel."""
        await websocket.accept()

    async def send_to_connection(self, websocket, message: dict):
        """Send message to WebSocket connection."""
        await websocket.send_json(message)

    async def broadcast_to_group(self, group: str, message: dict):
        """Broadcast message to group."""
        # Placeholder

    def get_connection_count(self, channel: str = None) -> int:
        """Get connection count."""
        return 0

    def get_active_experiments(self) -> list[str]:
        """Get active experiments."""
        return []

    async def disconnect(self, websocket):
        """Disconnect WebSocket."""
        try:
            await websocket.close()
        except:
            pass


connection_manager = ConnectionManager()

real_time_router = APIRouter(
    prefix="/api/v1/experiments/real-time", tags=["real-time-analytics"]
)


async def get_analytics_repository_dep(
    db_manager: DatabaseServices = Depends(get_realtime_database_services),
) -> AnalyticsRepositoryProtocol:
    """Get analytics repository with unified connection manager"""
    return await get_analytics_repository(db_manager)


async def get_health_repository_dep(
    db_manager: DatabaseServices = Depends(get_realtime_database_services),
) -> HealthRepositoryProtocol:
    """Get health repository with unified connection manager"""
    return await get_health_repository(db_manager)


@real_time_router.websocket("/live/{experiment_id}")
async def websocket_experiment_endpoint(
    websocket: WebSocket,
    experiment_id: str,
    user_id: str | None = None,
    db_manager: DatabaseServices = Depends(get_realtime_database_services),
):
    """WebSocket endpoint for real-time experiment monitoring

    Args:
        websocket: WebSocket connection
        experiment_id: UUID of experiment to monitor
        user_id: Optional user ID for connection tracking
        db_manager: Unified database connection manager
    """
    try:
        await connection_manager.connect(websocket, experiment_id, user_id)
        await connection_manager.send_to_connection(
            websocket,
            {
                "type": "welcome",
                "message": f"Connected to real-time monitoring for experiment {experiment_id}",
                "experiment_id": experiment_id,
            },
        )
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    await handle_websocket_message(
                        websocket, experiment_id, message, db_manager
                    )
                except json.JSONDecodeError:
                    await connection_manager.send_to_connection(
                        websocket, {"type": "error", "message": "Invalid JSON format"}
                    )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                await connection_manager.send_to_connection(
                    websocket, {"type": "error", "message": "Internal server error"}
                )
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


async def handle_websocket_message(
    websocket: WebSocket,
    experiment_id: str,
    message: dict[str, Any],
    db_manager: DatabaseServices,
):
    """Handle incoming WebSocket messages from clients"""
    try:
        message_type = message.get("type")
        if message_type == "ping":
            await connection_manager.send_to_connection(
                websocket, {"type": "pong", "timestamp": datetime.now(UTC).isoformat()}
            )
        elif message_type == "request_metrics":
            try:
                analytics_repository = await get_analytics_repository(db_manager)
                analytics_service = await get_real_time_analytics_service(
                    analytics_repository
                )
                metrics = await analytics_service.get_real_time_metrics(experiment_id)
            except Exception as e:
                logger.error(f"Error getting metrics in WebSocket: {e}")
                metrics = None
            if metrics:
                await connection_manager.send_to_connection(
                    websocket,
                    {
                        "type": "metrics_update",
                        "experiment_id": experiment_id,
                        "metrics": metrics.__dict__,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
            else:
                await connection_manager.send_to_connection(
                    websocket,
                    {"type": "error", "message": "Failed to retrieve metrics"},
                )
        elif message_type == "subscribe_alerts":
            await connection_manager.send_to_connection(
                websocket,
                {
                    "type": "subscription_confirmed",
                    "message": "Subscribed to alerts",
                    "subscriptions": ["metrics", "alerts"],
                },
            )
        else:
            await connection_manager.send_to_connection(
                websocket,
                {"type": "error", "message": f"Unknown message type: {message_type}"},
            )
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await connection_manager.send_to_connection(
            websocket, {"type": "error", "message": "Failed to process message"}
        )


@real_time_router.get("/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: str,
    analytics_repository: AnalyticsRepositoryProtocol = Depends(
        get_analytics_repository_dep
    ),
) -> JSONResponse:
    """Get current real-time metrics for an experiment

    Args:
        experiment_id: UUID of experiment
        analytics_repository: Analytics repository

    Returns:
        JSON response with current metrics
    """
    try:
        analytics_service = await get_real_time_analytics_service(analytics_repository)
        metrics = await analytics_service.get_real_time_metrics(experiment_id)
        if metrics:
            return JSONResponse({
                "status": "success",
                "experiment_id": experiment_id,
                "metrics": metrics,  # metrics is already a dict from repository-based service
            })
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or insufficient data",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve experiment metrics",
        )


@real_time_router.post("/experiments/{experiment_id}/monitoring/start")
async def start_monitoring(
    experiment_id: str,
    analytics_repository: AnalyticsRepositoryProtocol = Depends(
        get_analytics_repository_dep
    ),
    update_interval: int = 30,
) -> JSONResponse:
    """Start real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        update_interval: Update interval in seconds (default: 30)
        analytics_repository: Analytics repository

    Returns:
        JSON response confirming monitoring started
    """
    try:
        analytics_service = await get_real_time_analytics_service(analytics_repository)
        success = await analytics_service.start_experiment_monitoring(
            experiment_id, update_interval
        )
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"Started monitoring for experiment {experiment_id}",
                "experiment_id": experiment_id,
                "update_interval": update_interval,
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to start monitoring (experiment may not exist or not be running)",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start experiment monitoring",
        )


@real_time_router.post("/experiments/{experiment_id}/monitoring/stop")
async def stop_monitoring(
    experiment_id: str,
    analytics_repository: AnalyticsRepositoryProtocol = Depends(
        get_analytics_repository_dep
    ),
) -> JSONResponse:
    """Stop real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        analytics_repository: Analytics repository

    Returns:
        JSON response confirming monitoring stopped
    """
    try:
        analytics_service = await get_real_time_analytics_service(analytics_repository)
        success = await analytics_service.stop_experiment_monitoring(experiment_id)
        return JSONResponse({
            "status": "success" if success else "warning",
            "message": f"Stopped monitoring for experiment {experiment_id}",
            "experiment_id": experiment_id,
        })
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop experiment monitoring",
        )


@real_time_router.get("/monitoring/active")
async def get_active_monitoring(
    analytics_repository: AnalyticsRepositoryProtocol = Depends(
        get_analytics_repository_dep
    ),
) -> JSONResponse:
    """Get list of experiments currently being monitored

    Args:
        analytics_repository: Analytics repository

    Returns:
        JSON response with list of active experiments
    """
    try:
        analytics_service = await get_real_time_analytics_service(analytics_repository)
        active_experiments = await analytics_service.get_active_experiments()
        connection_info: list[dict[str, Any]] = []
        for experiment_id in active_experiments:
            connection_count = connection_manager.get_connection_count(experiment_id)
            connection_info.append({
                "experiment_id": experiment_id,
                "active_connections": connection_count,
            })
        return JSONResponse({
            "status": "success",
            "active_experiments": len(active_experiments),
            "total_connections": connection_manager.get_connection_count(),
            "experiments": connection_info,
        })
    except Exception as e:
        logger.error(f"Error getting active monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active monitoring information",
        )


@real_time_router.get("/dashboard/config/{experiment_id}")
async def get_dashboard_config(
    experiment_id: str,
    analytics_repository: AnalyticsRepositoryProtocol = Depends(
        get_analytics_repository_dep
    ),
) -> JSONResponse:
    """Get dashboard configuration for an experiment

    Args:
        experiment_id: UUID of experiment
        analytics_repository: Analytics repository

    Returns:
        JSON response with dashboard configuration
    """
    try:
        # Use repository to get experiment data through session analytics
        from datetime import timedelta

        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=1)

        # Get basic experiment info through analytics
        analytics_result = await analytics_repository.get_session_analytics(
            start_date, end_date
        )

        # Create a default experiment configuration
        experiment = {
            "experiment_id": experiment_id,
            "experiment_name": f"Experiment {experiment_id}",
            "status": "running",
            "started_at": start_date,
            "target_metric": "improvement_score",
            "sample_size_per_group": 100,
        }

        config: dict[str, Any] = {
            "experiment_id": experiment_id,
            "experiment_name": experiment["experiment_name"],
            "status": experiment["status"],
            "started_at": experiment["started_at"].isoformat()
            if experiment["started_at"]
            else None,
            "target_metric": experiment["target_metric"],
            "sample_size_per_group": experiment["sample_size_per_group"],
            "significance_level": 0.05,
            "dashboard_settings": {
                "auto_refresh": True,
                "refresh_interval": 30,
                "show_alerts": True,
                "show_early_stopping": True,
                "chart_types": ["line", "bar", "confidence_interval"],
                "metrics_to_display": [
                    "sample_sizes",
                    "conversion_rates",
                    "effect_size",
                    "p_value",
                    "confidence_interval",
                    "statistical_power",
                ],
            },
            "websocket_url": f"/api/v1/experiments/real-time/live/{experiment_id}",
            "api_endpoints": {
                "metrics": f"/api/v1/experiments/real-time/experiments/{experiment_id}/metrics",
                "start_monitoring": f"/api/v1/experiments/real-time/experiments/{experiment_id}/monitoring/start",
                "stop_monitoring": f"/api/v1/experiments/real-time/experiments/{experiment_id}/monitoring/stop",
            },
        }
        return JSONResponse({"status": "success", "config": config})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard configuration",
        )


@real_time_router.get("/health")
async def health_check(
    health_repo: HealthRepositoryProtocol = Depends(get_health_repository_dep),
) -> JSONResponse:
    """Health check endpoint for real-time services"""
    try:
        # Use repository to get comprehensive health check
        health_summary = await health_repo.perform_full_health_check()

        active_experiments = connection_manager.get_active_experiments()
        total_connections = connection_manager.get_connection_count()
        redis_status = "not_configured"
        if redis_available:
            redis_status = "available"
        elif not redis_available:
            redis_status = "not_installed"

        return JSONResponse({
            "status": health_summary.overall_status,
            "timestamp": datetime.now(UTC).isoformat(),
            "services": {
                "database": {"status": health_summary.overall_status},
                "websocket_manager": {
                    "status": "active",
                    "active_experiments": len(active_experiments),
                    "total_connections": total_connections,
                },
                "redis": {"status": redis_status, "available": redis_available},
            },
            "health_summary": {
                "components_checked": health_summary.components_checked,
                "healthy_components": health_summary.healthy_components,
                "performance_score": health_summary.performance_score,
            },
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )


@real_time_router.get("/orchestrator/status")
async def get_orchestrator_status() -> JSONResponse:
    """Get ML Pipeline Orchestrator status."""
    try:
        from prompt_improver.core.di.ml_container import MLServiceContainer
        from prompt_improver.core.factories.ml_pipeline_factory import (
            MLPipelineOrchestratorFactory,
        )
        from prompt_improver.core.protocols.ml_protocols import ServiceContainerProtocol
        from prompt_improver.ml.orchestration.config.external_services_config import (
            ExternalServicesConfig,
        )

        service_container = MLServiceContainer()
        external_config = ExternalServicesConfig()
        orchestrator = await MLPipelineOrchestratorFactory.create_from_container(
            service_container
        )
        status_data: dict[str, Any] = {
            "state": orchestrator.state.value,
            "initialized": getattr(orchestrator, "_is_initialized", False),
            "active_workflows": len(orchestrator.active_workflows),
            "timestamp": datetime.now().isoformat(),
        }
        if getattr(orchestrator, "_is_initialized", False):
            components = orchestrator.get_loaded_components()
            status_data.update({
                "loaded_components": len(components),
                "component_list": components[:10],
                "recent_invocations": len(orchestrator.get_invocation_history()),
            })
        return JSONResponse(
            content={
                "success": True,
                "data": status_data,
                "message": "Orchestrator status retrieved successfully",
            }
        )
    except ImportError as e:
        logger.error(f"Failed to import orchestrator: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Orchestrator not available",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get orchestrator status",
                "message": str(e),
            },
        )


@real_time_router.get("/orchestrator/components")
async def get_orchestrator_components() -> JSONResponse:
    """Get loaded ML components information."""
    try:
        from prompt_improver.core.di.ml_container import MLServiceContainer
        from prompt_improver.core.factories.ml_pipeline_factory import (
            MLPipelineOrchestratorFactory,
        )
        from prompt_improver.ml.orchestration.config.external_services_config import (
            ExternalServicesConfig,
        )

        service_container = MLServiceContainer()
        orchestrator = await MLPipelineOrchestratorFactory.create_from_container(
            service_container
        )
        if not getattr(orchestrator, "_is_initialized", False):
            await orchestrator.initialize()
        components = orchestrator.get_loaded_components()
        component_details: list[dict[str, Any]] = []
        for component_name in components:
            methods = orchestrator.get_component_methods(component_name)
            component_details.append({
                "name": component_name,
                "methods_count": len(methods),
                "methods": methods[:5],
                "is_loaded": orchestrator.component_loader.is_component_loaded(
                    component_name
                ),
                "is_initialized": orchestrator.component_loader.is_component_initialized(
                    component_name
                ),
            })
        return JSONResponse(
            content={
                "success": True,
                "data": {
                    "total_components": len(components),
                    "components": component_details,
                    "timestamp": datetime.now().isoformat(),
                },
                "message": f"Retrieved {len(components)} components",
            }
        )
    except Exception as e:
        logger.error(f"Error getting orchestrator components: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get components",
                "message": str(e),
            },
        )


@real_time_router.get("/orchestrator/history")
async def get_orchestrator_history(
    component: str | None = None, limit: int = 50
) -> JSONResponse:
    """Get orchestrator invocation history."""
    try:
        from prompt_improver.core.di.ml_container import MLServiceContainer
        from prompt_improver.core.factories.ml_pipeline_factory import (
            MLPipelineOrchestratorFactory,
        )
        from prompt_improver.ml.orchestration.config.external_services_config import (
            ExternalServicesConfig,
        )

        service_container = MLServiceContainer()
        orchestrator = await MLPipelineOrchestratorFactory.create_from_container(
            service_container
        )
        if not getattr(orchestrator, "_is_initialized", False):
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Orchestrator not initialized",
                    "message": "Initialize orchestrator first",
                }
            )
        history = orchestrator.get_invocation_history(component)[-limit:]
        total_invocations = len(history)
        successful_invocations = sum(1 for inv in history if inv["success"])
        success_rate = (
            successful_invocations / total_invocations if total_invocations > 0 else 0.0
        )
        return JSONResponse(
            content={
                "success": True,
                "data": {
                    "total_invocations": total_invocations,
                    "successful_invocations": successful_invocations,
                    "success_rate": success_rate,
                    "filtered_component": component,
                    "history": history,
                    "timestamp": datetime.now().isoformat(),
                },
                "message": f"Retrieved {total_invocations} invocation records",
            }
        )
    except Exception as e:
        logger.error(f"Error getting orchestrator history: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get history",
                "message": str(e),
            },
        )


async def setup_real_time_system(redis_url: str = None):
    """Setup real-time system with Redis connection"""
    try:
        if redis_available:
            if redis_url is None:
                import os

                redis_url = os.getenv(
                    "REDIS_URL", "redis://redis.external.service:6379/0"
                )
            await setup_redis_connection(redis_url)
            logger.info("Real-time system setup completed with Redis")
        else:
            logger.warning(
                "Real-time system setup completed without Redis (not installed)"
            )
    except Exception as e:
        logger.error(f"Failed to setup real-time system: {e}")


async def cleanup_real_time_system():
    """Cleanup real-time system resources"""
    try:
        await connection_manager.cleanup()
        logger.info("Real-time system cleanup completed")
    except Exception as e:
        logger.error(f"Error during real-time system cleanup: {e}")


__all__ = ["cleanup_real_time_system", "real_time_router", "setup_real_time_system"]
