"""Real-time API Endpoints for A/B Testing Analytics
Provides WebSocket and REST endpoints for live experiment monitoring
"""

import logging
from datetime import datetime, UTC
from typing import Any, Optional

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

# Make Redis import optional with proper error handling
try:
    import coredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    coredis = None

from ..database import get_session
from ..database.models import ABExperiment
from ..utils.websocket_manager import connection_manager, setup_redis_connection

logger = logging.getLogger(__name__)

# Create router for real-time endpoints
real_time_router = APIRouter(
    prefix="/api/v1/experiments/real-time", tags=["real-time-analytics"]
)

@real_time_router.websocket("/live/{experiment_id}")
async def websocket_experiment_endpoint(
    websocket: WebSocket, experiment_id: str, user_id: str | None = None
):
    """WebSocket endpoint for real-time experiment monitoring

    Args:
        websocket: WebSocket connection
        experiment_id: UUID of experiment to monitor
        user_id: Optional user ID for connection tracking
    """
    try:
        # Accept WebSocket connection
        await connection_manager.connect(websocket, experiment_id, user_id)

        # Send welcome message
        await connection_manager.send_to_connection(
            websocket,
            {
                "type": "welcome",
                "message": f"Connected to real-time monitoring for experiment {experiment_id}",
                "experiment_id": experiment_id,
            },
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()

                # Parse and handle client messages
                try:
                    import json

                    message = json.loads(data)
                    await handle_websocket_message(websocket, experiment_id, message)
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
        # Clean up connection
        await connection_manager.disconnect(websocket)

async def handle_websocket_message(
    websocket: WebSocket, experiment_id: str, message: dict[str, Any]
):
    """Handle incoming WebSocket messages from clients"""
    try:
        message_type = message.get("type")

        if message_type == "ping":
            # Respond to ping with pong
            await connection_manager.send_to_connection(
                websocket, {"type": "pong", "timestamp": datetime.now(UTC).isoformat()}
            )

        elif message_type == "request_metrics":
            # Send current metrics using proper async session pattern
            async_session_factory = get_async_session_factory()
            async with async_session_factory() as db_session:
                analytics_service = await get_real_time_analytics_service(db_session)
                metrics = await analytics_service.get_real_time_metrics(experiment_id)

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
            # Client wants to subscribe to alerts (already subscribed by default)
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
    experiment_id: str, db_session: AsyncSession = Depends(get_session)
) -> JSONResponse:
    """Get current real-time metrics for an experiment

    Args:
        experiment_id: UUID of experiment
        db_session: Database session

    Returns:
        JSON response with current metrics
    """
    try:
        analytics_service = await get_real_time_analytics_service(db_session)
        metrics = await analytics_service.get_real_time_metrics(experiment_id)

        if metrics:
            return JSONResponse({
                "status": "success",
                "experiment_id": experiment_id,
                "metrics": {
                    "experiment_id": metrics.experiment_id,
                    "timestamp": metrics.timestamp.isoformat(),
                    "sample_sizes": {
                        "control": metrics.control_sample_size,
                        "treatment": metrics.treatment_sample_size,
                        "total": metrics.total_sample_size,
                    },
                    "means": {
                        "control": metrics.control_mean,
                        "treatment": metrics.treatment_mean,
                    },
                    "statistical_analysis": {
                        "effect_size": metrics.effect_size,
                        "p_value": metrics.p_value,
                        "confidence_interval": [
                            metrics.confidence_interval_lower,
                            metrics.confidence_interval_upper,
                        ],
                        "statistical_significance": metrics.statistical_significance,
                        "statistical_power": metrics.statistical_power,
                    },
                    "progress": {
                        "completion_percentage": metrics.completion_percentage,
                        "estimated_days_remaining": metrics.estimated_days_remaining,
                    },
                    "quality": {
                        "balance_ratio": metrics.balance_ratio,
                        "data_quality_score": metrics.data_quality_score,
                    },
                    "early_stopping": {
                        "recommendation": metrics.early_stopping_recommendation,
                        "confidence": metrics.early_stopping_confidence,
                    },
                },
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
    experiment_id: str, db_session: AsyncSession = Depends(get_session), update_interval: int = 30
) -> JSONResponse:
    """Start real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        update_interval: Update interval in seconds (default: 30)
        db_session: Database session

    Returns:
        JSON response confirming monitoring started
    """
    try:
        analytics_service = await get_real_time_analytics_service(db_session)
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
async def stop_monitoring(experiment_id: str, db_session: AsyncSession = Depends(get_session)) -> JSONResponse:
    """Stop real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        db_session: Database session

    Returns:
        JSON response confirming monitoring stopped
    """
    try:
        analytics_service = await get_real_time_analytics_service(db_session)
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
async def get_active_monitoring(db_session: AsyncSession = Depends(get_session)) -> JSONResponse:
    """Get list of experiments currently being monitored

    Args:
        db_session: Database session

    Returns:
        JSON response with list of active experiments
    """
    try:
        analytics_service = await get_real_time_analytics_service(db_session)
        active_experiments = await analytics_service.get_active_experiments()

        # Get connection counts
        connection_info = []
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
    experiment_id: str, db_session: AsyncSession = Depends(get_session)
) -> JSONResponse:
    """Get dashboard configuration for an experiment

    Args:
        experiment_id: UUID of experiment
        db_session: Database session

    Returns:
        JSON response with dashboard configuration
    """
    try:
        # Get experiment details for dashboard configuration
        stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
        result = await db_session.execute(stmt)
        experiment = result.scalar_one_or_none()

        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
            )

        # Build dashboard configuration
        config = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.experiment_name,
            "status": experiment.status,
            "started_at": experiment.started_at.isoformat()
            if experiment.started_at
            else None,
            "target_metric": experiment.target_metric,
            "sample_size_per_group": experiment.sample_size_per_group,
            "significance_level": 0.05,  # Default or from experiment config
            "dashboard_settings": {
                "auto_refresh": True,
                "refresh_interval": 30,  # seconds
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
async def health_check() -> JSONResponse:
    """Health check endpoint for real-time services"""
    try:
        # Check WebSocket manager
        active_experiments = connection_manager.get_active_experiments()
        total_connections = connection_manager.get_connection_count()

        # Check Redis connection if available
        redis_status = "not_configured"
        if REDIS_AVAILABLE and connection_manager.redis_client:
            try:
                await connection_manager.redis_client.ping()
                redis_status = "connected"
            except Exception:
                redis_status = "disconnected"
        elif not REDIS_AVAILABLE:
            redis_status = "not_installed"

        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "services": {
                "websocket_manager": {
                    "status": "active",
                    "active_experiments": len(active_experiments),
                    "total_connections": total_connections,
                },
                "redis": {"status": redis_status, "available": REDIS_AVAILABLE},
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

# ML Pipeline Orchestrator Integration (Phase 6)
@real_time_router.get("/orchestrator/status")
async def get_orchestrator_status() -> JSONResponse:
    """Get ML Pipeline Orchestrator status."""
    try:
        # Import orchestrator
        from ..ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from ..ml.orchestration.config.orchestrator_config import OrchestratorConfig

        # Initialize orchestrator (lightweight check)
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)

        # Get status information
        status_data = {
            "state": orchestrator.state.value,
            "initialized": orchestrator._is_initialized,
            "active_workflows": len(orchestrator.active_workflows),
            "timestamp": datetime.now().isoformat()
        }

        # If initialized, get component information
        if orchestrator._is_initialized:
            components = orchestrator.get_loaded_components()
            status_data.update({
                "loaded_components": len(components),
                "component_list": components[:10],  # First 10 components
                "recent_invocations": len(orchestrator.get_invocation_history())
            })

        return JSONResponse(content={
            "success": True,
            "data": status_data,
            "message": "Orchestrator status retrieved successfully"
        })

    except ImportError as e:
        logger.error(f"Failed to import orchestrator: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Orchestrator not available",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get orchestrator status",
                "message": str(e)
            }
        )

@real_time_router.get("/orchestrator/components")
async def get_orchestrator_components() -> JSONResponse:
    """Get loaded ML components information."""
    try:
        from ..ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from ..ml.orchestration.config.orchestrator_config import OrchestratorConfig

        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)

        if not orchestrator._is_initialized:
            await orchestrator.initialize()

        components = orchestrator.get_loaded_components()
        component_details = []

        for component_name in components:
            methods = orchestrator.get_component_methods(component_name)
            component_details.append({
                "name": component_name,
                "methods_count": len(methods),
                "methods": methods[:5],  # First 5 methods
                "is_loaded": orchestrator.component_loader.is_component_loaded(component_name),
                "is_initialized": orchestrator.component_loader.is_component_initialized(component_name)
            })

        return JSONResponse(content={
            "success": True,
            "data": {
                "total_components": len(components),
                "components": component_details,
                "timestamp": datetime.now().isoformat()
            },
            "message": f"Retrieved {len(components)} components"
        })

    except Exception as e:
        logger.error(f"Error getting orchestrator components: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get components",
                "message": str(e)
            }
        )

@real_time_router.get("/orchestrator/history")
async def get_orchestrator_history(
    component: Optional[str] = None,
    limit: int = 50
) -> JSONResponse:
    """Get orchestrator invocation history."""
    try:
        from ..ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from ..ml.orchestration.config.orchestrator_config import OrchestratorConfig

        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)

        if not orchestrator._is_initialized:
            return JSONResponse(content={
                "success": False,
                "error": "Orchestrator not initialized",
                "message": "Initialize orchestrator first"
            })

        history = orchestrator.get_invocation_history(component)[-limit:]

        # Calculate success rate
        total_invocations = len(history)
        successful_invocations = sum(1 for inv in history if inv["success"])
        success_rate = successful_invocations / total_invocations if total_invocations > 0 else 0.0

        return JSONResponse(content={
            "success": True,
            "data": {
                "total_invocations": total_invocations,
                "successful_invocations": successful_invocations,
                "success_rate": success_rate,
                "filtered_component": component,
                "history": history,
                "timestamp": datetime.now().isoformat()
            },
            "message": f"Retrieved {total_invocations} invocation records"
        })

    except Exception as e:
        logger.error(f"Error getting orchestrator history: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get history",
                "message": str(e)
            }
        )

# Startup and shutdown events for the real-time system
async def setup_real_time_system(redis_url: str = "redis://localhost:6379"):
    """Setup real-time system with Redis connection"""
    try:
        if REDIS_AVAILABLE:
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

# Export the router
__all__ = ["cleanup_real_time_system", "real_time_router", "setup_real_time_system"]
