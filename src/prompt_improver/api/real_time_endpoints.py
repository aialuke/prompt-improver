"""Real-time API Endpoints for A/B Testing Analytics
Provides WebSocket and REST endpoints for live experiment monitoring
"""

import json
import logging
from datetime import datetime, UTC
from typing import Any, Optional, Dict, List

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
redis_available = True
try:
    import coredis
except ImportError:
    redis_available = False
    coredis = None

from ..database import get_unified_manager_async_modern, UnifiedConnectionManager
from ..database.models import ABExperiment
from ..database.analytics_query_interface import AnalyticsQueryInterface
from ..utils.websocket_manager import connection_manager, setup_redis_connection

# Import real-time analytics service with graceful fallback
analytics_available = True
try:
    from ..performance.analytics.real_time_analytics import get_real_time_analytics_service
except ImportError:
    analytics_available = False

    # Create a functional analytics service using existing APES infrastructure
    async def get_real_time_analytics_service(db_session: AsyncSession) -> Any:
        """Functional analytics service using AnalyticsQueryInterface and ABExperiment model"""

        class FunctionalAnalyticsService:
            """Real analytics service implementation using existing APES database infrastructure"""

            def __init__(self, db_session: AsyncSession):
                self.db_session = db_session
                self.analytics = AnalyticsQueryInterface(db_session)
                self.logger = logging.getLogger(__name__)

            async def get_real_time_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
                """Get real-time metrics for an experiment using database queries"""
                try:
                    # Query the ABExperiment table for the specific experiment
                    # Convert experiment_id to int if needed (ABExperiment.id is typically int)
                    try:
                        exp_id = int(experiment_id)
                    except ValueError:
                        # If experiment_id is not numeric, try string comparison
                        query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)
                    else:
                        query = select(ABExperiment).where(ABExperiment.id == exp_id)
                    result = await self.db_session.execute(query)
                    experiment = result.scalar_one_or_none()

                    if not experiment:
                        self.logger.warning(f"Experiment {experiment_id} not found")
                        return None

                    # Calculate metrics based on experiment data
                    current_time = datetime.now(UTC)
                    duration_hours = 0.0
                    if experiment.started_at:
                        duration_hours = (current_time - experiment.started_at).total_seconds() / 3600

                    # Build real metrics from experiment data
                    metrics = {
                        "experiment_id": experiment_id,
                        "experiment_name": experiment.experiment_name,
                        "status": experiment.status,
                        "current_sample_size": experiment.current_sample_size,
                        "target_sample_size": experiment.sample_size_per_group * 2 if experiment.sample_size_per_group else 0,
                        "completion_percentage": self._calculate_completion_percentage(experiment),
                        "duration_hours": duration_hours,
                        "target_metric": experiment.target_metric,
                        "significance_threshold": experiment.significance_threshold,
                        "results": experiment.results or {},
                        "metadata": experiment.experiment_metadata or {},
                        "last_updated": current_time.isoformat()
                    }

                    # Add statistical analysis if results are available
                    if experiment.results:
                        metrics.update(self._extract_statistical_metrics(experiment.results))

                    return metrics

                except Exception as e:
                    self.logger.error(f"Error getting real-time metrics for experiment {experiment_id}: {e}")
                    return None

            async def start_experiment_monitoring(self, experiment_id: str, update_interval: int) -> bool:
                """Start monitoring for an experiment by updating its configuration"""
                try:
                    # Convert experiment_id to appropriate type
                    try:
                        exp_id = int(experiment_id)
                        query = select(ABExperiment).where(ABExperiment.id == exp_id)
                    except ValueError:
                        query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)

                    result = await self.db_session.execute(query)
                    experiment = result.scalar_one_or_none()

                    if not experiment:
                        self.logger.warning(f"Cannot start monitoring: experiment {experiment_id} not found")
                        return False

                    # Update experiment metadata to include monitoring configuration
                    metadata = experiment.experiment_metadata or {}
                    metadata.update({
                        "monitoring_enabled": True,
                        "update_interval_seconds": update_interval,
                        "monitoring_started_at": datetime.now(UTC).isoformat(),
                        "last_monitoring_update": datetime.now(UTC).isoformat()
                    })

                    # Update the experiment in the database
                    experiment.experiment_metadata = metadata
                    await self.db_session.commit()

                    self.logger.info(f"Started monitoring for experiment {experiment_id} with {update_interval}s interval")
                    return True

                except Exception as e:
                    self.logger.error(f"Error starting monitoring for experiment {experiment_id}: {e}")
                    await self.db_session.rollback()
                    return False

            async def stop_experiment_monitoring(self, experiment_id: str) -> bool:
                """Stop monitoring for an experiment by updating its configuration"""
                try:
                    # Convert experiment_id to appropriate type
                    try:
                        exp_id = int(experiment_id)
                        query = select(ABExperiment).where(ABExperiment.id == exp_id)
                    except ValueError:
                        query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)

                    result = await self.db_session.execute(query)
                    experiment = result.scalar_one_or_none()

                    if not experiment:
                        self.logger.warning(f"Cannot stop monitoring: experiment {experiment_id} not found")
                        return False

                    # Update experiment metadata to disable monitoring
                    metadata = experiment.experiment_metadata or {}
                    metadata.update({
                        "monitoring_enabled": False,
                        "monitoring_stopped_at": datetime.now(UTC).isoformat(),
                        "last_monitoring_update": datetime.now(UTC).isoformat()
                    })

                    # Update the experiment in the database
                    experiment.experiment_metadata = metadata
                    await self.db_session.commit()

                    self.logger.info(f"Stopped monitoring for experiment {experiment_id}")
                    return True

                except Exception as e:
                    self.logger.error(f"Error stopping monitoring for experiment {experiment_id}: {e}")
                    await self.db_session.rollback()
                    return False

            async def get_active_experiments(self) -> List[str]:
                """Get list of active experiment IDs from the database"""
                try:
                    # Query for experiments with active statuses (using simple status check)
                    query = select(ABExperiment).where(ABExperiment.status == "running")
                    result = await self.db_session.execute(query)
                    experiments = result.scalars().all()

                    # Return list of experiment IDs (convert to string for consistency)
                    active_experiment_ids = [str(exp.id) for exp in experiments]

                    self.logger.info(f"Found {len(active_experiment_ids)} active experiments")
                    return active_experiment_ids

                except Exception as e:
                    self.logger.error(f"Error getting active experiments: {e}")
                    return []

            def _calculate_completion_percentage(self, experiment: ABExperiment) -> float:
                """Calculate completion percentage based on sample sizes"""
                if not experiment.sample_size_per_group:
                    return 0.0

                target_total = experiment.sample_size_per_group * 2  # Control + treatment
                current_total = experiment.current_sample_size

                if target_total <= 0:
                    return 0.0

                percentage = min((current_total / target_total) * 100, 100.0)
                return round(percentage, 2)

            def _extract_statistical_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
                """Extract statistical metrics from experiment results"""
                statistical_metrics = {}

                # Extract common statistical fields if they exist
                if "p_value" in results:
                    statistical_metrics["p_value"] = results["p_value"]
                if "effect_size" in results:
                    statistical_metrics["effect_size"] = results["effect_size"]
                if "confidence_interval" in results:
                    statistical_metrics["confidence_interval"] = results["confidence_interval"]
                if "statistical_significance" in results:
                    statistical_metrics["statistical_significance"] = results["statistical_significance"]
                if "control_mean" in results:
                    statistical_metrics["control_mean"] = results["control_mean"]
                if "treatment_mean" in results:
                    statistical_metrics["treatment_mean"] = results["treatment_mean"]

                return statistical_metrics

        return FunctionalAnalyticsService(db_session)

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
            # Send current metrics using UnifiedConnectionManager
            db_manager = get_unified_manager_async_modern()
            try:
                async with db_manager.get_async_session() as session:
                    analytics_service = await get_real_time_analytics_service(session)
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
    experiment_id: str, db_manager: UnifiedConnectionManager = Depends(get_unified_manager_async_modern)
) -> JSONResponse:
    """Get current real-time metrics for an experiment

    Args:
        experiment_id: UUID of experiment
        db_manager: Database manager

    Returns:
        JSON response with current metrics
    """
    async with db_manager.get_async_session() as session:
        try:
            analytics_service = await get_real_time_analytics_service(session)
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
    experiment_id: str, db_manager: UnifiedConnectionManager = Depends(get_unified_manager_async_modern), update_interval: int = 30
) -> JSONResponse:
    """Start real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        update_interval: Update interval in seconds (default: 30)
        db_manager: Database manager

    Returns:
        JSON response confirming monitoring started
    """
    async with db_manager.get_async_session() as session:
        try:
            analytics_service = await get_real_time_analytics_service(session)
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
async def stop_monitoring(experiment_id: str, db_manager: UnifiedConnectionManager = Depends(get_unified_manager_async_modern)) -> JSONResponse:
    """Stop real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        db_manager: Database manager

    Returns:
        JSON response confirming monitoring stopped
    """
    async with db_manager.get_async_session() as session:
        try:
            analytics_service = await get_real_time_analytics_service(session)
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
async def get_active_monitoring(db_manager: UnifiedConnectionManager = Depends(get_unified_manager_async_modern)) -> JSONResponse:
    """Get list of experiments currently being monitored

    Args:
        db_manager: Database manager

    Returns:
        JSON response with list of active experiments
    """
    async with db_manager.get_async_session() as session:
        try:
            analytics_service = await get_real_time_analytics_service(session)
            active_experiments = await analytics_service.get_active_experiments()

        # Get connection counts
        connection_info: List[Dict[str, Any]] = []
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
    experiment_id: str, db_manager: UnifiedConnectionManager = Depends(get_unified_manager_async_modern)
) -> JSONResponse:
    """Get dashboard configuration for an experiment

    Args:
        experiment_id: UUID of experiment
        db_manager: Database manager

    Returns:
        JSON response with dashboard configuration
    """
    async with db_manager.get_async_session() as session:
        try:
            # Get experiment details for dashboard configuration
            stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
            result = await session.execute(stmt)
        experiment = result.scalar_one_or_none()

        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
            )

        # Build dashboard configuration
        config: Dict[str, Any] = {
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
        if redis_available and connection_manager.redis_client:
            try:
                await connection_manager.redis_client.ping()
                redis_status = "connected"
            except Exception:
                redis_status = "disconnected"
        elif not redis_available:
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
                "redis": {"status": redis_status, "available": redis_available},
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
        status_data: Dict[str, Any] = {
            "state": orchestrator.state.value,
            "initialized": getattr(orchestrator, '_is_initialized', False),
            "active_workflows": len(orchestrator.active_workflows),
            "timestamp": datetime.now().isoformat()
        }

        # If initialized, get component information
        if getattr(orchestrator, '_is_initialized', False):
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

        if not getattr(orchestrator, '_is_initialized', False):
            await orchestrator.initialize()

        components = orchestrator.get_loaded_components()
        component_details: List[Dict[str, Any]] = []

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

        if not getattr(orchestrator, '_is_initialized', False):
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
        if redis_available:
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
