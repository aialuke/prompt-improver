"""Real-time API Endpoints for A/B Testing Analytics
Provides WebSocket and REST endpoints for live experiment monitoring
"""
import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from prompt_improver.database import UnifiedConnectionManager, get_unified_manager_async_modern
from prompt_improver.database.analytics_query_interface import AnalyticsQueryInterface
from prompt_improver.database.models import ABExperiment
from prompt_improver.utils.websocket_manager import connection_manager, setup_redis_connection
redis_available = True
try:
    import coredis
except ImportError:
    redis_available = False
    coredis = None
analytics_available = True
try:
    from prompt_improver.performance.analytics.real_time_analytics import get_real_time_analytics_service
except ImportError:
    analytics_available = False

    async def get_real_time_analytics_service(db_session: AsyncSession) -> Any:
        """Functional analytics service using AnalyticsQueryInterface and ABExperiment model"""

        class FunctionalAnalyticsService:
            """Real analytics service implementation using existing APES database infrastructure"""

            def __init__(self, db_session: AsyncSession):
                self.db_session = db_session
                self.analytics = AnalyticsQueryInterface(db_session)
                self.logger = logging.getLogger(__name__)

            async def get_real_time_metrics(self, experiment_id: str) -> dict[str, Any] | None:
                """Get real-time metrics for an experiment using database queries"""
                try:
                    try:
                        exp_id = int(experiment_id)
                    except ValueError:
                        query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)
                    else:
                        query = select(ABExperiment).where(ABExperiment.id == exp_id)
                    result = await self.db_session.execute(query)
                    experiment = result.scalar_one_or_none()
                    if not experiment:
                        self.logger.warning('Experiment %s not found', experiment_id)
                        return None
                    current_time = datetime.now(UTC)
                    duration_hours = 0.0
                    if experiment.started_at:
                        duration_hours = (current_time - experiment.started_at).total_seconds() / 3600
                    metrics = {'experiment_id': experiment_id, 'experiment_name': experiment.experiment_name, 'status': experiment.status, 'current_sample_size': experiment.current_sample_size, 'target_sample_size': experiment.sample_size_per_group * 2 if experiment.sample_size_per_group else 0, 'completion_percentage': self._calculate_completion_percentage(experiment), 'duration_hours': duration_hours, 'target_metric': experiment.target_metric, 'significance_threshold': experiment.significance_threshold, 'results': experiment.results or {}, 'metadata': experiment.experiment_metadata or {}, 'last_updated': current_time.isoformat()}
                    if experiment.results:
                        metrics.update(self._extract_statistical_metrics(experiment.results))
                    return metrics
                except Exception as e:
                    self.logger.error('Error getting real-time metrics for experiment %s: %s', experiment_id, e)
                    return None

            async def start_experiment_monitoring(self, experiment_id: str, update_interval: int) -> bool:
                """Start monitoring for an experiment by updating its configuration"""
                try:
                    try:
                        exp_id = int(experiment_id)
                        query = select(ABExperiment).where(ABExperiment.id == exp_id)
                    except ValueError:
                        query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)
                    result = await self.db_session.execute(query)
                    experiment = result.scalar_one_or_none()
                    if not experiment:
                        self.logger.warning('Cannot start monitoring: experiment %s not found', experiment_id)
                        return False
                    metadata = experiment.experiment_metadata or {}
                    metadata.update({'monitoring_enabled': True, 'update_interval_seconds': update_interval, 'monitoring_started_at': datetime.now(UTC).isoformat(), 'last_monitoring_update': datetime.now(UTC).isoformat()})
                    experiment.experiment_metadata = metadata
                    await self.db_session.commit()
                    self.logger.info('Started monitoring for experiment %s with %ss interval', experiment_id, update_interval)
                    return True
                except Exception as e:
                    self.logger.error('Error starting monitoring for experiment %s: %s', experiment_id, e)
                    await self.db_session.rollback()
                    return False

            async def stop_experiment_monitoring(self, experiment_id: str) -> bool:
                """Stop monitoring for an experiment by updating its configuration"""
                try:
                    try:
                        exp_id = int(experiment_id)
                        query = select(ABExperiment).where(ABExperiment.id == exp_id)
                    except ValueError:
                        query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)
                    result = await self.db_session.execute(query)
                    experiment = result.scalar_one_or_none()
                    if not experiment:
                        self.logger.warning('Cannot stop monitoring: experiment %s not found', experiment_id)
                        return False
                    metadata = experiment.experiment_metadata or {}
                    metadata.update({'monitoring_enabled': False, 'monitoring_stopped_at': datetime.now(UTC).isoformat(), 'last_monitoring_update': datetime.now(UTC).isoformat()})
                    experiment.experiment_metadata = metadata
                    await self.db_session.commit()
                    self.logger.info('Stopped monitoring for experiment %s', experiment_id)
                    return True
                except Exception as e:
                    self.logger.error('Error stopping monitoring for experiment %s: %s', experiment_id, e)
                    await self.db_session.rollback()
                    return False

            async def get_active_experiments(self) -> list[str]:
                """Get list of active experiment IDs from the database"""
                try:
                    query = select(ABExperiment).where(ABExperiment.status == 'running')
                    result = await self.db_session.execute(query)
                    experiments = result.scalars().all()
                    active_experiment_ids = [str(exp.id) for exp in experiments]
                    self.logger.info('Found %s active experiments', len(active_experiment_ids))
                    return active_experiment_ids
                except Exception as e:
                    self.logger.error('Error getting active experiments: %s', e)
                    return []

            def _calculate_completion_percentage(self, experiment: ABExperiment) -> float:
                """Calculate completion percentage based on sample sizes"""
                if not experiment.sample_size_per_group:
                    return 0.0
                target_total = experiment.sample_size_per_group * 2
                current_total = experiment.current_sample_size
                if target_total <= 0:
                    return 0.0
                percentage = min(current_total / target_total * 100, 100.0)
                return round(percentage, 2)

            def _extract_statistical_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
                """Extract statistical metrics from experiment results"""
                statistical_metrics = {}
                if 'p_value' in results:
                    statistical_metrics['p_value'] = results['p_value']
                if 'effect_size' in results:
                    statistical_metrics['effect_size'] = results['effect_size']
                if 'confidence_interval' in results:
                    statistical_metrics['confidence_interval'] = results['confidence_interval']
                if 'statistical_significance' in results:
                    statistical_metrics['statistical_significance'] = results['statistical_significance']
                if 'control_mean' in results:
                    statistical_metrics['control_mean'] = results['control_mean']
                if 'treatment_mean' in results:
                    statistical_metrics['treatment_mean'] = results['treatment_mean']
                return statistical_metrics
        return FunctionalAnalyticsService(db_session)
logger = logging.getLogger(__name__)
real_time_router = APIRouter(prefix='/api/v1/experiments/real-time', tags=['real-time-analytics'])

@real_time_router.websocket('/live/{experiment_id}')
async def websocket_experiment_endpoint(websocket: WebSocket, experiment_id: str, user_id: str | None=None, db_manager: UnifiedConnectionManager=Depends(get_unified_manager_async_modern)):
    """WebSocket endpoint for real-time experiment monitoring

    Args:
        websocket: WebSocket connection
        experiment_id: UUID of experiment to monitor
        user_id: Optional user ID for connection tracking
        db_manager: Unified database connection manager
    """
    try:
        await connection_manager.connect(websocket, experiment_id, user_id)
        await connection_manager.send_to_connection(websocket, {'type': 'welcome', 'message': f'Connected to real-time monitoring for experiment {experiment_id}', 'experiment_id': experiment_id})
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    await handle_websocket_message(websocket, experiment_id, message, db_manager)
                except json.JSONDecodeError:
                    await connection_manager.send_to_connection(websocket, {'type': 'error', 'message': 'Invalid JSON format'})
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error('Error in WebSocket loop: %s', e)
                await connection_manager.send_to_connection(websocket, {'type': 'error', 'message': 'Internal server error'})
    except Exception as e:
        logger.error('WebSocket connection error: %s', e)
    finally:
        await connection_manager.disconnect(websocket)

async def handle_websocket_message(websocket: WebSocket, experiment_id: str, message: dict[str, Any], db_manager: UnifiedConnectionManager):
    """Handle incoming WebSocket messages from clients"""
    try:
        message_type = message.get('type')
        if message_type == 'ping':
            await connection_manager.send_to_connection(websocket, {'type': 'pong', 'timestamp': datetime.now(UTC).isoformat()})
        elif message_type == 'request_metrics':
            try:
                async with db_manager.get_async_session() as session:
                    analytics_service = await get_real_time_analytics_service(session)
                    metrics = await analytics_service.get_real_time_metrics(experiment_id)
            except Exception as e:
                logger.error('Error getting metrics in WebSocket: %s', e)
                metrics = None
            if metrics:
                await connection_manager.send_to_connection(websocket, {'type': 'metrics_update', 'experiment_id': experiment_id, 'metrics': metrics.__dict__, 'timestamp': datetime.now(UTC).isoformat()})
            else:
                await connection_manager.send_to_connection(websocket, {'type': 'error', 'message': 'Failed to retrieve metrics'})
        elif message_type == 'subscribe_alerts':
            await connection_manager.send_to_connection(websocket, {'type': 'subscription_confirmed', 'message': 'Subscribed to alerts', 'subscriptions': ['metrics', 'alerts']})
        else:
            await connection_manager.send_to_connection(websocket, {'type': 'error', 'message': f'Unknown message type: {message_type}'})
    except Exception as e:
        logger.error('Error handling WebSocket message: %s', e)
        await connection_manager.send_to_connection(websocket, {'type': 'error', 'message': 'Failed to process message'})

@real_time_router.get('/experiments/{experiment_id}/metrics')
async def get_experiment_metrics(experiment_id: str, db_manager: UnifiedConnectionManager=Depends(get_unified_manager_async_modern)) -> JSONResponse:
    """Get current real-time metrics for an experiment

    Args:
        experiment_id: UUID of experiment
        db_manager: Database manager

    Returns:
        JSON response with current metrics
    """
    try:
        async with db_manager.get_async_session() as session:
            analytics_service = await get_real_time_analytics_service(session)
            metrics = await analytics_service.get_real_time_metrics(experiment_id)
            if metrics:
                return JSONResponse({'status': 'success', 'experiment_id': experiment_id, 'metrics': {'experiment_id': metrics.experiment_id, 'timestamp': metrics.timestamp.isoformat(), 'sample_sizes': {'control': metrics.control_sample_size, 'treatment': metrics.treatment_sample_size, 'total': metrics.total_sample_size}, 'means': {'control': metrics.control_mean, 'treatment': metrics.treatment_mean}, 'statistical_analysis': {'effect_size': metrics.effect_size, 'p_value': metrics.p_value, 'confidence_interval': [metrics.confidence_interval_lower, metrics.confidence_interval_upper], 'statistical_significance': metrics.statistical_significance, 'statistical_power': metrics.statistical_power}, 'progress': {'completion_percentage': metrics.completion_percentage, 'estimated_days_remaining': metrics.estimated_days_remaining}, 'quality': {'balance_ratio': metrics.balance_ratio, 'data_quality_score': metrics.data_quality_score}, 'early_stopping': {'recommendation': metrics.early_stopping_recommendation, 'confidence': metrics.early_stopping_confidence}}})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Experiment not found or insufficient data')
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error getting experiment metrics: %s', e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to retrieve experiment metrics')

@real_time_router.post('/experiments/{experiment_id}/monitoring/start')
async def start_monitoring(experiment_id: str, db_manager: UnifiedConnectionManager=Depends(get_unified_manager_async_modern), update_interval: int=30) -> JSONResponse:
    """Start real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        update_interval: Update interval in seconds (default: 30)
        db_manager: Database manager

    Returns:
        JSON response confirming monitoring started
    """
    try:
        async with db_manager.get_async_session() as session:
            analytics_service = await get_real_time_analytics_service(session)
            success = await analytics_service.start_experiment_monitoring(experiment_id, update_interval)
            if success:
                return JSONResponse({'status': 'success', 'message': f'Started monitoring for experiment {experiment_id}', 'experiment_id': experiment_id, 'update_interval': update_interval})
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Failed to start monitoring (experiment may not exist or not be running)')
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error starting monitoring: %s', e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to start experiment monitoring')

@real_time_router.post('/experiments/{experiment_id}/monitoring/stop')
async def stop_monitoring(experiment_id: str, db_manager: UnifiedConnectionManager=Depends(get_unified_manager_async_modern)) -> JSONResponse:
    """Stop real-time monitoring for an experiment

    Args:
        experiment_id: UUID of experiment
        db_manager: Database manager

    Returns:
        JSON response confirming monitoring stopped
    """
    try:
        async with db_manager.get_async_session() as session:
            analytics_service = await get_real_time_analytics_service(session)
            success = await analytics_service.stop_experiment_monitoring(experiment_id)
            return JSONResponse({'status': 'success' if success else 'warning', 'message': f'Stopped monitoring for experiment {experiment_id}', 'experiment_id': experiment_id})
    except Exception as e:
        logger.error('Error stopping monitoring: %s', e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to stop experiment monitoring')

@real_time_router.get('/monitoring/active')
async def get_active_monitoring(db_manager: UnifiedConnectionManager=Depends(get_unified_manager_async_modern)) -> JSONResponse:
    """Get list of experiments currently being monitored

    Args:
        db_manager: Database manager

    Returns:
        JSON response with list of active experiments
    """
    try:
        async with db_manager.get_async_session() as session:
            analytics_service = await get_real_time_analytics_service(session)
            active_experiments = await analytics_service.get_active_experiments()
            connection_info: list[dict[str, Any]] = []
            for experiment_id in active_experiments:
                connection_count = connection_manager.get_connection_count(experiment_id)
                connection_info.append({'experiment_id': experiment_id, 'active_connections': connection_count})
            return JSONResponse({'status': 'success', 'active_experiments': len(active_experiments), 'total_connections': connection_manager.get_connection_count(), 'experiments': connection_info})
    except Exception as e:
        logger.error('Error getting active monitoring: %s', e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to retrieve active monitoring information')

@real_time_router.get('/dashboard/config/{experiment_id}')
async def get_dashboard_config(experiment_id: str, db_manager: UnifiedConnectionManager=Depends(get_unified_manager_async_modern)) -> JSONResponse:
    """Get dashboard configuration for an experiment

    Args:
        experiment_id: UUID of experiment
        db_manager: Database manager

    Returns:
        JSON response with dashboard configuration
    """
    try:
        async with db_manager.get_async_session() as session:
            stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            if not experiment:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Experiment not found')
            config: dict[str, Any] = {'experiment_id': experiment_id, 'experiment_name': experiment.experiment_name, 'status': experiment.status, 'started_at': experiment.started_at.isoformat() if experiment.started_at else None, 'target_metric': experiment.target_metric, 'sample_size_per_group': experiment.sample_size_per_group, 'significance_level': 0.05, 'dashboard_settings': {'auto_refresh': True, 'refresh_interval': 30, 'show_alerts': True, 'show_early_stopping': True, 'chart_types': ['line', 'bar', 'confidence_interval'], 'metrics_to_display': ['sample_sizes', 'conversion_rates', 'effect_size', 'p_value', 'confidence_interval', 'statistical_power']}, 'websocket_url': f'/api/v1/experiments/real-time/live/{experiment_id}', 'api_endpoints': {'metrics': f'/api/v1/experiments/real-time/experiments/{experiment_id}/metrics', 'start_monitoring': f'/api/v1/experiments/real-time/experiments/{experiment_id}/monitoring/start', 'stop_monitoring': f'/api/v1/experiments/real-time/experiments/{experiment_id}/monitoring/stop'}}
            return JSONResponse({'status': 'success', 'config': config})
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error getting dashboard config: %s', e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to retrieve dashboard configuration')

@real_time_router.get('/health')
async def health_check() -> JSONResponse:
    """Health check endpoint for real-time services"""
    try:
        active_experiments = connection_manager.get_active_experiments()
        total_connections = connection_manager.get_connection_count()
        redis_status = 'not_configured'
        if redis_available and connection_manager.redis_client:
            try:
                await connection_manager.redis_client.ping()
                redis_status = 'connected'
            except Exception:
                redis_status = 'disconnected'
        elif not redis_available:
            redis_status = 'not_installed'
        return JSONResponse({'status': 'healthy', 'timestamp': datetime.now(UTC).isoformat(), 'services': {'websocket_manager': {'status': 'active', 'active_experiments': len(active_experiments), 'total_connections': total_connections}, 'redis': {'status': redis_status, 'available': redis_available}}})
    except Exception as e:
        logger.error('Health check error: %s', e)
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={'status': 'unhealthy', 'error': str(e), 'timestamp': datetime.now(UTC).isoformat()})

@real_time_router.get('/orchestrator/status')
async def get_orchestrator_status() -> JSONResponse:
    """Get ML Pipeline Orchestrator status."""
    try:
        from prompt_improver.core.di.ml_container import MLServiceContainer
        from prompt_improver.core.factories.ml_pipeline_factory import MLPipelineOrchestratorFactory
        from prompt_improver.core.protocols.ml_protocols import ServiceContainerProtocol
        from prompt_improver.ml.orchestration.config.external_services_config import ExternalServicesConfig
        service_container = MLServiceContainer()
        external_config = ExternalServicesConfig()
        orchestrator = await MLPipelineOrchestratorFactory.create_from_container(service_container)
        status_data: dict[str, Any] = {'state': orchestrator.state.value, 'initialized': getattr(orchestrator, '_is_initialized', False), 'active_workflows': len(orchestrator.active_workflows), 'timestamp': datetime.now().isoformat()}
        if getattr(orchestrator, '_is_initialized', False):
            components = orchestrator.get_loaded_components()
            status_data.update({'loaded_components': len(components), 'component_list': components[:10], 'recent_invocations': len(orchestrator.get_invocation_history())})
        return JSONResponse(content={'success': True, 'data': status_data, 'message': 'Orchestrator status retrieved successfully'})
    except ImportError as e:
        logger.error('Failed to import orchestrator: %s', e)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Orchestrator not available', 'message': str(e)})
    except Exception as e:
        logger.error('Error getting orchestrator status: %s', e)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Failed to get orchestrator status', 'message': str(e)})

@real_time_router.get('/orchestrator/components')
async def get_orchestrator_components() -> JSONResponse:
    """Get loaded ML components information."""
    try:
        from prompt_improver.core.di.ml_container import MLServiceContainer
        from prompt_improver.core.factories.ml_pipeline_factory import MLPipelineOrchestratorFactory
        from prompt_improver.ml.orchestration.config.external_services_config import ExternalServicesConfig
        service_container = MLServiceContainer()
        orchestrator = await MLPipelineOrchestratorFactory.create_from_container(service_container)
        if not getattr(orchestrator, '_is_initialized', False):
            await orchestrator.initialize()
        components = orchestrator.get_loaded_components()
        component_details: list[dict[str, Any]] = []
        for component_name in components:
            methods = orchestrator.get_component_methods(component_name)
            component_details.append({'name': component_name, 'methods_count': len(methods), 'methods': methods[:5], 'is_loaded': orchestrator.component_loader.is_component_loaded(component_name), 'is_initialized': orchestrator.component_loader.is_component_initialized(component_name)})
        return JSONResponse(content={'success': True, 'data': {'total_components': len(components), 'components': component_details, 'timestamp': datetime.now().isoformat()}, 'message': f'Retrieved {len(components)} components'})
    except Exception as e:
        logger.error('Error getting orchestrator components: %s', e)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Failed to get components', 'message': str(e)})

@real_time_router.get('/orchestrator/history')
async def get_orchestrator_history(component: str | None=None, limit: int=50) -> JSONResponse:
    """Get orchestrator invocation history."""
    try:
        from prompt_improver.core.di.ml_container import MLServiceContainer
        from prompt_improver.core.factories.ml_pipeline_factory import MLPipelineOrchestratorFactory
        from prompt_improver.ml.orchestration.config.external_services_config import ExternalServicesConfig
        service_container = MLServiceContainer()
        orchestrator = await MLPipelineOrchestratorFactory.create_from_container(service_container)
        if not getattr(orchestrator, '_is_initialized', False):
            return JSONResponse(content={'success': False, 'error': 'Orchestrator not initialized', 'message': 'Initialize orchestrator first'})
        history = orchestrator.get_invocation_history(component)[-limit:]
        total_invocations = len(history)
        successful_invocations = sum((1 for inv in history if inv['success']))
        success_rate = successful_invocations / total_invocations if total_invocations > 0 else 0.0
        return JSONResponse(content={'success': True, 'data': {'total_invocations': total_invocations, 'successful_invocations': successful_invocations, 'success_rate': success_rate, 'filtered_component': component, 'history': history, 'timestamp': datetime.now().isoformat()}, 'message': f'Retrieved {total_invocations} invocation records'})
    except Exception as e:
        logger.error('Error getting orchestrator history: %s', e)
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Failed to get history', 'message': str(e)})

async def setup_real_time_system(redis_url: str=None):
    """Setup real-time system with Redis connection"""
    try:
        if redis_available:
            if redis_url is None:
                import os
                redis_url = os.getenv('REDIS_URL', 'redis://redis.external.service:6379/0')
            await setup_redis_connection(redis_url)
            logger.info('Real-time system setup completed with Redis')
        else:
            logger.warning('Real-time system setup completed without Redis (not installed)')
    except Exception as e:
        logger.error('Failed to setup real-time system: %s', e)

async def cleanup_real_time_system():
    """Cleanup real-time system resources"""
    try:
        await connection_manager.cleanup()
        logger.info('Real-time system cleanup completed')
    except Exception as e:
        logger.error('Error during real-time system cleanup: %s', e)
__all__ = ['cleanup_real_time_system', 'real_time_router', 'setup_real_time_system']
