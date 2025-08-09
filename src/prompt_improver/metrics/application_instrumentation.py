"""Application Instrumentation for Business Metrics Collection.

Provides automatic instrumentation of existing application components
to collect real business metrics without requiring manual code changes.
"""
import asyncio
import inspect
import logging
import time
import uuid
from typing import Any, Dict, Optional
from prompt_improver.metrics.business_intelligence_metrics import CostType, FeatureCategory
# Legacy integration_middleware removed; replace decorator usage with no-ops or OTEL adapters as needed
from prompt_improver.monitoring.opentelemetry.metrics import get_business_metrics, get_ml_metrics, get_database_metrics
# Provide minimal decorator shims

def track_cost_operation(*args, **kwargs):
    def deco(fn):
        async def wrapper(*f_args, **f_kwargs):
            return await fn(*f_args, **f_kwargs)
        return wrapper
    return deco

def track_feature_usage(*args, **kwargs):
    def deco(fn):
        async def wrapper(*f_args, **f_kwargs):
            return await fn(*f_args, **f_kwargs)
        return wrapper
    return deco

from prompt_improver.metrics.ml_metrics import ModelInferenceStage, PromptCategory
from prompt_improver.metrics.performance_metrics import CacheType, DatabaseOperation
from prompt_improver.performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
logger = logging.getLogger(__name__)

class PromptImprovementInstrumentation:
    """Automatically instrument prompt improvement services to collect ML metrics."""

    @staticmethod
    def instrument_prompt_service(service_class: Any) -> Any:
        """Instrument a prompt improvement service class to automatically collect metrics.

        Usage:
            @PromptImprovementInstrumentation.instrument_prompt_service
            class PromptImprovementService:
                async def improve_prompt(self, prompt: str) -> dict:
                    # Your existing implementation
                    pass
        """
        improvement_methods = ['improve_prompt', 'enhance_prompt', 'optimize_prompt', 'analyze_prompt', 'suggest_improvements', 'refactor_prompt']
        for method_name in improvement_methods:
            if hasattr(service_class, method_name):
                original_method = getattr(service_class, method_name)
                category = PromptCategory.CLARITY
                if 'specific' in method_name.lower():
                    category = PromptCategory.SPECIFICITY
                elif 'context' in method_name.lower():
                    category = PromptCategory.CONTEXT
                elif 'structure' in method_name.lower():
                    category = PromptCategory.STRUCTURE
                elif 'role' in method_name.lower():
                    category = PromptCategory.ROLE_BASED
                elif 'few_shot' in method_name.lower():
                    category = PromptCategory.FEW_SHOT
                elif 'chain' in method_name.lower() or 'cot' in method_name.lower():
                    category = PromptCategory.CHAIN_OF_THOUGHT
                elif 'xml' in method_name.lower():
                    category = PromptCategory.XML_ENHANCEMENT
                instrumented_method = track_ml_operation(category=category, stage=ModelInferenceStage.MODEL_FORWARD, model_name='prompt_improver_v1')(original_method)
                setattr(service_class, method_name, instrumented_method)
                logger.info('Instrumented %s.%s for ML metrics', service_class.__name__, method_name)
        return service_class

class MLPipelineInstrumentation:
    """Automatically instrument ML pipeline components."""

    @staticmethod
    def instrument_ml_service(service_class: Any) -> Any:
        """Instrument ML service class to automatically collect metrics."""
        ml_methods = ['train_model', 'predict', 'inference', 'evaluate_model', 'preprocess_data', 'postprocess_results', 'feature_extraction']
        for method_name in ml_methods:
            if hasattr(service_class, method_name):
                original_method = getattr(service_class, method_name)
                category = FeatureCategory.ML_ANALYTICS
                instrumented_method = track_feature_usage(feature_name=method_name, feature_category=category)(track_cost_operation(operation_type=f'ml_{method_name}', cost_type=CostType.ML_INFERENCE, estimated_cost_per_unit=0.01)(original_method))
                setattr(service_class, method_name, instrumented_method)
                logger.info('Instrumented %s.%s for ML metrics', service_class.__name__, method_name)
        return service_class

class APIServiceInstrumentation:
    """Automatically instrument API service classes."""

    @staticmethod
    def instrument_api_service(service_class: Any, feature_category: FeatureCategory=FeatureCategory.API_INTEGRATION) -> Any:
        """Instrument API service class to collect business metrics."""
        api_methods = [name for name in dir(service_class) if not name.startswith('_') and callable(getattr(service_class, name))]
        for method_name in api_methods:
            original_method = getattr(service_class, method_name)
            if hasattr(original_method, '_instrumented') or not inspect.isfunction(original_method):
                continue
            instrumented_method = track_feature_usage(feature_name=f'api_{method_name}', feature_category=feature_category)(track_cost_operation(operation_type=f'api_{method_name}', cost_type=CostType.COMPUTE, estimated_cost_per_unit=0.001)(original_method))
            instrumented_method._instrumented = True
            setattr(service_class, method_name, instrumented_method)
            logger.info('Instrumented %s.%s for API metrics', service_class.__name__, method_name)
        return service_class

class DatabaseInstrumentation:
    """Automatically instrument database operations."""

    @staticmethod
    def instrument_database_class(db_class: Any) -> Any:
        """Instrument database class to automatically track query performance."""
        db_methods = ['execute', 'fetch', 'fetchone', 'fetchall', 'fetchmany', 'insert', 'update', 'delete', 'select', 'query', 'create', 'drop', 'alter']
        for method_name in db_methods:
            if hasattr(db_class, method_name):
                original_method = getattr(db_class, method_name)
                operation_type = DatabaseOperation.SELECT
                if method_name in ['insert', 'create']:
                    operation_type = DatabaseOperation.INSERT
                elif method_name in ['update', 'alter']:
                    operation_type = DatabaseOperation.UPDATE
                elif method_name in ['delete', 'drop']:
                    operation_type = DatabaseOperation.DELETE

                async def instrumented_method(self: Any, query_or_sql: Any, *args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    table_name = 'unknown'
                    rows_affected = 0
                    success = False
                    error_type = None
                    if isinstance(query_or_sql, str):
                        query_str = query_or_sql.lower()
                        if 'from ' in query_str:
                            parts = query_str.split('from ')
                            if len(parts) > 1:
                                table_part = parts[1].split()[0]
                                table_name = table_part.strip('`"[]')
                        elif any((keyword in query_str for keyword in ['insert into', 'update ', 'delete from'])):
                            for keyword in ['insert into ', 'update ', 'delete from ']:
                                if keyword in query_str:
                                    parts = query_str.split(keyword)
                                    if len(parts) > 1:
                                        table_name = parts[1].split()[0].strip('`"[]')
                                    break
                    try:
                        result = await original_method(self, query_or_sql, *args, **kwargs)
                        success = True
                        error_type = None
                        rows_affected = 0
                        if hasattr(result, 'rowcount'):
                            rows_affected = result.rowcount
                        elif isinstance(result, list):
                            rows_affected = len(result)
                        elif result is not None:
                            rows_affected = 1
                    except Exception as e:
                        result = None
                        success = False
                        error_type = type(e).__name__
                        rows_affected = 0
                        raise
                    finally:
                        end_time = time.time()
                        execution_time_ms = (end_time - start_time) * 1000
                        await db_metrics.track_query(query=str(query_or_sql)[:500], operation_type=operation_type, table_name=table_name, execution_time_ms=execution_time_ms, rows_affected=rows_affected, success=success, error_type=error_type)
                    return result
                if not inspect.iscoroutinefunction(original_method):

                    def sync_instrumented_method(self: Any, query_or_sql: Any, *args: Any, **kwargs: Any) -> Any:
                        start_time = time.time()
                        table_name = 'unknown'
                        rows_affected = 0
                        success = False
                        error_type = None
                        if isinstance(query_or_sql, str):
                            query_str = query_or_sql.lower()
                            if 'from ' in query_str:
                                parts = query_str.split('from ')
                                if len(parts) > 1:
                                    table_part = parts[1].split()[0]
                                    table_name = table_part.strip('`"[]')
                        try:
                            result = original_method(self, query_or_sql, *args, **kwargs)
                            success = True
                            error_type = None
                            rows_affected = 0
                            if hasattr(result, 'rowcount'):
                                rows_affected = result.rowcount
                            elif isinstance(result, list):
                                rows_affected = len(result)
                            elif result is not None:
                                rows_affected = 1
                        except Exception as e:
                            result = None
                            success = False
                            error_type = type(e).__name__
                            rows_affected = 0
                            raise
                        finally:
                            end_time = time.time()
                            execution_time_ms = (end_time - start_time) * 1000

                            async def submit_db_tracking():
                                task_manager = get_background_task_manager()
                                await task_manager.submit_enhanced_task(task_id=f'db_track_{operation_type.value}_{str(uuid.uuid4())[:8]}', coroutine=db_metrics.track_query(query=str(query_or_sql)[:500], operation_type=operation_type, table_name=table_name, execution_time_ms=execution_time_ms, rows_affected=rows_affected, success=success, error_type=error_type), priority=TaskPriority.NORMAL, tags={'service': 'metrics', 'type': 'tracking', 'component': 'database', 'table': table_name})
                            loop = asyncio.get_event_loop()
                            loop.create_task(submit_db_tracking())
                        return result
                    setattr(db_class, method_name, sync_instrumented_method)
                else:
                    setattr(db_class, method_name, instrumented_method)
                logger.info('Instrumented %s.%s for database metrics', db_class.__name__, method_name)
        return db_class

class CacheInstrumentation:
    """Automatically instrument cache operations."""

    @staticmethod
    def instrument_cache_class(cache_class: Any, cache_type: CacheType=CacheType.APPLICATION) -> Any:
        """Instrument cache class to automatically track cache performance."""
        cache_methods = ['get', 'set', 'delete', 'exists', 'clear', 'expire']
        for method_name in cache_methods:
            if hasattr(cache_class, method_name):
                original_method = getattr(cache_class, method_name)

                async def instrumented_method(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    hit = False
                    success = False
                    try:
                        result = await original_method(self, key, *args, **kwargs)
                        if method_name == 'get':
                            hit = result is not None
                        success = True
                    except Exception:
                        result = None
                        success = False
                        raise
                    finally:
                        end_time = time.time()
                        response_time_ms = (end_time - start_time) * 1000
                        await cache_metrics.track_cache_operation(cache_type=cache_type, operation=method_name, key=str(key), hit=hit, response_time_ms=response_time_ms, success=success)
                    return result
                if not inspect.iscoroutinefunction(original_method):

                    def sync_instrumented_method(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
                        start_time = time.time()
                        hit = False
                        success = False
                        try:
                            result = original_method(self, key, *args, **kwargs)
                            if method_name == 'get':
                                hit = result is not None
                            success = True
                        except Exception:
                            result = None
                            success = False
                            raise
                        finally:
                            end_time = time.time()
                            response_time_ms = (end_time - start_time) * 1000

                            async def submit_cache_tracking():
                                task_manager = get_background_task_manager()
                                await task_manager.submit_enhanced_task(task_id=f'cache_track_{method_name}_{str(uuid.uuid4())[:8]}', coroutine=cache_metrics.track_cache_operation(cache_type=cache_type, operation=method_name, key=str(key), hit=hit, response_time_ms=response_time_ms, success=success), priority=TaskPriority.NORMAL, tags={'service': 'metrics', 'type': 'tracking', 'component': 'cache', 'operation': method_name})
                            loop = asyncio.get_event_loop()
                            loop.create_task(submit_cache_tracking())
                        return result
                    setattr(cache_class, method_name, sync_instrumented_method)
                else:
                    setattr(cache_class, method_name, instrumented_method)
                logger.info('Instrumented %s.%s for cache metrics', cache_class.__name__, method_name)
        return cache_class

class BatchProcessingInstrumentation:
    """Automatically instrument batch processing operations."""

    @staticmethod
    def instrument_batch_processor(processor_class: Any) -> Any:
        """Instrument batch processor to track performance and costs."""
        batch_methods = ['process_batch', 'run_batch', 'execute_batch', 'process_items', 'handle_batch']
        for method_name in batch_methods:
            if hasattr(processor_class, method_name):
                original_method = getattr(processor_class, method_name)
                instrumented_method = track_feature_usage(feature_name=f'batch_{method_name}', feature_category=FeatureCategory.BATCH_PROCESSING)(track_cost_operation(operation_type=f'batch_{method_name}', cost_type=CostType.COMPUTE, estimated_cost_per_unit=0.05)(original_method))
                setattr(processor_class, method_name, instrumented_method)
                logger.info('Instrumented %s.%s for batch metrics', processor_class.__name__, method_name)
        return processor_class

def auto_instrument_module(module: Any, instrumentation_config: dict[str, Any] | None=None) -> None:
    """Automatically discover and instrument classes in a module based on naming patterns.

    Args:
        module: The module to instrument
        instrumentation_config: Configuration for instrumentation behavior
    """
    _ = instrumentation_config or {}
    for name in dir(module):
        obj = getattr(module, name)
        if not inspect.isclass(obj):
            continue
        class_name = name.lower()
        if any((keyword in class_name for keyword in ['prompt', 'improvement', 'enhance'])):
            PromptImprovementInstrumentation.instrument_prompt_service(obj)
        elif any((keyword in class_name for keyword in ['ml', 'model', 'inference', 'predict'])):
            MLPipelineInstrumentation.instrument_ml_service(obj)
        elif any((keyword in class_name for keyword in ['api', 'endpoint', 'router', 'handler'])):
            category = FeatureCategory.API_INTEGRATION
            if 'ml' in class_name:
                category = FeatureCategory.ML_ANALYTICS
            elif 'auth' in class_name:
                category = FeatureCategory.AUTHENTICATION
            APIServiceInstrumentation.instrument_api_service(obj, category)
        elif any((keyword in class_name for keyword in ['database', 'db', 'repository', 'dao'])):
            DatabaseInstrumentation.instrument_database_class(obj)
        elif any((keyword in class_name for keyword in ['cache', 'redis', 'memcache'])):
            cache_type = CacheType.REDIS if 'redis' in class_name else CacheType.APPLICATION
            CacheInstrumentation.instrument_cache_class(obj, cache_type)
        elif any((keyword in class_name for keyword in ['batch', 'processor', 'worker'])):
            BatchProcessingInstrumentation.instrument_batch_processor(obj)

def instrument_application_startup():
    """Instrument the application at startup to automatically collect metrics
    from existing business operations.
    """
    try:
        logger.info('Starting application instrumentation for business metrics...')
        try:
            from prompt_improver.core.services.prompt_improvement import PromptImprovementService
            PromptImprovementInstrumentation.instrument_prompt_service(PromptImprovementService)
            logger.info('Instrumented PromptImprovementService')
        except ImportError:
            logger.debug('PromptImprovementService not found')
        try:
            from prompt_improver.ml.core.ml_integration import MLService
            MLPipelineInstrumentation.instrument_ml_service(MLService)
            logger.info('Instrumented MLService')
        except ImportError:
            logger.debug('MLService not found')
        try:
            from prompt_improver.api import analytics_endpoints, health, real_time_endpoints
            auto_instrument_module(health)
            auto_instrument_module(analytics_endpoints)
            auto_instrument_module(real_time_endpoints)
            logger.info('Instrumented API modules')
        except ImportError:
            logger.debug('Some API modules not found')
        try:
            from prompt_improver.database import connection
            auto_instrument_module(connection)
            logger.info('Instrumented database modules')
        except ImportError:
            logger.debug('Database modules not found')
        try:
            from prompt_improver.core.config import AppConfig
            CacheInstrumentation.instrument_cache_class(RedisCache, CacheType.REDIS)
            logger.info('Instrumented RedisCache')
        except ImportError:
            logger.debug('RedisCache not found')
        logger.info('Application instrumentation completed successfully')
    except Exception as e:
        logger.error('Failed to instrument application: %s', e)

def create_custom_instrumentor(target_class: Any, feature_category: FeatureCategory, cost_type: CostType=CostType.COMPUTE, estimated_cost: float=0.001) -> Any:
    """Create a custom instrumentor for specific classes that don't fit standard patterns.

    Args:
        target_class: The class to instrument
        feature_category: Feature category for business intelligence
        cost_type: Type of cost for cost tracking
        estimated_cost: Estimated cost per operation

    Returns:
        Instrumented class
    """
    methods = [name for name in dir(target_class) if not name.startswith('_') and callable(getattr(target_class, name))]
    for method_name in methods:
        original_method = getattr(target_class, method_name)
        if not inspect.isfunction(original_method) and (not inspect.ismethod(original_method)):
            continue
        instrumented_method = track_feature_usage(feature_name=f'custom_{method_name}', feature_category=feature_category)(track_cost_operation(operation_type=f'custom_{method_name}', cost_type=cost_type, estimated_cost_per_unit=estimated_cost)(original_method))
        setattr(target_class, method_name, instrumented_method)
    logger.info('Custom instrumentation applied to %s', target_class.__name__)
    return target_class
