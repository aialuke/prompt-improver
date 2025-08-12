"""Usage Examples for APES Monitoring Integration
Shows how to integrate metrics collection into the application.
"""

from fastapi import FastAPI

# Legacy record_* helpers removed in unified middleware path; OTEL emission handled internally.
from src.prompt_improver.monitoring.opentelemetry.metrics import get_business_metrics

from prompt_improver.monitoring import get_metrics, health_router
from prompt_improver.monitoring.http.unified_http_middleware import (
    UnifiedHTTPMetricsMiddleware,
)

app = FastAPI(title="APES ML Pipeline Orchestrator")
app.add_middleware(
    UnifiedHTTPMetricsMiddleware,
    enable_in_memory_analytics=True,
    journey_labels="hashed",
)
app.include_router(health_router, prefix="/api/v1")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics()


async def ml_prediction_example(model_name: str, input_data: dict):
    """Example of recording ML prediction metrics."""
    import time

    start_time = time.time()
    try:
        result = await some_ml_model.predict(input_data)
        duration = time.time() - start_time
        # OTEL emission handled inside middleware; explicit record_* helpers removed
        return result
    except Exception as e:
        # OTEL emission handled inside middleware
        raise


async def database_operation_example():
    """Example of recording database operation metrics."""
    import time

    start_time = time.time()
    try:
        result = await database.query("SELECT * FROM prompts")
        duration = time.time() - start_time
        # OTEL emission handled via database wrappers
        return result
    except Exception as e:
        duration = time.time() - start_time
        record_database_query("SELECT", "prompts", duration)
        raise


async def cache_operation_example(key: str):
    """Example of recording cache operation metrics."""
    from prompt_improver.utils.redis_cache import get_cache

    cache = await get_cache()
    try:
        value = await cache.get(key)
        if value is not None:
            record_cache_hit("redis", "user_sessions")
            return value
        record_cache_miss("redis", "user_sessions")
        value = await fetch_from_database(key)
        await cache.set(key, value, expire=3600)
        return value
    except Exception as e:
        record_cache_miss("redis", "user_sessions")
        raise


async def batch_processing_monitor():
    """Example of monitoring batch processing queues."""
    from prompt_improver.ml.optimization.batch.batch_processor import BatchProcessor

    processor = BatchProcessor()
    while True:
        queue_size = await processor.get_queue_size()
        record_batch_queue_size("ml_training", queue_size)
        await asyncio.sleep(30)


def start_metrics_server(port: int = 8001):
    """Start standalone Prometheus metrics server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
    import os
    host = os.getenv('METRICS_HOST', 'localhost')
    print(f"Metrics available at http://{host}:{port}/metrics")


business_metrics = get_business_metrics()


def update_business_metrics():
    """Update custom business metrics using OpenTelemetry."""
    active_users = get_active_user_count()
    business_metrics.record_user_activity(
        user_id=f"user_{active_users}", activity_type="active_session"
    )
    business_metrics.record_feature_flag_evaluation(
        flag_name="prompt_improvement_clarity", enabled=True
    )
    business_metrics.record_feature_flag_evaluation(
        flag_name="prompt_improvement_specificity", enabled=True
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
