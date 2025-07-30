"""
Usage Examples for APES Monitoring Integration
Shows how to integrate metrics collection into the application.
"""

from fastapi import FastAPI
# REMOVED prometheus_client for OpenTelemetry consolidation
from prompt_improver.monitoring import OpenTelemetryMiddleware, health_router, get_metrics

# Example 1: Basic FastAPI integration
app = FastAPI(title="APES ML Pipeline Orchestrator")

# Add metrics middleware
app.add_middleware(OpenTelemetryMiddleware, metrics_enabled=True)

# Add health check endpoints
app.include_router(health_router, prefix="/api/v1")

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics()

# Example 2: Manual metrics recording in ML operations
from prompt_improver.monitoring.metrics_middleware import (
    record_ml_prediction,
    record_ml_error,
    record_database_query,
    record_cache_hit,
    record_cache_miss
)

async def ml_prediction_example(model_name: str, input_data: dict):
    """Example of recording ML prediction metrics."""
    import time
    
    start_time = time.time()
    
    try:
        # Your ML prediction logic here
        result = await some_ml_model.predict(input_data)
        
        # Record successful prediction
        duration = time.time() - start_time
        record_ml_prediction(model_name, "v1.0", duration)
        
        return result
        
    except Exception as e:
        # Record ML error
        record_ml_error(model_name, "v1.0", type(e).__name__)
        raise

async def database_operation_example():
    """Example of recording database operation metrics."""
    import time
    
    start_time = time.time()
    
    try:
        # Your database operation here
        result = await database.query("SELECT * FROM prompts")
        
        # Record query metrics
        duration = time.time() - start_time
        record_database_query("SELECT", "prompts", duration)
        
        return result
        
    except Exception as e:
        # Still record the query attempt
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
        else:
            record_cache_miss("redis", "user_sessions")
            # Fetch from database and cache
            value = await fetch_from_database(key)
            await cache.set(key, value, expire=3600)
            return value
            
    except Exception as e:
        record_cache_miss("redis", "user_sessions")
        raise

# Example 3: Batch processing metrics
from prompt_improver.monitoring.metrics_middleware import record_batch_queue_size

async def batch_processing_monitor():
    """Example of monitoring batch processing queues."""
    from prompt_improver.ml.optimization.batch.batch_processor import BatchProcessor
    
    processor = BatchProcessor()
    
    # Record queue size periodically
    while True:
        queue_size = await processor.get_queue_size()
        record_batch_queue_size("ml_training", queue_size)
        
        await asyncio.sleep(30)  # Update every 30 seconds

# Example 4: Starting metrics server (alternative to FastAPI endpoint)
def start_metrics_server(port: int = 8001):
    """Start standalone Prometheus metrics server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
    print(f"Metrics available at http://localhost:{port}/metrics")

# Example 5: Custom metric collectors using OpenTelemetry
from src.prompt_improver.monitoring.opentelemetry.metrics import get_business_metrics

# Get OpenTelemetry business metrics instance
business_metrics = get_business_metrics()

def update_business_metrics():
    """Update custom business metrics using OpenTelemetry."""
    # Your business logic here
    active_users = get_active_user_count()

    # Record business metrics using OpenTelemetry
    business_metrics.record_user_activity(
        user_id=f"user_{active_users}",
        activity_type="active_session"
    )

    # Record specific improvement types
    business_metrics.record_feature_flag_evaluation(
        flag_name="prompt_improvement_clarity",
        enabled=True
    )
    business_metrics.record_feature_flag_evaluation(
        flag_name="prompt_improvement_specificity",
        enabled=True
    )

if __name__ == "__main__":
    # Example usage
    import uvicorn
    
    # Start the application with metrics
    uvicorn.run(app, host="0.0.0.0", port=8000)
