"""Pure MCP Server implementation for the Adaptive Prompt Enhancement System (APES).
Provides prompt enhancement via Model Context Protocol with stdio transport.
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from prompt_improver.database import get_session
from prompt_improver.ml.optimization.batch.batch_processor import (
    BatchProcessor,
    BatchProcessorConfig,
    periodic_batch_processor_coroutine,
)
from prompt_improver.performance.analytics.analytics import AnalyticsService
from prompt_improver.performance.monitoring.health.background_manager import (
    get_background_task_manager,
)
from prompt_improver.core.services.prompt_improvement import PromptImprovementService
from prompt_improver.utils.event_loop_benchmark import run_startup_benchmark
from prompt_improver.utils.event_loop_manager import (
    get_event_loop_manager,
    setup_uvloop,
)
from prompt_improver.utils.redis_cache import (
    start_cache_subscriber,
    stop_cache_subscriber,
)
from prompt_improver.utils.session_event_loop import get_session_wrapper
from prompt_improver.utils.session_store import SessionStore
from prompt_improver.performance.optimization.performance_optimizer import (
    get_performance_optimizer,
    measure_mcp_operation,
    measure_database_operation
)
from prompt_improver.performance.monitoring.performance_monitor import (
    get_performance_monitor,
    record_mcp_operation
)
from prompt_improver.utils.multi_level_cache import get_cache, cached_get
from prompt_improver.performance.optimization.response_optimizer import optimize_mcp_response
from prompt_improver.performance.optimization.async_optimizer import get_async_optimizer

# Configure logging to stderr for MCP protocol compliance
# MCP servers using stdio transport should log to stderr, not stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP(
    name="APES - Adaptive Prompt Enhancement System",
    description="AI-powered prompt optimization service using ML-driven rules",
)

# Initialize services
prompt_service = PromptImprovementService()
analytics_service = AnalyticsService()
batch_processor = BatchProcessor(BatchProcessorConfig(
    batch_size=10,
    batch_timeout=30,
    max_attempts=3,
    concurrency=3,
    enable_circuit_breaker=True,
    enable_dead_letter_queue=True,
    enable_opentelemetry=True
))


# Register periodic batch processing task as background task
# This will be registered once the event loop is running
async def register_periodic_batch_processor():
    """Register the periodic batch processor as a background task."""
    try:
        background_task_manager = get_background_task_manager()
        await background_task_manager.submit_task(
            task_id="periodic_batch_processor",
            coroutine=periodic_batch_processor_coroutine,
            batch_processor=batch_processor,
        )
        logger.info("Periodic batch processor registered successfully")
    except Exception as e:
        logger.error(f"Failed to register periodic batch processor: {e}")
        # Fallback: create the task directly
        asyncio.create_task(periodic_batch_processor_coroutine(batch_processor))


# Schedule the registration to run when the event loop is available
if hasattr(asyncio, "get_running_loop"):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(register_periodic_batch_processor())
    except RuntimeError:
        # No running loop yet, will be registered during server startup
        pass

# Initialize session store with TTL
session_store = SessionStore(
    maxsize=1000,  # Max 1000 concurrent sessions
    ttl=3600,  # 1 hour TTL
    cleanup_interval=300,  # Cleanup every 5 minutes
)


class PromptEnhancementRequest(BaseModel):
    """Request model for prompt enhancement"""

    prompt: str = Field(..., description="The prompt to enhance")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context for enhancement"
    )
    session_id: str | None = Field(default=None, description="Session ID for tracking")


class PromptStorageRequest(BaseModel):
    """Request model for storing prompt data"""

    original: str = Field(..., description="The original prompt")
    enhanced: str = Field(..., description="The enhanced prompt")
    metrics: dict[str, Any] = Field(..., description="Success metrics")
    session_id: str | None = Field(default=None, description="Session ID for tracking")


@mcp.tool()
async def improve_prompt(
    prompt: str = Field(..., description="The prompt to enhance"),
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context"
    ),
    session_id: str | None = Field(default=None, description="Session ID for tracking"),
) -> dict[str, Any]:
    """Enhance a prompt using ML-optimized rules.

    This tool applies data-driven rules to improve prompt clarity, specificity,
    and effectiveness. Response time is optimized for <200ms.

    Args:
        prompt: The prompt text to enhance
        context: Optional context information
        session_id: Optional session ID for tracking
        ctx: MCP context (provided by framework)

    Returns:
        Enhanced prompt with processing metrics and applied rules
    """
    # Track start time for fallback metrics
    start_time = time.time()
    
    # Use performance optimizer for comprehensive measurement
    async with measure_mcp_operation(
        "improve_prompt",
        prompt_length=len(prompt),
        has_context=context is not None,
        session_id=session_id
    ) as perf_metrics:
        try:
            # Get session wrapper for performance monitoring
            session_wrapper = get_session_wrapper(session_id or "default")

            # Use session wrapper for performance context
            async with session_wrapper.performance_context("prompt_improvement"):
                # Get database session with performance measurement
                async with measure_database_operation("get_session"):
                    async with get_session() as db_session:
                        # Use the existing prompt improvement service
                        result = await prompt_service.improve_prompt(
                            prompt=prompt,
                            user_context=context,
                            session_id=session_id,
                            db_session=db_session,
                        )

                        # Add comprehensive performance metrics
                        result["processing_time_ms"] = perf_metrics.duration_ms or 0
                        result["mcp_transport"] = "stdio"
                        result["performance_target_met"] = (perf_metrics.duration_ms or 0) < 200

                        # Add event loop info
                        loop_manager = get_event_loop_manager()
                        result["event_loop_info"] = {
                            "type": loop_manager.get_loop_type(),
                            "uvloop_enabled": loop_manager.is_uvloop_enabled(),
                        }

            # Store the prompt data for ML training (real data priority 100)
            if result.get("improved_prompt") and result["improved_prompt"] != prompt:
                # Store asynchronously using BackgroundTaskManager
                task_manager = get_background_task_manager()
                task_id = f"store_prompt_{session_id or result.get('session_id', 'unknown')}_{int(time.time())}"
                try:
                    await task_manager.submit_task(
                        task_id=task_id,
                        coroutine=_store_prompt_data,
                        original=prompt,
                        enhanced=result["improved_prompt"],
                        metrics=result.get("metrics", {}),
                        session_id=session_id or result.get("session_id"),
                        priority=100,  # Real data priority
                    )
                except Exception as e:
                    # Fallback to direct execution if task manager fails
                    logger.warning(f"Background task submission failed: {e}")
                    asyncio.create_task(
                        _store_prompt_data(
                            original=prompt,
                            enhanced=result["improved_prompt"],
                            metrics=result.get("metrics", {}),
                            session_id=session_id or result.get("session_id"),
                            priority=100,
                        )
                    )

            return result

        except Exception as e:
            # Return graceful error response
            return {
                "improved_prompt": prompt,  # Fallback to original
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "fallback": True,
            }


@mcp.tool()
async def store_prompt(
    original: str = Field(..., description="The original prompt"),
    enhanced: str = Field(..., description="The enhanced prompt"),
    metrics: dict[str, Any] = Field(..., description="Success metrics"),
    session_id: str | None = Field(default=None, description="Session ID"),
) -> dict[str, Any]:
    """Store prompt interaction data for ML training.

    This tool captures real prompt data with priority 100 for training
    the ML models to improve rule effectiveness over time.

    Args:
        original: The original prompt text
        enhanced: The enhanced prompt text
        metrics: Performance and success metrics
        session_id: Optional session ID
        ctx: MCP context (provided by framework)

    Returns:
        Storage confirmation with priority level
    """
    try:
        await _store_prompt_data(
            original=original,
            enhanced=enhanced,
            metrics=metrics,
            session_id=session_id,
            priority=100,  # Real data priority
        )

        return {
            "status": "stored",
            "priority": 100,
            "data_source": "real",
            "session_id": session_id,
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "priority": 0}


@mcp.tool()
async def get_session(
    session_id: str = Field(..., description="Session ID to retrieve"),
) -> dict[str, Any]:
    """Get session data by ID.

    Args:
        session_id: Session ID to retrieve
        ctx: MCP context (provided by framework)

    Returns:
        Session data if found, error message otherwise
    """
    try:
        session_data = await session_store.get(session_id)
        if session_data is not None:
            return {
                "status": "found",
                "session_id": session_id,
                "data": session_data,
            }
        return {
            "status": "not_found",
            "session_id": session_id,
            "error": "Session not found or expired",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "session_id": session_id}


@mcp.tool()
async def set_session(
    session_id: str = Field(..., description="Session ID to set"),
    data: dict[str, Any] = Field(..., description="Session data to store"),
) -> dict[str, Any]:
    """Set session data by ID.

    Args:
        session_id: Session ID to set
        data: Session data to store
        ctx: MCP context (provided by framework)

    Returns:
        Confirmation of session storage
    """
    try:
        success = await session_store.set(session_id, data)
        if success:
            return {
                "status": "stored",
                "session_id": session_id,
                "data_keys": list(data.keys()) if isinstance(data, dict) else [],
            }
        return {
            "status": "error",
            "error": "Failed to store session",
            "session_id": session_id,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "session_id": session_id}


@mcp.tool()
async def touch_session(
    session_id: str = Field(..., description="Session ID to touch"),
) -> dict[str, Any]:
    """Touch session to extend its TTL.

    Args:
        session_id: Session ID to touch
        ctx: MCP context (provided by framework)

    Returns:
        Confirmation of session touch
    """
    try:
        success = await session_store.touch(session_id)
        if success:
            return {
                "status": "touched",
                "session_id": session_id,
                "message": "Session TTL extended",
            }
        return {
            "status": "not_found",
            "session_id": session_id,
            "error": "Session not found",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "session_id": session_id}


@mcp.tool()
async def delete_session(
    session_id: str = Field(..., description="Session ID to delete"),
) -> dict[str, Any]:
    """Delete session by ID.

    Args:
        session_id: Session ID to delete
        ctx: MCP context (provided by framework)

    Returns:
        Confirmation of session deletion
    """
    try:
        success = await session_store.delete(session_id)
        if success:
            return {
                "status": "deleted",
                "session_id": session_id,
            }
        return {
            "status": "not_found",
            "session_id": session_id,
            "error": "Session not found",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "session_id": session_id}


@mcp.resource("apes://rule_status")
async def get_rule_status() -> dict[str, Any]:
    """Get current rule effectiveness and status.

    This resource provides real-time information about rule performance,
    effectiveness scores, and active experiments.

    Returns:
        Current rule status and effectiveness metrics
    """
    try:
        async with get_session() as db_session:
            # Get rule effectiveness stats
            rule_stats = await analytics_service.get_rule_effectiveness(
                days=7, min_usage_count=10, db_session=db_session
            )

            # Get active rules metadata
            rule_metadata = await prompt_service.get_rules_metadata(
                enabled_only=True, db_session=db_session
            )

            return {
                "active_rules": len(rule_metadata),
                "rule_effectiveness": [
                    {
                        "rule_id": stat.rule_id,
                        "effectiveness_score": stat.effectiveness_score,
                        "usage_count": stat.usage_count,
                        "improvement_rate": stat.improvement_rate,
                    }
                    for stat in rule_stats
                ],
                "last_updated": time.time(),
                "status": "operational",
            }

    except Exception as e:
        return {"status": "error", "error": str(e), "active_rules": 0}


@mcp.resource("apes://session_store/status")
async def get_session_store_status() -> dict[str, Any]:
    """Get session store statistics and status.

    Returns:
        Session store statistics including size, TTL, and cleanup status
    """
    try:
        stats = await session_store.stats()
        return {
            "status": "operational",
            "session_stats": stats,
            "last_updated": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.resource("apes://health/live")
async def health_live() -> dict[str, Any]:
    """Check if the services are live with event loop latency."""
    try:
        # Measure event loop latency
        start_time = time.time()
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0)  # Yield control to measure loop responsiveness
        event_loop_latency = (time.time() - start_time) * 1000  # Convert to ms

        # Check background task manager queue size
        task_manager = get_background_task_manager()
        background_queue_size = task_manager.get_queue_size()

        # Use global batch processor to check queue size
        training_queue_size = await get_training_queue_size(batch_processor)

        return {
            "status": "live",
            "event_loop_latency_ms": event_loop_latency,
            "training_queue_size": training_queue_size,
            "background_queue_size": background_queue_size,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.resource("apes://health/ready")
async def health_ready() -> dict[str, Any]:
    """Check if the services are ready with DB connectivity and comprehensive checks."""
    try:
        # Database connectivity check
        db_start_time = time.time()
        async with get_session() as db_session:
            db_result = await db_session.execute("SELECT 1")
            db_ready = db_result.fetchone() is not None
        db_check_time = (time.time() - db_start_time) * 1000

        # Event loop latency check
        loop_start_time = time.time()
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0)
        event_loop_latency = (time.time() - loop_start_time) * 1000

        # Training queue size check
        training_queue_size = await get_training_queue_size(batch_processor)

        # Determine overall readiness
        ready = (
            db_ready and event_loop_latency < 100
        )  # Less than 100ms latency threshold

        return {
            "status": "ready" if ready else "not ready",
            "db_connectivity": {
                "ready": db_ready,
                "response_time_ms": db_check_time,
            },
            "event_loop_latency_ms": event_loop_latency,
            "training_queue_size": training_queue_size,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.resource("apes://health/queue")
async def health_queue() -> dict[str, Any]:
    """Check queue health with comprehensive metrics including length, retry backlog, and latency."""
    try:
        from ..performance.monitoring.health import get_health_service

        # Get health service and ensure queue checker is available
        health_service = get_health_service()

        # Ensure queue checker is loaded (handle circular import issues)
        if not health_service.ensure_queue_checker():
            return {
                "status": "warning",
                "error": "Queue health checker not available due to import restrictions",
                "message": "Queue health monitoring temporarily unavailable",
                "timestamp": time.time(),
                "queue_length": 0,
                "retry_backlog": 0,
                "avg_latency_ms": 0.0,
            }

        # Run queue-specific check
        queue_result = await health_service.run_specific_check("queue")

        # Format response in standardized health format
        response = {
            "status": queue_result.status.value,
            "message": queue_result.message,
            "timestamp": queue_result.timestamp.isoformat()
            if queue_result.timestamp
            else time.time(),
        }

        # Add response time if available
        if queue_result.response_time_ms is not None:
            response["response_time_ms"] = queue_result.response_time_ms

        # Add error information if present
        if queue_result.error:
            response["error"] = queue_result.error

        # Add detailed metrics if available
        if queue_result.details:
            response["metrics"] = queue_result.details

            # Extract key metrics to top level for easy access
            details = queue_result.details
            response["queue_length"] = details.get("total_queue_backlog", 0)
            response["retry_backlog"] = details.get("retry_count", 0)
            response["avg_latency_ms"] = details.get("avg_processing_latency_ms", 0.0)
            response["capacity_utilization"] = details.get(
                "queue_capacity_utilization", 0.0
            )
            response["success_rate"] = details.get("success_rate", 1.0)
            response["throughput_per_second"] = details.get(
                "throughput_per_second", 0.0
            )

        return response

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Queue health check failed",
            "timestamp": time.time(),
            "queue_length": 0,
            "retry_backlog": 0,
            "avg_latency_ms": 0.0,
        }


async def _store_prompt_data(
    original: str,
    enhanced: str,
    metrics: dict[str, Any],
    session_id: str | None,
    priority: int,
) -> None:
    """Internal helper to store prompt data asynchronously via BatchProcessor."""
    try:
        # Create record for batch processing
        record = {
            "original": original,
            "enhanced": enhanced,
            "metrics": metrics,
            "session_id": session_id,
            "data_source": "real",
            "priority": priority,
        }

        # Enqueue with priority for batch processing
        await batch_processor.enqueue(record, priority)

    except Exception as e:
        # Log error but don't raise - this is async background task
        logger.error(f"Error enqueueing prompt data: {e}")


# Helper function to check batch processor queue sizes
async def get_training_queue_size(batch_processor: BatchProcessor) -> int:
    """Get the size of the training queue."""
    return batch_processor.get_queue_size()


# Startup initialization
async def initialize_event_loop_optimization():
    """Initialize event loop optimization and run startup benchmark."""
    try:
        # Setup uvloop if available
        uvloop_enabled = setup_uvloop()
        if uvloop_enabled:
            logger.info("uvloop enabled for enhanced performance")
        else:
            logger.info("Using standard asyncio event loop")

        # Start cache subscriber for pattern.invalidate events
        try:
            await start_cache_subscriber()
            logger.info("Cache subscriber started for pattern.invalidate events")
        except Exception as e:
            logger.warning(f"Cache subscriber initialization failed: {e}")

        # Run startup benchmark
        benchmark_results = await run_startup_benchmark()

        logger.info(
            f"Event loop benchmark: {benchmark_results['avg_ms']:.2f}ms average latency"
        )

        if benchmark_results["meets_target"]:
            logger.info("Event loop meets <200ms target latency")
        else:
            logger.warning("Event loop exceeds 200ms target latency")

        return benchmark_results

    except Exception as e:
        logger.error(f"Event loop initialization failed: {e}")
        return None


# Add new MCP tool for event loop benchmarking
@mcp.tool()
async def benchmark_event_loop(
    operation_type: str = Field(
        default="sleep_yield", description="Type of benchmark operation"
    ),
    samples: int = Field(default=100, description="Number of samples to collect"),
) -> dict[str, Any]:
    """Run event loop benchmark to measure performance.

    Args:
        operation_type: Type of operation to benchmark
        samples: Number of samples to collect
        ctx: MCP context (provided by framework)

    Returns:
        Benchmark results with latency statistics
    """
    try:
        from prompt_improver.utils.event_loop_benchmark import EventLoopBenchmark

        benchmark = EventLoopBenchmark()
        results = await benchmark.run_latency_benchmark(
            samples=samples, operation_type=operation_type
        )

        return results

    except Exception as e:
        return {"status": "error", "error": str(e)}


# Add new MCP resource for event loop status
@mcp.resource("apes://event_loop/status")
async def get_event_loop_status() -> dict[str, Any]:
    """Get current event loop status and performance metrics.

    Returns:
        Event loop status including type, performance, and session metrics
    """
    try:
        loop_manager = get_event_loop_manager()
        session_manager = get_session_manager()

        # Get loop information
        loop_info = loop_manager.get_loop_info()

        # Get session statistics
        session_count = session_manager.get_session_count()

        # Run quick performance check
        performance_metrics = await loop_manager.benchmark_loop_latency(samples=10)

        return {
            "status": "operational",
            "loop_info": loop_info,
            "performance": performance_metrics,
            "sessions": {
                "active_count": session_count,
                "manager_available": True,
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def run_performance_benchmark(
    samples_per_operation: int = Field(default=50, description="Number of samples per operation"),
    include_validation: bool = Field(default=True, description="Include performance validation")
) -> dict[str, Any]:
    """Run comprehensive performance benchmark and validation.

    This tool runs a complete performance benchmark to validate that the
    <200ms response time target is achieved across all MCP operations.

    Args:
        samples_per_operation: Number of samples to collect per operation
        include_validation: Whether to include full validation suite

    Returns:
        Comprehensive benchmark results and performance metrics
    """
    try:
        from prompt_improver.utils.performance_validation import run_performance_validation
        from prompt_improver.utils.performance_benchmark import run_mcp_performance_benchmark

        logger.info(f"Starting performance benchmark with {samples_per_operation} samples")

        # Run baseline benchmark
        baseline_results = await run_mcp_performance_benchmark(samples_per_operation)

        # Run validation if requested
        validation_results = None
        if include_validation:
            validation_results = await run_performance_validation(samples_per_operation)

        # Get current performance stats
        monitor = get_performance_monitor()
        current_stats = monitor.get_current_performance_status()

        # Get optimization stats
        cache_stats = get_cache("prompt").get_performance_stats()

        return {
            "benchmark_timestamp": time.time(),
            "samples_per_operation": samples_per_operation,
            "baseline_results": {
                name: baseline.to_dict() if hasattr(baseline, 'to_dict') else str(baseline)
                for name, baseline in baseline_results.items()
            },
            "validation_results": validation_results,
            "current_performance": current_stats,
            "cache_performance": cache_stats,
            "target_compliance": {
                "target_ms": 200,
                "operations_meeting_target": sum(
                    1 for baseline in baseline_results.values()
                    if hasattr(baseline, 'meets_target') and baseline.meets_target(200)
                ),
                "total_operations": len(baseline_results)
            },
            "optimization_summary": {
                "optimizations_enabled": [
                    "uvloop_event_loop",
                    "multi_level_caching",
                    "database_optimization",
                    "response_compression",
                    "async_optimization",
                    "performance_monitoring"
                ],
                "performance_grade": current_stats.get("performance_grade", "N/A")
            }
        }

    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        return {
            "error": str(e),
            "benchmark_timestamp": time.time(),
            "status": "failed"
        }


@mcp.tool()
async def get_performance_status() -> dict[str, Any]:
    """Get current performance status and optimization metrics.

    Returns:
        Current performance metrics, cache statistics, and optimization status
    """
    try:
        # Get performance monitor status
        monitor = get_performance_monitor()
        performance_status = monitor.get_current_performance_status()

        # Get cache performance
        cache_stats = get_cache("prompt").get_performance_stats()

        # Get response optimizer stats
        from prompt_improver.utils.response_optimizer import get_response_optimizer
        response_stats = get_response_optimizer().get_optimization_stats()

        # Get active alerts
        active_alerts = monitor.get_active_alerts()

        return {
            "timestamp": time.time(),
            "performance_status": performance_status,
            "cache_performance": cache_stats,
            "response_optimization": response_stats,
            "active_alerts": active_alerts,
            "optimization_health": {
                "meets_200ms_target": performance_status.get("meets_200ms_target", False),
                "cache_hit_rate": cache_stats.get("overall_hit_rate", 0),
                "error_rate": performance_status.get("error_rate_percent", 0),
                "performance_grade": performance_status.get("performance_grade", "N/A")
            }
        }

    except Exception as e:
        logger.error(f"Failed to get performance status: {e}")
        return {"error": str(e), "timestamp": time.time()}


# ML Pipeline Orchestrator Tools (Phase 6)
@mcp.tool()
async def get_orchestrator_status() -> dict[str, Any]:
    """Get ML Pipeline Orchestrator status and component information.
    
    Returns:
        Orchestrator state, loaded components, and workflow information
    """
    try:
        # Import orchestrator
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        # Initialize orchestrator
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Get basic status
        status = {
            "timestamp": time.time(),
            "state": orchestrator.state.value,
            "initialized": orchestrator._is_initialized,
            "active_workflows": len(orchestrator.active_workflows),
        }
        
        # If initialized, get detailed information
        if orchestrator._is_initialized:
            components = orchestrator.get_loaded_components()
            history = orchestrator.get_invocation_history()
            
            status.update({
                "loaded_components": len(components),
                "component_list": components,
                "recent_invocations": len(history),
                "success_rate": sum(1 for inv in history if inv["success"]) / len(history) if history else 0.0,
                "component_health": {comp: True for comp in components}  # Simplified health check
            })
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get orchestrator status: {e}")
        return {"error": str(e), "timestamp": time.time()}


@mcp.tool()
async def initialize_orchestrator() -> dict[str, Any]:
    """Initialize the ML Pipeline Orchestrator and load all components.
    
    Returns:
        Initialization result with loaded component information
    """
    try:
        # Import orchestrator
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        # Initialize orchestrator
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Initialize if not already done
        if not orchestrator._is_initialized:
            await orchestrator.initialize()
        
        # Get results
        components = orchestrator.get_loaded_components()
        
        return {
            "timestamp": time.time(),
            "success": True,
            "state": orchestrator.state.value,
            "loaded_components": len(components),
            "component_list": components,
            "message": f"Orchestrator initialized with {len(components)} components"
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        return {"error": str(e), "timestamp": time.time(), "success": False}


@mcp.tool()
async def run_ml_training_workflow(training_data: str = "sample training data") -> dict[str, Any]:
    """Run a complete ML training workflow using the orchestrator.
    
    Args:
        training_data: Input training data for the workflow
        
    Returns:
        Training workflow results and performance metrics
    """
    try:
        # Import orchestrator
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        # Initialize orchestrator
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Initialize if not done
        if not orchestrator._is_initialized:
            await orchestrator.initialize()
        
        # Run training workflow
        start_time = time.time()
        results = await orchestrator.run_training_workflow(training_data)
        execution_time = time.time() - start_time
        
        # Get workflow history
        history = orchestrator.get_invocation_history()
        recent_history = history[-10:]  # Last 10 invocations
        
        return {
            "timestamp": time.time(),
            "success": True,
            "execution_time": execution_time,
            "workflow_results": results,
            "steps_completed": len(results),
            "recent_invocations": len(recent_history),
            "invocation_details": recent_history,
            "message": f"Training workflow completed in {execution_time:.2f}s"
        }
        
    except Exception as e:
        logger.error(f"Training workflow failed: {e}")
        return {"error": str(e), "timestamp": time.time(), "success": False}


@mcp.tool()
async def run_ml_evaluation_workflow(evaluation_data: str = "sample evaluation data") -> dict[str, Any]:
    """Run a complete ML evaluation workflow using the orchestrator.
    
    Args:
        evaluation_data: Input evaluation data for the workflow
        
    Returns:
        Evaluation workflow results and analysis
    """
    try:
        # Import orchestrator
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        # Initialize orchestrator
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Initialize if not done
        if not orchestrator._is_initialized:
            await orchestrator.initialize()
        
        # Run evaluation workflow
        start_time = time.time()
        results = await orchestrator.run_evaluation_workflow(evaluation_data)
        execution_time = time.time() - start_time
        
        return {
            "timestamp": time.time(),
            "success": True,
            "execution_time": execution_time,
            "evaluation_results": results,
            "analysis_steps": len(results),
            "message": f"Evaluation workflow completed in {execution_time:.2f}s"
        }
        
    except Exception as e:
        logger.error(f"Evaluation workflow failed: {e}")
        return {"error": str(e), "timestamp": time.time(), "success": False}


@mcp.tool()
async def invoke_ml_component(component_name: str, method_name: str, **kwargs) -> dict[str, Any]:
    """Invoke a specific method on a loaded ML component.
    
    Args:
        component_name: Name of the ML component to invoke
        method_name: Method name to call on the component
        **kwargs: Additional arguments for the method
        
    Returns:
        Component invocation result and execution details
    """
    try:
        # Import orchestrator
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        # Initialize orchestrator
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Initialize if not done
        if not orchestrator._is_initialized:
            await orchestrator.initialize()
        
        # Check if component is loaded
        components = orchestrator.get_loaded_components()
        if component_name not in components:
            return {
                "error": f"Component '{component_name}' not found",
                "available_components": components,
                "timestamp": time.time(),
                "success": False
            }
        
        # Get available methods
        methods = orchestrator.get_component_methods(component_name)
        if method_name not in methods:
            return {
                "error": f"Method '{method_name}' not found on component '{component_name}'",
                "available_methods": methods,
                "timestamp": time.time(),
                "success": False
            }
        
        # Invoke the component method
        start_time = time.time()
        result = await orchestrator.invoke_component(component_name, method_name, **kwargs)
        execution_time = time.time() - start_time
        
        return {
            "timestamp": time.time(),
            "success": True,
            "component_name": component_name,
            "method_name": method_name,
            "execution_time": execution_time,
            "result": result,
            "message": f"Successfully invoked {component_name}.{method_name}"
        }
        
    except Exception as e:
        logger.error(f"Component invocation failed: {e}")
        return {"error": str(e), "timestamp": time.time(), "success": False}


# Main entry point for stdio transport
if __name__ == "__main__":
    # Setup event loop optimization before running server
    import asyncio

    async def main():
        # Initialize event loop optimization
        await initialize_event_loop_optimization()

        # Additional startup tasks can be added here
        logger.info("Starting APES MCP Server with stdio transport...")

        # Note: mcp.run() is synchronous, so we can't easily integrate it here
        # The uvloop setup happens before this point

    # Run startup initialization
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}")

    # Run the MCP server with stdio transport
    logger.info("APES MCP Server ready with optimized event loop")
    mcp.run(transport="stdio")
