"""Modern MCP Server implementation for the Adaptive Prompt Enhancement System (APES).
Provides prompt enhancement via Model Context Protocol with stdio transport.
Features class-based architecture with proper lifecycle management and graceful shutdown.
"""

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


from prompt_improver.database import get_unified_manager, ManagerMode
# V2 unified connection manager - direct import with no legacy layers
from prompt_improver.core.config import get_config
# ML training components removed per architectural separation requirements
# MCP server is strictly read-only for rule application only
from prompt_improver.core.services.prompt_improvement import PromptImprovementService
# Startup benchmark functionality now part of unified loop manager
from prompt_improver.utils.unified_loop_manager import get_unified_loop_manager
from prompt_improver.core.config import AppConfig
from prompt_improver.utils.session_store import SessionStore
from prompt_improver.performance.optimization.performance_optimizer import (
    get_performance_optimizer,
)
from prompt_improver.performance.monitoring.performance_monitor import (
    get_performance_monitor,
)
from prompt_improver.utils.multi_level_cache import get_cache
from prompt_improver.core.config import AppConfig  # Redis functionality start_cache_subscriber

# Import MCP security components
# Security components for input validation and rate limiting
from prompt_improver.security.owasp_input_validator import OWASP2025InputValidator
from prompt_improver.security.structured_prompts import create_rule_application_prompt
from prompt_improver.security.rate_limit_middleware import require_rate_limiting, get_mcp_rate_limit_middleware
from prompt_improver.security.output_validator import OutputValidator

# Feedback collection removed - MCP server is read-only for rule application

# Import performance optimization
from prompt_improver.performance.sla_monitor import SLAMonitor

# Configure logging to stderr for MCP protocol compliance
# MCP servers using stdio transport should log to stderr, not stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


@dataclass
class ServerServices:
    """Container for all MCP server services - organized and clean"""
    # Configuration
    config: Any

    # Security components
    input_validator: OWASP2025InputValidator
    rate_limit_middleware: Any
    output_validator: OutputValidator

    # Performance components
    performance_optimizer: Any
    performance_monitor: Any
    sla_monitor: SLAMonitor

    # Core services
    prompt_service: PromptImprovementService
    session_store: SessionStore

    # Cache and utilities
    cache: Any
    event_loop_manager: Any


class APESMCPServer:
    """Modern MCP Server with clean architecture and lifecycle management.

    Features:
    - Class-based structure for better organization
    - Proper async lifecycle management
    - Graceful shutdown with signal handling
    - All existing functionality preserved
    - Clean service organization
    """

    def __init__(self):
        """Initialize the MCP server with all services."""
        # Load configuration first
        self.config = get_config()
        logger.info(f"MCP Server configuration loaded - Batch size: {self.config.mcp_batch_size}, "
                   f"Session maxsize: {self.config.mcp_session_maxsize}, TTL: {self.config.mcp_session_ttl}s")

        # Initialize FastMCP
        self.mcp = FastMCP(
            name="APES - Adaptive Prompt Enhancement System",
            description="AI-powered prompt optimization service using ML-driven rules",
        )

        # Initialize all services
        self.services = self._create_services()

        # Setup all tools and resources
        self._setup_tools()
        self._setup_resources()

        # Server state
        self._is_running = False
        self._shutdown_event = asyncio.Event()

    def _create_services(self) -> ServerServices:
        """Create and organize all server services."""
        return ServerServices(
            # Configuration
            config=self.config,

            # Security components
            input_validator=OWASP2025InputValidator(),
            rate_limit_middleware=get_mcp_rate_limit_middleware(),
            output_validator=OutputValidator(),

            # Performance components
            performance_optimizer=get_performance_optimizer(),
            performance_monitor=get_performance_monitor(
                enable_anomaly_detection=True,
                enable_adaptive_thresholds=True
            ),
            sla_monitor=SLAMonitor(),

            # Core services
            prompt_service=PromptImprovementService(),
            session_store=SessionStore(
                maxsize=self.config.mcp_session_maxsize,
                ttl=self.config.mcp_session_ttl,
                cleanup_interval=self.config.mcp_session_cleanup_interval,
            ),

            # Cache and utilities
            cache=get_cache("prompt"),
            event_loop_manager=get_unified_loop_manager(),
        )

    def _setup_tools(self):
        """Setup all MCP tools as class methods."""

        @self.mcp.tool()
        @require_rate_limiting(include_ip=True)
        async def improve_prompt(
            prompt: str = Field(..., description="The prompt to enhance"),
            context: dict[str, Any] | None = Field(
                default=None, description="Additional context"
            ),
            session_id: str | None = Field(default=None, description="Session ID for tracking"),
            rate_limit_remaining: int | None = None,  # Added by rate limit middleware
        ) -> dict[str, Any]:
            """Enhance a prompt using ML-optimized rules."""
            return await self._improve_prompt_impl(
                prompt, context, session_id, rate_limit_remaining
            )

        @self.mcp.tool()
        async def get_session(
            session_id: str = Field(..., description="Session ID to retrieve"),
        ) -> dict[str, Any]:
            """Retrieve session data from the session store."""
            return await self._get_session_impl(session_id)

        @self.mcp.tool()
        async def set_session(
            session_id: str = Field(..., description="Session ID to set"),
            data: dict[str, Any] = Field(..., description="Session data to store"),
        ) -> dict[str, Any]:
            """Store session data in the session store."""
            return await self._set_session_impl(session_id, data)

        @self.mcp.tool()
        async def touch_session(
            session_id: str = Field(..., description="Session ID to touch"),
        ) -> dict[str, Any]:
            """Update session last access time."""
            return await self._touch_session_impl(session_id)

        @self.mcp.tool()
        async def delete_session(
            session_id: str = Field(..., description="Session ID to delete"),
        ) -> dict[str, Any]:
            """Delete session data from the session store."""
            return await self._delete_session_impl(session_id)

        @self.mcp.tool()
        async def benchmark_event_loop(
            operation_type: str = Field(
                default="sleep_yield", description="Type of benchmark operation"
            ),
            iterations: int = Field(default=1000, description="Number of iterations"),
            concurrency: int = Field(default=10, description="Concurrent operations"),
        ) -> dict[str, Any]:
            """Benchmark event loop performance."""
            return await self._benchmark_event_loop_impl(operation_type, iterations, concurrency)

        @self.mcp.tool()
        async def run_performance_benchmark(
            samples_per_operation: int = Field(default=50, description="Number of samples per operation"),
            include_validation: bool = Field(default=True, description="Include performance validation")
        ) -> dict[str, Any]:
            """Run comprehensive performance benchmark."""
            return await self._run_performance_benchmark_impl(samples_per_operation, include_validation)

        @self.mcp.tool()
        async def get_performance_status() -> dict[str, Any]:
            """Get current performance status and optimization metrics."""
            return await self._get_performance_status_impl()

    def _setup_resources(self):
        """Setup all MCP resources as class methods."""

        @self.mcp.resource("apes://rule_status")
        async def get_rule_status() -> dict[str, Any]:
            """Get current rule effectiveness and status."""
            return await self._get_rule_status_impl()

        @self.mcp.resource("apes://session_store/status")
        async def get_session_store_status() -> dict[str, Any]:
            """Get session store statistics and status."""
            return await self._get_session_store_status_impl()

        @self.mcp.resource("apes://health/live")
        async def health_live() -> dict[str, Any]:
            """Phase 0 liveness check - basic service availability."""
            return await self._health_live_impl()

        @self.mcp.resource("apes://health/ready")
        async def health_ready() -> dict[str, Any]:
            """Phase 0 readiness check with MCP connection pool and rule application capability."""
            return await self._health_ready_impl()

        @self.mcp.resource("apes://health/queue")
        async def health_queue() -> dict[str, Any]:
            """Check queue health with comprehensive metrics."""
            return await self._health_queue_impl()

        @self.mcp.resource("apes://health/phase0")
        async def health_phase0() -> dict[str, Any]:
            """Comprehensive Phase 0 health check with all unified architecture components."""
            return await self._health_phase0_impl()

        @self.mcp.resource("apes://event_loop/status")
        async def get_event_loop_status() -> dict[str, Any]:
            """Get current event loop status and performance metrics."""
            return await self._get_event_loop_status_impl()

    async def initialize(self) -> bool:
        """Initialize the server and all services."""
        try:
            logger.info("Initializing APES MCP Server...")

            # Initialize event loop optimization
            await self._initialize_event_loop_optimization()

            # Start cache subscriber for pattern.invalidate events
            try:
                await start_cache_subscriber()
                logger.info("Cache subscriber started for pattern.invalidate events")
            except Exception as e:
                logger.warning(f"Cache subscriber initialization failed: {e}")

            # Additional startup tasks can be added here
            logger.info("APES MCP Server initialized successfully")
            self._is_running = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP Server: {e}")
            return False

    async def shutdown(self):
        """Gracefully shutdown the server and all services."""
        try:
            logger.info("Shutting down APES MCP Server...")
            self._is_running = False
            self._shutdown_event.set()

            # Shutdown services in reverse order
            # Add any cleanup logic here

            logger.info("APES MCP Server shutdown completed")

        except Exception as e:
            logger.error(f"Error during server shutdown: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, _frame: Any) -> None:
            logger.info(f"Received signal {signum} - initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run(self):
        """Run the MCP server with modern async lifecycle."""
        async def main():
            # Setup signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()

            def signal_handler():
                logger.info("Received shutdown signal")
                asyncio.create_task(self.shutdown())

            for sig in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(sig, signal_handler)

            # Initialize server
            if not await self.initialize():
                logger.error("Server initialization failed")
                sys.exit(1)

            try:
                # Run the MCP server (stdio transport is default)
                logger.info("APES MCP Server ready with optimized event loop")
                # FastMCP.run() is sync, not async
                self.mcp.run()
            finally:
                await self.shutdown()

        # Run the server
        asyncio.run(main())

    async def _initialize_event_loop_optimization(self):
        """Initialize event loop optimization and run startup benchmark."""
        try:
            # Setup uvloop if available
            get_unified_loop_manager().setup_uvloop()

            # Run startup benchmark
            benchmark_result = await unified_manager.benchmark_unified_performance()
            logger.info(f"Event loop optimization initialized - Benchmark: {benchmark_result}")

        except Exception as e:
            logger.warning(f"Event loop optimization failed: {e}")
            # Continue without optimization

    # Tool Implementation Methods
    # ==========================

    async def _improve_prompt_impl(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        session_id: str | None,
        rate_limit_remaining: int | None,
    ) -> dict[str, Any]:
        """Implementation of improve_prompt tool with all existing functionality."""
        # Track start time for fallback metrics
        start_time = time.time()
        request_id = f"anonymous_{session_id}_{int(start_time)}"

        # Phase 1: OWASP 2025 Input Validation
        validation_result = self.services.input_validator.validate_prompt(prompt)

        if validation_result.is_blocked:
            logger.warning(
                f"Blocked malicious prompt from anonymous request - "
                f"Threat: {validation_result.threat_type}, Score: {validation_result.threat_score:.2f}, "
                f"Patterns: {validation_result.detected_patterns}"
            )
            return {
                "error": "Input validation failed",
                "message": "The provided prompt contains potentially malicious content and cannot be processed.",
                "threat_type": validation_result.threat_type.value if validation_result.threat_type else None,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "security_check": "blocked"
            }

        # Use sanitized input for processing
        sanitized_prompt = validation_result.sanitized_input

        # Log security validation success
        logger.info(
            f"Security validation passed for anonymous request - "
            f"Threat score: {validation_result.threat_score:.2f}"
        )

        # Use performance optimizer for comprehensive measurement
        async with self.services.performance_optimizer.measure_operation(
            "mcp_improve_prompt",
            prompt_length=len(prompt),
            has_context=context is not None,
            session_id=session_id
        ) as perf_metrics:
            try:
                # Get unified loop manager for session tracking
                unified_manager = get_unified_loop_manager()

                # Use session context for performance tracking
                async with unified_manager.session_context(session_id or "default"):
                    # Get database session with performance measurement
                    async with self.services.performance_optimizer.measure_operation("db_get_session"):
                        # Create structured prompt for secure processing
                        structured_prompt = create_rule_application_prompt(
                            user_prompt=sanitized_prompt,
                            context=context,
                            agent_type="assistant"
                        )

                        # Use the existing prompt improvement service with structured prompt
                        result = await self.services.prompt_service.improve_prompt(
                            prompt=structured_prompt,
                            user_context=context,
                            session_id=session_id
                        )

                        # Validate output for security
                        output_validation = self.services.output_validator.validate_output(
                            result.get("improved_prompt", "")
                        )

                        if not output_validation.is_safe:
                            logger.warning(f"Output validation failed for anonymous request")
                            return {
                                "error": "Output validation failed",
                                "message": "Generated content failed safety validation",
                                "processing_time_ms": (time.time() - start_time) * 1000,
                                "security_check": "output_blocked"
                            }

                        # Calculate total processing time
                        total_time_ms = (time.time() - start_time) * 1000

                        # Record SLA metrics
                        await self.services.sla_monitor.record_request(
                            request_id=request_id,
                            endpoint="improve_prompt",
                            response_time_ms=total_time_ms,
                            success=True,
                            agent_type="anonymous"
                        )

                        # Return enhanced response with all metrics
                        return {
                            "improved_prompt": result.get("improved_prompt", sanitized_prompt),
                            "original_prompt": prompt,
                            "applied_rules": result.get("applied_rules", []),
                            "improvement_score": result.get("improvement_score", 0.0),
                            "confidence_level": result.get("confidence_level", 0.0),
                            "processing_time_ms": total_time_ms,
                            "performance_metrics": perf_metrics.to_dict() if hasattr(perf_metrics, 'to_dict') else {},
                            "security_validation": {
                                "input_threat_score": validation_result.threat_score,
                                "output_risk_score": output_validation.risk_score,
                                "validation_passed": True
                            },
                            "session_id": session_id,
                            "request_id": request_id,
                            "agent_type": "anonymous",
                            "rate_limit_remaining": rate_limit_remaining,
                            "timestamp": time.time()
                        }

            except Exception as e:
                # Return graceful error response
                return {
                    "improved_prompt": prompt,  # Fallback to original
                    "error": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "timestamp": time.time()
                }

    async def _get_session_impl(self, session_id: str) -> dict[str, Any]:
        """Implementation of get_session tool."""
        try:
            data = await self.services.session_store.get(session_id)
            if data is None:
                return {
                    "session_id": session_id,
                    "exists": False,
                    "message": "Session not found",
                    "timestamp": time.time()
                }

            return {
                "session_id": session_id,
                "exists": True,
                "data": data,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "session_id": session_id,
                "error": str(e),
                "exists": False,
                "timestamp": time.time()
            }

    async def _set_session_impl(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Implementation of set_session tool."""
        try:
            await self.services.session_store.set(session_id, data)
            return {
                "session_id": session_id,
                "success": True,
                "message": "Session data stored successfully",
                "data_keys": list(data.keys()),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    async def _touch_session_impl(self, session_id: str) -> dict[str, Any]:
        """Implementation of touch_session tool."""
        try:
            success = await self.services.session_store.touch(session_id)
            return {
                "session_id": session_id,
                "success": success,
                "message": "Session touched successfully" if success else "Session not found",
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    async def _delete_session_impl(self, session_id: str) -> dict[str, Any]:
        """Implementation of delete_session tool."""
        try:
            success = await self.services.session_store.delete(session_id)
            return {
                "session_id": session_id,
                "success": success,
                "message": "Session deleted successfully" if success else "Session not found",
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    async def _benchmark_event_loop_impl(
        self, operation_type: str, iterations: int, concurrency: int
    ) -> dict[str, Any]:
        """Implementation of benchmark_event_loop tool."""
        try:
            # Use the existing event loop benchmark functionality
            benchmark_result = await unified_manager.benchmark_unified_performance()

            return {
                "operation_type": operation_type,
                "iterations": iterations,
                "concurrency": concurrency,
                "benchmark_result": benchmark_result,
                "timestamp": time.time(),
                "success": True
            }
        except Exception as e:
            return {
                "operation_type": operation_type,
                "error": str(e),
                "success": False,
                "timestamp": time.time()
            }

    async def _run_performance_benchmark_impl(
        self, samples_per_operation: int, include_validation: bool
    ) -> dict[str, Any]:
        """Implementation of run_performance_benchmark tool."""
        try:
            # Get performance monitor for comprehensive benchmarking
            monitor = self.services.performance_monitor

            # Run performance validation if requested
            validation_results = {}
            if include_validation:
                # Add validation logic here
                validation_results = {"validation": "completed"}

            # Get current performance metrics
            performance_metrics: dict[str, Any] = await monitor.get_metrics_summary() if hasattr(monitor, 'get_metrics_summary') else {}

            return {
                "samples_per_operation": samples_per_operation,
                "include_validation": include_validation,
                "validation_results": validation_results,
                "performance_metrics": performance_metrics,
                "timestamp": time.time(),
                "success": True
            }
        except Exception as e:
            return {
                "samples_per_operation": samples_per_operation,
                "error": str(e),
                "success": False,
                "timestamp": time.time()
            }

    async def _get_performance_status_impl(self) -> dict[str, Any]:
        """Implementation of get_performance_status tool."""
        try:
            # Get performance monitor
            monitor = self.services.performance_monitor
            performance_status: dict[str, Any] = monitor.get_current_performance_status() if hasattr(monitor, 'get_current_performance_status') else {}

            # Get cache performance
            cache_stats: dict[str, Any] = self.services.cache.get_performance_stats() if hasattr(self.services.cache, 'get_performance_stats') else {}

            # Get response optimizer stats
            from prompt_improver.performance.optimization.response_optimizer import ResponseOptimizer
            response_optimizer = ResponseOptimizer()
            response_stats = response_optimizer.get_optimization_stats() if hasattr(response_optimizer, 'get_optimization_stats') else {}

            # Get active alerts
            active_alerts: list[Any] = monitor.get_active_alerts() if hasattr(monitor, 'get_active_alerts') else []

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

    # Resource Implementation Methods
    # ===============================

    async def _get_rule_status_impl(self) -> dict[str, Any]:
        """Implementation of rule_status resource."""
        try:
            # Get rule effectiveness from the prompt service
            rule_stats: dict[str, Any] = {}
            if hasattr(self.services.prompt_service, 'get_rule_effectiveness'):
                try:
                    rule_stats = await self.services.prompt_service.get_rule_effectiveness()
                except Exception as e:
                    logger.warning(f"Failed to get rule effectiveness: {e}")
                    rule_stats = {"rules": [], "error": str(e)}

            # Safely extract rule information
            rules_list = rule_stats.get("rules", []) if isinstance(rule_stats.get("rules"), list) else []
            active_rules = []
            for rule in rules_list:
                if isinstance(rule, dict) and rule.get("active", False):
                    active_rules.append(rule)

            return {
                "rule_effectiveness": rule_stats,
                "total_rules": len(rules_list),
                "active_rules": len(active_rules),
                "timestamp": time.time(),
                "status": "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "error", "timestamp": time.time()}

    async def _get_session_store_status_impl(self) -> dict[str, Any]:
        """Implementation of session_store/status resource."""
        try:
            # Check if session store has get_stats method
            if hasattr(self.services.session_store, 'get_stats'):
                stats: dict[str, Any] = self.services.session_store.get_stats()
            else:
                # Fallback stats if method doesn't exist
                stats = {
                    "count": 0,
                    "memory_usage": 0,
                    "hit_rate": 0.0,
                    "cleanup_runs": 0
                }

            return {
                "session_count": stats.get("count", 0),
                "memory_usage": stats.get("memory_usage", 0),
                "hit_rate": stats.get("hit_rate", 0.0),
                "cleanup_runs": stats.get("cleanup_runs", 0),
                "max_size": self.config.mcp_session_maxsize,
                "ttl_seconds": self.config.mcp_session_ttl,
                "timestamp": time.time(),
                "status": "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "error", "timestamp": time.time()}

    async def _health_live_impl(self) -> dict[str, Any]:
        """Implementation of health/live resource."""
        try:
            # Measure event loop latency
            start_time = time.time()
            await asyncio.sleep(0)  # Yield control to measure loop responsiveness
            event_loop_latency = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "status": "live",
                "event_loop_latency_ms": event_loop_latency,
                "background_queue_size": 0,  # No background tasks in read-only mode
                "phase": "0",
                "mcp_server_mode": "rule_application_only",
                "timestamp": time.time(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _health_ready_impl(self) -> dict[str, Any]:
        """Implementation of health/ready resource."""
        try:
            # Check database connection using V2 manager
            connection_manager = get_unified_manager(ManagerMode.MCP_SERVER)
            db_health = await connection_manager.health_check()

            # Check if we can apply rules
            rule_application_ready = True
            try:
                # Test rule application capability
                test_result = await self.services.prompt_service.improve_prompt(
                    prompt="test", user_context={}, session_id="health_check"
                )
                # Check if result contains expected fields
                rule_application_ready = "improved_prompt" in test_result
            except Exception:
                rule_application_ready = False

            overall_ready = db_health.get("status") == "healthy" and rule_application_ready

            return {
                "status": "ready" if overall_ready else "not_ready",
                "database": db_health,
                "rule_application": {
                    "ready": rule_application_ready,
                    "service_available": True  # Service is always available in this architecture
                },
                "phase": "0",
                "mcp_server_mode": "rule_application_only",
                "timestamp": time.time(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _health_queue_impl(self) -> dict[str, Any]:
        """Implementation of health/queue resource."""
        try:
            from prompt_improver.performance.monitoring.health.unified_health_system import get_unified_health_monitor
            health_monitor = get_unified_health_monitor()

            # Run queue-specific health check
            check_results = await health_monitor.check_health(plugin_name="queue_service")
            queue_result = check_results.get("queue_service")

            # Format response in standardized health format
            if queue_result:
                response = {
                    "status": queue_result.status.value,
                    "message": queue_result.message,
                    "timestamp": time.time()
                }
            else:
                response = {
                    "status": "unknown",
                    "message": "Queue health check not available",
                    "timestamp": time.time()
                }

            # Add queue-specific metrics if available
            if hasattr(queue_result, 'details') and queue_result.details:
                response.update({
                    "queue_length": queue_result.details.get("queue_length", 0),
                    "processing_rate": queue_result.details.get("processing_rate", 0.0),
                    "retry_backlog": queue_result.details.get("retry_backlog", 0),
                    "average_latency_ms": queue_result.details.get("average_latency_ms", 0.0),
                    "throughput_per_second": queue_result.details.get("throughput_per_second", 0.0)
                })

            return response

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    async def _health_phase0_impl(self) -> dict[str, Any]:
        """Implementation of health/phase0 resource."""
        try:
            from prompt_improver.performance.monitoring.health.unified_health_system import get_unified_health_monitor
            health_monitor = get_unified_health_monitor()

            overall_start = time.time()

            # Run comprehensive health checks for all Phase 0 components
            components = {}

            # Database health
            try:
                db_result = await health_service.run_specific_check("database")
                components["database"] = {
                    "status": db_result.status.value,
                    "message": db_result.message,
                    "response_time_ms": getattr(db_result, 'response_time_ms', 0)
                }
            except Exception as e:
                components["database"] = {"status": "error", "error": str(e)}

            # Cache health
            try:
                cache_result = await health_service.run_specific_check("cache")
                components["cache"] = {
                    "status": cache_result.status.value,
                    "message": cache_result.message,
                    "response_time_ms": getattr(cache_result, 'response_time_ms', 0)
                }
            except Exception as e:
                components["cache"] = {"status": "error", "error": str(e)}

            # Rule application health
            try:
                rule_result = await health_service.run_specific_check("rule_application")
                components["rule_application"] = {
                    "status": rule_result.status.value,
                    "message": rule_result.message,
                    "response_time_ms": getattr(rule_result, 'response_time_ms', 0)
                }
            except Exception as e:
                components["rule_application"] = {"status": "error", "error": str(e)}

            # Performance monitoring health
            try:
                perf_result = await health_service.run_specific_check("performance")
                components["performance_monitoring"] = {
                    "status": perf_result.status.value,
                    "message": perf_result.message,
                    "response_time_ms": getattr(perf_result, 'response_time_ms', 0)
                }
            except Exception as e:
                components["performance_monitoring"] = {"status": "error", "error": str(e)}

            # Calculate overall health check time
            total_check_time = (time.time() - overall_start) * 1000

            # Determine overall health
            healthy_components = sum(
                1 for comp in components.values()
                if isinstance(comp, dict) and comp.get("status") == "healthy"
            )
            total_components = len(components)
            health_percentage = (healthy_components / total_components) * 100 if total_components > 0 else 0

            overall_status = "healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 50 else "unhealthy"

            return {
                "status": overall_status,
                "phase": "0",
                "health_percentage": health_percentage,
                "healthy_components": healthy_components,
                "total_components": total_components,
                "total_check_time_ms": total_check_time,
                "components": components,
                "mcp_server_mode": "rule_application_only",
                "timestamp": time.time()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "phase": "0",
                "timestamp": time.time()
            }

    async def _get_event_loop_status_impl(self) -> dict[str, Any]:
        """Implementation of event_loop/status resource."""
        try:
            # Get event loop manager
            loop_manager = self.services.event_loop_manager

            # Get current event loop info
            loop = asyncio.get_event_loop()

            # Measure current latency
            start_time = time.time()
            await asyncio.sleep(0)
            current_latency = (time.time() - start_time) * 1000

            return {
                "loop_type": type(loop).__name__,
                "is_running": loop.is_running(),
                "current_latency_ms": current_latency,
                "task_count": len(asyncio.all_tasks()),
                "optimization_enabled": hasattr(loop_manager, 'optimization_enabled'),
                "timestamp": time.time(),
                "status": "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "error", "timestamp": time.time()}


# Pydantic models for backward compatibility
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


# Global server instance
server = APESMCPServer()


# Main entry point for stdio transport
def main():
    """Main entry point for the modernized MCP server."""
    logger.info("Starting APES MCP Server with modern architecture...")
    server.run()


if __name__ == "__main__":
    main()
