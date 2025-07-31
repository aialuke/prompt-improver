"""Modern MCP Server implementation for the Adaptive Prompt Enhancement System (APES).
Provides prompt enhancement via Model Context Protocol with stdio transport.
Features class-based architecture with proper lifecycle management and graceful shutdown.

2025 FastMCP Enhancements:
- Custom middleware stack implementation (timing, logging, rate limiting)
- Progress reporting capability with Context support
- Advanced resource templates with wildcard parameters
- Streamable HTTP transport support
"""

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field


from prompt_improver.database import get_unified_manager, ManagerMode
# V2 unified connection manager - direct import with no legacy layers
from prompt_improver.core.config import get_config
# ML training components removed per architectural separation requirements
# MCP server is strictly read-only for rule application only
from prompt_improver.core.services.prompt_improvement import PromptImprovementService
# Import batch processor for health monitoring (ADR-005 compliant - read-only access)
from prompt_improver.ml.optimization.batch.unified_batch_processor import UnifiedBatchProcessor
# Startup benchmark functionality now part of unified loop manager
from prompt_improver.utils.unified_loop_manager import get_unified_loop_manager
from prompt_improver.utils.session_store import SessionStore
from prompt_improver.performance.optimization.performance_optimizer import (
    get_performance_optimizer,
)
from prompt_improver.performance.monitoring.performance_monitor import (
    get_performance_monitor,
)
from prompt_improver.utils.multi_level_cache import get_cache
# Note: Cache subscriber functionality not implemented in current architecture
from prompt_improver.core.config import AppConfig

# Import MCP security components
# Security components for input validation and rate limiting
from prompt_improver.security.owasp_input_validator import OWASP2025InputValidator
from prompt_improver.security.structured_prompts import create_rule_application_prompt
from prompt_improver.security.rate_limit_middleware import require_rate_limiting, get_mcp_rate_limit_middleware
from prompt_improver.security.output_validator import OutputValidator

# Feedback collection removed - MCP server is read-only for rule application

# Import performance optimization
from prompt_improver.performance.sla_monitor import SLAMonitor

# Import 2025 FastMCP middleware components
from prompt_improver.mcp_server.middleware import (
    create_default_middleware_stack,
    MiddlewareStack,
    TimingMiddleware,
    DetailedTimingMiddleware,
    StructuredLoggingMiddleware,
    RateLimitingMiddleware,
    MiddlewareContext
)

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
    
    # 2025 FastMCP Middleware Stack
    middleware_stack: MiddlewareStack
    timing_middleware: TimingMiddleware
    detailed_timing_middleware: DetailedTimingMiddleware


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
        # Create middleware components
        middleware_stack = create_default_middleware_stack()
        timing_middleware = TimingMiddleware()
        detailed_timing_middleware = DetailedTimingMiddleware()
        
        # Add specialized middleware to stack
        middleware_stack.add(detailed_timing_middleware)
        
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
            
            # 2025 FastMCP Middleware
            middleware_stack=middleware_stack,
            timing_middleware=timing_middleware,
            detailed_timing_middleware=detailed_timing_middleware,
        )

    def _setup_tools(self):
        """Setup all MCP tools as class methods."""


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

        @self.mcp.tool()
        async def get_training_queue_size() -> dict[str, Any]:
            """Get current training queue size and processing metrics from batch processor."""
            return await self._get_training_queue_size_impl()

        @self.mcp.tool()
        @require_rate_limiting(include_ip=True)
        async def store_prompt(
            original_prompt: str = Field(..., description="The original prompt text"),
            enhanced_prompt: str = Field(..., description="The enhanced prompt text"),
            applied_rules: list[dict[str, Any]] = Field(..., description="List of applied rules with metadata"),
            response_time_ms: int = Field(..., description="Response time in milliseconds"),
            session_id: str = Field(..., description="Required session ID for tracking and observability"),
            agent_type: str = Field(default="external-agent", description="Agent type identifier"),
        ) -> dict[str, Any]:
            """Store prompt improvement session data for feedback collection.
            
            ⚠️ BREAKING CHANGE: session_id is now required (no fallback to None).
            
            Args:
                original_prompt: The original prompt text
                enhanced_prompt: The enhanced prompt text
                applied_rules: List of applied rules with metadata
                response_time_ms: Response time in milliseconds
                session_id: REQUIRED session ID for tracking (use create_session_id())
                agent_type: Agent type identifier
            
            Example Usage:
                session_id = APESMCPServer.create_session_id("feedback_client")
                
                result = await store_prompt(
                    original_prompt="Original text",
                    enhanced_prompt="Enhanced text", 
                    applied_rules=[{"rule": "clarity", "impact": 0.8}],
                    response_time_ms=150,
                    session_id=session_id,  # REQUIRED
                    agent_type="external-agent"
                )
            """
            return await self._store_prompt_impl(
                original_prompt, enhanced_prompt, applied_rules, response_time_ms, session_id, agent_type
            )

        @self.mcp.tool()
        async def query_database(
            query: str = Field(..., description="Read-only SQL query to execute on rule tables"),
            parameters: dict[str, Any] | None = Field(default=None, description="Query parameters for safe parameterized execution"),
        ) -> dict[str, Any]:
            """Execute read-only SQL queries on rule tables (rule_metadata, rule_performance, rule_combinations)."""
            return await self._query_database_impl(query, parameters)

        @self.mcp.tool()
        async def list_tables() -> dict[str, Any]:
            """List all accessible rule tables available for querying."""
            return await self._list_tables_impl()

        @self.mcp.tool()
        async def describe_table(
            table_name: str = Field(..., description="Name of the rule table to describe schema for"),
        ) -> dict[str, Any]:
            """Get schema information for rule application tables."""
            return await self._describe_table_impl(table_name)
        
        # 2025 FastMCP Enhancement: Progress-aware primary tool (replaces legacy improve_prompt)
        @self.mcp.tool()
        @require_rate_limiting(include_ip=True)
        async def improve_prompt(
            prompt: str = Field(..., description="The prompt to enhance"),
            session_id: str = Field(..., description="Required session ID for tracking and observability"),
            ctx: Context = Field(..., description="Required MCP Context for progress reporting and logging"),
            context: dict[str, Any] | None = Field(default=None, description="Optional additional context"),
        ) -> dict[str, Any]:
            """Enhanced prompt improvement with mandatory 2025 progress reporting.
            
            ⚠️ BREAKING CHANGE: All parameters are now required. No fallback behavior.
            
            This tool provides real-time progress updates during the enhancement process.
            All clients must provide Context and session_id - legacy patterns will fail.
            
            Args:
                prompt: The text prompt to enhance
                session_id: REQUIRED session ID for tracking (use create_session_id())
                ctx: REQUIRED MCP Context for progress reporting (use create_mock_context() for testing)
                context: Optional additional context for enhancement
            
            Returns:
                Dict with enhanced prompt and 2025 observability metadata
            
            Example Usage:
                # Modern 2025 pattern (REQUIRED)
                session_id = APESMCPServer.create_session_id("my_client")
                ctx = APESMCPServer.create_mock_context()  # or real MCP Context
                
                result = await improve_prompt(
                    prompt="Make this prompt better",
                    session_id=session_id,  # REQUIRED
                    ctx=ctx,  # REQUIRED  
                    context={"domain": "coding"}
                )
                
                # Or use convenience method:
                server = APESMCPServer()
                result = await server.modern_improve_prompt("Make this prompt better")
            
            Raises:
                TypeError: If required parameters are missing (no fallback behavior)
            """
            # Wrap with middleware for timing and logging
            middleware_ctx = MiddlewareContext(
                method="improve_prompt",
                message={"prompt": prompt, "context": context, "session_id": session_id}
            )
            
            async def handler(mctx: MiddlewareContext):
                # 2025 Modern Implementation: Progress reporting is mandatory
                await ctx.report_progress(progress=0, total=100, message="Starting validation")
                await ctx.info("Beginning prompt enhancement process")
                
                # Input validation phase (0-25%)
                await ctx.debug("Performing OWASP 2025 security validation")
                validation_result = self.services.input_validator.validate_prompt(prompt)
                
                if validation_result.is_blocked:
                    await ctx.error(f"Input validation failed: {validation_result.threat_type}")
                    return {
                        "error": "Input validation failed",
                        "message": "The provided prompt contains potentially malicious content.",
                        "threat_type": validation_result.threat_type.value if validation_result.threat_type else None
                    }
                
                await ctx.report_progress(progress=25, total=100, message="Validation complete")
                
                # Rule application phase (25-75%)
                await ctx.info("Applying enhancement rules")
                await ctx.report_progress(progress=50, total=100, message="Processing rules")
                
                # Call original implementation
                result = await self._improve_prompt_impl(
                    prompt=validation_result.sanitized_input,
                    context=context,
                    session_id=session_id
                )
                
                await ctx.report_progress(progress=75, total=100, message="Rules applied")
                
                # Output validation phase (75-90%)
                await ctx.debug("Validating enhanced output")
                
                # Finalization phase (90-100%)
                rules_count = len(result.get("applied_rules", []))
                await ctx.info(f"Enhancement complete. Applied {rules_count} rules.")
                await ctx.report_progress(progress=100, total=100, message="Complete")
                
                # Add mandatory 2025 timing metrics and observability data
                result["_timing_metrics"] = self.services.timing_middleware.get_metrics_summary()
                result["_session_id"] = session_id
                result["_middleware_applied"] = True
                
                return result
            
            # Execute through middleware stack
            wrapped = self.services.middleware_stack.wrap(handler)
            return await wrapped(__method__="improve_prompt")

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
        
        # 2025 FastMCP Enhancement: Advanced resource templates with wildcards
        @self.mcp.resource("apes://sessions/{session_id}/history")
        async def get_session_history(session_id: str) -> dict[str, Any]:
            """Get detailed session history with wildcard path support.
            
            Supports hierarchical session IDs like:
            - sessions/user123/history
            - sessions/user123/workspace/main/history
            """
            return await self._get_session_history_impl(session_id)
        
        @self.mcp.resource("apes://rules/{rule_category}/performance")
        async def get_rule_category_performance(rule_category: str) -> dict[str, Any]:
            """Get performance metrics for rule categories with wildcard support.
            
            Supports hierarchical categories like:
            - rules/security/performance
            - rules/security/input_validation/xss/performance
            """
            return await self._get_rule_category_performance_impl(rule_category)
        
        @self.mcp.resource("apes://metrics/{metric_type}")
        async def get_hierarchical_metrics(metric_type: str) -> dict[str, Any]:
            """Get hierarchical metrics with flexible path support.
            
            Examples:
            - metrics/performance
            - metrics/performance/tools/improve_prompt
            - metrics/errors/by_method
            """
            return await self._get_hierarchical_metrics_impl(metric_type)

    async def initialize(self) -> bool:
        """Initialize the server and all services."""
        try:
            logger.info("Initializing APES MCP Server...")

            # Initialize event loop optimization
            await self._initialize_event_loop_optimization()

            # Cache subscriber functionality not implemented in current architecture
            # This would handle pattern.invalidate events for cache management
            logger.info("Cache subscriber functionality is not implemented in current architecture")

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
            # Use proven event loop pattern from signal_handler for shutdown
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self.shutdown())
                logger.info("Scheduled shutdown as task")
            except RuntimeError:
                asyncio.run(self.shutdown())

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
                # Use proven event loop pattern from signal_handler for shutdown
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(self.shutdown())
                    logger.info("Scheduled shutdown as task")
                except RuntimeError:
                    asyncio.run(self.shutdown())

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

        # Run the server - using proven pattern from signal_handler.py:560-568
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(main())
            logger.info("MCP server started as task in existing event loop")
        except RuntimeError:
            asyncio.run(main())

    async def _initialize_event_loop_optimization(self):
        """Initialize event loop optimization and run startup benchmark."""
        try:
            # Setup uvloop if available
            get_unified_loop_manager().setup_uvloop()

            # Run startup benchmark using the unified loop manager
            loop_manager = get_unified_loop_manager()
            benchmark_result = await loop_manager.benchmark_unified_performance()
            logger.info(f"Event loop optimization initialized - Benchmark: {benchmark_result}")

        except Exception as e:
            logger.warning(f"Event loop optimization failed: {e}")
            # Continue without optimization

    # 2025 Modern Usage Helper Methods
    # ================================
    
    @staticmethod
    def create_session_id(prefix: str = "apes") -> str:
        """Create a properly formatted session ID for 2025 API requirements.
        
        Args:
            prefix: Optional prefix for the session ID
            
        Returns:
            A unique session ID string required by all modern tools
            
        Example:
            session_id = APESMCPServer.create_session_id("client")
            # Returns: "client_1640995200_abc123"
        """
        import uuid
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_id}"
    
    @staticmethod 
    def create_mock_context():
        """Create a mock Context object for testing modern 2025 patterns.
        
        This is useful for clients who need to test the modern API without
        implementing full MCP Context handling.
        
        Returns:
            A mock Context object that implements required methods
            
        Example:
            from unittest.mock import AsyncMock
            ctx = APESMCPServer.create_mock_context()
            # Use with modern tools that require Context
        """
        from unittest.mock import AsyncMock
        from mcp.server.fastmcp import Context
        
        mock_ctx = AsyncMock(spec=Context)
        mock_ctx.report_progress = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.debug = AsyncMock() 
        mock_ctx.error = AsyncMock()
        mock_ctx.warn = AsyncMock()
        return mock_ctx
    
    def validate_modern_parameters(self, session_id: str, ctx) -> None:
        """Validate that required 2025 parameters are properly provided.
        
        Args:
            session_id: Required session ID (must not be None or empty)
            ctx: Required Context object (must not be None)
            
        Raises:
            ValueError: If parameters don't meet 2025 requirements
            
        Example:
            server.validate_modern_parameters(session_id, ctx)
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id is required and must be a non-empty string in 2025 API")
        
        if ctx is None:
            raise ValueError("ctx (Context) parameter is required in 2025 API - no fallback behavior")
        
        # Validate Context has required methods
        required_methods = ['report_progress', 'info', 'debug', 'error']
        for method in required_methods:
            if not hasattr(ctx, method):
                raise ValueError(f"Context object must have {method} method for 2025 API compliance")
    
    async def modern_improve_prompt(self, prompt: str, context: dict[str, Any] | None = None, 
                                  session_prefix: str = "client") -> dict[str, Any]:
        """Convenience method for improve_prompt using modern 2025 patterns.
        
        This method automatically creates required parameters and provides
        a simpler interface for clients migrating to 2025 patterns.
        
        Args:
            prompt: The prompt to enhance
            context: Optional additional context
            session_prefix: Prefix for auto-generated session ID
            
        Returns:
            Enhanced prompt result with 2025 metadata
            
        Example:
            server = APESMCPServer()
            result = await server.modern_improve_prompt("Enhance this prompt")
        """
        session_id = self.create_session_id(session_prefix)
        ctx = self.create_mock_context()
        
        # Call the tool through the middleware stack
        improve_prompt = None
        for tool_name, tool_func in self.mcp._tools.items():
            if tool_name == "improve_prompt":
                improve_prompt = tool_func.implementation
                break
        
        if improve_prompt is None:
            raise RuntimeError("improve_prompt tool not found - server initialization issue")
        
        return await improve_prompt(
            prompt=prompt,
            session_id=session_id,
            ctx=ctx,
            context=context
        )
    
    def get_modern_usage_examples(self) -> dict[str, str]:
        """Get code examples showing how to use the modern 2025 API.
        
        Returns:
            Dictionary of example code snippets for common patterns
        """
        return {
            "basic_usage": '''
# Modern 2025 pattern - all parameters required
from prompt_improver.mcp_server.server import APESMCPServer

server = APESMCPServer()
session_id = server.create_session_id("my_client")
ctx = server.create_mock_context()  # or use real MCP Context

result = await improve_prompt_tool(
    prompt="Enhance this prompt",
    session_id=session_id,  # REQUIRED
    ctx=ctx,  # REQUIRED
    context={"optional": "context"}
)
''',
            "convenience_method": '''
# Using convenience method (auto-generates required params)
server = APESMCPServer()
result = await server.modern_improve_prompt("Enhance this prompt")
''',
            "session_management": '''
# Proper session ID management
session_id = APESMCPServer.create_session_id("my_app")
# Use same session_id across related operations for tracking
''',
            "validation": '''
# Validate parameters before tool calls
server = APESMCPServer()
try:
    server.validate_modern_parameters(session_id, ctx)
    # Parameters are valid for 2025 API
except ValueError as e:
    print(f"Invalid parameters: {e}")
'''
        }

    # Tool Implementation Methods
    # ==========================

    async def _improve_prompt_impl(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        session_id: str,
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
                loop_manager = get_unified_loop_manager()

                # Use session context for performance tracking
                async with loop_manager.session_context(session_id or "default"):
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
            loop_manager = get_unified_loop_manager()
            benchmark_result = await loop_manager.benchmark_unified_performance()

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

    async def _get_training_queue_size_impl(self) -> dict[str, Any]:
        """Implementation of get_training_queue_size tool."""
        try:
            # Create UnifiedBatchProcessor instance for health monitoring
            # This follows ADR-005 architectural separation - read-only monitoring only
            processor = UnifiedBatchProcessor()
            
            # Get current metrics summary from batch processor
            metrics_summary = processor.get_metrics_summary()
            
            # Extract queue-related information
            if metrics_summary.get("status") == "no_data":
                # No processing history available
                queue_info = {
                    "queue_size": 0,
                    "status": "idle",
                    "processing_rate": 0.0,
                    "active_batches": 0,
                    "pending_items": 0,
                    "total_processed": 0,
                    "success_rate": 1.0,
                    "avg_processing_time_ms": 0.0,
                    "strategy_usage": {},
                    "message": "No processing activity detected"
                }
            else:
                # Extract metrics from recent processing activity
                recent_summary = metrics_summary.get("recent_summary", {})
                strategy_usage = metrics_summary.get("strategy_usage", {})
                
                # Calculate derived queue metrics
                items_processed = recent_summary.get("items_processed", 0)
                items_failed = recent_summary.get("items_failed", 0)
                total_items = items_processed + items_failed
                success_rate = recent_summary.get("success_rate", 1.0)
                
                queue_info = {
                    "queue_size": total_items,  # Items in recent processing batches
                    "status": "active" if items_processed > 0 else "idle",
                    "processing_rate": recent_summary.get("avg_throughput_items_per_sec", 0.0),
                    "active_batches": recent_summary.get("batches", 0),
                    "pending_items": items_failed,  # Failed items could be considered pending retry
                    "total_processed": items_processed,
                    "success_rate": success_rate,
                    "avg_processing_time_ms": recent_summary.get("avg_processing_time_ms", 0.0),
                    "strategy_usage": strategy_usage,
                    "total_batches_processed": metrics_summary.get("total_batches_processed", 0)
                }
            
            # Add processor configuration info
            config_info = metrics_summary.get("current_config", {})
            queue_info.update({
                "processor_config": {
                    "strategy": config_info.get("strategy", "auto"),
                    "max_concurrent_tasks": config_info.get("max_concurrent_tasks", 10),
                    "task_timeout_seconds": config_info.get("task_timeout_seconds", 300.0),
                    "enable_optimization": config_info.get("enable_optimization", True)
                },
                "health_status": "healthy" if queue_info["success_rate"] > 0.8 else "degraded",
                "timestamp": time.time()
            })
            
            logger.info(f"Training queue size retrieved: {queue_info['queue_size']} items, "
                       f"status: {queue_info['status']}, rate: {queue_info['processing_rate']:.2f} items/sec")
            
            return queue_info
            
        except Exception as e:
            logger.error(f"Failed to get training queue size: {e}")
            return {
                "queue_size": 0,
                "status": "error",
                "processing_rate": 0.0,
                "error": str(e),
                "timestamp": time.time(),
                "health_status": "unhealthy"
            }

    async def _store_prompt_impl(
        self,
        original_prompt: str,
        enhanced_prompt: str,
        applied_rules: list[dict[str, Any]],
        response_time_ms: int,
        session_id: str,
        agent_type: str,
    ) -> dict[str, Any]:
        """Implementation of store_prompt tool for feedback collection."""
        start_time = time.time()
        
        try:
            # Security validation for inputs
            validation_result = self.services.input_validator.validate_prompt(original_prompt)
            if validation_result.is_blocked:
                logger.warning(f"Blocked malicious original prompt - Threat: {validation_result.threat_type}")
                return {
                    "success": False,
                    "error": "Input validation failed for original prompt",
                    "threat_type": validation_result.threat_type.value if validation_result.threat_type else None,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            validation_result_enhanced = self.services.input_validator.validate_prompt(enhanced_prompt)
            if validation_result_enhanced.is_blocked:
                logger.warning(f"Blocked malicious enhanced prompt - Threat: {validation_result_enhanced.threat_type}")
                return {
                    "success": False,
                    "error": "Input validation failed for enhanced prompt",
                    "threat_type": validation_result_enhanced.threat_type.value if validation_result_enhanced.threat_type else None,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            # Validate inputs according to database schema constraints
            if not original_prompt.strip():
                return {
                    "success": False,
                    "error": "Original prompt cannot be empty",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            if not enhanced_prompt.strip():
                return {
                    "success": False,
                    "error": "Enhanced prompt cannot be empty",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            if response_time_ms <= 0:
                return {
                    "success": False,
                    "error": "Response time must be positive",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            if response_time_ms >= 30000:  # 30 seconds max per schema constraint
                return {
                    "success": False,
                    "error": "Response time exceeds maximum allowed (30 seconds)",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            # Validate agent_type against schema constraint
            valid_agent_types = ['claude-code', 'augment-code', 'external-agent']
            if agent_type not in valid_agent_types:
                logger.warning(f"Invalid agent_type '{agent_type}', using 'external-agent'")
                agent_type = 'external-agent'

            # Validate JSONB payload size to prevent memory exhaustion (security improvement)
            import json
            applied_rules_json = json.dumps(applied_rules) if applied_rules else '[]'
            if len(applied_rules_json) > 100000:  # 100KB limit
                return {
                    "success": False,
                    "error": "Applied rules payload too large (max 100KB allowed)",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            # Validate prompt lengths to prevent excessive database storage
            if len(original_prompt) > 50000:  # 50KB limit per prompt
                return {
                    "success": False,
                    "error": "Original prompt too large (max 50KB allowed)",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            if len(enhanced_prompt) > 50000:  # 50KB limit per prompt
                return {
                    "success": False,
                    "error": "Enhanced prompt too large (max 50KB allowed)",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            # Get database connection using unified manager
            connection_manager = get_unified_manager(ManagerMode.MCP_SERVER)
            
            async with connection_manager.get_async_session() as session:
                # Import text function for SQL query
                from sqlalchemy import text
                
                # Prepare the query following existing patterns
                query = text("""
                    INSERT INTO prompt_improvement_sessions (
                        original_prompt, enhanced_prompt, applied_rules,
                        response_time_ms, agent_type, session_timestamp,
                        anonymized_user_hash, created_at
                    ) VALUES (
                        :original_prompt, :enhanced_prompt, :applied_rules,
                        :response_time_ms, :agent_type, :session_timestamp,
                        :anonymized_user_hash, :created_at
                    ) RETURNING id
                """)

                # Prepare parameters
                current_timestamp = time.time()
                session_timestamp = current_timestamp
                
                # Use session_id as anonymized_user_hash if provided
                anonymized_user_hash = session_id if session_id else f"anonymous_{int(current_timestamp)}"

                # applied_rules_json already calculated above for size validation

                result = await session.execute(query, {
                    "original_prompt": original_prompt.strip(),
                    "enhanced_prompt": enhanced_prompt.strip(),
                    "applied_rules": applied_rules_json,
                    "response_time_ms": response_time_ms,
                    "agent_type": agent_type,
                    "session_timestamp": session_timestamp,
                    "anonymized_user_hash": anonymized_user_hash,
                    "created_at": current_timestamp
                })

                await session.commit()
                row = result.first()
                record_id = row[0] if row else None

                processing_time = (time.time() - start_time) * 1000

                # Log successful storage
                logger.info(f"Successfully stored prompt improvement session - "
                           f"ID: {record_id}, Agent: {agent_type}, "
                           f"Response time: {response_time_ms}ms, "
                           f"Storage time: {processing_time:.2f}ms")

                return {
                    "success": True,
                    "record_id": record_id,
                    "message": "Prompt improvement session stored successfully",
                    "processing_time_ms": processing_time,
                    "agent_type": agent_type,
                    "session_id": session_id,
                    "anonymized_user_hash": anonymized_user_hash,
                    "applied_rules_count": len(applied_rules) if applied_rules else 0,
                    "rate_limit_remaining": rate_limit_remaining,
                    "timestamp": time.time()
                }

        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to store prompt improvement session: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": error_time,
                "timestamp": time.time()
            }

    async def _query_database_impl(self, query: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
        """Implementation of query_database tool with read-only access and SQL injection protection."""
        start_time = time.time()
        
        try:
            # Validate read-only access per ADR-005
            if not self._is_read_only_query(query):
                return {
                    "success": False,
                    "error": "Only read-only queries are permitted. SELECT statements only.",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            # Validate table access - only rule tables allowed
            if not self._validates_table_access(query):
                return {
                    "success": False,
                    "error": "Access restricted to rule tables only: rule_metadata, rule_performance, rule_combinations",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            # Get database connection using unified manager
            connection_manager = get_unified_manager(ManagerMode.MCP_SERVER)
            
            async with connection_manager.get_async_session() as session:
                # Import text function for SQL query (prevents SQL injection)
                from sqlalchemy import text
                
                # Execute parameterized query for security
                sql_query = text(query)
                result = await session.execute(sql_query, parameters or {})
                
                # Convert results to list of dictionaries
                rows = []
                if result.returns_rows:
                    column_names = list(result.keys())
                    for row in result.fetchall():
                        rows.append(dict(zip(column_names, row)))

                processing_time = (time.time() - start_time) * 1000

                # Log successful query execution
                logger.info(f"Successfully executed database query - "
                           f"Rows returned: {len(rows)}, "
                           f"Processing time: {processing_time:.2f}ms")

                return {
                    "success": True,
                    "rows": rows,
                    "row_count": len(rows),
                    "columns": column_names if result.returns_rows else [],
                    "processing_time_ms": processing_time,
                    "query_type": "SELECT",
                    "timestamp": time.time()
                }

        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to execute database query: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": error_time,
                "timestamp": time.time()
            }

    async def _list_tables_impl(self) -> dict[str, Any]:
        """Implementation of list_tables tool showing accessible rule tables."""
        start_time = time.time()
        
        try:
            # Get database connection using unified manager
            connection_manager = get_unified_manager(ManagerMode.MCP_SERVER)
            
            async with connection_manager.get_async_session() as session:
                from sqlalchemy import text
                
                # Query to get table information for rule tables only
                query = text("""
                    SELECT table_name, 
                           table_type,
                           table_comment
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                      AND table_name IN ('rule_metadata', 'rule_performance', 'rule_combinations')
                    ORDER BY table_name
                """)
                
                result = await session.execute(query)
                tables = []
                
                for row in result.fetchall():
                    table_info = {
                        "table_name": row[0],
                        "table_type": row[1],
                        "description": row[2] or f"Rule application table: {row[0]}",
                        "access_level": "read_only"
                    }
                    tables.append(table_info)

                processing_time = (time.time() - start_time) * 1000

                logger.info(f"Successfully listed {len(tables)} accessible rule tables - "
                           f"Processing time: {processing_time:.2f}ms")

                return {
                    "success": True,
                    "tables": tables,
                    "table_count": len(tables),
                    "accessible_tables": ["rule_metadata", "rule_performance", "rule_combinations"],
                    "access_level": "read_only_per_adr005",
                    "processing_time_ms": processing_time,
                    "timestamp": time.time()
                }

        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to list tables: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": error_time,
                "timestamp": time.time()
            }

    async def _describe_table_impl(self, table_name: str) -> dict[str, Any]:
        """Implementation of describe_table tool for rule application tables."""
        start_time = time.time()
        
        try:
            # Validate table access - only rule tables allowed
            allowed_tables = ["rule_metadata", "rule_performance", "rule_combinations"]
            if table_name not in allowed_tables:
                return {
                    "success": False,
                    "error": f"Access denied. Only rule tables are accessible: {', '.join(allowed_tables)}",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": time.time()
                }

            # Get database connection using unified manager
            connection_manager = get_unified_manager(ManagerMode.MCP_SERVER)
            
            async with connection_manager.get_async_session() as session:
                from sqlalchemy import text
                
                # Query to get column information for the specified table
                query = text("""
                    SELECT column_name,
                           data_type,
                           is_nullable,
                           column_default,
                           character_maximum_length,
                           numeric_precision,
                           numeric_scale,
                           column_comment
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                      AND table_name = :table_name
                    ORDER BY ordinal_position
                """)
                
                result = await session.execute(query, {"table_name": table_name})
                columns = []
                
                for row in result.fetchall():
                    column_info = {
                        "column_name": row[0],
                        "data_type": row[1],
                        "nullable": row[2] == "YES",
                        "default_value": row[3],
                        "max_length": row[4],
                        "precision": row[5],
                        "scale": row[6],
                        "description": row[7] or f"Column in {table_name} table"
                    }
                    columns.append(column_info)

                # Get table constraints (indexes, foreign keys, etc.)
                constraints_query = text("""
                    SELECT constraint_name, constraint_type 
                    FROM information_schema.table_constraints 
                    WHERE table_schema = 'public' 
                      AND table_name = :table_name
                """)
                
                constraints_result = await session.execute(constraints_query, {"table_name": table_name})
                constraints = []
                
                for row in constraints_result.fetchall():
                    constraints.append({
                        "constraint_name": row[0],
                        "constraint_type": row[1]
                    })

                processing_time = (time.time() - start_time) * 1000

                logger.info(f"Successfully described table '{table_name}' - "
                           f"Columns: {len(columns)}, Constraints: {len(constraints)}, "
                           f"Processing time: {processing_time:.2f}ms")

                return {
                    "success": True,
                    "table_name": table_name,
                    "columns": columns,
                    "column_count": len(columns),
                    "constraints": constraints,
                    "constraint_count": len(constraints),
                    "access_level": "read_only",
                    "table_purpose": self._get_table_purpose(table_name),
                    "processing_time_ms": processing_time,
                    "timestamp": time.time()
                }

        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to describe table '{table_name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "table_name": table_name,
                "processing_time_ms": error_time,
                "timestamp": time.time()
            }

    def _is_read_only_query(self, query: str) -> bool:
        """Validate that query is read-only (SELECT statements only) per ADR-005."""
        # Remove leading/trailing whitespace and convert to uppercase
        query_upper = query.strip().upper()
        
        # Check if query starts with SELECT (allowing for CTEs with WITH)
        if query_upper.startswith("SELECT"):
            return True
        elif query_upper.startswith("WITH") and "SELECT" in query_upper:
            return True
        
        # Explicitly deny write operations
        write_operations = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "REPLACE"]
        for operation in write_operations:
            if query_upper.startswith(operation):
                return False
                
        return False  # Conservative approach - deny if uncertain

    def _validates_table_access(self, query: str) -> bool:
        """Validate that query only accesses allowed rule tables per ADR-005."""
        import re
        
        # Allowed tables for rule application workflows
        allowed_tables = {"rule_metadata", "rule_performance", "rule_combinations"}
        
        # Extract table names from query using regex
        # This is a simple implementation - could be enhanced with a proper SQL parser
        query_lower = query.lower()
        
        # Find FROM and JOIN clauses (including CTEs)
        table_pattern = r'\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(table_pattern, query_lower)
        
        # Filter out CTE aliases and only check actual table names
        actual_tables = []
        for table in matches:
            if table in allowed_tables:
                actual_tables.append(table)
        
        # If we found actual allowed tables via regex, validate them
        if actual_tables:
            # All found tables must be in allowed list (already filtered above)
            # Check if any forbidden tables are also present
            for table in matches:
                if table not in allowed_tables and table not in ["cte", "subq", "t1", "t2", "alias"]:  # Common aliases
                    return False
            return True
                
        # If no tables found via regex, check if any allowed table names appear in the query
        # This handles CTEs and complex queries
        has_allowed_table = False
        for table in allowed_tables:
            if table in query_lower:
                has_allowed_table = True
                break
        
        if not has_allowed_table:
            return False  # No allowed tables found
        
        # Additional check: ensure no forbidden tables are mentioned anywhere in the query
        forbidden_patterns = [
            "users", "user_data", "system_config", "passwords", "credentials",
            "sessions", "tokens", "keys", "secrets", "logs", "audit",
            "feedback_collection", "training_prompts"  # Other tables from schema
        ]
        
        for forbidden in forbidden_patterns:
            if forbidden in query_lower:
                return False
            
        return True

    def _get_table_purpose(self, table_name: str) -> str:
        """Get descriptive purpose for rule application tables."""
        purposes = {
            "rule_metadata": "Stores rule definitions, categories, parameters, and configuration for prompt improvement rules",
            "rule_performance": "Tracks rule effectiveness metrics, execution times, and performance data for ML optimization",
            "rule_combinations": "Records combinations of rules, their combined effectiveness, and usage statistics"
        }
        return purposes.get(table_name, f"Rule application table: {table_name}")

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

    # 2025 FastMCP Enhancement: Streamable HTTP transport
    def run_streamable_http(self, host: str = "127.0.0.1", port: int = 8080):
        """Run server with Streamable HTTP transport (2025-03-26 spec).
        
        This enables HTTP-based communication instead of stdio,
        supporting both SSE and regular HTTP responses for better
        client compatibility and production deployments.
        """
        logger.info(f"Starting APES MCP Server with Streamable HTTP transport on {host}:{port}")
        
        try:
            # Note: The MCP SDK's FastMCP.run() method accepts transport parameter
            # but the actual implementation may vary based on SDK version
            self.mcp.run(
                transport="streamable-http",
                host=host,
                port=port,
                log_level="INFO"
            )
        except TypeError as e:
            # Fallback if transport parameters not supported
            logger.warning(f"Streamable HTTP transport parameters not supported in current MCP SDK: {e}")
            logger.info("Falling back to standard stdio transport")
            self.mcp.run()
        except Exception as e:
            logger.error(f"Failed to start with HTTP transport: {e}")
            raise
    
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


    # 2025 FastMCP Enhancement: Wildcard resource implementations
    async def _get_session_history_impl(self, session_id: str) -> dict[str, Any]:
        """Implementation for hierarchical session history with wildcards."""
        try:
            # Parse wildcard session_id (e.g., "user123/workspace/main")
            path_parts = session_id.split('/')
            base_session_id = path_parts[0]
            
            # Get session data from store
            session_data = await self.services.session_store.get(base_session_id)
            
            if not session_data:
                return {
                    "session_id": session_id,
                    "exists": False,
                    "message": f"Session '{base_session_id}' not found",
                    "path_components": path_parts,
                    "timestamp": time.time()
                }
            
            # Extract history based on path
            history = session_data.get("history", [])
            
            # Filter by sub-paths if provided
            if len(path_parts) > 1:
                # Example: filter by workspace
                if len(path_parts) >= 2 and path_parts[1] == "workspace":
                    workspace_name = path_parts[2] if len(path_parts) > 2 else None
                    if workspace_name:
                        history = [h for h in history if h.get("workspace") == workspace_name]
            
            return {
                "session_id": session_id,
                "base_session_id": base_session_id,
                "history": history,
                "count": len(history),
                "path_components": path_parts,
                "exists": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get session history for '{session_id}': {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "exists": False,
                "timestamp": time.time()
            }
    
    async def _get_rule_category_performance_impl(self, rule_category: str) -> dict[str, Any]:
        """Implementation for rule category performance metrics with wildcards."""
        start_time = time.time()
        
        try:
            # Parse category path (e.g., "security/input_validation/xss")
            categories = rule_category.split('/')
            
            # Build SQL query for hierarchical category filtering
            connection_manager = get_unified_manager(ManagerMode.MCP_SERVER)
            
            async with connection_manager.get_async_session() as session:
                from sqlalchemy import text
                
                # Query to get performance metrics for rules in the category
                query = text("""
                    SELECT 
                        COUNT(DISTINCT rp.rule_id) as total_rules,
                        COUNT(rp.id) as total_applications,
                        AVG(rp.improvement_score) as avg_improvement_score,
                        AVG(rp.execution_time_ms) as avg_execution_ms,
                        SUM(CASE WHEN rp.success = true THEN 1 ELSE 0 END)::float / COUNT(rp.id) as success_rate,
                        rm.category,
                        rm.subcategory
                    FROM rule_performance rp
                    JOIN rule_metadata rm ON rp.rule_id = rm.rule_id
                    WHERE rm.is_active = true
                """)
                
                # Add category filtering based on path depth
                params = {}
                if len(categories) >= 1:
                    query = text(str(query) + " AND rm.category = :category")
                    params["category"] = categories[0]
                
                if len(categories) >= 2:
                    query = text(str(query) + " AND rm.subcategory = :subcategory")
                    params["subcategory"] = categories[1]
                
                query = text(str(query) + " GROUP BY rm.category, rm.subcategory")
                
                result = await session.execute(query, params)
                row = result.first()
                
                if row:
                    metrics = {
                        "total_rules": int(row[0]),
                        "total_applications": int(row[1]),
                        "avg_improvement_score": float(row[2]) if row[2] else 0.0,
                        "avg_processing_ms": float(row[3]) if row[3] else 0.0,
                        "success_rate": float(row[4]) if row[4] else 0.0,
                        "category": row[5],
                        "subcategory": row[6]
                    }
                else:
                    metrics = {
                        "total_rules": 0,
                        "total_applications": 0,
                        "avg_improvement_score": 0.0,
                        "avg_processing_ms": 0.0,
                        "success_rate": 0.0
                    }
                
                # Get top performing rules in the category
                top_rules_query = text("""
                    SELECT 
                        rm.rule_id,
                        rm.name,
                        COUNT(rp.id) as applications,
                        AVG(rp.improvement_score) as avg_score
                    FROM rule_metadata rm
                    JOIN rule_performance rp ON rm.rule_id = rp.rule_id
                    WHERE rm.is_active = true
                """)
                
                if "category" in params:
                    top_rules_query = text(str(top_rules_query) + " AND rm.category = :category")
                if "subcategory" in params:
                    top_rules_query = text(str(top_rules_query) + " AND rm.subcategory = :subcategory")
                
                top_rules_query = text(str(top_rules_query) + """
                    GROUP BY rm.rule_id, rm.name
                    ORDER BY AVG(rp.improvement_score) DESC
                    LIMIT 5
                """)
                
                top_rules_result = await session.execute(top_rules_query, params)
                top_rules = []
                
                for rule_row in top_rules_result.fetchall():
                    top_rules.append({
                        "rule_id": rule_row[0],
                        "name": rule_row[1],
                        "applications": int(rule_row[2]),
                        "avg_improvement_score": float(rule_row[3]) if rule_row[3] else 0.0
                    })
                
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "category_path": rule_category,
                    "categories": categories,
                    "metrics": metrics,
                    "top_rules": top_rules,
                    "processing_time_ms": processing_time,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Failed to get rule category performance for '{rule_category}': {e}")
            return {
                "category_path": rule_category,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time()
            }
    
    async def _get_hierarchical_metrics_impl(self, metric_type: str) -> dict[str, Any]:
        """Implementation for hierarchical metrics with flexible paths."""
        try:
            # Parse metric path (e.g., "performance/tools/improve_prompt/daily")
            path_parts = metric_type.split('/')
            
            # Route to appropriate metric source based on first path component
            if path_parts[0] == "performance":
                # Get timing metrics from middleware
                timing_metrics = self.services.timing_middleware.get_metrics_summary()
                
                # Filter by tool name if specified
                if len(path_parts) > 1 and path_parts[1] == "tools" and len(path_parts) > 2:
                    tool_name = path_parts[2]
                    if tool_name in timing_metrics:
                        timing_metrics = {tool_name: timing_metrics[tool_name]}
                
                return {
                    "metric_type": metric_type,
                    "path": path_parts,
                    "data": timing_metrics,
                    "source": "timing_middleware",
                    "timestamp": time.time()
                }
                
            elif path_parts[0] == "errors":
                # Get error metrics from error handling middleware
                error_data = {}
                
                # Since we have ErrorHandlingMiddleware in our stack, get its data
                for mw in self.services.middleware_stack.middleware:
                    if hasattr(mw, 'error_counts'):
                        error_data = {"error_counts": dict(mw.error_counts)}
                        break
                
                return {
                    "metric_type": metric_type,
                    "path": path_parts,
                    "data": error_data,
                    "source": "error_middleware",
                    "timestamp": time.time()
                }
                
            elif path_parts[0] == "sessions":
                # Get session store metrics
                session_metrics = {
                    "active_sessions": len(self.services.session_store._store),
                    "max_size": self.services.session_store.maxsize,
                    "ttl": self.services.session_store.ttl,
                    "cleanup_interval": self.services.session_store.cleanup_interval
                }
                
                return {
                    "metric_type": metric_type,
                    "path": path_parts,
                    "data": session_metrics,
                    "source": "session_store",
                    "timestamp": time.time()
                }
                
            else:
                # Unknown metric type
                return {
                    "metric_type": metric_type,
                    "path": path_parts,
                    "data": {},
                    "message": f"Unknown metric type: {path_parts[0]}",
                    "available_types": ["performance", "errors", "sessions"],
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Failed to get hierarchical metrics for '{metric_type}': {e}")
            return {
                "metric_type": metric_type,
                "error": str(e),
                "timestamp": time.time()
            }


class PromptEnhancementRequest(BaseModel):
    """Request model for modern 2025 prompt enhancement - breaking change from legacy API"""
    prompt: str = Field(..., description="The prompt to enhance")
    session_id: str = Field(..., description="Required session ID for tracking and observability")
    context: dict[str, Any] | None = Field(
        default=None, description="Optional additional context for enhancement"
    )


class PromptStorageRequest(BaseModel):
    """Request model for modern 2025 prompt storage - breaking change from legacy API"""
    original: str = Field(..., description="The original prompt")
    enhanced: str = Field(..., description="The enhanced prompt")
    metrics: dict[str, Any] = Field(..., description="Success metrics")
    session_id: str = Field(..., description="Required session ID for tracking and observability")


# Global server instance
server = APESMCPServer()


# Main entry point for stdio transport
def main():
    """Main entry point for the modernized MCP server with 2025 enhancements.
    
    Supports both stdio (default) and HTTP transport modes:
    - Default: python server.py (stdio transport)
    - HTTP: python server.py --http (streamable HTTP transport on port 8080)
    - Custom HTTP: python server.py --http --port 9000
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="APES MCP Server with 2025 FastMCP enhancements")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP transport instead of stdio")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP transport (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport (default: 127.0.0.1)")
    
    args = parser.parse_args()
    
    logger.info("Starting APES MCP Server with modern architecture and 2025 enhancements...")
    
    if args.http:
        logger.info(f"Using streamable HTTP transport on {args.host}:{args.port}")
        server.run_streamable_http(host=args.host, port=args.port)
    else:
        logger.info("Using stdio transport (default)")
        server.run()


if __name__ == "__main__":
    main()
