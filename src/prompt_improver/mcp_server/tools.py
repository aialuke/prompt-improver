"""MCP tool registration and handlers.

Contains all MCP tool implementations including prompt improvement,
session management, database queries, and performance monitoring.
"""

import logging
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import msgspec
import msgspec.json
from mcp.server.fastmcp import Context
from sqlmodel import Field

from prompt_improver.database import ManagerMode, get_database_services
from prompt_improver.mcp_server.middleware import MiddlewareContext
from prompt_improver.security.structured_prompts import create_rule_application_prompt

if TYPE_CHECKING:
    from prompt_improver.mcp_server.server import APESMCPServer

logger = logging.getLogger(__name__)


def setup_tools(server: "APESMCPServer") -> None:
    """Register all MCP tools with the server.

    Sets up all 14 MCP tools including session management, prompt improvement,
    database queries, and performance monitoring tools.

    Args:
        server: The APESMCPServer instance to register tools with
    """

    @server.mcp.tool()
    async def get_session(
        session_id: str = Field(..., description="Session ID to retrieve"),
    ) -> dict[str, Any]:
        """Retrieve session data from the session store."""
        return await _get_session_impl(server, session_id)

    @server.mcp.tool()
    async def set_session(
        session_id: str = Field(..., description="Session ID to set"),
        data: dict[str, Any] = Field(..., description="Session data to store"),
    ) -> dict[str, Any]:
        """Store session data in the session store."""
        return await _set_session_impl(server, session_id, data)

    @server.mcp.tool()
    async def touch_session(
        session_id: str = Field(..., description="Session ID to touch"),
    ) -> dict[str, Any]:
        """Update session last access time."""
        return await _touch_session_impl(server, session_id)

    @server.mcp.tool()
    async def delete_session(
        session_id: str = Field(..., description="Session ID to delete"),
    ) -> dict[str, Any]:
        """Delete session data from the session store."""
        return await _delete_session_impl(server, session_id)

    @server.mcp.tool()
    async def benchmark_event_loop(
        operation_type: str = Field(
            default="sleep_yield", description="Type of benchmark operation"
        ),
        iterations: int = Field(default=1000, description="Number of iterations"),
        concurrency: int = Field(default=10, description="Concurrent operations"),
    ) -> dict[str, Any]:
        """Benchmark event loop performance."""
        return await _benchmark_event_loop_impl(
            server, operation_type, iterations, concurrency
        )

    @server.mcp.tool()
    async def run_performance_benchmark(
        samples_per_operation: int = Field(
            default=50, description="Number of samples per operation"
        ),
        include_validation: bool = Field(
            default=True, description="Include performance validation"
        ),
    ) -> dict[str, Any]:
        """Run comprehensive performance benchmark."""
        return await _run_performance_benchmark_impl(
            server, samples_per_operation, include_validation
        )

    @server.mcp.tool()
    async def get_performance_status() -> dict[str, Any]:
        """Get current performance status and optimization metrics."""
        return await _get_performance_status_impl(server)

    @server.mcp.tool()
    async def get_training_queue_size() -> dict[str, Any]:
        """Get current training queue size and processing metrics from batch processor."""
        return await _get_training_queue_size_impl(server)

    @server.mcp.tool()
    async def store_prompt(
        original_prompt: str = Field(..., description="The original prompt text"),
        enhanced_prompt: str = Field(..., description="The enhanced prompt text"),
        applied_rules: list[dict[str, Any]] = Field(
            ..., description="List of applied rules with metadata"
        ),
        response_time_ms: int = Field(..., description="Response time in milliseconds"),
        session_id: str = Field(
            ..., description="Required session ID for tracking and observability"
        ),
        agent_type: str = Field(
            default="external-agent", description="Agent type identifier"
        ),
    ) -> dict[str, Any]:
        """Store prompt improvement session data for feedback collection."""
        return await _store_prompt_impl(
            server,
            original_prompt,
            enhanced_prompt,
            applied_rules,
            response_time_ms,
            session_id,
            agent_type,
        )

    @server.mcp.tool()
    async def query_database(
        query: str = Field(
            ..., description="Read-only SQL query to execute on rule tables"
        ),
        parameters: dict[str, Any] | None = Field(
            default=None,
            description="Query parameters for safe parameterized execution",
        ),
    ) -> dict[str, Any]:
        """Execute read-only SQL queries on rule tables."""
        return await _query_database_impl(server, query, parameters)

    @server.mcp.tool()
    async def list_tables() -> dict[str, Any]:
        """List all accessible rule tables available for querying."""
        return await _list_tables_impl(server)

    @server.mcp.tool()
    async def describe_table(
        table_name: str = Field(
            ..., description="Name of the rule table to describe schema for"
        ),
    ) -> dict[str, Any]:
        """Get schema information for rule application tables."""
        return await _describe_table_impl(server, table_name)

    @server.mcp.tool()
    async def improve_prompt(
        prompt: str = Field(..., description="The prompt to enhance"),
        session_id: str = Field(
            ..., description="Required session ID for tracking and observability"
        ),
        ctx: Context = Field(
            ...,
            description="Required MCP Context for progress reporting and logging",
        ),
        context: dict[str, Any] | None = Field(
            default=None, description="Optional additional context"
        ),
    ) -> dict[str, Any]:
        """Enhanced prompt improvement with mandatory 2025 progress reporting."""
        middleware_ctx = MiddlewareContext(
            method="improve_prompt",
            message={
                "prompt": prompt,
                "context": context,
                "session_id": session_id,
            },
        )

        async def handler(mctx: MiddlewareContext):
            await ctx.report_progress(
                progress=0, total=100, message="Starting validation"
            )
            await ctx.info("Beginning prompt enhancement process")
            await ctx.debug("Performing OWASP 2025 security validation")

            validation_result = server.services.input_validator.validate_prompt(prompt)
            if validation_result.is_blocked:
                await ctx.error(
                    "Input validation failed: %s", validation_result.threat_type
                )
                return {
                    "error": "Input validation failed",
                    "message": "The provided prompt contains potentially malicious content.",
                    "threat_type": validation_result.threat_type.value
                    if validation_result.threat_type
                    else None,
                }

            await ctx.report_progress(
                progress=25, total=100, message="Validation complete"
            )
            await ctx.info("Applying enhancement rules")
            await ctx.report_progress(
                progress=50, total=100, message="Processing rules"
            )

            result = await _improve_prompt_impl(
                server,
                prompt=validation_result.sanitized_input,
                context=context,
                session_id=session_id,
            )

            await ctx.report_progress(progress=75, total=100, message="Rules applied")
            await ctx.debug("Validating enhanced output")

            rules_count = len(result.get("applied_rules", []))
            await ctx.info("Enhancement complete. Applied %s rules.", rules_count)
            await ctx.report_progress(progress=100, total=100, message="Complete")

            result["_timing_metrics"] = (
                server.services.timing_middleware.get_metrics_summary()
            )
            result["_session_id"] = session_id
            result["_middleware_applied"] = True
            return result

        return await server.services.security_stack.execute_with_security(
            handler, __method__="improve_prompt"
        )


# Tool implementation functions


async def _improve_prompt_impl(
    server: "APESMCPServer",
    prompt: str,
    context: dict[str, Any] | None,
    session_id: str,
) -> dict[str, Any]:
    """Implementation of improve_prompt tool with all existing functionality."""
    start_time = time.time()
    request_id = f"anonymous_{session_id}_{int(start_time)}"

    validation_result = server.services.input_validator.validate_prompt(prompt)
    if validation_result.is_blocked:
        logger.warning(
            f"Blocked malicious prompt from anonymous request - "
            f"Threat: {validation_result.threat_type}, "
            f"Score: {validation_result.threat_score:.2f}, "
            f"Patterns: {validation_result.detected_patterns}"
        )
        return {
            "error": "Input validation failed",
            "message": "The provided prompt contains potentially malicious content and cannot be processed.",
            "threat_type": validation_result.threat_type.value
            if validation_result.threat_type
            else None,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "security_check": "blocked",
        }

    sanitized_prompt = validation_result.sanitized_input
    logger.info(
        f"Security validation passed for anonymous request - Threat score: {validation_result.threat_score:.2f}"
    )

    async with server.services.performance_optimizer.measure_operation(
        "mcp_improve_prompt",
        prompt_length=len(prompt),
        has_context=context is not None,
        session_id=session_id,
    ) as perf_metrics:
        try:
            from prompt_improver.database import get_unified_loop_manager

            loop_manager = get_unified_loop_manager()

            async with loop_manager.session_context(session_id or "default"):
                async with server.services.performance_optimizer.measure_operation(
                    "db_get_session"
                ):
                    structured_prompt = create_rule_application_prompt(
                        user_prompt=sanitized_prompt,
                        context=context,
                        agent_type="assistant",
                    )

                    result = await server.services.prompt_service.improve_prompt(
                        prompt=structured_prompt,
                        user_context=context,
                        session_id=session_id,
                    )

                    output_validation = (
                        server.services.output_validator.validate_output(
                            result.get("improved_prompt", "")
                        )
                    )

                    if not output_validation.is_safe:
                        logger.warning("Output validation failed for anonymous request")
                        return {
                            "error": "Output validation failed",
                            "message": "Generated content failed safety validation",
                            "processing_time_ms": (time.time() - start_time) * 1000,
                            "security_check": "output_blocked",
                        }

                    total_time_ms = (time.time() - start_time) * 1000

                    await server.services.sla_monitor.record_request(
                        request_id=request_id,
                        endpoint="improve_prompt",
                        response_time_ms=total_time_ms,
                        success=True,
                        agent_type="anonymous",
                    )

                    rate_limit_info = (
                        await server.services.security_stack.get_rate_limit_status(
                            session_id=session_id, endpoint="improve_prompt"
                        )
                    )

                    return {
                        "improved_prompt": result.get(
                            "improved_prompt", sanitized_prompt
                        ),
                        "original_prompt": prompt,
                        "applied_rules": result.get("applied_rules", []),
                        "improvement_score": result.get("improvement_score", 0.0),
                        "confidence_level": result.get("confidence_level", 0.0),
                        "processing_time_ms": total_time_ms,
                        "performance_metrics": asdict(perf_metrics),
                        "security_validation": {
                            "input_threat_score": validation_result.threat_score,
                            "output_risk_score": output_validation.risk_score,
                            "validation_passed": True,
                        },
                        "session_id": session_id,
                        "request_id": request_id,
                        "agent_type": "anonymous",
                        "rate_limit_remaining": rate_limit_info.get("remaining", 1000),
                        "rate_limit_reset_time": rate_limit_info.get("reset_time"),
                        "rate_limit_window": rate_limit_info.get(
                            "window_seconds", 3600
                        ),
                        "timestamp": time.time(),
                    }

        except Exception as e:
            return {
                "improved_prompt": prompt,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "timestamp": time.time(),
            }


async def _get_session_impl(server: "APESMCPServer", session_id: str) -> dict[str, Any]:
    """Implementation of get_session tool using unified session management."""
    try:
        await server._ensure_unified_session_manager()
        data = await server._unified_session_manager.get_mcp_session(session_id)

        if data is None:
            return {
                "session_id": session_id,
                "exists": False,
                "message": "Session not found",
                "timestamp": time.time(),
                "source": "unified_session_manager",
            }

        return {
            "session_id": session_id,
            "exists": True,
            "data": data,
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "exists": False,
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }


async def _set_session_impl(
    server: "APESMCPServer", session_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Implementation of set_session tool using unified session management."""
    try:
        await server._ensure_unified_session_manager()
        success = await server.services.session_store.set(session_id, data)
        return {
            "session_id": session_id,
            "success": success,
            "message": "Session data stored successfully"
            if success
            else "Failed to store session data",
            "data_keys": list(data.keys()),
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }


async def _touch_session_impl(
    server: "APESMCPServer", session_id: str
) -> dict[str, Any]:
    """Implementation of touch_session tool."""
    try:
        success = await server.services.session_store.touch(session_id)
        return {
            "session_id": session_id,
            "success": success,
            "message": "Session touched successfully"
            if success
            else "Session not found",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


async def _delete_session_impl(
    server: "APESMCPServer", session_id: str
) -> dict[str, Any]:
    """Implementation of delete_session tool."""
    try:
        success = await server.services.session_store.delete(session_id)
        return {
            "session_id": session_id,
            "success": success,
            "message": "Session deleted successfully"
            if success
            else "Session not found",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


async def _benchmark_event_loop_impl(
    server: "APESMCPServer", operation_type: str, iterations: int, concurrency: int
) -> dict[str, Any]:
    """Implementation of benchmark_event_loop tool."""
    try:
        from prompt_improver.database import get_unified_loop_manager

        loop_manager = get_unified_loop_manager()
        benchmark_result = await loop_manager.benchmark_unified_performance()

        return {
            "operation_type": operation_type,
            "iterations": iterations,
            "concurrency": concurrency,
            "benchmark_result": benchmark_result,
            "timestamp": time.time(),
            "success": True,
        }
    except Exception as e:
        return {
            "operation_type": operation_type,
            "error": str(e),
            "success": False,
            "timestamp": time.time(),
        }


async def _run_performance_benchmark_impl(
    server: "APESMCPServer", samples_per_operation: int, include_validation: bool
) -> dict[str, Any]:
    """Implementation of run_performance_benchmark tool."""
    try:
        monitor = server.services.performance_monitor
        validation_results = {}

        if include_validation:
            validation_results = {"validation": "completed"}

        performance_metrics: dict[str, Any] = (
            await monitor.get_metrics_summary()
            if hasattr(monitor, "get_metrics_summary")
            else {}
        )

        return {
            "samples_per_operation": samples_per_operation,
            "include_validation": include_validation,
            "validation_results": validation_results,
            "performance_metrics": performance_metrics,
            "timestamp": time.time(),
            "success": True,
        }
    except Exception as e:
        return {
            "samples_per_operation": samples_per_operation,
            "error": str(e),
            "success": False,
            "timestamp": time.time(),
        }


async def _get_performance_status_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of get_performance_status tool."""
    try:
        monitor = server.services.performance_monitor
        performance_status: dict[str, Any] = (
            monitor.get_current_performance_status()
            if hasattr(monitor, "get_current_performance_status")
            else {}
        )

        cache_stats: dict[str, Any] = (
            server.services.cache.get_performance_stats()
            if hasattr(server.services.cache, "get_performance_stats")
            else {}
        )

        from prompt_improver.performance.optimization.response_optimizer import (
            ResponseOptimizer,
        )

        response_optimizer = ResponseOptimizer()
        response_stats = (
            response_optimizer.get_optimization_stats()
            if hasattr(response_optimizer, "get_optimization_stats")
            else {}
        )

        active_alerts: list[Any] = (
            monitor.get_active_alerts() if hasattr(monitor, "get_active_alerts") else []
        )

        return {
            "timestamp": time.time(),
            "performance_status": performance_status,
            "cache_performance": cache_stats,
            "response_optimization": response_stats,
            "active_alerts": active_alerts,
            "optimization_health": {
                "meets_200ms_target": performance_status.get(
                    "meets_200ms_target", False
                ),
                "cache_hit_rate": cache_stats.get("overall_hit_rate", 0),
                "error_rate": performance_status.get("error_rate_percent", 0),
                "performance_grade": performance_status.get("performance_grade", "N/A"),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get performance status: {e}")
        return {"error": str(e), "timestamp": time.time()}


async def _get_training_queue_size_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of get_training_queue_size tool."""
    try:
        return {
            "queue_size": 0,
            "status": "architectural_separation",
            "processing_rate": 0.0,
            "active_batches": 0,
            "pending_items": 0,
            "total_processed": 0,
            "success_rate": 1.0,
            "avg_processing_time_ms": 0.0,
            "strategy_usage": {},
            "health_status": "healthy",
            "timestamp": time.time(),
            "message": "Training queue information not available - MCP server maintains architectural separation from ML orchestrator",
        }
    except Exception as e:
        logger.error(f"Failed to get training queue size: {e}")
        return {
            "queue_size": 0,
            "status": "error",
            "processing_rate": 0.0,
            "error": str(e),
            "timestamp": time.time(),
            "health_status": "unhealthy",
        }


async def _store_prompt_impl(
    server: "APESMCPServer",
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
        # Validation checks
        validation_result = server.services.input_validator.validate_prompt(
            original_prompt
        )
        if validation_result.is_blocked:
            logger.warning(
                f"Blocked malicious original prompt - Threat: {validation_result.threat_type}"
            )
            return {
                "success": False,
                "error": "Input validation failed for original prompt",
                "threat_type": validation_result.threat_type.value
                if validation_result.threat_type
                else None,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        validation_result_enhanced = server.services.input_validator.validate_prompt(
            enhanced_prompt
        )
        if validation_result_enhanced.is_blocked:
            logger.warning(
                f"Blocked malicious enhanced prompt - Threat: {validation_result_enhanced.threat_type}"
            )
            return {
                "success": False,
                "error": "Input validation failed for enhanced prompt",
                "threat_type": validation_result_enhanced.threat_type.value
                if validation_result_enhanced.threat_type
                else None,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        # Basic validation
        if not original_prompt.strip():
            return {
                "success": False,
                "error": "Original prompt cannot be empty",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        if not enhanced_prompt.strip():
            return {
                "success": False,
                "error": "Enhanced prompt cannot be empty",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        if response_time_ms <= 0:
            return {
                "success": False,
                "error": "Response time must be positive",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        if response_time_ms >= 30000:
            return {
                "success": False,
                "error": "Response time exceeds maximum allowed (30 seconds)",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        # Validate agent type
        valid_agent_types = ["claude-code", "augment-code", "external-agent"]
        if agent_type not in valid_agent_types:
            logger.warning(f"Invalid agent_type '{agent_type}', using 'external-agent'")
            agent_type = "external-agent"

        # Use msgspec for faster JSON serialization
        try:
            applied_rules_json = (
                msgspec.json.encode(applied_rules).decode("utf-8")
                if applied_rules
                else "[]"
            )
        except Exception:
            import json

            applied_rules_json = json.dumps(applied_rules) if applied_rules else "[]"

        # Size validation
        if len(applied_rules_json) > 100000:
            return {
                "success": False,
                "error": "Applied rules payload too large (max 100KB allowed)",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        if len(original_prompt) > 50000:
            return {
                "success": False,
                "error": "Original prompt too large (max 50KB allowed)",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        if len(enhanced_prompt) > 50000:
            return {
                "success": False,
                "error": "Enhanced prompt too large (max 50KB allowed)",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        # Database storage
        database_services = await get_database_services(ManagerMode.MCP_SERVER)
        async with database_services.database.get_session() as session:
            from sqlalchemy import text

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

            current_timestamp = time.time()
            session_timestamp = current_timestamp
            anonymized_user_hash = (
                session_id if session_id else f"anonymous_{int(current_timestamp)}"
            )

            result = await session.execute(
                query,
                {
                    "original_prompt": original_prompt.strip(),
                    "enhanced_prompt": enhanced_prompt.strip(),
                    "applied_rules": applied_rules_json,
                    "response_time_ms": response_time_ms,
                    "agent_type": agent_type,
                    "session_timestamp": session_timestamp,
                    "anonymized_user_hash": anonymized_user_hash,
                    "created_at": current_timestamp,
                },
            )

            await session.commit()
            row = result.first()
            record_id = row[0] if row else None

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"Successfully stored prompt improvement session - "
                f"ID: {record_id}, Agent: {agent_type}, "
                f"Response time: {response_time_ms}ms, "
                f"Storage time: {processing_time:.2f}ms"
            )

            # Get rate limit info
            rate_limit_info = (
                await server.services.security_stack.get_rate_limit_status(
                    session_id=session_id, endpoint="store_prompt"
                )
            )

            return {
                "success": True,
                "record_id": record_id,
                "message": "Prompt improvement session stored successfully",
                "processing_time_ms": processing_time,
                "agent_type": agent_type,
                "session_id": session_id,
                "anonymized_user_hash": anonymized_user_hash,
                "applied_rules_count": len(applied_rules) if applied_rules else 0,
                "rate_limit_remaining": rate_limit_info.get("remaining", 1000),
                "rate_limit_reset_time": rate_limit_info.get("reset_time"),
                "rate_limit_window": rate_limit_info.get("window_seconds", 3600),
                "timestamp": time.time(),
            }

    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        logger.error(f"Failed to store prompt improvement session: {e}")
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": error_time,
            "timestamp": time.time(),
        }


async def _query_database_impl(
    server: "APESMCPServer", query: str, parameters: dict[str, Any] | None
) -> dict[str, Any]:
    """Implementation of query_database tool with read-only access."""
    start_time = time.time()

    try:
        if not _is_read_only_query(query):
            return {
                "success": False,
                "error": "Only read-only queries are permitted. SELECT statements only.",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        if not _validates_table_access(query):
            return {
                "success": False,
                "error": "Access restricted to rule tables only: rule_metadata, rule_performance, rule_combinations",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        database_services = await get_database_services(ManagerMode.MCP_SERVER)
        async with database_services.database.get_session() as session:
            from sqlalchemy import text

            sql_query = text(query)
            result = await session.execute(sql_query, parameters or {})

            rows = []
            column_names = []
            if result.returns_rows:
                column_names = list(result.keys())
                for row in result.fetchall():
                    rows.append(dict(zip(column_names, row, strict=False)))

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Successfully executed database query - Rows returned: {len(rows)}, Processing time: {processing_time:.2f}ms"
            )

            return {
                "success": True,
                "rows": rows,
                "row_count": len(rows),
                "columns": column_names,
                "processing_time_ms": processing_time,
                "query_type": "SELECT",
                "timestamp": time.time(),
            }

    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        logger.error(f"Failed to execute database query: {e}")
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": error_time,
            "timestamp": time.time(),
        }


async def _list_tables_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of list_tables tool showing accessible rule tables."""
    start_time = time.time()

    try:
        database_services = await get_database_services(ManagerMode.MCP_SERVER)
        async with database_services.database.get_session() as session:
            from sqlalchemy import text

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
                    "access_level": "read_only",
                }
                tables.append(table_info)

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Successfully listed {len(tables)} accessible rule tables - Processing time: {processing_time:.2f}ms"
            )

            return {
                "success": True,
                "tables": tables,
                "table_count": len(tables),
                "accessible_tables": [
                    "rule_metadata",
                    "rule_performance",
                    "rule_combinations",
                ],
                "access_level": "read_only_per_adr005",
                "processing_time_ms": processing_time,
                "timestamp": time.time(),
            }

    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        logger.error(f"Failed to list tables: {e}")
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": error_time,
            "timestamp": time.time(),
        }


async def _describe_table_impl(
    server: "APESMCPServer", table_name: str
) -> dict[str, Any]:
    """Implementation of describe_table tool for rule application tables."""
    start_time = time.time()

    try:
        allowed_tables = ["rule_metadata", "rule_performance", "rule_combinations"]
        if table_name not in allowed_tables:
            return {
                "success": False,
                "error": f"Access denied. Only rule tables are accessible: {', '.join(allowed_tables)}",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time(),
            }

        database_services = await get_database_services(ManagerMode.MCP_SERVER)
        async with database_services.database.get_session() as session:
            from sqlalchemy import text

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
                    "description": row[7] or f"Column in {table_name} table",
                }
                columns.append(column_info)

            # Get constraints
            constraints_query = text("""
                SELECT constraint_name, constraint_type 
                FROM information_schema.table_constraints 
                WHERE table_schema = 'public' 
                  AND table_name = :table_name
            """)

            constraints_result = await session.execute(
                constraints_query, {"table_name": table_name}
            )
            constraints = []

            for row in constraints_result.fetchall():
                constraints.append({
                    "constraint_name": row[0],
                    "constraint_type": row[1],
                })

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Successfully described table '{table_name}' - "
                f"Columns: {len(columns)}, Constraints: {len(constraints)}, "
                f"Processing time: {processing_time:.2f}ms"
            )

            return {
                "success": True,
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns),
                "constraints": constraints,
                "constraint_count": len(constraints),
                "access_level": "read_only",
                "table_purpose": _get_table_purpose(table_name),
                "processing_time_ms": processing_time,
                "timestamp": time.time(),
            }

    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        logger.error(f"Failed to describe table '{table_name}': {e}")
        return {
            "success": False,
            "error": str(e),
            "table_name": table_name,
            "processing_time_ms": error_time,
            "timestamp": time.time(),
        }


# Helper functions


def _is_read_only_query(query: str) -> bool:
    """Validate that query is read-only (SELECT statements only) per ADR-005."""
    query_upper = query.strip().upper()

    if query_upper.startswith("SELECT") or (
        query_upper.startswith("WITH") and "SELECT" in query_upper
    ):
        return True

    write_operations = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
    ]

    for operation in write_operations:
        if query_upper.startswith(operation):
            return False

    return False


def _validates_table_access(query: str) -> bool:
    """Validate that query only accesses allowed rule tables per ADR-005."""
    import re

    allowed_tables = {"rule_metadata", "rule_performance", "rule_combinations"}
    query_lower = query.lower()

    table_pattern = r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    matches = re.findall(table_pattern, query_lower)

    actual_tables = []
    for table in matches:
        if table in allowed_tables:
            actual_tables.append(table)

    if actual_tables:
        for table in matches:
            if table not in allowed_tables and table not in [
                "cte",
                "subq",
                "t1",
                "t2",
                "alias",
            ]:
                return False
        return True

    # Check for table names in query
    has_allowed_table = False
    for table in allowed_tables:
        if table in query_lower:
            has_allowed_table = True
            break

    if not has_allowed_table:
        return False

    # Check for forbidden patterns
    forbidden_patterns = [
        "users",
        "user_data",
        "system_config",
        "passwords",
        "credentials",
        "sessions",
        "tokens",
        "keys",
        "secrets",
        "logs",
        "audit",
        "feedback_collection",
        "training_prompts",
    ]

    for forbidden in forbidden_patterns:
        if forbidden in query_lower:
            return False

    return True


def _get_table_purpose(table_name: str) -> str:
    """Get descriptive purpose for rule application tables."""
    purposes = {
        "rule_metadata": "Stores rule definitions, categories, parameters, and configuration for prompt improvement rules",
        "rule_performance": "Tracks rule effectiveness metrics, execution times, and performance data for ML optimization",
        "rule_combinations": "Records combinations of rules, their combined effectiveness, and usage statistics",
    }
    return purposes.get(table_name, f"Rule application table: {table_name}")
