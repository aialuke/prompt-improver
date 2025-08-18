"""MCP resource registration and providers.

Contains all MCP resource implementations including health checks,
metrics monitoring, and hierarchical resource path handling.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from prompt_improver.repositories.protocols.session_manager_protocol import (
    SessionManagerProtocol,
)

if TYPE_CHECKING:
    from prompt_improver.mcp_server.protocols import MCPServerProtocol as APESMCPServer

logger = logging.getLogger(__name__)


def setup_resources(server: "APESMCPServer") -> None:
    """Register all MCP resources with the server.

    Sets up all 9 MCP resources including health checks, status monitoring,
    and hierarchical metrics with wildcard path support.

    Args:
        server: The APESMCPServer instance to register resources with
    """

    @server.mcp.resource("apes://rule_status")
    async def get_rule_status() -> dict[str, Any]:
        """Get current rule effectiveness and status."""
        return await _get_rule_status_impl(server)

    @server.mcp.resource("apes://session_store/status")
    async def get_session_store_status() -> dict[str, Any]:
        """Get session store statistics and status."""
        return await _get_session_store_status_impl(server)

    @server.mcp.resource("apes://health/live")
    async def health_live() -> dict[str, Any]:
        """Phase 0 liveness check - basic service availability."""
        return await _health_live_impl(server)

    @server.mcp.resource("apes://health/ready")
    async def health_ready() -> dict[str, Any]:
        """Phase 0 readiness check with MCP connection pool and rule application capability."""
        return await _health_ready_impl(server)

    @server.mcp.resource("apes://health/queue")
    async def health_queue() -> dict[str, Any]:
        """Check queue health with comprehensive metrics."""
        return await _health_queue_impl(server)

    @server.mcp.resource("apes://health/phase0")
    async def health_phase0() -> dict[str, Any]:
        """Comprehensive Phase 0 health check with all unified architecture components."""
        return await _health_phase0_impl(server)

    @server.mcp.resource("apes://event_loop/status")
    async def get_event_loop_status() -> dict[str, Any]:
        """Get current event loop status and performance metrics."""
        return await _get_event_loop_status_impl(server)

    @server.mcp.resource("apes://sessions/{session_id}/history")
    async def get_session_history(session_id: str) -> dict[str, Any]:
        """Get detailed session history with wildcard path support.

        Supports hierarchical session IDs like:
        - sessions/user123/history
        - sessions/user123/workspace/main/history
        """
        return await _get_session_history_impl(server, session_id)

    @server.mcp.resource("apes://rules/{rule_category}/performance")
    async def get_rule_category_performance(rule_category: str) -> dict[str, Any]:
        """Get performance metrics for rule categories with wildcard support.

        Supports hierarchical categories like:
        - rules/security/performance
        - rules/security/input_validation/xss/performance
        """
        return await _get_rule_category_performance_impl(server, rule_category)

    @server.mcp.resource("apes://metrics/{metric_type}")
    async def get_hierarchical_metrics(metric_type: str) -> dict[str, Any]:
        """Get hierarchical metrics with flexible path support.

        Examples:
        - metrics/performance
        - metrics/performance/tools/improve_prompt
        - metrics/errors/by_method
        """
        return await _get_hierarchical_metrics_impl(server, metric_type)


# Resource implementation functions


async def _get_rule_status_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of rule_status resource."""
    try:
        rule_stats: dict[str, Any] = {}

        if hasattr(server.services.prompt_service, "get_rule_effectiveness"):
            try:
                rule_stats = (
                    await server.services.prompt_service.get_rule_effectiveness()
                )
            except Exception as e:
                logger.warning(f"Failed to get rule effectiveness: {e}")
                rule_stats = {"rules": [], "error": str(e)}

        rules_list = (
            rule_stats.get("rules", [])
            if isinstance(rule_stats.get("rules"), list)
            else []
        )

        active_rules = []
        for rule in rules_list:
            if isinstance(rule, dict) and rule.get("active", False):
                active_rules.append(rule)

        return {
            "rule_effectiveness": rule_stats,
            "total_rules": len(rules_list),
            "active_rules": len(active_rules),
            "timestamp": time.time(),
            "status": "healthy",
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "timestamp": time.time()}


async def _get_session_store_status_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of session_store/status resource."""
    try:
        if hasattr(server.services.session_store, "get_performance_stats"):
            stats: dict[str, Any] = server.services.session_store.get_performance_stats()
        else:
            stats = {
                "count": 0,
                "memory_usage": 0,
                "hit_rate": 0.0,
                "cleanup_runs": 0,
            }

        return {
            "session_count": stats.get("count", 0),
            "memory_usage": stats.get("memory_usage", 0),
            "hit_rate": stats.get("hit_rate", 0.0),
            "cleanup_runs": stats.get("cleanup_runs", 0),
            "max_size": server.config.mcp_session_maxsize,
            "ttl_seconds": server.config.mcp_session_ttl,
            "timestamp": time.time(),
            "status": "healthy",
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "timestamp": time.time()}


async def _health_live_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of health/live resource."""
    try:
        start_time = time.time()
        await asyncio.sleep(0)
        event_loop_latency = (time.time() - start_time) * 1000

        return {
            "status": "live",
            "event_loop_latency_ms": event_loop_latency,
            "background_queue_size": 0,
            "phase": "0",
            "mcp_server_mode": "rule_application_only",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _health_ready_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of health/ready resource."""
    try:
        async with await server.services.get_database_session() as session:
            from sqlalchemy import text
            # Basic database connectivity check
            await session.execute(text("SELECT 1"))
            health_results = {"database": "healthy"}
        db_health = health_results.get("database", "unknown")

        # Test rule application capability
        rule_application_ready = True
        try:
            test_result = await server.services.prompt_service.improve_prompt(
                prompt="test", user_context={}, session_id="health_check"
            )
            rule_application_ready = "improved_prompt" in test_result
        except Exception:
            rule_application_ready = False

        overall_ready = db_health.get("status") == "healthy" and rule_application_ready

        return {
            "status": "ready" if overall_ready else "not_ready",
            "database": db_health,
            "rule_application": {
                "ready": rule_application_ready,
                "service_available": True,
            },
            "phase": "0",
            "mcp_server_mode": "rule_application_only",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _health_queue_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of health/queue resource."""
    try:
        from prompt_improver.performance.monitoring.health.unified_health_system import (
            get_unified_health_monitor,
        )

        health_monitor = get_unified_health_monitor()
        check_results = await health_monitor.check_health(plugin_name="queue_service")
        queue_result = check_results.get("queue_service")

        if queue_result:
            response = {
                "status": queue_result.status.value,
                "message": queue_result.message,
                "timestamp": time.time(),
            }
        else:
            response = {
                "status": "unknown",
                "message": "Queue health check not available",
                "timestamp": time.time(),
            }

        if hasattr(queue_result, "details") and queue_result.details:
            response.update({
                "queue_length": queue_result.details.get("queue_length", 0),
                "processing_rate": queue_result.details.get("processing_rate", 0.0),
                "retry_backlog": queue_result.details.get("retry_backlog", 0),
                "average_latency_ms": queue_result.details.get(
                    "average_latency_ms", 0.0
                ),
                "throughput_per_second": queue_result.details.get(
                    "throughput_per_second", 0.0
                ),
            })

        return response
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": time.time()}


async def _health_phase0_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of health/phase0 resource."""
    try:
        from prompt_improver.performance.monitoring.health.unified_health_system import (
            get_unified_health_monitor,
        )

        health_monitor = get_unified_health_monitor()
        overall_start = time.time()
        components = {}

        # Database health check
        try:
            db_result = await health_monitor.run_specific_check("database")
            components["database"] = {
                "status": db_result.status.value,
                "message": db_result.message,
                "response_time_ms": getattr(db_result, "response_time_ms", 0),
            }
        except Exception as e:
            components["database"] = {"status": "error", "error": str(e)}

        # Cache health check
        try:
            cache_result = await health_monitor.run_specific_check("cache")
            components["cache"] = {
                "status": cache_result.status.value,
                "message": cache_result.message,
                "response_time_ms": getattr(cache_result, "response_time_ms", 0),
            }
        except Exception as e:
            components["cache"] = {"status": "error", "error": str(e)}

        # Rule application health check
        try:
            rule_result = await health_monitor.run_specific_check("rule_application")
            components["rule_application"] = {
                "status": rule_result.status.value,
                "message": rule_result.message,
                "response_time_ms": getattr(rule_result, "response_time_ms", 0),
            }
        except Exception as e:
            components["rule_application"] = {"status": "error", "error": str(e)}

        # Performance monitoring health check
        try:
            perf_result = await health_monitor.run_specific_check("performance")
            components["performance_monitoring"] = {
                "status": perf_result.status.value,
                "message": perf_result.message,
                "response_time_ms": getattr(perf_result, "response_time_ms", 0),
            }
        except Exception as e:
            components["performance_monitoring"] = {
                "status": "error",
                "error": str(e),
            }

        total_check_time = (time.time() - overall_start) * 1000

        # Calculate health percentage
        healthy_components = sum(
            1
            for comp in components.values()
            if isinstance(comp, dict) and comp.get("status") == "healthy"
        )
        total_components = len(components)
        health_percentage = (
            healthy_components / total_components * 100 if total_components > 0 else 0
        )

        overall_status = (
            "healthy"
            if health_percentage >= 80
            else "degraded"
            if health_percentage >= 50
            else "unhealthy"
        )

        return {
            "status": overall_status,
            "phase": "0",
            "health_percentage": health_percentage,
            "healthy_components": healthy_components,
            "total_components": total_components,
            "total_check_time_ms": total_check_time,
            "components": components,
            "mcp_server_mode": "rule_application_only",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "phase": "0",
            "timestamp": time.time(),
        }


async def _get_event_loop_status_impl(server: "APESMCPServer") -> dict[str, Any]:
    """Implementation of event_loop/status resource."""
    try:
        loop_manager = server.services.event_loop_manager
        loop = asyncio.get_event_loop()

        start_time = time.time()
        await asyncio.sleep(0)
        current_latency = (time.time() - start_time) * 1000

        return {
            "loop_type": type(loop).__name__,
            "is_running": loop.is_running(),
            "current_latency_ms": current_latency,
            "task_count": len(asyncio.all_tasks()),
            "optimization_enabled": hasattr(loop_manager, "optimization_enabled"),
            "timestamp": time.time(),
            "status": "healthy",
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "timestamp": time.time()}


async def _get_session_history_impl(
    server: "APESMCPServer", session_id: str
) -> dict[str, Any]:
    """Implementation for hierarchical session history with wildcards."""
    try:
        path_parts = session_id.split("/")
        base_session_id = path_parts[0]

        session_data = await server.services.session_store.get_session(base_session_id)
        if not session_data:
            return {
                "session_id": session_id,
                "exists": False,
                "message": f"Session '{base_session_id}' not found",
                "path_components": path_parts,
                "timestamp": time.time(),
            }

        history = session_data.get("history", [])

        # Apply hierarchical filtering
        if len(path_parts) > 1:
            if len(path_parts) >= 2 and path_parts[1] == "workspace":
                workspace_name = path_parts[2] if len(path_parts) > 2 else None
                if workspace_name:
                    history = [
                        h for h in history if h.get("workspace") == workspace_name
                    ]

        return {
            "session_id": session_id,
            "base_session_id": base_session_id,
            "history": history,
            "count": len(history),
            "path_components": path_parts,
            "exists": True,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Failed to get session history for '{session_id}': {e}")
        return {
            "session_id": session_id,
            "error": str(e),
            "exists": False,
            "timestamp": time.time(),
        }


async def _get_rule_category_performance_impl(
    server: "APESMCPServer", rule_category: str
) -> dict[str, Any]:
    """Implementation for rule category performance metrics with wildcards."""
    start_time = time.time()

    try:
        categories = rule_category.split("/")
        async with await server.services.get_database_session() as session:
            from sqlalchemy import text

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

            params = {}

            # Apply hierarchical filtering
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
                    "subcategory": row[6],
                }
            else:
                metrics = {
                    "total_rules": 0,
                    "total_applications": 0,
                    "avg_improvement_score": 0.0,
                    "avg_processing_ms": 0.0,
                    "success_rate": 0.0,
                }

            # Get top performing rules in this category
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
                top_rules_query = text(
                    str(top_rules_query) + " AND rm.category = :category"
                )

            if "subcategory" in params:
                top_rules_query = text(
                    str(top_rules_query) + " AND rm.subcategory = :subcategory"
                )

            top_rules_query = text(
                str(top_rules_query)
                + """
                GROUP BY rm.rule_id, rm.name
                ORDER BY AVG(rp.improvement_score) DESC
                LIMIT 5
            """
            )

            top_rules_result = await session.execute(top_rules_query, params)
            top_rules = []

            for rule_row in top_rules_result.fetchall():
                top_rules.append({
                    "rule_id": rule_row[0],
                    "name": rule_row[1],
                    "applications": int(rule_row[2]),
                    "avg_improvement_score": float(rule_row[3]) if rule_row[3] else 0.0,
                })

            processing_time = (time.time() - start_time) * 1000

            return {
                "category_path": rule_category,
                "categories": categories,
                "metrics": metrics,
                "top_rules": top_rules,
                "processing_time_ms": processing_time,
                "timestamp": time.time(),
            }

    except Exception as e:
        logger.error(
            f"Failed to get rule category performance for '{rule_category}': {e}"
        )
        return {
            "category_path": rule_category,
            "error": str(e),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "timestamp": time.time(),
        }


async def _get_hierarchical_metrics_impl(
    server: "APESMCPServer", metric_type: str
) -> dict[str, Any]:
    """Implementation for hierarchical metrics with flexible paths."""
    try:
        path_parts = metric_type.split("/")

        if path_parts[0] == "performance":
            timing_metrics = server.services.timing_middleware.get_metrics_summary()

            # Apply hierarchical filtering
            if (
                len(path_parts) > 1
                and path_parts[1] == "tools"
                and (len(path_parts) > 2)
            ):
                tool_name = path_parts[2]
                if tool_name in timing_metrics:
                    timing_metrics = {tool_name: timing_metrics[tool_name]}

            return {
                "metric_type": metric_type,
                "path": path_parts,
                "data": timing_metrics,
                "source": "timing_middleware",
                "timestamp": time.time(),
            }

        elif path_parts[0] == "errors":
            error_data = {}
            if hasattr(server.services.security_stack, "get_error_metrics"):
                error_data = server.services.security_stack.get_error_metrics()
            else:
                error_data = {"error_counts": {}}

            return {
                "metric_type": metric_type,
                "path": path_parts,
                "data": error_data,
                "source": "error_middleware",
                "timestamp": time.time(),
            }

        elif path_parts[0] == "sessions":
            cache_stats = server.services.session_store.get_stats()
            session_metrics = {
                "active_sessions": cache_stats.get("l1_cache_size", 0),
                "max_size": cache_stats.get("l1_max_size", 1000),
                "ttl": cache_stats.get("l2_default_ttl", 3600),
                "cache_implementation": "unified_cache_facade",
                "hit_rate": cache_stats.get("overall_hit_rate", 0.0),
            }

            return {
                "metric_type": metric_type,
                "path": path_parts,
                "data": session_metrics,
                "source": "session_store",
                "timestamp": time.time(),
            }

        # Unknown metric type
        return {
            "metric_type": metric_type,
            "path": path_parts,
            "data": {},
            "message": f"Unknown metric type: {path_parts[0]}",
            "available_types": ["performance", "errors", "sessions"],
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get hierarchical metrics for '{metric_type}': {e}")
        return {
            "metric_type": metric_type,
            "error": str(e),
            "timestamp": time.time(),
        }
