"""
Async test context manager for ML testing.

This module contains async context management utilities extracted from conftest.py
to maintain clean architecture and separation of concerns.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class AsyncTestContextManager:
    """Async context manager for comprehensive fixture lifecycle management.

    Provides automatic setup/teardown coordination, resource tracking,
    and error handling for complex async test scenarios.
    """
    
    def __init__(self):
        self.active_contexts = {}
        self.resource_registry = {}
        self.cleanup_order = []

    @asynccontextmanager
    async def managed_test_lifecycle(
        self, test_name: str, services: list[Any], cleanup_timeout: int = 30
    ):
        """Manage complete test lifecycle with automatic cleanup."""
        start_time = time.perf_counter()
        context_id = f"{test_name}_{id(self)}"
        self.active_contexts[context_id] = {
            "test_name": test_name,
            "services": services,
            "start_time": start_time,
            "status": "initializing",
        }
        try:
            init_tasks = []
            for service in services:
                if hasattr(service, "initialize") and asyncio.iscoroutinefunction(
                    service.initialize
                ):
                    init_tasks.append(service.initialize())
            if init_tasks:
                await asyncio.gather(*init_tasks)
            self.active_contexts[context_id]["status"] = "running"
            yield {
                "context_id": context_id,
                "services": {
                    f"service_{i}": service for i, service in enumerate(services)
                },
                "start_time": start_time,
            }
            self.active_contexts[context_id]["status"] = "completed"
        except Exception as e:
            self.active_contexts[context_id]["status"] = "error"
            self.active_contexts[context_id]["error"] = str(e)
            raise
        finally:
            cleanup_tasks = []
            for service in reversed(services):
                if hasattr(service, "shutdown") and asyncio.iscoroutinefunction(
                    service.shutdown
                ):
                    cleanup_tasks.append(service.shutdown())
            if cleanup_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*cleanup_tasks, return_exceptions=True),
                        timeout=cleanup_timeout,
                    )
                except TimeoutError:
                    logger.warning(f"Cleanup timeout for context {context_id}")
            self.active_contexts[context_id]["duration_ms"] = (
                time.perf_counter() - start_time
            ) * 1000

    async def register_resource(
        self,
        resource_name: str,
        resource: Any,
        cleanup_function: Callable | None = None,
    ) -> None:
        """Register a resource for automatic cleanup."""
        self.resource_registry[resource_name] = {
            "resource": resource,
            "cleanup_function": cleanup_function,
            "registered_at": aware_utc_now().isoformat(),
        }
        self.cleanup_order.append(resource_name)

    async def cleanup_all_resources(self) -> dict[str, Any]:
        """Clean up all registered resources."""
        cleanup_results = {"successful": [], "failed": []}
        for resource_name in reversed(self.cleanup_order):
            if resource_name in self.resource_registry:
                resource_info = self.resource_registry[resource_name]
                try:
                    if resource_info["cleanup_function"]:
                        cleanup_func = resource_info["cleanup_function"]
                        if asyncio.iscoroutinefunction(cleanup_func):
                            await cleanup_func(resource_info["resource"])
                        else:
                            cleanup_func(resource_info["resource"])
                    cleanup_results["successful"].append(resource_name)
                except Exception as e:
                    cleanup_results["failed"].append({
                        "resource": resource_name,
                        "error": str(e),
                    })
        self.resource_registry.clear()
        self.cleanup_order.clear()
        return cleanup_results

    def get_active_contexts(self) -> dict[str, dict[str, Any]]:
        """Get information about all active test contexts."""
        return self.active_contexts.copy()