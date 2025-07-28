"""Event loop manager with uvloop integration and performance optimization.

This module provides utilities for managing asyncio event loops with uvloop
integration when available, along with performance monitoring and optimization.
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

class EventLoopManager:
    """Manages asyncio event loop with uvloop integration and performance monitoring."""

    def __init__(self):
        self._uvloop_available = False
        self._uvloop_enabled = False
        self._loop_type = "asyncio"
        self._policy_set = False
        self._performance_metrics: dict[str, Any] = {}

    def setup_uvloop(self, force: bool = False) -> bool:
        """Setup uvloop event loop policy when available.

        Following uvloop best practices:
        - Use asyncio.set_event_loop_policy(uvloop.EventLoopPolicy()) for Python 3.11+
        - Provide fallback to standard asyncio when uvloop unavailable
        - Setup should happen before any event loop operations

        Args:
            force: Force uvloop setup even if already configured

        Returns:
            True if uvloop was successfully configured, False otherwise
        """
        if self._policy_set and not force:
            logger.debug("Event loop policy already set, skipping")
            return self._uvloop_enabled

        try:
            import uvloop

            self._uvloop_available = True

            # Check if we already have a running loop (which would prevent policy change)
            try:
                asyncio.get_running_loop()
                logger.warning("Cannot set uvloop policy: event loop already running")
                # Still detect if current loop is uvloop
                current_loop = asyncio.get_running_loop()
                if "uvloop" in type(current_loop).__name__.lower():
                    self._uvloop_enabled = True
                    self._loop_type = "uvloop"
                    logger.info("Detected existing uvloop event loop")
                return self._uvloop_enabled
            except RuntimeError:
                # No running loop, safe to set policy
                pass

            # Set uvloop as the event loop policy (best practice for Python 3.11+)
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            self._loop_type = "uvloop"
            self._uvloop_enabled = True
            self._policy_set = True
            logger.info("uvloop event loop policy configured successfully")
            return True

        except ImportError:
            logger.info("uvloop not available, using standard asyncio event loop")
            self._uvloop_available = False
            self._uvloop_enabled = False
            self._loop_type = "asyncio"
            return False
        except Exception as e:
            logger.warning(f"Failed to configure uvloop: {e}")
            self._uvloop_available = False
            self._uvloop_enabled = False
            self._loop_type = "asyncio"
            return False

    def create_event_loop(self) -> asyncio.AbstractEventLoop:
        """Create a new event loop with optimal configuration.

        Returns:
            Configured event loop instance
        """
        if self._uvloop_enabled:
            try:
                import uvloop

                loop = uvloop.new_event_loop()
                logger.debug("Created new uvloop event loop")
                return loop
            except Exception as e:
                logger.warning(f"Failed to create uvloop, falling back to asyncio: {e}")

        # Fallback to standard asyncio
        loop = asyncio.new_event_loop()
        logger.debug("Created new asyncio event loop")
        return loop

    def get_loop_info(self) -> dict[str, Any]:
        """Get information about the current event loop.

        Returns:
            Dictionary containing loop information
        """
        try:
            loop = asyncio.get_running_loop()
            loop_type = type(loop).__name__
            uvloop_detected = "uvloop" in loop_type.lower()

            return {
                "loop_type": loop_type,
                "uvloop_available": self._uvloop_available,
                "uvloop_enabled": self._uvloop_enabled,
                "uvloop_detected": uvloop_detected,
                "policy_set": self._policy_set,
                "loop_running": True,
                "loop_debug": loop.get_debug(),
            }
        except RuntimeError:
            return {
                "loop_type": "none",
                "uvloop_available": self._uvloop_available,
                "uvloop_enabled": self._uvloop_enabled,
                "uvloop_detected": False,
                "policy_set": self._policy_set,
                "loop_running": False,
                "loop_debug": False,
            }

    async def benchmark_loop_latency(self, samples: int = 100) -> dict[str, float]:
        """Benchmark event loop latency.

        Args:
            samples: Number of samples to collect

        Returns:
            Dictionary with latency statistics in milliseconds
        """
        latencies = []

        for _ in range(samples):
            start = time.perf_counter()
            await asyncio.sleep(0)  # Yield control to event loop
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        if not latencies:
            return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "samples": 0}

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        metrics = {
            "avg_ms": avg_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
            "samples": len(latencies),
            "total_time_ms": sum(latencies),
        }

        # Store performance metrics
        self._performance_metrics["latency_benchmark"] = metrics

        logger.info(
            f"Event loop latency benchmark: "
            f"avg={avg_latency:.2f}ms, min={min_latency:.2f}ms, max={max_latency:.2f}ms "
            f"(samples={len(latencies)})"
        )

        return metrics

    async def benchmark_task_throughput(
        self, task_count: int = 1000
    ) -> dict[str, float]:
        """Benchmark task creation and execution throughput.

        Args:
            task_count: Number of tasks to create and execute

        Returns:
            Dictionary with throughput statistics
        """

        async def dummy_task():
            """Dummy task for benchmarking."""
            await asyncio.sleep(0.001)  # Small delay
            return True

        start_time = time.perf_counter()

        # Create and execute tasks
        tasks = [asyncio.create_task(dummy_task()) for _ in range(task_count)]
        results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        successful_tasks = sum(1 for result in results if result)
        throughput = successful_tasks / (duration_ms / 1000)  # tasks per second

        metrics = {
            "task_count": task_count,
            "successful_tasks": successful_tasks,
            "duration_ms": duration_ms,
            "throughput_per_second": throughput,
            "avg_task_time_ms": duration_ms / task_count,
        }

        # Store performance metrics
        self._performance_metrics["throughput_benchmark"] = metrics

        logger.info(
            f"Event loop throughput benchmark: "
            f"{successful_tasks}/{task_count} tasks in {duration_ms:.2f}ms "
            f"({throughput:.2f} tasks/sec)"
        )

        return metrics

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get stored performance metrics.

        Returns:
            Dictionary containing all performance metrics
        """
        return self._performance_metrics.copy()

    def is_uvloop_enabled(self) -> bool:
        """Check if uvloop is enabled.

        Returns:
            True if uvloop is enabled, False otherwise
        """
        return self._uvloop_enabled

    def get_loop_type(self) -> str:
        """Get the current loop type.

        Returns:
            String describing the loop type
        """
        return self._loop_type

# Global instance
_event_loop_manager = EventLoopManager()

def get_event_loop_manager() -> EventLoopManager:
    """Get the global event loop manager instance.

    Returns:
        Global EventLoopManager instance
    """
    return _event_loop_manager

def setup_uvloop(force: bool = False) -> bool:
    """Setup uvloop event loop policy when available.

    Args:
        force: Force uvloop setup even if already configured

    Returns:
        True if uvloop was successfully configured, False otherwise
    """
    return _event_loop_manager.setup_uvloop(force=force)

async def benchmark_event_loop(
    latency_samples: int = 100, throughput_tasks: int = 1000
) -> dict[str, Any]:
    """Run comprehensive event loop benchmarks.

    Args:
        latency_samples: Number of samples for latency test
        throughput_tasks: Number of tasks for throughput test

    Returns:
        Dictionary with benchmark results
    """
    manager = get_event_loop_manager()

    # Run latency benchmark
    latency_metrics = await manager.benchmark_loop_latency(latency_samples)

    # Run throughput benchmark
    throughput_metrics = await manager.benchmark_task_throughput(throughput_tasks)

    # Get loop info
    loop_info = manager.get_loop_info()

    return {
        "loop_info": loop_info,
        "latency": latency_metrics,
        "throughput": throughput_metrics,
        "timestamp": time.time(),
    }
