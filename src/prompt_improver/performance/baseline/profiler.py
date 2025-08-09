"""Continuous profiling system for performance monitoring."""

import asyncio
import cProfile
import functools
import io
import logging
import pstats
import sys
import threading
import time
import tracemalloc
import uuid
import weakref
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, ParamSpec, Tuple, TypeVar, Union

from prompt_improver.performance.baseline.models import ProfileData

# Enhanced background task management
from prompt_improver.performance.monitoring.health.background_manager import (
    TaskPriority,
    get_background_task_manager,
)

# Advanced profiling
try:
    import py_spy

    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


logger = logging.getLogger(__name__)


class ProfilerConfig:
    """Configuration for the continuous profiler."""

    def __init__(
        self,
        enable_cpu_profiling: bool = True,
        enable_memory_profiling: bool = True,
        enable_call_tracing: bool = True,
        sampling_interval: float = 0.01,  # 10ms
        profile_duration: int = 60,  # 60 seconds
        memory_threshold_mb: float = 100.0,  # Profile when memory increases by 100MB
        slow_function_threshold_ms: float = 50.0,  # Functions taking >50ms
        max_call_stack_depth: int = 20,
        output_directory: Path | None = None,
    ):
        """Initialize profiler configuration.

        Args:
            enable_cpu_profiling: Enable CPU time profiling
            enable_memory_profiling: Enable memory usage profiling
            enable_call_tracing: Enable function call tracing
            sampling_interval: Profiling sample interval in seconds
            profile_duration: Duration of profiling sessions in seconds
            memory_threshold_mb: Memory increase threshold for triggering profiling
            slow_function_threshold_ms: Threshold for identifying slow functions
            max_call_stack_depth: Maximum depth for call stack analysis
            output_directory: Directory to store profile outputs
        """
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_call_tracing = enable_call_tracing
        self.sampling_interval = sampling_interval
        self.profile_duration = profile_duration
        self.memory_threshold_mb = memory_threshold_mb
        self.slow_function_threshold_ms = slow_function_threshold_ms
        self.max_call_stack_depth = max_call_stack_depth
        self.output_directory = output_directory or Path("./profiles")
        self.output_directory.mkdir(parents=True, exist_ok=True)


class FunctionCallData:
    """Data structure for tracking function calls."""

    def __init__(
        self, function_name: str, module_name: str, filename: str, line_number: int
    ):
        self.function_name = function_name
        self.module_name = module_name
        self.filename = filename
        self.line_number = line_number
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0
        self.memory_usage = []
        self.last_called = None

    def record_call(self, duration: float, memory_mb: float | None = None):
        """Record a function call."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_called = datetime.now(UTC)

        if memory_mb is not None:
            self.memory_usage.append(memory_mb)

    @property
    def average_time(self) -> float:
        """Get average call time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "filename": self.filename,
            "line_number": self.line_number,
            "call_count": self.call_count,
            "total_time_ms": self.total_time * 1000,
            "average_time_ms": self.average_time * 1000,
            "min_time_ms": self.min_time * 1000 if self.min_time != float("inf") else 0,
            "max_time_ms": self.max_time * 1000,
            "memory_usage_mb": self.memory_usage,
            "last_called": self.last_called.isoformat() if self.last_called else None,
        }


class ContinuousProfiler:
    """Continuous profiling system for performance monitoring.

    Provides ongoing profiling of CPU usage, memory consumption, and function
    call patterns to identify performance hotspots and bottlenecks.
    """

    def __init__(self, config: ProfilerConfig | None = None):
        """Initialize continuous profiler.

        Args:
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        self._running = False
        self._profiling_task: asyncio.Task | None = None
        self._lock = threading.Lock()

        # Profiling state
        self._current_profile: cProfile.Profile | None = None
        self._function_calls: dict[str, FunctionCallData] = {}
        self._memory_snapshots: list[tuple[datetime, float]] = []
        self._cpu_samples: list[tuple[datetime, float]] = []

        # Active profiling sessions
        self._active_profiles: dict[str, ProfileData] = {}

        # Setup memory tracking if enabled
        if self.config.enable_memory_profiling and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Started memory tracing")

        logger.info(
            f"ContinuousProfiler initialized (CPU: {self.config.enable_cpu_profiling}, Memory: {self.config.enable_memory_profiling})"
        )

    async def start_profiling(self) -> None:
        """Start continuous profiling."""
        if self._running:
            logger.warning("Profiling already running")
            return

        self._running = True
        # Use EnhancedBackgroundTaskManager for profiling loop
        task_manager = get_background_task_manager()
        self._profiling_task_id = await task_manager.submit_enhanced_task(
            task_id=f"baseline_profiler_loop_{str(uuid.uuid4())[:8]}",
            coroutine=self._profiling_loop(),
            priority=TaskPriority.NORMAL,
            tags={
                "service": "performance",
                "type": "profiling",
                "component": "continuous_profiler",
            },
        )
        logger.info("Started continuous profiling")

    async def stop_profiling(self) -> None:
        """Stop continuous profiling."""
        if not self._running:
            return

        self._running = False

        if self._profiling_task:
            self._profiling_task.cancel()
            try:
                await self._profiling_task
            except asyncio.CancelledError:
                pass

        # Stop any active cProfile session
        if self._current_profile:
            self._current_profile.disable()
            self._current_profile = None

        logger.info("Stopped continuous profiling")

    def get_profiler_status(self) -> dict[str, Any]:
        """Get profiler status information."""
        return {
            "running": self._running,
            "config": {
                "cpu_profiling_enabled": self.config.enable_cpu_profiling,
                "memory_profiling_enabled": self.config.enable_memory_profiling,
                "call_tracing_enabled": self.config.enable_call_tracing,
                "sampling_interval": self.config.sampling_interval,
                "profile_duration": self.config.profile_duration,
            },
            "profile_count": len(self._profile_history),
            "output_directory": str(self.config.output_directory),
        }

    async def generate_profile_report(self) -> dict[str, Any]:
        """Generate a comprehensive profiling report."""
        if not self._profile_history:
            return {"error": "No profile data available"}

        latest_profile = self._profile_history[-1]

        # Basic statistics
        report = {
            "timestamp": latest_profile.timestamp.isoformat(),
            "session_id": latest_profile.session_id,
            "duration_seconds": latest_profile.duration_seconds,
            "cpu_time_seconds": latest_profile.cpu_time_seconds,
            "memory_peak_mb": latest_profile.memory_peak_mb,
            "memory_current_mb": latest_profile.memory_current_mb,
            "function_count": len(latest_profile.call_stats),
            "top_functions": [],
        }

        # Top functions by execution time
        sorted_functions = sorted(
            latest_profile.call_stats.items(),
            key=lambda x: x[1].get("cumulative_time", 0),
            reverse=True,
        )[:10]

        for func_name, stats in sorted_functions:
            report["top_functions"].append({
                "function": func_name,
                "calls": stats.get("call_count", 0),
                "total_time": stats.get("cumulative_time", 0),
                "per_call_time": stats.get("cumulative_time", 0)
                / max(stats.get("call_count", 1), 1),
            })

        return report


# Context managers for profiling specific blocks
@contextmanager
def profile_block(name: str = "block"):
    """Context manager for profiling a code block."""
    profiler = cProfile.Profile()
    start_time = time.time()

    try:
        profiler.enable()
        yield profiler
    finally:
        profiler.disable()
        end_time = time.time()

        # Create profile output
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        logger.info("Profile block '{name}' completed in %ss", end_time - start_time:.3f)
        logger.debug("Profile output:\n%s", stream.getvalue())


@asynccontextmanager
async def profile_async_block(name: str = "async_block"):
    """Context manager for profiling an async code block."""
    profiler = cProfile.Profile()
    start_time = time.time()

    try:
        profiler.enable()
        yield profiler
    finally:
        profiler.disable()
        end_time = time.time()

        # Create profile output
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        logger.info(
            f"Async profile block '{name}' completed in {end_time - start_time:.3f}s"
        )
        logger.debug("Profile output:\n%s", stream.getvalue())


# Decorator for profiling functions
P = ParamSpec("P")
T = TypeVar("T")


def profile(func: Callable[P, T] | None = None, *, name: str | None = None):
    """Decorator for profiling function execution."""

    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        profile_name = name or f"{f.__module__}.{f.__name__}"

        if asyncio.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                async with profile_async_block(profile_name):
                    return await f(*args, **kwargs)

            return async_wrapper

        @functools.wraps(f)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with profile_block(profile_name):
                return f(*args, **kwargs)

        return sync_wrapper

    if func is None:
        return decorator
    return decorator(func)


# Global profiler instance
_global_profiler: ContinuousProfiler | None = None


def get_profiler(config: ProfilerConfig | None = None) -> ContinuousProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = ContinuousProfiler(config or ProfilerConfig())
    return _global_profiler


async def start_continuous_profiling(config: ProfilerConfig | None = None) -> None:
    """Start continuous profiling with optional configuration."""
    profiler = get_profiler(config)
    await profiler.start_profiling()


async def stop_continuous_profiling() -> None:
    """Stop continuous profiling."""
    global _global_profiler
    if _global_profiler:
        await _global_profiler.stop_profiling()


def get_performance_summary() -> dict[str, Any]:
    """Get a summary of performance profiling data."""
    global _global_profiler
    if _global_profiler is None:
        return {"error": "Profiler not initialized"}

    status = _global_profiler.get_profiler_status()

    # Add summary information
    if (
        hasattr(_global_profiler, "_profile_history")
        and _global_profiler._profile_history
    ):
        latest_profile = _global_profiler._profile_history[-1]
        status.update({
            "latest_profile": {
                "timestamp": latest_profile.timestamp.isoformat(),
                "duration_seconds": latest_profile.duration_seconds,
                "cpu_time_seconds": latest_profile.cpu_time_seconds,
                "memory_peak_mb": latest_profile.memory_peak_mb,
                "function_count": len(latest_profile.call_stats),
            }
        })

    return status
