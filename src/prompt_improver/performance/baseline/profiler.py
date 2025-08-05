"""Continuous profiling system for performance monitoring."""

import asyncio
import cProfile
import functools
import io
import logging
import pstats
import sys
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import uuid
import threading
import weakref

# Enhanced background task management
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

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

from .models import ProfileData

logger = logging.getLogger(__name__)

class ProfilerConfig:
    """Configuration for the continuous profiler."""
    
    def __init__(
        self,
        enable_cpu_profiling: bool = True,
        enable_memory_profiling: bool = True,
        enable_call_tracing: bool = True,
        sampling_interval: float = 0.01,  # 10ms
        profile_duration: int = 60,       # 60 seconds
        memory_threshold_mb: float = 100.0,  # Profile when memory increases by 100MB
        slow_function_threshold_ms: float = 50.0,  # Functions taking >50ms
        max_call_stack_depth: int = 20,
        output_directory: Optional[Path] = None
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
    
    def __init__(self, function_name: str, module_name: str, filename: str, line_number: int):
        self.function_name = function_name
        self.module_name = module_name
        self.filename = filename
        self.line_number = line_number
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.memory_usage = []
        self.last_called = None
    
    def record_call(self, duration: float, memory_mb: Optional[float] = None):
        """Record a function call."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_called = datetime.now(timezone.utc)
        
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)
    
    @property
    def average_time(self) -> float:
        """Get average call time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'module_name': self.module_name,
            'filename': self.filename,
            'line_number': self.line_number,
            'call_count': self.call_count,
            'total_time_ms': self.total_time * 1000,
            'average_time_ms': self.average_time * 1000,
            'min_time_ms': self.min_time * 1000 if self.min_time != float('inf') else 0,
            'max_time_ms': self.max_time * 1000,
            'memory_usage_mb': self.memory_usage,
            'last_called': self.last_called.isoformat() if self.last_called else None
        }

class ContinuousProfiler:
    """Continuous profiling system for performance monitoring.
    
    Provides ongoing profiling of CPU usage, memory consumption, and function
    call patterns to identify performance hotspots and bottlenecks.
    """
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        """Initialize continuous profiler.
        
        Args:
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        self._running = False
        self._profiling_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Profiling state
        self._current_profile: Optional[cProfile.Profile] = None
        self._function_calls: Dict[str, FunctionCallData] = {}
        self._memory_snapshots: List[Tuple[datetime, float]] = []
        self._cpu_samples: List[Tuple[datetime, float]] = []
        
        # Active profiling sessions
        self._active_profiles: Dict[str, ProfileData] = {}

        # Setup memory tracking if enabled
        if self.config.enable_memory_profiling and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Started memory tracing")

        logger.info(f"ContinuousProfiler initialized (CPU: {self.config.enable_cpu_profiling}, Memory: {self.config.enable_memory_profiling})")

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
            tags={"service": "performance", "type": "profiling", "component": "continuous_profiler"}
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