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
        self._active_profiles: Dict[str, ProfileData] = {}\n        \n        # Setup memory tracking if enabled\n        if self.config.enable_memory_profiling and not tracemalloc.is_tracing():\n            tracemalloc.start()\n            logger.info(\"Started memory tracing\")\n        \n        logger.info(f\"ContinuousProfiler initialized (CPU: {self.config.enable_cpu_profiling}, Memory: {self.config.enable_memory_profiling})\")\n\n    async def start_profiling(self) -> None:\n        \"\"\"Start continuous profiling.\"\"\"\n        if self._running:\n            logger.warning(\"Profiling already running\")\n            return\n        \n        self._running = True\n        self._profiling_task = asyncio.create_task(self._profiling_loop())\n        logger.info(\"Started continuous profiling\")\n\n    async def stop_profiling(self) -> None:\n        \"\"\"Stop continuous profiling.\"\"\"\n        if not self._running:\n            return\n        \n        self._running = False\n        \n        if self._profiling_task:\n            self._profiling_task.cancel()\n            try:\n                await self._profiling_task\n            except asyncio.CancelledError:\n                pass\n        \n        # Stop any active cProfile session\n        if self._current_profile:\n            self._current_profile.disable()\n            self._current_profile = None\n        \n        logger.info(\"Stopped continuous profiling\")\n\n    async def _profiling_loop(self) -> None:\n        \"\"\"Main profiling loop.\"\"\"\n        while self._running:\n            try:\n                # Start a profiling session\n                profile_data = await self._run_profiling_session()\n                \n                # Save profile data\n                if profile_data:\n                    await self._save_profile_data(profile_data)\n                \n                # Wait before next session\n                await asyncio.sleep(self.config.profile_duration)\n                \n            except Exception as e:\n                logger.error(f\"Error in profiling loop: {e}\")\n                await asyncio.sleep(10)  # Wait before retrying\n\n    async def _run_profiling_session(self) -> Optional[ProfileData]:\n        \"\"\"Run a single profiling session.\"\"\"\n        session_id = str(uuid.uuid4())\n        start_time = datetime.now(timezone.utc)\n        \n        logger.debug(f\"Starting profiling session {session_id}\")\n        \n        # Initialize profile data\n        profile_data = ProfileData(\n            profile_id=session_id,\n            operation_name=\"continuous_profiling\",\n            start_time=start_time,\n            end_time=start_time,  # Will be updated\n            duration_ms=0.0       # Will be updated\n        )\n        \n        session_start = time.time()\n        \n        # Start CPU profiling if enabled\n        if self.config.enable_cpu_profiling:\n            self._current_profile = cProfile.Profile()\n            self._current_profile.enable()\n        \n        # Collect data during the session\n        end_time = session_start + self.config.profile_duration\n        sample_count = 0\n        \n        while time.time() < end_time and self._running:\n            # Collect memory snapshot\n            if self.config.enable_memory_profiling:\n                await self._collect_memory_snapshot(profile_data)\n            \n            # Collect CPU sample\n            if PSUTIL_AVAILABLE:\n                await self._collect_cpu_sample(profile_data)\n            \n            sample_count += 1\n            await asyncio.sleep(self.config.sampling_interval)\n        \n        # Stop CPU profiling\n        if self._current_profile:\n            self._current_profile.disable()\n            \n            # Analyze CPU profile\n            await self._analyze_cpu_profile(profile_data, self._current_profile)\n            \n            self._current_profile = None\n        \n        # Update timing\n        profile_data.end_time = datetime.now(timezone.utc)\n        profile_data.duration_ms = (time.time() - session_start) * 1000\n        profile_data.metadata['sample_count'] = sample_count\n        \n        logger.debug(f\"Completed profiling session {session_id} ({sample_count} samples)\")\n        \n        return profile_data\n\n    async def _collect_memory_snapshot(self, profile_data: ProfileData) -> None:\n        \"\"\"Collect memory usage snapshot.\"\"\"\n        try:\n            timestamp = datetime.now(timezone.utc)\n            \n            # Get tracemalloc data if available\n            if tracemalloc.is_tracing():\n                current, peak = tracemalloc.get_traced_memory()\n                snapshot = {\n                    'timestamp': timestamp.isoformat(),\n                    'current_mb': current / (1024 * 1024),\n                    'peak_mb': peak / (1024 * 1024),\n                    'source': 'tracemalloc'\n                }\n                profile_data.memory_snapshots.append(snapshot)\n            \n            # Get psutil data if available\n            if PSUTIL_AVAILABLE:\n                import psutil\n                process = psutil.Process()\n                memory_info = process.memory_info()\n                snapshot = {\n                    'timestamp': timestamp.isoformat(),\n                    'rss_mb': memory_info.rss / (1024 * 1024),\n                    'vms_mb': memory_info.vms / (1024 * 1024),\n                    'source': 'psutil'\n                }\n                profile_data.memory_snapshots.append(snapshot)\n        \n        except Exception as e:\n            logger.debug(f\"Failed to collect memory snapshot: {e}\")\n\n    async def _collect_cpu_sample(self, profile_data: ProfileData) -> None:\n        \"\"\"Collect CPU usage sample.\"\"\"\n        try:\n            if PSUTIL_AVAILABLE:\n                import psutil\n                cpu_percent = psutil.cpu_percent()\n                profile_data.cpu_samples.append(cpu_percent)\n        \n        except Exception as e:\n            logger.debug(f\"Failed to collect CPU sample: {e}\")\n\n    async def _analyze_cpu_profile(self, profile_data: ProfileData, profiler: cProfile.Profile) -> None:\n        \"\"\"Analyze CPU profiling results.\"\"\"\n        try:\n            # Get profile statistics\n            s = io.StringIO()\n            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')\n            ps.print_stats()\n            \n            # Parse statistics\n            stats_output = s.getvalue()\n            profile_data.metadata['cpu_profile_stats'] = stats_output\n            \n            # Extract function call information\n            for func_info, (cc, nc, tt, ct, callers) in profiler.get_stats().items():\n                filename, line_number, function_name = func_info\n                \n                # Skip built-in functions and profiler overhead\n                if '<built-in>' in filename or 'profiler' in filename.lower():\n                    continue\n                \n                call_data = {\n                    'filename': filename,\n                    'line_number': line_number,\n                    'function_name': function_name,\n                    'call_count': cc,\n                    'total_time': tt,\n                    'cumulative_time': ct,\n                    'time_per_call': tt / cc if cc > 0 else 0,\n                    'cumulative_per_call': ct / cc if cc > 0 else 0\n                }\n                \n                profile_data.function_calls.append(call_data)\n                \n                # Identify slow functions\n                if (tt / cc if cc > 0 else 0) * 1000 > self.config.slow_function_threshold_ms:\n                    slow_func = {\n                        'function_name': function_name,\n                        'filename': filename,\n                        'line_number': line_number,\n                        'avg_time_ms': (tt / cc if cc > 0 else 0) * 1000,\n                        'total_time_ms': tt * 1000,\n                        'call_count': cc\n                    }\n                    profile_data.slow_functions.append(slow_func)\n            \n            # Sort slow functions by average time\n            profile_data.slow_functions.sort(key=lambda x: x['avg_time_ms'], reverse=True)\n            \n        except Exception as e:\n            logger.error(f\"Failed to analyze CPU profile: {e}\")\n\n    async def _save_profile_data(self, profile_data: ProfileData) -> None:\n        \"\"\"Save profile data to storage.\"\"\"\n        try:\n            # Create filename with timestamp\n            timestamp_str = profile_data.start_time.strftime(\"%Y%m%d_%H%M%S\")\n            filename = f\"profile_{timestamp_str}_{profile_data.profile_id[:8]}.json\"\n            filepath = self.config.output_directory / filename\n            \n            # Convert to dict and save\n            import json\n            profile_dict = {\n                'profile_id': profile_data.profile_id,\n                'operation_name': profile_data.operation_name,\n                'start_time': profile_data.start_time.isoformat(),\n                'end_time': profile_data.end_time.isoformat(),\n                'duration_ms': profile_data.duration_ms,\n                'function_calls': profile_data.function_calls,\n                'memory_snapshots': profile_data.memory_snapshots,\n                'cpu_samples': profile_data.cpu_samples,\n                'slow_functions': profile_data.slow_functions,\n                'memory_allocations': profile_data.memory_allocations,\n                'tags': profile_data.tags,\n                'metadata': profile_data.metadata\n            }\n            \n            with open(filepath, 'w') as f:\n                json.dump(profile_dict, f, indent=2, default=str)\n            \n            logger.debug(f\"Saved profile data to {filepath}\")\n        \n        except Exception as e:\n            logger.error(f\"Failed to save profile data: {e}\")\n\n    @contextmanager\n    def profile_function(self, function_name: str, operation_name: Optional[str] = None):\n        \"\"\"Context manager for profiling specific functions.\"\"\"\n        start_time = time.time()\n        memory_before = None\n        \n        # Get memory before if tracking is enabled\n        if self.config.enable_memory_profiling and tracemalloc.is_tracing():\n            current, _ = tracemalloc.get_traced_memory()\n            memory_before = current / (1024 * 1024)\n        \n        try:\n            yield\n        finally:\n            duration = time.time() - start_time\n            \n            # Get memory after\n            memory_after = None\n            if self.config.enable_memory_profiling and tracemalloc.is_tracing():\n                current, _ = tracemalloc.get_traced_memory()\n                memory_after = current / (1024 * 1024)\n            \n            # Record function call\n            self._record_function_call(\n                function_name, duration, memory_before, memory_after, operation_name\n            )\n\n    @asynccontextmanager\n    async def profile_async_function(self, function_name: str, operation_name: Optional[str] = None):\n        \"\"\"Async context manager for profiling async functions.\"\"\"\n        start_time = time.time()\n        memory_before = None\n        \n        # Get memory before if tracking is enabled\n        if self.config.enable_memory_profiling and tracemalloc.is_tracing():\n            current, _ = tracemalloc.get_traced_memory()\n            memory_before = current / (1024 * 1024)\n        \n        try:\n            yield\n        finally:\n            duration = time.time() - start_time\n            \n            # Get memory after\n            memory_after = None\n            if self.config.enable_memory_profiling and tracemalloc.is_tracing():\n                current, _ = tracemalloc.get_traced_memory()\n                memory_after = current / (1024 * 1024)\n            \n            # Record function call\n            self._record_function_call(\n                function_name, duration, memory_before, memory_after, operation_name\n            )\n\n    def _record_function_call(\n        self, \n        function_name: str, \n        duration: float,\n        memory_before: Optional[float],\n        memory_after: Optional[float],\n        operation_name: Optional[str]\n    ) -> None:\n        \"\"\"Record a function call for analysis.\"\"\"\n        with self._lock:\n            # Get caller information\n            frame = sys._getframe(2)  # Go up 2 frames to get actual caller\n            module_name = frame.f_globals.get('__name__', 'unknown')\n            filename = frame.f_code.co_filename\n            line_number = frame.f_lineno\n            \n            # Create function key\n            func_key = f\"{module_name}.{function_name}:{line_number}\"\n            \n            # Get or create function call data\n            if func_key not in self._function_calls:\n                self._function_calls[func_key] = FunctionCallData(\n                    function_name, module_name, filename, line_number\n                )\n            \n            # Record the call\n            memory_delta = None\n            if memory_before is not None and memory_after is not None:\n                memory_delta = memory_after - memory_before\n            \n            self._function_calls[func_key].record_call(duration, memory_delta)\n\n    def profile_decorator(self, operation_name: Optional[str] = None):\n        \"\"\"Decorator for profiling functions.\"\"\"\n        def decorator(func):\n            if asyncio.iscoroutinefunction(func):\n                @functools.wraps(func)\n                async def async_wrapper(*args, **kwargs):\n                    async with self.profile_async_function(func.__name__, operation_name):\n                        return await func(*args, **kwargs)\n                return async_wrapper\n            else:\n                @functools.wraps(func)\n                def sync_wrapper(*args, **kwargs):\n                    with self.profile_function(func.__name__, operation_name):\n                        return func(*args, **kwargs)\n                return sync_wrapper\n        return decorator\n\n    async def create_operation_profile(\n        self, \n        operation_name: str,\n        duration_seconds: int = 30\n    ) -> ProfileData:\n        \"\"\"Create a focused profile for a specific operation.\"\"\"\n        profile_id = str(uuid.uuid4())\n        start_time = datetime.now(timezone.utc)\n        \n        logger.info(f\"Starting operation profile '{operation_name}' for {duration_seconds}s\")\n        \n        # Create profile data\n        profile_data = ProfileData(\n            profile_id=profile_id,\n            operation_name=operation_name,\n            start_time=start_time,\n            end_time=start_time,  # Will be updated\n            duration_ms=0.0,       # Will be updated\n            tags={'operation': operation_name, 'focused': True}\n        )\n        \n        # Store as active profile\n        self._active_profiles[profile_id] = profile_data\n        \n        # Start profiling\n        session_start = time.time()\n        profiler = None\n        \n        if self.config.enable_cpu_profiling:\n            profiler = cProfile.Profile()\n            profiler.enable()\n        \n        # Collect samples during the operation\n        end_time = session_start + duration_seconds\n        while time.time() < end_time:\n            if self.config.enable_memory_profiling:\n                await self._collect_memory_snapshot(profile_data)\n            \n            if PSUTIL_AVAILABLE:\n                await self._collect_cpu_sample(profile_data)\n            \n            await asyncio.sleep(self.config.sampling_interval)\n        \n        # Stop profiling\n        if profiler:\n            profiler.disable()\n            await self._analyze_cpu_profile(profile_data, profiler)\n        \n        # Update timing\n        profile_data.end_time = datetime.now(timezone.utc)\n        profile_data.duration_ms = (time.time() - session_start) * 1000\n        \n        # Remove from active profiles\n        del self._active_profiles[profile_id]\n        \n        # Save profile\n        await self._save_profile_data(profile_data)\n        \n        logger.info(f\"Completed operation profile '{operation_name}'\")\n        return profile_data\n\n    def get_function_call_summary(self, limit: int = 20) -> List[Dict[str, Any]]:\n        \"\"\"Get summary of function call statistics.\"\"\"\n        with self._lock:\n            # Sort by total time\n            sorted_functions = sorted(\n                self._function_calls.values(),\n                key=lambda x: x.total_time,\n                reverse=True\n            )\n            \n            return [func.to_dict() for func in sorted_functions[:limit]]\n\n    def get_slow_functions(self, limit: int = 10) -> List[Dict[str, Any]]:\n        \"\"\"Get functions that are consistently slow.\"\"\"\n        with self._lock:\n            slow_functions = []\n            \n            for func_data in self._function_calls.values():\n                if func_data.average_time * 1000 > self.config.slow_function_threshold_ms:\n                    slow_functions.append(func_data.to_dict())\n            \n            # Sort by average time\n            slow_functions.sort(key=lambda x: x['average_time_ms'], reverse=True)\n            \n            return slow_functions[:limit]\n\n    def get_memory_hotspots(self) -> List[Dict[str, Any]]:\n        \"\"\"Get functions with high memory usage.\"\"\"\n        with self._lock:\n            memory_hotspots = []\n            \n            for func_data in self._function_calls.values():\n                if func_data.memory_usage:\n                    avg_memory = sum(func_data.memory_usage) / len(func_data.memory_usage)\n                    max_memory = max(func_data.memory_usage)\n                    \n                    if max_memory > self.config.memory_threshold_mb / 10:  # 10% of threshold\n                        hotspot = func_data.to_dict()\n                        hotspot['average_memory_mb'] = avg_memory\n                        hotspot['max_memory_mb'] = max_memory\n                        memory_hotspots.append(hotspot)\n            \n            # Sort by max memory usage\n            memory_hotspots.sort(key=lambda x: x['max_memory_mb'], reverse=True)\n            \n            return memory_hotspots\n\n    def get_profiler_status(self) -> Dict[str, Any]:\n        \"\"\"Get current profiler status and statistics.\"\"\"\n        with self._lock:\n            return {\n                'running': self._running,\n                'config': {\n                    'cpu_profiling': self.config.enable_cpu_profiling,\n                    'memory_profiling': self.config.enable_memory_profiling,\n                    'call_tracing': self.config.enable_call_tracing,\n                    'sampling_interval': self.config.sampling_interval,\n                    'profile_duration': self.config.profile_duration\n                },\n                'statistics': {\n                    'tracked_functions': len(self._function_calls),\n                    'active_profiles': len(self._active_profiles),\n                    'memory_snapshots': len(self._memory_snapshots),\n                    'cpu_samples': len(self._cpu_samples)\n                },\n                'capabilities': {\n                    'tracemalloc': tracemalloc.is_tracing(),\n                    'psutil': PSUTIL_AVAILABLE,\n                    'py_spy': PY_SPY_AVAILABLE,\n                    'memory_profiler': MEMORY_PROFILER_AVAILABLE\n                }\n            }\n\n    async def reset_statistics(self) -> None:\n        \"\"\"Reset all collected profiling statistics.\"\"\"\n        with self._lock:\n            self._function_calls.clear()\n            self._memory_snapshots.clear()\n            self._cpu_samples.clear()\n        \n        logger.info(\"Reset profiler statistics\")\n\n# Global profiler instance\n_global_profiler: Optional[ContinuousProfiler] = None\n\ndef get_profiler() -> ContinuousProfiler:\n    \"\"\"Get the global profiler instance.\"\"\"\n    global _global_profiler\n    if _global_profiler is None:\n        _global_profiler = ContinuousProfiler()\n    return _global_profiler\n\ndef set_profiler(profiler: ContinuousProfiler) -> None:\n    \"\"\"Set the global profiler instance.\"\"\"\n    global _global_profiler\n    _global_profiler = profiler\n\n# Convenience decorators and functions\ndef profile(operation_name: Optional[str] = None):\n    \"\"\"Decorator for profiling functions.\"\"\"\n    profiler = get_profiler()\n    return profiler.profile_decorator(operation_name)\n\n@contextmanager\ndef profile_block(operation_name: str):\n    \"\"\"Context manager for profiling code blocks.\"\"\"\n    profiler = get_profiler()\n    with profiler.profile_function(operation_name):\n        yield\n\n@asynccontextmanager\nasync def profile_async_block(operation_name: str):\n    \"\"\"Async context manager for profiling code blocks.\"\"\"\n    profiler = get_profiler()\n    async with profiler.profile_async_function(operation_name):\n        yield\n\nasync def start_continuous_profiling() -> None:\n    \"\"\"Start the global continuous profiler.\"\"\"\n    profiler = get_profiler()\n    await profiler.start_profiling()\n\nasync def stop_continuous_profiling() -> None:\n    \"\"\"Stop the global continuous profiler.\"\"\"\n    profiler = get_profiler()\n    await profiler.stop_profiling()\n\ndef get_performance_summary() -> Dict[str, Any]:\n    \"\"\"Get a summary of performance statistics.\"\"\"\n    profiler = get_profiler()\n    return {\n        'slow_functions': profiler.get_slow_functions(5),\n        'memory_hotspots': profiler.get_memory_hotspots()[:5],\n        'top_functions': profiler.get_function_call_summary(10),\n        'profiler_status': profiler.get_profiler_status()\n    }