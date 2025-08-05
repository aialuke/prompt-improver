"""Unified Event Loop Manager for high-performance async operations.

This module provides comprehensive event loop management with session-scoped
tracking, performance benchmarking, and uvloop optimization support.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set, TypeVar, Coroutine, cast
from dataclasses import dataclass, field

from ..performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

logger = logging.getLogger(__name__)

T = TypeVar("T")

__version__ = "2.0.0"
__all__ = [
    "UnifiedLoopManager",
    "SessionMetrics",
    "get_unified_loop_manager",
]


@dataclass
class SessionMetrics:
    """Session-specific metrics tracking."""
    operations_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    error_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_operation_at: Optional[float] = None


class UnifiedLoopManager:
    """High-performance event loop manager with session tracking and uvloop support.
    
    Features:
    - Automatic uvloop optimization when available
    - Session-scoped performance metrics and tracking
    - Comprehensive benchmarking capabilities
    - Task lifecycle management per session
    - Thread-safe singleton pattern for global instance
    
    This manager follows Python 3.11+ best practices for event loop management
    and provides production-ready async infrastructure.
    """

    def __init__(self) -> None:
        """Initialize unified loop manager with performance optimization and session tracking."""
        # Global event loop state
        self._uvloop_available = False
        self._uvloop_enabled = False
        self._loop_type = "asyncio"
        self._policy_set = False
        self._performance_metrics: Dict[str, Any] = {}
        
        # Session management state
        self._sessions: Dict[str, SessionMetrics] = {}
        self._session_active_tasks: Dict[str, Set[asyncio.Task[Any]]] = {}
        self._default_timeout = 30.0
        
        logger.info("UnifiedLoopManager initialized")

    # ============================================================================
    # Event Loop Configuration and Optimization
    # ============================================================================

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

                uvloop_instance = uvloop.new_event_loop()
                logger.debug("Created new uvloop event loop")
                return cast(asyncio.AbstractEventLoop, uvloop_instance)
            except Exception as e:
                logger.warning(f"Failed to create uvloop, falling back to asyncio: {e}")

        # Fallback to standard asyncio
        asyncio_loop = asyncio.new_event_loop()
        logger.debug("Created new asyncio event loop")
        return asyncio_loop

    def get_loop_info(self) -> Dict[str, Any]:
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
                "unified_manager": True,
                "active_sessions": len(self._sessions),
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
                "unified_manager": True,
                "active_sessions": len(self._sessions),
            }

    async def benchmark_loop_latency(self, samples: int = 100) -> Dict[str, float]:
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

    async def benchmark_task_throughput(self, task_count: int = 1000) -> Dict[str, float]:
        """Benchmark task creation and execution throughput.

        Args:
            task_count: Number of tasks to create and execute

        Returns:
            Dictionary with throughput statistics
        """

        async def dummy_task() -> bool:
            """Dummy task for benchmarking."""
            await asyncio.sleep(0.001)  # Small delay
            return True

        start_time = time.perf_counter()

        # Create and execute tasks via EnhancedBackgroundTaskManager for batch operations
        task_manager = get_background_task_manager()
        
        # Submit all tasks as a batch operation
        task_coroutines = [dummy_task() for _ in range(task_count)]
        batch_task_id = await task_manager.submit_enhanced_task(
            task_id=f"benchmark_batch_{str(uuid.uuid4())[:8]}",
            coroutine=asyncio.gather(*task_coroutines),
            priority=TaskPriority.LOW,
            tags={
                "service": "benchmarking", 
                "type": "throughput_test", 
                "component": "unified_loop_manager",
                "task_count": str(task_count)
            }
        )
        
        # Wait for batch completion and get results
        task_status = await task_manager.wait_for_completion(batch_task_id, timeout=30.0)
        if task_status.status != "completed":
            raise RuntimeError(f"Benchmark batch failed with status: {task_status.status}")
        
        results = task_status.result

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

    def get_performance_metrics(self) -> Dict[str, Any]:
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

    # ============================================================================  
    # Session Management and Tracking
    # ============================================================================

    def get_session_wrapper(self, session_id: str) -> Dict[str, Any]:
        """Get or create session-specific metrics and state.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary containing session state and methods
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMetrics()
            self._session_active_tasks[session_id] = set()
            logger.debug(f"Created new session metrics for {session_id}")

        metrics = self._sessions[session_id]
        
        return {
            "session_id": session_id,
            "metrics": {
                "operations_count": metrics.operations_count,
                "total_time_ms": metrics.total_time_ms,
                "avg_time_ms": metrics.avg_time_ms,
                "min_time_ms": metrics.min_time_ms if metrics.min_time_ms != float("inf") else 0,
                "max_time_ms": metrics.max_time_ms,
                "error_count": metrics.error_count,
                "created_at": metrics.created_at,
                "last_operation_at": metrics.last_operation_at,
            },
            "active_tasks": len(self._session_active_tasks[session_id]),
        }

    def remove_session(self, session_id: str) -> None:
        """Remove a session and its metrics.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            if session_id in self._session_active_tasks:
                del self._session_active_tasks[session_id]
            logger.debug(f"Removed session {session_id}")

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and cancel its tasks.

        Args:
            session_id: Session identifier
        """
        if session_id not in self._sessions:
            logger.debug(f"Session {session_id} not found, nothing to clean up")
            return
            
        # Cancel all active tasks for this session
        if session_id in self._session_active_tasks:
            tasks = self._session_active_tasks[session_id].copy()
            cancelled_count = 0
            error_count = 0
            
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        cancelled_count += 1
                    except Exception as e:
                        error_count += 1
                        logger.warning(
                            f"Error cancelling task in session {session_id}: {e}",
                            exc_info=True
                        )
                        
            if cancelled_count > 0 or error_count > 0:
                logger.info(
                    f"Session {session_id} cleanup: "
                    f"{cancelled_count} tasks cancelled, {error_count} errors"
                )
            
        self.remove_session(session_id)
        logger.info(f"Cleaned up session {session_id}")

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions and their metrics.

        Returns:
            Dictionary of session information
        """
        return {
            session_id: self.get_session_wrapper(session_id)
            for session_id in self._sessions.keys()
        }

    def get_session_count(self) -> int:
        """Get number of active sessions.

        Returns:
            Number of active sessions
        """
        return len(self._sessions)

    def set_default_timeout(self, seconds: float) -> None:
        """Set default timeout for new sessions.

        Args:
            seconds: Default timeout in seconds
        """
        self._default_timeout = seconds

    async def benchmark_all_sessions(self) -> Dict[str, Dict[str, float]]:
        """Benchmark latency for all active sessions.

        Returns:
            Dictionary of session benchmarks
        """
        benchmarks = {}

        for session_id in self._sessions.keys():
            # Run a mini benchmark for each session
            start_time = time.perf_counter()
            await asyncio.sleep(0)  # Minimal operation
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            
            benchmarks[session_id] = {
                "session_latency_ms": latency_ms,
                "operations_count": self._sessions[session_id].operations_count,
                "avg_time_ms": self._sessions[session_id].avg_time_ms,
            }

        return benchmarks

    # ============================================================================
    # Integrated Operations and Context Management
    # ============================================================================

    async def run_with_session_tracking(
        self, 
        session_id: str, 
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = None
    ) -> T:
        """Run a coroutine with session-specific tracking and metrics.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
            timeout: Optional timeout in seconds

        Returns:
            Result of the coroutine execution
        """
        # Ensure session exists
        if session_id not in self._sessions:
            self.get_session_wrapper(session_id)

        metrics = self._sessions[session_id]
        timeout = timeout or self._default_timeout

        start_time = time.perf_counter()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(coro, timeout=timeout)
            
            # Update metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            metrics.operations_count += 1
            metrics.total_time_ms += duration_ms
            metrics.avg_time_ms = metrics.total_time_ms / metrics.operations_count
            metrics.min_time_ms = min(metrics.min_time_ms, duration_ms)
            metrics.max_time_ms = max(metrics.max_time_ms, duration_ms)
            metrics.last_operation_at = end_time
            
            return result
            
        except Exception as e:
            metrics.error_count += 1
            logger.error(f"Session {session_id} operation failed: {e}")
            raise

    @asynccontextmanager
    async def session_context(self, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Context manager for session-scoped operations.

        Args:
            session_id: Session identifier

        Yields:
            Session wrapper information
        """
        session_wrapper = self.get_session_wrapper(session_id)
        try:
            yield session_wrapper
        finally:
            # Could add cleanup logic here if needed
            pass

    async def benchmark_unified_performance(
        self,
        latency_samples: int = 100,
        throughput_tasks: int = 1000,
        session_count: int = 5
    ) -> Dict[str, Any]:
        """Run comprehensive benchmarks for both global and session performance.

        Args:
            latency_samples: Number of samples for latency test
            throughput_tasks: Number of tasks for throughput test  
            session_count: Number of test sessions to create

        Returns:
            Dictionary with comprehensive benchmark results
        """
        # Run global benchmarks
        latency_metrics = await self.benchmark_loop_latency(latency_samples)
        throughput_metrics = await self.benchmark_task_throughput(throughput_tasks)

        # Create test sessions and benchmark them
        test_sessions = [f"test_session_{i}" for i in range(session_count)]
        for session_id in test_sessions:
            # Run a test operation in each session
            await self.run_with_session_tracking(
                session_id, 
                asyncio.sleep(0.001), 
                timeout=1.0
            )

        session_benchmarks = await self.benchmark_all_sessions()

        # Get loop info
        loop_info = self.get_loop_info()

        # Clean up test sessions
        for session_id in test_sessions:
            await self.cleanup_session(session_id)

        return {
            "loop_info": loop_info,
            "global_latency": latency_metrics,
            "global_throughput": throughput_metrics,
            "session_benchmarks": session_benchmarks,
            "unified_manager_version": __version__,
            "timestamp": time.time(),
        }


# ============================================================================
# Global Instance and Factory Function
# ============================================================================

# Global unified loop manager instance with thread-safe initialization
_unified_loop_manager: Optional[UnifiedLoopManager] = None
_manager_lock = threading.Lock()

def get_unified_loop_manager() -> UnifiedLoopManager:
    """Get the global unified loop manager instance (thread-safe singleton).

    Returns:
        Global UnifiedLoopManager instance
    """
    global _unified_loop_manager
    
    if _unified_loop_manager is None:
        with _manager_lock:
            # Double-check pattern for thread safety
            if _unified_loop_manager is None:
                _unified_loop_manager = UnifiedLoopManager()
                
    return _unified_loop_manager

