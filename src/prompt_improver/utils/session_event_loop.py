"""Session-scoped event loop wrapper for MCP session optimization.

This module provides a session-aware wrapper around asyncio operations,
offering per-session performance monitoring and optimization for MCP interactions.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Set, TypeVar, Union

from prompt_improver.utils.event_loop_manager import get_event_loop_manager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SessionEventLoopWrapper:
    """Session-scoped wrapper for asyncio operations with performance monitoring."""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize session event loop wrapper.
        
        Args:
            session_id: Optional session identifier for tracking
        """
        self.session_id = session_id or f"session_{id(self)}"
        self._event_loop_manager = get_event_loop_manager()
        self._session_metrics: Dict[str, Any] = {
            "operations_count": 0,
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0,
            "min_time_ms": float("inf"),
            "max_time_ms": 0.0,
            "error_count": 0,
            "created_at": time.time(),
            "last_operation_at": None,
        }
        self._active_tasks: Set[asyncio.Task] = set()
        self._timeout_seconds = 30.0  # Default timeout for operations
        
    def set_timeout(self, seconds: float) -> None:
        """Set default timeout for session operations.
        
        Args:
            seconds: Timeout in seconds
        """
        self._timeout_seconds = seconds
    
    async def run_with_timeout(
        self, 
        coro, 
        timeout: Optional[float] = None,
        operation_name: str = "operation"
    ) -> T:
        """Run a coroutine with timeout and performance tracking.
        
        Args:
            coro: Coroutine to execute
            timeout: Optional timeout override
            operation_name: Name of the operation for logging
            
        Returns:
            Result of the coroutine
            
        Raises:
            asyncio.TimeoutError: If operation times out
        """
        timeout_value = timeout or self._timeout_seconds
        start_time = time.perf_counter()
        
        try:
            # Run with timeout
            result = await asyncio.wait_for(coro, timeout=timeout_value)
            
            # Track success metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(duration_ms, success=True)
            
            logger.debug(
                f"Session {self.session_id} {operation_name} completed in {duration_ms:.2f}ms"
            )
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(duration_ms, success=False)
            
            logger.warning(
                f"Session {self.session_id} {operation_name} timed out after {timeout_value}s"
            )
            raise
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(duration_ms, success=False)
            
            logger.error(
                f"Session {self.session_id} {operation_name} failed after {duration_ms:.2f}ms: {e}"
            )
            raise
    
    def create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Create a task with session tracking.
        
        Args:
            coro: Coroutine to create task from
            name: Optional task name
            
        Returns:
            Created task
        """
        task = asyncio.create_task(coro, name=name)
        self._active_tasks.add(task)
        
        # Add callback to remove from active tasks when done
        task.add_done_callback(self._active_tasks.discard)
        
        logger.debug(f"Session {self.session_id} created task {name or task.get_name()}")
        return task
    
    async def gather_with_timeout(
        self, 
        *coros, 
        timeout: Optional[float] = None,
        return_exceptions: bool = False
    ) -> list:
        """Gather coroutines with timeout and session tracking.
        
        Args:
            *coros: Coroutines to gather
            timeout: Optional timeout override
            return_exceptions: Whether to return exceptions instead of raising
            
        Returns:
            List of results
        """
        timeout_value = timeout or self._timeout_seconds
        
        return await self.run_with_timeout(
            asyncio.gather(*coros, return_exceptions=return_exceptions),
            timeout=timeout_value,
            operation_name=f"gather({len(coros)} tasks)"
        )
    
    @asynccontextmanager
    async def performance_context(
        self, 
        operation_name: str = "context_operation"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Context manager for performance monitoring.
        
        Args:
            operation_name: Name of the operation
            
        Yields:
            Performance tracking dictionary
        """
        start_time = time.perf_counter()
        context_metrics = {
            "operation_name": operation_name,
            "start_time": start_time,
            "session_id": self.session_id,
        }
        
        try:
            yield context_metrics
            
            # Success metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            context_metrics["duration_ms"] = duration_ms
            context_metrics["success"] = True
            self._update_metrics(duration_ms, success=True)
            
            logger.debug(
                f"Session {self.session_id} {operation_name} context completed in {duration_ms:.2f}ms"
            )
            
        except Exception as e:
            # Error metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            context_metrics["duration_ms"] = duration_ms
            context_metrics["success"] = False
            context_metrics["error"] = str(e)
            self._update_metrics(duration_ms, success=False)
            
            logger.error(
                f"Session {self.session_id} {operation_name} context failed after {duration_ms:.2f}ms: {e}"
            )
            raise
    
    async def sleep_with_monitoring(self, seconds: float) -> None:
        """Sleep with session monitoring.
        
        Args:
            seconds: Seconds to sleep
        """
        await self.run_with_timeout(
            asyncio.sleep(seconds),
            timeout=seconds + 1.0,  # Allow some buffer
            operation_name=f"sleep({seconds}s)"
        )
    
    def _update_metrics(self, duration_ms: float, success: bool) -> None:
        """Update session performance metrics.
        
        Args:
            duration_ms: Operation duration in milliseconds
            success: Whether the operation succeeded
        """
        self._session_metrics["operations_count"] += 1
        self._session_metrics["total_time_ms"] += duration_ms
        self._session_metrics["last_operation_at"] = time.time()
        
        if not success:
            self._session_metrics["error_count"] += 1
        
        # Update timing statistics
        if duration_ms < self._session_metrics["min_time_ms"]:
            self._session_metrics["min_time_ms"] = duration_ms
        if duration_ms > self._session_metrics["max_time_ms"]:
            self._session_metrics["max_time_ms"] = duration_ms
        
        # Update average
        self._session_metrics["avg_time_ms"] = (
            self._session_metrics["total_time_ms"] / self._session_metrics["operations_count"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get session performance metrics.
        
        Returns:
            Dictionary containing session metrics
        """
        return {
            **self._session_metrics,
            "active_tasks": len(self._active_tasks),
            "timeout_seconds": self._timeout_seconds,
            "session_age_seconds": time.time() - self._session_metrics["created_at"],
        }
    
    def get_active_tasks(self) -> Set[asyncio.Task]:
        """Get active tasks for this session.
        
        Returns:
            Set of active tasks
        """
        return self._active_tasks.copy()
    
    async def cancel_all_tasks(self) -> None:
        """Cancel all active tasks for this session."""
        if not self._active_tasks:
            return
        
        logger.info(f"Cancelling {len(self._active_tasks)} active tasks for session {self.session_id}")
        
        # Cancel all tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        
        self._active_tasks.clear()
    
    async def benchmark_session_latency(self, samples: int = 50) -> Dict[str, float]:
        """Benchmark session-specific latency.
        
        Args:
            samples: Number of samples to collect
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for i in range(samples):
            start_time = time.perf_counter()
            
            # Simple async operation
            await asyncio.sleep(0.001)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        if not latencies:
            return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "samples": 0}
        
        return {
            "avg_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "samples": len(latencies),
            "session_id": self.session_id,
        }
    
    def __str__(self) -> str:
        """String representation of the session wrapper."""
        return f"SessionEventLoopWrapper(session_id={self.session_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"SessionEventLoopWrapper("
            f"session_id={self.session_id}, "
            f"operations={self._session_metrics['operations_count']}, "
            f"active_tasks={len(self._active_tasks)}, "
            f"avg_time={self._session_metrics['avg_time_ms']:.2f}ms"
            f")"
        )


class SessionEventLoopManager:
    """Manager for session-scoped event loop wrappers."""
    
    def __init__(self):
        self._sessions: Dict[str, SessionEventLoopWrapper] = {}
        self._default_timeout = 30.0
    
    def get_session_wrapper(self, session_id: str) -> SessionEventLoopWrapper:
        """Get or create a session wrapper.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session wrapper instance
        """
        if session_id not in self._sessions:
            wrapper = SessionEventLoopWrapper(session_id)
            wrapper.set_timeout(self._default_timeout)
            self._sessions[session_id] = wrapper
            logger.debug(f"Created new session wrapper for {session_id}")
        
        return self._sessions[session_id]
    
    def remove_session(self, session_id: str) -> None:
        """Remove a session wrapper.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Removed session wrapper for {session_id}")
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and cancel its tasks.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            wrapper = self._sessions[session_id]
            await wrapper.cancel_all_tasks()
            del self._sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")
    
    def get_all_sessions(self) -> Dict[str, SessionEventLoopWrapper]:
        """Get all active sessions.
        
        Returns:
            Dictionary of session wrappers
        """
        return self._sessions.copy()
    
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
        
        for session_id, wrapper in self._sessions.items():
            benchmarks[session_id] = await wrapper.benchmark_session_latency()
        
        return benchmarks


# Global session manager
_session_manager = SessionEventLoopManager()


def get_session_manager() -> SessionEventLoopManager:
    """Get the global session manager.
    
    Returns:
        Global SessionEventLoopManager instance
    """
    return _session_manager


def get_session_wrapper(session_id: str) -> SessionEventLoopWrapper:
    """Get or create a session wrapper.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session wrapper instance
    """
    return _session_manager.get_session_wrapper(session_id)
