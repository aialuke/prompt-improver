"""
Progress reporting utilities for CLI operations.
Unified progress reporting extracted from existing patterns in console.py, batch processors, training workflows, and TUI widgets.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

@dataclass
class ProgressMetrics:
    """Metrics for progress tracking."""
    items_processed: int = 0
    total_items: Optional[int] = None
    processing_time_ms: float = 0.0
    throughput_items_per_sec: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ProgressReporter:
    """
    Unified progress reporting extracted from existing patterns.
    
    Combines patterns from:
    - console.py: create_progress_bar function
    - batch processors: BatchMetrics and processing tracking
    - training workflows: experiment tracking and metrics
    - TUI widgets: performance metrics and real-time updates
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._active_progress: Optional[Progress] = None
        self._active_tasks: Dict[str, int] = {}
        self._metrics: Dict[str, ProgressMetrics] = {}
        self._start_times: Dict[str, float] = {}

    def create_progress_bar(
        self,
        description: str = "Processing",
        show_percentage: bool = True,
        show_time: bool = True,
        show_speed: bool = False,
    ) -> Progress:
        """
        Create a standardized progress bar for CLI operations.
        
        Extracted from existing create_progress_bar function in console.py.
        
        Args:
            description: Progress description
            show_percentage: Show percentage completion
            show_time: Show elapsed and remaining time
            show_speed: Show processing speed
            
        Returns:
            Configured Progress instance
        """
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        ]
        
        if show_percentage:
            columns.append(TaskProgressColumn())
        
        if show_time:
            columns.extend([
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ])
        
        if show_speed:
            columns.append(TextColumn("[progress.percentage]{task.speed} items/s"))

        return Progress(
            *columns,
            console=self.console,
            expand=True,
        )

    @contextmanager
    def progress_context(
        self,
        description: str = "Processing",
        total: Optional[int] = None,
        **kwargs
    ):
        """
        Context manager for progress tracking.
        
        Args:
            description: Task description
            total: Total number of items to process
            **kwargs: Additional arguments for create_progress_bar
        """
        progress = self.create_progress_bar(description, **kwargs)
        
        with progress:
            task_id = progress.add_task(description, total=total)
            self._active_progress = progress
            self._active_tasks[description] = task_id
            self._start_times[description] = time.time()
            self._metrics[description] = ProgressMetrics(total_items=total)
            
            try:
                yield self._create_progress_updater(description, task_id)
            finally:
                self._finalize_metrics(description)
                if description in self._active_tasks:
                    del self._active_tasks[description]
                if description in self._start_times:
                    del self._start_times[description]
                self._active_progress = None

    def _create_progress_updater(self, description: str, task_id: int):
        """Create progress updater function for context manager."""
        def update_progress(
            advance: int = 1,
            completed: Optional[int] = None,
            error: bool = False,
            retry: bool = False,
            **kwargs
        ):
            """Update progress for the current task."""
            if self._active_progress is None:
                return
            
            # Update metrics
            metrics = self._metrics[description]
            if advance:
                metrics.items_processed += advance
            if completed is not None:
                metrics.items_processed = completed
            if error:
                metrics.error_count += 1
            if retry:
                metrics.retry_count += 1
            
            # Calculate throughput
            elapsed = time.time() - self._start_times[description]
            if elapsed > 0:
                metrics.throughput_items_per_sec = metrics.items_processed / elapsed
                metrics.processing_time_ms = elapsed * 1000
            
            # Update progress bar
            if completed is not None:
                self._active_progress.update(task_id, completed=completed, **kwargs)
            else:
                self._active_progress.advance(task_id, advance, **kwargs)
        
        return update_progress

    def _finalize_metrics(self, description: str):
        """Finalize metrics for completed task."""
        if description in self._metrics:
            metrics = self._metrics[description]
            metrics.timestamp = datetime.now(timezone.utc)
            
            # Calculate final throughput
            if description in self._start_times:
                elapsed = time.time() - self._start_times[description]
                if elapsed > 0:
                    metrics.throughput_items_per_sec = metrics.items_processed / elapsed
                    metrics.processing_time_ms = elapsed * 1000

    def get_metrics(self, description: str) -> Optional[ProgressMetrics]:
        """Get metrics for a specific task."""
        return self._metrics.get(description)

    def get_all_metrics(self) -> Dict[str, ProgressMetrics]:
        """Get all tracked metrics."""
        return self._metrics.copy()

    async def track_async_operation(
        self,
        operation_name: str,
        async_func,
        *args,
        total_items: Optional[int] = None,
        **kwargs
    ):
        """
        Track an async operation with progress reporting.
        
        Args:
            operation_name: Name of the operation
            async_func: Async function to execute
            *args: Arguments for async_func
            total_items: Total number of items to process
            **kwargs: Keyword arguments for async_func
            
        Returns:
            Result of async_func
        """
        with self.progress_context(operation_name, total=total_items) as update_progress:
            start_time = time.time()
            
            try:
                # Execute async function
                result = await async_func(*args, **kwargs)
                
                # Mark as completed
                if total_items:
                    update_progress(completed=total_items)
                else:
                    update_progress(advance=1)
                
                return result
                
            except Exception as e:
                update_progress(error=True)
                raise
            
            finally:
                elapsed = time.time() - start_time
                self.console.print(
                    f"✅ {operation_name} completed in {elapsed:.2f}s",
                    style="green"
                )

    def report_batch_metrics(self, batch_id: str, metrics: ProgressMetrics):
        """
        Report batch processing metrics.
        
        Extracted from batch processor patterns.
        
        Args:
            batch_id: Batch identifier
            metrics: Batch processing metrics
        """
        self.console.print(f"\n📊 Batch {batch_id} Metrics:", style="bold blue")
        self.console.print(f"  Items Processed: {metrics.items_processed}")
        if metrics.total_items:
            completion = (metrics.items_processed / metrics.total_items) * 100
            self.console.print(f"  Completion: {completion:.1f}%")
        
        self.console.print(f"  Processing Time: {metrics.processing_time_ms:.1f}ms")
        self.console.print(f"  Throughput: {metrics.throughput_items_per_sec:.1f} items/sec")
        
        if metrics.error_count > 0:
            self.console.print(f"  Errors: {metrics.error_count}", style="red")
        if metrics.retry_count > 0:
            self.console.print(f"  Retries: {metrics.retry_count}", style="yellow")

    def simple_spinner(self, description: str = "Processing..."):
        """
        Simple spinner for indeterminate progress.
        
        Returns:
            Context manager for spinner
        """
        return self.console.status(description, spinner="dots")
