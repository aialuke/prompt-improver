"""Enhanced Workflow Completion Manager for APES CLI
Implements 2025 best practices for workflow monitoring and graceful shutdown.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator


class WorkflowStopMode(Enum):
    """Workflow stopping modes."""

    GRACEFUL = "graceful"
    FORCE = "force"
    TIMEOUT = "timeout"


@dataclass
class WorkflowMonitorConfig:
    """Configuration for workflow monitoring and completion waiting."""

    poll_interval: float = 2.0
    progress_update_interval: float = 5.0
    exponential_backoff: bool = True
    max_poll_interval: float = 10.0
    backoff_multiplier: float = 1.2
    timeout_warning_threshold: float = 0.8


@dataclass
class WorkflowCompletionResult:
    """Result of workflow completion waiting."""

    workflow_id: str
    status: str
    completed: bool
    duration_seconds: float
    timeout_reached: bool
    error: str | None = None
    final_state: str | None = None
    progress_data: dict[str, Any] | None = None


class WorkflowService:
    """Workflow service implementing Clean Architecture and 2025 best practices for:
    - Configurable timeout management with exponential backoff
    - Rich progress monitoring with real-time updates
    - Graceful vs force shutdown modes
    - Comprehensive workflow status tracking
    - Integration with signal handling for interruption.
    """

    def __init__(
        self, cli_orchestrator: CLIOrchestrator, console: Console | None = None
    ) -> None:
        self.cli_orchestrator = cli_orchestrator
        self.console = console or Console()
        self.logger = logging.getLogger(__name__)
        self.active_workflows: set[str] = set()
        self.workflow_start_times: dict[str, datetime] = {}
        self.workflow_configs: dict[str, WorkflowMonitorConfig] = {}
        self.monitoring_tasks: dict[str, asyncio.Task] = {}
        self.shutdown_requested = False

    async def wait_for_workflow_completion(
        self,
        workflow_id: str,
        timeout: int = 3600,
        config: WorkflowMonitorConfig | None = None,
        show_progress: bool = True,
    ) -> WorkflowCompletionResult:
        """Wait for workflow completion with enhanced monitoring and timeout management.

        Args:
            workflow_id: Workflow identifier to monitor
            timeout: Maximum time to wait in seconds
            config: Optional monitoring configuration
            show_progress: Whether to show rich progress indicators

        Returns:
            WorkflowCompletionResult with completion details
        """
        if config is None:
            config = WorkflowMonitorConfig()
        start_time = datetime.now(UTC)
        self.workflow_start_times[workflow_id] = start_time
        self.workflow_configs[workflow_id] = config
        self.active_workflows.add(workflow_id)
        self.logger.info(
            f"Starting workflow completion monitoring for {workflow_id} (timeout: {timeout}s)"
        )
        try:
            if show_progress:
                return await self._wait_with_progress_display(
                    workflow_id, timeout, config
                )
            return await self._wait_with_polling(workflow_id, timeout, config)
        except asyncio.CancelledError:
            self.logger.info(f"Workflow monitoring cancelled for {workflow_id}")
            return WorkflowCompletionResult(
                workflow_id=workflow_id,
                status="cancelled",
                completed=False,
                duration_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                timeout_reached=False,
                error="Monitoring cancelled",
            )
        finally:
            self.active_workflows.discard(workflow_id)
            self.workflow_start_times.pop(workflow_id, None)
            self.workflow_configs.pop(workflow_id, None)
            if workflow_id in self.monitoring_tasks:
                task = self.monitoring_tasks.pop(workflow_id)
                if not task.done():
                    task.cancel()

    async def _wait_with_progress_display(
        self, workflow_id: str, timeout: int, config: WorkflowMonitorConfig
    ) -> WorkflowCompletionResult:
        """Wait for workflow completion with rich progress display."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            progress_task = progress.add_task(
                f"Monitoring workflow {workflow_id}", total=timeout
            )
            start_time = datetime.now(UTC)
            current_poll_interval = config.poll_interval
            while True:
                elapsed = (datetime.now(UTC) - start_time).total_seconds()
                if elapsed >= timeout:
                    progress.update(progress_task, completed=timeout)
                    return WorkflowCompletionResult(
                        workflow_id=workflow_id,
                        status="timeout",
                        completed=False,
                        duration_seconds=elapsed,
                        timeout_reached=True,
                        error=f"Workflow did not complete within {timeout}s",
                    )
                progress.update(progress_task, completed=elapsed)
                try:
                    status_result = await self.cli_orchestrator.get_workflow_status(
                        workflow_id
                    )
                    if status_result.get("completed", False):
                        progress.update(progress_task, completed=timeout)
                        return WorkflowCompletionResult(
                            workflow_id=workflow_id,
                            status="completed",
                            completed=True,
                            duration_seconds=elapsed,
                            timeout_reached=False,
                            final_state=status_result.get("state"),
                            progress_data=status_result,
                        )
                    current_status = status_result.get("state", "unknown")
                    progress.update(
                        progress_task,
                        description=f"Monitoring workflow {workflow_id} - {current_status}",
                    )
                except Exception as e:
                    self.logger.warning(f"Error checking workflow status: {e}")
                if config.exponential_backoff:
                    current_poll_interval = min(
                        current_poll_interval * config.backoff_multiplier,
                        config.max_poll_interval,
                    )
                await asyncio.sleep(current_poll_interval)

    async def _wait_with_polling(
        self, workflow_id: str, timeout: int, config: WorkflowMonitorConfig
    ) -> WorkflowCompletionResult:
        """Wait for workflow completion with simple polling (no progress display)."""
        start_time = datetime.now(UTC)
        current_poll_interval = config.poll_interval
        last_status_log = 0
        while True:
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed >= timeout:
                return WorkflowCompletionResult(
                    workflow_id=workflow_id,
                    status="timeout",
                    completed=False,
                    duration_seconds=elapsed,
                    timeout_reached=True,
                    error=f"Workflow did not complete within {timeout}s",
                )
            if elapsed - last_status_log >= config.progress_update_interval:
                self.logger.info(
                    f"Workflow {workflow_id} still running ({elapsed:.1f}s elapsed)"
                )
                last_status_log = elapsed
            try:
                status_result = await self.cli_orchestrator.get_workflow_status(
                    workflow_id
                )
                if status_result.get("completed", False):
                    return WorkflowCompletionResult(
                        workflow_id=workflow_id,
                        status="completed",
                        completed=True,
                        duration_seconds=elapsed,
                        timeout_reached=False,
                        final_state=status_result.get("state"),
                        progress_data=status_result,
                    )
            except Exception as e:
                self.logger.warning(f"Error checking workflow status: {e}")
            if config.exponential_backoff:
                current_poll_interval = min(
                    current_poll_interval * config.backoff_multiplier,
                    config.max_poll_interval,
                )
            await asyncio.sleep(current_poll_interval)
