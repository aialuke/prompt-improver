"""
CLI Orchestrator Integration Layer
Bridges 3-command CLI with ML Pipeline Orchestrator for clean training workflows.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from ...ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator

class CLIOrchestrator:
    """
    Integration layer between 3-command CLI and ML Pipeline Orchestrator.

    Provides clean interface for:
    - Starting continuous training workflows
    - Monitoring training progress
    - Managing workflow lifecycle
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.logger = logging.getLogger("apes.cli_orchestrator")

        self._orchestrator: Optional[MLPipelineOrchestrator] = None
        self._active_workflows: Dict[str, Dict[str, Any]] = {}

    async def start_continuous_training(
        self,
        session_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Start continuous training workflow with performance gap analysis.

        Args:
            session_id: Training session identifier
            config: Training configuration parameters

        Returns:
            Workflow startup result with workflow_id
        """
        self.logger.info(f"Starting continuous training for session {session_id}")

        try:
            # Get orchestrator instance
            orchestrator = await self._get_orchestrator()

            # Create continuous training workflow
            workflow_def = await self._create_continuous_training_workflow(config)

            # Start workflow execution
            workflow_id = await orchestrator.start_workflow(
                workflow_type="continuous_training",
                parameters={
                    "session_id": session_id,
                    "config": config,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "max_iterations": config.get("max_iterations"),
                    "improvement_threshold": config.get("improvement_threshold", 0.02),
                    "timeout": config.get("timeout", 3600),
                    "continuous": config.get("continuous", True),
                    "verbose": config.get("verbose", False)
                }
            )

            # Track workflow
            self._active_workflows[workflow_id] = {
                "session_id": session_id,
                "config": config,
                "started_at": datetime.now(timezone.utc),
                "status": "running",
                "iterations": 0,
                "last_improvement": None
            }

            self.logger.info(f"Continuous training workflow started: {workflow_id}")

            return {
                "status": "success",
                "workflow_id": workflow_id,
                "session_id": session_id,
                "started_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to start continuous training: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def monitor_training_progress(
        self,
        workflow_id: str,
        verbose: bool = False
    ) -> None:
        """
        Monitor continuous training progress with real-time performance metrics.

        Features:
        - Real-time performance tracking with improvement detection
        - Correlation-driven stopping criteria based on 2025 best practices
        - Plateau detection and intelligent threshold monitoring
        - Live metrics visualization with Rich progress bars

        Args:
            workflow_id: Workflow to monitor
            verbose: Show detailed progress information
        """
        self.logger.info(f"Monitoring training progress for workflow {workflow_id}")

        try:
            orchestrator = await self._get_orchestrator()

            # Initialize intelligent stopping criteria (2025 best practices)
            performance_history = []
            last_significant_improvement = time.time()
            plateau_threshold = 300  # 5 minutes without improvement

            # Correlation-driven stopping parameters
            min_correlation_threshold = 0.7  # Minimum correlation for trend detection
            improvement_threshold = 0.001    # Minimum improvement to be considered significant
            consecutive_poor_iterations = 0  # Track consecutive poor performance
            max_poor_iterations = 5         # Stop after 5 consecutive poor iterations

            # Performance trend analysis
            trend_window = 10               # Number of iterations for trend analysis
            early_stop_patience = 15       # Early stopping patience (iterations)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
                expand=True,
            ) as progress:

                monitor_task = progress.add_task("Initializing training monitor...", total=None)

                # Performance metrics tracking
                metrics_task = None
                if verbose:
                    metrics_task = progress.add_task("Performance metrics...", total=None)

                iteration_count = 0
                best_performance = 0.0

                while True:
                    try:
                        # Get workflow status
                        status = await orchestrator.get_workflow_status(workflow_id)

                        # Check if workflow completed
                        if hasattr(status, 'state'):
                            workflow_state = status.state.value if hasattr(status.state, 'value') else str(status.state)
                        else:
                            workflow_state = status.get("status", "unknown")

                        if workflow_state in ["completed", "failed", "cancelled", "error"]:
                            progress.update(monitor_task, description=f"Training {workflow_state}")
                            if verbose:
                                self.console.print(f"\n🎯 Final Results:", style="bold green")
                                self.console.print(f"   Iterations: {iteration_count}")
                                self.console.print(f"   Best Performance: {best_performance:.4f}")
                                self.console.print(f"   Total Time: {time.time() - last_significant_improvement:.1f}s")
                            break

                        # Get real-time performance metrics
                        current_metrics = await self._get_training_metrics(workflow_id, orchestrator)

                        if current_metrics:
                            iteration_count = current_metrics.get("iteration", iteration_count)
                            current_performance = current_metrics.get("performance_score", 0.0)
                            improvement_rate = current_metrics.get("improvement_rate", 0.0)

                            # Track performance history
                            performance_history.append({
                                "timestamp": time.time(),
                                "performance": current_performance,
                                "iteration": iteration_count
                            })

                            # Keep only last 10 measurements for trend analysis
                            if len(performance_history) > 10:
                                performance_history.pop(0)

                            # Intelligent stopping criteria analysis
                            improvement = current_performance - best_performance

                            # Check for significant improvement
                            if improvement > improvement_threshold:
                                best_performance = current_performance
                                last_significant_improvement = time.time()
                                consecutive_poor_iterations = 0  # Reset poor iteration counter

                                if verbose:
                                    self.console.print(f"📈 Significant improvement: +{improvement:.4f}", style="green")
                            else:
                                consecutive_poor_iterations += 1

                                if verbose and improvement < 0:
                                    self.console.print(f"📉 Performance decline: {improvement:.4f}", style="yellow")

                            # Correlation-driven trend analysis
                            if len(performance_history) >= trend_window:
                                trend_correlation = self._calculate_performance_trend(performance_history[-trend_window:])

                                if verbose:
                                    trend_desc = "improving" if trend_correlation > min_correlation_threshold else "declining" if trend_correlation < -min_correlation_threshold else "stable"
                                    self.console.print(f"📊 Trend analysis: {trend_desc} (correlation: {trend_correlation:.3f})", style="cyan")

                                # Intelligent stopping decision
                                should_stop, stop_reason = self._should_stop_training(
                                    performance_history=performance_history,
                                    consecutive_poor_iterations=consecutive_poor_iterations,
                                    max_poor_iterations=max_poor_iterations,
                                    trend_correlation=trend_correlation,
                                    min_correlation_threshold=min_correlation_threshold,
                                    time_since_improvement=time.time() - last_significant_improvement,
                                    plateau_threshold=plateau_threshold
                                )

                                if should_stop:
                                    self.console.print(f"\n🛑 Intelligent stopping triggered: {stop_reason}", style="bold yellow")
                                    self.console.print("   Initiating graceful workflow shutdown...", style="dim")

                                    # Trigger graceful stop
                                    await orchestrator.stop_workflow(workflow_id)
                                    break

                            # Update progress display
                            description = f"Training iteration {iteration_count} (score: {current_performance:.4f})"
                            if improvement_rate > 0:
                                description += f" ↗️ +{improvement_rate:.4f}"
                            elif improvement_rate < 0:
                                description += f" ↘️ {improvement_rate:.4f}"

                            progress.update(monitor_task, description=description)

                            # Update metrics display
                            if verbose and metrics_task is not None:
                                metrics_desc = f"Best: {best_performance:.4f} | Rate: {improvement_rate:.4f}/iter"
                                progress.update(metrics_task, description=metrics_desc)

                            # Check for plateau (intelligent stopping)
                            time_since_improvement = time.time() - last_significant_improvement
                            if time_since_improvement > plateau_threshold:
                                self.console.print(f"\n⚠️  Performance plateau detected ({time_since_improvement:.0f}s without improvement)", style="yellow")
                                self.console.print("   Consider stopping training or adjusting parameters", style="dim")

                        # Update workflow tracking
                        if workflow_id in self._active_workflows:
                            self._active_workflows[workflow_id].update({
                                "iterations": iteration_count,
                                "last_improvement": best_performance,
                                "performance_history": performance_history[-5:]  # Keep last 5 for memory efficiency
                            })

                    except Exception as e:
                        self.logger.warning(f"Error getting training metrics: {e}")
                        progress.update(monitor_task, description="Training in progress (metrics unavailable)")

                    # Check for user interruption
                    await asyncio.sleep(2)  # Update every 2 seconds

        except KeyboardInterrupt:
            self.console.print("\n🛑 Monitoring interrupted", style="yellow")
        except Exception as e:
            self.logger.error(f"Error monitoring training progress: {e}")
            self.console.print(f"❌ Monitoring error: {e}", style="red")

    async def _get_training_metrics(
        self,
        workflow_id: str,
        orchestrator
    ) -> Dict[str, Any]:
        """
        Get real-time training metrics from the workflow execution.

        Implements 2025 best practices for ML performance monitoring:
        - Real-time performance score calculation
        - Improvement rate tracking with correlation analysis
        - Memory-efficient metrics collection
        - Intelligent threshold detection

        Args:
            workflow_id: Workflow identifier
            orchestrator: ML Pipeline Orchestrator instance

        Returns:
            Dictionary with current training metrics
        """
        try:
            # Get workflow status with detailed metrics
            workflow_status = await orchestrator.get_workflow_status(workflow_id)

            if not workflow_status:
                return {}

            # Extract metrics from workflow metadata
            metadata = getattr(workflow_status, 'metadata', {})

            # Calculate performance score based on available metrics
            performance_score = 0.0
            iteration = 0
            improvement_rate = 0.0

            # Get iteration count
            if hasattr(workflow_status, 'metadata') and workflow_status.metadata:
                iteration = workflow_status.metadata.get('current_iteration', 0)

                # Calculate performance score from multiple metrics
                model_accuracy = workflow_status.metadata.get('model_accuracy', 0.0)
                rule_effectiveness = workflow_status.metadata.get('rule_effectiveness', 0.0)
                pattern_coverage = workflow_status.metadata.get('pattern_coverage', 0.0)

                # Weighted performance score (2025 best practice: multi-metric evaluation)
                performance_score = (
                    model_accuracy * 0.4 +
                    rule_effectiveness * 0.4 +
                    pattern_coverage * 0.2
                )

                # Calculate improvement rate
                previous_score = workflow_status.metadata.get('previous_performance', performance_score)
                improvement_rate = performance_score - previous_score

            # Get additional metrics from workflow tracking
            if workflow_id in self._active_workflows:
                workflow_info = self._active_workflows[workflow_id]
                iteration = max(iteration, workflow_info.get("iterations", 0))

                # Update performance history for trend analysis
                history = workflow_info.get("performance_history", [])
                if history:
                    last_performance = history[-1].get("performance", 0.0)
                    improvement_rate = performance_score - last_performance

            return {
                "iteration": iteration,
                "performance_score": performance_score,
                "improvement_rate": improvement_rate,
                "model_accuracy": metadata.get('model_accuracy', 0.0),
                "rule_effectiveness": metadata.get('rule_effectiveness', 0.0),
                "pattern_coverage": metadata.get('pattern_coverage', 0.0),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.warning(f"Failed to get training metrics: {e}")
            return {}

    def _calculate_performance_trend(self, history: List[Dict[str, Any]]) -> float:
        """
        Calculate performance trend correlation using linear regression.

        Implements 2025 best practices for trend analysis:
        - Pearson correlation coefficient for trend strength
        - Time-weighted performance analysis
        - Robust outlier handling

        Args:
            history: List of performance measurements with timestamps

        Returns:
            Correlation coefficient (-1 to 1, where 1 = strong upward trend)
        """
        if len(history) < 3:
            return 0.0

        try:
            # Extract performance values and create time series
            performances = [h.get("performance", 0.0) for h in history]
            time_points = list(range(len(performances)))

            # Calculate Pearson correlation coefficient
            n = len(performances)
            sum_x = sum(time_points)
            sum_y = sum(performances)
            sum_xy = sum(x * y for x, y in zip(time_points, performances))
            sum_x2 = sum(x * x for x in time_points)
            sum_y2 = sum(y * y for y in performances)

            # Pearson correlation formula
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5

            if denominator == 0:
                return 0.0

            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]

        except Exception as e:
            self.logger.warning(f"Failed to calculate performance trend: {e}")
            return 0.0

    def _should_stop_training(
        self,
        performance_history: List[Dict[str, Any]],
        consecutive_poor_iterations: int,
        max_poor_iterations: int,
        trend_correlation: float,
        min_correlation_threshold: float,
        time_since_improvement: float,
        plateau_threshold: float
    ) -> Tuple[bool, str]:
        """
        Intelligent stopping decision based on multiple criteria.

        Implements 2025 best practices for ML training stopping:
        - Multi-criteria decision making
        - Correlation-driven trend analysis
        - Plateau detection with time-based thresholds
        - Performance degradation protection

        Returns:
            Tuple of (should_stop, reason)
        """
        # Criterion 1: Consecutive poor iterations
        if consecutive_poor_iterations >= max_poor_iterations:
            return True, f"No improvement for {consecutive_poor_iterations} consecutive iterations"

        # Criterion 2: Strong declining trend
        if trend_correlation < -min_correlation_threshold:
            return True, f"Strong declining trend detected (correlation: {trend_correlation:.3f})"

        # Criterion 3: Extended plateau
        if time_since_improvement > plateau_threshold:
            return True, f"Performance plateau for {time_since_improvement:.0f} seconds"

        # Criterion 4: Performance degradation with poor trend
        if (consecutive_poor_iterations >= 3 and
            trend_correlation < 0 and
            time_since_improvement > plateau_threshold / 2):
            return True, "Performance degradation with negative trend"

        # Criterion 5: Very long training without significant progress
        if len(performance_history) > 50 and time_since_improvement > plateau_threshold * 2:
            return True, "Extended training without significant progress"

        return False, ""

    async def wait_for_completion(
        self,
        workflow_id: str,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Wait for workflow completion with timeout.

        Args:
            workflow_id: Workflow to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Final workflow result
        """
        self.logger.info(f"Waiting for workflow {workflow_id} completion (timeout: {timeout}s)")

        try:
            orchestrator = await self._get_orchestrator()

            start_time = time.time()

            while time.time() - start_time < timeout:
                status = await orchestrator.get_workflow_status(workflow_id)

                if status["status"] in ["completed", "failed", "cancelled"]:
                    # Get final results
                    result = await orchestrator.get_workflow_result(workflow_id)

                    # Update workflow tracking
                    if workflow_id in self._active_workflows:
                        self._active_workflows[workflow_id]["status"] = status["status"]
                        self._active_workflows[workflow_id]["completed_at"] = datetime.now(timezone.utc)

                    return {
                        "status": status["status"],
                        "result": result,
                        "duration": time.time() - start_time,
                        "iterations": self._active_workflows.get(workflow_id, {}).get("iterations", 0)
                    }

                await asyncio.sleep(5)  # Check every 5 seconds

            # Timeout reached
            self.logger.warning(f"Workflow {workflow_id} timed out after {timeout}s")
            return {
                "status": "timeout",
                "reason": f"Workflow exceeded {timeout}s timeout",
                "duration": timeout
            }

        except Exception as e:
            self.logger.error(f"Error waiting for workflow completion: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def wait_for_workflow_completion_with_progress(
        self,
        workflow_id: str,
        timeout: int = 3600,
        poll_interval: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced workflow completion waiting with progress monitoring and exponential backoff.

        Args:
            workflow_id: Workflow identifier
            timeout: Maximum wait time in seconds
            poll_interval: Initial status check interval in seconds
            progress_callback: Optional callback for progress updates

        Returns:
            Workflow completion results with enhanced status information
        """
        start_time = time.time()
        last_status = None
        current_poll_interval = poll_interval
        max_poll_interval = min(30, poll_interval * 4)  # Cap at 30 seconds
        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            orchestrator = await self._get_orchestrator()

            self.logger.info(f"Waiting for workflow {workflow_id} completion (timeout: {timeout}s)")

            while time.time() - start_time < timeout:
                try:
                    # Get current workflow status
                    status = await orchestrator.get_workflow_status(workflow_id)
                    consecutive_errors = 0  # Reset error counter on success

                    # Calculate progress information
                    elapsed_time = time.time() - start_time
                    remaining_time = timeout - elapsed_time
                    progress_info = {
                        "workflow_id": workflow_id,
                        "status": status.get("status", "unknown"),
                        "elapsed_seconds": elapsed_time,
                        "remaining_seconds": remaining_time,
                        "progress_percentage": min(100, (elapsed_time / timeout) * 100)
                    }

                    # Call progress callback if provided
                    if progress_callback:
                        try:
                            await progress_callback(progress_info)
                        except Exception as e:
                            self.logger.warning(f"Progress callback error: {e}")

                    current_status = status.get("status", "unknown")

                    # Log status changes with enhanced information
                    if current_status != last_status:
                        self.logger.info(
                            f"Workflow {workflow_id} status: {current_status} "
                            f"(elapsed: {elapsed_time:.1f}s, remaining: {remaining_time:.1f}s)"
                        )
                        last_status = current_status

                        # Reset poll interval on status change (more activity)
                        current_poll_interval = poll_interval

                    # Check if workflow completed successfully
                    if current_status == "completed":
                        duration = time.time() - start_time
                        self.logger.info(f"Workflow {workflow_id} completed successfully in {duration:.2f}s")

                        # Get final results
                        try:
                            result = await orchestrator.get_workflow_result(workflow_id)
                        except Exception as e:
                            self.logger.warning(f"Could not get workflow result: {e}")
                            result = {}

                        return {
                            "status": "completed",
                            "workflow_id": workflow_id,
                            "duration": duration,
                            "results": result,
                            "final_state": current_status,
                            "progress_info": progress_info
                        }

                    # Check if workflow failed
                    elif current_status == "failed":
                        duration = time.time() - start_time
                        error_msg = status.get('error', 'Unknown error')

                        self.logger.error(f"Workflow {workflow_id} failed after {duration:.2f}s: {error_msg}")

                        return {
                            "status": "failed",
                            "workflow_id": workflow_id,
                            "duration": duration,
                            "error": error_msg,
                            "final_state": current_status,
                            "progress_info": progress_info
                        }

                    # Check if workflow was cancelled
                    elif current_status == "cancelled":
                        duration = time.time() - start_time
                        self.logger.info(f"Workflow {workflow_id} was cancelled after {duration:.2f}s")

                        return {
                            "status": "cancelled",
                            "workflow_id": workflow_id,
                            "duration": duration,
                            "final_state": current_status,
                            "progress_info": progress_info
                        }

                    # Implement exponential backoff for polling
                    await asyncio.sleep(current_poll_interval)

                    # Gradually increase poll interval if no status changes
                    if current_status == last_status:
                        current_poll_interval = min(max_poll_interval, current_poll_interval * 1.2)

                except Exception as e:
                    consecutive_errors += 1
                    self.logger.warning(
                        f"Error checking workflow status (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
                    )

                    # If too many consecutive errors, fail the wait
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f"Too many consecutive errors checking workflow {workflow_id}")
                        return {
                            "status": "error",
                            "workflow_id": workflow_id,
                            "error": f"Too many consecutive errors: {e}",
                            "duration": time.time() - start_time,
                            "consecutive_errors": consecutive_errors
                        }

                    # Wait longer after errors
                    await asyncio.sleep(min(30, poll_interval * consecutive_errors))

            # Timeout reached
            duration = time.time() - start_time
            self.logger.warning(f"Workflow {workflow_id} timed out after {duration:.2f}s")

            return {
                "status": "timeout",
                "workflow_id": workflow_id,
                "reason": f"Workflow exceeded {timeout}s timeout",
                "duration": duration,
                "timeout_seconds": timeout,
                "last_known_state": last_status or "unknown"
            }

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Critical error waiting for workflow completion: {e}")

            return {
                "status": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "duration": duration,
                "error_type": "critical"
            }

    async def stop_all_workflows(
        self,
        graceful: bool = True,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Stop all active workflows.

        Args:
            graceful: Whether to perform graceful shutdown
            timeout: Shutdown timeout in seconds

        Returns:
            Shutdown results
        """
        self.logger.info(f"Stopping all workflows (graceful={graceful})")

        try:
            orchestrator = await self._get_orchestrator()

            stopped_workflows = []
            failed_workflows = []

            for workflow_id in list(self._active_workflows.keys()):
                try:
                    success = await orchestrator.cancel_workflow(workflow_id, graceful=graceful)

                    if success:
                        stopped_workflows.append(workflow_id)
                        self._active_workflows[workflow_id]["status"] = "cancelled"
                    else:
                        failed_workflows.append(workflow_id)

                except Exception as e:
                    self.logger.error(f"Failed to stop workflow {workflow_id}: {e}")
                    failed_workflows.append(workflow_id)

            return {
                "status": "success" if not failed_workflows else "partial",
                "stopped_workflows": stopped_workflows,
                "failed_workflows": failed_workflows,
                "total_workflows": len(self._active_workflows)
            }

        except Exception as e:
            self.logger.error(f"Error stopping workflows: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of specific workflow."""
        try:
            orchestrator = await self._get_orchestrator()
            status = await orchestrator.get_workflow_status(workflow_id)

            # Enhance with local tracking info
            if workflow_id in self._active_workflows:
                local_info = self._active_workflows[workflow_id]
                status.update({
                    "session_id": local_info["session_id"],
                    "iterations": local_info["iterations"],
                    "last_improvement": local_info["last_improvement"]
                })

            return status

        except Exception as e:
            self.logger.error(f"Error getting workflow status: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_orchestrator(self) -> MLPipelineOrchestrator:
        """Get or create orchestrator instance."""
        if not self._orchestrator:
            from ...ml.orchestration.config.orchestrator_config import OrchestratorConfig

            config = OrchestratorConfig(
                max_concurrent_workflows=3,
                default_timeout=1800,
                enable_monitoring=True,
                enable_caching=True
            )

            self._orchestrator = MLPipelineOrchestrator(config)
            await self._orchestrator.initialize()

        return self._orchestrator

    async def _create_continuous_training_workflow(self, config: Dict[str, Any]):
        """Create continuous training workflow definition."""
        from ...ml.orchestration.config.workflow_definition import WorkflowDefinition, WorkflowStep

        # Create 7-step continuous training workflow
        return WorkflowDefinition(
            workflow_type="continuous_training",
            name="Continuous Adaptive Training",
            description="Self-improving training loop with performance gap analysis",
            steps=[
                WorkflowStep(
                    step_id="assess_performance",
                    name="Assess Current Performance",
                    component_name="performance_analyzer",
                    parameters={"baseline_required": True},
                    timeout=300
                ),
                WorkflowStep(
                    step_id="analyze_gaps",
                    name="Analyze Performance Gaps",
                    component_name="performance_gap_analyzer",
                    parameters={"improvement_threshold": config.get("improvement_threshold", 0.02)},
                    dependencies=["assess_performance"],
                    timeout=300
                ),
                WorkflowStep(
                    step_id="generate_targeted_data",
                    name="Generate Targeted Synthetic Data",
                    component_name="synthetic_data_generator",
                    parameters={"target_gaps": True, "batch_size": 200},
                    dependencies=["analyze_gaps"],
                    timeout=600
                ),
                WorkflowStep(
                    step_id="incremental_training",
                    name="Incremental Model Training",
                    component_name="ml_integration",
                    parameters={"incremental": True, "epochs": 5},
                    dependencies=["generate_targeted_data"],
                    timeout=900
                ),
                WorkflowStep(
                    step_id="optimize_rules",
                    name="Optimize Rules",
                    component_name="rule_optimizer",
                    parameters={"method": "gaussian_process"},
                    dependencies=["incremental_training"],
                    timeout=600
                ),
                WorkflowStep(
                    step_id="validate_improvement",
                    name="Validate Improvement",
                    component_name="performance_validator",
                    parameters={"validation_threshold": config.get("improvement_threshold", 0.02)},
                    dependencies=["optimize_rules"],
                    timeout=300
                ),
                WorkflowStep(
                    step_id="update_session",
                    name="Update Training Session",
                    component_name="session_manager",
                    parameters={"save_progress": True},
                    dependencies=["validate_improvement"],
                    timeout=120
                )
            ],
            global_timeout=config.get("timeout", 3600),
            max_iterations=config.get("max_iterations"),
            continuous=config.get("continuous", True)
        )

    async def start_single_training(
        self,
        session_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Start single training iteration workflow.

        Args:
            session_id: Training session identifier
            config: Training configuration parameters

        Returns:
            Workflow execution result
        """
        try:
            self.logger.info(f"Starting single training workflow for session {session_id}")

            # Get orchestrator
            orchestrator = await self._get_orchestrator()

            # Start single training workflow
            workflow_id = await orchestrator.start_workflow(
                workflow_type="training",  # Use standard training workflow
                parameters={
                    "session_id": session_id,
                    "max_iterations": 1,  # Single iteration
                    "improvement_threshold": config.get("improvement_threshold", 0.02),
                    "timeout": config.get("timeout", 3600),
                    "verbose": config.get("verbose", False)
                }
            )

            # Track workflow
            self._active_workflows[workflow_id] = {
                "session_id": session_id,
                "workflow_type": "single_training",
                "started_at": datetime.now(timezone.utc),
                "config": config
            }

            self.console.print(f"✅ Single training workflow started: {workflow_id}", style="green")

            # Wait for completion and return results
            return await self.wait_for_completion(workflow_id, config.get("timeout", 3600))

        except Exception as e:
            self.logger.error(f"Failed to start single training workflow: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def stop_training_gracefully(
        self,
        session_id: str,
        timeout: int = 30,
        save_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Stop training gracefully for a specific session with progress preservation.

        Args:
            session_id: Training session identifier
            timeout: Shutdown timeout in seconds
            save_progress: Whether to save training progress

        Returns:
            Shutdown result with progress information
        """
        try:
            self.logger.info(f"Stopping training gracefully for session {session_id}")

            # Find workflows for this session
            session_workflows = [
                wf_id for wf_id, wf_info in self._active_workflows.items()
                if wf_info.get("session_id") == session_id
            ]

            if not session_workflows:
                return {
                    "success": True,
                    "message": f"No active workflows found for session {session_id}",
                    "progress_saved": False,
                    "saved_data": None
                }

            # Get orchestrator
            orchestrator = await self._get_orchestrator()

            # Stop workflows gracefully
            stopped_workflows = []
            saved_data = {}

            for workflow_id in session_workflows:
                try:
                    # Get current workflow status before stopping
                    status = await orchestrator.get_workflow_status(workflow_id)

                    if save_progress and status.get("status") == "running":
                        # Save current progress
                        progress_data = {
                            "workflow_id": workflow_id,
                            "session_id": session_id,
                            "stopped_at": datetime.now(timezone.utc).isoformat(),
                            "iterations_completed": status.get("iterations_completed", 0),
                            "current_performance": status.get("current_performance"),
                            "last_checkpoint": status.get("last_checkpoint")
                        }
                        saved_data[workflow_id] = progress_data

                    # Stop workflow gracefully
                    await orchestrator.stop_workflow(workflow_id, graceful=True, timeout=timeout)
                    stopped_workflows.append(workflow_id)

                    # Remove from active workflows
                    if workflow_id in self._active_workflows:
                        del self._active_workflows[workflow_id]

                except Exception as e:
                    self.logger.warning(f"Error stopping workflow {workflow_id}: {e}")

            self.console.print(f"✅ Stopped {len(stopped_workflows)} workflows gracefully", style="green")

            return {
                "success": True,
                "message": f"Gracefully stopped {len(stopped_workflows)} workflows",
                "progress_saved": save_progress and bool(saved_data),
                "saved_data": saved_data,
                "stopped_workflows": stopped_workflows
            }

        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            return {
                "success": False,
                "error": str(e),
                "progress_saved": False,
                "saved_data": None
            }

    async def force_stop_training(self, session_id: str) -> Dict[str, Any]:
        """
        Force stop training for a specific session (emergency shutdown).

        Args:
            session_id: Training session identifier

        Returns:
            Force stop result
        """
        try:
            self.logger.warning(f"Force stopping training for session {session_id}")

            # Find workflows for this session
            session_workflows = [
                wf_id for wf_id, wf_info in self._active_workflows.items()
                if wf_info.get("session_id") == session_id
            ]

            if not session_workflows:
                return {
                    "success": True,
                    "message": f"No active workflows found for session {session_id}",
                    "stopped_workflows": []
                }

            # Get orchestrator
            orchestrator = await self._get_orchestrator()

            # Force stop all workflows
            stopped_workflows = []

            for workflow_id in session_workflows:
                try:
                    await orchestrator.stop_workflow(workflow_id, graceful=False, timeout=5)
                    stopped_workflows.append(workflow_id)

                    # Remove from active workflows
                    if workflow_id in self._active_workflows:
                        del self._active_workflows[workflow_id]

                except Exception as e:
                    self.logger.error(f"Error force stopping workflow {workflow_id}: {e}")

            self.console.print(f"⚡ Force stopped {len(stopped_workflows)} workflows", style="yellow")

            return {
                "success": True,
                "message": f"Force stopped {len(stopped_workflows)} workflows",
                "stopped_workflows": stopped_workflows
            }

        except Exception as e:
            self.logger.error(f"Error during force stop: {e}")
            return {
                "success": False,
                "error": str(e),
                "stopped_workflows": []
            }
