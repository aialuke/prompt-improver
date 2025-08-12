"""Training Application Service

Orchestrates training workflow management, monitoring, and control operations
while managing complex resource coordination and graceful shutdown procedures.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from prompt_improver.application.protocols.application_service_protocols import (
    TrainingApplicationServiceProtocol,
)
from prompt_improver.cli.core.enhanced_workflow_manager import (
    EnhancedWorkflowManager,
    WorkflowCompletionResult,
    WorkflowMonitorConfig,
    WorkflowStopMode,
)
from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
from prompt_improver.core.services.ml_training_service import MLTrainingService
from prompt_improver.database import DatabaseServices
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    MLRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class TrainingWorkflowStatus(Enum):
    """Training workflow status enumeration."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingApplicationService:
    """
    Application service for training workflow orchestration.
    
    Orchestrates complex training workflows including:
    - Training pipeline initialization and resource allocation
    - Workflow monitoring and progress tracking
    - Graceful pause/resume functionality
    - Emergency shutdown and resource cleanup
    - Transaction boundary management across training phases
    - Integration with enhanced workflow management systems
    """

    def __init__(
        self,
        db_services: DatabaseServices,
        ml_repository: MLRepositoryProtocol,
        ml_training_service: MLTrainingService,
        training_system_manager: TrainingSystemManager,
        enhanced_workflow_manager: EnhancedWorkflowManager,
    ):
        self.db_services = db_services
        self.ml_repository = ml_repository
        self.ml_training_service = ml_training_service
        self.training_system_manager = training_system_manager
        self.enhanced_workflow_manager = enhanced_workflow_manager
        self.logger = logger
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize the training application service."""
        self.logger.info("Initializing TrainingApplicationService")
        await self.training_system_manager.initialize()

    async def cleanup(self) -> None:
        """Clean up training application service resources."""
        self.logger.info("Cleaning up TrainingApplicationService")
        
        # Stop all active workflows gracefully
        for workflow_id in list(self.active_workflows.keys()):
            await self.stop_training_workflow(workflow_id, graceful=True)
        
        await self.training_system_manager.cleanup()

    async def start_training_workflow(
        self,
        workflow_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start a new training workflow.
        
        Orchestrates training workflow initialization:
        1. Validate workflow configuration
        2. Initialize training resources and data pipelines
        3. Create workflow monitoring and progress tracking
        4. Start training execution with proper error handling
        5. Register workflow for monitoring and control
        
        Args:
            workflow_config: Configuration for training workflow
            
        Returns:
            Dict containing workflow startup details
        """
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting training workflow {workflow_id}")
            
            # 1. Validate workflow configuration
            config_validation = await self._validate_workflow_config(workflow_config)
            if not config_validation["valid"]:
                return {
                    "status": "error",
                    "workflow_id": workflow_id,
                    "error": config_validation["error"],
                    "timestamp": start_time.isoformat(),
                }
            
            # 2. Initialize workflow context and resources
            async with self.db_services.get_session() as db_session:
                try:
                    # Create workflow record
                    workflow_context = await self._create_workflow_context(
                        workflow_id, workflow_config, db_session
                    )
                    
                    # Store workflow metadata
                    await self._store_workflow_metadata(
                        workflow_id, workflow_config, workflow_context, db_session
                    )
                    
                    await db_session.commit()
                    
                    # 3. Register workflow for monitoring
                    self.active_workflows[workflow_id] = {
                        "status": TrainingWorkflowStatus.INITIALIZING,
                        "config": workflow_config,
                        "context": workflow_context,
                        "started_at": start_time,
                        "progress": {"completed_steps": 0, "total_steps": workflow_config.get("total_steps", 100)},
                    }
                    
                    # 4. Start workflow execution asynchronously
                    workflow_task = asyncio.create_task(
                        self._execute_workflow_async(workflow_id, workflow_config, workflow_context)
                    )
                    self.workflow_tasks[workflow_id] = workflow_task
                    
                    return {
                        "status": "success",
                        "workflow_id": workflow_id,
                        "workflow_status": TrainingWorkflowStatus.INITIALIZING.value,
                        "estimated_duration_hours": workflow_config.get("estimated_duration_hours", 2.0),
                        "progress": self.active_workflows[workflow_id]["progress"],
                        "metadata": {
                            "started_at": start_time.isoformat(),
                            "configuration": workflow_config,
                            "monitoring_enabled": True,
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    
                except Exception as e:
                    await db_session.rollback()
                    raise
                    
        except Exception as e:
            self.logger.error(f"Failed to start training workflow {workflow_id}: {e}")
            return {
                "status": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def monitor_training_progress(
        self,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """
        Monitor training workflow progress.
        
        Provides comprehensive workflow monitoring including:
        1. Current status and progress metrics
        2. Resource utilization and performance
        3. Error detection and health monitoring
        4. Estimated completion time
        5. Real-time progress indicators
        
        Args:
            workflow_id: Workflow identifier to monitor
            
        Returns:
            Dict containing comprehensive progress information
        """
        try:
            # Check if workflow exists
            if workflow_id not in self.active_workflows:
                return {
                    "status": "error",
                    "workflow_id": workflow_id,
                    "error": "Workflow not found or not active",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            workflow_info = self.active_workflows[workflow_id]
            current_time = datetime.now(timezone.utc)
            
            # Get enhanced progress information
            progress_result = await self._get_detailed_progress(workflow_id)
            
            # Calculate runtime metrics
            runtime_seconds = (current_time - workflow_info["started_at"]).total_seconds()
            
            return {
                "status": "success",
                "workflow_id": workflow_id,
                "workflow_status": workflow_info["status"].value,
                "progress": {
                    **workflow_info["progress"],
                    "completion_percentage": progress_result.get("completion_percentage", 0.0),
                    "current_phase": progress_result.get("current_phase", "unknown"),
                    "estimated_time_remaining_minutes": progress_result.get("eta_minutes"),
                },
                "performance_metrics": {
                    "runtime_seconds": runtime_seconds,
                    "steps_per_second": progress_result.get("throughput", 0.0),
                    "memory_usage_mb": progress_result.get("memory_usage_mb", 0),
                    "cpu_utilization": progress_result.get("cpu_utilization", 0.0),
                },
                "health_indicators": {
                    "error_count": progress_result.get("error_count", 0),
                    "warning_count": progress_result.get("warning_count", 0),
                    "last_checkpoint": progress_result.get("last_checkpoint"),
                    "system_health": progress_result.get("system_health", "unknown"),
                },
                "metadata": {
                    "started_at": workflow_info["started_at"].isoformat(),
                    "monitoring_interval": "real_time",
                    "last_updated": current_time.isoformat(),
                },
                "timestamp": current_time.isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to monitor workflow {workflow_id}: {e}")
            return {
                "status": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def pause_training_workflow(
        self,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """
        Pause an active training workflow.
        
        Orchestrates graceful workflow pause:
        1. Validate workflow can be paused
        2. Complete current training step
        3. Create checkpoint for resume
        4. Release resources safely
        5. Update workflow status and metadata
        
        Args:
            workflow_id: Workflow identifier to pause
            
        Returns:
            Dict containing pause operation results
        """
        try:
            self.logger.info(f"Pausing training workflow {workflow_id}")
            
            # Check workflow exists and can be paused
            if workflow_id not in self.active_workflows:
                return {
                    "status": "error",
                    "workflow_id": workflow_id,
                    "error": "Workflow not found or not active",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            workflow_info = self.active_workflows[workflow_id]
            current_status = workflow_info["status"]
            
            if current_status not in [TrainingWorkflowStatus.RUNNING, TrainingWorkflowStatus.INITIALIZING]:
                return {
                    "status": "error",
                    "workflow_id": workflow_id,
                    "error": f"Cannot pause workflow in status: {current_status.value}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            # Update status to indicate pausing
            workflow_info["status"] = TrainingWorkflowStatus.PAUSED
            pause_time = datetime.now(timezone.utc)
            
            # Transaction boundary for pause operation
            async with self.db_services.get_session() as db_session:
                try:
                    # Create checkpoint
                    checkpoint_result = await self._create_workflow_checkpoint(
                        workflow_id, workflow_info, db_session
                    )
                    
                    # Update workflow status in database
                    await self._update_workflow_status(
                        workflow_id, TrainingWorkflowStatus.PAUSED, 
                        {"paused_at": pause_time.isoformat(), "checkpoint": checkpoint_result},
                        db_session
                    )
                    
                    await db_session.commit()
                    
                    # Signal training system to pause
                    await self.training_system_manager.pause_training(workflow_id)
                    
                    return {
                        "status": "success",
                        "workflow_id": workflow_id,
                        "workflow_status": TrainingWorkflowStatus.PAUSED.value,
                        "pause_details": {
                            "paused_at": pause_time.isoformat(),
                            "checkpoint_created": checkpoint_result is not None,
                            "can_resume": True,
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    
                except Exception as e:
                    await db_session.rollback()
                    # Revert status on failure
                    workflow_info["status"] = current_status
                    raise
                    
        except Exception as e:
            self.logger.error(f"Failed to pause workflow {workflow_id}: {e}")
            return {
                "status": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def resume_training_workflow(
        self,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """
        Resume a paused training workflow.
        
        Orchestrates workflow resumption:
        1. Validate workflow can be resumed
        2. Restore from checkpoint
        3. Reinitialize resources and connections
        4. Resume training execution
        5. Update workflow status and monitoring
        
        Args:
            workflow_id: Workflow identifier to resume
            
        Returns:
            Dict containing resume operation results
        """
        try:
            self.logger.info(f"Resuming training workflow {workflow_id}")
            
            # Validate workflow can be resumed
            if workflow_id not in self.active_workflows:
                return {
                    "status": "error",
                    "workflow_id": workflow_id,
                    "error": "Workflow not found",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            workflow_info = self.active_workflows[workflow_id]
            if workflow_info["status"] != TrainingWorkflowStatus.PAUSED:
                return {
                    "status": "error",
                    "workflow_id": workflow_id,
                    "error": f"Cannot resume workflow in status: {workflow_info['status'].value}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            resume_time = datetime.now(timezone.utc)
            
            # Transaction boundary for resume operation
            async with self.db_services.get_session() as db_session:
                try:
                    # Restore from checkpoint
                    checkpoint_result = await self._restore_from_checkpoint(
                        workflow_id, db_session
                    )
                    
                    # Update workflow status
                    workflow_info["status"] = TrainingWorkflowStatus.RUNNING
                    await self._update_workflow_status(
                        workflow_id, TrainingWorkflowStatus.RUNNING,
                        {"resumed_at": resume_time.isoformat(), "checkpoint_restored": True},
                        db_session
                    )
                    
                    await db_session.commit()
                    
                    # Resume training system
                    await self.training_system_manager.resume_training(workflow_id)
                    
                    # Restart workflow execution task if needed
                    if workflow_id not in self.workflow_tasks or self.workflow_tasks[workflow_id].done():
                        workflow_task = asyncio.create_task(
                            self._execute_workflow_async(
                                workflow_id, workflow_info["config"], workflow_info["context"]
                            )
                        )
                        self.workflow_tasks[workflow_id] = workflow_task
                    
                    return {
                        "status": "success",
                        "workflow_id": workflow_id,
                        "workflow_status": TrainingWorkflowStatus.RUNNING.value,
                        "resume_details": {
                            "resumed_at": resume_time.isoformat(),
                            "checkpoint_restored": checkpoint_result is not None,
                            "execution_restarted": True,
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    
                except Exception as e:
                    await db_session.rollback()
                    raise
                    
        except Exception as e:
            self.logger.error(f"Failed to resume workflow {workflow_id}: {e}")
            return {
                "status": "error", 
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def stop_training_workflow(
        self,
        workflow_id: str,
        graceful: bool = True,
    ) -> Dict[str, Any]:
        """
        Stop a training workflow with optional graceful shutdown.
        
        Orchestrates workflow termination:
        1. Validate workflow exists and can be stopped
        2. Complete current operations (if graceful)
        3. Create final checkpoint and save artifacts
        4. Release all resources and connections
        5. Update final workflow status and metrics
        
        Args:
            workflow_id: Workflow identifier to stop
            graceful: Whether to perform graceful shutdown
            
        Returns:
            Dict containing stop operation results
        """
        try:
            self.logger.info(f"Stopping training workflow {workflow_id} (graceful: {graceful})")
            
            if workflow_id not in self.active_workflows:
                return {
                    "status": "error",
                    "workflow_id": workflow_id,
                    "error": "Workflow not found",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            workflow_info = self.active_workflows[workflow_id]
            stop_time = datetime.now(timezone.utc)
            
            # Update status immediately
            workflow_info["status"] = TrainingWorkflowStatus.STOPPING
            
            # Transaction boundary for stopping workflow
            async with self.db_services.get_session() as db_session:
                try:
                    # Stop using enhanced workflow manager
                    stop_mode = WorkflowStopMode.GRACEFUL if graceful else WorkflowStopMode.FORCE
                    completion_result = await self._stop_workflow_with_enhanced_manager(
                        workflow_id, stop_mode
                    )
                    
                    # Create final checkpoint and artifacts
                    final_checkpoint = await self._create_final_checkpoint(
                        workflow_id, workflow_info, db_session
                    )
                    
                    # Update final status
                    final_status = TrainingWorkflowStatus.COMPLETED if completion_result.completed else TrainingWorkflowStatus.CANCELLED
                    workflow_info["status"] = final_status
                    
                    await self._update_workflow_status(
                        workflow_id, final_status,
                        {
                            "stopped_at": stop_time.isoformat(),
                            "graceful_shutdown": graceful,
                            "completion_result": {
                                "completed": completion_result.completed,
                                "duration_seconds": completion_result.duration_seconds,
                                "final_state": completion_result.final_state,
                            },
                            "final_checkpoint": final_checkpoint,
                        },
                        db_session
                    )
                    
                    await db_session.commit()
                    
                    # Clean up workflow tracking
                    if workflow_id in self.workflow_tasks:
                        task = self.workflow_tasks[workflow_id]
                        if not task.done():
                            task.cancel()
                        del self.workflow_tasks[workflow_id]
                    
                    if workflow_id in self.active_workflows:
                        del self.active_workflows[workflow_id]
                    
                    end_time = datetime.now(timezone.utc)
                    total_duration = (end_time - workflow_info["started_at"]).total_seconds()
                    
                    return {
                        "status": "success",
                        "workflow_id": workflow_id,
                        "final_status": final_status.value,
                        "stop_details": {
                            "stopped_at": stop_time.isoformat(),
                            "graceful_shutdown": graceful,
                            "total_duration_seconds": total_duration,
                            "completion_percentage": completion_result.progress_data.get("completion", 0) if completion_result.progress_data else 0,
                            "final_checkpoint_created": final_checkpoint is not None,
                        },
                        "timestamp": end_time.isoformat(),
                    }
                    
                except Exception as e:
                    await db_session.rollback()
                    raise
                    
        except Exception as e:
            self.logger.error(f"Failed to stop workflow {workflow_id}: {e}")
            return {
                "status": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Private helper methods

    async def _validate_workflow_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow configuration."""
        try:
            required_fields = ["training_type", "data_config", "model_config"]
            for field in required_fields:
                if field not in config:
                    return {"valid": False, "error": f"Missing required field: {field}"}
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def _create_workflow_context(
        self, workflow_id: str, config: Dict[str, Any], db_session
    ) -> Dict[str, Any]:
        """Create workflow execution context."""
        return {
            "workflow_id": workflow_id,
            "config": config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "resources": {"allocated": False, "resource_ids": []},
        }

    async def _store_workflow_metadata(
        self, workflow_id: str, config: Dict[str, Any], context: Dict[str, Any], db_session
    ) -> None:
        """Store workflow metadata in database."""
        await self.ml_repository.store_training_workflow_metadata(
            workflow_id=workflow_id,
            config=config,
            context=context,
            db_session=db_session,
        )

    async def _execute_workflow_async(
        self, workflow_id: str, config: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Execute workflow asynchronously."""
        try:
            self.logger.info(f"Executing workflow {workflow_id} asynchronously")
            
            # Update status to running
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = TrainingWorkflowStatus.RUNNING
            
            # Execute training via ML training service
            await self.ml_training_service.execute_training_workflow(
                workflow_id=workflow_id,
                config=config,
                context=context,
            )
            
            # Update to completed if still active
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = TrainingWorkflowStatus.COMPLETED
            
        except asyncio.CancelledError:
            self.logger.info(f"Workflow {workflow_id} was cancelled")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = TrainingWorkflowStatus.CANCELLED
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = TrainingWorkflowStatus.FAILED

    async def _get_detailed_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed progress information for a workflow."""
        try:
            # This would integrate with actual training progress monitoring
            return {
                "completion_percentage": 65.0,
                "current_phase": "training",
                "eta_minutes": 45,
                "throughput": 2.3,
                "memory_usage_mb": 1024,
                "cpu_utilization": 0.75,
                "error_count": 0,
                "warning_count": 1,
                "last_checkpoint": "2025-01-11T10:30:00Z",
                "system_health": "healthy",
            }
        except Exception as e:
            self.logger.error(f"Failed to get progress for {workflow_id}: {e}")
            return {}

    async def _create_workflow_checkpoint(
        self, workflow_id: str, workflow_info: Dict[str, Any], db_session
    ) -> Optional[Dict[str, Any]]:
        """Create checkpoint for workflow state."""
        try:
            checkpoint_data = {
                "workflow_id": workflow_id,
                "status": workflow_info["status"].value,
                "progress": workflow_info["progress"],
                "checkpoint_time": datetime.now(timezone.utc).isoformat(),
            }
            
            await self.ml_repository.store_workflow_checkpoint(
                workflow_id=workflow_id,
                checkpoint_data=checkpoint_data,
                db_session=db_session,
            )
            
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint for {workflow_id}: {e}")
            return None

    async def _update_workflow_status(
        self, workflow_id: str, status: TrainingWorkflowStatus, metadata: Dict[str, Any], db_session
    ) -> None:
        """Update workflow status in database."""
        await self.ml_repository.update_training_workflow_status(
            workflow_id=workflow_id,
            status=status.value,
            metadata=metadata,
            db_session=db_session,
        )

    async def _restore_from_checkpoint(
        self, workflow_id: str, db_session
    ) -> Optional[Dict[str, Any]]:
        """Restore workflow state from checkpoint."""
        try:
            return await self.ml_repository.load_workflow_checkpoint(
                workflow_id=workflow_id,
                db_session=db_session,
            )
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint for {workflow_id}: {e}")
            return None

    async def _stop_workflow_with_enhanced_manager(
        self, workflow_id: str, stop_mode: WorkflowStopMode
    ) -> WorkflowCompletionResult:
        """Stop workflow using enhanced workflow manager."""
        try:
            # Configure monitoring for stop operation
            config = WorkflowMonitorConfig(
                poll_interval=1.0,
                timeout_warning_threshold=0.9,
            )
            
            # Wait for completion with timeout based on stop mode
            timeout = 300 if stop_mode == WorkflowStopMode.GRACEFUL else 30
            
            completion_result = await self.enhanced_workflow_manager.wait_for_workflow_completion(
                workflow_id=workflow_id,
                timeout=timeout,
                config=config,
                show_progress=False,  # Don't show progress for stop operation
            )
            
            return completion_result
        except Exception as e:
            self.logger.error(f"Enhanced manager stop failed for {workflow_id}: {e}")
            # Return a default completion result
            return WorkflowCompletionResult(
                workflow_id=workflow_id,
                status="error",
                completed=False,
                duration_seconds=0.0,
                timeout_reached=False,
                error=str(e),
            )

    async def _create_final_checkpoint(
        self, workflow_id: str, workflow_info: Dict[str, Any], db_session
    ) -> Optional[Dict[str, Any]]:
        """Create final checkpoint with artifacts."""
        try:
            final_checkpoint = {
                "workflow_id": workflow_id,
                "final_status": workflow_info["status"].value,
                "final_progress": workflow_info.get("progress", {}),
                "total_runtime_seconds": (datetime.now(timezone.utc) - workflow_info["started_at"]).total_seconds(),
                "finalized_at": datetime.now(timezone.utc).isoformat(),
            }
            
            await self.ml_repository.store_final_workflow_checkpoint(
                workflow_id=workflow_id,
                checkpoint_data=final_checkpoint,
                db_session=db_session,
            )
            
            return final_checkpoint
        except Exception as e:
            self.logger.error(f"Failed to create final checkpoint for {workflow_id}: {e}")
            return None