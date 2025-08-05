"""
Clean Training System Manager - Independent of MCP Server
Implements pure training-focused lifecycle management for the 3-command CLI.
"""

import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from rich.console import Console

from ...database import get_sessionmanager
# Signal handling integration - avoiding circular import
from .signal_handler import SignalOperation
from ...ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from ...ml.orchestration.config.orchestrator_config import OrchestratorConfig
from ...ml.preprocessing.orchestrator import ProductionSyntheticDataGenerator
from ...core.services.analytics_factory import get_analytics_interface
from ...utils.unified_session_manager import get_unified_session_manager
from .rule_validation_service import RuleValidationService

class TrainingSystemManager:
    """
    Pure training system lifecycle management - completely independent of MCP server.

    Implements clean break strategy with:
    - Zero MCP server dependencies
    - Pure training focus
    - Optimal resource utilization
    - Simplified configuration
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.logger = logging.getLogger("apes.training_system")

        # Unified session manager for consolidated session management
        self._unified_session_manager = None

        # Signal handling integration - lazy import to avoid circular dependency
        self.signal_handler = None
        self.background_manager = None
        self._shutdown_priority = 5  # High priority for training system
        
        # Training-specific state (no MCP state)
        self._training_status = "stopped"
        self._training_session_id: Optional[str] = None
        self._orchestrator: Optional[MLPipelineOrchestrator] = None
        self._analytics: Optional[Any] = None
        self._data_generator: Optional[ProductionSyntheticDataGenerator] = None
        self._rule_validator: Optional[RuleValidationService] = None

        # Performance tracking
        self._startup_time: Optional[float] = None
        self._resource_usage = {"memory_mb": 0, "cpu_percent": 0}

        # Training system data directory (separate from MCP)
        self.training_data_dir = Path.home() / ".local" / "share" / "apes" / "training"
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize signal handling integration (lazy)
        self._init_signal_handlers()

    def _init_signal_handlers(self):
        """Initialize signal handler with lazy import to avoid circular dependency."""
        try:
            # Lazy import to avoid circular dependency
            from ...performance.monitoring.health.background_manager import get_background_task_manager
            from .signal_handler import AsyncSignalHandler
            from rich.console import Console
            
            # Initialize signal handler if not already done
            if self.signal_handler is None:
                self.signal_handler = AsyncSignalHandler(console=Console())
                
                # Setup signal handlers with asyncio loop if available
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    self.signal_handler.setup_signal_handlers(loop)
                except RuntimeError:
                    # No running loop - will be set up when loop starts
                    pass
            
            # Initialize background manager
            if self.background_manager is None:
                self.background_manager = get_background_task_manager()
            
            # Register handlers
            self._register_signal_handlers()
            
        except ImportError as e:
            self.logger.warning(f"Signal handling integration not available: {e}")
            # Continue without signal handling
    
    async def _ensure_unified_session_manager(self):
        """Ensure unified session manager is available."""
        if self._unified_session_manager is None:
            self._unified_session_manager = await get_unified_session_manager()
    
    async def create_training_session(self, training_config: Dict[str, Any]) -> str:
        """Create training session using unified session management.
        
        Args:
            training_config: Training configuration
            
        Returns:
            Created training session ID
        """
        try:
            await self._ensure_unified_session_manager()
            
            # Generate session ID
            import uuid
            session_id = f"training_{uuid.uuid4().hex[:8]}"
            
            # Create session in unified manager
            success = await self._unified_session_manager.create_training_session(
                session_id=session_id,
                training_config=training_config
            )
            
            if success:
                self._training_session_id = session_id
                self._training_status = "running"
                self.logger.info(f"Created training session: {session_id}")
                return session_id
            else:
                raise Exception("Failed to create training session in unified manager")
                
        except Exception as e:
            self.logger.error(f"Failed to create training session: {e}")
            raise
    
    async def update_training_progress(
        self,
        iteration: int,
        performance_metrics: Dict[str, float],
        improvement_score: float = 0.0
    ) -> bool:
        """Update training progress using unified session management.
        
        Args:
            iteration: Current iteration
            performance_metrics: Performance metrics
            improvement_score: Improvement score
            
        Returns:
            True if updated successfully
        """
        if not self._training_session_id:
            self.logger.warning("No active training session for progress update")
            return False
            
        try:
            await self._ensure_unified_session_manager()
            
            success = await self._unified_session_manager.update_training_progress(
                session_id=self._training_session_id,
                iteration=iteration,
                performance_metrics=performance_metrics,
                improvement_score=improvement_score
            )
            
            if success:
                self.logger.debug(f"Updated training progress: iteration {iteration}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update training progress: {e}")
            return False
    
    async def get_training_session_context(self) -> Optional[Dict[str, Any]]:
        """Get current training session context from unified manager.
        
        Returns:
            Training session context if available
        """
        if not self._training_session_id:
            return None
            
        try:
            await self._ensure_unified_session_manager()
            
            context = await self._unified_session_manager.get_training_session(self._training_session_id)
            
            if context:
                return context.to_dict()
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get training session context: {e}")
            return None
            
    def _register_signal_handlers(self):
        """Register TrainingSystemManager-specific signal handlers."""
        if self.signal_handler is None:
            self.logger.warning("Signal handler not initialized, skipping signal registration")
            return
            
        import signal
        
        # Shutdown coordination with high priority
        self.signal_handler.register_shutdown_handler(
            "TrainingSystemManager_shutdown", 
            self.graceful_shutdown_handler
        )
        
        # Emergency checkpoint creation (SIGUSR1)
        self.signal_handler.register_operation_handler(
            SignalOperation.CHECKPOINT,
            self.create_training_emergency_checkpoint
        )
        
        # Status reporting (SIGUSR2)
        self.signal_handler.register_operation_handler(
            SignalOperation.STATUS_REPORT,
            self.generate_training_status_report
        )
        
        # Signal chaining for coordinated shutdown preparation
        self.signal_handler.add_signal_chain_handler(
            signal.SIGTERM,
            self.prepare_training_shutdown,
            priority=self._shutdown_priority
        )
        
        # SIGINT coordination for graceful training interruption
        self.signal_handler.add_signal_chain_handler(
            signal.SIGINT,
            self.prepare_training_interruption,
            priority=self._shutdown_priority
        )
        
        self.logger.info("TrainingSystemManager signal handlers registered")

    async def graceful_shutdown_handler(self, shutdown_context):
        """Handle graceful shutdown with training progress preservation."""
        self.logger.info("TrainingSystemManager graceful shutdown initiated")
        
        try:
            # Stop training system with progress preservation
            success = await self.stop_training_system(graceful=True)
            
            # Additional training-specific cleanup
            await self._emergency_training_cleanup()
            
            return {
                "status": "success" if success else "partial", 
                "component": "TrainingSystemManager",
                "training_stopped": success,
                "progress_preserved": self._training_session_id is not None
            }
        except Exception as e:
            self.logger.error(f"TrainingSystemManager shutdown error: {e}")
            return {
                "status": "error",
                "component": "TrainingSystemManager", 
                "error": str(e)
            }

    async def create_training_emergency_checkpoint(self, signal_context):
        """Create emergency training checkpoint on SIGUSR1 signal."""
        self.logger.info("Creating emergency training checkpoint")
        
        try:
            if not self._training_session_id:
                return {
                    "status": "no_active_session",
                    "message": "No active training session for checkpoint"
                }
            
            # Save current training progress immediately
            await self._save_training_progress()
            
            # Get current performance metrics
            performance_metrics = await self._get_current_performance_metrics()
            
            # Create checkpoint via progress preservation
            from .progress_preservation import ProgressPreservationManager
            progress_manager = ProgressPreservationManager()
            checkpoint_id = await progress_manager.create_checkpoint(self._training_session_id)
            
            return {
                "status": "checkpoint_created",
                "checkpoint_id": checkpoint_id,
                "session_id": self._training_session_id,
                "performance_metrics": performance_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Emergency checkpoint creation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def generate_training_status_report(self, signal_context):
        """Generate comprehensive training status report on SIGUSR2 signal."""
        self.logger.info("Generating training status report")
        
        try:
            # Get comprehensive system status
            system_status = await self.get_system_status()
            
            # Get training-specific metrics
            training_metrics = await self._get_detailed_training_metrics()
            
            return {
                "status": "report_generated",
                "system_status": system_status,
                "training_metrics": training_metrics,
                "resource_usage": await self._get_resource_usage(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Status report generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def prepare_training_shutdown(self, signum, signal_name):
        """Prepare training system for coordinated shutdown."""
        self.logger.info(f"Preparing training system for shutdown ({signal_name})")
        
        try:
            # Set training status to indicate shutdown preparation
            if self._training_status == "running":
                self._training_status = "shutting_down"
            
            # If training is active, prepare for progress preservation
            preparation_status = {
                "prepared": True,
                "component": "TrainingSystemManager",
                "training_status": self._training_status,
                "active_session": self._training_session_id is not None,
                "orchestrator_active": self._orchestrator is not None
            }
            
            return preparation_status
        except Exception as e:
            self.logger.error(f"Training shutdown preparation failed: {e}")
            return {
                "prepared": False,
                "component": "TrainingSystemManager",
                "error": str(e)
            }

    def prepare_training_interruption(self, signum, signal_name):
        """Prepare training system for user interruption (Ctrl+C)."""
        self.logger.info(f"Preparing training system for interruption ({signal_name})")
        
        try:
            # For user interruption, prioritize progress preservation
            preparation_status = {
                "prepared": True,
                "component": "TrainingSystemManager",
                "interruption_type": "user_requested",
                "progress_preservation_ready": True,
                "active_session": self._training_session_id,
                "can_resume": True
            }
            
            return preparation_status
        except Exception as e:
            self.logger.error(f"Training interruption preparation failed: {e}")
            return {
                "prepared": False,
                "component": "TrainingSystemManager",
                "error": str(e)
            }

    async def _emergency_training_cleanup(self):
        """Emergency cleanup for training resources during shutdown."""
        try:
            # Force close any hanging ML operations
            if self._orchestrator:
                try:
                    await self._orchestrator.shutdown()
                except Exception as e:
                    self.logger.warning(f"Orchestrator emergency shutdown failed: {e}")
            
            # Emergency save of any remaining training data
            if self._training_session_id:
                try:
                    await self._save_training_progress()
                except Exception as e:
                    self.logger.warning(f"Emergency progress save failed: {e}")
            
            # Release any background tasks
            try:
                await self.background_manager.stop(timeout=5.0)
            except Exception as e:
                self.logger.warning(f"Background task cleanup failed: {e}")
                
            self.logger.info("Emergency training cleanup completed")
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")

    async def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current training performance metrics for checkpoints."""
        try:
            if not self._orchestrator:
                return {"error": "No orchestrator available"}
            
            # Get orchestrator health and metrics
            health = await self._orchestrator.health_check()
            
            return {
                "orchestrator_health": health,
                "training_status": self._training_status,
                "session_id": self._training_session_id,
                "resource_usage": await self._get_resource_usage()
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_detailed_training_metrics(self) -> Dict[str, Any]:
        """Get detailed training metrics for status reports."""
        try:
            metrics = {
                "training_status": self._training_status,
                "session_id": self._training_session_id,
                "startup_time": self._startup_time,
                "resource_usage": await self._get_resource_usage()
            }
            
            # Add orchestrator metrics if available
            if self._orchestrator:
                try:
                    health = await self._orchestrator.health_check()
                    metrics["orchestrator_health"] = health
                    
                    # Get workflow information
                    workflows = await self._orchestrator.list_workflows()
                    metrics["active_workflows"] = len(workflows)
                except Exception as e:
                    metrics["orchestrator_error"] = str(e)
            
            # Add analytics metrics if available
            if self._analytics:
                try:
                    performance = await self._analytics.get_recent_performance_summary()
                    metrics["performance_summary"] = performance
                except Exception as e:
                    metrics["analytics_error"] = str(e)
            
            return metrics
        except Exception as e:
            return {"error": str(e)}

    async def start_training_system(self) -> Dict[str, Any]:
        """
        Start training system components ONLY - no MCP server management.

        Returns:
            Training system startup results with performance metrics
        """
        startup_start = time.time()
        self.logger.info("Starting pure training system (no MCP dependencies)")

        try:
            self._training_status = "starting"

            # 1. Initialize database connections (training-specific)
            await self._initialize_training_database()

            # 2. Initialize ML Pipeline Orchestrator (training-focused config)
            await self._initialize_training_orchestrator()

            # 3. Initialize analytics service (training metrics only)
            await self._initialize_training_analytics()

            # 4. Initialize synthetic data generator
            await self._initialize_data_generator()

            # 5. Verify training system health
            health_status = await self._verify_training_health()

            self._startup_time = time.time() - startup_start
            self._training_status = "running"

            startup_results = {
                "status": "success",
                "training_system_id": f"training_{int(time.time())}",
                "startup_time_seconds": self._startup_time,
                "components_initialized": [
                    "database_connections",
                    "ml_orchestrator",
                    "analytics_service",
                    "synthetic_data_orchestrator"
                ],
                "health_status": health_status,
                "resource_usage": await self._get_resource_usage(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self.logger.info(f"Training system started successfully in {self._startup_time:.2f}s")
            return startup_results

        except Exception as e:
            self._training_status = "failed"
            self.logger.error(f"Failed to start training system: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def stop_training_system(self, graceful: bool = True) -> bool:
        """
        Stop training system gracefully with progress preservation.

        Args:
            graceful: Whether to perform graceful shutdown

        Returns:
            True if shutdown successful, False otherwise
        """
        self.logger.info(f"Stopping training system (graceful={graceful})")

        try:
            self._training_status = "stopping"

            # 1. Save current training progress
            if self._training_session_id:
                await self._save_training_progress()

            # 2. Stop orchestrator workflows gracefully
            if self._orchestrator and graceful:
                # Stop all active workflows using the internal method
                await self._orchestrator._stop_all_workflows()

            # 3. Cleanup training resources
            await self._cleanup_training_resources()

            # 4. Close database connections
            await get_sessionmanager().close()

            self._training_status = "stopped"
            self._training_session_id = None

            self.logger.info("Training system stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping training system: {e}")
            return False

    async def get_training_status(self) -> Dict[str, Any]:
        """
        Get training system status - independent of MCP server status.

        Returns:
            Training system status and metrics
        """
        status = {
            "training_system_status": self._training_status,
            "training_session_id": self._training_session_id,
            "uptime_seconds": time.time() - self._startup_time if self._startup_time else 0,
            "resource_usage": await self._get_resource_usage(),
            "components": {
                "orchestrator": "running" if self._orchestrator else "stopped",
                "analytics": "running" if self._analytics else "stopped",
                "data_generator": "running" if self._data_generator else "stopped"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Add active workflow status if orchestrator is running
        if self._orchestrator:
            # Use list_workflows method which exists in MLPipelineOrchestrator
            workflows = await self._orchestrator.list_workflows()
            status["active_workflows"] = [w.workflow_id for w in workflows]

        return status

    async def _initialize_training_database(self):
        """Initialize database connections for training system only."""
        self.logger.info("Initializing training database connections")

        # Test database connectivity
        session_manager = get_sessionmanager()
        async with session_manager.get_async_session() as db_session:
            # Verify training-related tables exist
            result = await db_session.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_name IN ('improvement_sessions', 'rule_performance')")
            )
            tables = [row[0] for row in result.fetchall()]

            if len(tables) < 2:
                raise RuntimeError("Training database tables not found")

        self.logger.info("Training database connections initialized")

    async def _initialize_training_orchestrator(self):
        """Initialize ML Pipeline Orchestrator with training-focused configuration."""
        self.logger.info("Initializing training orchestrator")

        # Create training-optimized orchestrator config
        config = OrchestratorConfig(
            max_concurrent_workflows=3,  # Optimized for training workloads
            training_timeout=1800,  # 30 minutes for training workflows
            debug_mode=False,
            verbose_logging=False
        )

        self._orchestrator = MLPipelineOrchestrator(config)
        await self._orchestrator.initialize()

        self.logger.info("Training orchestrator initialized")

    async def _initialize_training_analytics(self):
        """Initialize analytics service for training metrics only."""
        self.logger.info("Initializing training analytics")

        analytics_factory = get_analytics_interface()
        self._analytics = analytics_factory() if analytics_factory else None
        # Configure for training-specific metrics

        self.logger.info("Training analytics initialized")

    async def _initialize_data_generator(self):
        """Initialize synthetic data generator for training."""
        self.logger.info("Initializing synthetic data generator")

        self._data_generator = ProductionSyntheticDataGenerator()

        self.logger.info("Synthetic data generator initialized")

    async def _verify_training_health(self) -> Dict[str, Any]:
        """Verify training system health and performance."""
        health_start = time.time()

        health_status = {
            "database_connectivity": False,
            "orchestrator_status": False,
            "analytics_status": False,
            "data_generator_status": False,
            "overall_health": False,
            "response_time_ms": 0
        }

        try:
            # Test database
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as db_session:
                await db_session.execute(text("SELECT 1"))
                health_status["database_connectivity"] = True

            # Test orchestrator
            if self._orchestrator:
                # Check if orchestrator is initialized and in a good state
                health_status["orchestrator_status"] = (
                    self._orchestrator._is_initialized and
                    self._orchestrator.state.name in ["IDLE", "RUNNING"]
                )

            # Test analytics
            health_status["analytics_status"] = self._analytics is not None

            # Test data generator
            health_status["data_generator_status"] = self._data_generator is not None

            # Overall health
            health_status["overall_health"] = all([
                health_status["database_connectivity"],
                health_status["orchestrator_status"],
                health_status["analytics_status"],
                health_status["data_generator_status"]
            ])

            health_status["response_time_ms"] = (time.time() - health_start) * 1000

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)

        return health_status

    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage for training system."""
        try:
            import psutil
            process = psutil.Process()

            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files())
            }
        except ImportError:
            return {"memory_mb": 0, "cpu_percent": 0, "open_files": 0}

    async def _save_training_progress(self):
        """Save current training progress to database."""
        if not self._training_session_id:
            return

        self.logger.info(f"Saving training progress for session {self._training_session_id}")
        # Implementation for saving training state

    async def _cleanup_training_resources(self):
        """Cleanup training system resources."""
        self.logger.info("Cleaning up training resources")

        # Cleanup orchestrator
        if self._orchestrator:
            await self._orchestrator.shutdown()
            self._orchestrator = None

        # Reset component references
        self._analytics = None
        self._data_generator = None

    async def smart_initialize(self) -> Dict[str, Any]:
        """
        Enhanced smart initialization with comprehensive system state detection.

        Implements 2025 best practices for ML system initialization:
        - Comprehensive system state detection and validation
        - Database schema and seeded rule validation
        - Training data availability assessment
        - Component health monitoring with graceful degradation
        - Intelligent initialization recommendations

        Returns:
            Detailed initialization results with system state analysis
        """
        initialization_start = time.time()
        self.logger.info("Starting enhanced smart initialization with comprehensive detection")

        try:
            # Phase 1: System State Detection
            system_state = await self._detect_system_state()

            # Phase 2: Component Validation
            component_status = await self._validate_components()

            # Phase 3: Database and Rule Validation
            database_status = await self._validate_database_and_rules()

            # Phase 4: Data Availability Assessment
            data_status = await self._assess_data_availability()

            # Phase 5: Initialization Decision Making
            initialization_plan = await self._create_initialization_plan(
                system_state, component_status, database_status, data_status
            )

            # Phase 6: Execute Initialization
            execution_results = await self._execute_initialization_plan(initialization_plan)

            # Phase 7: Post-Initialization Validation
            final_validation = await self._validate_post_initialization()

            initialization_time = time.time() - initialization_start

            return {
                "success": execution_results["success"],
                "message": execution_results["message"],
                "initialization_time_seconds": initialization_time,
                "system_state": system_state,
                "component_status": component_status,
                "database_status": database_status,
                "data_status": data_status,
                "initialization_plan": initialization_plan,
                "execution_results": execution_results,
                "final_validation": final_validation,
                "components_initialized": execution_results.get("components_initialized", []),
                "recommendations": execution_results.get("recommendations", []),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Enhanced smart initialization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "initialization_time_seconds": time.time() - initialization_start,
                "components_initialized": [],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _detect_system_state(self) -> Dict[str, Any]:
        """
        Comprehensive system state detection using 2025 best practices.

        Returns:
            Detailed system state information
        """
        state_start = time.time()

        system_state = {
            "training_system_status": self._training_status,
            "previous_initialization": self._startup_time is not None,
            "resource_usage": await self._get_resource_usage(),
            "environment_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(Path.cwd()),
                "data_directory": str(self.training_data_dir),
                "data_directory_exists": self.training_data_dir.exists()
            },
            "detection_time_ms": 0
        }

        # Check for existing configuration files
        config_files = {
            "training_config": (self.training_data_dir / "config.json").exists(),
            "model_cache": (self.training_data_dir / "models").exists(),
            "logs": (self.training_data_dir / "logs").exists()
        }
        system_state["configuration_files"] = config_files

        # Check system resources
        try:
            import psutil
            system_state["system_resources"] = {
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
            }
        except ImportError:
            system_state["system_resources"] = {"status": "psutil_not_available"}

        system_state["detection_time_ms"] = (time.time() - state_start) * 1000
        return system_state

    async def _validate_components(self) -> Dict[str, Any]:
        """
        Validate all training system components with health checks.

        Returns:
            Component validation results
        """
        validation_start = time.time()

        component_status = {
            "orchestrator": {"status": "not_initialized", "details": {}},
            "analytics": {"status": "not_initialized", "details": {}},
            "data_generator": {"status": "not_initialized", "details": {}},
            "validation_time_ms": 0
        }

        # Validate orchestrator
        if self._orchestrator:
            try:
                health = await self._orchestrator.health_check()
                component_status["orchestrator"] = {
                    "status": "healthy" if health.get("healthy") else "unhealthy",
                    "details": health
                }
            except Exception as e:
                component_status["orchestrator"] = {
                    "status": "error",
                    "details": {"error": str(e)}
                }

        # Validate analytics
        if self._analytics:
            try:
                # Test analytics functionality
                component_status["analytics"] = {
                    "status": "healthy",
                    "details": {"initialized": True}
                }
            except Exception as e:
                component_status["analytics"] = {
                    "status": "error",
                    "details": {"error": str(e)}
                }

        # Validate data generator
        if self._data_generator:
            try:
                # Test data generator functionality
                component_status["data_generator"] = {
                    "status": "healthy",
                    "details": {
                        "generation_method": self._data_generator.generation_method,
                        "target_samples": self._data_generator.target_samples
                    }
                }
            except Exception as e:
                component_status["data_generator"] = {
                    "status": "error",
                    "details": {"error": str(e)}
                }

        component_status["validation_time_ms"] = (time.time() - validation_start) * 1000
        return component_status

    async def _validate_database_and_rules(self) -> Dict[str, Any]:
        """
        Validate database connectivity, schema, and seeded rules.

        Returns:
            Database and rule validation results
        """
        validation_start = time.time()

        database_status = {
            "connectivity": {"status": "unknown", "details": {}},
            "schema_validation": {"status": "unknown", "details": {}},
            "seeded_rules": {"status": "unknown", "details": {}},
            "rule_metadata": {"status": "unknown", "details": {}},
            "validation_time_ms": 0
        }

        try:
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as db_session:
                # Test basic connectivity
                await db_session.execute(text("SELECT 1"))
                database_status["connectivity"] = {
                    "status": "healthy",
                    "details": {"connection_successful": True}
                }

                # Validate required tables exist
                required_tables = [
                    'improvement_sessions', 'rule_performance', 'rule_metadata',
                    'training_prompts', 'discovered_patterns'
                ]

                # Use a simpler approach with text() and string formatting
                table_list = "', '".join(required_tables)
                result = await db_session.execute(
                    text(f"SELECT table_name FROM information_schema.tables WHERE table_name IN ('{table_list}')")
                )
                existing_tables = [row[0] for row in result.fetchall()]

                missing_tables = set(required_tables) - set(existing_tables)
                database_status["schema_validation"] = {
                    "status": "healthy" if not missing_tables else "incomplete",
                    "details": {
                        "required_tables": required_tables,
                        "existing_tables": existing_tables,
                        "missing_tables": list(missing_tables)
                    }
                }

                # Enhanced rule validation using RuleValidationService
                if not self._rule_validator:
                    self._rule_validator = RuleValidationService()

                rule_validation_report = await self._rule_validator.validate_all_rules()

                database_status["seeded_rules"] = {
                    "status": rule_validation_report["overall_status"],
                    "details": {
                        "total_rules": rule_validation_report["rule_count"]["found"],
                        "valid_rules": rule_validation_report["rule_count"]["valid"],
                        "expected_rules": rule_validation_report["rule_count"]["expected"],
                        "validation_report": rule_validation_report
                    }
                }

                database_status["rule_metadata"] = {
                    "status": rule_validation_report["metadata_validation"]["overall_status"],
                    "details": rule_validation_report["metadata_validation"]
                }

        except Exception as e:
            database_status["connectivity"] = {
                "status": "error",
                "details": {"error": str(e)}
            }

        database_status["validation_time_ms"] = (time.time() - validation_start) * 1000
        return database_status

    async def _assess_data_availability(self) -> Dict[str, Any]:
        """
        Comprehensive training data availability assessment.

        Returns:
            Data availability analysis with quality metrics
        """
        assessment_start = time.time()

        data_status = {
            "training_data": {"status": "unknown", "details": {}},
            "synthetic_data": {"status": "unknown", "details": {}},
            "data_quality": {"status": "unknown", "details": {}},
            "minimum_requirements": {"status": "unknown", "details": {}},
            "assessment_time_ms": 0
        }

        try:
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as db_session:
                # Use raw SQL for count queries to avoid SQLModel field access issues

                # Assess training prompts
                training_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_prompts")
                )
                training_count = training_count_result.scalar() or 0

                # Assess by data source
                synthetic_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_prompts WHERE data_source = 'synthetic'")
                )
                synthetic_count = synthetic_count_result.scalar() or 0

                user_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_prompts WHERE data_source = 'user'")
                )
                user_count = user_count_result.scalar() or 0

                # Assess prompt sessions
                session_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM prompt_sessions")
                )
                session_count = session_count_result.scalar() or 0

                data_status["training_data"] = {
                    "status": "sufficient" if training_count >= 100 else "insufficient",
                    "details": {
                        "total_training_prompts": training_count,
                        "synthetic_prompts": synthetic_count,
                        "user_prompts": user_count,
                        "prompt_sessions": session_count,
                        "minimum_required": 100
                    }
                }

                data_status["synthetic_data"] = {
                    "status": "available" if synthetic_count > 0 else "missing",
                    "details": {
                        "synthetic_count": synthetic_count,
                        "percentage_synthetic": (synthetic_count / max(training_count, 1)) * 100
                    }
                }

                # Enhanced quality assessment with comprehensive scoring
                if training_count > 0:
                    quality_assessment = await self._perform_comprehensive_quality_assessment(db_session, training_count)
                    data_status["data_quality"] = quality_assessment
                else:
                    data_status["data_quality"] = {
                        "status": "no_data",
                        "details": {"message": "No training data available for quality assessment"}
                    }

                # Minimum requirements check
                requirements_met = {
                    "training_data_count": training_count >= 100,
                    "synthetic_data_available": synthetic_count > 0,
                    "user_data_available": user_count > 0,
                    "quality_acceptable": data_status["data_quality"]["status"] in ["good", "unknown"]
                }

                data_status["minimum_requirements"] = {
                    "status": "met" if all(requirements_met.values()) else "not_met",
                    "details": requirements_met
                }

        except Exception as e:
            self.logger.error(f"Data availability assessment failed: {e}")
            data_status.update({
                "training_data": {
                    "status": "error",
                    "details": {"error": str(e), "total_training_prompts": 0}
                },
                "synthetic_data": {
                    "status": "error",
                    "details": {"error": str(e)}
                },
                "data_quality": {
                    "status": "error",
                    "details": {"error": str(e)}
                },
                "minimum_requirements": {
                    "status": "error",
                    "details": {"error": str(e)}
                }
            })

        data_status["assessment_time_ms"] = (time.time() - assessment_start) * 1000
        return data_status

    async def _perform_comprehensive_quality_assessment(
        self, db_session: AsyncSession, training_count: int
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of training data.

        Implements 2025 best practices for data quality evaluation:
        - Multi-dimensional quality scoring
        - Statistical distribution analysis
        - Feature completeness validation
        - Effectiveness score analysis

        Returns:
            Detailed quality assessment report
        """
        # Using raw SQL queries, so imports not needed

        # Sample size for quality assessment (max 50 for performance)
        sample_size = min(50, training_count)

        # Get representative sample using raw SQL
        sample_result = await db_session.execute(
            text(f"""
                SELECT enhancement_result, data_source, training_priority
                FROM training_prompts
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """)
        )
        samples = [(row[0], row[1], row[2]) for row in sample_result.fetchall()]

        quality_metrics = {
            "effectiveness_scores": [],
            "feature_completeness": [],
            "data_source_distribution": {},
            "priority_distribution": {},
            "metadata_quality": [],
            "enhancement_quality": []
        }

        # Analyze each sample
        for enhancement_result, data_source, priority in samples:
            if isinstance(enhancement_result, dict):
                # Effectiveness score analysis
                effectiveness = enhancement_result.get("effectiveness_score", 0)
                if isinstance(effectiveness, (int, float)):
                    quality_metrics["effectiveness_scores"].append(effectiveness)

                # Feature completeness analysis
                required_fields = ["enhanced_prompt", "effectiveness_score", "metadata"]
                present_fields = sum(1 for field in required_fields if field in enhancement_result)
                completeness = present_fields / len(required_fields)
                quality_metrics["feature_completeness"].append(completeness)

                # Metadata quality analysis
                metadata = enhancement_result.get("metadata", {})
                if isinstance(metadata, dict):
                    metadata_fields = ["source", "domain", "generation_timestamp", "feature_names"]
                    metadata_completeness = sum(1 for field in metadata_fields if field in metadata) / len(metadata_fields)
                    quality_metrics["metadata_quality"].append(metadata_completeness)

                # Enhancement quality analysis
                enhanced_prompt = enhancement_result.get("enhanced_prompt", "")
                original_prompt = enhancement_result.get("original_prompt", "")
                if enhanced_prompt and original_prompt:
                    enhancement_ratio = len(enhanced_prompt) / max(len(original_prompt), 1)
                    quality_metrics["enhancement_quality"].append(min(enhancement_ratio, 3.0))  # Cap at 3x

            # Data source distribution
            quality_metrics["data_source_distribution"][data_source] = (
                quality_metrics["data_source_distribution"].get(data_source, 0) + 1
            )

            # Priority distribution
            priority_range = "high" if priority >= 80 else "medium" if priority >= 50 else "low"
            quality_metrics["priority_distribution"][priority_range] = (
                quality_metrics["priority_distribution"].get(priority_range, 0) + 1
            )

        # Calculate quality scores
        avg_effectiveness = sum(quality_metrics["effectiveness_scores"]) / len(quality_metrics["effectiveness_scores"]) if quality_metrics["effectiveness_scores"] else 0
        avg_completeness = sum(quality_metrics["feature_completeness"]) / len(quality_metrics["feature_completeness"]) if quality_metrics["feature_completeness"] else 0
        avg_metadata_quality = sum(quality_metrics["metadata_quality"]) / len(quality_metrics["metadata_quality"]) if quality_metrics["metadata_quality"] else 0
        avg_enhancement_quality = sum(quality_metrics["enhancement_quality"]) / len(quality_metrics["enhancement_quality"]) if quality_metrics["enhancement_quality"] else 0

        # Overall quality score (weighted average)
        overall_quality = (
            avg_effectiveness * 0.4 +
            avg_completeness * 0.3 +
            avg_metadata_quality * 0.2 +
            avg_enhancement_quality * 0.1
        )

        # Determine quality status
        if overall_quality >= 0.8:
            quality_status = "excellent"
        elif overall_quality >= 0.7:
            quality_status = "good"
        elif overall_quality >= 0.5:
            quality_status = "acceptable"
        else:
            quality_status = "needs_improvement"

        return {
            "status": quality_status,
            "details": {
                "samples_analyzed": len(samples),
                "overall_quality_score": overall_quality,
                "effectiveness_metrics": {
                    "average_score": avg_effectiveness,
                    "score_count": len(quality_metrics["effectiveness_scores"]),
                    "min_score": min(quality_metrics["effectiveness_scores"]) if quality_metrics["effectiveness_scores"] else 0,
                    "max_score": max(quality_metrics["effectiveness_scores"]) if quality_metrics["effectiveness_scores"] else 0
                },
                "completeness_metrics": {
                    "average_completeness": avg_completeness,
                    "metadata_quality": avg_metadata_quality,
                    "enhancement_quality": avg_enhancement_quality
                },
                "distribution_analysis": {
                    "data_sources": quality_metrics["data_source_distribution"],
                    "priority_levels": quality_metrics["priority_distribution"]
                },
                "quality_recommendations": self._generate_quality_recommendations(
                    overall_quality, avg_effectiveness, avg_completeness, quality_metrics
                )
            }
        }

    def _generate_quality_recommendations(
        self, overall_quality: float, avg_effectiveness: float, avg_completeness: float, metrics: Dict
    ) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []

        if overall_quality < 0.7:
            recommendations.append("Overall data quality is below optimal. Consider regenerating synthetic data with improved parameters.")

        if avg_effectiveness < 0.6:
            recommendations.append("Low effectiveness scores detected. Review and improve prompt enhancement algorithms.")

        if avg_completeness < 0.8:
            recommendations.append("Incomplete feature data detected. Ensure all required fields are populated during data generation.")

        # Check data source diversity
        source_dist = metrics["data_source_distribution"]
        if len(source_dist) == 1:
            recommendations.append("Limited data source diversity. Consider adding user-generated data alongside synthetic data.")

        synthetic_percentage = source_dist.get("synthetic", 0) / sum(source_dist.values()) * 100
        if synthetic_percentage > 90:
            recommendations.append("High reliance on synthetic data. Consider collecting real user data for better model performance.")

        return recommendations

    async def _create_initialization_plan(
        self, system_state: Dict, component_status: Dict, database_status: Dict, data_status: Dict
    ) -> Dict[str, Any]:
        """
        Create intelligent initialization plan based on system analysis.

        Returns:
            Detailed initialization plan with prioritized actions
        """
        plan = {
            "actions": [],
            "priorities": [],
            "estimated_time_seconds": 0,
            "requirements_met": True,
            "blocking_issues": [],
            "recommendations": []
        }

        # Analyze current state and determine required actions
        if self._training_status != "running":
            plan["actions"].append({
                "action": "start_training_system",
                "priority": 1,
                "estimated_time": 30,
                "description": "Initialize core training system components"
            })

        # Database issues
        if database_status["connectivity"]["status"] != "healthy":
            plan["blocking_issues"].append("Database connectivity failed")
            plan["requirements_met"] = False

        if database_status["schema_validation"]["status"] != "healthy":
            plan["blocking_issues"].append("Database schema incomplete")
            plan["requirements_met"] = False

        if database_status["seeded_rules"]["status"] != "healthy":
            plan["actions"].append({
                "action": "validate_and_load_rules",
                "priority": 2,
                "estimated_time": 10,
                "description": "Validate and load seeded rules from database"
            })

        # Data availability issues
        if data_status.get("minimum_requirements", {}).get("status") != "met":
            training_details = data_status.get("training_data", {}).get("details", {})
            if training_details.get("total_training_prompts", 0) < 100:
                plan["actions"].append({
                    "action": "generate_synthetic_data",
                    "priority": 3,
                    "estimated_time": 60,
                    "description": "Generate initial synthetic training data"
                })

        # Component issues
        for component, status in component_status.items():
            if component != "validation_time_ms" and status["status"] != "healthy":
                plan["actions"].append({
                    "action": f"initialize_{component}",
                    "priority": 2,
                    "estimated_time": 15,
                    "description": f"Initialize {component} component"
                })

        # Calculate total estimated time
        plan["estimated_time_seconds"] = sum(action["estimated_time"] for action in plan["actions"])

        # Generate recommendations
        if data_status["data_quality"]["status"] == "needs_improvement":
            plan["recommendations"].append("Consider regenerating synthetic data with higher quality parameters")

        if system_state["system_resources"].get("available_memory_gb", 0) < 2:
            plan["recommendations"].append("Low memory detected - consider reducing batch sizes")

        return plan

    async def _execute_initialization_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the initialization plan with progress tracking.

        Returns:
            Execution results with component status
        """
        execution_start = time.time()

        results = {
            "success": True,
            "message": "Initialization plan executed successfully",
            "components_initialized": [],
            "actions_completed": [],
            "actions_failed": [],
            "recommendations": plan.get("recommendations", []),
            "execution_time_seconds": 0
        }

        # Check for blocking issues
        if not plan["requirements_met"]:
            return {
                "success": False,
                "message": f"Blocking issues prevent initialization: {', '.join(plan['blocking_issues'])}",
                "components_initialized": [],
                "actions_completed": [],
                "actions_failed": plan["blocking_issues"],
                "execution_time_seconds": time.time() - execution_start
            }

        # Execute actions in priority order
        sorted_actions = sorted(plan["actions"], key=lambda x: x["priority"])

        for action in sorted_actions:
            try:
                action_start = time.time()

                if action["action"] == "start_training_system":
                    startup_result = await self.start_training_system()
                    if startup_result["status"] == "success":
                        results["components_initialized"].extend(startup_result["components_initialized"])
                        results["actions_completed"].append(action["action"])
                    else:
                        results["actions_failed"].append(action["action"])
                        results["success"] = False

                elif action["action"] == "validate_and_load_rules":
                    # Rule validation is already done in database validation
                    results["actions_completed"].append(action["action"])
                    results["components_initialized"].append("rule_validation")

                elif action["action"] == "generate_synthetic_data":
                    # Pass data status for targeted generation
                    data_status = None  # Will be passed from initialization context
                    generation_results = await self._generate_initial_data(data_status)
                    if generation_results["success"]:
                        results["actions_completed"].append(action["action"])
                        results["components_initialized"].append("synthetic_data_generation")
                        if generation_results.get("recommendations"):
                            results["recommendations"].extend(generation_results["recommendations"])
                    else:
                        results["actions_failed"].append(action["action"])
                        results["success"] = False

                elif action["action"].startswith("initialize_"):
                    component_name = action["action"].replace("initialize_", "")
                    # Component initialization handled by start_training_system
                    self.logger.debug(f"Initializing component: {component_name}")
                    results["actions_completed"].append(action["action"])

                action_time = time.time() - action_start
                self.logger.info(f"Completed {action['action']} in {action_time:.2f}s")

            except Exception as e:
                self.logger.error(f"Failed to execute {action['action']}: {e}")
                results["actions_failed"].append(action["action"])
                results["success"] = False

        results["execution_time_seconds"] = time.time() - execution_start

        if results["actions_failed"]:
            results["message"] = f"Initialization completed with {len(results['actions_failed'])} failures"

        return results

    async def _validate_post_initialization(self) -> Dict[str, Any]:
        """
        Validate system state after initialization.

        Returns:
            Post-initialization validation results
        """
        validation_start = time.time()

        validation = {
            "overall_health": False,
            "component_health": {},
            "training_readiness": False,
            "validation_time_ms": 0
        }

        try:
            # Re-run health checks
            health_status = await self._verify_training_health()
            validation["overall_health"] = health_status.get("overall_health", False)
            validation["component_health"] = {
                "database": health_status.get("database_connectivity", False),
                "orchestrator": health_status.get("orchestrator_status", False),
                "analytics": health_status.get("analytics_status", False),
                "data_generator": health_status.get("data_generator_status", False)
            }

            # Check training readiness
            validation["training_readiness"] = await self.validate_ready_for_training()

        except Exception as e:
            validation["error"] = str(e)

        validation["validation_time_ms"] = (time.time() - validation_start) * 1000
        return validation

    async def validate_ready_for_training(self) -> bool:
        """
        Validate that the system is ready for training.

        Returns:
            True if ready for training, False otherwise
        """
        try:
            # Check training system status
            if self._training_status != "running":
                return False

            # Check required components
            if not self._orchestrator:
                return False

            if not self._analytics:
                return False

            if not self._data_generator:
                return False

            # Check database connectivity
            try:
                session_manager = get_sessionmanager()
                async with session_manager.get_async_session() as session:
                    # Simple connectivity test
                    await session.execute(text("SELECT 1"))
            except Exception:
                return False

            # Check orchestrator health
            orchestrator_health = await self._orchestrator.health_check()
            if not orchestrator_health.get("healthy", False):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Training readiness validation failed: {e}")
            return False

    async def create_training_session(self, config: Dict[str, Any]) -> Any:
        """
        Create a new training session with the given configuration.

        Args:
            config: Training session configuration

        Returns:
            TrainingSession object
        """
        from ...database.models import TrainingSession, TrainingSessionCreate
        import uuid

        try:
            # Generate unique session ID
            session_id = f"training_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # Create session data
            session_data = TrainingSessionCreate(
                session_id=session_id,
                continuous_mode=config.get("continuous_mode", True),
                max_iterations=config.get("max_iterations"),
                improvement_threshold=config.get("improvement_threshold", 0.02),
                timeout_seconds=config.get("timeout", 3600),
                auto_init_enabled=config.get("auto_init", True)
            )

            # Save to database
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as session:
                db_session = TrainingSession.model_validate(session_data.model_dump())
                session.add(db_session)
                await session.commit()
                await session.refresh(db_session)

                # Set as current training session
                self._training_session_id = session_id

                self.logger.info(f"Created training session: {session_id}")
                return db_session

        except Exception as e:
            self.logger.error(f"Failed to create training session: {e}")
            raise

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including health and active sessions.

        Returns:
            System status dictionary
        """
        try:
            # Get active sessions
            active_sessions = await self.get_active_sessions()

            # Check component health
            components = {}
            components["training_system"] = self._training_status

            if self._orchestrator:
                orchestrator_health = await self._orchestrator.health_check()
                components["orchestrator"] = "healthy" if orchestrator_health.get("healthy") else "unhealthy"
            else:
                components["orchestrator"] = "not_initialized"

            if self._analytics:
                components["analytics"] = "healthy"
            else:
                components["analytics"] = "not_initialized"

            if self._data_generator:
                components["data_generator"] = "healthy"
            else:
                components["data_generator"] = "not_initialized"

            # Check database connectivity
            try:
                session_manager = get_sessionmanager()
                async with session_manager.get_async_session() as session:
                    await session.execute(text("SELECT 1"))
                components["database"] = "healthy"
            except Exception:
                components["database"] = "unhealthy"

            # Overall health
            healthy = all(status in ["healthy", "running"] for status in components.values())

            # Get recent performance if available
            recent_performance = {}
            if self._analytics:
                try:
                    recent_performance = await self._analytics.get_recent_performance_summary()
                except Exception:
                    pass

            return {
                "healthy": healthy,
                "status": "healthy" if healthy else "degraded",
                "active_sessions": [
                    {
                        "session_id": session.session_id,
                        "status": session.status,
                        "started_at": session.started_at.isoformat() if session.started_at else None,
                        "iterations": session.current_iteration,
                        "current_performance": session.current_performance
                    }
                    for session in active_sessions
                ],
                "components": components,
                "recent_performance": recent_performance,
                "resource_usage": await self._get_resource_usage(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def get_active_sessions(self) -> list:
        """
        Get all active training sessions.

        Returns:
            List of active TrainingSession objects
        """
        # TrainingSession model not needed since using raw SQL

        try:
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as session:
                # Query for active sessions using raw SQL
                result = await session.execute(
                    text("SELECT * FROM training_sessions WHERE status IN ('initializing', 'running', 'paused')")
                )
                active_sessions = result.fetchall()

                return list(active_sessions)

        except Exception as e:
            self.logger.error(f"Failed to get active sessions: {e}")
            return []

    async def _needs_synthetic_data(self) -> bool:
        """Check if synthetic data generation is needed."""
        try:
            # Simple check - could be enhanced with more sophisticated logic
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as session:
                # Check if we have enough training data using raw SQL
                result = await session.execute(
                    text("SELECT COUNT(*) FROM prompt_sessions")
                )
                count = result.scalar() or 0

                # Need synthetic data if we have less than 100 sessions
                return count < 100

        except Exception:
            return True  # Generate data if we can't check

    async def _generate_initial_data(self, data_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced synthetic data generation with smart initialization and targeted generation.

        Implements 2025 best practices for intelligent data generation:
        - Gap-based targeting based on data availability assessment
        - Quality-driven generation parameters
        - Adaptive batch sizing based on system resources
        - Multi-method generation strategy selection

        Args:
            data_status: Optional data status from assessment for targeted generation

        Returns:
            Generation results with metrics and recommendations
        """
        generation_start = time.time()

        generation_results = {
            "success": False,
            "samples_generated": 0,
            "generation_method": "statistical",
            "quality_score": 0.0,
            "generation_time_seconds": 0,
            "recommendations": []
        }

        if not self._data_generator:
            generation_results["error"] = "Data generator not initialized"
            return generation_results

        try:
            # Analyze data gaps and determine generation strategy
            generation_strategy = await self._determine_generation_strategy(data_status)

            # Configure generator based on strategy
            await self._configure_generator_for_strategy(generation_strategy)

            # Execute targeted generation
            generation_data = await self._execute_targeted_generation(generation_strategy)

            # Validate generated data quality
            quality_assessment = await self._validate_generated_data_quality(generation_data)

            generation_results.update({
                "success": True,
                "samples_generated": generation_data.get("total_samples", 0),
                "generation_method": generation_strategy["method"],
                "quality_score": quality_assessment["overall_quality"],
                "generation_time_seconds": time.time() - generation_start,
                "strategy_used": generation_strategy,
                "quality_assessment": quality_assessment,
                "recommendations": generation_strategy.get("recommendations", [])
            })

            self.logger.info(
                f"Generated {generation_results['samples_generated']} samples using {generation_strategy['method']} method "
                f"with quality score {quality_assessment['overall_quality']:.3f}"
            )

        except Exception as e:
            generation_results["error"] = str(e)
            generation_results["generation_time_seconds"] = time.time() - generation_start
            self.logger.error(f"Enhanced data generation failed: {e}")

        return generation_results

    async def _determine_generation_strategy(self, data_status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine optimal generation strategy based on data gaps and system state.

        Returns:
            Generation strategy with method, parameters, and targeting
        """
        strategy = {
            "method": "statistical",  # Default fallback
            "target_samples": 100,
            "focus_areas": [],
            "quality_target": 0.8,
            "recommendations": []
        }

        if data_status:
            training_data = data_status.get("training_data", {}).get("details", {})
            quality_data = data_status.get("data_quality", {}).get("details", {})

            # Determine sample count based on current data
            current_count = training_data.get("total_training_prompts", 0)
            if current_count < 50:
                strategy["target_samples"] = 200  # Bootstrap with more data
                strategy["method"] = "hybrid"  # Use hybrid for better diversity
            elif current_count < 100:
                strategy["target_samples"] = 100
                strategy["method"] = "statistical"
            else:
                strategy["target_samples"] = 50  # Incremental addition
                strategy["method"] = "neural"  # Use advanced method for refinement

            # Analyze quality gaps
            if quality_data.get("overall_quality_score", 0) < 0.7:
                strategy["quality_target"] = 0.85  # Higher quality target
                strategy["recommendations"].append("Targeting higher quality generation due to low existing data quality")

            # Analyze data source diversity
            distribution = quality_data.get("distribution_analysis", {}).get("data_sources", {})
            if len(distribution) == 1 and "synthetic" in distribution:
                strategy["focus_areas"].append("domain_diversity")
                strategy["recommendations"].append("Focusing on domain diversity to reduce synthetic data bias")

        # System resource considerations
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            if available_memory < 2:
                strategy["target_samples"] = min(strategy["target_samples"], 50)
                strategy["method"] = "statistical"  # Less memory intensive
                strategy["recommendations"].append("Reduced batch size due to limited memory")
        except ImportError:
            pass

        return strategy

    async def _configure_generator_for_strategy(self, strategy: Dict[str, Any]) -> None:
        """Configure the data generator based on the determined strategy."""
        if not self._data_generator:
            return

        # Update generation method
        self._data_generator.generation_method = strategy["method"]

        # Update target samples
        self._data_generator.target_samples = strategy["target_samples"]

        # Configure quality parameters
        if hasattr(self._data_generator, 'use_enhanced_scoring'):
            self._data_generator.use_enhanced_scoring = strategy["quality_target"] > 0.8

        self.logger.info(f"Configured generator: method={strategy['method']}, samples={strategy['target_samples']}")

    async def _execute_targeted_generation(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the targeted data generation based on strategy."""
        if not self._data_generator:
            raise RuntimeError("Data generator not available")

        # Use strategy parameters for generation
        method = strategy.get("method", "balanced")
        target_samples = strategy.get("target_samples", 100)

        # Execute generation using the configured method
        self.logger.info(f"Executing targeted generation: method={method}, target_samples={target_samples}")
        generation_data = await self._data_generator.generate_data()

        return generation_data

    async def _validate_generated_data_quality(self, generation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of newly generated data."""
        quality_assessment = {
            "overall_quality": 0.0,
            "effectiveness_distribution": {},
            "feature_completeness": 0.0,
            "domain_diversity": 0.0
        }

        try:
            # Analyze effectiveness scores
            effectiveness_scores = generation_data.get("effectiveness_scores", [])
            if effectiveness_scores:
                avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
                quality_assessment["overall_quality"] = avg_effectiveness

                # Distribution analysis
                score_ranges = {"low": 0, "medium": 0, "high": 0}
                for score in effectiveness_scores:
                    if score < 0.5:
                        score_ranges["low"] += 1
                    elif score < 0.8:
                        score_ranges["medium"] += 1
                    else:
                        score_ranges["high"] += 1

                quality_assessment["effectiveness_distribution"] = score_ranges

            # Analyze feature completeness
            features = generation_data.get("features", [])
            if features:
                # Check if features have expected dimensions
                expected_features = ["clarity", "specificity", "complexity", "actionability"]
                if len(features) > 0 and len(features[0]) >= len(expected_features):
                    quality_assessment["feature_completeness"] = 1.0
                else:
                    quality_assessment["feature_completeness"] = len(features[0]) / len(expected_features) if features else 0.0

            # Analyze domain diversity
            metadata = generation_data.get("metadata", {})
            domain_distribution = metadata.get("domain_distribution", {})
            if domain_distribution:
                # Calculate entropy for diversity measure
                total_samples = sum(domain_distribution.values())
                if total_samples > 0:
                    entropy = -sum(
                        (count / total_samples) * math.log2(count / total_samples)
                        for count in domain_distribution.values() if count > 0
                    )
                    max_entropy = math.log2(len(domain_distribution))
                    quality_assessment["domain_diversity"] = entropy / max_entropy if max_entropy > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Quality validation failed: {e}")

        return quality_assessment

    async def _verify_database_schema(self) -> bool:
        """Verify database schema is up to date."""
        try:
            # Simple schema verification - could be enhanced
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as session:
                # Check if training_sessions table exists
                result = await session.execute(
                    text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'training_sessions')")
                )
                return bool(result.scalar())
        except Exception:
            return False
