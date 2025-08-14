"""Training Orchestrator Service - Clean Architecture Implementation

Implements training workflow orchestration and component coordination.
Extracted from training_system_manager.py (2109 lines) as part of decomposition.
"""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from sqlalchemy import text

from prompt_improver.cli.core.signal_handler import SignalOperation
from prompt_improver.cli.services.training_protocols import (
    TrainingMetricsProtocol,
    TrainingPersistenceProtocol,
    TrainingValidatorProtocol,
)
from prompt_improver.core.di.ml_container import MLServiceContainer
from prompt_improver.core.factories.ml_pipeline_factory import (
    MLPipelineOrchestratorFactory,
)
from prompt_improver.core.protocols.ml_protocols import (
    ComponentFactoryProtocol,
    ServiceContainerProtocol,
)
from prompt_improver.core.services.analytics_factory import get_analytics_interface
from prompt_improver.database import ManagerMode, get_database_services, get_sessionmanager
from prompt_improver.ml.orchestration.config.orchestrator_config import (
    OrchestratorConfig,
)
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator,
)
from prompt_improver.ml.preprocessing.orchestrator import (
    ProductionSyntheticDataGenerator,
)


class TrainingOrchestrator:
    """Training workflow orchestrator implementing Clean Architecture patterns.
    
    Responsibilities:
    - Core workflow orchestration and component coordination
    - Training system lifecycle management
    - Component initialization and health monitoring
    - Signal handling integration for graceful shutdown
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        validator: Optional[TrainingValidatorProtocol] = None,
        metrics: Optional[TrainingMetricsProtocol] = None,
        persistence: Optional[TrainingPersistenceProtocol] = None,
    ):
        self.console = console or Console()
        self.logger = logging.getLogger("apes.training_orchestrator")
        
        # Protocol-based dependencies
        self.validator = validator
        self.metrics = metrics
        self.persistence = persistence
        
        # Core orchestration state
        self._training_status = "stopped"
        self._training_session_id: Optional[str] = None
        self._orchestrator: Optional[MLPipelineOrchestrator] = None
        self._analytics: Any = None
        self._data_generator: Optional[ProductionSyntheticDataGenerator] = None
        
        # Performance tracking
        self._startup_time: Optional[float] = None
        
        # Training system data directory
        self.training_data_dir = Path.home() / ".local" / "share" / "apes" / "training"
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified session manager for database access
        self._unified_session_manager = None
        
        # Signal handling components
        self.signal_handler = None
        self.background_manager = None
        self._shutdown_priority = 5
        
        # Initialize signal handling
        self._init_signal_handlers()

    def _init_signal_handlers(self):
        """Initialize signal handler with lazy import to avoid circular dependency."""
        try:
            from prompt_improver.cli.core.signal_handler import AsyncSignalHandler
            from prompt_improver.performance.monitoring.health.background_manager import (
                get_background_task_manager,
            )

            if self.signal_handler is None:
                self.signal_handler = AsyncSignalHandler(console=self.console)
                
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    self.signal_handler.setup_signal_handlers(loop)
                except RuntimeError:
                    pass

            if self.background_manager is None:
                self.background_manager = get_background_task_manager()

            self._register_signal_handlers()

        except ImportError as e:
            self.logger.warning(f"Signal handling integration not available: {e}")

    def _register_signal_handlers(self):
        """Register orchestrator-specific signal handlers."""
        if self.signal_handler is None:
            self.logger.warning("Signal handler not initialized, skipping registration")
            return

        import signal

        # Shutdown coordination with high priority
        self.signal_handler.register_shutdown_handler(
            "TrainingOrchestrator_shutdown", self.graceful_shutdown_handler
        )

        # Emergency checkpoint creation (SIGUSR1)
        self.signal_handler.register_operation_handler(
            SignalOperation.CHECKPOINT, self.create_training_emergency_checkpoint
        )

        # Status reporting (SIGUSR2)
        self.signal_handler.register_operation_handler(
            SignalOperation.STATUS_REPORT, self.generate_training_status_report
        )

        # Signal chaining for coordinated shutdown preparation
        self.signal_handler.add_signal_chain_handler(
            signal.SIGTERM,
            self.prepare_training_shutdown,
            priority=self._shutdown_priority,
        )

        # SIGINT coordination for graceful training interruption
        self.signal_handler.add_signal_chain_handler(
            signal.SIGINT,
            self.prepare_training_interruption,
            priority=self._shutdown_priority,
        )

        self.logger.info("TrainingOrchestrator signal handlers registered")

    async def start_training_system(self) -> Dict[str, Any]:
        """Start training system components - orchestrates full initialization.
        
        Returns:
            Training system startup results with performance metrics
        """
        startup_start = time.time()
        self.logger.info("Starting training system orchestration")

        try:
            self._training_status = "starting"

            # Phase 1: Initialize database connections
            await self._initialize_training_database()

            # Phase 2: Initialize ML Pipeline Orchestrator
            await self._initialize_training_orchestrator()

            # Phase 3: Initialize analytics service
            await self._initialize_training_analytics()

            # Phase 4: Initialize synthetic data generator
            await self._initialize_data_generator()

            # Phase 5: Verify system health using metrics service
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
                    "synthetic_data_orchestrator",
                ],
                "health_status": health_status,
                "resource_usage": await self._get_resource_usage(),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            self.logger.info(
                f"Training system started successfully in {self._startup_time:.2f}s"
            )
            return startup_results

        except Exception as e:
            self._training_status = "failed"
            self.logger.error(f"Failed to start training system: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def stop_training_system(self, graceful: bool = True) -> bool:
        """Stop training system gracefully with progress preservation.
        
        Args:
            graceful: Whether to perform graceful shutdown
            
        Returns:
            True if shutdown successful, False otherwise
        """
        self.logger.info(f"Stopping training system (graceful={graceful})")

        try:
            self._training_status = "stopping"

            # Save current training progress via persistence service
            if self._training_session_id and self.persistence:
                await self.persistence.save_training_progress(self._training_session_id)

            # Stop orchestrator workflows gracefully
            if self._orchestrator and graceful:
                await self._orchestrator._stop_all_workflows()

            # Cleanup training resources
            await self._cleanup_training_resources()

            # Close database connections
            await get_sessionmanager().close()

            self._training_status = "stopped"
            self._training_session_id = None

            self.logger.info("Training system stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping training system: {e}")
            return False

    async def get_training_status(self) -> Dict[str, Any]:
        """Get training system status using metrics service.
        
        Returns:
            Training system status and metrics
        """
        status = {
            "training_system_status": self._training_status,
            "training_session_id": self._training_session_id,
            "uptime_seconds": time.time() - self._startup_time
            if self._startup_time
            else 0,
            "resource_usage": await self._get_resource_usage(),
            "components": {
                "orchestrator": "running" if self._orchestrator else "stopped",
                "analytics": "running" if self._analytics else "stopped",
                "data_generator": "running" if self._data_generator else "stopped",
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Add active workflow status if orchestrator is running
        if self._orchestrator:
            workflows = await self._orchestrator.list_workflows()
            status["active_workflows"] = [w.workflow_id for w in workflows]

        return status

    async def _ensure_database_services(self):
        """Ensure database services are available for orchestration."""
        if self._unified_session_manager is None:
            self._unified_session_manager = await get_database_services(
                ManagerMode.MCP_SERVER
            )

    async def _initialize_training_database(self):
        """Initialize database connections for training system."""
        self.logger.info("Initializing training database connections")

        session_manager = get_sessionmanager()
        async with session_manager.get_async_session() as db_session:
            result = await db_session.execute(
                text(
                    "SELECT table_name FROM information_schema.tables WHERE table_name IN ('improvement_sessions', 'rule_performance')"
                )
            )
            tables = [row[0] for row in result.fetchall()]

            if len(tables) < 2:
                raise RuntimeError("Training database tables not found")

        self.logger.info("Training database connections initialized")

    async def _initialize_training_orchestrator(self):
        """Initialize ML Pipeline Orchestrator with training-focused configuration."""
        self.logger.info("Initializing training orchestrator")

        config = OrchestratorConfig(
            max_concurrent_workflows=3,  # Optimized for training workloads
            training_timeout=1800,  # 30 minutes for training workflows
            debug_mode=False,
            verbose_logging=False,
        )

        self._orchestrator = MLPipelineOrchestrator(config)
        await self._orchestrator.initialize()

        self.logger.info("Training orchestrator initialized")

    async def _initialize_training_analytics(self):
        """Initialize analytics service for training metrics."""
        self.logger.info("Initializing training analytics")

        analytics_factory = get_analytics_interface()
        self._analytics = analytics_factory() if analytics_factory else None

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
            "response_time_ms": 0,
        }

        try:
            # Test database
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as db_session:
                await db_session.execute(text("SELECT 1"))
                health_status["database_connectivity"] = True

            # Test orchestrator
            if self._orchestrator:
                health_status["orchestrator_status"] = (
                    hasattr(self._orchestrator, "_is_initialized")
                    and self._orchestrator._is_initialized
                    and self._orchestrator.state.name in ["IDLE", "RUNNING"]
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
                health_status["data_generator_status"],
            ])

            health_status["response_time_ms"] = (time.time() - health_start) * 1000

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)

        return health_status

    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage for orchestration tracking."""
        try:
            import psutil

            process = psutil.Process()

            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
            }
        except ImportError:
            return {"memory_mb": 0, "cpu_percent": 0, "open_files": 0}

    async def _cleanup_training_resources(self):
        """Cleanup training system resources during shutdown."""
        self.logger.info("Cleaning up training resources")

        # Cleanup orchestrator
        if self._orchestrator:
            await self._orchestrator.shutdown()
            self._orchestrator = None

        # Reset component references
        self._analytics = None
        self._data_generator = None

    # Signal handling methods
    async def graceful_shutdown_handler(self, shutdown_context):
        """Handle graceful shutdown with training progress preservation."""
        self.logger.info("TrainingOrchestrator graceful shutdown initiated")

        try:
            success = await self.stop_training_system(graceful=True)
            await self._emergency_training_cleanup()

            return {
                "status": "success" if success else "partial",
                "component": "TrainingOrchestrator",
                "training_stopped": success,
                "progress_preserved": self._training_session_id is not None,
            }
        except Exception as e:
            self.logger.error(f"TrainingOrchestrator shutdown error: {e}")
            return {
                "status": "error",
                "component": "TrainingOrchestrator",
                "error": str(e),
            }

    async def create_training_emergency_checkpoint(self, signal_context):
        """Create emergency training checkpoint on SIGUSR1 signal."""
        self.logger.info("Creating emergency training checkpoint")

        try:
            if not self._training_session_id:
                return {
                    "status": "no_active_session",
                    "message": "No active training session for checkpoint",
                }

            # Use persistence service for checkpoint creation
            if self.persistence:
                checkpoint_id = await self.persistence.create_checkpoint(
                    self._training_session_id
                )

            # Get current performance metrics via metrics service
            performance_metrics = {}
            if self.metrics:
                performance_metrics = await self.metrics.get_current_performance_metrics()

            return {
                "status": "checkpoint_created",
                "checkpoint_id": checkpoint_id if self.persistence else "unavailable",
                "session_id": self._training_session_id,
                "performance_metrics": performance_metrics,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Emergency checkpoint creation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def generate_training_status_report(self, signal_context):
        """Generate comprehensive training status report on SIGUSR2 signal."""
        self.logger.info("Generating training status report")

        try:
            # Get comprehensive system status
            system_status = await self.get_training_status()

            # Get training-specific metrics via metrics service
            training_metrics = {}
            if self.metrics:
                training_metrics = await self.metrics.get_detailed_training_metrics()

            return {
                "status": "report_generated",
                "system_status": system_status,
                "training_metrics": training_metrics,
                "resource_usage": await self._get_resource_usage(),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Status report generation failed: {e}")
            return {"status": "error", "error": str(e)}

    def prepare_training_shutdown(self, signum, signal_name):
        """Prepare training system for coordinated shutdown."""
        self.logger.info(f"Preparing training system for shutdown ({signal_name})")

        try:
            if self._training_status == "running":
                self._training_status = "shutting_down"

            return {
                "prepared": True,
                "component": "TrainingOrchestrator",
                "training_status": self._training_status,
                "active_session": self._training_session_id is not None,
                "orchestrator_active": self._orchestrator is not None,
            }
        except Exception as e:
            self.logger.error(f"Training shutdown preparation failed: {e}")
            return {
                "prepared": False,
                "component": "TrainingOrchestrator",
                "error": str(e),
            }

    def prepare_training_interruption(self, signum, signal_name):
        """Prepare training system for user interruption (Ctrl+C)."""
        self.logger.info(f"Preparing training system for interruption ({signal_name})")

        try:
            return {
                "prepared": True,
                "component": "TrainingOrchestrator",
                "interruption_type": "user_requested",
                "progress_preservation_ready": True,
                "active_session": self._training_session_id,
                "can_resume": True,
            }
        except Exception as e:
            self.logger.error(f"Training interruption preparation failed: {e}")
            return {
                "prepared": False,
                "component": "TrainingOrchestrator",
                "error": str(e),
            }

    async def _emergency_training_cleanup(self):
        """Emergency cleanup for training resources during shutdown."""
        try:
            if self._orchestrator:
                try:
                    await self._orchestrator.shutdown()
                except Exception as e:
                    self.logger.warning(f"Orchestrator emergency shutdown failed: {e}")

            if self._training_session_id and self.persistence:
                try:
                    await self.persistence.save_training_progress(self._training_session_id)
                except Exception as e:
                    self.logger.warning(f"Emergency progress save failed: {e}")

            try:
                await self.background_manager.stop(timeout=5.0)
            except Exception as e:
                self.logger.warning(f"Background task cleanup failed: {e}")

            self.logger.info("Emergency training cleanup completed")
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")

    # Properties for external access
    @property
    def training_status(self) -> str:
        """Get current training status."""
        return self._training_status

    @property
    def training_session_id(self) -> Optional[str]:
        """Get current training session ID."""
        return self._training_session_id

    @property
    def orchestrator(self) -> Optional[MLPipelineOrchestrator]:
        """Get ML pipeline orchestrator instance."""
        return self._orchestrator

    # Add missing methods that the facade expects
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return await self.get_training_status()

    async def create_training_session(self, config: Dict[str, Any]) -> str:
        """Create training session via persistence service."""
        if self.persistence:
            return await self.persistence.create_training_session(config)
        else:
            raise RuntimeError("Persistence service not available")

    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get active sessions via persistence service."""
        if self.persistence:
            return await self.persistence.get_active_sessions()
        else:
            return []